import torch
import torchaudio
from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def _build_mel_spectrogram(audio_config: dict) -> torchaudio.transforms.MelSpectrogram:
    sample_rate = audio_config.get("sample_rate", 16000)
    fft_size = audio_config.get("fft_size", 512)
    win_length = audio_config.get("win_length", fft_size)
    hop_length = audio_config.get("hop_length", 160)
    n_mels = audio_config.get("num_mels", 64)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=fft_size,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        normalized=False,
    )
    preemphasis = audio_config.get("preemphasis")
    if preemphasis is not None:
        coeff = torch.tensor([-preemphasis, 1.0], dtype=torch.float32).view(1, 1, 2)

        class PreEmphasisTransform(nn.Module):
            def __init__(self, base, kernel):
                super().__init__()
                self.base = base
                self.register_buffer("kernel", kernel)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.conv1d(x.unsqueeze(1), self.kernel, padding=1)
                return self.base(x.squeeze(1))

        return PreEmphasisTransform(mel, coeff)
    return mel


class ResNetSpeakerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=(3, 4, 6, 3),
        num_filters=(32, 64, 128, 256),
        encoder_type="ASP",
        log_input=False,
        use_torch_spec=False,
        audio_config=None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config or {}
        self.proj_dim = proj_dim

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.inplanes = num_filters[0]
        self.layer1 = self._create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self._create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(input_dim)
        self.torch_spec = _build_mel_spectrogram(self.audio_config) if self.use_torch_spec else None

        outmap_size = int(self.input_dim / 8)
        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)
        self._init_layers()

    def _init_layers(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _create_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, l2_norm=False):
        x.squeeze_(1)
        if self.use_torch_spec:
            x = self.torch_spec(x)
        if self.log_input:
            x = (x + 1e-6).log()
        x = self.instancenorm(x).unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)
        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
