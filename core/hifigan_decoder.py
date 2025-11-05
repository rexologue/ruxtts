import logging
from typing import Any

import torch

from .generic_utils import load_fsspec
from .hifigan_generator import HifiganGenerator
from .resnet import ResNetSpeakerEncoder

logger = logging.getLogger(__name__)


class HifiDecoder(torch.nn.Module):
    def __init__(
        self,
        input_sample_rate: int = 22050,
        output_sample_rate: int = 24000,
        output_hop_length: int = 256,
        ar_mel_length_compression: int = 1024,
        decoder_input_dim: int = 1024,
        resblock_type_decoder: str = "1",
        resblock_dilation_sizes_decoder: list[list[int]] | None = None,
        resblock_kernel_sizes_decoder: list[int] | None = None,
        upsample_rates_decoder: list[int] | None = None,
        upsample_initial_channel_decoder: int = 512,
        upsample_kernel_sizes_decoder: list[int] | None = None,
        d_vector_dim: int = 512,
        cond_d_vector_in_each_upsampling_layer: bool = True,
        speaker_encoder_audio_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        if resblock_dilation_sizes_decoder is None:
            resblock_dilation_sizes_decoder = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if resblock_kernel_sizes_decoder is None:
            resblock_kernel_sizes_decoder = [3, 7, 11]
        if upsample_rates_decoder is None:
            upsample_rates_decoder = [8, 8, 2, 2]
        if upsample_kernel_sizes_decoder is None:
            upsample_kernel_sizes_decoder = [16, 16, 4, 4]
        if speaker_encoder_audio_config is None:
            speaker_encoder_audio_config = {
                "fft_size": 512,
                "win_length": 400,
                "hop_length": 160,
                "sample_rate": 16000,
                "preemphasis": 0.97,
                "num_mels": 64,
            }

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length
        self.ar_mel_length_compression = ar_mel_length_compression
        self.speaker_encoder_audio_config = speaker_encoder_audio_config

        self.waveform_decoder = HifiganGenerator(
            decoder_input_dim,
            1,
            resblock_type_decoder,
            resblock_dilation_sizes_decoder,
            resblock_kernel_sizes_decoder,
            upsample_kernel_sizes_decoder,
            upsample_initial_channel_decoder,
            upsample_rates_decoder,
            inference_padding=0,
            cond_channels=d_vector_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
            cond_in_each_up_layer=cond_d_vector_in_each_upsampling_layer,
        )
        self.speaker_encoder = ResNetSpeakerEncoder(
            input_dim=64,
            proj_dim=512,
            log_input=True,
            use_torch_spec=True,
            audio_config=speaker_encoder_audio_config,
        )

    def forward(self, latents: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        z = torch.nn.functional.interpolate(
            latents.transpose(1, 2),
            scale_factor=[self.ar_mel_length_compression / self.output_hop_length],
            mode="linear",
        ).squeeze(1)
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z,
                scale_factor=[self.output_sample_rate / self.input_sample_rate],
                mode="linear",
            ).squeeze(0)
        return self.waveform_decoder(z, g=g)

    @torch.inference_mode()
    def inference(self, c: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.forward(c, g=g)

    def load_checkpoint(self, checkpoint_path: str, eval: bool = False) -> None:  # pylint: disable=redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        state = state["model"]
        keys = list(state.keys())
        for key in keys:
            if not (key.startswith("waveform_decoder.") or key.startswith("speaker_encoder.")):
                del state[key]
        self.load_state_dict(state, strict=False)
        if eval:
            self.eval()
            assert not self.training
            self.waveform_decoder.remove_weight_norm()
