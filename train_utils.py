import random
import shutil

import torch
from torch import nn
from torch.optim.optimizer import StateDict

import numpy as np

from config import Cfg
from core.xtts import Xtts
from core.dvae import DiscreteVAE
from core.arch_utils import TorchMelSpectrogram
from core.generic_utils import map_value_list_dict, ValueListDict

def get_optimizer(cfg: Cfg, net: nn.Module) -> torch.optim.Optimizer:
    """Создаёт AdamW с двумя группами: decay и no_decay (bias, norms, embeddings)."""

    norm_modules: tuple[type, ...] = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.GroupNorm, nn.LayerNorm
    )

    emb_modules: tuple[type, ...] = (nn.Embedding, nn.EmbeddingBag)

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for module_name, m in net.named_modules():
        for k, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue

            full_name = f"{module_name}.{k}" if module_name else k

            is_norm = isinstance(m, norm_modules)
            is_emb  = isinstance(m, emb_modules)

            is_bias = (k == "bias") or k.endswith("bias")

            if is_norm or is_emb or is_bias:
                no_decay_params.append(p)
                no_decay_names.append(full_name)
            else:
                decay_params.append(p)
                decay_names.append(full_name)

    # На случай, если одна из групп пуста — это нормально.
    param_groups = [
        {"params": decay_params, "weight_decay": float(cfg.train.weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, cfg.train.lr, betas=cfg.train.betas, eps=cfg.train.eps)

    # для удобного логирования — не публичный атрибут, но полезно
    optimizer._group_names = [decay_names, no_decay_names]

    # в случае если мы продолжаем обучение - подгружаем чекпоинт (если таковой имеется)
    if cfg.exp.resume:
        ckpt = torch.load(cfg.model.path, weights_only=False)

        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = cfg.train.lr

    return optimizer

def get_scheduler(
    cfg: Cfg,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.MultiStepLR:
    """Find, initialize and return a Torch scheduler."""

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=cfg.train.milestones,
        gamma=cfg.train.gamma
    )

    if cfg.exp.resume:
        ckpt = torch.load(cfg.model.path, weights_only=False)

        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])

    return scheduler

def get_step(
    cfg: Cfg
) -> int:
    if not cfg.exp.resume:
        return 0
    
    ckpt = torch.load(cfg.model.path, weights_only=False)
    s = ckpt.get("step")

    if not s:
        return 0

    return s


def _mod_device_dtype(mod):
    p = next((p for p in mod.parameters() if p is not None), None)
    dev = p.device if p is not None else torch.device("cpu")
    dt = p.dtype if p is not None else torch.float32
    return dev, dt


@torch.no_grad()
def format_batch_on_device(
    batch: dict,
    dvae: DiscreteVAE,
    torch_mel_spectrogram_dvae: TorchMelSpectrogram,
    torch_mel_spectrogram_style_encoder: TorchMelSpectrogram,
):
    """
    Вычисляем:
      - cond_mels: [B, num_cond, n_mel, Tm]
      - audio_codes: DVAE codebook indices

    Совместимость:
      - Используем batch["conditioning"] формы [B, num_cond(=1), 1, T]
      - Дублируем "оригинальные" ключи text_inputs/text_lengths, чтобы код модели мог их читать.
    """
    # device/dtype DVAE
    dv_dev, dv_dt = _mod_device_dtype(dvae)
    if hasattr(torch_mel_spectrogram_dvae, "to"):
        torch_mel_spectrogram_dvae = torch_mel_spectrogram_dvae.to(dv_dev)
    if hasattr(torch_mel_spectrogram_style_encoder, "to"):
        torch_mel_spectrogram_style_encoder = torch_mel_spectrogram_style_encoder.to(dv_dev)

    # ------ conditioning → cond_mels ------
    cond = batch["conditioning"]  # [B, num_cond, 1, T]
    B, num_cond, C, T = cond.shape
    cond_reshaped = cond.view(B * num_cond, C, T).to(dv_dev, dtype=torch.float32)
    paired_conditioning_mel = torch_mel_spectrogram_style_encoder(cond_reshaped)  # [B*num_cond, n_mel, Tm]
    n_mel = torch_mel_spectrogram_style_encoder.n_mel_channels
    T_mel = paired_conditioning_mel.size(2)
    paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond, n_mel, T_mel)
    batch["cond_mels"] = paired_conditioning_mel  # оставляем на dv_dev

    # ------ DVAE mel + code indices ------
    wav = batch["wav"].to(dv_dev, dtype=torch.float32)  # [B, 1, Tw]
    dvae_mel_spec = torch_mel_spectrogram_dvae(wav)     # dtype float32 → приведём к dv_dt
    dvae_mel_spec = dvae_mel_spec.to(dv_dev, dtype=dv_dt)
    codes = dvae.get_codebook_indices(dvae_mel_spec)
    batch["audio_codes"] = codes

    # ------ совместимость ключей ------
    # Оригинал ждёт text_inputs/text_lengths; «твой» код читает text/text_len.
    batch["text_inputs"]  = batch.get("padded_text") if "padded_text" in batch else batch["text"]
    batch["text_lengths"] = batch.get("text_lengths") if "text_lengths" in batch else batch["text_len"]

    # Очистим лишнее тяжёлое:
    for k in ("padded_text", "wav", "conditioning"):
        if k in batch:
            del batch[k]

    return batch

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    xtts: Xtts,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
    step: int,
    cfg: Cfg
) -> None:    
    state = {
        "model": xtts.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
    }

    ckp_dir = cfg.exp.checkpoints_path / f"checkpoint_{step}"
    ckp_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, ckp_dir / "model.pth")

    # Copy necessary for model files
    shutil.copy(str(cfg.model.dvae), str(ckp_dir / cfg.model.dvae.name))
    shutil.copy(str(cfg.model.vocab), str(ckp_dir / cfg.model.vocab.name))
    shutil.copy(str(cfg.model.config), str(ckp_dir / cfg.model.config.name))
    shutil.copy(str(cfg.model.mel_stats), str(ckp_dir / cfg.model.mel_stats.name))
