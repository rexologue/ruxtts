import argparse

import csv
from typing import Optional

import numpy as np
from tqdm import tqdm
import soundfile as sf

import torch
from torch.utils.data import DataLoader

from core.xtts import Xtts
from core.generic_utils import load_fsspec
from core.gpt import GPT
from core.dvae import DiscreteVAE
from core.arch_utils import TorchMelSpectrogram

from config import load_cfg, load_xtts_cfg
from dataset import AudioDataset, collate_fn
from train_utils import (
    get_step, 
    get_optimizer,
    get_scheduler, 
    format_batch_on_device, 
    seed_all,
    save_checkpoint
)


def forward(
    model,  # GPT
    dvae,
    torch_mel_spectrogram_dvae,
    torch_mel_spectrogram_style_encoder,
    batch: dict,
    device,
):
    # Предобработка на девайсе
    batch = format_batch_on_device(
        batch,
        dvae,
        torch_mel_spectrogram_dvae,
        torch_mel_spectrogram_style_encoder,
    )

    # Достаём ключи (поддерживаем оба стиля)
    text_inputs = (batch["text_inputs"] if "text_inputs" in batch else batch["text"]).to(device)
    text_lengths = (batch["text_lengths"] if "text_lengths" in batch else batch["text_len"]).to(device)
    audio_codes = batch["audio_codes"].to(device)
    wav_lengths = (batch["wav_lengths"] if "wav_lengths" in batch else batch["wav_len"]).to(device)
    cond_mels = batch["cond_mels"].to(device)

    # ВАЖНО: передаём ровно один из cond_idxs/cond_lens
    cond_idxs: Optional[torch.Tensor] = batch.get("cond_idxs", None)
    cond_lens: Optional[torch.Tensor] = batch.get("cond_lens", None)

    if cond_idxs is not None:
        cond_idxs = cond_idxs.to(device)
        cond_lens = None
    elif cond_lens is not None:
        cond_lens = cond_lens.to(device)
        cond_idxs = None
    else:
        raise RuntimeError("Both cond_idxs and cond_lens are None — dataset/collate misconfigured.")

    losses = model(
        text_inputs=text_inputs,
        text_lengths=text_lengths,
        audio_codes=audio_codes,
        wav_lengths=wav_lengths,
        cond_mels=cond_mels,
        cond_idxs=cond_idxs,
        cond_lens=cond_lens,
    )
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    xtts_cfg = load_xtts_cfg(cfg.model.config)

    # reproducibility
    seed_all(cfg.train.seed)

    device = torch.device(cfg.train.device)

    # ---- init model pieces ----
    xtts = Xtts.init_from_config(xtts_cfg)
    xtts.load_checkpoint(xtts_cfg, checkpoint_dir=str(cfg.model.dir), eval=True)

    if xtts.gpt is None:
        raise RuntimeError("Cannot initialize XTTS GPT part")

    xtts.eval()                # остальное (энкодеры и т.п.) не тренируем
    xtts.to(device)

    # Подготавливаем рефернс аудио для валидации
    voice_settings = {
        "gpt_cond_len": xtts_cfg.gpt_cond_len,
        "gpt_cond_chunk_len": xtts_cfg.gpt_cond_chunk_len,
        "max_ref_length": xtts_cfg.max_ref_len,
        "sound_norm_refs": xtts_cfg.sound_norm_refs,
    }
    voice = xtts.clone_voice(speaker_wav=str(cfg.eval.ref_wav_path), **voice_settings)
    cond_gpt_latent = voice["gpt_conditioning_latents"]
    cond_speaker_embedding = voice["speaker_embedding"]

    model = xtts.gpt
    model.train()

    if xtts_cfg.model_args.gpt_use_perceiver_resampler:
        torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
            filter_length=2048,
            hop_length=256,
            win_length=1024,
            normalize=False,
            sampling_rate=xtts_cfg.audio.sample_rate,
            mel_fmin=0,
            mel_fmax=8000,
            n_mel_channels=80,
            mel_norm_file=str(cfg.model.mel_stats),
        )
    else:
        torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
            filter_length=4096,
            hop_length=1024,
            win_length=4096,
            normalize=False,
            sampling_rate=xtts_cfg.audio.sample_rate,
            mel_fmin=0,
            mel_fmax=8000,
            n_mel_channels=80,
            mel_norm_file=str(cfg.model.mel_stats),
        )

    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=xtts_cfg.model_args.gpt_num_audio_tokens - 2,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )
    dvae.eval().to(device)
    for p in dvae.parameters():
        p.requires_grad_(False)

    dvae_checkpoint = torch.load(cfg.model.dvae, map_location="cpu", weights_only=False)
    dvae.load_state_dict(dvae_checkpoint, strict=False)

    torch_mel_spectrogram_dvae = TorchMelSpectrogram(
        mel_norm_file=str(cfg.model.mel_stats),
        sampling_rate=xtts_cfg.audio.dvae_sample_rate,
    )
    # перенесём спеки на девайс
    torch_mel_spectrogram_dvae = torch_mel_spectrogram_dvae.to(device)
    torch_mel_spectrogram_style_encoder = torch_mel_spectrogram_style_encoder.to(device)

    # ---- optim & sched ----
    step = get_step(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    # ---- data ----
    ds = AudioDataset(
        cfg.data.metadata_csv,
        cfg.data.text_column,
        xtts.tokenizer,
        xtts_cfg,
    )
    dataloader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.train.num_workers > 0,
        drop_last=True,
    )

    # ---- train loop ----
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_steps = int(cfg.train.total_steps)
    grad_accum = max(1, int(cfg.train.grad_accum_steps))
    eval_every = int(cfg.train.eval_steps)

    pbar = tqdm(total=total_steps, initial=step, desc="train", dynamic_ncols=True)
    metrics_file = open(str(cfg.exp.metrics_path), "w", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=["step", "lt", "lm", "lr"], delimiter="|")
    metrics_writer.writeheader()

    acc_i = 0

    while step < total_steps:
        for batch in dataloader:
            # forward + loss (без AMP)
            loss_text, loss_mel, _ = forward(
                model,
                dvae,
                torch_mel_spectrogram_dvae,
                torch_mel_spectrogram_style_encoder,
                batch,
                device
            )

            loss_text_ce = loss_text * xtts_cfg.model_args.gpt_loss_text_ce_weight
            loss_mel_ce = loss_mel * xtts_cfg.model_args.gpt_loss_mel_ce_weight
            loss = (loss_text_ce + loss_mel_ce) / grad_accum

            # backward (с аккумулированием)
            loss.backward()
            acc_i += 1

            # шаг оптимизации каждые grad_accum итераций
            if acc_i == grad_accum:
                # grad clipping
                if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                metrics_dir = {
                    "step": step,
                    "lt": float(loss_text.detach().cpu()),
                    "lm": float(loss_mel.detach().cpu()),
                    "lr": float(scheduler.get_last_lr()[0]),
                }

                step += 1
                acc_i = 0

                pbar.update(1)
                pbar.set_postfix(metrics_dir)

                metrics_writer.writerow(metrics_dir)
                metrics_file.flush()

                # валидация/чекпоинт по расписанию
                if (eval_every > 0) and (step % eval_every == 0):
                    model.eval()
                    
                    # делаем чекпоинт
                    save_checkpoint(xtts, optimizer, scheduler, step, cfg)

                    # делаем сэмплы
                    samples_path = cfg.exp.eval_samples_path / f"samples_{step}"
                    samples_path.mkdir(parents=True, exist_ok=True)

                    with torch.no_grad():
                        for i, val_text in enumerate(cfg.eval.texts):
                            res = xtts.inference(
                                text=val_text["text"],
                                language=val_text["language"],
                                gpt_cond_latent=cond_gpt_latent,
                                speaker_embedding=cond_speaker_embedding,
                            )
                            wav = np.asarray(res["wav"], dtype=np.float32)
                            sf.write(samples_path / f"{i}.wav", wav, xtts_cfg.model_args.output_sample_rate)

                    model.train()

                if step >= total_steps:
                    break

    # Сохраняем финальный чекпоинт (-1 - знак конца)
    save_checkpoint(xtts, optimizer, scheduler, -1, cfg)

    pbar.close()
    metrics_file.close()


if __name__ == "__main__":
    main()
