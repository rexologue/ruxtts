"""Extract VQ token sequences for a dataset of audio files using a pretrained DiscreteVAE."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import load_xtts_cfg
from core.arch_utils import TorchMelSpectrogram
from core.dvae import DiscreteVAE
from core.xtts import load_audio
from train_utils import seed_all


LOGGER = logging.getLogger("extract_vq_tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.csv")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing XTTS assets (config.json, dvae.pth, mel_stats.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to store the generated token chunks (*.npy)",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        required=True,
        help="Number of .npy files to split the output into",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of audio files to process at once",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (defaults to CUDA if available)",
    )
    parser.add_argument(
        "--resume-chunks",
        type=int,
        default=0,
        help="Number of already generated chunks to skip before resuming",
    )
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.metadata.is_file():
        raise FileNotFoundError(f"metadata file not found: {args.metadata}")

    if args.num_splits <= 0:
        raise ValueError("num-splits must be a positive integer")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be a positive integer")

    required = {
        "config.json": args.model_dir / "config.json",
        "dvae.pth": args.model_dir / "dvae.pth",
        "mel_stats.pth": args.model_dir / "mel_stats.pth",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts: " + ", ".join(f"{name} ({required[name]})" for name in missing)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)


def _select_device(explicit: str | None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _init_models(
    model_dir: Path, device: torch.device
) -> tuple[DiscreteVAE, TorchMelSpectrogram, int, torch.dtype]:
    cfg = load_xtts_cfg(model_dir / "config.json")

    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=int(cfg.model_args.gpt_num_audio_tokens) - 2,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )
    dvae.eval().to(device)
    for param in dvae.parameters():
        param.requires_grad_(False)

    param_example = next(dvae.parameters(), None)

    checkpoint = torch.load(model_dir / "dvae.pth", map_location="cpu", weights_only=False)
    dvae.load_state_dict(checkpoint, strict=False)

    torch_mel = TorchMelSpectrogram(
        mel_norm_file=str(model_dir / "mel_stats.pth"),
        sampling_rate=int(cfg.audio.dvae_sample_rate),
    ).to(device)

    dvae_sample_rate = int(cfg.audio.dvae_sample_rate)
    return dvae, torch_mel, dvae_sample_rate, param_example.dtype if param_example is not None else torch.float32


def _batched(iterable: Iterable, batch_size: int) -> Iterable[list]:
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    _validate_paths(args)
    device = _select_device(args.device)
    LOGGER.info("Using device: %s", device)

    seed_all(args.seed)

    dvae, mel_transform, sample_rate, dvae_dtype = _init_models(args.model_dir, device)

    df = pd.read_csv(args.metadata, sep="|", low_memory=False)
    if "audio_path" not in df.columns:
        raise ValueError("metadata.csv must contain an 'audio_path' column")

    audio_paths = df["audio_path"].astype(str).tolist()
    total_audios = len(audio_paths)
    if total_audios == 0:
        LOGGER.warning("No audio files found in metadata: %s", args.metadata)
        return

    if args.resume_chunks < 0:
        raise ValueError("resume-chunks must be a non-negative integer")

    chunk_size = math.ceil(total_audios / args.num_splits)
    LOGGER.info("Total audios: %d | Chunk size: %d", total_audios, chunk_size)

    skipped: list[str] = []
    current_chunk: list[dict[str, object]] = []
    resume_chunks = min(args.resume_chunks, args.num_splits)
    chunk_idx = resume_chunks
    skip_count = min(total_audios, resume_chunks * chunk_size)
    if skip_count:
        LOGGER.info(
            "Resuming from chunk %d; skipping %d audio entries from metadata.",
            resume_chunks,
            skip_count,
        )
        audio_paths = audio_paths[skip_count:]
    if not audio_paths:
        LOGGER.info("No new audio entries to process after applying resume offset.")
        return
    processed = 0

    def flush_chunk() -> None:
        nonlocal current_chunk, chunk_idx
        if not current_chunk:
            return
        if chunk_idx >= args.num_splits:
            LOGGER.warning("Reached maximum number of splits (%d); dropping %d remaining items.", args.num_splits, len(current_chunk))
            current_chunk = []
            return
        out_path = args.output_dir / f"tokens_part_{chunk_idx:03d}.npy"
        np.save(out_path, np.array(current_chunk, dtype=object), allow_pickle=True)
        LOGGER.info("Saved %d items to %s", len(current_chunk), out_path)
        current_chunk = []
        chunk_idx += 1

    metadata_root = args.metadata.parent

    total_batches = math.ceil(len(audio_paths) / args.batch_size)
    for batch_paths in tqdm(
        _batched(audio_paths, args.batch_size),
        total=total_batches,
        desc="Extracting tokens",
    ):
        wavs: list[torch.Tensor] = []
        valid_paths: list[str] = []
        for path_str in batch_paths:
            path = Path(path_str)
            if not path.is_absolute():
                path = (metadata_root / path).resolve()
            if not path.is_file():
                LOGGER.warning("Audio file missing: %s", path)
                skipped.append(path_str)
                continue
            try:
                wav = load_audio(str(path), sample_rate)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to load %s: %s", path, exc)
                skipped.append(path_str)
                continue
            if wav is None or wav.ndim != 2:
                LOGGER.warning("Unexpected audio tensor shape for %s", path)
                skipped.append(path_str)
                continue
            wavs.append(wav)
            valid_paths.append(path_str)

        if not wavs:
            continue

        max_len = max(wav.shape[-1] for wav in wavs)
        padded_wavs = [F.pad(wav, (0, max_len - wav.shape[-1])) if wav.shape[-1] < max_len else wav for wav in wavs]

        with torch.no_grad():
            batch_wav = torch.stack(padded_wavs).to(device, dtype=torch.float32)
            mel = mel_transform(batch_wav)
            mel = mel.to(device=device, dtype=dvae_dtype)
            codes = dvae.get_codebook_indices(mel)

        codes_np = codes.detach().cpu().numpy()
        for path_str, token_arr in zip(valid_paths, codes_np):
            current_chunk.append(
                {
                    "audio_path": path_str,
                    "tokens": np.asarray(token_arr, dtype=np.int64),
                }
            )
            processed += 1
            if len(current_chunk) >= chunk_size and chunk_idx < args.num_splits - 1:
                flush_chunk()

    if not current_chunk and chunk_idx == 0:
        LOGGER.warning("No tokens were generated. Check logs for skipped files.")
        return

    flush_chunk()

    if chunk_idx < args.num_splits:
        LOGGER.info("Requested %d splits but only wrote %d due to limited data.", args.num_splits, chunk_idx)

    LOGGER.info("Generated tokens for %d new audio files; skipped %d.", processed, len(skipped))
    if skip_count:
        LOGGER.info("Resume option skipped %d previously processed audio files.", skip_count)

    if skipped:
        LOGGER.info("Skipped %d files due to errors or missing data.", len(skipped))


if __name__ == "__main__":
    main()
