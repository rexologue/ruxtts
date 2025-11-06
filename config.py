from __future__ import annotations

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, fields

from core.xtts_config import XttsArgs, XttsAudioConfig, XttsConfig


@dataclass
class ModelCfg:
    dir: Path
    path: Path
    dvae: Path
    vocab: Path
    config: Path
    mel_stats: Path


@dataclass
class DataCfg:
    metadata_csv: Path
    text_column: str = "text"
    use_sampler: bool = False
    sampler_char_border: int = 0
    sampler_upsamle_prob: float = 0.5


@dataclass
class TrainCfg:
    device: str = "cuda:0"
    num_workers: int = 4
    seed: int = 42

    # Batch stuff
    batch_size: int = 16
    grad_accum_steps: int = 1

    # Steps stuff
    total_steps: int = 100000
    eval_steps: int = 2000

    # Optim (AdamW)
    lr: float = 5e-6
    betas: tuple[float, float] = (0.9, 0.96)
    weight_decay: float = 1e-2
    eps: float = 1e-8

    # Scheduler (MultiStepLR)
    milestones: list[int] = field(default_factory=lambda: [50_000, 150_000, 300_000])
    gamma: float = 0.1

    # General params
    grad_clip_norm: float = 1.0


@dataclass
class ExpCfg:
    exp_dir: Path
    checkpoints_path: Path
    eval_samples_path: Path
    metrics_path: Path
    resume: bool = False


@dataclass
class EvalCfg:
    texts: list[dict[str, str]]
    ref_wav_path: Path


@dataclass
class NeptuneCfg:
    project: str
    api_token: str | None = None
    experiment_name: str | None = None
    dependencies_path: Path | None = None
    run_id: str | None = None
    tags: list[str] | None = None
    env_path: Path | None = None


@dataclass
class Cfg:
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    exp: ExpCfg
    eval: EvalCfg
    neptune: NeptuneCfg | None = None


def load_cfg(path: str | Path) -> Cfg:
    # Load yaml config
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    # Extract dir
    model_dir = Path(y["model_dir"])

    if not model_dir.exists():
        raise ValueError(f"model_dir is not set or incorrect: {str(model_dir)}")

    # Init models
    model = ModelCfg(
        dir=model_dir,
        path=(model_dir / "model.pth"),
        dvae=(model_dir / "dvae.pth"),
        vocab=(model_dir / "vocab.json"),
        config=(model_dir / "config.json"),
        mel_stats=(model_dir / "mel_stats.pth")
    )

    # Check files
    if not model.path.exists():
        raise ValueError(f"Cannot find `model.pth` in {str(model_dir)}")
    if not model.dvae.exists():
        raise ValueError(f"Cannot find `dvae.pth` in {str(model_dir)}")
    if not model.vocab.exists():
        raise ValueError(f"Cannot find `vocab.json` in {str(model_dir)}")
    if not model.config.exists():
        raise ValueError(f"Cannot find `config.json` in {str(model_dir)}")
    if not model.mel_stats.exists():
        raise ValueError(f"Cannot find `mel_stats.pth` in {str(model_dir)}")

    # Init another configs
    data = DataCfg(**y["data"])
    train = TrainCfg(**y["train"])

    # Extract experiment dir
    exp_dir = Path(y["exp"]["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Experiment config
    exp = ExpCfg(
        exp_dir=exp_dir,
        checkpoints_path=(exp_dir / "checkpoints"),
        eval_samples_path=(exp_dir / "samples"),
        metrics_path=(exp_dir / "metrics.csv"),
        resume=bool(y["exp"]["resume"])
    )

    # Make dirs
    exp.eval_samples_path.mkdir(parents=True, exist_ok=True)
    exp.checkpoints_path.mkdir(parents=True, exist_ok=True)

    # Load evals texts
    with open(y["eval"]["texts_json_path"], "r", encoding="utf-8") as f:
        eval_list = json.load(f)

    # Eval config
    eval = EvalCfg(
        texts=eval_list,
        ref_wav_path=Path(y["eval"]["ref_wav_path"])
    )

    # Check reference audio file existence
    if not eval.ref_wav_path.exists():
        raise ValueError(f"Cannot find refernce audio file {str(eval.ref_wav_path)}")

    neptune_cfg = None
    if y.get("neptune"):
        neptune_raw = y["neptune"]
        neptune_cfg = NeptuneCfg(
            project=neptune_raw["project"],
            api_token=neptune_raw.get("api_token"),
            experiment_name=neptune_raw.get("experiment_name"),
            dependencies_path=Path(neptune_raw["dependencies_path"]) if neptune_raw.get("dependencies_path") else None,
            run_id=neptune_raw.get("run_id"),
            tags=neptune_raw.get("tags"),
            env_path=Path(neptune_raw["env_path"]) if neptune_raw.get("env_path") else None,
        )

    return Cfg(model=model, data=data, train=train, exp=exp, eval=eval, neptune=neptune_cfg)


def load_xtts_cfg(config_path: Path) -> XttsConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def _filter_kwargs(cls, data: dict) -> dict:
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in data.items() if k in allowed}

    model_args = raw.get("model_args", {})
    audio_args = raw.get("audio", {})
    config_kwargs = _filter_kwargs(XttsConfig, raw)

    config = XttsConfig(**config_kwargs)
    
    if isinstance(model_args, dict):
        config.model_args = XttsArgs(**_filter_kwargs(XttsArgs, model_args))

    if isinstance(audio_args, dict):
        config.audio = XttsAudioConfig(**_filter_kwargs(XttsAudioConfig, audio_args))

    return config
