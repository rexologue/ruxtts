from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Callable, TypeVar, TypeAlias

import fsspec
import torch
from packaging.version import Version

_T = TypeVar("_T")
_R = TypeVar("_R")

ValueListDict: TypeAlias = _T | list[_T] | dict[str, _T]

def exists(val: _T | None) -> bool:
    return val is not None


def default(val: _T | None, d: _T | Callable[[], _T]) -> _T:
    if exists(val):
        return val
    return d() if callable(d) else d  # type: ignore[return-value]


def is_pytorch_at_least_2_4() -> bool:
    return Version(torch.__version__) >= Version("2.4.0")


def warn_synthesize_config_deprecated() -> None:
    warnings.warn("Passing `config` to synthesize() is deprecated.", DeprecationWarning, stacklevel=2)


def warn_synthesize_speaker_id_deprecated() -> None:
    warnings.warn("Use `speaker` instead of `speaker_id`.", DeprecationWarning, stacklevel=2)


def load_fsspec(
    path: str | os.PathLike[Any],
    map_location: str | torch.device | dict[str, str] | None = None,
    *,
    cache: bool = True,  # kept for API compatibility
    **kwargs: Any,
) -> Any:
    path = Path(path)
    if path.exists():
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)

    with fsspec.open(str(path), "rb") as stream:
        return torch.load(stream, map_location=map_location, weights_only=False, **kwargs)

def map_value_list_dict(obj: ValueListDict[_T], fn: Callable[[_T], _R]) -> ValueListDict[_R]:
    """Apply `fn` to obj, list of obj, or dict of obj and return the same structure."""
    if isinstance(obj, list):
        return [fn(v) for v in obj]
    if isinstance(obj, dict):
        return {k: fn(v) for k, v in obj.items()}
    return fn(obj)