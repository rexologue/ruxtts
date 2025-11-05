"""Utilities for experiment logging backends."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

try:  # pragma: no cover - optional dependency
    import neptune
    from neptune.utils import stringify_unsupported
except ImportError:  # pragma: no cover - optional dependency
    neptune = None  # type: ignore
    stringify_unsupported = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
    USE_DOT_ENV = True
except ImportError:  # pragma: no cover - optional dependency
    USE_DOT_ENV = False


class BaseLogger(ABC):
    """A base experiment logger class."""

    @abstractmethod
    def __init__(self, config: Any):
        """Logs git commit id, dvc hash, environment."""

    @abstractmethod
    def log_hyperparameters(self, params: dict[str, Any]):
        """Persist experiment hyperparameters."""

    @abstractmethod
    def save_metrics(
        self,
        type_set: str,
        metric_name: Union[list[str], str],
        metric_value: Union[list[float], float],
        step: Optional[int] = None,
    ) -> None:
        """Save metrics to the underlying logger."""

    @abstractmethod
    def save_plot(self, *args: Any, **kwargs: Any) -> None:
        """Save visual artefacts such as plots."""

    @abstractmethod
    def add_tag(self, tag: str) -> None:
        """Attach a tag to the current run."""

    @abstractmethod
    def stop(self) -> None:
        """Finalize logger resources."""


class NeptuneLogger(BaseLogger):
    """A neptune.ai experiment logger class."""

    def __init__(self, config: Any):
        if neptune is None or stringify_unsupported is None:  # pragma: no cover - guarded import
            raise ImportError(
                "Neptune logger requested but the `neptune` package is not installed."
            )

        env_path = getattr(config, "env_path", None)
        if env_path and USE_DOT_ENV:
            load_dotenv(str(env_path))

        api_token = getattr(config, "api_token", None) or os.environ.get("NEPTUNE_API_TOKEN")
        if api_token:
            os.environ.setdefault("NEPTUNE_API_TOKEN", api_token)

        dependencies = getattr(config, "dependencies_path", None)
        if isinstance(dependencies, Path):
            dependencies = str(dependencies)

        run_kwargs = {
            "project": getattr(config, "project", None),
            "api_token": os.environ.get("NEPTUNE_API_TOKEN"),
            "name": getattr(config, "experiment_name", None),
            "dependencies": dependencies,
            "with_id": getattr(config, "run_id", None),
            "tags": getattr(config, "tags", None),
        }
        run_kwargs = {k: v for k, v in run_kwargs.items() if v is not None}
        self.run = neptune.init_run(**run_kwargs)

    def _prepare_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert dataclasses into plain dictionaries."""

        def _convert(value: Any) -> Any:
            if is_dataclass(value):
                return {k: _convert(v) for k, v in asdict(value).items()}
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            return value

        return {k: _convert(v) for k, v in params.items()}

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Model hyperparameters logging."""
        prepared = self._prepare_params(params)
        self.run["hyperparameters"] = stringify_unsupported(prepared)

    def save_metrics(
        self,
        type_set: str,
        metric_name: Union[list[str], str],
        metric_value: Union[list[float], float],
        step: Optional[int] = None,
    ) -> None:
        if isinstance(metric_name, list):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f"{type_set}/{p_n}"].log(p_v, step=step)
        else:
            self.run[f"{type_set}/{metric_name}"].log(metric_value, step=step)

    def save_plot(self, type_set: str, plot_name: str, plt_fig: Any) -> None:
        self.run[f"{type_set}/{plot_name}"].append(plt_fig)

    def add_tag(self, tag: str) -> None:
        self.run["sys/tags"].add(tag)

    def stop(self) -> None:
        self.run.stop()


def filter_metrics(metrics: dict[str, Optional[float]]) -> dict[str, float]:
    """Drop metrics with missing values."""

    return {k: float(v) for k, v in metrics.items() if v is not None}
