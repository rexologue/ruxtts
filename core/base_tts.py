import torch
from torch import nn
from coqpit import Coqpit


class BaseTTS(nn.Module):
    """Minimal base class providing device bookkeeping for inference-only models."""

    def __init__(self, config: Coqpit):
        super().__init__()
        self.config = config
        self.args = getattr(config, "model_args", config)
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        self._device = self.device
        return module

    def cpu(self):  # type: ignore[override]
        return self.to(torch.device("cpu"))

    def cuda(self, device: int | None = None):  # type: ignore[override]
        target = torch.device("cuda" if device is None else f"cuda:{device}")
        return self.to(target)

    def clone_voice(self, *args, **kwargs):
        voice, _ = self._clone_voice(*args, **kwargs)
        return voice

    def _clone_voice(self, *args, **kwargs):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError
