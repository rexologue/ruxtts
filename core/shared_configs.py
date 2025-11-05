from dataclasses import dataclass
from typing import Any

from coqpit import Coqpit


@dataclass
class BaseTTSConfig(Coqpit):
    """Light-weight configuration container shared by inference components."""

    model: str = "tts"
    description: str | None = None
    model_args: Any = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
