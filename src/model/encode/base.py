"""Swappable tokenizer/encoder bundle; loading policy belongs to the factory."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from torch import Tensor, nn

from model.config import RunConfig
from model.input.windows import TokenizerAdapter


class WindowEncoder(nn.Module, ABC):
    native_dim: int

    @abstractmethod
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode windows as [W, native_dim] summaries, preserving autograd."""

    @abstractmethod
    def adapter_state_dict(self) -> dict[str, Tensor]:
        """Return adapter weights only, never the frozen base model."""

    @abstractmethod
    def load_adapter_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore compatible adapter weights or reject incompatible state."""


@dataclass(frozen=True, slots=True)
class LoadedBackend:
    encoder: WindowEncoder
    tokenizer: TokenizerAdapter
    config: RunConfig


BackendFactory = Callable[[RunConfig], LoadedBackend]
