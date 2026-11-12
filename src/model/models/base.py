"""Source-only pair classification interfaces; decoding belongs elsewhere."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn

from model.input.types import PairInputs


@dataclass(frozen=True, slots=True)
class PairLogits:
    """Raw coarse [B,3] and fine [B,5] logits in canonical label order."""

    coarse: Tensor
    fine: Tensor


class PairModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: PairInputs) -> PairLogits:
        """Predict from source tensors without targets, gating or thresholds."""
