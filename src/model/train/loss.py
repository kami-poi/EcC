"""Pair-level joint supervision normalized across an entire effective batch."""

import math
from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from model.input.types import TargetsBatch
from model.models.base import PairLogits


@dataclass(frozen=True, slots=True)
class LossParts:
    total: Tensor
    coarse: Tensor
    fine: Tensor


def compute_loss(logits: PairLogits, targets: TargetsBatch, *, n_total: int,
                 n_modify: int, lambda_fine: float = 1.0) -> LossParts:
    """Return this microbatch's contribution using global pair denominators.

    Label dimensions are averaged before summing pairs. ``fine`` is unweighted;
    only ``total`` applies lambda_fine. Never gate supervision by predictions.
    """
    if type(n_total) is not int or n_total < 1:
        raise ValueError("n_total must be a positive integer")
    if type(n_modify) is not int or not 0 <= n_modify <= n_total:
        raise ValueError("n_modify must be an integer between zero and n_total")
    if targets.coarse.shape[0] > n_total or int(targets.modify_mask.sum()) > n_modify:
        raise ValueError("global denominators cannot be smaller than the microbatch counts")
    if not math.isfinite(lambda_fine) or lambda_fine < 0:
        raise ValueError("lambda_fine must be finite and nonnegative")
    # FP32 reductions keep half-precision logits numerically stable while the
    # cast retains gradients to mixed-precision model parameters.
    coarse = F.binary_cross_entropy_with_logits(logits.coarse.float(), targets.coarse.float(),
                                                reduction="none").mean(dim=1).sum() / n_total
    selected = targets.modify_mask
    if selected.any():
        fine = F.binary_cross_entropy_with_logits(logits.fine[selected].float(), targets.fine[selected].float(),
                                                  reduction="none").mean(dim=1).sum() / n_modify
    else:
        # All fine heads participate even when this microbatch has no Modify.
        fine = logits.fine.float().sum() * 0
    return LossParts(coarse + lambda_fine * fine, coarse, fine)
