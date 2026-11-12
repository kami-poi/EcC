"""AdamW parameter ownership and linear update-count scheduling."""

import math

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from model.config import TrainConfig
from model.models.base import PairModel


def build_optimizer(model: PairModel, cfg: TrainConfig) -> Optimizer:
    """Group trainable encoder adapters and task modules with configured decay."""
    encoder, task = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if name.startswith("encoder.") else task).append(parameter)
    groups = []
    for parameters, lr in ((encoder, cfg.lora_lr), (task, cfg.task_lr)):
        if parameters:
            groups.append({"params": parameters, "lr": lr})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return AdamW(groups, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: Optimizer, *, total_steps: int, warmup_ratio: float) -> LRScheduler:
    """Build linear warmup/decay; the caller steps after successful updates.

    Warmup uses floor(total_steps * warmup_ratio), matching integer update
    counts; small runs may have no warmup. Rates remain zero past total_steps.
    """
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be finite and in [0,1)")
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    return LambdaLR(optimizer, multiplier)
