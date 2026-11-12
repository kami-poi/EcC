"""One optimizer update over source pairs with globally normalized gradients."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from model.config import TrainConfig
from model.input.batch import collate_pairs
from model.input.types import PreparedPair
from model.models.base import PairModel

from .loss import compute_loss


@dataclass(frozen=True, slots=True)
class StepResult:
    total: float
    coarse: float
    fine: float
    n_pairs: int
    n_modify: int


def _execution(model: PairModel) -> tuple[torch.device, torch.dtype]:
    """Use loaded device and declared compute dtype, then parameter inference.

    Quantized HF encoders declare compute_dtype because storage parameters can
    be int8/FP32 while their requested compute dtype is BF16/FP16. Custom
    encoders may omit it and use their first floating parameter's dtype.
    """
    encoder = getattr(model, "encoder", model)
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("model has no parameters")
    dtype = getattr(encoder, "compute_dtype", None)
    if dtype is None:
        floating = next((p for p in encoder.parameters() if p.is_floating_point()), parameter)
        dtype = floating.dtype
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("training compute dtype must be FP32, BF16 or FP16")
    return parameter.device, dtype


def train_effective_batch(model: PairModel, pairs: Sequence[PreparedPair], optimizer: Optimizer,
                          cfg: TrainConfig, *, pad_token_id: int,
                          scaler: GradScaler | None = None) -> StepResult:
    """Accumulate microbatches, then unscale/clip/update once.

    The outer training loop owns the scheduler and the reusable FP16 scaler.
    It must advance its scheduler after a successful effective update (and
    account for GradScaler overflow skips). No gold metadata enters forward.
    """
    n_total = len(pairs)
    if not 0 < n_total <= cfg.effective_batch:
        raise ValueError("effective batch must contain between 1 and effective_batch pairs")
    if any(pair.record.targets is None for pair in pairs):
        raise ValueError("effective batch requires supervision for every pair")
    n_modify = sum(pair.record.targets.modify_mask for pair in pairs)
    device, dtype = _execution(model)
    if dtype == torch.float16 and (scaler is None or not scaler.is_enabled()):
        raise ValueError("FP16 training requires an enabled GradScaler")
    if dtype != torch.float16 and scaler is not None:
        raise ValueError("a scaler is only supported for FP16 training")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = coarse = fine = 0.0
    for start in range(0, n_total, cfg.microbatch):
        inputs, targets = collate_pairs(pairs[start:start + cfg.microbatch], pad_token_id)
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            parts = compute_loss(model(inputs), targets, n_total=n_total, n_modify=n_modify,
                                 lambda_fine=cfg.lambda_fine)
        if scaler is None:
            parts.total.backward()
        else:
            scaler.scale(parts.total).backward()
        total += parts.total.detach().item()
        coarse += parts.coarse.detach().item()
        fine += parts.fine.detach().item()
    if scaler is not None:
        scaler.unscale_(optimizer)
    clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.clip_norm)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return StepResult(total, coarse, fine, n_total, n_modify)
