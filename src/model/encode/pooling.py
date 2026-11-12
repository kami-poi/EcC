"""Last valid token pooling independent of padding direction."""

import torch
from torch import Tensor


def last_valid_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError("hidden [W,L,D] and mask [W,L] shapes must agree")
    if hidden.device != mask.device:
        raise ValueError("hidden and mask must be on the same device")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only zero/one validity values")
    valid = mask.bool()
    if hidden.shape[1] == 0 or not valid.any(dim=1).all():
        raise ValueError("every window must have at least one valid token")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~valid, -1).amax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
