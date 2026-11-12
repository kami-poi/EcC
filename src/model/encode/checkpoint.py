"""Microbatch pooled summaries with differentiable activation recomputation."""

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .base import WindowEncoder


def encode_windows(encoder: WindowEncoder, ids: Tensor, mask: Tensor, *,
                   microbatch: int = 1, checkpoint: bool = True) -> Tensor:
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    if type(checkpoint) is not bool:
        raise ValueError("checkpoint must be boolean")
    if ids.ndim != 2 or mask.shape != ids.shape or ids.device != mask.device:
        raise ValueError("window ids and mask must have matching [W,L] shapes/devices")
    if ids.shape[0] == 0:
        parameter = next(encoder.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
        return torch.empty((0, encoder.native_dim), device=ids.device, dtype=dtype)
    recompute = checkpoint and torch.is_grad_enabled() and any(p.requires_grad for p in encoder.parameters())
    outputs = []
    for start in range(0, ids.shape[0], microbatch):
        batch_ids, batch_mask = ids[start:start + microbatch], mask[start:start + microbatch]
        if recompute:
            # Non-reentrant mode tracks trainable parameters even when token
            # IDs/masks and frozen embeddings do not require gradients.
            output = activation_checkpoint(encoder, batch_ids, batch_mask,
                                           use_reentrant=False, preserve_rng_state=True)
        else:
            output = encoder(batch_ids, batch_mask)
        outputs.append(output)
    return torch.cat(outputs, dim=0)
