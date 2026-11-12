"""Role/position fusion and pair-local, mask-aware window aggregation."""

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from model.config import ModelConfig
from model.labels import COARSE_LABELS, FINE_LABELS

LABEL_COUNT = len(COARSE_LABELS) + len(FINE_LABELS)


class WindowFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.projection = nn.Linear(cfg.native_dim, cfg.projection_dim)
        self.norm = nn.LayerNorm(cfg.projection_dim)
        self.role_embedding = nn.Embedding(3, cfg.projection_dim)
        self.position_projection = nn.Linear(4, cfg.projection_dim)

    def forward(self, summaries: Tensor, roles: Tensor, positions: Tensor) -> Tensor:
        """Return [N,D] features without detaching native encoder summaries."""
        if summaries.ndim != 2 or summaries.shape[1] != self.projection.in_features:
            raise ValueError("summaries must have shape [N,native_dim]")
        if roles.shape != (summaries.shape[0],) or roles.dtype != torch.int64:
            raise ValueError("roles must be int64 [N]")
        if roles.device != summaries.device or positions.device != summaries.device:
            raise ValueError("summary, role and position devices must match")
        if ((roles < 0) | (roles >= 3)).any():
            raise ValueError("hierarchical windows must use PRE, POST or DIFF roles")
        if positions.shape != (summaries.shape[0], 4) or not positions.is_floating_point():
            raise ValueError("positions must be floating [N,4]")
        # The backend may be BF16 while new task modules remain FP32. A
        # differentiable cast also works when inference has no autocast context.
        projected = self.norm(self.projection(summaries.to(self.projection.weight.dtype)))
        positional = self.position_projection(positions.to(self.position_projection.weight.dtype))
        return projected + self.role_embedding(roles) + positional


def pad_windows(features: Tensor, pair_index: Tensor, pair_count: int) -> tuple[Tensor, Tensor]:
    """Pack all [N,D] windows as [B,Wmax,D], with True for valid slots.

    Input windows may be interleaved; order within each pair is preserved.
    Every pair must contain a window, including source-side EMPTY placeholders.
    """
    if type(pair_count) is not int or pair_count < 1:
        raise ValueError("pair_count must be a positive integer")
    if features.ndim != 2 or pair_index.shape != (features.shape[0],):
        raise ValueError("features and pair_index must have [N,D] and [N] shapes")
    if pair_index.dtype != torch.int64 or pair_index.device != features.device:
        raise ValueError("pair_index must be int64 on the features device")
    if ((pair_index < 0) | (pair_index >= pair_count)).any():
        raise ValueError("window pair_index is outside the batch")
    counts = torch.bincount(pair_index, minlength=pair_count)
    if (counts == 0).any():
        raise ValueError("each pair must have at least one valid window")
    padded = pad_sequence([features[pair_index == i] for i in range(pair_count)], batch_first=True)
    valid = torch.arange(padded.shape[1], device=features.device)[None] < counts[:, None]
    return padded, valid


class WindowAggregator(nn.Module):
    """Return [B,8,D] label features from fused padded windows.

    Transformer uses pair-local self-attention followed by eight learned
    label queries. The mean ablation has no Transformer or query parameters;
    it supplies the same masked mean to each separate linear label readout.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.aggregation = cfg.aggregation
        self.dim = cfg.projection_dim
        if cfg.aggregation == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.projection_dim, nhead=cfg.heads, dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout, activation=cfg.activation, batch_first=True,
                norm_first=cfg.norm_first,
            )
            # Pre-norm is intentional; disable the nested-tensor optimization
            # that is unavailable for it, avoiding a misleading torch warning.
            self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.layers,
                                                     enable_nested_tensor=False)
            self.label_queries = nn.Parameter(torch.empty(LABEL_COUNT, self.dim))
            nn.init.normal_(self.label_queries, std=1 / math.sqrt(self.dim))

    def forward(self, windows: Tensor, valid_mask: Tensor) -> Tensor:
        if windows.ndim != 3 or windows.shape[2] != self.dim:
            raise ValueError("windows must have shape [B,W,projection_dim]")
        if (valid_mask.shape != windows.shape[:2] or valid_mask.dtype != torch.bool
                or valid_mask.device != windows.device):
            raise ValueError("valid_mask must be boolean [B,W] on the windows device")
        if windows.shape[0] == 0 or not valid_mask.any(dim=1).all():
            raise ValueError("each pair must have at least one valid window")
        # Zero invalid slots even if callers supplied NaNs or arbitrary values.
        masked = windows.masked_fill(~valid_mask[:, :, None], 0)
        if self.aggregation == "mean":
            mean = masked.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True)
            return mean[:, None].expand(-1, LABEL_COUNT, -1)
        contextual = self.transformer(masked, src_key_padding_mask=~valid_mask)
        scores = torch.einsum("bwd,ld->blw", contextual, self.label_queries) / math.sqrt(self.dim)
        scores = scores.masked_fill(~valid_mask[:, None], -torch.inf)
        attention = scores.softmax(dim=-1)
        # These weights summarize valid windows, not localization supervision.
        return torch.einsum("blw,bwd->bld", attention, contextual)
