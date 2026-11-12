"""Shared window encoder plus hierarchical three/five classification."""

import torch
from torch import nn

from model.config import ModelConfig, RunConfig
from model.encode.base import WindowEncoder
from model.encode.checkpoint import encode_windows
from model.input.types import PairInputs
from model.labels import COARSE_LABELS, FINE_LABELS

from .aggregate import WindowAggregator, WindowFusion, pad_windows
from .base import PairLogits, PairModel


class HierarchicalClassifier(PairModel):
    def __init__(self, encoder: WindowEncoder, cfg: ModelConfig, *,
                 window_microbatch: int = 1, window_checkpoint: bool = True):
        super().__init__()
        if cfg.architecture != "hierarchical":
            raise ValueError("HierarchicalClassifier cannot use concat architecture")
        if encoder.native_dim != cfg.native_dim:
            raise ValueError("encoder native_dim must match model native_dim")
        if type(window_microbatch) is not int or window_microbatch < 1:
            raise ValueError("window_microbatch must be a positive integer")
        if type(window_checkpoint) is not bool:
            raise ValueError("window_checkpoint must be boolean")
        self.encoder = encoder
        self.window_microbatch = window_microbatch
        self.window_checkpoint = window_checkpoint
        self.fusion = WindowFusion(cfg)
        self.aggregation = WindowAggregator(cfg)
        self.heads = nn.ModuleList(nn.Linear(cfg.projection_dim, 1)
                                  for _ in (*COARSE_LABELS, *FINE_LABELS))
        # Match the already-loaded encoder's device while preserving its dtype
        # and retaining FP32 task weights, including for quantized backends.
        parameter = next(encoder.parameters(), None)
        if parameter is not None:
            for module in (self.fusion, self.aggregation, self.heads):
                module.to(device=parameter.device)

    def forward(self, inputs: PairInputs) -> PairLogits:
        if inputs.side_presence.ndim != 2 or inputs.side_presence.shape[1] != 2:
            raise ValueError("side_presence must have shape [B,2]")
        native = encode_windows(self.encoder, inputs.input_ids, inputs.attention_mask,
                                microbatch=self.window_microbatch, checkpoint=self.window_checkpoint)
        fused = self.fusion(native, inputs.roles, inputs.positions)
        padded, valid = pad_windows(fused, inputs.pair_index, inputs.side_presence.shape[0])
        label_features = self.aggregation(padded, valid)
        logits = torch.cat([head(label_features[:, i]) for i, head in enumerate(self.heads)], dim=1)
        coarse_count = len(COARSE_LABELS)
        return PairLogits(logits[:, :coarse_count], logits[:, coarse_count:])


def build_model(encoder: WindowEncoder, cfg: RunConfig) -> PairModel:
    """Build the implemented architecture using the supplied shared backend."""
    if cfg.model.architecture == "concat":
        from .concat import ConcatClassifier
        return ConcatClassifier(encoder, cfg.model, window_microbatch=cfg.window.window_microbatch,
                                window_checkpoint=cfg.window.window_checkpoint)
    return HierarchicalClassifier(encoder, cfg.model, window_microbatch=cfg.window.window_microbatch,
                                 window_checkpoint=cfg.window.window_checkpoint)
