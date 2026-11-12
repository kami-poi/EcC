"""One PAIR summary, native projection and eight linear readouts."""
import torch
from torch import nn

from model.config import ModelConfig
from model.encode.base import WindowEncoder
from model.encode.checkpoint import encode_windows
from model.input.types import PairInputs, ROLE_IDS
from model.labels import COARSE_LABELS, FINE_LABELS
from .base import PairLogits, PairModel


class ConcatClassifier(PairModel):
    def __init__(self, encoder: WindowEncoder, cfg: ModelConfig, *,
                 window_microbatch: int = 1, window_checkpoint: bool = True):
        super().__init__()
        if cfg.architecture != "concat" or encoder.native_dim != cfg.native_dim:
            raise ValueError("concat architecture and matching native_dim are required")
        if type(window_microbatch) is not int or window_microbatch < 1:
            raise ValueError("window_microbatch must be a positive integer")
        if type(window_checkpoint) is not bool:
            raise ValueError("window_checkpoint must be boolean")
        self.encoder = encoder
        self.window_microbatch = window_microbatch
        self.window_checkpoint = window_checkpoint
        self.projection = nn.Linear(cfg.native_dim, cfg.projection_dim)
        self.heads = nn.ModuleList(nn.Linear(cfg.projection_dim, 1) for _ in (*COARSE_LABELS, *FINE_LABELS))
        parameter = next(encoder.parameters(), None)
        if parameter is not None:
            self.projection.to(device=parameter.device)
            self.heads.to(device=parameter.device)

    def forward(self, inputs: PairInputs) -> PairLogits:
        if inputs.side_presence.ndim != 2 or inputs.side_presence.shape[1] != 2:
            raise ValueError("side_presence must have shape [B,2]")
        batch = inputs.side_presence.shape[0]
        expected = torch.arange(batch, device=inputs.pair_index.device)
        if (not torch.equal(inputs.pair_index, expected) or len(inputs.input_ids) != batch
                or inputs.roles.shape != expected.shape or not torch.all(inputs.roles == ROLE_IDS["PAIR"])):
            raise ValueError("concat requires exactly one PAIR window per pair in batch order")
        native = encode_windows(self.encoder, inputs.input_ids, inputs.attention_mask,
                                microbatch=self.window_microbatch, checkpoint=self.window_checkpoint)
        features = self.projection(native.to(self.projection.weight.dtype))
        logits = torch.cat([head(features) for head in self.heads], dim=1)
        return PairLogits(logits[:, :3], logits[:, 3:])
