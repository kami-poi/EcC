"""Deterministic source/token fixtures shared by model tests."""

from collections.abc import Sequence

from model.data.schema import PairRecord
from model.labels import encode_labels, normalize_code

import torch
from torch import nn

from model.encode.base import LoadedBackend, WindowEncoder
from model.encode.pooling import last_valid_pool


class FakeTokenizer:
    """One token per Python character; wrapper adds explicit BOS/EOS."""

    pad_token_id = 0
    special_token_rule = "BOS=1; EOS=2"

    def tokenize(self, text: str):
        return tuple(ord(char) + 10 for char in text), tuple((i, i + 1) for i in range(len(text)))

    def wrap_ids(self, content_ids: Sequence[int], role: str):
        prefix, _ = self.tokenize(
            "Instruct: Represent this source-code window for exception-handling change classification.\n"
            f"Query: {role}\n"
        )
        suffix, _ = self.tokenize("\nSUMMARY")
        return (1, *prefix, *content_ids, *suffix, 2)


class FakeWindowEncoder(WindowEncoder):
    """Reconstructible frozen hashed embeddings; no Unicode-sized vocabulary."""

    def __init__(self, native_dim: int = 8, *, adaptation: str = "lora", dropout: float = 0.2):
        super().__init__()
        self.native_dim = native_dim
        weights = torch.sin(torch.arange(257 * native_dim, dtype=torch.float32).reshape(257, native_dim) / 13)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        self.adapter = nn.Linear(native_dim, native_dim, bias=False) if adaptation == "lora" else None
        if self.adapter is not None:
            with torch.no_grad():
                self.adapter.weight.copy_(torch.eye(native_dim) / 10)
        self.dropout = nn.Dropout(dropout)
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.embedding.eval()
        if self.adapter is None:
            self.dropout.eval()
        return self

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids.remainder(257))
        if self.adapter is not None:
            hidden = hidden + self.adapter(self.dropout(hidden))
        return last_valid_pool(hidden, attention_mask)

    def adapter_state_dict(self):
        return {} if self.adapter is None else {key: value.detach().clone() for key, value in self.adapter.state_dict().items()}

    def load_adapter_state_dict(self, state):
        if self.adapter is None:
            if state:
                raise ValueError("frozen encoder has no adapter")
        else:
            self.adapter.load_state_dict(state, strict=True)


def fake_backend_factory(cfg):
    """Inject this callable at backend boundaries; it never contacts HF."""
    from dataclasses import replace

    revision = "0" * 40
    config = replace(cfg, encoder=replace(cfg.encoder, revision=revision, tokenizer_revision=revision))
    return LoadedBackend(FakeWindowEncoder(cfg.model.native_dim, adaptation=cfg.encoder.adaptation,
                                           dropout=cfg.encoder.lora_dropout), FakeTokenizer(), config)


def make_record(pre: str | None, post: str | None, *, labels: Sequence[str],
                subtypes: Sequence[str] = ()) -> PairRecord:
    return PairRecord("fixture", pre, post,
                      encode_labels(labels, subtypes, has_pre=normalize_code(pre) is not None,
                                    has_post=normalize_code(post) is not None),
                      "fixture-repo", "fixture-commit", "fixture-source", "C#")
