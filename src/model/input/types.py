"""Source mappings for the language-neutral textual diff."""

from dataclasses import dataclass
from typing import Literal

import torch

from model.data.schema import PairRecord

Role = Literal["PRE", "POST", "DIFF", "PAIR"]
# Stable embedding indices, including the reserved concat role.
ROLE_IDS = {"PRE": 0, "POST": 1, "DIFF": 2, "PAIR": 3}


@dataclass(frozen=True)
class DiffSpan:
    """One serialized source line, with zero-based, half-open ranges.

    ``diff_range`` includes the kind prefix and any framing newline.
    ``content_range`` selects only the exact source line, including its original
    line ending if present. Character offsets index Python strings, not bytes.
    An absent source side has a zero-width line range at its alignment boundary.
    """

    diff_range: tuple[int, int]
    pre_lines: tuple[int, int]
    post_lines: tuple[int, int]
    kind: Literal["ADD", "DEL", "CTX"]
    content_range: tuple[int, int]


@dataclass(frozen=True)
class DiffDocument:
    """Serialized hunks and mappings for every included source line.

    Hunk headers, footers and the NO_TEXT_CHANGE marker have no source spans.
    """

    text: str
    spans: tuple[DiffSpan, ...]
    hunk_count: int


@dataclass(frozen=True)
class Window:
    """An exact token slice, wrapped without decoding/reencoding.

    content_range indexes unwrapped role tokens. character_range indexes the
    original role text in Python code points; for is_placeholder=True both
    ranges instead index the synthetic EMPTY text. text is that character
    slice, not a reconstruction of the ids. Overlapping tokenizer offsets may
    therefore make adjacent windows' text slices overlap as well.
    """

    input_ids: tuple[int, ...]
    role: Role
    content_range: tuple[int, int]
    pre_lines: tuple[int, int]
    post_lines: tuple[int, int]
    position: tuple[float, float, float, float]
    character_range: tuple[int, int]
    text: str
    prefix: str
    suffix: str
    special_token_rule: str
    is_placeholder: bool = False


@dataclass(frozen=True)
class PreparedPair:
    """Preparation/audit data; only collated PairInputs enter the model.

    token_counts count original PRE/POST and serialized DIFF content tokens,
    before wrappers or overlap, including ablated roles; missing sides count 0.
    coverage is unique token coverage per PRE/POST/DIFF: None means ablated,
    1.0 for an enabled absent side means EMPTY is fully represented.
    change_positions contains normalized (pre_start, pre_end, post_start,
    post_end) for every ADD/DEL span in serialized order, excluding CTX.
    """

    record: PairRecord
    windows: tuple[Window, ...]
    token_counts: tuple[int, int, int]
    coverage: tuple[float | None, float | None, float | None]
    change_positions: tuple[tuple[float, float, float, float], ...]
    source_lengths: tuple[int, int]
    input_mode: str
    diff: DiffDocument


@dataclass(frozen=True)
class PairInputs:
    """Source-derived tensors only. N windows, B pairs, padded length L.

    int64: input_ids [N,L], roles/pair_index [N]; bool: attention_mask [N,L],
    side_presence [B,2]; float32: positions [N,4]. Collation creates CPU
    tensors; to(device) preserves layouts/dtypes and returns a new instance.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    roles: torch.Tensor
    positions: torch.Tensor
    pair_index: torch.Tensor
    side_presence: torch.Tensor

    def to(self, device: str | torch.device) -> "PairInputs":
        return PairInputs(**{name: value.to(device) for name, value in vars(self).items()})


@dataclass(frozen=True)
class TargetsBatch:
    """float32 coarse [B,3]/fine [B,5]; bool modify_mask [B]."""

    coarse: torch.Tensor
    fine: torch.Tensor
    modify_mask: torch.Tensor

    def to(self, device: str | torch.device) -> "TargetsBatch":
        return TargetsBatch(**{name: value.to(device) for name, value in vars(self).items()})
