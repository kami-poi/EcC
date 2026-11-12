"""Full source/diff coverage with tokenizer-aware total window budgets."""

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import Protocol

from model.config import RunConfig, WindowConfig
from model.data.schema import PairRecord
from model.errors import ConfigError, InputError
from model.labels import normalize_code

from .diff import make_diff
from .types import DiffDocument, PreparedPair, Role, Window

PREFIX_TEMPLATE = (
    "Instruct: Represent this source-code window for exception-handling change classification.\n"
    "Query: {ROLE}\n"
)
SUFFIX = "\nSUMMARY"
EMPTY = "EMPTY"


class TokenizerAdapter(Protocol):
    pad_token_id: int

    def tokenize(self, text: str) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]: ...

    def wrap_ids(self, content_ids: Sequence[int], role: str) -> tuple[int, ...]: ...


class HFTokenizerAdapter:
    """Adapt a loaded HF fast tokenizer; never load/download a backend here.

    Prefix/content/suffix are encoded separately with no special tokens. Fixed
    non-source probes verify a continuous single-sequence edge-special template
    through the public tokenizer call and its special_tokens_mask. Wrapping
    then concatenates that exact template around the existing content IDs.
    Offsets must be available; no decode fallback or truncation is permitted.
    """

    def __init__(self, tokenizer):
        if not getattr(tokenizer, "is_fast", False):
            raise ConfigError("source windows require a fast tokenizer with character offsets")
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ConfigError("tokenizer must define pad_token_id")
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self._special_prefix, self._special_suffix = self._special_template()
        self.special_token_rule = (
            f"edge_special_prefix={self._special_prefix!r}; edge_special_suffix={self._special_suffix!r}; "
            "edge_special_prefix + prefix_ids + content_ids + suffix_ids + edge_special_suffix; "
            f"special_tokens_map={getattr(tokenizer, 'special_tokens_map', {})!r}"
        )
        self._prefix_ids: dict[str, tuple[int, ...]] = {}
        self._suffix_ids, _ = self.tokenize(SUFFIX)

    def _special_template(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        template = None
        # The first probe is a single ordinary ASCII token on the supported
        # backends; the second exercises multiple tokens and punctuation.
        for probe in ("a", "tokenizer wrapping probe. 0123456789"):
            plain, _ = self.tokenize(probe)
            try:
                encoded = self.tokenizer(probe, add_special_tokens=True,
                                         return_special_tokens_mask=True,
                                         truncation=False, padding=False)
                ids, mask = tuple(encoded["input_ids"]), tuple(encoded["special_tokens_mask"])
            except (KeyError, NotImplementedError, TypeError, ValueError) as exc:
                raise ConfigError("tokenizer must expose its single-sequence special template") from exc
            if len(ids) != len(mask) or any(value not in (0, 1) for value in mask) or 0 not in mask:
                raise ConfigError("unsupported tokenizer special template mask")
            first = mask.index(0)
            last = len(mask) - mask[::-1].index(0)
            if any(mask[first:last]) or ids[first:last] != plain:
                raise ConfigError("tokenizer special template must preserve continuous content IDs")
            edges = ids[:first], ids[last:]
            if template is not None and edges != template:
                raise ConfigError("tokenizer special template must have content-independent edge tokens")
            template = edges
        return template

    def tokenize(self, text: str) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        try:
            encoded = self.tokenizer(text, add_special_tokens=False,
                                     return_offsets_mapping=True, truncation=False)
            ids = tuple(encoded["input_ids"])
            offsets = tuple(tuple(offset) for offset in encoded["offset_mapping"])
        except (KeyError, NotImplementedError, TypeError, ValueError) as exc:
            raise ConfigError("tokenizer must support character offsets without truncation") from exc
        _validate_offsets(text, ids, offsets)
        return ids, offsets

    def wrap_ids(self, content_ids: Sequence[int], role: str) -> tuple[int, ...]:
        if role not in self._prefix_ids:
            self._prefix_ids[role], _ = self.tokenize(PREFIX_TEMPLATE.format(ROLE=role))
        return (*self._special_prefix, *self._prefix_ids[role], *content_ids,
                *self._suffix_ids, *self._special_suffix)


def _validate_offsets(text, ids, offsets) -> None:
    if len(ids) != len(offsets) or (text and not ids):
        raise ConfigError("tokenizer must provide one character offset per content token")
    previous = 0
    for offset in offsets:
        if (len(offset) != 2 or any(type(value) is not int for value in offset)
                or not 0 <= offset[0] <= offset[1] <= len(text) or offset[0] < previous):
            raise ConfigError("tokenizer offsets must be ordered Python character ranges")
        previous = offset[0]


def _roles(cfg: WindowConfig) -> tuple[Role, ...]:
    if cfg.input_mode == "full_only":
        return ("PRE", "POST")
    if cfg.input_mode == "diff_only":
        return ("DIFF",)
    return ("PRE", "POST", "DIFF")


def _content_budget(tokenizer: TokenizerAdapter, cfg: WindowConfig, role: str) -> int:
    budget = cfg.window_tokens - len(tokenizer.wrap_ids((), role))
    if budget <= 0:
        raise ConfigError(f"{role} wrapper exhausts the complete window budget")
    if cfg.overlap_tokens >= budget:
        raise ConfigError(f"{role} overlap must be smaller than its content budget ({budget})")
    return budget


def validate_tokenizer_budget(tokenizer: TokenizerAdapter, cfg: WindowConfig) -> None:
    """Fail before preparing content if any enabled role cannot advance."""
    for role in _roles(cfg):
        _content_budget(tokenizer, cfg, role)


def _line_ends(text: str) -> tuple[int, ...]:
    ends = []
    offset = 0
    for line in text.splitlines(keepends=True):
        offset += len(line)
        ends.append(offset)
    return tuple(ends)


def _line_range(ends: tuple[int, ...], start: int, end: int) -> tuple[int, int]:
    if start == end:
        boundary = bisect_right(ends, start)
        return boundary, boundary
    return bisect_right(ends, start), bisect_left(ends, end) + 1


def _position(pre, post, totals) -> tuple[float, float, float, float]:
    return (pre[0] / max(totals[0], 1), pre[1] / max(totals[0], 1),
            post[0] / max(totals[1], 1), post[1] / max(totals[1], 1))


def _diff_lines(diff: DiffDocument, span_ends, start, end):
    pre, post = [], []
    # Map source records, including their prefixes/framing; structural-only
    # windows have no source extent. Never parse marker-like source text.
    for index in range(bisect_right(span_ends, start), len(diff.spans)):
        span = diff.spans[index]
        if span.diff_range[0] >= end:
            break
        pre.append(span.pre_lines)
        post.append(span.post_lines)
    if not pre:
        return (0, 0), (0, 0)
    return (min(r[0] for r in pre), max(r[1] for r in pre)), (
        min(r[0] for r in post), max(r[1] for r in post))


def prepare_pair(record: PairRecord, tokenizer: TokenizerAdapter, cfg: WindowConfig) -> PreparedPair:
    validate_tokenizer_budget(tokenizer, cfg)
    pre, post = normalize_code(record.pre_code), normalize_code(record.post_code)
    if pre is None and post is None:
        raise InputError("both source-code sides are absent")
    # Preserve raw input text for the diff, including whitespace-only sides.
    diff = make_diff(record.pre_code, record.post_code, context_lines=cfg.diff_context_lines)
    texts = {"PRE": pre or EMPTY, "POST": post or EMPTY, "DIFF": diff.text}
    encoded = {}
    for role, text in texts.items():
        ids, offsets = tokenizer.tokenize(text)
        _validate_offsets(text, ids, offsets)
        encoded[role] = ids, offsets
    line_ends = _line_ends(record.pre_code or ""), _line_ends(record.post_code or "")
    totals = len(line_ends[0]), len(line_ends[1])
    span_ends = tuple(span.diff_range[1] for span in diff.spans)
    windows = []
    coverage: list[float | None] = [None, None, None]
    for role in _roles(cfg):
        ids, offsets = encoded[role]
        budget = _content_budget(tokenizer, cfg, role)
        step = budget - cfg.overlap_tokens
        covered_end = 0
        for start in range(0, len(ids), step):
            end = min(start + budget, len(ids))
            wrapped = tokenizer.wrap_ids(ids[start:end], role)
            if len(wrapped) > cfg.window_tokens:
                raise ConfigError(f"{role} wrapper exceeds total budget for an actual content slice")
            char_start = min(offset[0] for offset in offsets[start:end])
            char_end = max(offset[1] for offset in offsets[start:end])
            placeholder = (role == "PRE" and pre is None) or (role == "POST" and post is None)
            pre_lines, post_lines = (0, 0), (0, 0)
            if role == "PRE" and not placeholder:
                pre_lines = _line_range(line_ends[0], char_start, char_end)
            elif role == "POST" and not placeholder:
                post_lines = _line_range(line_ends[1], char_start, char_end)
            elif role == "DIFF":
                pre_lines, post_lines = _diff_lines(diff, span_ends, char_start, char_end)
            windows.append(Window(
                input_ids=wrapped, role=role, content_range=(start, end),
                pre_lines=pre_lines, post_lines=post_lines,
                position=_position(pre_lines, post_lines, totals),
                character_range=(char_start, char_end), text=texts[role][char_start:char_end],
                prefix=PREFIX_TEMPLATE.format(ROLE=role), suffix=SUFFIX,
                special_token_rule=getattr(tokenizer, "special_token_rule",
                                          f"{type(tokenizer).__name__}.wrap_ids(content_ids, role)"),
                is_placeholder=placeholder,
            ))
            covered_end = end
            if end == len(ids):
                break
        coverage[("PRE", "POST", "DIFF").index(role)] = covered_end / len(ids)
    return PreparedPair(
        record=record, windows=tuple(windows),
        token_counts=(len(encoded["PRE"][0]) if pre is not None else 0,
                      len(encoded["POST"][0]) if post is not None else 0,
                      len(encoded["DIFF"][0])),
        coverage=tuple(coverage),
        change_positions=tuple(_position(span.pre_lines, span.post_lines, totals)
                               for span in diff.spans if span.kind != "CTX"),
        source_lengths=(len(record.pre_code or ""), len(record.post_code or "")),
        input_mode=cfg.input_mode, diff=diff,
    )


def prepare_model_input(record: PairRecord, tokenizer: TokenizerAdapter, cfg: RunConfig) -> PreparedPair:
    """Shared architecture dispatch for training, evaluation and prediction."""
    if cfg.model.architecture == "concat":
        return prepare_concat_pair(record, tokenizer, cfg.window)
    return prepare_pair(record, tokenizer, cfg.window)


def prepare_concat_pair(record: PairRecord, tokenizer: TokenizerAdapter, cfg: WindowConfig) -> PreparedPair:
    """Prefix-truncate PRE -> POST -> DIFF, retaining the PAIR summary wrapper.

    Stream markers and source streams are tokenized independently. Coverage
    excludes markers/wrappers; source IDs are sliced without decode/re-encode.
    Original lengths and all change spans survive truncation for diagnostics.
    """
    if cfg.input_mode != "full_diff":
        raise ConfigError("concat requires full_diff input_mode")
    budget = cfg.window_tokens - len(tokenizer.wrap_ids((), "PAIR"))
    if budget <= 0:
        raise ConfigError("PAIR wrapper exhausts the complete window budget")
    pre, post = normalize_code(record.pre_code), normalize_code(record.post_code)
    if pre is None and post is None:
        raise InputError("both source-code sides are absent")
    diff = make_diff(record.pre_code, record.post_code, context_lines=cfg.diff_context_lines)
    texts = (pre or EMPTY, post or EMPTY, diff.text)
    content, offsets, serialized = [], [], ""
    counts, coverage, ranges = [], [], []
    for i, (role, text) in enumerate(zip(("PRE", "POST", "DIFF"), texts)):
        marker = ("" if i == 0 else "\n") + role + "\n"
        for segment, source in ((marker, False), (text, True)):
            ids, spans = tokenizer.tokenize(segment)
            _validate_offsets(segment, ids, spans)
            start = len(content)
            content.extend(ids)
            offsets.extend((a + len(serialized), b + len(serialized)) for a, b in spans)
            serialized += segment
            if source:
                kept = max(0, min(len(ids), budget - start))
                coverage.append(kept / len(ids))
                counts.append(0 if (i == 0 and pre is None) or (i == 1 and post is None) else len(ids))
                ranges.append((0, max((b for _, b in spans[:kept]), default=0)))
    end = min(budget, len(content))
    wrapped = tokenizer.wrap_ids(content[:end], "PAIR")
    if len(wrapped) > cfg.window_tokens:
        raise ConfigError("PAIR wrapper exceeds total budget for actual content")
    line_ends = _line_ends(record.pre_code or ""), _line_ends(record.post_code or "")
    totals = len(line_ends[0]), len(line_ends[1])
    pre_lines = _line_range(line_ends[0], *ranges[0]) if pre is not None else (0, 0)
    post_lines = _line_range(line_ends[1], *ranges[1]) if post is not None else (0, 0)
    char_end = max((b for _, b in offsets[:end]), default=0)
    window = Window(wrapped, "PAIR", (0, end), pre_lines, post_lines,
                    _position(pre_lines, post_lines, totals), (0, char_end), serialized[:char_end],
                    PREFIX_TEMPLATE.format(ROLE="PAIR"), SUFFIX,
                    getattr(tokenizer, "special_token_rule", f"{type(tokenizer).__name__}.wrap_ids(content_ids, role)"))
    return PreparedPair(record, (window,), tuple(counts), tuple(coverage),
                        tuple(_position(span.pre_lines, span.post_lines, totals)
                              for span in diff.spans if span.kind != "CTX"),
                        (len(record.pre_code or ""), len(record.post_code or "")), cfg.input_mode, diff)
