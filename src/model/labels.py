"""Five shared fine labels and explicit projection of legacy supervision."""

from collections.abc import Sequence
from dataclasses import dataclass

from model.errors import DataError, InputError

COARSE_LABELS = ("Add", "Modify", "Delete")
FINE_LABELS = (
    "try_body_changed",
    "catch_added",
    "catch_removed",
    "catch_type_changed",
    "catch_body_changed",
)
REMOVED_FINALLY_LABELS = (
    "finally_added",
    "finally_removed",
    "finally_body_changed",
)
LEGACY_FINE_LABELS = (*FINE_LABELS, *REMOVED_FINALLY_LABELS)
REMOVED_FILTER_LABEL = "catch_filter_changed"
FINE_LABEL_VERSION = "eh-change-5"


@dataclass(frozen=True, slots=True)
class Targets:
    coarse: tuple[float, ...]
    fine: tuple[float, ...]
    modify_mask: bool
    removed_filter: bool
    removed_finally: tuple[str, ...] = ()


def normalize_code(value: str | None) -> str | None:
    """Treat blank code as absent without rewriting nonblank source."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InputError("code must be a string or null")
    return value if value.strip() else None


def _label_set(values: Sequence[str], allowed: tuple[str, ...], name: str) -> set[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise DataError(f"{name} must be an explicit array of strings")
    if any(not isinstance(value, str) for value in values):
        raise DataError(f"{name} must contain only strings")
    result = set(values)
    if len(result) != len(values):
        raise DataError(f"{name} contains duplicate labels")
    unknown = result - set(allowed)
    if unknown:
        raise DataError(f"{name} contains unknown labels: {sorted(unknown)}")
    return result


def encode_labels(
    labels: Sequence[str],
    subtypes: Sequence[str] | None,
    *,
    has_pre: bool,
    has_post: bool,
) -> Targets:
    """Validate supervision and emit targets in the fixed three/five order.

    Modify requires an explicit subtype array. An empty array, including one
    produced by filter/finally-only migration, is known negative fine
    supervision. Projection never changes the coarse Modify label.
    """
    if type(has_pre) is not bool or type(has_post) is not bool:
        raise InputError("side presence must be boolean")
    if not has_pre and not has_post:
        raise InputError("both source-code sides are absent")
    coarse = _label_set(labels, (*COARSE_LABELS, "None"), "labels")
    if "None" in coarse and len(coarse) > 1:
        raise DataError("None cannot be combined with positive labels")
    coarse.discard("None")
    if not has_pre and coarse - {"Add"}:
        raise DataError("a post-only pair can only have Add or None")
    if not has_post and coarse - {"Delete"}:
        raise DataError("a pre-only pair can only have Delete or None")
    modify = "Modify" in coarse
    if subtypes is None:
        if modify:
            raise DataError("Modify requires an explicit modify_subtypes array")
        fine = set()
    else:
        fine = _label_set(subtypes, (*LEGACY_FINE_LABELS, REMOVED_FILTER_LABEL), "modify_subtypes")
    if fine and not modify:
        raise DataError("modify_subtypes require the Modify coarse label")
    return Targets(
        coarse=tuple(float(label in coarse) for label in COARSE_LABELS),
        fine=tuple(float(label in fine) for label in FINE_LABELS),
        modify_mask=modify,
        removed_filter=REMOVED_FILTER_LABEL in fine,
        removed_finally=tuple(label for label in REMOVED_FINALLY_LABELS if label in fine),
    )
