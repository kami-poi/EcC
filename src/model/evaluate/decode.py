"""Thresholded predictions with source-side legality and Modify gating."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math

import numpy as np

from model.data.schema import PairRecord
from model.labels import COARSE_LABELS, FINE_LABELS, normalize_code
from model.models.base import PairLogits


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    split: str
    language: str
    coarse_support: tuple[int, ...]
    fine_support: tuple[int, ...]

    def __post_init__(self):
        if self.split != "val" or self.language != "csharp":
            raise ValueError("calibration provenance requires split=val and language=csharp")
        for name, size in (("coarse_support", len(COARSE_LABELS)), ("fine_support", len(FINE_LABELS))):
            values = tuple(getattr(self, name))
            if len(values) != size or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{name} requires {size} nonnegative integer counts")
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class Thresholds:
    coarse: tuple[float, ...] = (.5,) * len(COARSE_LABELS)
    fine: tuple[float, ...] = (.5,) * len(FINE_LABELS)
    provenance: CalibrationProvenance | None = None

    def __post_init__(self):
        if self.provenance is not None and not isinstance(self.provenance, CalibrationProvenance):
            raise ValueError("provenance must be CalibrationProvenance or None")
        for name, size in (("coarse", len(COARSE_LABELS)), ("fine", len(FINE_LABELS))):
            values = tuple(getattr(self, name))
            if len(values) != size or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError(f"{name} thresholds must contain {size} finite values in [0,1]")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict:
        return {"coarse": list(self.coarse), "fine": list(self.fine)}

    @classmethod
    def from_dict(cls, value: dict) -> "Thresholds":
        if set(value) != {"coarse", "fine"}:
            raise ValueError("thresholds require exactly coarse and fine")
        return cls(tuple(value["coarse"]), tuple(value["fine"]))


@dataclass(frozen=True, slots=True)
class Prediction:
    coarse_scores: tuple[float, ...]
    fine_scores: tuple[float, ...]
    coarse_labels: tuple[str, ...]
    fine_labels: tuple[str, ...]

    def to_dict(self) -> dict:
        return {key: list(value) for key, value in asdict(self).items()}


def probabilities(logits: PairLogits, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Detach logits once; keep raw scores even for labels illegal on one side."""
    arrays = []
    for value, size in ((logits.coarse, len(COARSE_LABELS)), (logits.fine, len(FINE_LABELS))):
        if tuple(value.shape) != (n, size):
            raise ValueError(f"logits must have shape {(n, size)}")
        if not value.isfinite().all().item():
            raise ValueError("logits must be finite")
        arrays.append(value.detach().double().sigmoid().cpu().numpy())
    return tuple(arrays)


def legal_coarse(records: Sequence[PairRecord]) -> np.ndarray:
    allowed = np.ones((len(records), len(COARSE_LABELS)), dtype=bool)
    for i, record in enumerate(records):
        pre, post = normalize_code(record.pre_code), normalize_code(record.post_code)
        if pre is None and post is None:
            raise ValueError("both source-code sides are absent")
        if pre is None:
            allowed[i] = [label == "Add" for label in COARSE_LABELS]
        elif post is None:
            allowed[i] = [label == "Delete" for label in COARSE_LABELS]
    return allowed


def decode(logits: PairLogits, records: Sequence[PairRecord], thresholds: Thresholds) -> list[Prediction]:
    coarse, fine = probabilities(logits, len(records))
    selected = (coarse >= np.asarray(thresholds.coarse)) & legal_coarse(records)
    modify_index = COARSE_LABELS.index("Modify")
    result = []
    for i in range(len(records)):
        coarse_labels = tuple(label for label, yes in zip(COARSE_LABELS, selected[i]) if yes)
        fine_labels = tuple(label for label, score, threshold in zip(FINE_LABELS, fine[i], thresholds.fine) if score >= threshold) if selected[i, modify_index] else ()
        result.append(Prediction(tuple(map(float, coarse[i])), tuple(map(float, fine[i])), coarse_labels or ("None",), fine_labels))
    return result
