"""Two-stage, per-label F1 calibration restricted to C# validation data."""

from collections.abc import Sequence

import numpy as np

from model.data.schema import PairRecord
from model.evaluate.decode import CalibrationProvenance, Thresholds, legal_coarse, probabilities
from model.evaluate.metrics import gold_arrays
from model.labels import COARSE_LABELS, FINE_LABELS
from model.models.base import PairLogits

CANDIDATES = tuple(i / 20 for i in range(1, 20))


def _best_threshold(gold: np.ndarray, scores: np.ndarray, gate: np.ndarray) -> float:
    if not gold.any():
        return .5
    best = None
    for threshold in CANDIDATES:
        predicted = (scores >= threshold) & gate
        tp = int(np.sum(gold & predicted))
        fp = int(np.sum(~gold & predicted))
        fn = int(np.sum(gold & ~predicted))
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.
        # Integer distance prevents binary-float noise changing symmetric ties.
        key = (-f1, abs(round(threshold * 20) - 10), threshold)
        if best is None or key < best[0]:
            best = key, threshold
    return best[1]


def calibrate_thresholds(records: Sequence[PairRecord], logits: PairLogits, *, split: str, language: str) -> Thresholds:
    if split != "val" or language != "csharp":
        raise ValueError("threshold calibration requires split=val and language=csharp")
    if any(record.language not in (None, "csharp") for record in records):
        raise ValueError("threshold calibration cannot include transfer-language records")
    coarse_scores, fine_scores = probabilities(logits, len(records))
    gold_coarse, gold_fine, _ = gold_arrays(records)
    allowed = legal_coarse(records)
    coarse = tuple(_best_threshold(gold_coarse[:, i], coarse_scores[:, i], allowed[:, i]) for i in range(len(COARSE_LABELS)))
    modify_index = COARSE_LABELS.index("Modify")
    modify_gate = allowed[:, modify_index] & (coarse_scores[:, modify_index] >= coarse[modify_index])
    fine = tuple(_best_threshold(gold_fine[:, i], fine_scores[:, i], modify_gate) for i in range(len(FINE_LABELS)))
    provenance = CalibrationProvenance(split, language,
                                       tuple(int(v) for v in gold_coarse.sum(axis=0)),
                                       tuple(int(v) for v in gold_fine.sum(axis=0)))
    return Thresholds(coarse, fine, provenance)
