"""Set metrics and continuous AP rankings, with explicit fixed label universes."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from model.data.schema import PairRecord
from model.evaluate.decode import Prediction, Thresholds, decode, probabilities
from model.labels import COARSE_LABELS, FINE_LABELS, FINE_LABEL_VERSION
from model.models.base import PairLogits


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    counts: dict
    metrics: dict
    diagnostics: dict
    provenance: dict
    predictions: tuple[Prediction, ...]

    def to_dict(self) -> dict:
        return {"counts": self.counts, "metrics": self.metrics, "diagnostics": self.diagnostics, "provenance": self.provenance, "predictions": [p.to_dict() for p in self.predictions]}


def gold_arrays(records: Sequence[PairRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if any(record.targets is None for record in records):
        raise ValueError("evaluation and calibration require gold targets for every record")
    coarse = np.asarray([record.targets.coarse for record in records], dtype=bool).reshape(-1, len(COARSE_LABELS))
    fine = np.asarray([record.targets.fine for record in records], dtype=bool).reshape(-1, len(FINE_LABELS))
    modify = np.asarray([record.targets.modify_mask for record in records], dtype=bool)
    return coarse, fine, modify


def binary_metrics(gold: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> dict:
    tp = int(np.sum(gold & predicted))
    fp = int(np.sum(~gold & predicted))
    fn = int(np.sum(gold & ~predicted))
    support = int(np.sum(gold))
    return {"tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "pr_auc": float(average_precision_score(gold, scores)) if support else None}


def label_metrics(gold: np.ndarray, predicted: np.ndarray, scores: np.ndarray, labels: tuple[str, ...]) -> dict:
    per_label = {label: binary_metrics(gold[:, i], predicted[:, i], scores[:, i]) for i, label in enumerate(labels)}
    micro = binary_metrics(gold.reshape(-1), predicted.reshape(-1), scores.reshape(-1))
    return {"n_samples": len(gold), "per_label": per_label,
            "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels),
            "micro_f1": micro["f1"], "micro": micro,
            "support": int(gold.sum()),
            "subset_match": float(np.all(gold == predicted, axis=1).mean()) if len(gold) else 0.}


def evaluate(records: Sequence[PairRecord], logits: PairLogits, thresholds: Thresholds, *, language: str) -> EvaluationReport:
    if language not in {"csharp", "java", "cpp"}:
        raise ValueError("language must be csharp, java or cpp")
    predictions = tuple(decode(logits, records, thresholds))
    coarse_scores, fine_scores = probabilities(logits, len(records))
    gold_coarse, gold_fine, gold_modify = gold_arrays(records)
    predicted_coarse = np.asarray([[label in p.coarse_labels for label in COARSE_LABELS] for p in predictions], dtype=bool).reshape(-1, len(COARSE_LABELS))
    predicted_fine = np.asarray([[label in p.fine_labels for label in FINE_LABELS] for p in predictions], dtype=bool).reshape(-1, len(FINE_LABELS))
    raw_fine_sets = fine_scores >= np.asarray(thresholds.fine)
    modify_index = COARSE_LABELS.index("Modify")
    end_scores = coarse_scores[:, modify_index, None] * fine_scores
    coarse = label_metrics(gold_coarse, predicted_coarse, coarse_scores, COARSE_LABELS)
    end = label_metrics(gold_fine, predicted_fine, end_scores, FINE_LABELS)
    conditional = label_metrics(gold_fine[gold_modify], raw_fine_sets[gold_modify], fine_scores[gold_modify], FINE_LABELS)
    gold_none, predicted_none = ~gold_coarse.any(axis=1), ~predicted_coarse.any(axis=1)
    none = binary_metrics(gold_none, predicted_none, np.prod(1 - coarse_scores, axis=1))
    metrics = {"coarse": coarse, "coarse_macro_f1": coarse["macro_f1"],
               "fine_end_to_end": end, "fine_end_to_end_macro_f1": end["macro_f1"],
               "fine_conditional": conditional, "fine_common_end_to_end": end,
               "fine_common_conditional": conditional, "none": none,
               "subset_match": float((np.all(gold_coarse == predicted_coarse, axis=1) & np.all(gold_fine == predicted_fine, axis=1)).mean()) if records else 0.}
    return EvaluationReport(
        {"n_samples": len(records), "n_gold_modify": int(gold_modify.sum()), "n_predicted_modify": int(predicted_coarse[:, modify_index].sum()), "coarse_support": {label: int(gold_coarse[:, i].sum()) for i, label in enumerate(COARSE_LABELS)}, "fine_support": {label: int(gold_fine[:, i].sum()) for i, label in enumerate(FINE_LABELS)}},
        metrics,
        {},
        {"language": language, "fine_label_version": FINE_LABEL_VERSION, "coarse_labels": list(COARSE_LABELS), "fine_labels": list(FINE_LABELS), "common_fine_labels": list(FINE_LABELS), "thresholds": thresholds.to_dict(), "pr_auc_definition": "average_precision_score (AP)", "ranking_scores": {"coarse": "raw sigmoid(coarse)", "fine_end_to_end": "sigmoid(Modify) * sigmoid(fine)", "fine_conditional": "raw sigmoid(fine)", "none": "product(1 - sigmoid(coarse))"}, "set_predictions": "per-label thresholds, source-side legality, predicted Modify gate; conditional fine uses gold Modify subset and ungated fine thresholds"},
        predictions)
