"""Strict external predictions and source-aware, repository-group summaries."""
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from model.data.jsonl import _unique_object
from model.data.schema import PairRecord
from model.evaluate.decode import Thresholds
from model.evaluate.metrics import EvaluationReport, evaluate, gold_arrays
from model.input.types import PreparedPair
from model.labels import COARSE_LABELS, FINE_LABELS, LEGACY_FINE_LABELS, REMOVED_FILTER_LABEL
from model.models.base import PairLogits

_LENGTH_BUCKETS = (("0-1024", 0, 1024), ("1025-4096", 1025, 4096),
                   ("4097-16384", 4097, 16384), ("16385+", 16385, math.inf))
_METRICS = ("coarse_macro_f1", "fine_end_to_end_macro_f1", "fine_common_end_to_end_macro_f1", "subset_match")


def read_external_predictions(path: Path, records: Sequence[PairRecord]) -> PairLogits:
    """Require exactly the requested IDs; reorder rows/columns by identity.

    An unsupported one-sided input must be a missing/invalid export and fails
    validation. No target is read and no prediction is synthesized. Legacy
    eight/nine-way fine heads are projected by declared label names; removed
    filter/finally outputs are discarded only for retrospective comparisons.
    """
    expected = [r.record_id for r in records]
    if len(set(expected)) != len(expected):
        raise ValueError("requested records contain duplicate record_id")
    found = {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(row, dict) or set(row) != {
                "record_id", "coarse_label_order", "fine_label_order", "coarse_logits", "fine_logits"
            }:
                raise ValueError("external row requires the five declared fields")
            identity = row["record_id"]
            if not isinstance(identity, str) or not identity or identity in found:
                raise ValueError("missing or duplicate record_id")
            projected = []
            for name, labels in (("coarse", COARSE_LABELS), ("fine", FINE_LABELS)):
                order, values = row[f"{name}_label_order"], row[f"{name}_logits"]
                if (not isinstance(order, list) or any(not isinstance(v, str) for v in order)
                        or len(set(order)) != len(order)):
                    raise ValueError(f"invalid {name} label order")
                allowed = [set(labels)]
                if name == "fine":
                    allowed.extend([set(labels) | {REMOVED_FILTER_LABEL},
                                    set(LEGACY_FINE_LABELS), set(LEGACY_FINE_LABELS) | {REMOVED_FILTER_LABEL}])
                if set(order) not in allowed:
                    raise ValueError(f"missing/unknown {name} label declaration")
                if (not isinstance(values, list) or len(values) != len(order)
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                    raise ValueError(f"invalid {name} logit dimensions or values")
                projected.append([values[order.index(label)] for label in labels])
            found[identity] = projected
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ValueError(f"external prediction line {line_no}: {exc}") from exc
    missing, extra = set(expected) - found.keys(), found.keys() - set(expected)
    if missing or extra:
        raise ValueError(f"external coverage failed: requested={len(expected)}, predicted={len(set(expected) & found.keys())}, "
                         f"failed={len(missing)}, missing_ids={sorted(missing)}, extra_ids={sorted(extra)}")
    # Float64 preserves finite JSON logits outside float32's representable range.
    return PairLogits(torch.tensor([found[r][0] for r in expected], dtype=torch.float64).reshape(-1, len(COARSE_LABELS)),
                      torch.tensor([found[r][1] for r in expected], dtype=torch.float64).reshape(-1, len(FINE_LABELS)))


def _bootstrap_indices(groups, rng, samples):
    """Sample G whole independent groups G times, with multiplicity."""
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    values = list(members.values())
    for _ in range(samples):
        yield [index for selected in rng.integers(0, len(values), len(values)) for index in values[selected]]


def _metric_values(report):
    return {"coarse_macro_f1": report.metrics["coarse_macro_f1"],
            "fine_end_to_end_macro_f1": report.metrics["fine_end_to_end_macro_f1"],
            "fine_common_end_to_end_macro_f1": report.metrics["fine_common_end_to_end"]["macro_f1"],
            "subset_match": report.metrics["subset_match"]}


def _support(report):
    return {"coarse_macro_f1": report.metrics["coarse"]["support"],
            "fine_end_to_end_macro_f1": report.metrics["fine_end_to_end"]["support"],
            "fine_common_end_to_end_macro_f1": report.metrics["fine_common_end_to_end"]["support"],
            "subset_match": report.counts["n_samples"]}


def _reasons(report):
    return {name: "no positive labels" for name, support in _support(report).items() if not support}


def _stratum(pairs, logits, thresholds, language, indices):
    if not indices:
        return {"counts": {"n_samples": 0}, "metrics": None, "diagnostics": {}, "reason": "empty stratum"}
    report = evaluate([pairs[i].record for i in indices],
                      PairLogits(logits.coarse[indices], logits.fine[indices]), thresholds, language=language)
    return {"counts": report.counts, "metrics": report.metrics, "diagnostics": report.diagnostics,
            "metric_reasons": _reasons(report), "reason": None}


def _change_buckets(pair):
    buckets = set()
    for position in pair.change_positions:
        for start, end in (position[:2], position[2:]):
            if end > start:
                midpoint = (start + end) / 2
                buckets.add("beginning" if midpoint < 1/3 else "middle" if midpoint < 2/3 else "end")
    return buckets or {"no_text_change"}


def _cross_window(pair):
    # A source-defined change extent per side, tested against actual full-source
    # windows. DIFF-only/PAIR observations cannot establish this diagnostic.
    source_windows = [w for w in pair.windows if w.role in {"PRE", "POST"} and not w.is_placeholder]
    if not source_windows:
        return "unavailable"
    for role, source in (("PRE", pair.record.pre_code), ("POST", pair.record.post_code)):
        boundaries = [0]
        for line in (source or "").splitlines(keepends=True):
            boundaries.append(boundaries[-1] + len(line))
        lines = [span.pre_lines if role == "PRE" else span.post_lines
                 for span in pair.diff.spans if span.kind != "CTX"]
        changes = [(boundaries[start], boundaries[end]) for start, end in lines if end > start]
        if not changes:
            continue
        extent = min(p[0] for p in changes), max(p[1] for p in changes)
        windows = [w for w in source_windows if w.role == role]
        if windows and not any(w.character_range[0] <= extent[0] and w.character_range[1] >= extent[1] for w in windows):
            return "cross_window"
    return "within_window" if pair.change_positions else "no_text_change"


def _bootstrap(report, records, groups, samples, seed, reason):
    support = _support(report)
    values = {name: [] for name in _METRICS}
    if reason is None:
        coarse, fine, _ = gold_arrays(records)
        cp = np.asarray([[label in p.coarse_labels for label in COARSE_LABELS] for p in report.predictions])
        fp = np.asarray([[label in p.fine_labels for label in FINE_LABELS] for p in report.predictions])
        for indices in _bootstrap_indices(groups, np.random.default_rng(seed), samples):
            for name, gold, pred in (("coarse_macro_f1", coarse[indices], cp[indices]),
                                    ("fine_end_to_end_macro_f1", fine[indices], fp[indices]),
                                    ("fine_common_end_to_end_macro_f1", fine[indices], fp[indices])):
                if gold.any():
                    tp = (gold & pred).sum(axis=0)
                    denominator = gold.sum(axis=0) + pred.sum(axis=0)
                    values[name].append(float(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0).mean()))
            values["subset_match"].append(float(((coarse[indices] == cp[indices]).all(axis=1)
                                                 & (fine[indices] == fp[indices]).all(axis=1)).mean()))
    return {name: {"estimate": _metric_values(report)[name], "support": support[name],
                   "ci95": np.percentile(values[name], [2.5, 97.5]).tolist() if values[name] else None,
                   "valid_samples": len(values[name]), "invalid_samples": samples - len(values[name]),
                   "reason": reason or ("no positive labels" if not support[name] else
                                        "no valid supported resamples" if not values[name] else None)} for name in _METRICS}


def summarize_predictions(pairs: Sequence[PreparedPair], logits: PairLogits, thresholds: Thresholds, *,
                          language: str, bootstrap_seed: int = 0, bootstrap_samples: int = 1000,
                          repo_groups: Mapping[str, str] | None = None) -> EvaluationReport:
    """Evaluate one language; append descriptive strata and group CIs.

    Missing repository metadata disables CIs, never substitutes sample-level
    resampling. Explicit mappings must cover every observed repository.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    records = [p.record for p in pairs]
    aliases = {"C#": "csharp", "Java": "java", "C++": "cpp"}
    if any(r.language is not None and aliases.get(r.language, r.language) != language for r in records):
        raise ValueError("summaries require a single matching language")
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("summaries require unique record_id")
    if repo_groups is not None and (not isinstance(repo_groups, Mapping)
            or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                   for k, v in repo_groups.items())
            or any(r.repo_id is not None and r.repo_id not in repo_groups for r in records)):
        raise ValueError("repo_groups must cover every repository with nonblank group IDs")
    groups = [repo_groups[r.repo_id] if repo_groups is not None and r.repo_id is not None else r.repo_id for r in records]
    missing = any(not isinstance(g, str) or not g.strip() for g in groups)
    n_groups = len(set(groups)) if not missing else None
    reason = "missing repo_id" if missing else "fewer than two independent groups" if (n_groups or 0) < 2 else None
    report = evaluate(records, logits, thresholds, language=language)
    report.diagnostics["metric_reasons"] = _reasons(report)
    report.diagnostics["length_buckets"] = {
        label: _stratum(pairs, logits, thresholds, language, [i for i, p in enumerate(pairs) if low <= sum(p.token_counts[:2]) <= high])
        for label, low, high in _LENGTH_BUCKETS}
    report.diagnostics["change_positions"] = {
        label: _stratum(pairs, logits, thresholds, language, [i for i, p in enumerate(pairs) if label in _change_buckets(p)])
        for label in ("beginning", "middle", "end", "no_text_change")}
    report.diagnostics["cross_window"] = {
        label: _stratum(pairs, logits, thresholds, language, [i for i, p in enumerate(pairs) if _cross_window(p) == label])
        for label in ("cross_window", "within_window", "no_text_change", "unavailable")}
    report.diagnostics["coverage"] = {
        "truncated_pairs": sum(any(c is not None and c < 1 for c in p.coverage) for p in pairs),
        "PAIR_window_count": sum(w.role == "PAIR" for p in pairs for w in p.windows),
        "streams": {role: {"enabled_pairs": sum(p.coverage[j] is not None for p in pairs),
                             "omitted_pairs": sum(p.coverage[j] is None for p in pairs),
                             "mean_unique_coverage": float(np.mean([p.coverage[j] for p in pairs if p.coverage[j] is not None]))
                                 if any(p.coverage[j] is not None for p in pairs) else None}
                    for j, role in enumerate(("PRE", "POST", "DIFF"))},
        "records": [{"record_id": p.record.record_id, "source_id": p.record.source_id, "repo_id": p.record.repo_id,
                     "token_counts": list(p.token_counts), "unique_coverage": list(p.coverage),
                     "window_count": len(p.windows), "input_mode": p.input_mode} for p in pairs]}
    report.diagnostics["bootstrap"] = _bootstrap(report, records, groups, bootstrap_samples, bootstrap_seed, reason)
    report.provenance.update(bootstrap={"unit": "repo_group" if repo_groups is not None else "repo_id", "n_groups": n_groups,
        "seed": bootstrap_seed, "samples": bootstrap_samples, "confidence": .95, "method": "percentile",
        "reason": reason, "small_group_warning": "fewer than five independent groups" if n_groups is not None and n_groups < 5 else None},
        record_ids=[r.record_id for r in records],
        dataset_signature=hashlib.sha256(json.dumps([(r.record_id, r.repo_id, r.source_id, r.pre_code, r.post_code,
            r.targets.coarse, r.targets.fine) for r in records], ensure_ascii=False).encode()).hexdigest(),
        length_definition="original PRE + POST content token counts, excluding DIFF/wrappers",
        change_position_definition="changed source-line midpoint thirds; a pair may appear in multiple thirds",
        cross_window_definition="changed extent on either source side cannot fit any single actual full-source window; diff-only/concat unavailable",
        label_source="unspecified; caller must attach audited AST-rule or manual provenance")
    return report


def summarize_seeds(reports: Sequence[EvaluationReport]) -> dict:
    """Training-seed mean and sample SD, separate from repository uncertainty."""
    if not reports:
        raise ValueError("at least one seed report is required")
    first = reports[0]
    for report in reports:
        if any(report.provenance.get(key) != first.provenance.get(key)
               for key in ("language", "fine_labels", "dataset_signature")):
            raise ValueError("seed reports must use the same language, label universe and ordered dataset")
    return {"n_seeds": len(reports), "language": first.provenance["language"],
            "metrics": {name: {"mean": float(np.mean([_metric_values(r)[name] for r in reports])),
                               "sample_std": float(np.std([_metric_values(r)[name] for r in reports], ddof=1)) if len(reports) > 1 else None,
                               "support": _support(first)[name],
                               "reason": "fewer than two seeds" if len(reports) < 2 else _reasons(first).get(name)}
                        for name in _METRICS}}
