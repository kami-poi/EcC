"""External coverage and grouped scientific summaries use real evaluation."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from helpers import FakeTokenizer, make_record
from model.config import WindowConfig
from model.evaluate.decode import Thresholds
from model.evaluate.report import read_external_predictions, summarize_predictions, summarize_seeds, _bootstrap_indices
from model.input.windows import prepare_concat_pair, prepare_pair
from model.labels import COARSE_LABELS, FINE_LABELS, LEGACY_FINE_LABELS
from model.models.base import PairLogits


def records():
    base = make_record("old", "new", labels=["Modify"], subtypes=[FINE_LABELS[0]])
    return [replace(base, record_id=str(i), source_id=f"source-{i}", repo_id=repo, language="cpp")
            for i, repo in enumerate(["a", "a", "b", "c"])]


def exported(record):
    return dict(record_id=record.record_id, coarse_label_order=list(COARSE_LABELS), fine_label_order=list(FINE_LABELS),
                coarse_logits=[-2., 2., -2.], fine_logits=list(range(5)))


@pytest.mark.parametrize("bad", ["missing", "duplicate", "extra", "unknown_label", "duplicate_label", "dimension", "missing_order", "unsupported"])
def test_external_missing_duplicate_and_misaligned_ids(tmp_path, bad):
    recs = records()
    rows = [exported(r) for r in recs]
    if bad == "missing": rows.pop()
    elif bad == "duplicate": rows.append(rows[0])
    elif bad == "extra": rows.append(dict(rows[0], record_id="extra"))
    elif bad == "unknown_label": rows[0]["coarse_label_order"][0] = "unknown"
    elif bad == "duplicate_label": rows[0]["fine_label_order"][0] = FINE_LABELS[1]
    elif bad == "dimension": rows[0]["fine_logits"].pop()
    elif bad == "missing_order": rows[0].pop("coarse_label_order")
    else: rows[0]["fine_logits"] = None
    path = tmp_path / "external.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError): read_external_predictions(path, recs)


def test_external_projects_truthful_permutation_and_legacy_filter(tmp_path):
    recs = records()
    rows = [exported(r) for r in reversed(recs)]
    for row in rows:
        row["coarse_label_order"].reverse()
        row["coarse_logits"] = [30., 20., 10.]
        row["fine_label_order"] = ["catch_filter_changed", *reversed(FINE_LABELS)]
        row["fine_logits"] = [99., *reversed(range(5))]
    path = tmp_path / "external.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = read_external_predictions(path, [replace(r, targets=None) for r in recs])
    assert result.coarse.tolist() == [[10., 20., 30.]] * 4
    assert result.fine.tolist() == [list(range(5))] * 4


@pytest.mark.parametrize("with_filter", [False, True])
def test_external_legacy_finally_projection_uses_names_not_positions(tmp_path, with_filter):
    recs = records()
    old_order = [*LEGACY_FINE_LABELS, *(["catch_filter_changed"] if with_filter else [])]
    scores = {name: float(i) for i, name in enumerate(FINE_LABELS)}
    rows = []
    for rec in reversed(recs):
        row = exported(rec)
        row["fine_label_order"] = list(reversed(old_order))
        row["fine_logits"] = [scores.get(name, 999.) for name in row["fine_label_order"]]
        rows.append(row)
    path = tmp_path / "legacy.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    projected = read_external_predictions(path, [replace(rec, targets=None) for rec in recs])
    assert projected.fine.tolist() == [list(range(5))] * len(recs)


def test_external_partial_legacy_finally_declaration_is_rejected(tmp_path):
    rec = records()[0]
    row = exported(rec)
    row["fine_label_order"].append("finally_added")
    row["fine_logits"].append(999.)
    path = tmp_path / "partial-legacy.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="label declaration"):
        read_external_predictions(path, [rec])


def test_repository_bootstrap_keeps_groups():
    draws = list(_bootstrap_indices(["ab", "ab", "ab", "c"], np.random.default_rng(0), 40))
    assert any(len(indices) != len(set(indices)) for indices in draws)
    for indices in draws:
        assert indices.count(0) == indices.count(1) == indices.count(2)
        assert indices.count(0) + indices.count(3) == 2
    pairs = [prepare_pair(r, FakeTokenizer(), WindowConfig()) for r in records()]
    report = summarize_predictions(pairs, PairLogits(torch.ones(4, 3), torch.ones(4, 5)), Thresholds(),
                                   language="cpp", bootstrap_samples=10, repo_groups={"a":"ab", "b":"ab", "c":"c"})
    assert report.provenance["bootstrap"]["n_groups"] == 2
    assert report.provenance["bootstrap"]["unit"] == "repo_group"
    assert not any(key.startswith("illegal_finally") for key in report.diagnostics)
    assert all(len(p.fine_scores) == 5 for p in report.predictions)
    assert report.diagnostics["bootstrap"]["coarse_macro_f1"]["valid_samples"] == 10
    json.dumps(report.to_dict(), allow_nan=False)


def test_length_buckets_use_original_code_length():
    pair = prepare_concat_pair(make_record("a"*2500, "b"*2500, labels=["Modify"]), FakeTokenizer(), WindowConfig())
    report = summarize_predictions([pair], PairLogits(torch.ones(1, 3), torch.ones(1, 5)), Thresholds(), language="csharp", bootstrap_samples=5)
    assert report.diagnostics["length_buckets"]["4097-16384"]["counts"]["n_samples"] == 1
    assert report.diagnostics["coverage"]["truncated_pairs"] == 1
    assert report.diagnostics["bootstrap"]["coarse_macro_f1"]["reason"] == "fewer than two independent groups"
    assert report.diagnostics["length_buckets"]["0-1024"]["reason"] == "empty stratum"


def test_metadata_missing_and_no_positive_support_are_explicit():
    pair = prepare_pair(replace(make_record("same", "same", labels=["None"]), repo_id=None, language="java"), FakeTokenizer(), WindowConfig())
    report = summarize_predictions([pair], PairLogits(torch.zeros(1, 3), torch.zeros(1, 5)), Thresholds(), language="java", bootstrap_samples=3)
    assert report.diagnostics["bootstrap"]["coarse_macro_f1"]["reason"] == "missing repo_id"
    assert report.diagnostics["metric_reasons"]["fine_end_to_end_macro_f1"] == "no positive labels"
    with pytest.raises(ValueError, match="repo_groups"):
        summarize_predictions([replace(pair, record=replace(pair.record, repo_id="x"))],
                              PairLogits(torch.zeros(1, 3), torch.zeros(1, 5)), Thresholds(), language="java", repo_groups={})


def test_seed_statistics_are_separate_and_validate_comparability():
    pairs = [prepare_pair(r, FakeTokenizer(), WindowConfig()) for r in records()]
    reports = [summarize_predictions(pairs, PairLogits(torch.full((4, 3), v), torch.full((4, 5), v)),
                                    Thresholds(), language="cpp", bootstrap_samples=2) for v in [-2., 2.]]
    summary = summarize_seeds(reports)
    expected = [r.metrics["coarse_macro_f1"] for r in reports]
    assert summary["metrics"]["coarse_macro_f1"]["mean"] == pytest.approx(np.mean(expected))
    assert summary["metrics"]["coarse_macro_f1"]["sample_std"] == pytest.approx(np.std(expected, ddof=1))
    assert summarize_seeds(reports[:1])["metrics"]["coarse_macro_f1"]["sample_std"] is None
    with pytest.raises(ValueError): summarize_seeds([])


def test_cross_window_diagnostic_handles_one_long_changed_line():
    pair = prepare_pair(make_record("a" * 2500, "b" * 2500, labels=["Modify"]), FakeTokenizer(), WindowConfig())
    report = summarize_predictions([pair], PairLogits(torch.ones(1, 3), torch.ones(1, 5)),
                                   Thresholds(), language="csharp", bootstrap_samples=2)
    assert report.diagnostics["cross_window"]["cross_window"]["counts"]["n_samples"] == 1


def test_external_one_sided_missing_is_not_synthesized(tmp_path):
    rec = make_record(None, "new", labels=["Add"])
    path = tmp_path / "unsupported.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="requested=1, predicted=0, failed=1"):
        read_external_predictions(path, [rec])
