import hashlib
import json
from dataclasses import asdict

import pytest

from model.data.audit import audit_dataset


def record(repo="repo-a", source="source-a", pre="a", post="b", language="C#", **extra):
    return {"repo_id": repo, "commit_sha": "commit", "source_id": source, "language": language,
            "pre_code": pre, "post_code": post, "labels": ["Modify"],
            "modify_subtypes": ["catch_filter_changed"], **extra}


def manifest(tmp_path, entries=None, **extra):
    entries = entries or [("train", [record()], {})]
    files, repos, splits = [], {}, {}
    for index, (split, rows, overrides) in enumerate(entries):
        path = tmp_path / f"pairs-{index}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        files.append({"path": path.name, "split": split, "language": "C#",
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_records": len(rows), **overrides})
        for row in rows:
            if row.get("repo_id"):
                repos[row["repo_id"]] = row["repo_id"]
                splits[row["repo_id"]] = split
    payload = {"schema_version": 1, "label_version": "eh-change-9", "label_source": "ast_rules",
               "rule_version": "rules-1", "split_seed": 0, "sampling_seed": 1,
               "repo_groups": repos, "group_splits": splits, "files": files, **extra}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_formal_audit_counts_and_hashes(tmp_path):
    path = manifest(tmp_path)
    report = audit_dataset(path)
    assert report.passed
    assert report.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report.counts["total"] == 1
    assert report.counts["coarse"] == {"Add": 0, "Modify": 1, "Delete": 0}
    assert report.counts["fine"]["catch_added"] == 0
    assert report.counts["filter_only"] == 1
    assert report.counts["formal"] is True
    assert "pairs-0.jsonl" in report.file_sha256
    json.dumps(asdict(report))


@pytest.mark.parametrize("pre,post,issue", [("a", "b", "exact_pair_overlap"), ("b", "a", "reversed_pair_overlap")])
def test_split_overlap_and_reversed_pair(tmp_path, pre, post, issue):
    path = manifest(tmp_path, [("train", [record()], {}),
                              ("test", [record("repo-b", "source-b", pre, post)], {})])
    report = audit_dataset(path)
    assert not report.passed
    assert any(issue in item for item in report.issues)


@pytest.mark.parametrize("repo,source,issue", [("repo-a", "different", "group_overlap"),
                                               ("repo-b", "source-a", "source_overlap")])
def test_shared_group_or_source_across_splits(tmp_path, repo, source, issue):
    report = audit_dataset(manifest(tmp_path, [("train", [record()], {}),
                                             ("val", [record(repo, source, "c", "d")], {})]))
    assert not report.passed
    assert any(issue in item for item in report.issues)


def test_manifest_hash_mismatch(tmp_path):
    report = audit_dataset(manifest(tmp_path, [("train", [record()], {"sha256": "0" * 64})]))
    assert not report.passed
    assert any("sha256" in item for item in report.issues)


def test_missing_provenance_cannot_pass_formal(tmp_path):
    row = record()
    for name in ("repo_id", "commit_sha", "source_id", "language"):
        del row[name]
    path = manifest(tmp_path, [("train", [row], {})])
    report = audit_dataset(path)
    assert not report.passed
    assert any("provenance" in item for item in report.issues)
    debug = audit_dataset(path, mode="debug")
    assert debug.passed
    assert debug.counts["formal"] is False


def test_debug_still_rejects_label_errors(tmp_path):
    report = audit_dataset(manifest(tmp_path, [("train", [record(modify_subtypes=None)], {})]), mode="debug")
    assert not report.passed
    assert any("modify_subtypes" in item for item in report.issues)


def test_label_version_mismatch(tmp_path):
    report = audit_dataset(manifest(tmp_path, label_version="eh-change-8"))
    assert not report.passed
    assert any("label_version" in item for item in report.issues)


@pytest.mark.parametrize("version", ["eh-change-8", "eh-change-9"])
def test_legacy_finally_projection_is_counted_without_dropping_records(tmp_path, version):
    from model.labels import FINE_LABELS, REMOVED_FINALLY_LABELS

    subtypes = [[name] for name in REMOVED_FINALLY_LABELS]
    subtypes += [["catch_added", "finally_added"], ["catch_filter_changed", "finally_removed"]]
    if version == "eh-change-8":
        subtypes.pop()
    rows = [record(source=f"s-{i}", pre=f"a-{i}", post=f"b-{i}", modify_subtypes=names)
            for i, names in enumerate(subtypes)]
    report = audit_dataset(manifest(tmp_path, [("train", rows, {})], label_version=version))
    assert report.passed, report.issues
    assert report.counts["total"] == report.counts["coarse"]["Modify"] == len(rows)
    assert report.counts["fine_label_order"] == list(FINE_LABELS)
    assert report.counts["fine_label_version"] == "eh-change-5"
    assert report.counts["finally_only"] == 3
    assert report.counts["removed_finally"] == len(rows)
    assert report.counts["removed_finally_labels"]["finally_added"] == 2
    assert report.counts["fine"]["catch_added"] == 1
    assert report.counts["empty_fine_modify"] == len(rows) - 1
    assert report.counts["filter_only"] == 0


def test_new_five_label_manifest_accepts_only_retained_supervision(tmp_path):
    rows = [record(modify_subtypes=["catch_added"])]
    report = audit_dataset(manifest(tmp_path, [("train", rows, {})], label_version="eh-change-5"))
    assert report.passed, report.issues


@pytest.mark.parametrize("removed", ["finally_added", "finally_removed", "finally_body_changed", "catch_filter_changed"])
def test_new_five_label_manifest_rejects_misdeclared_legacy_supervision(tmp_path, removed):
    path = manifest(tmp_path, [("train", [record(modify_subtypes=[removed])], {})], label_version="eh-change-5")
    report = audit_dataset(path)
    assert not report.passed
    assert any("eh-change-5 contains removed" in issue for issue in report.issues)


def test_cross_language_manual_test_override(tmp_path):
    row = record("repo-java", "java-source", "x", "y", "Java", modify_subtypes=["catch_added"],
                 label_source="manual", annotation_version="review-1")
    path = manifest(tmp_path, [("train", [record()], {}),
                              ("test", [row], {"language": "Java", "label_source": "manual", "annotation_version": "review-1"})])
    assert audit_dataset(path).passed


@pytest.mark.parametrize("split", ["train", "val"])
def test_supervised_splits_must_be_csharp(tmp_path, split):
    row = record(language="Java")
    report = audit_dataset(manifest(tmp_path, [(split, [row], {"language": "Java"})]))
    assert not report.passed
    assert any("C#" in item for item in report.issues)


@pytest.mark.parametrize("override,issue", [({"language": "Java"}, "language"),
                                          ({"label_source": "manual"}, "label_source"),
                                          ({"rule_version": "other"}, "rule_version")])
def test_record_metadata_must_match_declaration(tmp_path, override, issue):
    report = audit_dataset(manifest(tmp_path, [("train", [record(**override)], {})]))
    assert not report.passed
    assert any(issue in item for item in report.issues)


def test_whitespace_is_not_removed_for_pair_audit(tmp_path):
    path = manifest(tmp_path, [("train", [record(pre=" a ")], {}),
                              ("test", [record("repo-b", "source-b")], {})])
    assert audit_dataset(path).passed


def test_missing_marker_does_not_collide_with_literal_string(tmp_path):
    left = record(pre=None, labels=["Add"], modify_subtypes=[])
    right = record("repo-b", "source-b", pre="null", labels=["Add"], modify_subtypes=[])
    assert audit_dataset(manifest(tmp_path, [("train", [left], {}), ("test", [right], {})])).passed


@pytest.mark.parametrize("extra,issue", [({"schema_version": 2}, "schema_version"),
                                       ({"sampling_seed": None}, "sampling_seed"),
                                       ({"rule_version": None}, "rule_version"),
                                       ({"group_splits": {}}, "group_splits")])
def test_incomplete_manifest_cannot_pass(tmp_path, extra, issue):
    report = audit_dataset(manifest(tmp_path, **extra))
    assert not report.passed
    assert any(issue in item for item in report.issues)


def test_record_count_mismatch(tmp_path):
    report = audit_dataset(manifest(tmp_path, [("train", [record()], {"n_records": 2})]))
    assert not report.passed
    assert any("n_records" in item for item in report.issues)


def test_file_cannot_override_manifest_label_version(tmp_path):
    report = audit_dataset(manifest(tmp_path, [("train", [record()], {"label_version": "eh-change-8"})]))
    assert not report.passed
    assert any("label_version" in item for item in report.issues)


def test_distinct_repositories_in_same_group_cannot_cross_splits(tmp_path):
    path = manifest(tmp_path, [("train", [record()], {}),
                              ("test", [record("repo-b", "source-b", "c", "d")], {})],
                    repo_groups={"repo-a": "family", "repo-b": "family"}, group_splits={"family": "train"})
    report = audit_dataset(path)
    assert not report.passed
    assert any("group_overlap" in item for item in report.issues)


def test_audit_uses_explicit_field_map(tmp_path):
    row = record()
    row["repo_full_name"] = row.pop("repo_id")
    path = manifest(tmp_path, [("train", [row], {})], repo_groups={"repo-a": "group-a"},
                    group_splits={"group-a": "train"}, field_map={"repo_id": "repo_full_name"})
    assert audit_dataset(path).passed


def test_review_report_hash_is_checked(tmp_path):
    review = tmp_path / "review.json"
    review.write_text("{}", encoding="utf-8")
    report = audit_dataset(manifest(tmp_path, review_report={"path": review.name, "sha256": "0" * 64}))
    assert not report.passed
    assert any("review_report" in item and "sha256" in item for item in report.issues)


def test_manual_source_requires_annotation_version(tmp_path):
    report = audit_dataset(manifest(tmp_path, label_source="manual"))
    assert not report.passed
    assert any("annotation_version" in item for item in report.issues)


def test_declaration_path_cannot_escape_manifest_directory(tmp_path):
    report = audit_dataset(manifest(tmp_path, [("train", [record()], {"path": "../escape.jsonl"})]))
    assert not report.passed
    assert any("escapes" in item for item in report.issues)


def test_malformed_manifest_returns_failed_report(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{", encoding="utf-8")
    assert not audit_dataset(path).passed


@pytest.mark.parametrize("bad_path", ["bad\x00.jsonl", "bad\ud800.jsonl"])
def test_invalid_dataset_path_returns_contextual_failed_report(tmp_path, bad_path):
    path = manifest(tmp_path, [("train", [record()], {"path": bad_path})])
    report = audit_dataset(path)
    assert not report.passed
    assert any("files[0]" in issue and "path" in issue for issue in report.issues)


@pytest.mark.parametrize("bad_path", ["bad\x00.json", "bad\ud800.json"])
@pytest.mark.parametrize("attachment", ["manifest", "file"])
def test_invalid_review_report_path_returns_contextual_failed_report(tmp_path, bad_path, attachment):
    review = {"path": bad_path, "sha256": "0" * 64}
    if attachment == "manifest":
        path = manifest(tmp_path, review_report=review)
    else:
        path = manifest(tmp_path, [("train", [record()], {"review_report": review})])
    report = audit_dataset(path)
    assert not report.passed
    expected_context = "manifest" if attachment == "manifest" else "files[0]"
    assert any(expected_context in issue and "review_report" in issue and "path" in issue
               for issue in report.issues)


def fine_exception_manifest(tmp_path, *, exception_changes=None, row_changes=None, policy_changes=None):
    train = record(record_id="train-record", modify_subtypes=["finally_removed"])
    val = record(source="val-source", pre="c", post="d", record_id="val-record",
                 modify_subtypes=["finally_removed"])
    val.update(row_changes or {})
    exception = {"record_id": "val-record", "source_id": "val-source", "repo_id": "repo-a",
                 "original_split": "train", "assigned_split": "val", "fine_labels": ["finally_removed"]}
    exception.update(exception_changes or {})
    policy = {"repository_isolation": "fine_coverage_exceptions", "evaluation_distribution": "fine_coverage_supplemented",
              "record_exceptions": [exception]}
    policy.update(policy_changes or {})
    return manifest(tmp_path, [("train", [train], {}), ("val", [val], {})],
                    label_version="eh-change-8", repo_groups={"repo-a": "repo-a"},
                    group_splits={"repo-a": "train"}, split_policy=policy)


def test_declared_fine_repository_exception_passes_and_marks_relaxed_isolation(tmp_path):
    report = audit_dataset(fine_exception_manifest(tmp_path))
    assert report.passed, report.issues
    assert report.counts["formal"] is True
    assert report.counts["repository_isolated"] is False
    assert report.counts["repository_exceptions"] == 1
    assert report.counts["evaluation_distribution"] == "fine_coverage_supplemented"


def test_none_cap_policy_with_fine_repository_exceptions_is_audited(tmp_path):
    report = audit_dataset(fine_exception_manifest(tmp_path, policy_changes={
        "evaluation_distribution": "none_capped_with_fine_coverage"}))
    assert report.passed, report.issues
    assert report.counts["repository_isolated"] is False
    assert report.counts["evaluation_distribution"] == "none_capped_with_fine_coverage"


def test_none_cap_policy_alone_keeps_repository_isolation(tmp_path):
    path = manifest(tmp_path, split_policy={"repository_isolation": "strict", "evaluation_distribution": "none_capped",
                                           "record_exceptions": []})
    report = audit_dataset(path)
    assert report.passed, report.issues
    assert report.counts["repository_isolated"] is True
    assert report.counts["evaluation_distribution"] == "none_capped"


@pytest.mark.parametrize("changes", [{"record_id": "unmatched"}, {"source_id": "incorrect"},
                                     {"repo_id": "unknown"}, {"original_split": "test"},
                                     {"assigned_split": "train"}, {"fine_labels": []},
                                     {"fine_labels": ["unknown"]}, {"fine_labels": ["catch_added"]},
                                     {"fine_labels": ["finally_removed", "finally_removed"]}])
def test_exception_cannot_authorize_unmatched_identity_or_supervision(tmp_path, changes):
    report = audit_dataset(fine_exception_manifest(tmp_path, exception_changes=changes))
    assert not report.passed
    assert any("exception" in issue for issue in report.issues)


@pytest.mark.parametrize("changes", [{"repository_isolation": "strict"},
                                     {"repository_isolation": "ignore_everything"},
                                     {"evaluation_distribution": "natural"}, {"record_exceptions": None}])
def test_invalid_exception_policy_does_not_disable_group_checks(tmp_path, changes):
    report = audit_dataset(fine_exception_manifest(tmp_path, policy_changes=changes))
    assert not report.passed
    assert any("group_overlap" in issue for issue in report.issues)


@pytest.mark.parametrize("changes,issue", [({"source_id": "source-a"}, "source_overlap"),
                                          ({"pre_code": "a", "post_code": "b"}, "exact_pair_overlap"),
                                          ({"pre_code": "b", "post_code": "a"}, "reversed_pair_overlap")])
def test_fine_exception_keeps_source_and_pair_leakage_checks(tmp_path, changes, issue):
    exception_changes = {"source_id": changes["source_id"]} if "source_id" in changes else None
    report = audit_dataset(fine_exception_manifest(tmp_path, exception_changes=exception_changes, row_changes=changes))
    assert not report.passed
    assert any(issue in item for item in report.issues)


def test_exception_authorizes_only_its_record_not_the_entire_group(tmp_path):
    path = fine_exception_manifest(tmp_path)
    data = json.loads(path.read_text())
    pairs = tmp_path / "pairs-1.jsonl"
    with pairs.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record(source="undeclared-source", pre="e", post="f", record_id="undeclared",
                                      modify_subtypes=["finally_removed"])) + "\n")
    data["files"][1].update(n_records=2, sha256=hashlib.sha256(pairs.read_bytes()).hexdigest())
    path.write_text(json.dumps(data))
    report = audit_dataset(path)
    assert not report.passed
    assert any("undeclared" in issue and "group" in issue for issue in report.issues)


def test_duplicate_repository_exception_ids_are_rejected(tmp_path):
    path = fine_exception_manifest(tmp_path)
    data = json.loads(path.read_text())
    data["split_policy"]["record_exceptions"] *= 2
    path.write_text(json.dumps(data))
    report = audit_dataset(path)
    assert not report.passed
    assert any("duplicate exception" in issue for issue in report.issues)


def test_evaluation_metadata_preserves_relaxed_isolation_policy(tmp_path):
    from argparse import Namespace
    from model.cli import _eval_data

    path = fine_exception_manifest(tmp_path)
    data = json.loads(path.read_text())
    data["files"][1]["split"] = "test"
    data["split_policy"]["record_exceptions"][0]["assigned_split"] = "test"
    path.write_text(json.dumps(data))
    _, _, metadata = _eval_data(Namespace(manifest=path, data=tmp_path / "pairs-1.jsonl", language="csharp"))
    assert metadata["repository_isolated"] is False
    assert metadata["repository_exceptions"] == 1
    assert metadata["evaluation_distribution"] == "fine_coverage_supplemented"
