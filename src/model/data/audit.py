"""Manifest integrity, declared provenance and cross-split leakage checks."""

import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from model.data.jsonl import _json_snapshot, _load_rows, _read_snapshot
from model.errors import DataError
from model.labels import (COARSE_LABELS, FINE_LABELS, FINE_LABEL_VERSION,
                          LEGACY_FINE_LABELS, REMOVED_FINALLY_LABELS, normalize_code)

_SPLITS = ("train", "val", "test")
_VERSIONS = ("eh-change-9", "eh-change-8", FINE_LABEL_VERSION)
_SOURCES = ("ast_rules", "manual")


@dataclass(frozen=True, slots=True)
class AuditReport:
    passed: bool
    manifest_sha256: str
    file_sha256: dict[str, str]
    counts: dict
    issues: list[str]


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _path(root: Path, value: object) -> Path:
    if not _text(value):
        raise DataError("path must be relative to the manifest directory")
    if "\x00" in value or any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise DataError("path contains a NUL character or lone Unicode surrogate")
    if Path(value).is_absolute() or PureWindowsPath(value).drive:
        raise DataError("path must be relative to the manifest directory")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise DataError("path escapes the manifest directory")
    return resolved


def _repository_exceptions(data: dict, repos: dict, groups: dict, issues: list[str]) -> tuple[dict, dict]:
    policy = data.get("split_policy", {"repository_isolation": "strict", "evaluation_distribution": "natural",
                                      "record_exceptions": []})
    if not isinstance(policy, dict):
        issues.append("split_policy must be an object")
        return {}, {}
    isolation = policy.get("repository_isolation")
    distribution = policy.get("evaluation_distribution", "natural")
    entries = policy.get("record_exceptions", [])
    distributions = ("natural", "fine_coverage_supplemented", "none_capped", "none_capped_with_fine_coverage")
    fine_distributions = ("fine_coverage_supplemented", "none_capped_with_fine_coverage")
    if isolation not in ("strict", "fine_coverage_exceptions") or distribution not in distributions:
        issues.append("split_policy has an invalid isolation or evaluation distribution")
        return {}, {}
    if not isinstance(entries, list):
        issues.append("split_policy.record_exceptions must be an array")
        return {}, policy
    if entries and (isolation != "fine_coverage_exceptions" or distribution not in fine_distributions):
        issues.append("record exceptions require fine_coverage_exceptions and a fine coverage evaluation distribution")
        return {}, policy
    exceptions = {}
    for index, entry in enumerate(entries):
        context = f"split_policy.record_exceptions[{index}]"
        if not isinstance(entry, dict) or any(not _text(entry.get(key)) for key in
                ("record_id", "source_id", "repo_id", "original_split", "assigned_split")):
            issues.append(f"{context}: exception identity must be complete")
            continue
        fine = entry.get("fine_labels")
        allowed_fine = FINE_LABELS if data.get("label_version") == FINE_LABEL_VERSION else LEGACY_FINE_LABELS
        if (not isinstance(fine, list) or not fine or any(not isinstance(name, str) or name not in allowed_fine for name in fine)
                or len(set(fine)) != len(fine)):
            issues.append(f"{context}: fine_labels must be unique known positive types")
            continue
        if (entry["original_split"] not in _SPLITS or entry["assigned_split"] not in ("val", "test")
                or entry["original_split"] == entry["assigned_split"]
                or groups.get(repos.get(entry["repo_id"])) != entry["original_split"]):
            issues.append(f"{context}: exception must move an original repository record to a different holdout")
            continue
        if entry["record_id"] in exceptions:
            issues.append(f"{context}: duplicate exception record_id")
            continue
        exceptions[entry["record_id"]] = entry
    return exceptions, policy


def audit_dataset(manifest: Path, *, mode: str = "formal") -> AuditReport:
    """Return all detected issues; debug success is explicitly nonformal.

    Declaration versions live in the manifest, with explicit file overrides.
    Optional repeated provenance fields on records must agree with declarations.
    """
    if mode not in ("formal", "debug"):
        raise DataError("audit mode must be formal or debug")
    formal = mode == "formal"
    issues, file_hashes = [], {}
    counts = {"formal": formal, "total": 0, "splits": dict.fromkeys(_SPLITS, 0),
              "coarse": dict.fromkeys(COARSE_LABELS, 0), "fine": dict.fromkeys(FINE_LABELS, 0),
              "filter_only": 0, "removed_filter": 0, "none": 0,
              "finally_only": 0, "removed_finally": 0,
              "removed_finally_labels": dict.fromkeys(REMOVED_FINALLY_LABELS, 0),
              "fine_label_version": FINE_LABEL_VERSION, "fine_label_order": list(FINE_LABELS),
              "pre_only": 0, "post_only": 0, "empty_fine_modify": 0}
    digest = ""

    def report() -> AuditReport:
        return AuditReport(not issues, digest, file_hashes, counts, issues)

    try:
        data, digest = _json_snapshot(manifest)
        if not isinstance(data, dict):
            raise DataError("manifest must be a JSON object")
    except DataError as exc:
        issues.append(f"manifest: {exc}")
        return report()
    root = Path(manifest).resolve().parent
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        issues.append("schema_version must be 1")
    if data.get("label_version") not in _VERSIONS:
        issues.append(f"label_version must be one of {_VERSIONS}")
    for name in ("split_seed", "sampling_seed"):
        if type(data.get(name)) is not int or data[name] < 0:
            issues.append(f"{name} must be a nonnegative integer")

    def provenance(declaration: dict, context: str) -> dict:
        source = declaration.get("label_source")
        if source not in _SOURCES:
            if formal or source is not None:
                issues.append(f"{context}: provenance label_source must be one of {_SOURCES}")
            return {"label_source": source}
        version_name = "rule_version" if source == "ast_rules" else "annotation_version"
        version = declaration.get(version_name)
        if not _text(version) and (formal or version is not None):
            issues.append(f"{context}: provenance {version_name} must be nonblank")
        return {"label_source": source, version_name: version}

    provenance(data, "manifest")
    repos, groups = data.get("repo_groups"), data.get("group_splits")
    if not isinstance(repos, dict) or any(not _text(k) or not _text(v) for k, v in repos.items()):
        issues.append("repo_groups must map repo_id to nonblank group_id")
        repos = {}
    if not isinstance(groups, dict) or any(not _text(k) or v not in _SPLITS for k, v in groups.items()):
        issues.append("group_splits must map group_id to train/val/test")
        groups = {}
    for group in set(repos.values()):
        if group not in groups:
            issues.append(f"group_splits missing group {group}")
    exceptions, split_policy = _repository_exceptions(data, repos, groups, issues)
    used_exceptions = set()
    counts.update(repository_isolated=not bool(exceptions), repository_exceptions=0,
                  evaluation_distribution=split_policy.get("evaluation_distribution", "natural"))

    def check_hash(entry: dict, context: str) -> tuple[Path, bytes] | None:
        try:
            path = _path(root, entry.get("path"))
            raw, actual = _read_snapshot(path)
            file_hashes[entry["path"]] = actual
            expected = entry.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected) or actual != expected.lower():
                issues.append(f"{context}: sha256 mismatch or missing hash")
            return path, raw
        except (OSError, DataError) as exc:
            issues.append(f"{context}: {exc}")
            return None

    def review(declaration: dict, context: str) -> None:
        if "review_report" in declaration:
            entry = declaration["review_report"]
            if not isinstance(entry, dict):
                issues.append(f"{context}: review_report must declare path and sha256")
            else:
                check_hash(entry, f"{context}: review_report")

    review(data, "manifest")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        issues.append("files must be a nonempty array")
        return report()
    seen_files, seen_groups, seen_sources, seen_pairs = set(), {}, {}, {}

    def across(seen: dict, key: object, split: str, context: str, issue: str) -> None:
        previous = seen.setdefault(key, {})
        for old_split, old_context in previous.items():
            if old_split != split:
                issues.append(f"{issue}: {context} conflicts with {old_context}")
        previous.setdefault(split, context)

    for index, entry in enumerate(files):
        context = f"files[{index}]"
        if not isinstance(entry, dict):
            issues.append(f"{context}: file declaration must be an object")
            continue
        snapshot = check_hash(entry, context)
        if snapshot is None:
            continue
        path, raw = snapshot
        if path in seen_files:
            issues.append(f"{context}: duplicate file path")
        seen_files.add(path)
        split, language = entry.get("split"), entry.get("language")
        if split not in _SPLITS:
            issues.append(f"{context}: split must be train/val/test")
            continue
        if not _text(language):
            if formal or language is not None:
                issues.append(f"{context}: provenance language must be nonblank")
        elif formal and split in ("train", "val") and language != "C#":
            issues.append(f"{context}: formal train/val must declare C#")
        effective = {**data, **entry}
        if "label_version" in entry and entry["label_version"] != data.get("label_version"):
            issues.append(f"{context}: label_version cannot override manifest declaration")
        declared = provenance(effective, context)
        declared["label_version"] = data.get("label_version")
        review(entry, context)
        try:
            rows = _load_rows(path, field_map=effective.get("field_map"), snapshot=raw)
        except DataError as exc:
            issues.append(f"{context}: {exc}")
            continue
        if type(entry.get("n_records")) is not int or entry["n_records"] < 0 or len(rows) != entry["n_records"]:
            issues.append(f"{context}: n_records mismatch or invalid count (actual {len(rows)})")
        for record, row in rows:
            location = f"{entry['path']}:{record.record_id}"
            counts["total"] += 1
            counts["splits"][split] += 1
            targets = record.targets
            for name, positive in zip(COARSE_LABELS, targets.coarse):
                counts["coarse"][name] += int(positive)
            for name, positive in zip(FINE_LABELS, targets.fine):
                counts["fine"][name] += int(positive)
            counts["none"] += int(not any(targets.coarse))
            counts["removed_filter"] += int(targets.removed_filter)
            counts["filter_only"] += int(targets.removed_filter and not any(targets.fine) and not targets.removed_finally)
            counts["removed_finally"] += int(bool(targets.removed_finally))
            counts["finally_only"] += int(bool(targets.removed_finally) and not any(targets.fine) and not targets.removed_filter)
            for name in targets.removed_finally:
                counts["removed_finally_labels"][name] += 1
            has_pre = normalize_code(record.pre_code) is not None
            has_post = normalize_code(record.post_code) is not None
            counts["pre_only"] += int(has_pre and not has_post)
            counts["post_only"] += int(has_post and not has_pre)
            counts["empty_fine_modify"] += int(targets.modify_mask and not any(targets.fine))
            if data.get("label_version") == "eh-change-8" and targets.removed_filter:
                issues.append(f"{location}: label_version eh-change-8 contains legacy catch_filter_changed")
            if data.get("label_version") == FINE_LABEL_VERSION and (targets.removed_filter or targets.removed_finally):
                issues.append(f"{location}: label_version {FINE_LABEL_VERSION} contains removed fine labels")
            for name in ("repo_id", "commit_sha", "source_id", "language"):
                if not _text(getattr(record, name)) and formal:
                    issues.append(f"{location}: missing provenance {name}")
            if record.language is not None and language is not None and record.language != language:
                issues.append(f"{location}: language differs from file declaration")
            for name, expected in declared.items():
                if name in row and row[name] != expected:
                    issues.append(f"{location}: {name} differs from declaration")
            other_version = "annotation_version" if declared["label_source"] == "ast_rules" else "rule_version"
            if other_version in row and row[other_version] is not None:
                issues.append(f"{location}: {other_version} conflicts with label_source")
            exception = exceptions.get(record.record_id)
            allowed_exception = False
            if exception is not None and exception["assigned_split"] == split:
                fine = {name for name, positive in zip(FINE_LABELS, targets.fine) if positive}
                fine.update(targets.removed_finally)
                allowed_exception = (record.repo_id == exception["repo_id"] and record.source_id == exception["source_id"]
                                     and fine == set(exception["fine_labels"]) and targets.coarse[1]
                                     and has_pre and has_post and language == "C#")
                if not allowed_exception:
                    issues.append(f"{location}: fine repository exception differs from its record identity or supervision")
                else:
                    used_exceptions.add(record.record_id)
                    counts["repository_exceptions"] += 1
            if record.repo_id is not None:
                group = repos.get(record.repo_id)
                if group is None:
                    issues.append(f"{location}: repo_groups missing repo_id {record.repo_id}")
                else:
                    if groups.get(group) != split and not allowed_exception:
                        issues.append(f"{location}: group_splits mismatch for {group}")
                    if not allowed_exception:
                        across(seen_groups, group, split, location, "group_overlap")
            if record.source_id is not None:
                across(seen_sources, record.source_id, split, location, "source_overlap")
            # Raw strings and None are separate values. Never trim or normalize.
            pair = (record.pre_code, record.post_code)
            across(seen_pairs, pair, split, location, "exact_pair_overlap")
            if pair[0] != pair[1]:
                for old_split, old_location in seen_pairs.get((pair[1], pair[0]), {}).items():
                    if old_split != split:
                        issues.append(f"reversed_pair_overlap: {location} conflicts with {old_location}")
    for record_id in sorted(set(exceptions) - used_exceptions):
        issues.append(f"unused or unmatched fine repository exception: {record_id}")
    return report()
