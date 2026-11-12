"""Explicit top-level JSONL adaptation with physical-line diagnostics."""

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

from model.data.schema import PairRecord
from model.errors import DataError, InputError
from model.labels import encode_labels, normalize_code

_FIELDS = {"record_id", "pre_code", "post_code", "labels", "modify_subtypes",
           "repo_id", "commit_sha", "source_id", "language"}


def _mapping(field_map: Mapping[str, str] | None) -> dict[str, str]:
    if field_map is None:
        return {}
    if not isinstance(field_map, Mapping) or any(
        key not in _FIELDS or not isinstance(value, str) or not value.strip()
        for key, value in field_map.items()
    ):
        raise DataError("field_map must map canonical field names to nonblank top-level input names")
    return dict(field_map)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _record(row: dict, mapping: dict, require_labels: bool) -> PairRecord:
    value = {name: row[mapping.get(name, name)] for name in _FIELDS if mapping.get(name, name) in row}
    for name in ("pre_code", "post_code"):
        if name not in value:
            raise DataError(f"missing required source field: {name}")
        if value[name] is not None and not isinstance(value[name], str):
            raise DataError(f"{name} must be a string or null")
    pre, post = value["pre_code"], value["post_code"]
    has_pre, has_post = normalize_code(pre) is not None, normalize_code(post) is not None
    if not has_pre and not has_post:
        raise DataError("both source-code sides are absent")
    for name in ("record_id", "repo_id", "commit_sha", "source_id", "language"):
        item = value.get(name)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise DataError(f"{name} must be a nonblank string or null")
    targets = None
    if require_labels:
        if "labels" not in value:
            raise DataError("missing required labels")
        targets = encode_labels(value["labels"], value.get("modify_subtypes"), has_pre=has_pre, has_post=has_post)
    source = value.get("source_id")
    record_id = value.get("record_id") or hashlib.sha256(
        json.dumps([source, pre, post], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PairRecord(record_id, pre, post, targets, value.get("repo_id"),
                      value.get("commit_sha"), source, value.get("language"))


def _read_snapshot(path: Path) -> tuple[bytes, str]:
    """Read once; the digest identifies exactly the bytes consumed by parsers."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise DataError(f"cannot read data {path}: {exc}") from exc
    return raw, hashlib.sha256(raw).hexdigest()


def _json_snapshot(path: Path) -> tuple[object, str]:
    raw, digest = _read_snapshot(path)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object), digest
    except (UnicodeError, json.JSONDecodeError, DataError) as exc:
        raise DataError(f"cannot parse JSON {path}: {exc}") from exc


def _load_rows(path: Path, *, require_labels: bool = True,
               field_map: Mapping[str, str] | None = None,
               snapshot: bytes | None = None) -> list[tuple[PairRecord, dict]]:
    mapping = _mapping(field_map)
    records, seen = [], set()
    try:
        raw = _read_snapshot(path)[0] if snapshot is None else snapshot
        # TextIO preserves the original CR/LF universal-newline behavior. Unlike
        # str.splitlines(), it leaves U+2028 and U+0085 inside JSON source strings.
        with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline=None) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_pairs_hook=_unique_object)
                    if not isinstance(row, dict):
                        raise DataError("record must be a JSON object")
                    record = _record(row, mapping, require_labels)
                    if record.record_id in seen:
                        raise DataError(f"duplicate record_id: {record.record_id}")
                    seen.add(record.record_id)
                    records.append((record, row))
                except (json.JSONDecodeError, DataError, InputError) as exc:
                    raise DataError(f"{path}:{line_number}: {exc}") from exc
    except (OSError, UnicodeError) as exc:
        raise DataError(f"cannot read records {path}: {exc}") from exc
    return records


def load_records(path: Path, *, require_labels: bool = True,
                 field_map: Mapping[str, str] | None = None) -> list[PairRecord]:
    """Load raw sources; unlabeled prediction explicitly returns targets=None."""
    return [record for record, _ in _load_rows(path, require_labels=require_labels, field_map=field_map)]


def load_records_snapshot(path: Path, *, require_labels: bool = True,
                          field_map: Mapping[str, str] | None = None) -> tuple[list[PairRecord], str]:
    """Return normalized records and the digest of their immutable input bytes."""
    raw, digest = _read_snapshot(path)
    rows = _load_rows(path, require_labels=require_labels, field_map=field_map, snapshot=raw)
    return [record for record, _ in rows], digest
