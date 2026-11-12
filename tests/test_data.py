import hashlib
import json

import pytest

from model.data.jsonl import load_records
from model.errors import DataError


def write_pairs(path, rows):
    path.write_text("\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_load_filter_only_and_one_sided(tmp_path):
    path = write_pairs(tmp_path / "pairs.jsonl", [
        {"pre_code": " a ", "post_code": "b", "labels": ["Modify"],
         "modify_subtypes": ["catch_filter_changed"], "ast": {"ignored": True}},
        {"pre_code": None, "post_code": "new()", "labels": ["Add"]},
    ])
    records = load_records(path)
    assert len(records) == 2
    assert records[0].pre_code == " a "
    assert records[0].targets.coarse == (0., 1., 0.)
    assert records[0].targets.fine == (0.,) * 5
    assert records[0].targets.removed_filter
    assert records[1].pre_code is None
    assert records[1].targets.coarse == (1., 0., 0.)


def test_missing_subtypes_is_not_empty(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text('{"pre_code":"a","post_code":"b","labels":["Modify"]}\n', encoding="utf-8")
    with pytest.raises(DataError, match="modify_subtypes"):
        load_records(path)


def test_empty_subtypes_are_known_negative(tmp_path):
    row = {"pre_code": "a", "post_code": "b", "labels": ["Modify"], "modify_subtypes": []}
    assert load_records(write_pairs(tmp_path / "x", [row]))[0].targets.fine == (0.,) * 5


def test_json_error_uses_physical_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('\n{"pre_code":"a","post_code":"b","labels":[]}\n\nBAD\n', encoding="utf-8")
    with pytest.raises(DataError, match=r"broken.jsonl:4"):
        load_records(path)


def test_field_map_and_stable_id(tmp_path):
    row = {"before": "a", "after": "b", "origin": "source-1", "repo_full_name": "owner/repo", "labels": []}
    mapping = {"pre_code": "before", "post_code": "after", "source_id": "origin", "repo_id": "repo_full_name"}
    path = write_pairs(tmp_path / "x", [row])
    record = load_records(path, field_map=mapping)[0]
    assert record.repo_id == "owner/repo"
    expected = hashlib.sha256(b'["source-1","a","b"]').hexdigest()
    assert record.record_id == expected
    assert load_records(path, field_map=mapping)[0].record_id == expected
    with pytest.raises(DataError, match="pre_code"):
        load_records(path)


@pytest.mark.parametrize("row,match", [
    ({"post_code": "b", "labels": []}, "pre_code"),
    ({"pre_code": 2, "post_code": "b", "labels": []}, "pre_code"),
    ({"pre_code": None, "post_code": " ", "labels": []}, "absent"),
    ({"pre_code": "a", "post_code": "b"}, "labels"),
    ({"pre_code": "a", "post_code": "b", "labels": [], "repo_id": 2}, "repo_id"),
    ({"pre_code": "a", "post_code": "b", "labels": ["unknown"]}, "labels"),
])
def test_bad_supervision_is_contextual_data_error(tmp_path, row, match):
    with pytest.raises(DataError, match=match):
        load_records(write_pairs(tmp_path / "x", [row]))


def test_prediction_has_no_targets(tmp_path):
    path = write_pairs(tmp_path / "x", [{"pre_code": None, "post_code": "b"}])
    assert load_records(path, require_labels=False)[0].targets is None


def test_duplicate_record_id_is_rejected(tmp_path):
    rows = [{"record_id": "same", "pre_code": "a", "post_code": x, "labels": []} for x in ("b", "c")]
    with pytest.raises(DataError, match="duplicate record_id"):
        load_records(write_pairs(tmp_path / "x", rows))


@pytest.mark.parametrize("mapping", [{"typo": "pre_code"}, {"pre_code": ""}, {"pre_code": 1}])
def test_invalid_field_mapping_fails(tmp_path, mapping):
    with pytest.raises(DataError, match="field_map"):
        load_records(write_pairs(tmp_path / "x", []), field_map=mapping)
