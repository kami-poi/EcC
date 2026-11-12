import pytest

from model.errors import DataError, InputError
from model.labels import FINE_LABELS, REMOVED_FINALLY_LABELS, encode_labels, normalize_code


def test_filter_only_remains_known_negative_modify():
    target = encode_labels(["Modify"], ["catch_filter_changed"], has_pre=True, has_post=True)
    assert target.coarse == (0.0, 1.0, 0.0)
    assert target.fine == (0.0,) * 5
    assert target.modify_mask and target.removed_filter


def test_multilabel_encoding_preserves_fixed_logit_order():
    target = encode_labels(
        ["Delete", "Modify", "Add"],
        ["finally_body_changed", "catch_added", "catch_filter_changed"],
        has_pre=True, has_post=True,
    )
    assert target.coarse == (1.0, 1.0, 1.0)
    assert target.fine == (0.0, 1.0, 0.0, 0.0, 0.0)
    assert target.removed_finally == ("finally_body_changed",)
    assert target.modify_mask and target.removed_filter


@pytest.mark.parametrize("labels", [[], ["None"]])
def test_none_encodes_all_negative(labels):
    target = encode_labels(labels, None, has_pre=True, has_post=True)
    assert target.coarse == (0.0, 0.0, 0.0)
    assert target.fine == (0.0,) * 5
    assert not target.modify_mask and not target.removed_filter


def test_explicit_empty_subtypes_are_known_negatives():
    target = encode_labels(["Modify"], [], has_pre=True, has_post=True)
    assert target.modify_mask and not target.removed_filter
    assert target.fine == (0.0,) * 5


@pytest.mark.parametrize("subtype,want", [
    ("try_body_changed", (1.0, 0.0, 0.0, 0.0, 0.0)),
    ("catch_added", (0.0, 1.0, 0.0, 0.0, 0.0)),
    ("catch_removed", (0.0, 0.0, 1.0, 0.0, 0.0)),
    ("catch_type_changed", (0.0, 0.0, 0.0, 1.0, 0.0)),
    ("catch_body_changed", (0.0, 0.0, 0.0, 0.0, 1.0)),
])
def test_each_subtype_uses_its_declared_logit_position(subtype, want):
    assert encode_labels(["Modify"], [subtype], has_pre=True, has_post=True).fine == want


@pytest.mark.parametrize("subtypes", [[label] for label in REMOVED_FINALLY_LABELS] + [list(REMOVED_FINALLY_LABELS)])
def test_finally_only_projects_to_known_negative_fine_without_changing_coarse(subtypes):
    target = encode_labels(["Modify"], subtypes, has_pre=True, has_post=True)
    assert target.coarse == (0., 1., 0.) and target.modify_mask
    assert target.fine == (0.,) * len(FINE_LABELS)
    assert target.removed_finally == tuple(subtypes)
    assert not target.removed_filter


@pytest.mark.parametrize("labels,subtypes", [
    (["None", "Add"], None), (["Add", "Add"], None), (["Other"], None),
    (["Modify"], None), (["Modify"], ["unknown"]),
    (["Modify"], ["catch_added", "catch_added"]),
    (["Modify"], ["catch_filter_changed", "catch_filter_changed"]),
    (["Add"], ["catch_added"]), (["None"], ["catch_filter_changed"]),
    (["None"], ["finally_added"]), (["Modify"], ["finally_removed", "finally_removed"]),
    ("Add", None), ([1], None), (["Modify"], "catch_added"),
])
def test_malformed_supervision_is_rejected(labels, subtypes):
    with pytest.raises(DataError):
        encode_labels(labels, subtypes, has_pre=True, has_post=True)


@pytest.mark.parametrize("has_pre,has_post,legal", [(False, True, "Add"), (True, False, "Delete")])
def test_single_sided_supervision_allows_only_directional_label(has_pre, has_post, legal):
    target = encode_labels([legal], [], has_pre=has_pre, has_post=has_post)
    assert target.coarse == ((1.0, 0.0, 0.0) if legal == "Add" else (0.0, 0.0, 1.0))
    for label in {"Add", "Modify", "Delete"} - {legal}:
        with pytest.raises(DataError):
            encode_labels([label], [] if label == "Modify" else None, has_pre=has_pre, has_post=has_post)


def test_both_absent_is_invalid_input():
    with pytest.raises(InputError):
        encode_labels(["None"], None, has_pre=False, has_post=False)


def test_presence_flags_require_booleans():
    with pytest.raises(InputError):
        encode_labels([], None, has_pre=1, has_post=True)


@pytest.mark.parametrize("value", [None, "", " \r\n", "\t\u2003"])
def test_blank_code_is_absent(value):
    assert normalize_code(value) is None


@pytest.mark.parametrize("value", ["{}", " \r\n{}\t", "// Unicode 中文\r\n{}"])
def test_normalization_preserves_nonblank_code_verbatim(value):
    assert normalize_code(value) == value


@pytest.mark.parametrize("value", [0, {}, []])
def test_nonstring_code_is_rejected(value):
    with pytest.raises(InputError):
        normalize_code(value)
