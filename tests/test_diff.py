from time import perf_counter

import pytest

from model.input.diff import make_diff


def test_existing_try_catch_edit_is_only_text_alignment():
    before = "try {\n  old();\n} catch (E e) {\n  recover();\n}\n"
    after = "try {\n  new();\n} catch (E e) {\n  recover();\n}\n"
    diff = make_diff(before, after)
    assert diff.hunk_count == 1
    assert [span.kind for span in diff.spans] == ["CTX", "DEL", "ADD", "CTX", "CTX", "CTX"]
    deleted, added = diff.spans[1:3]
    assert diff.text[slice(*deleted.content_range)] == "  old();\n"
    assert diff.text[slice(*added.content_range)] == "  new();\n"
    assert deleted.pre_lines == (1, 2) and deleted.post_lines == (1, 1)
    assert added.pre_lines == (2, 2) and added.post_lines == (1, 2)
    renamed = make_diff(before.replace("try", "xyz").replace("catch", "abcde"),
                        after.replace("try", "xyz").replace("catch", "abcde"))
    assert diff.spans == renamed.spans
    assert diff.text.replace("try", "xyz").replace("catch", "abcde") == renamed.text


def test_merge_overlapping_context():
    before = "".join(f"line {i}\n" for i in range(16))
    after = before.replace("line 4\n", "changed 4\n").replace("line 9\n", "changed 9\n")
    diff = make_diff(before, after)
    assert diff.hunk_count == 1
    assert diff.text.startswith("HUNK 1:13 1:13\n")
    assert [span.pre_lines for span in diff.spans if span.kind == "CTX"] == [
        (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13),
    ]
    assert make_diff(before, after, context_lines=1).hunk_count == 2


def test_repeated_lines_tail_change(record_property):
    before = "}\n" * 400 + "old_tail();"
    after = "}\n" * 400 + "new_tail();"
    start = perf_counter()
    diff = make_diff(before, after)
    record_property("repeated_lines_diff_seconds", perf_counter() - start)
    assert "old_tail();" in diff.text and "new_tail();" in diff.text
    assert diff.hunk_count == 1
    assert any(span.post_lines[1] == 401 for span in diff.spans)
    assert [span.kind for span in diff.spans] == ["CTX", "CTX", "CTX", "DEL", "ADD"]


def test_repeated_lines_between_changes_remain_context():
    before = "old_head\n" + "}\n" * 400 + "old_tail"
    after = "new_head\n" + "}\n" * 400 + "new_tail"
    diff = make_diff(before, after)
    assert diff.hunk_count == 2
    assert len(diff.spans) == 10
    assert [span.pre_lines for span in diff.spans if span.kind == "CTX"] == [
        (1, 2), (2, 3), (3, 4), (398, 399), (399, 400), (400, 401),
    ]


@pytest.mark.parametrize("before,after,kind,pre_lines,post_lines", [
    (None, "new", "ADD", (0, 0), (0, 1)),
    ("old", None, "DEL", (0, 1), (0, 0)),
    ("", " \n", "ADD", (0, 0), (0, 1)),
])
def test_one_sided_and_no_change_markers(before, after, kind, pre_lines, post_lines):
    diff = make_diff(before, after)
    assert diff.hunk_count == 1
    assert len(diff.spans) == 1
    span = diff.spans[0]
    assert (span.kind, span.pre_lines, span.post_lines) == (kind, pre_lines, post_lines)
    assert diff.text[slice(*span.content_range)] == (after if kind == "ADD" else before)
    for unchanged in [None, "", " \r\n", "try { unchanged(); }"]:
        neutral = make_diff(unchanged, unchanged)
        assert neutral.text == "NO_TEXT_CHANGE\n"
        assert neutral.spans == () and neutral.hunk_count == 0
    assert make_diff(None, "").text == "NO_TEXT_CHANGE\n"


@pytest.mark.parametrize("before,after,kind,pre_lines,post_lines,payload", [
    ("head\ntail", "head\ninsert\ntail", "ADD", (1, 1), (1, 2), "insert\n"),
    ("head\nremove\ntail", "head\ntail", "DEL", (1, 2), (1, 1), "remove\n"),
    ("head\n", "head\ninsert", "ADD", (1, 1), (1, 2), "insert"),
    ("head\nremove", "head\n", "DEL", (1, 2), (1, 1), "remove"),
])
def test_zero_width_line_boundaries(before, after, kind, pre_lines, post_lines, payload):
    diff = make_diff(before, after, context_lines=0)
    assert len(diff.spans) == 1
    span = diff.spans[0]
    assert (span.kind, span.pre_lines, span.post_lines) == (kind, pre_lines, post_lines)
    assert diff.text[slice(*span.content_range)] == payload
    assert diff.text[slice(*span.diff_range)] == f"{kind} {payload}" + ("" if payload.endswith("\n") else "\n")


def test_keyword_decoys_and_unicode_preserved():
    before = '// try catch finally 假提示\r\nstring s = "catch 🐍";\r\n旧行();'
    after = '// try catch finally 假提示\r\nstring s = "catch 🐍";\r\n新行();'
    diff = make_diff(before, after)
    assert diff.text == (
        'HUNK 0:3 0:3\nCTX // try catch finally 假提示\r\n'
        'CTX string s = "catch 🐍";\r\nDEL 旧行();\nADD 新行();\nEND_HUNK\n'
    )
    assert [diff.text[slice(*span.content_range)] for span in diff.spans] == [
        '// try catch finally 假提示\r\n', 'string s = "catch 🐍";\r\n', '旧行();', '新行();',
    ]
    assert diff.spans[0].diff_range == (13, 43)
    assert diff.spans[0].content_range == (17, 43)
    assert diff.spans[1].content_range == (47, 70)
    assert diff.spans[2].content_range == (74, 79)
    assert diff.spans[3].content_range == (84, 89)


def test_line_ending_only_change_and_marker_like_source_preserved():
    diff = make_diff("ADD fake\r\nNO_TEXT_CHANGE", "ADD fake\nNO_TEXT_CHANGE")
    assert [span.kind for span in diff.spans] == ["DEL", "ADD", "CTX"]
    assert [diff.text[slice(*span.content_range)] for span in diff.spans] == [
        "ADD fake\r\n", "ADD fake\n", "NO_TEXT_CHANGE",
    ]


def test_all_changed_regions_and_long_single_line_are_retained():
    before = "a\n" + "x\n" * 20 + "中" * 10000 + " old"
    after = "b\n" + "x\n" * 20 + "中" * 10000 + " new"
    diff = make_diff(before, after)
    assert diff.hunk_count == 2
    changed = [diff.text[slice(*span.content_range)] for span in diff.spans if span.kind != "CTX"]
    assert changed == ["a\n", "b\n", "中" * 10000 + " old", "中" * 10000 + " new"]


@pytest.mark.parametrize("context", [-1, 1.5, True, "3", None])
def test_invalid_context_rejected(context):
    with pytest.raises(ValueError, match="context_lines"):
        make_diff("a", "b", context_lines=context)
