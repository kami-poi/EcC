from dataclasses import replace

import pytest

from helpers import FakeTokenizer, make_record
from model.config import RunConfig, WindowConfig
from model.errors import ConfigError, InputError
from model.input.windows import (
    HFTokenizerAdapter, prepare_model_input, prepare_pair, validate_tokenizer_budget,
)


def assert_complete(prepared, tokenizer, texts):
    for role, text in texts.items():
        ids, _ = tokenizer.tokenize(text)
        seen = set()
        for window in prepared.windows:
            if window.role != role:
                continue
            start, end = window.content_range
            seen.update(range(start, end))
            assert window.input_ids == tokenizer.wrap_ids(ids[start:end], role)
            assert window.text == text[slice(*window.character_range)]
        assert seen == set(range(len(ids)))


def test_wrapped_budget_and_unique_coverage():
    tokenizer = FakeTokenizer()
    prepared = prepare_pair(make_record("a" * 5000, "b" * 5000, labels=["Modify"],
                                       subtypes=["try_body_changed"]), tokenizer, WindowConfig())
    assert prepared.coverage == (1.0, 1.0, 1.0)
    assert all(len(window.input_ids) <= 2048 for window in prepared.windows)
    assert {window.role for window in prepared.windows} == {"PRE", "POST", "DIFF"}
    assert_complete(prepared, tokenizer, {"PRE": "a" * 5000, "POST": "b" * 5000,
                                        "DIFF": prepared.diff.text})
    pre = [w for w in prepared.windows if w.role == "PRE"]
    assert pre[0].content_range[1] - pre[1].content_range[0] == 256
    assert len(pre[0].input_ids) == 2048
    assert prepared.token_counts == (5000, 5000, len(prepared.diff.text))


def test_long_single_line_unicode_tail():
    pre = "}\u4e2d\U0001f40d" * 2000 + "old"
    post = "}\u4e2d\U0001f40d" * 2000 + "new"
    prepared = prepare_pair(make_record(pre, post, labels=["Modify"]), FakeTokenizer(), WindowConfig())
    assert_complete(prepared, FakeTokenizer(), {"PRE": pre, "POST": post, "DIFF": prepared.diff.text})
    tail = [w for w in prepared.windows if w.role == "POST"][-1]
    assert tail.text.endswith("new")
    assert tail.post_lines == (0, 1) and tail.position == (0.0, 0.0, 0.0, 1.0)
    assert prepared.source_lengths == (6003, 6003)


def test_all_hunks_in_diff():
    pre = "old_head\n" + "}\n" * 400 + "old_tail"
    post = pre.replace("old", "new")
    prepared = prepare_pair(make_record(pre, post, labels=["Modify"]), FakeTokenizer(),
                            WindowConfig(window_tokens=160, overlap_tokens=10))
    assert prepared.diff.hunk_count == 2
    assert prepared.change_positions == ((0.0, 1 / 402, 0.0, 0.0),
                                         (1 / 402, 1 / 402, 0.0, 1 / 402),
                                         (401 / 402, 1.0, 401 / 402, 401 / 402),
                                         (1.0, 1.0, 401 / 402, 1.0))
    assert_complete(prepared, FakeTokenizer(), {"DIFF": prepared.diff.text})
    assert any(w.pre_lines[1] == 402 for w in prepared.windows if w.role == "DIFF")


def test_empty_side_and_neutral_diff():
    absent = prepare_pair(make_record(" \r\n", "new", labels=["Add"]), FakeTokenizer(), WindowConfig())
    pre = [w for w in absent.windows if w.role == "PRE"]
    assert len(pre) == 1 and pre[0].text == "EMPTY"
    assert pre[0].pre_lines == (0, 0)
    assert pre[0].is_placeholder and pre[0].content_range == (0, 5)
    assert pre[0].character_range == (0, 5)
    assert absent.token_counts[0] == 0 and absent.coverage == (1.0, 1.0, 1.0)
    neutral = prepare_pair(make_record("same\r\n", "same\r\n", labels=["None"]),
                           FakeTokenizer(), WindowConfig())
    assert neutral.diff.text == "NO_TEXT_CHANGE\n" and neutral.change_positions == ()
    assert neutral.windows[-1].position == (0.0, 0.0, 0.0, 0.0)


def test_raw_blank_side_diff_positions_stay_normalized():
    pair = prepare_pair(make_record(" \n \n", "new", labels=["Add"]), FakeTokenizer(), WindowConfig())
    assert "DEL  \nDEL  \n" in pair.diff.text
    assert all(0 <= value <= 1 for w in pair.windows for value in w.position)
    assert all(0 <= value <= 1 for position in pair.change_positions for value in position)
    assert pair.token_counts[0] == 0 and pair.source_lengths == (4, 3)


def test_token_slices_preserve_contextual_ids():
    class ContextTokenizer(FakeTokenizer):
        def tokenize(self, text):
            return tuple(10000 + i for i in range(len(text))), tuple((i, i + 1) for i in range(len(text)))
    pair = prepare_pair(make_record("a" * 5000, "b" * 5000, labels=["Modify"]),
                        ContextTokenizer(), WindowConfig())
    assert_complete(pair, ContextTokenizer(), {"PRE": "a" * 5000, "POST": "b" * 5000,
                                             "DIFF": pair.diff.text})


@pytest.mark.parametrize("offsets", [(), ((0, 99),), ((1, 0),)])
def test_unavailable_or_invalid_offsets_fail(offsets):
    class BadOffsets(FakeTokenizer):
        def tokenize(self, text):
            return (42,), offsets
    with pytest.raises(ConfigError, match="offset"):
        prepare_pair(make_record("a", "b", labels=["Modify"]), BadOffsets(), WindowConfig())


@pytest.mark.parametrize("mode,roles,counts,coverage", [
    ("full_only", {"PRE", "POST"}, (3, 3, 38), (1.0, 1.0, None)),
    ("diff_only", {"DIFF"}, (3, 3, 38), (None, None, 1.0)),
])
def test_modes_and_shared_entry(mode, roles, counts, coverage):
    cfg = RunConfig(window=WindowConfig(input_mode=mode))
    pair = prepare_model_input(make_record("old", "new", labels=["Modify"]), FakeTokenizer(), cfg)
    assert pair.input_mode == mode and {w.role for w in pair.windows} == roles
    assert pair.token_counts == counts and pair.coverage == coverage


def test_wrapper_exhausts_budget():
    with pytest.raises(ConfigError, match="wrapper"):
        validate_tokenizer_budget(FakeTokenizer(), WindowConfig(window_tokens=100, overlap_tokens=0))


def test_overlap_exceeds_content_budget():
    with pytest.raises(ConfigError, match="overlap"):
        validate_tokenizer_budget(FakeTokenizer(), WindowConfig(window_tokens=150, overlap_tokens=40))


def test_source_line_offsets_and_gold_independence():
    record = make_record("a\r\nb\u2028c", "a\r\nb\u2028d", labels=["Modify"])
    cfg = WindowConfig()
    a = prepare_pair(record, FakeTokenizer(), cfg)
    b = prepare_pair(replace(record, targets=None, repo_id="different", language="Java"), FakeTokenizer(), cfg)
    assert a.windows == b.windows
    pre = [w for w in a.windows if w.role == "PRE"]
    assert pre[0].pre_lines == (0, 3)
    with pytest.raises(InputError, match="both"):
        prepare_pair(replace(record, pre_code=None, post_code=" "), FakeTokenizer(), cfg)


def test_hf_fast_offsets_and_separate_wrapping():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "old": 2, "new": 3,
                                 "[BOS]": 4, "[EOS]": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A [EOS]",
                                                 special_tokens=[("[BOS]", 4), ("[EOS]", 5)])
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    adapter = HFTokenizerAdapter(fast)
    ids, offsets = adapter.tokenize("old 新🐍 new")
    assert ids == (2, 0, 0, 3) and offsets == ((0, 3), (4, 5), (5, 6), (7, 10))
    pair = prepare_pair(make_record("old", "new", labels=["Modify"]), adapter, WindowConfig())
    window = pair.windows[0]
    prefix = fast(window.prefix, add_special_tokens=False)["input_ids"]
    suffix = fast(window.suffix, add_special_tokens=False)["input_ids"]
    assert window.input_ids == (4, *prefix, 2, *suffix, 5)
    assert window.input_ids == tuple(fast(window.prefix + "old" + window.suffix,
                                         add_special_tokens=True)["input_ids"])
    with pytest.raises(ConfigError, match="fast"):
        HFTokenizerAdapter(replace_like_slow())


def replace_like_slow():
    class Slow:
        is_fast = False
    return Slow()


@pytest.mark.parametrize("inconsistent", [False, True])
def test_hf_rejects_noncontinuous_or_inconsistent_special_template(inconsistent):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[SPECIAL]": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[SPECIAL] $A" if inconsistent else "$A [SPECIAL] $A",
        special_tokens=[("[SPECIAL]", 2)])
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")

    class ContentDependentTemplate:
        def __getattr__(self, name):
            return getattr(fast, name)

        def __call__(self, text, **kwargs):
            encoded = fast(text, **kwargs)
            if kwargs.get("add_special_tokens") and len(text) > 1:
                encoded["input_ids"].insert(0, 2)
                encoded["special_tokens_mask"].insert(0, 1)
            return encoded

    with pytest.raises(ConfigError, match="template"):
        HFTokenizerAdapter(ContentDependentTemplate() if inconsistent else fast)
