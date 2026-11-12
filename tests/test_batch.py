from dataclasses import fields, replace

import pytest
import torch

from helpers import FakeTokenizer, make_record
from model.config import WindowConfig
from model.errors import InputError
from model.input.batch import collate_pairs
from model.input.types import PairInputs, TargetsBatch
from model.input.windows import prepare_pair


def test_pair_indices_and_gold_separation():
    tokenizer = FakeTokenizer()
    pairs = [prepare_pair(make_record("old" * 1000, "new", labels=["Modify"],
                                     subtypes=["catch_added"]), tokenizer, WindowConfig()),
             prepare_pair(make_record(None, "new", labels=["Add"]), tokenizer, WindowConfig())]
    inputs, targets = collate_pairs(pairs, tokenizer.pad_token_id)
    n = len(pairs[0].windows)
    assert inputs.pair_index.tolist() == [0] * n + [1] * 3
    assert inputs.roles.tolist() == [0, 0, 1, 2, 2] + [0, 1, 2]
    assert inputs.side_presence.tolist() == [[True, True], [False, True]]
    assert inputs.positions.shape == (n + 3, 4)
    for index, window in enumerate(pairs[0].windows + pairs[1].windows):
        assert inputs.input_ids[index, :len(window.input_ids)].tolist() == list(window.input_ids)
        assert inputs.attention_mask[index].sum().item() == len(window.input_ids)
        assert inputs.input_ids[index, len(window.input_ids):].eq(0).all()
        assert inputs.positions[index].tolist() == pytest.approx(window.position)
    assert targets.coarse.tolist() == [[0, 1, 0], [1, 0, 0]]
    assert targets.fine.tolist() == [[0, 1, 0, 0, 0], [0] * 5]
    assert targets.modify_mask.tolist() == [True, False]
    assert {f.name for f in fields(PairInputs)} == {
        "input_ids", "attention_mask", "roles", "positions", "pair_index", "side_presence"}
    assert {f.name for f in fields(TargetsBatch)} == {"coarse", "fine", "modify_mask"}
    assert inputs.input_ids.dtype == inputs.roles.dtype == inputs.pair_index.dtype == torch.int64
    assert inputs.attention_mask.dtype == inputs.side_presence.dtype == torch.bool
    assert inputs.positions.dtype == targets.coarse.dtype == targets.fine.dtype == torch.float32
    assert targets.modify_mask.dtype == torch.bool
    assert all(getattr(inputs, f.name).device.type == "cpu" for f in fields(PairInputs))
    moved_inputs, moved_targets = inputs.to("meta"), targets.to("meta")
    for original, moved in [(inputs, moved_inputs), (targets, moved_targets)]:
        for f in fields(original):
            value, transferred = getattr(original, f.name), getattr(moved, f.name)
            assert transferred.device.type == "meta"
            assert value.shape == transferred.shape and value.dtype == transferred.dtype
    changed = [replace(p, record=replace(p.record, targets=None, language="Java", repo_id="other")) for p in pairs]
    unlabeled, absent = collate_pairs(changed, 0)
    assert absent is None
    assert all(torch.equal(getattr(inputs, f.name), getattr(unlabeled, f.name)) for f in fields(PairInputs))


def test_mixed_supervision_and_empty_batch_rejected():
    pair = prepare_pair(make_record("a", "b", labels=["Modify"]), FakeTokenizer(), WindowConfig())
    with pytest.raises(InputError, match="supervis"):
        collate_pairs([pair, replace(pair, record=replace(pair.record, targets=None))], 0)
    with pytest.raises(InputError, match="empty"):
        collate_pairs([], 0)
