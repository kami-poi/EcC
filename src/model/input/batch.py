"""Flatten source windows, keeping all supervision in a separate target batch."""

from collections.abc import Sequence

import torch

from model.errors import InputError
from model.labels import normalize_code

from .types import ROLE_IDS, PairInputs, PreparedPair, TargetsBatch


def collate_pairs(pairs: Sequence[PreparedPair], pad_token_id: int) -> tuple[PairInputs, TargetsBatch | None]:
    if not pairs:
        raise InputError("cannot collate an empty pair batch")
    supervised = [pair.record.targets is not None for pair in pairs]
    if any(supervised) and not all(supervised):
        raise InputError("cannot mix supervised and unsupervised pairs in one batch")
    if any(not pair.windows for pair in pairs):
        raise InputError("each pair must have at least one window")
    windows = [window for pair in pairs for window in pair.windows]
    width = max(len(window.input_ids) for window in windows)
    ids = torch.full((len(windows), width), pad_token_id, dtype=torch.int64)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, window in enumerate(windows):
        length = len(window.input_ids)
        ids[index, :length] = torch.tensor(window.input_ids, dtype=torch.int64)
        mask[index, :length] = True
    inputs = PairInputs(
        input_ids=ids, attention_mask=mask,
        roles=torch.tensor([ROLE_IDS[window.role] for window in windows], dtype=torch.int64),
        positions=torch.tensor([window.position for window in windows], dtype=torch.float32),
        pair_index=torch.tensor([index for index, pair in enumerate(pairs) for _ in pair.windows],
                                dtype=torch.int64),
        side_presence=torch.tensor([(normalize_code(pair.record.pre_code) is not None,
                                     normalize_code(pair.record.post_code) is not None) for pair in pairs],
                                    dtype=torch.bool),
    )
    if not all(supervised):
        return inputs, None
    targets = [pair.record.targets for pair in pairs]
    return inputs, TargetsBatch(
        coarse=torch.tensor([target.coarse for target in targets], dtype=torch.float32),
        fine=torch.tensor([target.fine for target in targets], dtype=torch.float32),
        modify_mask=torch.tensor([target.modify_mask for target in targets], dtype=torch.bool),
    )
