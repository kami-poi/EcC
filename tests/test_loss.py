"""Joint BCE uses pair-level denominators, including filter-only gold."""

import math

import pytest
import torch

from model.input.types import TargetsBatch
from model.labels import encode_labels
from model.models.base import PairLogits
from model.train.loss import compute_loss


def test_filter_only_contributes_negative_fine_loss():
    gold = encode_labels(["Modify"], ["catch_filter_changed"], has_pre=True, has_post=True)
    logits = PairLogits(torch.zeros(1, 3, requires_grad=True), torch.zeros(1, 5, requires_grad=True))
    targets = TargetsBatch(torch.tensor([gold.coarse]), torch.tensor([gold.fine]), torch.tensor([gold.modify_mask]))
    parts = compute_loss(logits, targets, n_total=5, n_modify=2, lambda_fine=2)
    assert parts.coarse.item() == pytest.approx(math.log(2) / 5)
    assert parts.fine.item() == pytest.approx(math.log(2) / 2)
    assert parts.total.item() == pytest.approx(math.log(2) * 1.2)
    parts.total.backward()
    torch.testing.assert_close(logits.fine.grad, torch.full((1, 5), 1 / 10))


def test_fine_supervision_masks_non_modify_pairs_before_reduction():
    logits = PairLogits(torch.zeros(2, 3, requires_grad=True),
                        torch.tensor([[0.] * 5, [20.] * 5], requires_grad=True))
    targets = TargetsBatch(torch.zeros(2, 3), torch.zeros(2, 5), torch.tensor([True, False]))
    parts = compute_loss(logits, targets, n_total=5, n_modify=2)
    assert parts.coarse.item() == pytest.approx(2 * math.log(2) / 5)
    assert parts.fine.item() == pytest.approx(math.log(2) / 2)
    parts.total.backward()
    torch.testing.assert_close(logits.fine.grad[0], torch.full((5,), 1 / 20))
    torch.testing.assert_close(logits.fine.grad[1], torch.zeros(5))


@pytest.mark.parametrize("n_modify", [0, 2])
def test_zero_modify_retains_finite_gradient_graph(n_modify):
    logits = PairLogits(torch.zeros(2, 3, requires_grad=True), torch.randn(2, 5, requires_grad=True))
    targets = TargetsBatch(torch.zeros(2, 3), torch.zeros(2, 5), torch.zeros(2, dtype=torch.bool))
    parts = compute_loss(logits, targets, n_total=5, n_modify=n_modify)
    assert parts.fine.item() == 0
    parts.total.backward()
    assert torch.isfinite(logits.coarse.grad).all()
    torch.testing.assert_close(logits.fine.grad, torch.zeros(2, 5))


@pytest.mark.parametrize("n_total,n_modify", [(0, 0), (1, -1), (1, 2), (1, 0)])
def test_invalid_global_denominators_are_rejected(n_total, n_modify):
    logits = PairLogits(torch.zeros(1, 3), torch.zeros(1, 5))
    targets = TargetsBatch(torch.zeros(1, 3), torch.zeros(1, 5), torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError):
        compute_loss(logits, targets, n_total=n_total, n_modify=n_modify)
