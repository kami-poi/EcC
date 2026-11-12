import pytest
import torch

from model.encode.pooling import last_valid_pool


def test_left_and_right_padding_select_last_valid():
    hidden = torch.arange(8.0).reshape(2, 4, 1)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    assert last_valid_pool(hidden, mask).flatten().tolist() == [3.0, 5.0]


def test_holes_and_gradients_select_the_actual_last_token():
    hidden = torch.arange(8.0).reshape(2, 4, 1).requires_grad_()
    last_valid_pool(hidden, torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]])).sum().backward()
    assert hidden.grad.flatten().tolist() == [0, 0, 1, 0, 0, 1, 0, 0]


@pytest.mark.parametrize("mask", [torch.zeros(2, 4), torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])])
def test_all_zero_mask_is_rejected(mask):
    with pytest.raises(ValueError, match="valid"):
        last_valid_pool(torch.zeros(2, 4, 3), mask)


def test_invalid_mask_and_shapes_are_rejected():
    with pytest.raises(ValueError):
        last_valid_pool(torch.zeros(2, 4, 3), torch.ones(2, 3))
    with pytest.raises(ValueError):
        last_valid_pool(torch.zeros(2, 4, 3), torch.full((2, 4), 2))
