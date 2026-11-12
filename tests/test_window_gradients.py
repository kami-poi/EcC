import pytest
import torch
from torch import nn

from helpers import FakeWindowEncoder
from model.encode.checkpoint import encode_windows


def windows():
    return torch.tensor([[1, 101, 2, 0], [0, 1, 0x1F600 + 10, 2], [1, 99, 100, 2]]), torch.tensor(
        [[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)


def test_frozen_inputs_still_train_adapter():
    encoder = FakeWindowEncoder()
    ids, mask = windows()
    output = encode_windows(encoder, ids, mask)
    output.square().sum().backward()
    assert output.shape == (3, 8)
    assert encoder.embedding.weight.grad is None
    assert encoder.adapter.weight.grad.abs().sum() > 0


def test_checkpoint_actually_recomputes_each_trainable_microbatch():
    class CountingEncoder(FakeWindowEncoder):
        calls = 0

        def forward(self, ids, mask):
            self.calls += 1
            return super().forward(ids, mask)

    ids, mask = windows()
    encoder = CountingEncoder()
    output = encode_windows(encoder, ids, mask, microbatch=2)
    assert encoder.calls == 2
    output.square().sum().backward()
    assert encoder.calls == 4


@pytest.mark.parametrize("microbatch", [1, 2, 5])
def test_checkpoint_preserves_dropout_gradients(microbatch):
    ids, mask = windows()
    outputs, gradients = [], []
    for recompute in (False, True):
        encoder = FakeWindowEncoder(dropout=0.4)
        torch.manual_seed(91)
        output = encode_windows(encoder, ids, mask, microbatch=microbatch, checkpoint=recompute)
        output.square().sum().backward()
        outputs.append(output.detach())
        gradients.append(encoder.adapter.weight.grad)
        assert encoder.embedding.weight.grad is None
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-5, atol=1e-6)
    assert gradients[0].abs().sum() > 0


def test_frozen_encoder_has_only_downstream_gradients():
    encoder = FakeWindowEncoder(adaptation="frozen")
    downstream = nn.Sequential(encoder, nn.Linear(8, 1), nn.Dropout(0.3))
    downstream.train()
    ids, mask = windows()
    summaries = encode_windows(encoder, ids, mask)
    downstream[1](summaries).sum().backward()
    assert not summaries.requires_grad
    assert all(param.grad is None for param in encoder.parameters())
    assert downstream[1].weight.grad.abs().sum() > 0
    assert downstream[1].training and downstream[2].training
    assert not encoder.embedding.training and not encoder.dropout.training


def test_microbatch_validation_and_empty_windows():
    ids, mask = windows()
    encoder = FakeWindowEncoder()
    with pytest.raises(ValueError):
        encode_windows(encoder, ids, mask, microbatch=0)
    with pytest.raises(ValueError):
        encode_windows(encoder, ids, mask[:, :2])
    assert encode_windows(encoder, ids[:0], mask[:0]).shape == (0, 8)
