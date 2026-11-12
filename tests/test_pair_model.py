"""End-to-end classification keeps raw labels and tail-window gradients."""

from dataclasses import replace

import pytest
import torch

from model.config import EncoderConfig, ModelConfig, RunConfig, WindowConfig
from model.input.types import PairInputs
from model.models.base import PairLogits, PairModel
from model.models.pair import HierarchicalClassifier, build_model
from helpers import FakeWindowEncoder, fake_backend_factory


def config(aggregation="transformer"):
    return RunConfig(encoder=EncoderConfig(device="cpu", dtype="fp32"),
                     model=ModelConfig(native_dim=8, projection_dim=16, ffn_dim=32,
                                       dropout=0.1, aggregation=aggregation),
                     window=WindowConfig(window_microbatch=2, window_checkpoint=True))


def inputs(counts=(3, 9)):
    n = sum(counts)
    return PairInputs(torch.arange(1, n * 4 + 1).reshape(n, 4),
                      torch.tensor([[True, True, False, False]] * n),
                      torch.arange(n).remainder(3), torch.linspace(0, 1, n * 4).reshape(n, 4),
                      torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)),
                      torch.ones(len(counts), 2, dtype=torch.bool))


@pytest.mark.parametrize("aggregation", ["transformer", "mean"])
def test_output_shapes_and_adapter_gradient(aggregation):
    torch.manual_seed(17)
    cfg = config(aggregation)
    backend = fake_backend_factory(cfg)
    model = build_model(backend.encoder, cfg)
    assert isinstance(model, PairModel)
    summaries = []

    def capture(module, args):
        args[0].retain_grad()
        summaries.append(args[0])

    handle = model.fusion.register_forward_pre_hook(capture)
    result = model(inputs())
    handle.remove()
    assert isinstance(result, PairLogits)
    assert result.coarse.shape == (2, 3)
    assert result.fine.shape == (2, 5)
    loss = result.coarse.square().sum() + result.fine.square().sum()
    loss.backward()
    assert summaries[0].shape == (12, 8)
    assert torch.count_nonzero(summaries[0].grad[-1]) > 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad) > 0, name
        else:
            assert parameter.grad is None, name
    assert not backend.encoder.embedding.weight.requires_grad


@pytest.mark.parametrize("aggregation", ["transformer", "mean"])
def test_padding_and_other_pair_do_not_affect_prediction(aggregation):
    torch.manual_seed(8)
    cfg = config(aggregation)
    model = build_model(fake_backend_factory(cfg).encoder, cfg).eval()
    batch = inputs()
    single = PairInputs(batch.input_ids[:3], batch.attention_mask[:3], batch.roles[:3],
                        batch.positions[:3], batch.pair_index[:3], batch.side_presence[:1])
    with torch.no_grad():
        alone = model(single)
        together = model(batch)
        changed_ids = batch.input_ids.clone()
        changed_ids[~batch.attention_mask] = 123456
        changed_ids[3:] = 987654
        changed = model(replace(batch, input_ids=changed_ids))
    for field in ("coarse", "fine"):
        torch.testing.assert_close(getattr(together, field)[:1], getattr(alone, field), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(getattr(changed, field)[:1], getattr(alone, field), atol=2e-6, rtol=2e-6)


def test_roles_and_positions_change_predictions_and_outputs_are_raw():
    torch.manual_seed(21)
    cfg = config("mean")
    model = build_model(fake_backend_factory(cfg).encoder, cfg).eval()
    batch = inputs((1,))
    original = model(batch)
    role_changed = model(replace(batch, roles=torch.tensor([1])))
    position_changed = model(replace(batch, positions=batch.positions.flip(1)))
    assert not torch.allclose(original.coarse, role_changed.coarse)
    assert not torch.allclose(original.fine, position_changed.fine)
    with torch.no_grad():
        for i, head in enumerate(model.heads):
            head.weight.zero_()
            head.bias.fill_(i - 5)
    result = model(batch)
    torch.testing.assert_close(result.coarse, torch.tensor([[-5., -4., -3.]]))
    torch.testing.assert_close(result.fine, torch.tensor([[-2., -1., 0., 1., 2.]]))


@pytest.mark.parametrize("autocast", [False, True])
def test_bf16_native_summaries_work_with_fp32_task_weights(autocast):
    torch.manual_seed(25)
    cfg = config()
    encoder = FakeWindowEncoder(8).to(dtype=torch.bfloat16)
    model = build_model(encoder, cfg)
    assert model.fusion.projection.weight.dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        output = model(inputs((3,)))
        loss = output.coarse.float().square().sum() + output.fine.float().square().sum()
    loss.backward()
    assert encoder.adapter.weight.grad is not None
    assert torch.count_nonzero(encoder.adapter.weight.grad) > 0
    assert torch.count_nonzero(model.fusion.projection.weight.grad) > 0
    model.eval()
    with torch.no_grad():
        inference = model(inputs((3,)))
    assert torch.isfinite(inference.coarse).all()
    assert torch.isfinite(inference.fine).all()


def test_frozen_encoder_stays_frozen_and_mean_removes_aggregation_parameters():
    cfg = config()
    model = build_model(FakeWindowEncoder(8, adaptation="frozen"), cfg)
    mean = build_model(FakeWindowEncoder(8, adaptation="frozen"), replace(cfg, model=replace(cfg.model, aggregation="mean")))
    output = model(inputs((3,)))
    (output.coarse.sum() + output.fine.sum()).backward()
    assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    assert sum(p.numel() for p in mean.aggregation.parameters()) == 0
    main_task = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("encoder."))
    mean_task = sum(p.numel() for name, p in mean.named_parameters() if not name.startswith("encoder."))
    assert main_task - mean_task == sum(p.numel() for p in model.aggregation.parameters())


def test_main_rejects_concat_role_dimension_mismatch_and_concat_architecture():
    cfg = config()
    model = build_model(fake_backend_factory(cfg).encoder, cfg)
    with pytest.raises(ValueError, match="PRE.*POST.*DIFF"):
        model(replace(inputs((1,)), roles=torch.tensor([3])))
    with pytest.raises(ValueError, match="native_dim"):
        HierarchicalClassifier(FakeWindowEncoder(7), cfg.model)
    with pytest.raises(ValueError, match="concat"):
        HierarchicalClassifier(FakeWindowEncoder(8), replace(cfg.model, architecture="concat"))
