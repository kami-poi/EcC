"""Real pair models accumulate identical losses regardless of splitting."""

import copy
import math
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from helpers import FakeTokenizer, FakeWindowEncoder, make_record
from model.config import EncoderConfig, ModelConfig, RunConfig, WindowConfig
from model.input.batch import collate_pairs
from model.input.windows import prepare_model_input
from model.models.pair import build_model
from model.train.loss import compute_loss
from model.train.optim import build_optimizer
from model.train.step import train_effective_batch


def fixture():
    cfg = RunConfig(encoder=EncoderConfig(device="cpu", dtype="fp32", lora_dropout=0),
                    model=ModelConfig(native_dim=8, projection_dim=8, ffn_dim=16, dropout=0),
                    window=WindowConfig(window_tokens=256, overlap_tokens=16, window_checkpoint=False))
    tokenizer = FakeTokenizer()
    records = [make_record("a", "b", labels=["Modify"], subtypes=["try_body_changed"]),
               make_record("a" * 350, "b" * 370, labels=["Modify"], subtypes=["catch_filter_changed"]),
               make_record(None, "new", labels=["Add"]),
               make_record("old", None, labels=["Delete"]),
               make_record("same", "same", labels=["None"])]
    pairs = [prepare_model_input(record, tokenizer, cfg) for record in records]
    torch.manual_seed(7)
    model = build_model(FakeWindowEncoder(8, dropout=0), cfg)
    return cfg, pairs, model


@pytest.mark.parametrize("microbatch", [1, 2])
def test_microbatch_gradients_equal_full_batch(microbatch):
    cfg, pairs, model = fixture()
    full = copy.deepcopy(model)
    inputs, targets = collate_pairs(pairs, 0)
    expected = compute_loss(full(inputs), targets, n_total=5, n_modify=2)
    expected.total.backward()
    result = train_effective_batch(model, pairs, build_optimizer(model, cfg.train),
                                   replace(cfg.train, microbatch=microbatch, clip_norm=1e6), pad_token_id=0)
    assert (result.n_pairs, result.n_modify) == (5, 2)
    for field in ("total", "coarse", "fine"):
        assert getattr(result, field) == pytest.approx(getattr(expected, field).item(), rel=1e-5)
    for (name, parameter), (_, reference) in zip(model.named_parameters(), full.named_parameters()):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-5, atol=1e-6, msg=name)
        else:
            assert parameter.grad is None


def test_no_window_count_loss_weight():
    cfg, pairs, model = fixture()
    assert len(pairs[1].windows) > len(pairs[0].windows)
    with torch.no_grad():
        for head in model.heads:
            head.weight.zero_()
            head.bias.zero_()
    result = train_effective_batch(model, pairs, build_optimizer(model, cfg.train),
                                   replace(cfg.train, microbatch=2), pad_token_id=0)
    assert result.coarse == pytest.approx(math.log(2))
    assert result.fine == pytest.approx(math.log(2))
    assert result.total == pytest.approx(2 * math.log(2))


def test_tail_batch_updates_zeroes_and_clips_once():
    cfg, pairs, model = fixture()
    optimizer = build_optimizer(model, cfg.train)
    before = model.heads[0].bias.detach().clone()
    with patch.object(optimizer, "step", wraps=optimizer.step) as step, \
         patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero, \
         patch("model.train.step.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
        train_effective_batch(model, pairs, optimizer, replace(cfg.train, microbatch=2), pad_token_id=0)
    assert step.call_count == zero.call_count == clip.call_count == 1
    assert not torch.equal(model.heads[0].bias.detach(), before)


def test_no_modify_batch_has_zero_fine_loss_and_finite_head_gradients():
    cfg, pairs, model = fixture()
    result = train_effective_batch(model, pairs[2:], build_optimizer(model, cfg.train), cfg.train, pad_token_id=0)
    assert result.n_pairs == 3 and result.n_modify == 0 and result.fine == 0
    assert math.isfinite(result.total)
    for head in model.heads[3:]:
        assert head.weight.grad is not None
        torch.testing.assert_close(head.weight.grad, torch.zeros_like(head.weight))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_explicit_compute_dtype_wins_over_fp32_parameters(dtype):
    cfg, pairs, model = fixture()
    model.encoder.compute_dtype = dtype
    assert all(p.dtype == torch.float32 for p in model.parameters())
    outputs = []
    handle = model.register_forward_hook(lambda module, args, result: outputs.append(result.coarse.dtype))
    scaler = torch.amp.GradScaler("cpu") if dtype == torch.float16 else None
    before = model.heads[0].bias.detach().clone()
    result = train_effective_batch(model, pairs[2:], build_optimizer(model, cfg.train),
                                   cfg.train, pad_token_id=0, scaler=scaler)
    handle.remove()
    assert outputs == [dtype] * 3
    assert math.isfinite(result.total)
    assert not torch.equal(before, model.heads[0].bias.detach())


def test_bf16_parameter_dtype_inference_uses_autocast():
    cfg, pairs, model = fixture()
    model.encoder.to(dtype=torch.bfloat16)
    outputs = []
    handle = model.register_forward_hook(lambda module, args, result: outputs.append(result.coarse.dtype))
    result = train_effective_batch(model, pairs[2:], build_optimizer(model, cfg.train), cfg.train, pad_token_id=0)
    handle.remove()
    assert outputs == [torch.bfloat16] * 3
    assert math.isfinite(result.total)


def test_fp16_requires_scaler_before_changing_optimizer_state():
    cfg, pairs, model = fixture()
    model.encoder.compute_dtype = torch.float16
    optimizer = build_optimizer(model, cfg.train)
    with patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero:
        with pytest.raises(ValueError, match="(?i)scaler"):
            train_effective_batch(model, pairs, optimizer, cfg.train, pad_token_id=0)
    zero.assert_not_called()


def test_scaler_unscales_before_single_gradient_clip_and_update():
    cfg, pairs, model = fixture()
    model.encoder.compute_dtype = torch.float16
    scaler = torch.amp.GradScaler("cpu", init_scale=32)
    optimizer = build_optimizer(model, cfg.train)
    events = []
    original_unscale, original_step = scaler.unscale_, scaler.step
    original_clip = torch.nn.utils.clip_grad_norm_

    def unscale(optimizer):
        events.append("unscale")
        return original_unscale(optimizer)

    def clip(parameters, max_norm):
        events.append("clip")
        return original_clip(parameters, max_norm)

    def step(optimizer):
        events.append("step")
        return original_step(optimizer)

    with patch.object(scaler, "unscale_", side_effect=unscale), \
         patch.object(scaler, "step", side_effect=step), \
         patch("model.train.step.clip_grad_norm_", side_effect=clip):
        train_effective_batch(model, pairs, optimizer, cfg.train, pad_token_id=0, scaler=scaler)
    assert events == ["unscale", "clip", "step"]


@pytest.mark.parametrize("kind", ["empty", "oversized", "unsupervised"])
def test_invalid_effective_batch_rejected_before_zero_grad(kind):
    cfg, pairs, model = fixture()
    if kind == "empty":
        pairs = []
    elif kind == "oversized":
        pairs = pairs * 4
    else:
        pairs = [replace(pairs[0], record=replace(pairs[0].record, targets=None))]
    optimizer = build_optimizer(model, cfg.train)
    with patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as zero:
        with pytest.raises(ValueError):
            train_effective_batch(model, pairs, optimizer, cfg.train, pad_token_id=0)
    zero.assert_not_called()
