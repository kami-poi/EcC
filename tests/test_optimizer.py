"""Optimizer covers trainable parameters and scheduler advances by updates."""

import pytest
import torch

from helpers import FakeWindowEncoder
from model.config import ModelConfig, TrainConfig
from model.models.pair import HierarchicalClassifier
from model.train.optim import build_optimizer, build_scheduler


@pytest.mark.parametrize("adaptation", ["lora", "frozen"])
def test_optimizer_groups_cover_trainable_parameters_once(adaptation):
    model = HierarchicalClassifier(FakeWindowEncoder(8, adaptation=adaptation),
                                   ModelConfig(native_dim=8, projection_dim=8, ffn_dim=16))
    cfg = TrainConfig(lora_lr=0.002, task_lr=0.003, weight_decay=0.04)
    optimizer = build_optimizer(model, cfg)
    assert isinstance(optimizer, torch.optim.AdamW)
    members = [p for group in optimizer.param_groups for p in group["params"]]
    expected = [p for p in model.parameters() if p.requires_grad]
    assert len(members) == len({id(p) for p in members}) == len(expected)
    assert {id(p) for p in members} == {id(p) for p in expected}
    for group in optimizer.param_groups:
        assert group["weight_decay"] == cfg.weight_decay
        for parameter in group["params"]:
            name = next(name for name, p in model.named_parameters() if p is parameter)
            assert group["lr"] == (cfg.lora_lr if name.startswith("encoder.") else cfg.task_lr)


def test_linear_scheduler_warmup_then_decay_per_optimizer_update():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=10, warmup_ratio=0.2)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates == pytest.approx([0, .05, .1, .0875, .075, .0625, .05, .0375, .025, .0125, 0])


def test_scheduler_without_warmup_starts_at_full_rate():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=2, warmup_ratio=0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.1)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.05)


@pytest.mark.parametrize("steps,ratio", [(0, .05), (2, -1), (2, 1)])
def test_scheduler_rejects_invalid_duration(steps, ratio):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    with pytest.raises(ValueError):
        build_scheduler(optimizer, total_steps=steps, warmup_ratio=ratio)
