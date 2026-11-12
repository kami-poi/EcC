from dataclasses import replace
import json

import pytest
import torch

from model.config import EncoderConfig, ModelConfig, RunConfig, WindowConfig
from model.errors import ResourceError
from helpers import fake_backend_factory, make_record


def config():
    return RunConfig(encoder=EncoderConfig(dtype="fp32"),
                     model=ModelConfig(native_dim=8, projection_dim=8, heads=2,
                                       ffn_dim=16, layers=1, dropout=0),
                     window=WindowConfig(window_tokens=256, overlap_tokens=16))


def records():
    return [replace(make_record("a" * n, "a" * n + "b", labels=["Modify"],
                                subtypes=["try_body_changed"]), record_id=f"pair-{n}")
            for n in (10, 100, 400)]


def setup_cuda(monkeypatch):
    from model.benchmark import gpu
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    monkeypatch.setattr(gpu, "_environment", lambda: {"gpu": "stub"})
    monkeypatch.setattr(gpu, "load_backend", fake_backend_factory)
    return gpu, events


def test_benchmark_requires_cuda(tmp_path, monkeypatch):
    from model.benchmark.gpu import benchmark_pairs
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ResourceError, match="CUDA"):
        benchmark_pairs(config(), records(), tmp_path / "bench")
    assert not (tmp_path / "bench").exists()


def test_metrics_record_full_coverage_and_failure_id(tmp_path, monkeypatch):
    gpu, _ = setup_cuda(monkeypatch)
    original = gpu.prepare_model_input
    def fail_second(record, *args):
        if record.record_id == "pair-100":
            raise RuntimeError("injected window failure")
        return original(record, *args)
    monkeypatch.setattr(gpu, "prepare_model_input", fail_second)
    result = gpu.benchmark_pairs(config(), records(), tmp_path / "failed")
    assert result["passed"] is False
    assert [r["record_id"] for r in result["pairs"]] == ["pair-10", "pair-100", "pair-400"]
    assert result["pairs"][1]["failure"]["message"] == "injected window failure"
    for row in (result["pairs"][0], result["pairs"][2]):
        assert row["coverage"] == [1, 1, 1]
        assert row["windows"] >= 3
        assert row["original_chars"][0] > 0
        assert row["encoded_valid_tokens"] > sum(row["source_tokens"])
    assert json.loads((tmp_path / "failed" / "benchmark.json").read_text())["passed"] is False


def test_memory_reset_and_synchronization_boundaries(tmp_path, monkeypatch):
    gpu, events = setup_cuda(monkeypatch)
    result = gpu.benchmark_pairs(config(), records(), tmp_path / "bench")
    assert result["passed"] is True
    assert events.count("reset") == 6  # inference and training for every pair
    assert events[0] == "sync"
    assert all(events[i-1] == "sync" and events[i+1] == "sync"
               for i, event in enumerate(events) if event == "reset")
    for row in result["pairs"]:
        assert row["inference"]["peak_allocated_bytes"] == 100
        assert row["training"]["peak_reserved_bytes"] == 200
        assert row["gradients"]["adapter_nonzero"]
        assert row["gradients"]["frozen_base_has_grad"] is False
        assert row["inference"]["seconds"] > 0
        assert row["inference"]["tokens_per_second"] > 0
    assert result["roundtrip"]["passed"]


def test_benchmark_rejects_non_training_or_incomplete_main_path(tmp_path, monkeypatch):
    gpu, _ = setup_cuda(monkeypatch)
    with pytest.raises(ValueError, match="three"):
        gpu.benchmark_pairs(config(), records()[:2], tmp_path / "short")
    cfg = replace(config(), window=replace(config().window, input_mode="diff_only"))
    with pytest.raises(ValueError, match="full_diff"):
        gpu.benchmark_pairs(cfg, records(), tmp_path / "ablated")
    invalid = records()
    invalid[1] = replace(invalid[1], targets=None)
    with pytest.raises(ValueError, match="supervised"):
        gpu.benchmark_pairs(config(), invalid, tmp_path / "unlabeled")


def test_timing_encloses_synchronized_work(monkeypatch):
    from model.benchmark import gpu
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    clocks = iter((10.0, 12.0))
    def clock():
        events.append("clock")
        return next(clocks)
    monkeypatch.setattr(gpu.time, "perf_counter", clock)
    def work():
        events.append("work")
        return 3
    value, metrics = gpu._timed(work, memory=True)
    assert events == ["sync", "reset", "sync", "clock", "work", "sync", "clock"]
    assert value == 3 and metrics["seconds"] == 2


def test_oom_retains_all_ids_and_capacity_advice(tmp_path, monkeypatch):
    gpu, _ = setup_cuda(monkeypatch)
    def oom(_cfg):
        raise torch.cuda.OutOfMemoryError("injected OOM")
    monkeypatch.setattr(gpu, "load_backend", oom)
    report = gpu.benchmark_pairs(config(), records(), tmp_path / "oom")
    assert not report["passed"]
    assert [row["record_id"] for row in report["pairs"]] == [r.record_id for r in records()]
    assert all("failure" in row for row in report["pairs"])
    assert "Increase" in report["failure"]["capacity_advice"]
    assert json.loads((tmp_path / "oom/benchmark.json").read_text())["passed"] is False
