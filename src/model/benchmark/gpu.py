"""Synchronized full-coverage CUDA benchmark of supervised training-source pairs."""

from collections.abc import Sequence
from dataclasses import asdict
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch
from torch.amp import GradScaler

from model.config import RunConfig
from model.data.schema import PairRecord
from model.encode.hf import load_backend
from model.errors import ResourceError
from model.input.batch import collate_pairs
from model.input.windows import prepare_model_input
from model.models.pair import build_model
from model.train.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from model.train.loss import compute_loss
from model.train.optim import build_optimizer
from model.train.step import _execution, train_effective_batch


def _environment():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "packages": {name: version(name) for name in ("transformers", "peft", "safetensors")},
            "gpu": properties.name, "total_memory_bytes": properties.total_memory,
            "device_index": torch.cuda.current_device(),
            "capability": list(torch.cuda.get_device_capability()),
            "bf16_supported": torch.cuda.is_bf16_supported(), "nvidia_smi": driver}


def _timed(call, *, memory=False):
    torch.cuda.synchronize()
    if memory:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    metrics = {"seconds": time.perf_counter() - start}
    if memory:
        metrics.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return value, metrics


def _forward(model, prepared, pad):
    device, dtype = _execution(model)
    inputs, _ = collate_pairs([prepared], pad)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        return model(inputs.to(device))


def _predict(model, prepared, pad):
    model.eval()
    with torch.no_grad():
        logits = _forward(model, prepared, pad)
    values = torch.cat((logits.coarse, logits.fine), dim=1).float().cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError("nonfinite inference logits")
    return values


def _warmup(model, prepared, cfg, pad):
    model.train()
    device, dtype = _execution(model)
    inputs, targets = collate_pairs([prepared], pad)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        loss = compute_loss(model(inputs.to(device)), targets.to(device), n_total=1,
                            n_modify=int(prepared.record.targets.modify_mask),
                            lambda_fine=cfg.train.lambda_fine).total
    loss.backward()
    model.zero_grad(set_to_none=True)


def _gradients(model):
    adapter = [(name, p) for name, p in model.encoder.named_parameters() if p.requires_grad]
    frozen = [(name, p) for name, p in model.encoder.named_parameters() if not p.requires_grad]
    norms = {name: None if p.grad is None else float(p.grad.detach().float().norm().item())
             for name, p in adapter}
    finite = all(value is None or math.isfinite(value) for value in norms.values())
    norms = {name: value if value is None or math.isfinite(value) else None
             for name, value in norms.items()}
    return {"adapter_nonzero": bool(adapter) and finite and any(
                value is not None and value > 0 for value in norms.values()),
            "adapter_gradients_finite": finite, "adapter_grad_norms": norms,
            "frozen_parameter_count": sum(p.numel() for _, p in frozen),
            "frozen_base_has_grad": any(p.grad is not None for _, p in frozen)}


def _stage_hooks(model, timings):
    handles = []
    for name, module in (("window_encoding_seconds", model.encoder),
                         ("aggregation_seconds", model.aggregation)):
        starts = []
        def before(_module, _args, starts=starts):
            torch.cuda.synchronize()
            starts.append(time.perf_counter())
        def after(_module, _args, _output, starts=starts, name=name):
            torch.cuda.synchronize()
            timings[name] += time.perf_counter() - starts.pop()
        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
    return handles


def _failure(exc):
    result = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError, ResourceError)):
        result["capacity_advice"] = "Increase GPU/memory capacity; preserve complete source and diff coverage."
    return result


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")


def benchmark_pairs(cfg: RunConfig, records: Sequence[PairRecord], output: Path) -> dict:
    """Measure exactly three explicitly supplied short/mid/long training examples.

    Runtime failures retain every record ID and invalidate the complete result.
    Inference stage hooks include synchronization overhead; throughput uses
    actual nonpadding encoded tokens, including overlap, wrappers and diff.
    Training timing includes forward/backward, clipping and one optimizer update.
    """
    if cfg.encoder.device != "cuda" or not torch.cuda.is_available():
        raise ResourceError("benchmark requires actual CUDA; no CPU fallback")
    if len(records) != 3 or len({r.record_id for r in records}) != 3:
        raise ValueError("benchmark requires exactly three distinct short/mid/long training pairs")
    if any(r.targets is None for r in records):
        raise ValueError("benchmark requires supervised training-source records")
    lengths = [len(r.pre_code or "") + len(r.post_code or "") for r in records]
    if lengths != sorted(lengths) or len(set(lengths)) != 3:
        raise ValueError("supply short/mid/long records in increasing original source length")
    if cfg.window.input_mode != "full_diff" or cfg.model.architecture != "hierarchical":
        raise ValueError("benchmark acceptance requires hierarchical full_diff complete coverage")
    if cfg.encoder.adaptation != "lora":
        raise ValueError("benchmark acceptance requires LoRA adaptation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [{"record_id": r.record_id, "size": size, "source_id": r.source_id,
             "repo_id": r.repo_id, "commit_sha": r.commit_sha, "language": r.language,
             "original_chars": [len(r.pre_code or ""), len(r.post_code or "")],
             "source_pair_sha256": hashlib.sha256(json.dumps(
                 [r.pre_code, r.post_code], ensure_ascii=False).encode("utf-8")).hexdigest()}
            for r, size in zip(records, ("short", "mid", "long"))]
    report = {"passed": False, "formal": False, "purpose": "engineering GPU validation",
              "environment": {}, "config": asdict(cfg), "warmup_count": 0,
              "token_throughput_definition": "nonpadding encoded tokens including overlap/wrappers/diff",
              "pairs": rows, "roundtrip": {"passed": False}}
    try:
        report["environment"] = _environment()
        torch.manual_seed(cfg.train.seed)
        backend, report["backend_load"] = _timed(lambda: load_backend(cfg))
        cfg = backend.config
        report["config"] = asdict(cfg)
        _write(output / "config.json", report["config"])
        model = build_model(backend.encoder, cfg)
        report["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_trainable": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
            "storage_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "compute_dtype": str(_execution(model)[1]),
            "lora_modules": list(getattr(model.encoder, "lora_module_names", ())) }
        prepared = []
        for record, row in zip(records, rows):
            try:
                pair, timing = _timed(lambda: prepare_model_input(record, backend.tokenizer, cfg))
                row.update(preprocessing=timing, windows=len(pair.windows),
                           windows_by_role={role: sum(w.role == role for w in pair.windows)
                                            for role in ("PRE", "POST", "DIFF")},
                           source_tokens=list(pair.token_counts), coverage=list(pair.coverage),
                           encoded_valid_tokens=sum(len(w.input_ids) for w in pair.windows),
                           window_lengths=[len(w.input_ids) for w in pair.windows])
                if pair.coverage != (1.0, 1.0, 1.0):
                    raise ValueError("incomplete source or diff coverage")
                prepared.append(pair)
            except Exception as exc:
                row["failure"] = _failure(exc)
                prepared.append(None)
        first = next((p for p in prepared if p is not None), None)
        if first is not None:
            _, report["warmup"] = _timed(lambda: _warmup(model, first, cfg, backend.tokenizer.pad_token_id))
            report["warmup_count"] = 1
        optimizer = build_optimizer(model, cfg.train)
        device, dtype = _execution(model)
        scaler = GradScaler(device.type) if dtype == torch.float16 else None
        for pair, row in zip(prepared, rows):
            if pair is None:
                continue
            handles = []
            try:
                stages = {"window_encoding_seconds": 0.0, "aggregation_seconds": 0.0}
                handles = _stage_hooks(model, stages)
                _, row["inference"] = _timed(
                    lambda: _predict(model, pair, backend.tokenizer.pad_token_id), memory=True)
                row["inference"].update(stages)
                for handle in handles:
                    handle.remove()
                handles = []
                result, row["training"] = _timed(lambda: train_effective_batch(
                    model, [pair], optimizer, cfg.train, pad_token_id=backend.tokenizer.pad_token_id,
                    scaler=scaler), memory=True)
                values = asdict(result)
                row["training"]["loss"] = {key: value if not isinstance(value, float) or math.isfinite(value)
                                           else None for key, value in values.items()}
                if any(isinstance(value, float) and not math.isfinite(value) for value in values.values()):
                    raise RuntimeError("nonfinite training loss")
                for name in ("inference", "training"):
                    seconds = row[name]["seconds"]
                    row[name].update(pairs_per_second=1 / seconds,
                                     tokens_per_second=row["encoded_valid_tokens"] / seconds)
                row["gradients"] = _gradients(model)
                if not row["gradients"]["adapter_nonzero"] or row["gradients"]["frozen_base_has_grad"]:
                    raise RuntimeError("adapter nonzero/frozen-base gradient contract failed")
            except Exception as exc:
                row["failure"] = _failure(exc)
            finally:
                for handle in handles:
                    handle.remove()
        if all("failure" not in row for row in rows):
            expected = [_predict(model, pair, backend.tokenizer.pad_token_id) for pair in prepared]
            special_rule = backend.tokenizer.special_token_rule
            hashes = {row["record_id"]: row["source_pair_sha256"] for row in rows}
            save_checkpoint(output / "model" / "roundtrip", model, cfg,
                            CheckpointState(0, 3, 0.0, hashes, special_rule))
            del model, optimizer, backend, scaler
            gc.collect()
            torch.cuda.empty_cache()
            bundle = load_checkpoint(output / "model" / "roundtrip", device=cfg.encoder.device,
                                     backend_factory=load_backend)
            actual = [_predict(bundle.model, prepare_model_input(record, bundle.tokenizer, bundle.config),
                               bundle.tokenizer.pad_token_id) for record in records]
            rtol, atol = (1e-5, 1e-6) if cfg.encoder.dtype == "fp32" else (1e-2, 1e-2)
            report["roundtrip"] = {"passed": all(torch.allclose(a, b, rtol=rtol, atol=atol)
                                                   for a, b in zip(expected, actual)),
                "rtol": rtol, "atol": atol, "max_absolute_error": max(
                    (a - b).abs().max().item() for a, b in zip(expected, actual)),
                "expected_logits": [v.tolist() for v in expected],
                "reloaded_logits": [v.tolist() for v in actual],
                "checkpoint_digest": bundle.checkpoint_digest,
                "checkpoint": "model/roundtrip", "reload": "pinned backend and PEFT adapter state"}
            report["passed"] = report["roundtrip"]["passed"]
    except Exception as exc:
        report["failure"] = _failure(exc)
        for row in rows:
            row.setdefault("failure", {"type": "BenchmarkAborted", "message": str(exc)})
    _write(output / "benchmark.json", report)
    return report
