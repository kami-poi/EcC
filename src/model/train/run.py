"""Audited train/validation orchestration with fixed-threshold best selection.

Checkpoint weights are inference artifacts; RNG snapshots document the run,
but this entry point intentionally offers no interruption/resume interface.
"""

from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import random

import numpy as np
import torch
from torch.amp import GradScaler

from model.config import RunConfig
from model.data.audit import _path
from model.data.jsonl import _json_snapshot, load_records_snapshot
from model.encode.base import BackendFactory
from model.encode.hf import load_backend
from model.errors import ArtifactError, DataError, ResourceError
from model.evaluate.decode import Thresholds
from model.evaluate.metrics import evaluate
from model.evaluate.thresholds import calibrate_thresholds
from model.input.batch import collate_pairs
from model.input.windows import prepare_model_input
from model.labels import COARSE_LABELS, FINE_LABELS, FINE_LABEL_VERSION
from model.models.base import PairLogits
from model.models.pair import build_model

from .checkpoint import CheckpointState, load_checkpoint, save_checkpoint, save_thresholds
from .optim import build_optimizer, build_scheduler
from .step import _execution, train_effective_batch


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def _append(path, payload):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def _data(cfg):
    """Consume an existing audit; never rerun an audit that opens test labels."""
    if cfg.data.train is None or cfg.data.val is None:
        raise DataError("training requires explicit train and val paths")
    paths = {split: Path(getattr(cfg.data, split)).resolve() for split in ("train", "val")}
    if paths["train"] == paths["val"]:
        raise DataError("train and val must be distinct files")
    hashes = {}
    mappings = {split: cfg.data.field_map for split in paths}
    counts = {}
    declarations = {}
    if cfg.data.mode == "formal":
        if cfg.data.manifest is None or cfg.data.audit is None:
            raise DataError("formal training requires existing manifest and passed audit")
        manifest_path, audit_path = Path(cfg.data.manifest).resolve(), Path(cfg.data.audit).resolve()
        manifest, hashes["manifest"] = _json_snapshot(manifest_path)
        audit, hashes["audit"] = _json_snapshot(audit_path)
        if (not isinstance(audit, dict) or audit.get("passed") is not True or audit.get("issues") != []
                or not isinstance(audit.get("counts"), dict) or audit["counts"].get("formal") is not True
                or not isinstance(audit.get("file_sha256"), dict)):
            raise DataError("formal training requires a passed formal audit with no issues")
        if hashes["manifest"] != audit.get("manifest_sha256"):
            raise DataError("manifest hash differs from passed audit")
        if (audit["counts"].get("fine_label_version") != FINE_LABEL_VERSION
                or audit["counts"].get("fine_label_order") != list(FINE_LABELS)):
            raise DataError("audit uses an incompatible fine label profile; rerun check-data for eh-change-5")
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
            raise DataError("invalid audited manifest schema")
        for split, path in paths.items():
            matches = [entry for entry in manifest["files"] if isinstance(entry, dict)
                       and _path(manifest_path.parent, entry.get("path")) == path]
            if len(matches) != 1 or matches[0].get("split") != split or matches[0].get("language") != "C#":
                raise DataError(f"{split} path must have one audited C# {split} declaration")
            entry = matches[0]
            declarations[split] = entry
            declared = entry.get("field_map", manifest.get("field_map", {}))
            if cfg.data.field_map and cfg.data.field_map != declared:
                raise DataError(f"{split} field_map differs from audited declaration")
            mappings[split], counts[split] = declared, entry.get("n_records")
    records = {}
    for split, path in paths.items():
        records[split], hashes[split] = load_records_snapshot(path, field_map=mappings[split])
        if cfg.data.mode == "formal":
            entry = declarations[split]
            if hashes[split] != entry.get("sha256", "").lower() or hashes[split] != audit["file_sha256"].get(entry["path"]):
                raise DataError(f"{split} hash differs from manifest or audit")
    for split, values in records.items():
        if not values:
            raise DataError(f"{split} must contain at least one supervised pair")
        if split in counts and len(values) != counts[split]:
            raise DataError(f"{split} record count differs from audited declaration")
        for record in values:
            if record.language not in (None, "C#", "csharp"):
                raise DataError(f"{split}:{record.record_id}: supervised training/validation must be C#")
            if cfg.data.mode == "formal" and (record.language != "C#" or any(
                    getattr(record, name) is None for name in ("repo_id", "commit_sha", "source_id"))):
                raise DataError(f"{split}:{record.record_id}: formal source provenance is missing")
    return records, hashes


def _rng():
    numpy = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": numpy[0], "keys": numpy[1].tolist(), "position": numpy[2],
                      "has_gauss": numpy[3], "cached_gaussian": numpy[4]},
            "torch_cpu": torch.get_rng_state().tolist(),
            "torch_cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def _environment():
    result = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
              "numpy": np.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    for package in ("transformers", "peft", "safetensors", "scikit-learn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _context(pairs):
    return [{"record_id": pair.record.record_id, "window_count": len(pair.windows)} for pair in pairs]


def _infer(model, pairs, cfg, pad_token_id, logs):
    """Detach validation logits only; training embeddings are never cached."""
    device, dtype = _execution(model)
    model.eval()
    coarse, fine = [], []
    with torch.no_grad():
        for start in range(0, len(pairs), cfg.train.microbatch):
            batch = pairs[start:start + cfg.train.microbatch]
            try:
                inputs, _ = collate_pairs(batch, pad_token_id)
                with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                    logits = model(inputs.to(device))
                coarse.append(logits.coarse.float().cpu())
                fine.append(logits.fine.float().cpu())
            except (torch.cuda.OutOfMemoryError, MemoryError, ResourceError) as exc:
                _failure(logs, "validation", _context(batch), cfg, exc)
    return PairLogits(torch.cat(coarse), torch.cat(fine))


def _failure(logs, phase, records, cfg, exc):
    _write(logs / "failure.json", {"phase": phase, "records": records, "config": asdict(cfg),
                                  "error_type": type(exc).__name__, "error": str(exc)})
    raise ResourceError("training aborted after resource failure; increase compute/memory capacity") from exc


def run_training(cfg: RunConfig, output: Path, *, backend_factory: BackendFactory | None = None) -> Path:
    """Create one run directory; use only train/val, then calibrate reloaded best.

    CLI checkpoint defaults resolve model/best with model/thresholds.json.
    Formal runs consume the upstream passed audit; debug outputs are nonformal.
    """
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"run output must be new or empty: {output}")
    records, hashes = _data(cfg)
    output.mkdir(parents=True, exist_ok=True)
    logs, models = output / "logs", output / "model"
    logs.mkdir()
    models.mkdir()
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)
    initial_rng = _rng()
    _write(output / "rng_state.json", {"initial": initial_rng, "final": None})
    _write(logs / "environment.json", _environment())
    formal = cfg.data.mode == "formal"
    _write(output / "provenance.json", {"formal": formal, "mode": cfg.data.mode, "data_hashes": hashes,
        "manifest": cfg.data.manifest, "audit": cfg.data.audit, "train": cfg.data.train, "val": cfg.data.val})
    _write(output / "labels.json", {"coarse": COARSE_LABELS, "fine": FINE_LABELS})
    factory = backend_factory or load_backend
    try:
        backend = factory(cfg)
        cfg = backend.config
        model = build_model(backend.encoder, cfg)
    except (torch.cuda.OutOfMemoryError, MemoryError, ResourceError) as exc:
        _failure(logs, "backend", [], cfg, exc)
    _write(output / "config.json", asdict(cfg))
    prepared = {}
    for split, values in records.items():
        prepared[split] = []
        for record in values:
            try:
                prepared[split].append(prepare_model_input(record, backend.tokenizer, cfg))
            except (torch.cuda.OutOfMemoryError, MemoryError, ResourceError) as exc:
                _failure(logs, "prepare", [{"record_id": record.record_id, "window_count": None}], cfg, exc)
    # Source metadata remains intact; only evaluator's spelling is normalized.
    val_records = [replace(record, language="csharp") if record.language == "C#" else record
                   for record in records["val"]]
    optimizer = build_optimizer(model, cfg.train)
    total_steps = math.ceil(len(prepared["train"]) / cfg.train.effective_batch) * cfg.train.epochs
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=cfg.train.warmup_ratio)
    device, dtype = _execution(model)
    scaler = GradScaler(device.type) if dtype == torch.float16 else None
    special_rule, pad_token_id = backend.tokenizer.special_token_rule, backend.tokenizer.pad_token_id
    best_score, best_epoch, stale, successful_steps = -math.inf, 0, 0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        order = list(range(len(prepared["train"])))
        random.shuffle(order)
        for start in range(0, len(order), cfg.train.effective_batch):
            batch = [prepared["train"][i] for i in order[start:start + cfg.train.effective_batch]]
            scale_before = scaler.get_scale() if scaler is not None else None
            try:
                result = train_effective_batch(model, batch, optimizer, cfg.train,
                                               pad_token_id=pad_token_id, scaler=scaler)
            except (torch.cuda.OutOfMemoryError, MemoryError, ResourceError) as exc:
                _failure(logs, "train", _context(batch), cfg, exc)
            updated = scaler is None or scaler.get_scale() >= scale_before
            if updated:
                successful_steps += 1
                scheduler.step()
            payload = {name: value if not isinstance(value, float) or math.isfinite(value) else None
                       for name, value in asdict(result).items()}
            payload.update(epoch=epoch, step=successful_steps, optimizer_updated=updated,
                           record_ids=[p.record.record_id for p in batch])
            _append(logs / "train.jsonl", payload)
        logits = _infer(model, prepared["val"], cfg, pad_token_id, logs)
        report = evaluate(val_records, logits, Thresholds(), language="csharp")
        score = (report.metrics["coarse_macro_f1"] + report.metrics["fine_end_to_end_macro_f1"]) / 2
        if not math.isfinite(score):
            raise ValueError("fixed-threshold selection score must be finite")
        improved = score > best_score
        state = CheckpointState(epoch, successful_steps, score, hashes, special_rule)
        checkpoint_name = "last"
        try:
            save_checkpoint(models / "last", model, cfg, state)
            if improved:
                checkpoint_name = "best"
                save_checkpoint(models / "best", model, cfg, state)
        except (torch.cuda.OutOfMemoryError, MemoryError, ResourceError) as exc:
            _failure(logs, f"save_{checkpoint_name}", _context(batch), cfg, exc)
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        _append(logs / "epochs.jsonl", {"epoch": epoch, "step": successful_steps, "selection_score": score,
            "metrics": report.metrics, "thresholds": Thresholds().to_dict(), "improved": improved,
            "nonimproving_epochs": stale, "formal": formal})
        if stale >= cfg.train.patience:
            break
    # Drop training owners before loading another backbone on the same device.
    del model, optimizer, scheduler, scaler, backend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        bundle = load_checkpoint(models / "best", device=cfg.encoder.device, thresholds_path=None, backend_factory=factory)
    except (torch.cuda.OutOfMemoryError, MemoryError, ResourceError) as exc:
        _failure(logs, "reload_best", [], cfg, exc)
    except ArtifactError as exc:
        # Restoration may wrap a resource error; retain capacity reporting
        # while ordinary tensor/schema failures remain artifact errors.
        if isinstance(exc.__cause__, (torch.cuda.OutOfMemoryError, MemoryError, ResourceError)):
            _failure(logs, "reload_best", [], cfg, exc.__cause__)
        raise
    best_val = []
    for record in records["val"]:
        try:
            best_val.append(prepare_model_input(record, bundle.tokenizer, bundle.config))
        except (torch.cuda.OutOfMemoryError, MemoryError, ResourceError) as exc:
            _failure(logs, "prepare_best", [{"record_id": record.record_id, "window_count": None}], bundle.config, exc)
    logits = _infer(bundle.model, best_val, bundle.config, bundle.tokenizer.pad_token_id, logs)
    thresholds = calibrate_thresholds(val_records, logits, split="val", language="csharp")
    save_thresholds(models / "thresholds.json", thresholds, checkpoint_digest=bundle.checkpoint_digest)
    final_report = evaluate(val_records, logits, thresholds, language="csharp")
    final_report.provenance.update(formal=formal, checkpoint_digest=bundle.checkpoint_digest, selected_epoch=best_epoch)
    _write(logs / "validation.json", final_report.to_dict())
    for record, prediction in zip(records["val"], final_report.predictions):
        _append(logs / "validation_predictions.jsonl", {"record_id": record.record_id, "split": "val",
            "checkpoint_digest": bundle.checkpoint_digest, **prediction.to_dict()})
    _write(output / "rng_state.json", {"initial": initial_rng, "final": _rng()})
    _write(logs / "run.json", {"status": "complete", "formal": formal, "seed": cfg.train.seed,
        "best_epoch": best_epoch, "best_score": best_score, "stopped_epoch": epoch,
        "successful_steps": successful_steps, "checkpoint_digest": bundle.checkpoint_digest,
        "paths": {"best": "model/best", "last": "model/last", "thresholds": "model/thresholds.json",
                  "config": "config.json", "provenance": "provenance.json", "labels": "labels.json",
                  "rng_state": "rng_state.json", "train_log": "logs/train.jsonl", "epochs_log": "logs/epochs.jsonl",
                  "environment": "logs/environment.json", "validation": "logs/validation.json",
                  "validation_predictions": "logs/validation_predictions.jsonl"}})
    return output
