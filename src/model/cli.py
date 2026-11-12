"""Audited command-line workflows and calibrated source-only prediction."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import torch

from model.config import load_config
from model.data.audit import _path, audit_dataset
from model.data.jsonl import _json_snapshot, load_records, load_records_snapshot
from model.data.schema import PairRecord
from model.encode.base import BackendFactory
from model.errors import ArtifactError, ConfigError, DataError, InputError, ResourceError
from model.evaluate.decode import Prediction, decode
from model.evaluate.report import summarize_predictions
from model.input.batch import collate_pairs
from model.input.windows import prepare_model_input
from model.models.base import PairLogits
from model.train.checkpoint import InferenceBundle, _read_generation, load_checkpoint
from model.train.run import run_training
from model.train.step import _execution

_LANGUAGES = {"C#": "csharp", "Java": "java", "C++": "cpp",
              "csharp": "csharp", "java": "java", "cpp": "cpp"}
_RESOURCE_ERRORS = (torch.cuda.OutOfMemoryError, MemoryError, ResourceError)


def _parser():
    parser = argparse.ArgumentParser(prog="model", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-data", help="audit a dataset manifest before encoder loading")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--mode", choices=("formal", "debug"), default="formal")
    train = commands.add_parser("train", help="train and calibrate one audited run")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("eval", help="evaluate an independently audited test file")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--language", choices=("csharp", "java", "cpp"), required=True)
    prediction = commands.add_parser("predict", help="predict unlabeled source pairs")
    prediction.add_argument("--input", type=Path, required=True)
    for command in (evaluation, prediction):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--thresholds", type=Path)
        command.add_argument("--output", type=Path, required=True)
    return parser


def _new_output(path):
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def _jsonl(path, values):
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def _bundle(args, backend_factory):
    thresholds = args.thresholds if args.thresholds is not None else args.checkpoint.parent / "thresholds.json"
    if not thresholds.is_file():
        raise ArtifactError(f"calibrated thresholds are required: {thresholds}")
    # Use the persisted execution device. Loading revalidates all hashes/config.
    folder, _, _ = _read_generation(args.checkpoint)
    device = load_config(folder / "config.json").encoder.device
    bundle = load_checkpoint(args.checkpoint, device=device, thresholds_path=thresholds, backend_factory=backend_factory)
    if not bundle.calibrated:
        raise ArtifactError("inference requires a calibrated checkpoint bundle")
    return bundle


def _pair_logits(bundle, record, *, prepared=None):
    pair = prepare_model_input(record, bundle.tokenizer, bundle.config)
    if prepared is not None:
        prepared.append(pair)
    inputs, _ = collate_pairs([pair], bundle.tokenizer.pad_token_id)
    device, dtype = _execution(bundle.model)
    bundle.model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        logits = bundle.model(inputs.to(device))
    return PairLogits(logits.coarse.float().cpu(), logits.fine.float().cpu())


def predict_pair(bundle: InferenceBundle, pre_code: str | None, post_code: str | None) -> Prediction:
    """Predict from two raw source strings; calibration is mandatory."""
    if not bundle.calibrated:
        raise ArtifactError("prediction requires a calibrated checkpoint bundle")
    for source in (pre_code, post_code):
        if source is not None and not isinstance(source, str):
            raise InputError("source code must be a string or null")
    record = PairRecord("prediction", pre_code, post_code, None, None, None, None, None)
    return decode(_pair_logits(bundle, record), [record], bundle.thresholds)[0]


def _prediction(record, prediction, digest, **provenance):
    return {"record_id": record.record_id, "labels": list(prediction.coarse_labels),
            "modify_subtypes": list(prediction.fine_labels), "coarse_scores": list(prediction.coarse_scores),
            "fine_scores": list(prediction.fine_scores), "checkpoint_digest": digest, **provenance}


def _eval_data(args):
    audit = audit_dataset(args.manifest, mode="formal")
    if not audit.passed:
        raise DataError("evaluation audit failed: " + "; ".join(audit.issues))
    manifest, manifest_hash = _json_snapshot(args.manifest)
    if manifest_hash != audit.manifest_sha256:
        raise DataError("evaluation manifest changed after audit")
    root = args.manifest.resolve().parent
    entries = [entry for entry in manifest["files"] if _path(root, entry.get("path")) == args.data.resolve()]
    if len(entries) != 1 or entries[0].get("split") != "test":
        raise DataError("evaluation data must have one audited test declaration")
    entry = entries[0]
    if _LANGUAGES.get(entry["language"]) != args.language:
        raise DataError("evaluation language differs from manifest declaration")
    records, data_hash = load_records_snapshot(args.data, field_map=entry.get("field_map", manifest.get("field_map")))
    if not records:
        raise DataError("evaluation data must contain at least one supervised pair")
    if data_hash != audit.file_sha256[entry["path"]] or len(records) != entry["n_records"]:
        raise DataError("evaluation data changed after audit")
    # Normalize evaluation metadata only; raw source text is never modified.
    records = [replace(record, language=args.language) for record in records]
    return records, manifest, {"data_sha256": data_hash, "manifest_sha256": audit.manifest_sha256,
                               "repository_isolated": audit.counts.get("repository_isolated", True),
                               "repository_exceptions": audit.counts.get("repository_exceptions", 0),
                               "evaluation_distribution": audit.counts.get("evaluation_distribution", "natural")}


def _eval(args, backend_factory):
    records, manifest, hashes = _eval_data(args)
    args.output.mkdir(parents=True)
    predictions, coarse, fine, prepared = [], [], [], []
    bundle = None
    digest = None
    active_record = None
    try:
        # Keep the verified identity even if backend loading exhausts memory.
        _, digest, _ = _read_generation(args.checkpoint)
        bundle = _bundle(args, backend_factory)
        for record in records:
            active_record = record
            logits = _pair_logits(bundle, record, prepared=prepared)
            prediction = decode(logits, [record], bundle.thresholds)[0]
            predictions.append(_prediction(record, prediction, bundle.checkpoint_digest, **hashes))
            coarse.append(logits.coarse)
            fine.append(logits.fine)
        report = summarize_predictions(prepared, PairLogits(torch.cat(coarse), torch.cat(fine)), bundle.thresholds,
                                       language=args.language, repo_groups=manifest["repo_groups"])
    except (ArtifactError, ConfigError, DataError, InputError, OSError, ValueError, RuntimeError, MemoryError) as exc:
        pairs_by_id = {pair.record.record_id: pair for pair in prepared}
        failures = []
        for record in records[len(predictions):]:
            pair = pairs_by_id.get(record.record_id)
            cfg = bundle.config if bundle is not None and record is active_record else None
            failures.append({"record_id": record.record_id, "reason": str(exc), "error_type": type(exc).__name__,
                             "checkpoint_digest": digest, **hashes,
                             "window_count": len(pair.windows) if pair is not None else None,
                             "window_config": asdict(cfg.window) if cfg is not None else None,
                             "model_config": asdict(cfg.model) if cfg is not None else None,
                             "encoder_config": asdict(cfg.encoder) if cfg is not None else None})
        _jsonl(args.output / "predictions.jsonl", predictions)
        _jsonl(args.output / "failures.jsonl", failures)
        _json(args.output / "metrics.json", {
            "counts": {"n_requested": len(records), "n_predicted": len(predictions), "n_failed": len(failures)},
            "metrics": None, "diagnostics": {"failure_context": failures}, "provenance": {
                "status": "failed", "formal": False, "language": args.language, "checkpoint_digest": digest, **hashes,
                "training": {"mode": bundle.config.data.mode, "formal": bundle.config.data.mode == "formal"}
                    if bundle is not None else None,
                "evaluation": {"formal": True, **hashes}}})
        raise
    training_formal = bundle.config.data.mode == "formal"
    report.provenance.update(status="complete", formal=training_formal,
                             training={"mode": bundle.config.data.mode, "formal": training_formal},
                             evaluation={"formal": True, **hashes},
                             checkpoint_digest=bundle.checkpoint_digest, **hashes)
    report.provenance["parameter_counts"] = {
        "total": sum(p.numel() for p in bundle.model.parameters()),
        "trainable": sum(p.numel() for p in bundle.model.parameters() if p.requires_grad),
        "encoder_total": sum(p.numel() for p in bundle.model.encoder.parameters()),
        "encoder_trainable": sum(p.numel() for p in bundle.model.encoder.parameters() if p.requires_grad),
        "task_trainable": sum(p.numel() for name, p in bundle.model.named_parameters()
                              if not name.startswith("encoder.") and p.requires_grad)}
    report.provenance["model_config"] = asdict(bundle.config.model)
    report.provenance["encoder_adaptation"] = bundle.config.encoder.adaptation
    report.provenance["window_config"] = asdict(bundle.config.window)
    entry = next(item for item in manifest["files"] if _path(args.manifest.resolve().parent, item["path"]) == args.data.resolve())
    report.provenance["label_source"] = entry.get("label_source", manifest.get("label_source"))
    for field in ("rule_version", "annotation_version", "review_report"):
        report.provenance[field] = entry.get(field, manifest.get(field))
    report.counts.update(n_requested=len(records), n_predicted=len(predictions), n_failed=0)
    _json(args.output / "metrics.json", {key: value for key, value in report.to_dict().items() if key != "predictions"})
    _jsonl(args.output / "predictions.jsonl", predictions)
    _jsonl(args.output / "failures.jsonl", [])


def _dispatch(args, backend_factory):
    _new_output(args.output)
    if args.command == "check-data":
        report = audit_dataset(args.manifest, mode=args.mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _json(args.output, asdict(report))
        return 0 if report.passed else 2
    if args.command == "train":
        cfg = load_config(args.config)
        root = args.config.resolve().parent
        paths = {name: str((root / getattr(cfg.data, name)).resolve()) if getattr(cfg.data, name) is not None else None
                 for name in ("train", "val", "manifest", "audit")}
        cfg = replace(cfg, data=replace(cfg.data, **paths))
        # Atomically claim the CLI's new directory; the direct runner may use
        # the empty directory owned by this invocation.
        args.output.mkdir(parents=True)
        run_training(cfg, args.output, backend_factory=backend_factory)
    elif args.command == "eval":
        _eval(args, backend_factory)
    else:
        records = load_records(args.input, require_labels=False)
        bundle = _bundle(args, backend_factory)
        results = [_prediction(record, predict_pair(bundle, record.pre_code, record.post_code), bundle.checkpoint_digest)
                   for record in records]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _jsonl(args.output, results)
    return 0


def main(argv: list[str] | None = None, *, backend_factory: BackendFactory | None = None) -> int:
    """Return 0 on success, 2 for invalid requests/artifacts, 1 for run failure."""
    try:
        args = _parser().parse_args(argv)
        return _dispatch(args, backend_factory)
    except SystemExit as exc:
        return int(exc.code)
    except ArtifactError as exc:
        if isinstance(exc.__cause__, _RESOURCE_ERRORS):
            print(f"resource failure: {exc.__cause__}; increase compute/memory capacity", file=sys.stderr)
            return 1
        print(f"artifact error: {exc}", file=sys.stderr)
        return 2
    except _RESOURCE_ERRORS as exc:
        print(f"resource failure: {exc}; increase compute/memory capacity", file=sys.stderr)
        return 1
    except (ConfigError, DataError, InputError, OSError, ValueError) as exc:
        print(f"invalid request: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"runtime failure: {exc}", file=sys.stderr)
        return 1
