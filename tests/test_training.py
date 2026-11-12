"""Training orchestration exercises real CPU model updates and artifacts."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from helpers import fake_backend_factory
from model.config import DataConfig, EncoderConfig, ModelConfig, RunConfig, TrainConfig, WindowConfig
from model.data.audit import audit_dataset
from model.data.jsonl import load_records
from model.errors import ArtifactError, DataError, ResourceError
from model.evaluate.thresholds import calibrate_thresholds
from model.input.batch import collate_pairs
from model.input.windows import prepare_model_input
from model.train import run
from model.train import checkpoint
from model.train.checkpoint import load_checkpoint
from model.train.run import run_training
from model.labels import COARSE_LABELS, FINE_LABELS
from model.models.base import PairLogits


def read(path):
    return json.loads(path.read_text())


def state(path):
    pointer = read(path / "CURRENT.json")
    return read(path / "generations" / pointer["generation"] / "state.json")["state"]


def row(index, split):
    pre, post = f"try old_{split}_{index}();", f"try new_{split}_{index}();"
    labels, subtypes = ["Modify"], ["try_body_changed"]
    if index % 6 == 1:
        subtypes = ["catch_filter_changed"]
    elif index % 6 == 2:
        pre, labels, subtypes = None, ["Add"], []
    elif index % 6 == 3:
        post, labels, subtypes = None, ["Delete"], []
    elif index % 6 == 4:
        labels = ["Add", "Modify"]
    elif index % 6 == 5:
        post, labels, subtypes = pre, ["None"], []
    return dict(record_id=f"{split}-{index}", pre_code=pre, post_code=post, labels=labels,
                modify_subtypes=subtypes, repo_id=f"repo-{split}", commit_sha="commit",
                source_id=f"source-{split}-{index}", language="C#")


def fixture(tmp_path, *, n_train=3, epochs=1, mode="formal"):
    files = []
    for split, count in [("train", n_train), ("val", 6), ("test", 1)]:
        path = tmp_path / f"{split}.jsonl"
        rows = [row(i, split) for i in range(count)]
        if split == "test":
            rows[0]["language"] = "Java"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files.append(dict(path=path.name, split=split, language="Java" if split == "test" else "C#",
                          n_records=count, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(schema_version=1, label_version="eh-change-9", label_source="ast_rules",
        rule_version="test-rules", split_seed=0, sampling_seed=0,
        repo_groups={f"repo-{s}": s for s in ("train", "val", "test")},
        group_splits={s: s for s in ("train", "val", "test")}, files=files)))
    audit = audit_dataset(manifest)
    assert audit.passed, audit.issues
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(asdict(audit)))
    return RunConfig(data=DataConfig(train=str(tmp_path / "train.jsonl"), val=str(tmp_path / "val.jsonl"),
                                    manifest=str(manifest), audit=str(audit_path), mode=mode),
                     encoder=EncoderConfig(device="cpu", dtype="fp32", lora_dropout=0),
                     model=ModelConfig(native_dim=8, projection_dim=8, layers=1, heads=2,
                                       ffn_dim=16, dropout=0, aggregation="mean"),
                     window=WindowConfig(window_tokens=256, overlap_tokens=16, window_checkpoint=False),
                     train=TrainConfig(epochs=epochs, seed=3))


def test_old_eight_label_audit_requires_fresh_five_label_validation(tmp_path):
    cfg = fixture(tmp_path)
    audit_path = Path(cfg.data.audit)
    audit = read(audit_path)
    audit["counts"].pop("fine_label_version")
    audit["counts"].pop("fine_label_order")
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(DataError, match="incompatible fine label profile.*check-data"):
        run_training(cfg, tmp_path / "run", backend_factory=lambda c: pytest.fail("loaded weights before audit check"))


def trajectory(monkeypatch):
    original = run.evaluate
    scores, thresholds = iter([.2, .3, .3, .25, .29]), []

    def evaluate(records, logits, limits, *, language):
        report = original(records, logits, limits, language=language)
        if len(thresholds) < 5:
            thresholds.append(limits)
            score = next(scores)
            report.metrics.update(coarse_macro_f1=score * 2, fine_end_to_end_macro_f1=0.,
                                  fine_conditional_macro_f1=.99)
        return report

    monkeypatch.setattr(run, "evaluate", evaluate)
    return thresholds


def test_training_reads_only_train_and_val(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    forbidden = (tmp_path / "test.jsonl").resolve()
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.resolve() != forbidden, "runner opened test labels"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory) == tmp_path / "run"
    assert read(tmp_path / "run/provenance.json")["data_hashes"]["train"] == hashlib.sha256(Path(cfg.data.train).read_bytes()).hexdigest()


def test_best_uses_fixed_threshold_score(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, n_train=17, epochs=10)
    thresholds = trajectory(monkeypatch)
    output = run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    best = state(output / "model/best")
    assert best["epoch"] == 2 and best["selection_score"] == pytest.approx(.3)
    assert len(thresholds) == 5
    assert all(t.coarse == (.5,) * 3 and t.fine == (.5,) * 5 for t in thresholds)


def test_patience_and_last_partial_batch(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, n_train=17, epochs=10)
    trajectory(monkeypatch)
    original, counts = run.train_effective_batch, []

    def step(model, pairs, *args, **kwargs):
        counts.append(len(pairs))
        return original(model, pairs, *args, **kwargs)

    monkeypatch.setattr(run, "train_effective_batch", step)
    output = run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    assert counts == [16, 1] * 5
    assert state(output / "model/last")["epoch"] == 5
    summary = read(output / "logs/run.json")
    assert summary["successful_steps"] == 10
    assert summary["best_epoch"] == 2 and summary["stopped_epoch"] == 5


def test_calibration_reloads_best(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, n_train=17, epochs=10)
    trajectory(monkeypatch)
    captured = []

    def calibrate(records, logits, **kwargs):
        captured.append(logits)
        assert all(record.language == "csharp" for record in records)
        return calibrate_thresholds(records, logits, **kwargs)

    monkeypatch.setattr(run, "calibrate_thresholds", calibrate)
    output = run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    records = load_records(Path(cfg.data.val))
    predictions = []
    for name in ("best", "last"):
        bundle = load_checkpoint(output / f"model/{name}", device="cpu", backend_factory=fake_backend_factory)
        inputs, _ = collate_pairs([prepare_model_input(r, bundle.tokenizer, bundle.config) for r in records], 0)
        with torch.no_grad():
            predictions.append(bundle.model(inputs))
    torch.testing.assert_close(captured[0].coarse, predictions[0].coarse)
    assert not torch.allclose(captured[0].coarse, predictions[1].coarse)
    calibrated = load_checkpoint(output / "model/best", device="cpu", thresholds_path=output / "model/thresholds.json",
                                 backend_factory=fake_backend_factory)
    assert calibrated.calibrated


def test_existing_run_is_not_overwritten(tmp_path):
    cfg = fixture(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "old.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        run_training(cfg, output, backend_factory=lambda c: pytest.fail("loaded weights before output check"))
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


def test_small_real_training_updates_adapter_and_heads_and_writes_artifacts(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, n_train=6)
    starts = []
    original_build, heads = run.build_model, []

    def build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        heads.append(model.heads[0].weight.detach().clone())
        return model

    monkeypatch.setattr(run, "build_model", build)

    def factory(requested):
        backend = fake_backend_factory(requested)
        starts.append(backend.encoder.adapter.weight.detach().clone())
        return backend

    (tmp_path / "run").mkdir()
    output = run_training(cfg, tmp_path / "run", backend_factory=factory)
    bundle = load_checkpoint(output / "model/best", device="cpu", backend_factory=fake_backend_factory)
    assert not torch.equal(bundle.model.encoder.adapter.weight, starts[0])
    assert not torch.equal(bundle.model.heads[0].weight, heads[0])
    steps = [json.loads(line) for line in (output / "logs/train.jsonl").read_text().splitlines()]
    assert steps[0]["n_pairs"] == 6 and steps[0]["n_modify"] == 3
    assert steps[0]["optimizer_updated"] is True
    for name in ["config.json", "provenance.json", "labels.json", "rng_state.json", "model/thresholds.json",
                 "logs/environment.json", "logs/epochs.jsonl", "logs/run.json",
                 "logs/validation.json", "logs/validation_predictions.jsonl"]:
        assert (output / name).is_file(), name
    assert read(output / "config.json")["encoder"]["revision"] == "0" * 40
    assert read(output / "logs/run.json")["formal"] is True
    assert read(output / "labels.json") == {"coarse": list(COARSE_LABELS), "fine": list(FINE_LABELS)}
    rng = read(output / "rng_state.json")
    assert rng["initial"]["python"] == json.loads(json.dumps(random.Random(cfg.train.seed).getstate()))
    assert rng["initial"]["numpy"]["keys"] == np.random.RandomState(cfg.train.seed).get_state()[1].tolist()
    assert rng["initial"]["torch_cpu"] == torch.Generator().manual_seed(cfg.train.seed).get_state().tolist()
    for snapshot in (rng["initial"], rng["final"]):
        assert set(snapshot) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert len(snapshot["python"]) == 3
        assert len(snapshot["numpy"]["keys"]) == 624
        assert len(snapshot["torch_cpu"]) > 0
    records = [json.loads(line) for line in (output / "logs/validation_predictions.jsonl").read_text().splitlines()]
    assert [r["record_id"] for r in records] == [f"val-{i}" for i in range(6)]


@pytest.mark.parametrize("changed", ["train", "val", "manifest", "audit"])
def test_stale_or_failed_audit_rejected_before_backend(tmp_path, changed):
    cfg = fixture(tmp_path)
    path = Path(getattr(cfg.data, changed))
    if changed == "audit":
        payload = read(path)
        payload["passed"] = False
        path.write_text(json.dumps(payload))
    else:
        path.write_text(path.read_text() + "\n")
    with pytest.raises(DataError):
        run_training(cfg, tmp_path / "run", backend_factory=lambda c: pytest.fail("unaudited backend load"))


def test_debug_mode_marks_nonformal_without_fabricating_metadata(tmp_path):
    cfg = fixture(tmp_path, mode="debug")
    rows = [row(i, "train") for i in range(3)]
    for value in rows:
        for key in ("repo_id", "source_id", "commit_sha", "language"):
            value.pop(key)
    Path(cfg.data.train).write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = replace(cfg, data=replace(cfg.data, audit=None, manifest=None))
    output = run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    assert read(output / "logs/run.json")["formal"] is False
    assert read(output / "logs/validation.json")["provenance"]["formal"] is False


def test_resource_failure_records_context_and_aborts(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)

    def exhausted(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("controlled test OOM")

    monkeypatch.setattr(run, "train_effective_batch", exhausted)
    with pytest.raises(ResourceError, match="capacity"):
        run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    failure = read(tmp_path / "run/logs/failure.json")
    assert failure["phase"] == "train"
    assert {r["record_id"] for r in failure["records"]} == {f"train-{i}" for i in range(3)}
    assert all(r["window_count"] > 0 for r in failure["records"])
    assert failure["config"]["train"]["effective_batch"] == 16


def test_best_window_preparation_resource_failure_has_record_context(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    original, calls = run.prepare_model_input, []

    def prepare(record, *args):
        calls.append(record.record_id)
        if len(calls) > 9:  # Three train plus six val preparations precede reload.
            raise MemoryError("controlled best preparation exhaustion")
        return original(record, *args)

    monkeypatch.setattr(run, "prepare_model_input", prepare)
    with pytest.raises(ResourceError):
        run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    failure = read(tmp_path / "run/logs/failure.json")
    assert failure["phase"] == "prepare_best"
    assert failure["records"] == [{"record_id": "val-0", "window_count": None}]


def test_wrapped_best_reload_oom_is_reported_as_resource_failure(tmp_path):
    cfg = fixture(tmp_path)
    calls = []

    def factory(requested):
        calls.append(True)
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("controlled best reload OOM")
        return fake_backend_factory(requested)

    with pytest.raises(ResourceError):
        run_training(cfg, tmp_path / "run", backend_factory=factory)
    failure = read(tmp_path / "run/logs/failure.json")
    assert failure["phase"] == "reload_best"
    assert failure["error_type"] == "OutOfMemoryError"


@pytest.mark.parametrize("name", ["last", "best"])
@pytest.mark.parametrize("error", [torch.cuda.OutOfMemoryError, MemoryError, ResourceError])
def test_checkpoint_publication_resource_failure_has_context(tmp_path, monkeypatch, name, error):
    cfg = fixture(tmp_path)
    original = checkpoint.save_file

    def save(tensors, destination, *args, **kwargs):
        if name in Path(destination).parts:
            raise error("controlled checkpoint publication exhaustion")
        return original(tensors, destination, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "save_file", save)
    with pytest.raises(ResourceError, match="capacity") as raised:
        run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    assert isinstance(raised.value.__cause__, error)
    failure = read(tmp_path / "run/logs/failure.json")
    assert failure["phase"] == f"save_{name}"
    assert failure["error_type"] == error.__name__
    assert {r["record_id"] for r in failure["records"]} == {f"train-{i}" for i in range(3)}
    assert all(r["window_count"] > 0 for r in failure["records"])
    assert failure["config"]["encoder"]["revision"] == "0" * 40
    assert not (tmp_path / "run/logs/run.json").exists()


@pytest.mark.parametrize("name", ["last", "best"])
def test_checkpoint_publication_preserves_ordinary_artifact_errors(tmp_path, monkeypatch, name):
    cfg = fixture(tmp_path)
    original = checkpoint.save_file
    failure = ArtifactError("ordinary checkpoint publication error")

    def save(tensors, destination, *args, **kwargs):
        if name in Path(destination).parts:
            raise failure
        return original(tensors, destination, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "save_file", save)
    with pytest.raises(ArtifactError) as raised:
        run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    assert raised.value is failure
    assert not (tmp_path / "run/logs/failure.json").exists()


def test_seed_reproduces_order_and_best_predictions(tmp_path):
    cfg = fixture(tmp_path, n_train=17)
    outputs = [run_training(cfg, tmp_path / name, backend_factory=fake_backend_factory) for name in ("a", "b")]
    assert (outputs[0] / "logs/train.jsonl").read_bytes() == (outputs[1] / "logs/train.jsonl").read_bytes()
    assert (outputs[0] / "logs/validation_predictions.jsonl").read_bytes() == (outputs[1] / "logs/validation_predictions.jsonl").read_bytes()
    assert read(outputs[0] / "rng_state.json")["initial"] == read(outputs[1] / "rng_state.json")["initial"]


def test_fp16_reuses_scaler_and_skips_scheduler_on_overflow(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, n_train=17)
    cfg = replace(cfg, encoder=replace(cfg.encoder, dtype="fp16"))
    scalers, advances = [], []
    original_step, original_scheduler = run.train_effective_batch, run.build_scheduler

    def factory(requested):
        backend = fake_backend_factory(requested)
        backend.encoder.compute_dtype = torch.float16
        return backend

    monkeypatch.setattr(run, "GradScaler", lambda *args, **kwargs: torch.amp.GradScaler("cpu", init_scale=32))

    def scheduler(*args, **kwargs):
        result = original_scheduler(*args, **kwargs)
        advance = result.step
        def step():
            advances.append(True)
            advance()
        monkeypatch.setattr(result, "step", step)
        return result

    def update(model, pairs, *args, **kwargs):
        scalers.append(kwargs["scaler"])
        if len(scalers) == 1:
            handle = model.register_forward_hook(lambda module, args, logits:
                PairLogits(logits.coarse * float("inf"), logits.fine))
            try:
                return original_step(model, pairs, *args, **kwargs)
            finally:
                handle.remove()
        return original_step(model, pairs, *args, **kwargs)

    monkeypatch.setattr(run, "build_scheduler", scheduler)
    monkeypatch.setattr(run, "train_effective_batch", update)
    output = run_training(cfg, tmp_path / "run", backend_factory=factory)
    assert len(scalers) == 2 and scalers[0] is scalers[1]
    assert advances == [True]
    assert read(output / "logs/run.json")["successful_steps"] == 1
    steps = [json.loads(line) for line in (output / "logs/train.jsonl").read_text().splitlines()]
    assert [s["optimizer_updated"] for s in steps] == [False, True]
