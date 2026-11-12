"""Behavioral regressions for consumed provenance and complete diagnostics."""
from argparse import Namespace
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError

from helpers import FakeTokenizer, fake_backend_factory, make_record
from model import cli, main
from model.data.audit import audit_dataset
from model.data.jsonl import load_records
from model.errors import ArtifactError, DataError
from model.evaluate.decode import Thresholds
from model.evaluate.report import summarize_predictions
from model.input.windows import prepare_pair
from model.config import WindowConfig
from model.models.base import PairLogits
from model.train import checkpoint, run
from test_audit import manifest, record
from test_checkpoint import fixture as checkpoint_fixture, generation, read_json
from test_training import fixture, read, row
from test_workflow import jsonlines, train


def replace_after_read(monkeypatch, target, consumed, replacement):
    """An opened stream keeps its bytes while a producer replaces the path."""
    original = Path.open
    reads = []
    def raced(path, mode="r", *args, **kwargs):
        if path.resolve() == target.resolve() and mode in ("r", "rb") and not reads:
            reads.append(mode)
            with original(target, "wb") as output:
                output.write(replacement)
            return io.BytesIO(consumed) if mode == "rb" else io.StringIO(consumed.decode("utf-8"), newline=None)
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", raced)
    return reads


def test_audit_hash_and_counts_use_one_replaced_snapshot(tmp_path, monkeypatch):
    path = manifest(tmp_path)
    data = tmp_path / "pairs-0.jsonl"
    original = data.read_bytes()
    changed = json.loads(original)
    changed.update(labels=["None"], modify_subtypes=[])
    replace_after_read(monkeypatch, data, original, (json.dumps(changed) + "\n").encode())
    report = audit_dataset(path)
    assert report.passed, report.issues
    assert report.file_sha256[data.name] == hashlib.sha256(original).hexdigest()
    assert report.counts["coarse"]["Modify"] == 1
    assert report.counts["none"] == 0


def test_formal_train_hash_and_records_use_one_same_count_snapshot(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    data = Path(cfg.data.train)
    original = data.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["pre_code"] = "replacement source();"
    replace_after_read(monkeypatch, data, original, "".join(json.dumps(r) + "\n" for r in rows).encode())
    records, hashes = run._data(cfg)
    assert records["train"][0].pre_code == row(0, "train")["pre_code"]
    assert hashes["train"] == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("name", ["manifest", "audit"])
def test_training_json_metadata_hashes_consumed_snapshot(tmp_path, monkeypatch, name):
    cfg = fixture(tmp_path)
    path = Path(getattr(cfg.data, name))
    original = path.read_bytes()
    # The declaration remains semantically identical, but exact bytes change.
    replace_after_read(monkeypatch, path, original, original + b" \n")
    _, hashes = run._data(cfg)
    assert hashes[name] == hashlib.sha256(original).hexdigest()


def test_eval_rejects_transient_same_count_replacement_even_if_restored(tmp_path, monkeypatch):
    cfg = fixture(tmp_path)
    manifest_path, data = Path(cfg.data.manifest), tmp_path / "test.jsonl"
    audit = audit_dataset(manifest_path)
    original = data.read_bytes()
    changed = json.loads(original)
    changed["pre_code"] = "transient replacement();"
    replacement = (json.dumps(changed) + "\n").encode()
    monkeypatch.setattr(cli, "audit_dataset", lambda *args, **kwargs: audit)
    replace_after_read(monkeypatch, data, replacement, original)
    with pytest.raises(DataError, match="changed after audit"):
        cli._eval_data(Namespace(manifest=manifest_path, data=data, language="java"))


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_snapshot_parser_keeps_unicode_and_physical_lines(tmp_path, newline):
    path = tmp_path / "unicode.jsonl"
    source = "code\u2028inside\u0085string"
    value = json.dumps(dict(pre_code=source, post_code="new", labels=[]), ensure_ascii=False)
    path.write_bytes((newline + value + newline + newline).encode())
    assert load_records(path)[0].pre_code == source
    path.write_bytes((newline + value + newline + newline + "BAD" + newline).encode())
    with pytest.raises(DataError, match=r"unicode.jsonl:4"):
        load_records(path)


def test_debug_checkpoint_eval_preserves_separate_provenance(tmp_path):
    cfg = fixture(tmp_path, mode="debug")
    output = run.run_training(cfg, tmp_path / "run", backend_factory=fake_backend_factory)
    report = tmp_path / "report"
    assert main(["eval", "--checkpoint", str(output / "model/best"), "--data", str(tmp_path / "test.jsonl"),
                 "--manifest", cfg.data.manifest, "--language", "java", "--output", str(report)],
                backend_factory=fake_backend_factory) == 0
    provenance = read(report / "metrics.json")["provenance"]
    assert provenance["formal"] is False
    assert provenance["training"] == {"mode": "debug", "formal": False}
    assert provenance["evaluation"]["formal"] is True
    assert provenance["evaluation"]["data_sha256"] == hashlib.sha256((tmp_path / "test.jsonl").read_bytes()).hexdigest()


def corrupt_tensor(path, name):
    folder = generation(path)
    (folder / name).write_bytes(b"malformed safetensors")
    complete = read_json(folder / "COMPLETE.json")
    complete["files"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    digest = checkpoint._digest_files(complete["files"])
    complete["digest"] = digest
    (folder / "COMPLETE.json").write_text(json.dumps(complete))
    pointer = read_json(path / "CURRENT.json")
    pointer["digest"] = digest
    (path / "CURRENT.json").write_text(json.dumps(pointer))
    return digest


@pytest.mark.parametrize("name", ["task.safetensors", "adapter.safetensors"])
def test_checksummed_malformed_tensor_is_artifact_error(tmp_path, name):
    model, cfg, state, _ = checkpoint_fixture()
    checkpoint.save_checkpoint(tmp_path, model, cfg, state)
    corrupt_tensor(tmp_path, name)
    with pytest.raises(ArtifactError) as error:
        checkpoint.load_checkpoint(tmp_path, device="cpu", backend_factory=fake_backend_factory)
    assert isinstance(error.value.__cause__, SafetensorError)


def test_tensor_restoration_runtime_is_still_artifact_error(tmp_path, monkeypatch):
    model, cfg, state, _ = checkpoint_fixture()
    checkpoint.save_checkpoint(tmp_path, model, cfg, state)
    original = fake_backend_factory
    def factory(cfg):
        backend = original(cfg)
        def invalid(_):
            raise RuntimeError("corrupt adapter tensor state")
        backend.encoder.load_adapter_state_dict = invalid
        return backend
    with pytest.raises(ArtifactError) as error:
        checkpoint.load_checkpoint(tmp_path, device="cpu", backend_factory=factory)
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize("command", ["predict", "eval"])
def test_malformed_tensor_cli_exit_and_eval_failure_ledger(tmp_path, command):
    output = train(tmp_path)
    best = output / "model/best"
    digest = corrupt_tensor(best, "task.safetensors")
    thresholds_path = output / "model/thresholds.json"
    thresholds = read(thresholds_path)
    thresholds["checkpoint_digest"] = digest
    thresholds_path.write_text(json.dumps(thresholds))
    report = tmp_path / "result"
    args = [command, "--checkpoint", str(best), "--output", str(report)]
    if command == "predict":
        args += ["--input", str(tmp_path / "test.jsonl")]
    else:
        args += ["--data", str(tmp_path / "test.jsonl"), "--manifest", str(tmp_path / "manifest.json"), "--language", "java"]
    assert main(args, backend_factory=fake_backend_factory) == 2
    if command == "eval":
        failures = jsonlines(report / "failures.jsonl")
        assert [r["record_id"] for r in failures] == ["test-0"]
        assert failures[0]["error_type"] == "ArtifactError" and failures[0]["checkpoint_digest"] == digest
        assert failures[0]["window_count"] is None and failures[0]["window_config"] is None
        assert read(report / "metrics.json")["metrics"] is None


def test_eval_preparation_failure_has_null_window_and_known_config(tmp_path, monkeypatch):
    output = train(tmp_path)
    def exhausted(*args):
        raise MemoryError("controlled preparation exhaustion")
    monkeypatch.setattr(cli, "prepare_model_input", exhausted)
    report = tmp_path / "report"
    assert main(["eval", "--checkpoint", str(output / "model/best"), "--data", str(tmp_path / "test.jsonl"),
                 "--manifest", str(tmp_path / "manifest.json"), "--language", "java", "--output", str(report)],
                backend_factory=fake_backend_factory) == 1
    failure = jsonlines(report / "failures.jsonl")[0]
    assert failure["window_count"] is None
    assert failure["window_config"]["window_tokens"] == 256
    assert failure["encoder_config"]["device"] == "cpu"


def test_eval_forward_oom_retains_prepared_context_and_all_ids(tmp_path, monkeypatch):
    output = train(tmp_path)
    data = tmp_path / "test.jsonl"
    data.write_text("".join(json.dumps(dict(row(i, "test"), language="Java")) + "\n" for i in range(3)))
    manifest_path = tmp_path / "manifest.json"
    value = read(manifest_path)
    value["files"][-1].update(n_records=3, sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(value))
    original = cli._bundle
    def bundle(args, factory):
        result = original(args, factory)
        forward = result.model.forward
        calls = 0
        def exhausted(inputs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise torch.cuda.OutOfMemoryError("controlled forward exhaustion")
            return forward(inputs)
        monkeypatch.setattr(result.model, "forward", exhausted)
        return result
    monkeypatch.setattr(cli, "_bundle", bundle)
    report = tmp_path / "report"
    assert main(["eval", "--checkpoint", str(output / "model/best"), "--data", str(data),
                 "--manifest", str(manifest_path), "--language", "java", "--output", str(report)],
                backend_factory=fake_backend_factory) == 1
    failures = jsonlines(report / "failures.jsonl")
    assert [r["record_id"] for r in failures] == ["test-1", "test-2"]
    assert failures[0]["window_count"] > 0
    assert failures[0]["window_config"]["window_tokens"] == 256
    assert failures[0]["model_config"]["architecture"] == "hierarchical"
    assert failures[1]["window_count"] is None
    assert failures[1]["window_config"] is None and failures[1]["model_config"] is None
    metrics = read(report / "metrics.json")
    assert metrics["metrics"] is None
    assert metrics["counts"] == dict(n_requested=3, n_predicted=1, n_failed=2)
    assert metrics["diagnostics"]["failure_context"] == failures


@pytest.mark.parametrize("language", ["csharp", "java", "cpp"])
def test_shared_fine_profile_is_preserved_in_strata(language):
    records = [replace(make_record("old", "new", labels=["None"]), record_id=str(i), language=language) for i in range(3)]
    logits = PairLogits(torch.tensor([[-2., 2., -2.]] * 3), torch.zeros(3, 5))
    pairs = [prepare_pair(r, FakeTokenizer(), WindowConfig()) for r in records]
    report = summarize_predictions(pairs, logits, Thresholds(), language=language, bootstrap_samples=1)
    assert not any(key.startswith("illegal_finally") for key in report.diagnostics)
    stratum = report.diagnostics["length_buckets"]["0-1024"]
    assert stratum["diagnostics"] == {}
    assert stratum["metrics"]["fine_end_to_end"]["per_label"] == report.metrics["fine_end_to_end"]["per_label"]
    assert len(report.metrics["fine_end_to_end"]["per_label"]) == 5


def test_audit_counts_normalized_sides_and_all_empty_modify(tmp_path):
    rows = [record(source=f"s-{i}", pre=pre, post=post, labels=labels, modify_subtypes=fine)
            for i, (pre, post, labels, fine) in enumerate([
                (" \t", "added", ["Add"], []), ("deleted", "\n ", ["Delete"], []),
                ("a", "b", ["Modify"], []), ("c", "d", ["Modify"], ["catch_filter_changed"]),
                ("e", "f", ["Modify"], ["catch_filter_changed", "catch_added"])])]
    report = audit_dataset(manifest(tmp_path, [("train", rows, {})]))
    assert report.passed, report.issues
    assert report.counts["pre_only"] == 1 and report.counts["post_only"] == 1
    assert report.counts["empty_fine_modify"] == 2
    assert report.counts["filter_only"] == 1 and report.counts["removed_filter"] == 2


def test_bootstrap_explains_zero_valid_draws_with_original_support():
    positive = replace(make_record("a", "b", labels=["Modify"], subtypes=["try_body_changed"]), repo_id="positive", record_id="p")
    negative = replace(make_record("c", "d", labels=["None"]), repo_id="negative", record_id="n")
    pairs = [prepare_pair(r, FakeTokenizer(), WindowConfig()) for r in (positive, negative)]
    # NumPy seed 0 selects the negative group twice in this single draw.
    report = summarize_predictions(pairs, PairLogits(torch.zeros(2, 3), torch.zeros(2, 5)), Thresholds(),
                                   language="csharp", bootstrap_seed=0, bootstrap_samples=1)
    for metric in ("coarse_macro_f1", "fine_end_to_end_macro_f1", "fine_common_end_to_end_macro_f1"):
        item = report.diagnostics["bootstrap"][metric]
        assert item["support"] > 0 and item["valid_samples"] == 0 and item["invalid_samples"] == 1
        assert item["ci95"] is None and item["reason"] == "no valid supported resamples"
