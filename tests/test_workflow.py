"""CLI integration uses real model updates and persisted inference artifacts."""

from dataclasses import asdict, replace
import hashlib
import json

import pytest
import torch

from helpers import fake_backend_factory
from model import main, predict_pair
from model.errors import ArtifactError
from model.train.checkpoint import load_checkpoint
from test_training import fixture, read, row


def config_file(tmp_path):
    cfg = fixture(tmp_path)
    payload = asdict(cfg)
    for name in ("train", "val", "manifest", "audit"):
        payload["data"][name] = getattr(cfg.data, name).rsplit("/", 1)[-1]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def train(tmp_path):
    path = config_file(tmp_path)
    output = tmp_path / "run"
    assert main(["train", "--config", str(path), "--output", str(output)],
                backend_factory=fake_backend_factory) == 0
    return output


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_train_eval_predict_fake_roundtrip(tmp_path, monkeypatch):
    path = config_file(tmp_path)
    # Relative config paths must work independently of the caller's cwd.
    monkeypatch.chdir(tmp_path.parent)
    audit = tmp_path / "checked.json"
    assert main(["check-data", "--manifest", str(tmp_path / "manifest.json"), "--output", str(audit),
                 "--mode", "formal"], backend_factory=fake_backend_factory) == 0
    assert read(audit)["passed"] is True
    output = tmp_path / "run"
    assert main(["train", "--config", str(path), "--output", str(output)],
                backend_factory=fake_backend_factory) == 0
    best = output / "model/best"
    digest = read(best / "CURRENT.json")["digest"]
    assert read(output / "model/thresholds.json")["checkpoint_digest"] == digest
    report = tmp_path / "report"
    args = ["eval", "--checkpoint", str(best), "--data", str(tmp_path / "test.jsonl"),
            "--manifest", str(tmp_path / "manifest.json"), "--language", "java", "--output", str(report)]
    assert main(args, backend_factory=fake_backend_factory) == 0
    metrics = read(report / "metrics.json")
    assert metrics["counts"]["n_samples"] == 1
    assert metrics["provenance"]["language"] == "java"
    assert metrics["provenance"]["checkpoint_digest"] == digest
    assert metrics["provenance"]["data_sha256"] == hashlib.sha256((tmp_path / "test.jsonl").read_bytes()).hexdigest()
    assert (report / "failures.jsonl").read_text() == ""
    pairs = tmp_path / "pairs.jsonl"
    values = [dict(record_id="second", pre_code=None, post_code="plain();"),
              dict(record_id="first", pre_code="old();", post_code=None)]
    pairs.write_text("".join(json.dumps(value) + "\n" for value in values))
    predictions = tmp_path / "predictions.jsonl"
    assert main(["predict", "--checkpoint", str(best), "--input", str(pairs), "--output", str(predictions)],
                backend_factory=fake_backend_factory) == 0
    results = jsonlines(predictions)
    assert [value["record_id"] for value in results] == ["second", "first"]
    for value in results + jsonlines(report / "predictions.jsonl"):
        assert len(value["coarse_scores"]) == 3 and len(value["fine_scores"]) == 5
        assert value["checkpoint_digest"] == digest
        assert isinstance(value["labels"], list) and isinstance(value["modify_subtypes"], list)
    assert main(args, backend_factory=fake_backend_factory) == 2


def test_predict_needs_no_language_or_ast(tmp_path, monkeypatch):
    output = train(tmp_path)
    bundle = load_checkpoint(output / "model/best", device="cpu", thresholds_path=output / "model/thresholds.json",
                             backend_factory=fake_backend_factory)
    # Force independently known outputs through actual model inference.
    with torch.no_grad():
        for parameter in bundle.model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    from model.evaluate.decode import Thresholds
    absent = replace(bundle, thresholds=Thresholds((1.,) * 3, (1.,) * 5))
    none = predict_pair(absent, "same();", "same();")
    assert none.coarse_labels == ("None",) and none.fine_labels == ()
    assert none.coarse_scores == (.5,) * 3 and none.fine_scores == (.5,) * 5
    from model import cli
    monkeypatch.setattr(cli, "_bundle", lambda args, factory: absent)
    pairs = tmp_path / "only-source.jsonl"
    pairs.write_text('{"record_id":"none","pre_code":"same();","post_code":"same();"}\n')
    predicted = tmp_path / "none.jsonl"
    assert main(["predict", "--checkpoint", str(output / "model/best"), "--input", str(pairs),
                 "--output", str(predicted)]) == 0
    assert jsonlines(predicted)[0]["labels"] == ["None"]
    assert jsonlines(predicted)[0]["modify_subtypes"] == []
    modified = replace(bundle, thresholds=Thresholds((1., 0., 1.), (1.,) * 5))
    assert predict_pair(modified, "old();", "new();").coarse_labels == ("Modify",)
    assert predict_pair(modified, "old();", "new();").fine_labels == ()
    with pytest.raises(ArtifactError, match="calibrat"):
        predict_pair(replace(bundle, calibrated=False), "old();", "new();")


def test_wrong_checkpoint_threshold_digest_rejected_before_backend(tmp_path):
    output = train(tmp_path)
    thresholds = output / "model/thresholds.json"
    payload = read(thresholds)
    payload["checkpoint_digest"] = "f" * 64
    thresholds.write_text(json.dumps(payload))
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"pre_code":"old();","post_code":"new();"}\n')
    def forbidden(cfg):
        pytest.fail("mismatched calibration reached encoder loading")
    assert main(["predict", "--checkpoint", str(output / "model/best"), "--thresholds", str(thresholds),
                 "--input", str(pairs), "--output", str(tmp_path / "pred.jsonl")], backend_factory=forbidden) == 2


def test_missing_thresholds_rejected_for_inference(tmp_path):
    output = train(tmp_path)
    (output / "model/thresholds.json").unlink()
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"pre_code":"old();","post_code":"new();"}\n')
    def forbidden(cfg):
        pytest.fail("missing thresholds must be rejected before encoder loading")
    assert main(["predict", "--checkpoint", str(output / "model/best"), "--input", str(pairs),
                 "--output", str(tmp_path / "pred.jsonl")], backend_factory=forbidden) == 2


def test_eval_audits_test_hash_and_language_before_backend(tmp_path):
    output = train(tmp_path)
    def forbidden(cfg):
        pytest.fail("unaudited evaluation reached encoder loading")
    args = ["eval", "--checkpoint", str(output / "model/best"), "--data", str(tmp_path / "test.jsonl"),
            "--manifest", str(tmp_path / "manifest.json"), "--output", str(tmp_path / "report")]
    assert main(args + ["--language", "cpp"], backend_factory=forbidden) == 2
    with (tmp_path / "test.jsonl").open("a") as stream:
        stream.write(" \n")
    assert main(args + ["--language", "java"], backend_factory=forbidden) == 2


@pytest.mark.parametrize("language,formal_name", [("csharp", "C#"), ("java", "Java"), ("cpp", "C++")])
def test_eval_uses_manifest_field_mapping_and_normalizes_metadata(tmp_path, language, formal_name):
    output = train(tmp_path)
    value = row(0, "test")
    value["language"] = formal_name
    value["before"], value["after"] = value.pop("pre_code"), value.pop("post_code")
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(value) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = read(manifest_path)
    manifest["files"][-1].update(language=formal_name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                field_map={"pre_code": "before", "post_code": "after"})
    manifest_path.write_text(json.dumps(manifest))
    report = tmp_path / "report"
    assert main(["eval", "--checkpoint", str(output / "model/best"), "--data", str(path),
                 "--manifest", str(manifest_path), "--language", language, "--output", str(report)],
                backend_factory=fake_backend_factory) == 0
    assert read(report / "metrics.json")["provenance"]["language"] == language


def test_eval_resource_failure_accounts_for_all_records(tmp_path, monkeypatch):
    output = train(tmp_path)
    path = tmp_path / "test.jsonl"
    rows = [dict(row(i, "test"), language="Java") for i in range(3)]
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))
    manifest_path = tmp_path / "manifest.json"
    manifest = read(manifest_path)
    manifest["files"][-1].update(n_records=3, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    from model import cli
    original = cli._pair_logits
    calls = 0
    def exhausted(bundle, record, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.cuda.OutOfMemoryError("controlled evaluation capacity failure")
        return original(bundle, record, **kwargs)
    monkeypatch.setattr(cli, "_pair_logits", exhausted)
    report = tmp_path / "report"
    assert main(["eval", "--checkpoint", str(output / "model/best"), "--data", str(path),
                 "--manifest", str(manifest_path), "--language", "java", "--output", str(report)],
                backend_factory=fake_backend_factory) == 1
    assert [value["record_id"] for value in jsonlines(report / "predictions.jsonl")] == ["test-0"]
    failures = jsonlines(report / "failures.jsonl")
    assert [value["record_id"] for value in failures] == ["test-1", "test-2"]
    assert all(value["reason"] and value["checkpoint_digest"] for value in failures)
    metrics = read(report / "metrics.json")
    assert metrics["counts"] == {"n_requested": 3, "n_predicted": 1, "n_failed": 2}
    assert metrics["metrics"] is None and metrics["provenance"]["status"] == "failed"


def test_eval_backend_resource_failure_retains_checkpoint_hash(tmp_path):
    output = train(tmp_path)
    digest = read(output / "model/best/CURRENT.json")["digest"]
    def exhausted(cfg):
        raise torch.cuda.OutOfMemoryError("controlled load exhaustion")
    report = tmp_path / "report"
    assert main(["eval", "--checkpoint", str(output / "model/best"), "--data", str(tmp_path / "test.jsonl"),
                 "--manifest", str(tmp_path / "manifest.json"), "--language", "java", "--output", str(report)],
                backend_factory=exhausted) == 1
    assert (report / "predictions.jsonl").read_text() == ""
    failures = jsonlines(report / "failures.jsonl")
    assert failures[0]["record_id"] == "test-0"
    assert failures[0]["checkpoint_digest"] == digest
    assert failures[0]["window_count"] is None
    assert failures[0]["window_config"] is None and failures[0]["model_config"] is None
    assert failures[0]["encoder_config"] is None
    assert read(report / "metrics.json")["provenance"]["checkpoint_digest"] == digest


def test_eval_rejects_manifest_changed_after_audit(tmp_path, monkeypatch):
    output = train(tmp_path)
    from model import cli
    original = cli.audit_dataset
    def raced(path, **kwargs):
        report = original(path, **kwargs)
        value = read(path)
        value["split_seed"] = 42
        path.write_text(json.dumps(value))
        return report
    monkeypatch.setattr(cli, "audit_dataset", raced)
    def forbidden(cfg):
        pytest.fail("changed manifest reached backend")
    assert main(["eval", "--checkpoint", str(output / "model/best"), "--data", str(tmp_path / "test.jsonl"),
                 "--manifest", str(tmp_path / "manifest.json"), "--language", "java",
                 "--output", str(tmp_path / "report")], backend_factory=forbidden) == 2
