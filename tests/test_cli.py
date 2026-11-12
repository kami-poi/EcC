"""CLI validation and run/resource exit categories."""

import json

import pytest
import torch

from model import main
from model.errors import ResourceError
from test_workflow import config_file, train


def test_check_data_before_encoder_load(tmp_path):
    path = config_file(tmp_path)
    records = tmp_path / "train.jsonl"
    value = json.loads(records.read_text().splitlines()[0])
    value["labels"] = ["invalid"]
    records.write_text(json.dumps(value) + "\n")
    def forbidden(cfg):
        pytest.fail("invalid labels reached the encoder factory")
    assert main(["check-data", "--manifest", str(tmp_path / "manifest.json"), "--mode", "formal",
                 "--output", str(tmp_path / "bad-audit.json")], backend_factory=forbidden) == 2
    assert json.loads((tmp_path / "bad-audit.json").read_text())["passed"] is False
    assert main(["train", "--config", str(path), "--output", str(tmp_path / "run")],
                backend_factory=forbidden) == 2


@pytest.mark.parametrize("argv", [[], ["unknown"], ["train"], ["eval", "--language", "python"],
                                  ["predict", "--fake-backend"], ["train", "--config", "absent", "--output", "unused"]])
def test_invalid_cli_exit_codes(argv):
    assert main(argv) == 2


def test_cli_help_is_success():
    assert main(["--help"]) == 0


@pytest.mark.parametrize("exception", [torch.cuda.OutOfMemoryError, MemoryError, ResourceError, RuntimeError])
def test_runtime_and_resource_exit_codes(tmp_path, capsys, exception):
    path = config_file(tmp_path)
    def broken(cfg):
        raise exception("controlled failure")
    assert main(["train", "--config", str(path), "--output", str(tmp_path / "run")],
                backend_factory=broken) == 1
    if exception is not RuntimeError:
        assert "capacity" in capsys.readouterr().err


@pytest.mark.parametrize("exception", [torch.cuda.OutOfMemoryError, MemoryError, ResourceError, RuntimeError])
def test_checkpoint_backend_failure_classification(tmp_path, capsys, exception):
    output = train(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"pre_code":"old();","post_code":"new();"}\n')
    def broken(cfg):
        raise exception("controlled checkpoint backend failure")
    code = main(["predict", "--checkpoint", str(output / "model/best"), "--input", str(pairs),
                 "--output", str(tmp_path / "pred.jsonl")], backend_factory=broken)
    assert code == 1
    if exception is not RuntimeError:
        assert "capacity" in capsys.readouterr().err


def test_predict_invalid_json_line_and_existing_output_rejected_before_backend(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{"pre_code":"old();","post_code":"new();"}\nBAD\n')
    output = tmp_path / "pred.jsonl"
    def forbidden(cfg):
        pytest.fail("invalid prediction request reached backend")
    args = ["predict", "--checkpoint", "absent", "--input", str(pairs), "--output", str(output)]
    assert main(args, backend_factory=forbidden) == 2
    assert not output.exists()
    output.write_text("keep\n")
    assert main(args, backend_factory=forbidden) == 2
    assert output.read_text() == "keep\n"


def test_check_data_preserves_output_created_during_audit(tmp_path, monkeypatch):
    config_file(tmp_path)
    from model import cli
    original = cli.audit_dataset
    output = tmp_path / "competing-audit.json"
    def competing_audit(path, **kwargs):
        report = original(path, **kwargs)
        output.write_text("competing output\n")
        return report
    monkeypatch.setattr(cli, "audit_dataset", competing_audit)
    assert main(["check-data", "--manifest", str(tmp_path / "manifest.json"),
                 "--output", str(output), "--mode", "formal"]) == 2
    assert output.read_text() == "competing output\n"


def test_train_rejects_directory_created_during_config_loading(tmp_path, monkeypatch):
    config = config_file(tmp_path)
    from model import cli
    from helpers import fake_backend_factory
    original = cli.load_config
    output = tmp_path / "competing-run"
    def competing_config(path):
        cfg = original(path)
        output.mkdir()
        return cfg
    monkeypatch.setattr(cli, "load_config", competing_config)
    assert main(["train", "--config", str(config), "--output", str(output)],
                backend_factory=fake_backend_factory) == 2
    assert list(output.iterdir()) == []
