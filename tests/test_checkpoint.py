"""Checkpoint persistence tests use deterministic CPU/FP32 backends only."""

from dataclasses import replace
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file

from model.config import EncoderConfig, ModelConfig, RunConfig, WindowConfig
from model.encode.base import LoadedBackend
from model.errors import ArtifactError
from model.evaluate.decode import CalibrationProvenance, Thresholds
from model.evaluate.thresholds import calibrate_thresholds
from model.input.batch import collate_pairs
from model.input.windows import prepare_pair
from model.labels import COARSE_LABELS, FINE_LABELS, LEGACY_FINE_LABELS
from model.models.base import PairLogits
from model.models.pair import build_model
from model.train import checkpoint
from model.train.checkpoint import CheckpointState, load_checkpoint, save_checkpoint, save_thresholds
from helpers import fake_backend_factory, make_record


def fixture(adaptation="lora"):
    cfg = RunConfig(encoder=EncoderConfig(device="cpu", dtype="fp32", adaptation=adaptation),
                    model=ModelConfig(native_dim=8, projection_dim=8, heads=2, ffn_dim=16),
                    window=WindowConfig(window_tokens=160, overlap_tokens=4))
    backend = fake_backend_factory(cfg)
    model = build_model(backend.encoder, backend.config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(.123)
    record = make_record("try { old(); }", "try { new(); }", labels=["Modify"], subtypes=[FINE_LABELS[0]])
    inputs, _ = collate_pairs([prepare_pair(record, backend.tokenizer, cfg.window)], 0)
    state = CheckpointState(epoch=2, step=7, selection_score=.72,
                            data_hashes={"train": "a" * 64, "val": "b" * 64},
                            tokenizer_special_token_rule=backend.tokenizer.special_token_rule)
    return model, backend.config, state, inputs


def read_json(path):
    return json.loads(path.read_text())


def generation(path):
    pointer = read_json(path / "CURRENT.json")
    return path / "generations" / pointer["generation"]


def rewrite_artifact(path, name, mutate):
    """Re-sign deliberate incompatible artifacts to exercise schema, not checksums."""
    folder = generation(path)
    payload = read_json(folder / name)
    mutate(payload)
    (folder / name).write_text(json.dumps(payload))
    complete = read_json(folder / "COMPLETE.json")
    complete["files"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(complete["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    complete["digest"] = digest
    (folder / "COMPLETE.json").write_text(json.dumps(complete))
    pointer = read_json(path / "CURRENT.json")
    pointer["digest"] = digest
    (path / "CURRENT.json").write_text(json.dumps(pointer))


@pytest.mark.parametrize("adaptation", ["lora", "frozen"])
def test_roundtrip_predictions_and_adapter(tmp_path, adaptation):
    model, cfg, state, inputs = fixture(adaptation)
    path = tmp_path / "best"
    before = model(inputs)
    save_checkpoint(path, model, cfg, state)
    bundle = load_checkpoint(path, device="cpu", backend_factory=fake_backend_factory)
    after = bundle.model(inputs)
    torch.testing.assert_close(after.coarse, before.coarse)
    torch.testing.assert_close(after.fine, before.fine)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(bundle.model.state_dict()[key], value)
    assert not bundle.calibrated and bundle.thresholds == Thresholds()
    assert bundle.config == cfg and not bundle.model.training
    folder = generation(path)
    assert all(not name.startswith("encoder.") for name in load_file(folder / "task.safetensors"))
    assert (folder / "adapter.safetensors").exists() == (adaptation == "lora")
    assert read_json(folder / "state.json")["state"]["data_hashes"] == state.data_hashes
    assert read_json(folder / "state.json")["environment"]["torch"]


@pytest.mark.parametrize("filter_label", [False, True])
def test_old_eight_or_nine_label_head_rejected_before_loading_base(tmp_path, filter_label):
    model, cfg, state, _ = fixture()
    save_checkpoint(tmp_path, model, cfg, state)
    old_labels = [*LEGACY_FINE_LABELS, *(["catch_filter_changed"] if filter_label else [])]
    rewrite_artifact(tmp_path, "labels.json", lambda value: value.update(fine=old_labels))
    with pytest.raises(ArtifactError, match="three/five.*retraining"):
        load_checkpoint(tmp_path, device="cpu", backend_factory=lambda cfg: pytest.fail("loaded incompatible base"))


def test_old_eight_label_model_version_rejected_before_loading_base(tmp_path):
    model, cfg, state, _ = fixture()
    save_checkpoint(tmp_path, model, cfg, state)
    rewrite_artifact(tmp_path, "state.json", lambda value: value["state"].update(model_version="eh-change-3-8-v1"))
    with pytest.raises(ArtifactError, match="version"):
        load_checkpoint(tmp_path, device="cpu", backend_factory=lambda cfg: pytest.fail("loaded incompatible base"))


def test_threshold_digest_mismatch(tmp_path):
    model, cfg, state, _ = fixture()
    save_checkpoint(tmp_path / "best", model, cfg, state)
    thresholds = Thresholds(provenance=CalibrationProvenance("val", "csharp", (0, 2, 0), (1, 0, 0, 0, 0)))
    file = tmp_path / "thresholds.json"
    save_thresholds(file, thresholds, checkpoint_digest="f" * 64)
    with pytest.raises(ArtifactError, match="digest"):
        load_checkpoint(tmp_path / "best", device="cpu", thresholds_path=file, backend_factory=fake_backend_factory)
    digest = read_json(tmp_path / "best" / "CURRENT.json")["digest"]
    save_thresholds(file, thresholds, checkpoint_digest=digest)
    loaded = load_checkpoint(tmp_path / "best", device="cpu", thresholds_path=file, backend_factory=fake_backend_factory)
    assert loaded.calibrated and loaded.thresholds == thresholds
    reasons = read_json(file)["no_positive_reasons"]
    assert reasons["coarse"] == ["no_positive_examples", None, "no_positive_examples"]
    assert reasons["fine"][0] is None  # .5 with positive support is not an absent class.


@pytest.mark.parametrize("stage", ["tensor", "publish"])
def test_atomic_failure_preserves_previous_checkpoint(tmp_path, monkeypatch, stage):
    model, cfg, state, inputs = fixture()
    save_checkpoint(tmp_path, model, cfg, state)
    previous = (tmp_path / "CURRENT.json").read_bytes()
    expected = model(inputs)
    if stage == "tensor":
        def fail(*args, **kwargs):
            args[1].write_bytes(b"partial safetensors write")
            raise OSError("interrupted tensor write")
        monkeypatch.setattr(checkpoint, "save_file", fail)
    else:
        original = checkpoint.os.replace
        def fail(source, destination):
            if str(destination).endswith("CURRENT.json"):
                raise OSError("interrupted pointer publication")
            return original(source, destination)
        monkeypatch.setattr(checkpoint.os, "replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint(tmp_path, model, cfg, replace(state, step=8))
    assert (tmp_path / "CURRENT.json").read_bytes() == previous
    bundle = load_checkpoint(tmp_path, device="cpu", backend_factory=fake_backend_factory)
    torch.testing.assert_close(bundle.model(inputs).fine, expected.fine)
    with pytest.raises(ArtifactError):
        load_checkpoint(generation(tmp_path), device="cpu", backend_factory=fake_backend_factory)
    for staging in (tmp_path / "generations").glob(".tmp-*"):
        with pytest.raises(ArtifactError):
            load_checkpoint(staging, device="cpu", backend_factory=fake_backend_factory)


@pytest.mark.parametrize("change", ["revision", "tokenizer_revision", "window", "wrapper", "native_dim", "lora_modules"])
def test_revision_and_window_config_mismatch(tmp_path, change):
    model, cfg, state, _ = fixture()
    save_checkpoint(tmp_path, model, cfg, state)
    def wrong_backend(requested):
        backend = fake_backend_factory(requested)
        if change in ("revision", "tokenizer_revision"):
            config = replace(backend.config, encoder=replace(backend.config.encoder, **{change: "1" * 40}))
            return LoadedBackend(backend.encoder, backend.tokenizer, config)
        if change == "window":
            return LoadedBackend(backend.encoder, backend.tokenizer,
                                 replace(backend.config, window=replace(backend.config.window, overlap_tokens=3)))
        if change == "wrapper":
            backend.tokenizer.special_token_rule = "changed"
        if change == "native_dim":
            backend.encoder.native_dim = 9
        if change == "lora_modules":
            backend.encoder.lora_module_names = ("different",)
        return backend
    with pytest.raises(ArtifactError, match="mismatch"):
        load_checkpoint(tmp_path, device="cpu", backend_factory=wrong_backend)


def test_corrupt_or_unpublished_generation_rejected(tmp_path):
    model, cfg, state, _ = fixture()
    save_checkpoint(tmp_path, model, cfg, state)
    folder = generation(tmp_path)
    (folder / "task.safetensors").write_bytes(b"broken")
    with pytest.raises(ArtifactError, match="digest"):
        load_checkpoint(tmp_path, device="cpu", backend_factory=fake_backend_factory)
    (tmp_path / "CURRENT.json").unlink()
    with pytest.raises(ArtifactError):
        load_checkpoint(tmp_path, device="cpu", backend_factory=fake_backend_factory)


def test_unpinned_revision_and_model_version_rejected(tmp_path):
    model, cfg, state, _ = fixture()
    with pytest.raises(ArtifactError, match="revision"):
        save_checkpoint(tmp_path, model, replace(cfg, encoder=replace(cfg.encoder, revision="main")), state)
    with pytest.raises(ArtifactError, match="version"):
        save_checkpoint(tmp_path, model, cfg, replace(state, model_version="legacy-nine"))


def test_calibration_provenance_and_threshold_validation(tmp_path):
    record = replace(make_record("a", "b", labels=["Modify"], subtypes=[FINE_LABELS[0]]), language="csharp")
    logits = PairLogits(torch.zeros(1, 3), torch.zeros(1, 5))
    thresholds = calibrate_thresholds([record], logits, split="val", language="csharp")
    assert thresholds.provenance == CalibrationProvenance("val", "csharp", (0, 1, 0), (1, 0, 0, 0, 0))
    assert set(thresholds.to_dict()) == {"coarse", "fine"}
    assert Thresholds.from_dict(thresholds.to_dict()).provenance is None
    with pytest.raises(ArtifactError, match="provenance"):
        save_thresholds(tmp_path / "t.json", Thresholds(), checkpoint_digest="a" * 64)
    with pytest.raises(ValueError):
        CalibrationProvenance("test", "csharp", (0, 0, 0), (0,) * 5)
    with pytest.raises(ValueError):
        CalibrationProvenance("val", "csharp", (0, True, -1), (0,) * 5)
    with pytest.raises(ValueError):
        CalibrationProvenance("val", "csharp", (0,), (0,) * 5)


@pytest.mark.parametrize("missing", ["model_version", "config_section"])
def test_incomplete_artifact_schema_cannot_use_defaults(tmp_path, missing):
    model, cfg, state, _ = fixture()
    save_checkpoint(tmp_path, model, cfg, state)
    if missing == "model_version":
        rewrite_artifact(tmp_path, "state.json", lambda value: value["state"].pop("model_version"))
    else:
        rewrite_artifact(tmp_path, "config.json", lambda value: value.pop("data"))
    with pytest.raises(ArtifactError, match="fields|configuration"):
        load_checkpoint(tmp_path, device="cpu", backend_factory=fake_backend_factory)


def test_same_content_has_reproducible_digest(tmp_path):
    model, cfg, state, _ = fixture()
    save_checkpoint(tmp_path / "best", model, cfg, state)
    save_checkpoint(tmp_path / "last", model, cfg, state)
    first = read_json(tmp_path / "best" / "CURRENT.json")
    second = read_json(tmp_path / "last" / "CURRENT.json")
    assert first["generation"] != second["generation"]
    assert first["digest"] == second["digest"]


def test_threshold_support_and_reasons_cannot_disagree(tmp_path):
    model, cfg, state, _ = fixture()
    save_checkpoint(tmp_path / "best", model, cfg, state)
    digest = read_json(tmp_path / "best" / "CURRENT.json")["digest"]
    provenance = CalibrationProvenance("val", "csharp", (0, 1, 0), (0,) * 5)
    with pytest.raises(ArtifactError, match="no-positive"):
        save_thresholds(tmp_path / "t.json", Thresholds((.4, .5, .5), provenance=provenance), checkpoint_digest=digest)
    save_thresholds(tmp_path / "t.json", Thresholds(provenance=provenance), checkpoint_digest=digest)
    payload = read_json(tmp_path / "t.json")
    payload["no_positive_reasons"]["coarse"][1] = "no_positive_examples"
    (tmp_path / "t.json").write_text(json.dumps(payload))
    with pytest.raises(ArtifactError, match="reasons"):
        load_checkpoint(tmp_path / "best", device="cpu", thresholds_path=tmp_path / "t.json", backend_factory=fake_backend_factory)
