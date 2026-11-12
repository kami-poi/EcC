"""Baseline contracts through shared preparation, model and persistence."""
from dataclasses import replace
import json

import pytest
import torch

from helpers import FakeTokenizer, fake_backend_factory, make_record
from model.config import EncoderConfig, ModelConfig, RunConfig, WindowConfig
from model.errors import ConfigError
from model.evaluate.decode import Thresholds
from model.evaluate.metrics import evaluate
from model.input.batch import collate_pairs
from model.input.windows import prepare_concat_pair, prepare_model_input, prepare_pair
from model.models.concat import ConcatClassifier
from model.models.pair import build_model


@pytest.mark.parametrize("adaptation,mode,aggregation", [
    ("lora", "full_diff", "transformer"), ("frozen", "full_diff", "transformer"),
    ("lora", "full_only", "transformer"), ("lora", "diff_only", "transformer"),
    ("lora", "full_diff", "mean")])
def test_frozen_and_input_ablation_keep_evaluation_contract(adaptation, mode, aggregation):
    cfg = RunConfig(encoder=EncoderConfig(device="cpu", dtype="fp32", adaptation=adaptation),
                    model=ModelConfig(native_dim=8, projection_dim=8, heads=2, aggregation=aggregation),
                    window=WindowConfig(input_mode=mode))
    backend = fake_backend_factory(cfg)
    model = build_model(backend.encoder, cfg).eval()
    record = make_record("old", "new", labels=["Modify"])
    pair = prepare_model_input(record, backend.tokenizer, cfg)
    inputs, _ = collate_pairs([pair], 0)
    logits = model(inputs)
    assert logits.coarse.shape == (1, 3) and logits.fine.shape == (1, 5)
    assert pair.token_counts == prepare_pair(record, backend.tokenizer, WindowConfig()).token_counts
    assert evaluate([record], logits, Thresholds(), language="cpp").counts["n_samples"] == 1
    if adaptation == "frozen":
        assert not any(p.requires_grad for p in model.encoder.parameters())
    if aggregation == "mean":
        assert not any("transformer" in name or "queries" in name for name, _ in model.named_parameters())


def test_concat_reports_real_truncation_coverage():
    tokenizer = FakeTokenizer()
    record = make_record("a" * 2500, "b" * 2500, labels=["Modify"])
    pair = prepare_concat_pair(record, tokenizer, WindowConfig())
    full = prepare_pair(record, tokenizer, WindowConfig())
    assert pair.token_counts == full.token_counts and sum(pair.token_counts[:2]) == 5000
    assert len(pair.windows) == 1 and pair.windows[0].role == "PAIR"
    assert len(pair.windows[0].input_ids) == 2048
    assert 0 < pair.coverage[0] < 1 and pair.coverage[1:] == (0., 0.)
    assert full.coverage == (1., 1., 1.)
    ids = pair.windows[0].input_ids
    suffix, _ = tokenizer.tokenize("\nSUMMARY")
    assert ids[-len(suffix)-1:-1] == suffix
    assert pair.change_positions == full.change_positions


def test_concat_complete_order_and_missing_side():
    tokenizer = FakeTokenizer()
    pair = prepare_concat_pair(make_record(None, "new", labels=["Add"]), tokenizer, WindowConfig())
    assert pair.windows[0].text.startswith("PRE\nEMPTY\nPOST\nnew\nDIFF\n")
    assert pair.token_counts[0] == 0 and pair.coverage == (1., 1., 1.)
    with pytest.raises(ConfigError, match="full_diff"):
        RunConfig(model=ModelConfig(architecture="concat"), window=WindowConfig(input_mode="diff_only"))


def test_concat_shared_dispatch_gradient_and_checkpoint(tmp_path):
    from model.train.checkpoint import CheckpointState, save_checkpoint, load_checkpoint
    cfg = RunConfig(encoder=EncoderConfig(device="cpu", dtype="fp32", lora_dropout=0),
                    model=ModelConfig(architecture="concat", native_dim=8, projection_dim=8, heads=2))
    backend = fake_backend_factory(cfg)
    model = build_model(backend.encoder, cfg).eval()
    assert isinstance(model, ConcatClassifier)
    assert not any("aggregation" in name or "fusion" in name for name, _ in model.named_parameters())
    pair = prepare_model_input(make_record("old", "new", labels=["Modify"]), backend.tokenizer, cfg)
    inputs, _ = collate_pairs([pair, pair], 0)
    logits = model(inputs)
    (logits.coarse.sum() + logits.fine.sum()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    state = CheckpointState(0, 0, 0., {"train": "a" * 64}, backend.tokenizer.special_token_rule)
    save_checkpoint(tmp_path, model, backend.config, state)
    loaded = load_checkpoint(tmp_path, device="cpu", backend_factory=fake_backend_factory)
    assert isinstance(loaded.model, ConcatClassifier)
    torch.testing.assert_close(loaded.model(inputs).fine, logits.fine)


def test_concat_cli_train_eval_and_predict(tmp_path):
    from model import main
    from test_workflow import config_file, read
    path = config_file(tmp_path)
    cfg = read(path)
    cfg["model"]["architecture"] = "concat"
    path.write_text(json.dumps(cfg))
    run = tmp_path / "run"
    assert main(["train", "--config", str(path), "--output", str(run)], backend_factory=fake_backend_factory) == 0
    report = tmp_path / "report"
    assert main(["eval", "--checkpoint", str(run / "model/best"), "--data", str(tmp_path / "test.jsonl"),
                 "--manifest", str(tmp_path / "manifest.json"), "--language", "java", "--output", str(report)],
                backend_factory=fake_backend_factory) == 0
    metrics = read(report / "metrics.json")
    assert metrics["provenance"]["parameter_counts"]["trainable"] > 0
    assert metrics["provenance"]["bootstrap"]["unit"] == "repo_group"
    assert metrics["diagnostics"]["coverage"]["PAIR_window_count"] == 1
    assert main(["predict", "--checkpoint", str(run / "model/best"), "--input", str(tmp_path / "test.jsonl"),
                 "--output", str(tmp_path / "pred.jsonl")], backend_factory=fake_backend_factory) == 0
