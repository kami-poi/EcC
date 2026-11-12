import json
from dataclasses import asdict

import pytest

from model.config import DataConfig, EncoderConfig, ModelConfig, RunConfig, TrainConfig, WindowConfig, load_config
from model.errors import ConfigError


def write_config(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_empty_config_provides_reproducible_defaults(tmp_path):
    cfg = load_config(write_config(tmp_path, {}))
    assert isinstance(cfg, RunConfig)
    assert cfg.data.mode == "formal"
    assert (cfg.window.window_tokens, cfg.window.overlap_tokens, cfg.window.diff_context_lines) == (2048, 256, 3)
    assert (cfg.window.window_microbatch, cfg.window.input_mode) == (1, "full_diff")
    assert (cfg.encoder.model_id, cfg.encoder.device, cfg.encoder.dtype) == ("Qwen/Qwen3-Embedding-0.6B", "cuda", "bf16")
    assert (cfg.encoder.adaptation, cfg.encoder.quantization) == ("lora", "none")
    assert (cfg.encoder.lora_r, cfg.encoder.lora_alpha, cfg.encoder.lora_dropout) == (16, 32, 0.05)
    assert (cfg.model.architecture, cfg.model.aggregation) == ("hierarchical", "transformer")
    assert (cfg.model.native_dim, cfg.model.projection_dim, cfg.model.layers, cfg.model.heads, cfg.model.ffn_dim) == (1024, 256, 2, 4, 1024)
    assert (cfg.model.dropout, cfg.model.activation, cfg.model.norm_first) == (0.1, "gelu", True)
    assert (cfg.train.lambda_fine, cfg.train.effective_batch, cfg.train.microbatch) == (1.0, 16, 1)
    assert (cfg.train.lora_lr, cfg.train.task_lr, cfg.train.weight_decay) == (1e-4, 3e-4, 0.01)
    assert (cfg.train.warmup_ratio, cfg.train.clip_norm, cfg.train.epochs, cfg.train.patience, cfg.train.seed) == (0.05, 1.0, 10, 3, 0)


def test_explicit_cpu_stub_and_ablation_config_roundtrips(tmp_path):
    payload = {
        "data": {"train": "train.jsonl", "val": "val.jsonl", "manifest": "manifest.json", "audit": "audit.json", "field_map": {"pre_code": "before"}, "mode": "debug"},
        "window": {"window_tokens": 128, "overlap_tokens": 16, "input_mode": "full_only"},
        "encoder": {"device": "cpu", "dtype": "fp32", "adaptation": "frozen", "revision": "abc", "tokenizer_revision": "def"},
        "model": {"native_dim": 12, "projection_dim": 8, "heads": 2, "aggregation": "mean"},
        "train": {"seed": 4},
    }
    cfg = load_config(write_config(tmp_path, payload))
    assert cfg.data.field_map == {"pre_code": "before"}
    assert cfg.encoder.device == "cpu" and cfg.encoder.dtype == "fp32"
    assert cfg.model.native_dim == 12 and cfg.train.seed == 4
    assert load_config(write_config(tmp_path, asdict(cfg))) == cfg


@pytest.mark.parametrize("payload", [
    {"widow": {}}, {"window": {"overlap_token": 16}}, {"encoder": {"dtpye": "bf16"}},
    {"data": {"trani": "x"}}, {"model": {"head": 4}}, {"train": {"epoch": 10}},
    [], {"window": None}, {"model": []}, {"train": {"epochs": True}},
    {"encoder": {"revision": 12}}, {"data": {"field_map": {"unknown": "x"}}},
    {"data": {"field_map": {"pre_code": ""}}}, {"data": {"train": 2}},
])
def test_config_rejects_unknown_fields_and_wrong_types(tmp_path, payload):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, payload))


@pytest.mark.parametrize("factory,kwargs", [
    (WindowConfig, {"window_tokens": 0}), (WindowConfig, {"window_tokens": 32769}),
    (WindowConfig, {"overlap_tokens": -1}), (WindowConfig, {"window_tokens": 256}),
    (WindowConfig, {"diff_context_lines": -1}), (WindowConfig, {"window_microbatch": 0}),
    (WindowConfig, {"input_mode": "truncate"}), (WindowConfig, {"window_tokens": 512.0}),
    (WindowConfig, {"window_checkpoint": 1}),
    (EncoderConfig, {"dtype": "float16"}), (EncoderConfig, {"device": "auto"}),
    (EncoderConfig, {"lora_r": 0}), (EncoderConfig, {"lora_alpha": -1}),
    (EncoderConfig, {"lora_dropout": 1.0}), (EncoderConfig, {"model_id": ""}),
    (EncoderConfig, {"adaptation": "full"}), (EncoderConfig, {"quantization": "int8"}),
    (ModelConfig, {"native_dim": 0}), (ModelConfig, {"projection_dim": 255}),
    (ModelConfig, {"heads": 0}), (ModelConfig, {"layers": 0}),
    (ModelConfig, {"ffn_dim": -1}), (ModelConfig, {"dropout": float("nan")}),
    (ModelConfig, {"architecture": "unknown"}), (ModelConfig, {"aggregation": "max"}),
    (ModelConfig, {"norm_first": "true"}),
    (TrainConfig, {"effective_batch": 0}), (TrainConfig, {"microbatch": 17}),
    (TrainConfig, {"epochs": 0}), (TrainConfig, {"patience": -1}),
    (TrainConfig, {"lora_lr": 0}), (TrainConfig, {"task_lr": float("inf")}),
    (TrainConfig, {"weight_decay": -1}), (TrainConfig, {"warmup_ratio": 1}),
    (TrainConfig, {"lambda_fine": -1}), (TrainConfig, {"clip_norm": 0}),
    (TrainConfig, {"seed": -1}), (DataConfig, {"mode": "production"}),
    (TrainConfig, {"task_lr": 10**400}),
])
def test_numeric_and_enum_validation_precedes_loading(factory, kwargs):
    with pytest.raises(ConfigError):
        factory(**kwargs)


def test_maximum_context_and_zero_overlap_are_valid():
    cfg = WindowConfig(window_tokens=32768, overlap_tokens=0)
    assert cfg.window_tokens == 32768 and cfg.overlap_tokens == 0


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_supported_precision_is_explicit(dtype):
    assert EncoderConfig(dtype=dtype).dtype == dtype


def test_invalid_json_and_missing_config_are_config_errors(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.json")


def test_duplicate_json_fields_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"train": {"seed": 0, "seed": 4}}', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_run_config_requires_validated_section_types():
    with pytest.raises(ConfigError):
        RunConfig(window={"window_tokens": 0})
