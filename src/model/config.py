"""Strict, dependency-free run configuration checked before backend loading.

Data paths remain strings as supplied, making dataclasses.asdict output JSON
serializable. Relative paths are interpreted by the consuming run's directory.
Tokenizer-specific packaging overhead is validated after tokenizer loading.
"""

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

from model.errors import ConfigError


def _integer(name: str, value: int, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: float, *, minimum: float = 0, positive: bool = False, upper: float | None = None) -> None:
    if type(value) not in (int, float):
        raise ConfigError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ConfigError(f"{name} must be a finite number")
    if value < minimum or (positive and value == minimum) or (upper is not None and value >= upper):
        raise ConfigError(f"{name} is outside its valid range")


def _choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{name} must be one of {choices}")


def _text(name: str, value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a nonblank string")


@dataclass(frozen=True, slots=True)
class DataConfig:
    train: str | None = None
    val: str | None = None
    manifest: str | None = None
    audit: str | None = None
    field_map: dict[str, str] = field(default_factory=dict)
    mode: str = "formal"

    def __post_init__(self) -> None:
        for name in ("train", "val", "manifest", "audit"):
            _text(f"data.{name}", getattr(self, name), optional=True)
        _choice("data.mode", self.mode, ("formal", "debug"))
        canonical = {
            "record_id", "pre_code", "post_code", "labels", "modify_subtypes",
            "repo_id", "commit_sha", "source_id", "language",
        }
        if not isinstance(self.field_map, dict):
            raise ConfigError("data.field_map must be a canonical-to-input field mapping")
        for key, value in self.field_map.items():
            if key not in canonical:
                raise ConfigError(f"unknown canonical field in data.field_map: {key!r}")
            _text(f"data.field_map.{key}", value)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    window_tokens: int = 2048
    overlap_tokens: int = 256
    diff_context_lines: int = 3
    window_microbatch: int = 1
    input_mode: str = "full_diff"
    window_checkpoint: bool = True

    def __post_init__(self) -> None:
        _integer("window.window_tokens", self.window_tokens)
        _integer("window.overlap_tokens", self.overlap_tokens, minimum=0)
        _integer("window.diff_context_lines", self.diff_context_lines, minimum=0)
        _integer("window.window_microbatch", self.window_microbatch)
        if not self.overlap_tokens < self.window_tokens <= 32768:
            raise ConfigError("window must satisfy overlap_tokens < window_tokens <= 32768")
        _choice("window.input_mode", self.input_mode, ("full_diff", "full_only", "diff_only"))
        if type(self.window_checkpoint) is not bool:
            raise ConfigError("window.window_checkpoint must be boolean")


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str | None = None
    tokenizer_revision: str | None = None
    device: str = "cuda"
    dtype: str = "bf16"
    adaptation: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantization: str = "none"

    def __post_init__(self) -> None:
        _text("encoder.model_id", self.model_id)
        _text("encoder.revision", self.revision, optional=True)
        _text("encoder.tokenizer_revision", self.tokenizer_revision, optional=True)
        _choice("encoder.device", self.device, ("cuda", "cpu"))
        _choice("encoder.dtype", self.dtype, ("bf16", "fp16", "fp32"))
        _choice("encoder.adaptation", self.adaptation, ("lora", "frozen"))
        _choice("encoder.quantization", self.quantization, ("none", "nf4"))
        _integer("encoder.lora_r", self.lora_r)
        _integer("encoder.lora_alpha", self.lora_alpha)
        _number("encoder.lora_dropout", self.lora_dropout, upper=1)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: str = "hierarchical"
    aggregation: str = "transformer"
    native_dim: int = 1024
    projection_dim: int = 256
    layers: int = 2
    heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True

    def __post_init__(self) -> None:
        _choice("model.architecture", self.architecture, ("hierarchical", "concat"))
        _choice("model.aggregation", self.aggregation, ("transformer", "mean"))
        for name in ("native_dim", "projection_dim", "layers", "heads", "ffn_dim"):
            _integer(f"model.{name}", getattr(self, name))
        if self.projection_dim % self.heads:
            raise ConfigError("model.projection_dim must be divisible by model.heads")
        _number("model.dropout", self.dropout, upper=1)
        _choice("model.activation", self.activation, ("gelu",))
        if type(self.norm_first) is not bool:
            raise ConfigError("model.norm_first must be boolean")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    lambda_fine: float = 1.0
    effective_batch: int = 16
    microbatch: int = 1
    lora_lr: float = 1e-4
    task_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    clip_norm: float = 1.0
    epochs: int = 10
    patience: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("effective_batch", "microbatch", "epochs", "patience"):
            _integer(f"train.{name}", getattr(self, name))
        _integer("train.seed", self.seed, minimum=0)
        if self.microbatch > self.effective_batch:
            raise ConfigError("train.microbatch cannot exceed train.effective_batch")
        for name in ("lora_lr", "task_lr", "clip_norm"):
            _number(f"train.{name}", getattr(self, name), positive=True)
        for name in ("lambda_fine", "weight_decay"):
            _number(f"train.{name}", getattr(self, name))
        _number("train.warmup_ratio", self.warmup_ratio, upper=1)


@dataclass(frozen=True, slots=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        for name, expected in _SECTIONS.items():
            if not isinstance(getattr(self, name), expected):
                raise ConfigError(f"{name} must be a {expected.__name__}")
        if self.model.architecture == "concat" and self.window.input_mode != "full_diff":
            raise ConfigError("concat requires full_diff input_mode")


_SECTIONS = {
    "data": DataConfig, "window": WindowConfig, "encoder": EncoderConfig,
    "model": ModelConfig, "train": TrainConfig,
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate configuration field: {key}")
        result[key] = value
    return result


def load_config(path: Path) -> RunConfig:
    """Load nested JSON, rejecting misspellings, wrong types and bad budgets."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("configuration must be a JSON object")
    unknown = payload.keys() - _SECTIONS.keys()
    if unknown:
        raise ConfigError(f"unknown configuration sections: {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTIONS.items():
        value = payload.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"{name} must be a JSON object")
        unknown = value.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ConfigError(f"unknown {name} fields: {sorted(unknown)}")
        sections[name] = cls(**value)
    return RunConfig(**sections)
