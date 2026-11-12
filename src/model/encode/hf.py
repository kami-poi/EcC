"""Pinned Qwen hidden-state backend with optional supervised LoRA adaptation."""

import importlib.util
import re
from collections.abc import Mapping
from dataclasses import replace

import torch
from torch import Tensor, nn

from model.config import RunConfig
from model.errors import ConfigError
from model.input.windows import HFTokenizerAdapter, validate_tokenizer_budget

from .base import LoadedBackend, WindowEncoder
from .pooling import last_valid_pool

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class HFWindowEncoder(WindowEncoder):
    def __init__(self, backbone: nn.Module, *, native_dim: int, frozen: bool,
                 lora_module_names: tuple[str, ...] = (), compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.native_dim = native_dim
        self.backbone = backbone
        self.frozen = frozen
        self.lora_module_names = lora_module_names
        # Quantized training may promote stored floating parameters to FP32.
        # Keep configured computation precision available to the training step.
        self.compute_dtype = compute_dtype
        self.backbone.config.use_cache = False
        if frozen:
            self.backbone.requires_grad_(False)
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(~attention_mask.bool(), 0)
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=positions, use_cache=False,
                               output_hidden_states=False, return_dict=True).last_hidden_state
        return last_valid_pool(hidden, attention_mask)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        if self.frozen:
            return {}
        from peft import get_peft_model_state_dict

        return {key: value.detach().clone() for key, value in get_peft_model_state_dict(self.backbone).items()}

    def load_adapter_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if self.frozen:
            if state:
                raise ValueError("frozen encoder has no adapter")
            return
        from peft import set_peft_model_state_dict

        expected = self.adapter_state_dict()
        if state.keys() != expected.keys() or any(
                not isinstance(state[key], Tensor) or state[key].shape != value.shape
                for key, value in expected.items()):
            raise ValueError("incompatible adapter keys or shapes")
        result = set_peft_model_state_dict(self.backbone, dict(state))
        if result.unexpected_keys:
            raise ValueError(f"unexpected adapter keys: {result.unexpected_keys}")


def _resolved_sha(api, model_id: str, revision: str | None) -> str:
    sha = api.model_info(model_id, revision=revision).sha
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None:
        raise ConfigError("Hugging Face model revision did not resolve to a commit SHA")
    return sha


def _quantization(cfg: RunConfig, dtype):
    from transformers import BitsAndBytesConfig

    if cfg.encoder.device != "cuda" or dtype not in (torch.bfloat16, torch.float16):
        raise ConfigError("NF4 requires an explicit CUDA bf16/fp16 encoder")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ConfigError("NF4 requires the optional quant extra (bitsandbytes)")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 6:
        raise ConfigError("NF4 requires a compatible CUDA GPU")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ConfigError("configured bf16 NF4 compute requires CUDA bf16 support")
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)


def load_backend(cfg: RunConfig) -> LoadedBackend:
    """Validate lightweight artifacts before loading weights; never fallback."""
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    dtype = DTYPES[cfg.encoder.dtype]
    quantization = _quantization(cfg, dtype) if cfg.encoder.quantization == "nf4" else None
    api = HfApi()
    model_sha = _resolved_sha(api, cfg.encoder.model_id, cfg.encoder.revision)
    tokenizer_sha = (model_sha if cfg.encoder.tokenizer_revision is None else
                     _resolved_sha(api, cfg.encoder.model_id, cfg.encoder.tokenizer_revision))
    resolved = replace(cfg, encoder=replace(cfg.encoder, revision=model_sha, tokenizer_revision=tokenizer_sha))
    tokenizer = HFTokenizerAdapter(AutoTokenizer.from_pretrained(
        cfg.encoder.model_id, revision=tokenizer_sha, use_fast=True))
    model_config = AutoConfig.from_pretrained(cfg.encoder.model_id, revision=model_sha)
    validate_tokenizer_budget(tokenizer, cfg.window)
    if getattr(model_config, "hidden_size", None) != 1024 or cfg.model.native_dim != 1024:
        raise ConfigError("Qwen3-Embedding-0.6B must provide the complete native 1024 dimensions")
    context = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(context, int) or cfg.window.window_tokens > context:
        raise ConfigError("complete window budget exceeds model context")
    model_config.use_cache = False
    kwargs = dict(revision=model_sha, config=model_config, torch_dtype=dtype)
    if quantization is not None:
        kwargs.update(quantization_config=quantization, device_map={"": cfg.encoder.device})
    backbone = AutoModel.from_pretrained(cfg.encoder.model_id, **kwargs)
    if quantization is None:
        backbone.to(cfg.encoder.device)
    matched = ()
    if cfg.encoder.adaptation == "lora":
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if quantization is not None:
            # Window recomputation is managed outside the backbone so pooled
            # summaries, rather than every hidden state, cross this boundary.
            backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
        matched = tuple(name for name, module in backbone.named_modules()
                        if name.rsplit(".", 1)[-1] in LORA_TARGETS)
        if not matched:
            raise ConfigError("no requested Qwen projection modules matched LoRA targets")
        backbone = get_peft_model(backbone, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=cfg.encoder.lora_r,
            lora_alpha=cfg.encoder.lora_alpha, lora_dropout=cfg.encoder.lora_dropout,
            target_modules=list(LORA_TARGETS), bias="none"))
    encoder = HFWindowEncoder(backbone, native_dim=model_config.hidden_size,
                              frozen=cfg.encoder.adaptation == "frozen", lora_module_names=matched,
                              compute_dtype=dtype)
    return LoadedBackend(encoder, tokenizer, resolved)
