from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from helpers import FakeWindowEncoder, fake_backend_factory, make_record
from model.config import EncoderConfig, ModelConfig, RunConfig, WindowConfig
from model.encode.hf import HFWindowEncoder, load_backend
from model.errors import ConfigError
from model.input.batch import collate_pairs
from model.input.windows import HFTokenizerAdapter, prepare_model_input

MODEL_SHA, TOKENIZER_SHA = "a" * 40, "b" * 40
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class TinyBackbone(nn.Module):
    """Only replaces downloaded weights; PEFT injection remains real."""
    def __init__(self):
        super().__init__()
        from transformers import Qwen3Config
        self.config = Qwen3Config(hidden_size=1024, max_position_embeddings=32768)
        self.layers = nn.ModuleList([nn.ModuleDict({key: nn.Linear(4, 4, bias=False) for key in TARGETS})])
        self.last_call = None

    def forward(self, input_ids, attention_mask, **kwargs):
        self.last_call = dict(kwargs, attention_mask=attention_mask)
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4) / 100
        for projection in self.layers[0].values():
            hidden = hidden + projection(hidden)
        return SimpleNamespace(last_hidden_state=hidden.repeat(1, 1, 256))


@pytest.fixture
def hf_downloads(monkeypatch):
    import huggingface_hub
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    raw = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "[BOS]": 3, "[EOS]": 4}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    raw.post_processor = TemplateProcessing(single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 3), ("[EOS]", 4)])
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]")
    events = []
    config = transformers.Qwen3Config(hidden_size=1024, max_position_embeddings=32768)

    def resolve(self, repo_id, *, revision=None):
        events.append(("resolve", revision))
        return SimpleNamespace(sha=TOKENIZER_SHA if revision == "tokenizer-tag" else MODEL_SHA)

    def load_tokenizer(name, **kwargs):
        events.append(("tokenizer", kwargs))
        return tokenizer

    def load_config(name, **kwargs):
        events.append(("config", kwargs))
        return config

    def load_model(name, **kwargs):
        events.append(("weights", kwargs))
        return TinyBackbone()

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", resolve)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    return events, config


def cpu_config(**kwargs):
    return RunConfig(encoder=EncoderConfig(device="cpu", dtype="fp32", **kwargs))


def test_fake_backend_closes_input_encoder_loop_without_hf(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **k: pytest.fail("fake tried HF"))
    backend = fake_backend_factory(RunConfig(encoder=EncoderConfig(device="cpu", dtype="fp32"), model=ModelConfig(native_dim=8)))
    prepared = prepare_model_input(make_record("旧😀", "新😀", labels=["Modify"]), backend.tokenizer, backend.config)
    inputs, _ = collate_pairs([prepared], backend.tokenizer.pad_token_id)
    assert backend.encoder(inputs.input_ids, inputs.attention_mask).shape == (3, 8)
    assert backend.config.encoder.revision == backend.config.encoder.tokenizer_revision == "0" * 40


@pytest.mark.parametrize("dtype, expected", [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)])
def test_backend_pins_revision_and_loads_tokenizer_config_before_weights(hf_downloads, dtype, expected):
    events, _ = hf_downloads
    cfg = RunConfig(encoder=EncoderConfig(device="cpu", dtype=dtype, revision="model-tag"))
    backend = load_backend(cfg)
    assert isinstance(backend.tokenizer, HFTokenizerAdapter)
    assert backend.config is not cfg and cfg.encoder.revision == "model-tag"
    assert backend.config.encoder.revision == backend.config.encoder.tokenizer_revision == MODEL_SHA
    assert [event[0] for event in events] == ["resolve", "tokenizer", "config", "weights"]
    assert events[1][1]["revision"] == events[2][1]["revision"] == events[3][1]["revision"] == MODEL_SHA
    assert events[1][1]["use_fast"] is True
    assert events[3][1]["torch_dtype"] == expected
    assert backend.encoder.compute_dtype == expected
    assert backend.encoder.native_dim == 1024
    assert {name.rsplit(".", 1)[-1] for name in backend.encoder.lora_module_names} == set(TARGETS)
    assert all(not param.requires_grad for name, param in backend.encoder.named_parameters() if "lora_" not in name)


def test_explicit_tokenizer_revision_is_resolved_separately(hf_downloads):
    backend = load_backend(cpu_config(tokenizer_revision="tokenizer-tag"))
    assert backend.config.encoder.revision == MODEL_SHA
    assert backend.config.encoder.tokenizer_revision == TOKENIZER_SHA


@pytest.mark.parametrize("invalid", ["context", "dimension", "budget"])
def test_invalid_lightweight_configuration_rejected_before_weights(hf_downloads, invalid):
    events, model_config = hf_downloads
    cfg = cpu_config()
    if invalid == "context":
        model_config.max_position_embeddings = 1024
    elif invalid == "dimension":
        model_config.hidden_size = 768
    else:
        cfg = replace(cfg, window=WindowConfig(window_tokens=8, overlap_tokens=0))
    with pytest.raises(ConfigError):
        load_backend(cfg)
    assert "weights" not in [event[0] for event in events]


def test_forward_builds_padding_independent_positions_and_disables_cache():
    backbone = TinyBackbone()
    encoder = HFWindowEncoder(backbone, native_dim=1024, frozen=True)
    ids = torch.tensor([[0, 0, 3, 4], [3, 4, 0, 0]])
    mask = ids.ne(0)
    encoder.train()
    output = encoder(ids, mask)
    assert output.shape == (2, 1024)
    torch.testing.assert_close(output[0], output[1])
    assert backbone.last_call["position_ids"].tolist() == [[0, 0, 0, 1], [0, 1, 0, 0]]
    assert backbone.last_call["use_cache"] is False
    assert backbone.last_call["output_hidden_states"] is False
    assert not backbone.training and not output.requires_grad
    assert encoder.adapter_state_dict() == {}
    with pytest.raises(ValueError):
        encoder.load_adapter_state_dict({"unexpected": torch.zeros(1)})


def test_peft_adapter_roundtrip_excludes_backbone_and_restores_predictions(hf_downloads):
    backend = load_backend(cpu_config(lora_dropout=0))
    encoder = backend.encoder.eval()
    ids, mask = torch.tensor([[2, 3, 4]]), torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        for name, param in encoder.named_parameters():
            if "lora_B" in name:
                param.fill_(0.02)
    expected = encoder(ids, mask).detach()
    state = encoder.adapter_state_dict()
    assert state and all("lora_" in name for name in state)
    with torch.no_grad():
        for name, param in encoder.named_parameters():
            if "lora_" in name:
                param.zero_()
    assert not torch.allclose(encoder(ids, mask), expected)
    encoder.load_adapter_state_dict(state)
    torch.testing.assert_close(encoder(ids, mask), expected)
    with pytest.raises(ValueError):
        encoder.load_adapter_state_dict({})


def test_frozen_backend_has_no_adapter_or_trainable_backbone(hf_downloads):
    backend = load_backend(cpu_config(adaptation="frozen"))
    backend.encoder.train()
    assert not backend.encoder.backbone.training
    assert not any(param.requires_grad for param in backend.encoder.parameters())
    assert backend.encoder.adapter_state_dict() == {}
    assert backend.encoder.lora_module_names == ()


def test_nf4_requires_cuda_and_explicit_low_precision(hf_downloads):
    with pytest.raises(ConfigError):
        load_backend(cpu_config(quantization="nf4"))
    assert "weights" not in [event[0] for event in hf_downloads[0]]


@pytest.mark.parametrize("dtype, expected", [("bf16", torch.bfloat16), ("fp16", torch.float16)])
def test_nf4_double_quant_compute_dtype_and_prepare_before_lora(hf_downloads, monkeypatch, dtype, expected):
    import peft
    import transformers
    import model.encode.hf as hf
    events, _ = hf_downloads
    original_get = peft.get_peft_model
    monkeypatch.setattr(hf.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kwargs: SimpleNamespace(**kwargs))

    def prepare(model, **kwargs):
        events.append(("prepare", kwargs))
        return model

    def inject(model, config):
        events.append(("inject", config))
        return original_get(model, config)

    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", prepare)
    monkeypatch.setattr(peft, "get_peft_model", inject)
    backend = load_backend(RunConfig(encoder=EncoderConfig(device="cuda", dtype=dtype, quantization="nf4")))
    assert backend.encoder.compute_dtype == expected
    weights = next(value for key, value in events if key == "weights")
    quant = weights["quantization_config"]
    assert quant.load_in_4bit and quant.bnb_4bit_use_double_quant
    assert quant.bnb_4bit_quant_type == "nf4" and quant.bnb_4bit_compute_dtype == expected
    assert weights["device_map"] == {"": "cuda"}
    assert [key for key, _ in events][-2:] == ["prepare", "inject"]
    assert next(value for key, value in events if key == "prepare")["use_gradient_checkpointing"] is False


def test_weight_initialization_failure_propagates_without_fallback(hf_downloads, monkeypatch):
    import transformers
    calls = []

    def failed_load(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("weight initialization failed")

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", failed_load)
    with pytest.raises(RuntimeError, match="weight initialization failed"):
        load_backend(cpu_config())
    assert len(calls) == 1


def test_no_lora_projection_matches_is_rejected(hf_downloads, monkeypatch):
    import transformers
    backbone = TinyBackbone()
    backbone.layers = nn.ModuleList()
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: backbone)
    with pytest.raises(ConfigError, match="matched"):
        load_backend(cpu_config())


def test_real_tiny_qwen_peft_adapter_receives_gradients():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3Config, Qwen3Model
    from model.encode.checkpoint import encode_windows

    backbone = Qwen3Model(Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                    head_dim=4, max_position_embeddings=32))
    backbone = get_peft_model(backbone, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=2,
                                                 lora_alpha=4, lora_dropout=0.2, target_modules=list(TARGETS)))
    encoder = HFWindowEncoder(backbone, native_dim=8, frozen=False)
    ids = torch.tensor([[1, 2, 3, 0], [0, 1, 4, 5]])
    encode_windows(encoder, ids, ids.ne(0)).square().sum().backward()
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for name, param in encoder.named_parameters() if "lora_" in name)
    assert all(param.grad is None for name, param in encoder.named_parameters() if "lora_" not in name)
