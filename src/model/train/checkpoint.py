"""Versioned inference artifacts, published through an atomic generation pointer.

Only CURRENT.json selects a completed generation. Interrupted staging directories
and completed but unpublished generations are never discovered or loaded.
These artifacts intentionally contain no optimizer or training-resume state.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import re
import uuid

from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
import torch

from model.config import RunConfig, load_config
from model.encode.base import BackendFactory
from model.encode.hf import load_backend
from model.errors import ArtifactError
from model.evaluate.decode import CalibrationProvenance, Thresholds
from model.input.windows import EMPTY, PREFIX_TEMPLATE, SUFFIX, TokenizerAdapter
from model.labels import COARSE_LABELS, FINE_LABELS
from model.models.base import PairModel
from model.models.pair import build_model

FORMAT_VERSION = 1
MODEL_VERSION = "eh-change-3-5-v1"


def _sha(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


@dataclass(frozen=True, slots=True)
class CheckpointState:
    epoch: int
    step: int
    selection_score: float
    data_hashes: dict[str, str]
    tokenizer_special_token_rule: str
    model_version: str = MODEL_VERSION

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.epoch, self.step)):
            raise ArtifactError("epoch and step must be nonnegative integers")
        if type(self.selection_score) not in (int, float) or not math.isfinite(self.selection_score):
            raise ArtifactError("selection_score must be finite")
        if (not isinstance(self.data_hashes, dict) or not self.data_hashes
                or any(not isinstance(k, str) or not k or not _sha(v) for k, v in self.data_hashes.items())):
            raise ArtifactError("data_hashes must name SHA256 data checksums")
        if not isinstance(self.tokenizer_special_token_rule, str) or not self.tokenizer_special_token_rule.strip():
            raise ArtifactError("tokenizer_special_token_rule must be explicit")
        if self.model_version != MODEL_VERSION:
            raise ArtifactError("incompatible model version")


@dataclass(frozen=True, slots=True)
class InferenceBundle:
    model: PairModel
    tokenizer: TokenizerAdapter
    config: RunConfig
    thresholds: Thresholds
    checkpoint_digest: str
    calibrated: bool


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _write_json(path, value):
    with path.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError(f"duplicate artifact field: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ArtifactError(f"invalid {name} fields")


def _sync_dir(path):
    # POSIX directory fsync makes rename publication durable across crashes.
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _digest_files(files):
    return hashlib.sha256(_json_bytes(files)).hexdigest()


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _labels():
    return {"coarse": list(COARSE_LABELS), "fine": list(FINE_LABELS)}


def _wrapper():
    return {"prefix_template": PREFIX_TEMPLATE, "suffix": SUFFIX, "empty": EMPTY}


def _module_names(encoder):
    names = getattr(encoder, "lora_module_names", None)
    if names is None:
        names = sorted({name.rsplit(".", 1)[0] for name, param in encoder.named_parameters()
                        if param.requires_grad})
    return list(names)


def _environment():
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "cuda": torch.version.cuda}
    for name in ("transformers", "peft", "safetensors", "numpy"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _pinned(cfg):
    if not _sha(cfg.encoder.revision, 40) or not _sha(cfg.encoder.tokenizer_revision, 40):
        raise ArtifactError("model and tokenizer revision must be resolved commit SHAs")


def save_checkpoint(path: Path, model: PairModel, cfg: RunConfig, state: CheckpointState) -> None:
    """Save resolved config and adapter/task weights; publish only after fsync.

    The caller passes backend.config and tokenizer.special_token_rule in state.
    Data hashes are checksums from the audited run; no dataset is reread here.
    """
    _pinned(cfg)
    CheckpointState(**asdict(state))
    if (model.encoder.native_dim != cfg.model.native_dim
            or model.window_microbatch != cfg.window.window_microbatch
            or model.window_checkpoint != cfg.window.window_checkpoint):
        raise ArtifactError("model/config mismatch")
    adapter = model.encoder.adapter_state_dict()
    if bool(adapter) != (cfg.encoder.adaptation == "lora"):
        raise ArtifactError("adapter/config mismatch")
    task = {key: value for key, value in model.state_dict().items() if not key.startswith("encoder.")}
    path = Path(path)
    generations = path / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    staging = generations / (".tmp-" + identifier)
    staging.mkdir()
    files = ["task.safetensors", "config.json", "labels.json", "state.json"]
    tensors = {"task.safetensors": task}
    if adapter:
        files.append("adapter.safetensors")
        tensors["adapter.safetensors"] = adapter
    for name, values in tensors.items():
        save_file({key: value.detach().cpu().contiguous().clone() for key, value in values.items()}, staging / name)
        with (staging / name).open("rb") as stream:
            os.fsync(stream.fileno())
    _write_json(staging / "config.json", asdict(cfg))
    _write_json(staging / "labels.json", _labels())
    _write_json(staging / "state.json", {
        "format_version": FORMAT_VERSION, "state": asdict(state), "wrapper": _wrapper(),
        "native_dim": model.encoder.native_dim, "lora_module_names": _module_names(model.encoder),
        "environment": _environment(),
    })
    checksums = {name: _file_digest(staging / name) for name in files}
    digest = _digest_files(checksums)
    _write_json(staging / "COMPLETE.json", {"format_version": FORMAT_VERSION, "files": checksums, "digest": digest})
    _sync_dir(staging)
    os.replace(staging, generations / identifier)
    _sync_dir(generations)
    pointer = path / (".CURRENT-" + identifier + ".tmp")
    _write_json(pointer, {"format_version": FORMAT_VERSION, "generation": identifier, "digest": digest})
    os.replace(pointer, path / "CURRENT.json")
    _sync_dir(path)


def _read_generation(path):
    pointer = _read_json(path / "CURRENT.json")
    _keys(pointer, ("format_version", "generation", "digest"), "checkpoint pointer")
    if (type(pointer["format_version"]) is not int or pointer["format_version"] != FORMAT_VERSION
            or not _sha(pointer["generation"], 32) or not _sha(pointer["digest"])):
        raise ArtifactError("incompatible checkpoint pointer version/generation/digest")
    folder = path / "generations" / pointer["generation"]
    complete = _read_json(folder / "COMPLETE.json")
    _keys(complete, ("format_version", "files", "digest"), "completion marker")
    if type(complete["format_version"]) is not int or complete["format_version"] != FORMAT_VERSION:
        raise ArtifactError("incompatible completion version")
    checksums = complete["files"]
    required = {"task.safetensors", "config.json", "labels.json", "state.json"}
    if not isinstance(checksums, dict) or set(checksums) not in (required, required | {"adapter.safetensors"}):
        raise ArtifactError("invalid generation file manifest")
    if (complete["digest"] != pointer["digest"] or _digest_files(checksums) != pointer["digest"]
            or any(not _sha(value) or _file_digest(folder / name) != value for name, value in checksums.items())):
        raise ArtifactError("checkpoint digest mismatch")
    return folder, pointer["digest"], checksums


def _no_positive_reasons(provenance):
    return {name: ["no_positive_examples" if count == 0 else None for count in getattr(provenance, name + "_support")]
            for name in ("coarse", "fine")}


def _validate_no_positive(thresholds):
    for name in ("coarse", "fine"):
        if any(count == 0 and value != .5 for count, value in
               zip(getattr(thresholds.provenance, name + "_support"), getattr(thresholds, name))):
            raise ArtifactError("no-positive classes must retain threshold 0.5")


def save_thresholds(path: Path, thresholds: Thresholds, *, checkpoint_digest: str) -> None:
    if thresholds.provenance is None:
        raise ArtifactError("threshold calibration provenance is required")
    if not _sha(checkpoint_digest):
        raise ArtifactError("checkpoint_digest must be SHA256")
    _validate_no_positive(thresholds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    _write_json(staging, {"format_version": FORMAT_VERSION, "model_version": MODEL_VERSION,
                         "checkpoint_digest": checkpoint_digest, "labels": _labels(),
                         "thresholds": thresholds.to_dict(), "provenance": asdict(thresholds.provenance),
                         "no_positive_reasons": _no_positive_reasons(thresholds.provenance)})
    os.replace(staging, path)
    _sync_dir(path.parent)


def _load_thresholds(path, digest):
    payload = _read_json(path)
    _keys(payload, ("format_version", "model_version", "checkpoint_digest", "labels", "thresholds",
                    "provenance", "no_positive_reasons"), "threshold artifact")
    if type(payload["format_version"]) is not int or payload["format_version"] != FORMAT_VERSION or payload["model_version"] != MODEL_VERSION:
        raise ArtifactError("incompatible thresholds version")
    if payload["checkpoint_digest"] != digest:
        raise ArtifactError("threshold checkpoint digest mismatch")
    if payload["labels"] != _labels():
        raise ArtifactError("threshold label schema mismatch")
    provenance = CalibrationProvenance(**payload["provenance"])
    thresholds = replace(Thresholds.from_dict(payload["thresholds"]), provenance=provenance)
    if payload["no_positive_reasons"] != _no_positive_reasons(provenance):
        raise ArtifactError("threshold no-positive reasons mismatch")
    _validate_no_positive(thresholds)
    return thresholds


def load_checkpoint(path: Path, *, device: str, thresholds_path: Path | None = None,
                    backend_factory: BackendFactory | None = None) -> InferenceBundle:
    """Rebuild the pinned base, validate contracts, then restore strict weights.

    Without thresholds this is preparation for validation/calibration only.
    Formal eval/predict callers MUST require bundle.calibrated to be true.
    Only device may differ from the persisted resolved backend configuration.
    """
    try:
        folder, digest, files = _read_generation(Path(path))
        if _read_json(folder / "labels.json") != _labels():
            raise ArtifactError("incompatible label names/order (expected three/five); legacy eight-label checkpoints require retraining")
        metadata = _read_json(folder / "state.json")
        _keys(metadata, ("format_version", "state", "wrapper", "native_dim", "lora_module_names", "environment"), "checkpoint metadata")
        if type(metadata["format_version"]) is not int or metadata["format_version"] != FORMAT_VERSION:
            raise ArtifactError("incompatible checkpoint version")
        _keys(metadata["state"], CheckpointState.__dataclass_fields__, "checkpoint state")
        state = CheckpointState(**metadata["state"])
        if metadata["wrapper"] != _wrapper():
            raise ArtifactError("input wrapper mismatch")
        cfg = load_config(folder / "config.json")
        if _read_json(folder / "config.json") != asdict(cfg):
            raise ArtifactError("checkpoint configuration must include all resolved fields")
        _pinned(cfg)
        if metadata["native_dim"] != cfg.model.native_dim:
            raise ArtifactError("native_dim configuration mismatch")
        if ("adapter.safetensors" in files) != (cfg.encoder.adaptation == "lora"):
            raise ArtifactError("adapter manifest mismatch")
        thresholds = Thresholds() if thresholds_path is None else _load_thresholds(Path(thresholds_path), digest)
        requested = replace(cfg, encoder=replace(cfg.encoder, device=device))
    except ArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise ArtifactError(f"cannot load checkpoint {path}: {exc}") from exc
    # Construction failures are runtime failures, independent of artifact data.
    backend = (backend_factory or load_backend)(requested)
    try:
        if backend.config != requested:
            raise ArtifactError("backend revision/configuration mismatch")
        if (backend.encoder.native_dim != metadata["native_dim"]
                or _module_names(backend.encoder) != metadata["lora_module_names"]
                or getattr(backend.tokenizer, "special_token_rule", None) != state.tokenizer_special_token_rule):
            raise ArtifactError("backend native_dim/LoRA/tokenizer wrapper mismatch")
        model = build_model(backend.encoder, requested)
        adapter = load_file(folder / "adapter.safetensors") if "adapter.safetensors" in files else {}
        model.encoder.load_adapter_state_dict(adapter)
        task = load_file(folder / "task.safetensors")
        expected = {key: value for key, value in model.state_dict().items() if not key.startswith("encoder.")}
        if task.keys() != expected.keys() or any(task[key].shape != value.shape or task[key].dtype != value.dtype for key, value in expected.items()):
            raise ArtifactError("task tensor names/shapes/dtypes mismatch")
        # The only omitted keys are the base/adapter state owned by the encoder.
        result = model.load_state_dict(task, strict=False)
        if result.unexpected_keys or any(not key.startswith("encoder.") for key in result.missing_keys):
            raise ArtifactError("task state mismatch")
        model.eval()
        return InferenceBundle(model, backend.tokenizer, backend.config, thresholds, digest, thresholds_path is not None)
    except ArtifactError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError) as exc:
        raise ArtifactError(f"cannot load checkpoint {path}: {exc}") from exc
