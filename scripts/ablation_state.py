"""Atomic batch bookkeeping, without importing the neural model or its deps."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


class BatchError(Exception):
    pass


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BatchError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(job):
    path = job / "status.json"
    if not path.exists():
        return {"schema_version": 1, "history": []}
    state = read(path)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise BatchError(f"invalid batch status: {path}; preserve it and use a different output root")
    return state


def save_state(job, state):
    state["updated_at"] = timestamp()
    atomic_write(job / "status.json", state)


def checkpoint_digest(run):
    """Check published generation checksums without loading model weights."""
    pointer = read(run / "model/best/CURRENT.json")
    generation = pointer["generation"]
    if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
        raise BatchError("invalid best generation")
    folder = run / "model/best/generations" / generation
    complete = read(folder / "COMPLETE.json")
    files = complete["files"]
    if not isinstance(files, dict) or not {"config.json", "labels.json", "state.json", "task.safetensors"} <= files.keys():
        raise BatchError("incomplete checkpoint file declaration")
    if digest(files) != pointer["digest"] or complete["digest"] != pointer["digest"]:
        raise BatchError("inconsistent checkpoint digest")
    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise BatchError("invalid checkpoint file name")
        if file_hash(folder / name) != expected:
            raise BatchError(f"checkpoint checksum mismatch: {name}")
    return pointer["digest"]


def training_result(record):
    if not record:
        return None
    try:
        if file_hash(record["config"]) != record["config_sha256"]:
            return None
        run = Path(record["output"])
        result = read(run / "logs/run.json")
        identity = checkpoint_digest(run)
        thresholds = read(run / "model/thresholds.json")
        provenance = read(run / "provenance.json")
        if (result.get("status") != "complete" or result.get("checkpoint_digest") != identity
                or thresholds.get("checkpoint_digest") != identity
                or provenance.get("data_hashes") != record["data_hashes"]):
            return None
        for name in ("config.json", "labels.json", "logs/validation.json", "logs/epochs.jsonl"):
            if not (run / name).is_file():
                return None
        return identity
    except (OSError, ValueError, KeyError, TypeError, BatchError):
        return None


def evaluation_result(record):
    if not record:
        return False
    try:
        output = Path(record["output"])
        report = read(output / "metrics.json")
        counts, provenance = report["counts"], report["provenance"]
        expected = record["expected"]
        if (report.get("metrics") is None or provenance.get("status") != "complete"
                or counts.get("n_failed") != 0
                or counts.get("n_requested") != expected["n_records"]
                or counts.get("n_predicted") != expected["n_records"]
                or any(provenance.get(key) != expected[key] for key in
                       ("checkpoint_digest", "data_sha256", "manifest_sha256", "language"))):
            return False
        if not (output / "failures.jsonl").is_file() or (output / "failures.jsonl").read_text().strip():
            return False
        identities = set()
        ordered = []
        with (output / "predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                identity = row["record_id"]
                if identity in identities or row.get("checkpoint_digest") != expected["checkpoint_digest"]:
                    return False
                identities.add(identity)
                ordered.append(identity)
        return len(identities) == expected["n_records"] and ordered == provenance.get("record_ids")
    except (OSError, ValueError, KeyError, TypeError, BatchError):
        return False


def next_attempt(job, phase):
    base = job / "attempts"
    numbers = [int(path.name.split("-")[-1]) for path in base.glob(f"{phase}-*")
               if re.fullmatch(rf"{phase}-[0-9]+", path.name)]
    return base / f"{phase}-{max(numbers, default=0)+1:04d}"


def evaluation_input(config, args, hashes):
    manifest = Path(config["data"]["manifest"])
    declaration = read(manifest)
    names = {"C#": "csharp", "csharp": "csharp", "Java": "java", "java": "java", "C++": "cpp", "cpp": "cpp"}
    candidates = [entry for entry in declaration["files"]
                  if entry.get("split") == "test" and names.get(entry.get("language")) == args.language]
    if args.test_data:
        requested = Path(args.test_data).resolve()
        candidates = [entry for entry in candidates if (manifest.parent / entry["path"]).resolve() == requested]
    if len(candidates) != 1:
        raise BatchError("require one matching test file; specify --test-data and --language explicitly")
    entry = candidates[0]
    path = (manifest.parent / entry["path"]).resolve()
    if not path.is_relative_to(manifest.parent):
        raise BatchError("test data escapes its manifest directory")
    actual = file_hash(path)
    if actual != entry["sha256"].lower():
        raise BatchError(f"test data hash differs from manifest: {path}")
    return {"data": str(path), "manifest": str(manifest), "data_sha256": actual,
            "manifest_sha256": hashes["manifest"], "language": args.language, "n_records": entry["n_records"]}


def emit(action, job, record=None, evaluation=None):
    values = [action, str(job)]
    if action == "train":
        values += [record["config"], record["output"], record["console_log"]]
    elif action == "eval":
        values += [record["checkpoint"], evaluation["data"], evaluation["manifest"], evaluation["language"],
                   record["output"], record["console_log"]]
    sys.stdout.buffer.write(b"\0".join(value.encode() for value in values) + b"\0")


def plan(args):
    source = Path(args.config).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source.stem):
        raise BatchError("experiment config basename must contain only letters, digits, '.', '_' or '-'")
    config = read(source)
    data = config.get("data", {})
    hashes = {}
    for name in ("train", "val", "manifest", "audit"):
        value = data.get(name)
        if not isinstance(value, str) or not value:
            raise BatchError(f"config requires data.{name}: {source}")
        path = (source.parent / value).resolve()
        hashes[name] = file_hash(path)
        data[name] = str(path)
    config["data"] = data
    if args.revision:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
            raise BatchError("--revision must be an immutable 40-character commit SHA")
        config.setdefault("encoder", {}).update(revision=args.revision.lower(), tokenizer_revision=args.revision.lower())
    seed = config.get("train", {}).get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise BatchError("train.seed must be a nonnegative integer")
    job = Path(args.root).resolve() / source.stem / f"seed-{seed}"
    fingerprint = digest({"config": config, "data_hashes": hashes})
    evaluation = None if args.train_only else evaluation_input(config, args, hashes)
    state = load_state(job)
    previous = state.get("train")
    identity = training_result(previous) if previous and previous.get("fingerprint") == fingerprint else None
    if args.force or identity is None:
        attempt = next_attempt(job, "train")
        record = {"fingerprint": fingerprint, "config": str(attempt / source.name), "output": str(attempt / "run"),
                  "data_hashes": hashes, "console_log": str(attempt / "train.log"), "status": "pending", "created_at": timestamp()}
        if not args.dry_run:
            attempt.mkdir(parents=True, exist_ok=False)
            atomic_write(Path(record["config"]), config)
            record["config_sha256"] = file_hash(record["config"])
            for phase in ("train", "eval"):
                if state.get(phase):
                    state["history"].append({"phase": phase, **state[phase]})
            state.update(config_source=str(source), train=record, eval=None)
            save_state(job, state)
        emit("train", job, record)
        return
    previous.update(status="complete", checkpoint_digest=identity)
    if args.train_only:
        if not args.dry_run:
            save_state(job, state)
        emit("skip", job)
        return
    expected = {key: value for key, value in evaluation.items() if key not in ("data", "manifest")}
    expected["checkpoint_digest"] = identity
    fingerprint = digest(expected)
    previous_eval = state.get("eval")
    if previous_eval and previous_eval.get("fingerprint") == fingerprint and evaluation_result(previous_eval):
        previous_eval["status"] = "complete"
        if not args.dry_run:
            save_state(job, state)
        emit("skip", job)
        return
    attempt = next_attempt(job, "eval")
    record = {"fingerprint": fingerprint, "checkpoint": str(Path(previous["output"]) / "model/best"),
              "output": str(attempt / "report"), "console_log": str(attempt / "eval.log"), "expected": expected,
              "status": "pending", "created_at": timestamp()}
    if not args.dry_run:
        attempt.mkdir(parents=True, exist_ok=False)
        if previous_eval:
            state["history"].append({"phase": "eval", **previous_eval})
        state["eval"] = record
        save_state(job, state)
    emit("eval", job, record, evaluation)


def mark(args):
    job = Path(args.job)
    state = load_state(job)
    record = state[args.phase]
    if args.running:
        record.update(status="running", started_at=timestamp(), runner_pid=args.pid)
        save_state(job, state)
        return
    good = args.exit_code == 0 and (training_result(record) if args.phase == "train" else evaluation_result(record))
    record.update(status="complete" if good else "interrupted" if args.exit_code in (130, 143) else "failed",
                  exit_code=args.exit_code, finished_at=timestamp())
    if good and args.phase == "train":
        record["checkpoint_digest"] = good
    if args.exit_code == 0 and not good:
        record["reason"] = "command exited successfully but complete artifacts failed validation"
    save_state(job, state)
    if not good:
        raise BatchError(f"{args.phase} incomplete; preserved output: {record['output']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("plan")
    preparation.add_argument("--config", required=True)
    preparation.add_argument("--root", required=True)
    preparation.add_argument("--language", choices=("csharp", "java", "cpp"), default="csharp")
    preparation.add_argument("--test-data")
    preparation.add_argument("--revision")
    preparation.add_argument("--train-only", action="store_true")
    preparation.add_argument("--dry-run", action="store_true")
    preparation.add_argument("--force", action="store_true")
    status = commands.add_parser("mark")
    status.add_argument("--job", required=True)
    status.add_argument("--phase", choices=("train", "eval"), required=True)
    status.add_argument("--running", action="store_true")
    status.add_argument("--pid", type=int)
    status.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    try:
        plan(args) if args.command == "plan" else mark(args)
        return 0
    except (OSError, ValueError, KeyError, TypeError, BatchError) as exc:
        print(f"batch state error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
