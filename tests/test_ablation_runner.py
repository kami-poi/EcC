"""Exercise the Linux batch runner's recovery without starting GPU experiments."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_ablations.sh"
pytestmark = pytest.mark.skipif(
    os.name != "posix" or any(shutil.which(tool) is None for tool in ("bash", "flock", "setsid")),
    reason="batch runner targets the remote Linux server",
)

# The command double publishes the same envelope/checksums as the real CLI.
# The final test also exercises real CPU training/evaluation through the CLI.
FAKE_UV = r'''#!__PYTHON__
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert sys.argv[1:4] == ['run', '--no-sync', 'model']
parser = argparse.ArgumentParser()
parser.add_argument('phase')
for option in ('config', 'output', 'checkpoint', 'data', 'manifest', 'language'):
    parser.add_argument('--' + option)
args = parser.parse_args(sys.argv[4:])
if args.phase == 'train':
    name = Path(args.config).stem
else:
    name = Path(args.checkpoint).parents[5].name
with Path(os.environ['FAKE_CALL_LOG']).open('a') as stream:
    stream.write(json.dumps(dict(phase=args.phase, name=name, output=args.output,
                               checkpoint=args.checkpoint, config=args.config)) + '\n')
if os.environ.get('FAKE_BLOCK') == args.phase:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    write(os.environ['FAKE_PIDS'], dict(parent=os.getpid(), child=child.pid))
    time.sleep(120)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
if os.environ.get('FAKE_FAIL_' + args.phase.upper()) == name:
    (output / 'partial.txt').write_text('interrupted work must be preserved')
    raise SystemExit(1)
if os.environ.get('FAKE_REAL') == '1':
    # Let the CLI create its own new output directory.
    output.rmdir()
    from model import main
    from helpers import fake_backend_factory
    raise SystemExit(main(sys.argv[4:], backend_factory=fake_backend_factory))
if args.phase == 'train':
    config = read(args.config)
    files = {'config.json': config, 'labels.json': {'version': 'test'},
             'state.json': {'state': {}}, 'task.safetensors': {'weights': 'stub'}}
    folder = output / 'model/best/generations' / ('a' * 32)
    for key, value in files.items():
        write(folder / key, value)
    hashes = {key: sha(folder / key) for key in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write(folder / 'COMPLETE.json', dict(files=hashes, digest=digest))
    write(output / 'model/best/CURRENT.json', dict(generation='a'*32, digest=digest))
    write(output / 'model/thresholds.json', dict(checkpoint_digest=digest))
    write(output / 'logs/run.json', dict(status='complete', checkpoint_digest=digest))
    write(output / 'provenance.json', dict(data_hashes={key: sha(config['data'][key])
                                                     for key in ('train', 'val', 'manifest', 'audit')}))
    for key in ('config.json', 'labels.json', 'logs/validation.json', 'logs/epochs.jsonl'):
        write(output / key, {})
else:
    digest = read(Path(args.checkpoint) / 'CURRENT.json')['digest']
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    provenance = dict(status='complete', language=args.language, checkpoint_digest=digest,
                      data_sha256=sha(args.data), manifest_sha256=sha(args.manifest),
                      record_ids=[row['record_id'] for row in rows])
    write(output / 'metrics.json', dict(metrics={}, counts=dict(n_requested=len(rows),
          n_predicted=len(rows), n_failed=0), provenance=provenance))
    (output / 'predictions.jsonl').write_text(''.join(json.dumps(dict(record_id=row['record_id'],
                               checkpoint_digest=digest)) + '\n' for row in rows))
    (output / 'failures.jsonl').write_text('')
'''


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def batch(tmp_path):
    data, configs = tmp_path / "data", tmp_path / "configs"
    data.mkdir()
    configs.mkdir()
    files = []
    for split in ("train", "val", "test"):
        path = data / f"{split}.jsonl"
        path.write_text(json.dumps(dict(record_id=f"{split}-1")) + "\n")
        files.append(dict(path=path.name, split=split, language="C#", n_records=1,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write(data / "manifest.json", dict(files=files))
    write(data / "audit.json", dict(passed=True))
    for name in ("qwen3_06b", "frozen", "fullonly", "diffonly", "mean", "concat"):
        write(configs / f"{name}.json", dict(
            data={key: f"../data/{key}.{'jsonl' if key in ('train', 'val') else 'json'}"
                  for key in ("train", "val", "manifest", "audit")},
            train=dict(seed=0), encoder=dict(adaptation=name)))
    uv = tmp_path / "fake-uv"
    uv.write_text(FAKE_UV.replace("__PYTHON__", sys.executable))
    uv.chmod(0o755)
    environment = dict(os.environ, UV_EXECUTABLE=str(uv), PYTHON_EXECUTABLE=sys.executable,
                       FAKE_CALL_LOG=str(tmp_path / "calls.jsonl"), FAKE_PIDS=str(tmp_path / "pids.json"))
    command = ["bash", str(SCRIPT), "--config-dir", str(configs),
               "--output-root", str(tmp_path / "batch output")]
    return dict(root=tmp_path, configs=configs, output=tmp_path / "batch output", env=environment, command=command)


def invoke(batch, *options, **environment):
    return subprocess.run(batch["command"] + list(options), env=dict(batch["env"], **environment),
                          capture_output=True, text=True, timeout=60)


def calls(batch):
    path = batch["root"] / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def status(batch, name="frozen", seed=0):
    return read(batch["output"] / name / f"seed-{seed}" / "status.json")


def expect_success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_queue_complete_then_skip_and_normalize_paths(batch):
    expect_success(invoke(batch, "--revision", "a" * 40))
    assert [(row["phase"], row["name"]) for row in calls(batch)] == [
        (phase, name) for name in ("frozen", "fullonly", "diffonly", "mean", "concat")
        for phase in ("train", "eval")]
    snapshot = read(status(batch)["train"]["config"])
    assert all(Path(path).is_absolute() for path in snapshot["data"].values())
    assert snapshot["encoder"]["revision"] == snapshot["encoder"]["tokenizer_revision"] == "a" * 40
    expect_success(invoke(batch, "--revision", "a" * 40))
    assert len(calls(batch)) == 10


def test_training_failure_continues_queue_and_restart_retries_only_failure(batch):
    result = invoke(batch, FAKE_FAIL_TRAIN="fullonly")
    assert result.returncode == 1
    assert [(row["phase"], row["name"]) for row in calls(batch)] == [
        ("train", "frozen"), ("eval", "frozen"), ("train", "fullonly"),
        ("train", "diffonly"), ("eval", "diffonly"), ("train", "mean"), ("eval", "mean"),
        ("train", "concat"), ("eval", "concat")]
    previous = status(batch, "fullonly")["train"]
    assert previous["status"] == "failed"
    expect_success(invoke(batch))
    assert [(r["phase"], r["name"]) for r in calls(batch)[9:]] == [("train", "fullonly"), ("eval", "fullonly")]
    assert (Path(previous["output"]) / "partial.txt").is_file()
    assert status(batch, "fullonly")["history"][0]["output"] == previous["output"]


def test_failed_evaluation_reuses_training_checkpoint(batch):
    assert invoke(batch, "frozen", FAKE_FAIL_EVAL="frozen").returncode == 1
    previous = status(batch)
    expect_success(invoke(batch, "frozen"))
    assert [row["phase"] for row in calls(batch)] == ["train", "eval", "eval"]
    assert status(batch)["train"]["output"] == previous["train"]["output"]
    assert (Path(previous["eval"]["output"]) / "partial.txt").is_file()


def test_config_change_retrains_only_that_experiment_and_keeps_outputs(batch):
    expect_success(invoke(batch, "frozen", "mean"))
    old = status(batch, "mean")["train"]["output"]
    config = read(batch["configs"] / "mean.json")
    config["train"]["epochs"] = 11
    write(batch["configs"] / "mean.json", config)
    expect_success(invoke(batch, "frozen", "mean"))
    assert [(r["phase"], r["name"]) for r in calls(batch)[4:]] == [("train", "mean"), ("eval", "mean")]
    assert Path(old).is_dir()


def test_data_bytes_change_invalidates_training(batch):
    expect_success(invoke(batch, "frozen", "--train-only"))
    with (batch["root"] / "data/train.jsonl").open("a") as stream:
        stream.write(" \n")
    expect_success(invoke(batch, "frozen", "--train-only"))
    assert [r["phase"] for r in calls(batch)] == ["train", "train"]


@pytest.mark.parametrize("artifact", ["model/thresholds.json", "task.safetensors"])
def test_incomplete_or_corrupt_training_artifacts_are_not_skipped(batch, artifact):
    expect_success(invoke(batch, "frozen"))
    run = Path(status(batch)["train"]["output"])
    if artifact.endswith("safetensors"):
        (run / "model/best/generations" / ("a" * 32) / artifact).write_text("corrupt")
    else:
        (run / artifact).unlink()
    expect_success(invoke(batch, "frozen"))
    assert [r["phase"] for r in calls(batch)] == ["train", "eval", "train", "eval"]


def test_incomplete_evaluation_retries_evaluation_only(batch):
    expect_success(invoke(batch, "frozen"))
    output = Path(status(batch)["eval"]["output"])
    (output / "predictions.jsonl").write_text("")
    expect_success(invoke(batch, "frozen"))
    assert [r["phase"] for r in calls(batch)] == ["train", "eval", "eval"]


def test_finished_artifacts_are_adopted_after_status_publication_crash(batch):
    expect_success(invoke(batch, "frozen"))
    path = batch["output"] / "frozen/seed-0/status.json"
    value = read(path)
    for phase in ("train", "eval"):
        value[phase]["status"] = "running"
    write(path, value)
    expect_success(invoke(batch, "frozen"))
    assert len(calls(batch)) == 2
    assert status(batch)["train"]["status"] == status(batch)["eval"]["status"] == "complete"


def test_dry_run_does_not_write_state_or_run_commands(batch):
    result = invoke(batch, "--include-main", "--dry-run")
    expect_success(result)
    assert "TRAIN qwen3_06b" in result.stdout and "TRAIN concat" in result.stdout
    assert not batch["output"].exists() and not calls(batch)


def test_selective_rerun_does_not_exit_on_nonmatching_experiments(batch):
    expect_success(invoke(batch, "frozen", "mean"))
    expect_success(invoke(batch, "frozen", "mean", "--rerun", "mean"))
    assert [(r["phase"], r["name"]) for r in calls(batch)[4:]] == [("train", "mean"), ("eval", "mean")]


def test_train_only_defers_evaluation_and_fail_fast(batch):
    expect_success(invoke(batch, "frozen", "--train-only"))
    expect_success(invoke(batch, "frozen"))
    assert [r["phase"] for r in calls(batch)] == ["train", "eval"]
    assert invoke(batch, "fullonly", "mean", "--fail-fast", FAKE_FAIL_TRAIN="fullonly").returncode == 1
    assert calls(batch)[-1]["name"] == "fullonly"


def test_seed_isolated_jobs_and_invalid_queue(batch):
    expect_success(invoke(batch, "frozen", "--train-only"))
    config = read(batch["configs"] / "frozen.json")
    config["train"]["seed"] = 4
    write(batch["configs"] / "frozen.json", config)
    expect_success(invoke(batch, "frozen", "--train-only"))
    assert status(batch, seed=0)["train"]["status"] == status(batch, seed=4)["train"]["status"] == "complete"
    assert invoke(batch, "frozen", "frozen.json").returncode == 2
    assert invoke(batch, "frozen", "--rerun", "mean").returncode == 2


def test_preparation_failure_isolated_and_test_file_verified(batch):
    result = invoke(batch, "missing", "frozen")
    assert result.returncode == 1
    assert status(batch)["eval"]["status"] == "complete"
    with (batch["root"] / "data/test.jsonl").open("a") as stream:
        stream.write(" \n")
    assert invoke(batch, "mean").returncode == 1
    assert len(calls(batch)) == 2


def await_file(path, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.is_file():
            return read(path)
        if process.poll() is not None:
            pytest.fail(f"process ended early: {process.communicate()}")
        time.sleep(.05)
    pytest.fail(f"timed out waiting for {path}")


def test_batch_lock_and_interrupt_entire_training_process_group(batch):
    process = subprocess.Popen(batch["command"] + ["frozen"],
                               env=dict(batch["env"], FAKE_BLOCK="train"),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        pids = await_file(batch["root"] / "pids.json", process)
        result = invoke(batch, "mean")
        assert result.returncode == 2 and "Another batch" in result.stderr
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, stdout + stderr
        assert status(batch)["train"]["status"] == "interrupted"
        # A descendant may be a zombie until init reaps it, but must not keep running.
        for pid in pids.values():
            proc_status = Path(f"/proc/{pid}/status")
            if proc_status.exists():
                assert "State:\tZ" in proc_status.read_text()
        expect_success(invoke(batch, "frozen"))
        assert [r["phase"] for r in calls(batch)] == ["train", "train", "eval"]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        if (batch["root"] / "pids.json").is_file():
            for pid in read(batch["root"] / "pids.json").values():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_runner_accepts_real_training_and_evaluation_artifacts(batch):
    from test_training import fixture
    data = batch["root"] / "data"
    cfg = fixture(data)
    payload = asdict(cfg)
    payload["data"] = {**payload["data"], **{
        key: f"../data/{Path(payload['data'][key]).name}" for key in ("train", "val", "manifest", "audit")}}
    write(batch["configs"] / "frozen.json", payload)
    expect_success(invoke(batch, "frozen", "--language", "java", FAKE_REAL="1"))
    assert status(batch, seed=3)["train"]["status"] == "complete"
    assert status(batch, seed=3)["eval"]["status"] == "complete"
    expect_success(invoke(batch, "frozen", "--language", "java", FAKE_REAL="1"))
    assert len(calls(batch)) == 2
