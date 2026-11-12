"""Benchmark three explicitly selected training-source pairs on real CUDA."""

import argparse
import hashlib
import json
from pathlib import Path

from model.benchmark.gpu import benchmark_pairs
from model.config import load_config
from model.data.jsonl import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_records(args.data, field_map=cfg.data.field_map)
    report = benchmark_pairs(cfg, records, args.output)
    report["data_file"] = str(args.data.resolve())
    with args.data.open("rb") as stream:
        report["data_file_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output),
                      "failure": report.get("failure"),
                      "pair_failures": [{"record_id": row["record_id"], "failure": row["failure"]}
                                        for row in report["pairs"] if "failure" in row]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
