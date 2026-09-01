#!/usr/bin/env python3
"""Sanitize run metadata and summarize the supplied SAGE JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def load_and_sanitize(path: Path, run_name: str) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("npc_url", None)
            metadata.pop("timestamp", None)
            metadata["npc_model"] = run_name
        records.append(record)
    return records


def summarize(records: list[dict]) -> dict:
    scores = [float(record["emo_point"]) for record in records]
    states = [str(record["emo_state"]).removeprefix("Emotion-") for record in records]
    return {
        "n": len(scores),
        "sage_overall": statistics.fmean(scores),
        "success_rate": 100.0 * sum(score >= 100 for score in scores) / len(scores),
        "failure_rate": 100.0 * sum(score < 10 for score in scores) / len(scores),
        **{f"state_{state}": states.count(state) for state in ("S", "A", "B", "C", "F")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for index, path in enumerate(args.files, 1):
        run_name = f"ours_run_{index}"
        records = load_and_sanitize(path, run_name)
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        rows.append({"run": run_name, **summarize(records)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    values = [row["sage_overall"] for row in rows]
    print(f"runs={len(rows)} mean={statistics.fmean(values):.4f} sample_sd={statistics.stdev(values):.4f}")


if __name__ == "__main__":
    main()
