#!/usr/bin/env python3
"""Verify released SAGE trajectories against the published aggregates."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAGE_DIR = ROOT / "results" / "sage"


def summarize(path: Path) -> dict[str, float]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    scores = [float(record["emo_point"]) for record in records]
    return {
        "n": float(len(scores)),
        "sage_overall": statistics.fmean(scores),
        "success_rate": 100.0 * sum(score >= 100 for score in scores) / len(scores),
        "failure_rate": 100.0 * sum(score < 10 for score in scores) / len(scores),
    }


def require_close(label: str, actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise SystemExit(f"{label}: expected {expected}, recomputed {actual}")


def main() -> None:
    with (SAGE_DIR / "summary.csv").open(encoding="utf-8", newline="") as stream:
        expected_rows = list(csv.DictReader(stream))

    recomputed = []
    for row in expected_rows:
        run = row["run"]
        summary = summarize(SAGE_DIR / f"{run}.jsonl")
        require_close(f"{run}.n", summary["n"], float(row["n"]))
        for field in ("sage_overall", "success_rate", "failure_rate"):
            require_close(f"{run}.{field}", summary[field], float(row[field]))
        recomputed.append(summary)

    overalls = [row["sage_overall"] for row in recomputed]
    require_close("three-run mean", statistics.fmean(overalls), 79.23666666666666)
    require_close("three-run sample SD", statistics.stdev(overalls), 1.562188635643381)
    print(
        "[result-check] runs=3 scenarios_per_run=100 "
        f"mean={statistics.fmean(overalls):.4f} sample_sd={statistics.stdev(overalls):.4f}"
    )


if __name__ == "__main__":
    main()
