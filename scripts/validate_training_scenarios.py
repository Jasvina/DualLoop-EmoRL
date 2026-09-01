#!/usr/bin/env python3
"""Fail fast when the training-scenario manifest violates the paper protocol."""

import argparse
import json
from collections import Counter


REQUIRED_FIELDS = ("id", "player", "scene", "task")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-rows", type=int, default=500)
    parser.add_argument("--expected-intents", type=int, default=8)
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at line {line_number}: {exc}") from exc
            missing = [key for key in REQUIRED_FIELDS if not str(row.get(key, "")).strip()]
            if missing:
                raise SystemExit(f"line {line_number} missing required fields: {missing}")
            rows.append(row)

    if len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} scenarios, found {len(rows)}")
    ids = [str(row["id"]).strip() for row in rows]
    if len(set(ids)) != len(ids):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise SystemExit(f"duplicate scenario ids: {duplicates[:10]}")
    intents = Counter(str(row["task"]).strip() for row in rows)
    if len(intents) != args.expected_intents:
        raise SystemExit(f"expected {args.expected_intents} intents, found {len(intents)}: {dict(intents)}")
    print(f"[data-check] rows={len(rows)} intents={len(intents)} counts={dict(intents)}")


if __name__ == "__main__":
    main()
