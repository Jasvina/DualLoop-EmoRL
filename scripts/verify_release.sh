#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 scripts/validate_training_scenarios.py \
  --input data/train_scenarios.jsonl \
  --expected-rows 500 \
  --expected-intents 8
python3 -m unittest discover -s tests -v
python3 scripts/verify_released_results.py
bash -n launch/train_dual_loop_qwen3_8b_n4.sh
shasum -a 256 -c MANIFEST.sha256
