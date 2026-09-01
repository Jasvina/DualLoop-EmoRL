# DualLoop-EmoRL

Official implementation and reproducibility artifacts for **Dual-Loop
Self-Evolution via Verifiable Emotion Feedback for Multi-Turn Empathetic
Dialogue**.

DualLoop-EmoRL uses the same verified emotion outcomes at two nested time
scales. The inner loop optimizes a multi-turn dialogue policy with continuous
emotion rewards. The outer loop estimates the policy-relative utility of
interaction conditions and reallocates the next rollout budget toward the
policy's evolving competence boundary.

## Repository contents

```text
method/
  emotion_boundary_controller.py  # hierarchical boundary controller
  interaction_state_prompt.py      # exact 3 x 2 x 4 state instructions
  rollout_adapter.py               # group-shared sampling and feedback hook
launch/
  train_dual_loop_qwen3_8b_n4.sh   # released Qwen3-8B training configuration
data/
  train_scenarios.jsonl            # deterministic 500-scenario manifest
results/sage/
  ours_run_{1,2,3}.jsonl           # raw SAGE outputs for three training runs
  summary.csv                      # run-level aggregate results
scripts/                           # validation and result summarization
tests/                             # controller and prompt unit tests
paper/                             # public paper and technical supplement source
```

This repository intentionally excludes model checkpoints, API credentials,
benchmark packages, experiment-tracking logs, and generic RL infrastructure.
The released modules contain the paper-specific contribution; GRPO, Ray/FSDP,
and vLLM are provided by the compatible training runtime selected by the user.

## Paper-to-code map

| Paper component | Implementation |
|---|---|
| 24 interaction states | `method/interaction_state_prompt.py` |
| Group-shared scenario and state | `method/rollout_adapter.py` |
| Hard group pass rate | `EmotionBoundaryController.passed` and `update_group` |
| Hierarchical intent-state estimate | `EmotionBoundaryController.sampling_distribution` |
| Boundary utility and uncertainty bonus | `EmotionBoundaryController.sampling_distribution` |
| Uniform initialization and rehearsal | `EmotionBoundaryController.sampling_distribution` |
| Persistent controller history | SQLite operations in `emotion_boundary_controller.py` |
| Released optimization configuration | `launch/train_dual_loop_qwen3_8b_n4.sh` |

## Installation and validation

The method-only checks require Python 3.10+ and NumPy:

```bash
python -m pip install -r requirements.txt
python scripts/validate_training_scenarios.py \
  --input data/train_scenarios.jsonl \
  --expected-rows 500 --expected-intents 8
python -m unittest discover -s tests -v
```

## Training

The launcher records the released Qwen3-8B setup. Supply a compatible VERL
runtime, a Qwen3-8B checkpoint, an output directory, and credentials for the
frozen user-simulator/verifier service:

```bash
export VERL_CODE_DIR=/path/to/compatible/verl/runtime
export TRAIN_MODEL_PATH=/path/to/Qwen3-8B
export OUTPUT_DIR=/path/to/output
export PLAYER_API_BASE=https://your-endpoint.example/v1
export PLAYER_API_KEY=YOUR_KEY
bash launch/train_dual_loop_qwen3_8b_n4.sh
```

The controller database uses a local POSIX path under `/tmp` by default. To
resume an interrupted run, preserve the database together with the policy
checkpoint and set `EGSEC_CONTROLLER_STATE_FILE` to its path.

The released configuration uses 500 scenarios, batch size 16, GRPO group size
4, 500 optimizer steps, learning rate `3e-7`, and eight dialogue turns. The
controller performs 1,600 completed groups of uniform initialization before
adaptive allocation. Remaining optimization and controller values are explicit
in the launcher and can be overridden through environment variables.

## Released results

The three independent complete-method SAGE runs achieve Overall scores of
`81.02`, `78.58`, and `78.11`, for a mean of `79.24` and sample standard
deviation of `1.56`. Raw per-scenario outputs are under `results/sage/`.

To regenerate the summary from sanitized raw outputs:

```bash
python scripts/sanitize_and_summarize_sage.py --help
```

## Data provenance

`data/train_scenarios.jsonl` is the deterministic 500-record training manifest
derived from the public RLVER scenarios. RLVER is distributed in Tencent's
[`digitalhuman`](https://github.com/Tencent/digitalhuman) repository under the
MIT License. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution. Benchmark
packages and model weights are not redistributed here and remain governed by
their respective upstream terms.

## Paper source

The public manuscript source is in `paper/`, and the extended technical
supplement is in `paper/supplement/`. From `paper/`, compile the manuscript
with the standard BibTeX sequence:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Citation

Please use the metadata in [`CITATION.cff`](CITATION.cff). A BibTeX entry can
also be exported directly from GitHub's **Cite this repository** menu.

## License

The original code in this repository is released under the Apache License 2.0.
Third-party data, models, benchmarks, templates, and infrastructure retain
their original licenses and terms.
