# DualLoop-EmoRL

Official implementation and research artifacts for **Dual-Loop Self-Evolution
via Verifiable Emotion Feedback for Multi-Turn Empathetic Dialogue**.

The method uses one source of trajectory-level emotion feedback in two nested
loops: an inner GRPO loop improves the dialogue policy, while an outer
controller reallocates subsequent interaction conditions around the policy's
current competence boundary. The user simulator and verifier remain frozen.

## Highlights

- 24 controllable interaction states: 3 disclosure levels x 2 activation
  levels x 4 trust levels.
- One scenario and interaction state shared by every rollout in a GRPO group.
- Hierarchical controller over 8 support intents x 24 states (192 units).
- Boundary utility, uncertainty-guided exploration, and uniform rehearsal.
- Persistent, process-safe controller history backed by SQLite.
- Exact 500-scenario training manifest and three independent SAGE result files.
- Public preprint source, figures, aggregate results, and diagnostic exports.

## Repository Layout

```text
method/                   Controller and rollout integration
launch/                   Released Qwen3-8B training configuration
data/                     Deterministic 500-scenario training manifest
results/sage/             Three complete-method SAGE runs and summary
results/controller/       Controller dynamics and unit snapshots
results/state_validation/ Interaction-state validation records
results/analysis/         Aggregate analysis outputs
scripts/                  Validation and result utilities
tests/                    Controller and state-prompt unit tests
paper/                    Public paper source, references, and figures
docs/                     Reproducibility and provenance notes
```

An additional, separately documented **SAGEVariant-PassCtrl** experiment is
available under `release/` with a transfer archive under `artifacts/`. It uses a
different profile-level controller and is not evidence for, or a replacement
for, the paper's 24-state hierarchical method. See
[`docs/SAGEVAR_PASSCTRL.md`](docs/SAGEVAR_PASSCTRL.md) for its exact scope.

Start with the [artifact index](docs/ARTIFACT_INDEX.md) for a claim-by-claim
map from the paper to released code, data, and results. The exact environment,
hyperparameters, resume behavior, and validation commands are documented in
[the reproducibility notes](docs/REPRODUCIBILITY.md). The
[public-release scope](docs/RELEASE_SCOPE.md) records what is included,
what must be obtained from upstream providers, and what is intentionally not
published. See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for runtime versions,
[`docs/INTEGRATION.md`](docs/INTEGRATION.md) for the controller hooks, and
[`docs/EVALUATION.md`](docs/EVALUATION.md) for evaluation protocols and scaling.

## Paper-to-Code Map

| Paper component | Source |
|---|---|
| 24 interaction states | `method/interaction_state_prompt.py` |
| Group-shared scenario and state | `method/rollout_adapter.py` |
| Hard group pass rate | `EmotionBoundaryController.passed` and `update_group` |
| Hierarchical intent-state estimate | `EmotionBoundaryController.sampling_distribution` |
| Boundary utility and uncertainty bonus | `EmotionBoundaryController.sampling_distribution` |
| Uniform initialization and rehearsal | `EmotionBoundaryController.sampling_distribution` |
| Persistent controller history | SQLite operations in `emotion_boundary_controller.py` |
| Optimization invocation | `launch/train_dual_loop_qwen3_8b_n4.sh` |

## Design Decision: Why the User Is Frozen

Early variants jointly optimized user-side tokens with rewards derived from the
assistant's final emotion score. Across binary success-rate, dense boundary,
hand-written user-quality proxy, and latent-state variants, this produced sparse
credit, non-stationary training, or proxy optimization without held-out SAGE
improvement. The released method therefore treats the user simulator as a
frozen environment and adapts only the distribution over explicit interaction
conditions. This keeps scenario semantics and emotional dynamics stable while
still moving training experience toward the current policy boundary.

The negative-results record and the reasoning behind the final design are in
[`docs/SELF_PLAY_EXPERIMENTS.md`](docs/SELF_PLAY_EXPERIMENTS.md). Exact reward,
threshold, grouping, and resume semantics are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

The repository contains the method-specific contribution rather than a fork of
the full RL stack. GRPO, Ray/FSDP, and vLLM are supplied by a compatible
[VERL](https://github.com/verl-project/verl) / RLVER runtime. This keeps the
release focused and makes the changes introduced by this work easy to inspect.
Runtime authors can use `docs/INTEGRATION.md` for the two required controller
hooks and the invariants that preserve group-relative training.

## Quick Validation

The method-only checks require Python 3.10+, NumPy, and SQLite:

```bash
python -m pip install -r requirements.txt
python scripts/validate_training_scenarios.py \
  --input data/train_scenarios.jsonl \
  --expected-rows 500 --expected-intents 8
python -m unittest discover -s tests -v
```

The checks do not require model weights, GPUs, or access to the simulator API.
Run every released integrity check, including result recomputation and
manifest verification, with:

```bash
bash scripts/verify_release.sh
```

## Training

Provide a compatible runtime, model directory, output directory, and simulator
endpoint. No credentials or private service URLs are stored in this repository.

```bash
export VERL_CODE_DIR=/path/to/compatible/verl/runtime
export TRAIN_MODEL_PATH=/path/to/Qwen3-8B
export OUTPUT_DIR=/path/to/output
export PLAYER_API_BASE=https://your-endpoint.example/v1
export PLAYER_API_KEY=YOUR_KEY
bash launch/train_dual_loop_qwen3_8b_n4.sh
```

The released configuration uses 500 optimizer steps, batch size 16, GRPO group
size 4, learning rate `3e-7`, AdamW weight decay `0.01`, warm-up ratio `0.02`,
one PPO epoch, clip ratio `0.2`, gradient clip `10.0`, and policy/actor KL
coefficients `0.01` with `low_var_kl`. Policy temperature and top-p are `1.0`;
the maximum dialogue length is 8 turns.

The controller uses 1,600 completed groups of uniform initialization, success
threshold 50, hierarchical shrinkage and state prior 4, uncertainty weight
`0.15`, uniform rehearsal `0.10`, sampling temperature `1.0`, and minimum score
`0.05`. Incomplete groups and groups with non-finite rewards are not recorded.

## Released Results

| Run | SAGE Overall | Success | Failure |
|---|---:|---:|---:|
| 1 | 81.02 | 49.0 | 9.0 |
| 2 | 78.58 | 50.0 | 13.0 |
| 3 | 78.11 | 48.0 | 12.0 |
| **Mean / sample SD** | **79.24 / 1.56** | **49.0** | **11.33** |

`results/paper_main_results.csv` contains the protocol-matched aggregate table,
and `results/ablation_summary.csv` records the component ablations.
Raw complete-method SAGE trajectories are retained in `results/sage/`; service
URLs, timestamps, local paths, and experiment-tracking identifiers were removed.
`python scripts/verify_released_results.py` recomputes their principal metrics
and verifies the reported three-run mean and sample standard deviation.

## Paper

The public paper source is in `paper/main.tex`. From `paper/`, compile with:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The technical supplement and recovered appendix source are under
`paper/supplement/`. Final submission PDFs are not treated as canonical build
artifacts; the tracked LaTeX source and figure assets are.

## Data and Model Provenance

`data/train_scenarios.jsonl` is the exact deterministic 500-record manifest
derived from the public RLVER training scenarios cited in the paper. Dataset,
benchmark, model, and API terms remain governed by their upstream licenses and
providers. Model checkpoints and third-party benchmark packages are not
redistributed here.
`docs/RELEASE_SCOPE.md` lists the included artifacts and external dependencies.

## License

Method code and repository-authored utilities are released under the Apache
License 2.0. The training implementation interoperates with VERL, which is also
Apache-2.0 licensed. See `NOTICE` for attribution. Data and third-party assets
remain subject to their original terms.

## Citation

```bibtex
@article{wei2026dualloop,
  title={Dual-Loop Self-Evolution via Verifiable Emotion Feedback for Multi-Turn Empathetic Dialogue},
  author={Wei, Yi and Jiang, Shuo and Dou, Huaixia and Zhu, Jie and Li, Junhui and Guo, Lifan and Chen, Feng and Zhang, Chi},
  year={2026}
}
```
