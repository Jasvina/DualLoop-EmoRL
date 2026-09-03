# Artifact Index

This index connects the paper's claims to the exact public artifacts used to
implement or support them. Paths are relative to the repository root.

## Method

| Paper element | Public artifact | What to inspect |
|---|---|---|
| 24 interaction states | `method/interaction_state_prompt.py` | The 3 disclosure, 2 activation, and 4 trust levels and their simulator-only prompts |
| Scenario-state composition | `method/rollout_adapter.py` | Online composition without changing the original scenario semantics |
| Group-shared environment | `method/rollout_adapter.py` | Reuse of one scenario and state across all rollouts in a group |
| Hard group pass rate | `method/emotion_boundary_controller.py` | Thresholding and cumulative group updates |
| 192 intent-state units | `method/emotion_boundary_controller.py` | Statistics indexed by 8 support intents and 24 states |
| Hierarchical evidence sharing | `method/emotion_boundary_controller.py` | Unit evidence shrunk toward the corresponding cross-intent state estimate |
| Boundary utility | `method/emotion_boundary_controller.py` | Highest utility around an estimated pass rate of 0.5 |
| Uncertainty exploration | `method/emotion_boundary_controller.py` | Visit-count-dependent exploration bonus |
| Uniform rehearsal | `method/emotion_boundary_controller.py` | Probability mixture that retains full support |
| Resume support | `method/emotion_boundary_controller.py` | WAL-enabled SQLite state and cumulative counters |
| Released training invocation | `launch/train_dual_loop_qwen3_8b_n4.sh` | Model, rollout, optimizer, and controller overrides |

## Data and Results

| Evidence | Public artifact | Scope |
|---|---|---|
| Training manifest | `data/train_scenarios.jsonl` | Exact ordered set of 500 training scenarios with 8 native support intents |
| Main paper table | `results/paper_main_results.csv` | Protocol-matched aggregate values reported in the paper |
| Independent SAGE runs | `results/sage/ours_run_1.jsonl`, `ours_run_2.jsonl`, `ours_run_3.jsonl` | Sanitized per-scenario outputs from three independently trained policies |
| SAGE summary | `results/sage/summary.csv` | Recomputed run-level mean and sample standard deviation |
| Controller evolution | `results/controller/controller_evolution_to350.csv` | Stepwise controller diagnostics |
| Controller phases | `results/controller/controller_phase_diagnostics.csv` | Uniform-initialization and adaptive-phase summaries |
| State snapshots | `results/controller/controller_state24_snapshots.csv` | Aggregated statistics for all 24 interaction states |
| Unit snapshots | `results/controller/controller_unit192_snapshots.csv` | Statistics for all 192 intent-state units |
| State validation | `results/state_validation/` | Predictions, human review, confusion matrices, and aggregate summary |
| Additional analyses | `results/analysis/` | Intent-level and paired-scenario analyses |

## Paper and Supplement

- `paper/main.tex`: canonical public manuscript source.
- `paper/references.bib`: bibliography used by the manuscript.
- `paper/Figures/`: final manuscript figures in PDF and PNG form.
- `paper/supplement/main.tex`: technical-supplement entry point.
- `paper/supplement/RecoveredAppendix.tex`: recovered extended appendix source.

## Verification Utilities

- `scripts/validate_training_scenarios.py` checks manifest size, identifiers,
  required fields, and intent coverage.
- `scripts/sanitize_and_summarize_sage.py` removes machine-specific metadata
  and recomputes SAGE summaries.
- `tests/test_emotion_boundary_controller.py` checks controller updates,
  sampling, persistence, and invalid-group handling.
- `tests/test_interaction_state_prompt.py` checks the 24-state construction and
  prompt behavior.
- `MANIFEST.sha256` provides content hashes for released research artifacts.

## Recommended Reading Order

1. Read `README.md` for the method and quick start.
2. Read `method/interaction_state_prompt.py` and
   `method/emotion_boundary_controller.py` for the core contribution.
3. Run the method-only validation commands in `docs/REPRODUCIBILITY.md`.
4. Inspect `results/paper_main_results.csv` and `results/sage/summary.csv`.
5. Use the controller and state-validation exports for deeper analysis.
