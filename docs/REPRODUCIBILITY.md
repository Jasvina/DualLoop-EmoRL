# Reproducibility Notes

## Released Scope

This repository releases the method-specific code, exact training manifest,
configuration, unit tests, complete-method SAGE result files, aggregate paper
results, controller exports, state-validation records, and public paper source.
It does not redistribute model checkpoints, third-party benchmark packages,
private API credentials, or the generic RL runtime.

## Environment

- Base policy: Qwen3-8B
- User simulator and emotion verifier: frozen DeepSeek-V3 services
- Training hardware: 4 x NVIDIA A800 80GB
- Optimizer: AdamW, weight decay 0.01
- Scenario batch size: 16
- GRPO group size: 4
- Training steps: 500
- Learning rate: 3e-7
- LR warm-up ratio: 0.02
- PPO epochs: 1
- Clip ratio: 0.2
- Gradient clip: 10.0
- KL controller / actor KL loss: 0.01 / 0.01
- KL loss type: low_var_kl
- Policy temperature / top-p: 1.0 / 1.0
- Maximum dialogue turns: 8

The exact command-line overrides are in
`launch/train_dual_loop_qwen3_8b_n4.sh`.

## Controller

The 500 complete scenarios retain eight native support intents. Each scenario
is composed online with one of 24 simulator-only interaction states (3 x 2 x 4),
forming 192 intent-state controller units. All rollouts in one GRPO group share
the same scenario and state. The continuous terminal emotion outcome updates the
policy; the thresholded group pass rate updates the controller once per complete
group.

Controller history is cumulative and stored in a WAL-enabled SQLite database.
For resumed training, preserve the database and pass its path through
`EGSEC_CONTROLLER_STATE_FILE` together with the matching policy checkpoint.

## Evaluation Boundaries

Training scenarios and the 100-scenario SAGE test set are disjoint. SAGE,
ESC-Eval, EIBench, ESConv, and human evaluation use the protocols described in
the paper. Third-party evaluation code and model weights should be obtained from
their official sources and remain under their original licenses.

## Integrity Checks

```bash
python scripts/validate_training_scenarios.py \
  --input data/train_scenarios.jsonl \
  --expected-rows 500 --expected-intents 8
python -m unittest discover -s tests -v
bash -n launch/train_dual_loop_qwen3_8b_n4.sh
```
