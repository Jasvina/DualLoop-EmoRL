# Reproducibility Notes

## Released Scope

This repository releases the method-specific code, exact training manifest,
configuration, unit tests, complete-method SAGE result files, aggregate paper
results, controller exports, state-validation records, and public paper source.
It does not redistribute model checkpoints, third-party benchmark packages,
private API credentials, or the generic RL runtime.

For a claim-by-claim artifact map, see `docs/ARTIFACT_INDEX.md`. For the full
inclusion and exclusion policy, see `docs/RELEASE_SCOPE.md`.

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

## Reward and Threshold Semantics

The inner and outer loops consume different transformations of the same
verified trajectory outcome:

```text
complete dialogue -> terminal SAGE emotion R in [0, 100]
                  -> assistant policy reward: continuous R
                  -> controller observation: 1[R >= eta]
```

The assistant receives continuous terminal emotion. No thresholded reward is
used for its policy gradient. The frozen user simulator receives no policy
loss, and visible user tokens are never updated.

The released controller uses `eta = 50`. This is a reference criterion for
estimating whether successful and unsuccessful assistant behavior coexist
under one controlled condition; it is not a universal definition of an
emotionally successful conversation. All four trajectories in a GRPO group
share the same scenario and interaction state. Their binary outcomes are
averaged into one group pass-rate observation. A stabilized historical rate
near 0.5 receives high boundary utility because the current policy remains
inconsistent there.

Matched runs with thresholds 40, 50, and 60 reached SAGE Overall scores of
77.46, 78.11, and 77.72. The center value was retained, while the close results
show that the method does not depend on a narrowly tuned cutoff. Thresholding
changes future condition allocation only; it does not add model calls, alter
the verifier, or replace the continuous assistant reward.

## What Evolves

The term "dual-loop self-evolution" refers to a policy changing its own future
training distribution through verified outcomes:

1. The inner loop updates only the assistant policy from continuous emotion.
2. The outer loop updates cumulative intent-state statistics from completed
   groups and reallocates future rollout conditions.
3. The user simulator, emotion verifier, scenario facts, and state realization
   rules stay frozen.

This is intentionally different from two-policy adversarial self-play. The
failed co-training variants and their diagnostics are recorded in
`docs/SELF_PLAY_EXPERIMENTS.md`.

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

To verify the byte-level integrity of the released research artifacts, run:

```bash
sha256sum -c MANIFEST.sha256
```
