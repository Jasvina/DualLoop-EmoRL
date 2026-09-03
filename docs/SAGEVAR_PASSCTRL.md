# SAGEVariant-PassCtrl: Method and Reproducibility

## Status

This document describes the corrected main-method package in
`release/work_sagevar_passctrl_train500_n4`. It supersedes the earlier
three-difficulty/1,500-profile experiment. The current main method has no
TopVar gate and never permanently deletes a profile.

## Data construction

The data pipeline starts from the fixed 100 complete SAGE profiles. Each seed
is used only as a source for generation and evaluation; the original seed row
is not inserted into the training pool. An API model generates five
surface-level variants for every seed:

```text
100 SAGE seeds x 5 generated variants = 500 training profiles
```

Every generated profile retains the seed's emotional-support topic, core
conflict, hidden theme, `main_cha`, `cha_group`, task, overall difficulty, and
the complete SAGE reaction structure. It may vary name, age, occupation,
speaking style, life context, and event details. The generation prompt forbids
SEAD-style scalar profiles and requires explicit high/medium/low emotion
behavior plus the emotion response when an NPC matches or deviates from the
hidden theme.

The released JSONL passed these structural checks:

- exactly 500 rows and 500 unique profile IDs;
- exactly 100 source IDs and five variants per source;
- variant indices 1 through 5 for every source;
- all required SAGE/RLVER fields present;
- no duplicate `player + scene` pairs.

The generated pool is
`release/work_sagevar_passctrl_train500_n4/data/train_profile_sage_variants_500.jsonl`.
The reproducible generator is
`release/work_sagevar_passctrl_train500_n4/scripts/generate_sage_variants_500.py`.
It reads `IDEALAB_API_KEY` from the environment; credentials are never stored
in the repository.

## Training flow

For every optimizer step, the dataset supplies eight profile groups. Each
group samples one complete profile, then creates four rollouts under that same
profile (`GRPO_N=4`). The profile ID is also the GRPO group ID, so rollouts from
different profiles are not mixed when group-relative advantages are computed.

The user simulator receives the complete SAGE profile. The assistant receives
only the normal dialogue context; it does not receive profile IDs, controller
scores, success labels, or discarded difficulty metadata.

The assistant reward is:

```text
assistant reward = 0.92 * normalized emotion reward
                 + 0.08 * counselor auxiliary reward
```

The 0.08 auxiliary component supplies a small dense signal when the four
emotion rewards become identical. It is held fixed across controller
ablations so that the controller remains the only changed variable.

## Pass-rate controller

The controller maintains independent history for every generated profile ID.
During the first 100 optimizer steps, every worker samples uniformly from the
entire 500-profile pool while collecting controller observations. This is not
an easy-to-hard curriculum.

For one profile group, let the four final emotion scores be
`10, 40, 70, 90`. A rollout succeeds when its emotion score is at least 50, so
this group has two successes and a pass rate of 0.5. Its training value is:

```text
training_value = max(0.05, 1 - 2 * abs(pass_rate - 0.5))
```

With four rollouts, the mapping is:

| Pass rate | Training value |
|---:|---:|
| 0.00 | 0.05 |
| 0.25 | 0.50 |
| 0.50 | 1.00 |
| 0.75 | 0.50 |
| 1.00 | 0.05 |

For each profile, `sample_score` is the arithmetic mean of all historical
group training values, floored at 0.05. After warmup, the 500 scores are
normalized into sampling probabilities. Profiles near the current competence
boundary (pass rate near 0.5) are sampled more often. Always-failed and
always-succeeded profiles remain reachable through the 0.05 floor; they are
not deleted.

## Fixed experiment configuration

| Setting | Value |
|---|---:|
| Training steps | 500 |
| Profile groups per step | 8 |
| Rollouts per group | 4 |
| Effective rollouts per step | 32 |
| Learning rate | `1e-6` |
| Actor warmup ratio | `0.02` |
| KL controller coefficient | `0.001` |
| Actor KL-loss coefficient | `0.001` |
| PPO epochs | 1 |
| Max prompt length | 25,000 |
| Max response length | 7,000 |
| Per-turn length | 7,000 |
| Max turns | 8 |
| API workers | 16 |
| Pass threshold | 50 |
| Controller warmup | 100 optimizer steps |
| Minimum sample score | 0.05 |
| Counselor auxiliary weight | 0.08 |

The response-length metrics and critic/value masks use
`attention_mask * generation_mask` when available. This keeps padding outside
assistant token counts and critic loss without changing the controller,
reward, data, or sampling equations.

## Removed design

The current launcher does not build or read `train_profile_sead3_1500.jsonl`.
It does not append `diff_text`, easy/medium/hard labels, cooperation, trust, or
emotion-intensity scalar descriptions to the user prompt. Any package or log
containing `3diff`, `sead3`, or `train1500` belongs to the superseded design.

## Resume limitation

The released package uses worker-local in-memory controller statistics and
`trainer.resume_mode=disable`. Run the 500 steps uninterrupted. If checkpoint
resume is required, first add rank-sharded save/load for `_PROFILE_STEP` and
`_PROFILE_STATS`, then switch resume mode deliberately. Resuming only model and
optimizer state after adaptive sampling has started would reset the sampling
history and would not be a strictly continuous controller run.

## Experiment isolation

For the controller ablation, use the same 500-profile file, reward mixture,
GRPO group size, lengths, batch size, optimizer settings, and random-seed
policy. Remove only adaptive profile weighting and sample uniformly throughout.
Do not compare against the superseded 1,500-profile data as a controller
ablation because that changes both data and sampling.
