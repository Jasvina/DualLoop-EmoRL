# SAGE-Variant-500 PassCtrl Package

This package is a cleaned main-method variant for the corrected SAGE/RLVER profile setting.

## Method

- Training profile pool: `data/train_profile_sage_variants_500.jsonl`
- The package includes the generated 500-line SAGE-variant JSONL.
- Original SAGE 100 seed profiles are not used for training.
- Profile sampling uses a sample-level pass-rate controller keyed by each generated profile `id`.
- Warmup is full-pool uniform sampling for 100 steps.
- After warmup, sampling is weighted by historical training value:
  `training_value = max(0.05, 1 - 2 * abs(pass_rate - 0.5))`.
- Low-value profiles are soft downweighted only. They are not permanently deleted.

## Removed From The Old 3-Difficulty Version

- No `build_sead3_profiles.py` invocation at launch.
- No `train_profile_sead3_1500.jsonl`.
- No `diff_text`, `easy/medium/hard`, or difficulty prompt injection.
- No `full_sample_id` grouping.

## Run

```bash
export PLAYER_API_BASE=https://your-simulator-endpoint.example/v1
export PLAYER_API_KEY=YOUR_KEY
export TRAIN_MODEL_PATH=/path/to/Qwen3-8B
export OUTPUT_DIR=/path/to/training/output
bash launch_sead_train500_n4_dlc.sh
```

The launch filename retains `sead` for compatibility with earlier server
commands. The profile data is SAGE-style; there are no SEAD difficulty labels
or `easy/medium/hard` prompt additions in this package.

## Important Experiment Invariants

- `VIRTUAL_DATASET_SIZE=500`
- `BATCH_SIZE=8`
- `GRPO_N=4`
- `MAX_RESPONSE_LEN=7000`
- `PER_TURN_LEN=7000`
- `PLAYER_MAX_WORKERS=16`
- `SEAD_PASSCTRL_WARMUP_STEPS=100`
- `DUOROLE_SUCCESS_THRESHOLD=50`
- `DUOROLE_COUNSELOR_AUX_WEIGHT=0.08`
- The four rollouts for one group share the same profile `id`.
- The assistant never receives profile-controller metadata.

## Reproducing The Profile Pool

The generation utility is included as
`scripts/generate_sage_variants_500.py`. It reads API credentials from
`IDEALAB_API_KEY`; no key is stored in the package.
