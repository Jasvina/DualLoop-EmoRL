# Server Runbook: SAGEVariant-PassCtrl

## Use this package

Use `artifacts/sagevar_passctrl_train500_n4_code_with_profiles.zip`. Do not use
older archives containing `3diff`, `sead3`, or `train1500`.

After transfer:

```bash
unzip sagevar_passctrl_train500_n4_code_with_profiles.zip
cd work_sagevar_passctrl_train500_n4
```

The archive already contains the complete runtime code and the 500-row profile
file. It does not require a data-building step.

## Required environment

Set deployment-specific paths and credentials in the shell. Do not edit a key
into the launcher.

```bash
export TRAIN_MODEL_PATH=/path/to/Qwen3-8B
export OUTPUT_DIR=/path/to/rlver-output
export PLAYER_API_BASE=https://your-simulator-endpoint.example/v1
export PLAYER_API_KEY=YOUR_KEY
export PLAYER_MODEL_NAME=deepseek-v3
export SCORER_MODEL_NAME=deepseek-v3
```

Review `TRAINING_GPUS`, Ray ports, and the Python environment path in
`launch_sead_train500_n4_dlc.sh` for the target machine. The launch filename is
retained for server-command compatibility; its data path and active method are
SAGEVariant-PassCtrl.

## Preflight checks

```bash
wc -l data/train_profile_sage_variants_500.jsonl
bash -n launch_sead_train500_n4_dlc.sh
python3 -m py_compile \
  code/verl/workers/rollout/vllm_rollout/sage_player_simulator.py \
  code/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_duorole_sr_v2.py \
  code/verl/trainer/ppo/ray_trainer_duorole_sr_v2.py \
  code/verl/workers/critic/dp_critic.py
```

The row count must be 500. Confirm these launcher values before running:

```text
VIRTUAL_DATASET_SIZE=500
BATCH_SIZE=8
GRPO_N=4
MAX_RESPONSE_LEN=7000
PER_TURN_LEN=7000
PLAYER_MAX_WORKERS=16
SEAD_PASSCTRL_WARMUP_STEPS=100
DUOROLE_SUCCESS_THRESHOLD=50
DUOROLE_COUNSELOR_AUX_WEIGHT=0.08
```

## Launch

```bash
nohup bash launch_sead_train500_n4_dlc.sh \
  > sagevar_passctrl_console.log 2>&1 &
echo $! > sagevar_passctrl.pid
```

Follow progress with:

```bash
tail -f sagevar_passctrl_console.log
```

## Expected behavior

Before step 100, controller mode should report
`warmup_uniform_full_pool`. After the 100-step boundary it should report
`adaptive_sample_score`. The controller seen-profile count should increase,
and score statistics should eventually show non-uniform values while retaining
a minimum near 0.05.

The response-mask fix is active when `response_length/mean` is not pinned to
7,000, `response_length/clip_ratio` is not permanently 1.0, and
`duorole/assistant_response_length_mean` is present. It should approximately
equal `duorole/assistant_mask_tokens / duorole/trajectory_rows`.

## Health checks and stop conditions

At steps 1-10, verify finite reward, loss, KL, gradient norm, and assistant
advantage standard deviation. Stop immediately for NaN/Inf, empty assistant
masks, repeated API failure, or a continuously exploding gradient norm/KL.

At step 100 and the first adaptive steps, verify the mode transition and a
changing sampling-score distribution. If mode stays in warmup, controller step
stays at zero, only one profile is seen, or all scores remain exactly 1.0,
stop and inspect the controller path before spending the rest of the run.

`final_user_mask_tokens=0` and `user_adv_std=0` are expected because this
experiment trains the assistant policy while the SAGE user simulator remains
an environment.

## Resume policy

This archive defaults to `trainer.resume_mode=disable` and does not persist
worker-local controller history. Prefer one uninterrupted run. Do not resume an
adaptive-stage checkpoint unless the server version has a verified
rank-sharded controller-state save/load implementation; otherwise model state
would resume while the profile distribution history resets.
