#!/usr/bin/env bash
# Dual-loop self-evolution, Qwen3-8B, GRPO N=4.
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_CODE_DIR="${VERL_CODE_DIR:?Set VERL_CODE_DIR to a compatible VERL training runtime}"
TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:?Set TRAIN_MODEL_PATH to the base model directory}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the experiment output directory}"
RUN_NAME="${RUN_NAME:-dual_loop_qwen3_8b_train500_n4}"
CKPT_DIR="${OUTPUT_DIR}/${RUN_NAME}"

PLAYER_API_KEY="${PLAYER_API_KEY:?Set PLAYER_API_KEY for the user simulator/verifier service}"
PLAYER_API_BASE="${PLAYER_API_BASE:?Set PLAYER_API_BASE for the user simulator/verifier service}"
PLAYER_MODEL_NAME="${PLAYER_MODEL_NAME:-deepseek-v3}"
SCORER_MODEL_NAME="${SCORER_MODEL_NAME:-deepseek-v3}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRPO_N="${GRPO_N:-4}"
LR="${LR:-3e-7}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-25000}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-6000}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-32000}"
PER_TURN_LEN="${PER_TURN_LEN:-6000}"
MAX_TURNS="${MAX_TURNS:-8}"
N_GPUS="${N_GPUS:-4}"
TRAINING_GPUS="${TRAINING_GPUS:-0,1,2,3}"
RAY_PORT="${RAY_PORT:-6399}"
SAVE_FREQ="${SAVE_FREQ:-5}"

# Controller units are completed GRPO groups, not optimizer steps.
EGSEC_CONTROLLER_DIR="${EGSEC_CONTROLLER_DIR:-/tmp/egsec_controller_${RUN_NAME}}"
mkdir -p "${EGSEC_CONTROLLER_DIR}"
export EGSEC_CONTROLLER_STATE_FILE="${EGSEC_CONTROLLER_STATE_FILE:-${EGSEC_CONTROLLER_DIR}/egsec_controller.sqlite3}"
# 16 scenario groups/optimizer step x 100 initialization steps = 1600 groups.
export EGSEC_WARMUP_GROUPS="${EGSEC_WARMUP_GROUPS:-1600}"
export EGSEC_SUCCESS_THRESHOLD="${EGSEC_SUCCESS_THRESHOLD:-50}"
export EGSEC_HIERARCHICAL_SHRINKAGE="${EGSEC_HIERARCHICAL_SHRINKAGE:-4}"
export EGSEC_STATE_PRIOR="${EGSEC_STATE_PRIOR:-4}"
export EGSEC_UNCERTAINTY_WEIGHT="${EGSEC_UNCERTAINTY_WEIGHT:-0.15}"
export EGSEC_UNIFORM_MIX="${EGSEC_UNIFORM_MIX:-0.10}"
export EGSEC_SAMPLING_TEMPERATURE="${EGSEC_SAMPLING_TEMPERATURE:-1.0}"
export EGSEC_MIN_SCORE="${EGSEC_MIN_SCORE:-0.05}"
export EGSEC_GROUP_SIZE="${EGSEC_GROUP_SIZE:-${GRPO_N}}"

export CUDA_VISIBLE_DEVICES="${TRAINING_GPUS}"
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true NCCL_DEBUG=WARN
export RLVER_OUTPUT_DIR="${CKPT_DIR}"
export RLVER_BASE_DIR="${PACKAGE_DIR}"
export PLAYER_API_BASE PLAYER_API_KEY PLAYER_MODEL_NAME SCORER_MODEL_NAME
export PLAYER_MAX_WORKERS="${PLAYER_MAX_WORKERS:-64}"
export DUOROLE_API_WORKERS="${DUOROLE_API_WORKERS:-32}"
export DUOROLE_USER_LAMBDA=0.0 DUOROLE_SUCCESS_THRESHOLD=50
export SIMULATOR_TYPE=sage SAGE_EMO_INIT=30 SAGE_CHANGE_CLAMP_DISABLE=1
export SEAD_PROFILE_FILE="${PACKAGE_DIR}/data/train_scenarios.jsonl"
export EGSEC_SCENARIO_LIMIT=500
export SEAD_ALLOW_EXTREME=0 SEAD_PROFILE_SEED=2026
export SEAD_PROFILE_POOL_SEED=20260701
export PYTHONPATH="${VERL_CODE_DIR}:${PACKAGE_DIR}/method:${PYTHONPATH:-}"

mkdir -p "${CKPT_DIR}/controller"
python3 "${PACKAGE_DIR}/scripts/validate_training_scenarios.py" \
  --input "${PACKAGE_DIR}/data/train_scenarios.jsonl" --expected-rows 500 --expected-intents 8
cd "${VERL_CODE_DIR}"
ray start --head --port="${RAY_PORT}" --num-gpus="${N_GPUS}" \
  --temp-dir="${RAY_TMPDIR:-/tmp/ray_egsec}" --dashboard-port="${RAY_DASHBOARD_PORT:-8279}" >/dev/null
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

python3 -m verl.trainer.main_ppo_duorole_sr_v2 \
  +data.virtual_dataset_size=500 +data.val_virtual_dataset_size=25 \
  data.prompt_key=prompt data.train_batch_size="${BATCH_SIZE}" data.val_batch_size="${BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.return_raw_chat=True algorithm.adv_estimator=grpo algorithm.kl_ctrl.kl_coef=0.01 \
  actor_rollout_ref.thinking=False actor_rollout_ref.model.path="${TRAIN_MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
  actor_rollout_ref.actor.grad_clip=10.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  +actor_rollout_ref.actor.use_loss_generation_mask=True \
  actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
  +actor_rollout_ref.rollout.environment.name=url_environment \
  +actor_rollout_ref.rollout.environment.per_turn_length="${PER_TURN_LEN}" \
  +actor_rollout_ref.rollout.environment.max_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.n="${GRPO_N}" actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.disable_log_stats=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN}" \
  trainer.project_name=dual_loop_self_evolution trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${CKPT_DIR}" trainer.logger="['console']" \
  +trainer.val_before_train=False trainer.n_gpus_per_node="${N_GPUS}" trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" trainer.save_rollout=True trainer.test_freq=0 \
  trainer.total_epochs=999999 trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.resume_mode=auto 2>&1 | tee -a "${CKPT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
