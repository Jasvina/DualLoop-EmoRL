#!/bin/bash
# RLVER-SAGEVariant pass-rate controller DLC train500 (SAGE-style variants, N=4)
set -e
set -u

TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:?Set TRAIN_MODEL_PATH to the base model directory}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the training output directory}"
RUN_NAME="rlver_sagevar_passctrl_dsv3user_train500_n4"
CKPT_DIR="${OUTPUT_DIR}/${RUN_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/code"

RLVER_VLLM_MEMORY_UTIL=0.45
PLAYER_API_BASE="${PLAYER_API_BASE:?Set PLAYER_API_BASE to the simulator API endpoint}"
PLAYER_API_KEY="${PLAYER_API_KEY:?Set PLAYER_API_KEY before launching training}"
PLAYER_MODEL_NAME="${PLAYER_MODEL_NAME:-deepseek-v3}"
SCORER_MODEL_NAME="${SCORER_MODEL_NAME:-deepseek-v3}"

TOTAL_TRAINING_STEPS=500
BATCH_SIZE=8
GRPO_N=4
LR=1e-6
ACTOR_WARMUP_RATIO=0.02
TEMPERATURE=1.0
KL_COEF=0.001
KL_LOSS_COEF=0.001
KL_LOSS_TYPE="low_var_kl"
PPO_EPOCHS=1

MAX_PROMPT_LEN=25000
MAX_RESPONSE_LEN=7000
PPO_MAX_TOKEN_LEN=40000
PER_TURN_LEN=7000
MAX_TURNS=8

VIRTUAL_DATASET_SIZE=500
VAL_VIRTUAL_DATASET_SIZE=25

TRAINING_GPUS="0,1,2,3"
N_GPUS=4
NNODES=1
IF_THINK=False
SAVE_FREQ=5
PLAYER_MAX_WORKERS=16
SIMULATOR_TYPE=sage
SAGE_EMO_INIT=30
SAGE_CHANGE_CLAMP_DISABLE=1

RAY_PORT=6396
RAY_DASHBOARD_PORT=8276
RAY_TEMP_DIR=/tmp/ray_sead_passctrl_n4

# SEAD pass-rate controller specific
DUOROLE_USER_LAMBDA=0.0
DUOROLE_COUNSELOR_AUX_WEIGHT=0.08
DUOROLE_API_WORKERS=32
DUOROLE_SUCCESS_THRESHOLD=50
SEAD_PROFILE_SEED=2026
SEAD_PROFILE_POOL_SEED=20260701
SEAD_BEHAVIOR_COMBOS_PER_STATE=20
SEAD_PROFILE_MIN_COUNT=2
SEAD_STATS_KEY_LEVEL=id
SEAD_PASSCTRL_WARMUP_STEPS=100
LOCAL_PROFILE_GROUPS_PER_STEP=$(((BATCH_SIZE + N_GPUS - 1) / N_GPUS))
SEAD_PASSCTRL_WARMUP_GROUPS=$((SEAD_PASSCTRL_WARMUP_STEPS * LOCAL_PROFILE_GROUPS_PER_STEP))
SEAD_PASSCTRL_MIN_SCORE=0.05
SEAD_PROFILE_POOL_SIZE=500
DUOROLE_LAZY_SIMULATOR_INIT=1

export CUDA_VISIBLE_DEVICES=${TRAINING_GPUS}
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export HYDRA_FULL_ERROR=1
export RAY_record_ref_creation_sites=1
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export RLVER_OUTPUT_DIR="${OUTPUT_DIR}"
export RLVER_BASE_DIR="${SCRIPT_DIR}"
export PLAYER_API_BASE PLAYER_API_KEY PLAYER_MODEL_NAME SCORER_MODEL_NAME
export PLAYER_MAX_WORKERS
export IF_THINK
export SIMULATOR_TYPE
export SAGE_EMO_INIT SAGE_CHANGE_CLAMP_DISABLE
export DUOROLE_USER_LAMBDA DUOROLE_COUNSELOR_AUX_WEIGHT DUOROLE_SUCCESS_THRESHOLD DUOROLE_API_WORKERS
export DUOROLE_LAZY_SIMULATOR_INIT
export SEAD_PROFILE_SEED
export SEAD_PROFILE_POOL_SEED SEAD_BEHAVIOR_COMBOS_PER_STATE SEAD_PROFILE_MIN_COUNT
export SEAD_STATS_KEY_LEVEL SEAD_PASSCTRL_WARMUP_STEPS SEAD_PASSCTRL_WARMUP_GROUPS SEAD_PASSCTRL_MIN_SCORE SEAD_PROFILE_POOL_SIZE

export PATH="/dev/shm/sage_env/bin:$PATH"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

SCRIPT_START=$(date +%s)
log() { echo "[$(date +%H:%M:%S)] $*"; }
elapsed() { local now=$(date +%s); local secs=$((now - SCRIPT_START)); printf "%dm%02ds" $((secs/60)) $((secs%60)); }
cleanup() { log "Cleanup..."; pkill -9 -f "gcs-address=.*:${RAY_PORT}" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "${CKPT_DIR}"
mkdir -p "${SCRIPT_DIR}/data"

SEAD_PROFILE_FILE="${SCRIPT_DIR}/data/train_profile_sage_variants_500.jsonl"
if [[ ! -s "${SEAD_PROFILE_FILE}" ]]; then
    echo "[ERROR] missing generated SAGE-variant profile file: ${SEAD_PROFILE_FILE}"
    echo "Please copy the generated 500-line train_profile_sage_variants_500.jsonl there before running."
    exit 1
fi
export SEAD_PROFILE_FILE

echo "┌──────────────────────────────────────────────┐"
echo "│  RLVER-SAGEVariant PassCtrl train500 N=4     │"
echo "│  Data: 500 SAGE-style generated variants     │"
echo "│  Sample-level controller                                 │"
echo "│  Warmup: full-pool uniform 100 steps   │"
echo "│  Counselor reward: emotion + small aux       │"
echo "└──────────────────────────────────────────────┘"

cd "${CODE_DIR}"

log "[1/2] Starting Ray..."
pkill -9 -f "gcs-address=.*:${RAY_PORT}" 2>/dev/null || true
sleep 2
ray start --head --port=${RAY_PORT} --num-gpus=${N_GPUS} --temp-dir=${RAY_TEMP_DIR} --dashboard-port=${RAY_DASHBOARD_PORT} > /dev/null 2>&1
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
log "  Ray ready [$(elapsed)]"

log "[2/2] SAGEVariant-PassCtrl GRPO training ${TOTAL_TRAINING_STEPS} steps..."
python3 -m verl.trainer.main_ppo_duorole_sr_v2 \
    +data.virtual_dataset_size=${VIRTUAL_DATASET_SIZE} \
    +data.val_virtual_dataset_size=${VAL_VIRTUAL_DATASET_SIZE} \
    data.prompt_key=prompt \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.return_raw_chat=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    actor_rollout_ref.thinking=${IF_THINK} \
    actor_rollout_ref.model.path=${TRAIN_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${ACTOR_WARMUP_RATIO} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.use_loss_generation_mask=True \
    actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
    +actor_rollout_ref.rollout.environment.name=url_environment \
    +actor_rollout_ref.rollout.environment.per_turn_length=${PER_TURN_LEN} \
    +actor_rollout_ref.rollout.environment.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=${GRPO_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${RLVER_VLLM_MEMORY_UTIL} \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
    trainer.project_name=rlver_sagevar_passctrl_train \
    trainer.experiment_name=${RUN_NAME} \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.logger="['console']" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.save_rollout=True \
    trainer.test_freq=0 \
    trainer.total_epochs=999999 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.resume_mode=disable \
    2>&1 | tee -a "${CKPT_DIR}/train_dlc_$(date +%Y%m%d_%H%M%S).log"

log "Done [$(elapsed)]"
