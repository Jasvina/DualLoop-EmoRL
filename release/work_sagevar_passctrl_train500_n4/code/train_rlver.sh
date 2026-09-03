
set -e
set -u

WANDB_TOKEN=YOUR_WANDB_TOKEN
RUN_NAME=YOUR_RUN_NAME
IF_THINK=False
export RAY_record_ref_creation_sites=1
export HYDRA_FULL_ERROR=1
pip install torchdata
mkdir -p YOUR_DIR_TO_SAVE_CKPTS
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
    "env_vars": {
   "YOUR_ENV_VARS"
    },
    "working_dir": "./",
    "pip": ["latex2sympy2", "word2number", "timeout_decorator"]
    }' -- PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    data.virtual_dataset_size=32000 \
    data.val_virtual_dataset_size=320 \
    data.prompt_key=prompt \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=5000 \
    data.max_response_length=10000 \
    data.return_raw_chat=True \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.thinking=$IF_THINK \
    actor_rollout_ref.model.path=YOUR_BASE_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.use_loss_generation_mask=True \
    actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
    +actor_rollout_ref.rollout.environment.name=url_environment \
    +actor_rollout_ref.rollout.environment.per_turn_length=5000 \
    +actor_rollout_ref.rollout.environment.max_turns=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48000 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48000 \
    critic.optim.lr=5e-6 \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="YOUR_CRITIC_MODEL_PATH" \
    critic.ppo_mini_batch_size=32 \
    critic.model.use_remove_padding=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.ppo_max_token_len_per_gpu=48000 \
    critic.forward_max_token_len_per_gpu=48000 \
    trainer.project_name=verl \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir="YOUR_DIR_TO_SAVE_CKPTS" \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=5 \
    trainer.save_rollout=True \
    trainer.test_freq=999999 \
    trainer.total_epochs=999999 \
    trainer.total_training_steps=1000 \
    2>&1 | tee -a "YOUR_DIR_TO_SAVE_CKPTS/train.log"
