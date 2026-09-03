# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import json
import uuid
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from tqdm import tqdm

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from tensordict import TensorDict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.workers.rollout.vllm_rollout.scene_generator import SceneGenerator
from verl.utils.dataset.rl_dataset_selfplay_v1 import RLHFDataset, collate_fn
# from verl.utils.dialogue.dialogue_client import DialogueClient  # removed: module not present
from verl.utils.torch_functional import get_eos_mask
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    generation_mask = data.batch['generation_mask']
    response_mask = (attention_mask * generation_mask)[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        generation_mask = data.batch['generation_mask']
        response_mask = (attention_mask * generation_mask)[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        print("uid:",index)
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = (batch.batch['attention_mask']*batch.batch['generation_mask'])[:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    sequence_original_reward = batch.batch['original_reward'].sum(-1)
    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']
    
    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    def _safe_stat(t, fn):
        if t.numel() == 0:
            return 0.0
        return fn(t).detach().item()

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    uid = batch.non_tensor_batch['uid']
    sequence_reward_std = []
    for uid_ in uid:
        inds = torch.tensor((uid == uid_), dtype=torch.bool)
        sequence_reward_ = sequence_reward[inds]
        sequence_reward_std.append(sequence_reward_.std())
    sequence_reward_std = torch.stack(sequence_reward_std)
    print(f'max response_length:{torch.max(response_length).detach().item()}')

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/original_rewards/mean':
            torch.mean(sequence_original_reward).detach().item(),
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        'critic/rewards/std':
            torch.std(sequence_reward).detach().item(),
        'critic/rewards/per_data_std':
            torch.mean(sequence_reward_std).detach().item(),
        'critic/rewards/positive_ratio':
            torch.mean((sequence_original_reward>0.4).float()).detach().item(),
        'critic/rewards/success':
            torch.mean((sequence_original_reward>=1).float()).detach().item(),
        'critic/rewards/failure':
            torch.mean((sequence_original_reward<0.1).float()).detach().item(),
        # adv
        'critic/advantages/mean':
            _safe_stat(valid_adv, torch.mean),
        'critic/advantages/max':
            _safe_stat(valid_adv, torch.max),
        'critic/advantages/min':
            _safe_stat(valid_adv, torch.min),
        # returns
        'critic/returns/mean':
            _safe_stat(valid_returns, torch.mean),
        'critic/returns/max':
            _safe_stat(valid_returns, torch.max),
        'critic/returns/min':
            _safe_stat(valid_returns, torch.min),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
            if num_tokens_of_section[name] > 0
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """


    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 environment=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config

        self.environment = environment
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        assert not (self.config.algorithm.adv_estimator != 'grpo' and self.config.trainer.trajectory_injection), "trajectory_injection is only supported for GRPO"

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        
        # 检查是否使用虚拟数据集模式
        use_virtual_dataset = True
        
        if use_virtual_dataset:
            # 使用虚拟数据集，不依赖实际文件
            from verl.utils.dataset.rl_dataset import VirtualRLHFDataset
            
            # 计算虚拟数据集大小：基于总训练步数和batch size
            virtual_size = self.config.data.get('virtual_dataset_size', 
                                              self.config.trainer.total_training_steps * self.config.data.train_batch_size)
            
            print(f"Using virtual dataset with size: {virtual_size}")
            
            self.train_dataset = VirtualRLHFDataset(virtual_size=virtual_size,
                                                   tokenizer=self.tokenizer,
                                                   prompt_key=self.config.data.prompt_key,
                                                   response_key=self.config.data.response_key,
                                                   max_prompt_length=self.config.data.max_prompt_length,
                                                   max_response_length=self.config.data.max_response_length,
                                                   return_raw_chat=self.config.data.get('return_raw_chat', False),
                                                   truncation='error',
                                                   trajectory_injection=self.config.trainer.trajectory_injection)
        else:
            # 使用传统的RLHFDataset
            print("Using traditional RLHFDataset with parquet files")
            self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                           tokenizer=self.tokenizer,
                                           prompt_key=self.config.data.prompt_key,
                                           response_key=self.config.data.response_key,
                                           max_prompt_length=self.config.data.max_prompt_length,
                                           max_response_length=self.config.data.max_response_length,
                                           filter_prompts=True,
                                           return_raw_chat=self.config.data.get('return_raw_chat', False),
                                           truncation='error',
                                           trajectory_injection=self.config.trainer.trajectory_injection)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        if use_virtual_dataset:
            # 验证数据集也使用虚拟模式，但通常较小
            val_virtual_size = self.config.data.get('val_virtual_dataset_size', 
                                                  self.config.data.get('val_batch_size', 32) * 10)  # 默认10个批次用于验证
            
            print(f"Using virtual validation dataset with size: {val_virtual_size}")
            
            self.val_dataset = VirtualRLHFDataset(virtual_size=val_virtual_size,
                                                 tokenizer=self.tokenizer,
                                                 prompt_key=self.config.data.prompt_key,
                                                 response_key=self.config.data.response_key,
                                                 max_prompt_length=self.config.data.max_prompt_length,
                                                 max_response_length=self.config.data.max_response_length,
                                                 return_raw_chat=self.config.data.get('return_raw_chat', False),
                                                 truncation='error')
        else:
            self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         response_key=self.config.data.response_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         max_response_length=self.config.data.max_response_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps


    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in tqdm(self.val_dataloader, desc="Validation"):
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys = ['input_ids', 'attention_mask', 'position_ids']
            if self.config.actor_rollout_ref.rollout.name in ['vllm_multi_turn', 'vllm_multi_turn_via_chat']:
                print("使用多轮对话rollout")
                non_tensor_batch_keys = ['raw_prompt']
            else:
                non_tensor_batch_keys = None

            test_gen_batch = test_batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            _,reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(entity_name=self.config.trainer.entity_name,
                          project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          run_id=self.config.trainer.run_id,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # [SELFPLAY_V1] Initialize scene generator if in self-play mode
        self._selfplay_mode = os.environ.get("SELFPLAY_MODE", "") == "true"
        # [V1 dynT PATCH] Default T0=50 (aligns with Qwen3-8B baseline mean=48.03)
        self._selfplay_threshold = int(os.environ.get("SELFPLAY_SUCCESS_THRESHOLD", "50"))
        self._selfplay_skip_count = 0  # consecutive skipped steps counter (legacy, for compat)
        # [V1 dynT PATCH] Dynamic threshold scheduler state + tunables (all env-var driven)
        self._sr_history = []
        self._selfplay_consecutive_skip = 0
        self._selfplay_total_skip = 0
        self._t_floor = int(os.environ.get("SELFPLAY_T_FLOOR", "40"))
        self._t_ceiling = int(os.environ.get("SELFPLAY_T_CEILING", "100"))
        self._sr_target_low = float(os.environ.get("SELFPLAY_SR_TARGET_LOW", "0.35"))
        self._sr_target_high = float(os.environ.get("SELFPLAY_SR_TARGET_HIGH", "0.65"))
        self._sr_window = int(os.environ.get("SELFPLAY_SR_WINDOW", "5"))
        self._t_step = int(os.environ.get("SELFPLAY_T_STEP", "5"))
        if self._selfplay_mode:
            _base_dir = os.environ.get("RLVER_BASE_DIR", "/mnt/data/weiyi/RLVER_new/RLVER")
            _profile_path = os.path.join(_base_dir, "data/train_profile.jsonl")
            self._scene_generator = SceneGenerator(_profile_path, seed=42)
            print(f"[SELFPLAY_V1] Enabled. T0={self._selfplay_threshold}, T_range=[{self._t_floor},{self._t_ceiling}], "
                  f"target_sr=[{self._sr_target_low},{self._sr_target_high}], window={self._sr_window}, "
                  f"profile={_profile_path}")

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            print(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # recompute `start_epoch`
        num_steps_per_epoch = len(self.train_dataloader)
        start_epoch = (self.global_steps - 1) // num_steps_per_epoch
        saved_steps={}

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            tqdm_initial = (self.global_steps - 1) % num_steps_per_epoch
            for batch_dict in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1} / {self.config.trainer.total_epochs}", initial=tqdm_initial):
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys = ['input_ids', 'attention_mask', 'position_ids']
                if self.config.actor_rollout_ref.rollout.name in ['vllm_multi_turn', 'vllm_multi_turn_via_chat']:
                    non_tensor_batch_keys = ["raw_prompt","simulator"]
                else:
                    non_tensor_batch_keys = None
                gen_batch = batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        # [SELFPLAY_V1] Dynamic scene generation
                        if self._selfplay_mode:
                            _batch_size = self.config.data.train_batch_size
                            _gen_msgs = self._scene_generator.build_generation_messages(_batch_size * 3)
                            # Use DashScope API (same as player simulator) for scene generation
                            import requests as _requests
                            _api_base = os.environ.get("PLAYER_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                            _api_key = os.environ.get("PLAYER_API_KEY", "EMPTY")
                            _api_model = os.environ.get("PLAYER_MODEL_NAME", "deepseek-v3")
                            _gen_texts = []
                            for _msg in _gen_msgs:
                                try:
                                    _resp = _requests.post(
                                        f"{_api_base}/chat/completions",
                                        json={"model": _api_model, "messages": _msg, "temperature": 1.0, "max_tokens": 512},
                                        headers={"Authorization": f"Bearer {_api_key}", "Content-Type": "application/json"},
                                        timeout=30)
                                    _resp.raise_for_status()
                                    _gen_texts.append(_resp.json()["choices"][0]["message"]["content"])
                                except Exception as _e:
                                    pass
                            _valid_profiles = self._scene_generator.parse_generated_texts(_gen_texts)
                            if len(_valid_profiles) < _batch_size:
                                _fallback = self._scene_generator.get_fallback_profiles(_batch_size - len(_valid_profiles))
                                _valid_profiles.extend(_fallback)
                            _valid_profiles = _valid_profiles[:_batch_size]
                            from verl.workers.rollout.vllm_rollout.sage_player_simulator_selfplay_v1 import PlayerSimulator as SPPlayerSimulator
                            _sim_dir = os.environ.get("RLVER_OUTPUT_DIR", "/tmp") + "/simulator"
                            for _i, _prof in enumerate(_valid_profiles):
                                if _i < len(gen_batch.non_tensor_batch["simulator"]):
                                    gen_batch.non_tensor_batch["simulator"][_i] = SPPlayerSimulator(_sim_dir, external_profile=_prof)
                            print(f"[SELFPLAY_V1] Generated {len(_valid_profiles)} scenes "
                                  f"(valid={self._scene_generator.total_generated}, failed={self._scene_generator.total_failed})")
                        print("generate_sequences start")
                        start_time = time.time()
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        end_time = time.time()
                        if self.config.trainer.save_rollout:
                            rollout_dir = os.path.join(os.environ.get("RLVER_OUTPUT_DIR", "/data/js/DigitalHuman/RLVER/output") + f"/{self.config.trainer.experiment_name}", f'global_step_{self.global_steps}')
                            os.makedirs(rollout_dir, exist_ok=True)  # 确保目录存在
                            with open(os.path.join(rollout_dir, 'rollout.jsonl'), "w", encoding="utf-8") as f:
                                for i in range(len(gen_batch_output)):
                                    f.write(json.dumps(gen_batch_output.non_tensor_batch['messages'][i], ensure_ascii=False) + "\n")
                                    f.write(json.dumps(gen_batch_output.non_tensor_batch['emo_point'][i], ensure_ascii=False) + "\n")
                        print(f"generate_sequences end, time: {end_time - start_time} seconds")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            _,reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    with _timer('reward', timing_raw):
                        print("reward_fn start")
                        start_time = time.time()
                        original_reward_tensor,reward_tensor = self.reward_fn(batch)
                        if self.config.trainer.trajectory_injection:
                            reward_tensor_aggr = reward_tensor.sum(dim=-1)
                            grouped_reward_tensor = reward_tensor_aggr.view(-1, self.config.actor_rollout_ref.rollout.n)
                            grouped_reward_tensor_mean = grouped_reward_tensor.mean(dim=-1)
                            grouped_reward_tensor_argmin = grouped_reward_tensor.argmin(dim=-1)
                            inject_group_id = (grouped_reward_tensor_mean <= self.config.trainer.trajectory_injection_threshold).nonzero()[:, 0]
                            inject_item_id = torch.tensor([
                                igi * self.config.actor_rollout_ref.rollout.n + grouped_reward_tensor_argmin[igi]
                                for igi in inject_group_id
                            ])
                            num_trajectory_injection = inject_item_id.shape[0]

                            if num_trajectory_injection > 0:
                                max_prompt_length = batch.batch['prompts'].shape[-1]
                                max_response_length = batch.batch['responses'].shape[-1]
                                num_of_responses = batch.batch['responses'].shape[0]

                                # inject the ground truth response into the batch
                                batch.batch['responses'][inject_item_id] = batch.batch['gt_response'][inject_item_id]

                                # modify the `input_ids`
                                batch.batch['input_ids'] = torch.cat([batch.batch['prompts'], batch.batch['responses']], dim=-1)

                                # modify the `attention_mask`
                                prompt_attention_mask = batch.batch['attention_mask'][:, :max_prompt_length]
                                response_attention_mask = get_eos_mask(response_id=batch.batch['responses'], eos_token=self.tokenizer.eos_token_id, dtype=batch.batch['attention_mask'].dtype)
                                batch.batch['attention_mask'] = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)

                                # modify the `position_ids`
                                prompt_position_ids = batch.batch['position_ids'][:, :max_prompt_length]
                                delta_position_id = torch.arange(1, max_response_length + 1, device=prompt_position_ids.device).unsqueeze(0).repeat(num_of_responses, 1)
                                response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
                                batch.batch['position_ids'] = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

                                # modify the `reward_tensor`
                                rows = torch.arange(num_of_responses)
                                cols = batch.batch['attention_mask'][:, max_prompt_length:].sum(dim=-1) - 1
                                reward_tensor_aggr[inject_item_id] = self.config.trainer.trajectory_injection_reward
                                reward_tensor[rows, cols] = reward_tensor_aggr

                            metrics['critic/trajectory_injection_num'] = num_trajectory_injection
                            print(f"# Trajectory injection: {num_trajectory_injection}")
                        batch.batch['original_reward'] = original_reward_tensor
                        batch.batch['token_level_scores'] = reward_tensor
                        print(f"reward_fn end, time: {time.time() - start_time} seconds")

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        print("compute_log_prob start")
                        start_time = time.time()
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        end_time = time.time()
                        print(f"compute_log_prob end, time: {end_time - start_time} seconds")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            print("compute_ref_log_prob start")
                            start_time = time.time()
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            end_time = time.time()
                            print(f"compute_ref_log_prob end, time: {end_time - start_time} seconds")
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            print("compute_values start")
                            start_time = time.time()
                            values = self.critic_wg.compute_values(batch)
                            end_time = time.time()
                            print(f"compute_values end, time: {end_time - start_time} seconds")
                            batch = batch.union(values)
                        print("batch.batch['values']shape",batch.batch['values'].shape)
                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            print("apply_kl_penalty start")
                            start_time = time.time()
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            end_time = time.time()
                            print(f"apply_kl_penalty end, time: {end_time - start_time} seconds")
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        print("batch.batch['token_level_rewards']shape",batch.batch['token_level_rewards'].shape)

                        # compute advantages, executed on the driver process
                        # [SELFPLAY_V1] Extreme difficulty filtering
                        if self._selfplay_mode:
                            _n = self.config.actor_rollout_ref.rollout.n
                            _T = self._selfplay_threshold
                            _emo_points = batch.non_tensor_batch["emo_point"]
                            _uids = batch.non_tensor_batch["uid"]
                            _total_batch = len(_uids)
                            # Group by uid (each uid = one scene, n trajectories)
                            from collections import defaultdict as _ddict
                            _uid_indices = _ddict(list)
                            for _idx in range(_total_batch):
                                _uid_indices[_uids[_idx]].append(_idx)
                            _valid_indices = []
                            _filter_fail = 0
                            _filter_success = 0
                            _sr_list = []
                            for _uid, _idxs in _uid_indices.items():
                                _emos = [float(_emo_points[_i]) for _i in _idxs]
                                _success_count = sum(1 for _e in _emos if _e >= _T)
                                _sr = _success_count / len(_idxs)
                                _sr_list.append(_sr)
                                if _sr == 0.0:
                                    _filter_fail += 1
                                    continue
                                if _sr == 1.0:
                                    _filter_success += 1
                                    continue
                                _valid_indices.extend(_idxs)
                            _n_scenes = len(_uid_indices)
                            _n_valid = len(_valid_indices) // _n if _n > 0 else 0
                            _avg_sr = sum(_sr_list) / len(_sr_list) if _sr_list else 0
                            print(f"[SELFPLAY_V1] Filter: total_scenes={_n_scenes}, valid={_n_valid}, "
                                  f"filter_all_fail={_filter_fail}, filter_all_success={_filter_success}, "
                                  f"avg_sr={_avg_sr:.3f}")
                            # [V1 dynT PATCH] Update sliding window of avg_sr for bidirectional T adjustment
                            self._sr_history.append(float(_avg_sr))
                            if len(self._sr_history) > self._sr_window:
                                self._sr_history.pop(0)
                            if len(self._sr_history) >= self._sr_window:
                                _recent_sr = sum(self._sr_history) / len(self._sr_history)
                                _old_T = self._selfplay_threshold
                                if _recent_sr > self._sr_target_high:
                                    self._selfplay_threshold = min(self._t_ceiling, self._selfplay_threshold + self._t_step)
                                elif _recent_sr < self._sr_target_low:
                                    self._selfplay_threshold = max(self._t_floor, self._selfplay_threshold - self._t_step)
                                if self._selfplay_threshold != _old_T:
                                    print(f"[SELFPLAY_V1] T adjusted {_old_T}->{self._selfplay_threshold} "
                                          f"(recent_sr={_recent_sr:.3f} window={self._sr_window})")

                            if _n_valid >= 2:
                                # Keep only valid indices - use reorder to preserve all fields and meta_info
                                _keep = torch.tensor(sorted(_valid_indices), dtype=torch.long)
                                _old_meta = batch.meta_info
                                batch.reorder(_keep)
                                batch.meta_info = _old_meta
                                # [V1 dynT PATCH] Reset consecutive skip counter on real step
                                self._selfplay_consecutive_skip = 0
                                self._selfplay_skip_count = 0  # legacy compat
                            else:
                                # [V1 dynT PATCH] Track consecutive skips (separate from sr-based T adjust)
                                self._selfplay_consecutive_skip += 1
                                self._selfplay_total_skip += 1
                                self._selfplay_skip_count += 1  # legacy compat
                                print(f"[SELFPLAY_V1] Too few valid scenes ({_n_valid}), "
                                      f"skip step (consecutive={self._selfplay_consecutive_skip}, "
                                      f"T={self._selfplay_threshold})")
                                # [V1 dynT PATCH] Hard safety cap: 10 consecutive skips -> model broken
                                _max_consec_skip = int(os.environ.get("SELFPLAY_MAX_CONSECUTIVE_SKIP", "10"))
                                if self._selfplay_consecutive_skip >= _max_consec_skip:
                                    raise RuntimeError(
                                        f"[SELFPLAY_V1] {self._selfplay_consecutive_skip} consecutive skipped steps "
                                        f"(T={self._selfplay_threshold}). Model likely broken (check prior grad NaN). "
                                        f"Aborting to avoid empty-ckpt loop. "
                                        f"Set SELFPLAY_MAX_CONSECUTIVE_SKIP=999 to disable."
                                    )
                                # [P0-3 PATCH] Do NOT increment global_steps on a skipped step.
                                continue
                        print("compute_advantage start")
                        start_time = time.time()
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        end_time = time.time()
                        print(f"compute_advantage end, time: {end_time - start_time} seconds")

                    # update critic
                    if self.use_critic:
                        print("update_critic start")
                        start_time = time.time()
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                        end_time = time.time()
                        print(f"update_critic end, time: {end_time - start_time} seconds")

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        print("update_actor start")
                        start_time = time.time()
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        end_time = time.time()
                        print(f"update_actor end, time: {end_time - start_time} seconds")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                        saved_steps[self.global_steps] = True


                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                print("metrics",metrics)
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return

                self.global_steps += 1