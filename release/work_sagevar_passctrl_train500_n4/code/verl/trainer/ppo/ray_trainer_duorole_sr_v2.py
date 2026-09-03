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
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
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
    response_mask = attention_mask[:, -response_length:]

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
        response_mask = attention_mask[:, -response_length:]
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


def _duorole_sr_value(sr: float) -> float:
    if sr == 0.0 or sr == 1.0:
        return -0.5
    if 0.4 <= sr <= 0.6:
        return 1.0
    if (0.2 < sr < 0.4) or (0.6 < sr < 0.8):
        return 0.5
    return 0.0


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _bell_score(x: float, center: float, width: float) -> float:
    if width <= 1e-6:
        return 0.0
    return float(np.exp(-((x - center) ** 2) / (2.0 * width * width)))


def _duorole_usp_proxy_score(messages, player_profile: str, scene_profile: str, final_emo: float) -> tuple[float, dict]:
    """
    No-API USP proxy inspired by the co-evo document:
    response / emotion / information / fidelity effective score.
    """
    if not isinstance(messages, (list, tuple)):
        return 0.0, {"resp": 0.0, "emo": 0.0, "info": 0.0, "fid": 0.0, "effect": 0.0}

    user_texts = [_safe_text(m.get("content", "")) for m in messages
                  if isinstance(m, dict) and m.get("role") == "user"]
    if not user_texts:
        return 0.0, {"resp": 0.0, "emo": 0.0, "info": 0.0, "fid": 0.0, "effect": 0.0}

    joined_user = " ".join(user_texts)
    all_text = " ".join(_safe_text(m.get("content", "")) for m in messages if isinstance(m, dict))

    n_user = len(user_texts)
    avg_len = float(np.mean([len(t) for t in user_texts])) if user_texts else 0.0
    unique_ratio = len(set(user_texts)) / max(n_user, 1)

    # Response quality: enough turns, reasonable length, low repetition.
    turn_score = 1.0 - float(np.exp(-n_user / 3.0))
    length_score = _bell_score(avg_len, center=45.0, width=35.0)
    repeat_score = 0.5 + 0.5 * unique_ratio
    resp = _clamp01(0.45 * turn_score + 0.35 * length_score + 0.20 * repeat_score)

    # Emotion expression: emotional markers + punctuation intensity.
    emo_markers = [
        "难受", "委屈", "烦", "生气", "崩溃", "害怕", "担心", "想哭", "郁闷", "累",
        "无语", "难过", "失望", "烦死", "别这样", "求你", "受不了", "心累", "失眠",
        "压抑", "糟心"
    ]
    marker_hits = sum(joined_user.count(tok) for tok in emo_markers)
    punct_hits = joined_user.count("!") + joined_user.count("？") + joined_user.count("?") + joined_user.count("…")
    emo = _clamp01(0.55 * np.tanh(marker_hits / max(1.0, n_user)) + 0.45 * np.tanh(punct_hits / max(1.0, n_user)))
    if avg_len > 180:
        emo *= 0.85

    # Information richness: diversity + small anchor overlap.
    token_like = [t for t in joined_user.replace("，", " ").replace("。", " ").replace("！", " ").replace("？", " ").split() if t]
    char_div = len(set(joined_user)) / max(1, len(joined_user))
    token_div = len(set(token_like)) / max(1, len(token_like))
    anchors = []
    for source in (_safe_text(player_profile), _safe_text(scene_profile)):
        for token in source.replace("\n", " ").replace("，", " ").replace("。", " ").split():
            token = token.strip()
            if len(token) >= 2:
                anchors.append(token)
    anchor_tokens = list(set(anchors[:80]))
    if anchors:
        hit = sum(1 for token in anchor_tokens if token in joined_user)
        anchor_match = hit / max(1.0, float(len(anchor_tokens)))
    else:
        anchor_match = 0.25
    info = _clamp01(0.45 * char_div + 0.35 * token_div + 0.20 * anchor_match)

    # Fidelity / naturalness: stay in role and avoid leaking internals.
    bad_markers = [
        "系统", "system", "assistant", "助手", "NPC", "模拟用户", "人物画像",
        "当前困境", "情绪规划", "Response:", "Thinking:", "Origin:", "Change:"
    ]
    leak_count = sum(1 for marker in bad_markers if marker in all_text)
    role_clean = max(0.0, 1.0 - 0.18 * leak_count)
    if unique_ratio < 0.5:
        role_clean *= 0.8
    if "我是助手" in all_text or "我可以帮你" in all_text:
        role_clean *= 0.7
    fidelity = _clamp01(0.55 * anchor_match + 0.45 * role_clean)

    effect = _clamp01(0.30 * resp + 0.30 * emo + 0.20 * info + 0.20 * fidelity)
    reward = (effect - 0.75) * 2.5 + 0.10 * (1.0 - _clamp01(final_emo / 100.0)) + 2.0 * max(0.0, 0.60 - fidelity)
    return float(reward), {
        "resp": float(resp),
        "emo": float(emo),
        "info": float(info),
        "fid": float(fidelity),
        "effect": float(effect),
    }


def _duorole_counselor_aux_score(messages) -> tuple[float, dict]:
    if not isinstance(messages, (list, tuple)):
        return 0.0, {"len": 0.0, "emp": 0.0, "spec": 0.0, "rep": 0.0}

    assistant_texts = [
        _safe_text(m.get("content", ""))
        for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    if not assistant_texts:
        return 0.0, {"len": 0.0, "emp": 0.0, "spec": 0.0, "rep": 0.0}

    avg_len = float(np.mean([len(t) for t in assistant_texts]))
    # Replies that are too short are often empty comfort; too long caused previous instability.
    len_score = _bell_score(avg_len, center=85.0, width=55.0)

    joined = " ".join(assistant_texts)
    empathy_markers = [
        "理解", "听起来", "确实", "难受", "委屈", "不容易", "辛苦", "在意",
        "担心", "害怕", "失望", "压力", "我能感受到", "你不是"
    ]
    emp_hits = sum(1 for marker in empathy_markers if marker in joined)
    emp_score = _clamp01(emp_hits / 4.0)

    question_count = joined.count("？") + joined.count("?")
    concrete_punct = joined.count("，") + joined.count("。") + joined.count("；")
    spec_score = _clamp01(0.55 * np.tanh(concrete_punct / max(1.0, len(assistant_texts) * 4.0))
                          + 0.45 * np.tanh(question_count / max(1.0, len(assistant_texts) * 1.5)))

    unique_ratio = len(set(assistant_texts)) / max(len(assistant_texts), 1)
    rep_score = _clamp01(unique_ratio)

    aux = _clamp01(0.35 * len_score + 0.30 * emp_score + 0.20 * spec_score + 0.15 * rep_score)
    return float(aux), {
        "len": float(len_score),
        "emp": float(emp_score),
        "spec": float(spec_score),
        "rep": float(rep_score),
    }


def _zscore(values: np.ndarray):
    values = values.astype(np.float32)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - float(values.mean())) / (std + 1e-6)


def build_duorole_rewards(data: DataProto, success_threshold: float, user_lambda: float):
    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    generation_mask = data.batch['generation_mask']
    response_mask = (attention_mask * generation_mask)[:, -response_length:]
    user_generation_mask = data.batch['user_generation_mask']
    user_response_mask = (attention_mask * user_generation_mask)[:, -response_length:]

    scene_uids = np.array([str(x) for x in data.non_tensor_batch['scene_uid']], dtype=object)
    trajectory_uids = np.array([str(x) for x in data.non_tensor_batch['trajectory_uid']], dtype=object)
    emo_points = np.array([float(x) for x in data.non_tensor_batch['emo_point']], dtype=np.float32)
    messages_list = data.non_tensor_batch.get('messages', np.array([None] * len(data), dtype=object))
    player_profiles = data.non_tensor_batch.get('player_profile', np.array([""] * len(data), dtype=object))
    scene_profiles = data.non_tensor_batch.get('scene_profile', np.array([""] * len(data), dtype=object))

    trajectory_to_emo = {}
    trajectory_to_scene = {}
    for traj_uid in sorted(set(trajectory_uids.tolist())):
        vals = emo_points[trajectory_uids == traj_uid]
        if len(vals) == 0:
            continue
        trajectory_to_emo[traj_uid] = float(vals[0])
        trajectory_to_scene[traj_uid] = str(scene_uids[trajectory_uids == traj_uid][0])

    scene_to_assistant_emos = {}
    for scene_uid in sorted(set(scene_uids.tolist())):
        vals = [emo for traj_uid, emo in trajectory_to_emo.items()
                if trajectory_to_scene.get(traj_uid) == scene_uid]
        scene_to_assistant_emos[scene_uid] = np.array(vals, dtype=np.float32)

    scene_to_sr = {}
    scene_to_user_reward = {}
    for scene_uid, vals in scene_to_assistant_emos.items():
        if len(vals) == 0:
            scene_to_sr[scene_uid] = 0.0
            scene_to_user_reward[scene_uid] = -0.5
            continue
        sr = float(np.mean(vals >= success_threshold))
        scene_to_sr[scene_uid] = sr
        scene_to_user_reward[scene_uid] = _duorole_sr_value(sr)

    assistant_scalar_scores = torch.zeros(len(data), dtype=torch.float32, device=responses.device)
    user_scalar_scores = torch.zeros(len(data), dtype=torch.float32, device=responses.device)
    user_usp_scores = torch.zeros(len(data), dtype=torch.float32, device=responses.device)
    user_resp_scores = torch.zeros(len(data), dtype=torch.float32, device=responses.device)
    user_emo_scores = torch.zeros(len(data), dtype=torch.float32, device=responses.device)
    user_info_scores = torch.zeros(len(data), dtype=torch.float32, device=responses.device)
    user_fid_scores = torch.zeros(len(data), dtype=torch.float32, device=responses.device)
    counselor_aux_scores = torch.zeros(len(data), dtype=torch.float32, device=responses.device)
    counselor_aux_weight = float(os.environ.get("DUOROLE_COUNSELOR_AUX_WEIGHT", "0.08"))
    for i in range(len(data)):
        emotion_reward = float(emo_points[i]) / 100.0
        counselor_aux, _aux_parts = _duorole_counselor_aux_score(messages_list[i])
        counselor_aux_scores[i] = counselor_aux
        assistant_scalar_scores[i] = (1.0 - counselor_aux_weight) * emotion_reward + counselor_aux_weight * counselor_aux
        usp_reward, usp_parts = _duorole_usp_proxy_score(messages_list[i], player_profiles[i], scene_profiles[i], float(emo_points[i]))
        user_scalar_scores[i] = usp_reward
        user_usp_scores[i] = usp_parts["effect"]
        user_resp_scores[i] = usp_parts["resp"]
        user_emo_scores[i] = usp_parts["emo"]
        user_info_scores[i] = usp_parts["info"]
        user_fid_scores[i] = usp_parts["fid"]

    token_level_scores = torch.zeros_like(responses, dtype=torch.float32, device=responses.device)
    user_token_level_scores = torch.zeros_like(responses, dtype=torch.float32, device=responses.device)
    valid_lengths = response_mask.sum(dim=-1).long().clamp(min=1)
    user_valid_lengths = user_response_mask.sum(dim=-1).long().clamp(min=1)
    for i in range(len(data)):
        token_level_scores[i, valid_lengths[i] - 1] = assistant_scalar_scores[i]
        if user_response_mask[i].sum() > 0:
            user_token_level_scores[i, user_valid_lengths[i] - 1] = user_scalar_scores[i]

    uid = []
    for i in range(len(data)):
        uid.append(str(scene_uids[i]))
    data.non_tensor_batch['uid'] = np.array(uid, dtype=object)
    data.non_tensor_batch['duorole_assistant_reward'] = np.array(
        [float(x) for x in assistant_scalar_scores.cpu().tolist()], dtype=object)
    data.non_tensor_batch['duorole_counselor_aux'] = np.array(
        [float(x) for x in counselor_aux_scores.cpu().tolist()], dtype=object)
    data.non_tensor_batch['duorole_sr'] = np.array([scene_to_sr.get(scene_uids[i], 0.0) for i in range(len(data))],
                                                   dtype=object)
    data.non_tensor_batch['duorole_user_reward'] = np.array(
        [float(x) for x in user_scalar_scores.cpu().tolist()], dtype=object)
    data.non_tensor_batch['duorole_user_usp'] = np.array(
        [float(x) for x in user_usp_scores.cpu().tolist()], dtype=object)
    data.non_tensor_batch['duorole_user_resp'] = np.array(
        [float(x) for x in user_resp_scores.cpu().tolist()], dtype=object)
    data.non_tensor_batch['duorole_user_emo'] = np.array(
        [float(x) for x in user_emo_scores.cpu().tolist()], dtype=object)
    data.non_tensor_batch['duorole_user_info'] = np.array(
        [float(x) for x in user_info_scores.cpu().tolist()], dtype=object)
    data.non_tensor_batch['duorole_user_fid'] = np.array(
        [float(x) for x in user_fid_scores.cpu().tolist()], dtype=object)
    data.batch['original_reward'] = token_level_scores.clone()
    data.batch['token_level_scores'] = token_level_scores
    data.batch['user_token_level_scores'] = user_token_level_scores
    metrics = {
        'duorole/trajectory_rows': float(len(data)),
        'duorole/assistant_mask_tokens': float(response_mask.sum().detach().item()),
        'duorole/final_user_mask_tokens': float(user_response_mask.sum().detach().item()),
        'duorole/assistant_reward_mean': float(np.mean(emo_points / 100.0)) if len(emo_points) else 0.0,
        'duorole/assistant_mixed_reward_mean': float(np.mean(assistant_scalar_scores.detach().cpu().numpy())) if len(assistant_scalar_scores) else 0.0,
        'duorole/counselor_aux_mean': float(np.mean(counselor_aux_scores.detach().cpu().numpy())) if len(counselor_aux_scores) else 0.0,
        'duorole/counselor_aux_weight': float(counselor_aux_weight),
        'duorole/user_reward_mean': float(np.mean(user_scalar_scores.detach().cpu().numpy())) if len(user_scalar_scores) else 0.0,
        'duorole/user_usp_mean': float(np.mean(user_usp_scores.detach().cpu().numpy())) if len(user_usp_scores) else 0.0,
        'duorole/user_resp_mean': float(np.mean(user_resp_scores.detach().cpu().numpy())) if len(user_resp_scores) else 0.0,
        'duorole/user_emo_mean': float(np.mean(user_emo_scores.detach().cpu().numpy())) if len(user_emo_scores) else 0.0,
        'duorole/user_info_mean': float(np.mean(user_info_scores.detach().cpu().numpy())) if len(user_info_scores) else 0.0,
        'duorole/user_fid_mean': float(np.mean(user_fid_scores.detach().cpu().numpy())) if len(user_fid_scores) else 0.0,
        'duorole/sr_mean': float(np.mean(list(scene_to_sr.values()))) if scene_to_sr else 0.0,
        'duorole/sr_min': float(np.min(list(scene_to_sr.values()))) if scene_to_sr else 0.0,
        'duorole/sr_max': float(np.max(list(scene_to_sr.values()))) if scene_to_sr else 0.0,
        'duorole/unique_trajectories': float(len(trajectory_to_emo)),
    }
    if 'profile_difficulty' in data.non_tensor_batch:
        profile_difficulty = np.array([float(x) for x in data.non_tensor_batch['profile_difficulty']], dtype=np.float32)
        profile_success = np.array([float(x) for x in data.non_tensor_batch.get('profile_success', np.zeros(len(data)))], dtype=np.float32)
        profile_target = np.array([float(x) for x in data.non_tensor_batch.get('profile_next_target_difficulty', np.zeros(len(data)))], dtype=np.float32)
        profile_pid_error = np.array([float(x) for x in data.non_tensor_batch.get('profile_next_pid_error', np.zeros(len(data)))], dtype=np.float32)
        profile_cooperation = np.array([float(x) for x in data.non_tensor_batch.get('profile_cooperation', np.zeros(len(data)))], dtype=np.float32)
        profile_emotion = np.array([float(x) for x in data.non_tensor_batch.get('profile_emotion_intensity', np.zeros(len(data)))], dtype=np.float32)
        profile_trust = np.array([float(x) for x in data.non_tensor_batch.get('profile_trust', np.zeros(len(data)))], dtype=np.float32)
        profile_pool_size = np.array([float(x) for x in data.non_tensor_batch.get('profile_pool_size', np.zeros(len(data)))], dtype=np.float32)
        metrics.update({
            'profile/difficulty_mean': float(np.mean(profile_difficulty)) if len(profile_difficulty) else 0.0,
            'profile/difficulty_min': float(np.min(profile_difficulty)) if len(profile_difficulty) else 0.0,
            'profile/difficulty_max': float(np.max(profile_difficulty)) if len(profile_difficulty) else 0.0,
            'profile/success_rate': float(np.mean(profile_success)) if len(profile_success) else 0.0,
            'profile/target_difficulty': float(profile_target[-1]) if len(profile_target) else 0.0,
            'profile/pid_error': float(profile_pid_error[-1]) if len(profile_pid_error) else 0.0,
            'profile/cooperation_mean': float(np.mean(profile_cooperation)) if len(profile_cooperation) else 0.0,
            'profile/emotion_intensity_mean': float(np.mean(profile_emotion)) if len(profile_emotion) else 0.0,
            'profile/trust_mean': float(np.mean(profile_trust)) if len(profile_trust) else 0.0,
            'profile/pool_size': float(profile_pool_size[-1]) if len(profile_pool_size) else 0.0,
        })
        print(
            "[RLVER-SEAD-PassCtrl][reward] "
            f"rows={len(data)} "
            f"assistant_reward_mean={metrics['duorole/assistant_reward_mean']:.4f} "
            f"assistant_mixed_reward_mean={metrics['duorole/assistant_mixed_reward_mean']:.4f} "
            f"counselor_aux_mean={metrics['duorole/counselor_aux_mean']:.4f} "
            f"duorole_sr_mean={metrics['duorole/sr_mean']:.4f} "
            f"profile_success_rate={metrics['profile/success_rate']:.4f} "
            f"profile_target_difficulty={metrics['profile/target_difficulty']:.4f} "
            f"profile_difficulty_mean={metrics['profile/difficulty_mean']:.4f} "
            f"profile_pool_size={metrics['profile/pool_size']:.0f} "
            f"assistant_mask_tokens={metrics['duorole/assistant_mask_tokens']:.0f} "
            f"user_mask_tokens={metrics['duorole/final_user_mask_tokens']:.0f}"
        )
    return data, metrics


def compute_duorole_advantage(data: DataProto, user_lambda: float):
    responses = data.batch['responses']
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    generation_mask = data.batch['generation_mask']
    response_mask = (attention_mask * generation_mask)[:, -response_length:]
    user_generation_mask = data.batch['user_generation_mask']
    user_response_mask = (attention_mask * user_generation_mask)[:, -response_length:]
    scene_uids = np.array([str(x) for x in data.non_tensor_batch['scene_uid']], dtype=object)
    trajectory_uids = np.array([str(x) for x in data.non_tensor_batch['trajectory_uid']], dtype=object)
    assistant_rewards = np.array([float(x) for x in data.non_tensor_batch['duorole_assistant_reward']], dtype=np.float32)
    user_rewards = np.array([float(x) for x in data.non_tensor_batch['duorole_user_reward']], dtype=np.float32)

    advantages = torch.zeros_like(data.batch['token_level_rewards'])
    returns = torch.zeros_like(data.batch['token_level_rewards'])
    user_advantages = torch.zeros_like(data.batch['token_level_rewards'])

    for scene_uid in sorted(set(scene_uids.tolist())):
        scene_trajs = sorted(set(trajectory_uids[scene_uids == scene_uid].tolist()))
        if not scene_trajs:
            continue
        traj_rewards = []
        for traj_uid in scene_trajs:
            idxs = np.where(trajectory_uids == traj_uid)[0]
            traj_rewards.append(float(assistant_rewards[idxs[0]]) if len(idxs) > 0 else 0.0)
        traj_adv = _zscore(np.array(traj_rewards, dtype=np.float32))
        for traj_uid, adv in zip(scene_trajs, traj_adv):
            row_idx_np = np.where(trajectory_uids == traj_uid)[0]
            row_idx = torch.tensor(row_idx_np, dtype=torch.long, device=advantages.device)
            advantages[row_idx] = response_mask[row_idx] * float(adv)
            returns[row_idx] = response_mask[row_idx] * float(adv)

    for scene_uid in sorted(set(scene_uids.tolist())):
        scene_trajs = sorted(set(trajectory_uids[scene_uids == scene_uid].tolist()))
        if not scene_trajs:
            continue
        traj_rewards = []
        for traj_uid in scene_trajs:
            idxs = np.where(trajectory_uids == traj_uid)[0]
            traj_rewards.append(float(user_rewards[idxs[0]]) if len(idxs) > 0 else 0.0)
        traj_adv = _zscore(np.array(traj_rewards, dtype=np.float32))
        for traj_uid, adv in zip(scene_trajs, traj_adv):
            row_idx_np = np.where(trajectory_uids == traj_uid)[0]
            row_idx = torch.tensor(row_idx_np, dtype=torch.long, device=advantages.device)
            user_advantages[row_idx] = user_response_mask[row_idx] * float(adv)

    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
    data.batch['user_advantages'] = user_advantages
    valid_user = torch.masked_select(user_advantages, user_response_mask.bool())
    valid_asst = torch.masked_select(advantages, response_mask.bool())
    metrics = {
        'duorole/user_adv_std': torch.std(valid_user).detach().item() if valid_user.numel() > 1 else 0.0,
        'duorole/assistant_adv_std': torch.std(valid_asst).detach().item() if valid_asst.numel() > 1 else 0.0,
        'duorole/user_lambda': float(user_lambda),
    }
    return data, metrics


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    max_response_length = batch.batch['responses'].shape[-1]

    attention_mask = batch.batch['attention_mask']
    prompt_mask = attention_mask[:, :-max_response_length]
    if 'generation_mask' in batch.batch:
        response_mask = (attention_mask * batch.batch['generation_mask'])[:, -max_response_length:]
    else:
        response_mask = attention_mask[:, -max_response_length:]

    ones_percentage = response_mask.float().mean().item() * 100

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

    attention_mask = batch.batch['attention_mask']
    prompt_mask = attention_mask[:, :-max_response_length].bool()
    if 'generation_mask' in batch.batch:
        response_mask = (attention_mask * batch.batch['generation_mask'])[:, -max_response_length:].bool()
    else:
        response_mask = attention_mask[:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

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
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
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
        'duorole/assistant_response_length_mean':
            torch.mean(response_length).detach().item(),
        'duorole/assistant_response_length_max':
            torch.max(response_length).detach().item(),
        'duorole/assistant_response_length_min':
            torch.min(response_length).detach().item(),
        'duorole/assistant_response_length_clip_ratio':
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
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
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
        
        use_virtual_dataset = True
        
        if use_virtual_dataset:
            from verl.utils.dataset.rl_dataset import VirtualRLHFDataset
            
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
            val_virtual_size = self.config.data.get('val_virtual_dataset_size', 
                                                  self.config.data.get('val_batch_size', 32) * 10)  
            
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

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
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
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        if batch_size % world_size == 0:
            global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                            k_partitions=world_size,
                                                            equal_size=True)
        else:
            print(f"Warning: Batch size {batch_size} is not divisible by world_size {world_size}. Using unequal partitioning.")
            global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                          k_partitions=world_size,
                                                          equal_size=False)
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

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
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
            # 输入prompt，从train_dataloader
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
                # print("non_tensor_batch_keys",non_tensor_batch_keys)
                gen_batch = batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        start_time = time.time()
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        end_time = time.time()
                        print(f"generate_sequences end, time: {end_time - start_time} seconds")
                        if self.config.trainer.save_rollout:
                            import os as _os2, json as _json2
                            _rollout_dir = _os2.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
                            _os2.makedirs(_rollout_dir, exist_ok=True)
                            _rollout_path = _os2.path.join(_rollout_dir, "rollout.jsonl")
                            with open(_rollout_path, "w") as _rf:
                                for _i in range(len(gen_batch_output)):
                                    _msgs = gen_batch_output.non_tensor_batch.get('messages', [])
                                    _emo = gen_batch_output.non_tensor_batch.get('emo_point', [])
                                    _row = {"messages": _msgs[_i] if _i < len(_msgs) else [], "emo_point": float(_emo[_i]) if _i < len(_emo) else 0.0}
                                    _rf.write(_json2.dumps(_row, ensure_ascii=False) + "\n")
                            print(f"[rollout] saved to {_rollout_path}")
                    
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            _,reward_baseline_tensor = self.reward_fn(batch,self.global_steps)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch = gen_batch_output
                    world_size = self.actor_rollout_wg.world_size
                    batch_size = len(batch.batch)
                    if batch_size % world_size != 0:
                        pad_size = world_size - (batch_size % world_size)
                        print(f"Padding batch from size {batch_size} to {batch_size + pad_size} to be divisible by world_size {world_size}")
                        batch, _ = pad_dataproto_to_divisor(batch, world_size)

                    print("batch", batch)
                    with _timer('reward', timing_raw):
                        print("duorole reward start")
                        start_time = time.time()
                        threshold = float(os.environ.get("DUOROLE_SUCCESS_THRESHOLD",
                                                         os.environ.get("SELFPLAY_SUCCESS_THRESHOLD", "50")))
                        user_lambda = float(os.environ.get("DUOROLE_USER_LAMBDA", "0.1"))
                        batch, reward_metrics = build_duorole_rewards(batch,
                                                                       success_threshold=threshold,
                                                                       user_lambda=user_lambda)
                        metrics.update(reward_metrics)
                        print("duorole token_level_scores", batch.batch['token_level_scores'])
                        print(f"duorole reward end, time: {time.time() - start_time} seconds")

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    print("batch.meta_info['global_token_num']",batch.meta_info['global_token_num'])
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

                    with _timer('adv', timing_raw):
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
                        print("compute_duorole_advantage start")
                        start_time = time.time()
                        user_lambda = float(os.environ.get("DUOROLE_USER_LAMBDA", "0.1"))
                        batch, adv_metrics = compute_duorole_advantage(batch, user_lambda=user_lambda)
                        metrics.update(adv_metrics)
                        end_time = time.time()
                        print(f"compute_duorole_advantage end, time: {end_time - start_time} seconds")

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
                # Auto-write metrics to NAS for remote monitoring
                try:
                    import json as _json_metrics
                    _metrics_file = os.path.join(self.config.trainer.default_local_dir, "metrics.jsonl")
                    _metrics_out = {"step": self.global_steps}
                    _metrics_out.update({k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()})
                    with open(_metrics_file, "a") as _f:
                        _f.write(_json_metrics.dumps(_metrics_out, ensure_ascii=False) + "\n")
                except:
                    pass
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
