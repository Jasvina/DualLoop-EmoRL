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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import subprocess
import copy
import json
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length, pad_sequence_to_length, get_final_eos_mask

from verl.utils.py_functional import to_1d_np_array
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

import requests
import time
from concurrent.futures import ThreadPoolExecutor
from verl.workers.rollout.vllm_rollout.system_prompt import *
import numpy as np

import os as _os
if _os.environ.get("SIMULATOR_TYPE", "rlver") == "sage":
    from verl.workers.rollout.vllm_rollout.sage_player_simulator import *
else:
    from verl.workers.rollout.vllm_rollout.hard_player_simulator_dsv3 import *


import os
# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _remove_trailing_pad_tokens(pad_token_id, token_ids: torch.Tensor) -> torch.Tensor:
    # remove the right padding in the token_id
    non_pad_token_locs=torch.nonzero(token_ids != pad_token_id, as_tuple=False)
    assert len(non_pad_token_locs)>0,"No non-pad tokens: "+str(token_ids)
    max_non_pad_token_loc=non_pad_token_locs.max()
    return token_ids[:max_non_pad_token_loc+1]

def _remove_prepending_messages(token_ids: torch.Tensor, message_end_id: int, n_skip_messages: int) -> torch.Tensor:
    # only keep from the n_skip_messages+1 th appearance of message_end_id
    if n_skip_messages==0:
        return token_ids
    message_end_locs=torch.nonzero(token_ids == message_end_id, as_tuple=False)
    if len(message_end_locs)<n_skip_messages:
        assert False,"Not enough messages"
    return token_ids[message_end_locs[n_skip_messages-1]+1:]

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    assert all([t.ndim==1 for t in tensor_list])
    max_len=max([t.size(0) for t in tensor_list])
    padded_tensor_list=[]
    for t in tensor_list:
        padded_tensor_list.append(torch.cat([t,torch.tensor([pad_token_id]*(max_len-t.size(0)),device=t.device,dtype=t.dtype)],dim=0))
    return torch.stack(padded_tensor_list,dim=0)

class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self._sead_profile_states = self._build_sead_profile_states()
        self._sead_profile_rng = np.random.default_rng(int(_os.environ.get("SEAD_PROFILE_SEED", "2026")))
        self._sead_target_sr = 0.5
        self._sead_target_difficulty = 0.0
        self._sead_min_difficulty = 0.0
        self._sead_max_difficulty = 0.0
        self._sead_last_batch_sr = self._sead_target_sr
        self._sead_last_controller_error = 0.0
        self._sead_state_stats = {
            state["state_id"]: {"n": 0, "success": 0}
            for state in self._sead_profile_states
        }

    def _sage_behavior_library(self):
        return {
            "target_sensitivity": [
                "只有助手具体回应自己的真实困境时，才会觉得被理解。",
                "对泛泛的安慰不太买账，会追问对方到底懂不懂。",
                "如果助手没有贴合隐藏主题，会明显冷淡下来。",
                "比较容易接受真诚的共情，但不喜欢空洞鼓励。",
            ],
            "interpretation_bias": [
                "容易把建议理解成说教。",
                "容易觉得对方在敷衍自己。",
                "会反复确认助手是不是认真在听。",
                "愿意把细致、具体的回应理解为善意。",
            ],
            "emotion_volatility": [
                "会出现强烈自责。",
                "容易灾难化地理解问题。",
                "情绪波动明显，可能突然变得很低落。",
                "情绪低落但表达克制，不会夸张宣泄。",
            ],
            "response_attitude": [
                "默认先否定或犹豫，再慢慢补充。",
                "中性观望，需要助手多回应几轮才会松动。",
                "被理解后会逐渐正向回应。",
                "对建议有抵触，容易说“没用”。",
            ],
            "help_seeking_goal": [
                "主要想被听见，而不是立刻要解决方案。",
                "想确认自己是不是做错了。",
                "想要具体、能执行的下一步建议。",
                "想试探助手是否可靠，再决定是否继续透露。",
            ],
            "speaking_style": [
                "回复短，口语化，信息量少。",
                "经常犹豫，表达不完整。",
                "会用反问表达不信任。",
                "一次只透露一点，不主动展开太多。",
            ],
            "disclosure_pace": [
                "不愿一开始透露关键细节。",
                "说到重要处会停顿或收回。",
                "如果被理解，会逐步补充背景。",
                "愿意较快说明具体事件和自己的困扰。",
            ],
        }

    def _sample_behavior_tags_for_state(self, rng, cooperation, emotion_intensity, trust):
        library = self._sage_behavior_library()

        def choose(group, hard_idx=None, easy_idx=None):
            options = library[group]
            if hard_idx is not None and easy_idx is not None:
                if cooperation <= 1 or trust <= 1 or emotion_intensity >= 3:
                    pool = [options[i] for i in hard_idx]
                elif cooperation >= 3 and trust >= 4 and emotion_intensity <= 1:
                    pool = [options[i] for i in easy_idx]
                else:
                    pool = options
            else:
                pool = options
            return pool[int(rng.integers(len(pool)))]

        tags = [
            choose("target_sensitivity", hard_idx=[0, 1, 2], easy_idx=[3]),
            choose("interpretation_bias", hard_idx=[0, 1, 2], easy_idx=[3]),
            choose("emotion_volatility", hard_idx=[0, 1, 2], easy_idx=[3]),
            choose("response_attitude", hard_idx=[0, 1, 3], easy_idx=[2]),
            choose("help_seeking_goal"),
            choose("speaking_style", hard_idx=[0, 1, 2, 3], easy_idx=[3]),
            choose("disclosure_pace", hard_idx=[0, 1], easy_idx=[2, 3]),
        ]
        keep = int(rng.integers(4, min(len(tags), 6) + 1))
        idxs = rng.choice(len(tags), size=keep, replace=False)
        return [tags[int(i)] for i in idxs]

    def _build_sead_profile_states(self):
        states = []
        combos_per_state = int(_os.environ.get("SEAD_BEHAVIOR_COMBOS_PER_STATE", "20"))
        rng = np.random.default_rng(int(_os.environ.get("SEAD_PROFILE_POOL_SEED", "20260701")))
        for cooperation in range(5):
            for emotion_intensity in range(4):
                for trust in range(6):
                    difficulty = float((4 - cooperation) + emotion_intensity + (5 - trust))
                    state_id = f"c{cooperation}_e{emotion_intensity}_t{trust}"
                    for combo_idx in range(combos_per_state):
                        profile_id = f"{state_id}_b{combo_idx}"
                        states.append({
                            "profile_id": profile_id,
                            "state_id": state_id,
                            "cooperation": cooperation,
                            "emotion_intensity": emotion_intensity,
                            "trust": trust,
                            "difficulty": difficulty,
                            "behavior_tags": self._sample_behavior_tags_for_state(
                                rng, cooperation, emotion_intensity, trust),
                        })
        return states

    def _sample_sead_profile_state_disabled(self):
        raise RuntimeError("Disabled: sampling is handled by sample_profile_with_controller(profile id pool).")


    def _format_sead_profile_state(self, profile_state):
        cooperation = int(profile_state["cooperation"])
        emotion_intensity = int(profile_state["emotion_intensity"])
        trust = int(profile_state["trust"])
        behavior_rules = []
        if cooperation <= 1:
            behavior_rules.append("对建议有抵触，容易说“没用”或先否定。")
        elif cooperation >= 3:
            behavior_rules.append("愿意继续聊，也会尝试回答助手的问题。")
        else:
            behavior_rules.append("会回应助手，但需要被慢慢带动。")
        if emotion_intensity >= 3:
            behavior_rules.append("情绪强烈，表达里会有明显自责、崩溃或灾难化。")
        elif emotion_intensity <= 1:
            behavior_rules.append("情绪低落但表达克制，不会突然大段宣泄。")
        else:
            behavior_rules.append("会自然表达难受，但不要夸张。")
        if trust <= 1:
            behavior_rules.append("不太信任助手，会质疑对方是否真的理解自己。")
        elif trust >= 4:
            behavior_rules.append("比较信任助手，愿意透露更多具体信息。")
        else:
            behavior_rules.append("信任感一般，先观察助手的反应再决定透露多少。")
        for tag in profile_state.get("behavior_tags", []):
            behavior_rules.append(str(tag))
        rules = "\n".join([f"- {r}" for r in behavior_rules])
        return f"""合成用户状态（由 Profile Controller 采样，助手不可见）：
- 合作度：{cooperation}/4
- 情绪强度：{emotion_intensity}/3
- 信任度：{trust}/5
- 难度：{profile_state['difficulty']:.1f}/12

行为要求：
{rules}

请只把这些状态自然体现在说话方式里，不要直接说出合作度、情绪强度、信任度、难度或上述行为标签。"""

    def _update_sead_controller_stats_disabled(self, profile_states, successes):
        return


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)


class vLLMMultiTurnViaChatRollout_think(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.tokenizer = tokenizer
        self.total_length = config.prompt_length + config.response_length
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.environment.per_turn_length
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False


        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        
        #Important:
        #This is multi turn, so we need to set n=1 for sampling params, as we will manually batch n since some samplings might terminate earlier.
        kwargs['n']=1

        #print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self._sead_profile_states = []
        self._sead_profile_rng = np.random.default_rng(int(_os.environ.get("SEAD_PROFILE_SEED", "2026")))
        self._sead_target_sr = 0.5
        self._sead_target_difficulty = 0.0
        self._sead_last_batch_sr = self._sead_target_sr
        self._sead_last_controller_error = 0.0
        self._sead_rollout_step = 0
        self._sead_last_controller_mode = "init"
        self._sead_stats_key_level = "id"
        self._sead_state_stats = {}
        print(
            "[RLVER-SEAD-PassCtrl][init] "
            f"user_source=DSV3_API assistant_source=Qwen_vLLM scorer=DSV3_SAGE "
            f"profile_file={_os.environ.get('SEAD_PROFILE_FILE', '')} "
            f"warmup_steps={_os.environ.get('SEAD_PASSCTRL_WARMUP_STEPS', '150')} "
            f"target_pass_rate={self._sead_target_sr:.3f} "
            f"stats_key_level={self._sead_stats_key_level}"
        )

    def _sage_behavior_library(self):
        return {
            "target_sensitivity": [
                "只有助手具体回应自己的真实困境时，才会觉得被理解。",
                "对泛泛的安慰不太买账，会追问对方到底懂不懂。",
                "如果助手没有贴合隐藏主题，会明显冷淡下来。",
                "比较容易接受真诚的共情，但不喜欢空洞鼓励。",
            ],
            "interpretation_bias": [
                "容易把建议理解成说教。",
                "容易觉得对方在敷衍自己。",
                "会反复确认助手是不是认真在听。",
                "愿意把细致、具体的回应理解为善意。",
            ],
            "emotion_volatility": [
                "会出现强烈自责。",
                "容易灾难化地理解问题。",
                "情绪波动明显，可能突然变得很低落。",
                "情绪低落但表达克制，不会夸张宣泄。",
            ],
            "response_attitude": [
                "默认先否定或犹豫，再慢慢补充。",
                "中性观望，需要助手多回应几轮才会松动。",
                "被理解后会逐渐正向回应。",
                "对建议有抵触，容易说“没用”。",
            ],
            "help_seeking_goal": [
                "主要想被听见，而不是立刻要解决方案。",
                "想确认自己是不是做错了。",
                "想要具体、能执行的下一步建议。",
                "想试探助手是否可靠，再决定是否继续透露。",
            ],
            "speaking_style": [
                "回复短，口语化，信息量少。",
                "经常犹豫，表达不完整。",
                "会用反问表达不信任。",
                "一次只透露一点，不主动展开太多。",
            ],
            "disclosure_pace": [
                "不愿一开始透露关键细节。",
                "说到重要处会停顿或收回。",
                "如果被理解，会逐步补充背景。",
                "愿意较快说明具体事件和自己的困扰。",
            ],
        }

    def _sample_behavior_tags_for_state(self, rng, cooperation, emotion_intensity, trust):
        library = self._sage_behavior_library()

        def choose(group, hard_idx=None, easy_idx=None):
            options = library[group]
            if hard_idx is not None and easy_idx is not None:
                if cooperation <= 1 or trust <= 1 or emotion_intensity >= 3:
                    pool = [options[i] for i in hard_idx]
                elif cooperation >= 3 and trust >= 4 and emotion_intensity <= 1:
                    pool = [options[i] for i in easy_idx]
                else:
                    pool = options
            else:
                pool = options
            return pool[int(rng.integers(len(pool)))]

        tags = [
            choose("target_sensitivity", hard_idx=[0, 1, 2], easy_idx=[3]),
            choose("interpretation_bias", hard_idx=[0, 1, 2], easy_idx=[3]),
            choose("emotion_volatility", hard_idx=[0, 1, 2], easy_idx=[3]),
            choose("response_attitude", hard_idx=[0, 1, 3], easy_idx=[2]),
            choose("help_seeking_goal"),
            choose("speaking_style", hard_idx=[0, 1, 2, 3], easy_idx=[3]),
            choose("disclosure_pace", hard_idx=[0, 1], easy_idx=[2, 3]),
        ]
        keep = int(rng.integers(4, min(len(tags), 6) + 1))
        idxs = rng.choice(len(tags), size=keep, replace=False)
        return [tags[int(i)] for i in idxs]

    def _build_sead_profile_states(self):
        states = []
        combos_per_state = int(_os.environ.get("SEAD_BEHAVIOR_COMBOS_PER_STATE", "20"))
        rng = np.random.default_rng(int(_os.environ.get("SEAD_PROFILE_POOL_SEED", "20260701")))
        for cooperation in range(5):
            for emotion_intensity in range(4):
                for trust in range(6):
                    difficulty = float((4 - cooperation) + emotion_intensity + (5 - trust))
                    state_id = f"c{cooperation}_e{emotion_intensity}_t{trust}"
                    for combo_idx in range(combos_per_state):
                        profile_id = f"{state_id}_b{combo_idx}"
                        states.append({
                            "profile_id": profile_id,
                            "state_id": state_id,
                            "cooperation": cooperation,
                            "emotion_intensity": emotion_intensity,
                            "trust": trust,
                            "difficulty": difficulty,
                            "behavior_tags": self._sample_behavior_tags_for_state(
                                rng, cooperation, emotion_intensity, trust),
                        })
        return states

    def _sead_stats_key(self, profile_state):
        key_level = getattr(self, "_sead_stats_key_level", "state_id")
        return profile_state.get(key_level, profile_state.get("state_id", profile_state.get("profile_id", "")))

    def _sead_warmup_bucket(self):
        # PassCtrl uses uniform sampling over all 1500 static profiles during
        # SEAD_PASSCTRL_WARMUP_STEPS.  No staged easy/medium curriculum here.
        return None, None, None

    def _sample_from_difficulty_range(self, min_difficulty, max_difficulty):
        candidates = [
            s for s in self._sead_profile_states
            if min_difficulty <= float(s.get("difficulty", 0.0)) <= max_difficulty
        ]
        if not candidates:
            candidates = self._sead_profile_states
        idx = int(self._sead_profile_rng.integers(len(candidates)))
        return candidates[idx]

    def _sead_state_count_summary(self):
        seen = []
        ideal = easy = hard = 0
        for stats in self._sead_state_stats.values():
            n = int(stats.get("n", 0))
            if n <= 0:
                continue
            sr = float(stats.get("success", 0)) / max(float(n), 1.0)
            seen.append(sr)
            if 0.25 <= sr <= 0.45:
                ideal += 1
            elif sr > 0.60:
                easy += 1
            elif sr < 0.20:
                hard += 1
        return len(seen), ideal, easy, hard

    def _sample_sead_profile_state_disabled(self):
        raise RuntimeError("Disabled: sampling is handled by sample_profile_with_controller(profile id pool).")


    def _format_sead_profile_state(self, profile_state):
        cooperation = int(profile_state["cooperation"])
        emotion_intensity = int(profile_state["emotion_intensity"])
        trust = int(profile_state["trust"])
        behavior_rules = []
        if cooperation <= 1:
            behavior_rules.append("对建议有抵触，容易说“没用”或先否定。")
        elif cooperation >= 3:
            behavior_rules.append("愿意继续聊，也会尝试回答助手的问题。")
        else:
            behavior_rules.append("会回应助手，但需要被慢慢带动。")
        if emotion_intensity >= 3:
            behavior_rules.append("情绪强烈，表达里会有明显自责、崩溃或灾难化。")
        elif emotion_intensity <= 1:
            behavior_rules.append("情绪低落但表达克制，不会突然大段宣泄。")
        else:
            behavior_rules.append("会自然表达难受，但不要夸张。")
        if trust <= 1:
            behavior_rules.append("不太信任助手，会质疑对方是否真的理解自己。")
        elif trust >= 4:
            behavior_rules.append("比较信任助手，愿意透露更多具体信息。")
        else:
            behavior_rules.append("信任感一般，先观察助手的反应再决定透露多少。")
        for tag in profile_state.get("behavior_tags", []):
            behavior_rules.append(str(tag))
        rules = "\n".join([f"- {r}" for r in behavior_rules])
        return f"""合成用户状态（由 Profile Controller 采样，助手不可见）：
- 合作度：{cooperation}/4
- 情绪强度：{emotion_intensity}/3
- 信任度：{trust}/5
- 难度：{profile_state['difficulty']:.1f}/12

行为要求：
{rules}

请只把这些状态自然体现在说话方式里，不要直接说出合作度、情绪强度、信任度、难度或上述行为标签。"""

    def _update_sead_controller_stats_disabled(self, profile_states, successes):
        return


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def get_n_tokens(self,prompt,add_generation_prompt=False):
        return len(self.tokenizer.apply_chat_template(prompt,
                                                      tokenize=True,
                                                      add_generation_prompt=add_generation_prompt,
                                                      enable_thinking=False,
                                                      return_dict=False))
    
    def tokenize_with_role_masks(self, messages):
        n_messages=len(messages)
        tokenized_messages=self.tokenizer.apply_chat_template(messages,
                                                              tokenize=True,
                                                              add_generation_prompt=False,
                                                              enable_thinking=False,
                                                              return_dict=False)
        head=0
        assistant_mask=[]
        user_mask=[]
        for i_last_message in range(n_messages):
            if (i_last_message!=n_messages-1) and (messages[i_last_message+1]["role"]=="assistant"):
                is_next_assistant=True
            else:
                is_next_assistant=False
            last_message_role=messages[i_last_message]["role"]
            n_tokens_with_last_message=self.get_n_tokens(messages[:i_last_message+1],add_generation_prompt=is_next_assistant)
            n_add=n_tokens_with_last_message-head
            if last_message_role=="assistant":
                assistant_mask.append(torch.ones(n_add,dtype=torch.bool))
            else:
                assistant_mask.append(torch.zeros(n_add,dtype=torch.bool))
            if last_message_role=="user":
                user_mask.append(torch.ones(n_add,dtype=torch.bool))
            else:
                user_mask.append(torch.zeros(n_add,dtype=torch.bool))
            head+=n_add
        assistant_mask=torch.cat(assistant_mask,dim=0)
        user_mask=torch.cat(user_mask,dim=0)
        assert len(assistant_mask)==len(tokenized_messages), "Bug: assistant mask length mismatch"
        assert len(user_mask)==len(tokenized_messages), "Bug: user mask length mismatch"
        return tokenized_messages,assistant_mask,user_mask

    def tokenize_with_assistant_mask(self,messages):
        tokenized_messages, assistant_mask, _ = self.tokenize_with_role_masks(messages)
        return tokenized_messages, assistant_mask


    def extract_content(self,content):
        if "</think>" in content:
            extracted_content = content.split("</think>")[-1].strip()
        else:
            extracted_content = content
        if "你：" in extracted_content:
            extracted_content = extracted_content.split("你：")[-1].strip()
        elif "你:" in extracted_content:
            extracted_content = extracted_content.split("你:")[-1].strip()
        return extracted_content

    def _duorole_user_messages(self, simulator, messages, planning, profile_state=None):
        role = simulator.role
        history = role.get("history", [])
        history_text = []
        for msg in history[-8:]:
            name = "用户" if msg["role"] == "user" else "助手"
            history_text.append(f"{name}: {msg['content']}")
        planning_text = ""
        if planning:
            planning_text = planning.get("analyse", "") or planning.get("activity", "") or str(planning)
        else:
            planning_text = "这是对话开始，请用一句简短、自然、符合人设的倾诉开启对话。"
        profile_text = ""
        if profile_state is not None:
            profile_text = "\n\n" + self._format_sead_profile_state(profile_state)
        user_system = f"""你正在扮演一名情感困境中的真实用户。你不是助手，不要安慰别人。

人物画像：
{role.get('player', '')}

当前困境：
{role.get('scene', '')}

当前情绪状态：{getattr(simulator, 'emo_state', '')} / {getattr(simulator, 'emo_point', '')}

上一轮情绪规划：
{planning_text}
{profile_text}

请根据人物画像、当前困境、历史对话和情绪状态，生成下一句用户回复。
要求：自然、简短、口语化；不要说“我是模拟用户”；不要泄露系统设定；不要一次性说完所有隐藏主题。"""
        content = "\n".join(history_text) if history_text else "对话刚开始。"
        return [{"role": "system", "content": user_system}, {"role": "user", "content": content}]

    def _duorole_assistant_messages(self, raw_system_message, history):
        messages = [{"role": raw_system_message.get("role", "system"), "content": raw_system_message.get("content", "")}]
        messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in history])
        return messages

    def _duorole_tokenize_prompt(self, messages, device):
        prompt_text = self.tokenizer.apply_chat_template(messages,
                                                         add_generation_prompt=True,
                                                         tokenize=False,
                                                         enable_thinking=False)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_text,
            tokenizer=self.tokenizer,
            max_length=self.config.prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation='left')
        position_ids = compute_position_id_with_mask(attention_mask)
        return input_ids[0].to(device), attention_mask[0].to(device), position_ids[0].to(device)

    def _duorole_build_trajectory_sample(self, prompt_ids, prompt_attention_mask, prompt_position_ids,
                                         eos_token_id, device, generated_turns=None):
        if not generated_turns:
            return None

        last_user_idx = -1
        for i, (role, _) in enumerate(generated_turns):
            if role == "user":
                last_user_idx = i

        response_tokens = []
        response_asst_mask = []
        response_user_mask = []
        for i, (role, toks) in enumerate(generated_turns):
            response_tokens.extend(toks)
            if role == "assistant":
                response_asst_mask.extend([True] * len(toks))
            else:
                response_asst_mask.extend([False] * len(toks))
            response_user_mask.extend([False] * len(toks))

        assert len(response_tokens) == len(response_asst_mask) == len(response_user_mask), \
            f"mask length mismatch: tokens={len(response_tokens)} asst={len(response_asst_mask)} user={len(response_user_mask)}"

        response_len = len(response_tokens)
        if response_len <= 0:
            print(f"[DuoRole-SR] skip empty response: n_turns={len(generated_turns)}")
            return None

        assistant_mask_tokens = sum(response_asst_mask)
        user_mask_tokens = sum(response_user_mask)
        if response_len > self.config.response_length:
            print(f"[DuoRole-SR] skip overlength traj: response_len={response_len} "
                  f"limit={self.config.response_length} asst_mask={assistant_mask_tokens} user_mask={user_mask_tokens}")
            return None
        if assistant_mask_tokens <= 0:
            print(f"[DuoRole-SR] skip zero-mask traj: response_len={response_len} "
                  f"asst_mask={assistant_mask_tokens} user_mask={user_mask_tokens} "
                  f"n_turns={len(generated_turns)}")
            return None

        response = torch.tensor(response_tokens, device=device, dtype=prompt_ids.dtype)
        response_generation_mask = torch.tensor(response_asst_mask, device=device, dtype=prompt_attention_mask.dtype)
        response_user_mask_t = torch.tensor(response_user_mask, device=device, dtype=prompt_attention_mask.dtype)

        if response.shape[0] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, self.config.response_length, 0)
            response_user_mask_t = pad_sequence_to_length(response_user_mask_t, self.config.response_length, 0)

        response_attention_mask = get_final_eos_mask(response_id=response.unsqueeze(0),
                                                     eos_token=eos_token_id,
                                                     dtype=prompt_attention_mask.dtype)[0]
        response_length = response.size(0)
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        response_position_ids = prompt_position_ids[-1:] + delta_position_id
        input_ids = torch.cat([prompt_ids, response], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
        generation_mask = torch.cat([torch.zeros_like(prompt_attention_mask), response_generation_mask], dim=-1)
        user_generation_mask = torch.cat([torch.zeros_like(prompt_attention_mask), response_user_mask_t], dim=-1)
        return {
            "prompts": prompt_ids,
            "responses": response,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "generation_mask": generation_mask,
            "user_generation_mask": user_generation_mask,
        }
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx=prompts.batch['input_ids']#just for device, size
        batch_size = idx.size(0)
        player_simulators=prompts.non_tensor_batch['simulator']
        
        attention_mask=prompts.batch['attention_mask']
        position_ids=prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        print(f"eos_token_id: {eos_token_id}")
        print(f"self.tokenizer.eos_token_id: {self.tokenizer.eos_token_id}")

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        raw_prompt=prompts.non_tensor_batch['raw_prompt']
        n=1 if prompts.meta_info.get('validate',False) else self.config.n

        samples = []
        non_tensors = []
        device = idx.device
        max_turns = self.config.environment.max_turns
        prompt_ids_by_scene = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]
        prefix_lengths = [len(x) for x in prompt_ids_by_scene]

        trajectories = []
        for scene_idx in range(batch_size):
            raw_prompt_i_raw = raw_prompt[scene_idx]
            raw_prompt_i = []
            if raw_prompt_i_raw is not None:
                for m in raw_prompt_i_raw:
                    if not hasattr(m, "items"):
                        continue
                    d = {}
                    for k, v in m.items():
                        k_s = str(k)
                        if isinstance(v, (list, dict)):
                            d[k_s] = v
                        else:
                            d[k_s] = str(v)
                    raw_prompt_i.append(d)
            raw_system_message = raw_prompt_i[0] if len(raw_prompt_i) > 0 else {"role": "system", "content": ""}
            try:
                player_simulators[scene_idx].role = player_simulators[scene_idx].generate_role("eq")
            except Exception as exc:
                print(f"[RLVER-SEAD-PassCtrl] actor-side profile sample failed at scene={scene_idx}: {exc}")
            for rollout_idx in range(n):
                simulator = player_simulators[scene_idx].clone()
                simulator.role["history"] = []
                role = simulator.role
                profile_id = str(role.get("profile_id", role.get("id", f"scene_{scene_idx}")))
                profile_state = {
                    "id": profile_id,
                    "profile_id": profile_id,
                    "state_id": profile_id,
                    "source_id": str(role.get("source_id", "")),
                    "variant_index": str(role.get("variant_index", "")),
                    "sead_profile_prompt": "",
                    "controller_mode": str(role.get("controller_mode", "")),
                    "cooperation": -1,
                    "emotion_intensity": -1,
                    "trust": -1,
                    "difficulty": 0.0,
                    "behavior_tags": [],
                }
                self._sead_last_controller_mode = profile_state.get("controller_mode") or self._sead_last_controller_mode
                scene_uid = profile_state["profile_id"]
                trajectories.append({
                    "scene_idx": scene_idx,
                    "rollout_idx": rollout_idx,
                    "scene_uid": scene_uid,
                    "trajectory_uid": f"{scene_uid}_rollout_{rollout_idx}",
                    "raw_prompt_messages": list(raw_prompt_i),
                    "raw_system_message": raw_system_message,
                    "simulator": simulator,
                    "planning": {},
                    "last_user_text": "",
                    "dialogue_turns": 0,
                    "done": False,
                    "generated_turns": [],
                    "profile_state": profile_state,
                })

        active = list(range(len(trajectories)))
        for turn_idx in range(max_turns):
            if not active:
                break

            _api_workers = int(_os.environ.get("DUOROLE_API_WORKERS", "1"))

            def _generate_dsv3_user(traj_idx):
                traj = trajectories[traj_idx]
                simulator = traj["simulator"]
                profile_state = traj.get("profile_state") or {}
                simulator.role["sead_profile_prompt"] = profile_state.get("sead_profile_prompt", "")
                try:
                    simulator.role = simulator.player_reply(simulator.role, traj["planning"])
                    user_text = str(simulator.role["history"][-1]["content"]).strip()
                    user_token_ids = self.tokenizer.encode(user_text, add_special_tokens=False)
                    return traj_idx, user_text, user_token_ids, None
                except Exception as exc:
                    return traj_idx, "", [], exc

            if _api_workers > 1 and len(active) > 1:
                with ThreadPoolExecutor(max_workers=_api_workers) as pool:
                    futures = [pool.submit(_generate_dsv3_user, traj_idx) for traj_idx in active]
                    user_results = [f.result() for f in futures]
            else:
                user_results = [_generate_dsv3_user(traj_idx) for traj_idx in active]

            for traj_idx, user_text, user_token_ids, exc in user_results:
                traj = trajectories[traj_idx]
                if exc is not None:
                    print(f"[DuoRole-SR] DSV3 user failed at scene={traj['scene_idx']} rollout={traj['rollout_idx']} turn={turn_idx}: {exc}")
                    traj["done"] = True
                    continue
                if not user_text:
                    print(f"[DuoRole-SR] empty user text at scene={traj['scene_idx']} rollout={traj['rollout_idx']} turn={turn_idx}")
                    traj["done"] = True
                    continue
                traj["last_user_text"] = user_text
                traj["generated_turns"].append(("user", list(user_token_ids)))

            active = [traj_idx for traj_idx in active if not trajectories[traj_idx].get("done", False)]
            if not active:
                break

            assistant_messages_batch = []
            assistant_prompt_parts = []
            for traj_idx in active:
                traj = trajectories[traj_idx]
                simulator = traj["simulator"]
                assistant_messages = self._duorole_assistant_messages(traj["raw_system_message"],
                                                                      simulator.role["history"])
                assistant_prompt_ids, assistant_attention_mask, assistant_position_ids = self._duorole_tokenize_prompt(
                    assistant_messages, device)
                assistant_messages_batch.append(assistant_messages)
                assistant_prompt_parts.append((assistant_prompt_ids, assistant_attention_mask, assistant_position_ids))

            with self.update_sampling_params(**kwargs):
                assert self.sampling_params.n == 1, "n should be 1 for multi-turn"
                assistant_outputs = self.inference_engine.chat(
                    messages=assistant_messages_batch,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})

            next_active = []
            def _process_turn(local_idx, output):
                traj_idx = active[local_idx]
                traj = trajectories[traj_idx]
                simulator = traj["simulator"]
                assistant_token_ids = output.outputs[0].token_ids
                assistant_text = self.tokenizer.decode(assistant_token_ids, skip_special_tokens=True).strip()
                if not assistant_text:
                    print(f"[DuoRole-SR] empty assistant text at scene={traj['scene_idx']} rollout={traj['rollout_idx']} turn={turn_idx}")
                    traj["done"] = True
                    return (traj_idx, False)
                simulator.role["history"].append({"role": "assistant", "content": assistant_text})
                traj["generated_turns"].append(("assistant", list(assistant_token_ids)))

                simulator.role, traj["planning"] = simulator.planning_reply(simulator.role)
                traj["dialogue_turns"] += 1

                user_text = traj.get("last_user_text", "")
                if "再见" in user_text or "拜拜" in user_text or simulator.emo_point <= 0:
                    traj["done"] = True
                    return (traj_idx, False)
                return (traj_idx, True)

            if _api_workers > 1 and len(assistant_outputs) > 1:
                with ThreadPoolExecutor(max_workers=_api_workers) as pool:
                    futures = [pool.submit(_process_turn, i, assistant_outputs[i])
                               for i in range(len(assistant_outputs))]
                    results = [f.result() for f in futures]
            else:
                results = [_process_turn(i, assistant_outputs[i])
                           for i in range(len(assistant_outputs))]

            for traj_idx, is_active in results:
                if is_active:
                    next_active.append(traj_idx)
            active = next_active

        sampled_profile_states = []
        sampled_profile_successes = []
        success_threshold = float(_os.environ.get("DUOROLE_SUCCESS_THRESHOLD", "50"))
        for traj in trajectories:
            if not traj["simulator"].role.get("history"):
                continue
            simulator = traj["simulator"]
            if not any(msg.get("role") == "assistant" for msg in simulator.role["history"]):
                continue
            final_emo = float(simulator.emo_point)
            profile_success = float(final_emo >= success_threshold)
            scene_idx = traj["scene_idx"]
            clean_prompt_ids, clean_prompt_attention_mask, clean_prompt_position_ids = self._duorole_tokenize_prompt(
                [traj["raw_system_message"]], device)
            sample = self._duorole_build_trajectory_sample(
                prompt_ids=clean_prompt_ids,
                prompt_attention_mask=clean_prompt_attention_mask,
                prompt_position_ids=clean_prompt_position_ids,
                eos_token_id=eos_token_id,
                device=device,
                generated_turns=traj["generated_turns"])
            if sample is None:
                continue
            samples.append(sample)
            profile_state = traj.get("profile_state") or {}
            sampled_profile_states.append(profile_state)
            sampled_profile_successes.append(profile_success)
            non_tensors.append({
                "messages": copy.deepcopy(simulator.role["history"]),
                "emo_point": final_emo,
                "dialogue_turns": traj["dialogue_turns"],
                "scene_uid": traj["scene_uid"],
                "trajectory_uid": traj["trajectory_uid"],
                "rollout_idx": traj["rollout_idx"],
                "player_profile": str(simulator.role.get("player", "")),
                "scene_profile": str(simulator.role.get("scene", "")),
                "profile_id": str(profile_state.get("profile_id", "")),
                "profile_state_id": str(profile_state.get("state_id", "")),
                "profile_cooperation": int(profile_state.get("cooperation", -1)),
                "profile_emotion_intensity": int(profile_state.get("emotion_intensity", -1)),
                "profile_trust": int(profile_state.get("trust", -1)),
                "profile_difficulty": float(profile_state.get("difficulty", 0.0)),
                "profile_behavior_tags": json.dumps(profile_state.get("behavior_tags", []), ensure_ascii=False),
                "profile_pool_size": int(_os.environ.get("SEAD_PROFILE_POOL_SIZE", "500")),
                "profile_success": profile_success,
                "profile_controller_target_difficulty_disabled": float(self._sead_target_difficulty),
                "profile_batch_success_rate": float(self._sead_last_batch_sr),
                "profile_controller_error": float(self._sead_last_controller_error),
                "profile_controller_mode": str(profile_state.get("controller_mode") or self._sead_last_controller_mode),
                "profile_stats_key_level": str(self._sead_stats_key_level),
            })
            simulator.data_for_save = copy.deepcopy(simulator.role)
            try:
                simulator.save_player_data()
            except Exception as exc:
                print(f"[DuoRole-SR] save_player_data skipped: {exc}")

        if not samples:
            raise RuntimeError("[DuoRole-SR] rollout produced no trajectory samples")

        old_target_difficulty = 0.0
        profile_to_scores = {}
        for item in non_tensors:
            profile_to_scores.setdefault(str(item.get("profile_id", "")), []).append(float(item.get("emo_point", 0.0)))
        controller_updates = []
        for profile_id, scores in profile_to_scores.items():
            try:
                update = update_profile_controller(
                    profile_id,
                    scores,
                    success_threshold=float(_os.environ.get("DUOROLE_SUCCESS_THRESHOLD", "50")),
                )
                controller_updates.append(update)
            except Exception as exc:
                print(f"[RLVER-SEAD-PassCtrl] controller update skipped for {profile_id}: {exc}")
        controller_summary = get_profile_controller_summary()
        self._sead_rollout_step = int(controller_summary.get("controller_step", self._sead_rollout_step))
        self._sead_last_batch_sr = float(np.mean(sampled_profile_successes)) if sampled_profile_successes else self._sead_last_batch_sr
        self._sead_last_controller_error = self._sead_last_batch_sr - 0.5
        for item in non_tensors:
            item["profile_next_controller_target_difficulty_disabled"] = float(self._sead_target_difficulty)
            item["profile_next_batch_success_rate"] = float(self._sead_last_batch_sr)
            item["profile_next_controller_error"] = float(self._sead_last_controller_error)
            item["profile_controller_seen_count"] = int(controller_summary.get("seen_profile_count", 0))
            item["profile_controller_score_mean"] = float(controller_summary.get("score_mean", 1.0))

        if sampled_profile_states:
            profile_difficulties = np.array(
                [float(s.get("difficulty", 0.0)) for s in sampled_profile_states],
                dtype=np.float32,
            )
            profile_cooperations = np.array(
                [float(s.get("cooperation", -1)) for s in sampled_profile_states],
                dtype=np.float32,
            )
            profile_emotions = np.array(
                [float(s.get("emotion_intensity", -1)) for s in sampled_profile_states],
                dtype=np.float32,
            )
            profile_trusts = np.array(
                [float(s.get("trust", -1)) for s in sampled_profile_states],
                dtype=np.float32,
            )
            profile_success_rate = float(np.mean(sampled_profile_successes)) if sampled_profile_successes else 0.0
            seen_states, ideal_states, easy_states, hard_states = self._sead_state_count_summary()
            unique_state_count = len(set(str(s.get("state_id", "")) for s in sampled_profile_states))
            unique_profile_count = len(set(str(s.get("profile_id", "")) for s in sampled_profile_states))
            example_profiles = []
            for s in sampled_profile_states[:3]:
                example_profiles.append(
                    f"{s.get('profile_id', s.get('state_id', ''))}:D={float(s.get('difficulty', 0.0)):.1f},"
                    f"c/e/t={s.get('cooperation', -1)}/{s.get('emotion_intensity', -1)}/{s.get('trust', -1)}"
                )
            example_tags = sampled_profile_states[0].get("behavior_tags", []) if sampled_profile_states else []
            print(
                "[RLVER-SAGEVariant-PassCtrl][rollout] "
                f"rows={len(samples)} user_source=DSV3_API assistant_source=Qwen_vLLM scorer=DSV3_SAGE "
                f"controller_mode={self._sead_last_controller_mode} "
                f"stats_key_level={self._sead_stats_key_level} "
                f"rollout_step={int(self._sead_rollout_step)} "
                f"profile_pool_size={int(_os.environ.get('SEAD_PROFILE_POOL_SIZE', '500'))} "
                f"profile_success_rate={profile_success_rate:.3f} "
                f"target_sr={self._sead_target_sr:.3f} "
                f"passctrl_error={self._sead_last_controller_error:.3f} "
                f"difficulty_mean={float(np.mean(profile_difficulties)):.3f} "
                f"difficulty_min={float(np.min(profile_difficulties)):.3f} "
                f"difficulty_max={float(np.max(profile_difficulties)):.3f} "
                f"cooperation_mean={float(np.mean(profile_cooperations)):.3f} "
                f"emotion_intensity_mean={float(np.mean(profile_emotions)):.3f} "
                f"trust_mean={float(np.mean(profile_trusts)):.3f} "
                f"unique_state_count={unique_state_count} "
                f"unique_profile_count={unique_profile_count} "
                f"cr_seen_state_count={seen_states} "
                f"cr_ideal_state_count={ideal_states} "
                f"cr_easy_state_count={easy_states} "
                f"cr_hard_state_count={hard_states} "
                f"controller_seen_profile_count={int(controller_summary.get('seen_profile_count', 0))} "
                f"controller_score_mean={float(controller_summary.get('score_mean', 1.0)):.3f} "
                f"controller_pass_rate_mean={float(controller_summary.get('pass_rate_mean', 0.5)):.3f} "
                f"examples={' | '.join(example_profiles)} "
                f"first_behavior_tags={json.dumps(example_tags, ensure_ascii=False)}"
            )

        batch = TensorDict(
            {
                'prompts': torch.stack([s["prompts"] for s in samples]).contiguous(),
                'responses': torch.stack([s["responses"] for s in samples]).contiguous(),
                'input_ids': torch.stack([s["input_ids"] for s in samples]).contiguous(),
                'attention_mask': torch.stack([s["attention_mask"] for s in samples]).contiguous(),
                'position_ids': torch.stack([s["position_ids"] for s in samples]).contiguous(),
                'generation_mask': torch.stack([s["generation_mask"] for s in samples]).contiguous(),
                'user_generation_mask': torch.stack([s["user_generation_mask"] for s in samples]).contiguous(),
            },
            batch_size=len(samples),
        )
        non_tensor_batch = {key: to_1d_np_array([item[key] for item in non_tensors]) for key in non_tensors[0].keys()}
        if int(batch['generation_mask'].sum().item()) <= 0:
            raise RuntimeError("[DuoRole-SR] zero training mask detected; aborting to avoid empty training batch")
        print(f"[DuoRole-SR] rollout rows={len(samples)} trajectories, assistant_mask_tokens={int(batch['generation_mask'].sum().item())}, final_user_mask_tokens={int(batch['user_generation_mask'].sum().item())}, profile_batch_sr={self._sead_last_batch_sr:.3f}")
        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
