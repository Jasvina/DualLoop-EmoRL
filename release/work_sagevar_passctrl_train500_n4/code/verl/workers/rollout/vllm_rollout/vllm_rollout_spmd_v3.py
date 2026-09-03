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
from verl.workers.rollout.vllm_rollout.system_prompt import *

import os as _os
if _os.environ.get("SIMULATOR_TYPE", "rlver") == "sage":
    from verl.workers.rollout.vllm_rollout.sage_player_simulator import *
else:
    from verl.workers.rollout.vllm_rollout.hard_player_simulator_dsv3 import *

import copy

# ═══════════════════════════════════════════════════════════════════════════════
# [V3] Dual-role self-play helpers
# ═══════════════════════════════════════════════════════════════════════════════

V3_USER_SYSTEM_PROMPT = """你是一个有情感困境的用户。你将根据以下人物画像和对话背景扮演角色，和一个情感支持NPC对话。

# 你的人物画像
{player}

# 对话背景
{scene}

# 对话要求
- 像真实用户一样表达困境和感受
- 逐步透露信息，不要一次性说完
- 根据NPC的回复自然地回应
- 不要角色泄漏（不要说"我是模拟用户""根据设定"等）
- 保持人物性格一致性
- 回复要简短自然，像真人聊天，不要太长"""


def _v3_build_user_messages(history, profile):
    """Build messages for Qwen3-8B to generate a user turn."""
    system_content = V3_USER_SYSTEM_PROMPT.format(
        player=profile.get("player", "一个有情感困境的人"),
        scene=profile.get("scene", "日常生活中遇到了困难"),
    )
    msgs = [{"role": "system", "content": system_content}]
    for msg in history:
        msgs.append({"role": msg["role"], "content": msg["content"]})
    return msgs


def _v3_realism_check(user_reply, profile, history):
    """Rule-based realism check. Returns (score 0~1, penalty_reasons)."""
    score = 1.0
    penalties = []
    if not user_reply or len(user_reply.strip()) < 5:
        score -= 0.5; penalties.append("too_short")
    if len(user_reply) > 500:
        score -= 0.3; penalties.append("too_long")
    _leakage = ["我是模拟", "根据设定", "作为AI", "我是AI", "角色扮演", "模拟用户",
                "按照要求", "人物设定", "根据剧本", "我的任务是"]
    for kw in _leakage:
        if kw in user_reply:
            return -1.0, [f"role_leakage:{kw}"]
    sentences = [s.strip() for s in user_reply.replace("。", ".").split(".") if s.strip()]
    if len(sentences) >= 2 and len(set(sentences)) < len(sentences) * 0.5:
        score -= 0.4; penalties.append("repetition")
    _asst_phrases = ["我理解你的感受", "别担心", "我在这里陪你", "你愿意多说一些吗"]
    if sum(1 for p in _asst_phrases if p in user_reply) >= 2:
        score -= 0.3; penalties.append("sounds_like_assistant")
    return max(-1.0, min(1.0, score)), penalties


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
        return len(self.tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=add_generation_prompt))
    
    def tokenize_with_assistant_mask(self,messages):
        n_messages=len(messages)
        tokenized_messages=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
        head=0
        assistant_mask=[]
        user_mask=[]  # [V3] Track user-generated tokens separately
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
                user_mask.append(torch.zeros(n_add,dtype=torch.bool))
            elif last_message_role=="user":
                assistant_mask.append(torch.zeros(n_add,dtype=torch.bool))
                user_mask.append(torch.ones(n_add,dtype=torch.bool))  # [V3] user tokens
            else:
                # system / other roles → neither
                assistant_mask.append(torch.zeros(n_add,dtype=torch.bool))
                user_mask.append(torch.zeros(n_add,dtype=torch.bool))
            head+=n_add
        assistant_mask=torch.cat(assistant_mask,dim=0)
        user_mask=torch.cat(user_mask,dim=0)
        assert len(assistant_mask)==len(tokenized_messages), "Bug: assistant mask length mismatch"
        assert len(user_mask)==len(tokenized_messages), "Bug: user mask length mismatch"
        return tokenized_messages, assistant_mask, user_mask


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

        raw_prompt=prompts.non_tensor_batch['raw_prompt']
        # rank=vllm_ps.get_tensor_model_parallel_src_rank()

        rank=vllm_ps.get_tensor_model_parallel_rank()

        n=1 if prompts.meta_info.get('validate',False) else self.config.n

        todo=list(range(batch_size*n))

        # 准备对话列表
        messagess=[]
        prefix_lengths=[]
        expanded_player_simulators=[]
        idx_list_=[]
        attention_mask_list_=[]
        position_ids_list_=[]
        for i_batch in range(batch_size):
            raw_prompt_i_batch=raw_prompt[i_batch]
            n_tokens_prefix=self.get_n_tokens(raw_prompt_i_batch,add_generation_prompt=True)#计算token数量

            for _ in range(n):
                messagess.append(list(raw_prompt_i_batch))  # 确保是列表
                prefix_lengths.append(n_tokens_prefix)
                expanded_player_simulators.append(player_simulators[i_batch].clone())
                idx_list_.append(idx[i_batch])
                attention_mask_list_.append(attention_mask[i_batch])
                position_ids_list_.append(position_ids[i_batch])
            # print("messagess type:",type(messagess))
            #messagess:list[dict],1*batch_size
        player_simulators=expanded_player_simulators
        idx = torch.stack(idx_list_)
        attention_mask = torch.stack(attention_mask_list_)
        position_ids = torch.stack(position_ids_list_)
        print(f"idx shape: {idx.shape}")
        turn_count=0
        all_seqs = [[] for _ in range(batch_size*n)]
        all_attention_masks = [[] for _ in range(batch_size*n)]
        all_position_ids = [[] for _ in range(batch_size*n)]
        all_prompts = [[] for _ in range(batch_size*n)]
        all_responses = [[] for _ in range(batch_size*n)]
        all_turn_roles = [[] for _ in range(batch_size*n)]  # [V3] "user" or "assistant" per sample
        dialogue_turns = [0] * batch_size*n
        # print(f"Initial batch_size: {batch_size}")
        # print(f"Initial dialogue_turns: {dialogue_turns}")
        todo_to_batch_idx = {}
        for i in range(len(todo)):
            todo_to_batch_idx[todo[i]] = todo[i] % (batch_size*n)
        print("todo_to_batch_idx:",todo_to_batch_idx)
        dialogue_history = []
        contents = [[] for _ in range(batch_size*n)]
        while True:
            turn_count+=1
            
            if turn_count!=1:

                idx_list = []
                for i in range(len(idx)):
                    idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))
            messagess_todo=[]
            for i_batch_sample in todo:
                messages_=[{"role":msg["role"],"content":msg["content"]} for msg in messagess[i_batch_sample]]#raw_prompt的role是user
                messagess_todo.append(messages_)
            with self.update_sampling_params(**kwargs):
                assert self.sampling_params.n==1,"n should be 1 for multi-turn"

                outputs = self.inference_engine.chat(
                    messages=messagess_todo,
                    sampling_params=self.sampling_params,
                    use_tqdm=False)
            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
            if not response:
                raise ValueError("response is empty,messagess_todo:",messagess_todo)
            response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.environment.per_turn_length).to(idx.device)

            assert response.shape[1] <= self.config.environment.per_turn_length,"Bug: response too long from vllm: "+str(response.shape)

            
            seq = torch.cat([idx, response], dim=-1) 
            for i, i_batch_sample in enumerate(todo):
                original_idx = todo_to_batch_idx[i_batch_sample]  
                all_responses[original_idx].append(response[i])
                all_seqs[original_idx].append(seq[i])
                all_turn_roles[original_idx].append("assistant")  # [V3] mark role
            response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            current_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
            
            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).repeat(len(todo), 1)  
            response_position_ids = position_ids[:, -1:] + delta_position_id
            current_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            
            for i, i_batch_sample in enumerate(todo):
                original_idx = todo_to_batch_idx[i_batch_sample]
                all_attention_masks[original_idx].append(current_attention_mask[i])
                all_position_ids[original_idx].append(current_position_ids[i])
                all_prompts[original_idx].append(idx[i])


            assert len(todo)==len(response)
            for i_batch_sample,response_ in zip(todo,response):
                response_text=self.tokenizer.decode(response_,skip_special_tokens=True)
                content = response_text
                contents[i_batch_sample].append(content)
                if turn_count > 1:  
                    prompt_data = next_prompt_to_tokenize[todo.index(i_batch_sample)] 
                    dialogue_data = {"prompt": prompt_data, "response": content}
                    dialogue_history.append(dialogue_data)
                extracted_content_1=self.extract_content(content)
                messagess[i_batch_sample].append({"role":"assistant","content":extracted_content_1})
 
            messagess_todo=[]
            for i_batch_sample in todo:
                messagess_todo.append(messagess[i_batch_sample])
                
            # [V3] Dual-role or original user reply generation
            _v3_mode = os.environ.get("V3_DUAL_ROLE", "") == "true"
            env_response_batched = []
            
            if _v3_mode:
                # --- V3: Qwen3-8B generates user turn via vLLM ---
                user_messagess = []
                for i_batch_sample in todo:
                    _profile = getattr(player_simulators[i_batch_sample], 'role', {})
                    _user_msgs = _v3_build_user_messages(messagess[i_batch_sample], _profile)
                    user_messagess.append(_user_msgs)
                
                # Generate user replies (same vLLM engine, user system prompt)
                from vllm import SamplingParams as _SP
                _user_sp = _SP(temperature=0.7, top_p=0.95, max_tokens=300, n=1)
                user_outputs = self.inference_engine.chat(
                    messages=user_messagess,
                    sampling_params=_user_sp,
                    use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})
                
                # [V3 FULL] Process user outputs: store as independent training samples
                user_response_tokens = []
                for output in user_outputs:
                    user_response_tokens.append(list(output.outputs[0].token_ids))
                
                # Pad user responses to same length
                _user_resp_tensor = pad_2d_list_to_length(
                    user_response_tokens, self.pad_token_id,
                    max_length=self.config.environment.per_turn_length).to(idx.device)
                
                # Build user sample: (prompt_before_user, user_response)
                # prompt_before_user = current idx (which is the prompt used for assistant this turn)
                # + the assistant response just generated (already in messagess)
                # Actually, for user turn the "prompt" is the full history up to assistant's last reply
                # which is exactly `seq` = idx + assistant_response from above
                _user_seq = torch.cat([seq, _user_resp_tensor[:len(todo)]], dim=-1)
                _user_resp_attn = get_eos_mask(response_id=_user_resp_tensor[:len(todo)],
                                               eos_token=eos_token_id, dtype=attention_mask.dtype)
                _user_full_attn = torch.cat((current_attention_mask[:len(todo)], _user_resp_attn), dim=-1)
                _user_resp_len = _user_resp_tensor.shape[1]
                _user_delta_pos = torch.arange(1, _user_resp_len + 1, device=position_ids.device)
                _user_delta_pos = _user_delta_pos.unsqueeze(0).repeat(len(todo), 1)
                _user_resp_pos = current_position_ids[:len(todo), -1:] + _user_delta_pos
                _user_full_pos = torch.cat([current_position_ids[:len(todo)], _user_resp_pos], dim=-1)
                
                # Store user samples into all_* lists
                for i, i_batch_sample in enumerate(todo):
                    original_idx_val = todo_to_batch_idx[i_batch_sample]
                    all_responses[original_idx_val].append(_user_resp_tensor[i])
                    all_seqs[original_idx_val].append(_user_seq[i])
                    all_attention_masks[original_idx_val].append(_user_full_attn[i])
                    all_position_ids[original_idx_val].append(_user_full_pos[i])
                    all_prompts[original_idx_val].append(seq[i])  # prompt for user = seq (history + asst reply)
                    all_turn_roles[original_idx_val].append("user")
                    dialogue_turns[original_idx_val] += 1
                
                # Now process user text for conversation history + emotion tracking
                for i, (output, i_batch_sample) in enumerate(zip(user_outputs, todo)):
                    user_text = self.tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)
                    
                    # Realism check
                    _profile = getattr(player_simulators[i_batch_sample], 'role', {})
                    _realism, _penalties = _v3_realism_check(user_text, _profile, messagess[i_batch_sample])
                    if not hasattr(player_simulators[i_batch_sample], '_v3_realism_scores'):
                        player_simulators[i_batch_sample]._v3_realism_scores = []
                    player_simulators[i_batch_sample]._v3_realism_scores.append(_realism)
                    
                    # DSV3 emotion tracking
                    _asst_last = messagess[i_batch_sample][-1]["content"] if messagess[i_batch_sample] else ""
                    try:
                        player_simulators[i_batch_sample].role["history"].append(
                            {"role": "assistant", "content": _asst_last})
                        _temp_data = copy.deepcopy(player_simulators[i_batch_sample].role)
                        _temp_data["history"].append({"role": "user", "content": user_text})
                        if len(_temp_data["history"]) > 1:
                            _temp_data, _ = player_simulators[i_batch_sample].planning_reply(_temp_data)
                            player_simulators[i_batch_sample].emo_point = _temp_data["emo_point"]
                            player_simulators[i_batch_sample].emo_state = _temp_data["emo_state"]
                        player_simulators[i_batch_sample].role["history"].append(
                            {"role": "user", "content": user_text})
                    except Exception as _e:
                        print(f"[V3] Emotion tracking error: {_e}")
                    
                    env_response_batched.append({"role": "user", "content": user_text})
            else:
                # --- Original: DSV3 generates user reply ---
                for i in range(len(messagess_todo)):
                    msg = messagess_todo[i].copy()
                    swapped_messages = []
                    assert msg[-1]["role"]=="assistant"
                    swapped_messages={"role":"user","content":msg[-1]["content"]}
                    
                    env_response_message = player_simulators[i].reply(swapped_messages["content"])
                    env_response_batched.append(env_response_message)

            todo_=todo.copy()
            assert len(todo_)==len(env_response_batched)
            for i_batch_sample, env_response in zip(todo_, env_response_batched):
                assert env_response["role"] == "user"
                messagess[i_batch_sample].append(env_response)
                
                original_idx = todo_to_batch_idx[i_batch_sample]
                dialogue_turns[original_idx] += 1
                
                if dialogue_turns[original_idx] > self.config.environment.max_turns:
                    todo.remove(i_batch_sample)
                    continue
                if "再见" in env_response["content"] or "拜拜" in env_response["content"] or player_simulators[i_batch_sample].emo_point<=0:
                    todo.remove(i_batch_sample)
                    continue
            
            if len(todo)==0:
                break
            next_prompt_to_tokenize=[]
            for i_batch_sample in todo:
                dialog_messages = []
                for msg in messagess[i_batch_sample]:
                    dialog_messages.append({"role":msg["role"],"content":msg["content"]})
                next_prompt_to_tokenize.append(dialog_messages)

            end_dialog = False
            idx_device=idx.device
            attention_mask_device=attention_mask.device
            position_ids_device=position_ids.device
            
            valid_todo = []
            valid_dialogs = []
            
            for i, i_batch_sample in enumerate(todo):
                dialog = next_prompt_to_tokenize[i]
                prompt_with_chat_template = self.tokenizer.apply_chat_template(dialog, add_generation_prompt=True, tokenize=False)
                if len(prompt_with_chat_template) <= self.config.prompt_length:
                    valid_todo.append(i_batch_sample)
                    valid_dialogs.append(dialog)
                else:
                    print(f"错误：提示文本过长，长度为 {len(prompt_with_chat_template)}，最大允许长度为 {self.config.prompt_length}")
            
            todo = valid_todo
            next_prompt_to_tokenize = valid_dialogs
            
            if len(todo) == 0:
                break
                
            idx = []
            attention_mask = []
            position_ids = []
            
            for i, dialog in enumerate(next_prompt_to_tokenize):
                nxt_input_ids, nxt_attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=self.tokenizer.apply_chat_template(dialog, add_generation_prompt=True, tokenize=False),
                    tokenizer=self.tokenizer,
                    max_length=self.config.prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation='error')

                next_position_ids = compute_position_id_with_mask(nxt_attention_mask)

                idx.append(nxt_input_ids[0])
                attention_mask.append(nxt_attention_mask[0])
                position_ids.append(next_position_ids[0])
            
            idx = torch.stack(idx, dim=0).to(idx_device)
            attention_mask = torch.stack(attention_mask, dim=0).to(attention_mask_device)
            position_ids = torch.stack(position_ids, dim=0).to(position_ids_device)

     
        emo_point_list = []
        for i in range(len(player_simulators)):
            emo_point_list.append(player_simulators[i].emo_point)
            player_simulators[i].save_player_data()

        all_seqs_flat = []
        all_prompts_flat = []
        all_responses_flat = []
        all_attention_masks_flat = []
        all_position_ids_flat = []
        
        for i in range(batch_size*n):
            all_seqs_flat.extend(all_seqs[i])
            all_prompts_flat.extend(all_prompts[i])
            all_responses_flat.extend(all_responses[i])
            all_attention_masks_flat.extend(all_attention_masks[i])
            all_position_ids_flat.extend(all_position_ids[i])
        
        final_seq = torch.stack(all_seqs_flat)
        final_prompts = torch.stack(all_prompts_flat)
        final_responses = torch.stack(all_responses_flat)
        final_attention_mask = torch.stack(all_attention_masks_flat)
        final_position_ids = torch.stack(all_position_ids_flat)

        new_batch_size = sum(dialogue_turns)  
        batch = TensorDict(
            {
                'prompts': final_prompts.contiguous(),
                'responses': final_responses.contiguous(),
                'input_ids': final_seq.contiguous(),
                'attention_mask': final_attention_mask.contiguous(),
                'position_ids': final_position_ids.contiguous(),
            },
            batch_size=new_batch_size,
            )  
        expanded_messagess = []
        expanded_emo_point_list = []
        expanded_dialogue_turns = []
        print("batch_size:",batch_size)
        print("dialogue_turns:",dialogue_turns)
        for i in range(batch_size*n):
            for _ in range(dialogue_turns[i]): 
                expanded_messagess.append(messagess[i])
                expanded_emo_point_list.append(emo_point_list[i])
                expanded_dialogue_turns.append(dialogue_turns[i])  
        print("expanded_messagess:",expanded_messagess)
        print("expanded_emo_point_list:",expanded_emo_point_list)
        print("expanded_dialogue_turns:",expanded_dialogue_turns)
        non_tensor_batch = {
            'messages': to_1d_np_array(expanded_messagess),
            'emo_point': to_1d_np_array(expanded_emo_point_list),
            'dialogue_turns': to_1d_np_array(expanded_dialogue_turns),
        }
        # [V3] Add turn_role and realism_score to output
        if os.environ.get("V3_DUAL_ROLE", "") == "true":
            # Expand turn_roles (same pattern as dialogue_turns expansion)
            expanded_turn_roles = []
            expanded_realism = []
            for i in range(batch_size * n):
                _roles = all_turn_roles[i] if i < len(all_turn_roles) else []
                _realism_scores = getattr(player_simulators[i], '_v3_realism_scores', [])
                _realism_idx = 0
                for j in range(len(_roles)):
                    expanded_turn_roles.append(_roles[j])
                    if _roles[j] == "user":
                        _r = _realism_scores[_realism_idx] if _realism_idx < len(_realism_scores) else 0.5
                        expanded_realism.append(_r)
                        _realism_idx += 1
                    else:
                        # assistant turn: realism not applicable, use emo_point as reward (handled in trainer)
                        expanded_realism.append(1.0)  # placeholder, trainer will use emo_point for assistant
            non_tensor_batch['turn_role'] = to_1d_np_array(expanded_turn_roles)
            non_tensor_batch['realism_score'] = to_1d_np_array(expanded_realism)
        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

