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
import os as _os
from concurrent.futures import ThreadPoolExecutor
from verl.workers.rollout.vllm_rollout.system_prompt import *
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

    def _duorole_user_messages(self, simulator, messages, planning):
        import json as _json
        role = simulator.role
        history = role.get("history", [])

        target_prompt_str = target_prompt

        emo_state = getattr(simulator, 'emo_state', 'Emotion-C')
        emo_point = getattr(simulator, 'emo_point', 30)

        if not planning:
            planning_text = "请你以一个简短的回复开启倾诉"
        else:
            planning_text = (
                "对NPC回复的客观分析：\n" + planning.get("TargetCompletion", "") +
                "\n对NPC回复的主观分析：\n" + planning.get("activity", "") + planning.get("analyse", "")
            )

        template = """你是一个演员。你将根据剧本中的人物画像和对话背景扮演角色和一个NPC进行对话。

# 你的任务
*你目的是在对话中扮演好人物画像和对话背景构成的角色
*你需要根据你实时变换的emotion，结合人物画像和对话背景中的相关定义，选择不同的对话策略，完成符合角色特征的回复

# 你的对话目的
*{target}

# Emotion
*你会收到当前你的emotion，你一共有5档emotion，越高代表此时你的对话情绪越高，对话情绪由对话参与度和情绪构成，代表了演员是否享受、投入当前对话
*emotion影响你不同的说话风格和语气以及回应方式，根据不同emotion按照对话背景内定义的角色不同emotion下的反应进行回复：
Emotion-S: 你的emotion已经达到了最高，你可以感谢NPC并用"再见"或"拜拜"告别，直接结束对话。
Emotion-A：高emotion，此时你的对话感受较为正面，你的反馈也比较正面。
Emotion-B：中emotion，此时你没有正面或者负面的感受。
Emotion-C：低emotion，此时你的对话感受较为负面，你的反馈也比较负面。
Emotion-F：你的情绪已到达最负面，不想继续对话。此时你要用"再见"或"拜拜"告别，直接结束对话。

# 你应该区分Emotion和对NPC最新回复感受，Emotion代表你的当前的对话情绪，对NPC回复的感受代表你对NPC回复的即时感受，你需要结合两者生成回复。

# 回复思路
*你会收到当前你对NPC最新回复的详细感受，包含客观分析部分和主观分析部分，你要结合人物画像、对话背景、隐藏主题和详细感受来分析，并决定回复内容。
*分析内容，应该包含以下4个维度：
1.根据你的详细感受和当前Emotion，结合隐藏主题，结合对话背景内定义的角色不同emotion下的反应，当前的回复态度偏向应该是正面、无偏向还是负面？
2.根据你的详细感受和当前Emotion，结合隐藏主题，你的本次回复目标应该是？（注意，你不需要针对NPC的每一句话做出回应，你可以稍微透露你的需求，但不可以主动泄露隐藏主题）
3.根据人物画像中说话风格的相关定义，结合对话背景内定义的角色不同emotion下的反应和你的回复态度以及回复目标，你的说话语气、风格应该是？
4.根据人物画像和对话背景以及隐藏主题，结合你的详细感受以及前三轮分析，你的说话方式和内容应该是？（注意：如果根据人设你是被动型，则你的说话方式应该是被动、不主动提问）
*回复内容，根据分析结果生成初始回复，回复内容要尽可能简洁，不要一次包含过多信息量。
*改造内容，你需要参照下述规则改造你的回复让其更真实，从而得到最终回复：
1.你需要说话简洁，真实的回复一般不会包含太长的句子
2.真实的回复应该更多使用语气词、口语化用语，语法也更随意。
** 部分口语化用语示例："笑死"、"哇塞"、"牛逼"、"简直烦死了"、"真的假的"、"。。。"
3.真实的回复不会直接陈述自己的情绪，而是将情绪蕴含在回复中，用语气表达自己的情绪
4.你绝对不可以使用"我真的觉得……""我真的不知道……""我真的快撑不住了"这些句子，你不应该用"真的"、"根本"来表述你的情绪
5.在表达情绪或观点时，尽量从对话背景中提取新的信息辅助表达
6.你不应该生成和对话上下文中相似的回复

# 输出内容：
*你需要按照回复思路中的分析版块，首先进行4个维度分析
*然后你需要**逐步**按照分析内容并遵顼注意事项生成初始回复，回复中的信息量来源于对话背景和你的联想，你不应该一次性谈论太多事件或内容
*随后你需要根据改造内容分析你应该如何针对初始回复进行改造
*最后你需要根据分析改造初始回复生成最终回复

# 输出格式:
Thinking:
[分析内容]
Origin:
[初始回复]
Change:
[改造分析]
Response:
[最终回复]


# 发言风格
你的发言需要严格遵守"玩家画像"中描述的人物设定和背景。
你的性格和发言风格要遵循"习惯和行为特点"的描述
如果发言要符合你的人物形象，比如负面的人物形象需要你进行负面的发言。
你的语气要符合你的年龄

* 你的发言要遵守以下5条准则
1. 发言必须简洁、随意、自然,按照自然对话进行交流。
2. 不许一次提问超过两个问题。
3. 不允许重复之前说过的回复或者进行相似的回复。
4. 在发言时，可以自然的使用一些口语化词汇
5. 你的发言应该精简，不准过长


#人物画像：
{player_type}

#当前对话背景：
{player_topic}

**这是上下文内容
{dialog_history}

**这是你和NPC的最新对话
{new_history}

**这是你对NPC最新回复的详细感受
{planning}

**这是你当前的Emotion
{emotion}

你生成的[回复]部分不允许和历史记录过于相似，不许过长，不许主动转移话题。
"""

        user_system = (
            template
            .replace("{target}", target_prompt_str)
            .replace("{emotion}", emo_state)
            .replace("{player_type}", role.get('player', ''))
            .replace("{player_topic}", role.get('scene', ''))
            .replace("{planning}", planning_text)
        )

        if not history:
            user_system = user_system.replace(
                "{dialog_history}",
                "对话开始，你是玩家，请你先发起话题，用简短的回复开启倾诉",
            ).replace("{new_history}", "")
        else:
            mapping = {"user": "你", "assistant": "NPC"}
            old_his = [{"role": mapping.get(m["role"], m["role"]), "content": m["content"]} for m in history[:-2]]
            new_his = [{"role": mapping.get(m["role"], m["role"]), "content": m["content"]} for m in history[-2:]]
            user_system = user_system.replace(
                "{dialog_history}", _json.dumps(old_his, ensure_ascii=False, indent=2)
            ).replace(
                "{new_history}", _json.dumps(new_his, ensure_ascii=False, indent=2)
            )

        return [{"role": "system", "content": user_system}]

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
            if role == "user":
                response_user_mask.extend([True] * len(toks))
            else:
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
        if assistant_mask_tokens <= 0 or user_mask_tokens <= 0:
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
            for rollout_idx in range(n):
                simulator = player_simulators[scene_idx].clone()
                simulator.role["history"] = []
                scene_uid = f"scene_{scene_idx}"
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
                })

        active = list(range(len(trajectories)))
        for turn_idx in range(max_turns):
            if not active:
                break

            user_messages_batch = []
            user_prompt_parts = []
            for traj_idx in active:
                traj = trajectories[traj_idx]
                simulator = traj["simulator"]
                user_messages = self._duorole_user_messages(simulator, simulator.role["history"], traj["planning"])
                user_prompt_ids, user_attention_mask, user_position_ids = self._duorole_tokenize_prompt(
                    user_messages, device)
                user_messages_batch.append(user_messages)
                user_prompt_parts.append((user_prompt_ids, user_attention_mask, user_position_ids))

            with self.update_sampling_params(**kwargs):
                assert self.sampling_params.n == 1, "n should be 1 for multi-turn"
                user_outputs = self.inference_engine.chat(
                    messages=user_messages_batch,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False})

            for local_idx, output in enumerate(user_outputs):
                traj_idx = active[local_idx]
                traj = trajectories[traj_idx]
                simulator = traj["simulator"]
                user_token_ids = output.outputs[0].token_ids
                user_text = self.tokenizer.decode(user_token_ids, skip_special_tokens=True).strip()
                if "Response:" in user_text:
                    user_text = user_text.split("Response:")[-1].strip("\n").strip("[").strip("]").strip("\u201c").strip("\u201d").strip()
                elif "Response" in user_text:
                    user_text = user_text.split("Response")[-1].strip(":").strip("\n").strip("[").strip("]").strip("\u201c").strip("\u201d").strip()
                if not user_text:
                    print(f"[DuoRole-SR] empty user text at scene={traj['scene_idx']} rollout={traj['rollout_idx']} turn={turn_idx}")
                    traj["done"] = True
                    continue
                user_token_ids = self.tokenizer.encode(user_text, add_special_tokens=False)
                simulator.role["history"].append({"role": "user", "content": user_text})
                traj["last_user_text"] = user_text
                traj["generated_turns"].append(("user", list(user_token_ids)))

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
            _api_workers = int(_os.environ.get("DUOROLE_API_WORKERS", "2"))

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

        for traj in trajectories:
            if not traj["simulator"].role.get("history"):
                continue
            simulator = traj["simulator"]
            if not any(msg.get("role") == "assistant" for msg in simulator.role["history"]):
                continue
            final_emo = float(simulator.emo_point)
            scene_idx = traj["scene_idx"]
            sample = self._duorole_build_trajectory_sample(
                prompt_ids=idx[scene_idx],
                prompt_attention_mask=attention_mask[scene_idx],
                prompt_position_ids=position_ids[scene_idx],
                eos_token_id=eos_token_id,
                device=device,
                generated_turns=traj["generated_turns"])
            if sample is None:
                continue
            samples.append(sample)
            non_tensors.append({
                "messages": copy.deepcopy(simulator.role["history"]),
                "emo_point": final_emo,
                "dialogue_turns": traj["dialogue_turns"],
                "scene_uid": traj["scene_uid"],
                "trajectory_uid": traj["trajectory_uid"],
                "rollout_idx": traj["rollout_idx"],
            })
            simulator.data_for_save = copy.deepcopy(simulator.role)
            try:
                simulator.save_player_data()
            except Exception as exc:
                print(f"[DuoRole-SR] save_player_data skipped: {exc}")

        if not samples:
            raise RuntimeError("[DuoRole-SR] rollout produced no trajectory samples")

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
        if int(batch['generation_mask'].sum().item()) <= 0 or int(batch['user_generation_mask'].sum().item()) <= 0:
            raise RuntimeError("[DuoRole-SR] zero training mask detected; aborting to avoid empty training batch")
        print(f"[DuoRole-SR] rollout rows={len(samples)} trajectories, assistant_mask_tokens={int(batch['generation_mask'].sum().item())}, final_user_mask_tokens={int(batch['user_generation_mask'].sum().item())}")
        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
