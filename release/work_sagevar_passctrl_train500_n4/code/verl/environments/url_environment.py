from verl import DataProto
import requests
import torch
import numpy as np


class URLEnvironment():

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer


    def get_reward_batched(self, data: DataProto):  #batched
        messages_batched = []
        reward_locs = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            messages = data_item.non_tensor_batch['messages']
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            messages_batched.append(messages)

            attention_mask = data_item.batch['attention_mask']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = attention_mask[prompt_length:].sum()
            reward_locs.append(valid_response_length - 1)

        # reward_batched = requests.post(url, json=payload).json()
        reward_batched = data.non_tensor_batch['emo_point']/100
        reward_batched = np.maximum(reward_batched, 0)
        original_reward_batched = reward_batched.copy()



        original_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        penalized_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            original_reward_tensor[i, reward_locs[i]] = original_reward_batched[i]
            penalized_reward_tensor[i, reward_locs[i]] = reward_batched[i]
        
        return original_reward_tensor, penalized_reward_tensor
