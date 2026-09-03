#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAGE Environment for verl RL Training
Integrates SAGE's emotional dialogue simulation with verl's PPO training
"""

import torch
import numpy as np
import os
import json
import sys
import requests
import re
from typing import List, Dict, Any

# Add parent directory to path to import SAGE modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl import DataProto


class SAGEEnvironment():
    """
    SAGE Environment for multi-turn emotional dialogue RL training.
    Uses NPC response API and emotion-based reward calculation.
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = config.get('tokenizer', None)
        
        # Environment configuration
        self.per_turn_length = config.get('per_turn_length', 5000)
        self.max_turns = config.get('max_turns', 8)
        
        # NPC API configuration (the model being trained)
        self.npc_api_url = os.getenv("SAGE_NPC_API_URL", "http://localhost:8100/v1/chat/completions")
        self.npc_model_name = os.getenv("SAGE_NPC_MODEL_NAME", "qwen3-8b")
        
        # Judge API configuration (for reward calculation)
        self.judge_api_url = os.getenv("SAGE_JUDGE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.judge_api_model = os.getenv("SAGE_JUDGE_API_MODEL", "deepseek-v3")
        self.judge_api_key = os.getenv("SAGE_JUDGE_API_KEY", "")
        
        # Emotion scoring configuration
        self.emo_count = {"Emotion-S": 100, "Emotion-A": 70, "Emotion-B": 40, "Emotion-C": 10}
        
        print(f"[SAGEEnvironment] NPC API URL: {self.npc_api_url}")
        print(f"[SAGEEnvironment] Judge API URL: {self.judge_api_url}")
        print(f"[SAGEEnvironment] Max turns: {self.max_turns}")

    def call_npc_api(self, messages: List[Dict[str, str]]) -> str:
        """Call NPC API to get response"""
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.npc_model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(
                self.npc_api_url,
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[ERROR] NPC API call failed: {e}")
            return "I'm sorry, I couldn't process that."

    def calculate_emotion_reward(self, conversation: List[Dict[str, str]], profile: Dict[str, Any]) -> float:
        """
        Calculate emotion-based reward using SAGE's emotion scoring system.
        Simulates the emotion analyzer to compute final emotion score.
        """
        # Initial emotion
        emo_point = 40  # Start at Emotion-B
        emo_state = "Emotion-B"
        
        # Use judge API to evaluate emotion changes for each turn
        for i in range(1, len(conversation), 2):  # Skip user messages, evaluate NPC responses
            if i >= len(conversation):
                break
                
            # Get conversation up to this point
            partial_convo = conversation[:i+1]
            
            # Call judge API to evaluate emotion change
            emotion_change = self._evaluate_emotion_change(partial_convo, profile, emo_point)
            
            # Update emotion
            emo_point += emotion_change
            emo_point = max(0, min(100, emo_point))
            
            # Update emotion state
            for emo_state_name in ["Emotion-S", "Emotion-A", "Emotion-B", "Emotion-C"]:
                if emo_point >= self.emo_count[emo_state_name]:
                    emo_state = emo_state_name
                    break
            if emo_point < 10:
                emo_state = 'Emotion-F'
        
        # Final reward is normalized emotion score
        reward = emo_point / 100.0
        
        return reward

    def _evaluate_emotion_change(self, conversation: List[Dict[str, str]], 
                                 profile: Dict[str, Any], 
                                 current_emotion: int) -> int:
        """Use judge API to evaluate emotion change based on NPC's latest response"""
        
        judge_prompt = f"""You are an emotion analyzer for a multi-turn emotional dialogue system.

Character Profile:
{profile.get('player', 'Unknown')}

Background:
{profile.get('scene', 'Unknown')}

Current Emotion: {current_emotion}/100

Conversation History:
{json.dumps(conversation, ensure_ascii=False, indent=2)}

Based on the character's profile, background, and the NPC's latest response, analyze how the character's emotion should change.
Consider:
1. Does the NPC's response show empathy and understanding?
2. Does it address the character's emotional needs?
3. Is the response appropriate for the character's current emotional state?

Output ONLY a single integer between -10 and +10 representing the emotion change.
Positive means emotion improves, negative means it worsens."""
        
        messages = [
            {"role": "system", "content": "You are an emotion analyzer. Output only an integer between -10 and +10."},
            {"role": "user", "content": judge_prompt}
        ]
        
        headers = {
            "Authorization": f"Bearer {self.judge_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.judge_api_model,
            "messages": messages,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(
                self.judge_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract integer from response
            numbers = re.findall(r'[+-]?\d+', content)
            if numbers:
                change = int(numbers[0])
                return max(-10, min(10, change))
            else:
                return 0
        except Exception as e:
            print(f"[WARNING] Emotion evaluation failed: {e}")
            return 0

    def get_reward_batched(self, data: DataProto):
        """
        Calculate rewards for a batch of conversations.
        Returns reward tensors for PPO training.
        """
        reward_batched = []
        reward_locs = []
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Extract conversation messages
            messages = data_item.non_tensor_batch.get('messages', [])
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            
            # Extract profile information
            profile = data_item.non_tensor_batch.get('profile', {
                'player': 'Unknown',
                'scene': 'Unknown'
            })
            if isinstance(profile, np.ndarray):
                profile = profile.item() if profile.size == 1 else profile.tolist()
            
            # Calculate emotion-based reward
            reward = self.calculate_emotion_reward(messages, profile)
            reward_batched.append(reward)
            
            # Find reward location (end of response)
            attention_mask = data_item.batch['attention_mask']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = attention_mask[prompt_length:].sum()
            reward_locs.append(valid_response_length - 1)
        
        # Convert to numpy array
        reward_batched = np.array(reward_batched)
        reward_batched = np.maximum(reward_batched, 0)  # Ensure non-negative
        original_reward_batched = reward_batched.copy()
        
        # Create reward tensors
        original_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        penalized_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            original_reward_tensor[i, reward_locs[i]] = original_reward_batched[i]
            penalized_reward_tensor[i, reward_locs[i]] = reward_batched[i]
        
        return original_reward_tensor, penalized_reward_tensor
