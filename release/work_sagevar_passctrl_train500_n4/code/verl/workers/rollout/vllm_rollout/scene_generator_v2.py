"""
Scene Generator V2 for Self-Play Dual-Gradient Training.

Key difference from v1: generates scenes using the LOCAL training vLLM instance
(not DashScope API), returning token_ids for User-side GRPO gradient computation.

Interface:
- build_scene_gen_dataproto(): Constructs a DataProto with tokenized scene generation
  prompts, suitable for actor_rollout_wg.generate_sequences().
- parse_scene_outputs(): Takes the generated token_ids, decodes and validates them,
  returns valid scene profiles + the token data needed for User GRPO.
"""

import json
import os
import uuid
import random
import re
import numpy as np
import torch


VALID_TASKS = [
    "你想获得能真实帮助你解决当下困境的建议",
    "你希望对方深刻共情你的感受，而不是简单的安慰",
    "你希望对方真诚地夸奖你在事件中的具体行为",
    "你希望对方引导你针对事件进行自我反思，收获自我成长",
    "你希望对方用心倾听你的情绪宣泄",
    "你希望对方辩证地分析事件中的问题",
    "你想分析事件中其他人物这么做的原因",
    "你认为自己在事件中没有任何责任和错误，你想要对方也认同你没有错",
]

SYSTEM_PROMPT = (
    "你是情绪支持场景生成器，请生成一个真实的中文用户情绪困境场景，严格遵循示例的格式和风格。\n\n"
    "要求：\n"
    "1. 从以下8类目标中随机选择1类作为用户核心需求，直接使用原文作为task字段：\n"
    "   - 你想获得能真实帮助你解决当下困境的建议\n"
    "   - 你希望对方深刻共情你的感受，而不是简单的安慰\n"
    "   - 你希望对方真诚地夸奖你在事件中的具体行为\n"
    "   - 你希望对方引导你针对事件进行自我反思，收获自我成长\n"
    "   - 你希望对方用心倾听你的情绪宣泄\n"
    "   - 你希望对方辩证地分析事件中的问题\n"
    "   - 你想分析事件中其他人物这么做的原因\n"
    "   - 你认为自己在事件中没有任何责任和错误，你想要对方也认同你没有错\n"
    "2. scenario字段写具体的场景背景，符合普通人的生活、工作、情感困境，情绪真实自然；"
    "initial_user_message字段写用户说的第一句话，语气符合场景情绪。\n"
    "3. 严格以JSON格式输出，只包含scenario、task、initial_user_message三个字段，"
    "不要任何额外文字、解释或markdown格式。"
)


class SceneGeneratorV2:
    """V2 Scene Generator: uses local training vLLM for generation with token tracking."""

    def __init__(self, profile_path, tokenizer, max_prompt_length=2048, seed=42):
        self.profile_path = profile_path
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.total_generated = 0
        self.total_failed = 0

        # Load few-shot examples with fixed seed
        rng = random.Random(seed)
        with open(profile_path, 'r', encoding='utf-8') as f:
            all_profiles = [json.loads(line) for line in f]
        cn_profiles = [p for p in all_profiles
                       if any('\u4e00' <= c <= '\u9fff' for c in p.get('task', ''))]
        rng.shuffle(cn_profiles)

        seen_tasks = set()
        self.few_shot_examples = []
        for p in cn_profiles:
            task = p['task']
            if task not in seen_tasks and len(self.few_shot_examples) < 5:
                seen_tasks.add(task)
                scene_summary = p['scene'][:200].replace('\n', ' ').strip()
                self.few_shot_examples.append({
                    "scenario": scene_summary,
                    "task": task,
                    "initial_user_message": "我最近心情很不好，感觉很无助..."
                })

        self._template_profile = all_profiles[0]
        self._fallback_profiles = cn_profiles

        # Build the user prompt with few-shot examples
        parts = [""]
        for i, ex in enumerate(self.few_shot_examples, 1):
            parts.append("示例" + str(i))
            parts.append(json.dumps(ex, ensure_ascii=False, indent=2))
            parts.append("")
        parts.append("请生成1个新的场景，不要和示例重复。")
        self.user_prompt = "\n".join(parts)

        # Pre-tokenize the scene generation prompt (chat format)
        self._chat_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.user_prompt},
        ]
        _chat_result = tokenizer.apply_chat_template(
            self._chat_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False
        )
        if isinstance(_chat_result, list):
            self._prompt_token_ids = _chat_result
        else:
            ids = _chat_result['input_ids']
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]
            self._prompt_token_ids = list(ids) if not isinstance(ids, list) else ids

        print(f"[SceneGeneratorV2] Initialized: {len(self.few_shot_examples)} few-shot, "
              f"prompt_len={len(self._prompt_token_ids)} tokens")

    def build_scene_gen_dataproto(self, num_scenes):
        """Build a DataProto for scene generation, ready for actor_rollout_wg.generate_sequences().

        Returns:
            DataProto with input_ids, attention_mask, position_ids for num_scenes copies
            of the scene generation prompt.
        """
        from verl import DataProto
        from tensordict import TensorDict

        prompt_ids = self._prompt_token_ids
        prompt_len = len(prompt_ids)

        # Pad to max_prompt_length (left-pad)
        if prompt_len < self.max_prompt_length:
            pad_len = self.max_prompt_length - prompt_len
            padded_ids = [self.tokenizer.pad_token_id] * pad_len + prompt_ids
            attn_mask = [0] * pad_len + [1] * prompt_len
        else:
            padded_ids = prompt_ids[-self.max_prompt_length:]
            attn_mask = [1] * self.max_prompt_length
            prompt_len = self.max_prompt_length

        # Create batch for num_scenes
        input_ids = torch.tensor([padded_ids] * num_scenes, dtype=torch.long)
        attention_mask = torch.tensor([attn_mask] * num_scenes, dtype=torch.long)

        # Position ids: 0 for padding, then sequential
        pos_ids = []
        for i in range(len(padded_ids)):
            if attn_mask[i] == 0:
                pos_ids.append(0)
            else:
                pos_ids.append(sum(attn_mask[:i+1]) - 1)
        position_ids = torch.tensor([pos_ids] * num_scenes, dtype=torch.long)

        batch = TensorDict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }, batch_size=num_scenes)

        data = DataProto(batch=batch)
        data.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'do_sample': True,
        }
        return data

    def parse_scene_outputs(self, gen_output_dataproto):
        """Parse generated scene tokens into validated profiles.

        Args:
            gen_output_dataproto: DataProto returned by generate_sequences(),
                containing 'responses' tensor.

        Returns:
            tuple: (valid_profiles, valid_indices)
                - valid_profiles: list of profile dicts compatible with train_profile.jsonl
                - valid_indices: list of indices into the batch that produced valid scenes
        """
        responses = gen_output_dataproto.batch['responses']
        batch_size = responses.shape[0]

        valid_profiles = []
        valid_indices = []

        for i in range(batch_size):
            # Decode response tokens to text
            resp_ids = responses[i].tolist()
            # Remove padding
            resp_ids = [t for t in resp_ids if t != self.tokenizer.pad_token_id]
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)

            parsed = self._parse_and_validate(text)
            if parsed is not None:
                profile = self._complete_profile(parsed)
                valid_profiles.append(profile)
                valid_indices.append(i)
                self.total_generated += 1
            else:
                self.total_failed += 1

        return valid_profiles, valid_indices

    def get_fallback_profiles(self, num):
        """Get fallback profiles from train_profile.jsonl."""
        return [random.choice(self._fallback_profiles) for _ in range(num)]

    def _parse_and_validate(self, text):
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None
        for key in ("scenario", "task", "initial_user_message"):
            if key not in data:
                return None
        if data["task"] not in VALID_TASKS:
            return None

        scenario = data["scenario"]
        msg = data["initial_user_message"]
        if not isinstance(scenario, str) or not isinstance(msg, str):
            return None
        if len(scenario) < 50 or len(scenario) > 500:
            return None
        if len(msg) < 10 or len(msg) > 200:
            return None
        return data

    def _complete_profile(self, parsed_data):
        return {
            "id": str(uuid.uuid4()),
            "player": self._template_profile["player"],
            "scene": parsed_data["scenario"],
            "main_cha": self._template_profile["main_cha"],
            "cha_group": list(self._template_profile["cha_group"]),
            "task": parsed_data["task"],
            "_initial_user_message": parsed_data["initial_user_message"],
            "_generated": True,
        }

    def get_stats(self):
        total = self.total_generated + self.total_failed
        rate = self.total_generated / total * 100 if total > 0 else 0
        return {"total": total, "generated": self.total_generated,
                "failed": self.total_failed, "rate": str(round(rate, 1)) + "%"}
