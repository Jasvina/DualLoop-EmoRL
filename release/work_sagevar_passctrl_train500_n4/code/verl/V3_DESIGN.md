# V3 实现设计文档（伪代码级，确认后逐文件写真代码）

## 核心改动：rollout 从"模型+API"变成"同一模型双角色"

### 当前 RLVER rollout 循环
```python
while turn_count < max_turns:
    # Step A: NPC (Qwen3-8B) 生成 assistant 回复
    outputs = vllm.chat(messages, role="assistant")
    assistant_reply = decode(outputs)
    messages.append({"role": "assistant", "content": assistant_reply})
    
    # Step B: User (DSV3 API) 生成 user 回复
    user_reply = player_simulator.reply(assistant_reply)  # ← 调外部 API
    messages.append({"role": "user", "content": user_reply})
    
    turn_count += 1
```

### V3 rollout 循环（改动）
```python
while turn_count < max_turns:
    # Step A: Assistant turn — 同一模型，assistant role prompt
    asst_messages = build_messages(history, role="assistant", profile=scene)
    outputs = vllm.chat(asst_messages, role="assistant")
    assistant_reply = decode(outputs)
    history.append({"role": "assistant", "content": assistant_reply})
    
    # Step B: User turn — 同一模型，user role prompt  ← V3 核心改动
    user_messages = build_messages(history, role="user", profile=scene)
    outputs = vllm.chat(user_messages, role="user")      # ← 不再调 API
    user_reply = decode(outputs)
    history.append({"role": "user", "content": user_reply})
    
    turn_count += 1

# Step C: 对话结束后，用冻结 judge 给两侧打分
emotion_score = emotion_judge.score(history, profile)     # → assistant reward
realism_score = realism_judge.score(history, profile)     # → user reward
```

### 关键设计决策

#### 1. User role prompt 怎么设计
```python
USER_SYSTEM_PROMPT = """你是一个有情感困境的用户。你将根据以下人物画像和对话背景扮演角色，和一个情感支持NPC对话。

# 你的人物画像
{profile.player}

# 对话背景
{profile.scene}

# 当前情绪状态
emotion: {current_emo_point}

# 对话要求
- 像真实用户一样表达困境和感受
- 逐步透露信息，不要一次性说完
- 根据 NPC 的回复自然地回应
- 不要角色泄漏（不要说"我是模拟用户"等）
- 保持人物性格一致性
"""
```

#### 2. Token Mask 区分 user/assistant
```python
# 一条完整轨迹的 token 序列：
# [system_tokens] [user_turn1] [asst_turn1] [user_turn2] [asst_turn2] ...

# 需要生成两个 mask：
assistant_mask = [0,0,...,0, 0,...,0, 1,...,1, 0,...,0, 1,...,1, ...]
user_mask      = [0,0,...,0, 1,...,1, 0,...,0, 1,...,1, 0,...,0, ...]

# L_assistant 只对 assistant_mask=1 的 token 算 policy gradient
# L_user 只对 user_mask=1 的 token 算 policy gradient
```

#### 3. Reward 来源（冻结 judge，不是训练中的模型自评）

##### Emotion Judge（给 assistant reward）
- **输入**：完整对话历史 + profile
- **输出**：emo_final (0-100)
- **实现**：DSV3 API（只做评分，不参与对话生成）
- **计算**：`r_asst = emo_final / 100`
- **跟 V1/RLVER 保持一致**

##### Realism Judge（给 user reward）
- **输入**：完整对话历史 + profile + user 每轮回复
- **输出**：realism_score (0-1)
- **实现**：DSV3 API（冻结，只做评分）
- **计算**：`r_user = realism_score - penalties`
- **硬惩罚**：角色泄漏/格式错乱/与 profile 矛盾 → -1.0

#### 4. Advantage 计算
```python
# K=4 条 rollout per scene，跟 V1/RLVER 一样
# Assistant advantage: 组内 z-score on r_asst
A_asst = (r_asst - mean(r_asst_group)) / std(r_asst_group)

# User advantage: 组内 z-score on r_user  
A_user = (r_user - mean(r_user_group)) / std(r_user_group)
```

#### 5. Loss 计算
```python
L_asst = policy_gradient(assistant_tokens, A_asst) + kl_loss * 0.001
L_user = policy_gradient(user_tokens, A_user) + kl_loss * 0.001

L_total = L_asst + λ_user * L_user   # λ_user = 0.1
L_total.backward()
optimizer.step()
```

#### 6. 训练流程
```
Step N:

1. 准备 scene（从固定库抽取，或用 V1 的 self-play 生成）
2. 多轮对话 rollout（Qwen3-8B 交替扮演 user 和 assistant）
   - 每个 scene × K=4 条轨迹
   - 每条轨迹 max 8 turns
3. Emotion Judge 打分 → assistant reward
4. Realism Judge 打分 → user reward  
5. dynT filter（可选，保留 V1 设计）
6. 分别计算 A_asst 和 A_user
7. 分别对 assistant_tokens 和 user_tokens 算 loss
8. Joint backward

```

### 工程实现要点

#### 7. vLLM 双角色生成
- 同一个 vLLM engine
- 每个 turn 交替换 system prompt（user prompt / assistant prompt）
- 用 `chat_template` 区分角色
- 一个 turn 生成后拼回 history，再传下一个 turn

#### 8. dynT / filter 在 V3 中的角色
- **保留 dynT**：用 emotion_judge 的 emo_final 计算 SR
- **filter 逻辑跟 V1 一致**：SR=0/1 弃，中间保留
- T 动态调节：保持 SR ≈ 0.5

#### 9. 超参建议
```
λ_user = 0.1（专家建议起步）
max_turns = 8
K = 4（rollout per scene）
user_temperature = 0.7
assistant_temperature = 0.7（跟 RLVER 的 1.0 不同？待定）
warmup = 0（V3 不需要 warmup，两边同时开始）
```

### 需要新建的文件
1. `vllm_rollout_spmd_v3.py` — 双角色交替生成
2. `ray_trainer_v3.py` — 双 loss 训练循环
3. `dp_actor_v3.py` — user/assistant token mask 分离 backward
4. `judges/emotion_judge.py` — 冻结 emotion 评分（复用现有 SAGE planning_reply 逻辑）
5. `judges/realism_judge.py` — 冻结 realism 评分（新写 prompt）
6. `main_ppo_v3.py` — 入口
7. `launch_v3.sh` — launcher

### 最大风险点
1. **rollout 速度**：每轮要生成两次（user + assistant），比 RLVER 慢 ~2x
2. **reward 调用量**：每条轨迹结束后要调两次 judge（emotion + realism），API 压力翻倍
3. **user 生成质量**：Qwen3-8B 未经训练可能一开始当不好 user → 前几步训练信号差
4. **梯度冲突**：user reward 和 assistant reward 可能让梯度方向不一致
