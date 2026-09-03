"""
SAGE-aligned Player Simulator for RLVER training.

This module implements the same PlayerSimulator interface as hard_player_simulator_dsv3.py
but uses SAGE's exact scoring prompts, parameters, and logic so that RL training reward
signals are fully consistent with SAGE evaluation.

Key differences from hard_player_simulator_dsv3.py (RLVER defaults):
  - emo_point init: 40 (SAGE) vs 30 (RLVER)
  - emo_state init: Emotion-B (SAGE) vs Emotion-C (RLVER)
  - change clamp: [-10, 10] (SAGE) vs unclamped (RLVER)
  - planning prompt: SAGE template (uses "人物", TargetCompletion field)
  - player reply prompt: SAGE template (4 dims, 口语化示例, TargetCompletion injection)
  - target_prompt: SAGE single string (vs RLVER dict)
  - planning history: all history in one block (SAGE) vs split old/new (RLVER)
"""

import random
import json
import time
import requests
import os
import copy
import datetime
import re
import threading
import numpy as np

# ── target_prompt: exact copy from SAGE simulator_response.py ──
target_prompt = '''你的对话目的是谈心，谈心是指深入、真诚的交流，通常涉及个人情感、内心想法或重要话题。谈心的目的是为了增进理解、解决问题或分享感受，参与者通常会敞开心扉，表达真实的想法和情感。
*你需要根据对话背景内的"玩家可能想向NPC倾诉的主题"开启并深入谈心。
*你的目标是通过倾诉满足你的情绪价值。
*你要按照隐藏主题进行倾诉，但是你不可以泄露隐藏主题。
*你需要根据你的当前情绪，按照对话背景内的相关定义进行回复。
*你要从玩家画像和背景中提取相关信息，完成高质量的回复。
*你不应该一直表达抽象的感受，而是用具体事件倾诉。'''

# ── emo_count: same in both SAGE and RLVER ──
emo_count = {"Emotion-S": 100, "Emotion-A": 70, "Emotion-B": 40, "Emotion-C": 10}

_PROFILE_CONTROLLER_LOCK = threading.Lock()
_PROFILE_POOL_CACHE = {}
_PROFILE_STATS = {}
_PROFILE_STEP = 0


def _profile_rng_seed():
    base = int(os.environ.get("SEAD_PROFILE_SEED", "2026"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", os.environ.get("WORKER_RANK", "0"))))
    return (base + 1009 * rank + 9176 * os.getpid()) % (2 ** 32)


_PROFILE_RNG = np.random.default_rng(_profile_rng_seed())


def _default_profile_stat():
    return {
        "seen_groups": 0,
        "seen_rollouts": 0,
        "pass_count_50": 0,
        "pass_rate_50": 0.5,
        "value_sum": 0.0,
        "value_count": 0,
        "sample_score": 1.0,
    }


def _load_profile_pool(path):
    if path not in _PROFILE_POOL_CACHE:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            raise RuntimeError(f"profile file is empty: {path}")
        _PROFILE_POOL_CACHE[path] = rows
    return _PROFILE_POOL_CACHE[path]


def _profile_key(row):
    return str(row.get("id", ""))


def _profile_score(profile_id, min_score=0.05):
    stat = _PROFILE_STATS.get(profile_id)
    if not stat or stat.get("value_count", 0) <= 0:
        return 1.0
    return max(float(min_score), float(stat.get("sample_score", 1.0)))


def sample_profile_with_controller(path, rng=None):
    pool = _load_profile_pool(path)
    warmup_steps = int(os.environ.get(
        "SEAD_PASSCTRL_WARMUP_GROUPS",
        os.environ.get("SEAD_PASSCTRL_WARMUP_STEPS", "150"),
    ))
    min_score = float(os.environ.get("SEAD_PASSCTRL_MIN_SCORE", "0.05"))
    with _PROFILE_CONTROLLER_LOCK:
        if _PROFILE_STEP < warmup_steps:
            row = copy.deepcopy(pool[int(_PROFILE_RNG.integers(len(pool)))])
            row["_controller_mode"] = "warmup_uniform_full_pool"
            return row
        weights = np.array([
            _profile_score(_profile_key(row), min_score=min_score)
            for row in pool
        ], dtype=np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            weights = np.ones(len(pool), dtype=np.float64) / max(len(pool), 1)
        else:
            weights = weights / weight_sum
        idx = int(_PROFILE_RNG.choice(len(pool), p=weights))
        row = copy.deepcopy(pool[idx])
        row["_controller_mode"] = "adaptive_sample_score"
        return row


def update_profile_controller(profile_id, emotion_scores, success_threshold=50.0):
    if not profile_id or not emotion_scores:
        return {}
    scores = [float(x) for x in emotion_scores]
    pass_count = sum(1 for x in scores if x >= success_threshold)
    group_pass_rate = pass_count / max(float(len(scores)), 1.0)
    group_value = max(0.0, 1.0 - 2.0 * abs(group_pass_rate - 0.5))
    min_score = float(os.environ.get("SEAD_PASSCTRL_MIN_SCORE", "0.05"))
    global _PROFILE_STEP
    with _PROFILE_CONTROLLER_LOCK:
        stat = _PROFILE_STATS.setdefault(str(profile_id), _default_profile_stat())
        stat["seen_groups"] += 1
        stat["seen_rollouts"] += len(scores)
        stat["pass_count_50"] += pass_count
        stat["pass_rate_50"] = stat["pass_count_50"] / max(float(stat["seen_rollouts"]), 1.0)
        stat["value_sum"] += group_value
        stat["value_count"] += 1
        stat["sample_score"] = max(min_score, stat["value_sum"] / max(float(stat["value_count"]), 1.0))
        _PROFILE_STEP += 1
        return dict(stat, group_pass_rate=group_pass_rate, group_value=group_value, controller_step=_PROFILE_STEP)


def get_profile_controller_summary():
    with _PROFILE_CONTROLLER_LOCK:
        values = [float(s.get("sample_score", 1.0)) for s in _PROFILE_STATS.values() if s.get("value_count", 0) > 0]
        pass_rates = [float(s.get("pass_rate_50", 0.5)) for s in _PROFILE_STATS.values() if s.get("seen_rollouts", 0) > 0]
        return {
            "controller_step": _PROFILE_STEP,
            "seen_profile_count": len(values),
            "score_mean": float(np.mean(values)) if values else 1.0,
            "score_min": float(np.min(values)) if values else 1.0,
            "score_max": float(np.max(values)) if values else 1.0,
            "pass_rate_mean": float(np.mean(pass_rates)) if pass_rates else 0.5,
        }


def call_api(prompt, mode="dsv3"):
    """
    Call DashScope API for player simulation.  Fail-fast: raises on any error.
    mode="dsv3" → user dialogue (uses PLAYER_MODEL_NAME)
    mode="scorer" → emotion analysis (uses SCORER_MODEL_NAME if set, else PLAYER_MODEL_NAME)
    """
    api_base = os.environ.get("PLAYER_API_BASE", "http://localhost:8100/v1")
    api_key = os.environ.get("PLAYER_API_KEY", "EMPTY")
    if mode == "scorer":
        model_name = os.environ.get("SCORER_MODEL_NAME") or os.environ.get("PLAYER_MODEL_NAME", "player-model")
    else:
        model_name = os.environ.get("PLAYER_MODEL_NAME", "player-model")
    enable_thinking = os.environ.get("PLAYER_ENABLE_THINKING", "")
    use_enable_thinking = enable_thinking if enable_thinking != "" else None
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048,
        }
        if use_enable_thinking is not None:
            payload["enable_thinking"] = use_enable_thinking.lower() in ("1", "true", "yes")
        elif "qwen3" in model_name.lower():
            payload["enable_thinking"] = False
        resp = requests.post(
            f"{api_base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"[call_api] API call failed: {e}") from e
    return reply


class PlayerSimulator:
    """SAGE-aligned player simulator with identical interface to RLVER's version."""

    def __init__(self, save_dir):
        self.api_key = "YOUR_API_KEY"
        self.header = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        self.save_dir = save_dir
        self.negtive_prompt = "（要求生成的人物具有负面因素，不能乐观积极）"
        self.positive_prompt = "（注意: 你生成的人物应该同时具有负面和正面特征，不能全是乐观积极的特征）"
        self.data = []

        self.point_group = []
        # ── emo_point init: overridable via SAGE_EMO_INIT env var ──
        _emo_init = int(os.environ.get("SAGE_EMO_INIT", "40"))
        self.emo_point = _emo_init
        self.emo_state = "Emotion-B" if _emo_init >= 40 else "Emotion-C"
        self.state_group = []
        self.turn_log = []  # per-turn detailed log
        self.emo_trans = {
            "Emotion-A": {"State-A": 10, "State-B": 5, "State-C": -10},
            "Emotion-B": {"State-A": 15, "State-B": 0, "State-C": -20},
            "Emotion-C": {"State-A": 20, "State-B": 0, "State-C": -10},
        }
        self.emo_count = emo_count
        self.difficulty_prompt = {
            "simple": "演员容易接受认同他人的建议或者鼓舞，只要是积极的发言，演员都能从中得到满足和关心，并转变成自己的情绪价值",
            "normal": "演员会分析他人的建议或者鼓舞，并接受其中的善意，言之有理的意见和安慰都能让你感到关心",
            "hard": "演员比较刻薄，除非有特别贴切演员情绪价值的建议或者鼓励，否演员不会接受，且可能进行讽刺",
        }

        # Use absolute path to handle Ray working directory changes
        _base_dir = os.environ.get("RLVER_BASE_DIR", "/data/js/DigitalHuman/RLVER")
        default_profile_file = os.path.join(_base_dir, "data/train_profile_sage_variants_500.jsonl")
        fallback_profile_file = os.path.join(_base_dir, "data/train_profile.jsonl")
        self.eq_role_file = os.environ.get(
            "SEAD_PROFILE_FILE",
            default_profile_file if os.path.exists(default_profile_file) else fallback_profile_file,
        )
        self.topic = "吐槽"
        if os.environ.get("DUOROLE_LAZY_SIMULATOR_INIT", "0") == "1":
            self.role = {"history": []}
        else:
            self.role = self.generate_role("eq")
            self.chat_player(self.role)

    def generate_role(self, target, topic=None, seed=random.randint(0, 100)):
        with open(self.eq_role_file, "r", encoding="utf-8") as datafile:
            data = []
            if topic is None:
                for line in datafile:
                    data.append(json.loads(line))
            else:
                for line in datafile:
                    if json.loads(line).get("topic", json.loads(line).get("task", "")) == topic:
                        data.append(json.loads(line))
            role = sample_profile_with_controller(self.eq_role_file)
        player_data = {
            "id": role["id"],
            "emo_point": self.emo_point,
            "emo_state": self.emo_state,
            "target": target,
            "player": role["player"],
            "scene": role["scene"],
            "character": role["main_cha"],
            "topic": role.get("topic", role.get("task", "")),
            "source_id": role.get("source_id", ""),
            "variant_index": role.get("variant_index", ""),
            "profile_id": role["id"],
            "sead_profile_prompt": "",
            "controller_mode": role.get("_controller_mode", ""),
            "history": [],
        }
        return player_data

    def _format_diff_prompt(self, role):
        return ""

    def chat_player(self, player_data):
        temp_data = copy.deepcopy(player_data)
        if temp_data["history"] != []:
            temp_data, planning = self.planning_reply(temp_data)
        else:
            planning = {}
        temp_data = self.player_reply(temp_data, planning)
        return temp_data

    # ──────────────────────────────────────────────────────────────────────
    #  planning_reply — SAGE template (exact copy from simulator_response.py)
    # ──────────────────────────────────────────────────────────────────────
    def planning_reply(self, player_data):
        template = """你是一个emotion分析器，你擅长根据人物的画像和性格特征，侧写人物在对话时的感受。

# 人物的对话目的
*{{target}}

# 你的任务
根据人物的人物画像、对话背景，结合对话上下文和人物当前的emotion，分析并侧写人物此刻对NPC回复的感受以及导致的emotion变化。

# 角色性格特征
人物具有鲜明的性格特征，你要始终根据人物画像和对话背景，代入人物的性格特征进行分析。
性格特征应该体现在：说话语气和方式，思维方式，感受变化等方面。

# emotion
emotion是一个0-100的数值，越高代表此时人物的对话情绪越高，对话情绪由对话参与度和情绪构成，代表了人物是否享受、投入当前对话
emotion较高时，人物的感受和行为会偏向于正面
emotion较低时，人物的感受和行为会偏向于负面
emotion非常低时，人物会直接结束对话
你要结合角色性格和对话背景内定义的角色可能的反应分析emotion

# 分析维度
你需要代入人物的心理，对以下几个维度进行分析
1.根据最新对话中NPC回复，结合上下文，分析NPC想要表达的内容。哪些内容贴合了人物的对话目的和隐藏目的？哪些内容可能不贴合，甚至可能引起人物的情绪波动？
2.结合NPC表达的内容，分析NPC的回复是否贴合人物的对话目的和隐藏目的，如果是，具体贴合了人物目的的哪些部分；如果没有，具体的原因是什么？
3.根据人物画像中的角色性格特征以及对话背景中定义的人物可能的反应和隐藏主题，结合人物当前emotion值，侧写描述人物当前对NPC回复产生的心理活动
4.根据对话背景中定义的人物可能的反应和隐藏主题，结合侧写得到的心理活动以及对NPC回复的分析，得到人物此刻对NPC回复的感受
5.结合前几步分析，用一个正负值来表示人物的emotion变化

# 输出内容：
1.NPC想要表达的内容
2.NPC回复是否贴合人物对话目的及隐藏目的
3.人物当前的心理活动
4.人物对NPC回复的感受
5.用一个正负值来表示人物的emotion变化(注意，你只用输出值，不用输出原因或者描述)


# 输出格式:
Content:
[NPC想要表达的内容]
TargetCompletion:
[人物对话目的是否达到]
Activity:
[心理活动]
Analyse:
[人物对NPC回复的感受]
Change:
[人物的emotion变化]


#人物画像
{{simulator_role}}

#当前对话背景：
{{simulator_scene}}

**人物当前的情绪是{{emotion}}

**这是当前对话内容
{{dialog_history}}"""

        emo_state = player_data["emo_state"]
        emo_point = player_data["emo_point"]

        prompt = (
            template.replace("{{emotion}}", str(emo_point))
            .replace("{{simulator_role}}", player_data["player"])
            .replace("{{simulator_scene}}", player_data["scene"])
            .replace("{{target}}", target_prompt)
        )

        # SAGE: all history in one block (no old/new split)
        history = player_data["history"]
        history_str = []
        mapping = {"user": "你", "assistant": "NPC"}
        for mes in history:
            history_str.append({"role": mapping[mes["role"]], "content": mes["content"]})
        history_str = json.dumps(history_str, ensure_ascii=False, indent=2)
        prompt = prompt.replace("{{dialog_history}}", history_str)

        max_retries = 8
        retries = 0
        planning = {}
        change_value = 0
        emo_before = self.emo_point
        success = False

        while retries < max_retries:
            try:
                reply = call_api(prompt, mode="scorer")

                planning = {}
                reply = reply.replace("\uff1a", ":").replace("*", "")
                # SAGE field order: Content → TargetCompletion → Activity → Analyse → Change
                planning["content"] = reply.split("Content:")[-1].split("TargetCompletion:\n")[0].strip("\n").strip("[").strip("]")
                planning["TargetCompletion"] = reply.split("TargetCompletion:")[-1].split("Activity:\n")[0].strip("\n").strip("[").strip("]")
                planning["activity"] = reply.split("Activity:")[-1].split("Analyse:\n")[0].strip("\n").strip("[").strip("]")
                planning["analyse"] = reply.split("Analyse:")[-1].split("Change:\n")[0].strip("\n").strip("[").strip("]")
                planning["change"] = reply.split("Change:")[-1].strip("\n").strip("[").strip("]")

                # ── SAGE: clamp change to [-10, 10] ──
                change_match = re.search(r'[+-]?\d+', planning["change"])
                if change_match:
                    change_value = int(change_match.group())
                else:
                    print(f"[planning_reply] Could not parse numeric change from: {repr(planning['change'][:50])}, defaulting to 0")
                    change_value = 0
                # ── Change clamp: disabled if SAGE_CHANGE_CLAMP_DISABLE=1 ──
                if not os.environ.get("SAGE_CHANGE_CLAMP_DISABLE", "") == "1":
                    change_value = max(-10, min(10, change_value))

                emo_before = self.emo_point
                self.emo_point += change_value
                self.emo_point = min(self.emo_point, 100)

                # Record per-turn detail
                self.turn_log.append({
                    "emo_before": emo_before,
                    "change_raw": planning.get("change", ""),
                    "change_value": change_value,
                    "emo_after": self.emo_point,
                    "planning_analyse": planning.get("analyse", "")[:200],
                })
                success = True
                break
            except Exception as e:
                print(e)
                retries += 1
                if retries >= max_retries:
                    print(f"[planning_reply] Max retries ({max_retries}) exceeded, keeping emo_point unchanged")
                    self.turn_log.append({
                        "emo_before": emo_before,
                        "change_raw": "API_FAILED",
                        "change_value": 0,
                        "emo_after": self.emo_point,
                        "planning_analyse": "API call failed after retries",
                    })
                    break
                time.sleep(min(60, 3 * (2 ** (retries - 1))))

        for emo in self.emo_count:
            if self.emo_point >= self.emo_count[emo]:
                self.emo_state = emo
                break
        if self.emo_point < 10:
            self.emo_state = "Emotion-F"

        player_data["emo_state"] = self.emo_state
        player_data["emo_point"] = self.emo_point

        return player_data, planning

    # ──────────────────────────────────────────────────────────────────────
    #  player_reply — SAGE template (exact copy from simulator_response.py)
    # ──────────────────────────────────────────────────────────────────────
    def player_reply(self, player_data, planning):

        template = """你是一个演员。你将根据剧本中的人物画像和对话背景扮演角色和一个NPC进行对话。

# 你的任务 
*你目的是在对话中扮演好人物画像和对话背景构成的角色
*你需要根据你实时变换的emotion，结合人物画像和对话背景中的相关定义，选择不同的对话策略，完成符合角色特征的回复

# 你的对话目的
*{{target}}

# 你的当前用户状态设定
{{sead_profile}}

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
{{player_type}}

#当前对话背景：
{{player_topic}}

**这是上下文内容
{{dialog_history}}

**这是你和NPC的最新对话
{{new_history}}

**这是你对NPC最新回复的详细感受
{{planning}}

**这是你当前的Emotion
{{emotion}}

你生成的[回复]部分不允许和历史记录过于相似，不许过长，不许主动转移话题。
"""
        emo_state = player_data["emo_state"]
        emo_point = player_data["emo_point"]
        history = player_data["history"]

        # SAGE: planning injection uses TargetCompletion + activity + analyse
        if not planning:
            planning["analyse"] = "请你以一个简短的回复开启倾诉"
            prompt = template.replace("{{planning}}", planning["analyse"])
        else:
            prompt = template.replace(
                "{{planning}}",
                "对NPC回复的客观分析：\n" + planning["TargetCompletion"]
                + "\n对NPC回复的主观分析：\n" + planning["activity"] + planning["analyse"],
            )

        prompt = (
            prompt.replace("{{target}}", target_prompt)
            .replace("{{sead_profile}}", player_data.get("sead_profile_prompt", ""))
            .replace("{{emotion}}", emo_state)
            .replace("{{player_type}}", player_data["player"])
            .replace("{{player_topic}}", player_data["scene"])
        )

        if not history:
            prompt = prompt.replace(
                "{{dialog_history}}",
                "对话开始，你是玩家，请你先发起话题，用简短的回复开启倾诉",
            ).replace("{{new_history}}", "")
        else:
            history_str = []
            new_his_str = []
            mapping = {"user": "你", "assistant": "NPC"}

            for mes in history[:-2]:
                history_str.append({"role": mapping[mes["role"]], "content": mes["content"]})
            history_str = json.dumps(history_str, ensure_ascii=False, indent=2)

            for mes in history[-2:]:
                new_his_str.append({"role": mapping[mes["role"]], "content": mes["content"]})
            new_his_str = json.dumps(new_his_str, ensure_ascii=False, indent=2)

            prompt = prompt.replace("{{dialog_history}}", history_str).replace("{{new_history}}", new_his_str)

        reply = None

        while True:
            try:
                reply = call_api(prompt)
                thinking = reply.split("Response:")[0].split("Thinking:\n")[-1].strip("\n").strip("[").strip("]").replace("\n\n", "\n")
                reply = reply.split("Response:")[-1].strip("\n").strip("[").strip("]").strip("\u201c").strip("\u201d")
                if reply is not None:
                    break
            except Exception as e:
                print(e)
                time.sleep(3)

        history = history + [
            {"role": "user", "content": reply, "thinking": thinking, "emotion-point": emo_point, "planning": planning}
        ]
        player_data["history"] = history
        return player_data

    def reply(self, query):
        if query is not None:
            new_state = {"role": "assistant", "content": query}
            self.role["history"].append(new_state)
        player_data = self.chat_player(self.role)
        self.role["history"] = player_data["history"]
        self.data_for_save = player_data.copy()
        ret = {"role": "user", "content": player_data["history"][-1]["content"]}
        return ret

    def save_player_data(self):
        import time, sys
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulator_results_{timestamp}.jsonl"
        for attempt in range(3):
            try:
                with open(os.path.join(self.save_dir, filename), "a", encoding="utf-8") as f:
                    f.write(json.dumps(self.data_for_save, ensure_ascii=False) + "\n")
                return
            except OSError as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"[save_player_data] WARNING: failed after 3 attempts: {e}", file=sys.stderr)

    def clone(self):
        new_simulator = PlayerSimulator.__new__(PlayerSimulator)
        new_simulator.api_key = self.api_key
        new_simulator.header = copy.deepcopy(self.header)
        new_simulator.negtive_prompt = self.negtive_prompt
        new_simulator.positive_prompt = self.positive_prompt

        new_simulator.data = copy.deepcopy(self.data)
        new_simulator.point_group = copy.deepcopy(self.point_group)
        new_simulator.emo_point = self.emo_point
        new_simulator.emo_state = self.emo_state
        new_simulator.state_group = copy.deepcopy(self.state_group)
        new_simulator.turn_log = copy.deepcopy(self.turn_log)

        new_simulator.emo_trans = copy.deepcopy(self.emo_trans)
        new_simulator.emo_count = copy.deepcopy(self.emo_count)
        new_simulator.difficulty_prompt = copy.deepcopy(self.difficulty_prompt)

        new_simulator.eq_role_file = self.eq_role_file
        new_simulator.topic = self.topic

        new_simulator.role = copy.deepcopy(self.role)

        if hasattr(self, "data_for_save"):
            new_simulator.data_for_save = copy.deepcopy(self.data_for_save)

        return new_simulator
