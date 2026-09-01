"""Deterministic natural-language realization of the 24 EG-SEC states."""

from __future__ import annotations

from typing import Mapping


DISCLOSURE_RULES = {
    1: ("低", "不要主动完整透露困境的关键细节。回复相对简短，一次只披露少量信息；"
        "只有在助手持续表现出理解后，才逐步补充更深层的感受和背景。"),
    2: ("中", "愿意说明基本困境并回应助手的问题，但不会主动透露所有敏感细节。"
        "是否进一步披露，取决于助手的回应是否具体、尊重且贴合当前感受。"),
    3: ("高", "愿意较主动地描述事件、感受和困扰，也愿意回答助手的问题。"
        "可以较快补充关键背景，但仍保持符合人物画像和真实对话节奏。"),
}

ACTIVATION_RULES = {
    1: ("中等", "表现出清晰但相对克制的负面情绪。能够较完整地组织语言，"
        "不要频繁使用极端化、崩溃式或灾难化表达。"),
    2: ("高", "表现出明显而强烈的负面情绪，可能出现自责、焦虑、愤怒或灾难化理解。"
        "表达可以更急促或波动，但不能脱离原始场景，也不要无依据地升级为极端危机。"),
}

TRUST_RULES = {
    1: ("很低", "对助手是否真正理解自己抱有明显怀疑。可能质疑、试探或拒绝泛泛安慰；"
        "在助手给出具体且贴合情境的回应之前，不轻易接受其判断或建议。"),
    2: ("低", "对助手保持谨慎，愿意继续对话但不会立即接受安慰或建议。"
        "会观察助手是否认真倾听，再决定是否相信其回应。"),
    3: ("中", "对助手持中性、观望态度。能够接受合理的共情回应，"
        "但遇到空洞、说教或偏离问题的回答时会表现出保留。"),
    4: ("高", "倾向于相信助手的善意和理解，愿意积极回应贴合情境的安慰或建议，"
        "并在感到被理解后自然表现出更强的接受度。"),
}


def format_interaction_state(profile_state: Mapping) -> str:
    disclosure = int(profile_state["cooperation"])
    activation = int(profile_state["emotion_intensity"])
    trust = int(profile_state["trust"])
    try:
        disclosure_label, disclosure_rule = DISCLOSURE_RULES[disclosure]
        activation_label, activation_rule = ACTIVATION_RULES[activation]
        trust_label, trust_rule = TRUST_RULES[trust]
    except KeyError as exc:
        raise ValueError(
            f"invalid EG-SEC state: disclosure={disclosure}, activation={activation}, trust={trust}"
        ) from exc

    return f"""隐式互动条件（仅供用户模拟器使用，助手不可见）：

【倾诉开放度：{disclosure_label}】
{disclosure_rule}

【情绪激活度：{activation_label}】
{activation_rule}

【关系信任度：{trust_label}】
{trust_rule}

请同时遵守以上三项要求，并自然体现在披露节奏、情绪表达、对回应的信任和支持接受方式中。
人物背景、事件事实和真正的支持意图必须以原始场景为准，不得新增、替换或改变。
不要直接说出任何维度名称、等级、状态编号、难度或系统设定。"""
