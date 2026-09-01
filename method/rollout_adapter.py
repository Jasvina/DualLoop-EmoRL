"""Minimal rollout-side integration for the dual-loop controller."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

try:
    from .emotion_boundary_controller import (
        EmotionBoundaryController,
        stable_scene_key,
        support_intent_from_role,
    )
    from .interaction_state_prompt import format_interaction_state
except ImportError:  # Direct execution with method/ on PYTHONPATH.
    from emotion_boundary_controller import (
        EmotionBoundaryController,
        stable_scene_key,
        support_intent_from_role,
    )
    from interaction_state_prompt import format_interaction_state


def interaction_states() -> list[dict]:
    """Return the complete 3 x 2 x 4 interaction-state space."""
    states = []
    for disclosure, activation, trust in itertools.product(range(1, 4), range(1, 3), range(1, 5)):
        states.append({
            "profile_id": f"d{disclosure}_a{activation}_t{trust}",
            "state_id": f"d{disclosure}_a{activation}_t{trust}",
            "cooperation": disclosure,
            "emotion_intensity": activation,
            "trust": trust,
        })
    return states


@dataclass(frozen=True)
class SharedGroupCondition:
    support_intent: str
    scene_key: str
    state: Mapping
    simulator_instruction: str


class DualLoopRolloutAdapter:
    """Sample one condition per GRPO group and update it after all rollouts."""

    def __init__(self, controller: EmotionBoundaryController):
        self.controller = controller
        self.candidates = interaction_states()

    def sample_group(self, scenario: Mapping, rng: np.random.Generator) -> SharedGroupCondition:
        support_intent = support_intent_from_role(scenario)
        state, _ = self.controller.sample(support_intent, self.candidates, rng)
        return SharedGroupCondition(
            support_intent=support_intent,
            scene_key=stable_scene_key(scenario),
            state=state,
            simulator_instruction=format_interaction_state(state),
        )

    def update_group(self, condition: SharedGroupCondition, rewards: Sequence[float]) -> dict:
        return self.controller.update_group(
            condition.support_intent,
            condition.state,
            rewards,
        )
