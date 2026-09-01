import importlib.util
import itertools
import os
import sys
import unittest


MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "method", "interaction_state_prompt.py",
)
SPEC = importlib.util.spec_from_file_location("interaction_state_prompt", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class InteractionStatePromptTest(unittest.TestCase):
    def test_all_24_prompts_are_unique_and_deterministic(self):
        prompts = []
        for disclosure, activation, trust in itertools.product(range(1, 4), range(1, 3), range(1, 5)):
            state = {
                "cooperation": disclosure,
                "emotion_intensity": activation,
                "trust": trust,
            }
            first = MODULE.format_interaction_state(state)
            second = MODULE.format_interaction_state(state)
            self.assertEqual(first, second)
            prompts.append(first)
        self.assertEqual(len(prompts), 24)
        self.assertEqual(len(set(prompts)), 24)

    def test_support_goal_is_not_invented_by_state_prompt(self):
        prompt = MODULE.format_interaction_state({
            "cooperation": 1, "emotion_intensity": 2, "trust": 1,
        })
        self.assertNotIn("想要具体、能执行的下一步建议", prompt)
        self.assertNotIn("主要想被听见", prompt)
        self.assertIn("真正的支持意图必须以原始场景为准", prompt)

    def test_invalid_level_fails_fast(self):
        with self.assertRaises(ValueError):
            MODULE.format_interaction_state({
                "cooperation": 4, "emotion_intensity": 2, "trust": 1,
            })


if __name__ == "__main__":
    unittest.main()
