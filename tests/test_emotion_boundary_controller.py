import os
import importlib.util
import sys
import tempfile
import unittest

import numpy as np

MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "method", "emotion_boundary_controller.py",
)
SPEC = importlib.util.spec_from_file_location("emotion_boundary_controller", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ControllerConfig = MODULE.ControllerConfig
EmotionBoundaryController = MODULE.EmotionBoundaryController


def candidates():
    return [
        {"profile_id": "p0", "state_id": "s0"},
        {"profile_id": "p1", "state_id": "s1"},
        {"profile_id": "p2", "state_id": "s2"},
    ]


class ControllerTest(unittest.TestCase):
    def make_controller(self, path, warmup=2):
        return EmotionBoundaryController(ControllerConfig(
            state_file=path, warmup_groups=warmup, uncertainty_weight=0.0,
            uniform_mix=0.1, shrinkage=0.0, state_prior=0.0,
        ))

    def test_warmup_is_uniform_and_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.sqlite3")
            controller = self.make_controller(path)
            probs, info = controller.sampling_distribution("deep-empathy", candidates())
            np.testing.assert_allclose(probs, [1 / 3] * 3)
            self.assertEqual(info["mode"], "warmup_uniform")
            self.assertEqual(info["effective_pool"], 3.0)
            controller.update_group("deep-empathy", candidates()[0], [10, 40, 70, 90])
            self.assertEqual(self.make_controller(path).groups_seen(), 1)

    def test_boundary_environment_receives_larger_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(os.path.join(directory, "state.sqlite3"), warmup=0)
            controller.update_group("deep-empathy", candidates()[0], [48, 49, 51, 52])
            controller.update_group("deep-empathy", candidates()[1], [90, 91, 92, 93])
            controller.update_group("deep-empathy", candidates()[2], [5, 6, 7, 8])
            probs, info = controller.sampling_distribution("deep-empathy", candidates())
            self.assertEqual(info["mode"], "adaptive_intent_state")
            self.assertGreater(probs[0], probs[1])
            self.assertGreater(probs[0], probs[2])

    def test_hard_pass_rate_and_triangular_value(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(os.path.join(directory, "state.sqlite3"), warmup=0)
            result = controller.update_group(
                "deep-empathy", candidates()[0], [10, 40, 70, 90]
            )
            self.assertEqual(result["group_pass_rate"], 0.5)
            probs, info = controller.sampling_distribution("deep-empathy", candidates())
            detail = next(x for x in info["candidate_details"] if x["state_id"] == "s0")
            self.assertEqual(detail["outcome_mean"], 0.5)
            self.assertEqual(detail["boundary"], 1.0)

    def test_group_update_counts_one_group_and_all_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(os.path.join(directory, "state.sqlite3"), warmup=0)
            controller.update_group("deep-empathy", candidates()[0], [10, 40, 70, 90])
            with controller._connect() as conn:
                row = conn.execute(
                    "SELECT n_groups,n_rollouts FROM intent_state_stats"
                ).fetchone()
            self.assertEqual(row, (1, 4))

    def test_history_is_cumulative_not_last_group_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(os.path.join(directory, "state.sqlite3"), warmup=0)
            controller.update_group("deep-empathy", candidates()[0], [10, 40, 70, 90])
            controller.update_group("deep-empathy", candidates()[0], [90, 90, 90, 90])
            with controller._connect() as conn:
                row = conn.execute(
                    "SELECT n_groups,n_rollouts,outcome_sum FROM intent_state_stats"
                ).fetchone()
            self.assertEqual(row[0], 2)
            self.assertEqual(row[1], 8)
            self.assertAlmostEqual(row[2], 1.5)
            _, info = controller.sampling_distribution("deep-empathy", candidates())
            detail = next(x for x in info["candidate_details"] if x["state_id"] == "s0")
            self.assertAlmostEqual(detail["outcome_mean"], 0.75)

    def test_same_state_has_separate_intent_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(os.path.join(directory, "state.sqlite3"), warmup=0)
            controller.update_group("deep-empathy", candidates()[0], [50, 50, 50, 50])
            controller.update_group("actionable-advice", candidates()[0], [90, 90, 90, 90])
            with controller._connect() as conn:
                state_count = conn.execute("SELECT COUNT(*) FROM state_stats").fetchone()[0]
                arm_count = conn.execute("SELECT COUNT(*) FROM intent_state_stats").fetchone()[0]
            self.assertEqual(state_count, 1)
            self.assertEqual(arm_count, 2)

    def test_incomplete_group_is_not_written(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(os.path.join(directory, "state.sqlite3"), warmup=0)
            result = controller.update_group("deep-empathy", candidates()[0], [10, 90])
            self.assertTrue(result["skipped"])
            self.assertEqual(result["skip_reason"], "incomplete_group")
            self.assertEqual(controller.groups_seen(), 0)
            with controller._connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM intent_state_stats").fetchone()[0]
            self.assertEqual(count, 0)

    def test_non_finite_reward_is_not_written(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(os.path.join(directory, "state.sqlite3"), warmup=0)
            result = controller.update_group(
                "deep-empathy", candidates()[0], [10, 40, 70, float("nan")]
            )
            self.assertTrue(result["skipped"])
            self.assertEqual(result["skip_reason"], "non_finite_reward")
            self.assertEqual(controller.groups_seen(), 0)

    def test_hierarchical_prior_excludes_current_intent_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = EmotionBoundaryController(ControllerConfig(
                state_file=os.path.join(directory, "state.sqlite3"),
                warmup_groups=0,
                uncertainty_weight=0.0,
                uniform_mix=0.0,
                shrinkage=4.0,
                state_prior=4.0,
            ))
            state = candidates()[0]
            controller.update_group("deep-empathy", state, [100, 100, 100, 100])
            _, info = controller.sampling_distribution("deep-empathy", candidates())
            detail = next(x for x in info["candidate_details"] if x["state_id"] == "s0")
            # No other intent has observed s0, so the state prior remains 0.5:
            # (one arm group at 1.0 + four prior groups at 0.5) / five groups.
            self.assertAlmostEqual(detail["outcome_mean"], 0.6)

            controller.update_group("actionable-advice", state, [0, 0, 0, 0])
            _, info = controller.sampling_distribution("deep-empathy", candidates())
            detail = next(x for x in info["candidate_details"] if x["state_id"] == "s0")
            # Other-intent state estimate is (0 + 4*0.5)/(1+4)=0.4;
            # current arm estimate is then (1 + 4*0.4)/(1+4)=0.52.
            self.assertAlmostEqual(detail["outcome_mean"], 0.52)

    def test_duplicate_candidate_state_ids_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(os.path.join(directory, "state.sqlite3"), warmup=0)
            duplicate = [candidates()[0], dict(candidates()[0])]
            with self.assertRaises(ValueError):
                controller.sampling_distribution("deep-empathy", duplicate)


if __name__ == "__main__":
    unittest.main()
