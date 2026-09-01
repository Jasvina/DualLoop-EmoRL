# Runtime Integration

The controller is independent of VERL. A multi-turn GRPO runtime needs only two hooks: one before generating a rollout group and one after receiving the group's terminal emotion rewards.

```python
import numpy as np

from method.emotion_boundary_controller import EmotionBoundaryController
from method.rollout_adapter import DualLoopRolloutAdapter

controller = EmotionBoundaryController()
adapter = DualLoopRolloutAdapter(controller)
rng = np.random.default_rng(2026)

# Once per GRPO group: all group members reuse this condition.
condition = adapter.sample_group(scenario, rng)
simulator_prompt = condition.simulator_instruction

# Generate K multi-turn rollouts with the same scenario and simulator prompt.
rewards = generate_group_and_score(
    scenario=scenario,
    simulator_instruction=simulator_prompt,
    group_size=controller.config.group_size,
)

# Once after all K terminal emotion rewards are available.
diagnostics = adapter.update_group(condition, rewards)
```

## Required Invariants

1. Call `sample_group` exactly once per GRPO group, not once per rollout.
2. Keep the selected scenario and interaction state identical for every rollout in that group.
3. Append `simulator_instruction` only to the simulator context. Never expose it to the assistant policy.
4. Use the continuous terminal emotion reward for policy optimization.
5. Pass the same rewards to `update_group`; the controller performs the hard threshold conversion internally.
6. Update only complete groups. Incomplete or non-finite groups are rejected by the controller.
7. Reuse the same SQLite state file when resuming training.

These invariants preserve the paper's separation between policy learning and curriculum adaptation.
