"""Persistent intent-conditioned controller for dual-loop self-evolution."""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    state_file: str
    warmup_groups: int = 1600
    success_threshold: float = 50.0
    shrinkage: float = 4.0
    state_prior: float = 4.0
    uncertainty_weight: float = 0.15
    uniform_mix: float = 0.10
    sampling_temperature: float = 1.0
    min_score: float = 0.05
    group_size: int = 4

    @classmethod
    def from_env(cls) -> "ControllerConfig":
        return cls(
            state_file=os.environ.get(
                "EGSEC_CONTROLLER_STATE_FILE",
                os.path.join(os.environ.get("RLVER_OUTPUT_DIR", "."), "egsec_controller.sqlite3"),
            ),
            warmup_groups=int(os.environ.get("EGSEC_WARMUP_GROUPS", "1600")),
            success_threshold=float(os.environ.get("EGSEC_SUCCESS_THRESHOLD", "50")),
            shrinkage=max(float(os.environ.get("EGSEC_HIERARCHICAL_SHRINKAGE", "4")), 0.0),
            state_prior=max(float(os.environ.get("EGSEC_STATE_PRIOR", "4")), 0.0),
            uncertainty_weight=max(float(os.environ.get("EGSEC_UNCERTAINTY_WEIGHT", "0.15")), 0.0),
            uniform_mix=min(max(float(os.environ.get("EGSEC_UNIFORM_MIX", "0.10")), 0.0), 1.0),
            sampling_temperature=max(float(os.environ.get("EGSEC_SAMPLING_TEMPERATURE", "1.0")), 1e-6),
            min_score=max(float(os.environ.get("EGSEC_MIN_SCORE", "0.05")), 0.0),
            group_size=max(int(os.environ.get("EGSEC_GROUP_SIZE", "4")), 1),
        )


def support_intent_from_role(role: Mapping) -> str:
    """Read the native hidden support intent without exposing it to the policy."""
    intent = role.get("task") or role.get("topic")
    if not intent:
        raise ValueError("EG-SEC requires every training scenario to provide task/topic")
    return str(intent).strip()


def stable_scene_key(role: Mapping) -> str:
    """Create a collision-resistant rollout-group key from scenario semantics."""
    raw = "\x1f".join(str(role.get(key, "")) for key in ("id", "player", "scene", "topic"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class EmotionBoundaryController:
    """Global SQLite controller over 8 support intents x 24 interaction states."""

    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig.from_env()
        os.makedirs(os.path.dirname(os.path.abspath(self.config.state_file)), exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.state_file, timeout=60.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=60000")
        try:
            # Ray workers may initialize simultaneously. WAL activation briefly
            # takes an exclusive lock, so retry it after installing busy_timeout.
            for attempt in range(10):
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 9:
                        raise
                    time.sleep(0.1 * (attempt + 1))
            conn.execute("PRAGMA synchronous=FULL")
            return conn
        except Exception:
            conn.close()
            raise

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state_stats (
                    state_id TEXT PRIMARY KEY,
                    n_groups INTEGER NOT NULL DEFAULT 0,
                    n_rollouts INTEGER NOT NULL DEFAULT 0,
                    outcome_sum REAL NOT NULL DEFAULT 0,
                    reward_sum REAL NOT NULL DEFAULT 0,
                    reward_sq_sum REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS intent_state_stats (
                    arm_id TEXT PRIMARY KEY,
                    support_intent TEXT NOT NULL,
                    state_id TEXT NOT NULL,
                    n_groups INTEGER NOT NULL DEFAULT 0,
                    n_rollouts INTEGER NOT NULL DEFAULT 0,
                    outcome_sum REAL NOT NULL DEFAULT 0,
                    reward_sum REAL NOT NULL DEFAULT 0,
                    reward_sq_sum REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    UNIQUE(support_intent, state_id)
                );
                CREATE INDEX IF NOT EXISTS idx_intent_state_intent
                    ON intent_state_stats(support_intent);
                """
            )
            conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('groups_seen','0')")
            conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version','2')")

    def groups_seen(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='groups_seen'").fetchone()
        return int(row[0]) if row else 0

    def passed(self, reward: float) -> float:
        """Binary controller outcome; the continuous reward remains the policy signal."""
        return float(float(reward) >= self.config.success_threshold)

    @staticmethod
    def _arm_id(support_intent: str, state_id: str) -> str:
        raw = f"{support_intent}\x1f{state_id}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]

    def _state_rows(self, conn: sqlite3.Connection) -> Dict[str, tuple]:
        rows = conn.execute(
            "SELECT state_id,n_groups,n_rollouts,outcome_sum,reward_sum,reward_sq_sum FROM state_stats"
        ).fetchall()
        return {str(row[0]): tuple(row[1:]) for row in rows}

    def _intent_rows(self, conn: sqlite3.Connection, support_intent: str) -> Dict[str, tuple]:
        rows = conn.execute(
            """SELECT state_id,n_groups,n_rollouts,outcome_sum,reward_sum,reward_sq_sum
               FROM intent_state_stats WHERE support_intent=?""",
            (support_intent,),
        ).fetchall()
        return {str(row[0]): tuple(row[1:]) for row in rows}

    def sampling_distribution(
        self, support_intent: str, candidates: Sequence[Mapping]
    ) -> tuple[np.ndarray, dict]:
        if not candidates:
            raise ValueError("EG-SEC requires at least one candidate state")
        state_ids = [str(candidate["state_id"]) for candidate in candidates]
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("EG-SEC candidate state IDs must be unique")
        seen = self.groups_seen()
        n_candidates = len(candidates)
        if seen < self.config.warmup_groups:
            probs = np.full(n_candidates, 1.0 / n_candidates, dtype=np.float64)
            return probs, {
                "mode": "warmup_uniform",
                "groups_seen": seen,
                "support_intent": support_intent,
                "effective_pool": float(n_candidates),
            }

        with self._connect() as conn:
            state_rows = self._state_rows(conn)
            intent_rows = self._intent_rows(conn, support_intent)

        scores: List[float] = []
        details: List[dict] = []
        total_groups = max(seen, 1)
        for candidate in candidates:
            state_id = str(candidate["state_id"])
            s_n, _, s_sum, _, _ = state_rows.get(state_id, (0, 0, 0.0, 0.0, 0.0))
            a_n, _, a_sum, _, _ = intent_rows.get(state_id, (0, 0, 0.0, 0.0, 0.0))

            # Shrink toward this state's history in other intents. Excluding the
            # current arm prevents using its observations twice.
            other_n = max(float(s_n) - float(a_n), 0.0)
            other_sum = float(s_sum) - float(a_sum)
            state_denom = other_n + self.config.state_prior
            state_mean = (
                (other_sum + self.config.state_prior * 0.5) / state_denom
                if state_denom > 0 else 0.5
            )

            arm_denom = float(a_n) + self.config.shrinkage
            outcome_mean = (
                (float(a_sum) + self.config.shrinkage * state_mean) / arm_denom
                if arm_denom > 0 else state_mean
            )
            boundary = max(0.0, 1.0 - 2.0 * abs(outcome_mean - 0.5))
            uncertainty = math.sqrt(math.log(total_groups + 1.0) / (float(a_n) + 1.0))
            score = max(self.config.min_score, boundary + self.config.uncertainty_weight * uncertainty)
            scores.append(score)
            details.append({
                "state_id": state_id,
                "outcome_mean": outcome_mean,
                "boundary": boundary,
                "uncertainty": uncertainty,
                "n_groups": int(a_n),
            })

        logits = np.log(np.asarray(scores, dtype=np.float64)) / self.config.sampling_temperature
        logits -= float(np.max(logits))
        adaptive = np.exp(logits)
        adaptive /= float(np.sum(adaptive))
        uniform = np.full(n_candidates, 1.0 / n_candidates, dtype=np.float64)
        probs = (1.0 - self.config.uniform_mix) * adaptive + self.config.uniform_mix * uniform
        probs /= float(np.sum(probs))
        return probs, {
            "mode": "adaptive_intent_state",
            "groups_seen": seen,
            "support_intent": support_intent,
            "entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
            "effective_pool": float(1.0 / np.sum(probs ** 2)),
            "candidate_details": details,
        }

    def sample(
        self,
        support_intent: str,
        candidates: Sequence[Mapping],
        rng: np.random.Generator,
    ) -> tuple[dict, dict]:
        probs, diagnostics = self.sampling_distribution(support_intent, candidates)
        index = int(rng.choice(len(candidates), p=probs))
        state = dict(candidates[index])
        diagnostics = dict(diagnostics)
        diagnostics["selected_probability"] = float(probs[index])
        diagnostics["selected_state_id"] = str(state["state_id"])
        return state, diagnostics

    def update_group(
        self,
        support_intent: str,
        profile_state: Mapping,
        rewards: Iterable[float],
    ) -> dict:
        rewards = [float(reward) for reward in rewards]
        if not rewards:
            return {}
        if len(rewards) != self.config.group_size:
            return {
                "skipped": True,
                "skip_reason": "incomplete_group",
                "observed_rollouts": len(rewards),
                "expected_rollouts": self.config.group_size,
                "support_intent": support_intent,
                "state_id": str(profile_state["state_id"]),
            }
        if not all(math.isfinite(reward) for reward in rewards):
            return {
                "skipped": True,
                "skip_reason": "non_finite_reward",
                "observed_rollouts": len(rewards),
                "expected_rollouts": self.config.group_size,
                "support_intent": support_intent,
                "state_id": str(profile_state["state_id"]),
            }
        outcomes = [self.passed(reward) for reward in rewards]
        state_id = str(profile_state["state_id"])
        arm_id = self._arm_id(support_intent, state_id)
        values = (
            1,
            len(rewards),
            float(np.mean(outcomes)),
            float(np.sum(rewards)),
            float(np.sum(np.square(rewards))),
        )

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO state_stats(state_id,n_groups,n_rollouts,outcome_sum,reward_sum,reward_sq_sum)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(state_id) DO UPDATE SET
                     n_groups=n_groups+excluded.n_groups,
                     n_rollouts=n_rollouts+excluded.n_rollouts,
                     outcome_sum=outcome_sum+excluded.outcome_sum,
                     reward_sum=reward_sum+excluded.reward_sum,
                     reward_sq_sum=reward_sq_sum+excluded.reward_sq_sum""",
                (state_id,) + values,
            )
            conn.execute(
                """INSERT INTO intent_state_stats(
                     arm_id,support_intent,state_id,n_groups,n_rollouts,outcome_sum,
                     reward_sum,reward_sq_sum,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(support_intent,state_id) DO UPDATE SET
                     n_groups=n_groups+excluded.n_groups,
                     n_rollouts=n_rollouts+excluded.n_rollouts,
                     outcome_sum=outcome_sum+excluded.outcome_sum,
                     reward_sum=reward_sum+excluded.reward_sum,
                     reward_sq_sum=reward_sq_sum+excluded.reward_sq_sum,
                     updated_at=excluded.updated_at""",
                (arm_id, support_intent, state_id) + values + (time.time(),),
            )
            conn.execute("UPDATE meta SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT) WHERE key='groups_seen'")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

        return {
            "skipped": False,
            "group_outcome_mean": float(np.mean(outcomes)),
            "group_pass_rate": float(np.mean(outcomes)),
            "group_reward_mean": float(np.mean(rewards)),
            "group_reward_std": float(np.std(rewards)),
            "groups_seen": self.groups_seen(),
            "support_intent": support_intent,
            "state_id": state_id,
        }

    def snapshot(self) -> dict:
        with self._connect() as conn:
            groups_seen = int(conn.execute("SELECT value FROM meta WHERE key='groups_seen'").fetchone()[0])
            state_count = int(conn.execute("SELECT COUNT(*) FROM state_stats").fetchone()[0])
            arm_count = int(conn.execute("SELECT COUNT(*) FROM intent_state_stats").fetchone()[0])
            intent_count = int(conn.execute("SELECT COUNT(DISTINCT support_intent) FROM intent_state_stats").fetchone()[0])
        return {
            "groups_seen": groups_seen,
            "state_count": state_count,
            "arm_count": arm_count,
            "intent_count": intent_count,
            "state_file": self.config.state_file,
        }
