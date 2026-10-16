"""Tests for src/reward_wrappers.py (Phase 2, Step 1).

Covered:
* ``builtin`` mode is bit-identical to the raw environment;
* the potential-based shaping term telescopes (near-zero over a hover, and
  algebraically exact over a scripted changing sequence);
* the hover bonus fires exactly when its conditions hold (boundary cases);
* the wrapper always exposes original/shaping per step and per episode.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from src.reward_wrappers import MODES, ShapingRewardWrapper

GAMMA = 0.999


class ScriptedEnv(gym.Env):
    """Minimal env: ignores actions and steps through a scripted obs list.

    ``obs_sequence[0]`` is the reset observation; step ``t`` transitions INTO
    ``obs_sequence[t]``. The last observation repeats if the episode runs
    past the end of the script. Episodes terminate after ``length`` steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, obs_sequence, reward: float = 0.0, length: int = 1000):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64)
        self.action_space = spaces.Discrete(4)
        self.obs_sequence = list(obs_sequence)
        self.reward = reward
        self.length = length
        self._steps = 0
        self._idx = 0

    def _obs_at(self, i: int) -> np.ndarray:
        return self.obs_sequence[min(i, len(self.obs_sequence) - 1)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self._idx = 0
        return self._obs_at(0), {}

    def step(self, action):
        self._steps += 1
        self._idx += 1
        terminated = self._steps >= self.length
        return self._obs_at(self._idx), self.reward, terminated, False, {}


def obs8(x=0.0, y=0.5, vx=0.0, vy=0.0, angle=0.0) -> np.ndarray:
    """A LunarLander-shaped observation with defaults at a hover state."""
    return np.array([x, y, vx, vy, angle, 0.0, 0.0, 0.0])


def test_builtin_mode_matches_raw_env():
    """``builtin`` (control arm) returns exactly the built-in environment."""
    actions = [2, 0, 1, 0, 3, 2, 0, 0, 1, 2] * 5  # 50 steps, no landing
    raw = gym.make("LunarLander-v3")
    wrapped = ShapingRewardWrapper(gym.make("LunarLander-v3"), mode="builtin")
    raw_obs, _ = raw.reset(seed=12345)
    wrapped_obs, _ = wrapped.reset(seed=12345)
    assert np.allclose(raw_obs, wrapped_obs)

    for action in actions:
        raw_step = raw.step(action)
        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert np.allclose(raw_step[0], obs), "observations diverged in builtin mode"
        assert float(raw_step[1]) == reward, "rewards diverged in builtin mode"
        assert raw_step[2] == terminated and raw_step[3] == truncated
        assert info["original_reward"] == float(raw_step[1])
        assert info["shaping"] == 0.0
        if terminated:
            break
    raw.close()
    wrapped.close()


def test_potential_telescopes_over_hover():
    """Constant state (perfect hover) => tiny total shaping, ~per-frame 0.

    Each term is (gamma - 1) * Phi; over 200 hover frames the whole shaping
    stream is ~0.1, where a flat +1/frame hover bonus would pay ~200.
    """
    hover = obs8(x=0.0, y=0.5)  # Phi = -(1.0 * 0.5) = -0.5
    env = ShapingRewardWrapper(
        ScriptedEnv([hover], reward=0.0, length=200),
        mode="potential_shaped",
        gamma=GAMMA,
        potential_cost=0.0,
    )
    env.reset()
    terms = []
    for _ in range(200):
        _, _, terminated, _, info = env.step(0)
        terms.append(info["shaping"])
        assert not terminated or info["episode_shaping"]
    total = sum(terms)
    phi = env.potential(hover)
    expected_per_frame = (GAMMA - 1.0) * phi
    assert all(abs(t - expected_per_frame) < 1e-12 for t in terms)
    # Telescoping: bounded total, dwarfed by a flat per-frame bonus.
    assert abs(total) < 1.0
    assert abs(total) < 0.01 * 200 * 1.0  # < 1% of the hover-bonus equivalent


def test_potential_sum_matches_telescoping_identity():
    """Undiscounted potential sum = gamma*sum(Phi_t) - sum(Phi_{t-1})."""
    rng = np.random.default_rng(0)
    seq = [obs8(x=rng.uniform(-1, 1), y=rng.uniform(0.2, 1.5),
                vx=rng.uniform(-0.5, 0.5), vy=rng.uniform(-0.5, 0.5),
                angle=rng.uniform(-0.5, 0.5)) for _ in range(30)]
    env = ShapingRewardWrapper(
        ScriptedEnv(seq, reward=0.0, length=len(seq)),
        mode="potential_shaped",
        gamma=GAMMA,
        potential_cost=0.0,
    )
    env.reset()
    for _ in range(len(seq) - 1):
        env.step(0)
    phis = [env.potential(o) for o in seq]
    # Transitions t=1..T-1: term_t = gamma*Phi_t - Phi_{t-1}.
    expected = GAMMA * sum(phis[1:]) - sum(phis[:-1])
    assert abs(env._episode_shaping - expected) < 1e-9


def test_hover_bonus_fires_exactly_on_conditions():
    """The bonus fires iff |angle|, |vx|, |vy| are all under threshold."""
    eps = 1e-6
    cases = [
        (obs8(), True),                                        # hover state
        (obs8(angle=0.2 - eps), True),                         # angle boundary
        (obs8(angle=0.2 + eps), False),                        # just over
        (obs8(vx=0.3 - eps), True),                            # vx boundary
        (obs8(vx=0.3 + eps), False),
        (obs8(vy=0.3 - eps), True),                            # vy boundary
        (obs8(vy=0.3 + eps), False),
        (obs8(x=1.5, y=2.0), True),  # exploitable: anywhere, position ignored
        (obs8(angle=0.5), False),       # tilted
        (obs8(vx=1.0, vy=1.0), False),  # drifting fast
    ]
    scripted = [obs8()] + [case for case, _ in cases]
    env = ShapingRewardWrapper(
        ScriptedEnv(scripted, reward=0.0, length=len(scripted) + 1),
        mode="hover_bonus",
    )
    env.reset()
    for (_, expected_fire) in cases:
        _, reward, _, _, info = env.step(0)
        assert info["shaping"] == (1.0 if expected_fire else 0.0)
        assert reward == info["shaping"]  # original reward was 0.0


def test_time_penalty_and_episode_accumulators():
    """time_penalty subtracts a constant and accumulates per episode."""
    env = ShapingRewardWrapper(
        ScriptedEnv([obs8()], reward=1.0, length=10),
        mode="time_penalty",
        time_penalty=0.05,
    )
    env.reset()
    for step in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == pytest.approx(1.0 - 0.05)
        assert info["original_reward"] == 1.0
        assert info["shaping"] == pytest.approx(-0.05)
        if terminated:
            assert info["episode_original"] == pytest.approx(10.0)
            assert info["episode_shaping"] == pytest.approx(-0.5)
    assert terminated and not truncated


def test_unknown_mode_raises():
    """An unknown mode fails loudly rather than silently passing through."""
    with pytest.raises(ValueError, match="unknown reward mode"):
        ShapingRewardWrapper(ScriptedEnv([obs8()]), mode="not_a_mode")


def test_all_modes_are_tested_constants():
    """MODES stays in sync with the modes the lab trains."""
    assert MODES == ("builtin", "time_penalty", "hover_bonus", "potential_shaped")
