"""Reward shaping wrappers for the LunarLander lab (Phase 2).

A single :class:`ShapingRewardWrapper` sits between the raw environment and
the training algorithm. On every step it takes the built-in reward ``r`` and
returns ``r + shaping``, while recording in ``info``:

* ``original_reward`` -- the built-in reward on this step (before shaping),
* ``shaping``         -- the shaping term added on this step,
* ``episode_original`` / ``episode_shaping`` -- running totals, emitted on
  the terminal/truncating step so that Monitor (via ``info_keywords``) can
  log both per-episode totals in its CSV.

Wrapper order matters (verified in ``stable_baselines3/common/env_util.py``):
``make_vec_env`` applies ``Monitor`` FIRST and any ``wrapper_class`` on top,
i.e. ``wrapper_class(Monitor(env))``. Passing this wrapper as
``wrapper_class`` therefore makes Monitor log the ORIGINAL reward while the
agent trains on the shaped one. To log BOTH rewards, build the stack
explicitly as
``Monitor(ShapingRewardWrapper(gym.make(...)), info_keywords=("episode_original", "episode_shaping"))``:
then the CSV column ``r`` is the SHAPED reward and the keyword columns carry
the per-episode original reward and shaping total. Whichever order is used,
the per-step and per-episode totals in ``info`` always let the original
reward be recovered and evaluated -- evaluation itself never uses this
wrapper.

Modes
-----
``builtin``
    No change (control arm; identical hyperparameters otherwise).
``time_penalty``
    Subtract ``time_penalty`` every frame (start: 0.05).
``hover_bonus``
    Add ``hover_bonus`` every frame in which the lander is upright and slow:
    ``|angle| < hover_angle`` and ``|vx|, |vy| < hover_speed``. The condition
    ignores WHERE the lander is, so the bonus can be farmed by hovering in
    place anywhere above the pad -- deliberately exploitable. Thresholds
    come from measured percentiles of a converged baseline policy's
    trajectories (|angle| p95 ~= 0.21 rad, |vx| p95 ~= 0.33, |vy| p95 ~=
    0.30 over all frames; a good descent satisfies the criterion ~87% of the
    time, so the bonus is discoverable AND exploitable).
``potential_shaped``
    Add the potential-based term ``gamma * Phi(s') - Phi(s)`` with
    ``Phi = -(dist_w * ||(x, y)|| + speed_w * ||(vx, vy)|| + tilt_w * |angle|)``
    and subtract a small constant ``potential_cost`` per frame.

Why the potential form cannot be farmed by holding still
--------------------------------------------------------
Summed over an episode, the potential terms telescope: for a constant
state (a perfect hover), each term equals ``(gamma - 1) * Phi``, i.e. the
total grows as ``(1 - gamma) * T * |Phi|`` and never per-frame as a flat
bonus does. With ``gamma = 0.999`` and ``|Phi| ~ 2``, hovering for the full
1,000-step episode yields only ~2 points of shaping, versus ~1,000 from the
flat hover bonus. Phi is also bounded (distances/speeds/angles are bounded),
so the shaped contribution stays small no matter how long the episode runs.
The term pays for *improvement* (moving closer to the pad, slowing down,
levelling out), never for merely being in a good state.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

MODES = ("builtin", "time_penalty", "hover_bonus", "potential_shaped")

# Observation indices of LunarLander-v3 (8-dim).
X, Y, VX, VY, ANGLE = 0, 1, 2, 3, 4


class ShapingRewardWrapper(gym.Wrapper):
    """Apply a configurable reward-shaping mode on top of the built-in reward.

    The built-in reward is always preserved in ``info["original_reward"]``
    and accumulated per episode, so the original reward can be logged and
    evaluated even while training on the shaped reward.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "builtin",
        gamma: float = 0.999,
        *,
        time_penalty: float = 0.05,
        hover_bonus: float = 1.0,
        hover_angle: float = 0.2,
        hover_speed: float = 0.3,
        potential_dist_w: float = 1.0,
        potential_speed_w: float = 2.0,
        potential_tilt_w: float = 1.0,
        potential_cost: float = 0.01,
    ) -> None:
        super().__init__(env)
        if mode not in MODES:
            raise ValueError(f"unknown reward mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.gamma = float(gamma)
        self.time_penalty = float(time_penalty)
        self.hover_bonus = float(hover_bonus)
        self.hover_angle = float(hover_angle)
        self.hover_speed = float(hover_speed)
        self.potential_dist_w = float(potential_dist_w)
        self.potential_speed_w = float(potential_speed_w)
        self.potential_tilt_w = float(potential_tilt_w)
        self.potential_cost = float(potential_cost)
        self._prev_potential: float | None = None
        self._episode_original = 0.0
        self._episode_shaping = 0.0

    # ------------------------------------------------------------------ #
    # Shaping computations (pure functions of the observation)           #
    # ------------------------------------------------------------------ #

    def potential(self, obs: np.ndarray) -> float:
        """Potential Phi(state): negative weighted distance-to-pad-state.

        Phi is <= 0 and approaches 0 as the lander nears the pad, slows down
        and levels out. Bounded by construction, so potential-based shaping
        terms cannot grow without bound (see module docstring).
        """
        dist = float(np.hypot(obs[X], obs[Y]))
        speed = float(np.hypot(obs[VX], obs[VY]))
        tilt = abs(float(obs[ANGLE]))
        return -(
            self.potential_dist_w * dist
            + self.potential_speed_w * speed
            + self.potential_tilt_w * tilt
        )

    def is_hovering(self, obs: np.ndarray) -> bool:
        """Upright and slow, regardless of position (the exploitable part)."""
        return (
            abs(float(obs[ANGLE])) < self.hover_angle
            and abs(float(obs[VX])) < self.hover_speed
            and abs(float(obs[VY])) < self.hover_speed
        )

    def _shaping_term(self, obs: np.ndarray, done: bool) -> float:
        """Return the shaping term for the transition INTO ``obs``."""
        if self.mode == "builtin":
            return 0.0
        if self.mode == "time_penalty":
            return -self.time_penalty
        if self.mode == "hover_bonus":
            return self.hover_bonus if self.is_hovering(obs) else 0.0
        # potential_shaped
        assert self._prev_potential is not None, "reset() must be called first"
        term = self.gamma * self.potential(obs) - self._prev_potential
        if done:
            # Terminal states are not carried forward: drop the running
            # potential so a later (buggy) extra step cannot reuse it.
            self._prev_potential = None
        else:
            self._prev_potential = self.potential(obs)
        return term - self.potential_cost

    # ------------------------------------------------------------------ #
    # gym API                                                            #
    # ------------------------------------------------------------------ #

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self.potential(obs)
        self._episode_original = 0.0
        self._episode_shaping = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        shaping = self._shaping_term(obs, done)
        self._episode_original += float(reward)
        self._episode_shaping += float(shaping)
        # Copy so the raw env's info dict is never mutated.
        info = dict(info)
        info["original_reward"] = float(reward)
        info["shaping"] = float(shaping)
        if done:
            info["episode_original"] = self._episode_original
            info["episode_shaping"] = self._episode_shaping
        return obs, float(reward) + shaping, terminated, truncated, info


def wrapper_from_config(env: gym.Env, reward_cfg: dict, gamma: float) -> ShapingRewardWrapper:
    """Build a ShapingRewardWrapper from the config's ``reward`` block.

    Keeps every parameter in one place (the YAML) while the wrapper itself
    stays config-agnostic and unit-testable.
    """
    params = dict(reward_cfg.get("params", {}))
    return ShapingRewardWrapper(
        env,
        mode=reward_cfg["mode"],
        gamma=gamma,
        time_penalty=params.get("time_penalty", 0.05),
        hover_bonus=params.get("hover_bonus", 1.0),
        hover_angle=params.get("hover_angle", 0.2),
        hover_speed=params.get("hover_speed", 0.3),
        potential_dist_w=params.get("potential_dist_w", 1.0),
        potential_speed_w=params.get("potential_speed_w", 2.0),
        potential_tilt_w=params.get("potential_tilt_w", 1.0),
        potential_cost=params.get("potential_cost", 0.01),
    )
