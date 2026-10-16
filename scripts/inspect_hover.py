"""Diagnose a trained policy for reward hacking by hovering (Phase 2).

Loads a PPO checkpoint, runs N deterministic episodes on the ORIGINAL
(unwrapped) LunarLander-v3 environment, and reports:

* episode length (mean / max) and the timeout rate (episodes truncated at
  the 1000-step limit -- the signature of hover farming),
* landing / crash rates (from the sign of the +-100 termination reward),
* mean original (built-in) reward,
* the fraction of frames the lander spends in the hover state
  (|angle| < angle_thresh and |vx|,|vy| < speed_thresh), plus the mean
  length of consecutive hover runs.

Example::

    .venv\\Scripts\\python.exe scripts\\inspect_hover.py --episodes 20 \\
        --model results/runs/hover_bonus/seed_0/models/ppo_lander_seed0.zip
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

# Make src/ importable when this script is run directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.reward_wrappers import ANGLE, VX, VY

EPISODE_LIMIT = 1000
SEED_BASE = 1000


def hover_mask(observations: np.ndarray, angle_thresh: float, speed_thresh: float) -> np.ndarray:
    """Boolean mask: True for frames in the hover state (upright and slow).

    Mirrors ShapingRewardWrapper.is_hovering so diagnostics use the same
    definition as the reward being diagnosed.
    """
    return (
        (np.abs(observations[:, ANGLE]) < angle_thresh)
        & (np.abs(observations[:, VX]) < speed_thresh)
        & (np.abs(observations[:, VY]) < speed_thresh)
    )


def mean_hover_run_length(mask: np.ndarray) -> float:
    """Mean length of consecutive True runs in a boolean mask."""
    if mask.size == 0 or not mask.any():
        return 0.0
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = edges[1::2] - edges[::2]
    return float(runs.mean())


def run_episodes(
    model,
    env: gym.Env,
    n_episodes: int,
    angle_thresh: float,
    speed_thresh: float,
) -> list[dict]:
    """Run n_episodes deterministic episodes; return per-episode records."""
    records = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=SEED_BASE + episode)
        frames = []
        total_reward, length, outcome = 0.0, 0, "timeout"
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(np.asarray(obs, dtype=np.float64))
            total_reward += float(reward)
            length += 1
            if terminated:
                outcome = "landed" if reward > 0 else "crashed"
        mask = hover_mask(np.array(frames), angle_thresh, speed_thresh)
        records.append(
            {
                "episode": episode,
                "reward": total_reward,
                "length": length,
                "outcome": outcome,
                "hover_fraction": float(mask.mean()),
                "hover_run_length": mean_hover_run_length(mask),
            }
        )
    return records


def compute_stats(records: list[dict]) -> dict:
    """Pool per-episode records into summary statistics."""
    lengths = np.array([r["length"] for r in records])
    rewards = np.array([r["reward"] for r in records])
    outcomes = [r["outcome"] for r in records]
    return {
        "episodes": len(records),
        "mean_reward": float(rewards.mean()),
        "sem_reward": float(rewards.std(ddof=1) / np.sqrt(len(records))) if len(records) > 1 else 0.0,
        "mean_length": float(lengths.mean()),
        "max_length": int(lengths.max()),
        "timeout_rate": float(np.mean(lengths >= EPISODE_LIMIT)),
        "landed_rate": float(np.mean([o == "landed" for o in outcomes])),
        "crashed_rate": float(np.mean([o == "crashed" for o in outcomes])),
        "hover_fraction": float(np.mean([r["hover_fraction"] for r in records])),
        "hover_run_length": float(np.mean([r["hover_run_length"] for r in records])),
    }


def print_stats(name: str, stats: dict) -> None:
    """Print one diagnostics block."""
    print(f"\n{name} ({stats['episodes']} episodes)")
    print(f"  original reward : {stats['mean_reward']:+.1f} +/- {stats['sem_reward']:.1f}")
    print(f"  episode length  : mean {stats['mean_length']:.0f}, max {stats['max_length']}")
    print(
        f"  outcomes        : landed {stats['landed_rate']:.0%}, "
        f"crashed {stats['crashed_rate']:.0%}, timeout {stats['timeout_rate']:.0%}"
    )
    print(
        f"  hover state     : {stats['hover_fraction']:.0%} of frames, "
        f"mean run {stats['hover_run_length']:.0f} frames"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a PPO .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--angle", type=float, default=0.2, help="hover |angle| threshold (rad)")
    parser.add_argument("--speed", type=float, default=0.3, help="hover |vx|,|vy| threshold")
    args = parser.parse_args()

    from stable_baselines3 import PPO  # heavy import kept out of helper scope

    model = PPO.load(args.model, device="cpu")
    env = gym.make("LunarLander-v3")
    records = run_episodes(model, env, args.episodes, args.angle, args.speed)
    env.close()
    stats = compute_stats(records)
    print_stats(os.path.basename(args.model), stats)


if __name__ == "__main__":
    main()
