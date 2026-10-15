"""Evaluate trained PPO agents on LunarLander-v3 with the original reward.

Runs a fixed number of deterministic episodes per seed on the *unmodified*
built-in reward and records, for every episode, the total reward, length,
outcome (landed / crashed / timeout) and the engine-fuel proxy. Episode
initial states come from a fixed seed sequence shared by every model, so
different models (or a later reward-shaped variant) are compared on the same
set of initial states.

Per-seed results are written to ``results/evaluations/ppo_seed_<S>_<src>.json``
for ``scripts/aggregate.py`` to roll up.

Example:

    .venv\\Scripts\\python.exe scripts\\evaluate.py --seeds 0 1 2 --episodes 100
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

ENV_ID = "LunarLander-v3"
RESULTS_DIR = "results"
DEFAULT_EPISODES = 100
# Every episode is reset with seed_base + episode index, so all models see
# the same 100 initial states (paired comparison across seeds/models).
SEED_BASE = 1000
MAIN_ENGINE_COST = 0.30  # per frame the main engine fires (from env source)
SIDE_ENGINE_COST = 0.03  # per frame a side engine fires (from env source)

OUTCOME_LANDED = "landed"
OUTCOME_CRASHED = "crashed"
OUTCOME_TIMEOUT = "timeout"


def find_model(seed: int, model_source: str) -> str:
    """Return the path to the requested model artifact for ``seed``."""
    run_dir = os.path.join(RESULTS_DIR, "runs", f"seed_{seed}")
    if model_source == "best":
        path = os.path.join(run_dir, "best", "best_model.zip")
    else:
        path = os.path.join(run_dir, "models", f"ppo_lander_seed{seed}.zip")
    if not os.path.exists(path):
        raise SystemExit(f"Model not found for seed {seed} ({model_source}): {path}")
    return path


def run_episode(model: PPO, env: gym.Env, episode_seed: int) -> dict:
    """Run one deterministic episode and return its measurements."""
    obs, _ = env.reset(seed=episode_seed)
    total_reward = 0.0
    length = 0
    main_firings = 0
    side_firings = 0
    last_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        last_reward = float(reward)
        length += 1
        # Discrete actions: 0 nothing, 1 left engine, 2 main engine, 3 right.
        if action == 2:
            main_firings += 1
        elif action in (1, 3):
            side_firings += 1

    if truncated:
        outcome = OUTCOME_TIMEOUT
    elif last_reward > 0:
        # The terminating step pays exactly +100 (landed) or -100 (crashed);
        # its sign discriminates the outcome even if a small shaping term is
        # added on the same step.
        outcome = OUTCOME_LANDED
    else:
        outcome = OUTCOME_CRASHED
    return {
        "reward": total_reward,
        "length": length,
        "outcome": outcome,
        "fuel_cost": MAIN_ENGINE_COST * main_firings + SIDE_ENGINE_COST * side_firings,
    }
def summarise(episodes: list[dict]) -> dict:
    """Return summary statistics over a list of episode measurements."""
    rewards = np.array([ep["reward"] for ep in episodes], dtype=float)
    lengths = np.array([ep["length"] for ep in episodes], dtype=float)
    fuels = np.array([ep["fuel_cost"] for ep in episodes], dtype=float)
    outcomes = [ep["outcome"] for ep in episodes]
    n = len(episodes)
    return {
        "episodes": n,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=1)) if n > 1 else 0.0,
        "stderr_reward": float(rewards.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "median_reward": float(np.median(rewards)),
        "mean_length": float(lengths.mean()),
        "mean_fuel_cost": float(fuels.mean()),
        # Fraction of episodes above the classic "solved" bar of 200 points.
        "solved_rate_ge200": float((rewards >= 200.0).mean()),
        "landed_rate": float(np.mean([o == OUTCOME_LANDED for o in outcomes])),
        "crash_rate": float(np.mean([o == OUTCOME_CRASHED for o in outcomes])),
        "timeout_rate": float(np.mean([o == OUTCOME_TIMEOUT for o in outcomes])),
    }


def evaluate_seed(seed: int, model_source: str, episodes: int, out_dir: str) -> dict:
    """Evaluate one seed's model for ``episodes`` episodes and save a JSON."""
    model_path = find_model(seed, model_source)
    model = PPO.load(model_path, device="cpu")
    env = gym.make(ENV_ID)
    results = [run_episode(model, env, SEED_BASE + i) for i in range(episodes)]
    env.close()

    payload = {
        "env_id": ENV_ID,
        "seed": seed,
        "model_source": model_source,
        "model_path": model_path,
        "deterministic": True,
        "episode_seed_base": SEED_BASE,
        "episodes": results,
        "summary": summarise(results),
    }
    out_path = os.path.join(out_dir, f"ppo_seed_{seed}_{model_source}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    summary = payload["summary"]
    print(
        f"Seed {seed} ({model_source}): mean reward {summary['mean_reward']:+.2f} "
        f"+/- {summary['stderr_reward']:.2f} SEM over {episodes} episodes | "
        f"landed {summary['landed_rate'] * 100:.0f}% | "
        f"crash {summary['crash_rate'] * 100:.0f}% | "
        f"timeout {summary['timeout_rate'] * 100:.0f}% | "
        f"mean length {summary['mean_length']:.0f}"
    )
    print(f"Saved: {out_path}")
    return payload


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="seeds to evaluate (default: all seeds found under results/runs)",
    )
    parser.add_argument(
        "--model",
        choices=["final", "best"],
        default="final",
        help="which model artifact to evaluate (default: final)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"episodes per seed (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(RESULTS_DIR, "evaluations"),
        help="directory for the per-seed JSON results",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate every requested seed and write per-seed JSON results."""
    args = parse_args()
    torch.set_num_threads(1)

    seeds = args.seeds
    if seeds is None:
        pattern = os.path.join(RESULTS_DIR, "runs", "seed_*")
        seeds = []
        for path in sorted(glob.glob(pattern)):
            match = re.search(r"seed_(\d+)$", path)
            if match:
                seeds.append(int(match.group(1)))
        if not seeds:
            raise SystemExit(f"No runs found under {pattern}")
        print(f"Found seeds: {seeds}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"Evaluating {args.model} models on {ENV_ID} (original reward), "
        f"{args.episodes} deterministic episodes each"
    )
    for seed in seeds:
        evaluate_seed(seed, args.model, args.episodes, args.out_dir)


if __name__ == "__main__":
    main()

