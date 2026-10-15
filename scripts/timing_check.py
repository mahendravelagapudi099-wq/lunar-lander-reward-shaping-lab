"""Measure PPO training throughput on LunarLander-v3 for CPU training.

Trains PPO for a fixed number of timesteps with 1, 8 and 16 parallel
environments (``make_vec_env``) and reports steps per second, then estimates
the wall time for the full 1,000,000-step baseline run. The hyperparameters
are the tuned RL Baselines3 Zoo values for LunarLander-v3
(https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml):

    n_envs: 16, n_steps: 1024, batch_size: 64, gae_lambda: 0.98,
    gamma: 0.999, n_epochs: 4, ent_coef: 0.01
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "LunarLander-v3"
ZOO_PPO_POLICY = "MlpPolicy"
ZOO_PPO_PARAMS: dict[str, object] = {
    "n_steps": 1024,
    "batch_size": 64,
    "gae_lambda": 0.98,
    "gamma": 0.999,
    "n_epochs": 4,
    "ent_coef": 0.01,
}
DEFAULT_TIMESTEPS = 50_000
DEFAULT_ENVS = (1, 8, 16)
RESULTS_DIR = os.path.join("results", "timing")


def benchmark(n_envs: int, total_timesteps: int, seed: int) -> dict[str, object]:
    """Train PPO for ``total_timesteps`` and return throughput numbers."""
    # Single-threaded torch: this machine is a 12-core big.LITTLE laptop CPU
    # and for the small (64, 64) policy the 12-thread intra-op parallelism
    # costs far more in sync overhead than it gains. This matches the setting
    # that scripts/train.py uses for the real runs.
    torch.set_num_threads(1)
    env = make_vec_env(ENV_ID, n_envs=n_envs, seed=seed)
    model = PPO(
        ZOO_PPO_POLICY,
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        **ZOO_PPO_PARAMS,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed = time.perf_counter() - start
    env.close()
    return {
        "n_envs": n_envs,
        "total_timesteps": total_timesteps,
        "wall_seconds": round(elapsed, 2),
        "steps_per_second": round(total_timesteps / elapsed, 1),
        "est_1m_steps_seconds": round(1_000_000 / (total_timesteps / elapsed), 1),
    }


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    minutes, seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"


def main() -> None:
    """Run the throughput benchmark for each environment count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--envs", type=int, nargs="+", default=list(DEFAULT_ENVS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"PPO hyperparameters (RL Baselines3 Zoo for {ENV_ID}):")
    print(f"  policy: {ZOO_PPO_POLICY}")
    for key, value in ZOO_PPO_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"  torch threads: 1 (single-threaded, see benchmark())")
    print(f"\nBenchmarking {args.timesteps:,} steps per configuration...\n")

    results = [benchmark(n_envs, args.timesteps, args.seed) for n_envs in args.envs]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "timing_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"{'n_envs':>6} | {'wall time':>10} | {'steps/sec':>9} | {'est. 1M steps':>13}")
    print("-" * 48)
    for row in results:
        print(
            f"{row['n_envs']:>6} | {format_duration(row['wall_seconds']):>10} | "
            f"{row['steps_per_second']:>9} | {format_duration(row['est_1m_steps_seconds']):>13}"
        )
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
