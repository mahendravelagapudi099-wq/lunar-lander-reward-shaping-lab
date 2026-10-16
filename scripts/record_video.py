"""Record deterministic landing videos of the trained PPO agents.

For each requested seed, loads the final (or best) model and records one
deterministic episode on LunarLander-v3, saved to
``videos/<mode>_seed<S>.mp4``. The episode uses the same initial state as
evaluation episode 0 (``SEED_BASE`` in ``scripts/evaluate.py``) so the video
shows a representative, reproducible landing. The episode plays out on the
ORIGINAL reward (no shaping wrapper), exactly like evaluation.

The baseline models live under ``results/runs/seed_<S>/``; a shaped mode
lives under ``results/runs/<mode>/seed_<S>/`` (train.py's per-mode layout),
selected with ``--mode``:

    .venv\\Scripts\\python.exe scripts\\record_video.py --seeds 0 1 2
    .venv\\Scripts\\python.exe scripts\\record_video.py --mode hover_bonus --seeds 0 1 2

(The Phase 1 baseline videos ``videos/ppo_lander_seed{0,1,2}.mp4`` keep their
original legacy names; ``--mode baseline`` writes the uniform
``baseline_seed<S>.mp4`` names.)
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

ENV_ID = "LunarLander-v3"
RESULTS_DIR = "results"
VIDEO_DIR = "videos"
BASELINE = "baseline"
MODES = ("baseline", "time_penalty", "hover_bonus", "potential_shaped")
SEED_BASE = 1000  # same initial state as evaluate.py's first episode


def find_model(seed: int, model_source: str, mode: str = BASELINE) -> str:
    """Return the path to the requested model artifact for ``seed``.

    The baseline lives under ``results/runs/seed_<S>/``; a shaped mode lives
    under ``results/runs/<mode>/seed_<S>/`` (train.py's per-mode layout), the
    same resolution as ``scripts/evaluate.py``.
    """
    run_dir = os.path.join(RESULTS_DIR, "runs", f"seed_{seed}")
    if mode != BASELINE:
        run_dir = os.path.join(RESULTS_DIR, "runs", mode, f"seed_{seed}")
    if model_source == "best":
        path = os.path.join(run_dir, "best", "best_model.zip")
    else:
        path = os.path.join(run_dir, "models", f"ppo_lander_seed{seed}.zip")
    if not os.path.exists(path):
        raise SystemExit(f"Model not found for seed {seed} ({model_source}, mode={mode}): {path}")
    return path


def record_seed(seed: int, model_source: str, mode: str = BASELINE) -> str | None:
    """Record one deterministic episode for ``seed`` and return its path."""
    model_path = find_model(seed, model_source, mode)
    video_name = f"{mode}_seed{seed}"
    model = PPO.load(model_path, device="cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = RecordVideo(
        env,
        VIDEO_DIR,
        name_prefix=video_name,
        episode_trigger=lambda episode: episode == 0,
        disable_logger=True,
    )

    obs, _ = env.reset(seed=SEED_BASE)
    total_reward = 0.0
    length = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        length += 1
    env.close()

    # RecordVideo writes "<prefix>-episode-0.mp4"; rename to the clean name.
    produced = os.path.join(VIDEO_DIR, f"{video_name}-episode-0.mp4")
    target = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
    if os.path.exists(produced):
        os.replace(produced, target)
    print(
        f"Seed {seed} ({mode}/{model_source}): recorded 1 episode, "
        f"original reward {total_reward:+.1f}, length {length} -> {target}"
    )
    return target if os.path.exists(target) else None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="seeds to record (default: 0 1 2)",
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default=BASELINE,
        help="reward mode whose runs to record (baseline = results/runs/seed_<S>)",
    )
    parser.add_argument(
        "--model",
        choices=["final", "best"],
        default="final",
        help="which model artifact to record (default: final)",
    )
    return parser.parse_args()


def main() -> None:
    """Record one video per requested seed."""
    args = parse_args()
    torch.set_num_threads(1)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    for seed in args.seeds:
        record_seed(seed, args.model, args.mode)


if __name__ == "__main__":
    main()
