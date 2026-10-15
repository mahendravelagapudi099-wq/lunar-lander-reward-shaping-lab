"""Train the PPO baseline on LunarLander-v3 (config-driven, resumable).

One invocation trains a single seed. Run three seeds as three parallel
processes, e.g.:

    .venv\\Scripts\\python.exe scripts\\train.py --seed 0
    .venv\\Scripts\\python.exe scripts\\train.py --seed 1
    .venv\\Scripts\\python.exe scripts\\train.py --seed 2

Re-running the same seed with ``--resume`` picks up from the latest
checkpoint (or the final model) and continues training until
``total_timesteps`` is reached.

Per-seed layout under the run directory::

    results/runs/seed_<S>/
        models/                 checkpoints + final model
        monitor/                per-env Monitor CSVs (episode reward/length)
        eval/                   evaluations.npz from the EvalCallback
        best/                   best model seen during evaluation
    TensorBoard logs go to results/tb/ppo_seed_<S>/.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time
import warnings

import gymnasium as gym
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

CONFIG_PATH = os.path.join("configs", "ppo_baseline.yaml")
RESULTS_DIR = "results"


def load_config(path: str) -> dict:
    """Load the YAML training config."""
    with open(path) as f:
        return yaml.safe_load(f)


def find_resume_checkpoint(model_dir: str, seed: int) -> str | None:
    """Return the path to resume from: final model, else newest checkpoint."""
    final = os.path.join(model_dir, f"ppo_lander_seed{seed}.zip")
    if os.path.exists(final):
        return final
    pattern = os.path.join(model_dir, f"ppo_lander_seed{seed}_*_steps.zip")
    checkpoints = []
    for path in glob.glob(pattern):
        match = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    return max(checkpoints)[1]

def main() -> None:
    """Train one seed of the PPO baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed: int = args.seed if args.seed is not None else cfg["seed"]
    total_timesteps: int = (
        args.total_timesteps
        if args.total_timesteps is not None
        else cfg["total_timesteps"]
    )
    n_envs: int = cfg["n_envs"]
    torch.set_num_threads(int(cfg["torch_threads"]))

    run_dir = os.path.join(RESULTS_DIR, "runs", f"seed_{seed}")
    model_dir = os.path.join(run_dir, "models")
    monitor_dir = os.path.join(run_dir, "monitor")
    eval_dir = os.path.join(run_dir, "eval")
    best_dir = os.path.join(run_dir, "best")
    for directory in (model_dir, monitor_dir, eval_dir, best_dir):
        os.makedirs(directory, exist_ok=True)

    vec_env_cls = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[cfg["vec_env"]]
    env = make_vec_env(
        cfg["env_id"],
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        vec_env_cls=vec_env_cls,
    )
    # Single-env evaluation on the original reward, deterministic, no render.
    eval_env = Monitor(
        gym.make(cfg["env_id"]),
        filename=os.path.join(eval_dir, "eval_monitor"),
    )

    hparams = dict(cfg["hyperparameters"])
    policy = hparams.pop("policy", "MlpPolicy")

    if args.resume:
        resume_path = find_resume_checkpoint(model_dir, seed)
        if resume_path is None:
            raise SystemExit(
                f"--resume requested but no checkpoint found under {model_dir}"
            )
        model = PPO.load(resume_path, env=env, device="cpu")
        done_steps = model.num_timesteps
        print(f"Resumed from {resume_path} ({done_steps:,} steps done)")
    else:
        model = PPO(
            policy=policy,
            env=env,
            seed=seed,
            device="cpu",
            verbose=1,
            tensorboard_log=os.path.join(RESULTS_DIR, "tb"),
            **hparams,
        )
        done_steps = 0

    remaining = max(total_timesteps - done_steps, 0)
    if remaining == 0:
        print(f"Seed {seed}: already reached {total_timesteps:,} steps, nothing to do.")
        env.close()
        return

    # Eval/checkpoint frequencies count vec env step() calls, not env steps.
    eval_freq = max(cfg["eval"]["freq_timesteps"] // n_envs, 1)
    save_freq = max(cfg["checkpoint"]["freq_timesteps"] // n_envs, 1)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=cfg["eval"]["episodes"],
        eval_freq=eval_freq,
        log_path=eval_dir,
        best_model_save_path=best_dir,
        deterministic=True,
        render=False,
        warn=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix=f"ppo_lander_seed{seed}",
    )

    print(
        f"Seed {seed}: training for {remaining:,} steps "
        f"({done_steps:,} -> {total_timesteps:,}), torch threads "
        f"{torch.get_num_threads()}, vec env: {cfg['vec_env']} x {n_envs}"
    )
    start = time.perf_counter()
    # Single-env eval is always a DummyVecEnv, so SB3 warns when the training
    # env is subprocess-based. The mismatch is intentional (a subprocess for
    # one sequential eval env would be pointless), so filter just this message.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Training and eval env are not of the same type",
            category=UserWarning,
        )
        model.learn(
            total_timesteps=remaining,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name=f"ppo_seed{seed}",
            reset_num_timesteps=not args.resume,
        )
    elapsed = time.perf_counter() - start

    final_path = os.path.join(model_dir, f"ppo_lander_seed{seed}.zip")
    model.save(final_path)
    env.close()

    steps_done = model.num_timesteps
    steps_per_sec = steps_done / max(elapsed, 1e-9)
    print(
        f"Seed {seed}: done. steps={steps_done:,} wall={elapsed / 60.0:.1f} min "
        f"({steps_per_sec:.1f} steps/s over this call)"
    )
    print(f"Final model: {final_path}")
    print(f"Best model:  {os.path.join(best_dir, 'best_model.zip')}")
    print(f"Evaluation history: {os.path.join(eval_dir, 'evaluations.npz')}")


if __name__ == "__main__":
    main()

