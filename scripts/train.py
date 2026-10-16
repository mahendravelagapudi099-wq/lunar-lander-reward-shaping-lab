"""Train PPO on LunarLander-v3 (config-driven, resumable, reward-mode aware).

One invocation trains a single seed. Run three seeds as three parallel
processes, e.g.:

    .venv\\Scripts\\python.exe scripts\\train.py --seed 0
    .venv\\Scripts\\python.exe scripts\\train.py --seed 1
    .venv\\Scripts\\python.exe scripts\\train.py --seed 2

For a shaped reward (Phase 2), pass ``--mode`` (or set ``reward.mode`` in
the config); the run then lands under a per-mode directory::

    .venv\\Scripts\\python.exe scripts\\train.py --mode time_penalty --seed 0

Re-running the same seed with ``--resume`` picks up from the latest
checkpoint (or the final model) and continues training until
``total_timesteps`` is reached.

Per-seed layout under the run directory::

    results/runs/<mode>/seed_<S>/   (mode omitted for Phase 1 baseline runs)
        models/                 checkpoints + final model
        monitor/                per-env Monitor CSVs ('r' = SHAPED reward,
                                plus episode_original/episode_shaping columns)
        eval/                   evaluations.npz from the EvalCallback
                                (always on the ORIGINAL, unshaped reward)
        best/                   best model seen during evaluation
        summary.json            run metadata (hyperparameters, wall time, ...)
    TensorBoard logs go to results/tb/ppo_<mode>_seed_<S>/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings

import gymnasium as gym
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Make src/ importable when this script is run directly (python scripts/train.py).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.reward_wrappers import MODES, wrapper_from_config

CONFIG_PATH = os.path.join("configs", "ppo_baseline.yaml")
RESULTS_DIR = "results"
# Monitor keyword columns logged for every mode (see src/reward_wrappers.py):
# 'r' in the CSV is the SHAPED reward; these columns carry the original.
MONITOR_INFO_KEYWORDS = ("episode_original", "episode_shaping")


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


def resolve_reward_cfg(args: argparse.Namespace, cfg: dict) -> dict:
    """Return the reward config whose mode is the *effective* training mode.

    The mode comes from ``--mode`` when it is given, otherwise from the
    config's ``reward.mode`` (``builtin`` when the block is absent). The
    resolved mode is always written into the returned config, because
    :func:`src.reward_wrappers.wrapper_from_config` builds the training
    wrapper from that config: letting the banner mode and the wrapper mode
    disagree previously trained ``--mode hover_bonus`` on the *builtin*
    reward (byte-identical policy weights to baseline, zero shaping in the
    monitor logs, wasted 3 x 25 min of training).
    """
    reward_cfg = dict(cfg.get("reward", {"mode": "builtin"}))
    if args.mode is not None:
        reward_cfg["mode"] = args.mode
    mode = reward_cfg.get("mode", "builtin")
    if mode not in MODES:
        raise SystemExit(f"unknown reward mode {mode!r}; expected one of {MODES}")
    return reward_cfg


def main() -> None:
    """Train one seed of the PPO baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument(
        "--mode",
        default=None,
        choices=MODES,
        help="reward mode (default: the config's reward.mode, or builtin)",
    )
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

    reward_cfg = resolve_reward_cfg(args, cfg)
    mode: str = reward_cfg["mode"]
    gamma = float(cfg["hyperparameters"]["gamma"])

    run_dir = os.path.join(RESULTS_DIR, "runs", mode, f"seed_{seed}")
    model_dir = os.path.join(run_dir, "models")
    monitor_dir = os.path.join(run_dir, "monitor")
    eval_dir = os.path.join(run_dir, "eval")
    best_dir = os.path.join(run_dir, "best")
    for directory in (model_dir, monitor_dir, eval_dir, best_dir):
        os.makedirs(directory, exist_ok=True)

    vec_env_cls = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[cfg["vec_env"]]

    # Training env: Monitor OUTSIDE the shaping wrapper, so that the Monitor
    # CSV's 'r' column is the SHAPED reward (what the agent optimizes) while
    # the info_keywords columns record the per-episode ORIGINAL reward and
    # shaping total. Seeding mirrors stable_baselines3's make_vec_env exactly
    # (action_space.seed(seed + rank) per env, then vec_env.seed(seed)).
    def make_env(rank: int):
        def _init() -> gym.Env:
            env = gym.make(cfg["env_id"], render_mode="rgb_array")
            env = wrapper_from_config(env, reward_cfg, gamma)
            env = Monitor(
                env,
                filename=os.path.join(monitor_dir, str(rank)),
                info_keywords=MONITOR_INFO_KEYWORDS,
            )
            if seed is not None:
                env.action_space.seed(seed + rank)
            return env

        return _init

    env = vec_env_cls([make_env(i) for i in range(n_envs)])
    if seed is not None:
        env.seed(seed)
    # Single-env evaluation on the original reward, deterministic, no render:
    # the eval env is NEVER wrapped with the shaping wrapper.
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
        f"Seed {seed} (mode={mode}): training for {remaining:,} steps "
        f"({done_steps:,} -> {total_timesteps:,}), torch threads "
        f"{torch.get_num_threads()}, vec env: {cfg['vec_env']} x {n_envs}"
    )
    est_sps = float(cfg.get("est_steps_per_sec", 700.0))
    print(
        f"Estimated wall time: ~{remaining / est_sps / 60.0:.0f} min "
        f"at ~{est_sps:.0f} steps/s (measured on this machine)"
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
            tb_log_name=f"ppo_{mode}_seed{seed}",
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

    summary = {
        "seed": seed,
        "reward": {
            "mode": mode,
            "params": reward_cfg.get("params", {}),
            "monitor_logs": "r = shaped reward; episode_original/episode_shaping columns",
            "eval_reward": "original (the shaping wrapper is never used for evaluation)",
        },
        "env_id": cfg["env_id"],
        "algorithm": "PPO",
        "hyperparameters": cfg["hyperparameters"],
        "vec_env": cfg["vec_env"],
        "n_envs": n_envs,
        "torch_threads": int(cfg["torch_threads"]),
        "total_timesteps_requested": total_timesteps,
        "timesteps_done": int(steps_done),
        "wall_minutes_this_call": round(elapsed / 60.0, 2),
        "steps_per_second": round(steps_per_sec, 1),
        "resumed": bool(args.resume),
        "resumed_from_steps": int(done_steps) if args.resume else 0,
        "final_model": final_path,
        "best_model": os.path.join(best_dir, "best_model.zip"),
        "evaluations": os.path.join(eval_dir, "evaluations.npz"),
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:     {summary_path}")


if __name__ == "__main__":
    main()

