"""Plot PPO learning curves for the baseline runs.

Reads the EvalCallback histories (``results/runs/seed_*/eval/evaluations.npz``
— deterministic evaluation on the original reward every 50k steps) and renders
one PNG with a line per seed plus a shaded standard-error band, plus a
smoothed version of each curve.

Output: ``results/figures/learning_curves.png``.

Example:

    .venv\\Scripts\\python.exe scripts\\plot_curves.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
OUT_NAME = "learning_curves.png"
# Number of evaluation points for the moving average (== 100k steps of data).
SMOOTH_WINDOW = 3
SEED_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")


def load_eval_curves() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {seed: (timesteps, mean eval reward per checkpoint)}."""
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    pattern = os.path.join(RESULTS_DIR, "runs", "seed_*", "eval", "evaluations.npz")
    for path in sorted(glob.glob(pattern)):
        match = re.search(r"seed_(\d+)", path)
        if match is None:
            continue
        data = np.load(path)
        timesteps = np.asarray(data["timesteps"], dtype=float)
        # results shape: (n_evals, n_episodes)
        means = np.asarray(data["results"], dtype=float).mean(axis=1)
        curves[int(match.group(1))] = (timesteps, means)
    if not curves:
        raise SystemExit(f"No evaluation curves found under {pattern}")
    return curves


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Simple centred moving average (edges shrink the window)."""
    if window <= 1 or values.size < 3:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def main() -> None:
    """Render the learning-curve figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=FIGURES_DIR, help="directory for the output PNG"
    )
    args = parser.parse_args()

    curves = load_eval_curves()
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for index, seed in enumerate(sorted(curves)):
        timesteps, means = curves[seed]
        color = SEED_COLORS[index % len(SEED_COLORS)]
        ax.plot(
            timesteps,
            smooth(means, SMOOTH_WINDOW),
            color=color,
            lw=2,
            label=f"seed {seed} (smoothed)",
        )
        ax.plot(timesteps, means, color=color, lw=1, alpha=0.35)
        plotted += 1

    ax.axhline(200.0, color="k", ls="--", lw=1, alpha=0.6)
    ax.text(
        timesteps[-1] * 0.02,
        205,
        "solved bar (200)",
        fontsize=8,
        alpha=0.7,
    )
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean evaluation reward (10 episodes, deterministic)")
    ax.set_title(f"PPO baseline on LunarLander-v3 ({plotted} seeds x 1M steps)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, OUT_NAME)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
