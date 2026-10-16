"""Compare evaluation results across reward-shaping modes (Phase 2).

Reads every per-seed evaluation JSON written by ``scripts/evaluate.py``
(shared episode seeds, always on the ORIGINAL reward, never the shaped one),
groups them via ``scripts/aggregate.py`` by reward mode and model source, and
renders one row per group:

* mean across seeds +/- the seed-level spread (the number to quote as
  "mode X reaches Y +/- Z over 3 seeds", since seeds are the replication unit)
* pooled mean over all episodes, landed / crashed / timeout rates,
  mean episode length, fuel cost and solved(>=200) rate.

The ``timeout %`` column is the hover-farming signal: timeouts only happen
when a policy hover-farms the shaping bonus until the 1000-step limit.

Outputs (all under ``results/evaluations/``):

* ``comparison.md`` / ``comparison.csv`` — the table above, one row per group
* ``comparison.png`` — mean-across-seeds bar chart with seed-level error bars
* ``paired_differences.md`` — for every non-baseline mode, the paired mean
  difference vs the baseline over the shared initial states (evaluate.py uses
  the same episode seeds for every model) with a bootstrap 95% CI; a CI
  straddling zero is stated as "within noise"
* ``learning_curves.png`` — per mode, the smoothed reward actually seen by
  the agent during training (shaped, Monitor ``r``) next to the original
  reward (Monitor ``episode_original``); a flat shaped curve with a falling
  original curve is the hover-farming signature

Example:

    .venv\\\\Scripts\\\\python.exe scripts\\\\compare.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

import aggregate

N_BOOTSTRAP = 10000

COMPARISON_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("mode", "mode", "s"),
    ("n_seeds", "seeds", "d"),
    ("n_episodes", "episodes", "d"),
    ("mean_across_seeds", "mean across seeds", "+.1f"),
    ("std_across_seeds", "+/-", ".1f"),
    ("pooled_mean_reward", "pooled mean", "+.1f"),
    ("landed_rate", "landed %", ".0%"),
    ("crash_rate", "crash %", ".0%"),
    ("timeout_rate", "timeout %", ".0%"),
    ("mean_length", "mean len", ".0f"),
    ("mean_fuel_cost", "fuel cost", ".1f"),
    ("solved_rate_ge200", "solved>=200 %", ".0%"),
)


def comparison_rows(groups: dict[str, list[dict]]) -> list[dict]:
    """Return one comparison row per (mode, model source) group."""
    rows: list[dict] = []
    for group, payloads in sorted(groups.items()):
        if not payloads:
            continue
        pooled = aggregate.pool_episodes(payloads)
        # All payloads in a group share the mode/source; take from the first
        # rather than re-parsing the "<mode>_<source>" key (modes contain `_`).
        mode = payloads[0].get("reward_mode", aggregate.BASELINE)
        model_source = payloads[0]["model_source"]
        rows.append(
            {
                "group": group,
                "mode": mode,
                "model_source": model_source,
                "seeds": [p["seed"] for p in payloads],
                "n_seeds": len(payloads),
                "n_episodes": pooled["episodes"],
                "mean_across_seeds": pooled["mean_across_seeds"],
                # One seed cannot give a seed-level spread; show 0, not NaN.
                "std_across_seeds": pooled["std_across_seeds"],
                "pooled_mean_reward": pooled["pooled_mean_reward"],
                "landed_rate": pooled["landed_rate"],
                "crash_rate": pooled["crash_rate"],
                "timeout_rate": pooled["timeout_rate"],
                "mean_length": pooled["mean_length"],
                "mean_fuel_cost": pooled["mean_fuel_cost"],
                "solved_rate_ge200": pooled["solved_rate_ge200"],
            }
        )
    return rows


def render_markdown(rows: list[dict]) -> str:
    """Render the comparison rows as a markdown table."""
    header = "| " + " | ".join(col[1] for col in COMPARISON_COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in COMPARISON_COLUMNS) + "|"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key, _, fmt in COMPARISON_COLUMNS:
            value = row[key]
            if fmt.endswith("%"):
                cells.append(format(value * 100.0, fmt[:-1] + "f"))
            else:
                cells.append(format(value, fmt))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: str) -> None:
    """Write the comparison rows as CSV (one row per group)."""
    keys = [col[0] for col in COMPARISON_COLUMNS if col[0] != "seeds"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def paired_difference(
    baseline_payloads: list[dict], shaped_payloads: list[dict],
    n_bootstrap: int = N_BOOTSTRAP, rng_seed: int = 0,
) -> dict:
    """Paired mean difference (shaped - baseline) over the shared states.

    evaluate.py gives every model the same episode seeds, so episode ``i`` of
    seed ``S`` sees the same initial state in both groups. We pool the paired
    episodes across seeds, bootstrap-resample the episode indices, and take
    the 2.5/97.5 percentiles of the mean difference. Only seeds present in
    BOTH groups contribute (same initial states).
    """
    base_by_seed = {p["seed"]: p for p in baseline_payloads}
    shared = sorted(set(base_by_seed) & {p["seed"] for p in shaped_payloads})
    base_rewards, shaped_rewards = [], []
    for seed in shared:
        b_eps = base_by_seed[seed]["episodes"]
        s_eps = next(p for p in shaped_payloads if p["seed"] == seed)["episodes"]
        n = min(len(b_eps), len(s_eps))
        base_rewards.extend(e["reward"] for e in b_eps[:n])
        shaped_rewards.extend(e["reward"] for e in s_eps[:n])
    base = np.asarray(base_rewards, dtype=float)
    shaped = np.asarray(shaped_rewards, dtype=float)
    if base.size == 0:
        raise ValueError("no shared seeds between baseline and shaped groups")
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, base.size, size=(n_bootstrap, base.size))
    boot = shaped[idx].mean(axis=1) - base[idx].mean(axis=1)
    return {
        "shared_seeds": shared,
        "n_paired_episodes": int(base.size),
        "mean_difference": float(shaped.mean() - base.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def paired_differences_report(groups: dict[str, list[dict]]) -> str:
    """Render one section per non-baseline mode: paired diff vs baseline + CI."""
    baseline = groups.get(f"{aggregate.BASELINE}_final")
    if baseline is None:
        return (
            "# Paired differences vs baseline\n\n"
            "No baseline evaluation found — run `scripts/evaluate.py` (no --mode) "
            "first, so every shaped mode can be paired against it.\n"
        )
    parts = [
        "# Paired differences vs baseline (original reward)\n",
        "Episodes are paired by initial state: every model was evaluated on the "
        "same episode seeds, so the difference removes initial-state luck. "
        "95% CIs are bootstrap (10,000 resamples of the paired episodes).\n",
    ]
    for group, payloads in sorted(groups.items()):
        mode = payloads[0].get("reward_mode", aggregate.BASELINE)
        if mode == aggregate.BASELINE:
            continue
        diff = paired_difference(baseline, payloads)
        verdict = "WITHIN NOISE" if diff["ci_low"] <= 0.0 <= diff["ci_high"] else "SIGNIFICANT"
        parts.append(
            f"\n## {mode}\n\n"
            f"- paired episodes: {diff['n_paired_episodes']} "
            f"(seeds {diff['shared_seeds']})\n"
            f"- mean difference vs baseline: {diff['mean_difference']:+.1f}\n"
            f"- 95% bootstrap CI: [{diff['ci_low']:+.1f}, {diff['ci_high']:+.1f}]\n"
            f"- verdict: **{verdict}** "
            f"(a CI straddling 0 means the difference is not distinguishable "
            f"from noise at this episode count)\n"
        )
    return "".join(parts)


def monitor_rewards(mode: str, results_dir: str = "results", window: int = 100) -> dict:
    """Smoothed training rewards for one mode from its Monitor CSVs.

    Returns ``{"shaped": [...], "original": [...]}`` — rolling means
    (``window`` episodes) over the episodes in training order across all
    seeds' Monitor CSVs. ``r`` is the reward the agent actually saw (shaped
    for a shaped mode); ``episode_original`` is the built-in reward.
    """
    mode_dir = "" if mode == aggregate.BASELINE else mode
    pattern = os.path.join(
        results_dir, "runs", mode_dir, "seed_*", "monitor", "*.monitor.csv"
    )
    shaped, original = [], []
    for path in sorted(glob.glob(pattern)):
        # Skip the leading '# {"t_start": ...}' metadata line; the NEXT line is
        # the CSV header. Baseline Monitor CSVs have only r,l,t (no
        # episode_original column) — for the unwrapped baseline, r IS the
        # original reward.
        with open(path) as f:
            lines = f.readlines()[1:]
        reader = csv.DictReader(lines)
        for row in reader:
            shaped.append(float(row["r"]))
            original.append(float(row.get("episode_original", row["r"])))
    if not shaped:
        return {"shaped": np.empty(0), "original": np.empty(0)}

    def smooth(values: list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size < window:
            return arr.mean(keepdims=True)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")

    return {"shaped": smooth(shaped), "original": smooth(original)}


def plot_bar_chart(rows: list[dict], path: str) -> None:
    """Bar chart of the mean across seeds per mode with seed-level error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [r["mode"] for r in rows]
    means = [r["mean_across_seeds"] for r in rows]
    errors = [r["std_across_seeds"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(modes, means, yerr=errors, capsize=5, color="#4a90d9")
    ax.set_ylabel("mean reward over 100 episodes/seed (original reward)")
    ax.set_title("Reward shaping vs baseline (error bars = spread across seeds)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(modes: list[str], path: str, results_dir: str = "results") -> None:
    """Per-mode training curves: shaped reward vs original reward."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(modes)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.6 * n), squeeze=False)
    for ax, mode in zip(axes[:, 0], modes):
        series = monitor_rewards(mode, results_dir=results_dir)
        if series["shaped"].size == 0:
            ax.set_title(f"{mode} (no Monitor CSVs found)")
            continue
        x = np.arange(series["shaped"].size)
        ax.plot(x, series["shaped"], label="reward seen by agent (Monitor r)", lw=1)
        ax.plot(
            x, series["original"], label="original reward", lw=1, ls="--", alpha=0.8
        )
        ax.set_title(mode)
        ax.set_xlabel("training episode (smoothed, window=100)")
        ax.set_ylabel("reward")
        ax.legend(fontsize=7)
    fig.suptitle(
        "Training rewards: shaped vs original "
        "(flat shaped + falling original = hover-farming)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        default=aggregate.DEFAULT_EVAL_DIR,
        help="directory containing the per-seed evaluation JSONs",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="results root (for the Monitor CSV learning curves)",
    )
    return parser.parse_args()


def main() -> None:
    """Print the mode comparison and write comparison.md."""
    args = parse_args()
    groups = aggregate.load_results(args.eval_dir)
    if not groups:
        raise SystemExit(
            f"No evaluation JSONs found under {args.eval_dir} "
            "(run scripts/evaluate.py first)"
        )
    rows = comparison_rows(groups)
    table = render_markdown(rows)
    title = (
        "# Reward-shaping comparison on the original reward\n\n"
        "All models were trained with identical hyperparameters and evaluated "
        "on the same shared episode seeds. `mean across seeds +/-` is the "
        "honest spread over seeds (the replication unit); `timeout %` is the "
        "hover-farming signal.\n\n"
    )
    print(title + table)
    out_path = os.path.join(args.eval_dir, "comparison.md")
    with open(out_path, "w") as f:
        f.write(title + table)
    csv_path = os.path.join(args.eval_dir, "comparison.csv")
    write_csv(rows, csv_path)
    json_path = os.path.join(args.eval_dir, "comparison.json")
    with open(json_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    paired = paired_differences_report(groups)
    paired_path = os.path.join(args.eval_dir, "paired_differences.md")
    with open(paired_path, "w") as f:
        f.write(paired)
    print(paired)
    bar_path = os.path.join(args.eval_dir, "comparison.png")
    plot_bar_chart(rows, bar_path)
    curves_path = os.path.join(args.eval_dir, "learning_curves.png")
    plot_learning_curves([r["mode"] for r in rows], curves_path, results_dir=args.results_dir)
    for path in (out_path, csv_path, json_path, paired_path, bar_path, curves_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
