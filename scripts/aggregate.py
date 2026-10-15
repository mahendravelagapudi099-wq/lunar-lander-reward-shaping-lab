"""Aggregate per-seed evaluation JSONs into a summary table.

Reads every ``ppo_seed_<S>_<src>.json`` written by ``scripts/evaluate.py``
from the evaluations directory, groups them by model source (``final``,
``best``, ...), and writes:

* ``summary.json`` — the full structured aggregate (per-seed and pooled
  statistics), and
* ``summary.md`` — a compact markdown table for the README/report.

Pooled statistics pool all episodes of all seeds; ``mean_across_seeds``
additionally treats each seed's mean as one sample, which is the honest
spread when quoting "the baseline reaches X +/- Y over 3 seeds".

Example:

    .venv\\Scripts\\python.exe scripts\\aggregate.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

RESULTS_DIR = "results"
DEFAULT_EVAL_DIR = os.path.join(RESULTS_DIR, "evaluations")
RESULT_PATTERN = "ppo_seed_*_*.json"

# Columns of the markdown table: (stat key, header, format).
TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("seed", "seed", "s"),
    ("episodes", "episodes", "d"),
    ("mean_reward", "mean reward", "+.1f"),
    ("stderr_reward", "SEM", ".1f"),
    ("min_reward", "min", "+.1f"),
    ("max_reward", "max", "+.1f"),
    ("landed_rate", "landed %", ".0%"),
    ("crash_rate", "crash %", ".0%"),
    ("timeout_rate", "timeout %", ".0%"),
    ("mean_length", "mean len", ".0f"),
    ("mean_fuel_cost", "fuel cost", ".1f"),
)


def load_results(eval_dir: str) -> dict[str, list[dict]]:
    """Group all evaluation JSONs by their model source, seeds ordered."""
    groups: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, RESULT_PATTERN))):
        with open(path) as f:
            payload = json.load(f)
        groups.setdefault(payload["model_source"], []).append(payload)
    for payloads in groups.values():
        payloads.sort(key=lambda p: p["seed"])
    return groups


def pool_episodes(payloads: list[dict]) -> dict:
    """Pool every episode of every seed into aggregate statistics."""
    rewards = np.array(
        [ep["reward"] for p in payloads for ep in p["episodes"]], dtype=float
    )
    lengths = np.array(
        [ep["length"] for p in payloads for ep in p["episodes"]], dtype=float
    )
    fuels = np.array(
        [ep["fuel_cost"] for p in payloads for ep in p["episodes"]], dtype=float
    )
    outcomes = [ep["outcome"] for p in payloads for ep in p["episodes"]]
    seed_means = np.array([p["summary"]["mean_reward"] for p in payloads])

    def rate(name: str) -> float:
        return float(np.mean([o == name for o in outcomes]))

    return {
        "episodes": int(rewards.size),
        "pooled_mean_reward": float(rewards.mean()),
        "pooled_std_reward": float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0,
        "mean_across_seeds": float(seed_means.mean()),
        # Spread of per-seed means (n-1 denominator; with 3 seeds this is the
        # number to quote as "+/-" for the baseline).
        "std_across_seeds": float(seed_means.std(ddof=1))
        if seed_means.size > 1
        else 0.0,
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "solved_rate_ge200": float((rewards >= 200.0).mean()),
        "landed_rate": rate("landed"),
        "crash_rate": rate("crashed"),
        "timeout_rate": rate("timeout"),
        "mean_length": float(lengths.mean()),
        "mean_fuel_cost": float(fuels.mean()),
    }
def markdown_table(rows: list[dict], pooled: dict, model_source: str) -> str:
    """Render per-seed rows plus a pooled row as a markdown table."""
    header = "| " + " | ".join(col[1] for col in TABLE_COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key, _, fmt in TABLE_COLUMNS:
            if key == "seed":
                cells.append(str(row["seed"]))
                continue
            value = row["summary"][key]
            if fmt.endswith("%"):
                # Rates are fractions; render as percentages.
                cells.append(format(value * 100.0, fmt[:-1] + "f"))
            else:
                cells.append(format(value, fmt))
        lines.append("| " + " | ".join(cells) + " |")
    pooled_row = ["*pooled*"]
    pooled_row.append(format(pooled["episodes"], "d"))
    pooled_row.append(format(pooled["pooled_mean_reward"], "+.1f"))
    pooled_row.append("-")
    pooled_row.append(format(pooled["min_reward"], "+.1f"))
    pooled_row.append(format(pooled["max_reward"], "+.1f"))
    pooled_row.append(format(pooled["landed_rate"] * 100, ".0f"))
    pooled_row.append(format(pooled["crash_rate"] * 100, ".0f"))
    pooled_row.append(format(pooled["timeout_rate"] * 100, ".0f"))
    pooled_row.append(format(pooled["mean_length"], ".0f"))
    pooled_row.append(format(pooled["mean_fuel_cost"], ".1f"))
    lines.append("| " + " | ".join(pooled_row) + " |")
    return (
        f"### Model: `{model_source}` on {rows[0]['env_id']} (original reward)\n\n"
        + "\n".join(lines)
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        default=DEFAULT_EVAL_DIR,
        help="directory containing the per-seed evaluation JSONs",
    )
    return parser.parse_args()


def main() -> None:
    """Aggregate evaluation results and write summary.json / summary.md."""
    args = parse_args()
    groups = load_results(args.eval_dir)
    if not groups:
        raise SystemExit(
            f"No evaluation JSONs found under {args.eval_dir} "
            f"(run scripts/evaluate.py first)"
        )

    aggregate: dict[str, object] = {"eval_dir": args.eval_dir, "by_model_source": {}}
    markdown_parts: list[str] = []
    for model_source, payloads in sorted(groups.items()):
        pooled = pool_episodes(payloads)
        aggregate["by_model_source"][model_source] = {
            "seeds": [p["seed"] for p in payloads],
            "per_seed": [p["summary"] for p in payloads],
            "pooled": pooled,
        }
        markdown_parts.append(markdown_table(payloads, pooled, model_source))
        print(
            f"{model_source}: {len(payloads)} seeds, "
            f"pooled mean reward {pooled['pooled_mean_reward']:+.1f} over "
            f"{pooled['episodes']} episodes; across seeds "
            f"{pooled['mean_across_seeds']:+.1f} +/- {pooled['std_across_seeds']:.1f}; "
            f"landed {pooled['landed_rate'] * 100:.0f}%, "
            f"crash {pooled['crash_rate'] * 100:.0f}%, "
            f"solved(>=200) {pooled['solved_rate_ge200'] * 100:.0f}%"
        )

    json_path = os.path.join(args.eval_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    md_path = os.path.join(args.eval_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# PPO baseline evaluation on the original reward\n\n")
        f.write("\n".join(markdown_parts))
        f.write(
            "\n`SEM` = standard error of the per-seed mean over its episodes; "
            "`fuel cost` = 0.3 * main + 0.03 * side engine firings.\n"
        )
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()

