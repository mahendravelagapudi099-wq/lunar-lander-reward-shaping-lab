"""Tests for the cross-mode comparison report."""

import json
import sys

import pytest

import compare


def payload(seed: int, rewards: list[float], mode: str = "baseline") -> dict:
    """Build a synthetic per-seed evaluation payload from a reward list."""
    episodes = [
        {"reward": r, "length": 100, "outcome": "landed", "fuel_cost": 5.0}
        for r in rewards
    ]
    mean = sum(rewards) / len(rewards)
    return {
        "env_id": "LunarLander-v3",
        "reward_mode": mode,
        "seed": seed,
        "model_source": "final",
        "model_path": f"results/runs/seed_{seed}/models/ppo_lander_seed{seed}.zip",
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "mean_reward": mean,
            "stderr_reward": 1.0,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "mean_length": 100.0,
            "mean_fuel_cost": 5.0,
            "solved_rate_ge200": sum(r >= 200 for r in rewards) / len(rewards),
            "landed_rate": 1.0,
            "crash_rate": 0.0,
            "timeout_rate": 0.0,
        },
    }


def test_comparison_rows_one_row_per_group():
    """Each (mode, source) group becomes one row with seed-level spread."""
    groups = {
        "baseline_final": [payload(0, [200.0, 200.0]), payload(1, [100.0, 100.0])],
        "time_penalty_final": [payload(0, [250.0, 250.0], mode="time_penalty")],
    }
    rows = compare.comparison_rows(groups)
    assert [r["mode"] for r in rows] == ["baseline", "time_penalty"]
    base, shaped = rows
    assert base["n_seeds"] == 2 and base["n_episodes"] == 4
    assert base["mean_across_seeds"] == pytest.approx(150.0)
    assert base["std_across_seeds"] == pytest.approx(50.0 * 2**0.5)
    assert base["pooled_mean_reward"] == pytest.approx(150.0)
    # Shaped mode reports its own mode even though the group key has an `_`.
    assert shaped["mode"] == "time_penalty"
    assert shaped["n_seeds"] == 1
    # A single seed cannot give a seed-level spread; it must be 0, not NaN.
    assert shaped["std_across_seeds"] == 0.0


def test_paired_difference_recovers_constant_offset():
    """A +10 offset on every paired episode gives diff +10 and a CI above 0."""
    base_rewards = [100.0, 105.0, 95.0, 102.0, 98.0]
    baseline = [payload(0, base_rewards), payload(1, base_rewards)]
    shaped = [
        payload(0, [r + 10.0 for r in base_rewards]),
        payload(1, [r + 10.0 for r in base_rewards]),
    ]
    for p in shaped:
        p["reward_mode"] = "time_penalty"
    diff = compare.paired_difference(baseline, shaped, n_bootstrap=2000)
    assert diff["mean_difference"] == pytest.approx(10.0)
    assert diff["n_paired_episodes"] == 10
    assert diff["ci_high"] >= diff["ci_low"]
    assert diff["ci_low"] > 0.0  # a constant shift is not noise


def test_paired_difference_noise_gives_ci_straddling_zero():
    """Same-shaped rewards on both sides -> the CI must straddle zero."""
    same = [payload(0, [100.0, 200.0, 150.0, 175.0])]
    diff = compare.paired_difference(same, list(same), n_bootstrap=2000)
    assert diff["mean_difference"] == pytest.approx(0.0)
    assert diff["ci_low"] <= 0.0 <= diff["ci_high"]


def test_paired_difference_unshared_seeds_is_empty():
    """A ValueError (not a crash) when the groups share no seeds."""
    with pytest.raises(ValueError):
        compare.paired_difference([payload(0, [100.0])], [payload(1, [100.0])])


def test_write_csv_has_one_row_per_group(tmp_path):
    """The CSV mirrors the markdown table."""
    rows = [
        {
            "mode": "baseline", "n_seeds": 3, "n_episodes": 300,
            "mean_across_seeds": 250.0, "std_across_seeds": 3.5,
            "pooled_mean_reward": 254.6, "landed_rate": 0.99,
            "crash_rate": 0.0, "timeout_rate": 0.01,
            "mean_length": 366.0, "mean_fuel_cost": 53.7,
            "solved_rate_ge200": 0.99,
        }
    ]
    out = tmp_path / "comparison.csv"
    compare.write_csv(rows, str(out))
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("mode,n_seeds")
    assert lines[1].startswith("baseline,3,300")
    assert "seeds" not in lines[0].split(",")  # seed list stays out of the CSV


def test_monitor_rewards_reads_and_smooths(tmp_path):
    """Monitor CSVs are parsed (header skipped) and smoothed."""
    runs = tmp_path / "results" / "runs" / "time_penalty" / "seed_0" / "monitor"
    runs.mkdir(parents=True)
    header = '# {"t_start": 1}\nr,l,t,episode_original,episode_shaping\n'
    rows = "".join(f"{i * 10},{5},{i * 2},{i * 10},{-5.0}\n" for i in range(10))
    (runs / "0.monitor.csv").write_text(header + rows)
    series = compare.monitor_rewards("time_penalty", results_dir=str(tmp_path / "results"))
    # r rises in steps of 10 (0,10,...,90): a 100-window series is short, so
    # the fallback single-mean branch must kick in.
    assert len(series["shaped"]) == 1
    assert series["shaped"][0] == pytest.approx(45.0)
    assert series["original"][0] == pytest.approx(45.0)
    # No CSVs for a missing mode -> empty series, no exception.
    missing = compare.monitor_rewards("hover_bonus", results_dir=str(tmp_path / "results"))
    assert missing["shaped"].size == 0 and missing["original"].size == 0
    # Baseline Monitor CSVs have only r,l,t: the original reward is r itself.
    base_runs = tmp_path / "results" / "runs" / "seed_0" / "monitor"
    base_runs.mkdir(parents=True)
    (base_runs / "0.monitor.csv").write_text(header.replace(",episode_original,episode_shaping", "") + rows)
    base = compare.monitor_rewards("baseline", results_dir=str(tmp_path / "results"))
    assert base["shaped"] == base["original"]


def test_render_markdown_includes_mode_and_signal_columns():
    """The table exposes mode, mean +/- and the timeout (hover) signal."""
    rows = [
        {
            "mode": "time_penalty",
            "n_seeds": 3,
            "n_episodes": 300,
            "mean_across_seeds": 250.0,
            "std_across_seeds": 12.0,
            "pooled_mean_reward": 248.0,
            "landed_rate": 0.99,
            "crash_rate": 0.01,
            "timeout_rate": 0.0,
            "mean_length": 310.0,
            "mean_fuel_cost": 40.0,
            "solved_rate_ge200": 0.9,
        }
    ]
    table = compare.render_markdown(rows)
    for cell in ("time_penalty", "250.0", "12.0", "99", "0", "310"):
        assert cell in table
    assert "timeout %" in table
    assert "solved>=200 %" in table


def test_main_reports_all_modes(tmp_path, monkeypatch, capsys):
    """main() reads evaluation JSONs, prints and saves the comparison."""
    for mode, rewards in (("baseline", [200.0, 100.0]), ("hover_bonus", [220.0, 80.0])):
        data = payload(0, rewards, mode=mode)
        name = f"ppo_seed_0_final.json" if mode == "baseline" else f"ppo_hover_bonus_seed_0_final.json"
        (tmp_path / name).write_text(json.dumps(data))
    monkeypatch.setattr(compare.aggregate, "DEFAULT_EVAL_DIR", str(tmp_path))
    sys.argv = ["compare.py", "--eval-dir", str(tmp_path), "--results-dir", str(tmp_path)]
    compare.main()
    out = capsys.readouterr().out
    assert "baseline" in out and "hover_bonus" in out
    assert (tmp_path / "comparison.md").exists()
    saved = json.loads((tmp_path / "comparison.json").read_text())
    assert [r["mode"] for r in saved["rows"]] == ["baseline", "hover_bonus"]
