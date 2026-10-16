"""Tests for the aggregation script: grouping, pooling and table rendering."""

import json

import pytest

import aggregate


def payload(seed: int, rewards: list[float], source: str = "final") -> dict:
    """Build a synthetic per-seed evaluation payload from a reward list."""
    episodes = [
        {"reward": r, "length": 100, "outcome": "landed", "fuel_cost": 5.0}
        for r in rewards
    ]
    mean = sum(rewards) / len(rewards)
    return {
        "env_id": "LunarLander-v3",
        "seed": seed,
        "model_source": source,
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


def test_pool_episodes_pools_all_episodes():
    """Pooled stats use every episode of every seed."""
    payloads = [payload(0, [100.0, 200.0]), payload(1, [300.0, 100.0])]
    pooled = aggregate.pool_episodes(payloads)
    assert pooled["episodes"] == 4
    assert pooled["pooled_mean_reward"] == pytest.approx(175.0)
    assert pooled["min_reward"] == pytest.approx(100.0)
    assert pooled["max_reward"] == pytest.approx(300.0)
    assert pooled["landed_rate"] == pytest.approx(1.0)


def test_pool_episodes_seed_level_spread():
    """The across-seeds spread uses per-seed means, not raw episodes."""
    payloads = [payload(0, [200.0, 200.0]), payload(1, [100.0, 100.0])]
    pooled = aggregate.pool_episodes(payloads)
    assert pooled["mean_across_seeds"] == pytest.approx(150.0)
    # ddof=1: deviations are +50/-50 from the 150 mean, one free degree.
    assert pooled["std_across_seeds"] == pytest.approx((50.0**2 + 50.0**2) ** 0.5)


def test_load_results_groups_by_mode_and_model_source(tmp_path):
    """JSONs are grouped by reward mode + model source and ordered by seed."""
    for seed in (2, 0, 1):
        data = payload(seed, [100.0, 200.0], source="best")
        (tmp_path / f"ppo_seed_{seed}_best.json").write_text(json.dumps(data))
    (tmp_path / "summary.json").write_text("{}")  # must be ignored
    groups = aggregate.load_results(str(tmp_path))
    # Legacy JSONs without reward_mode are treated as baseline.
    assert list(groups.keys()) == ["baseline_best"]
    assert [p["seed"] for p in groups["baseline_best"]] == [0, 1, 2]


def test_load_results_separates_shaped_modes(tmp_path):
    """Baseline and shaped-mode JSONs land in different groups."""
    for seed in (0, 1):
        data = payload(seed, [100.0, 200.0])
        data["reward_mode"] = "time_penalty"
        (tmp_path / f"ppo_time_penalty_seed_{seed}_final.json").write_text(
            json.dumps(data)
        )
    baseline = payload(0, [300.0, 400.0])
    (tmp_path / "ppo_seed_0_final.json").write_text(json.dumps(baseline))
    groups = aggregate.load_results(str(tmp_path))
    assert sorted(groups.keys()) == ["baseline_final", "time_penalty_final"]
    assert [p["seed"] for p in groups["time_penalty_final"]] == [0, 1]


def test_markdown_table_contains_seed_rows_and_pooled_row():
    """The rendered table lists every seed plus one pooled row."""
    payloads = [payload(0, [100.0]), payload(1, [300.0])]
    pooled = aggregate.pool_episodes(payloads)
    table = aggregate.markdown_table(payloads, pooled, "final")
    assert "mean reward" in table
    assert "| 0 |" in table
    assert "| 1 |" in table
    assert "*pooled*" in table
