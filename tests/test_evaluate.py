"""Tests for the evaluation script's statistics, episodes and JSON output."""

import json

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

import evaluate


def episode(reward: float, outcome: str, length: int = 100, fuel: float = 10.0) -> dict:
    """Build a synthetic episode measurement."""
    return {"reward": reward, "length": length, "outcome": outcome, "fuel_cost": fuel}


@pytest.fixture(scope="module")
def tiny_model():
    """A very small, briefly-trained PPO model for episode-level tests."""
    env = gym.make("LunarLander-v3")
    model = PPO(
        "MlpPolicy",
        env,
        seed=0,
        n_steps=256,
        batch_size=64,
        n_epochs=1,
        policy_kwargs={"net_arch": [16]},
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=512)
    env.close()
    return model


def test_summarise_computes_mean_and_rates():
    """Mean/std and outcome rates come out exactly for a fixed sample."""
    episodes = [
        episode(100.0, "landed"),
        episode(300.0, "landed"),
        episode(-100.0, "crashed"),
    ]
    stats = evaluate.summarise(episodes)
    assert stats["episodes"] == 3
    assert stats["mean_reward"] == pytest.approx(100.0)
    assert stats["std_reward"] == pytest.approx(200.0)
    assert stats["stderr_reward"] == pytest.approx(200.0 / (3.0**0.5))
    assert stats["median_reward"] == pytest.approx(100.0)
    assert stats["landed_rate"] == pytest.approx(2 / 3)
    assert stats["crash_rate"] == pytest.approx(1 / 3)
    assert stats["timeout_rate"] == pytest.approx(0.0)
    assert stats["solved_rate_ge200"] == pytest.approx(1 / 3)


def test_summarise_single_episode_has_no_std():
    """One episode cannot have a sample std; both error terms must be 0."""
    stats = evaluate.summarise([episode(5.0, "landed")])
    assert stats["std_reward"] == 0.0
    assert stats["stderr_reward"] == 0.0


def test_run_episode_records_valid_measurements(tiny_model):
    """One episode returns a well-formed measurement on this machine."""
    env = gym.make("LunarLander-v3")
    result = evaluate.run_episode(tiny_model, env, 123)
    env.close()
    assert result["outcome"] in {"landed", "crashed", "timeout"}
    assert 1 <= result["length"] <= 1000
    assert result["fuel_cost"] >= 0.0


def test_run_episode_is_deterministic_for_fixed_seed(tiny_model):
    """A fixed episode seed reproduces the same episode exactly."""
    first = evaluate.run_episode(tiny_model, gym.make("LunarLander-v3"), 7)
    second = evaluate.run_episode(tiny_model, gym.make("LunarLander-v3"), 7)
    assert first == second


def test_evaluate_seed_writes_json_with_expected_shape(tmp_path, monkeypatch, tiny_model):
    """evaluate_seed produces the JSON consumed by scripts/aggregate.py."""
    # Point the module's RESULTS_DIR at a fake run directory holding the model.
    fake_results = tmp_path / "results"
    model_dir = fake_results / "runs" / "seed_0" / "models"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "ppo_lander_seed0.zip"
    tiny_model.save(str(model_path))
    monkeypatch.setattr(evaluate, "RESULTS_DIR", str(fake_results))

    out_dir = tmp_path / "evaluations"
    payload = evaluate.evaluate_seed(0, "final", 3, str(out_dir))

    assert payload["seed"] == 0
    assert payload["model_source"] == "final"
    assert payload["reward_mode"] == "baseline"
    assert payload["model_path"].endswith("ppo_lander_seed0.zip")
    assert len(payload["episodes"]) == 3
    # Episode seeds follow the fixed base sequence -> reproducible evaluations.
    assert payload["episode_seed_base"] == evaluate.SEED_BASE

    saved = out_dir / "ppo_seed_0_final.json"
    assert saved.exists()
    with open(saved) as f:
        assert json.load(f)["summary"]["episodes"] == 3


def test_evaluate_seed_shaped_mode_uses_mode_subdir_and_filename(
    tmp_path, monkeypatch, tiny_model
):
    """A shaped --mode reads runs/<mode>/seed_<S>/ and writes ppo_<mode>_seed_...json."""
    fake_results = tmp_path / "results"
    model_dir = fake_results / "runs" / "time_penalty" / "seed_0" / "models"
    model_dir.mkdir(parents=True)
    tiny_model.save(str(model_dir / "ppo_lander_seed0.zip"))
    monkeypatch.setattr(evaluate, "RESULTS_DIR", str(fake_results))

    out_dir = tmp_path / "evaluations"
    payload = evaluate.evaluate_seed(0, "final", 3, str(out_dir), mode="time_penalty")

    assert payload["reward_mode"] == "time_penalty"
    assert "time_penalty" in payload["model_path"]
    saved = out_dir / "ppo_time_penalty_seed_0_final.json"
    assert saved.exists()
    assert not (out_dir / "ppo_seed_0_final.json").exists()
