"""Tests for the training entry point's helper functions."""

import argparse
import os

import pytest
import train


def test_load_config_returns_expected_values():
    """The baseline config exposes the documented keys and Zoo values."""
    cfg = train.load_config(os.path.join("configs", "ppo_baseline.yaml"))
    assert cfg["env_id"] == "LunarLander-v3"
    assert cfg["total_timesteps"] == 1_000_000
    assert cfg["n_envs"] == 16
    assert cfg["hyperparameters"]["policy"] == "MlpPolicy"
    assert cfg["hyperparameters"]["ent_coef"] == 0.01
    assert cfg["hyperparameters"]["gamma"] == 0.999


def test_find_resume_checkpoint_prefers_final_model(tmp_path):
    """The final model wins over any periodic checkpoint."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "ppo_lander_seed0.zip").write_bytes(b"final")
    (model_dir / "ppo_lander_seed0_50000_steps.zip").write_bytes(b"ck")
    (model_dir / "ppo_lander_seed0_100000_steps.zip").write_bytes(b"ck")
    picked = train.find_resume_checkpoint(str(model_dir), 0)
    assert picked.endswith("ppo_lander_seed0.zip")


def test_find_resume_checkpoint_picks_latest_when_no_final(tmp_path):
    """Without a final model, the highest-step checkpoint is used."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "ppo_lander_seed1_50000_steps.zip").write_bytes(b"ck")
    (model_dir / "ppo_lander_seed1_300000_steps.zip").write_bytes(b"ck")
    (model_dir / "ppo_lander_seed1_250000_steps.zip").write_bytes(b"ck")
    picked = train.find_resume_checkpoint(str(model_dir), 1)
    assert picked.endswith("ppo_lander_seed1_300000_steps.zip")


def test_find_resume_checkpoint_returns_none_when_empty(tmp_path):
    """An empty models directory yields None rather than an exception."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    assert train.find_resume_checkpoint(str(model_dir), 2) is None


def test_all_reward_configs_share_identical_hyperparameters():
    """The only thing that differs between modes is the reward block.

    This is the core methodological invariant of the Phase 2 comparison:
    every shaped mode must train with the baseline's exact recipe.
    """
    from src.reward_wrappers import MODES

    configs = {
        name: train.load_config(os.path.join("configs", name))
        for name in (
            "ppo_baseline.yaml",
            "ppo_time_penalty.yaml",
            "ppo_hover_bonus.yaml",
            "ppo_potential_shaped.yaml",
        )
    }
    baseline = configs["ppo_baseline.yaml"]
    for name, cfg in configs.items():
        assert cfg["hyperparameters"] == baseline["hyperparameters"], name
        assert cfg["total_timesteps"] == baseline["total_timesteps"], name
        assert cfg["n_envs"] == baseline["n_envs"], name
        assert cfg["vec_env"] == baseline["vec_env"], name
    shaped = {k: v for k, v in configs.items() if k != "ppo_baseline.yaml"}
    for name, cfg in shaped.items():
        assert cfg["reward"]["mode"] in MODES, name
        assert cfg["reward"]["mode"] != "builtin", "builtin must use the baseline config"
    modes = {cfg["reward"]["mode"] for cfg in shaped.values()}
    assert modes == {"time_penalty", "hover_bonus", "potential_shaped"}


def _reward_args(mode: str | None) -> argparse.Namespace:
    """A Namespace stub carrying only the reward-related CLI field."""
    return argparse.Namespace(mode=mode)


def test_resolve_reward_cfg_cli_mode_overrides_builtin_default():
    """--mode must flow into the reward config that builds the wrapper.

    Regression test for the bug that trained ``--mode`` runs on the builtin
    reward while the banner claimed the shaped mode. The baseline config has
    no reward block, so the CLI mode is the only source for the wrapper mode.
    """
    cfg = train.load_config(os.path.join("configs", "ppo_baseline.yaml"))
    resolved = train.resolve_reward_cfg(_reward_args("hover_bonus"), cfg)
    assert resolved["mode"] == "hover_bonus"
    assert "reward" not in cfg  # the input config must not be mutated


def test_resolve_reward_cfg_defaults_to_config_reward_block():
    """Without --mode, the config's reward mode wins and its params are kept."""
    cfg = train.load_config(os.path.join("configs", "ppo_time_penalty.yaml"))
    resolved = train.resolve_reward_cfg(_reward_args(None), cfg)
    assert resolved["mode"] == "time_penalty"
    assert resolved["params"]["time_penalty"] == 0.05


def test_resolve_reward_cfg_cli_mode_wins_over_config_block():
    """An explicit --mode is authoritative even when the config names another mode."""
    cfg = train.load_config(os.path.join("configs", "ppo_hover_bonus.yaml"))
    resolved = train.resolve_reward_cfg(_reward_args("potential_shaped"), cfg)
    assert resolved["mode"] == "potential_shaped"


def test_resolve_reward_cfg_rejects_unknown_mode():
    """An unknown mode must fail loudly before any training starts."""
    with pytest.raises(SystemExit):
        train.resolve_reward_cfg(_reward_args("not_a_reward_mode"), {})
