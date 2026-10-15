"""Tests for the training entry point's helper functions."""

import os

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
