"""Tests for the video recording script's model discovery and naming."""

import os

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

import record_video


@pytest.fixture(scope="module")
def tiny_model():
    """A very small, briefly-trained PPO model for recording tests."""
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


def _fake_run_tree(tmp_path, mode):
    """Create a fake results/runs tree for one mode and return its model dir."""
    model_dir = tmp_path / "results" / "runs"
    if mode != record_video.BASELINE:
        model_dir = model_dir / mode
    model_dir = model_dir / "seed_0" / "models"
    model_dir.mkdir(parents=True)
    return model_dir


def test_find_model_resolves_baseline_and_best(tmp_path, monkeypatch, tiny_model):
    """Baseline models live under runs/seed_<S>/; best uses the best/ dir."""
    run_dir = tmp_path / "results" / "runs" / "seed_0"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "best").mkdir()
    tiny_model.save(str(run_dir / "models" / "ppo_lander_seed0.zip"))
    tiny_model.save(str(run_dir / "best" / "best_model.zip"))
    monkeypatch.setattr(record_video, "RESULTS_DIR", str(tmp_path / "results"))

    assert record_video.find_model(0, "final").endswith(os.path.join("models", "ppo_lander_seed0.zip"))
    assert record_video.find_model(0, "best").endswith(os.path.join("best", "best_model.zip"))


def test_find_model_shaped_mode_uses_mode_subdir(tmp_path, monkeypatch, tiny_model):
    """A shaped --mode resolves runs/<mode>/seed_<S>/."""
    model_dir = _fake_run_tree(tmp_path, "time_penalty")
    tiny_model.save(str(model_dir / "ppo_lander_seed0.zip"))
    monkeypatch.setattr(record_video, "RESULTS_DIR", str(tmp_path / "results"))

    path = record_video.find_model(0, "final", mode="time_penalty")
    assert path.endswith(os.path.join("time_penalty", "seed_0", "models", "ppo_lander_seed0.zip"))


def test_find_model_missing_raises_system_exit(tmp_path, monkeypatch):
    """A missing run directory is a hard error, not a silent skip."""
    monkeypatch.setattr(record_video, "RESULTS_DIR", str(tmp_path / "results"))
    with pytest.raises(SystemExit) as excinfo:
        record_video.find_model(0, "final", mode="hover_bonus")
    assert "Model not found" in str(excinfo.value)


def test_record_seed_writes_mode_named_video(tmp_path, monkeypatch, tiny_model):
    """Recording writes videos/<mode>_seed<S>.mp4 with the episode's reward."""
    model_dir = _fake_run_tree(tmp_path, "hover_bonus")
    tiny_model.save(str(model_dir / "ppo_lander_seed0.zip"))
    monkeypatch.setattr(record_video, "RESULTS_DIR", str(tmp_path / "results"))
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(record_video, "VIDEO_DIR", str(video_dir))

    path = record_video.record_seed(0, "final", mode="hover_bonus")

    assert path == str(video_dir / "hover_bonus_seed0.mp4")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    # No leftover intermediate file keeps the videos/ directory clean.
    assert not (video_dir / "hover_bonus_seed0-episode-0.mp4").exists()


def test_record_seed_baseline_uses_uniform_naming(tmp_path, monkeypatch, tiny_model):
    """The baseline mode writes baseline_seed<S>.mp4 like every other mode."""
    model_dir = _fake_run_tree(tmp_path, record_video.BASELINE)
    tiny_model.save(str(model_dir / "ppo_lander_seed0.zip"))
    monkeypatch.setattr(record_video, "RESULTS_DIR", str(tmp_path / "results"))
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(record_video, "VIDEO_DIR", str(video_dir))

    path = record_video.record_seed(0, "final")
    assert os.path.basename(path) == "baseline_seed0.mp4"
