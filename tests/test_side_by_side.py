"""Tests for the side-by-side video composition script."""

import os

import pytest
from moviepy import ColorClip, VideoFileClip

import side_by_side


def _write_color_clip(path, color, size=(32, 24), duration=0.5):
    """Write a tiny solid-colour clip as a stand-in episode video."""
    clip = ColorClip(size, color=color).with_duration(duration)
    clip.write_videofile(str(path), fps=10, codec="libx264", logger=None)
    clip.close()
    return str(path)


def test_stack_writes_composite_with_both_panels(tmp_path):
    """Stacking produces a video as wide as both inputs and as long as the longer."""
    left = _write_color_clip(tmp_path / "a_seed0.mp4", (255, 0, 0), duration=0.5)
    right = _write_color_clip(tmp_path / "b_seed0.mp4", (0, 0, 255), duration=0.3)
    out = str(tmp_path / "out.mp4")

    result = side_by_side.stack(left, right, out)

    assert result == out
    assert os.path.exists(out)
    with VideoFileClip(out) as composite:
        assert composite.w == 64
        assert composite.h == 24
        assert composite.duration == pytest.approx(0.5, abs=0.05)


def test_main_requires_both_episode_videos(tmp_path, monkeypatch):
    """A missing input video is a hard error naming the file to record."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    _write_color_clip(video_dir / "hover_bonus_seed0.mp4", (0, 255, 0))
    monkeypatch.setattr(side_by_side, "VIDEO_DIR", str(video_dir))
    monkeypatch.setattr(
        "sys.argv",
        ["side_by_side.py", "--video-dir", str(video_dir), "--seed", "0"],
    )

    with pytest.raises(SystemExit) as excinfo:
        side_by_side.main()
    assert "potential_shaped_seed0.mp4" in str(excinfo.value)


def test_main_composes_requested_modes(tmp_path, monkeypatch):
    """--mode-a/--mode-b/--seed pick the inputs and the default output name."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    _write_color_clip(video_dir / "hover_bonus_seed1.mp4", (0, 255, 0), duration=0.4)
    _write_color_clip(video_dir / "potential_shaped_seed1.mp4", (255, 255, 0), duration=0.4)
    monkeypatch.setattr(
        "sys.argv",
        ["side_by_side.py", "--video-dir", str(video_dir), "--seed", "1"],
    )

    side_by_side.main()

    out = video_dir / "side_by_side_hover_bonus_vs_potential_shaped_seed1.mp4"
    assert out.exists()
    assert out.stat().st_size > 0
