"""Tests for scripts/inspect_hover.py helper functions."""

import numpy as np

from inspect_hover import compute_stats, hover_mask, mean_hover_run_length


def _obs(angle=0.0, vx=0.0, vy=0.0):
    return np.array([0.0, 0.5, vx, vy, angle, 0.0, 0.0, 0.0])


def test_hover_mask_boundary_conditions():
    """Hover = |angle| and both speeds strictly under the thresholds."""
    eps = 1e-6
    frames = np.array([
        _obs(),                    # hovering
        _obs(angle=0.2 - eps),     # just under the angle threshold
        _obs(angle=0.2 + eps),     # just over -> not hovering
        _obs(vx=0.3 + eps),        # just over vx -> not hovering
        _obs(vy=0.3 + eps),        # just over vy -> not hovering
        _obs(angle=1.0, vx=1.0, vy=1.0),  # clearly not hovering
    ])
    mask = hover_mask(frames, angle_thresh=0.2, speed_thresh=0.3)
    assert mask.tolist() == [True, True, False, False, False, False]


def test_mean_hover_run_length():
    """Consecutive True runs are averaged; empty/all-False masks give 0."""
    assert mean_hover_run_length(np.array([True, True, False, True])) == 1.5
    assert mean_hover_run_length(np.array([True] * 10)) == 10.0
    assert mean_hover_run_length(np.array([False] * 5)) == 0.0
    assert mean_hover_run_length(np.array([], dtype=bool)) == 0.0


def test_compute_stats_pools_records():
    """Timeout/landing rates and reward statistics come out as expected."""
    records = [
        {"length": 1000, "reward": +800.0, "outcome": "timeout",
         "hover_fraction": 0.9, "hover_run_length": 50.0},
        {"length": 1000, "reward": +820.0, "outcome": "timeout",
         "hover_fraction": 0.9, "hover_run_length": 60.0},
        {"length": 300, "reward": +250.0, "outcome": "landed",
         "hover_fraction": 0.6, "hover_run_length": 20.0},
        {"length": 250, "reward": -100.0, "outcome": "crashed",
         "hover_fraction": 0.1, "hover_run_length": 5.0},
    ]
    stats = compute_stats(records)
    assert stats["episodes"] == 4
    assert stats["mean_reward"] == np.mean([800.0, 820.0, 250.0, -100.0])
    assert stats["sem_reward"] > 0.0
    assert stats["mean_length"] == 637.5
    assert stats["max_length"] == 1000
    assert stats["timeout_rate"] == 0.5
    assert stats["landed_rate"] == 0.25
    assert stats["crashed_rate"] == 0.25
    assert stats["hover_fraction"] == np.mean([0.9, 0.9, 0.6, 0.1])
    assert stats["hover_run_length"] == np.mean([50.0, 60.0, 20.0, 5.0])
