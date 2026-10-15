"""Tests for the learning-curve plot script's data helpers."""

import numpy as np
import pytest

import plot_curves


def test_smooth_short_input_is_unchanged():
    """Inputs smaller than the window are returned as-is."""
    values = np.array([1.0, 2.0])
    result = plot_curves.smooth(values, 3)
    assert np.array_equal(result, values)


def test_smooth_flattens_noise():
    """Averaging reduces the spread of a noisy constant signal."""
    rng = np.random.default_rng(0)
    noisy = 50.0 + rng.normal(0, 10, size=200)
    smoothed = plot_curves.smooth(noisy, 5)
    assert smoothed.std() < noisy.std()
    assert abs(smoothed.mean() - 50.0) < 5.0


def test_smooth_constant_signal_stays_constant():
    """A constant input has nothing to smooth."""
    values = np.full(10, 3.0)
    assert np.allclose(plot_curves.smooth(values, 3), values)


def test_load_eval_curves_reads_checkpoint_histories(tmp_path, monkeypatch):
    """Evaluations are picked up as {seed: (timesteps, mean rewards)}."""
    runs = tmp_path / "runs"
    for seed, offset in ((0, 0.0), (1, 5.0)):
        eval_dir = runs / f"seed_{seed}" / "eval"
        eval_dir.mkdir(parents=True)
        timesteps = np.arange(1, 6) * 50_000
        results = np.full((5, 10), 100.0 + offset)
        np.savez(eval_dir / "evaluations.npz", timesteps=timesteps, results=results)
    monkeypatch.setattr(plot_curves, "RESULTS_DIR", str(tmp_path))

    curves = plot_curves.load_eval_curves()
    assert sorted(curves.keys()) == [0, 1]
    timesteps, means = curves[0]
    assert list(timesteps) == [50_000, 100_000, 150_000, 200_000, 250_000]
    assert np.allclose(means, 100.0)


def test_load_eval_curves_errors_without_any_runs(tmp_path, monkeypatch):
    """An empty results directory is a clear failure, not silent success."""
    monkeypatch.setattr(plot_curves, "RESULTS_DIR", str(tmp_path))
    with pytest.raises(SystemExit):
        plot_curves.load_eval_curves()
