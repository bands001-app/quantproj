import numpy as np
import pandas as pd
import pytest

from vtsmom.metrics import (
    annualised_vol,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def test_sharpe_matches_closed_form():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_is_scale_invariant():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 2000))
    assert sharpe_ratio(r) == pytest.approx(sharpe_ratio(r * 3.0), rel=1e-9)


def test_zero_vol_returns_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 50)))


def test_annualised_vol_scales_with_root_time():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    assert annualised_vol(r) == pytest.approx(0.01 * np.sqrt(252), rel=0.05)


def test_max_drawdown_known_path():
    # +100% then -50% returns to the start: drawdown is exactly -50%.
    r = pd.Series([1.0, -0.5])
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_is_non_positive():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert max_drawdown(r) <= 0.0


def test_psr_is_half_when_benchmark_equals_estimate():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    sr = sharpe_ratio(r)
    assert probabilistic_sharpe_ratio(r, sr) == pytest.approx(0.5, abs=1e-6)


def test_psr_decreases_as_benchmark_rises():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.0006, 0.01, 2000))
    assert probabilistic_sharpe_ratio(r, 0.0) > probabilistic_sharpe_ratio(r, 1.0)


def test_psr_rewards_longer_samples():
    """Same Sharpe, more observations, higher confidence."""
    rng = np.random.default_rng(5)
    base = rng.normal(0.0005, 0.01, 250)
    short = pd.Series(base)
    long = pd.Series(np.tile(base, 8))
    assert sharpe_ratio(long) == pytest.approx(sharpe_ratio(short), rel=1e-2)
    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)


def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(10, sr_variance=0.25)
    b = expected_max_sharpe(1000, sr_variance=0.25)
    assert 0 < a < b


def test_single_trial_needs_no_deflation():
    assert expected_max_sharpe(1, sr_variance=0.25) == 0.0


def test_deflated_sharpe_penalises_wide_searches():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.0004, 0.01, 2000))
    narrow = deflated_sharpe_ratio(r, n_trials=5, sr_variance=0.05)
    wide = deflated_sharpe_ratio(r, n_trials=5000, sr_variance=0.5)
    assert narrow > wide


def test_expected_max_sharpe_rejects_bad_input():
    with pytest.raises(ValueError):
        expected_max_sharpe(0, sr_variance=0.1)
    with pytest.raises(ValueError):
        expected_max_sharpe(10, sr_variance=-1.0)
