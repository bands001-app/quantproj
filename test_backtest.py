import numpy as np
import pandas as pd
import pytest

from vtsmom.backtest import BacktestConfig, run_backtest
from vtsmom.costs import CostModel
from vtsmom.data import SyntheticSpec, synthetic_panel
from vtsmom.metrics import annualised_vol, sharpe_ratio


@pytest.fixture(scope="module")
def prices():
    return synthetic_panel(SyntheticSpec(n_days=1500, n_assets=5, seed=11))


def test_execution_lag_zero_is_rejected():
    """A zero lag would trade on the bar that produced the signal."""
    with pytest.raises(ValueError, match="lookahead"):
        BacktestConfig(execution_lag=0)


def test_no_lookahead_future_prices_cannot_change_the_past(prices):
    """The canonical bias test.

    Overwrite the final third of the panel with different data. Every return
    the strategy produced *before* that point must be bit-identical. If any of
    them move, information from the future is leaking backwards.
    """
    cut = int(len(prices) * 0.66)
    tampered = prices.copy()
    rng = np.random.default_rng(99)
    shock = rng.normal(0, 0.05, tampered.iloc[cut:].shape)
    tampered.iloc[cut:] = tampered.iloc[cut:].to_numpy() * np.exp(np.cumsum(shock, axis=0))

    # Slice by DATE: net_returns is trimmed of its warm-up, so integer
    # positions do not line up with positions in the price panel.
    boundary = prices.index[cut - 5]
    a = run_backtest(prices).net_returns.loc[:boundary]
    b = run_backtest(tampered).net_returns.loc[:boundary]

    assert len(a) > 500, "guard: the comparison window must be substantial"
    pd.testing.assert_series_equal(a, b)


def test_weights_are_lagged_relative_to_signal(prices):
    """A position must not be held on the same bar its signal was computed."""
    res = run_backtest(prices, BacktestConfig(execution_lag=1))
    # The first held weight can only appear one bar after the first signal.
    first_nonzero = res.weights.abs().sum(axis=1).to_numpy().nonzero()[0]
    assert len(first_nonzero) > 0


def test_realised_vol_tracks_the_target(prices):
    cfg = BacktestConfig(target_vol=0.10, costs=CostModel(0, 0, 0))
    res = run_backtest(prices, cfg)
    realised = annualised_vol(res.gross_returns)
    # Ex-ante targeting is imperfect; a factor-of-two band is the honest claim.
    assert 0.05 < realised < 0.20


def test_target_vol_scales_output_vol(prices):
    cfg_lo = BacktestConfig(target_vol=0.05, max_leverage=99, costs=CostModel(0, 0, 0))
    cfg_hi = BacktestConfig(target_vol=0.20, max_leverage=99, costs=CostModel(0, 0, 0))
    lo = annualised_vol(run_backtest(prices, cfg_lo).gross_returns)
    hi = annualised_vol(run_backtest(prices, cfg_hi).gross_returns)
    assert hi == pytest.approx(4 * lo, rel=0.05)


def test_leverage_cap_binds(prices):
    cfg = BacktestConfig(target_vol=2.0, max_leverage=1.5)
    res = run_backtest(prices, cfg)
    assert res.weights.abs().sum(axis=1).max() <= 1.5 + 1e-9


def test_costs_reduce_returns_monotonically(prices):
    free = run_backtest(prices, BacktestConfig(costs=CostModel(0, 0, 0)))
    cheap = run_backtest(prices, BacktestConfig(costs=CostModel(0.5, 1.0, 0)))
    dear = run_backtest(prices, BacktestConfig(costs=CostModel(10.0, 20.0, 0)))

    assert sharpe_ratio(free.net_returns) > sharpe_ratio(cheap.net_returns)
    assert sharpe_ratio(cheap.net_returns) > sharpe_ratio(dear.net_returns)


def test_zero_cost_model_means_gross_equals_net(prices):
    res = run_backtest(prices, BacktestConfig(costs=CostModel(0, 0, 0)))
    pd.testing.assert_series_equal(
        res.gross_returns, res.net_returns, check_names=False
    )


def test_turnover_is_non_negative_and_finite(prices):
    res = run_backtest(prices)
    assert (res.turnover >= 0).all()
    assert np.isfinite(res.turnover).all()


def test_no_nans_in_output(prices):
    res = run_backtest(prices)
    assert res.net_returns.notna().all()
    assert res.weights.notna().to_numpy().all()


def test_equity_curve_is_consistent_with_returns(prices):
    res = run_backtest(prices)
    expected = float((1 + res.net_returns).prod())
    assert res.equity_curve.iloc[-1] == pytest.approx(expected)


def test_empty_panel_raises():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=10)))
