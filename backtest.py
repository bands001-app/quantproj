"""Vectorised backtest engine.

Design notes
------------
*Execution lag.* Signals are computed from data up to and including the close
of day ``t``, and the resulting weights are held over day ``t + lag``. The lag
is applied exactly once, here, via ``.shift(lag)``. This is the single most
common source of lookahead bias in backtests, so it is isolated and tested.

*Vol targeting at the portfolio level, not the asset level.* Sizing each leg to
its own vol target ignores correlation and delivers a portfolio whose realised
vol wanders with the correlation regime. Here, raw weights set relative risk
across the book, then one scalar rescales the whole book to the target.

*Ex-ante scaling only.* The scalar at time ``t`` uses a covariance estimate
built from returns up to ``t``. It is never fitted to the realised vol of the
period it scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vtsmom.costs import CostModel
from vtsmom.signals import clip_to_unit, ewma_vol, log_returns, momentum_score

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]


@dataclass(frozen=True)
class BacktestConfig:
    lookback: int = 126
    skip: int = 1
    vol_halflife: int = 32
    signal_cap: float = 2.0
    target_vol: float = 0.10  # annualised
    max_leverage: float = 3.0  # cap on gross exposure
    execution_lag: int = 1
    periods_per_year: int = 252
    costs: CostModel = field(default_factory=CostModel)

    def __post_init__(self) -> None:
        if self.execution_lag < 1:
            raise ValueError(
                "execution_lag must be >= 1; a lag of 0 trades on the same bar "
                "that generated the signal, which is lookahead"
            )
        if self.target_vol <= 0:
            raise ValueError("target_vol must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    config: BacktestConfig

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.net_returns).cumprod().rename("equity")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gross": self.gross_returns,
                "net": self.net_returns,
                "cost": self.costs,
                "turnover": self.turnover,
                "equity": self.equity_curve,
            }
        )


def _portfolio_vol_estimate(
    raw_weights: pd.DataFrame,
    returns: pd.DataFrame,
    halflife: int,
    periods_per_year: int,
) -> pd.Series:
    """Ex-ante annualised vol of the raw-weight portfolio.

    Uses an EWMA correlation matrix combined with EWMA per-asset vols. The
    correlation is estimated on standardised returns, which keeps the estimate
    stable when individual instruments go through vol spikes.
    """
    vol = ewma_vol(returns, halflife=halflife)
    standardised = (returns / vol.replace(0.0, np.nan)).clip(-5.0, 5.0)

    n = len(returns)
    out = np.full(n, np.nan)
    alpha = 1.0 - np.exp(np.log(0.5) / halflife)

    cov = None
    w_arr = raw_weights.to_numpy()
    s_arr = standardised.to_numpy()
    v_arr = vol.to_numpy()

    for t in range(n):
        x = s_arr[t]
        if np.isnan(x).any():
            continue
        outer = np.outer(x, x)
        cov = outer if cov is None else (1 - alpha) * cov + alpha * outer

        d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)

        w, v = w_arr[t], v_arr[t]
        if np.isnan(w).any() or np.isnan(v).any():
            continue
        wv = w * v
        var = float(wv @ corr @ wv)
        out[t] = np.sqrt(max(var, 0.0) * periods_per_year)

    return pd.Series(out, index=returns.index, name="port_vol")


def run_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run the vol-targeted time-series momentum strategy over a price panel."""
    config = config or BacktestConfig()
    if prices.shape[1] == 0:
        raise ValueError("price panel has no columns")

    returns = log_returns(prices)

    score = momentum_score(
        returns,
        lookback=config.lookback,
        skip=config.skip,
        vol_halflife=config.vol_halflife,
    )
    raw = clip_to_unit(score, cap=config.signal_cap)

    # Equal risk budget per active leg: normalise by the number of live signals.
    active = (raw != 0).sum(axis=1).replace(0, np.nan)
    raw = raw.div(active, axis=0).fillna(0.0)

    port_vol = _portfolio_vol_estimate(
        raw, returns, config.vol_halflife, config.periods_per_year
    )
    scalar = (config.target_vol / port_vol.replace(0.0, np.nan)).fillna(0.0)

    weights = raw.mul(scalar, axis=0)

    # Cap gross exposure. Scale the whole book down so relative bets are kept.
    gross = weights.abs().sum(axis=1)
    overshoot = (gross / config.max_leverage).clip(lower=1.0)
    weights = weights.div(overshoot, axis=0).fillna(0.0)

    # --- execution lag: the only place a shift is applied -------------------
    held = weights.shift(config.execution_lag).fillna(0.0)

    gross_returns = (held * returns).sum(axis=1).rename("gross")
    turnover = held.diff().abs().sum(axis=1).fillna(0.0).rename("turnover")

    costs = config.costs.apply(turnover, vol=port_vol / np.sqrt(config.periods_per_year))
    net_returns = (gross_returns - costs).rename("net")

    # Trim the warm-up period where no position could exist.
    first_live = held.abs().sum(axis=1).to_numpy().nonzero()[0]
    if len(first_live):
        start = returns.index[first_live[0]]
        gross_returns = gross_returns.loc[start:]
        net_returns = net_returns.loc[start:]
        costs = costs.loc[start:]
        turnover = turnover.loc[start:]
        held = held.loc[start:]

    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        costs=costs,
        turnover=turnover,
        weights=held,
        config=config,
    )
