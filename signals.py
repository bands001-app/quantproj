"""Signal and risk-estimate construction.

Every function here is strictly causal: the value at index ``t`` uses only
information available at the close of ``t``. Execution lag is applied later,
in ``backtest.run_backtest``, so that the lag assumption lives in exactly one
place and can be stress-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["log_returns", "ewma_vol", "momentum_score", "clip_to_unit"]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Continuously compounded returns. First row is NaN by construction."""
    return np.log(prices).diff()


def ewma_vol(
    returns: pd.DataFrame,
    halflife: int = 32,
    min_periods: int = 20,
    annualise: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Exponentially weighted return volatility.

    Halflife rather than span, because halflife is the parameter people can
    reason about ("how long until an observation matters half as much").
    """
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    vol = returns.ewm(halflife=halflife, min_periods=min_periods).std(bias=False)
    if annualise:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def momentum_score(
    returns: pd.DataFrame,
    lookback: int = 126,
    skip: int = 1,
    vol_halflife: int = 32,
) -> pd.DataFrame:
    """Risk-adjusted time-series momentum.

    The score is the trailing cumulative return over ``lookback`` periods,
    divided by the volatility of a return over that horizon. Dividing by
    ``vol * sqrt(lookback)`` puts every asset on a comparable t-statistic-like
    scale, so a single sizing rule works across instruments with very
    different risk.

    ``skip`` drops the most recent observations from the window. The classic
    reason is short-horizon reversal contaminating the momentum signal.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if skip < 0:
        raise ValueError("skip must be non-negative")

    cum = returns.rolling(lookback, min_periods=lookback).sum()
    if skip:
        cum = cum.shift(skip)

    vol = ewma_vol(returns, halflife=vol_halflife)
    if skip:
        vol = vol.shift(skip)

    scale = vol * np.sqrt(lookback)
    score = cum / scale.replace(0.0, np.nan)
    return score.replace([np.inf, -np.inf], np.nan)


def clip_to_unit(score: pd.DataFrame, cap: float = 2.0) -> pd.DataFrame:
    """Map a score onto [-1, 1] by clipping at +/- ``cap`` and rescaling.

    Clipping rather than a smooth squash (e.g. tanh) is deliberate: it keeps
    the mapping linear in the region where most observations live, so the
    relationship between signal strength and position size stays legible.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    return (score.clip(-cap, cap) / cap).fillna(0.0)
