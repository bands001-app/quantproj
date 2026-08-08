"""Performance, risk and statistical-significance metrics.

The headline number in most strategy write-ups is an in-sample Sharpe ratio
chosen as the best of many trials. That number is biased upward, and the bias
grows with the number of configurations tested. This module therefore reports
the Sharpe ratio alongside two corrections from Bailey & Lopez de Prado (2014):

- **Probabilistic Sharpe Ratio (PSR)** — the probability that the true Sharpe
  exceeds a benchmark, given the observed sample length, skewness and kurtosis.
  Non-normal returns are penalised: negative skew and fat tails widen the
  confidence interval around the estimate.
- **Deflated Sharpe Ratio (DSR)** — the PSR evaluated against a benchmark set
  to the *expected maximum* Sharpe under the null of no skill, given that
  ``n_trials`` configurations were searched. A DSR below ~0.95 means the result
  is not distinguishable from the best of a set of coin flips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "annualised_return",
    "annualised_vol",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "performance_summary",
]

_EULER_MASCHERONI = 0.5772156649015329


def _clean(returns: pd.Series) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    return arr[np.isfinite(arr)]


def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualised return."""
    r = _clean(returns)
    if r.size == 0:
        return np.nan
    total = float(np.prod(1.0 + r))
    if total <= 0:
        return -1.0
    return total ** (periods_per_year / r.size) - 1.0


def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _clean(returns)
    if r.size < 2:
        return np.nan
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0
) -> float:
    """Annualised Sharpe ratio. ``rf`` is an annualised risk-free rate."""
    r = _clean(returns)
    if r.size < 2:
        return np.nan
    excess = r - rf / periods_per_year
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int = 252, target: float = 0.0
) -> float:
    """Downside-deviation analogue of the Sharpe ratio."""
    r = _clean(returns)
    if r.size < 2:
        return np.nan
    downside = np.minimum(r - target / periods_per_year, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return np.nan
    return float(np.mean(r - target / periods_per_year) / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the compounded equity curve, as a
    negative fraction."""
    r = _clean(returns)
    if r.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return np.nan
    return annualised_return(returns, periods_per_year) / abs(mdd)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """P(true Sharpe > ``benchmark_sr``), both stated in annualised terms.

    Adjusts for sample length and for the third and fourth moments of the
    return distribution.
    """
    r = _clean(returns)
    n = r.size
    if n < 3:
        return np.nan

    sr = sharpe_ratio(returns, periods_per_year)
    if not np.isfinite(sr):
        return np.nan

    # Work in per-period units; the moment corrections are defined there.
    sr_p = sr / np.sqrt(periods_per_year)
    bench_p = benchmark_sr / np.sqrt(periods_per_year)

    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, bias=False, fisher=False))

    denom_sq = 1.0 - skew * sr_p + (kurt - 1.0) / 4.0 * sr_p**2
    if denom_sq <= 0:
        return np.nan

    z = (sr_p - bench_p) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(
    n_trials: int,
    sr_variance: float,
    periods_per_year: int = 252,
) -> float:
    """Expected maximum Sharpe under the null of zero true skill.

    ``sr_variance`` is the variance of the *annualised* Sharpe ratios across the
    trials that were run. Returns an annualised Sharpe.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    if sr_variance < 0:
        raise ValueError("sr_variance must be non-negative")

    n = float(n_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    factor = (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    return float(np.sqrt(sr_variance) * factor)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sr_variance: float,
    periods_per_year: int = 252,
) -> float:
    """PSR against the expected-maximum-Sharpe null.

    Interpret as: the probability the strategy has genuine skill, having
    accounted for the fact that ``n_trials`` configurations were searched and
    the best one reported. Values above 0.95 are the usual bar.
    """
    bench = expected_max_sharpe(n_trials, sr_variance, periods_per_year)
    return probabilistic_sharpe_ratio(returns, bench, periods_per_year)


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 252,
    turnover: pd.Series | None = None,
    n_trials: int | None = None,
    sr_variance: float | None = None,
) -> pd.Series:
    """One-stop metric table for a return series."""
    out: dict[str, float] = {
        "ann_return": annualised_return(returns, periods_per_year),
        "ann_vol": annualised_vol(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "skew": float(stats.skew(_clean(returns), bias=False))
        if _clean(returns).size > 2
        else np.nan,
        "excess_kurtosis": float(stats.kurtosis(_clean(returns), bias=False))
        if _clean(returns).size > 3
        else np.nan,
        "psr_vs_zero": probabilistic_sharpe_ratio(returns, 0.0, periods_per_year),
        "n_periods": float(_clean(returns).size),
    }

    if turnover is not None:
        out["ann_turnover"] = float(turnover.mean() * periods_per_year)

    if n_trials is not None and sr_variance is not None:
        out["deflated_sharpe"] = deflated_sharpe_ratio(
            returns, n_trials, sr_variance, periods_per_year
        )
        out["expected_max_sharpe_null"] = expected_max_sharpe(
            n_trials, sr_variance, periods_per_year
        )

    return pd.Series(out, name="performance")
