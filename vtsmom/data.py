"""Price data ingestion.

Two paths are supported:

1. ``load_csv`` — a thin, strict loader for a wide CSV of close prices
   (rows = dates, columns = instruments). This is what you point at real data.
2. ``synthetic_panel`` — a regime-switching geometric Brownian motion generator.
   It exists so the test suite and the demo script are fully reproducible and
   run offline. It is *not* a claim about market realism.

If you want live data, ``yfinance`` slots straight into ``load_csv``'s output
contract; see ``scripts/fetch_data.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["load_csv", "synthetic_panel", "SyntheticSpec"]


def load_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    """Load a wide close-price panel.

    Returns a DataFrame indexed by a sorted, unique, tz-naive DatetimeIndex with
    float columns. Raises on duplicated dates rather than silently aggregating,
    because a duplicated bar is nearly always an upstream join bug.
    """
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"expected a '{date_col}' column, found {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], utc=False)
    df = df.set_index(date_col).sort_index()

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()[:5]
        raise ValueError(f"duplicate timestamps in {path}, e.g. {list(dupes)}")

    df = df.astype(float)
    if (df <= 0).to_numpy().any():
        raise ValueError("non-positive prices present; log returns are undefined")

    df.index.name = "date"
    return df


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters for the synthetic panel.

    The generator alternates between a trending regime and a mean-reverting
    regime via a two-state Markov chain, so that a momentum strategy has
    something to find *and* something to lose money on.
    """

    n_days: int = 3000
    n_assets: int = 8
    trend_drift: float = 0.35 / 252  # annualised drift while trending
    chop_drift: float = 0.0
    daily_vol: float = 0.011
    vol_of_vol: float = 0.25
    p_stay_trend: float = 0.985
    p_stay_chop: float = 0.975
    ar_coef_chop: float = -0.12  # negative autocorrelation in the chop regime
    market_beta: float = 0.45  # shared factor loading across assets
    seed: int = 7


def synthetic_panel(spec: SyntheticSpec | None = None) -> pd.DataFrame:
    """Generate a reproducible close-price panel with trending / choppy regimes."""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n, k = spec.n_days, spec.n_assets

    # --- regime path (shared across assets, per-asset offset in sign) --------
    regime = np.zeros(n, dtype=bool)  # True == trending
    regime[0] = True
    for t in range(1, n):
        stay = spec.p_stay_trend if regime[t - 1] else spec.p_stay_chop
        regime[t] = regime[t - 1] if rng.random() < stay else not regime[t - 1]

    # Each asset picks a trend direction that persists for the length of a run.
    run_id = np.cumsum(np.r_[0, np.diff(regime.astype(int)) != 0])
    signs = rng.choice([-1.0, 1.0], size=(run_id.max() + 1, k))
    direction = signs[run_id]  # (n, k)

    # --- stochastic volatility ---------------------------------------------
    log_vol = np.log(spec.daily_vol) + spec.vol_of_vol * np.cumsum(
        rng.standard_normal((n, k)) * 0.05, axis=0
    )
    vol = np.exp(log_vol - log_vol.mean(axis=0) + np.log(spec.daily_vol))

    # --- returns ------------------------------------------------------------
    market = rng.standard_normal(n) * spec.daily_vol
    idio = rng.standard_normal((n, k))
    rets = np.zeros((n, k))

    for t in range(n):
        drift = spec.trend_drift if regime[t] else spec.chop_drift
        shock = vol[t] * idio[t] + spec.market_beta * market[t]
        rets[t] = drift * direction[t] + shock
        if not regime[t] and t > 0:
            rets[t] += spec.ar_coef_chop * rets[t - 1]

    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    index = pd.bdate_range("2008-01-01", periods=n, name="date")
    cols = [f"ASSET_{i:02d}" for i in range(k)]
    return pd.DataFrame(prices, index=index, columns=cols)
