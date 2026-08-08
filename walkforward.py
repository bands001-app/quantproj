"""Walk-forward parameter selection.

A single in-sample optimisation tells you almost nothing: the best parameter
set on a fixed history is the one that best fits that history's noise. Walk
forward instead — fit on a window, trade the next window out-of-sample, roll,
and stitch the out-of-sample segments into one continuous track record.

**Embargo.** The momentum signal at time ``t`` depends on the previous
``lookback + skip`` observations. If an out-of-sample fold begins immediately
after the in-sample window ends, its first signals are built from bars that
were part of the fit. The ``embargo`` parameter drops that many observations at
the fold boundary, which removes the overlap. This is the same idea as purging
in cross-validation for financial series.

The reported out-of-sample Sharpe is still not free — the *procedure* was
chosen by a human — but it is a far better estimate than any in-sample number,
and the distribution of in-sample Sharpes across the grid feeds directly into
the deflated Sharpe calculation in ``metrics``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vtsmom.backtest import BacktestConfig, run_backtest
from vtsmom.metrics import sharpe_ratio

__all__ = ["WalkForwardConfig", "WalkForwardResult", "walk_forward", "generate_folds"]


@dataclass(frozen=True)
class WalkForwardConfig:
    train_periods: int = 756  # ~3 years
    test_periods: int = 252  # ~1 year
    embargo: int = 130  # >= lookback + skip of the widest candidate
    expanding: bool = False  # False == rolling window
    base: BacktestConfig = field(default_factory=BacktestConfig)
    lookback_grid: Sequence[int] = (21, 42, 63, 126, 189, 252)
    vol_halflife_grid: Sequence[int] = (16, 32, 64)


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    oos_turnover: pd.Series
    selections: pd.DataFrame
    is_sharpes: pd.DataFrame
    config: WalkForwardConfig

    @property
    def n_trials(self) -> int:
        """Number of distinct configurations searched per fold."""
        return len(self.config.lookback_grid) * len(self.config.vol_halflife_grid)

    @property
    def sr_variance(self) -> float:
        """Dispersion of in-sample Sharpes attributable to the *search*.

        The deflated Sharpe ratio wants to know how far apart the candidate
        Sharpes were, because a wider spread means the maximum of the set is
        more likely to be high by luck alone.

        Variance is computed *within* each fold, across the parameter grid, and
        then averaged over folds. Pooling every fold's Sharpes into one sample
        instead would fold in the variation between market regimes, which is
        not search width and materially over-deflates the result.
        """
        per_fold = []
        for _, row in self.is_sharpes.iterrows():
            vals = row.to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 1:
                per_fold.append(np.var(vals, ddof=1))
        return float(np.mean(per_fold)) if per_fold else 0.0


def generate_folds(
    n: int, cfg: WalkForwardConfig
) -> Iterable[tuple[int, int, int, int]]:
    """Yield ``(train_start, train_end, test_start, test_end)`` index bounds.

    Bounds are half-open on the right. ``test_start`` already includes the
    embargo gap.
    """
    train_end = cfg.train_periods
    while True:
        test_start = train_end + cfg.embargo
        test_end = min(test_start + cfg.test_periods, n)
        if test_start >= n or test_end - test_start < cfg.test_periods // 4:
            break
        train_start = 0 if cfg.expanding else max(0, train_end - cfg.train_periods)
        yield train_start, train_end, test_start, test_end
        train_end += cfg.test_periods


def _with_params(base: BacktestConfig, lookback: int, halflife: int) -> BacktestConfig:
    return BacktestConfig(
        lookback=lookback,
        skip=base.skip,
        vol_halflife=halflife,
        signal_cap=base.signal_cap,
        target_vol=base.target_vol,
        max_leverage=base.max_leverage,
        execution_lag=base.execution_lag,
        periods_per_year=base.periods_per_year,
        costs=base.costs,
    )


def walk_forward(
    prices: pd.DataFrame, cfg: WalkForwardConfig | None = None
) -> WalkForwardResult:
    """Select parameters in-sample per fold, trade them out-of-sample."""
    cfg = cfg or WalkForwardConfig()
    n = len(prices)
    if n < cfg.train_periods + cfg.embargo + cfg.test_periods // 4:
        raise ValueError(
            f"need at least ~{cfg.train_periods + cfg.embargo} rows, got {n}"
        )

    oos_chunks: list[pd.Series] = []
    turnover_chunks: list[pd.Series] = []
    rows: list[dict] = []
    is_grid: list[pd.Series] = []

    for tr_s, tr_e, te_s, te_e in generate_folds(n, cfg):
        train = prices.iloc[tr_s:tr_e]

        scores: dict[tuple[int, int], float] = {}
        for lb in cfg.lookback_grid:
            if lb + cfg.base.skip >= len(train):
                continue
            for hl in cfg.vol_halflife_grid:
                res = run_backtest(train, _with_params(cfg.base, lb, hl))
                scores[(lb, hl)] = sharpe_ratio(
                    res.net_returns, cfg.base.periods_per_year
                )

        finite = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not finite:
            continue
        best_lb, best_hl = max(finite, key=finite.get)

        is_grid.append(
            pd.Series(
                {f"lb{k[0]}_hl{k[1]}": v for k, v in scores.items()},
                name=prices.index[te_s],
            )
        )

        # The test slice is warmed up with history so the signal is defined on
        # day one of the fold, but only the post-embargo returns are recorded.
        warmup = max(best_lb + cfg.base.skip + 5, 60)
        slice_start = max(0, te_s - warmup)
        test_slice = prices.iloc[slice_start:te_e]

        res = run_backtest(test_slice, _with_params(cfg.base, best_lb, best_hl))
        keep = prices.index[te_s : te_e]
        oos_chunks.append(res.net_returns.reindex(keep).dropna())
        turnover_chunks.append(res.turnover.reindex(keep).fillna(0.0))

        rows.append(
            {
                "test_start": prices.index[te_s],
                "test_end": prices.index[te_e - 1],
                "lookback": best_lb,
                "vol_halflife": best_hl,
                "is_sharpe": finite[(best_lb, best_hl)],
            }
        )

    if not oos_chunks:
        raise RuntimeError("no valid folds produced; check data length and config")

    oos = pd.concat(oos_chunks).sort_index().rename("oos_net")
    turn = pd.concat(turnover_chunks).sort_index().rename("turnover")

    return WalkForwardResult(
        oos_returns=oos[~oos.index.duplicated()],
        oos_turnover=turn[~turn.index.duplicated()],
        selections=pd.DataFrame(rows).set_index("test_start"),
        is_sharpes=pd.DataFrame(is_grid),
        config=cfg,
    )
