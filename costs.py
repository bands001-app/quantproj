"""Transaction costs.

Cost modelling is where most backtests quietly lie. This module keeps the model
explicit and cheap to vary, so that "does the edge survive costs?" is a
one-line sensitivity rather than a rewrite.

Cost per period, in units of portfolio return:

    cost_t = turnover_t * (commission + half_spread) + impact_coef * vol_t * turnover_t^1.5

The first term is linear in traded notional. The second is the square-root
impact law: trading 4x the size costs roughly 8x, not 4x. It scales with
instrument volatility because a more volatile book is more expensive to move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["CostModel"]

_BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Per-unit-notional trading costs.

    Parameters
    ----------
    commission_bps:
        Round-trip broker commission, in basis points of traded notional.
    half_spread_bps:
        Half the quoted bid-ask spread. Crossing the spread costs this much.
    impact_coef:
        Coefficient on the square-root impact term. Set to 0.0 to disable
        impact and model costs as purely linear.
    """

    commission_bps: float = 0.5
    half_spread_bps: float = 1.0
    impact_coef: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_bps", "half_spread_bps", "impact_coef"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def linear_rate(self) -> float:
        """Combined linear cost per unit of turnover."""
        return (self.commission_bps + self.half_spread_bps) * _BPS

    def apply(self, turnover: pd.Series, vol: pd.Series | None = None) -> pd.Series:
        """Cost drag per period, as a return series aligned to ``turnover``.

        ``turnover`` is the sum of absolute weight changes across the book.
        ``vol`` is the portfolio-level volatility estimate, required only when
        ``impact_coef`` is non-zero.
        """
        turnover = turnover.fillna(0.0).clip(lower=0.0)
        cost = turnover * self.linear_rate

        if self.impact_coef:
            if vol is None:
                raise ValueError("impact_coef is non-zero but no vol series supplied")
            impact = self.impact_coef * vol.reindex(turnover.index).fillna(0.0) * np.power(
                turnover, 1.5
            )
            cost = cost + impact

        return cost.rename("cost")
