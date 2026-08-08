"""End-to-end research run.

Produces the numbers quoted in the README:

1. A naive full-sample backtest (the number a careless write-up would report).
2. A purged walk-forward run (the number worth believing).
3. The deflated Sharpe ratio, which asks whether (2) survives the fact that a
   parameter grid was searched.
4. A cost sensitivity sweep, which asks at what cost level the edge dies.

Usage
-----
    python scripts/run_research.py                 # synthetic demo data
    python scripts/run_research.py --csv prices.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from vtsmom.backtest import BacktestConfig, run_backtest
from vtsmom.costs import CostModel
from vtsmom.data import SyntheticSpec, load_csv, synthetic_panel
from vtsmom.metrics import performance_summary, sharpe_ratio
from vtsmom.walkforward import WalkForwardConfig, walk_forward


def _fmt(s: pd.Series) -> str:
    return "\n".join(f"  {k:<26} {v:>10.4f}" for k, v in s.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="wide close-price CSV")
    ap.add_argument("--days", type=int, default=4000)
    ap.add_argument("--assets", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="write OOS returns to this CSV")
    args = ap.parse_args()

    if args.csv:
        prices = load_csv(args.csv)
        source = args.csv
    else:
        prices = synthetic_panel(
            SyntheticSpec(n_days=args.days, n_assets=args.assets, seed=args.seed)
        )
        source = "synthetic"

    print(f"\ndata: {source}  |  {prices.shape[0]} rows x {prices.shape[1]} assets")
    print(f"span: {prices.index[0].date()} -> {prices.index[-1].date()}")

    # --- 1. full-sample, default parameters --------------------------------
    full = run_backtest(prices)
    print("\n[1] FULL-SAMPLE BACKTEST (default params, in-sample -- optimistic)")
    print(_fmt(performance_summary(full.net_returns, turnover=full.turnover)))

    # --- 2. purged walk-forward --------------------------------------------
    wf_cfg = WalkForwardConfig(train_periods=756, test_periods=252, embargo=130)
    wf = walk_forward(prices, wf_cfg)

    print(f"\n[2] WALK-FORWARD, OUT-OF-SAMPLE ({len(wf.selections)} folds, "
          f"embargo={wf_cfg.embargo}d)")
    print(_fmt(performance_summary(wf.oos_returns, turnover=wf.oos_turnover)))

    print("\n    parameter selection by fold:")
    sel = wf.selections.copy()
    sel.index = sel.index.date
    print(sel.to_string(float_format=lambda x: f"{x:7.3f}"))

    # --- 3. deflation for the search ---------------------------------------
    dsr = performance_summary(
        wf.oos_returns, n_trials=wf.n_trials, sr_variance=wf.sr_variance
    )
    print(f"\n[3] MULTIPLE-TESTING ADJUSTMENT ({wf.n_trials} configs per fold)")
    print(f"  variance of in-sample Sharpes  {wf.sr_variance:>10.4f}")
    print(f"  expected max Sharpe under H0  {dsr['expected_max_sharpe_null']:>10.4f}")
    print(f"  deflated Sharpe ratio         {dsr['deflated_sharpe']:>10.4f}")
    verdict = "PASSES" if dsr["deflated_sharpe"] > 0.95 else "FAILS"
    print(f"  -> {verdict} the 0.95 skill threshold")

    # --- 4. cost sensitivity -------------------------------------------------
    print("\n[4] COST SENSITIVITY (full sample, bps per unit turnover)")
    rows = []
    for bps in (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0):
        cfg = BacktestConfig(costs=CostModel(commission_bps=bps / 2, half_spread_bps=bps / 2))
        res = run_backtest(prices, cfg)
        rows.append({"cost_bps": bps, "sharpe": sharpe_ratio(res.net_returns)})
    table = pd.DataFrame(rows)
    breakeven = table.loc[table["sharpe"] > 0, "cost_bps"].max()
    print(table.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"  -> edge survives to roughly {breakeven:.0f} bps round-trip")

    if args.out:
        wf.oos_returns.to_csv(args.out, header=True)
        print(f"\nwrote out-of-sample returns -> {args.out}")
    print()


if __name__ == "__main__":
    main()
