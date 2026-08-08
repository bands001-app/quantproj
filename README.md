# vtsmom — volatility-targeted time-series momentum

A backtesting and validation toolkit for a cross-asset time-series momentum
strategy, built to answer one question properly: **is the edge real, or is it
the best of many guesses?**

Most strategy repositories stop at an equity curve. The hard part is not
producing a Sharpe ratio — it is producing one you can defend. This repo puts
the validation machinery on equal footing with the strategy: purged
walk-forward selection, an explicit execution lag with a test that proves it,
a transaction-cost sensitivity sweep, and the deflated Sharpe ratio to price in
the cost of searching a parameter grid.

The headline result is a negative one, and it is reported as such.

[![ci](https://github.com/USERNAME/vtsmom/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/vtsmom/actions)

---

## Results

Run on the bundled synthetic panel (8 assets, ~4,000 business days, regime-switching
generator). Reproduce with `python scripts/run_research.py`.

| | Full-sample (in-sample) | Walk-forward (out-of-sample) |
|---|---|---|
| Annualised return | 11.3% | 11.3% |
| Annualised vol | 9.9% | 10.3% |
| **Sharpe** | **1.13** | **1.09** |
| Sortino | 1.70 | 1.61 |
| Max drawdown | −30.2% | −33.4% |
| Calmar | 0.37 | 0.34 |
| Annual turnover | 31.8× | 48.6× |

Realised volatility lands at 10.3% against a 10% target, so the ex-ante sizing
is doing its job. Out-of-sample performance holds up against in-sample, which
is the first thing worth checking and the point at which most write-ups stop.

### The finding: it does not survive deflation

18 parameter configurations were searched per fold. Under the null of zero
skill, the *expected maximum* Sharpe across that many trials is **1.02** —
which is most of the 1.09 that was actually achieved.

```
variance of in-sample Sharpes      0.3031
expected max Sharpe under H0       1.0206
deflated Sharpe ratio              0.5906   <- needs > 0.95
```

A deflated Sharpe of 0.59 means there is roughly a 41% chance a skill-free
strategy searching the same grid would have looked this good. **That is not
evidence of an edge.** The honest conclusion from this run is that the grid is
too wide for the amount of data, not that momentum works.

Two ways to fix it, both of which are real research directions rather than
knobs to turn until the number goes green: shrink the parameter grid so fewer
trials are spent, or extend the sample so the Sharpe estimate tightens.

### Cost sensitivity

| Round-trip cost (bps) | 0 | 1 | 2.5 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|---|
| Sharpe | 1.18 | 1.15 | 1.10 | 1.02 | 0.86 | 0.54 | −0.10 |

At ~48× annual turnover the strategy is cost-sensitive by construction. It
breaks even somewhere between 20 and 40 bps round-trip — comfortable for liquid
futures, fatal for anything with a wide spread. Any claim about this strategy
that does not name a cost assumption is meaningless.

---

## Method

**Signal.** Risk-adjusted time-series momentum: trailing cumulative return over
a lookback window, divided by `vol × √lookback` so that instruments with
different risk profiles land on a comparable scale. A one-day skip drops the
most recent bar to avoid short-horizon reversal contaminating the signal.

**Sizing.** Scores are clipped to [−1, 1], split across active legs, then the
*whole book* is scaled by a single ex-ante factor targeting 10% annualised
volatility. Portfolio-level rather than per-asset targeting is deliberate:
sizing each leg to its own vol target ignores correlation and produces a
portfolio whose realised vol drifts with the correlation regime. Gross exposure
is capped at 3×.

**Volatility estimate.** EWMA per-asset vols combined with an EWMA correlation
matrix estimated on standardised, winsorised returns — which keeps the
correlation stable when a single instrument goes through a vol spike.

**Execution.** Signals computed from the close of day *t* are held over day
*t+1*. The lag is applied in exactly one line of `backtest.py`, and a config
with `execution_lag=0` raises rather than silently running a lookahead backtest.

**Costs.** Linear commission and half-spread, plus an optional square-root
market-impact term scaled by instrument volatility — trading 4× the size costs
roughly 8×, not 4×.

### Validation

**Purged walk-forward.** Fit the parameter grid on a rolling 3-year window,
trade the best configuration out-of-sample for the following year, roll, and
stitch the out-of-sample segments into one continuous track record.

The **embargo** matters and is easy to get wrong. A momentum signal at time *t*
depends on the previous `lookback + skip` bars. If the out-of-sample fold starts
the day after the training window ends, its first signals are built from bars
that were part of the fit. A 130-day gap at each fold boundary removes that
overlap. This is the same idea as purging in cross-validation for time series.

**Deflated Sharpe ratio** (Bailey & López de Prado, 2014). The Probabilistic
Sharpe Ratio gives P(true Sharpe > benchmark) given the sample length, skewness
and kurtosis — non-normal returns are penalised. The deflated version sets that
benchmark to the expected maximum Sharpe under the null, given the number of
configurations searched.

One judgement call worth flagging: search-width variance is computed *within*
each fold across the grid, then averaged across folds. Pooling every fold's
Sharpes into one sample instead would fold in variation between market regimes,
which is not search width — that choice alone moves the deflated Sharpe from
0.01 to 0.59, so it is documented rather than buried.

### Testing

41 tests, 86% coverage. The one that matters most:

```python
def test_no_lookahead_future_prices_cannot_change_the_past(prices):
    """Overwrite the final third of the panel. Every return the strategy
    produced before that point must be bit-identical."""
    boundary = prices.index[cut - 5]
    a = run_backtest(prices).net_returns.loc[:boundary]
    b = run_backtest(tampered).net_returns.loc[:boundary]
    pd.testing.assert_series_equal(a, b)
```

If future data can change the past, this fails. It caught a real bug during
development — an off-by-one in how the warm-up period was trimmed.

Other properties under test: Sharpe against its closed form and its
scale-invariance, PSR equal to 0.5 when the benchmark equals the estimate,
PSR increasing with sample length at fixed Sharpe, expected-max-Sharpe monotone
in trial count, drawdown on a known path, vol targeting scaling linearly with
the target, the leverage cap binding, costs monotonically reducing Sharpe,
signal causality, and walk-forward folds being disjoint with the embargo gap
respected.

---

## Usage

```bash
git clone https://github.com/USERNAME/vtsmom.git
cd vtsmom
pip install -e ".[dev]"

pytest                              # 41 tests
python scripts/run_research.py      # full research pipeline
```

```python
from vtsmom import BacktestConfig, CostModel, run_backtest, performance_summary
from vtsmom.data import load_csv

prices = load_csv("data/prices.csv")     # wide panel: rows = dates, cols = tickers

result = run_backtest(prices, BacktestConfig(
    lookback=126,
    target_vol=0.10,
    max_leverage=3.0,
    costs=CostModel(commission_bps=0.5, half_spread_bps=1.0),
))

print(performance_summary(result.net_returns, turnover=result.turnover))
```

Walk-forward:

```python
from vtsmom import WalkForwardConfig, walk_forward
from vtsmom.metrics import deflated_sharpe_ratio

wf = walk_forward(prices, WalkForwardConfig(train_periods=756, test_periods=252))

print(wf.selections)          # chosen parameters per fold
print(deflated_sharpe_ratio(wf.oos_returns, wf.n_trials, wf.sr_variance))
```

### Real data

The package ships with a regime-switching synthetic generator so the tests and
demo run offline and deterministically. It is scaffolding, not a claim about
market realism. For real prices:

```bash
pip install yfinance
python scripts/fetch_data.py --tickers SPY TLT GLD DBC EFA IWM --out data/prices.csv
python scripts/run_research.py --csv data/prices.csv
```

---

## Layout

```
src/vtsmom/
  data.py          CSV loader + synthetic regime-switching generator
  signals.py       momentum score, EWMA volatility (strictly causal)
  costs.py         linear + square-root-impact cost model
  backtest.py      vectorised engine, vol targeting, execution lag
  metrics.py       Sharpe/Sortino/drawdown, PSR, deflated Sharpe
  walkforward.py   purged walk-forward with embargo
tests/             41 tests including the lookahead-bias check
scripts/           research pipeline + optional data fetcher
```

## Limitations

Stated plainly, because a strategy repo without a limitations section is a
sales pitch:

- Headline results are on **synthetic data**. The generator was written to
  contain a momentum edge, so finding one is not a discovery.
- **No survivorship-bias-free universe.** The real-data path uses whatever
  tickers you pass, which will be survivors.
- **Costs are a model, not fills.** No order book, no queue position, no
  partial fills, no borrow cost for shorts.
- **Daily bars only.** Intraday path within the day is invisible, so intraday
  risk limits cannot be evaluated.
- **The deflated Sharpe here fails its own threshold.** See above.

## References

- Bailey, D. and López de Prado, M. (2014). *The Deflated Sharpe Ratio*.
  Journal of Portfolio Management.
- Moskowitz, T., Ooi, Y. and Pedersen, L. (2012). *Time Series Momentum*.
  Journal of Financial Economics.
- Harvey, C. and Liu, Y. (2015). *Backtesting*. Journal of Portfolio Management.

MIT licensed.
