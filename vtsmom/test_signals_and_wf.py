import numpy as np
import pandas as pd
import pytest

from vtsmom.costs import CostModel
from vtsmom.data import SyntheticSpec, synthetic_panel
from vtsmom.signals import clip_to_unit, ewma_vol, log_returns, momentum_score
from vtsmom.walkforward import WalkForwardConfig, generate_folds, walk_forward


@pytest.fixture(scope="module")
def rets():
    return log_returns(synthetic_panel(SyntheticSpec(n_days=1200, n_assets=4, seed=21)))


# --------------------------------------------------------------------- signals


def test_momentum_is_causal(rets):
    """Perturbing the tail must not alter earlier signal values."""
    tampered = rets.copy()
    tampered.iloc[800:] += 0.05

    a = momentum_score(rets, lookback=126).iloc[:795]
    b = momentum_score(tampered, lookback=126).iloc[:795]
    pd.testing.assert_frame_equal(a, b)


def test_momentum_warmup_is_nan(rets):
    score = momentum_score(rets, lookback=126, skip=1)
    assert score.iloc[:126].isna().to_numpy().all()


def test_momentum_sign_follows_trend():
    idx = pd.bdate_range("2020-01-01", periods=300)
    up = pd.DataFrame({"A": np.full(300, 0.001)}, index=idx)
    # A constant series has zero vol, so add a small amount of noise.
    rng = np.random.default_rng(3)
    up["A"] += rng.normal(0, 0.002, 300)
    score = momentum_score(up, lookback=100, skip=1).dropna()
    assert score["A"].mean() > 0


def test_ewma_vol_recovers_known_sigma():
    rng = np.random.default_rng(4)
    r = pd.DataFrame({"A": rng.normal(0, 0.02, 4000)})
    est = ewma_vol(r, halflife=64).dropna()
    assert est["A"].mean() == pytest.approx(0.02, rel=0.15)


def test_ewma_vol_rejects_bad_halflife(rets):
    with pytest.raises(ValueError):
        ewma_vol(rets, halflife=0)


def test_clip_bounds_output(rets):
    w = clip_to_unit(momentum_score(rets), cap=2.0)
    assert w.abs().to_numpy().max() <= 1.0 + 1e-12
    assert w.notna().to_numpy().all()


# ----------------------------------------------------------------------- costs


def test_cost_is_linear_in_turnover():
    cm = CostModel(commission_bps=1.0, half_spread_bps=1.0, impact_coef=0.0)
    t = pd.Series([0.0, 0.1, 0.2])
    c = cm.apply(t)
    assert c.iloc[2] == pytest.approx(2 * c.iloc[1])
    assert c.iloc[0] == 0.0


def test_impact_is_superlinear():
    cm = CostModel(0, 0, impact_coef=0.5)
    t = pd.Series([0.1, 0.2])
    v = pd.Series([0.01, 0.01])
    c = cm.apply(t, v)
    assert c.iloc[1] > 2 * c.iloc[0]


def test_impact_without_vol_raises():
    with pytest.raises(ValueError, match="vol"):
        CostModel(0, 0, impact_coef=0.5).apply(pd.Series([0.1]))


def test_negative_cost_params_rejected():
    with pytest.raises(ValueError):
        CostModel(commission_bps=-1.0)


# ---------------------------------------------------------------- walk-forward


def test_folds_are_disjoint_and_ordered():
    cfg = WalkForwardConfig(train_periods=500, test_periods=250, embargo=100)
    folds = list(generate_folds(3000, cfg))
    assert len(folds) > 1
    for _, _, ts, te in folds:
        assert ts < te
    starts = [f[2] for f in folds]
    assert starts == sorted(starts)
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert prev[3] <= nxt[2]  # no overlap between test windows


def test_embargo_gap_is_respected():
    cfg = WalkForwardConfig(train_periods=500, test_periods=250, embargo=130)
    for _, tr_e, te_s, _ in generate_folds(3000, cfg):
        assert te_s - tr_e == 130


def test_rolling_window_has_fixed_train_length():
    cfg = WalkForwardConfig(train_periods=500, test_periods=250, embargo=50, expanding=False)
    for tr_s, tr_e, _, _ in generate_folds(3000, cfg):
        assert tr_e - tr_s == 500


def test_expanding_window_grows():
    cfg = WalkForwardConfig(train_periods=500, test_periods=250, embargo=50, expanding=True)
    lengths = [tr_e - tr_s for tr_s, tr_e, _, _ in generate_folds(3000, cfg)]
    assert lengths == sorted(lengths)
    assert lengths[-1] > lengths[0]


def test_walk_forward_end_to_end():
    prices = synthetic_panel(SyntheticSpec(n_days=2200, n_assets=4, seed=31))
    cfg = WalkForwardConfig(
        train_periods=600,
        test_periods=250,
        embargo=130,
        lookback_grid=(42, 126),
        vol_halflife_grid=(32,),
    )
    res = walk_forward(prices, cfg)

    assert len(res.oos_returns) > 200
    assert res.oos_returns.notna().all()
    assert res.oos_returns.index.is_monotonic_increasing
    assert not res.oos_returns.index.has_duplicates
    assert res.n_trials == 2
    assert res.sr_variance >= 0
    assert set(res.selections["lookback"]).issubset({42, 126})


def test_walk_forward_rejects_short_history():
    prices = synthetic_panel(SyntheticSpec(n_days=200, n_assets=3))
    with pytest.raises(ValueError, match="at least"):
        walk_forward(prices, WalkForwardConfig(train_periods=600))
