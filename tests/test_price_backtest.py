import numpy as np

from price_backtest import binom_p_two_sided, positions, score


def test_binomial_symmetric_and_bounded():
    assert binom_p_two_sided(5, 10) == 1.0
    assert abs(binom_p_two_sided(0, 10) - binom_p_two_sided(10, 10)) < 1e-12
    assert binom_p_two_sided(9, 10) < 0.05


def test_more_wind_means_short():
    assert positions(np.array([500.0, -300.0, 0.0])).tolist() == [-1.0, 1.0, 0.0]


def test_score_hit_rate_and_costs():
    signal = np.array([100.0, -100.0, 100.0, -100.0])
    dp = np.array([-5.0, 3.0, 2.0, -1.0])
    row = score("t", positions(signal), dp, signal)
    assert row["days"] == 4
    assert row["hit_rate_pct"] == 50.0
    assert row["pnl_per_day_cost_0"] == (5 + 3 - 2 - 1) / 4
    assert row["pnl_per_day_cost_1"] == (5 + 3 - 2 - 1) / 4 - 1
