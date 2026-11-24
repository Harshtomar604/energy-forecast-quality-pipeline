from datetime import date

import numpy as np
import pandas as pd

from quality import daily_completeness, drift_z, expected_periods, invalid_periods, last_sunday, out_of_range, status


def make(days, counts):
    rows = [(d, p) for d, n in zip(days, counts) for p in range(1, n + 1)]
    return pd.DataFrame(rows, columns=["day", "period"])


def test_clock_change_days():
    assert expected_periods(date(2025, 3, 30)) == 46
    assert expected_periods(date(2025, 10, 26)) == 50
    assert expected_periods(date(2026, 3, 29)) == 46
    assert expected_periods(date(2026, 10, 25)) == 50
    assert expected_periods(date(2026, 11, 17)) == 48


def test_last_sunday_handles_december():
    assert last_sunday(2026, 12) == date(2026, 12, 27)


def test_completeness_counts_missing_periods():
    days = [date(2026, 11, 16), date(2026, 11, 17)]
    out = daily_completeness(make(days, [48, 45]), days[0], days[1], "sp")
    assert out["present"].tolist() == [48, 45]
    assert out["expected"].tolist() == [48, 48]


def test_completeness_reports_absent_day_as_zero():
    out = daily_completeness(make([date(2026, 11, 16)], [48]), date(2026, 11, 16), date(2026, 11, 17), "sp")
    assert out["present"].tolist() == [48, 0]


def test_completeness_on_autumn_clock_change():
    d = date(2026, 10, 25)
    out = daily_completeness(make([d], [50]), d, d, "sp")
    assert out.loc[0, "expected"] == 50
    assert out.loc[0, "present"] == 50


def test_hourly_series_expects_24():
    d = date(2026, 3, 29)
    out = daily_completeness(make([d], [24]), d, d, "hour")
    assert out.loc[0, "expected"] == 24


def test_invalid_periods():
    df = pd.DataFrame({
        "day": [date(2026, 3, 29), date(2026, 3, 29), date(2026, 11, 17), date(2026, 11, 17)],
        "period": [46, 47, 48, 49],
    })
    assert invalid_periods(df, "sp") == 2


def test_out_of_range_counts_nulls():
    assert out_of_range(pd.Series([-1.0, 5.0, 100.0, np.nan]), 0, 50) == 3


def test_drift_flags_a_shift():
    days = pd.date_range("2026-01-01", periods=97).date
    values = [10.0 + (i % 3) for i in range(90)] + [50.0] * 7
    assert drift_z(pd.DataFrame({"day": days, "v": values}), "v") > 3


def test_drift_quiet_when_stable():
    days = pd.date_range("2026-01-01", periods=97).date
    values = [10.0 + (i % 3) for i in range(97)]
    assert abs(drift_z(pd.DataFrame({"day": days, "v": values}), "v")) < 3


def test_status_levels():
    assert status(1.0, 0, 0, 0.5) == "PASS"
    assert status(0.995, 0, 0, 0.5) == "WARN"
    assert status(1.0, 1, 0, 0.5) == "WARN"
    assert status(1.0, 0, 0, 4.0) == "WARN"
    assert status(0.97, 0, 0, 0.5) == "FAIL"
    assert status(1.0, 0, 1, 0.5) == "FAIL"
    assert status(1.0, 0, 0, np.nan) == "PASS"


def test_drift_uses_same_season_last_year():
    days = pd.date_range("2025-01-01", "2026-01-07").date
    values = [5.0 + (i % 3) if (d.month in (12, 1)) else 20.0 + (i % 3) for i, d in enumerate(days)]
    df = pd.DataFrame({"day": days, "v": values})
    assert abs(drift_z(df, "v")) < 3
