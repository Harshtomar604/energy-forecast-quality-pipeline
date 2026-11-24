from datetime import date

from ingest import window

TODAY = date(2026, 11, 24)


def test_default_window_is_last_7_days_before_lag():
    assert window(None, None, 1, TODAY) == (date(2026, 11, 17), date(2026, 11, 23))


def test_archive_lag_moves_window_back():
    assert window(None, None, 6, TODAY) == (date(2026, 11, 12), date(2026, 11, 18))


def test_explicit_end_is_capped_at_latest_available():
    assert window(date(2024, 11, 18), date(2026, 12, 31), 6, TODAY) == (date(2024, 11, 18), date(2026, 11, 18))


def test_backfill_start_with_default_end():
    assert window(date(2024, 11, 18), None, 1, TODAY) == (date(2024, 11, 18), date(2026, 11, 23))
