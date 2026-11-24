from briefing import extract_numbers, strip_dates, validate

FACTS = {"date": "2026-11-23", "wind_mean_mw": 9624, "price_mean_gbp_mwh": 195.9,
         "wind_change_vs_prev7_pct": -12.3, "lookback_days": 7, "quality_fail": 2}


def test_dates_are_not_counted_as_numbers():
    text = "On 2026-11-23 (23 November 2026, or Nov 23) wind averaged 9,624 MW."
    assert [n for _, n, _ in extract_numbers(text)] == [9624.0]


def test_series_names_are_ignored():
    assert extract_numbers("price_n2ex warned") == []


def test_exact_numbers_pass():
    text = "Wind averaged 9,624 MW, 12.3% lower than the previous 7 days. Price averaged 195.9 GBP/MWh; 2 series failed."
    checked, bad = validate(text, FACTS)
    assert checked == 5 and bad == []


def test_invented_number_is_caught():
    checked, bad = validate("Wind averaged 9,624 MW and price hit 250.0 GBP/MWh.", FACTS)
    assert bad == ["250.0"]


def test_rounding_changes_are_caught():
    _, bad = validate("Price averaged 196 GBP/MWh.", FACTS)
    assert bad == []
    _, bad = validate("Price averaged 195.6 GBP/MWh.", FACTS)
    assert bad == ["195.6"]


def test_gw_conversion_is_accepted():
    _, bad = validate("Wind averaged 9.624 GW.", FACTS)
    assert bad == []


def test_strip_dates_keeps_other_text():
    assert "MW" in strip_dates("23 Nov 2026 wind 5 MW")
