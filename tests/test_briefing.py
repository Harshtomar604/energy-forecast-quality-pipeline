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


def _facts():
    return {"date": "2026-11-23", "lookback_days": 7, "periods_expected": 48, "wind_mean_mw": 6692,
            "wind_min_mw": 4031, "wind_max_mw": 11466, "wind_mean_prev7_mw": 7963, "demand_mean_mw": 35034,
            "demand_peak_mw": 43134, "price_mean_gbp_mwh": 195.9, "price_min_gbp_mwh": 154.6,
            "price_max_gbp_mwh": 251.2, "wind_forecast_mae_mw": 1034, "wind_baseline_mae_mw": 6304,
            "demand_forecast_mae_mw": 1117, "demand_baseline_mae_mw": 1603, "quality_series_checked": 29,
            "quality_pass": 25, "quality_warn": 2, "quality_fail": 2}


def test_empty_llm_reply_falls_back_to_template(monkeypatch):
    import briefing
    monkeypatch.setenv("LLM_API_KEY", "test")
    calls = []

    def fake(messages):
        calls.append(1)
        return "", "fake-model", "stop"

    monkeypatch.setattr(briefing, "call_llm", fake)
    text, model, status = briefing.generate(_facts())
    assert len(calls) == 2
    assert model == "template"
    assert "too short" in status
    assert "6,692 MW" in text


def test_short_reply_is_rejected():
    from briefing import problems
    issues, _, _ = problems("Wind averaged 6,692 MW.", _facts())
    assert any("too short" in i for i in issues)


def test_template_handles_missing_facts():
    from briefing import template_briefing, validate
    facts = {**_facts(), "wind_mean_mw": None, "wind_min_mw": None, "wind_max_mw": None,
             "price_mean_gbp_mwh": None, "demand_forecast_mae_mw": None, "demand_baseline_mae_mw": None}
    text = template_briefing(facts)
    assert "n/a MW" in text and "None" not in text
    assert validate(text, facts)[1] == []
