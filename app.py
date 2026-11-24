import json
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
ART_DIR = ROOT / "artefacts"
MODEL_LABELS = {
    "baseline": "Baseline",
    "gbm_forecast_weather": "Gradient boosting (forecast weather)",
    "gbm_actual_weather": "Gradient boosting (actual weather, upper bound)",
}
BASELINES = {"wind": "persistence (the 09:30 value from the day before)", "demand": "the same half-hour one week earlier"}
STATUS_ICON = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}
ELEXON_SERIES = ["wind_onshore", "wind_offshore", "demand", "price_apx", "price_n2ex"]

alt.data_transformers.disable_max_rows()


@st.cache_data
def load(name):
    return duckdb.sql(f"SELECT * FROM '{(ART_DIR / name).as_posix()}.parquet'").df()


@st.cache_data
def load_meta():
    return json.loads((ART_DIR / "meta.json").read_text())


st.set_page_config(page_title="GB power: data quality and forecasts", layout="wide")
meta = load_meta()
st.title("GB power data: automated quality checks and day-ahead forecasts")
st.caption(
    f"Elexon BMRS and Open-Meteo data from {meta['data_first']} to {meta['data_last']}. "
    f"Last refreshed {pd.Timestamp(meta['generated_at']):%d %b %Y, %H:%M} UTC."
)

tab_b, tab_fc, tab_q, tab_p, tab_m = st.tabs(
    ["Daily briefing", "Forecasts", "Data quality", "Price research", "Method and limitations"])

with tab_b:
    path = ART_DIR / "briefing.json"
    if path.exists():
        b = json.loads(path.read_text())
        st.subheader(f"Briefing for {b['date']}")
        st.write(b["text"])
        if b["model"] == "template":
            st.info(f"Template briefing written by code, not an LLM ({b['status']}). "
                    f"{b['numbers_checked']} numbers checked against the source tables.")
        elif not b["unsupported_numbers"]:
            st.success(f"Written by {b['model']}, {b['status']}: all {b['numbers_checked']} numbers "
                       "match the source tables.")
        else:
            st.error(f"Unsupported numbers found: {b['unsupported_numbers']}")
        with st.expander("Facts given to the model"):
            st.json(b["facts"])
        st.caption("Code computes every fact from DuckDB, the LLM writes the text, then code extracts every number "
                   "from the text and matches it to a fact. A briefing with an unmatched number is retried once, "
                   "then replaced by a template.")
    else:
        st.info("No briefing has been generated yet.")

with tab_fc:
    target = st.radio("Target", ["wind", "demand"], horizontal=True,
                      format_func=lambda t: "Wind generation" if t == "wind" else "National demand")
    summary = load("forecast_summary").query("target == @target").set_index("model").loc[list(MODEL_LABELS)]
    base_mae = summary.loc["baseline", "mae"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline MAE", f"{base_mae:,.0f} MW", help=f"Baseline: {BASELINES[target]}.")
    c2.metric("Model MAE (forecast weather)", f"{summary.loc['gbm_forecast_weather', 'mae']:,.0f} MW")
    c3.metric("Error reduction vs baseline", f"{1 - summary.loc['gbm_forecast_weather', 'mae'] / base_mae:.1%}")
    c4.metric("Upper bound (actual weather)", f"{summary.loc['gbm_actual_weather', 'mae']:,.0f} MW",
              help="Same model fed actual weather, which would not be known a day ahead.")

    table = summary.reset_index().assign(model=lambda d: d["model"].map(MODEL_LABELS))
    table = table.rename(columns={"model": "Model", "n": "Half-hours tested", "mae": "MAE (MW)",
                                  "nmae_pct": "nMAE (%)", "mape_pct": "MAPE (%)"}).drop(columns="target")
    if target == "wind":
        table = table.drop(columns="MAPE (%)")
    st.dataframe(table.round(2), hide_index=True, width="stretch")

    st.subheader("Forecast vs actual")
    recent = load("backtest_recent").query("target == @target").copy()
    recent["start_time"] = pd.to_datetime(recent["start_time"], utc=True)
    last = recent["start_time"].max()
    first_day = st.slider("Week starting", min_value=recent["start_time"].min().date(),
                          max_value=(last - pd.Timedelta(days=7)).date(),
                          value=(last - pd.Timedelta(days=7)).date(), format="YYYY-MM-DD")
    chosen = st.multiselect("Models", list(MODEL_LABELS), default=["baseline", "gbm_forecast_weather"],
                            format_func=MODEL_LABELS.get)
    week = recent[(recent["start_time"] >= pd.Timestamp(first_day, tz="UTC"))
                  & (recent["start_time"] < pd.Timestamp(first_day, tz="UTC") + pd.Timedelta(days=7))]
    actual = week.drop_duplicates("start_time")[["start_time", "actual"]].rename(columns={"actual": "mw"})
    lines = pd.concat([
        actual.assign(series="Actual"),
        week[week["model"].isin(chosen)].assign(series=lambda d: d["model"].map(MODEL_LABELS))
        .rename(columns={"pred": "mw"})[["start_time", "mw", "series"]],
    ])
    chart = alt.Chart(lines).mark_line(strokeWidth=1.5).encode(
        x=alt.X("start_time:T", title=None),
        y=alt.Y("mw:Q", title="MW"),
        color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Actual"] + [MODEL_LABELS[m] for m in chosen],
                                        range=["#222222", "#9e9e9e", "#1f77b4", "#ff7f0e"][: len(chosen) + 1])),
        tooltip=[alt.Tooltip("start_time:T", format="%d %b %H:%M"), "series:N", alt.Tooltip("mw:Q", format=",.0f")],
    ).properties(height=360)
    st.altair_chart(chart, width="stretch")

    st.subheader("MAE by test month")
    monthly = load("forecast_monthly").query("target == @target").assign(model=lambda d: d["model"].map(MODEL_LABELS))
    bars = alt.Chart(monthly).mark_bar().encode(
        x=alt.X("fold:N", title="Test month"),
        xOffset=alt.XOffset("model:N", sort=list(MODEL_LABELS.values())),
        y=alt.Y("mae:Q", title="MAE (MW)"),
        color=alt.Color("model:N", title=None, sort=list(MODEL_LABELS.values()), legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(range=["#9e9e9e", "#1f77b4", "#ff7f0e"])),
        tooltip=["fold:N", "model:N", alt.Tooltip("mae:Q", format=",.0f"), "n:Q"],
    ).properties(height=320)
    st.altair_chart(bars, width="stretch")
    st.caption("Each month was forecast by a model trained only on data before that month (rolling-origin validation).")

with tab_q:
    card = load("quality_scorecard")
    counts = card["status"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("PASS", int(counts.get("PASS", 0)))
    c2.metric("WARN", int(counts.get("WARN", 0)))
    c3.metric("FAIL", int(counts.get("FAIL", 0)))

    view = card.assign(status=card["status"].map(STATUS_ICON), completeness=(card["completeness"] * 100).round(2))
    view = view.rename(columns={
        "series": "Series", "rows": "Rows", "expected": "Expected", "missing": "Missing",
        "completeness": "Completeness (%)", "gap_days": "Days with gaps", "invalid_periods": "Invalid periods",
        "out_of_range": "Out of range or null", "duplicates_resolved": "Duplicate keys resolved",
        "drift_z": "Drift z", "status": "Status"})
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Drift z": st.column_config.NumberColumn(
            "Drift z", help="Mean of the last 7 days compared with the same weeks a year earlier, "
                            "in standard deviations."),
    })
    st.caption("FAIL: completeness below 98% or invalid settlement periods. "
               "WARN: completeness below 99.9%, out-of-range or null values, or |drift z| above 3.")

    st.subheader("Daily completeness of Elexon series")
    daily = load("quality_daily").query("series in @ELEXON_SERIES").copy()
    daily["completeness"] = (daily["present"] / daily["expected"]).clip(upper=1)
    heat = alt.Chart(daily).mark_rect().encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("series:N", title=None, sort=ELEXON_SERIES),
        color=alt.Color("completeness:Q", title="Share of periods present",
                        scale=alt.Scale(domain=[0, 1], range=["#c62828", "#f9a825", "#e8f5e9"])),
        tooltip=[alt.Tooltip("day:T", format="%d %b %Y"), "series:N", "present:Q", "expected:Q"],
    ).properties(height=200)
    st.altair_chart(heat, width="stretch")
    st.caption("Each day should have 48 settlement periods: 46 on the spring clock-change day and 50 in autumn. "
               "All weather series are complete and are not shown.")

with tab_p:
    st.warning("Research test, not a trading strategy. The day-ahead auction already prices public wind forecasts, "
               "so beating yesterday's average price is not a tradable edge.")
    st.markdown("**Question:** does the day-ahead wind forecast tell us which way the next day's average APX price moves? "
                "Signal = forecast daily wind for tomorrow minus the last wind value known at 09:30 today; "
                "more wind expected means the price is predicted to fall.")
    res = load("price_backtest_results")
    res_view = res.rename(columns={
        "strategy": "Strategy", "days": "Days", "hit_rate_pct": "Hit rate (%)", "p_value_vs_50pct": "p-value vs 50%",
        "corr_signal_vs_price_change": "Correlation", "pnl_per_day_cost_0": "P&L/day, no cost (GBP)",
        "pnl_per_day_cost_1": "P&L/day, GBP 1 cost", "pnl_per_day_cost_2": "P&L/day, GBP 2 cost"})
    st.dataframe(res_view.round(3), hide_index=True, width="stretch")
    daily = load("price_backtest")
    scatter = alt.Chart(daily).mark_circle(size=28, opacity=0.6).encode(
        x=alt.X("signal_forecast:Q", title="Forecast wind change for tomorrow (MW)"),
        y=alt.Y("price_change:Q", title="Change in daily average price (GBP/MWh)"),
        tooltip=[alt.Tooltip("day:T", format="%d %b %Y"), alt.Tooltip("signal_forecast:Q", format=",.0f"),
                 alt.Tooltip("price_change:Q", format=",.1f")],
    ).properties(height=340)
    st.altair_chart(scatter, width="stretch")
    st.caption("Each point is one day. A downward slope means more expected wind goes with falling prices. "
               "Caveats: APX is a within-day index; daily averages hide intraday shape; one year of data; "
               "no slippage, liquidity or position limits; P&L is illustrative only.")

with tab_m:
    st.markdown("""
**Data**
- Elexon BMRS Insights API: actual wind generation (onshore and offshore), national demand outturn, market index price.
- Open-Meteo: actual weather (ERA5 archive) and the day-ahead forecast (Previous Runs API) at six points:
  three wind areas (North Sea, Scotland, Irish Sea) and three population centres (London, Birmingham, Manchester).

**Pipeline**
1. Ingestion into DuckDB, keyed on settlement date and period, so it is safe to re-run.
   Duplicate revisions from the API are counted and the latest publication is kept.
2. Quality checks per series: completeness (46/48/50 periods per day), invalid periods, ranges, nulls, drift.
3. Day-ahead forecasts with gradient boosting, tested on the last 12 months with rolling-origin validation
   against naive baselines; half-hours missing in the source are excluded, not imputed.
4. Price research: whether the forecast change in wind explains the direction of the next day's price.
5. Daily briefing: an LLM summarises the latest day from facts computed in code; every number it writes is
   checked against those facts before publication.

**Limitations**
- The day-ahead weather forecast is Open-Meteo's run from the previous day, which approximates but does not exactly
  match the information available at a real market gate closure.
- Demand is transmission-level outturn, which falls when embedded solar generates; solar is not yet a feature,
  and the model only knows bank holidays, not the whole Christmas period.
- The drift check compares with the same weeks a year earlier, so a series needs a year of history before
  seasonal change stops looking like drift.
- Six weather points are a coarse proxy for GB-wide wind and temperature.
""")
