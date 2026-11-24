from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

from ingest import POINTS

warnings.filterwarnings("ignore", message="The inplace keyword")
DB_PATH = Path("data/energy.duckdb")
OUT_PATH = Path("quality_output.txt")
DRIFT_RECENT_DAYS = 7
DRIFT_BASELINE_DAYS = 90
DRIFT_SEASON_HALF_WIDTH = 45
DRIFT_Z_LIMIT = 3.0

SERIES = [
    ("wind_onshore", "wind", "psr_type = 'Wind Onshore'", "mw", 0, 25000),
    ("wind_offshore", "wind", "psr_type = 'Wind Offshore'", "mw", 0, 25000),
    ("demand", "demand", "TRUE", "demand_mw", 10000, 60000),
    ("price_apx", "price", "provider = 'APXMIDP'", "price", -500, 4000),
    ("price_n2ex", "price", "provider = 'N2EXMIDP'", "price", -500, 4000),
] + [
    (f"{prefix}{point}_{short}", table, f"point = '{point}'", col, lo, hi)
    for table, prefix in (("weather", ""), ("weather_forecast", "fc_"))
    for point in POINTS
    for short, col, lo, hi in [("wind_ms", "wind_speed_100m", 0, 60), ("temp_c", "temperature_2m", -30, 40)]
]

DUPLICATES_SQL = """
    SELECT source, SUM(duplicate_keys)::INT
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY source, chunk_start, chunk_end ORDER BY run_at DESC) AS rn
        FROM ingest_log
    )
    WHERE rn = 1
    GROUP BY source
"""


def last_sunday(year, month):
    month_end = (date(year, month, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return month_end - timedelta(days=(month_end.weekday() + 1) % 7)


def expected_periods(d):
    if d == last_sunday(d.year, 3):
        return 46
    if d == last_sunday(d.year, 10):
        return 50
    return 48


def max_period(d, freq):
    return 24 if freq == "hour" else expected_periods(d)


def daily_completeness(df, start, end, freq):
    days = pd.date_range(start, end, freq="D").date
    present = df.groupby("day")["period"].nunique().reindex(days, fill_value=0)
    return pd.DataFrame({
        "day": days,
        "expected": [max_period(d, freq) for d in days],
        "present": present.to_numpy(),
    })


def invalid_periods(df, freq):
    limit = df["day"].map(lambda d: max_period(d, freq))
    return int(((df["period"] < 1) | (df["period"] > limit)).sum())


def out_of_range(s, lo, hi):
    return int(((s < lo) | (s > hi) | s.isna()).sum())


def drift_baseline(daily, recent_days=DRIFT_RECENT_DAYS):
    last = daily.index.max()
    centre = last - timedelta(days=365 + recent_days // 2)
    lo, hi = centre - timedelta(days=DRIFT_SEASON_HALF_WIDTH), centre + timedelta(days=DRIFT_SEASON_HALF_WIDTH)
    seasonal = daily[(daily.index >= lo) & (daily.index <= hi)]
    if len(seasonal) >= DRIFT_SEASON_HALF_WIDTH:
        return seasonal, "seasonal"
    return daily.iloc[-(recent_days + DRIFT_BASELINE_DAYS):-recent_days], "previous_90d"


def drift_z(df, col, recent_days=DRIFT_RECENT_DAYS):
    daily = df.groupby("day")[col].mean().sort_index()
    recent = daily.iloc[-recent_days:]
    base, _ = drift_baseline(daily, recent_days)
    if len(base) < 2 or base.std() == 0:
        return np.nan
    return float((recent.mean() - base.mean()) / base.std())


def status(completeness, bad_values, invalid, z):
    if completeness < 0.98 or invalid > 0:
        return "FAIL"
    if completeness < 0.999 or bad_values > 0 or (not np.isnan(z) and abs(z) > DRIFT_Z_LIMIT):
        return "WARN"
    return "PASS"


def load_series(con, table, where, col):
    if table.startswith("weather"):
        df = con.execute(f"SELECT time, {col} FROM {table} WHERE {where}").df()
        t = pd.to_datetime(df["time"], utc=True)
        df["day"], df["period"] = t.dt.date, t.dt.hour + 1
    else:
        df = con.execute(
            f"SELECT settlement_date AS day, settlement_period AS period, {col} FROM {table} WHERE {where}"
        ).df()
        df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def main():
    con = duckdb.connect(str(DB_PATH))
    windows = {src: (s, e) for src, s, e in con.execute(
        "SELECT source, MIN(chunk_start), MAX(chunk_end) FROM ingest_log GROUP BY source").fetchall()}
    duplicates = dict(con.execute(DUPLICATES_SQL).fetchall())
    run_at = datetime.now(timezone.utc)
    cards, dailies = [], []

    for name, table, where, col, lo, hi in SERIES:
        freq = "hour" if table.startswith("weather") else "sp"
        start, end = windows[table]
        df = load_series(con, table, where, col)
        daily = daily_completeness(df, start, end, freq)
        daily.insert(0, "series", name)
        dailies.append(daily)

        expected = int(daily["expected"].sum())
        missing = int((daily["expected"] - daily["present"]).clip(lower=0).sum())
        completeness = 1 - missing / expected
        invalid = invalid_periods(df, freq)
        bad = out_of_range(df[col], lo, hi)
        z = drift_z(df, col)
        cards.append({
            "run_at": run_at, "series": name, "rows": len(df), "expected": expected,
            "missing": missing, "completeness": round(completeness, 5),
            "gap_days": int((daily["present"] < daily["expected"]).sum()),
            "invalid_periods": invalid, "out_of_range": bad,
            "duplicates_resolved": duplicates.get(table, 0),
            "drift_z": round(z, 2), "status": status(completeness, bad, invalid, z),
        })

    scorecard = pd.DataFrame(cards)
    daily_df = pd.concat(dailies, ignore_index=True)
    con.execute("CREATE OR REPLACE TABLE quality_daily AS SELECT * FROM daily_df")
    con.execute("CREATE TABLE IF NOT EXISTS quality_scorecard AS SELECT * FROM scorecard WHERE FALSE")
    con.execute("INSERT INTO quality_scorecard SELECT * FROM scorecard")
    con.close()

    view = scorecard.drop(columns="run_at").assign(completeness=(scorecard["completeness"] * 100).round(2))
    worst = (daily_df.assign(missing=daily_df["expected"] - daily_df["present"])
             .query("missing > 0").sort_values("missing", ascending=False)
             .groupby("series").head(3).sort_values(["series", "missing"], ascending=[True, False]))
    lines = [f"window {src}: {s} -> {e}" for src, (s, e) in sorted(windows.items())] + [
        "drift baseline: same weeks one year earlier (+/- 45 days), previous 90 days if history is shorter",
        "",
        "=== quality scorecard (completeness in %) ===",
        view.to_string(index=False),
        "",
        "=== worst days per series (up to 3) ===",
        worst.to_string(index=False) if len(worst) else "none",
        "",
        "status counts: " + ", ".join(f"{k} {v}" for k, v in scorecard["status"].value_counts().items()),
    ]
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH}; tables quality_daily and quality_scorecard written")


if __name__ == "__main__":
    main()
