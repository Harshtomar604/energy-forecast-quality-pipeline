import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import warnings

import duckdb
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

warnings.filterwarnings("ignore", message="The inplace keyword")
DB_PATH = Path("data/energy.duckdb")
SCHEMA_PATH = Path("sql/schema.sql")
OUT_PATH = Path("ingest_output.txt")
ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"
METEO = "https://archive-api.open-meteo.com/v1/archive"
METEO_FORECAST = "https://previous-runs-api.open-meteo.com/v1/forecast"
CHUNK_DAYS = 6
WEATHER_CHUNK_DAYS = 60
REFRESH_DAYS = 7
LAG_DAYS = {"elexon": 1, "weather": 6, "weather_forecast": 1}
POINTS = {
    "north_sea": (54.0, 1.5), "scotland": (57.0, -4.0), "irish_sea": (53.8, -4.0),
    "london": (51.51, -0.13), "birmingham": (52.48, -1.90), "manchester": (53.48, -2.24),
}
RENAME = {
    "settlementDate": "settlement_date", "settlementPeriod": "settlement_period",
    "startTime": "start_time", "publishTime": "publish_time", "psrType": "psr_type",
    "quantity": "mw", "dataProvider": "provider",
    "initialDemandOutturn": "demand_mw", "initialTransmissionSystemDemandOutturn": "tsd_mw",
}
JOBS = [
    ("wind", ["settlement_date", "settlement_period", "psr_type"]),
    ("demand", ["settlement_date", "settlement_period"]),
    ("price", ["settlement_date", "settlement_period", "provider"]),
]

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))


def get_json(url, params):
    r = SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def elexon(path, params):
    j = get_json(f"{ELEXON}{path}", {**params, "format": "json"})
    df = pd.DataFrame(j["data"] if isinstance(j, dict) else j).rename(columns=RENAME)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    for col in ("start_time", "publish_time"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def by_time(path, d0, d1):
    df = elexon(path, {"from": f"{d0 - timedelta(days=1)}T00:00Z", "to": f"{d1 + timedelta(days=1)}T00:00Z"})
    return df[(df["settlement_date"] >= d0) & (df["settlement_date"] <= d1)]


def fetch_wind(d0, d1):
    df = by_time("/generation/actual/per-type/wind-and-solar", d0, d1)
    return df[df["psr_type"].str.startswith("Wind")]


def fetch_demand(d0, d1):
    return elexon("/demand/outturn", {"settlementDateFrom": str(d0), "settlementDateTo": str(d1)})


def fetch_price(d0, d1):
    return by_time("/balancing/pricing/market-index", d0, d1)


def meteo(url, suffix, d0, d1):
    frames = []
    for point, (lat, lon) in POINTS.items():
        j = get_json(url, {
            "latitude": lat, "longitude": lon, "start_date": str(d0), "end_date": str(d1),
            "hourly": f"wind_speed_100m{suffix},temperature_2m{suffix}", "wind_speed_unit": "ms", "timezone": "UTC",
        })
        h = pd.DataFrame(j["hourly"]).rename(columns=lambda c: c.removesuffix(suffix))
        h["time"] = pd.to_datetime(h["time"], utc=True)
        h["point"] = point
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def fetch_weather(d0, d1):
    return meteo(METEO, "", d0, d1)


def fetch_weather_forecast(d0, d1):
    return meteo(METEO_FORECAST, "_previous_day1", d0, d1)


def dedupe(df, keys):
    dups = int(df.duplicated(keys).sum())
    if "publish_time" in df:
        df = df.sort_values("publish_time")
    return df.drop_duplicates(keys, keep="last"), dups


def write(con, table, df):
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    con.execute(f"INSERT OR REPLACE INTO {table} SELECT {', '.join(cols)} FROM df")


def load(con, run_at, table, keys, fetch, d0, d1):
    raw = fetch(d0, d1)
    df, dups = dedupe(raw, keys)
    write(con, table, df)
    con.execute("INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                [run_at, table, d0, d1, len(raw), dups, len(df)])
    print(f"{datetime.now():%H:%M:%S} {table:16} {d0}..{d1} rows={len(df)}", flush=True)


def window(start, end, lag_days, today=None):
    latest = (today or date.today()) - timedelta(days=lag_days)
    end = min(end, latest) if end else latest
    start = start or end - timedelta(days=REFRESH_DAYS - 1)
    return start, end


def main(start=None, end=None):
    windows = {source: window(start, end, lag) for source, lag in LAG_DAYS.items()}
    DB_PATH.parent.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(SCHEMA_PATH.read_text())
    run_at = datetime.now(timezone.utc)
    fetchers = {"wind": fetch_wind, "demand": fetch_demand, "price": fetch_price}
    lines = [f"window {source}: {s} -> {e}" for source, (s, e) in windows.items()]

    e_start, e_end = windows["elexon"]
    d0 = e_start
    while d0 <= e_end:
        d1 = min(d0 + timedelta(days=CHUNK_DAYS - 1), e_end)
        for table, keys in JOBS:
            try:
                load(con, run_at, table, keys, fetchers[table], d0, d1)
            except Exception as e:
                lines.append(f"{table} {d0}..{d1} FAILED: {type(e).__name__}: {e}")
        d0 = d1 + timedelta(days=1)

    for table, fetch in (("weather", fetch_weather), ("weather_forecast", fetch_weather_forecast)):
        w_start, w_end = windows[table]
        d0 = w_start
        while d0 <= w_end:
            d1 = min(d0 + timedelta(days=WEATHER_CHUNK_DAYS - 1), w_end)
            try:
                load(con, run_at, table, ["point", "time"], fetch, d0, d1)
            except Exception as e:
                lines.append(f"{table} {d0}..{d1} FAILED: {type(e).__name__}: {e}")
            d0 = d1 + timedelta(days=1)

    this_run = con.execute("""
        SELECT source, SUM(rows_received)::INT AS received, SUM(duplicate_keys)::INT AS duplicates,
               SUM(rows_written)::INT AS written
        FROM ingest_log WHERE run_at = ? GROUP BY source ORDER BY source
    """, [run_at]).df()
    totals = con.execute("""
        SELECT 'wind' AS tbl, COUNT(*) AS rows, MIN(settlement_date) AS first, MAX(settlement_date) AS last FROM wind
        UNION ALL SELECT 'demand', COUNT(*), MIN(settlement_date), MAX(settlement_date) FROM demand
        UNION ALL SELECT 'price', COUNT(*), MIN(settlement_date), MAX(settlement_date) FROM price
        UNION ALL SELECT 'weather', COUNT(*), MIN(time)::DATE, MAX(time)::DATE FROM weather
        UNION ALL SELECT 'weather_forecast', COUNT(*), MIN(time)::DATE, MAX(time)::DATE FROM weather_forecast
    """).df()
    con.close()

    lines += ["", "=== this run ===", this_run.to_string(index=False),
              "", "=== table totals ===", totals.to_string(index=False)]
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat)
    p.add_argument("--end", type=date.fromisoformat)
    args = p.parse_args()
    main(args.start, args.end)
