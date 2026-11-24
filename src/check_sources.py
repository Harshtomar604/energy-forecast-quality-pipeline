from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"
METEO = "https://archive-api.open-meteo.com/v1/archive"
OUT_PATH = Path("check_sources_output.txt")

END = date.today() - timedelta(days=7)
START = END - timedelta(days=6)
FROM = f"{START}T00:00Z"
TO = f"{END + timedelta(days=1)}T00:00Z"

POINTS = {"north_sea": (54.0, 1.5), "scotland": (57.0, -4.0), "irish_sea": (53.8, -4.0)}


def get_rows(url, params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j["data"] if isinstance(j, dict) and "data" in j else j


def summary(name, s, expected):
    return (f"{name:10} rows={len(s)} (expected ~{expected}) | {s.index.min()} -> {s.index.max()} | "
            f"mean={s.mean():.1f} min={s.min():.1f} max={s.max():.1f} nulls={int(s.isna().sum())}")


def wind():
    df = pd.DataFrame(get_rows(f"{ELEXON}/generation/actual/per-type/wind-and-solar",
                               {"from": FROM, "to": TO, "format": "json"}))
    out = [f"wind columns: {df.columns.tolist()}", f"wind psrTypes: {df['psrType'].unique().tolist()}"]
    df = df[df["psrType"].str.contains("Wind")]
    df["t"] = pd.to_datetime(df["startTime"], utc=True)
    out.append(summary("wind_MW", df.groupby("t")["quantity"].sum(), 336))
    return out


def demand():
    df = pd.DataFrame(get_rows(f"{ELEXON}/demand/outturn",
                               {"settlementDateFrom": str(START), "settlementDateTo": str(END), "format": "json"}))
    out = [f"demand columns: {df.columns.tolist()}"]
    df["t"] = pd.to_datetime(df["startTime"], utc=True)
    out.append(summary("demand_MW", df.set_index("t")["initialDemandOutturn"], 336))
    return out


def price():
    df = pd.DataFrame(get_rows(f"{ELEXON}/balancing/pricing/market-index",
                               {"from": FROM, "to": TO, "format": "json"}))
    out = [f"price columns: {df.columns.tolist()}", f"price providers: {df['dataProvider'].unique().tolist()}"]
    df = df[df["dataProvider"] == "APXMIDP"]
    df["t"] = pd.to_datetime(df["startTime"], utc=True)
    out.append(summary("price", df.set_index("t")["price"], 336))
    return out


def weather():
    frames = []
    for lat, lon in POINTS.values():
        j = requests.get(METEO, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(START), "end_date": str(END),
            "hourly": "wind_speed_100m,temperature_2m",
            "wind_speed_unit": "ms", "timezone": "UTC",
        }, timeout=60).json()
        h = pd.DataFrame(j["hourly"])
        h["t"] = pd.to_datetime(h["time"], utc=True)
        frames.append(h.set_index("t")[["wind_speed_100m", "temperature_2m"]])
    avg = sum(frames) / len(frames)
    return [summary("wind_ms", avg["wind_speed_100m"], 168), summary("temp_C", avg["temperature_2m"], 168)]


def main():
    lines = [f"window: {START} -> {END}"]
    for fn in (wind, demand, price, weather):
        try:
            lines += fn()
        except Exception as e:
            lines.append(f"{fn.__name__.upper()} FAILED: {type(e).__name__}: {e}")
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()