from pathlib import Path

import pandas as pd
import requests

OUT_PATH = Path("check_forecast_weather_output.txt")
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
POINT = (54.0, 1.5)
WINDOWS = [("2024-11-18", "2024-11-24"), ("2025-06-01", "2025-06-07"), ("2026-11-11", "2026-11-17")]
HOURLY = "wind_speed_100m,wind_speed_100m_previous_day1,temperature_2m,temperature_2m_previous_day1"


def check(start, end):
    r = requests.get(URL, params={
        "latitude": POINT[0], "longitude": POINT[1], "start_date": start, "end_date": end,
        "hourly": HOURLY, "wind_speed_unit": "ms", "timezone": "UTC",
    }, timeout=60)
    if r.status_code != 200:
        return [f"{start} -> {end}: HTTP {r.status_code}: {r.text[:300]}"]
    h = pd.DataFrame(r.json()["hourly"])
    lines = [f"{start} -> {end}: rows={len(h)}"]
    for col in h.columns.drop("time"):
        s = h[col]
        lines.append(f"  {col:32} non-null={int(s.notna().sum()):3}  mean={s.mean():.2f}")
    if {"wind_speed_100m", "wind_speed_100m_previous_day1"} <= set(h.columns):
        both = h.dropna(subset=["wind_speed_100m", "wind_speed_100m_previous_day1"])
        if len(both):
            mae = (both["wind_speed_100m"] - both["wind_speed_100m_previous_day1"]).abs().mean()
            lines.append(f"  day-ahead wind speed MAE vs same-day run: {mae:.2f} m/s")
    return lines


def main():
    lines = []
    for start, end in WINDOWS:
        try:
            lines += check(start, end)
        except Exception as e:
            lines.append(f"{start} -> {end}: FAILED {type(e).__name__}: {e}")
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
