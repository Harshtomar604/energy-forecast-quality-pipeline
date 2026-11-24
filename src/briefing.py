import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import warnings

import duckdb
import requests

warnings.filterwarnings("ignore", message="The inplace keyword")
DB_PATH = Path("data/energy.duckdb")
ENV_PATH = Path(".env")
ART_PATH = Path("artefacts/briefing.json")
OUT_PATH = Path("briefing_output.txt")
LOOKBACK_DAYS = 7
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}(?:\s+\d{{4}})?\b",
    rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b",
]
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")

SYSTEM_PROMPT = """You write a short daily briefing on Great Britain's power data for an energy analyst.
Rules:
- Use only the facts in the JSON you are given. Do not add outside knowledge, causes or forecasts.
- Every number you write must appear in the JSON. Copy numbers exactly; do not round, convert units or calculate new ones.
- Keep units as given (MW, GBP/MWh, %). Write percentages with no space (16.0%) and use British English.
- For a negative change, drop the minus sign and say lower or fell (write "14.7% lower", not "-14.7% lower").
- 120 to 180 words, plain prose, no headings, no bullet points.
- Cover: wind, demand, price, how the day-ahead forecasts performed, and any data-quality issues."""


def load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.split(" #")[0].strip().strip("'\""))


def strip_dates(text):
    for pattern in DATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def extract_numbers(text):
    found = []
    for token in NUMBER.findall(strip_dates(text)):
        cleaned = token.replace(",", "")
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        found.append((token, float(cleaned), decimals))
    return found


def allowed_values(facts):
    values = []
    for v in facts.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            values += [abs(v), abs(v) / 1000]
    return values


def validate(text, facts):
    allowed = allowed_values(facts)
    unsupported, checked = [], 0
    for token, value, decimals in extract_numbers(text):
        checked += 1
        tol = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(abs(value) - a) <= tol for a in allowed):
            unsupported.append(token)
    return checked, unsupported


def one(con, sql, params):
    return con.execute(sql, params).fetchone()


def pct_change(new, old):
    return round(100 * (new - old) / abs(old), 1) if old else None


def build_facts(con):
    day = one(con, "SELECT MAX(settlement_date) FROM model_data WHERE demand_mw IS NOT NULL", [])[0]
    prev_start, prev_end = day - timedelta(days=LOOKBACK_DAYS), day - timedelta(days=1)
    stats = """SELECT AVG(wind_mw), MIN(wind_mw), MAX(wind_mw), COUNT(wind_mw), COUNT(*),
                      AVG(demand_mw), MAX(demand_mw), AVG(price), MIN(price), MAX(price)
               FROM model_data WHERE settlement_date BETWEEN ? AND ?"""
    w, wmin, wmax, wn, n, d, dmax, p, pmin, pmax = one(con, stats, [day, day])
    pw, *_, pd_, _, pp, _, _ = one(con, stats, [prev_start, prev_end])
    errors = dict(((t, m), mae) for t, m, mae in con.execute("""
        SELECT target, model, AVG(ABS(actual - pred)) FROM backtest_predictions
        WHERE CAST(settlement_date AS DATE) = ? AND model IN ('baseline', 'gbm_forecast_weather')
        GROUP BY target, model""", [day]).fetchall())
    mape = one(con, """SELECT 100 * AVG(ABS(actual - pred) / actual) FROM backtest_predictions
                       WHERE CAST(settlement_date AS DATE) = ? AND target = 'demand'
                       AND model = 'gbm_forecast_weather'""", [day])[0]
    card = con.execute("""SELECT series, status FROM quality_scorecard
                          WHERE run_at = (SELECT MAX(run_at) FROM quality_scorecard)""").fetchall()
    missing_today = con.execute("""SELECT series, expected - present FROM quality_daily
                                   WHERE day = ? AND present < expected ORDER BY series""", [day]).fetchall()

    def r(x, nd=0):
        return None if x is None else (round(x) if nd == 0 else round(x, nd))

    facts = {
        "date": str(day),
        "lookback_days": LOOKBACK_DAYS,
        "periods_expected": n,
        "wind_periods_present": wn,
        "wind_mean_mw": r(w), "wind_min_mw": r(wmin), "wind_max_mw": r(wmax),
        "wind_mean_prev7_mw": r(pw), "wind_change_vs_prev7_pct": pct_change(w, pw) if w and pw else None,
        "demand_mean_mw": r(d), "demand_peak_mw": r(dmax),
        "demand_mean_prev7_mw": r(pd_), "demand_change_vs_prev7_pct": pct_change(d, pd_) if d and pd_ else None,
        "price_mean_gbp_mwh": r(p, 1), "price_min_gbp_mwh": r(pmin, 1), "price_max_gbp_mwh": r(pmax, 1),
        "price_mean_prev7_gbp_mwh": r(pp, 1), "price_change_vs_prev7_pct": pct_change(p, pp) if p and pp else None,
        "wind_forecast_mae_mw": r(errors.get(("wind", "gbm_forecast_weather"))),
        "wind_baseline_mae_mw": r(errors.get(("wind", "baseline"))),
        "demand_forecast_mae_mw": r(errors.get(("demand", "gbm_forecast_weather"))),
        "demand_baseline_mae_mw": r(errors.get(("demand", "baseline"))),
        "demand_forecast_mape_pct": r(mape, 2),
        "quality_series_checked": len(card),
        "quality_pass": sum(s == "PASS" for _, s in card),
        "quality_warn": sum(s == "WARN" for _, s in card),
        "quality_fail": sum(s == "FAIL" for _, s in card),
        "quality_issues": [f"{series}: {s}" for series, s in card if s != "PASS"],
        "missing_periods_on_date": {series: int(m) for series, m in missing_today},
    }
    for series, m in missing_today:
        facts[f"missing_{series}"] = int(m)
    return facts


def template_briefing(f):
    parts = [f"On {f['date']}, wind generation averaged {f['wind_mean_mw']:,} MW "
             f"(range {f['wind_min_mw']:,} to {f['wind_max_mw']:,} MW), against {f['wind_mean_prev7_mw']:,} MW "
             f"over the previous {f['lookback_days']} days."]
    parts.append(f"Demand averaged {f['demand_mean_mw']:,} MW with a peak of {f['demand_peak_mw']:,} MW.")
    parts.append(f"The APX market index averaged {f['price_mean_gbp_mwh']} GBP/MWh "
                 f"(range {f['price_min_gbp_mwh']} to {f['price_max_gbp_mwh']}).")
    if f["wind_forecast_mae_mw"] is not None:
        parts.append(f"The day-ahead wind model's MAE was {f['wind_forecast_mae_mw']:,} MW "
                     f"against {f['wind_baseline_mae_mw']:,} MW for persistence; the demand model's MAE was "
                     f"{f['demand_forecast_mae_mw']:,} MW against {f['demand_baseline_mae_mw']:,} MW for last week's value.")
    parts.append(f"Data quality: {f['quality_pass']} of {f['quality_series_checked']} series passed, "
                 f"{f['quality_warn']} warned and {f['quality_fail']} failed.")
    return " ".join(parts)


def call_llm(messages):
    base = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    key, model = os.environ.get("LLM_API_KEY"), os.environ.get("LLM_MODEL")
    if not model:
        r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        names = sorted(m["id"] for m in r.json().get("data", []))
        raise RuntimeError(f"LLM_MODEL is not set. Models available to this key: {names}")
    r = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": messages, "temperature": 0.2}, timeout=90)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} for model {model!r}: {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip(), model


def generate(facts):
    if not os.environ.get("LLM_API_KEY"):
        return template_briefing(facts), "template", "skipped: LLM_API_KEY not set", [], 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(facts, indent=2)}]
    model, unsupported, checked = None, [], 0
    for attempt in (1, 2):
        try:
            text, model = call_llm(messages)
        except Exception as e:
            return template_briefing(facts), "template", f"LLM error: {type(e).__name__}: {e}"[:500], [], 0
        checked, unsupported = validate(text, facts)
        if not unsupported:
            return text, model, f"verified on attempt {attempt}", [], checked
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"These numbers are not in the JSON: {unsupported}. "
                                                 "Rewrite the briefing using only numbers from the JSON."}]
    return template_briefing(facts), "template", f"LLM output rejected: unsupported numbers {unsupported}", unsupported, checked


def main():
    load_env()
    con = duckdb.connect(str(DB_PATH))
    facts = build_facts(con)
    text, model, status, unsupported, checked = generate(facts)
    template_checked, template_bad = validate(text, facts)
    generated_at = datetime.now(timezone.utc)
    record = {"date": facts["date"], "generated_at": generated_at.isoformat(timespec="seconds"),
              "model": model, "status": status, "numbers_checked": template_checked,
              "unsupported_numbers": template_bad, "text": text, "facts": facts}

    con.execute("""CREATE TABLE IF NOT EXISTS briefings (
        date DATE, generated_at TIMESTAMPTZ, model VARCHAR, status VARCHAR,
        numbers_checked INTEGER, unsupported VARCHAR, text VARCHAR, facts VARCHAR)""")
    con.execute("INSERT INTO briefings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [facts["date"], generated_at, model, status, template_checked,
                 json.dumps(template_bad), text, json.dumps(facts)])
    con.close()
    ART_PATH.parent.mkdir(exist_ok=True)
    ART_PATH.write_text(json.dumps(record, indent=2))

    lines = [f"date: {facts['date']}", f"model: {model}", f"status: {status}",
             f"numbers checked in published text: {template_checked}, unsupported: {template_bad}",
             "", text, "", "facts:", json.dumps(facts, indent=2)]
    out = "\n".join(lines)
    OUT_PATH.write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
