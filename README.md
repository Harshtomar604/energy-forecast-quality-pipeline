# GB power data: automated quality checks, day-ahead forecasts and an LLM briefing

**Can a small, automated pipeline keep GB power data trustworthy and turn it into useful day-ahead forecasts?**
This project ingests two years of half-hourly Elexon and Open-Meteo data into DuckDB, checks its quality every day,
forecasts wind generation and national demand a day ahead, tests whether the wind forecast explains price moves,
and publishes an LLM-written daily briefing whose every number is checked in code.

**Live app:** https://energy-forecast-quality-pipeline-ht.streamlit.app/ (data refreshed daily by GitHub Actions)

## Results

Rolling-origin validation: 12 monthly test folds (Dec 2025 to Nov 2026), each forecast by a model trained only on
earlier data. Inputs are the **day-ahead** weather forecast (what was known the day before), not actual weather.

**Wind generation**

| Model | MAE (MW) | nMAE | vs baseline |
|---|---|---|---|
| Persistence (09:30 value the day before) | 4,550 | 45.7% | |
| **Gradient boosting, forecast weather** | **1,596** | **16.0%** | **−64.9%** |
| Gradient boosting, actual weather (upper bound) | 1,278 | 12.8% | −71.9% |

**National demand**

| Model | MAE (MW) | MAPE | vs baseline |
|---|---|---|---|
| Same half-hour one week earlier | 2,123 | 8.41% | |
| **Gradient boosting, forecast weather** | **1,518** | **6.10%** | **−28.5%** |
| Gradient boosting, actual weather (upper bound) | 1,457 | 5.90% | −31.4% |

- The wind model beats persistence in every test month. Weather-forecast error accounts for about a fifth of its error
  (1,596 vs 1,278 MW with perfect weather).
- For demand, better weather barely helps (1,518 vs 1,457 MW); the weak months are December (Christmas) and March.

## Data quality findings

The daily checks (completeness against 46/48/50 periods, invalid periods, ranges and nulls, seasonal drift) found:

- **An off-by-one in the API:** time-filtered Elexon endpoints treat `to` as inclusive (337 rows instead of 336),
  so every table is keyed on settlement date and period instead of timestamps.
- **A five-week hole in wind data** (19 May to 25 June 2026): about a third of half-hours missing. Re-fetching returned
  the same rows, so the gap is in the source. Wind is 97.8% complete overall; missing periods are excluded, not imputed.
- **2,874 duplicate revisions** of wind data (mostly March to April 2026); the latest publication is kept.
- **N2EX price** is missing whole days (29 to 30 June 2025), so the complete APX index is used.
- **A real price shift:** the drift check flags APX prices, which averaged GBP 201.1/MWh in 1 to 23 November 2026 against
  GBP 75.9/MWh a year earlier. Comparing with the same weeks last year stops seasonal temperature change from
  raising false alarms.

## Price research (not a trading strategy)

Signal: tomorrow's forecast daily wind minus the last value known at 09:30 today; more wind means the price is
predicted to fall. Over 314 days, the signal called the direction of the daily average APX price **74.8%** of the time
(p < 0.0001, correlation −0.67), rising to 90.5% for the strongest third of signals. An oracle using actual wind
scored 73.2%, so the forecast is good enough for this purpose.

**This is not evidence of a trading edge.** The day-ahead auction clears before delivery and already prices the same
public wind forecasts, so yesterday's average price is not a price anyone could trade at. A proper test needs the
day-ahead auction price, which this dataset does not include.

## LLM daily briefing with a hallucination check

1. Code computes the day's facts from DuckDB (wind, demand, price, forecast errors, quality status).
2. An LLM (`openai/gpt-oss-120b` on Groq's free tier) writes a 120 to 180 word briefing from those facts only.
3. Code extracts every number from the text (ignoring dates) and matches it to a fact, allowing display rounding and
   MW to GW conversion. Any unmatched number triggers one retry with the list of offenders; if that fails too,
   a template briefing written by code is published instead and the failure is recorded.

The check confirms that every number in the text exists in the facts, not that it is attached to the right quantity, so a misattributed number can still pass; the prompt therefore requires digits and exact series names.

## Pipeline

```
Elexon Insights API ─┐
                     ├─> ingest.py ─> DuckDB ─> quality.py ─> forecast.py ─> price_backtest.py ─> briefing.py
Open-Meteo APIs ─────┘   (keyed,               (scorecard)    (rolling        (research test)     (LLM + number
                          re-runnable)                         origin)                              check)
                                                       └──────────────> export_artefacts.py ─> artefacts/ ─> app.py
```

- **Ingestion** is safe to re-run: primary keys plus `INSERT OR REPLACE`, with an `ingest_log` of rows received,
  duplicates and rows written per chunk. Each source has its own window (Elexon to yesterday, the weather archive
  six days behind).
- **GitHub Actions** (`.github/workflows/daily.yml`) runs the tests and the pipeline every morning, caches the DuckDB
  file between runs, and commits refreshed artefacts; Streamlit Cloud redeploys from them.
- **The app** reads only `artefacts/` (about 200 KB of Parquet and JSON), so it starts instantly without the database.
- **Tests:** 26 pytest tests cover clock-change days, completeness, drift, ingest windows, the number checker and the
  backtest statistics.

## Data and licences

- **Elexon Insights Solution (BMRS) API:** wind generation by type, demand outturn, market index price.
  Use of the Insights APIs is governed by Elexon's BMRS Data Licence Terms and API Terms of Use.
- **Open-Meteo:** ERA5-based historical weather and the Previous Runs API for day-ahead forecasts, used under the
  free non-commercial API terms. Weather data by Open-Meteo.com, licensed under CC BY 4.0.

No raw data, database files or API keys are committed; everything is fetched by code.

## Limitations

- The day-ahead weather input is Open-Meteo's run from the previous day, which approximates but does not exactly match
  the information available at a real gate closure.
- Six weather points are a coarse proxy for GB-wide wind and temperature.
- Demand is transmission-level outturn, which falls when embedded solar generates; solar radiation and a
  Christmas-period flag are not yet features, and demand MAPE (6.1%) is well short of operator forecasts.
- One year of out-of-sample testing.

## How to run

```bash
conda create -n energy python=3.11 -y
conda activate energy
pip install -r requirements.txt

python src/ingest.py --start 2024-11-18   # first run: two-year backfill (a few minutes)
python run_pipeline.py                    # daily: ingest, quality, forecast, research, briefing, export
pytest -q
streamlit run app.py
```

For the LLM briefing, copy `.env.example` to `.env` and add a free Groq API key; without one, the template briefing
is used. In GitHub, store the key as the secret `LLM_API_KEY` and the model as the variable `LLM_MODEL`.

## Project structure

```
app.py                     Streamlit app (reads artefacts/ only)
run_pipeline.py            runs every step in order
sql/schema.sql             DuckDB tables and keys
sql/model_data.sql         one row per half-hour: targets, price, forecast and actual weather
src/ingest.py              Elexon and Open-Meteo ingestion
src/quality.py             quality checks and scorecard
src/forecast.py            day-ahead models and rolling-origin backtest
src/price_backtest.py      wind signal vs price direction (research)
src/briefing.py            LLM briefing with number check
src/export_artefacts.py    small Parquet/JSON files for the app
tests/                     pytest tests
.github/workflows/         daily GitHub Actions run
```
