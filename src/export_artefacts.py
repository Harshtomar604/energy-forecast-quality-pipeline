import json
from datetime import datetime, timezone
from pathlib import Path
import warnings

import duckdb

warnings.filterwarnings("ignore", message="The inplace keyword")
DB_PATH = Path("data/energy.duckdb")
ART_DIR = Path("artefacts")
OUT_PATH = Path("export_output.txt")
RECENT_DAYS = 90

EXPORTS = {
    "quality_scorecard": """
        SELECT * EXCLUDE (run_at) FROM quality_scorecard
        WHERE run_at = (SELECT MAX(run_at) FROM quality_scorecard)""",
    "quality_history": """
        SELECT run_at, status, COUNT(*) AS n_series FROM quality_scorecard
        GROUP BY run_at, status ORDER BY run_at""",
    "quality_daily": "SELECT * FROM quality_daily",
    "forecast_monthly": """
        SELECT * EXCLUDE (run_at) FROM forecast_metrics
        WHERE run_at = (SELECT MAX(run_at) FROM forecast_metrics)""",
    "forecast_summary": """
        SELECT target, model, COUNT(*) AS n,
               AVG(ABS(actual - pred)) AS mae,
               100 * AVG(ABS(actual - pred)) / AVG(actual) AS nmae_pct,
               CASE WHEN target = 'demand' THEN 100 * AVG(ABS(actual - pred) / actual) END AS mape_pct
        FROM backtest_predictions WHERE scored GROUP BY target, model""",
    "price_backtest": """
        SELECT day, signal_forecast, signal_oracle, price, price_prev, price_change, pnl_forecast_cost_1
        FROM price_backtest ORDER BY day""",
    "price_backtest_results": "SELECT * FROM price_backtest_results",
    "backtest_recent": f"""
        SELECT target, model, start_time, settlement_date, settlement_period, actual, pred
        FROM backtest_predictions WHERE pred IS NOT NULL
        AND settlement_date > (SELECT MAX(settlement_date) FROM backtest_predictions) - INTERVAL {RECENT_DAYS} DAY
        ORDER BY target, model, start_time""",
}


def main():
    ART_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    lines = []
    for name, sql in EXPORTS.items():
        path = ART_DIR / f"{name}.parquet"
        con.execute(f"COPY ({sql}) TO '{path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        rows = con.execute(f"SELECT COUNT(*) FROM '{path.as_posix()}'").fetchone()[0]
        lines.append(f"{name:18} rows={rows:>6}  size={path.stat().st_size / 1024:6.1f} KB")

    first, last = con.execute("SELECT MIN(settlement_date), MAX(settlement_date) FROM model_data").fetchone()
    con.close()
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_first": str(first),
        "data_last": str(last),
        "recent_days": RECENT_DAYS,
    }
    (ART_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    total = sum(p.stat().st_size for p in ART_DIR.iterdir()) / 1024
    lines += [f"meta.json written: {meta}", f"total artefacts: {total:.1f} KB"]
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
