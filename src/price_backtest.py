from math import comb
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="The inplace keyword")
DB_PATH = Path("data/energy.duckdb")
OUT_PATH = Path("price_backtest_output.txt")
MIN_PERIODS = 44
COSTS = [0.0, 1.0, 2.0]

DAILY_SQL = """
WITH fc AS (
    SELECT CAST(settlement_date AS DATE) AS day,
           AVG(pred) FILTER (WHERE model = 'gbm_forecast_weather') AS wind_fc,
           AVG(pred) FILTER (WHERE model = 'baseline') AS wind_last_known,
           COUNT(*) FILTER (WHERE model = 'gbm_forecast_weather') AS n_fc
    FROM backtest_predictions
    WHERE target = 'wind'
    GROUP BY 1
),
act AS (
    SELECT settlement_date AS day, AVG(wind_mw) AS wind_act, COUNT(wind_mw) AS n_wind,
           AVG(price) AS price, COUNT(price) AS n_price
    FROM model_data
    GROUP BY 1
)
SELECT a.day, f.wind_fc, f.wind_last_known, f.n_fc, a.wind_act, a.n_wind, a.price, a.n_price,
       LAG(a.price) OVER (ORDER BY a.day) AS price_prev
FROM act a
LEFT JOIN fc f USING (day)
ORDER BY a.day
"""


def binom_p_two_sided(k, n, p=0.5):
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(q for q in probs if q <= probs[k] * (1 + 1e-9)))


def positions(signal):
    return -np.sign(signal)


def score(name, pos, dp, signal=None):
    traded = pos != 0
    hits = int((np.sign(dp[traded]) == pos[traded]).sum())
    n = int(traded.sum())
    row = {"strategy": name, "days": n, "hit_rate_pct": 100 * hits / n, "p_value_vs_50pct": binom_p_two_sided(hits, n),
           "corr_signal_vs_price_change": np.corrcoef(signal, dp)[0, 1] if signal is not None else np.nan}
    for c in COSTS:
        row[f"pnl_per_day_cost_{c:g}"] = float(np.mean(pos * dp - c * traded))
    return row


def build_daily(con):
    df = con.execute(DAILY_SQL).df()
    df["price_change"] = df["price"] - df["price_prev"]
    df["signal_forecast"] = df["wind_fc"] - df["wind_last_known"]
    df["signal_oracle"] = df["wind_act"] - df["wind_last_known"]
    usable = (df["n_fc"] >= MIN_PERIODS) & (df["n_price"] >= MIN_PERIODS) & df["price_prev"].notna()
    df["oracle_usable"] = usable & (df["n_wind"] >= MIN_PERIODS)
    return df[usable].reset_index(drop=True)


def main():
    con = duckdb.connect(str(DB_PATH))
    df = build_daily(con)
    dp = df["price_change"].to_numpy()
    orc = df[df["oracle_usable"]]

    results = pd.DataFrame([
        score("always predict price up", np.ones(len(df)), dp),
        score("wind forecast signal", positions(df["signal_forecast"].to_numpy()), dp, df["signal_forecast"].to_numpy()),
        score("oracle: actual wind change", positions(orc["signal_oracle"].to_numpy()), orc["price_change"].to_numpy(),
              orc["signal_oracle"].to_numpy()),
    ])
    df["strength_tercile"] = pd.qcut(df["signal_forecast"].abs(), 3, labels=["weak", "medium", "strong"])
    df["hit"] = np.sign(dp) == positions(df["signal_forecast"].to_numpy())
    by_strength = df.groupby("strength_tercile", observed=True).agg(
        days=("hit", "size"), hit_rate_pct=("hit", lambda h: 100 * h.mean()),
        mean_abs_signal_mw=("signal_forecast", lambda s: s.abs().mean())).reset_index()

    df["pnl_forecast_cost_1"] = positions(df["signal_forecast"].to_numpy()) * dp - 1.0
    con.execute("CREATE OR REPLACE TABLE price_backtest AS SELECT * EXCLUDE (strength_tercile) FROM df")
    con.execute("CREATE OR REPLACE TABLE price_backtest_results AS SELECT * FROM results")
    con.close()

    lines = [
        "RESEARCH TEST, NOT A TRADING STRATEGY",
        f"days: {df['day'].min()} -> {df['day'].max()}, {len(df)} usable days "
        f"(>= {MIN_PERIODS} periods of forecast and price); oracle uses {len(orc)} days with complete actual wind",
        "signal: forecast daily wind for tomorrow minus the last value known at 09:30 today; "
        "more wind expected -> predict the daily average APX price falls",
        "P&L: +/-1 MWh on the change in daily average price, in GBP per day, after a round-trip cost in GBP/MWh",
        "",
        "=== direction tests ===",
        results.round(4).to_string(index=False),
        "",
        "=== wind forecast signal: hit rate by signal strength (terciles of |signal|) ===",
        by_strength.round(1).to_string(index=False),
        "",
        f"share of days the price rose: {100 * (dp > 0).mean():.1f}%",
        "interpretation: the signal explains day-on-day price moves, but it is NOT evidence of a trading edge.",
        "the day-ahead auction for tomorrow clears around 11:00 today and already prices the same public wind",
        "forecasts, so yesterday's average price is not a price anyone could trade at. A real test would compare",
        "the signal with the day-ahead auction price, which is not in this dataset.",
        "other caveats: the APX index is a within-day index; daily averages hide intraday shape; one year of data;",
        "no slippage, liquidity or position limits; costs and P&L are illustrative only",
    ]
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH}; tables price_backtest and price_backtest_results written")


if __name__ == "__main__":
    main()
