from datetime import datetime, timezone
from pathlib import Path
import warnings

import duckdb
import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", message="The inplace keyword")
DB_PATH = Path("data/energy.duckdb")
SQL_PATH = Path("sql/model_data.sql")
OUT_PATH = Path("forecast_output.txt")
SEED = 42
N_FOLDS = 12
PERSISTENCE_SP = 20
WIND_POINTS = ["north_sea", "scotland", "irish_sea", "london", "birmingham", "manchester"]
DEMAND_POINTS = ["london", "birmingham", "manchester"]
CALENDAR = ["settlement_period", "dow", "month", "holiday"]
TARGETS = {
    "wind": {"y": "wind_mw", "baseline": "wind_persistence"},
    "demand": {"y": "demand_mw", "baseline": "demand_lag7"},
}
WEATHER_SOURCES = {"gbm_forecast_weather": "fc", "gbm_actual_weather": "act"}


def feature_cols(target, src):
    if target == "wind":
        return [f"{src}_{p}_ws" for p in WIND_POINTS] + ["settlement_period", "month"]
    return [f"{src}_{p}_temp" for p in DEMAND_POINTS] + [f"{src}_pop_temp"] + CALENDAR + ["demand_lag2", "demand_lag7"]


def lag(df, col, days, period=None):
    src = df[["settlement_date", "settlement_period", col]].copy()
    src["settlement_date"] += pd.Timedelta(days=days)
    keys = ["settlement_date", "settlement_period"]
    if period is not None:
        src = src[src["settlement_period"] == period].drop(columns="settlement_period")
        keys = ["settlement_date"]
    return df[keys].merge(src, on=keys, how="left")[col].to_numpy()


def build_features(df):
    df = df.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    years = range(df["settlement_date"].dt.year.min(), df["settlement_date"].dt.year.max() + 1)
    bank_holidays = pd.to_datetime(list(holidays.UK(subdiv="England", years=years)))
    df["dow"] = df["settlement_date"].dt.day_of_week
    df["month"] = df["settlement_date"].dt.month
    df["holiday"] = df["settlement_date"].isin(bank_holidays).astype(int)
    for src in ("fc", "act"):
        df[f"{src}_pop_temp"] = df[[f"{src}_{p}_temp" for p in DEMAND_POINTS]].mean(axis=1)
    df["wind_persistence"] = lag(df, "wind_mw", 1, period=PERSISTENCE_SP)
    df["demand_lag2"] = lag(df, "demand_mw", 2)
    df["demand_lag7"] = lag(df, "demand_mw", 7)
    return df


def fold_starts(dates, n):
    first = (dates.max() - pd.DateOffset(months=n - 1)).to_period("M").to_timestamp()
    return pd.date_range(first, periods=n, freq="MS")


def metrics(target, y, p):
    err = np.abs(y - p)
    return {
        "n": len(y),
        "mae": err.mean(),
        "nmae_pct": 100 * err.mean() / y.mean(),
        "mape_pct": 100 * np.mean(err / y) if target == "demand" else np.nan,
    }


def backtest(df, target):
    y, base = TARGETS[target]["y"], TARGETS[target]["baseline"]
    preds = []
    for start in fold_starts(df["settlement_date"], N_FOLDS):
        end = start + pd.DateOffset(months=1)
        train = df[(df["settlement_date"] < start) & df[y].notna()]
        fc_weather = [c for c in feature_cols(target, "fc") if c.startswith("fc_")]
        act_weather = [c for c in feature_cols(target, "act") if c.startswith("act_")]
        test = df[(df["settlement_date"] >= start) & (df["settlement_date"] < end) & df[y].notna()
                  & df[base].notna() & df[fc_weather].notna().all(axis=1)]
        has_actual = test[act_weather].notna().all(axis=1).to_numpy()
        fold = test[["settlement_date", "settlement_period", "start_time", y]].rename(columns={y: "actual"})
        fold = fold.assign(target=target, fold=start.strftime("%Y-%m"), scored=has_actual)
        preds.append(fold.assign(model="baseline", pred=test[base].to_numpy()))
        for model, src in WEATHER_SOURCES.items():
            cols = feature_cols(target, src)
            gbm = HistGradientBoostingRegressor(random_state=SEED).fit(train[cols], train[y])
            p = gbm.predict(test[cols])
            if src == "act":
                p = np.where(has_actual, p, np.nan)
            preds.append(fold.assign(model=model, pred=p))
    return pd.concat(preds, ignore_index=True)


def summarise(target, preds):
    preds = preds[preds["scored"]]
    overall = pd.DataFrame([
        {"model": m, **metrics(target, g["actual"].to_numpy(), g["pred"].to_numpy())}
        for m, g in preds.groupby("model", sort=False)
    ])
    base_mae = overall.loc[overall["model"] == "baseline", "mae"].iloc[0]
    overall["skill_vs_baseline_pct"] = 100 * (1 - overall["mae"] / base_mae)
    monthly = (preds.assign(err=(preds["actual"] - preds["pred"]).abs())
               .pivot_table(index="fold", columns="model", values="err", aggfunc="mean"))
    monthly = monthly[list(overall["model"])]
    return overall, monthly


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute(SQL_PATH.read_text())
    df = build_features(con.execute("SELECT * FROM model_data").df())
    run_at = datetime.now(timezone.utc)

    lines = [f"data: {df['settlement_date'].min():%Y-%m-%d} -> {df['settlement_date'].max():%Y-%m-%d}, "
             f"{len(df)} settlement periods",
             f"rolling origin: {N_FOLDS} monthly folds, expanding training window, test month never seen in training",
             "scored rows: half-hours where all three models have a prediction (actual weather lags about 6 days)",
             ""]
    all_preds, all_metrics = [], []
    for target in TARGETS:
        preds = backtest(df, target)
        overall, monthly = summarise(target, preds)
        all_preds.append(preds)
        for fold, g in preds[preds["scored"]].groupby(["fold", "model"]):
            all_metrics.append({"run_at": run_at, "target": target, "fold": fold[0], "model": fold[1],
                                **metrics(target, g["actual"].to_numpy(), g["pred"].to_numpy())})
        unit = "MW"
        lines += [
            f"=== {target}: all test folds pooled (MAE in {unit}) ===",
            overall.round(2).to_string(index=False),
            "",
            f"=== {target}: MAE by test month ({unit}) ===",
            monthly.round(0).to_string(),
            "",
        ]

    predictions = pd.concat(all_preds, ignore_index=True)
    metric_rows = pd.DataFrame(all_metrics)
    con.execute("CREATE OR REPLACE TABLE backtest_predictions AS SELECT * FROM predictions")
    con.execute("CREATE TABLE IF NOT EXISTS forecast_metrics AS SELECT * FROM metric_rows WHERE FALSE")
    con.execute("INSERT INTO forecast_metrics SELECT * FROM metric_rows")
    con.close()

    lines.append("wind MAPE is not reported: actual wind can be near zero, which makes percentage error unstable")
    text = "\n".join(lines)
    OUT_PATH.write_text(text)
    print(text)
    print(f"\nsaved to {OUT_PATH}; tables backtest_predictions and forecast_metrics written")


if __name__ == "__main__":
    main()
