-- One row per settlement period: targets, price and weather aligned to the period's start hour.
-- fc_* = day-ahead forecast weather (what was known the day before); act_* = actual weather (upper bound only).

CREATE OR REPLACE TABLE wind_hh AS
SELECT settlement_date, settlement_period, SUM(mw) AS wind_mw
FROM wind
GROUP BY settlement_date, settlement_period
HAVING COUNT(*) = 2;

CREATE OR REPLACE TABLE fc_wide AS
PIVOT weather_forecast ON point USING first(wind_speed_100m) AS ws, first(temperature_2m) AS temp GROUP BY time;

CREATE OR REPLACE TABLE act_wide AS
PIVOT weather ON point USING first(wind_speed_100m) AS ws, first(temperature_2m) AS temp GROUP BY time;

CREATE OR REPLACE TABLE model_data AS
SELECT
    d.settlement_date,
    d.settlement_period,
    d.start_time,
    d.demand_mw,
    w.wind_mw,
    p.price,
    COLUMNS(f.* EXCLUDE (time)) AS 'fc_\0',
    COLUMNS(a.* EXCLUDE (time)) AS 'act_\0'
FROM demand d
LEFT JOIN wind_hh w USING (settlement_date, settlement_period)
LEFT JOIN price p
    ON p.settlement_date = d.settlement_date
   AND p.settlement_period = d.settlement_period
   AND p.provider = 'APXMIDP'
LEFT JOIN fc_wide f ON f.time = date_trunc('hour', d.start_time)
LEFT JOIN act_wide a ON a.time = date_trunc('hour', d.start_time)
ORDER BY d.start_time;
