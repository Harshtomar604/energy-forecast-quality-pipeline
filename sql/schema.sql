CREATE TABLE IF NOT EXISTS wind (
    settlement_date   DATE,
    settlement_period INTEGER,
    psr_type          VARCHAR,
    start_time        TIMESTAMPTZ,
    publish_time      TIMESTAMPTZ,
    mw                DOUBLE,
    PRIMARY KEY (settlement_date, settlement_period, psr_type)
);

CREATE TABLE IF NOT EXISTS demand (
    settlement_date   DATE,
    settlement_period INTEGER,
    start_time        TIMESTAMPTZ,
    publish_time      TIMESTAMPTZ,
    demand_mw         DOUBLE,
    tsd_mw            DOUBLE,
    PRIMARY KEY (settlement_date, settlement_period)
);

CREATE TABLE IF NOT EXISTS price (
    settlement_date   DATE,
    settlement_period INTEGER,
    provider          VARCHAR,
    start_time        TIMESTAMPTZ,
    price             DOUBLE,
    volume            DOUBLE,
    PRIMARY KEY (settlement_date, settlement_period, provider)
);

CREATE TABLE IF NOT EXISTS weather (
    point             VARCHAR,
    time              TIMESTAMPTZ,
    wind_speed_100m   DOUBLE,
    temperature_2m    DOUBLE,
    PRIMARY KEY (point, time)
);

CREATE TABLE IF NOT EXISTS weather_forecast (
    point             VARCHAR,
    time              TIMESTAMPTZ,
    wind_speed_100m   DOUBLE,
    temperature_2m    DOUBLE,
    PRIMARY KEY (point, time)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    run_at            TIMESTAMPTZ,
    source            VARCHAR,
    chunk_start       DATE,
    chunk_end         DATE,
    rows_received     INTEGER,
    duplicate_keys    INTEGER,
    rows_written      INTEGER
);
