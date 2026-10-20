-- AirGuard ingestion schema.
-- One Postgres database holds both the migrated historical CSVs and everything
-- GitHub Actions ingests going forward. Tables store data in roughly its source
-- shape -- export_for_modeling.py reshapes it into the CSVs build_features.py expects.

-- CPCB station pollutant readings (data.gov.in "Real time Air Quality Index"
-- resource). Long format: one row per station, pollutant and observation time.
CREATE TABLE IF NOT EXISTS cpcb_hourly (
    station       text        NOT NULL,
    city          text        NOT NULL DEFAULT 'Delhi',
    pollutant     text        NOT NULL,      -- e.g. PM2.5, PM10, NO2, NH3, SO2, CO, OZONE
    unit          text,
    value_avg     double precision,
    value_min     double precision,
    value_max     double precision,
    observed_at   timestamptz NOT NULL,
    source        text        NOT NULL DEFAULT 'data.gov.in',
    inserted_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (station, pollutant, observed_at, source)
);
CREATE INDEX IF NOT EXISTS ix_cpcb_hourly_time ON cpcb_hourly (observed_at);

-- Open-Meteo surface weather, both recently-observed hours (is_forecast = false,
-- from the API's past_days window) and forecast hours (is_forecast = true).
CREATE TABLE IF NOT EXISTS weather_surface_hourly (
    valid_time            timestamptz NOT NULL,
    is_forecast           boolean     NOT NULL,
    run_time              timestamptz NOT NULL,   -- when this value was fetched / forecast issued
    temperature_2m        double precision,
    relative_humidity_2m  double precision,
    surface_pressure_hpa  double precision,
    wind_speed_10m        double precision,
    wind_direction_10m    double precision,
    boundary_layer_height double precision,
    shortwave_radiation   double precision,
    precipitation         double precision,
    source                text        NOT NULL DEFAULT 'open-meteo',
    inserted_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (valid_time, source)
);
CREATE INDEX IF NOT EXISTS ix_weather_surface_time ON weather_surface_hourly (valid_time);

-- Open-Meteo pressure-level weather, same recent/forecast split.
CREATE TABLE IF NOT EXISTS weather_pressure_hourly (
    valid_time         timestamptz NOT NULL,
    is_forecast        boolean     NOT NULL,
    run_time           timestamptz NOT NULL,
    pressure_hpa       integer     NOT NULL,
    temperature        double precision,
    wind_speed         double precision,
    wind_direction     double precision,
    relative_humidity  double precision,
    geopotential_height double precision,
    -- specific_humidity (q) and vertical_velocity (w): NOT available from Open-Meteo's
    -- forecast or historical weather APIs at any pressure level. These two columns are
    -- only ever populated by ingestion/cds_backfill.py against real ERA5 pressure-level
    -- data; every Open-Meteo-sourced row leaves them NULL, same as before this column
    -- existed (export_for_modeling.py used to hardcode NaN here -- now it reads real
    -- values where a CDS backfill has landed them, and NaN everywhere else, unchanged).
    specific_humidity  double precision,
    vertical_velocity  double precision,
    source             text        NOT NULL DEFAULT 'open-meteo',
    inserted_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (valid_time, pressure_hpa, source)
);
CREATE INDEX IF NOT EXISTS ix_weather_pressure_time ON weather_pressure_hourly (valid_time);

-- The 12h, 4-grid-point box around Delhi (28.50/28.75 x 77.00/77.25, 700/800/900 hPa --
-- the same points era5_pressure_levels_12h_ist.csv uses). t/u/v/r only, no q or w here,
-- so (unlike weather_pressure_hourly) this is fully Open-Meteo-fetchable via multi-location
-- queries -- see ingestion/backfill_openmeteo_12h_grid.py.
CREATE TABLE IF NOT EXISTS weather_pressure12h_hourly (
    valid_time         timestamptz NOT NULL,
    is_forecast        boolean     NOT NULL,
    run_time           timestamptz NOT NULL,
    latitude           double precision NOT NULL,
    longitude          double precision NOT NULL,
    pressure_hpa       integer     NOT NULL,
    temperature        double precision,
    wind_speed         double precision,
    wind_direction     double precision,
    relative_humidity  double precision,
    source             text        NOT NULL DEFAULT 'open-meteo',
    inserted_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (valid_time, latitude, longitude, pressure_hpa, source)
);
CREATE INDEX IF NOT EXISTS ix_weather_pressure12h_time ON weather_pressure12h_hourly (valid_time);

-- CAMS global atmospheric composition, ADS forecast product (nearest grid point
-- to Delhi), same recent/forecast split by run_time vs valid_time.
CREATE TABLE IF NOT EXISTS cams_hourly (
    valid_time    timestamptz NOT NULL,
    -- CAMS (cams-global-atmospheric-composition-forecasts) is a forecast product line
    -- for every row this ingestion writes -- is_forecast is always true here and must
    -- NOT be derived from "valid_time has now passed", because a forecast's valid time
    -- being in the past does not make it an actual measurement. is_best_estimate marks
    -- the short-lead rows (lead_time_hours small, valid_time around "now" at fetch time)
    -- that export_for_modeling.py treats as the best available near-term estimate --
    -- still a model forecast, just the freshest one, not an observation.
    is_forecast   boolean     NOT NULL DEFAULT true,
    is_best_estimate boolean  NOT NULL DEFAULT false,
    lead_time_hours integer,              -- hours between this run's forecast issuance and valid_time
    run_time      timestamptz NOT NULL,   -- when this value was fetched (not the same as the forecast's own init time)
    cams_aod550    double precision,
    cams_pm25      double precision,
    cams_pm10      double precision,
    cams_no2_col   double precision,
    cams_o3_col_du double precision,
    source        text        NOT NULL DEFAULT 'cams-ads',
    inserted_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (valid_time, source)
);
-- Idempotent migration for databases created before is_best_estimate/lead_time_hours
-- existed (CREATE TABLE IF NOT EXISTS above is a no-op on an already-existing table).
ALTER TABLE cams_hourly ADD COLUMN IF NOT EXISTS is_best_estimate boolean NOT NULL DEFAULT false;
ALTER TABLE cams_hourly ADD COLUMN IF NOT EXISTS lead_time_hours integer;
CREATE INDEX IF NOT EXISTS ix_cams_hourly_time ON cams_hourly (valid_time);

-- Raw FIRMS fire detections (MODIS and VIIRS, NRT), Punjab-Haryana-Delhi NCR box.
-- Kept raw and un-aggregated: fire_features.py (already in your repo) turns the
-- combined historical archive + this table into fire_daily_features.csv, so its
-- transport-weighting logic does not need to be reimplemented here.
CREATE TABLE IF NOT EXISTS fire_detections_raw (
    id           bigserial PRIMARY KEY,
    satellite    text,
    instrument   text,
    acq_date     date        NOT NULL,
    acq_time     text        NOT NULL,   -- HHMM as given by FIRMS
    latitude     double precision NOT NULL,
    longitude    double precision NOT NULL,
    frp          double precision,
    confidence   text,
    daynight     text,
    -- matches the columns in your historical fire_archive_M-C61_817146.csv /
    -- fire_nrt_J2V-C2_817148.csv (both use brightness + bright_t31, MODIS adds "type").
    brightness   double precision,
    bright_t31   double precision,
    scan         double precision,
    track        double precision,
    version      text,
    source       text        NOT NULL DEFAULT 'firms-nrt',
    inserted_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (satellite, instrument, acq_date, acq_time, latitude, longitude)
);
CREATE INDEX IF NOT EXISTS ix_fire_detections_date ON fire_detections_raw (acq_date);

-- ---------------------------------------------------------------------------
-- Historical mirrors: one table per original CSV, same columns, loaded once by
-- migrate_historical.py. Kept separate from the live tables above because the
-- live sources (Open-Meteo, CAMS ADS forecast) do not have an identical shape
-- to the original ERA5/CAMS reanalysis CSVs (see export_for_modeling.py for
-- how the two are reconciled into one feature timeline).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS station_hourly_hist (
    station text NOT NULL, "timestamp" timestamptz NOT NULL,
    "PM25" double precision, "PM10" double precision, "NO" double precision, "NO2" double precision,
    "NOx" double precision, "NH3" double precision, "SO2" double precision, "CO" double precision,
    "O3" double precision, "Benzene" double precision, "Toluene" double precision,
    temperature double precision, humidity double precision, wind_speed double precision,
    wind_direction double precision, pressure double precision, solar_radiation double precision,
    total_rainfall double precision,
    PRIMARY KEY (station, "timestamp")
);

CREATE TABLE IF NOT EXISTS era5_surface_hist (
    "timestamp" timestamptz PRIMARY KEY,
    era5_pblh double precision, era5_inv_surface double precision, era5_inv_deep double precision,
    era5_lapse_925_700 double precision, era5_ventilation double precision, era5_t2m double precision,
    era5_rh double precision, era5_pressure double precision, era5_solar double precision,
    era5_rain double precision, era5_wind10 double precision,
    u10 double precision, v10 double precision, u925 double precision, v925 double precision
);

CREATE TABLE IF NOT EXISTS era5_pressure6h_hist (
    "timestamp" timestamptz NOT NULL, pressure_hpa integer NOT NULL,
    t double precision, u double precision, v double precision,
    r double precision, q double precision, w double precision,
    PRIMARY KEY ("timestamp", pressure_hpa)
);

CREATE TABLE IF NOT EXISTS era5_pressure12h_hist (
    "timestamp" timestamptz NOT NULL, latitude double precision NOT NULL, longitude double precision NOT NULL,
    pressure_hpa integer NOT NULL, t double precision, u double precision, v double precision, r double precision,
    PRIMARY KEY ("timestamp", latitude, longitude, pressure_hpa)
);

CREATE TABLE IF NOT EXISTS cams_3h_hist (
    "timestamp" timestamptz PRIMARY KEY,
    cams_aod550 double precision, cams_pm25 double precision, cams_pm10 double precision,
    cams_no2_col double precision, cams_o3_col_du double precision
);

-- One row per ingestion job run, for monitoring the GitHub Actions schedule.
CREATE TABLE IF NOT EXISTS ingestion_log (
    id           bigserial PRIMARY KEY,
    job          text        NOT NULL,
    started_at   timestamptz NOT NULL,
    finished_at  timestamptz,
    rows_written integer,
    status       text        NOT NULL,   -- 'ok' or 'error'
    message      text
);
