-- Simulated operator source database (the PostgreSQL behind the published household datasets).
-- Tables and columns follow ANALYTICAL_DATASETS_MASTER_GUIDE.md section 5 ("Source" / "Source field" columns).
-- Marked ASSUMED where the guide does not describe the source table; the real database may differ there.

CREATE SCHEMA IF NOT EXISTS household_reading;
CREATE SCHEMA IF NOT EXISTS household_billing;
CREATE SCHEMA IF NOT EXISTS household_consumer_new;
CREATE SCHEMA IF NOT EXISTS weather;      -- ASSUMED: weather arrives from outside the billing system (S3 in reality)
CREATE SCHEMA IF NOT EXISTS sim;          -- simulator internals (truth, pending arrivals, clock); never read by the pipeline

-- ---------------------------------------------------------------- lookups
CREATE TABLE IF NOT EXISTS household_reading.reading_source_types (
    id integer PRIMARY KEY, code text, name_uz text, name_ru text, name_crl text, status integer);
CREATE TABLE IF NOT EXISTS household_reading.meter_reading_statuses (
    id integer PRIMARY KEY, code text, name_uz text, name_ru text, name_crl text, status integer);
CREATE TABLE IF NOT EXISTS household_billing.coato_branches (
    code text PRIMARY KEY, parent_code text, name_uz text, level_id integer, location_id integer, status integer);

-- ---------------------------------------------------------------- readings (household_reading.reading_histories)
-- One row per meter reading. The pipeline reads it incrementally by id (ids grow with arrival).
CREATE TABLE IF NOT EXISTS household_reading.reading_histories (
    id                      bigint PRIMARY KEY,
    consumer_id             bigint NOT NULL,
    meter_id                bigint NOT NULL,
    coato_code              text   NOT NULL,
    reading_date            date   NOT NULL,
    reading_a_plus          numeric(18, 3) NOT NULL,   -- cumulative imported active energy register (stored units)
    cycle_a_plus            integer NOT NULL DEFAULT 0,
    reading_source_type_id  integer NOT NULL,          -- 4 = ASKUE (automatic)
    meter_reading_status_id integer NOT NULL,          -- 1 = normal
    created_at              timestamp NOT NULL         -- ASSUMED: arrival time in the source system
);
CREATE INDEX IF NOT EXISTS reading_histories_created ON household_reading.reading_histories (created_at);
CREATE INDEX IF NOT EXISTS reading_histories_meter ON household_reading.reading_histories (meter_id, reading_date);

-- ---------------------------------------------------------------- current customer snapshot (household_consumer_new.consumers)
CREATE TABLE IF NOT EXISTS household_consumer_new.consumers (
    id                        bigint PRIMARY KEY,
    coato_code                text NOT NULL,
    transformer_id            bigint,
    additional_substation_id  bigint,
    tariff_id                 integer,
    meter_id                  bigint,
    accounting_status_id      integer,
    consumer_status_id        integer,                 -- 1 = active, 2 = disconnected
    consumer_status_update_at timestamp,
    housing_type_id           integer,                 -- 1 = house, 2 = apartment
    housing_category_id       integer,
    gaz_connection            boolean,
    central_heating           boolean,
    residents_amount          integer,
    application_date          date,
    contraction_date          date,
    updated_at                timestamp NOT NULL        -- ASSUMED: last change (snapshot is updated in place)
);

-- ---------------------------------------------------------------- monthly customer state (household_billing.consumer_histories)
CREATE TABLE IF NOT EXISTS household_billing.consumer_histories (
    consumer_id                bigint  NOT NULL,
    meter_id                   bigint  NOT NULL,
    period                     date    NOT NULL,       -- first day of the effective month
    coato_code                 text    NOT NULL,
    transformation_coefficient numeric(10, 2) NOT NULL,
    meter_capacity             integer,
    meter_type_id              integer,
    askue_connection           boolean,
    tariff_id                  integer,
    consumer_status_id         integer,
    fider_id                   bigint,                 -- feeder
    disconnection_date         date,
    meter_installation_date    date,
    PRIMARY KEY (consumer_id, period)
);

-- ---------------------------------------------------------------- network (ASSUMED: feeder -> substation reference)
CREATE TABLE IF NOT EXISTS household_billing.network_feeders (
    fider_id bigint PRIMARY KEY, substation_id bigint NOT NULL, coato_code text NOT NULL);

-- ---------------------------------------------------------------- weather (ASSUMED layout)
CREATE TABLE IF NOT EXISTS weather.observed_daily (
    coato_code text NOT NULL, date date NOT NULL, t_mean real, t_min real, t_max real, arrived_at timestamp NOT NULL,
    PRIMARY KEY (coato_code, date));
CREATE TABLE IF NOT EXISTS weather.forecast_daily (         -- runs issued 00 UTC on issue_date, lead 0..15 days
    model text NOT NULL, issue_date date NOT NULL, coato_code text NOT NULL, lead_day integer NOT NULL,
    t_mean real, t_min real, t_max real, arrived_at timestamp NOT NULL,
    PRIMARY KEY (model, issue_date, coato_code, lead_day));

-- ---------------------------------------------------------------- simulator internals
CREATE TABLE IF NOT EXISTS sim.pending_readings (LIKE household_reading.reading_histories INCLUDING DEFAULTS);
ALTER TABLE sim.pending_readings DROP COLUMN IF EXISTS created_at;
ALTER TABLE sim.pending_readings ADD COLUMN IF NOT EXISTS arrival_date date;
ALTER TABLE sim.pending_readings ADD COLUMN IF NOT EXISTS seq bigint;
CREATE INDEX IF NOT EXISTS pending_arrival ON sim.pending_readings (arrival_date, seq);
CREATE TABLE IF NOT EXISTS sim.pending_consumers (LIKE household_consumer_new.consumers);
ALTER TABLE sim.pending_consumers ADD COLUMN IF NOT EXISTS effective_date date;
CREATE TABLE IF NOT EXISTS sim.pending_histories (LIKE household_billing.consumer_histories);
CREATE TABLE IF NOT EXISTS sim.pending_weather_obs (LIKE weather.observed_daily);
CREATE TABLE IF NOT EXISTS sim.pending_weather_fc (LIKE weather.forecast_daily);
CREATE TABLE IF NOT EXISTS sim.truth_node_daily (            -- true daily kWh per node (diagnostics only)
    node text NOT NULL, date date NOT NULL, kwh double precision, households integer, PRIMARY KEY (node, date));
CREATE TABLE IF NOT EXISTS sim.clock (
    id integer PRIMARY KEY DEFAULT 1 CHECK (id = 1), sim_date date NOT NULL, next_reading_id bigint NOT NULL,
    config jsonb, updated_at timestamp NOT NULL DEFAULT now());
