-- NexusGrid Database Schema
-- © 2026 Mandeep Sharma. All rights reserved.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Organizations
CREATE TABLE organizations (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name       VARCHAR(255) NOT NULL,
    type       VARCHAR(50)  NOT NULL,
    tier       VARCHAR(20)  DEFAULT 'demo',
    settings   JSONB        DEFAULT '{}',
    is_active  BOOLEAN      DEFAULT TRUE,
    created_at TIMESTAMPTZ  DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Users
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id      UUID REFERENCES organizations(id),
    email       VARCHAR(255) UNIQUE NOT NULL,
    name        VARCHAR(255) NOT NULL,
    role        VARCHAR(50)  DEFAULT 'viewer',
    hashed_pw   VARCHAR(255),
    is_active   BOOLEAN DEFAULT TRUE,
    last_login  TIMESTAMPTZ,
    preferences JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Facilities
CREATE TABLE facilities (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id        UUID REFERENCES organizations(id),
    name          VARCHAR(255) NOT NULL,
    city          VARCHAR(100),
    state         VARCHAR(50),
    country       VARCHAR(50) DEFAULT 'US',
    timezone      VARCHAR(50) DEFAULT 'America/New_York',
    sqft          INTEGER,
    facility_type VARCHAR(100),
    ai_score      FLOAT,
    is_active     BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ix_facilities_org_id ON facilities(org_id);

-- Tariffs
CREATE TABLE tariffs (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    utility        VARCHAR(255) NOT NULL,
    schedule_name  VARCHAR(255) NOT NULL,
    tariff_type    VARCHAR(50),
    base_rate      NUMERIC(10,6),
    demand_charge  NUMERIC(10,4),
    off_peak_rate  NUMERIC(10,6),
    on_peak_rate   NUMERIC(10,6),
    on_peak_hours  VARCHAR(255),
    state          CHAR(2)
);

-- Meters
CREATE TABLE meters (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id     UUID REFERENCES facilities(id),
    name            VARCHAR(255),
    meter_type      VARCHAR(50) DEFAULT 'electric',
    utility_account VARCHAR(100),
    tariff_id       UUID REFERENCES tariffs(id),
    unit            VARCHAR(20) DEFAULT 'kWh',
    is_active       BOOLEAN DEFAULT TRUE
);

-- Energy Readings (TimescaleDB hypertable)
CREATE TABLE energy_readings (
    id           BIGSERIAL,
    time         TIMESTAMPTZ NOT NULL,
    meter_id     UUID,
    facility_id  UUID,
    org_id       UUID,
    kw           NUMERIC(12,3),
    kwh          NUMERIC(12,4),
    power_factor NUMERIC(5,4),
    cost_usd     NUMERIC(12,4),
    is_anomaly   BOOLEAN DEFAULT FALSE,
    anomaly_score FLOAT
);
SELECT create_hypertable('energy_readings', 'time');
CREATE INDEX ix_readings_time_facility ON energy_readings(time, facility_id);
-- Compression
SELECT add_compression_policy('energy_readings', INTERVAL '7 days');

-- Utility Bills
CREATE TABLE utility_bills (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id         UUID REFERENCES facilities(id),
    org_id              UUID,
    period_start        TIMESTAMPTZ NOT NULL,
    period_end          TIMESTAMPTZ NOT NULL,
    total_kwh           NUMERIC(14,2),
    peak_kw             NUMERIC(10,2),
    energy_cost         NUMERIC(12,2),
    demand_charge       NUMERIC(12,2),
    distribution_charge NUMERIC(12,2),
    taxes_fees          NUMERIC(12,2),
    total_bill          NUMERIC(14,2),
    blended_rate        NUMERIC(10,6),
    load_factor         FLOAT,
    has_error           BOOLEAN DEFAULT FALSE,
    error_amount        NUMERIC(12,2) DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Forecasts
CREATE TABLE forecasts (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id    UUID,
    org_id         UUID,
    model_name     VARCHAR(100),
    horizon_days   INTEGER,
    forecast_start TIMESTAMPTZ,
    values         JSONB,
    accuracy_mape  FLOAT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Recommendations
CREATE TABLE recommendations (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id         UUID REFERENCES organizations(id),
    facility_id    UUID,
    rec_type       VARCHAR(100),
    title          VARCHAR(512),
    description    TEXT,
    annual_savings NUMERIC(14,2),
    confidence     FLOAT,
    effort         VARCHAR(20),
    payback_months FLOAT,
    status         VARCHAR(50) DEFAULT 'open',
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Alerts
CREATE TABLE alerts (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id   UUID REFERENCES facilities(id),
    org_id        UUID,
    severity      VARCHAR(20) NOT NULL,
    alert_type    VARCHAR(100),
    title         VARCHAR(512),
    message       TEXT,
    root_cause    TEXT,
    anomaly_score FLOAT,
    cost_impact   NUMERIC(12,2),
    status        VARCHAR(20) DEFAULT 'open',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ix_alerts_org_created ON alerts(org_id, created_at DESC);

-- Emissions
CREATE TABLE emissions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id     UUID REFERENCES facilities(id),
    org_id          UUID,
    period          VARCHAR(7),
    scope           INTEGER,
    tco2            NUMERIC(14,4),
    emission_factor NUMERIC(10,6),
    renewable_pct   FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ESG Targets
CREATE TABLE esg_targets (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id          UUID REFERENCES organizations(id),
    framework       VARCHAR(100),
    target_name     VARCHAR(255),
    target_year     INTEGER,
    baseline_value  FLOAT,
    current_value   FLOAT,
    target_value    FLOAT,
    unit            VARCHAR(50),
    on_track        BOOLEAN
);
