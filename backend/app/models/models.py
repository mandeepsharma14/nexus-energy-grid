"""
NexusGrid — SQLAlchemy Database Models
Complete data model for the energy intelligence platform.

© 2026 Mandeep Sharma. All rights reserved.
"""

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text,
    ForeignKey, JSON, Enum, BigInteger, Numeric, Index
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func
import uuid
import enum


class Base(DeclarativeBase):
    pass


def gen_uuid():
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════
class OrgType(enum.Enum):
    industrial   = "industrial"
    retail       = "retail"
    commercial   = "commercial"
    logistics    = "logistics"
    data_center  = "data_center"
    healthcare   = "healthcare"
    education    = "education"

class AlertSeverity(enum.Enum):
    critical = "critical"
    warning  = "warning"
    info     = "info"

class AlertStatus(enum.Enum):
    open         = "open"
    acknowledged = "acknowledged"
    resolved     = "resolved"

class RecommendationStatus(enum.Enum):
    open       = "open"
    approved   = "approved"
    in_progress = "in_progress"
    completed  = "completed"
    dismissed  = "dismissed"

class UserRole(enum.Enum):
    admin    = "admin"
    manager  = "manager"
    analyst  = "analyst"
    viewer   = "viewer"


# ═══════════════════════════════════════════════════════════
# TENANT / ORGANIZATION LAYER
# ═══════════════════════════════════════════════════════════
class Organization(Base):
    __tablename__ = "organizations"

    id          = Column(String(36), primary_key=True, default=gen_uuid)
    name        = Column(String(255), nullable=False)
    type        = Column(Enum(OrgType), nullable=False)
    industry    = Column(String(100))
    tier        = Column(String(20), default="demo")   # demo / starter / pro / enterprise
    settings    = Column(JSON, default={})
    metadata    = Column(JSON, default={})
    is_active   = Column(Boolean, default=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())

    facilities      = relationship("Facility",       back_populates="organization")
    users           = relationship("User",           back_populates="organization")
    recommendations = relationship("Recommendation", back_populates="organization")
    esg_targets     = relationship("ESGTarget",      back_populates="organization")


class User(Base):
    __tablename__ = "users"

    id           = Column(String(36), primary_key=True, default=gen_uuid)
    org_id       = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    email        = Column(String(255), unique=True, nullable=False)
    name         = Column(String(255), nullable=False)
    role         = Column(Enum(UserRole), default=UserRole.viewer)
    hashed_pw    = Column(String(255))
    is_active    = Column(Boolean, default=True)
    last_login   = Column(DateTime(timezone=True))
    preferences  = Column(JSON, default={})
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    organization = relationship("Organization", back_populates="users")

    __table_args__ = (Index("ix_users_org_id", "org_id"),)


# ═══════════════════════════════════════════════════════════
# FACILITY LAYER
# ═══════════════════════════════════════════════════════════
class Facility(Base):
    __tablename__ = "facilities"

    id           = Column(String(36), primary_key=True, default=gen_uuid)
    org_id       = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    name         = Column(String(255), nullable=False)
    address      = Column(String(512))
    city         = Column(String(100))
    state        = Column(String(50))
    zip_code     = Column(String(20))
    country      = Column(String(50), default="US")
    timezone     = Column(String(50), default="America/New_York")
    sqft         = Column(Integer)
    year_built   = Column(Integer)
    facility_type = Column(String(100))
    num_employees = Column(Integer)
    certifications = Column(ARRAY(String))   # LEED, ENERGY STAR, etc.
    ai_score     = Column(Float)             # 0-100 AI efficiency score
    metadata     = Column(JSON, default={})
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    organization  = relationship("Organization", back_populates="facilities")
    meters        = relationship("Meter",        back_populates="facility")
    utility_bills = relationship("UtilityBill",  back_populates="facility")
    alerts        = relationship("Alert",        back_populates="facility")
    emissions     = relationship("Emission",     back_populates="facility")

    __table_args__ = (Index("ix_facilities_org_id", "org_id"),)


class Meter(Base):
    __tablename__ = "meters"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    facility_id     = Column(String(36), ForeignKey("facilities.id"), nullable=False)
    name            = Column(String(255))
    meter_type      = Column(String(50))   # electric / gas / water / steam
    utility_account = Column(String(100))
    tariff_id       = Column(String(36), ForeignKey("tariffs.id"))
    unit            = Column(String(20), default="kWh")
    multiplier      = Column(Float, default=1.0)
    is_submeter     = Column(Boolean, default=False)
    parent_meter_id = Column(String(36), ForeignKey("meters.id"))
    location        = Column(String(255))
    install_date    = Column(DateTime(timezone=True))
    is_active       = Column(Boolean, default=True)
    metadata        = Column(JSON, default={})

    facility = relationship("Facility", back_populates="meters")
    tariff   = relationship("Tariff")
    readings = relationship("EnergyReading", back_populates="meter")


# ═══════════════════════════════════════════════════════════
# TARIFF
# ═══════════════════════════════════════════════════════════
class Tariff(Base):
    __tablename__ = "tariffs"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    utility         = Column(String(255), nullable=False)
    schedule_name   = Column(String(255), nullable=False)
    tariff_type     = Column(String(50))   # flat / tou / real_time / demand
    base_rate       = Column(Numeric(10, 6))  # $/kWh
    demand_charge   = Column(Numeric(10, 4))  # $/kW
    off_peak_rate   = Column(Numeric(10, 6))
    on_peak_rate    = Column(Numeric(10, 6))
    on_peak_hours   = Column(String(255))
    ratchet_pct     = Column(Float)
    tou_periods     = Column(JSON)
    effective_date  = Column(DateTime(timezone=True))
    expiry_date     = Column(DateTime(timezone=True))
    state           = Column(String(2))


# ═══════════════════════════════════════════════════════════
# TIME-SERIES DATA
# ═══════════════════════════════════════════════════════════
class EnergyReading(Base):
    """
    15-minute interval energy readings.
    This is a TimescaleDB hypertable in production.
    Partitioned by time for efficient queries.
    """
    __tablename__ = "energy_readings"

    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    time         = Column(DateTime(timezone=True), nullable=False)  # TimescaleDB time column
    meter_id     = Column(String(36), ForeignKey("meters.id"), nullable=False)
    facility_id  = Column(String(36), nullable=False)   # denormalized for query speed
    org_id       = Column(String(36), nullable=False)   # tenant isolation
    kw           = Column(Numeric(12, 3))
    kwh          = Column(Numeric(12, 4))
    kvar         = Column(Numeric(12, 3))              # reactive power
    power_factor = Column(Numeric(5, 4))
    voltage      = Column(Numeric(8, 2))
    cost_usd     = Column(Numeric(12, 4))
    is_anomaly   = Column(Boolean, default=False)
    anomaly_score = Column(Float)

    meter = relationship("Meter", back_populates="readings")

    __table_args__ = (
        Index("ix_readings_time_facility", "time", "facility_id"),
        Index("ix_readings_meter_time", "meter_id", "time"),
        Index("ix_readings_org_time", "org_id", "time"),
    )


# ═══════════════════════════════════════════════════════════
# UTILITY BILLS
# ═══════════════════════════════════════════════════════════
class UtilityBill(Base):
    __tablename__ = "utility_bills"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    facility_id     = Column(String(36), ForeignKey("facilities.id"), nullable=False)
    org_id          = Column(String(36), nullable=False)
    period_start    = Column(DateTime(timezone=True), nullable=False)
    period_end      = Column(DateTime(timezone=True), nullable=False)
    utility         = Column(String(255))
    total_kwh       = Column(Numeric(14, 2))
    peak_kw         = Column(Numeric(10, 2))
    energy_cost     = Column(Numeric(12, 2))
    demand_charge   = Column(Numeric(12, 2))
    distribution_charge = Column(Numeric(12, 2))
    taxes_fees      = Column(Numeric(12, 2))
    total_bill      = Column(Numeric(14, 2))
    blended_rate    = Column(Numeric(10, 6))
    load_factor     = Column(Float)
    has_error       = Column(Boolean, default=False)
    error_amount    = Column(Numeric(12, 2), default=0)
    raw_pdf_path    = Column(String(512))
    parsed_at       = Column(DateTime(timezone=True))
    charges_detail  = Column(JSON)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    facility = relationship("Facility", back_populates="utility_bills")
    errors   = relationship("BillingError", back_populates="bill")


class BillingError(Base):
    __tablename__ = "billing_errors"

    id           = Column(String(36), primary_key=True, default=gen_uuid)
    bill_id      = Column(String(36), ForeignKey("utility_bills.id"), nullable=False)
    error_type   = Column(String(100))   # wrong_rate / meter_misread / demand_error
    description  = Column(Text)
    expected_amt = Column(Numeric(12, 2))
    actual_amt   = Column(Numeric(12, 2))
    dispute_amt  = Column(Numeric(12, 2))
    status       = Column(String(50), default="open")  # open / disputed / resolved
    resolved_at  = Column(DateTime(timezone=True))
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    bill = relationship("UtilityBill", back_populates="errors")


# ═══════════════════════════════════════════════════════════
# AI / ML OUTPUTS
# ═══════════════════════════════════════════════════════════
class Forecast(Base):
    __tablename__ = "forecasts"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    facility_id     = Column(String(36), nullable=False)
    org_id          = Column(String(36), nullable=False)
    model_name      = Column(String(100))  # LSTM / Prophet / Ensemble
    horizon_days    = Column(Integer)
    forecast_start  = Column(DateTime(timezone=True))
    forecast_end    = Column(DateTime(timezone=True))
    values          = Column(JSON)         # [{ts, kwh, lower, upper}]
    cost_forecast   = Column(JSON)         # [{ts, cost, lower, upper}]
    accuracy_mape   = Column(Float)
    confidence      = Column(Float)
    features_used   = Column(ARRAY(String))
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


class Recommendation(Base):
    __tablename__ = "recommendations"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    org_id          = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    facility_id     = Column(String(36))
    rec_type        = Column(String(100))
    title           = Column(String(512))
    description     = Column(Text)
    annual_savings  = Column(Numeric(14, 2))
    monthly_savings = Column(Numeric(14, 2))
    confidence      = Column(Float)
    effort          = Column(String(20))       # low / medium / high
    payback_months  = Column(Float)
    roi_pct         = Column(Float)
    action_required = Column(Text)
    status          = Column(Enum(RecommendationStatus), default=RecommendationStatus.open)
    approved_by     = Column(String(36))
    completed_at    = Column(DateTime(timezone=True))
    model_version   = Column(String(50))
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    organization = relationship("Organization", back_populates="recommendations")


class Alert(Base):
    __tablename__ = "alerts"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    facility_id     = Column(String(36), ForeignKey("facilities.id"), nullable=False)
    org_id          = Column(String(36), nullable=False)
    severity        = Column(Enum(AlertSeverity), nullable=False)
    alert_type      = Column(String(100))
    title           = Column(String(512))
    message         = Column(Text)
    root_cause      = Column(Text)
    kw_reading      = Column(Float)
    baseline_kw     = Column(Float)
    anomaly_score   = Column(Float)
    cost_impact     = Column(Numeric(12, 2))
    status          = Column(Enum(AlertStatus), default=AlertStatus.open)
    acknowledged_by = Column(String(36))
    resolved_at     = Column(DateTime(timezone=True))
    metadata        = Column(JSON, default={})
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    facility = relationship("Facility", back_populates="alerts")

    __table_args__ = (
        Index("ix_alerts_org_created", "org_id", "created_at"),
        Index("ix_alerts_facility", "facility_id"),
    )


# ═══════════════════════════════════════════════════════════
# SUSTAINABILITY / CARBON
# ═══════════════════════════════════════════════════════════
class Emission(Base):
    __tablename__ = "emissions"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    facility_id     = Column(String(36), ForeignKey("facilities.id"), nullable=False)
    org_id          = Column(String(36), nullable=False)
    period          = Column(String(7))    # "2026-03"
    scope           = Column(Integer)      # 1 / 2 / 3
    tco2            = Column(Numeric(14, 4))
    emission_factor = Column(Numeric(10, 6))
    methodology     = Column(String(100))  # eGRID / GHG Protocol
    source          = Column(String(100))
    kwh_basis       = Column(Numeric(14, 2))
    renewable_pct   = Column(Float)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    facility = relationship("Facility", back_populates="emissions")


class ESGTarget(Base):
    __tablename__ = "esg_targets"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    org_id          = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    framework       = Column(String(100))   # SBTi / GHG Protocol / CDP
    target_name     = Column(String(255))
    target_year     = Column(Integer)
    baseline_year   = Column(Integer)
    baseline_value  = Column(Float)
    current_value   = Column(Float)
    target_value    = Column(Float)
    unit            = Column(String(50))
    progress_pct    = Column(Float)
    on_track        = Column(Boolean)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    organization = relationship("Organization", back_populates="esg_targets")


class CarbonCredit(Base):
    __tablename__ = "carbon_credits"

    id          = Column(String(36), primary_key=True, default=gen_uuid)
    org_id      = Column(String(36), nullable=False)
    credit_type = Column(String(50))    # REC / VER / CCA
    quantity_mt = Column(Float)
    vintage     = Column(Integer)
    registry    = Column(String(100))  # Gold Standard / Verra / APX
    project     = Column(String(255))
    cost_usd    = Column(Numeric(12, 2))
    retired     = Column(Boolean, default=False)
    retired_at  = Column(DateTime(timezone=True))
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
