"""
NexusGrid — Energy Data API Router
Endpoints for interval data, load profiles, and usage analytics.

© 2026 Mandeep Sharma. All rights reserved.
"""

from fastapi import APIRouter, Query, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

from app.core.config import settings

router = APIRouter()


# ── Response Models ───────────────────────────────────────
class IntervalPoint(BaseModel):
    timestamp: str
    kw: float
    kwh: float
    cost_usd: float
    is_anomaly: bool

class LoadProfilePoint(BaseModel):
    hour: int
    avg_kw: float
    max_kw: float
    min_kw: float

class UsageSummary(BaseModel):
    facility_id: str
    period_start: str
    period_end: str
    total_kwh: float
    total_cost_usd: float
    peak_kw: float
    avg_kw: float
    load_factor: float
    anomaly_count: int


# ── Data Loader (cached in demo mode) ────────────────────
_interval_cache = None

def get_interval_data() -> pd.DataFrame:
    global _interval_cache
    if _interval_cache is None:
        path = os.path.join(settings.DATA_DIR, "interval_data.csv")
        if os.path.exists(path):
            _interval_cache = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            # Fallback: generate synthetic on-the-fly
            _interval_cache = _generate_fallback_data()
    return _interval_cache


def _generate_fallback_data() -> pd.DataFrame:
    hours = pd.date_range(end=datetime.now(), periods=8760, freq="h")
    kw = np.abs(np.random.normal(5000, 800, len(hours)))
    return pd.DataFrame({
        "timestamp": hours,
        "facility_id": "F001",
        "facility_name": "Demo Facility",
        "kw": kw,
        "kwh": kw,
        "cost_usd": kw * 0.0674,
        "is_anomaly": np.random.random(len(hours)) < 0.02,
    })


# ── Endpoints ─────────────────────────────────────────────
@router.get("/usage", summary="Get interval energy readings")
async def get_usage(
    facility_id: Optional[str] = Query(None, description="Filter by facility ID"),
    org_id: Optional[str] = Query(None, description="Filter by org ID"),
    start: Optional[str] = Query(None, description="Start datetime ISO8601"),
    end: Optional[str] = Query(None, description="End datetime ISO8601"),
    resolution: str = Query("15min", description="15min / hourly / daily"),
    limit: int = Query(1000, le=50000),
):
    """
    Retrieve time-series energy readings with optional filtering and resampling.
    Supports 15-minute, hourly, and daily resolution.
    """
    df = get_interval_data()

    # Filter
    if facility_id:
        df = df[df["facility_id"] == facility_id]
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end)]

    # Resample
    if resolution == "hourly":
        df = df.set_index("timestamp").resample("h").agg(
            kw=("kw","mean"), kwh=("kwh","sum"), cost_usd=("cost_usd","sum"),
            is_anomaly=("is_anomaly","any")
        ).reset_index()
    elif resolution == "daily":
        df = df.set_index("timestamp").resample("D").agg(
            kw=("kw","mean"), kwh=("kwh","sum"), cost_usd=("cost_usd","sum"),
            is_anomaly=("is_anomaly","any")
        ).reset_index()

    df = df.tail(limit)
    return {
        "data": df.to_dict(orient="records"),
        "count": len(df),
        "resolution": resolution,
    }


@router.get("/summary", summary="Portfolio-level usage summary")
async def get_summary(
    org_id: Optional[str] = Query("arcelor_steel"),
    period: str = Query("mtd", description="mtd / ytd / last30 / last90"),
):
    """Aggregated usage KPIs for portfolio or single facility."""
    df = get_interval_data()

    now = df["timestamp"].max()
    if period == "mtd":
        start = now.replace(day=1, hour=0, minute=0, second=0)
    elif period == "last30":
        start = now - timedelta(days=30)
    elif period == "last90":
        start = now - timedelta(days=90)
    else:
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0)

    filtered = df[df["timestamp"] >= start]

    summaries = []
    for fid, grp in filtered.groupby("facility_id"):
        avg_kw = grp["kw"].mean()
        peak_kw = grp["kw"].max()
        total_kwh = grp["kwh"].sum()
        total_cost = grp["cost_usd"].sum()
        load_factor = avg_kw / peak_kw if peak_kw > 0 else 0

        summaries.append(UsageSummary(
            facility_id=fid,
            period_start=str(start),
            period_end=str(now),
            total_kwh=round(total_kwh, 2),
            total_cost_usd=round(total_cost, 2),
            peak_kw=round(peak_kw, 2),
            avg_kw=round(avg_kw, 2),
            load_factor=round(load_factor, 4),
            anomaly_count=int(grp["is_anomaly"].sum()),
        ))

    return {"period": period, "facilities": [s.dict() for s in summaries]}


@router.get("/load-profile", summary="24-hour average load profile")
async def get_load_profile(
    facility_id: Optional[str] = Query("F001"),
    segment: str = Query("all", description="all / weekday / weekend"),
):
    """Returns average hourly load profile for visualizing demand patterns."""
    df = get_interval_data()
    if facility_id:
        df = df[df["facility_id"] == facility_id]

    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    if segment == "weekday":
        df = df[~df["is_weekend"]]
    elif segment == "weekend":
        df = df[df["is_weekend"]]

    profile = df.groupby("hour").agg(
        avg_kw=("kw","mean"), max_kw=("kw","max"), min_kw=("kw","min")
    ).reset_index()

    return {
        "facility_id": facility_id,
        "segment": segment,
        "profile": profile.round(2).to_dict(orient="records"),
    }


@router.get("/peak-analysis", summary="Peak demand analysis")
async def get_peak_analysis(
    facility_id: Optional[str] = Query("F001"),
    months: int = Query(12),
):
    """Analyze monthly peak demand patterns and demand charge impact."""
    df = get_interval_data()
    if facility_id:
        df = df[df["facility_id"] == facility_id]

    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    monthly_peaks = df.groupby("month").agg(
        peak_kw=("kw","max"),
        avg_kw=("kw","mean"),
        total_kwh=("kwh","sum"),
        total_cost=("cost_usd","sum"),
    ).tail(months).reset_index()

    monthly_peaks["demand_charge_est"] = monthly_peaks["peak_kw"] * 12.80
    monthly_peaks["demand_pct_of_bill"] = (
        monthly_peaks["demand_charge_est"] / monthly_peaks["total_cost"]
    ).clip(0, 1)

    return {
        "facility_id": facility_id,
        "months": months,
        "monthly_peaks": monthly_peaks.round(2).to_dict(orient="records"),
        "avg_peak_kw": round(monthly_peaks["peak_kw"].mean(), 2),
        "max_peak_kw": round(monthly_peaks["peak_kw"].max(), 2),
    }
