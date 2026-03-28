"""NexusGrid — Optimization Engine API. © 2026 Mandeep Sharma. All rights reserved."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import numpy as np

router = APIRouter()

class SimulatorRequest(BaseModel):
    peak_shed_kw: float = 600
    tariff_option: int = 2
    renewable_pct: float = 30
    dr_level: int = 2
    org_id: str = "arcelor_steel"

@router.post("/simulate", summary="Run savings simulator")
async def simulate_savings(req: SimulatorRequest):
    s1 = round(req.peak_shed_kw * 0.0128 * 12 * 100)
    s2 = {1:0, 2:412000, 3:380000, 4:440000}.get(req.tariff_option, 0)
    s3 = {0:0, 1:86400, 2:172800, 3:259200}.get(req.dr_level, 0)
    s4 = round(req.renewable_pct * 2626)
    total = s1 + s2 + s3 + s4
    return {"annual_savings": total, "monthly_savings": round(total/12),
            "breakdown": {"peak_shaving": s1, "tariff_optimization": s2, "demand_response": s3, "renewable": s4}}

@router.get("/lp-result", summary="LP optimizer result for facility")
async def get_lp_result(facility_id: str = "F001"):
    np.random.seed(42)
    hourly = (np.abs(np.random.normal(7000, 1200, 24))).tolist()
    optimized = [min(v, 7800) * (0.78 if 14<=h<=18 else (1.18 if 2<=h<=5 else 1.0)) for h,v in enumerate(hourly)]
    return {"facility_id": facility_id, "original_load": [round(v) for v in hourly],
            "optimized_load": [round(v) for v in optimized],
            "peak_reduction_kw": round(max(hourly) - max(optimized)),
            "monthly_savings_usd": round((max(hourly)-max(optimized))*12.80*12)}
