"""NexusGrid — Digital Twin API. © 2026 Mandeep Sharma. All rights reserved."""
from fastapi import APIRouter, Query
from pydantic import BaseModel
import numpy as np

router = APIRouter()

class TwinScenario(BaseModel):
    facility_id: str = "F001"
    production_change_pct: float = 0
    hvac_setpoint_delta_f: float = 0
    solar_coverage_pct: float = 12

@router.post("/simulate", summary="Run digital twin scenario")
async def simulate_scenario(scenario: TwinScenario):
    cost_delta = round(scenario.production_change_pct * 8400 + scenario.hvac_setpoint_delta_f * -3200 + (scenario.solar_coverage_pct - 12) * -1800)
    carbon_delta = round(scenario.production_change_pct * 180 + scenario.hvac_setpoint_delta_f * -40)
    demand_delta = round(scenario.production_change_pct * 82 + scenario.hvac_setpoint_delta_f * -160)
    return {"facility_id": scenario.facility_id, "monthly_cost_delta": cost_delta, "carbon_delta_tco2": carbon_delta,
            "demand_delta_kw": demand_delta, "payback_months": round(abs(cost_delta)/max(abs(cost_delta)*0.1,1)*12) if cost_delta < 0 else None}

@router.get("/state", summary="Get current virtual facility state")
async def get_facility_state(facility_id: str = Query("F001")):
    np.random.seed(42)
    return {"facility_id": facility_id, "timestamp": "2026-03-28T14:00:00Z",
            "systems": {"hvac_kw": 2840, "hvac_capacity_pct": 68, "production_kw": 4120, "production_capacity_pct": 82,
                        "lighting_kw": 640, "compressed_air_kw": 820, "total_kw": 8420},
            "sync_age_seconds": 14}
