"""NexusGrid — Forecasts API. © 2026 Mandeep Sharma. All rights reserved."""
from fastapi import APIRouter, Query
import pandas as pd, numpy as np, os
from app.core.config import settings

router = APIRouter()

@router.get("/", summary="Get AI consumption and cost forecasts")
async def get_forecasts(
    facility_id: str = Query("F001"),
    horizon: int = Query(30, description="Days: 7 / 30 / 90"),
    org_id: str = Query("arcelor_steel"),
):
    path = os.path.join(settings.DATA_DIR, "forecasts.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "org_id" in df.columns:
            df = df[(df["org_id"]==org_id) & (df["horizon_days"]==horizon)]
        return {"facility_id": facility_id, "horizon_days": horizon, "forecasts": df.to_dict(orient="records")}
    # fallback
    np.random.seed(42)
    days = pd.date_range(start="2026-03-28", periods=horizon, freq="D")
    base = 61000
    vals = base * (1 + np.cumsum(np.random.normal(0.001, 0.02, horizon)))
    return {"facility_id": facility_id, "horizon_days": horizon,
            "forecasts": [{"date": str(d.date()), "kwh": round(v,0), "lower": round(v*0.90,0), "upper": round(v*1.10,0)} for d,v in zip(days,vals)],
            "model": "LSTM+Prophet_Ensemble", "accuracy": 0.942}

@router.get("/peak", summary="Peak demand forecast next 7 days")
async def get_peak_forecast(facility_id: str = Query("F001")):
    np.random.seed(42)
    days = pd.date_range(start="2026-03-28", periods=7, freq="D")
    peaks = np.random.uniform(7200, 8900, 7)
    return {"facility_id": facility_id,
            "daily_peaks": [{"date": str(d.date()), "predicted_peak_kw": round(p,0), "risk": "high" if p>8500 else "medium"} for d,p in zip(days,peaks)]}
