"""NexusGrid — Insights/Recommendations API. © 2026 Mandeep Sharma. All rights reserved."""
from fastapi import APIRouter, Query
import pandas as pd, os
from app.core.config import settings

router = APIRouter()

@router.get("/", summary="Get AI recommendations")
async def get_recommendations(org_id: str = Query("arcelor_steel"), status: str = Query("open")):
    path = os.path.join(settings.DATA_DIR, "recommendations.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "org_id" in df.columns: df = df[df["org_id"]==org_id]
        if "status" in df.columns and status != "all": df = df[df["status"]==status]
        return {"org_id": org_id, "count": len(df), "recommendations": df.to_dict(orient="records")}
    return {"org_id": org_id, "count": 0, "recommendations": []}

@router.get("/savings-potential", summary="Total savings opportunity")
async def get_savings_potential(org_id: str = Query("arcelor_steel")):
    path = os.path.join(settings.DATA_DIR, "recommendations.csv")
    total = 0
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "org_id" in df.columns: df = df[df["org_id"]==org_id]
        total = float(df["annual_savings_usd"].sum()) if "annual_savings_usd" in df.columns else 1060000
    return {"org_id": org_id, "annual_savings_potential": total, "monthly": round(total/12,0)}
