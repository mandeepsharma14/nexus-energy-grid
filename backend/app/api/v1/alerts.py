"""NexusGrid — Alerts API. © 2026 Mandeep Sharma. All rights reserved."""
from fastapi import APIRouter, Query
import pandas as pd, os
from app.core.config import settings

router = APIRouter()

@router.get("/", summary="Get active alerts")
async def get_alerts(org_id: str = Query("arcelor_steel"), severity: str = Query("all"), status: str = Query("open")):
    path = os.path.join(settings.DATA_DIR, "alerts.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "org_id" in df.columns: df = df[df["org_id"]==org_id]
        if severity != "all" and "severity" in df.columns: df = df[df["severity"]==severity]
        if status != "all" and "status" in df.columns: df = df[df["status"]==status]
        return {"count": len(df), "alerts": df.head(50).to_dict(orient="records")}
    return {"count": 0, "alerts": []}

@router.patch("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    return {"alert_id": alert_id, "status": "acknowledged", "message": "Alert acknowledged"}
