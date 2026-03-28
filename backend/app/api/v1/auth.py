"""NexusGrid — Auth API (Demo Mode). © 2026 Mandeep Sharma. All rights reserved."""
from fastapi import APIRouter
from pydantic import BaseModel
import time

router = APIRouter()

class DemoLoginRequest(BaseModel):
    org_id: str = "arcelor_steel"

@router.post("/demo-login", summary="Demo mode login (no password required)")
async def demo_login(req: DemoLoginRequest):
    return {"access_token": f"demo_token_{req.org_id}_{int(time.time())}",
            "token_type": "bearer", "org_id": req.org_id,
            "user": {"name": "Demo User", "role": "analyst", "email": "demo@nexusgrid.ai"},
            "expires_in": 86400, "mode": "demo"}

@router.get("/me")
async def get_me():
    return {"name": "Mandeep Sharma", "role": "admin", "org_id": "arcelor_steel", "email": "demo@nexusgrid.ai"}
