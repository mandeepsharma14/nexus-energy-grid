"""NexusGrid — Organizations API. © 2026 Mandeep Sharma. All rights reserved."""
from fastapi import APIRouter
import json, os
from app.core.config import settings

router = APIRouter()

@router.get("/", summary="List all demo organizations")
async def list_orgs():
    path = os.path.join(settings.DATA_DIR, "organizations.json")
    if os.path.exists(path):
        with open(path) as f: orgs = json.load(f)
        return {"organizations": list(orgs.keys()), "details": orgs}
    return {"organizations": ["arcelor_steel","megamart_retail","primeplex_cre","swiftlogix","vertex_dc"]}
