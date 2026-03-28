"""
NexusGrid — FastAPI Backend Application
Main entry point and app configuration.

© 2026 Mandeep Sharma. All rights reserved.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.core.config import settings
from app.api.v1 import (
    energy, forecasts, insights, optimize,
    carbon, copilot, twin, alerts, orgs, auth
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nexusgrid")

# ─── Application Factory ──────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="NexusGrid API",
        description="AI-Powered Energy Intelligence & Optimization Platform",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── Middleware ────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time"] = f"{duration:.1f}ms"
        return response

    # Multi-tenant middleware
    @app.middleware("http")
    async def tenant_middleware(request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID", "demo")
        request.state.tenant_id = tenant_id
        return await call_next(request)

    # ── Routers ───────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(auth.router,      prefix=f"{prefix}/auth",      tags=["Auth"])
    app.include_router(orgs.router,      prefix=f"{prefix}/orgs",      tags=["Organizations"])
    app.include_router(energy.router,    prefix=f"{prefix}/energy",    tags=["Energy"])
    app.include_router(forecasts.router, prefix=f"{prefix}/forecasts", tags=["Forecasts"])
    app.include_router(insights.router,  prefix=f"{prefix}/insights",  tags=["Insights"])
    app.include_router(optimize.router,  prefix=f"{prefix}/optimize",  tags=["Optimizer"])
    app.include_router(carbon.router,    prefix=f"{prefix}/carbon",    tags=["Carbon"])
    app.include_router(copilot.router,   prefix=f"{prefix}/copilot",   tags=["Copilot"])
    app.include_router(twin.router,      prefix=f"{prefix}/twin",      tags=["Digital Twin"])
    app.include_router(alerts.router,    prefix=f"{prefix}/alerts",    tags=["Alerts"])

    # ── Health & Root ─────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "NexusGrid API",
            "version": "2.0.0",
            "status": "healthy",
            "copyright": "© 2026 Mandeep Sharma. All rights reserved.",
        }

    @app.get("/health", tags=["System"])
    async def health():
        return {"status": "ok", "timestamp": time.time()}

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        return JSONResponse({"error": "Not found", "path": request.url.path}, status_code=404)

    @app.exception_handler(500)
    async def server_error(request: Request, exc):
        logger.error(f"Server error: {exc}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

    logger.info("NexusGrid API started — © 2026 Mandeep Sharma")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
