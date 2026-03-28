"""
NexusGrid — Application Configuration
Pydantic settings model — reads from environment variables.

© 2026 Mandeep Sharma. All rights reserved.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # ── App ─────────────────────────────────────────────
    APP_NAME: str = "NexusGrid"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/nexusgrid"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    DB_POOL_TIMEOUT: int = 30

    # ── Redis ─────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 300  # 5 minutes default

    # ── Auth / Security ───────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-in-production-please"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ── AI / LLM ──────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int = 1500

    # ── CORS ─────────────────────────────────────────────
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://nexusgrid-demo.vercel.app",
        "https://*.vercel.app",
    ]

    # ── Data ─────────────────────────────────────────────
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "../../../ml/data/generated")
    MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "../../../ml/models")

    # ── Pagination ────────────────────────────────────────
    DEFAULT_PAGE_SIZE: int = 100
    MAX_PAGE_SIZE: int = 10000

    # ── Feature Flags ─────────────────────────────────────
    DEMO_MODE: bool = True
    ENABLE_WEBSOCKET: bool = True
    ENABLE_CACHING: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
