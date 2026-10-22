"""Equine Edge API - dynamic endpoints for the React frontend.

Everything that CAN be precomputed is served as static JSON from Cloudflare
Pages. This API only handles what cannot: interactive Data Explorer filtering
over the historical parquet, and Betting Manager persistence.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .routers import bankroll, explorer

app = FastAPI(
    title="Equine Edge API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

app.include_router(explorer.router)
app.include_router(bankroll.router)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "data_dir": str(config.DATA_DIR),
        "explorer_parquet": config.EXPLORER_PARQUET,
    }
