from fastapi import APIRouter, HTTPException

from ..services import explorer_service
from ..services.explorer_service import ExplorerFilters

router = APIRouter(prefix="/api/explorer", tags=["explorer"])


@router.get("/filters")
def filters():
    try:
        return explorer_service.get_filter_domains()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Historical dataset not found: {exc}")


@router.post("/query")
def query(filters: ExplorerFilters):
    try:
        return explorer_service.query(filters)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Historical dataset not found: {exc}")
