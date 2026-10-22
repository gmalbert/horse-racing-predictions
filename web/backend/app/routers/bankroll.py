from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import date as Date
from typing import Literal

from ..services import bankroll_service

router = APIRouter(prefix="/api", tags=["bankroll"])


class BankrollUpdate(BaseModel):
    current_bankroll: float = Field(ge=0, allow_inf_nan=False)


class NewBet(BaseModel):
    bet_date: Date
    horse_name: str = Field(min_length=1, max_length=200, pattern=r"\S")
    race_id: str = ""
    win_probability: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    market_odds: float = Field(gt=1.0, allow_inf_nan=False)
    stake: float = Field(gt=0, allow_inf_nan=False)
    result: Literal["Pending", "Won", "Lost"] = "Pending"


class SettleRequest(BaseModel):
    result: Literal["Won", "Lost"]


@router.get("/bankroll")
def get_bankroll():
    return bankroll_service.load_state()


@router.put("/bankroll")
def update_bankroll(update: BankrollUpdate):
    return bankroll_service.update_bankroll(update.current_bankroll)


@router.get("/bets")
def list_bets():
    return bankroll_service.list_bets()


@router.post("/bets")
def create_bet(bet: NewBet):
    return bankroll_service.log_bet(bet.model_dump(mode="json"))


@router.post("/bets/{index}/settle")
def settle_bet(index: int, req: SettleRequest):
    result = bankroll_service.settle_bet(index, req.result)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@router.get("/kelly")
def kelly(
    date: Date = Query(..., description="YYYY-MM-DD"),
    bankroll: float = Query(1000.0, gt=0, allow_inf_nan=False),
    kelly_fraction: float = Query(0.25, gt=0, le=1.0, allow_inf_nan=False),
    min_edge: float = Query(0.05, ge=0, le=1.0, allow_inf_nan=False),
):
    return bankroll_service.kelly_recommendations(date.isoformat(), bankroll, kelly_fraction, min_edge)


@router.get("/performance")
def performance():
    return bankroll_service.performance()
