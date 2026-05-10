"""
fetch_betfair_us_odds.py
Fetches live US horse racing odds from the Betfair Exchange API.

Betfair's Exchange is a legitimate, free-to-use API (requires an account and
app key). This script:
  1. Authenticates with Betfair using username/password/app-key (no SSL certs
     required when using the non-interactive login endpoint).
  2. Lists all US horse-racing WIN markets available today.
  3. Fetches best back/lay prices for each runner.
  4. Saves the result to data/raw/betfair_us_odds_YYYY-MM-DD.json.
  5. Optionally overlays the fetched odds onto an existing
     data/processed/us_predictions_YYYY-MM-DD.csv to add
     betfair_back_odds / betfair_value_edge columns.

Setup:
    pip install betfairlightweight
    # Add to .env:
    #   BETFAIR_USERNAME=your@email.com
    #   BETFAIR_PASSWORD=your_password
    #   BETFAIR_APP_KEY=your_app_key

Usage:
    python scripts/fetch_betfair_us_odds.py [--date YYYY-MM-DD] [--overlay]
"""

import json
import os
import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# Betfair event type ID for Horse Racing
HORSE_RACING_TYPE_ID = "7"
US_COUNTRY = "US"


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

def _create_client():
    """
    Build an authenticated betfairlightweight APIClient.

    Uses environment variables:
      BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY

    Raises RuntimeError if credentials are missing.
    """
    import betfairlightweight

    username = os.getenv("BETFAIR_USERNAME", "").strip()
    password = os.getenv("BETFAIR_PASSWORD", "").strip()
    app_key  = os.getenv("BETFAIR_APP_KEY",  "").strip()

    if not all([username, password, app_key]):
        raise RuntimeError(
            "Betfair credentials not found in .env. "
            "Set BETFAIR_USERNAME, BETFAIR_PASSWORD, and BETFAIR_APP_KEY."
        )

    # betfairlightweight can authenticate without SSL certs when using the
    # non-interactive (username/password) endpoint on the exchange.
    client = betfairlightweight.APIClient(
        username=username,
        password=password,
        app_key=app_key,
    )
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Market fetching
# ─────────────────────────────────────────────────────────────────────────────

def get_us_win_markets(client, race_date: str | None = None) -> list:
    """
    List all US horse-racing WIN markets for a given date.

    Args:
        client:    authenticated betfairlightweight APIClient
        race_date: "YYYY-MM-DD" window to search; defaults to today

    Returns list of MarketCatalogue objects.
    """
    from betfairlightweight import filters

    if race_date is None:
        race_date = date.today().isoformat()

    # Build a time window covering the entire day in UTC
    day_start = datetime.fromisoformat(race_date).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )
    day_end = day_start + timedelta(days=1)

    market_filter = filters.market_filter(
        event_type_ids=[HORSE_RACING_TYPE_ID],
        market_countries=[US_COUNTRY],
        market_type_codes=["WIN"],
        market_start_time=filters.time_range(
            from_=day_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            to=day_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )

    # market_projection is passed as a plain list — not via filters module
    market_projection = [
        "COMPETITION",
        "EVENT",
        "EVENT_TYPE",
        "RUNNER_DESCRIPTION",
        "MARKET_START_TIME",
        "MARKET_DESCRIPTION",
    ]

    return client.betting.list_market_catalogue(
        filter=market_filter,
        market_projection=market_projection,
        max_results=200,
    )


def get_market_book(client, market_ids: list[str]) -> list:
    """
    Fetch best back/lay prices for a batch of markets.

    Args:
        client:     authenticated client
        market_ids: list of Betfair market IDs (max 40 per call recommended)

    Returns list of MarketBook objects.
    """
    from betfairlightweight import filters

    price_filter = filters.price_projection(
        price_data=["EX_BEST_OFFERS"],
    )
    # Betfair recommends ≤40 market IDs per call
    books = []
    for i in range(0, len(market_ids), 40):
        chunk = market_ids[i : i + 40]
        books.extend(
            client.betting.list_market_book(
                market_ids=chunk,
                price_projection=price_filter,
            )
        )
    return books


# ─────────────────────────────────────────────────────────────────────────────
# Data assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_us_card(client, race_date: str | None = None) -> list[dict]:
    """
    Fetch full US horse-racing card with live Betfair odds.

    Returns list of market dicts:
      {
        "market_id", "market_name", "event", "competition",
        "start_time", "status", "total_matched",
        "runners": [
          {
            "name", "selection_id", "best_back", "best_lay",
            "last_traded", "status", "implied_prob"
          }, ...
        ]
      }
    """
    markets = get_us_win_markets(client, race_date)
    print(f"  Found {len(markets)} US WIN markets")

    if not markets:
        return []

    # Build selection_id → name map for all markets
    id_to_name: dict[str, dict[int, str]] = {}
    for m in markets:
        id_to_name[m.market_id] = {
            r.selection_id: r.runner_name for r in (m.runners or [])
        }

    # Fetch live books in batches
    market_ids = [m.market_id for m in markets]
    books = get_market_book(client, market_ids)
    book_map = {b.market_id: b for b in books}

    # Build a catalogue lookup
    cat_map = {m.market_id: m for m in markets}

    results: list[dict] = []
    for mid, cat in cat_map.items():
        book = book_map.get(mid)

        runners_out: list[dict] = []
        if book:
            for runner in book.runners:
                best_back = None
                best_lay  = None
                if runner.ex:
                    backs = runner.ex.available_to_back
                    lays  = runner.ex.available_to_lay
                    best_back = backs[0].price if backs else None
                    best_lay  = lays[0].price  if lays  else None

                implied_prob = round(1 / best_back, 4) if best_back else None

                runners_out.append(
                    {
                        "selection_id":   runner.selection_id,
                        "name":           id_to_name.get(mid, {}).get(
                                              runner.selection_id, "Unknown"
                                          ),
                        "status":         str(runner.status),
                        "best_back":      best_back,
                        "best_lay":       best_lay,
                        "last_traded":    runner.last_price_traded,
                        "implied_prob":   implied_prob,
                    }
                )
        else:
            # No book data — include runners from catalogue without odds
            for sel_id, name in id_to_name.get(mid, {}).items():
                runners_out.append(
                    {
                        "selection_id": sel_id,
                        "name":         name,
                        "status":       "UNKNOWN",
                        "best_back":    None,
                        "best_lay":     None,
                        "last_traded":  None,
                        "implied_prob": None,
                    }
                )

        results.append(
            {
                "market_id":    mid,
                "market_name":  cat.market_name,
                "event":        cat.event.name if cat.event else "",
                "competition":  cat.competition.name if cat.competition else "",
                "start_time":   str(cat.market_start_time),
                "status":       str(book.status) if book else "UNKNOWN",
                "total_matched": float(book.total_matched or 0) if book else 0.0,
                "runners":      sorted(
                    runners_out,
                    key=lambda r: r["best_back"] or 9999,
                ),
            }
        )

    return sorted(results, key=lambda r: r["start_time"])


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_betfair_odds(data: list[dict], date_str: str) -> Path:
    """
    Persist Betfair odds to data/raw/betfair_us_odds_YYYY-MM-DD.json.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / f"betfair_us_odds_{date_str}.json"
    payload = {
        "date":       date_str,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source":     "betfair_exchange",
        "markets":    data,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved -> {out_path}")
    return out_path


def load_betfair_odds(date_str: str) -> dict | None:
    """
    Load previously saved Betfair odds.  Returns None if the file doesn't exist.
    """
    p = DATA_RAW / f"betfair_us_odds_{date_str}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Overlay onto US predictions CSV
# ─────────────────────────────────────────────────────────────────────────────

def overlay_betfair_odds(date_str: str) -> bool:
    """
    Join Betfair best-back odds onto the US predictions CSV.

    Adds columns:
      betfair_back_odds   — best available back price on Betfair
      betfair_implied_prob — 1 / betfair_back_odds
      betfair_value_edge  — model win_probability - betfair_implied_prob

    Matching is done by fuzzy horse name (lowercased, stripped).

    Returns True if the overlay was applied, False otherwise.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  [overlay] pandas not available — skipping overlay")
        return False

    preds_path = DATA_PROCESSED / f"us_predictions_{date_str}.csv"
    odds_path  = DATA_RAW / f"betfair_us_odds_{date_str}.json"

    if not preds_path.exists():
        print(f"  [overlay] No predictions file: {preds_path}")
        return False
    if not odds_path.exists():
        print(f"  [overlay] No Betfair odds file: {odds_path}")
        return False

    preds_df = pd.read_csv(preds_path)
    odds_data = json.loads(odds_path.read_text(encoding="utf-8"))

    # Build a flat name → best_back lookup
    odds_lookup: dict[str, float] = {}
    for market in odds_data.get("markets", []):
        for runner in market.get("runners", []):
            name = runner.get("name", "")
            back = runner.get("best_back")
            if name and back:
                key = name.strip().lower()
                # Keep shortest (most competitive) back price if duplicated
                if key not in odds_lookup or back < odds_lookup[key]:
                    odds_lookup[key] = back

    def _match_odds(horse_name: str) -> float | None:
        key = str(horse_name).strip().lower()
        return odds_lookup.get(key)

    preds_df["betfair_back_odds"]    = preds_df["horse"].apply(_match_odds)
    preds_df["betfair_implied_prob"] = preds_df["betfair_back_odds"].apply(
        lambda x: round(1 / x, 4) if pd.notna(x) and x > 0 else None
    )
    preds_df["betfair_value_edge"]   = (
        preds_df["win_probability"] - preds_df["betfair_implied_prob"]
    ).round(4)

    preds_df.to_csv(preds_path, index=False)

    matched = preds_df["betfair_back_odds"].notna().sum()
    print(
        f"  [overlay] Matched {matched}/{len(preds_df)} horses "
        f"with Betfair odds -> {preds_path}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Display helper
# ─────────────────────────────────────────────────────────────────────────────

def print_us_card(data: list[dict]) -> None:
    """Print a formatted US racing card from assembled market data."""
    print(f"\n{'='*70}")
    print(f"  BETFAIR EXCHANGE — US HORSE RACING ({len(data)} markets)")
    print(f"{'='*70}")

    for market in data:
        print(f"\n{market['event']:<35} {market['market_name']}")
        print(f"  Start : {market['start_time']}")
        print(f"  Status: {market['status']}   "
              f"Matched: £{market['total_matched']:,.0f}")
        print(f"  {'Runner':<32} {'Back':>7} {'Lay':>7} {'ImpProb':>8}")
        print(f"  {'-'*56}")
        for r in market["runners"]:
            back = f"{r['best_back']:.2f}" if r.get("best_back") else "  N/A"
            lay  = f"{r['best_lay']:.2f}"  if r.get("best_lay")  else "  N/A"
            prob = f"{r['implied_prob']*100:.1f}%" if r.get("implied_prob") else "  N/A"
            print(f"  {r['name']:<32} {back:>7} {lay:>7} {prob:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch live US horse racing odds from Betfair Exchange"
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date to fetch markets for (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help=(
            "After fetching odds, overlay them onto the "
            "us_predictions_YYYY-MM-DD.csv file."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the fetched odds card to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print(f"Betfair US Odds — {args.date}")
    print("=" * 60)

    try:
        client = _create_client()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print("  Logging in to Betfair...")
    try:
        client.login()
    except Exception as exc:
        print(f"  Login failed: {exc}")
        raise SystemExit(1) from exc

    try:
        data = assemble_us_card(client, args.date)
        save_betfair_odds(data, args.date)

        total_runners = sum(len(m["runners"]) for m in data)
        print(
            f"\nDone — {len(data)} markets, {total_runners} runners fetched"
        )

        if args.show:
            print_us_card(data)

        if args.overlay:
            overlay_betfair_odds(args.date)

    finally:
        try:
            client.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
