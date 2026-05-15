# US Track Scraping Execution Plan

This document is an implementation-first plan for scraping US racing tracks across Thoroughbred, Quarter Horse, and Harness.

Scope goals:
- Build reliable daily entries/results ingestion for as many tracks as possible.
- Prioritize easiest/highest-yield sources first.
- Use JSON endpoints where available before HTML scraping.
- Keep architecture modular so each track can be added with low risk.

---

## 1) Execution Strategy (How We Will Build)

Source priority order:
1. Official/licensed API if available (least brittle)
2. Hidden/public JSON endpoints
3. Static HTML tables
4. Dynamic JS pages (React hydration)

Implementation order:
1. Tier 1 static HTML tracks (fastest wins)
2. Tier 2 light parsing tracks
3. Hidden JSON endpoint tracks (high quality / high leverage)
4. Tier 3 annoying dynamic tracks (NYRA/CDI/Keeneland hardening)

Data model targets per race:
- track_code, track_name, race_date, race_number
- race_time, race_name, race_class, surface, distance, purse
- runner_name, program_number, jockey, trainer, ml_odds
- scratches (if available)
- source_name, source_url, fetched_at, parser_version

---

## 2) Architecture Plan

### 2.1 Scraper layers
- `scripts/us/sources/`:
  - one module per source family (`equibase_client.py`, `usta_client.py`, `nyra_client.py`, `cdi_client.py`, `tracksite_static_client.py`)
- `scripts/us/adapters/`:
  - per-track adapters mapping raw source payloads to canonical schema
- `scripts/us/pipeline/`:
  - orchestration for daily pull, retries, dedupe, validation

### 2.2 Canonical outputs
- `data/raw/us/<source>/<track>/<YYYY-MM-DD>.json`
- `data/processed/us_entries_<YYYY-MM-DD>.parquet`
- `data/processed/us_results_<YYYY-MM-DD>.parquet`
- `data/processed/us_scratches_<YYYY-MM-DD>.parquet` (optional)

### 2.3 Reliability requirements
- Per-source retry with exponential backoff
- Request pacing (2-5s, jitter)
- Idempotent writes (same day rerun safe)
- Parser version tagging for backward compatibility
- Drift checks (schema + row-level sanity thresholds)

---

## 3) Tiered Build Plan

## Phase A — Tier 1 (Extremely Easy: static HTML)

### Thoroughbred (Tier 1)
- Tampa Bay Downs
- Charles Town
- Mountaineer
- Canterbury Park
- Presque Isle Downs
- Parx
- Penn National
- Ellis Park
- Lone Star Park
- Sam Houston

### Quarter Horse (Tier 1)
- Los Alamitos
- Ruidoso Downs
- Delta Downs
- Evangeline Downs
- Zia Park
- Sunland Park

### Harness (Tier 1)
- Northfield Park
- Scioto Downs
- Rosecroft
- Running Aces
- Plainridge Park
- Cal Expo

Deliverables:
- Track adapters for all Tier 1 tracks.
- Daily pull job for entries + basic results where available.
- Coverage report: race count, runner count, parse success rate.

Acceptance criteria:
- 95%+ successful parses across a 7-day rolling window.
- Missing-field rate under 10% for core fields.

---

## Phase B — Tier 2 (Easy: light parsing)

### Thoroughbred (Tier 2)
- Gulfstream Park
- Santa Anita
- Del Mar
- Oaklawn
- Fair Grounds
- Monmouth Park

### Harness (Tier 2)
- Meadowlands
- Yonkers
- Hoosier Park

Deliverables:
- Light parser normalization for inconsistent labels/tables.
- Enhanced odds and purse extraction.

Acceptance criteria:
- Same as Phase A + stable extraction for race_time/surface/distance.

---

## Phase C — Tier 3 (Scrapeable but Annoying)

### Thoroughbred (Tier 3)
- NYRA (Aqueduct, Belmont, Saratoga)
- Churchill Downs
- Keeneland

Approach:
- Prefer exposed JSON/React hydration data over rendered DOM scraping.
- Keep Playwright fallback for UI-only pages.
- Add out-of-meet and 404 guards to avoid bad cards.

Deliverables:
- Robust adapters with anti-drift tests.
- Endpoint discovery docs for each track family.

Acceptance criteria:
- 90%+ successful parses on active meet days.
- Automatic stale data detection and warnings in UI.

---

## 4) Hidden JSON Endpoints Plan

## Phase D — JSON-first harvesting (high leverage)

### Harness (best JSON availability)
- Meadowlands
- Yonkers
- Hoosier Park
- Scioto Downs
- Northfield Park
- Pocono Downs
- Harrah’s Philadelphia
- Rosecroft
- Running Aces
- Tioga Downs
- Vernon Downs
- Plainridge Park

Expected fields:
- entries
- results
- scratches
- driver changes
- purse info
- post positions

### Thoroughbred JSON sources
- NYRA (React API endpoints)
- Churchill Downs Inc. (shared CDI API)
- FanDuel Racing (TVG internal JSON)
- TwinSpires (internal JSON)

### Quarter Horse JSON sources
- Los Alamitos
- Ruidoso Downs
- Remington QH

Deliverables:
- Endpoint registry (`docs/US_JSON_ENDPOINT_REGISTRY.md`) with:
  - URL pattern
  - method/auth
  - response schema
  - headers needed
  - known rate limits

Acceptance criteria:
- JSON endpoint used as primary source where stable.
- HTML parsing retained only as fallback.

---

## 5) Best Source Mapping (Execution Reference)

### Thoroughbred
- Churchill Downs -> Equibase + CDI JSON (fast updates)
- Keeneland -> Equibase (clean tables)
- Gulfstream -> HRN + TVG (clean HTML)
- Santa Anita -> Equibase (stable)
- Del Mar -> Equibase (stable)
- Oaklawn -> HRN (clean)
- Fair Grounds -> CDI JSON (very clean)
- Tampa Bay -> Track site (easy HTML)
- Parx -> Track site (clean tables)
- Penn National -> Track site (clean tables)
- Charles Town -> Track site (simple HTML)
- Mountaineer -> Track site (simple HTML)

### Quarter Horse
- Los Alamitos -> Track JSON (best QH data)
- Ruidoso -> Track JSON (clean)
- Remington QH -> Track JSON (clean)
- Delta Downs -> Equibase (reliable)
- Evangeline -> Equibase (reliable)
- Zia Park -> Track site (clean)
- Sunland -> Track site (clean)

### Harness
- Meadowlands -> USTA JSON (best harness data)
- Yonkers -> USTA JSON (clean)
- Hoosier -> USTA JSON (clean)
- Northfield -> USTA JSON (clean)
- Scioto -> USTA JSON (clean)
- Pocono -> USTA JSON (clean)
- Rosecroft -> USTA JSON (clean)
- Running Aces -> USTA JSON (clean)
- Plainridge -> USTA JSON (clean)
- Cal Expo -> Track site (clean)

---

## 6) Implementation Backlog (Practical Sequence)

Sprint 1 (3-5 days):
- Canonical schema + storage writer + validation layer
- Tier 1 Thoroughbred static adapters (first 5 tracks)
- Daily job + summary metrics

Sprint 2 (3-5 days):
- Remaining Tier 1 Thoroughbred + Quarter Horse
- Basic harness Tier 1 adapters
- Alerting for parser failures

Sprint 3 (4-7 days):
- Tier 2 tracks
- Race/result reconciliation logic
- Odds/purse enrichers

Sprint 4 (5-8 days):
- JSON endpoint registry and JSON-first clients
- Harness JSON wave (USTA-heavy tracks)

Sprint 5 (5-8 days):
- Tier 3 hard tracks (NYRA/CDI/Keeneland)
- Playwright fallbacks + anti-drift test fixtures

---

## 7) QA and Monitoring Plan

Automated checks per run:
- Non-empty output for active tracks
- Race count sanity vs prior day (delta threshold)
- Runner count sanity per race
- Mandatory field completeness checks

Parser drift detection:
- Snapshot fixtures in `tests/fixtures/us_sources/`
- Contract tests per adapter
- CI gate on parser changes

Operational dashboards (simple CSV/JSON summary okay initially):
- parse_success_rate by source
- tracks_active_today
- fields_missing_rate
- stale_source_age_minutes

---

## 8) Risk Register

Main risks:
- Site markup changes
- Dynamic pages with anti-bot behavior
- Hidden endpoint auth/header changes
- Legal/ToS constraints for specific sources

Mitigations:
- JSON-first where possible
- Fallback source chain per track
- Low request rate + retries + caching
- Source-level kill-switch and skip lists

---

## 9) Immediate Next Actions (Recommended)

1. Implement `tracksite_static_client.py` and onboard Tier 1 Thoroughbred first.
2. Build USTA JSON client and onboard top 5 harness tracks.
3. Create endpoint registry doc and capture NYRA/CDI/TVG/TwinSpires findings.
4. Add nightly run + morning validation report to avoid silent data gaps.
