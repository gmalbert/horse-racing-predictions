# US Horse Racing Scrapers

Daily entries pipeline for all US thoroughbred, quarter horse, and harness tracks
covered by DraftKings Racing (NH) / TwinSpires.

Aqueduct (AQU) is **intentionally excluded** — handled separately.

---

## File Layout

```
scrapers/
├── requirements.txt
├── us/
│   ├── run_daily_pipeline.py          ← master orchestrator (start here)
│   ├── utils/
│   │   └── common.py                  ← schema, HTTP client, writers
│   └── sources/
│       ├── equibase_client.py         ← Thoroughbred Tier 1+2+3 (static HTML)
│       ├── usta_client.py             ← Harness (USTA JSON + HTML fallback)
│       ├── cdi_client.py              ← Churchill Downs + Fair Grounds (CDI JSON)
│       ├── nyra_client.py             ← Belmont + Saratoga (NYRA JSON/Playwright)
│       ├── quarter_horse_client.py    ← QH tracks (JSON + HTML + Equibase)
│       ├── tracksite_static_client.py ← Static HTML track sites
│       └── playwright_dynamic_client.py ← JS-heavy sites (Keeneland, SA, Del Mar…)
```

---

## Track Coverage

### Thoroughbred — Equibase (`equibase_client.py`)

| Code | Track              | Tier |
|------|--------------------|------|
| TAM  | Tampa Bay Downs    | T1   |
| CT   | Charles Town       | T1   |
| MNR  | Mountaineer        | T1   |
| CBY  | Canterbury Park    | T1   |
| PID  | Presque Isle Downs | T1   |
| PRX  | Parx Racing        | T1   |
| PEN  | Penn National      | T1   |
| ELP  | Ellis Park         | T1   |
| LS   | Lone Star Park     | T1   |
| HOU  | Sam Houston        | T1   |
| SA   | Santa Anita        | T2   |
| DMR  | Del Mar            | T2   |
| GP   | Gulfstream Park    | T2   |
| OP   | Oaklawn Park       | T2   |
| FG   | Fair Grounds       | T2   |
| MTH  | Monmouth Park      | T2   |
| KEE  | Keeneland          | T3   |
| BEL  | Belmont Park       | T3   |
| SAR  | Saratoga           | T3   |
| CD   | Churchill Downs    | T3   |

### Harness — USTA (`usta_client.py`)

| Code | Track                   |
|------|-------------------------|
| M    | Meadowlands             |
| YO   | Yonkers Raceway         |
| HP   | Hoosier Park            |
| NP   | Northfield Park         |
| SD   | Scioto Downs            |
| RC   | Rosecroft               |
| RUN  | Running Aces            |
| PLN  | Plainridge Park         |
| CAL  | Cal Expo                |
| PCD  | Pocono Downs            |
| PHL  | Harrah's Philadelphia   |
| TGA  | Tioga Downs             |
| VD   | Vernon Downs            |
| FH   | Freehold Raceway        |
| ND   | Northville Downs        |

### CDI Tracks — `cdi_client.py`

| Code | Track            |
|------|------------------|
| CD   | Churchill Downs  |
| FG   | Fair Grounds     |
| TP   | Turfway Park     |

### NYRA Tracks — `nyra_client.py`

| Code | Track          |
|------|----------------|
| BEL  | Belmont Park   |
| SAR  | Saratoga       |

### Quarter Horse — `quarter_horse_client.py`

| Code | Track              | Method   |
|------|--------------------|----------|
| LAD  | Los Alamitos       | JSON     |
| RUI  | Ruidoso Downs      | JSON     |
| REM  | Remington Park     | JSON     |
| ZIA  | Zia Park           | HTML     |
| SUN  | Sunland Park       | HTML     |
| EVD  | Evangeline Downs   | Equibase |
| DEL  | Delta Downs        | Equibase |
| AZD  | Arizona Downs      | HTML     |

### Dynamic Tracks — `playwright_dynamic_client.py`

| Code | Track              |
|------|--------------------|
| KEE  | Keeneland          |
| SA   | Santa Anita        |
| DMR  | Del Mar            |
| LRL  | Laurel Park        |
| PIM  | Pimlico            |
| TAM  | Tampa Bay Downs    |
| CBY  | Canterbury Park    |
| PID  | Presque Isle Downs |

---

## Setup

```bash
pip install -r requirements.txt

# Only needed for Playwright tracks (KEE, SA, DMR, LRL, PIM, etc.)
playwright install chromium
```

---

## Running

```bash
# Run everything for today
python us/run_daily_pipeline.py

# Specific date
python us/run_daily_pipeline.py --date 2026-05-10

# Only specific source modules
python us/run_daily_pipeline.py --sources equibase usta

# Dry run (print plan only)
python us/run_daily_pipeline.py --dry-run

# Individual scrapers
python us/sources/equibase_client.py --date 2026-05-10
python us/sources/equibase_client.py --date 2026-05-10 --tracks TAM CT MNR

python us/sources/usta_client.py --date 2026-05-10 --tracks M YO HP

python us/sources/nyra_client.py --date 2026-05-10

python us/sources/cdi_client.py --date 2026-05-10

python us/sources/quarter_horse_client.py --date 2026-05-10 --tracks LAD RUI

python us/sources/playwright_dynamic_client.py --date 2026-05-10 --tracks KEE SA
```

---

## Outputs

```
output/
├── raw/                           # Raw source data (JSON or HTML) per track/date
│   ├── equibase/tam/2026-05-10.json
│   ├── usta/m/2026-05-10.json
│   └── ...
├── processed/                     # Canonical CSV outputs
│   ├── us_all_entries_2026-05-10.csv    # combined per-day output
│   ├── us_equibase_canonical_all.csv    # rolling equibase master
│   ├── us_harness_canonical_all.csv     # rolling harness master
│   ├── us_qh_canonical_all.csv          # rolling QH master
│   └── us_all_canonical_master.csv      # rolling all-sources master
└── reports/
    └── pipeline_report_2026-05-10.json  # per-run summary
```

### Canonical Schema (per row)

| Field           | Description                        |
|-----------------|------------------------------------|
| track_code      | Exchange/source track code         |
| track_name      | Full track name                    |
| race_date       | YYYY-MM-DD                         |
| race_number     | Integer race number                |
| race_time       | Post time (local)                  |
| race_name       | Race name (if available)           |
| race_class      | Race classification                |
| surface         | Dirt / Turf / Synthetic / Harness  |
| distance        | Distance string                    |
| purse           | Purse amount string                |
| program_number  | Program/post position number       |
| runner_name     | Horse name                         |
| jockey          | Jockey (or driver for harness)     |
| trainer         | Trainer name                       |
| ml_odds         | Morning line odds                  |
| scratched       | Boolean                            |
| breed           | Thoroughbred / Quarter Horse / Harness |
| source_name     | Which scraper produced this row    |
| source_url      | Source URL                         |
| fetched_at      | UTC timestamp                      |
| parser_version  | Version string                     |

---

## Architecture Notes

- **Source priority**: JSON endpoints beat HTML scrapers. Playwright is last resort.
- **Retry + pacing**: All HTTP clients use exponential backoff and 2–5s random delay.
- **Idempotent writes**: Re-running the same date appends to master CSV but
  per-day files are overwritten. Add dedup on `(track_code, race_date, race_number,
  runner_name)` if needed.
- **No data = no error**: Missing race cards (off-season, 404) log INFO, not ERROR.
- **Playwright optional**: All Playwright tracks also have Equibase/HTML fallback
  where possible. Skip playwright source if you don't need those tracks.

---

## Cron Example

```bash
# Run at 7 AM daily, log to file
0 7 * * * cd /path/to/scrapers && python us/run_daily_pipeline.py >> logs/pipeline.log 2>&1
```
