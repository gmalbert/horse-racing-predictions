# React + FastAPI Migration

The app now uses a **React frontend** backed by
**precomputed static JSON** plus a small **FastAPI service** for the two things
that cannot be precomputed: interactive Data Explorer filtering and Betting Manager persistence.

## View locally with Docker Compose

With Docker installed inside WSL, run this one command in PowerShell from the repository root:

```powershell
.\start.ps1
```

The launcher uses the default WSL distribution, builds and starts both services, waits
for health checks, and opens the browser. WSL starts automatically if it is stopped;
the Docker daemon must be configured to run in that distribution. No Docker Desktop is
required. Use `.\start.ps1 -Distribution Ubuntu -Port 8081` to select a distribution or port,
or add `-NoBrowser` to skip opening a browser.

From inside WSL, or any terminal with a working Docker CLI and Compose, use:

```powershell
docker compose up --build -d --wait
```

Open **http://localhost:8080**. API documentation is at **http://localhost:8080/api/docs**.
No API keys, local Python installation, or Node installation are required to view the
included data. The first build downloads the Node, Python, and nginx images and dependencies.

```powershell
docker compose logs -f          # view logs
docker compose down             # stop; preserves betting records
docker compose up --build -d --wait  # rebuild after code or JSON exports change
```

If port 8080 is occupied, use `.\start.ps1 -Port 8081`, then open http://localhost:8081.
For direct Compose commands, set `HRP_WEB_PORT` in the shell that runs Docker. The API is internal to Compose; nginx proxies `/api/`
through the same origin as the React app. Browser refreshes on nested routes work.

The repository's `data/` folder is mounted read-only. New betting records start with a
$1,000 USD bankroll and persist separately in the `betting-state` named volume. This avoids
modifying existing pipeline bankroll files. `docker compose down -v` **deletes** that volume;
use ordinary `down` to preserve it. To use existing bankroll files, set `HRP_STATE_DIR` to
a writable folder containing `bankroll_state.json` and `betting_history.csv`, or replace
the state volume with a bind mount. Any external automatic settlement process must share
that same writable state directory. Manual settlement works through Betting Manager.

The file-based state service runs one API worker and serializes write operations. Keep
one API replica; use a transactional database before scaling the writable service.

## Data availability

The refreshed export includes British/Irish predictions for **2026-10-21 and
2026-10-22**, US predictions for **2026-10-21** (NYRA coverage only), and statistics
through **2026-04-11**. US historical results now extend through October 20, with an
older gap still requiring backfill. The British results API rejected the configured
account because that endpoint requires a Standard plan. See
[the repair report](DATA_PIPELINE_REPAIRS_2026-10-21.md) for source and model limitations.
Exporting JSON does not fetch races or train a model. Dates remain visible in the
selectors; the historical-data messages have been removed from the layout.

To refresh after the existing prediction pipeline has produced new files:

```powershell
venv\Scripts\python.exe scripts\export_web_data.py
venv\Scripts\python.exe scripts\validate_web_data.py
docker compose up --build -d --wait
```

Validation checks strict JSON, required artifacts, prediction row/race totals, stats windows,
and whether exported dates match the latest upstream files. Optionally pass
`--max-age-days 2` to make stale content an error. It currently fails that optional freshness
threshold because the upstream inputs are historical.

## Converted workflows and verification

| Route | Available workflows |
|---|---|
| `/` | Five tabs; schedule, prediction dates and combined current cards, race filters, best/value bets, top 25, runner detail, exacta/trifecta, value calculator, fixture tiers, model insights |
| `/us-racing` | Five tabs; US cards, track/type/class/going/surface/distance filters, stakes calendar, graded race ranking, model provenance and accuracy |
| `/data-explorer` | Horse/course/jockey summaries, literal name search, all filter controls, raw result pagination; precomputed fallback when the dataset/API is unavailable |
| `/trainer-stats` | Five time windows; recent form, course records, partnerships, going patterns |
| `/jockey-stats` | Five time windows; season form, course records, partnerships, rolling form trends |
| `/betting-manager` | Kelly recommendations, bankroll editing, bet logging, settlement, history, P/L and ROI charts |
| `/backtest` | Walk-forward results, segment accuracy, segment ROI, calibration |

Model training, scraping, calibration, and prediction generation remain in the existing
offline scripts and workflows. The React app reads their exported outputs. The legacy
Streamlit source remains as a reference and is not needed by either web container.

```powershell
npm ci --prefix web/frontend
npm run build --prefix web/frontend
venv\Scripts\python.exe -m pip install -r web/backend/requirements-dev.txt
venv\Scripts\python.exe -m pytest web/tests/test_api.py -q
venv\Scripts\python.exe -m playwright install chromium
venv\Scripts\python.exe web/tests/smoke_web.py
# Or read-only browser checks against the running Compose app:
venv\Scripts\python.exe web/tests/smoke_web.py --url http://localhost:8080
```

The default smoke runner starts temporary API/production-preview servers with isolated
betting state, exercises every route and tab, logs and settles a test bet, checks reload
persistence, mobile navigation, and offline fallbacks, and records screenshots, errors,
and page-ready timings in `web/test-results/`. These are local React measurements, not a
controlled speed comparison against Streamlit. CI repeats the build, API tests, data
validation, browser tests, and a real Compose build/start/read-only smoke test.

Local verification on 2026-10-21: production build and 18 API/export tests passed; all seven
routes/30 tabs passed Chromium with zero console errors or failed requests. The WSL
Docker build and `start.ps1 -NoBrowser` launcher also passed: both containers are healthy,
and all seven routes/30 tabs passed the read-only browser smoke test through nginx at
http://localhost:8080 from Windows. Docker is enabled to start automatically in Ubuntu WSL.

The legacy pipeline suite passed 190 tests with two existing failures in unchanged
code/assets: `feature_columns.txt` contains 35 more features than the saved model, and
American odds conversion truncates a 2/3 favorite to -199 instead of -200. These are
upstream model/odds issues, separate from the web/API regression checks.

```
┌──────────────────────────────────────────────────────────────────────┐
│ PRECOMPUTE (existing pipeline + scripts/export_web_data.py)          │
│   predictions CSVs / parquets / model files                          │
│     └─> web/frontend/public/data/*.json  (committed to the repo)     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ git push
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ REACT FRONTEND - Cloudflare Pages (web/frontend)                     │
│   Serves ONLY precomputed values from /data/*.json                   │
│   7 pages: International · US Racing · Data Explorer · Trainer Stats │
│            Jockey Stats · Betting Manager · Backtest Results         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ /api/* (only when needed)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FASTAPI - Oracle Cloud Always-Free VM (web/backend)                  │
│   /api/explorer/*    server-side filtering of race_scores.parquet    │
│   /api/bankroll|bets|kelly|performance   Betting Manager persistence │
└──────────────────────────────────────────────────────────────────────┘
```

## What is precomputed (the "precalculate values" requirement)

`scripts/export_web_data.py` computes **everything the Streamlit app used to compute on the fly**
and writes it to `web/frontend/public/data/`:

| Artifact | Precomputed contents |
|---|---|
| `manifest.json` | available prediction dates, stats windows, generation time |
| `predictions/{date}.json` | per-horse ranks, Top-2/Top-3 %, exacta/trifecta probabilities, fair trifecta odds, favorites, best bet, value bets (edge-sorted), top 25, filter domains |
| `us_predictions/{date}.json` | same for US + model provenance |
| `schedule_upcoming.json` | BHA 2026 fixture list (upcoming only) |
| `fixtures_scored.json` | tier counts, score histogram bins, top courses, full fixture table, next 50 Tier-1 races |
| `us_stakes.json` / `us_schedule.json` | graded stakes calendar + US entries summary |
| `us_top_races.json` | predictability score per race (confidence ×40 + margin ×30 + field ×10 + grade/tier/purse bonuses − harness penalty), confidence labels |
| `model_insights.json` / `us_model_insights.json` | metadata, feature importance, calibration metrics, diagnostics, US accuracy rollups |
| `trainer_stats_{14,30,90,180,365}.json` | leaderboard, 14-day form, course specialists, partnerships, going patterns per window |
| `jockey_stats_{...}.json` | same + precomputed 30-ride rolling win-rate series |
| `backtest.json` | walk-forward folds + summary, segment accuracy/ROI tables, calibration bins |
| `explorer_overview.json` | landing aggregates + filter domains (used when the API is offline) |
| `betting_watchlist.json` | Tier 1 / Tier 2 watchlist tables |

If the daily pipeline is stale, the exporter automatically falls back to the **latest available**
prediction dates and the UI shows a staleness banner.

## Local development

```powershell
# 1. Export the precomputed data (requires the project venv)
venv\Scripts\python.exe scripts\export_web_data.py

# 2. Frontend (http://localhost:5173), from repository root
npm ci --prefix web/frontend
npm run dev --prefix web/frontend

# 3. Separate terminal, from repository root (http://localhost:8000)
venv\Scripts\python.exe -m uvicorn app.main:app --app-dir web/backend --reload --port 8000
```

The Vite dev and preview servers proxy `/api/*` to `localhost:8000`. If that port is
occupied, start FastAPI on another port and set `VITE_API_PROXY_TARGET` before starting
Vite. `VITE_API_BASE_URL` is an optional build-time override for a separately hosted API;
leave it unset for Compose. `HRP_DATA_DIR`, `HRP_RAW_DIR`, `HRP_STATE_DIR`, and
`HRP_CORS_ORIGINS` configure the backend. With no state override, native development uses
the pipeline's existing files under `data/processed`; set `HRP_STATE_DIR` for isolation.

## Cloudflare Pages deployment (frontend)

**Build settings** (Pages → Create project → connect repo):

| Setting | Value |
|---|---|
| Root directory | `web/frontend` |
| Build command | `npm run build` |
| Build output | `dist` |
| Environment variable | `VITE_API_BASE_URL` = `https://<your-oracle-host>` (or domain) |

SPA routing and caching are handled by `public/_redirects` and `public/_headers`.

Because the exported JSON is committed to the repo (see
`.github/workflows/export_web_data.yml`), every data refresh automatically triggers a Pages
rebuild when using git integration. Files under `/data/*` are cached at the edge for 5 minutes
(`must-revalidate`), so cached responses revalidate after a successful new deployment. A new Pages build/deployment is still required to publish changed JSON.

## Oracle Cloud deployment (FastAPI)

Run the existing Python container on a VM with sufficient RAM for pandas and the result dataset. Verify the host's current availability and pricing before provisioning. Before public use, add authentication or protect the application and API with Cloudflare Access; the current API exposes shared bankroll state without user authentication.

```bash
# 1. Sync the repo (or at least data/processed + web/backend) to the VM
git clone <repo> /srv/horse-racing && cd /srv/horse-racing

# 2. Build and run
cd web/backend
docker build -t equine-edge-api .
docker run -d --name equine-edge-api --restart unless-stopped \
  -p 8000:8000 \
  -v /srv/horse-racing/data:/data:ro \
  -v equine-edge-state:/state \
  -e HRP_CORS_ORIGINS="https://<your-pages-domain>.pages.dev" \
  equine-edge-api
```

Notes:
- `HRP_DATA_DIR=/data/processed` and `HRP_RAW_DIR=/data/raw` are set inside the image; the
  volume mount above is all that is required.
- Betting Manager writes `bankroll_state.json` / `betting_history.csv` to `/state` in the
  image. The named volume above must remain writable; race data can stay read-only.
- Put Caddy or nginx in front for TLS (`api.yourdomain.com`), then set
  `VITE_API_BASE_URL=https://api.yourdomain.com` in Cloudflare Pages and redeploy.
- The Data Explorer loads `race_scores.parquet` (~51 MB) into memory on first request —
  allow memory headroom for pandas and concurrent requests.
- If you later skip Oracle entirely, the site still works: the Data Explorer falls back to the
  precomputed overview and only the Betting Manager is unavailable.

## Keeping data fresh (pipeline integration)

The new workflow `.github/workflows/export_web_data.yml` runs daily and after the prediction
jobs, then commits the refreshed JSON. The existing workflows are untouched; if you want the
export to run inside them instead, add:

```yaml
      - name: Export web data
        run: python scripts/export_web_data.py --history 14
```

before their commit step, and include `web/frontend/public/data/**` in the committed paths.

## Project layout

```
web/
├── frontend/                 # React 19 + Vite 7 + TypeScript + Recharts
│   ├── public/
│   │   ├── data/             # ← precomputed JSON (generated, committed)
│   │   ├── _redirects        # SPA fallback for Cloudflare Pages
│   │   └── _headers          # edge cache rules
│   └── src/
│       ├── components/       # ui, predictions (race detail/value bets), stats
│       ├── lib/              # dataClient (static JSON), apiClient (FastAPI), types, format
│       └── pages/            # the 7 pages (one file per page)
└── backend/                  # FastAPI (Oracle Cloud)
    ├── app/
    │   ├── routers/          # explorer.py, bankroll.py
    │   ├── services/         # parquet filtering; bankroll/kelly/performance
    │   └── config.py         # env-driven paths + CORS
    └── Dockerfile
```
