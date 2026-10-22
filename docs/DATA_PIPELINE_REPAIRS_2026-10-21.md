# Data pipeline repairs — October 21, 2026

## Local data refresh

- British/Irish predictions: October 21 (359 eligible runners) and October 22
  (433 eligible runners), generated with the tracked JSON model. Scratches are excluded.
- US predictions: October 21, 62 runners across eight NYRA races. This is partial US
  coverage; empty track landing pages are not usable racecards.
- US historical results: 809,420 rows, including 13,544 newly fetched runner records
  for October 14–20. The latest result is October 20. This does **not** fill the gap
  between the previous May 9 cutoff and October 14.
- Trainer/jockey exports now use April 11 results instead of January 31, selecting
  the freshest compatible dataset. April results are still historical.

The static JSON must be exported before rebuilding the frontend container:

```powershell
python -X utf8 scripts/export_web_data.py --history 14
python scripts/validate_web_data.py
.\start.ps1
```

Starting the app does not scrape or train models. The existing GitHub pipelines and
local browser collector remain responsible for producing data.

## Repairs

- British/Irish prediction loads `horse_win_predictor.json` and its own ordered
  75-feature metadata. It no longer requires an ignored pickle or substitutes a
  constant probability after inference errors. Historical rows dated on or after the
  prediction date are excluded. Current field ratings, draw, and weights replace
  historical race context. Other regions are not scored with this British/Irish model.
- Model loading, missing cards, empty output, invalid probabilities, and inference
  errors fail the job. Feature-contract validation reads the same model metadata.
- US ingestion installs Chromium in Actions. Empty cards, cached failed fallbacks,
  and empty prediction output fail. Empty track placeholders cannot satisfy race
  coverage checks. Browser probes receive their own time budget after slower sources.
  Drift uses runner-bearing cards from the requested recent-day window, excluding
  future cards and obsolete snapshots rather than comparing against old placeholders.
- T1 subprocess failures propagate and source reports remain available as artifacts.
  UTF-8 subprocess output is explicit on Windows.
- TVG workflows retain both raw results and the updated US parquet. Backfill locates
  the newest missing historical interval instead of stopping at an unrelated old cache.
- Publication from `run_racing_post_browser.ps1 -Push -TriggerPredictions` uses an
  isolated clone of `-PublishBranch` (default `main`) and pushes only the collected
  racecards. It then dispatches predictions on that branch. Development commits and
  unrelated staged files in the source checkout are not published.
- Changed workflow commit messages include Markdown descriptions, and synchronization
  failures stop publication. Regression checks run in `Data Pipeline Checks`.
- The explorer export uses grouped sums rather than repeatedly calling Python for
  every horse group; its counts include nonfinishers as runs but not as places.

## Unresolved source and model limitations

An actual request for October 20 British results was rejected by The Racing API:
the configured account does not have the Standard plan required for `/results`.
No account subscription or credentials were changed. A supported results feed is
required before full British/Irish history and statistics can become current.

Several T1 track sites return 403 or provide pages without usable runner lists.
Installing Chromium fixes the missing-browser error but does not establish full US
coverage. The usable live output verified here is NYRA; other tracks need further
source/parser work. Guardrails now expose insufficient coverage rather than accepting
an empty collection as successful. Verified no-racing-day exceptions are not automatic.

Current-date predictions do not establish model accuracy or calibration. The available
engineered British/Irish historical feature dataset still ends January 31, and the
existing feature builder includes estimates for missing histories. Historical training,
feature freshness, and calibration need validation against refreshed results. The old
June 28 constant-fallback predictions have not been rewritten as historical model results.

These workflow repairs are local until committed and integrated into the publishing
branch. No remote push or workflow dispatch was performed during the repair.

## Validation

- 18 offline data/prediction regression tests and 18 API tests passed.
- Both new British/Irish CSVs contain varied finite model probabilities.
- Today's US ingestion and odds-coverage validations passed with real runner data.
- Web artifact validation checks exported dates against the newest source CSVs.
- The publication integration test uses local temporary Git repositories and verifies
  that an unrelated development commit cannot reach the publishing branch.
- The rebuilt Docker containers are healthy. Browser checks passed on all seven
  pages and 30 tabs, including mobile navigation and offline fallback handling.
- The served manifest reports British/Irish October 21–22, US October 21, and
  statistics through April 11. The Docker build disables optional npm audit/funding
  requests after an audit request stalled dependency installation.
