# Racing data freshness audit — 2026-10-21

## Scope and saved work

The React/FastAPI migration was committed locally as `aa3be3c` on
`coding/react-fast-api`, with a detailed Markdown commit body. It has not been pushed.
This subsequent audit reads local files, GitHub Actions logs, and today's archived US
ingestion report. It does not change prediction files, model artifacts, workflows,
scheduled tasks, remote branches, or betting records.

## Summary

| Area | What the app currently receives | Confirmed upstream issue |
|---|---|---|
| International predictions | Latest CSV: 2026-06-28 | Collection, dispatch, model loading, and model feature compatibility are disconnected |
| US predictions | Latest CSV: 2026-06-06 | Missing browser installation produces zero races; empty cards and failed source steps can still report success |
| Trainer/jockey statistics | Results through 2026-01-31 | Exporter prefers a stale engineered dataset ahead of newer base results |
| Historical US results | Local parquet ends 2026-05-09 | Daily fetch successfully updates an ephemeral parquet, but the workflow does not publish that parquet |

## 1. International: current racecards exist, prediction generation is broken

- Local `data/raw/racecards_2026-10-21.json` contains 46 races and 494 runners.
- The same path on GitHub's current default branch contains 70 races and 767 runners.
  These are different snapshots; the current collection date is not the blocker.
- The local browser collector log `logs/racing-post-browser/fetch-20261021-065003.log`
  ends with `fatal: The current branch coding/react-fast-api has no upstream branch.`
- In `run_racing_post_browser.ps1`, a failed `git push` throws before
  `-TriggerPredictions` can dispatch the prediction workflow. Dispatch also uses the
  currently checked-out branch, making this scheduled collector depend on development
  checkout state.
- `precompute_predictions.yml` now has only `workflow_dispatch`; it has no periodic
  schedule or automatic racecard-workflow completion trigger.
- The latest listed prediction run is **2026-08-25**, and is marked successful despite
  printing `Failed to load model: ... models/horse_win_predictor.pkl` for both dates.
- `predict_todays_races.py` loads only that pickle file and returns normally on failure.
  Pickles are ignored by Git, and `git ls-files models` confirms the repository contains
  JSON models, not the required pickle.

Evidence: [latest listed prediction run](https://github.com/gmalbert/horse-racing-predictions/actions/runs/32846035564),
[current prediction workflow](https://github.com/gmalbert/horse-racing-predictions/blob/main/.github/workflows/precompute_predictions.yml),
[current racecards](https://github.com/gmalbert/horse-racing-predictions/blob/main/data/raw/racecards_2026-10-21.json).

### Model integrity must be repaired before publishing new predictions

Read-only inspection found three different feature contracts:

| Artifact | Feature count |
|---|---:|
| Local ignored `horse_win_predictor.pkl` | 47 |
| Tracked `horse_win_predictor.json` | 75 |
| Nonempty entries in `feature_columns.txt` | 82 |

The predictor catches per-horse inference exceptions and substitutes a raw probability
of `0.158`. For the shipped 2026-06-28 CSV, reversing the script's shrinkage formula
produces exactly that fallback for **109 of 109 runners across nine races**. This strongly
suggests fallback predictions; these outputs need validation against a working model
before being relied on. Merely rerunning the current script can reproduce the defect.

Required repair: load the tracked portable model, obtain the exact feature names and
order from its own metadata, validate input features, and stop with a nonzero status
on model or inference failure. Then generate and validate today's/tomorrow's cards.

## 2. US: a green workflow is currently compatible with collecting no races

Today's [US ingestion run](https://github.com/gmalbert/horse-racing-predictions/actions/runs/39593809886)
is green, but its archived source report shows:

- NYRA cannot launch Chromium: the Playwright browser executable is missing.
- The T2/T3 probes report the same missing executable.
- The T1 child pipeline reports `FAILED (TB=False QH=False H=False)` while its process
  returns code zero. The archive does not include the detailed individual T1 source
  reports, so the underlying T1 fetch/parser failures are not yet established.
- Both daily cards contain zero races. `--allow-empty-racecards` permits this to pass
  the ingestion guardrails.
- The predictor logs `Racecard file contained no races — nothing to predict` and
  exits successfully without writing prediction CSVs.
- The commit step attempts to add raw cards/source reports that `.gitignore` excludes;
  errors are suppressed by `|| true`.

Required repair: install the matching browser before collection, propagate genuine
source failures, distinguish a verified no-racing day from collection failure, and
persist the intended source outputs. A browser installation alone does not establish
that every external source/parser works; validate actual race/runner counts afterward.

The downloaded report is retained under the ignored diagnostic folder:
`web/.runtime/stale-audit/us/raw/us_source_report_2026-10-21.json`.

## 3. Historical US results are fetched successfully and then discarded

Today's [TVG results run](https://github.com/gmalbert/horse-racing-predictions/actions/runs/39579354498)
reports:

- Loaded 795,876 existing rows, with dates through 2026-05-09.
- Fetched 1,578 races / 13,544 runner rows for the requested recent period.
- Saved 809,420 rows to `data/processed/us_races_cleaned.parquet` in the runner.

However, `fetch_us_tvg_results.yml` stages only `data/raw/tvg_results_*.json`.
Those raw files are ignored, and the updated parquet is not staged or uploaded as a
durable output. The weekly TVG backfill workflow has the same raw-only commit pattern.
Consequently, the next checkout starts again from the older historical parquet.

Required repair: publish a durable updated dataset (or durable raw results plus a
downstream rebuild), then verify its maximum result date after a fresh checkout.
Existing successful fetches do not currently establish a working end-to-end refresh.

## 4. Statistics select an older dataset; base results are also out of date

Local parquet metadata:

| File | Rows | Maximum date |
|---|---:|---|
| `race_scores_engineered.parquet` | 639,564 | 2026-01-31 |
| `race_scores_or_context.parquet` | 639,564 | 2026-01-31 |
| `race_scores_connections_v2.parquet` | 639,564 | 2026-01-31 |
| `race_scores.parquet` | 653,509 | 2026-04-11 |
| `all_gb_races.parquet` | 653,509 | 2026-04-11 |
| `all_gb_races_cleaned.parquet` | 653,509 | 2026-04-11 |
| `us_races_cleaned.parquet` | 795,876 | 2026-05-09 |

`scripts/export_web_data.py::_stats_source()` returns the first existing engineered
dataset rather than comparing result dates. For the available local files this selects
January instead of April. The trainer/jockey export only needs result fields, so it
should choose the freshest compatible result source.

Using the April file improves the statistics cutoff but does not make results current.
The weekly training workflow reads the existing historical files; it does not append
new GB result records to the base history. `append_new_race_data.py` exists as a manual
CSV ingestion utility and is not called by the scheduled workflows reviewed here.
The prediction-result capture workflow updates prediction/betting history rather than
establishing a current full historical base.

## Recommended repair order

1. Restore International model/feature compatibility and fail-on-error behavior.
   Current racecards are already available, making this the most direct route to
   validating fresh predictions without new scraping.
2. Decouple scheduled collection/dispatch from the active development branch; choose
   an explicit publishing checkout/ref and validate successful output publication.
3. Repair US browser setup and empty/failure semantics, then inspect individual T1
   failures and verify source coverage with real cards.
4. Persist the successful TVG result updates and restore historical GB result ingestion.
5. Select the freshest compatible statistics source, regenerate the web exports,
   rebuild the frontend container, and verify displayed dates against upstream data.
6. Gate pipeline success on expected outputs and source freshness, not merely process
   completion. Keep explicit exceptions for genuinely verified no-racing days.

No live scraping, prediction regeneration, workflow dispatch, push, or model retraining
was performed during this audit.
