# Model Deployment Handoff: US Pattern and International Rollout

## Purpose

This document explains the June 2026 US prediction deployment fix and provides
a repeatable process for deploying international race models through GitHub and
Streamlit.

Source pull request: [#66 — Fix US model deployment](https://github.com/gmalbert/horse-racing-predictions/pull/66)

## US incident summary

The US prediction command failed with:

```text
[ERROR] No model found at .../models/horse_win_predictor.pkl
```

This was not a racecard-fetching failure. `scripts/predict_us_races.py` loaded
the prediction model before reading the daily racecard, so execution stopped
before any race data was processed.

The underlying deployment mismatch was:

- `.gitignore` excludes `models/*.pkl`.
- The US predictor required `us_horse_model.pkl`, with
  `horse_win_predictor.pkl` as its fallback.
- Both pickle files existed locally but were absent from GitHub-based
  deployments.
- A tracked XGBoost JSON model existed for the UK model, but the US loader did
  not support JSON.
- When a US pickle was available locally, the loader incorrectly paired it with
  the UK `feature_columns.txt` rather than `us_feature_columns.txt`.

## Implemented solution

The US deployment now uses these artifacts in priority order:

| Priority | Model | Feature contract | Label |
|---:|---|---|---|
| 1 | `models/us_horse_model.json` | `models/us_feature_columns.txt` | US |
| 2 | `models/us_horse_model.pkl` | `models/us_feature_columns.txt` | US |
| 3 | `models/horse_win_predictor.json` | `models/feature_columns.txt` | UK base |
| 4 | `models/horse_win_predictor.pkl` | `models/feature_columns.txt` | UK base |

The following changes were made:

- Added the tracked `models/us_horse_model.json` artifact.
- Updated `scripts/predict_us_races.py` to load XGBoost JSON models.
- Paired US models with `us_feature_columns.txt`.
- Added a model/feature-count check before inference.
- Updated `scripts/train_us_model.py` to save a deployable JSON model after
  every training run while retaining an ignored local pickle for compatibility.
- Updated `pages/us_racing.py` so model readiness recognizes the JSON artifact.

The deployed US model and feature contract both contain 64 features. A real
`predict_proba` call was verified with XGBoost 3.0.4.

## International rollout checklist

### 1. Inventory the international model

Identify all of the following before changing code:

- prediction entry point;
- training entry point;
- local model filename and serialization format;
- exact feature-list file used during training;
- model type stored inside the pickle;
- UI or diagnostics that check whether the model exists;
- deployment mechanism and ignored file patterns.

Useful commands:

```bash
rg -n "international|horse_model|feature_columns|pickle.load|load_model" scripts pages
git check-ignore -v models/<international-model>.pkl
git ls-files models
ls -lh models
```

Do not assume that similarly named feature files are interchangeable. Model
availability and the feature contract must be treated as one versioned unit.

### 2. Confirm that JSON export is safe

The US conversion was safe because the pickle contained an `XGBClassifier` and
the predictor only required the underlying XGBoost booster.

Inspect the international pickle using the same pinned dependencies used to
train it. Confirm:

- the object is an XGBoost classifier or regressor;
- its booster feature count matches the intended feature-list length;
- no required preprocessing pipeline exists only inside the pickle;
- no calibrator, ensemble, or custom Python wrapper would be lost.

If the pickle contains a `Pipeline`, `CalibratedClassifierCV`, stacked ensemble,
custom transformer, or other wrapper, exporting only `model.get_booster()` is
not equivalent. In that case, either reconstruct and version every component or
store the complete artifact in approved release/object storage instead of
silently dropping model behavior.

### 3. Export the deployable artifact

For a plain trusted XGBoost model:

```python
import pickle

with open("models/<international-model>.pkl", "rb") as fh:
    model = pickle.load(fh)

model.get_booster().save_model("models/<international-model>.json")
```

Load it in the prediction process with:

```python
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("models/<international-model>.json")
```

Use the model class appropriate to the training objective. Never unpickle an
artifact from an untrusted source.

### 4. Make future training runs deployment-safe

Update the international training script so a successful training run writes:

- a tracked JSON model for deployment;
- the matching tracked feature-list file;
- metadata containing the training date, feature count, data volume, and model
  metric;
- an optional ignored pickle only when local compatibility requires it.

Example:

```python
json_path = MODELS_DIR / "<international-model>.json"
model.get_booster().save_model(str(json_path))

feature_path.write_text("\n".join(feature_cols))
```

The JSON and feature-list changes must be committed together after retraining.

### 5. Update prediction loading

Use explicit model/feature pairs rather than selecting a model and feature file
independently:

```python
candidates = [
    (international_json, international_features, "International"),
    (international_pickle, international_features, "International"),
    (base_json, base_features, "Base"),
]
```

After loading, fail clearly when the feature contract is incompatible:

```python
expected = model.get_booster().num_features()
if len(feature_cols) != expected:
    raise ValueError(
        f"Model expects {expected} features, but the feature file contains "
        f"{len(feature_cols)}"
    )
```

Do not catch inference errors and quietly emit uniform probabilities. A broken
model contract should be visible in logs and UI diagnostics.

### 6. Update operational diagnostics

Search dashboards, health checks, scheduled workflows, and documentation for
hard-coded `.pkl` paths. A deployment can work while the UI incorrectly says
the model is missing if these checks are not updated.

Diagnostics should report:

- selected artifact name;
- selected feature contract;
- model feature count;
- feature-file count;
- model label/version;
- actionable errors when loading fails.

### 7. Validate before publishing

Minimum validation:

```bash
python -m py_compile \
  scripts/<international-predictor>.py \
  scripts/<international-trainer>.py
```

Then verify in the deployment dependency version:

1. JSON loads successfully.
2. Model feature count equals feature-list length.
3. `predict_proba` or `predict` succeeds on a correctly ordered matrix.
4. A representative historical racecard completes end to end.
5. Output is not a field-wide uniform fallback.
6. The UI recognizes the JSON model.
7. A clean checkout contains every required artifact.

The final clean-checkout test is essential: local ignored files can conceal the
same defect that caused the US incident.

### 8. Publish safely

Keep the pull request limited to:

- deployable model artifact;
- matching feature contract if it changed;
- loader changes;
- training serialization changes;
- diagnostics and focused tests.

Before pushing:

```bash
git diff --check
git diff --cached --name-status
git check-ignore -v models/<international-model>.json
```

The JSON must not be ignored. GitHub blocks regular files above 100 MiB and
recommends keeping repository objects small. The US JSON is approximately
1.9 MiB and does not require Git LFS. Reassess storage when an international
artifact is materially larger or is updated frequently.

## Known follow-up from the US work

The tracked UK fallback currently has a separate contract mismatch:

- `horse_win_predictor.json` expects 75 features;
- the current `feature_columns.txt` contains 82 features.

The US loader now reports this mismatch clearly if it ever needs the UK
fallback. Normal US deployment is unaffected because the valid 64-feature US
JSON has priority. The UK model and feature artifacts should nevertheless be
realigned in a separate change.

## Operational boundaries

Deploying the model fixes model availability for all future dates. It does not
guarantee that daily international racecards were fetched successfully. Keep
these as distinct health checks:

1. data ingestion produced the dated racecard;
2. model and feature artifacts loaded;
3. inference produced the dated predictions;
4. the UI loaded the prediction output.

## Rollback

If the international JSON deployment fails:

1. capture the exact loader or feature-contract error;
2. revert the deployment pull request rather than deleting artifacts manually
   in production;
3. keep the previous known-good model available while correcting export or
   preprocessing compatibility;
4. repeat validation from a clean checkout before redeploying.

