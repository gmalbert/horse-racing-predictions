# Racing Post browser racecard collector

The browser collector supplements the API baseline with Racing Post racecard
data rendered in ordinary Chrome. It is designed for an authorized,
self-hosted Windows PC and intentionally contains no stealth, fingerprint
modification, proxy rotation, or challenge-solving behavior.

## Local scheduled setup

1. Install Google Chrome, Python 3.11, Git, and the GitHub CLI (`gh`) on the
   Windows PC.
2. Run `gh auth login` as the same Windows user that will run the task.
3. In Task Scheduler, create a daily task at 06:50 ET that runs only while
   that user is logged in. Chrome must be able to open in that desktop session.
4. Set **Program/script** to `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`; set **Arguments** to:

   ```text
   -NoProfile -ExecutionPolicy Bypass -File "C:\path\to\horse-racing-predictions\run_racing_post_browser.ps1" -Commit -Push -TriggerPredictions
   ```

5. Set **Start in** to the repository directory. Do not run the task as a
   Windows service.

## Optional persistent browser profile

Set the `RACING_POST_PROFILE_DIR` environment variable for the logged-in
Windows user to a dedicated directory, for example:

```text
C:\actions-data\racing-post-profile
```

The directory must not be the profile used by an already-running Chrome
instance. A dedicated profile preserves ordinary cookies and consent choices
between runs. It must not be committed to the repository.

## Local verification

For a full local morning run, use the PowerShell wrapper from the logged-in
desktop session:

```powershell
.\run_racing_post_browser.ps1 -Commit -Push -TriggerPredictions
```

It writes timestamped logs under `logs/racing-post-browser/`, prevents
overlapping runs, and only stages `data/raw/racecards_*.json`. After a
successful push, `-TriggerPredictions` dispatches **Precompute Daily
Predictions** on the same branch. Omit all three switches when you want a
collection-only run.

Run visible Chrome for one date:

```powershell
python scripts/fetch_racecards_browser.py --date 2026-08-25
```

Fetch today and tomorrow using the same Eastern Time date boundary as the
daily workflows:

```powershell
python scripts/fetch_racecards_browser.py --days 2 --timezone America/New_York
```

The collector refuses to replace an existing snapshot unless every discovered
race produces usable runner data. On failure it saves the rendered HTML and a
screenshot from the last page under `tmp/racing-post-browser/` for diagnosis.

The regular `fetch_racecards.yml` API workflow remains the baseline. The local
browser task runs afterward, commits a Racing Post snapshot only after full
validation, and then dispatches `Precompute Daily Predictions`. That workflow
generates predictions and pushes its own output commit. It is no longer
independently scheduled, which prevents it from racing the browser collection.
