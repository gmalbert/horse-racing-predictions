# Racing Post browser racecard collector

The browser collector supplements the API baseline with Racing Post racecard
data rendered in ordinary Chrome. It is designed for an authorized,
self-hosted Windows runner and intentionally contains no stealth, fingerprint
modification, proxy rotation, or challenge-solving behavior.

## Runner setup

1. Install Google Chrome and Python 3.11 on the Windows host.
2. Add a GitHub Actions self-hosted runner to this repository.
3. Add the custom runner label `racing-post-browser`. The workflow also
   requires the standard `self-hosted`, `Windows`, and `X64` labels.
4. Start the runner interactively in the logged-in desktop session. Do not run
   it as a Windows service: services cannot display Chrome in the user's
   desktop session.
5. Keep the runner online for the scheduled 11:50 UTC collection, or dispatch
   **Fetch Racing Post Racecards (Self-hosted Browser)** manually.

The workflow installs the Python requirements and bundled Chromium as a
fallback. The collector prefers installed system Chrome.

## Optional persistent browser profile

Set the repository Actions variable `RACING_POST_PROFILE_DIR` to a dedicated
directory on the runner, for example:

```text
C:\actions-data\racing-post-profile
```

The directory must not be the profile used by an already-running Chrome
instance. A dedicated profile preserves ordinary cookies and consent choices
between runs. It must not be committed to the repository.

## Local verification

For recurring local collection, use the PowerShell wrapper from the logged-in
desktop session:

```powershell
.\run_racing_post_browser.ps1 -Commit -Push
```

It writes timestamped logs under `logs/racing-post-browser/`, prevents
overlapping runs, and only stages `data/raw/racecards_*.json`. Omit `-Commit`
and `-Push` when you want a collection-only run. Schedule it with Windows Task
Scheduler using the same logged-in user account that can open Chrome.

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
race produces usable runner data. On workflow failure it uploads the rendered
HTML and screenshot from the last page as a seven-day diagnostic artifact.

The regular `fetch_racecards.yml` API workflow remains the baseline. The
self-hosted browser workflow runs afterward and commits a Racing Post snapshot
only after full validation.
