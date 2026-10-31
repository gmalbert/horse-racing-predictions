# US track entry collection

`scripts/us_track_sources.py` holds the track landing pages and the independent
Equibase/USTA identifiers. The three `pull_t1_*_tracks.py` commands and the
additional-source groups use `us_track_collector.py`.

## Source selection

Thoroughbred and quarter-horse collectors request the dated Equibase full card
on `tvg.equibase.com`, with `www.equibase.com` and the official track page as
fallbacks. Harness collectors use USTA's mobile track listing, select the link
whose text matches the requested full date, and load that card. They do not
invent an API endpoint or assume that a track's homepage contains runners.

Verified source examples:

- [Los Alamitos entries](https://www.losalamitos.com/ECE.aspx?ver=1)
- [Zia Park racing](https://www.ziaparkcasino.com/racing)
- [Sunland Park racing](https://sunland-park.com/racing/)
- [Delta Downs racing information](https://deltadowns.boydgaming.com/racing/racing-information)
- [Plainridge racing and its USTA entries link](https://www.plainridgeparkcasino.com/racing)
- [Running Aces racing](https://runaces.com/racing/live-racing/)
- [Scioto's official racing URL, listed by the Ohio commission](https://racing.ohio.gov/ohio-racetracks/scioto-downs)
- [Ellis Park's current racing page](https://www.ellisparkowensborogaming.com/racing/)

## Recovery and validation

Requests use HTTP/1.1 first. Transport failures can retry in a fresh browser
page; every page closes even if navigation fails. Chromium's HTTP/2 fallback
is disabled for the affected servers. TLS verification remains enabled.
Access-denied responses switch to independent public sources. They are not
retried indefinitely.

Each track has a 70-second budget and at most seven candidate URLs. Each breed
collects at most three tracks concurrently, and the master pipeline runs the
three breeds concurrently. Discovered links rank dated entries ahead of general
entry pages; results, news, purchases and unrelated domains are excluded or
deprioritized.

The provider's card heading must confirm the track and full requested date.
A date in a URL or calendar selector alone is insufficient. Runner tables use
named columns and retain program numbers, post positions, jockeys/drivers,
trainers, weights and morning-line odds. Scratched runners and USTA result cards
are excluded. Generic landing-page summaries cannot become new canonical
entry records. Historical summary parsing remains available for old snapshots.

## Diagnostics

Per-track JSON and uploaded group reports retain every attempted URL, final
redirect URL, HTTP status, transport, full exception and error category. A track
is `ok`, `not_listed`, or `collection_error`. `not_listed` means the provider has
no published card in the inspected date window; it does not assert that racing
was cancelled or that entries will never be published.

The all-sources report includes `collection_errors`. When no usable races are
collected, source failures exit with code 1. An empty run without collection
errors exits with code 2. Only the latter can defer tomorrow's optional workflow
publication. Any published cards still require the normal volume, odds and
drift checks.

Empty canonical outputs retain CSV headers. Blank legacy history CSVs can be
read and repaired by a subsequent append, and runner identity is included in
deduplication so runners in the same race do not overwrite one another.

## Verification on October 25, 2026

Live collection for October 26 recovered 50 races and 438 runners from
Mountaineer, Parx, Rosecroft, Plainridge and Yonkers. Replaying those captured
responses through the merger and ingestion validator passed the volume and
odds checks (420/438 runners had morning-line odds). The isolated replay had
no historical drift baseline. Prediction generation produced 438 rows using
the local model and default history features in the isolated directory.

Additional live checks recovered October 24 cards for Charles Town, Delta Downs
and Evangeline Downs, and October 25 cards for Los Alamitos and Northfield.
These local checks do not establish network access from a GitHub-hosted runner;
the workflow must be run after publication to verify that environment.

Run offline regressions with:

```powershell
python -m pytest tests/test_t1_parser_hardening.py tests/test_freshness_repairs.py tests/test_predict_us_races.py -q
```


## GitHub harness recovery on October 26, 2026

GitHub-hosted runners timed out connecting to `m.ustrotting.com`, losing
Rosecroft, Plainridge, and Yonkers despite collecting 17 thoroughbred races.
The collector now falls back to USTA's independent desktop host at
`https://racing.ustrotting.com/default.aspx`. Each track uses a separate HTTP
session, posts the server-issued ASP.NET form state, selects the requested
non-qualifying card date, and follows the site's full-card control.

Desktop parsing verifies the track and full date in each race heading,
uses named columns including their colspans, reads both alternating row
classes, and excludes scratches and result tables. The primary driver is
kept separately from alternates. Cards missing races advertised as entries
in the desktop summary are rejected. Diagnostics retain each desktop step.

The local desktop check recovered 13 Rosecroft races (113 runners),
11 Plainridge races (84 runners), and 9 Yonkers races (70 runners) for
October 26. GitHub-runner verification is recorded in the follow-up PR.
The ingestion validator rejects complete harness-source loss even if
thoroughbred volume meets its thresholds; confirmed unlisted dates remain
valid empty coverage for that source group.


A later GitHub runner could reach neither USTA host. Therefore USTA desktop
is a tertiary recovery path, and the public JSON used by DRF Harness entry
pages is now the independent provider fallback after a mobile failure.
Its track is selected from the provider's dated track list. Each race must
match the requested track, selected full date, and actual post timestamp
converted to America/New_York. Completed/cancelled races and scratches are
excluded; morning-line odds, full driver/trainer names, and runner positions
are retained. This avoids depending on Equibase's harness mirror, which does
not carry morning-line odds.

The pull-request verification directly calls this independent provider for
every collected harness track and compares race numbers. Zero tracks is a
failure, so a skipped comparison cannot be reported as successful recovery.
