# US Track Scraping Tracker

Operational tracker for implementing the US scraping roadmap.

How to use:
- Keep one owner per task (GitHub handle or initials).
- Move status through: Backlog -> In Progress -> Blocked -> Done.
- Keep acceptance criteria objective and testable.
- Update Last Updated when any task changes.

Last Updated: 2026-05-09 (T1-TB detailed execution cards added)

---

## Legend

Status values:
- Backlog
- In Progress
- Blocked
- Done

Priority values:
- P0 (critical path)
- P1 (high)
- P2 (normal)
- P3 (nice-to-have)

---

## Program Snapshot

| Area | Total | Done | In Progress | Blocked | Backlog |
|---|---:|---:|---:|---:|---:|
| Foundation | 6 | 0 | 0 | 0 | 6 |
| Tier 1 Tracks | 22 | 0 | 12 | 0 | 10 |
| Tier 2 Tracks | 9 | 0 | 0 | 0 | 9 |
| Tier 3 Tracks | 3 | 0 | 0 | 0 | 3 |
| JSON Endpoints | 15 | 0 | 0 | 0 | 15 |
| QA/Monitoring | 8 | 0 | 0 | 0 | 8 |

---

## Foundation (P0)

### F-01 Canonical schema and versioning
- [ ] Task ID: F-01
- Status: Backlog
- Priority: P0
- Owner: TBD
- Due: TBD
- Deliverable: canonical entries/results schema with parser_version and source fields
- Acceptance:
  - schema documented in code comments
  - all writers emit schema-compliant records

### F-02 Source client scaffolding
- [ ] Task ID: F-02
- Status: Backlog
- Priority: P0
- Owner: TBD
- Due: TBD
- Deliverable: source modules under scripts/us/sources/
- Acceptance:
  - module skeletons for equibase, usta, nyra, cdi, tracksite_static
  - shared retry/pacing utility consumed by all clients

### F-03 Adapter layer
- [ ] Task ID: F-03
- Status: Backlog
- Priority: P0
- Owner: TBD
- Due: TBD
- Deliverable: per-track adapters under scripts/us/adapters/
- Acceptance:
  - each adapter returns canonical schema
  - adapter unit tests pass for fixture payloads

### F-04 Storage writer + idempotency
- [ ] Task ID: F-04
- Status: Backlog
- Priority: P0
- Owner: TBD
- Due: TBD
- Deliverable: raw + processed writers with dedupe keys
- Acceptance:
  - rerunning same date/track does not duplicate rows
  - parquet outputs produced for entries/results

### F-05 Validation framework
- [ ] Task ID: F-05
- Status: Backlog
- Priority: P0
- Owner: TBD
- Due: TBD
- Deliverable: validation checks for row counts and mandatory fields
- Acceptance:
  - failing checks return non-zero exit code
  - validation summary written per run

### F-06 CI smoke suite
- [ ] Task ID: F-06
- Status: Backlog
- Priority: P1
- Owner: TBD
- Due: TBD
- Deliverable: tests for parser contracts and fixture regressions
- Acceptance:
  - CI job runs tests on PR
  - contract tests cover top 5 sources

---

## Tier 1 Track Rollout

### Thoroughbred (Tier 1)

| Track | Task ID | Status | Priority | Owner | Source Target | Notes |
|---|---|---|---|---|---|---|
| Tampa Bay Downs | T1-TB-01 | Backlog | P0 | TBD | Track site HTML | Easiest HTML |
| Charles Town | T1-TB-02 | Backlog | P1 | TBD | Track site HTML | Simple tables |
| Mountaineer | T1-TB-03 | Backlog | P1 | TBD | Track site HTML | Simple tables |
| Canterbury Park | T1-TB-04 | Backlog | P1 | TBD | Track site HTML | Static layouts |
| Presque Isle Downs | T1-TB-05 | Backlog | P1 | TBD | Track site HTML | Static layouts |
| Parx | T1-TB-06 | Backlog | P1 | TBD | Track site HTML | Clean tables |
| Penn National | T1-TB-07 | Backlog | P1 | TBD | Track site HTML | Clean tables |
| Ellis Park | T1-TB-08 | Backlog | P1 | TBD | Track site HTML | Seasonal |
| Lone Star Park | T1-TB-09 | Backlog | P1 | TBD | Track site HTML | Seasonal |
| Sam Houston | T1-TB-10 | Backlog | P1 | TBD | Track site HTML | Seasonal |

### Thoroughbred (Tier 1) Implementation Pack

Definition of done for each T1-TB task:
- [ ] Source URL and terms-of-use reviewed
- [ ] Parser implemented in scripts/ with stable selectors
- [ ] Fixture captured in tests/fixtures/us_sources/
- [ ] Parser unit test added and passing
- [ ] Adapter output validated against canonical schema
- [ ] Included in batch run and smoke-tested for one live date

#### T1-TB-01 Tampa Bay Downs
- [ ] Status: Backlog
- [ ] Priority: P0
- [ ] Owner: @gmalbert
- [ ] Due: 2026-05-23
- [ ] Sprint target: Sprint 1
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/tampa_bay_downs_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-02 Charles Town
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-05-23
- [ ] Sprint target: Sprint 1
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/charles_town_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-03 Mountaineer
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-05-23
- [ ] Sprint target: Sprint 1
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/mountaineer_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-04 Canterbury Park
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-06-13
- [ ] Sprint target: Sprint 2
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/canterbury_park_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-05 Presque Isle Downs
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-06-13
- [ ] Sprint target: Sprint 2
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/presque_isle_downs_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-06 Parx
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-06-13
- [ ] Sprint target: Sprint 2
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/parx_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-07 Penn National
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-06-13
- [ ] Sprint target: Sprint 2
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/penn_national_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-08 Ellis Park
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-06-13
- [ ] Sprint target: Sprint 2
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/ellis_park_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-09 Lone Star Park
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-06-13
- [ ] Sprint target: Sprint 2
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/lone_star_park_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

#### T1-TB-10 Sam Houston
- [ ] Status: Backlog
- [ ] Priority: P1
- [ ] Owner: @gmalbert
- [ ] Due: 2026-06-13
- [ ] Sprint target: Sprint 2
- [ ] Source URL confirmed and robots/terms reviewed
- [ ] Parser implemented for entries + race metadata
- [ ] Track mapped to canonical course name in adapter output
- [ ] Fixture saved to tests/fixtures/us_sources/sam_houston_entries.html
- [ ] Unit test added for parser success and empty-card behavior
- [ ] Included in batch smoke run for one live date

### Quarter Horse (Tier 1)

| Track | Task ID | Status | Priority | Owner | Source Target | Notes |
|---|---|---|---|---|---|---|
| Los Alamitos | T1-QH-01 | In Progress | P0 | @gmalbert | Track JSON/HTML | Playwright puller implemented |
| Ruidoso Downs | T1-QH-02 | In Progress | P1 | @gmalbert | Track JSON/HTML | Playwright puller implemented |
| Delta Downs | T1-QH-03 | In Progress | P1 | @gmalbert | Equibase/Track | Playwright puller implemented |
| Evangeline Downs | T1-QH-04 | In Progress | P1 | @gmalbert | Equibase/Track | Playwright puller implemented |
| Zia Park | T1-QH-05 | In Progress | P1 | @gmalbert | Track site HTML | Playwright puller implemented |
| Sunland Park | T1-QH-06 | In Progress | P1 | @gmalbert | Track site HTML | Playwright puller implemented |

### Quarter Horse (Tier 1) Implementation Pack

Definition of done for each T1-QH task:
- [x] Playwright puller created: scripts/pull_t1_qh_tracks.py
- [x] PDF extractor created: scripts/extract_t1_qh_pdf_artifacts.py
- [x] Canonical parser created: scripts/parse_t1_qh_tracksite.py
- [x] Master orchestrator integrated: run_t1_all_daily_pipeline.py
- [x] Tested end-to-end for 2026-05-09 (pipeline: OK)
- [ ] Fixture HTMLs captured for each track to tests/fixtures/us_sources/
- [ ] Per-track parser refinements based on live data
- [ ] Integrated into daily GitHub Actions scheduler

#### T1-QH-01 Los Alamitos
- [x] Status: In Progress
- [x] Priority: P0
- [x] Owner: @gmalbert
- [x] Due: 2026-05-23
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.losalamitosonline.com)
- [x] Playwright puller added to pull_t1_qh_tracks.py (LA entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Quarter Horse"
- [ ] Fixture saved to tests/fixtures/us_sources/los_alamitos_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-QH-02 Ruidoso Downs
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-05-23
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.ruidosodowns.com)
- [x] Playwright puller added to pull_t1_qh_tracks.py (RUD entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Quarter Horse"
- [ ] Fixture saved to tests/fixtures/us_sources/ruidoso_downs_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-QH-03 Delta Downs
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-06-13
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.deltadowns.com)
- [x] Playwright puller added to pull_t1_qh_tracks.py (DLD entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Quarter Horse"
- [ ] Fixture saved to tests/fixtures/us_sources/delta_downs_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-QH-04 Evangeline Downs
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-06-13
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.evangelinedowns.com)
- [x] Playwright puller added to pull_t1_qh_tracks.py (EVD entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Quarter Horse"
- [ ] Fixture saved to tests/fixtures/us_sources/evangeline_downs_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-QH-05 Zia Park
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-06-13
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.ziaparkracing.com)
- [x] Playwright puller added to pull_t1_qh_tracks.py (ZIA entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Quarter Horse"
- [ ] Fixture saved to tests/fixtures/us_sources/zia_park_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-QH-06 Sunland Park
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-06-13
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.sunland.com)
- [x] Playwright puller added to pull_t1_qh_tracks.py (SUN entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Quarter Horse"
- [ ] Fixture saved to tests/fixtures/us_sources/sunland_park_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

### Harness (Tier 1)

| Track | Task ID | Status | Priority | Owner | Source Target | Notes |
|---|---|---|---|---|---|---|
| Northfield Park | T1-H-01 | In Progress | P0 | @gmalbert | USTA JSON / Track | Playwright puller implemented |
| Scioto Downs | T1-H-02 | In Progress | P0 | @gmalbert | USTA JSON / Track | Playwright puller implemented |
| Rosecroft | T1-H-03 | In Progress | P1 | @gmalbert | USTA JSON / Track | Playwright puller implemented |
| Running Aces | T1-H-04 | In Progress | P1 | @gmalbert | USTA JSON / Track | Playwright puller implemented |
| Plainridge Park | T1-H-05 | In Progress | P1 | @gmalbert | USTA JSON / Track | Playwright puller implemented |
| Cal Expo | T1-H-06 | In Progress | P1 | @gmalbert | Track site HTML | Playwright puller implemented |

### Harness (Tier 1) Implementation Pack

Definition of done for each T1-H task:
- [x] Playwright puller created: scripts/pull_t1_h_tracks.py
- [x] PDF extractor created: scripts/extract_t1_h_pdf_artifacts.py
- [x] Canonical parser created: scripts/parse_t1_h_tracksite.py
- [x] Master orchestrator integrated: run_t1_all_daily_pipeline.py
- [x] Tested end-to-end for 2026-05-09 (pipeline: OK)
- [ ] Fixture HTMLs captured for each track to tests/fixtures/us_sources/
- [ ] Per-track parser refinements based on live data (includes driver field for harness)
- [ ] Integrated into daily GitHub Actions scheduler

#### T1-H-01 Northfield Park
- [x] Status: In Progress
- [x] Priority: P0
- [x] Owner: @gmalbert
- [x] Due: 2026-05-23
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.northfieldpark.com)
- [x] Playwright puller added to pull_t1_h_tracks.py (NTH entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Harness" and driver field
- [ ] Fixture saved to tests/fixtures/us_sources/northfield_park_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-H-02 Scioto Downs
- [x] Status: In Progress
- [x] Priority: P0
- [x] Owner: @gmalbert
- [x] Due: 2026-05-23
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.sciotodowns.com)
- [x] Playwright puller added to pull_t1_h_tracks.py (SCD entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Harness" and driver field
- [ ] Fixture saved to tests/fixtures/us_sources/scioto_downs_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-H-03 Rosecroft
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-06-13
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.rosecroft.com)
- [x] Playwright puller added to pull_t1_h_tracks.py (RSC entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Harness" and driver field
- [ ] Fixture saved to tests/fixtures/us_sources/rosecroft_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-H-04 Running Aces
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-06-13
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.runningacesharness.com)
- [x] Playwright puller added to pull_t1_h_tracks.py (RUN entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Harness" and driver field
- [ ] Fixture saved to tests/fixtures/us_sources/running_aces_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-H-05 Plainridge Park
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-06-13
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.plainridgepark.com)
- [x] Playwright puller added to pull_t1_h_tracks.py (PLN entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Harness" and driver field
- [ ] Fixture saved to tests/fixtures/us_sources/plainridge_park_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

#### T1-H-06 Cal Expo
- [x] Status: In Progress
- [x] Priority: P1
- [x] Owner: @gmalbert
- [x] Due: 2026-06-13
- [x] Sprint target: Sprint 3
- [x] Source URL confirmed (https://www.calfair.com)
- [x] Playwright puller added to pull_t1_h_tracks.py (CAL entry in TRACKS dict)
- [x] PDF extractor included in pipeline
- [x] Canonical parser returns summary_event rows with breed="Harness" and driver field
- [ ] Fixture saved to tests/fixtures/us_sources/cal_expo_entries.html
- [ ] Unit test added for parser success
- [ ] Included in batch smoke run for one live date

---

## Tier 2 Track Rollout

### Thoroughbred (Tier 2)

| Track | Task ID | Status | Priority | Owner | Source Target |
|---|---|---|---|---|---|
| Gulfstream Park | T2-TB-01 | Backlog | P1 | TBD | HRN + TVG |
| Santa Anita | T2-TB-02 | Backlog | P1 | TBD | Equibase |
| Del Mar | T2-TB-03 | Backlog | P1 | TBD | Equibase |
| Oaklawn | T2-TB-04 | Backlog | P1 | TBD | HRN |
| Fair Grounds | T2-TB-05 | Backlog | P1 | TBD | CDI JSON |
| Monmouth Park | T2-TB-06 | Backlog | P1 | TBD | Track/Equibase |

### Harness (Tier 2)

| Track | Task ID | Status | Priority | Owner | Source Target |
|---|---|---|---|---|---|
| Meadowlands | T2-H-01 | Backlog | P0 | TBD | USTA JSON |
| Yonkers | T2-H-02 | Backlog | P0 | TBD | USTA JSON |
| Hoosier Park | T2-H-03 | Backlog | P0 | TBD | USTA JSON |

---

## Tier 3 Track Rollout

| Track Group | Task ID | Status | Priority | Owner | Source Target | Notes |
|---|---|---|---|---|---|---|
| NYRA (AQU/BEL/SAR) | T3-01 | Backlog | P0 | TBD | NYRA React API + fallback DOM | Existing parser to harden |
| Churchill Downs | T3-02 | Backlog | P0 | TBD | CDI JSON + Equibase fallback | Dynamic endpoints |
| Keeneland | T3-03 | Backlog | P1 | TBD | Equibase + static endpoint probes | Dynamic wrappers |

---

## Hidden JSON Endpoint Program

| Source/Track | Task ID | Status | Priority | Owner | Endpoint Discovery | Parser Complete | Notes |
|---|---|---|---|---|---|---|---|
| Meadowlands | J-01 | Backlog | P0 | TBD | [ ] | [ ] | USTA JSON |
| Yonkers | J-02 | Backlog | P0 | TBD | [ ] | [ ] | USTA JSON |
| Hoosier Park | J-03 | Backlog | P0 | TBD | [ ] | [ ] | USTA JSON |
| Scioto Downs | J-04 | Backlog | P0 | TBD | [ ] | [ ] | USTA JSON |
| Northfield Park | J-05 | Backlog | P0 | TBD | [ ] | [ ] | USTA JSON |
| Pocono Downs | J-06 | Backlog | P1 | TBD | [ ] | [ ] | USTA JSON |
| Harrah's Philadelphia | J-07 | Backlog | P1 | TBD | [ ] | [ ] | USTA JSON |
| Rosecroft | J-08 | Backlog | P1 | TBD | [ ] | [ ] | USTA JSON |
| Running Aces | J-09 | Backlog | P1 | TBD | [ ] | [ ] | USTA JSON |
| Tioga Downs | J-10 | Backlog | P1 | TBD | [ ] | [ ] | USTA JSON |
| Vernon Downs | J-11 | Backlog | P1 | TBD | [ ] | [ ] | USTA JSON |
| Plainridge Park | J-12 | Backlog | P1 | TBD | [ ] | [ ] | USTA JSON |
| NYRA | J-13 | Backlog | P0 | TBD | [ ] | [ ] | React JSON endpoints |
| CDI family | J-14 | Backlog | P0 | TBD | [ ] | [ ] | Shared CDI API |
| TVG/TwinSpires | J-15 | Backlog | P1 | TBD | [ ] | [ ] | Internal JSON patterns |

---

## QA, Monitoring, and Ops

### Q-01 Parser contract fixtures
- [ ] Status: Backlog
- Priority: P0
- Owner: TBD
- Deliverable: fixture set under tests/fixtures/us_sources/
- Acceptance:
  - each source parser tested against saved fixture
  - schema and key-field assertions present

### Q-02 Drift detection
- [ ] Status: Backlog
- Priority: P0
- Owner: TBD
- Deliverable: drift checker script
- Acceptance:
  - alerts on major field drop or row count anomalies

### Q-03 Run report
- [ ] Status: Backlog
- Priority: P1
- Owner: TBD
- Deliverable: per-run JSON/CSV report
- Acceptance:
  - parse success by source and track
  - missing-field percentages

### Q-04 Scheduler
- [ ] Status: Backlog
- Priority: P1
- Owner: TBD
- Deliverable: daily schedule job
- Acceptance:
  - retry on transient failures
  - idempotent write behavior

### Q-05 Source kill-switch
- [ ] Status: Backlog
- Priority: P1
- Owner: TBD
- Deliverable: config-based source disable list
- Acceptance:
  - source can be disabled without code change

### Q-06 Data freshness guard
- [ ] Status: Backlog
- Priority: P1
- Owner: TBD
- Deliverable: stale-data warning in UI and logs
- Acceptance:
  - warning if expected daily data missing/old

### Q-07 Reconciliation checks
- [ ] Status: Backlog
- Priority: P2
- Owner: TBD
- Deliverable: entries vs results reconciliation
- Acceptance:
  - unmatched runner/race counts reported

### Q-08 Throughput and rate-limit controls
- [ ] Status: Backlog
- Priority: P2
- Owner: TBD
- Deliverable: request pacing policy + overrides
- Acceptance:
  - configurable per-source delays and concurrency caps

---

## Sprint Board (Move Tasks Weekly)

### Sprint 1
- [ ] F-01 Canonical schema and versioning
- [ ] F-02 Source client scaffolding
- [ ] F-03 Adapter layer
- [ ] F-04 Storage writer + idempotency
- [ ] T1-TB-01 Tampa Bay Downs
- [ ] T1-TB-02 Charles Town
- [ ] T1-TB-03 Mountaineer

### Sprint 2
- [ ] T1-TB-04 Canterbury Park
- [ ] T1-TB-05 Presque Isle Downs
- [ ] T1-TB-06 Parx
- [ ] T1-TB-07 Penn National
- [ ] T1-TB-08 Ellis Park
- [ ] T1-TB-09 Lone Star Park
- [ ] T1-TB-10 Sam Houston

### Sprint 3
- [ ] T1-QH-01 through T1-QH-06
- [ ] T1-H-01 through T1-H-06
- [ ] Q-01 Parser contract fixtures

### Sprint 4
- [ ] T2-TB-01 through T2-TB-06
- [ ] T2-H-01 through T2-H-03
- [ ] Q-02 Drift detection

### Sprint 5
- [ ] J-01 through J-12 (Harness JSON wave)
- [ ] J-13 NYRA JSON
- [ ] J-14 CDI family JSON
- [ ] J-15 TVG/TwinSpires JSON

### Sprint 6
- [ ] T3-01 NYRA hardening
- [ ] T3-02 Churchill Downs
- [ ] T3-03 Keeneland
- [ ] Q-03 through Q-08 operations hardening

---

## Notes / Decisions Log

- 2026-05-09: Initial tracker created from US execution plan.
- 2026-05-09: Implemented T1 Thoroughbred execution pack (T1-TB-01 to T1-TB-10) with per-track done criteria and sprint targets.
- 2026-05-09: Expanded all highlighted T1 Thoroughbred tasks with owner, due date, fixture paths, and parser/test/smoke-run checklist items.
- 2026-05-09: Implemented T1 Quarter Horse (6 tracks) and T1 Harness (6 tracks) breeds:
  - Created breed-specific Playwright pullers: pull_t1_qh_tracks.py, pull_t1_h_tracks.py
  - Created breed-specific PDF extractors: extract_t1_qh_pdf_artifacts.py, extract_t1_h_pdf_artifacts.py
  - Created breed-specific canonical parsers: parse_t1_qh_tracksite.py, parse_t1_h_tracksite.py
  - Created master orchestrator: run_t1_all_daily_pipeline.py (runs all 3 breeds in sequence)
  - Executed and validated end-to-end pipeline for 2026-05-09 (all breeds: OK)
  - Canonical outputs: us_t1_qh_canonical_all.csv, us_t1_h_canonical_all.csv (archival append mode)
  - Marked T1-QH-01 through T1-QH-06 and T1-H-01 through T1-H-06 as In Progress
- Add one bullet per architecture or source decision with date and rationale.
