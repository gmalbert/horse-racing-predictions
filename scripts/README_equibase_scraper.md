# Equibase Scraper

A Python scraper for [equibase.com](https://www.equibase.com) that pulls **entries, results, horse profiles, and race charts**.

---

## Install

```bash
pip install requests beautifulsoup4 lxml cloudscraper
```

> `cloudscraper` handles Equibase's Cloudflare-based bot protection automatically.

---

## Modes

### 1. Entries — today's race entries

```bash
python equibase_scraper.py --mode entries --date 2025-05-15
python equibase_scraper.py --mode entries --date 2025-05-15 --track AQU
```

### 2. Results — past race results

```bash
python equibase_scraper.py --mode results --date 2025-05-14
python equibase_scraper.py --mode results --date 2025-05-14 --track CD
```

### 3. Horse Profile

```bash
python equibase_scraper.py --mode horse --name "Justify"
python equibase_scraper.py --mode horse --name "American Pharoah"
```

### 4. Race Chart (past performance chart for a specific race)

```bash
python equibase_scraper.py --mode chart --track AQU --date 2025-05-14 --race 3
```

---

## All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | required | `entries`, `results`, `horse`, `chart` |
| `--date` | today | `YYYY-MM-DD` |
| `--track` | all tracks | 3-letter track code, e.g. `AQU`, `CD`, `SA` |
| `--race` | `1` | Race number (chart mode only) |
| `--name` | — | Horse name (horse mode only) |
| `--out` | `./output` | Output directory |
| `--format` | `csv` | `csv`, `json`, or `both` |
| `--delay` | `2.0` | Seconds between requests (be polite!) |

---

## Output

Files land in `./output/` by default:

```
output/
  entries_2025-05-15_AQU.csv
  results_2025-05-14_CD.json
  horse_Justify.csv
  chart_AQU_2025-05-14_R3.csv
```

---

## Extending It

The `EquibaseScraper` class is importable:

```python
from equibase_scraper import EquibaseScraper, EquibaseSession
from datetime import date

scraper = EquibaseScraper()

# Get today's entries at Aqueduct
entries = scraper.get_entries(date.today(), track="AQU")
for e in entries:
    print(e.horse_name, e.jockey, e.morning_line_odds)

# Get yesterday's results
from datetime import timedelta
results = scraper.get_results(date.today() - timedelta(days=1))
```

---

## Notes

- **Rate limiting**: defaults to 2-second delay + random jitter. Don't lower this aggressively.
- **Anti-bot**: `cloudscraper` mimics a real browser for Cloudflare challenges. If you get 403s, try upgrading: `pip install -U cloudscraper`.
- **HTML changes**: Equibase occasionally updates their page layout. If parsing returns empty results, inspect the page and update the CSS selectors in the `_parse_*` methods.
- **Legal**: Check Equibase's [Terms of Service](https://www.equibase.com/static/html/termsofuse.html) before scraping at scale.
