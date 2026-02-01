# Free Data Sources for UK Horse Racing

All sources listed are **free** (some with registration). No paid APIs or subscriptions required.

---

## 1. Betfair Historical Data (HIGHLY RECOMMENDED)

### What It Provides
- **Betfair Starting Price (BSP)** — market-determined fair odds
- **Betfair exchange prices** — pre-race price movements
- **Implied probabilities** — what the market thinks

### Why It's Critical
- BSP is the single best predictor of race outcomes
- Market wisdom aggregates all public information
- Can identify when your model disagrees with market (value bets)

### How to Access
```
URL: https://historicdata.betfair.com/
Cost: Free for basic data (registration required)
Format: CSV files by date
Coverage: UK & Ireland racing from 2016+
```

### Data Available
| Field | Description | Use Case |
|-------|-------------|----------|
| `BSP` | Betfair Starting Price | Primary odds benchmark |
| `BSP_Liability` | Amount matched at BSP | Market confidence |
| `Place_BSP` | BSP for place markets | Place probability |
| `IP_Max` | Max in-play price | Did horse drift/shorten? |

### Integration Steps
1. Register at historicdata.betfair.com
2. Download historical BSP data for UK horse racing
3. Match by date + course + race time + horse name
4. Create features: `market_prob`, `price_movement`, `bsp_rank`

---

## 2. Racing Post (Web Scraping)

### What It Provides
- **RPR (Racing Post Rating)** — independent of BHA ratings
- **Topspeed figures** — speed ratings
- **Going preferences** — symbols for each ground type
- **Sectional times** (some races)

### Legal Considerations
- Scraping for personal use is generally acceptable
- Do NOT scrape at scale or redistribute data
- Respect robots.txt and rate limits

### Sample Data Points
```
Horse: Galileo Gold
RPR: 118
Topspeed: 112
Going Prefs: G/GF (Good to Firm preferred)
Distance Prefs: 6f-1m (best at 7f)
C&D Winner: ✓
```

### Scraping Approach
```python
# Example (for personal use only)
import requests
from bs4 import BeautifulSoup

def get_rp_rating(horse_name, date):
    """Scrape RPR for a horse on given date."""
    # Note: Implement with proper rate limiting
    # and error handling
    pass
```

### Implementation Priority: 🟠 HIGH

---

## 3. Timeform (Limited Free Access)

### What It Provides
- Timeform ratings (very respected)
- Flags: "Horse in Form", "Big Run Expected", "Trainer Form"
- Pace predictions

### Free Access
```
URL: https://www.timeform.com/
Free: Limited race previews, some ratings visible
Registration: Email sign-up for more content
```

### Note
- Full Timeform requires subscription
- But race previews contain useful qualitative data
- Can scrape "Timeform Tips" sections

---

## 4. At The Races (ATR) Free Data

### What It Provides
- Racecard with trainer/jockey stats
- Recent form analysis
- Spotlight comments

### Access
```
URL: https://www.attheraces.com/
Cost: Free (registration optional)
Format: Web pages per race
```

### Useful Free Elements
- 14-day trainer form percentage
- Jockey/trainer combination stats
- Course form summaries

---

## 5. Sky Sports Racing / Racing TV

### What It Provides  
- Free racecards online
- Some analyst comments
- Trainer quotes

### Access
```
URL: https://www.skysports.com/racing
URL: https://www.racingtv.com/
Cost: Free for basic info
```

---

## 6. Met Office Weather Data

### What It Provides
- Historical weather by location
- Forecasts for race day going prediction

### Why Important
- Ground conditions affect races dramatically
- Can predict going changes 24-48 hours ahead
- Soft ground specialists, firm ground specialists

### API Access
```
URL: https://www.metoffice.gov.uk/services/data/datapoint
Cost: Free tier: 3,000 requests/day
Format: JSON/XML
```

### Features to Create
| Feature | Source | Impact |
|---------|--------|--------|
| `rain_24h` | Met Office | High |
| `going_forecast` | Derived | High |
| `temp_vs_seasonal` | Met Office | Low |

---

## 7. BHA (British Horseracing Authority)

### What It Provides
- Official race results (free)
- Horse/trainer/jockey lookup
- Handicap ratings (OR)

### Access
```
URL: https://www.britishhorseracing.com/
Format: Web pages, some API access
Cost: Free
```

---

## 8. Equibase (for US Racing if Expanding)

### What It Provides
- US racing results and entries
- Speed figures (Beyer)
- Pace ratings

### Access
```
URL: https://www.equibase.com/
Cost: Free basic data
```

---

## 9. Kaggle Datasets

### Relevant Datasets
```
https://www.kaggle.com/datasets - search "horse racing"
```

### Available Free Datasets
- UK/Irish racing results 2000-2020
- Betfair historical odds
- Hong Kong racing data
- Feature-engineered datasets

### Use Case
- Additional training data
- Benchmarking your model
- Cross-validation with external data

---

## 10. GitHub Open Source Projects

### Racing Data Projects
```
github.com/search?q=horse+racing+data
```

### Notable Repos
- Various scraping tools for Racing Post
- Betfair API wrappers
- Historical result datasets

---

## Integration Priority

| Source | Effort | Data Value | Priority |
|--------|--------|------------|----------|
| Betfair Historical | Low | Very High | 🔴 Week 1 |
| Met Office Weather | Low | High | 🔴 Week 1 |
| Racing Post (scrape) | Medium | Very High | 🟠 Week 2-3 |
| At The Races | Medium | Medium | 🟡 Week 4+ |
| Timeform | Medium | High | 🟡 Week 4+ |
| Kaggle Datasets | Low | Medium | 🟢 Anytime |

---

## Data Joining Strategy

### Key Matching Fields
```python
# Primary key for matching external data
match_key = {
    'date': '2026-01-30',
    'course': 'Kempton',
    'race_time': '14:00',
    'horse_name': 'Example Horse'
}
```

### Fuzzy Matching Needed
- Horse names may vary slightly across sources
- Course names need normalization
- Time zones must be handled consistently

### Recommended Approach
1. Create master horse/course lookup tables
2. Use fuzzy matching (fuzzywuzzy) for initial joins
3. Build confidence scores for matches
4. Manual review of low-confidence matches
