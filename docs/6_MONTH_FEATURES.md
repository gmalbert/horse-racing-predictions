# Horse Racing Predictions — 6-Month Feature Roadmap

## Month 1: Daily Racecard

- **Today's racecards page** — Display all UK/IRE races with model win probabilities, top 3 picks, and confidence tier.
- **Race type filter** — Filter by flat/NH, class, distance band, going.
- **Going updates** — Auto-detect "going description" changes from the API and flag races where conditions changed since predictions were made.
- **Best bet of the day** — Single highlighted pick with highest edge among all races.

## Month 2: Value & Odds Tools

- **Tote vs. SP comparison** — Compare model's implied price to Tote and bookmaker odds.
- **Exchange integration** — Betfair/Smarkets exchange price comparison for top picks.
- **Place market analysis** — Show each-way implied values alongside win market.
- **Over-round display** — Show market over-round as a transparency indicator.

## Month 3: Trainer & Jockey Analytics

- **Trainer stats page** — Form last 14 days, course specialists, yard patterns by going.
- **Jockey stats page** — Current season win%, course win%, jockey-trainer partnership table.
- **Course specialist finder** — Search by course for trainer/jockey combinations with strong historical records.

## Month 4: Historical Analysis

- **Backtest results page** — Model accuracy by race class, distance, going, and month.
- **Race replay index** — Link to past race replays alongside model retrospective analysis.
- **Handicap ratings tracker** — Weekly official handicap rating trends for tracked horses.

## Month 5: Bankroll Management

- **Kelly calculator UI** — Enter current bankroll; see recommended stakes for today's best bets.
- **P&L tracker** — Log actual bets and results; track ROI over rolling windows.
- **Drawdown alert** — Warning when bankroll falls 20%+ below peak.

## Month 6: Automation

- **Morning email** — Daily selections email at 7 AM with today's best bets.
- **GitHub Actions** — Nightly model inference run for tomorrow's races.
- **Race result capture** — Auto-fetch final results and SP; feed into CLV and P&L tracking.
