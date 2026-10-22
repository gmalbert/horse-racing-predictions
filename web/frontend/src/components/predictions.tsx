import { useMemo, useState } from "react";
import type { DayPayload, HorseRow, RaceDetail } from "../lib/types";
import { addDays, fmtDate, num, pct, signedPct } from "../lib/format";
import { Alert, DataTable, Expander, Metric, MetricGrid, useData, Loading, ErrorBox } from "./ui";
import { todayLocal } from "../lib/format";
import { combineDays } from "../lib/predictions";

/* ── Value verdict (same thresholds as the Streamlit app) ─────────────────── */
export function valueVerdict(edge: number): { label: string; kind: "success" | "warning" | "info" | "error" } {
  if (edge >= 0.05) return { label: "🎯 VALUE BET - model edge 5%+", kind: "success" };
  if (edge >= 0.02) return { label: "⚠️ MARGINAL - small edge, proceed with caution", kind: "warning" };
  if (edge >= -0.02) return { label: "⚖️ FAIR ODDS - no significant edge", kind: "info" };
  return { label: "❌ NO VALUE - bookmaker odds shorter than model", kind: "error" };
}

/* ── Best bet ──────────────────────────────────────────────────────────────── */
export function BestBetCard({ day }: { day: DayPayload }) {
  if (!day.best_bet) return null;
  const bb = day.best_bet;
  return (
    <div className="panel">
      <div className="panel-title">⭐ Best Bet of the Day</div>
      <div className="card" style={{ borderLeft: "4px solid var(--success)" }}>
        <div style={{ fontSize: 18, fontWeight: 800 }}>
          {bb.horse} <span className="caption">({bb.course} {bb.race_time})</span>
        </div>
        <MetricGrid>
          <Metric label="Win Probability" value={pct(bb.win_probability)} />
          <Metric label="Market Odds" value={num(bb.market_odds)} />
          <Metric label="Edge" value={signedPct(bb.edge)} />
        </MetricGrid>
      </div>
    </div>
  );
}

/* ── Value bets table ──────────────────────────────────────────────────────── */
export function ValueBetsTable({ day }: { day: DayPayload }) {
  if (!day.value_bets?.length) return null;
  return (
    <Expander title={`💰 Value Bets (${day.value_bets.length}) - sorted by model edge`}>
      <DataTable
        rows={day.value_bets as Array<Record<string, unknown>>}
        columns={[
          { key: "date", label: "Date", render: (r) => fmtDate(r.date) },
          { key: "race_time", label: "Time" },
          { key: "course", label: "Course" },
          { key: "horse", label: "Horse" },
          { key: "win_probability", label: "Model Win %", numeric: true, render: (r) => pct(r.win_probability) },
          { key: "market_odds", label: "Mkt Odds", numeric: true, render: (r) => num(r.market_odds) },
          { key: "implied_prob", label: "Mkt Implied %", numeric: true, render: (r) => pct(r.implied_prob) },
          { key: "edge", label: "Edge", numeric: true, render: (r) => signedPct(r.edge) },
          { key: "ev_per_unit", label: "EV/£1", numeric: true, render: (r) => num(r.ev_per_unit) },
          { key: "best_bookmaker", label: "Bookie" },
          { key: "odds_trend", label: "Trend" },
        ]}
      />
    </Expander>
  );
}

/* ── Top 25 ────────────────────────────────────────────────────────────────── */
export function Top25Table({ day }: { day: DayPayload }) {
  if (!day.top25?.length) return null;
  return (
    <div className="panel">
      <div className="panel-title">🏆 Top 25 Predictions Per Day</div>
      <DataTable
        maxHeight={560}
        rows={day.top25 as Array<Record<string, unknown>>}
        columns={[
          { key: "date", label: "Date", render: (r) => fmtDate(r.date) },
          { key: "race_time", label: "Time" },
          { key: "course", label: "Course" },
          { key: "horse", label: "Horse" },
          { key: "jockey", label: "Jockey" },
          { key: "win_probability", label: "Win %", numeric: true, render: (r) => pct(r.win_probability) },
          { key: "win_odds_fractional", label: "Win Odds" },
          { key: "place_probability", label: "Place %", numeric: true, render: (r) => pct(r.place_probability) },
          { key: "show_probability", label: "Show %", numeric: true, render: (r) => pct(r.show_probability) },
          { key: "market_odds", label: "Mkt Odds", numeric: true, render: (r) => num(r.market_odds) },
          { key: "edge", label: "Edge", numeric: true, render: (r) => signedPct(r.edge) },
          { key: "race_class", label: "Class" },
          { key: "distance_band", label: "Distance" },
          { key: "ofr", label: "OR", numeric: true },
        ]}
      />
    </div>
  );
}

/* ── Race detail ───────────────────────────────────────────────────────────── */
export function RaceDetailView({ race, variant }: { race: RaceDetail; variant: "gb" | "us" }) {
  const fav = race.favorites.favorite_cumulative;
  const hasMarket = race.horses.some((h) => h.market_odds != null);
  const hasBetfair = race.horses.some((h) => h.betfair_back_odds != null);
  const hasBreeding = race.horses.some((h) => h.sire || h.dam);

  const columns = [
    { key: "horse", label: "Horse" },
    { key: "jockey", label: "Jockey" },
    ...(variant === "us" ? [{ key: "trainer", label: "Trainer" }] : []),
    { key: "win_rank", label: "Win Rank", numeric: true },
    { key: "win_probability", label: "Win %", numeric: true, render: (r: HorseRow) => pct(r.win_probability) },
    { key: "win_odds_fractional", label: "Win Odds" },
    ...(variant === "us" ? [{ key: "ml_odds", label: "ML Odds" }] : []),
    { key: "place_rank", label: "Place Rank", numeric: true },
    { key: "place_probability", label: "Place %", numeric: true, render: (r: HorseRow) => pct(r.place_probability) },
    { key: "show_rank", label: "Show Rank", numeric: true },
    { key: "show_probability", label: "Show %", numeric: true, render: (r: HorseRow) => pct(r.show_probability) },
    { key: "top_2_prob", label: "Top 2 %", numeric: true, render: (r: HorseRow) => pct(r.top_2_prob) },
    { key: "top_3_prob", label: "Top 3 %", numeric: true, render: (r: HorseRow) => pct(r.top_3_prob) },
    ...(hasMarket ? [
      { key: "market_odds", label: "Mkt Odds", numeric: true, render: (r: HorseRow) => num(r.market_odds) },
      { key: "best_bookmaker", label: "Bookie" },
      { key: "edge", label: "Edge", numeric: true, render: (r: HorseRow) => signedPct(r.edge) },
      { key: "odds_trend", label: "Trend" },
      { key: "is_value_bet", label: "Value", render: (r: HorseRow) => (r.is_value_bet ? <span className="badge value">Value</span> : "") },
    ] : []),
    ...(hasBetfair ? [
      { key: "betfair_back_odds", label: "Betfair Back", numeric: true, render: (r: HorseRow) => num(r.betfair_back_odds) },
      { key: "betfair_value_edge", label: "BF Edge", numeric: true, render: (r: HorseRow) => signedPct(r.betfair_value_edge) },
    ] : []),
    { key: "age", label: "Age", numeric: true },
    { key: "weight_lbs", label: "Weight", numeric: true },
    { key: "ofr", label: "OR", numeric: true },
    { key: "form", label: "Recent Form" },
    ...(hasBreeding ? [{ key: "sire", label: "Sire" }, { key: "dam", label: "Dam" }] : []),
  ];

  return (
    <div className="panel">
      <div className="panel-title">
        🏁 {race.course} {race.race_time} {race.race_name ? `- ${race.race_name}` : ""}
      </div>
      <MetricGrid>
        {race.date && <Metric label="Date" value={fmtDate(race.date)} />}
        <Metric label="Runners" value={race.runners} />
        {race.race_class && <Metric label="Class" value={race.race_class} />}
        {race.distance_f != null && <Metric label="Distance" value={`${race.distance_f}f`} />}
        {race.distance_str && <Metric label="Distance" value={race.distance_str} />}
        {race.going && <Metric label="Going" value={race.going} />}
        {race.surface && <Metric label="Surface" value={race.surface} />}
      </MetricGrid>

      <div className="panel-title">🎯 Top Picks</div>
      <MetricGrid>
        <Metric label="Most Likely to WIN" value={race.favorites.win.horse} delta={pct(race.favorites.win.prob)} />
        <Metric label="Most Likely to PLACE" value={race.favorites.place.horse} delta={pct(race.favorites.place.prob)} />
        <Metric label="Most Likely to SHOW" value={race.favorites.show.horse} delta={pct(race.favorites.show.prob)} />
      </MetricGrid>

      {race.exacta_prob != null && (
        <>
          <div className="panel-title">🎲 Exacta / Trifecta Probabilities</div>
          <MetricGrid>
            <Metric label="Exacta (1-2 in order)" value={pct(race.exacta_prob)} delta={`${race.predicted_123[0]} → ${race.predicted_123[1]}`} />
            <Metric label="Trifecta (1-2-3 in order)" value={pct(race.trifecta_prob)} delta={race.predicted_123.join(" → ")} />
            <Metric label="Fair Trifecta Odds" value={race.fair_trifecta_odds != null ? `${race.fair_trifecta_odds.toFixed(1)}/1` : "N/A"} />
          </MetricGrid>
        </>
      )}

      <div className="panel-title">🏆 {fav.horse} - Finish in the Money</div>
      <MetricGrid>
        <Metric label="Win (1st)" value={pct(fav.win)} />
        <Metric label="Win or Place" value={pct(fav.top_2)} delta={`(+${pct(fav.top_2_increment)})`} />
        <Metric label="Win, Place or Show" value={pct(fav.top_3)} delta={`(+${pct(fav.top_3_increment)})`} />
      </MetricGrid>

      <div className="panel-title">🐎 All Horse Predictions</div>
      <p className="caption">
        Form reads right-to-left (rightmost = most recent). Model odds are fair value from model probabilities.
      </p>
      <DataTable columns={columns} rows={race.horses} rowKey={(r) => String(r.horse)} />
    </div>
  );
}

/* ── Value bet calculator ──────────────────────────────────────────────────── */
export function ValueBetCalculator({ race }: { race: RaceDetail }) {
  const firstWithMarket = race.horses.find((h) => h.market_odds != null);
  const [horseName, setHorseName] = useState(firstWithMarket?.horse ?? race.horses[0]?.horse ?? "");
  const [odds, setOdds] = useState<number>(firstWithMarket?.market_odds ?? 3.0);

  const horse = race.horses.find((h) => h.horse === horseName);
  if (!horse) return null;

  const bookmakerProb = 1 / odds;
  const edge = horse.win_probability - bookmakerProb;
  const verdict = valueVerdict(edge);

  return (
    <Expander title="🧮 Value Bet Calculator">
      <div className="filter-bar">
        <label className="field">
          <span>Horse</span>
          <select value={horseName} onChange={(e) => {
            setHorseName(e.target.value);
            const h = race.horses.find((x) => x.horse === e.target.value);
            if (h?.market_odds != null) setOdds(h.market_odds);
          }}>
            {race.horses.map((h) => <option key={h.horse} value={h.horse}>{h.horse}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Bookmaker Decimal Odds</span>
          <input type="number" min={1.01} step={0.1} value={odds} onChange={(e) => setOdds(Number(e.target.value) || 1.01)} />
        </label>
      </div>
      <MetricGrid>
        <Metric label="Model Win Probability" value={pct(horse.win_probability)} />
        <Metric label="Bookmaker Implied" value={pct(bookmakerProb)} />
        <Metric label="Model Edge" value={signedPct(edge)} />
      </MetricGrid>
      <Alert kind={verdict.kind}>{verdict.label}</Alert>
    </Expander>
  );
}

/* ── Whole "Today & Tomorrow" tab ──────────────────────────────────────────── */
export function PredictionsDayView({ day: sourceDay, variant }: { day: DayPayload; variant: "gb" | "us" }) {
  const [filters, setFilters] = useState<Record<string, string>>({});
  const distanceBand = (race: RaceDetail) => {
    const d = race.distance_f;
    return d == null ? "Unknown" : d < 6 ? "Sprint (<6f)" : d <= 8 ? "Mile (6-8f)" : d <= 12 ? "Middle (8-12f)" : "Staying (>12f)";
  };
  const fields = [
    { key: "course", label: variant === "gb" ? "Course" : "Track" },
    { key: "race_type", label: "Race type" },
    { key: "race_class", label: "Class" },
    { key: "going", label: "Going" },
    { key: "surface", label: "Surface" },
    { key: "distance_band", label: "Distance" },
  ];
  const raceValue = (race: RaceDetail, key: string) => key === "distance_band" ? distanceBand(race) : String(race[key as keyof RaceDetail] ?? "");
  const races = sourceDay.races.filter((r) => fields.every(({ key }) => !filters[key] || raceValue(r, key) === filters[key]));
  const filtered = Object.values(filters).some(Boolean);
  const horses = races.flatMap((r) => r.horses.map((h) => ({ ...h, date: r.date ?? sourceDay.date, course: r.course ?? "", race_time: r.race_time ?? "", race_class: r.race_class, distance_band: distanceBand(r) })));
  const ranked = [...horses].sort((a, b) => b.win_probability - a.win_probability);
  const dailyCounts = new Map<string, number>();
  const topPerDay = ranked.filter((horse) => {
    const count = dailyCounts.get(horse.date) ?? 0;
    dailyCounts.set(horse.date, count + 1);
    return count < 25;
  }).sort((a, b) => a.date.localeCompare(b.date) || b.win_probability - a.win_probability);
  const edged = horses.filter((h) => h.edge != null).sort((a, b) => (b.edge ?? 0) - (a.edge ?? 0));
  const day: DayPayload = !filtered ? sourceDay : {
    ...sourceDay, races,
    summary: { horses: horses.length, races: races.length, courses: new Set(races.map((r) => r.course)).size, has_odds: horses.some((h) => h.market_odds != null) },
    best_bet: edged[0] ?? null,
    top25: topPerDay,
    value_bets: edged.filter((h) => h.is_value_bet === true),
  };
  const raceLabels = useMemo(
    () => day.races.map((r, i) => ({
      i,
      label: `${r.date ?? day.date} ${r.race_time ?? "Time unavailable"} - ${r.course ?? "Course unavailable"}${r.race_name ? ` - ${r.race_name}` : ""} (${r.runners})`,
    })),
    [day]
  );
  const [raceIdx, setRaceIdx] = useState(0);
  const race = day.races[Math.min(raceIdx, day.races.length - 1)];

  return (
    <>
      <div className="filter-bar">
        {fields.map(({ key, label }) => {
          const options = [...new Set(sourceDay.races.map((r) => raceValue(r, key)).filter(Boolean))].sort();
          if (!options.length) return null;
          return <label className="field" key={key}><span>{label}</span>
            <select value={filters[key] ?? ""} onChange={(e) => { setFilters((f) => ({ ...f, [key]: e.target.value })); setRaceIdx(0); }}>
              <option value="">All</option>{options.map((option) => <option key={option}>{option}</option>)}
            </select>
          </label>;
        })}
        <button className="btn secondary" onClick={() => { setFilters({}); setRaceIdx(0); }}>Reset race filters</button>
      </div>
      {!day.races.length && <Alert>No races match these filters.</Alert>}
      <MetricGrid>
        <Metric label="Horses" value={day.summary.horses} />
        <Metric label="Races" value={day.summary.races} />
        <Metric label={variant === "gb" ? "Courses" : "Tracks"} value={day.summary.courses || (day.summary.races ? "—" : 0)} />
        <Metric label="Market Odds" value={day.summary.has_odds ? "✅ Merged" : "—"} />
      </MetricGrid>
      {variant === "us" && day.model_provenance?.source_model && (
        <p className="caption">
          Model: {day.model_provenance.source_model}
          {day.model_provenance.model_feature_count ? ` (${day.model_provenance.model_feature_count} features)` : ""}
        </p>
      )}

      <BestBetCard day={day} />
      <ValueBetsTable day={day} />
      <Top25Table day={day} />

      {day.races.length > 0 && (
        <>
          <div className="panel-title">🔎 Race-by-Race Detail</div>
          <div className="filter-bar">
            <label className="field race-picker">
              <span>Select race</span>
              <select value={raceIdx} onChange={(e) => setRaceIdx(Number(e.target.value))}>
                {raceLabels.map((r) => <option key={r.i} value={r.i}>{r.label}</option>)}
              </select>
            </label>
          </div>
          {race && (
            <>
              <RaceDetailView race={race} variant={variant} />
              <ValueBetCalculator key={`${day.date}-${race.course}-${race.race_time}`} race={race} />
            </>
          )}
        </>
      )}
    </>
  );
}

/** Date pills: Today / Tomorrow / latest-available fallback with stale banner. */
export function DayPicker({
  dates, selected, onSelect, today, combined = false,
}: { dates: string[]; selected: string | null; onSelect: (d: string) => void; today: string; combined?: boolean }) {
  if (!dates.length) return null;
  const labelFor = (d: string) =>
    d === today ? `Today (${d})` : d === addDays(today, 1) ? `Tomorrow (${d})` : d;
  return (
    <div className="filter-bar">
      <label className="field">
        <span>Date</span>
        <select value={selected ?? ""} onChange={(e) => onSelect(e.target.value)}>
          {combined && <option value="combined">Today & Tomorrow (combined)</option>}
          {dates.map((d) => <option key={d} value={d}>{labelFor(d)}</option>)}
        </select>
      </label>
      {selected && selected < today && (
        <p className="caption date-context">Viewing predictions from {selected}.</p>
      )}
    </div>
  );
}

/** Prefer the combined current cards, otherwise explicitly label historical data. */
export function PredictionDatesView({ dates, variant }: { dates: string[]; variant: "gb" | "us" }) {
  const today = todayLocal();
  const tomorrow = addDays(today, 1);
  const combined = dates.includes(today) && dates.includes(tomorrow);
  const [selected, setSelected] = useState<string | null>(combined ? "combined" : dates.includes(today) ? today : dates.at(-1) ?? null);
  const directory = variant === "gb" ? "predictions" : "us_predictions";
  const first = useData<DayPayload>(selected ? `${directory}/${selected === "combined" ? today : selected}.json` : null);
  const second = useData<DayPayload>(selected === "combined" ? `${directory}/${tomorrow}.json` : null);
  const error = first.error ?? second.error;
  const data = first.data && (selected !== "combined" || second.data)
    ? selected === "combined" ? combineDays([first.data, second.data!]) : first.data : null;
  if (!dates.length) return <Alert kind="warning">No predictions are available yet.</Alert>;
  return <>
    <DayPicker dates={dates} selected={selected} onSelect={setSelected} today={today} combined={combined} />
    {error != null && <ErrorBox error={error} onRetry={() => { first.reload(); second.reload(); }} />}
    {!data && !error && <Loading />}
    {data && <PredictionsDayView key={selected} day={data} variant={variant} />}
  </>;
}
