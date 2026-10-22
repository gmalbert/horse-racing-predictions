import type { DayPayload } from "./types";

/** Merge already-computed predictions without changing model probabilities. */
export function combineDays(days: DayPayload[]): DayPayload {
  const races = days.flatMap((d) => d.races.map((race) => ({ ...race, date: race.date ?? d.date })));
  const best = days.flatMap((d) => d.best_bet ? [d.best_bet] : []).sort((a, b) => (b.edge ?? 0) - (a.edge ?? 0));
  return {
    date: days.map((d) => d.date).join(" / "),
    races,
    summary: {
      horses: days.reduce((total, d) => total + d.summary.horses, 0),
      races: races.length,
      courses: new Set(races.map((r) => r.course)).size,
      has_odds: days.some((d) => d.summary.has_odds),
    },
    best_bet: best[0] ?? null,
    top25: days.flatMap((d) => d.top25.map((r): Record<string, unknown> => ({ ...r, date: r.date ?? d.date }))),
    value_bets: days.flatMap((d) => d.value_bets.map((r): Record<string, unknown> => ({ ...r, date: r.date ?? d.date })))
      .sort((a, b) => Number(b.edge) - Number(a.edge)),
    filters: days[0].filters,
    model_provenance: days[0].model_provenance,
  };
}
