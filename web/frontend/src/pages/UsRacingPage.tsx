import { useMemo, useState } from "react";
import { useData, Tabs, Loading, ErrorBox, Metric, MetricGrid, DataTable, Expander, Alert } from "../components/ui";
import { PredictionDatesView } from "../components/predictions";
import type { DayPayload, Manifest, UsTopRaces } from "../lib/types";
import { addDays, fmtDate, money, num, pctRaw, todayLocal } from "../lib/format";

interface UsSchedule {
  days: Array<{
    date: string; tracks: number; races: number; entries: number; scratches: number;
    tracks_table: Array<Record<string, unknown>>;
    race_times: Array<Record<string, unknown>>;
  }>;
}

interface UsStakes {
  total: number; upcoming: number; g1_remaining: number; g2g3_remaining: number;
  stakes: Array<Record<string, unknown>>;
}

interface UsModelInsights {
  us_model_exists: boolean;
  us_metadata?: Record<string, unknown>;
  us_training_rows?: number;
  us_training_file?: string;
  accuracy?: {
    days_tracked: number; races_evaluated: number;
    top1_accuracy: number | null; top3_accuracy: number | null;
    per_track: Array<Record<string, unknown>>;
    recent: Array<Record<string, unknown>>;
  };
}

/* ── Tab 1: Upcoming Schedule ─────────────────────────────────────────────── */
function ScheduleTab() {
  const { data, error, reload } = useData<UsSchedule>("us_schedule.json");
  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;
  if (!data.days?.length) return <Alert>No US entries are available for the coming days.</Alert>;
  return (
    <>
      {data.days.map((d) => (
        <Expander key={d.date} title={`📅 ${fmtDate(d.date)}`} defaultOpen>
          <MetricGrid>
            <Metric label="Tracks" value={d.tracks} />
            <Metric label="Races" value={d.races} />
            <Metric label="Entries" value={d.entries} />
            <Metric label="Scratches" value={d.scratches} />
          </MetricGrid>
          {d.tracks_table.length > 0 && (
            <DataTable
              rows={d.tracks_table}
              columns={[
                { key: "track_code", label: "Code" },
                { key: "track_name", label: "Track" },
                { key: "races", label: "Races", numeric: true },
                { key: "entries", label: "Entries", numeric: true },
              ]}
            />
          )}
          {d.race_times.length > 0 && (
            <Expander title="Race times by track">
              <DataTable
                rows={d.race_times}
                columns={[
                  { key: "track_name", label: "Track" },
                  { key: "race_number", label: "Race #", numeric: true },
                  { key: "race_time", label: "Post Time" },
                  { key: "race_name", label: "Name" },
                  { key: "surface", label: "Surface" },
                  { key: "distance", label: "Distance" },
                ]}
              />
            </Expander>
          )}
        </Expander>
      ))}
    </>
  );
}

/* ── Tab 2: Today & Tomorrow ──────────────────────────────────────────────── */
function TodayTomorrowTab({ manifest }: { manifest: Manifest }) {
  return <PredictionDatesView dates={manifest.us_prediction_dates ?? []} variant="us" />;
}

/* ── Tab 3: Predicted Fixtures (graded stakes) ────────────────────────────── */
function FixturesTab() {
  const { data, error, reload } = useData<UsStakes>("us_stakes.json");
  const [grade, setGrade] = useState("");
  const [window_, setWindow_] = useState("all");
  const today = todayLocal();

  const filtered = useMemo(() => {
    if (!data?.stakes) return [];
    return data.stakes.filter((s) => {
      if (grade && String(s.grade) !== grade) return false;
      const d = String(s.date ?? "").slice(0, 10);
      if (window_ === "30" && (d < today || d > addDays(today, 30))) return false;
      if (window_ === "90" && (d < today || d > addDays(today, 90))) return false;
      if (window_ === "upcoming" && d < today) return false;
      return true;
    });
  }, [data, grade, window_, today]);

  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;
  if (!data.stakes?.length) return <Alert>No graded stakes calendar is available yet.</Alert>;
  const grades = [...new Set(data.stakes.map((s) => String(s.grade ?? "")).filter(Boolean))].sort();

  return (
    <>
      <MetricGrid>
        <Metric label="Total Stakes" value={data.total} />
        <Metric label="Upcoming" value={data.upcoming} />
        <Metric label="G1 Remaining" value={data.g1_remaining} />
        <Metric label="G2/G3 Remaining" value={data.g2g3_remaining} />
      </MetricGrid>
      <div className="filter-bar">
        <label className="field">
          <span>Grade</span>
          <select value={grade} onChange={(e) => setGrade(e.target.value)}>
            <option value="">All grades</option>
            {grades.map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Window</span>
          <select value={window_} onChange={(e) => setWindow_(e.target.value)}>
            <option value="all">Full year</option>
            <option value="upcoming">All upcoming</option>
            <option value="30">Next 30 days</option>
            <option value="90">Next 90 days</option>
          </select>
        </label>
      </div>
      <DataTable
        maxHeight={560}
        rows={filtered}
        columns={[
          { key: "date", label: "Date", render: (r) => fmtDate(r.date) },
          { key: "race_name", label: "Race" },
          { key: "track", label: "Track" },
          { key: "grade", label: "Grade", render: (r) => <span className={`badge ${String(r.grade).includes("1") ? "tier1" : String(r.grade).includes("2") ? "tier2" : "info"}`}>{String(r.grade)}</span> },
          { key: "purse", label: "Purse", numeric: true, render: (r) => money(r.purse, "$") },
          { key: "distance", label: "Distance" },
          { key: "surface", label: "Surface" },
        ]}
      />
    </>
  );
}

/* ── Tab 4: Top Predictive Races ──────────────────────────────────────────── */
function TopRacesTab() {
  const { data, error, reload } = useData<UsTopRaces>("us_top_races.json");
  const [gradeFilter, setGradeFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [surfaceFilter, setSurfaceFilter] = useState("all");
  const [topN, setTopN] = useState(20);

  const filtered = useMemo(() => {
    if (!data?.races) return [];
    return data.races.filter((r) => {
      if (gradeFilter === "g1" && r.grade !== "G1") return false;
      if (gradeFilter === "g1g2" && !["G1", "G2"].includes(String(r.grade))) return false;
      if (gradeFilter === "graded" && r.grade === "-") return false;
      if (typeFilter !== "all" && r.race_type !== typeFilter) return false;
      if (surfaceFilter !== "all" && r.surface !== surfaceFilter) return false;
      return true;
    }).slice(0, topN);
  }, [data, gradeFilter, typeFilter, surfaceFilter, topN]);

  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;
  if (!data.races?.length) return <Alert>No US predictions available to rank yet.</Alert>;

  return (
    <>
      <p className="caption">
        Races ranked by a precomputed predictability score: model confidence in the winner, separation from
        the field, field size, track prestige, grade and purse.
      </p>
      {data.summary && (
        <MetricGrid>
          <Metric label="Races Ranked" value={data.summary.races_ranked} />
          <Metric label="Graded Stakes" value={data.summary.graded} />
          <Metric label="Tracks" value={data.summary.tracks} />
          <Metric label="Avg Top Win %" value={pctRaw(data.summary.avg_top_win_pct)} />
          <Metric label="Avg Separation" value={pctRaw(data.summary.avg_margin_pct)} />
        </MetricGrid>
      )}
      <div className="filter-bar">
        <label className="field"><span>Grade</span>
          <select value={gradeFilter} onChange={(e) => setGradeFilter(e.target.value)}>
            <option value="all">All</option><option value="g1">G1 only</option>
            <option value="g1g2">G1 + G2</option><option value="graded">Graded only</option>
          </select>
        </label>
        <label className="field"><span>Race type</span>
          <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
            <option value="all">All</option><option value="Thoroughbred">Thoroughbred</option>
            <option value="Harness">Harness</option><option value="QuarterHorse">Quarter Horse</option>
          </select>
        </label>
        <label className="field"><span>Surface</span>
          <select value={surfaceFilter} onChange={(e) => setSurfaceFilter(e.target.value)}>
            <option value="all">All</option><option value="Dirt">Dirt</option>
            <option value="Turf">Turf</option><option value="Synthetic">Synthetic</option>
          </select>
        </label>
        <label className="field"><span>Top N</span>
          <select value={topN} onChange={(e) => setTopN(Number(e.target.value))}>
            <option value={10}>10</option><option value={20}>20</option><option value={50}>50</option>
          </select>
        </label>
      </div>
      <DataTable
        maxHeight={620}
        rows={filtered}
        columns={[
          { key: "rank", label: "#", numeric: true },
          { key: "date", label: "Date", render: (r) => fmtDate(r.date) },
          { key: "course", label: "Track" },
          { key: "race_time", label: "Time" },
          { key: "race_name", label: "Race" },
          { key: "race_type", label: "Type" },
          { key: "grade", label: "Grade" },
          { key: "surface", label: "Surface" },
          { key: "distance", label: "Dist" },
          { key: "runners", label: "Runners", numeric: true },
          { key: "purse", label: "Purse", numeric: true, render: (r) => money(r.purse, "$") },
          { key: "top_horse", label: "Top Horse" },
          { key: "top_odds", label: "Odds" },
          { key: "top_win_pct", label: "Top Win %", numeric: true, render: (r) => pctRaw(r.top_win_pct) },
          { key: "margin_pct", label: "Margin", numeric: true, render: (r) => pctRaw(r.margin_pct) },
          { key: "score", label: "Score", numeric: true, render: (r) => num(r.score, 1) },
          { key: "confidence", label: "Confidence", render: (r) => <span className={`badge ${r.confidence === "High" ? "tier1" : r.confidence === "Medium" ? "tier2" : "tier3"}`}>{String(r.confidence)}</span> },
        ]}
      />
      <Expander title="📐 Scoring methodology">
        <pre className="small mono" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(data.scoring_methodology, null, 2)}</pre>
      </Expander>
    </>
  );
}

/* ── Tab 5: Model Insights ────────────────────────────────────────────────── */
function ModelInsightsTab() {
  const { data, error, reload } = useData<UsModelInsights>("us_model_insights.json");
  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;

  const acc = data.accuracy;
  const md = data.us_metadata ?? {};

  return (
    <>
      <Alert>US predictions are currently generated by the UK-trained model; a dedicated US model is in training.</Alert>
      <div className="panel-title">🇺🇸 US Model Readiness</div>
      <MetricGrid>
        <Metric label="US Model File" value={data.us_model_exists ? "✅ Present" : "⏳ Not trained yet"} />
        <Metric label="US Training Rows" value={data.us_training_rows?.toLocaleString() ?? "-"} delta={data.us_training_file} />
        <Metric label="Target" value="50,000" />
      </MetricGrid>
      {Object.keys(md).length > 0 && (
        <Expander title="US model metadata">
          <pre className="small mono" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(md, null, 2)}</pre>
        </Expander>
      )}

      {acc && (
        <>
          <div className="panel-title">🎯 Historical Accuracy (last 30 days tracked)</div>
          <MetricGrid>
            <Metric label="Days Tracked" value={acc.days_tracked} />
            <Metric label="Races Evaluated" value={acc.races_evaluated} />
            <Metric label="Top-1 Accuracy" value={acc.top1_accuracy != null ? pctRaw(acc.top1_accuracy * 100) : "-"} />
            <Metric label="Top-3 Accuracy" value={acc.top3_accuracy != null ? pctRaw(acc.top3_accuracy * 100) : "-"} />
          </MetricGrid>
          {acc.per_track.length > 0 && (
            <Expander title="Per-track Top-1 accuracy">
              <DataTable
                rows={acc.per_track}
                columns={[
                  { key: "course", label: "Track" },
                  { key: "races", label: "Races", numeric: true },
                  { key: "top1", label: "Top-1", numeric: true, render: (r) => pctRaw((r.top1 as number) * 100) },
                  { key: "top3", label: "Top-3", numeric: true, render: (r) => pctRaw((r.top3 as number) * 100) },
                ]}
              />
            </Expander>
          )}
        </>
      )}
    </>
  );
}

/* ── Page ─────────────────────────────────────────────────────────────────── */
export default function UsRacingPage() {
  const [tab, setTab] = useState("today");
  const { data: manifest, error, reload } = useData<Manifest>("manifest.json");

  return (
    <div>
      <div className="page-header">
        <h1>US Racing</h1>
        <div className="subtitle">US predictions, graded stakes calendar and track analytics</div>
      </div>
      <Tabs
        active={tab}
        onChange={setTab}
        tabs={[
          { id: "today", label: "🎲 Today & Tomorrow" },
          { id: "schedule", label: "📅 Upcoming Schedule" },
          { id: "fixtures", label: "📅 Predicted Fixtures" },
          { id: "top", label: "🎯 Top Predictive Races" },
          { id: "model", label: "📊 Model Insights" },
        ]}
      />
      {tab === "schedule" && <ScheduleTab />}
      {tab === "today" && (error ? <ErrorBox error={error} onRetry={reload} /> : manifest ? <TodayTomorrowTab manifest={manifest} /> : <Loading />)}
      {tab === "fixtures" && <FixturesTab />}
      {tab === "top" && <TopRacesTab />}
      {tab === "model" && <ModelInsightsTab />}
    </div>
  );
}
