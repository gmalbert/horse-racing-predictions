import { MultiSelect } from "../components/FilterSelect";
import { useMemo, useState } from "react";
import { useData, Tabs, Loading, ErrorBox, Metric, MetricGrid, DataTable, Expander, Alert } from "../components/ui";
import { PredictionDatesView } from "../components/predictions";
import type { DayPayload, FixturesPayload, Manifest, ModelInsights, SchedulePayload } from "../lib/types";
import { fmtDate, money, num, pct, todayLocal } from "../lib/format";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const TIER_COLORS: Record<string, string> = {
  "Tier 1: Focus": "#1e7d43",
  "Tier 2: Value": "#b07d10",
  "Tier 3: Avoid": "#b03030",
};

/* ── Tab 1: Upcoming Schedule ─────────────────────────────────────────────── */
function ScheduleTab() {
  const { data, error, reload } = useData<SchedulePayload>("schedule_upcoming.json");
  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;
  if (!data.fixtures?.length) return <Alert>No upcoming fixtures found. The fixture list may need refreshing.</Alert>;
  return (
    <>
      <MetricGrid>
        <Metric label="Total Fixtures" value={data.total_upcoming} />
        <Metric label="Courses" value={data.courses} />
        <Metric label="Turf Fixtures" value={data.turf_count} />
        <Metric label="Calendar Span" value={`${data.span_days} days`} />
      </MetricGrid>
      <DataTable
        maxHeight={640}
        rows={data.fixtures as unknown as Array<Record<string, unknown>>}
        columns={[
          { key: "Date", label: "Date", render: (r) => fmtDate(r.Date) },
          { key: "Course", label: "Course" },
          { key: "Time", label: "Time" },
          { key: "Type", label: "Type" },
          { key: "Surface", label: "Surface" },
        ]}
      />
    </>
  );
}

/* ── Tab 2: Today & Tomorrow ──────────────────────────────────────────────── */
function TodayTomorrowTab({ manifest }: { manifest: Manifest }) {
  return <PredictionDatesView dates={manifest.gb_prediction_dates ?? []} variant="gb" />;
}

/* ── Tab 3: Predicted Fixtures ────────────────────────────────────────────── */
function FixturesTab() {
  const { data, error, reload } = useData<FixturesPayload>("fixtures_scored.json");
  const [tierFilter, setTierFilter] = useState<string[]>([]);
  const [courseFilter, setCourseFilter] = useState("");
  const [minScore, setMinScore] = useState(0);

  const histData = useMemo(() => {
    if (!data?.histogram) return [];
    const { bin_edges, series } = data.histogram;
    return bin_edges.slice(0, -1).map((edge, i) => {
      const row: Record<string, string | number> = { bin: `${edge.toFixed(0)}-${bin_edges[i + 1].toFixed(0)}` };
      for (const [tier, counts] of Object.entries(series)) row[tier] = counts[i];
      return row;
    });
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    return (data.fixtures as Array<Record<string, unknown>>).filter((f) => {
      if (tierFilter.length && !tierFilter.includes(String(f.race_tier))) return false;
      if (courseFilter && f.course !== courseFilter) return false;
      if (typeof f.race_score === "number" && f.race_score < minScore) return false;
      return true;
    });
  }, [data, tierFilter, courseFilter, minScore]);

  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;
  if (!data.summary) return <Alert>No scored fixtures available.</Alert>;

  const tiers = data.tiers?.length ? data.tiers : Object.keys(TIER_COLORS);

  return (
    <>
      <MetricGrid>
        <Metric label="Total Fixtures" value={data.summary.total} />
        <Metric label="Tier 1: Focus" value={data.summary.tier1} />
        <Metric label="Tier 2: Value" value={data.summary.tier2} />
        <Metric label="Avg Score" value={num(data.summary.avg_score, 1)} />
      </MetricGrid>

      <Expander title="ℹ️ About Predicted Fixtures">
        <p className="caption">
          Upcoming races scored 0-100 by predicted class quality, prize money, course tier, field size and
          pattern-race status. Tier 1 (score ≥ 70) races historically offer the best betting value.
        </p>
      </Expander>

      <div className="panel-title">Score Distribution</div>
      <div className="chart-box" style={{ height: 300 }}>
        <ResponsiveContainer>
          <BarChart data={histData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" fontSize={11} />
            <YAxis />
            <Tooltip />
            <Legend />
            {tiers.map((tier) => (
              <Bar isAnimationActive={false} key={tier} dataKey={tier} stackId="a" fill={TIER_COLORS[tier] ?? "#6c8dd8"} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="panel-title">🏆 Top Courses by Score</div>
      <DataTable
        rows={data.top_courses as unknown as Array<Record<string, unknown>>}
        columns={[
          { key: "course", label: "Course" },
          { key: "total_races", label: "Races", numeric: true },
          { key: "avg_score", label: "Avg Score", numeric: true },
          { key: "max_score", label: "Max Score", numeric: true },
          { key: "tier1_count", label: "Tier 1", numeric: true },
        ]}
      />

      <div className="panel-title">📋 All Upcoming Fixtures</div>
      <div className="filter-bar">
        <MultiSelect label="Tier" options={tiers} values={tierFilter} onChange={setTierFilter} />
        <label className="field">
          <span>Course</span>
          <select value={courseFilter} onChange={(e) => setCourseFilter(e.target.value)}>
            <option value="">All courses</option>
            {data.courses.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Min Score</span>
          <input type="number" min={0} max={100} value={minScore} onChange={(e) => setMinScore(Number(e.target.value) || 0)} />
        </label>
      </div>
      <DataTable
        maxHeight={560}
        rows={filtered}
        columns={[
          { key: "date", label: "Date", render: (r) => fmtDate(r.date) },
          { key: "course", label: "Course" },
          { key: "class", label: "Class" },
          { key: "prize", label: "Prize", numeric: true, render: (r) => money(r.prize) },
          { key: "race_score", label: "Score", numeric: true, render: (r) => num(r.race_score, 1) },
          { key: "race_tier", label: "Tier", render: (r) => <span className={String(r.race_tier).startsWith("Tier 1") ? "badge tier1" : String(r.race_tier).startsWith("Tier 2") ? "badge tier2" : "badge tier3"}>{String(r.race_tier)}</span> },
          { key: "weekday", label: "Day" },
          { key: "surface", label: "Surface" },
        ]}
      />
    </>
  );
}

/* ── Tab 4: Top Predictive Races ──────────────────────────────────────────── */
function TopRacesTab() {
  const { data, error, reload } = useData<FixturesPayload>("fixtures_scored.json");
  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;
  const races = (data.top_tier1_races ?? []) as Array<Record<string, unknown>>;
  if (!races.length) return <Alert>No upcoming Tier 1 races found.</Alert>;
  return (
    <>
      <Alert kind="success">Next {races.length} upcoming Tier 1 Focus races (score ≥ 70), sorted by date.</Alert>
      <DataTable
        rows={races}
        columns={[
          { key: "date", label: "Date", render: (r) => fmtDate(r.date) },
          { key: "course", label: "Course" },
          { key: "class", label: "Class" },
          { key: "prize", label: "Prize", numeric: true, render: (r) => money(r.prize) },
          { key: "race_score", label: "Score", numeric: true, render: (r) => num(r.race_score, 1) },
        ]}
      />
    </>
  );
}

/* ── Tab 5: Model Insights ────────────────────────────────────────────────── */
function ModelInsightsTab() {
  const { data, error, reload } = useData<ModelInsights>("model_insights.json");
  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;

  const md = data.metadata ?? {};
  const cal = data.calibration;
  const diag = data.diagnostics as Record<string, unknown> | undefined;
  const fi = (data.feature_importance ?? []).slice().reverse();

  return (
    <>
      <MetricGrid>
        <Metric label="Model Type" value={String(md.model_type ?? "XGBoost")} />
        <Metric label="Features" value={data.n_features ?? "-"} />
        <Metric label="Trained" value={String(md.trained_date ?? "-").slice(0, 16)} />
      </MetricGrid>

      {cal?.metrics && (
        <>
          <div className="panel-title">🎯 Calibration</div>
          <MetricGrid>
            <Metric label="Brier Score (calibrated)" value={num(cal.metrics.brier_score_calibrated, 4)} />
            <Metric label="Calibration Gain" value={pct(cal.metrics.brier_improvement_pct != null ? cal.metrics.brier_improvement_pct / 100 : null)} />
            <Metric label="Calibration Samples" value={cal.n_calibration_samples ?? "-"} />
            <Metric label="Calibrated" value={String(cal.calibration_date ?? "-").slice(0, 10)} />
          </MetricGrid>
        </>
      )}

      {diag && (
        <>
          <div className="panel-title">📊 Latest Prediction Diagnostics</div>
          <MetricGrid>
            <Metric label="Date" value={String(diag.date ?? "-")} />
            <Metric label="Races" value={String(diag.total_races ?? "-")} />
            <Metric label="Horses" value={String(diag.total_horses ?? "-")} />
            <Metric label="Avg Field Size" value={num(diag.avg_field_size as number, 1)} />
          </MetricGrid>
          <Expander title="Raw diagnostics JSON">
            <pre className="small mono" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(diag, null, 2)}</pre>
          </Expander>
        </>
      )}

      {fi.length > 0 && (
        <>
          <div className="panel-title">🔬 Feature Importance (Top 15)</div>
          <div className="chart-box" style={{ height: 460 }}>
            <ResponsiveContainer>
              <BarChart data={fi} layout="vertical" margin={{ left: 8, right: 16 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={130} fontSize={12} />
                <Tooltip formatter={(v) => pct(v as number)} />
                <Bar isAnimationActive={false} dataKey="importance" fill="#3c5a99" radius={[0, 4, 4, 0]}>
                  {fi.map((_, i) => (
                    <Cell key={i} fill={`hsl(${222 + i * 6}, 45%, ${62 - i * 1.6}%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </>
  );
}

/* ── Page ─────────────────────────────────────────────────────────────────── */
export default function InternationalPage() {
  const [tab, setTab] = useState("today");
  const { data: manifest, error, reload } = useData<Manifest>("manifest.json");

  return (
    <div>
      <div className="page-header">
        <h1>International Racing</h1>
        <div className="subtitle">UK & Ireland predictions, fixtures and model insights</div>
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
