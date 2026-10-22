import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from "recharts";
import type { EntityStatsPayload } from "../lib/types";
import { num, pctRaw } from "../lib/format";
import { Alert, DataTable, Expander, Loading, ErrorBox, useData } from "./ui";

type Row = Record<string, unknown>;

function entityKey(payload: EntityStatsPayload): string {
  return payload.entity; // "trainer" | "jockey"
}

/* ── Leaderboard / recent-form charts ─────────────────────────────────────── */
function FormCharts({ rows, nameKey }: { rows: Row[]; nameKey: string }) {
  const topWinners = rows.slice(0, 15);
  const strikeRates = useMemo(
    () => rows.filter((r) => (r.runs as number) >= 5).sort((a, b) => (b.win_pct as number) - (a.win_pct as number)).slice(0, 15),
    [rows]
  );
  return (
    <div className="grid-2">
      <div className="chart-box" style={{ height: 380 }}>
        <div className="panel-title">Most Winners</div>
        <ResponsiveContainer>
          <BarChart data={topWinners} layout="vertical" margin={{ left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey={nameKey} width={120} fontSize={11} />
            <Tooltip />
            <Bar isAnimationActive={false} dataKey="wins" fill="#3c5a99" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-box" style={{ height: 380 }}>
        <div className="panel-title">Highest Strike Rate (5+ runs)</div>
        <ResponsiveContainer>
          <BarChart data={strikeRates} layout="vertical" margin={{ left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" unit="%" />
            <YAxis type="category" dataKey={nameKey} width={120} fontSize={11} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Bar isAnimationActive={false} dataKey="win_pct" fill="#6c8dd8" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const statCols = (nameKey: string, label: string) => [
  { key: nameKey, label },
  { key: "runs", label: "Runs", numeric: true },
  { key: "wins", label: "Wins", numeric: true },
  { key: "places", label: "Places", numeric: true },
  { key: "win_pct", label: "Win %", numeric: true, render: (r: Row) => pctRaw(r.win_pct) },
  { key: "place_pct", label: "Place %", numeric: true, render: (r: Row) => pctRaw(r.place_pct) },
];

/* ── Shared stats page body ────────────────────────────────────────────────── */
export function StatsPageBody({ payload }: { payload: EntityStatsPayload }) {
  const nameKey = entityKey(payload);
  const otherKey = nameKey === "trainer" ? "jockey" : "trainer";
  const label = nameKey === "trainer" ? "Trainer" : "Jockey";
  const otherLabel = nameKey === "trainer" ? "Jockey" : "Trainer";

  const [tab, setTab] = useState("form");
  const [nameSearch, setNameSearch] = useState("");
  const [minimumRuns, setMinimumRuns] = useState(0);
  const leaderboard = (payload.leaderboard ?? []).filter((row) =>
    String(row[nameKey]).toLowerCase().includes(nameSearch.trim().toLowerCase()) && Number(row.runs) >= minimumRuns
  );
  const [courseFilter, setCourseFilter] = useState("");
  const [entityFilter, setEntityFilter] = useState("");
  const [goingEntity, setGoingEntity] = useState<string>("");

  const specialists = useMemo(() => {
    let rows = payload.course_specialists ?? [];
    if (courseFilter) rows = rows.filter((r) => r.course === courseFilter);
    if (entityFilter) rows = rows.filter((r) => String(r[nameKey]).toLowerCase().includes(entityFilter.toLowerCase()));
    return rows;
  }, [payload, courseFilter, entityFilter, nameKey]);

  const eliteSpecialists = useMemo(
    () => specialists.filter((r) => (r.win_pct as number) >= 25).slice(0, 25),
    [specialists]
  );

  const partnerships = payload.partnerships ?? [];
  const elitePartnerships = useMemo(
    () => partnerships.filter((r) => (r.win_pct as number) >= 25 && (r.runs as number) >= 20).slice(0, 20),
    [partnerships]
  );

  const goingNames = Object.keys(payload.going_patterns ?? {});
  const goingName = goingEntity || goingNames[0] || "";
  const goingRows = payload.going_patterns?.[goingName] ?? [];

  const trends = payload.form_trends ?? {};
  const [trendName, setTrendName] = useState("");
  const trendKey = trendName || Object.keys(trends)[0] || "";
  const trend = trends[trendKey];
  const trendData = trend?.dates.map((d, i) => ({ date: d, win_pct: trend.rolling_win_pct[i] })) ?? [];

  const tabs = [
    { id: "form", label: nameKey === "trainer" ? "📊 Recent Form" : "📊 Current Season" },
    { id: "courses", label: "🏇 Course Records" },
    { id: "partners", label: `🤝 ${otherLabel} Partnerships` },
    ...(nameKey === "trainer"
      ? [{ id: "going", label: "🌦️ Going Patterns" }]
      : [{ id: "trends", label: "📈 Form Trends" }]),
  ];

  return (
    <>
      <p className="caption">
        Results through {payload.as_of} · {payload.window_days}-day window
      </p>
      <div className="tabs">
        {tabs.map((t) => (
          <button key={t.id} aria-pressed={tab === t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {tab === "form" && (
        <>
          <div className="filter-bar stats-search">
            <label className="field"><span>Search {label.toLowerCase()}</span><input type="search" placeholder={`Find a ${label.toLowerCase()}`} value={nameSearch} onChange={(event) => setNameSearch(event.target.value)} /></label>
            <label className="field"><span>Minimum runs</span><input type="number" min={0} step={1} value={minimumRuns} onChange={(event) => setMinimumRuns(Math.max(0, Number(event.target.value)))} /></label>
            <button className="btn secondary" onClick={() => { setNameSearch(""); setMinimumRuns(0); }}>Reset leaderboard</button>
          </div>
          {leaderboard.length > 0 && <FormCharts rows={leaderboard as Row[]} nameKey={nameKey} />}
          <div className="panel-title">{label} Leaderboard</div>
          <p className="caption">{leaderboard.length} of {payload.leaderboard.length} {label.toLowerCase()}s in this leaderboard</p>
          <DataTable maxHeight={520} rows={leaderboard as Row[]} columns={statCols(nameKey, label)} />
          {payload.recent_form_14d?.length > 0 && (
            <>
              <div className="panel-title">🔥 Hot Last 14 Days</div>
              <DataTable maxHeight={420} rows={payload.recent_form_14d as Row[]} columns={statCols(nameKey, label)} />
            </>
          )}
        </>
      )}

      {tab === "courses" && (
        <>
          <div className="filter-bar">
            <label className="field"><span>Course</span>
              <select value={courseFilter} onChange={(e) => setCourseFilter(e.target.value)}>
                <option value="">All courses</option>
                {payload.courses.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="field"><span>{label} contains</span>
              <input value={entityFilter} onChange={(e) => setEntityFilter(e.target.value)} placeholder="Type to filter..." />
            </label>
          </div>
          {eliteSpecialists.length > 0 && (
            <>
              <div className="panel-title">⭐ Top Specialists (25%+ strike rate)</div>
              <DataTable rows={eliteSpecialists} columns={[...statCols(nameKey, label).slice(0, 1), { key: "course", label: "Course" }, ...statCols(nameKey, label).slice(1)]} />
            </>
          )}
          <div className="panel-title">All Course Records (5+ runs)</div>
          <DataTable maxHeight={520} rows={specialists} columns={[...statCols(nameKey, label).slice(0, 1), { key: "course", label: "Course" }, ...statCols(nameKey, label).slice(1)]} />
        </>
      )}

      {tab === "partners" && (
        <>
          {elitePartnerships.length > 0 && (
            <>
              <div className="panel-title">⭐ Elite Partnerships (25%+ wins, 20+ runs)</div>
              <DataTable rows={elitePartnerships} columns={[
                { key: nameKey, label }, { key: otherKey, label: otherLabel },
                { key: "runs", label: "Runs", numeric: true }, { key: "wins", label: "Wins", numeric: true },
                { key: "win_pct", label: "Win %", numeric: true, render: (r: Row) => pctRaw(r.win_pct) },
              ]} />
            </>
          )}
          <div className="panel-title">All Partnerships (10+ runs)</div>
          <DataTable maxHeight={520} rows={partnerships as Row[]} columns={[
            { key: nameKey, label }, { key: otherKey, label: otherLabel },
            { key: "runs", label: "Runs", numeric: true },
            { key: "wins", label: "Wins", numeric: true },
            { key: "places", label: "Places", numeric: true },
            { key: "win_pct", label: "Win %", numeric: true, render: (r: Row) => pctRaw(r.win_pct) },
            { key: "place_pct", label: "Place %", numeric: true, render: (r: Row) => pctRaw(r.place_pct) },
          ]} />
        </>
      )}

      {tab === "going" && (
        <>
          {goingNames.length === 0 ? <Alert>No going data in this window.</Alert> : (
            <>
              <div className="filter-bar">
                <label className="field"><span>{label}</span>
                  <select value={goingName} onChange={(e) => setGoingEntity(e.target.value)}>
                    {goingNames.map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
              </div>
              <div className="chart-box" style={{ height: 320 }}>
                <div className="panel-title">Win % by Going - {goingName}</div>
                <ResponsiveContainer>
                  <BarChart data={goingRows}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="going_n" fontSize={11} />
                    <YAxis unit="%" />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Legend />
                    <Bar isAnimationActive={false} dataKey="win_pct" name="Win %" fill="#3c5a99" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <DataTable rows={goingRows} columns={[
                { key: "going_n", label: "Going" },
                { key: "runs", label: "Runs", numeric: true },
                { key: "wins", label: "Wins", numeric: true },
                { key: "places", label: "Places", numeric: true },
                { key: "win_pct", label: "Win %", numeric: true, render: (r: Row) => pctRaw(r.win_pct) },
              ]} />
            </>
          )}
        </>
      )}

      {tab === "trends" && (
        <>
          {Object.keys(trends).length === 0 ? <Alert>Not enough recent data for trend analysis.</Alert> : (
            <>
              <div className="filter-bar">
                <label className="field"><span>Jockey</span>
                  <select value={trendKey} onChange={(e) => setTrendName(e.target.value)}>
                    {Object.keys(trends).map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
              </div>
              {trend && (
                <>
                  <div className="metric-grid">
                    <div className="metric"><div className="label">Last 30 Rides Win %</div><div className="value">{trend.last30_win_pct}%</div></div>
                    <div className="metric"><div className="label">Wins (Last 30)</div><div className="value">{trend.last30_wins}</div></div>
                  </div>
                  <div className="chart-box" style={{ height: 340 }}>
                    <div className="panel-title">30-Ride Rolling Win % - {trendKey}</div>
                    <ResponsiveContainer>
                      <LineChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" fontSize={10} tick={{ display: "none" }} />
                        <YAxis unit="%" domain={[0, "auto"]} />
                        <Tooltip formatter={(v) => `${v}%`} />
                        <Line isAnimationActive={false} type="monotone" dataKey="win_pct" stroke="#3c5a99" dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </>
          )}
        </>
      )}
    </>
  );
}

/** Full page: window selector + data loading for one entity. */
export function EntityStatsPage({ entity, title, subtitle }: { entity: "trainer" | "jockey"; title: string; subtitle: string }) {
  const [window_, setWindow] = useState(90);
  const { data, error, reload } = useData<EntityStatsPayload>(`${entity}_stats_${window_}.json`);

  return (
    <div>
      <div className="page-header">
        <h1>{title}</h1>
        <div className="subtitle">{subtitle}</div>
      </div>
      <div className="filter-bar">
        <label className="field"><span>Time window</span>
          <select value={window_} onChange={(e) => setWindow(Number(e.target.value))}>
            {[14, 30, 90, 180, 365].map((w) => <option key={w} value={w}>Last {w} days</option>)}
          </select>
        </label>
      </div>
      {error != null && <ErrorBox error={error} onRetry={reload} />}
      {!data && !error && <Loading />}
      {data && <StatsPageBody key={window_} payload={data} />}
    </div>
  );
}
