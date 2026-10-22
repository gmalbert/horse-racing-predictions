import { MultiSelect } from "../components/FilterSelect";
import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../lib/apiClient";
import { useData, Loading, ErrorBox, Metric, MetricGrid, DataTable, Alert, Expander } from "../components/ui";
import type { ExplorerOverview } from "../lib/types";
import { fmtDate, money, num, pctRaw } from "../lib/format";

type Row = Record<string, unknown>;

interface Filters {
  years: number[]; courses: string[]; going: string[]; horse_search: string;
  finish_positions: number[]; top3_only: boolean;
  or_min?: number; or_max?: number;
  tiers: string[]; score_min?: number; score_max?: number;
  classes: string[]; min_prize?: number;
  field_size_min?: number; field_size_max?: number;
  surfaces: string[]; distance_band: string; pattern_only: boolean;
  age_bands: string[];
}

const EMPTY_FILTERS: Filters = {
  years: [], courses: [], going: [], horse_search: "", finish_positions: [], top3_only: false,
  tiers: [], classes: [], surfaces: [], distance_band: "All", pattern_only: false, age_bands: [],
};

const leaderCols = (key: string, label: string) => [
  { key, label },
  { key: "races", label: "Races", numeric: true },
  { key: "avg_finish", label: "Avg Finish", numeric: true },
  { key: "wins", label: "Wins", numeric: true },
  { key: "top3", label: "Top 3", numeric: true },
  { key: "win_rate_pct", label: "Win %", numeric: true, render: (r: Row) => pctRaw(r.win_rate_pct) },
];

/* ── Online explorer (FastAPI) ────────────────────────────────────────────── */
function ExplorerOnline({ overview }: { overview: ExplorerOverview }) {
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [appliedFilters, setAppliedFilters] = useState<Filters>(EMPTY_FILTERS);
  const requestId = useRef(0);
  const [view, setView] = useState("horses");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [result, setResult] = useState<Row | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);

  const runQuery = useCallback(async (v: string, f: Filters, p: number, ps: number) => {
    const id = ++requestId.current;
    setBusy(true);
    setErr(null);
    try {
      const body = {
        ...f,
        years: f.years.length ? f.years : null,
        courses: f.courses.length ? f.courses : null,
        going: f.going.length ? f.going : null,
        finish_positions: f.finish_positions.length ? f.finish_positions : null,
        tiers: f.tiers.length ? f.tiers : null,
        classes: f.classes.length ? f.classes : null,
        surfaces: f.surfaces.length ? f.surfaces : null,
        age_bands: f.age_bands.length ? f.age_bands : null,
        horse_search: f.horse_search || null,
        view: v, page: p, page_size: ps,
      };
      const res = await apiFetch<Row>("/api/explorer/query", { method: "POST", body: JSON.stringify(body) });
      if (id === requestId.current) setResult(res);
    } catch (e) {
      if (id === requestId.current) setErr(e);
    } finally {
      if (id === requestId.current) setBusy(false);
    }
  }, []);

  useEffect(() => {
    runQuery(view, appliedFilters, page, pageSize);
    return () => { requestId.current++; };
  }, [view, appliedFilters, page, pageSize, runQuery]);

  const set = <K extends keyof Filters>(k: K, v: Filters[K]) => setFilters((f) => ({ ...f, [k]: v }));
  const domains = overview.filter_domains;
  const overall = result?.overall as Row | undefined;

  const tabs = [
    { id: "horses", label: "🏇 Horses" },
    { id: "courses", label: "🏟️ Courses" },
    { id: "jockeys", label: "👤 Jockeys" },
    { id: "raw", label: "🗃️ Raw Data" },
  ];

  return (
    <>
      <form className="explorer-filters" onSubmit={(event) => { event.preventDefault(); setPage(1); setAppliedFilters({ ...filters }); }}>
        <div className="explorer-filter-heading">
          <div><h2>Filter results</h2><p>Search horses or narrow the results by course, year and going.</p></div>
        </div>
        <div className="explorer-filter-grid">
          <label className="field"><span>Horse name</span>
            <input value={filters.horse_search} onChange={(e) => set("horse_search", e.target.value)} placeholder="Search by horse name" />
          </label>
          <MultiSelect label="Course" options={domains.courses} values={filters.courses} onChange={(v) => set("courses", v)} />
          <MultiSelect label="Year" options={domains.years} values={filters.years} onChange={(v) => set("years", v.map(Number))} />
          <MultiSelect label="Going" options={domains.going ?? []} values={filters.going} onChange={(v) => set("going", v)} />
        </div>
        <details className="explorer-advanced">
          <summary>Advanced filters</summary>
          <div className="explorer-filter-grid">
            <MultiSelect label="Class" options={domains.classes ?? []} values={filters.classes} onChange={(v) => set("classes", v)} />
            <MultiSelect label="Surface" options={domains.surfaces ?? []} values={filters.surfaces} onChange={(v) => set("surfaces", v)} />
            <MultiSelect label="Tier" options={domains.tiers ?? []} values={filters.tiers} onChange={(v) => set("tiers", v)} />
            <MultiSelect label="Finish position" options={domains.finish_positions ?? Array.from({ length: 30 }, (_, i) => i + 1)} values={filters.finish_positions} onChange={(v) => set("finish_positions", v.map(Number))} />
            <MultiSelect label="Age band" options={domains.age_bands ?? []} values={filters.age_bands} onChange={(v) => set("age_bands", v)} />
            <label className="field"><span>Distance band</span>
              <select value={filters.distance_band} onChange={(e) => set("distance_band", e.target.value)}>
                {["All", "Sprint (5-7f)", "Mile (7-9f)", "Middle (9-12f)", "Long (12f+)"].map((band) => <option key={band}>{band}</option>)}
              </select>
            </label>
            <label className="field"><span>Min prize £</span>
              <input type="number" min={0} value={filters.min_prize ?? ""} onChange={(e) => set("min_prize", e.target.value ? Number(e.target.value) : undefined)} placeholder="Any prize" />
            </label>
          </div>
          <div className="explorer-range-grid">
            {([
              { title: "Official rating", min: "or_min", max: "or_max", minLabel: "OR min", maxLabel: "OR max" },
              { title: "Race score", min: "score_min", max: "score_max", minLabel: "Min score", maxLabel: "Max score" },
              { title: "Field size", min: "field_size_min", max: "field_size_max", minLabel: "Min field size", maxLabel: "Max field size" },
            ] as const).map((range) => <fieldset className="explorer-range" key={range.title}>
              <legend>{range.title}</legend>
              <label className="field"><span>From</span><input aria-label={range.minLabel} type="number" min={0} value={filters[range.min] ?? ""} onChange={(e) => set(range.min, e.target.value ? Number(e.target.value) : undefined)} placeholder="No minimum" /></label>
              <label className="field"><span>To</span><input aria-label={range.maxLabel} type="number" min={0} value={filters[range.max] ?? ""} onChange={(e) => set(range.max, e.target.value ? Number(e.target.value) : undefined)} placeholder="No maximum" /></label>
            </fieldset>)}
          </div>
          <label className="explorer-check"><input type="checkbox" checked={filters.pattern_only} onChange={(e) => set("pattern_only", e.target.checked)} /> Pattern races only</label>
        </details>
        <div className="explorer-filter-actions">
          <label className="explorer-check"><input type="checkbox" checked={filters.top3_only} onChange={(e) => set("top3_only", e.target.checked)} /> Top 3 finishers only</label>
          <div className="explorer-filter-buttons">
            <button type="button" className="btn secondary" onClick={() => { setFilters(EMPTY_FILTERS); setPage(1); setAppliedFilters({ ...EMPTY_FILTERS }); }}>Reset</button>
            <button type="submit" className="btn" disabled={busy}>{busy ? "Querying..." : "Apply Filters"}</button>
          </div>
        </div>
      </form>

      {overall && (
        <MetricGrid>
          <Metric label="Filtered Rows" value={(result?.total_rows as number)?.toLocaleString() ?? "-"} />
          <Metric label="Horses" value={(overall.horses as number)?.toLocaleString() ?? "-"} />
          <Metric label="Jockeys" value={(overall.jockeys as number)?.toLocaleString() ?? "-"} />
          <Metric label="Trainers" value={(overall.trainers as number)?.toLocaleString() ?? "-"} />
          <Metric label="Courses" value={(overall.courses as number)?.toLocaleString() ?? "-"} />
          <Metric label="Avg Finish" value={num(overall.avg_finish, 1)} />
        </MetricGrid>
      )}

      <div className="tabs">
        {tabs.map((t) => <button key={t.id} aria-pressed={view === t.id} className={`tab ${view === t.id ? "active" : ""}`} onClick={() => setView(t.id)}>{t.label}</button>)}
      </div>

      {err != null && <ErrorBox error={err} onRetry={() => runQuery(view, appliedFilters, page, pageSize)} />}
      {busy && !result && <Loading />}

      {result && view === "horses" && <DataTable maxHeight={560} rows={(result.horses ?? []) as Row[]} columns={leaderCols("horse", "Horse")} />}
      {result && view === "courses" && <DataTable maxHeight={560} rows={(result.courses ?? []) as Row[]} columns={leaderCols("course", "Course")} />}
      {result && view === "jockeys" && <DataTable maxHeight={560} rows={(result.jockeys ?? []) as Row[]} columns={leaderCols("jockey", "Jockey")} />}
      {result && view === "raw" && (
        <>
          <div className="filter-bar" style={{ justifyContent: "space-between" }}>
            <span className="caption">Page {result.page as number} of {result.total_pages as number} ({(result.total_rows as number).toLocaleString()} rows)</span>
            <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <button className="btn secondary" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>← Prev</button>
              <button className="btn secondary" disabled={page >= (result.total_pages as number)} onClick={() => setPage((p) => p + 1)}>Next →</button>
              <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }}>
                {[25, 50, 100, 200].map((n) => <option key={n} value={n}>{n}/page</option>)}
              </select>
            </span>
          </div>
          <DataTable
            maxHeight={560}
            rows={(result.rows ?? []) as Row[]}
            columns={[
              { key: "date", label: "Date", render: (r) => fmtDate(r.date) },
              { key: "course", label: "Course" },
              { key: "race_name", label: "Race" },
              { key: "horse", label: "Horse" },
              { key: "pos", label: "Pos", numeric: true },
              { key: "jockey", label: "Jockey" },
              { key: "trainer", label: "Trainer" },
              { key: "dist", label: "Dist" },
              { key: "going", label: "Going" },
              { key: "time", label: "Time" },
              { key: "or", label: "OR", numeric: true },
              { key: "prize", label: "Prize", numeric: true, render: (r) => money(r.prize) },
              { key: "class", label: "Class" },
              { key: "race_score", label: "Score", numeric: true, render: (r) => num(r.race_score, 1) },
            ]}
          />
        </>
      )}
    </>
  );
}

/* ── Offline fallback (static precomputed overview) ───────────────────────── */
function ExplorerOffline({ overview }: { overview: ExplorerOverview }) {
  return (
    <>
      <Alert kind="warning">
        Historical filtering is unavailable - showing the precomputed overview instead.
      </Alert>
      <MetricGrid>
        <Metric label="Rows (2021+)" value={(overview.summary.total_rows as number)?.toLocaleString()} />
        <Metric label="Horses" value={(overview.summary.horses as number)?.toLocaleString()} />
        <Metric label="Jockeys" value={(overview.summary.jockeys as number)?.toLocaleString()} />
        <Metric label="Trainers" value={(overview.summary.trainers as number)?.toLocaleString()} />
        <Metric label="Courses" value={overview.summary.courses} />
        <Metric label="Total Prize" value={money(overview.summary.total_prize)} />
      </MetricGrid>
      <div className="grid-2">
        <div>
          <div className="panel-title">🏇 Top Horses by Wins</div>
          <DataTable rows={overview.top_horses as Row[]} columns={leaderCols("horse", "Horse")} />
        </div>
        <div>
          <div className="panel-title">👤 Top Jockeys by Wins</div>
          <DataTable rows={overview.top_jockeys as Row[]} columns={leaderCols("jockey", "Jockey")} />
        </div>
      </div>
      <div className="mt">
        <div className="panel-title">🏟️ Top Courses</div>
        <DataTable rows={overview.top_courses as Row[]} columns={leaderCols("course", "Course")} />
      </div>
      <Expander title="Distance distribution">
        <DataTable rows={overview.distance_distribution as Row[]} columns={[{ key: "distance", label: "Distance" }, { key: "races", label: "Races", numeric: true }]} />
      </Expander>
    </>
  );
}

/* ── Page ─────────────────────────────────────────────────────────────────── */
export default function DataExplorerPage() {
  const { data: overview, error, reload } = useData<ExplorerOverview>("explorer_overview.json");
  const [apiStatus, setApiStatus] = useState<"unknown" | "online" | "offline">("unknown");
  const [domains, setDomains] = useState<ExplorerOverview["filter_domains"] | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch<ExplorerOverview["filter_domains"]>("/api/explorer/filters")
      .then((result) => { if (!cancelled) { setDomains(result); setApiStatus("online"); } })
      .catch(() => setApiStatus("offline"));
    return () => { cancelled = true; };
  }, []);

  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!overview) return <Loading />;

  return (
    <div>
      <div className="page-header">
        <h1>Data Explorer</h1>
        <div className="subtitle">
          Historical results (2021+) · {apiStatus === "online" ? "interactive filtering" : apiStatus === "offline" ? "precomputed overview (API offline)" : "checking API..."}
        </div>
      </div>
      {apiStatus === "online" ? <ExplorerOnline overview={{ ...overview, filter_domains: domains ?? overview.filter_domains }} /> : apiStatus === "offline" ? <ExplorerOffline overview={overview} /> : <Loading label="Checking historical data..." />}
    </div>
  );
}
