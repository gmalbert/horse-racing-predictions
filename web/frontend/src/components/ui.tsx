import { useEffect, useState, type ReactNode } from "react";
import { getData, invalidateCache } from "../lib/dataClient";

export function Loading({ label = "Loading data..." }: { label?: string }) {
  return <div className="loading" role="status">{label}</div>;
}

export function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return <div className="empty-state" role="status"><h2>{title}</h2><p>{children}</p></div>;
}

export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <div className="error-box">
      <p>Failed to load data: {error instanceof Error ? error.message : String(error)}</p>
      {onRetry && <button className="btn" onClick={onRetry}>Retry</button>}
    </div>
  );
}

export function Metric({ label, value, delta }: { label: string; value: ReactNode; delta?: ReactNode }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {delta != null && <div className="delta">{delta}</div>}
    </div>
  );
}

export function MetricGrid({ children }: { children: ReactNode }) {
  return <div className="metric-grid">{children}</div>;
}

export function Alert({ kind = "info", children }: { kind?: "info" | "success" | "warning" | "error"; children: ReactNode }) {
  return <div className={`alert ${kind === "info" ? "" : kind}`}>{children}</div>;
}

export function Expander({ title, children, defaultOpen = false }: { title: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="expander" open={defaultOpen}>
      <summary>{title}</summary>
      <div className="expander-body">{children}</div>
    </details>
  );
}

export function Tabs({ tabs, active, onChange }: { tabs: Array<{ id: string; label: string }>; active: string; onChange: (id: string) => void }) {
  return (
    <div className="tabs">
      {tabs.map((t) => (
        <button key={t.id} aria-pressed={t.id === active} className={`tab ${t.id === active ? "active" : ""}`} onClick={() => onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

/** Simple typed table. Columns map to record keys; render/format optional. */
export function DataTable<T extends object>({
  columns,
  rows,
  maxHeight,
  rowKey,
}: {
  columns: Array<{ key: string; label: ReactNode; numeric?: boolean; render?: (row: T) => ReactNode }>;
  rows: T[];
  maxHeight?: number;
  rowKey?: (row: T, i: number) => string | number;
}) {
  return (
    <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable results table" style={maxHeight ? { maxHeight, overflowY: "auto" } : undefined}>
      <table className="data">
        <thead>
          <tr>
            {columns.map((c) => (
              <th scope="col" key={c.key} className={c.numeric ? "num" : ""}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={rowKey ? rowKey(row, i) : i}>
              {columns.map((c) => (
                <td key={c.key} className={c.numeric ? "num" : ""}>
                  {c.render ? c.render(row) : ((row as Record<string, unknown>)[c.key] as ReactNode) ?? "-"}
                </td>
              ))}
            </tr>
          ))}
          {rows.length === 0 && (
            <tr><td colSpan={columns.length} className="table-empty">No results match your selection.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

/** Generic hook for fetching a precomputed JSON artifact. */
export function useData<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    if (!path) return;
    getData<T>(path)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(e); });
    return () => { cancelled = true; };
  }, [path, tick]);

  return { data, error, reload: () => {
    if (path) invalidateCache(path);
    setTick((t) => t + 1);
  } };
}
