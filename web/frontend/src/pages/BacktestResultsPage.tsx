import { useMemo, useState } from "react";
import { Line, LineChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend, Scatter, ScatterChart, ZAxis, ReferenceLine } from "recharts";
import { useData, Loading, ErrorBox, Metric, MetricGrid, DataTable, Alert, EmptyState } from "../components/ui";
import type { BacktestPayload } from "../lib/types";
import { num, pct, pctRaw } from "../lib/format";

type Row = Record<string, unknown>;

export default function BacktestResultsPage() {
  const { data, error, reload } = useData<BacktestPayload>("backtest.json");
  const [tab, setTab] = useState("walkforward");
  const [accSegment, setAccSegment] = useState("race_class");
  const [roiSegment, setRoiSegment] = useState("race_class");

  const folds = (data?.results?.folds ?? []) as Row[];
  const foldChart = folds.map((f) => ({ fold: `Fold ${f.fold}`, auc: f.auc, top1: (f.top1_accuracy as number) * 100, top3: (f.top3_accuracy as number) * 100 }));

  const segments = data?.historical?.segments ?? {};
  const accRows = useMemo(() => (segments[accSegment]?.accuracy ?? []) as Row[], [segments, accSegment]);
  const roiRows = useMemo(() => (segments[roiSegment]?.roi ?? []) as Row[], [segments, roiSegment]);

  const calibration = data?.historical?.calibration ?? [];
  const roiTotals = useMemo(() => {
    if (!roiRows.length) return null;
    const bets = roiRows.reduce((s, r) => s + (r.bets as number), 0);
    const profit = roiRows.reduce((s, r) => s + (r.total_profit as number), 0);
    return { bets, profit, roi: bets ? (profit / bets) * 100 : 0 };
  }, [roiRows]);

  if (error) return <ErrorBox error={error} onRetry={reload} />;
  if (!data) return <Loading />;

  const segOptions = Object.keys(segments);

  return (
    <div>
      <div className="page-header">
        <h1>Backtest Results</h1>
        <div className="subtitle">Walk-forward validation, accuracy by segment, ROI and calibration</div>
      </div>
      <div className="tabs">
        {[
          { id: "walkforward", label: "📈 Walk-Forward" },
          { id: "accuracy", label: "🎯 Accuracy Analysis" },
          { id: "roi", label: "💰 ROI Analysis" },
          { id: "calibration", label: "📉 Calibration" },
        ].map((t) => (
          <button key={t.id} aria-pressed={tab === t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {tab === "walkforward" && (
        <>
          {!folds.length ? (
            <EmptyState title="No walk-forward results yet">Validation results will appear here when a completed backtest is available.</EmptyState>
          ) : (
            <>
              {data.summary && (
                <MetricGrid>
                  <Metric label="Folds" value={data.summary.n_folds} />
                  <Metric label="Mean AUC" value={num(data.summary.auc_mean, 4)} delta={data.summary.auc_min != null && data.summary.auc_max != null ? `range ${num(data.summary.auc_min, 3)} - ${num(data.summary.auc_max, 3)}` : undefined} />
                  <Metric label="Mean Top-1" value={pct(data.summary.top1_mean)} />
                  <Metric label="Mean Top-3" value={pct(data.summary.top3_mean)} />
                  <Metric label="Mean Level-Stakes ROI" value={pctRaw(data.summary.level_roi_pct_mean)} />
                </MetricGrid>
              )}
              <div className="chart-box" style={{ height: 320 }}>
                <div className="panel-title">AUC by Fold</div>
                <ResponsiveContainer>
                  <LineChart data={foldChart}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="fold" />
                    <YAxis domain={["auto", "auto"]} />
                    <Tooltip />
                    <Legend />
                    <Line isAnimationActive={false} type="monotone" dataKey="auc" name="AUC" stroke="#3c5a99" strokeWidth={2} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <DataTable
                rows={folds}
                columns={[
                  { key: "fold", label: "Fold", numeric: true },
                  { key: "test_start", label: "Test Start" },
                  { key: "test_end", label: "Test End" },
                  { key: "test_size", label: "Test Rows", numeric: true },
                  { key: "auc", label: "AUC", numeric: true, render: (r) => num(r.auc, 4) },
                  { key: "top1_accuracy", label: "Top-1", numeric: true, render: (r) => pct(r.top1_accuracy) },
                  { key: "top3_accuracy", label: "Top-3", numeric: true, render: (r) => pct(r.top3_accuracy) },
                  { key: "level_roi_pct", label: "Level ROI %", numeric: true, render: (r) => num(r.level_roi_pct, 1) },
                  { key: "level_max_dd", label: "Max DD", numeric: true, render: (r) => num(r.level_max_dd, 0) },
                ]}
              />
            </>
          )}
        </>
      )}

      {tab === "accuracy" && (
        <>
          {!segOptions.length ? (
            <EmptyState title="No accuracy results yet">Accuracy analysis needs predictions with recorded race results.</EmptyState>
          ) : (
            <>
              <div className="filter-bar">
                <label className="field"><span>Analyze by</span>
                  <select value={accSegment} onChange={(e) => setAccSegment(e.target.value)}>
                    {segOptions.map((s) => <option key={s} value={s}>{s.replaceAll("_", " ")}</option>)}
                  </select>
                </label>
              </div>
              <div className="chart-box" style={{ height: 320 }}>
                <ResponsiveContainer>
                  <BarChart data={accRows}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="segment" fontSize={11} />
                    <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(v, name) => (name === "count" ? v : pct(v as number))} />
                    <Legend />
                    <Bar isAnimationActive={false} dataKey="accuracy" name="Accuracy (prob>0.5)" fill="#3c5a99" radius={[4, 4, 0, 0]} />
                    <Bar isAnimationActive={false} dataKey="actual_win_rate" name="Actual win rate" fill="#6c8dd8" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <DataTable
                rows={accRows}
                columns={[
                  { key: "segment", label: "Segment" },
                  { key: "count", label: "Predictions", numeric: true },
                  { key: "accuracy", label: "Accuracy", numeric: true, render: (r) => pct(r.accuracy) },
                  { key: "avg_prob", label: "Avg Prob", numeric: true, render: (r) => pct(r.avg_prob) },
                  { key: "actual_win_rate", label: "Actual Win Rate", numeric: true, render: (r) => pct(r.actual_win_rate) },
                ]}
              />
            </>
          )}
        </>
      )}

      {tab === "roi" && (
        <>
          {!segOptions.length ? (
            <EmptyState title="No ROI results yet">ROI analysis needs settled predictions with market odds.</EmptyState>
          ) : (
            <>
              <div className="filter-bar">
                <label className="field"><span>Analyze ROI by</span>
                  <select value={roiSegment} onChange={(e) => setRoiSegment(e.target.value)}>
                    {segOptions.map((s) => <option key={s} value={s}>{s.replaceAll("_", " ")}</option>)}
                  </select>
                </label>
              </div>
              {!roiRows.length ? (
                <Alert>No market odds joined to settled predictions for this segment yet.</Alert>
              ) : (
                <>
                  {roiTotals && (
                    <MetricGrid>
                      <Metric label="Total Bets (£1 level)" value={roiTotals.bets} />
                      <Metric label="Total P/L" value={`£${roiTotals.profit.toFixed(2)}`} />
                      <Metric label="Overall ROI" value={pctRaw(roiTotals.roi)} />
                    </MetricGrid>
                  )}
                  <div className="chart-box" style={{ height: 320 }}>
                    <ResponsiveContainer>
                      <BarChart data={roiRows}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="segment" fontSize={11} />
                        <YAxis unit="%" />
                        <Tooltip formatter={(v) => `${v}%`} />
                        <Bar isAnimationActive={false} dataKey="roi_pct" name="ROI %" fill="#1e7d43" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <DataTable
                    rows={roiRows}
                    columns={[
                      { key: "segment", label: "Segment" },
                      { key: "bets", label: "Bets", numeric: true },
                      { key: "total_profit", label: "P/L (£)", numeric: true, render: (r) => num(r.total_profit, 2) },
                      { key: "roi_pct", label: "ROI %", numeric: true, render: (r) => pctRaw(r.roi_pct) },
                    ]}
                  />
                </>
              )}
            </>
          )}
        </>
      )}

      {tab === "calibration" && (
        <>
          {!calibration.length ? (
            <EmptyState title="Not enough data for calibration">Calibration compares predicted probabilities with actual outcomes once enough races have settled.</EmptyState>
          ) : (
            <>
              <div className="chart-box" style={{ height: 380 }}>
                <div className="panel-title">Predicted vs Actual Win Rate (10 bins, min 10/bin)</div>
                <ResponsiveContainer>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="predicted_mean" name="Predicted" tickFormatter={(v) => pct(v, 0)} domain={[0, "auto"]} />
                    <YAxis type="number" dataKey="actual_rate" name="Actual" tickFormatter={(v) => pct(v, 0)} domain={[0, "auto"]} />
                    <ZAxis type="number" dataKey="count" range={[60, 400]} name="Count" />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v, name) => (name === "Count" ? v : pct(v as number))} />
                    <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#b03030" strokeDasharray="4 4" />
                    <Scatter data={calibration} fill="#3c5a99" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <Alert>
                Points on the diagonal mean perfect calibration. Above the line = the model is conservative
                (horses win more often than predicted); below = over-confident.
              </Alert>
              <DataTable
                rows={calibration as unknown as Row[]}
                columns={[
                  { key: "bin_label", label: "Probability Bin" },
                  { key: "predicted_mean", label: "Predicted Mean", numeric: true, render: (r) => pct(r.predicted_mean) },
                  { key: "actual_rate", label: "Actual Win Rate", numeric: true, render: (r) => pct(r.actual_rate) },
                  { key: "count", label: "Count", numeric: true },
                ]}
              />
            </>
          )}
        </>
      )}
    </div>
  );
}
