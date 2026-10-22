import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "../lib/apiClient";
import { Loading, Metric, MetricGrid, DataTable, Alert, EmptyState } from "../components/ui";
import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from "recharts";
import { fmtDate, num, pct, pctRaw, signedPct, todayLocal, usd } from "../lib/format";

type Row = Record<string, unknown>;

interface BankrollState {
  initial_bankroll: number; current_bankroll: number; peak_bankroll: number;
  total_bets: number; total_wins: number; total_staked: number; total_pnl: number;
}

/* ── Kelly tab ────────────────────────────────────────────────────────────── */
function KellyTab({ bankroll }: { bankroll: BankrollState }) {
  const [date, setDate] = useState(todayLocal());
  const [frac, setFrac] = useState(0.25);
  const [minEdge, setMinEdge] = useState(0.05);
  const [result, setResult] = useState<Row | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { setResult(null); }, [bankroll.current_bankroll]);

  const run = async () => {
    setBusy(true); setErr(null); setResult(null);
    try {
      const res = await apiFetch<Row>(`/api/kelly?date=${date}&bankroll=${bankroll.current_bankroll}&kelly_fraction=${frac}&min_edge=${minEdge}`);
      setResult(res);
    } catch (e) { setErr(e); } finally { setBusy(false); }
  };

  return (
    <>
      <form className="kelly-form card" onSubmit={(event) => { event.preventDefault(); run(); }}>
        <h2 className="card-title">Calculate a stake</h2>
        <p className="caption">Choose a date and adjust the staking limits to find bets with a model edge.</p>
        <div className="control-grid">
        <label className="field"><span>Prediction date</span>
          <input type="date" required value={date} onChange={(e) => { setDate(e.target.value); setResult(null); }} />
        </label>
        <label className="field"><span>Kelly fraction ({frac.toFixed(2)})</span>
          <input type="range" min={0.05} max={1} step={0.05} value={frac} onChange={(e) => { setFrac(Number(e.target.value)); setResult(null); }} />
        </label>
        <label className="field"><span>Min edge ({pct(minEdge, 0)})</span>
          <input type="range" min={0} max={0.2} step={0.01} value={minEdge} onChange={(e) => { setMinEdge(Number(e.target.value)); setResult(null); }} />
        </label>
        </div>
        <div className="card-actions"><span className="caption">Available bankroll: {usd(bankroll.current_bankroll)}</span><button className="btn" type="submit" disabled={busy || bankroll.current_bankroll <= 0}>{busy ? "Computing..." : "Get Recommendations"}</button></div>
        {bankroll.current_bankroll <= 0 && <p className="caption">Set a positive bankroll above to calculate stakes.</p>}
      </form>
      {err != null && <Alert kind="error">{String(err)}</Alert>}
      {result && !result.found && <EmptyState title="No predictions for this date">Choose another prediction date to calculate recommendations.</EmptyState>}
      {result && result.found && result.has_odds === false && <Alert kind="warning">Predictions for {date} have no market odds merged yet.</Alert>}
      {result && (result.recommendations as Row[])?.length > 0 && (
        <>
          <MetricGrid>
            <Metric label="Value Bets" value={result.value_bets as number} />
            <Metric label="Total Outlay" value={usd(result.total_outlay)} />
            <Metric label="Expected Profit" value={usd(result.expected_profit)} />
            <Metric label="Expected ROI" value={pctRaw(result.expected_roi_pct)} />
          </MetricGrid>
          <DataTable
            rows={(result.recommendations as Row[])}
            columns={[
              { key: "horse", label: "Horse" },
              { key: "race_time", label: "Time" },
              { key: "course", label: "Course" },
              { key: "win_probability", label: "Win %", numeric: true, render: (r) => pct(r.win_probability) },
              { key: "market_odds", label: "Odds", numeric: true, render: (r) => num(r.market_odds) },
              { key: "edge", label: "Edge", numeric: true, render: (r) => signedPct(r.edge) },
              { key: "stake", label: "Stake", numeric: true, render: (r) => usd(r.stake) },
              { key: "expected_profit", label: "Exp. Profit", numeric: true, render: (r) => usd(r.expected_profit) },
            ]}
          />
        </>
      )}
      {result && (result.recommendations as Row[])?.length === 0 && result.found && result.has_odds !== false && <EmptyState title="No qualifying bets">No bets meet the {pct(minEdge, 0)} minimum edge for {date}. Try another date or adjust the minimum edge.</EmptyState>}
    </>
  );
}

/* ── Log bet tab ──────────────────────────────────────────────────────────── */
function LogBetTab({ onLogged }: { onLogged: () => void }) {
  const [form, setForm] = useState({ bet_date: todayLocal(), horse_name: "", race_id: "", win_probability: "", market_odds: "", stake: "", result: "Pending" });
  const [msg, setMsg] = useState<{ kind: "success" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setMsg(null);
    try {
      await apiFetch("/api/bets", {
        method: "POST",
        body: JSON.stringify({
          bet_date: form.bet_date,
          horse_name: form.horse_name,
          race_id: form.race_id,
          win_probability: form.win_probability ? Number(form.win_probability) / 100 : 0,
          market_odds: Number(form.market_odds),
          stake: Number(form.stake),
          result: form.result,
        }),
      });
      setMsg({ kind: "success", text: `Logged bet on ${form.horse_name}.` });
      setForm((f) => ({ ...f, horse_name: "", race_id: "", stake: "", market_odds: "", win_probability: "" }));
      onLogged();
    } catch (err) {
      setMsg({ kind: "error", text: String(err) });
    } finally { setBusy(false); }
  };

  return (
    <form className="card bet-form" onSubmit={submit}>
      <h2 className="card-title">Log a bet</h2>
      <p className="caption">Enter your stake in US dollars. Leave the result pending to settle it later.</p>
      <div className="grid-2">
        <label className="field"><span>Date</span><input type="date" value={form.bet_date} onChange={(e) => setForm({ ...form, bet_date: e.target.value })} required /></label>
        <label className="field"><span>Horse name</span><input value={form.horse_name} onChange={(e) => setForm({ ...form, horse_name: e.target.value })} required /></label>
        <label className="field"><span>Race ID (optional)</span><input value={form.race_id} onChange={(e) => setForm({ ...form, race_id: e.target.value })} /></label>
        <label className="field"><span>Model win prob (%)</span><input type="number" min={0} max={100} step={0.1} value={form.win_probability} onChange={(e) => setForm({ ...form, win_probability: e.target.value })} /></label>
        <label className="field"><span>Market decimal odds</span><input type="number" min={1.01} step={0.01} value={form.market_odds} onChange={(e) => setForm({ ...form, market_odds: e.target.value })} required /></label>
        <label className="field"><span>Stake $</span><input type="number" min={0.5} step={0.5} value={form.stake} onChange={(e) => setForm({ ...form, stake: e.target.value })} required /></label>
        <label className="field"><span>Result</span>
          <select value={form.result} onChange={(e) => setForm({ ...form, result: e.target.value })}>
            <option>Pending</option><option>Won</option><option>Lost</option>
          </select>
        </label>
      </div>
      <div className="mt"><button className="btn" type="submit" disabled={busy}>{busy ? "Saving..." : "Log Bet"}</button></div>
      {msg && <Alert kind={msg.kind}>{msg.text}</Alert>}
    </form>
  );
}

/* ── Page ─────────────────────────────────────────────────────────────────── */
export default function BettingManagerPage() {
  const [tab, setTab] = useState("kelly");
  const [apiDown, setApiDown] = useState(false);
  const [state, setState] = useState<BankrollState | null>(null);
  const [bets, setBets] = useState<Row[]>([]);
  const [perf, setPerf] = useState<Row | null>(null);
  const [editBankroll, setEditBankroll] = useState("");
  const [actionError, setActionError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [settling, setSettling] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [betsRes, perfRes] = await Promise.all([
        apiFetch<{ bets: Row[]; state: BankrollState }>("/api/bets"),
        apiFetch<Row>("/api/performance"),
      ]);
      setBets(betsRes.bets);
      setState(betsRes.state);
      setEditBankroll(String(betsRes.state.current_bankroll));
      setPerf(perfRes);
      setApiDown(false);
    } catch {
      setApiDown(true);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const settle = async (id: number, result: string) => {
    if (settling !== null) return;
    setSettling(id);
    setActionError(null);
    try {
      await apiFetch(`/api/bets/${id}/settle`, { method: "POST", body: JSON.stringify({ result }) });
      await refresh();
    } catch (error) { setActionError(error); }
    finally { setSettling(null); }
  };

  const saveBankroll = async () => {
    const v = Number(editBankroll);
    if (!Number.isFinite(v) || v < 0 || !editBankroll.trim()) return;
    setSaving(true); setSaved(false); setActionError(null);
    try {
      await apiFetch("/api/bankroll", { method: "PUT", body: JSON.stringify({ current_bankroll: v }) });
      await refresh();
      setSaved(true);
    } catch (error) { setActionError(error); }
    finally { setSaving(false); }
  };

  const drawdown = state && state.peak_bankroll > 0
    ? (state.peak_bankroll - state.current_bankroll) / state.peak_bankroll
    : 0;

  const pendingBets = bets.filter((b) => b.result === "Pending");
  const totals = perf?.totals as Row | undefined;
  const series = (perf?.series ?? []) as Row[];
  const rolling = (perf?.rolling ?? []) as Row[];

  return (
    <div>
      <div className="page-header">
        <h1>Betting Manager</h1>
        <div className="subtitle">Kelly staking, bet logging and P&L tracking in US dollars</div>
      </div>

      {apiDown && (
        <Alert kind="error">
          Betting data is unavailable. Check that the API service is running.
          <button className="btn secondary" onClick={refresh}>Retry connection</button>
        </Alert>
      )}
      {actionError != null && <Alert kind="error">{String(actionError)}</Alert>}
      {!state && !apiDown && <Loading label="Loading betting account..." />}

      {state && (
        <>
          {drawdown >= 0.1 && (
            <Alert kind={drawdown >= 0.2 ? "error" : "warning"}>
              Drawdown alert: bankroll is {pct(drawdown)} below its peak ({usd(state.peak_bankroll)}).
            </Alert>
          )}
          <div className="metric-grid bankroll-summary">
            <Metric label="Current Bankroll" value={usd(state.current_bankroll)} />
            <Metric label="Peak" value={usd(state.peak_bankroll)} />
            <Metric label="Total P/L" value={usd(state.total_pnl)} />
          </div>
          <form className="bankroll-edit" onSubmit={(event) => { event.preventDefault(); saveBankroll(); }}>
            <label className="field" htmlFor="bankroll-amount"><span>Set bankroll (USD)</span>
              <input id="bankroll-amount" type="number" min={0} step={0.01} required value={editBankroll} onChange={(e) => { setEditBankroll(e.target.value); setSaved(false); }} />
            </label>
            <button type="submit" className="btn secondary" disabled={saving}>{saving ? "Saving..." : "Save"}</button>
            {saved && <span className="save-confirmation" role="status">Bankroll updated.</span>}
          </form>
        </>
      )}

      <div className="tabs">
        {[
          { id: "kelly", label: "🎯 Kelly Calculator" },
          { id: "log", label: "📝 Log Bet" },
          { id: "pnl", label: `📊 P&L Tracker${pendingBets.length ? ` (${pendingBets.length} open)` : ""}` },
          { id: "perf", label: "📈 Performance" },
        ].map((t) => (
          <button key={t.id} aria-pressed={tab === t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {tab === "kelly" && state && <KellyTab bankroll={state} />}
      {tab === "log" && !apiDown && <LogBetTab onLogged={refresh} />}

      {tab === "pnl" && (
        <>
          {pendingBets.length > 0 && (
            <>
              <div className="panel-title">⏳ Open Bets</div>
              <DataTable
                rows={pendingBets}
                columns={[
                  { key: "bet_date", label: "Date", render: (r) => fmtDate(r.bet_date) },
                  { key: "horse_name", label: "Horse" },
                  { key: "market_odds", label: "Odds", numeric: true },
                  { key: "stake", label: "Stake", numeric: true, render: (r) => usd(r.stake) },
                  { key: "id", label: "Settle", render: (r) => (
                    <span style={{ display: "flex", gap: 6 }}>
                      <button className="btn" disabled={settling !== null} onClick={() => settle(r.id as number, "Won")}>Won</button>
                      <button className="btn secondary" disabled={settling !== null} onClick={() => settle(r.id as number, "Lost")}>Lost</button>
                    </span>
                  ) },
                ]}
              />
            </>
          )}
          {totals && (
            <MetricGrid>
              <Metric label="Settled Bets" value={totals.bets as number} />
              <Metric label="Win Rate" value={totals.win_rate != null ? pct(totals.win_rate) : "-"} />
              <Metric label="Total P/L" value={usd(totals.pnl)} />
              <Metric label="ROI" value={totals.roi_pct != null ? pctRaw(totals.roi_pct) : "-"} />
            </MetricGrid>
          )}
          <div className="panel-title">Full History</div>
          {!bets.length && <EmptyState title="No bets logged yet">Use the Log Bet tab to add your first bet. Open and settled bets will appear here.</EmptyState>}
          {bets.length > 0 &&
          <DataTable
            maxHeight={480}
            rows={[...bets].reverse()}
            columns={[
              { key: "bet_date", label: "Date", render: (r) => fmtDate(r.bet_date) },
              { key: "horse_name", label: "Horse" },
              { key: "win_probability", label: "Model %", numeric: true, render: (r) => pct(r.win_probability) },
              { key: "market_odds", label: "Odds", numeric: true },
              { key: "stake", label: "Stake", numeric: true, render: (r) => usd(r.stake) },
              { key: "result", label: "Result", render: (r) => <span className={`badge ${r.result === "Won" ? "tier1" : r.result === "Lost" ? "tier3" : "info"}`}>{String(r.result)}</span> },
              { key: "profit", label: "Profit", numeric: true, render: (r) => usd(r.profit) },
            ]}
          />}
        </>
      )}

      {tab === "perf" && (
        <>
          {!series.length ? <EmptyState title="No settled bets yet">Performance charts appear once you settle a bet in the P&amp;L Tracker.</EmptyState> : (
            <>
              <div className="chart-box mb" style={{ height: 300 }}>
                <div className="panel-title">Cumulative P/L ($)</div>
                <ResponsiveContainer>
                  <LineChart data={series}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" fontSize={10} />
                    <YAxis />
                    <Tooltip />
                    <Line isAnimationActive={false} type="monotone" dataKey="cum_pnl" stroke="#3c5a99" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="grid-2">
                <div className="chart-box" style={{ height: 280 }}>
                  <div className="panel-title">Cumulative ROI %</div>
                  <ResponsiveContainer>
                    <LineChart data={series}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" fontSize={10} />
                      <YAxis unit="%" />
                      <Tooltip formatter={(v) => `${v}%`} />
                      <Line isAnimationActive={false} type="monotone" dataKey="cum_roi_pct" stroke="#1e7d43" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="chart-box" style={{ height: 280 }}>
                  <div className="panel-title">30-Day Rolling ROI %</div>
                  <ResponsiveContainer>
                    <LineChart data={rolling}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" fontSize={10} />
                      <YAxis unit="%" />
                      <Tooltip formatter={(v) => `${v}%`} />
                      <Line isAnimationActive={false} type="monotone" dataKey="rolling_roi_pct" stroke="#b07d10" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
