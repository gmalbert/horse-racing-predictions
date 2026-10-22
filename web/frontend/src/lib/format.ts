/** Display formatting helpers (values themselves are precomputed server-side). */

export const pct = (v: unknown, digits = 1): string =>
  typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : "-";

export const pctRaw = (v: unknown, digits = 1): string =>
  typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(digits)}%` : "-";

export const signedPct = (v: unknown, digits = 1): string =>
  typeof v === "number" && Number.isFinite(v)
    ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`
    : "-";

export const money = (v: unknown, symbol = "£"): string =>
  typeof v === "number" && Number.isFinite(v)
    ? `${symbol}${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
    : "-";

export const num = (v: unknown, digits = 2): string =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "-";

export const fmtDate = (v: unknown): string => {
  if (typeof v !== "string" || !v) return "-";
  const d = new Date(v.length === 10 ? `${v}T00:00:00` : v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short", year: "numeric" });
};

export const todayLocal = (): string => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

export const addDays = (dateStr: string, days: number): string => {
  const d = new Date(`${dateStr}T00:00:00`);
  d.setDate(d.getDate() + days);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

export const tierClass = (tier: unknown): string => {
  const t = String(tier ?? "");
  if (t.startsWith("Tier 1")) return "badge tier1";
  if (t.startsWith("Tier 2")) return "badge tier2";
  if (t.startsWith("Tier 3")) return "badge tier3";
  return "badge info";
};

export const usd = (value: unknown): string =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", { style: "currency", currency: "USD" })
    : "-";
