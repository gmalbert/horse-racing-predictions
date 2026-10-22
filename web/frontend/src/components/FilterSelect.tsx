import { useEffect, useId, useRef, useState } from "react";

export function MultiSelect({ label, options, values, onChange }: {
  label: string; options: Array<string | number>; values: Array<string | number>; onChange: (v: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const id = useId();
  const selected = values.map(String);
  const visible = options.map(String).filter((option) => option.toLowerCase().includes(search.toLowerCase()));
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  return (
    <div className="field filter-select" ref={root}>
      <span className="filter-label" id={`${id}-label`}>{label}</span>
      <button type="button" className="filter-trigger" ref={trigger} aria-label={label} aria-expanded={open} aria-controls={`${id}-options`} onClick={() => { setOpen(!open); setSearch(""); }}>
        <span className="filter-selection">{selected.length === 0 ? label === "Going" ? "Any going" : `All ${label.toLowerCase()}${label === "Class" ? "es" : "s"}` : selected.length === 1 ? selected[0] : `${selected.length} selected`}</span>
        <span aria-hidden="true">⌄</span>
      </button>
      {open && <div className="filter-options" id={`${id}-options`} role="group" aria-labelledby={`${id}-label`}>
        <input autoFocus aria-label={`Search ${label}`} placeholder={`Search ${label.toLowerCase()}…`} value={search} onChange={(e) => setSearch(e.target.value)} />
        <div className="filter-options-list">
          {visible.map((option) => <label className="filter-option" key={option}>
            <input type="checkbox" checked={selected.includes(option)} onChange={() => onChange(selected.includes(option) ? selected.filter((value) => value !== option) : [...selected, option])} />
            <span>{option}</span>
          </label>)}
          {visible.length === 0 && <p className="caption">No matching options</p>}
        </div>
        <div className="filter-options-actions">
          <button type="button" className="filter-text-button" onClick={() => onChange([])}>Clear selection</button>
          <button type="button" className="btn" onClick={() => { setOpen(false); trigger.current?.focus(); }}>Done</button>
        </div>
      </div>}
    </div>
  );
}
