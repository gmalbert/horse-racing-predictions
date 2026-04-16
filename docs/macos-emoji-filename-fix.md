# Streamlit 1.45+ Navigation — Horse Racing Predictions

## Status
- [x] Current repo page filenames are already plain ASCII (no emoji to rename)
- [x] Entry point migrated to `st.navigation()` / `st.Page()`

## Repo Page Structure

| File | Role | Page title |
|------|------|------------|
| `predictions.py` | Entry point + main page | 🏇 Horse Racing Predictions |
| `pages/data_explorer.py` | Historical data explorer | 📊 Data Explorer |

---

## What Changed in Streamlit 1.45+ (sidebar navigation)

Streamlit 1.45 replaced the automatic directory-scanning sidebar with an explicit
`st.navigation()` API. The key differences from the old approach:

| Old (`pages/` auto-scan) | New (`st.navigation()`) |
|--------------------------|-------------------------|
| Icons come from filename prefix (emoji or number) | Icons declared in `st.Page(icon=...)` |
| `set_page_config` called once per page file | Called **once only** in the entry point |
| `initial_sidebar_state` controls sidebar open/close | Sidebar is always shown when nav is present |
| No grouping | Pages can be grouped under headings |
| `st.logo()` optional | `st.logo()` + `with st.sidebar:` control brand placement |

In **Streamlit 1.51+** the `position` parameter of `st.navigation()` additionally accepts
`"hidden"` to suppress the sidebar nav entirely (useful for single-page or custom nav UIs).

---

## Step-by-Step Instructions

### 1. Confirm page filenames (no action needed for this repo)

Both page files are already plain ASCII:

```
predictions.py          ✅  no emoji
pages/data_explorer.py  ✅  no emoji
```

If you ever add a page with emoji in the filename on Windows, remember that macOS/Docker
will fail to load it. Always use plain filenames and declare icons via `st.Page(icon=...)`.

---

### 2. Remove `st.set_page_config()` from `pages/data_explorer.py`

When using `st.navigation()`, config must be called **once** in the entry point.
`pages/data_explorer.py` currently calls it inside its own `main()` — that call must be
removed once the entry point owns it.

**PowerShell (safe — uses absolute paths):**

```powershell
$base = "C:\Users\gmalb\Downloads\horse-racing-predictions\pages"
Get-ChildItem "$base\*.py" | ForEach-Object {
    $content = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    # Remove single-line set_page_config calls
    $updated = $content -replace '(?m)^\s*st\.set_page_config\(.*\)\r?\n', ''
    [System.IO.File]::WriteAllText($_.FullName, $updated, [System.Text.Encoding]::UTF8)
    Write-Host "Updated: $($_.Name)"
}
```

**Verify nothing remains:**

```powershell
Select-String -Path "pages\*.py" -Pattern "set_page_config"
# Should return no matches
```

---

### 3. Rewrite `predictions.py` entry point

Wrap the existing main-page body in a `predictions_page()` function, then declare the
two-page navigation at module level.

```python
"""
Horse Racing Predictions — entry point (Streamlit 1.45+).
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared.utils import BASE_DIR, LOGO_FILE, MODEL_FILE, SCORED_FIXTURES_FILE, get_now_local
# ... all other existing imports ...

# ── Called ONCE. Sub-pages must NOT call set_page_config. ──────────────────────
st.set_page_config(
    page_title="Horse Racing Predictions",
    page_icon="🏇",
    layout="wide",
    # No initial_sidebar_state needed — st.navigation() manages the sidebar.
)


def predictions_page():
    """Main predictions page — today & tomorrow."""
    # ── Logo (shown here because this is the main content page) ─────────────
    if LOGO_FILE.exists():
        st.image(str(LOGO_FILE), width=200)

    # All existing main() body code goes here (indented into this function).
    # Remove the old st.set_page_config() call — it now lives above.
    # ...
    pass


# ── Sidebar navigation (Streamlit 1.45+) ──────────────────────────────────────
pg = st.navigation(
    [
        st.Page(predictions_page,           title="Predictions",   icon="🏇", default=True),
        st.Page("pages/data_explorer.py",   title="Data Explorer", icon="📊"),
    ],
    position="sidebar",   # default; use "hidden" to suppress the sidebar nav
)

# ── Logo in sidebar ────────────────────────────────────────────────────────────
# The logo is already shown on the main Predictions page (above), so it does NOT
# need to be duplicated in the sidebar.
#
# If you ever remove the logo from the main page content, add it below the nav
# links like this (runs on every page render because it is in the entry point):
#
#   with st.sidebar:
#       if LOGO_FILE.exists():
#           st.image(str(LOGO_FILE), width=150)
#
# Do NOT use st.logo() for this — that places the image at the very top of the
# sidebar, above the page links, which is above rather than below them.

pg.run()
```

---

### 4. Remove the navigation hint banners

`pages/data_explorer.py` currently shows:

```python
st.info("💡 Use the sidebar to navigate back to the main Predictions page", icon="ℹ️")
```

Once `st.navigation()` is active the sidebar always shows both page links, so this banner
is redundant. Delete it.

Similarly, `predictions.py` has:

```python
st.info("📊 **Looking for data exploration?** Check out the **Data Explorer** page …")
```

Remove this too — the sidebar nav makes it self-evident.

---

### 5. Sidebar filters in `data_explorer.py`

`pages/data_explorer.py` uses `st.sidebar.header("Filters")` and several
`st.sidebar.*` widgets. These continue to work unchanged — `st.navigation()` renders the
page links at the top of the sidebar, and any `st.sidebar.*` content in the active page
renders below them automatically.

---

## Commit

```bash
git add predictions.py pages/data_explorer.py
git commit -m "feat: migrate to st.navigation() for Streamlit 1.45+ sidebar nav

- Wrapped predictions.py main body in predictions_page() function
- Added st.navigation() with two pages: Predictions + Data Explorer
- Removed st.set_page_config() from pages/data_explorer.py
- Removed redundant navigation hint banners
- Logo remains in main page content area (not duplicated in sidebar)"
git push
```

---

## ⚠️ Lessons Learned / Pitfalls

### The PowerShell working-directory trap
The `set_page_config` removal script **must use absolute paths** (or `Join-Path` with a
known base). If the shell's working directory was changed by a prior command that failed
silently, `[System.IO.File]::ReadAllText("filename.py")` resolves relative to the *wrong*
directory — it will silently read the wrong file (or fail), and then `WriteAllText` will
**overwrite the target file with the wrong content**.

**Safe pattern (always use absolute paths):**

```powershell
$base = "C:\Users\gmalb\Downloads\horse-racing-predictions\pages"   # always absolute
Get-ChildItem "$base\*.py" | ForEach-Object {
    # $_.FullName is always absolute — safe
    $content = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::UTF8)
    # ... transform ...
    [System.IO.File]::WriteAllText($_.FullName, $updated, [System.Text.Encoding]::UTF8)
}
```

### Verify file contents before committing

After any bulk file operation, spot-check before `git add`:

```powershell
Get-Content pages\data_explorer.py -Head 5
```

If the top of the file looks wrong, do **not** commit — restore from git:

```bash
git checkout -- pages/data_explorer.py
```

### Recovery when a page is corrupted

```bash
# Find the last good commit hash
git log --oneline -5

# Restore the pages/ directory from that commit
git checkout <good-commit-hash> -- pages/

# Redo the set_page_config removal correctly, then commit
```

### `st.navigation()` requires removing `set_page_config` from sub-pages
Streamlit raises `StreamlitSetPageConfigMustBeFirstCommandError` if any sub-page calls
`set_page_config`. This is safe to remove because `st.navigation()` inherits the config
(title, icon, layout) set in the entry point.

### `st.logo()` places the image *above* nav links, not below
`st.logo(image)` renders at the very top of the sidebar — above the page list. If you want
the logo **below** the two page links, use `with st.sidebar: st.image(...)` in the entry
point instead (placed after the `st.navigation(...)` call and before `pg.run()`). In this
repo the logo lives in the main page body, so neither call is needed in the sidebar.
