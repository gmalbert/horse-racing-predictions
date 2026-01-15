# Quick Reference: Automatic Prediction Generation

## 🔥 File Watcher (Set It and Forget It)

Start the watcher once and it will automatically generate predictions whenever you copy new racecards to `data/raw/`:

```powershell
# PowerShell
.\start_watcher.ps1

# Or CMD
start_watcher.bat
```

**What it does:**
- Monitors `data/raw/` folder continuously
- Auto-generates predictions when `racecards_YYYY-MM-DD.json` files appear
- Skips dates that already have predictions
- Runs until you press Ctrl+C

**Your workflow:**
1. Start watcher once: `.\start_watcher.ps1`
2. Copy racecards from other repo → `data/raw/`
3. Predictions generate automatically ✨
4. View in Streamlit: `streamlit run predictions.py`

---

## 📦 Batch Processing (One-Time Run)

Process all racecards at once without watching:

```powershell
.\run_batch_predictions.ps1
```

See [BATCH_PREDICTIONS.md](BATCH_PREDICTIONS.md) for complete documentation.
