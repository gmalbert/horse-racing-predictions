# Windows Task Scheduler Setup for Auto-Start Watcher

## Option 1: Run Watcher at Login (Recommended for Dev)

1. Press `Win + R` and type: `shell:startup`
2. Create a shortcut to `start_watcher.bat` in that folder
3. Watcher starts automatically when you log in

## Option 2: Run as Windows Service (Advanced)

Create a scheduled task:

```powershell
# Open Task Scheduler
taskschd.msc
```

Then:
1. **Create Task** (not Basic Task)
2. **General tab:**
   - Name: "Horse Racing Predictions Watcher"
   - Run whether user is logged on or not
   - Run with highest privileges
3. **Triggers tab:**
   - New → At startup
4. **Actions tab:**
   - New → Start a program
   - Program: `C:\Users\gmalb\Downloads\horse-racing-predictions\.venv\Scripts\python.exe`
   - Arguments: `scripts\watch_racecards.py`
   - Start in: `C:\Users\gmalb\Downloads\horse-racing-predictions`
5. **Conditions tab:**
   - Uncheck "Start only if on AC power"

## Option 3: Run Manually When Needed

Just double-click `start_watcher.bat` when you're about to copy racecards.

## Stopping the Watcher

- **Console:** Press `Ctrl+C`
- **Task Manager:** End the python.exe process running watch_racecards.py
- **Task Scheduler:** Right-click task → End

## Logs

The watcher outputs to console. To save logs:

```powershell
python scripts/watch_racecards.py > watcher.log 2>&1
```

Or modify `start_watcher.ps1`:
```powershell
python scripts\watch_racecards.py | Tee-Object -FilePath "watcher.log"
```
