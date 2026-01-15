#!/usr/bin/env pwsh
# Start racecard file watcher (PowerShell)
# Monitors data/raw/ for new racecards and auto-generates predictions

Write-Host ""
Write-Host "========================================"
Write-Host "  Racecard File Watcher"
Write-Host "========================================"
Write-Host ""
Write-Host "Monitoring: data\raw\"
Write-Host "Pattern: racecards_YYYY-MM-DD.json"
Write-Host ""
Write-Host "Press Ctrl+C to stop watching"
Write-Host ""

# Activate virtual environment
& .venv\Scripts\Activate.ps1

# Run watcher
python scripts\watch_racecards.py
