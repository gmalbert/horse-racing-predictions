#!/usr/bin/env pwsh
# Quick batch prediction generator for PowerShell
# Activates venv and runs batch prediction script

Write-Host ""
Write-Host "========================================"
Write-Host "  Batch Prediction Generator"
Write-Host "========================================"
Write-Host ""

# Activate virtual environment
& .venv\Scripts\Activate.ps1

# Run batch script
python scripts\batch_generate_predictions.py $args
