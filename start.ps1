# Start the complete app using Docker installed inside WSL.
[CmdletBinding()]
param(
    [string]$Distribution,
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'WSL is required by this launcher. Alternatively, run docker compose up --build -d --wait from a Docker-enabled terminal.'
}

$wslArguments = @()
if ($Distribution) {
    $wslArguments += @('--distribution', $Distribution)
}
$wslArguments += @('--cd', $PSScriptRoot, '--exec', 'env', "HRP_WEB_PORT=$Port", 'docker', 'compose')

Write-Host 'Building and starting Equine Edge in WSL...'
& wsl.exe @wslArguments up --build --detach --wait --wait-timeout 180
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Compose could not start the app. Check the output above. Docker must be running inside the selected WSL distribution.'
}

$appUrl = "http://localhost:$Port"
Write-Host "Equine Edge is ready: $appUrl"
Write-Host 'Betting records persist when the containers stop.'
if (-not $NoBrowser) {
    Start-Process -FilePath $appUrl
}
