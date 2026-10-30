# Manage collector VPN tunnels using Docker locally or inside WSL.
[CmdletBinding()]
param(
    [ValidateSet('Up', 'Down', 'Status', 'Logs')]
    [string]$Action = 'Up',
    [string]$Distribution
)

$ErrorActionPreference = 'Stop'
if ($Action -eq 'Up') {
    foreach ($name in @('racecards', 'odds', 'auxiliary')) {
        $configPath = Join-Path $PSScriptRoot "local\vpn\$name.conf"
        if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
            throw "Missing Proton configuration: $configPath. See docs/LOCAL_VPN_WORKFLOWS.md."
        }
    }
}

$composeArgs = @('compose', '-f', 'compose.vpn.yaml')
switch ($Action) {
    'Up' { $composeArgs += @('up', '--detach', '--wait', '--wait-timeout', '180') }
    'Down' { $composeArgs += 'down' }
    'Status' { $composeArgs += 'ps' }
    'Logs' { $composeArgs += @('logs', '--tail', '80') }
}

if (-not $Distribution -and (Get-Command docker -ErrorAction SilentlyContinue)) {
    Push-Location $PSScriptRoot
    try {
        & docker @composeArgs
        $composeExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} else {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw 'Docker or WSL is required. See docs/LOCAL_VPN_WORKFLOWS.md.'
    }
    $wslArguments = @()
    if ($Distribution) { $wslArguments += @('--distribution', $Distribution) }
    $wslArguments += @('--cd', $PSScriptRoot, '--exec', 'docker')
    & wsl.exe @wslArguments @composeArgs
    $composeExitCode = $LASTEXITCODE
}
if ($composeExitCode -ne 0) {
    throw 'VPN Compose command failed. Check the output above and ensure Docker is running.'
}
if ($Action -eq 'Up') {
    Write-Host 'Tunnels started. Verify distinct exits with: .\.venv\Scripts\python.exe scripts/run_with_vpn.py --check-all'
}
