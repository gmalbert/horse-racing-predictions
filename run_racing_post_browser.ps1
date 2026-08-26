<#
Run the visible-Chrome Racing Post collector locally.

Examples:
  .\run_racing_post_browser.ps1
  .\run_racing_post_browser.ps1 -Date 2026-08-25
  .\run_racing_post_browser.ps1 -Commit -Push

The script is intended to run as the logged-in Windows user (for example from
Task Scheduler), not as a Windows service. It does not enable headless mode or
any stealth/proxy/challenge-solving behavior.
#>

[CmdletBinding()]
param(
    [string]$Date,
    [ValidateRange(1, 7)]
    [int]$Days = 2,
    [string]$Timezone = 'America/New_York',
    [string]$ProfileDir = $env:RACING_POST_PROFILE_DIR,
    [switch]$Commit,
    [switch]$Push
)

$ErrorActionPreference = 'Stop'

if ($Date -and $PSBoundParameters.ContainsKey('Days')) {
    throw 'Use either -Date or -Days, not both.'
}

if ($Push) {
    $Commit = $true
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '.')).Path
$collector = Join-Path $repoRoot 'scripts\fetch_racecards_browser.py'
$logDir = Join-Path $repoRoot 'logs\racing-post-browser'
$null = New-Item -ItemType Directory -Force -Path $logDir

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logDir "fetch-$timestamp.log"
$lockPath = Join-Path $env:TEMP 'horse-racing-predictions-racing-post-browser.lock'
$lockStream = $null

try {
    try {
        $lockStream = [System.IO.File]::Open(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch [System.IO.IOException] {
        throw 'Another Racing Post browser collection is already running.'
    }

    $pythonCandidates = @(
        $env:PYTHON,
        (Join-Path $repoRoot '.venv\Scripts\python.exe'),
        'python'
    ) | Where-Object { $_ }

    $pythonExe = $null
    foreach ($candidate in $pythonCandidates) {
        if ([System.IO.Path]::IsPathRooted($candidate)) {
            if (Test-Path -LiteralPath $candidate) {
                $pythonExe = (Resolve-Path -LiteralPath $candidate).Path
                break
            }
        } else {
            $command = Get-Command $candidate -ErrorAction SilentlyContinue
            if ($command) {
                $pythonExe = $command.Source
                break
            }
        }
    }
    if (-not $pythonExe) {
        throw 'Python was not found. Set PYTHON or install Python 3.11.'
    }

    $collectorArgs = @('--timezone', $Timezone)
    if ($Date) {
        $collectorArgs += @('--date', $Date)
    } else {
        $collectorArgs += @('--days', $Days.ToString())
    }
    if ($ProfileDir) {
        $collectorArgs += @('--profile-dir', $ProfileDir)
    }

    "[$(Get-Date -Format o)] Starting Racing Post browser collection" | Tee-Object -FilePath $logPath
    "Repository: $repoRoot" | Tee-Object -FilePath $logPath -Append
    "Python: $pythonExe" | Tee-Object -FilePath $logPath -Append
    "Arguments: $($collectorArgs -join ' ')" | Tee-Object -FilePath $logPath -Append

    # In PowerShell 7, a native command's stderr can become a terminating
    # error under $ErrorActionPreference = 'Stop'. Keep it in the log so a
    # failed collection reports the real Python error instead of only its
    # first stderr line.
    $previousNativeCommandErrorPreference = $PSNativeCommandUseErrorActionPreference
    try {
        $PSNativeCommandUseErrorActionPreference = $false
        & $pythonExe $collector @collectorArgs 2>&1 | Tee-Object -FilePath $logPath -Append
        $collectorExitCode = $LASTEXITCODE
    } finally {
        $PSNativeCommandUseErrorActionPreference = $previousNativeCommandErrorPreference
    }
    if ($collectorExitCode -ne 0) {
        throw "Racecard collector failed with exit code $collectorExitCode. See $logPath"
    }

    if ($Commit) {
        Push-Location $repoRoot
        try {
            & git add -- 'data/raw/racecards_*.json'
            if ($LASTEXITCODE -ne 0) {
                throw 'git add failed.'
            }

            $stagedDiff = & git diff --cached --quiet
            if ($LASTEXITCODE -eq 0) {
                "[$(Get-Date -Format o)] No racecard changes to commit" | Tee-Object -FilePath $logPath -Append
            } else {
                $unexpectedFiles = @(
                    & git diff --cached --name-only |
                        Where-Object { $_ -and ($_ -notmatch '^data/raw/racecards_[^/]+\.json$') }
                )
                if ($unexpectedFiles.Count -gt 0) {
                    throw "Refusing to commit unrelated staged files: $($unexpectedFiles -join ', ')"
                }
                & git commit -m 'chore: refresh Racing Post browser racecards [skip ci]' 2>&1 | Tee-Object -FilePath $logPath -Append
                if ($LASTEXITCODE -ne 0) {
                    throw 'git commit failed.'
                }
                if ($Push) {
                    & git push 2>&1 | Tee-Object -FilePath $logPath -Append
                    if ($LASTEXITCODE -ne 0) {
                        throw 'git push failed.'
                    }
                }
            }
        } finally {
            Pop-Location
        }
    }

    "[$(Get-Date -Format o)] Completed successfully. Log: $logPath" | Tee-Object -FilePath $logPath -Append
    exit 0
} catch {
    "[$(Get-Date -Format o)] ERROR: $($_.Exception.Message)" | Tee-Object -FilePath $logPath -Append
    exit 1
} finally {
    if ($lockStream) {
        $lockStream.Dispose()
    }
}
