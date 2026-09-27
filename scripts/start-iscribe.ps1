<#
.SYNOPSIS
    Start iScribe in production mode on this Windows host.

.DESCRIPTION
    Loads configuration from .env, verifies the access key is set, and starts
    the service. The service binds to localhost by default; expose it with
    scripts/start-tunnel.ps1 (Cloudflare Tunnel) or a reverse proxy.

    All inference runs locally. No clinical audio or text leaves this machine.

.EXAMPLE
    .\scripts\start-iscribe.ps1
#>
[CmdletBinding()]
param(
    [string]$EnvFile = "$PSScriptRoot\..\.env"
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $Root

$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "Virtualenv not found at $Python. Create it first: python -m venv .venv"
}

if (-not (Test-Path $EnvFile)) {
    throw "Missing $EnvFile. Copy .env.example to .env and set ISCRIBE_ACCESS_TOKEN."
}

# Load .env into the process environment. Real environment variables win, so an
# operator can override any single value for one run without editing the file.
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
        $name, $value = $line.Split('=', 2)
        $name = $name.Trim()
        $value = $value.Trim().Trim('"')
        if ($value -and -not [Environment]::GetEnvironmentVariable($name)) {
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

if (-not $env:ISCRIBE_ACCESS_TOKEN) {
    throw "ISCRIBE_ACCESS_TOKEN is not set. Generate one with: .\scripts\new-access-key.ps1"
}

$env:ISCRIBE_ENV = if ($env:ISCRIBE_ENV) { $env:ISCRIBE_ENV } else { 'production' }

Write-Host "Starting iScribe ($env:ISCRIBE_ENV) on $($env:ISCRIBE_HOST):$($env:ISCRIBE_PORT)" -ForegroundColor Cyan
Write-Host "Data directory : $(if ($env:ISCRIBE_DATA_DIR) { $env:ISCRIBE_DATA_DIR } else { "$Root\data" })"
Write-Host "Inference      : local CPU ($env:ISCRIBE_WHISPER_MODEL) - no external AI service" -ForegroundColor Green
Write-Host ""

& $Python -m service
