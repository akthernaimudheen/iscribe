<#
.SYNOPSIS
    Generate a new iScribe access key and write it into .env.

.DESCRIPTION
    The key is generated locally and written only to .env, which is excluded
    from version control. It is printed once so it can be handed to the trial
    clinicians; it is not recoverable from anywhere else afterwards except .env.

    Run this again to rotate the key. Everyone signed in is logged out on the
    next restart, because session cookies are signed with the key.

.EXAMPLE
    .\scripts\new-access-key.ps1
#>
[CmdletBinding()]
param(
    [string]$EnvFile = "$PSScriptRoot\..\.env"
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..").Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'

$key = & $Python -c "import secrets; print(secrets.token_urlsafe(32))"
if (-not $key) { throw 'Key generation failed.' }

if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $Root '.env.example') $EnvFile
    Write-Host "Created .env from .env.example"
}

$lines = Get-Content $EnvFile
if ($lines -match '^ISCRIBE_ACCESS_TOKEN=') {
    $lines = $lines -replace '^ISCRIBE_ACCESS_TOKEN=.*', "ISCRIBE_ACCESS_TOKEN=$key"
} else {
    $lines += "ISCRIBE_ACCESS_TOKEN=$key"
}
Set-Content -Path $EnvFile -Value $lines -Encoding utf8

Write-Host ""
Write-Host "New access key written to .env" -ForegroundColor Green
Write-Host "Give this key to the trial clinicians. It is shown only once:" -ForegroundColor Yellow
Write-Host ""
Write-Host "    $key" -ForegroundColor White
Write-Host ""
Write-Host "Restart iScribe for it to take effect." -ForegroundColor Cyan
