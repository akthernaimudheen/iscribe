<#
.SYNOPSIS
    Expose the locally-running iScribe over HTTPS with a Cloudflare Tunnel.

.DESCRIPTION
    The tunnel makes an outbound connection to Cloudflare, so the hospital can
    reach iScribe without opening a firewall port or giving this machine a
    public IP. Clinical audio and notes are still processed entirely on this
    host; only the HTTPS session passes through Cloudflare.

    Two modes:

      -Quick   Ephemeral *.trycloudflare.com URL, no Cloudflare account.
               The URL changes every restart. Use for same-day testing.

      -Named   A stable hostname from your Cloudflare account. Use this for the
               actual trial, and put Cloudflare Access in front of the hostname
               so sign-in is tied to named hospital accounts rather than one
               shared key.

.EXAMPLE
    .\scripts\start-tunnel.ps1 -Quick

.EXAMPLE
    .\scripts\start-tunnel.ps1 -Named iscribe-trial
#>
[CmdletBinding(DefaultParameterSetName = 'Quick')]
param(
    [Parameter(ParameterSetName = 'Quick')]
    [switch]$Quick,

    [Parameter(ParameterSetName = 'Named', Mandatory = $true)]
    [string]$Named,

    [int]$Port = 8123
)

$ErrorActionPreference = 'Stop'

$cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cloudflared) {
    throw "cloudflared is not installed. Install it with: winget install Cloudflare.cloudflared"
}

# Refuse to publish a server that is not actually answering, so we never put a
# broken or unauthenticated endpoint on the public internet.
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
} catch {
    throw "iScribe is not responding on port $Port. Start it first: .\scripts\start-iscribe.ps1"
}
Write-Host "iScribe is healthy (uptime $($health.uptime_s)s)" -ForegroundColor Green

if ($PSCmdlet.ParameterSetName -eq 'Named') {
    Write-Host "Starting named tunnel '$Named' -> http://127.0.0.1:$Port" -ForegroundColor Cyan
    & $cloudflared tunnel run --url "http://127.0.0.1:$Port" $Named
} else {
    Write-Host "Starting quick tunnel -> http://127.0.0.1:$Port" -ForegroundColor Cyan
    Write-Host "The public URL is printed below. It changes on every restart." -ForegroundColor Yellow
    & $cloudflared tunnel --url "http://127.0.0.1:$Port"
}
