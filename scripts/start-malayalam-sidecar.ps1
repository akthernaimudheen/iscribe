<#
.SYNOPSIS
    Start the Malayalam ASR sidecar (experimental).

.DESCRIPTION
    Runs IndicConformer (and optionally IndicTrans2) in the GPU environment and
    exposes them on loopback for the iScribe service to call.

    A separate process is required, not a preference: IndicConformer needs
    Python 3.12 + the AI4Bharat NeMo fork + numpy 1.26, while iScribe runs
    Python 3.14 + faster-whisper + numpy 2.x. They cannot share an interpreter.

    Binds to 127.0.0.1 only. Audio is processed in memory and never stored here.
    Malayalam remains EXPERIMENTAL and is not part of the supported English
    clinical path.

.EXAMPLE
    .\scripts\start-malayalam-sidecar.ps1

.EXAMPLE
    .\scripts\start-malayalam-sidecar.ps1 -Device cpu -Port 8131
#>
[CmdletBinding()]
param(
    [int]$Port = 8131,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [ValidateSet('ctc', 'rnnt')][string]$Decoder = 'ctc',
    [switch]$NoTranslation
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $Root

$Python = 'C:\Users\akthe\scribe-gpu-venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "GPU environment not found at $Python. See docs/MALAYALAM_PIPELINE.md."
}

$Nemo = "$env:USERPROFILE\.cache\huggingface\hub\models--ai4bharat--indicconformer_stt_ml_hybrid_ctc_rnnt_large\snapshots\e96d81e42e5fe73f282b0322d422a48b461a4ca6\indicconformer_stt_ml_hybrid_rnnt_large.nemo"
if (-not (Test-Path $Nemo)) {
    throw "IndicConformer checkpoint not found at $Nemo"
}

$env:MALAYALAM_ASR_MODEL_PATH = $Nemo
$env:MALAYALAM_SIDECAR_DEVICE = $Device
$env:MALAYALAM_ASR_DECODER    = $Decoder
$env:SCRIBE_OFFLINE_MODE      = '1'
$env:HF_HUB_OFFLINE           = '1'
# No credential is needed or wanted: the models are already cached locally.
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue

$MtPath = 'C:\Users\akthe\scribe-models\indictrans2-indic-en-dist-200M'
if (-not $NoTranslation -and (Test-Path $MtPath)) {
    $env:INDIC_TRANS_MODEL_PATH = $MtPath
    Write-Host "Translation enabled (IndicTrans2 local snapshot)" -ForegroundColor Green
} else {
    Remove-Item Env:INDIC_TRANS_MODEL_PATH -ErrorAction SilentlyContinue
    Write-Host "Translation disabled - Malayalam transcript only" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Malayalam sidecar (EXPERIMENTAL - not for clinical use)" -ForegroundColor Cyan
Write-Host "  device   : $Device"
Write-Host "  decoder  : $Decoder"
Write-Host "  endpoint : http://127.0.0.1:$Port  (loopback only)"
Write-Host "  models load at startup; first request waits for /ready" -ForegroundColor DarkGray
Write-Host ""

& $Python -m uvicorn malayalam_sidecar.server:app --host 127.0.0.1 --port $Port
