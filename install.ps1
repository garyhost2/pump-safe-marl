$ErrorActionPreference = "Stop"
$UvVersion = "0.13.8"
Set-Location -Path $PSScriptRoot

function Say($m) { Write-Host "`n==> $m" }

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $local = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $local) {
        $uv = $local
    } else {
        Say "Installing uv $UvVersion into $env:USERPROFILE\.local\bin (no admin rights needed)"
        $env:UV_NO_MODIFY_PATH = "1"
        powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/$UvVersion/install.ps1 | iex"
        $uv = $local
    }
}

Say "Installing Python 3.11 and the exact package versions (first time: a few minutes)"
& $uv sync --frozen
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($args -contains "--no-data") {
    Say "Skipping the data download (run .\run.ps1 data later)"
} else {
    Say "Downloading and building the data (first time: about 3 minutes)"
    & $uv run --frozen python -m data_pipeline
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Say "Checking the installation"
& $uv run --frozen python cli.py check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Say "Done. Next: .\run.ps1 smoke   (or .\run.ps1 help)"
