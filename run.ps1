$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path $uv)) {
        Write-Error "uv is not installed yet. Run .\install.ps1 first."
        exit 1
    }
}
& $uv run --frozen python cli.py @args
exit $LASTEXITCODE
