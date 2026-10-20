$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
& (Get-Command python -ErrorAction Stop).Source -u (Join-Path $PSScriptRoot 'plan3_kaggle.py') @args
exit $LASTEXITCODE
