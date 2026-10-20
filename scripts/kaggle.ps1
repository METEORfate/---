$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
# Packaging only needs the standard library; Kaggle CLI retains its existing login.
$python = (Get-Command python -ErrorAction Stop).Source
& $python -u (Join-Path $PSScriptRoot 'kaggle.py') @args
exit $LASTEXITCODE
