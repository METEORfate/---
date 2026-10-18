param([double]$Hours = 10, [int]$Workers = 4)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
$env:HF_HOME = Join-Path $project '.cache\huggingface'
$env:TORCH_HOME = Join-Path $project '.cache\torch'
& (Join-Path $PSScriptRoot 'run.ps1') nightly --hours $Hours --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Nightly run stopped; inspect artifacts/nightly/state.json and resume after addressing the cause.' }
