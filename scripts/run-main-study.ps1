param([double]$Hours = 8, [int]$Workers = 2)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
$runtime = Get-Content -LiteralPath (Join-Path $project 'runtime.local.json') -Raw | ConvertFrom-Json
& $runtime.python -u (Join-Path $PSScriptRoot 'run-main-study.py') --hours $Hours --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Main study stopped; inspect artifacts/main_study/state.json and stderr.log.' }
