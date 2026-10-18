$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
$runtimeFile = Join-Path $project 'runtime.local.json'
$interpreter = if (Test-Path -LiteralPath $runtimeFile) {
    (Get-Content -Raw -LiteralPath $runtimeFile | ConvertFrom-Json).python
} else { Join-Path $project '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $interpreter)) { throw 'Interpreter is missing; run bootstrap.ps1 first.' }
& $interpreter -u (Join-Path $project 'scripts/run-p1p2.py') @args
exit $LASTEXITCODE
