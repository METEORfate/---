$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
$runtimeFile = Join-Path $project 'runtime.local.json'
if (Test-Path -LiteralPath $runtimeFile) {
    $runtime = Get-Content -Raw -LiteralPath $runtimeFile | ConvertFrom-Json
    $interpreter = $runtime.python
} else {
    $interpreter = Join-Path $project '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $interpreter)) { throw 'Run bootstrap.ps1 first; interpreter is missing.' }
& $interpreter -u (Join-Path $project 'run.py') @args
exit $LASTEXITCODE
