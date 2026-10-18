param([string]$Python = '3.12', [string]$EnvironmentPath = '')
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project
if (-not $EnvironmentPath) {
    $runtimeFile = Join-Path $project 'runtime.local.json'
    if (Test-Path -LiteralPath $runtimeFile) {
        $previousRuntime = Get-Content -Raw -LiteralPath $runtimeFile | ConvertFrom-Json
        $EnvironmentPath = Split-Path -Parent (Split-Path -Parent $previousRuntime.python)
    } else { $EnvironmentPath = Join-Path $project '.venv' }
}
$EnvironmentPath = [IO.Path]::GetFullPath($EnvironmentPath)
$interpreter = Join-Path $EnvironmentPath 'Scripts\python.exe'
$env:UV_CACHE_DIR = Join-Path (Split-Path -Parent $EnvironmentPath) 'uv-cache'
$env:HF_HOME = Join-Path $project '.cache\huggingface'
$env:TORCH_HOME = Join-Path $project '.cache\torch'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv or use an existing uv executable first.' }
if (-not (Test-Path -LiteralPath $interpreter)) {
    uv venv --python $Python $EnvironmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
uv pip install --python $interpreter torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
uv pip install --python $interpreter -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
@{python=$interpreter; uv_cache=$env:UV_CACHE_DIR} | ConvertTo-Json | Set-Content -LiteralPath 'runtime.local.json' -Encoding utf8
& $interpreter run.py environment
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }
$lockedPackages = uv pip freeze --python $interpreter
$lockedPackages = $lockedPackages | ForEach-Object {
    if ($_ -like 'torch @ *') { 'torch==2.10.0+cu128' } else { $_ }
}
@('--extra-index-url https://download.pytorch.org/whl/cu128') + $lockedPackages | Set-Content -LiteralPath requirements.lock.txt -Encoding utf8

