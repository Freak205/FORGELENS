param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("sync", "format", "format-check", "lint", "typecheck", "test", "verify", "smoke-train", "cuda", "download-aiforge-v2")]
    [string]$Task
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$StorageRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot ".."))
$Uv = Join-Path $StorageRoot "tools\uv.exe"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$env:UV_CACHE_DIR = Join-Path $StorageRoot ".cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $StorageRoot "python"
$env:UV_PYTHON_BIN_DIR = Join-Path $StorageRoot "python-bin"
$env:UV_NO_MANAGED_PYTHON = "1"
$env:TEMP = Join-Path $StorageRoot ".tmp"
$env:TMP = $env:TEMP
$env:HF_HOME = Join-Path $StorageRoot ".cache\huggingface"
$env:TORCH_HOME = Join-Path $StorageRoot ".cache\torch"
$env:XDG_CACHE_HOME = Join-Path $StorageRoot ".cache"

New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Set-Location $ProjectRoot

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

switch ($Task) {
    "sync" {
        & $Uv sync --extra dev --python $Python
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
    }
    "format" { Invoke-Python @("-m", "ruff", "format", "src", "tests", "scripts") }
    "format-check" {
        Invoke-Python @("-m", "ruff", "format", "--check", "src", "tests", "scripts")
    }
    "lint" { Invoke-Python @("-m", "ruff", "check", "src", "tests", "scripts") }
    "typecheck" { Invoke-Python @("-m", "mypy", "src", "scripts") }
    "test" { Invoke-Python @("-m", "pytest") }
    "smoke-train" { Invoke-Python @("scripts\train_smoke.py") }
    "cuda" {
        Invoke-Python @(
            "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
        )
    }
    "download-aiforge-v2" {
        Invoke-Python @("scripts\download_aiforge.py", "v2")
    }
    "verify" {
        & $PSCommandPath "format-check"
        & $PSCommandPath "lint"
        & $PSCommandPath "typecheck"
        & $PSCommandPath "test"
    }
}
