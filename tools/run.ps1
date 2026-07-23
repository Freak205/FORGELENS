param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("sync", "format", "format-check", "lint", "typecheck", "test", "verify", "smoke-train", "train-real-baseline", "train-unet-baseline", "train-residual-baseline", "benchmark-inference", "report-assets", "demo", "cuda", "download-aiforge-v2", "download-cord", "extract-cord", "manifest-cord", "build-cord-copy-move", "prepare-trufor")]
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
    "format" {
        Invoke-Python @("-m", "ruff", "format", "src", "tests", "scripts", "demo")
    }
    "format-check" {
        Invoke-Python @(
            "-m", "ruff", "format", "--check", "src", "tests", "scripts", "demo"
        )
    }
    "lint" { Invoke-Python @("-m", "ruff", "check", "src", "tests", "scripts", "demo") }
    "typecheck" { Invoke-Python @("-m", "mypy", "src", "scripts", "demo") }
    "test" { Invoke-Python @("-m", "pytest") }
    "smoke-train" { Invoke-Python @("scripts\train_smoke.py") }
    "train-real-baseline" {
        Invoke-Python @("scripts\train_real_baseline.py")
    }
    "train-unet-baseline" {
        Invoke-Python @(
            "scripts\train_real_baseline.py",
            "--config",
            "configs\training\cord_copy_move_unet.yaml"
        )
    }
    "train-residual-baseline" {
        Invoke-Python @(
            "scripts\train_real_baseline.py",
            "--config",
            "configs\training\cord_copy_move_residual.yaml"
        )
    }
    "benchmark-inference" {
        Invoke-Python @("scripts\benchmark_inference.py")
    }
    "report-assets" { Invoke-Python @("scripts\generate_report_assets.py") }
    "demo" { Invoke-Python @("demo\server.py") }
    "cuda" {
        Invoke-Python @(
            "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
        )
    }
    "download-aiforge-v2" {
        Invoke-Python @("scripts\download_aiforge.py", "v2")
    }
    "download-cord" {
        Invoke-Python @("scripts\download_cord.py")
    }
    "extract-cord" {
        $NodeRoot = Join-Path $StorageRoot "tools\node"
        $env:npm_config_cache = Join-Path $StorageRoot ".cache\npm"
        & "F:\node\npm.cmd" install --prefix $NodeRoot --package-lock=true `
            --save-exact hyparquet@1.26.2 hyparquet-compressors@1.1.1
        if ($LASTEXITCODE -ne 0) { throw "npm dependency install failed" }
        & "F:\node\node.exe" (Join-Path $ProjectRoot "scripts\extract_cord.mjs")
        if ($LASTEXITCODE -ne 0) { throw "CORD extraction failed" }
    }
    "manifest-cord" {
        Invoke-Python @("scripts\build_cord_manifest.py")
    }
    "build-cord-copy-move" {
        Invoke-Python @("scripts\build_cord_copy_move.py")
    }
    "prepare-trufor" {
        Invoke-Python @("scripts\prepare_trufor_inputs.py", "--split", "test")
    }
    "verify" {
        & $PSCommandPath "format-check"
        & $PSCommandPath "lint"
        & $PSCommandPath "typecheck"
        & $PSCommandPath "test"
    }
}
