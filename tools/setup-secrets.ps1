param(
    [switch]$HuggingFaceOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$StorageRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot ".."))
$SecretsRoot = Join-Path $StorageRoot "secrets"
$HfEncrypted = Join-Path $SecretsRoot "hf_token.dpapi"
$KagglePlain = Join-Path $SecretsRoot "kaggle.json"
$KaggleEncrypted = Join-Path $SecretsRoot "kaggle_json.dpapi"
$KaggleApiEncrypted = Join-Path $SecretsRoot "kaggle_api_token.dpapi"

New-Item -ItemType Directory -Path $SecretsRoot -Force | Out-Null

$HfSecret = Read-Host "Paste the Hugging Face READ token (input is hidden)" -AsSecureString
if ($HfSecret.Length -eq 0) {
    throw "Hugging Face token cannot be empty"
}
$HfSecret | ConvertFrom-SecureString | Set-Content -LiteralPath $HfEncrypted
Write-Output "Encrypted Hugging Face credential saved under F:\HYPERVERGE."

if (-not $HuggingFaceOnly) {
    if (Test-Path -LiteralPath $KagglePlain) {
        $KaggleText = Get-Content -LiteralPath $KagglePlain -Raw
        $KagglePayload = $KaggleText | ConvertFrom-Json
        if (-not $KagglePayload.username -or -not $KagglePayload.key) {
            throw "kaggle.json does not contain username and key"
        }
        $KaggleSecret = ConvertTo-SecureString $KaggleText -AsPlainText -Force
        $KaggleSecret | ConvertFrom-SecureString |
            Set-Content -LiteralPath $KaggleEncrypted
        Remove-Item -LiteralPath $KagglePlain -Force
        Write-Output "Legacy Kaggle credential encrypted; plaintext removed."
    }
    else {
        $KaggleApiSecret = Read-Host (
            "Paste the replacement Kaggle API token (input is hidden)"
        ) -AsSecureString
        if ($KaggleApiSecret.Length -eq 0) {
            throw "Kaggle API token cannot be empty"
        }
        $KaggleApiSecret | ConvertFrom-SecureString |
            Set-Content -LiteralPath $KaggleApiEncrypted
        Write-Output "Encrypted Kaggle API token saved under F:\HYPERVERGE."
    }
}
