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

New-Item -ItemType Directory -Path $SecretsRoot -Force | Out-Null

$HfSecret = Read-Host "Paste the Hugging Face READ token (input is hidden)" -AsSecureString
if ($HfSecret.Length -eq 0) {
    throw "Hugging Face token cannot be empty"
}
$HfSecret | ConvertFrom-SecureString | Set-Content -LiteralPath $HfEncrypted
Write-Output "Encrypted Hugging Face credential saved under F:\HYPERVERGE."

if (-not $HuggingFaceOnly) {
    if (-not (Test-Path -LiteralPath $KagglePlain)) {
        Write-Output "Kaggle file not found yet: $KagglePlain"
        Write-Output "Save Kaggle's downloaded kaggle.json there, then rerun this script."
        exit 0
    }
    $KaggleText = Get-Content -LiteralPath $KagglePlain -Raw
    $KagglePayload = $KaggleText | ConvertFrom-Json
    if (-not $KagglePayload.username -or -not $KagglePayload.key) {
        throw "kaggle.json does not contain username and key"
    }
    $KaggleSecret = ConvertTo-SecureString $KaggleText -AsPlainText -Force
    $KaggleSecret | ConvertFrom-SecureString |
        Set-Content -LiteralPath $KaggleEncrypted
    Remove-Item -LiteralPath $KagglePlain -Force
    Write-Output "Kaggle credential encrypted; plaintext kaggle.json removed."
}
