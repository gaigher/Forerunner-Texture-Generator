param(
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

Write-Host "== Forerunner Texture Generator: build exe =="

if ($InstallDeps) {
    Write-Host "Installation/mise a jour des dependances Python..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install pyinstaller
}

Write-Host "Nettoyage des anciens artefacts..."
if (Test-Path "build") { Remove-Item "build" -Recurse -Force }
if (Test-Path "dist") { Remove-Item "dist" -Recurse -Force }

Write-Host "Compilation PyInstaller (mode dossier onedir, sortie dans dist\FTG\)..."
python -m PyInstaller --noconfirm --clean "ForerunnerTextureGenerator.spec"

$appDir = "dist\FTG"
$exePath = Join-Path $appDir "ForerunnerTextureGenerator.exe"
if (-not (Test-Path $exePath)) {
    throw "Echec build: exe introuvable: $exePath"
}

Write-Host "Ajout des dossiers de base dans le paquet portable..."
foreach ($assetDir in @("lignes_construction_base", "metal_textures_base")) {
    if (-not (Test-Path $assetDir)) {
        throw "Echec build: dossier introuvable: $assetDir"
    }
    Copy-Item -Path $assetDir -Destination $appDir -Recurse -Force
}

Write-Host ""
Write-Host "Build termine."
Write-Host "Executable: $exePath"
Write-Host "Important: utilisez uniquement l'exe dans dist\FTG\... (pas build\...)."

$hash = Get-FileHash $exePath -Algorithm SHA256
Write-Host "SHA256: $($hash.Hash)"

$zipPath = "dist\ForerunnerTextureGenerator-win64.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path $appDir -DestinationPath $zipPath
Write-Host "Archive a diffuser: $zipPath"
