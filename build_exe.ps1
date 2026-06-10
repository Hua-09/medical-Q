$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "Building MedicalQA with PyInstaller..."
$env:PYTHONNOUSERSITE = "1"
python -m PyInstaller medical_qa_app.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Build output:"
Write-Host "  dist\MedicalQA\MedicalQA.exe"
Write-Host ""
Write-Host "Run command:"
Write-Host "  .\dist\MedicalQA\MedicalQA.exe"
