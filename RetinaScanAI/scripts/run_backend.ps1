# Starts the FastAPI backend in dev mode with auto-reload.
# Usage: from repo root -> .\scripts\run_backend.ps1
Set-Location "$PSScriptRoot\..\backend"
if (-not (Test-Path "models\retina_cnn_demo.pt")) {
    Write-Host "No trained model found. Generating synthetic data + training a demo model first..."
    python ml\generate_synthetic_data.py --out data\synthetic --n-per-class 80
    python ml\train.py --data-dir data\synthetic --epochs 8 --arch resnet18
}
uvicorn app.main:app --reload --port 8000
