# Environment Setup - Plate Vision (PowerShell)
# If script execution fails, run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Environment Setup - Plate Vision (Win)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Create directory structure
Write-Host "`n[1/5] Creating project directories..." -ForegroundColor Yellow
$folders = @("data", "data\images", "data\videos", "output", "models")
foreach ($folder in $folders) {
    if (!(Test-Path $folder)) { New-Item -ItemType Directory -Path $folder | Out-Null }
}

# 2. Create virtual environment
Write-Host "[2/5] Creating Python venv..." -ForegroundColor Yellow
if (!(Test-Path "venv")) {
    python -m venv venv
}

# 3. Activate venv
Write-Host "[3/5] Activating venv..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# 4. Install packages
Write-Host "[4/5] Installing packages (this may take a few minutes)..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Verification
Write-Host "`n[5/5] Verifying installation..." -ForegroundColor Yellow
python -c "from ultralytics import YOLO; from paddleocr import PaddleOCR; import cv2; print('OK - ready')"

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host " Done! To start developing, run:" -ForegroundColor Green
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green
