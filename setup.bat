@echo off
chcp 65001 >nul
echo ==========================================
echo  Environment Setup - Plate Vision (Windows)
echo ==========================================
echo.

REM 1. Create directory structure
echo [1/5] Creating project directories...
if not exist "data\images" mkdir "data\images"
if not exist "data\videos" mkdir "data\videos"
if not exist "output" mkdir "output"
if not exist "models" mkdir "models"

REM 2. Create virtual environment
echo [2/5] Creating Python venv...
if not exist "venv" (
    python -m venv venv
)

REM 3. Activate venv
echo [3/5] Activating venv...
call venv\Scripts\activate.bat

REM 4. Upgrade pip and install packages
echo [4/5] Installing packages (this may take a few minutes)...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 5. Verification
echo [5/5] Verifying installation...
python -c "from ultralytics import YOLO; from paddleocr import PaddleOCR; import cv2; print('OK - ready')"

echo.
echo ==========================================
echo  Done! To start developing, run:
echo    venv\Scripts\activate.bat
echo ==========================================
pause
