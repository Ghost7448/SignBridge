@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SignBridge Setup

echo ==========================================
echo         SignBridge - Setup
echo ==========================================
echo.

py -3.11 --version >nul 2>nul
if errorlevel 1 (
    echo Python 3.11 was not found.
    echo Install Python 3.11 and enable the Python Launcher, then retry.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Installing required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installation failed. Check your internet connection and Python version.
    pause
    exit /b 1
)

echo.
echo Setup completed. Start the app with run_app.bat.
pause
