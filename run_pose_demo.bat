@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SignBridge - Hand Pose Demo

if not exist ".venv\Scripts\python.exe" (
    echo The basic demo environment is missing.
    echo Run setup.bat first, then try again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo.
    echo The hand-pose demo closed with an error.
    pause
)
