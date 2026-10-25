@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SignBridge

if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first to install the project.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo.
    echo SignBridge closed with an error. Check the message above.
    pause
)
