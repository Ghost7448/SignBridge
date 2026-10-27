@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SignBridge

echo ================================================
echo                 SignBridge
echo        Hand Pose Recognition
echo ================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python Launcher not found. Install Python 3.11 from python.org.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo ERROR: Could not create Python 3.11 environment.
    pause
    exit /b 1
  )
)

echo Checking demo dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo ERROR: pip update failed. Check your internet connection.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-demo.txt
if errorlevel 1 (
  echo ERROR: Dependency installation failed. See the message above.
  pause
  exit /b 1
)

echo.
echo Launching SignBridge...
".venv\Scripts\python.exe" simple_signs_app.py
if errorlevel 1 (
  echo.
  echo SignBridge closed with an error. Copy the error text above.
  pause
)
