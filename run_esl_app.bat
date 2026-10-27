@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SignBridge ESL Setup and Launch

echo ================================================
echo       SignBridge - Egyptian Sign Language
echo ================================================
echo.

py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.11 was not found.
  echo Install Python 3.11 and enable the Python Launcher.
  echo Then double-click this file again.
  pause
  exit /b 1
)

if not exist ".venv-esl\Scripts\python.exe" (
  echo [1/3] Creating isolated Python environment...
  py -3.11 -m venv .venv-esl
  if errorlevel 1 (
    echo ERROR: Could not create the Python environment.
    pause
    exit /b 1
  )
) else (
  echo [1/3] Using existing ESL environment.
)

echo.
echo [2/3] Checking/installing required packages.
echo This may take several minutes on the first run.
".venv-esl\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo ERROR: pip upgrade failed. Check internet access and retry.
  pause
  exit /b 1
)
".venv-esl\Scripts\python.exe" -m pip install -r requirements-esl.txt
if errorlevel 1 (
  echo ERROR: Dependency installation failed.
  echo Check internet access, free disk space, and Python 3.11.
  pause
  exit /b 1
)

echo.
echo [3/3] Launching SignBridge ESL...
echo On the first run, click "تشغيل النظام" to download the model assets.
echo Keep this window open while using the app.
echo.
".venv-esl\Scripts\python.exe" esl_app.py
if errorlevel 1 (
  echo.
  echo SignBridge stopped with an error. Read the message above.
  pause
)
