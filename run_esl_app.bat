@echo off
cd /d "%~dp0"
if not exist ".venv-esl\Scripts\python.exe" (
  echo Setting up SignBridge ESL environment...
  py -3.11 -m venv .venv-esl
  if errorlevel 1 (
    echo Could not create Python 3.11 environment. Install Python 3.11 first.
    pause
    exit /b 1
  )
  ".venv-esl\Scripts\python.exe" -m pip install --upgrade pip
  ".venv-esl\Scripts\python.exe" -m pip install -r requirements-esl.txt
  if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection and Python version.
    pause
    exit /b 1
  )
)
".venv-esl\Scripts\python.exe" esl_app.py
if errorlevel 1 pause
