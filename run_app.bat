@echo off
chcp 65001 >nul
cd /d "%~dp0"
title SignBridge - Egyptian Sign Language

echo ================================================
echo                 SignBridge
echo       Egyptian Sign Language Recognition
echo ================================================
echo.
echo Starting the ESL research app...
echo First run: dependencies and model files may take several minutes to download.
echo Keep this window open while the app is running.
echo.

call "%~dp0run_esl_app.bat"
