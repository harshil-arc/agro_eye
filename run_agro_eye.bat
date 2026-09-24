@echo off
title Agro Eye - 100%% Offline Plant Vision AI
cd /d "%~dp0"
echo =======================================================
echo  Starting Agro Eye Plant Disease Monitor (YOLOv8)
echo  Connecting to USB Webcam and Offline Disease Model...
echo =======================================================
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
    pause
)
