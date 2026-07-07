@echo off
title VoxCPM2-Server-8808
cd /d C:\Users\lhai9\Desktop\voxcpm2
set PYTHONPATH=C:\Users\lhai9\Desktop\voxcpm2\src
echo Starting VoxCPM2...
echo.
E:\ziyuan\VoxCPM-2.0.2-20260505\jian27\python.exe app.py --model-id pretrained_models\VoxCPM2 --port 8808 --host 127.0.0.1
pause
