@echo off
chcp 65001 > nul
title Training Monitor - Auto Restart

echo ========================================
echo    Auto-Restart Training Script
echo ========================================
echo.

:loop
echo [%date% %time%] Starting/Resuming training...
echo.

python train_phase4_unet.py

echo.
echo [%date% %time%] Training stopped!
echo.

REM Check if training completed successfully (30 epochs)
if exist "results_phase4_optimized\checkpoint_epoch_30.pth" (
    echo ========================================
    echo    Training COMPLETED Successfully!
    echo    All 30 epochs finished!
    echo ========================================
    pause
    exit /b 0
)

echo Restarting in 10 seconds...
echo Press Ctrl+C to stop auto-restart.
timeout /t 10 /nobreak > nul

echo.
echo ========================================
goto loop
