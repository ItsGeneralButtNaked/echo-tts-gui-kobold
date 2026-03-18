@echo off
title WSL Network Configurator
color 0B
echo ========================================
echo  WSL Network Configuration
echo ========================================
echo.
echo This script writes .wslconfig and restarts WSL.
echo Run this ONCE before running install_ecko.bat
echo.

:: CHECK WSL EXISTS
wsl -l >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL not found. Please install WSL first.
    echo Run: wsl --install
    pause
    exit /b 1
)

:: WRITE WSLCONFIG
echo Writing %USERPROFILE%\.wslconfig...
(
echo [wsl2]
echo ipv6=false
echo.
echo [experimental]
echo networkingMode=mirrored
) > "%USERPROFILE%\.wslconfig"

echo Done.
echo.

:: SHUTDOWN WSL TO APPLY CONFIG
echo Shutting down WSL to apply config...
wsl --shutdown
echo WSL shut down. Config will apply on next WSL launch.
echo.
echo ========================================
echo  WSL configured. You can now run:
echo  install_ecko.bat
echo ========================================
echo.
pause
