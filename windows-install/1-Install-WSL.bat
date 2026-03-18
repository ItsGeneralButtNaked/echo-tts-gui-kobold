@echo off
title ECKO - WSL Installer
color 0A

echo ========================================
echo        ECKO WSL SETUP
echo ========================================
echo.

:: --- Check admin ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Please run as Administrator
    pause
    exit /b 1
)

:: --- Check WSL ---
where wsl >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo ========================================
    echo  WSL IS ALREADY INSTALLED
    echo ========================================
    echo  Good news - you don't need to do
    echo  anything here. WSL is ready to go.
    echo.
    echo  Next step: Run 2-Configure_WSL_Networking.bat
    echo ========================================
    echo.
) else (
    echo Installing WSL...
    wsl --install
    if %errorlevel% neq 0 (
        echo ERROR: WSL install failed
        pause
        exit /b 1
    )

    echo.
    echo ========================================
    echo REBOOT REQUIRED
    echo After reboot:
    echo 1. Complete Ubuntu setup
    echo 2. Run install_ecko.bat
    echo ========================================
    pause
    exit /b
)

:: --- Check Ubuntu ---
wsl -l | findstr /i "Ubuntu" >nul
if %errorlevel% neq 0 (
    echo Installing Ubuntu...
    wsl --install -d Ubuntu

    if %errorlevel% neq 0 (
        echo ERROR: Ubuntu install failed
        pause
        exit /b 1
    )

    echo.
    echo ========================================
    echo REBOOT REQUIRED
    echo Complete Ubuntu setup then run:
    echo install_ecko.bat
    echo ========================================
    pause
    exit /b
)

:: --- Check first run ---
wsl -l >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo FIRST RUN NOT COMPLETE
    echo Open Ubuntu and create user first
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo WSL SETUP COMPLETE
echo Now run: 2-Configure_WSL_Networking.bat
echo ========================================
pause
