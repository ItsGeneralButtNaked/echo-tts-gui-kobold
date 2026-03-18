@echo off
title KoboldCPP Installer
color 0A

setlocal enabledelayedexpansion

set LOGFILE=%USERPROFILE%\koboldcpp_install.log
set INSTALLDIR=%USERPROFILE%\KoboldCPP

echo ========================================
echo        KOBOLDCPP INSTALLER
echo ========================================
echo Log: %LOGFILE%
echo.

:: -------------------------------
:: CHECK ADMIN
:: -------------------------------
net session >nul 2>&1
if %errorlevel% neq 0 goto :admin_error

:: -------------------------------
:: CUDA DETECT
:: -------------------------------
echo Detecting GPU...
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set CUDA=1
    echo CUDA GPU detected - will install CUDA build
) else (
    set CUDA=0
    echo No NVIDIA GPU detected - will install CPU + Vulkan build
    echo (Vulkan gives GPU acceleration on AMD, Intel and non-CUDA NVIDIA cards)
)
echo [%date% %time%] CUDA=%CUDA% >> "%LOGFILE%"
echo.

:: -------------------------------
:: CREATE INSTALL DIR
:: -------------------------------
echo Creating install directory...
if not exist "%INSTALLDIR%" mkdir "%INSTALLDIR%"
if %errorlevel% neq 0 goto :mkdir_error
echo [%date% %time%] OK - mkdir %INSTALLDIR% >> "%LOGFILE%"
echo.

:: -------------------------------
:: CHECK CURL
:: -------------------------------
where curl >nul 2>&1
if %errorlevel% neq 0 goto :curl_error

:: -------------------------------
:: DOWNLOAD BINARY
:: -------------------------------
if %CUDA%==1 (
    echo Downloading KoboldCPP (CUDA build)...
    echo This may take a few minutes - the CUDA build is around 500MB.
    echo.
    if exist "%INSTALLDIR%\koboldcpp.exe" (
        echo Binary already exists, skipping download.
    ) else (
        curl -L --progress-bar -o "%INSTALLDIR%\koboldcpp.exe" "https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp.exe"
        if !errorlevel! neq 0 goto :download_error
    )
) else (
    echo Downloading KoboldCPP (CPU + Vulkan build)...
    echo This may take a few minutes.
    echo.
    if exist "%INSTALLDIR%\koboldcpp.exe" (
        echo Binary already exists, skipping download.
    ) else (
        curl -L --progress-bar -o "%INSTALLDIR%\koboldcpp.exe" "https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp_nocuda.exe"
        if !errorlevel! neq 0 goto :download_error
    )
)
echo [%date% %time%] OK - download >> "%LOGFILE%"
echo.

:: -------------------------------
:: CREATE DESKTOP SHORTCUT (launcher)
:: -------------------------------
echo Creating KoboldCPP desktop shortcut...
set DESKTOP=%USERPROFILE%\Desktop

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\mkshortcut_kcpp.vbs"
echo sLinkFile = "%DESKTOP%\KoboldCPP.lnk" >> "%TEMP%\mkshortcut_kcpp.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\mkshortcut_kcpp.vbs"
echo oLink.TargetPath = "%INSTALLDIR%\koboldcpp.exe" >> "%TEMP%\mkshortcut_kcpp.vbs"
echo oLink.WorkingDirectory = "%INSTALLDIR%" >> "%TEMP%\mkshortcut_kcpp.vbs"
echo oLink.Description = "Start KoboldCPP" >> "%TEMP%\mkshortcut_kcpp.vbs"
echo oLink.WindowStyle = 1 >> "%TEMP%\mkshortcut_kcpp.vbs"
echo oLink.Save >> "%TEMP%\mkshortcut_kcpp.vbs"
cscript //nologo "%TEMP%\mkshortcut_kcpp.vbs"
del "%TEMP%\mkshortcut_kcpp.vbs"
if %errorlevel% neq 0 goto :shortcut_error
echo [%date% %time%] OK - launcher shortcut >> "%LOGFILE%"

:: KoboldCPP Web UI shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\mkshortcut_kcpp_ui.vbs"
echo sLinkFile = "%DESKTOP%\KoboldCPP UI.lnk" >> "%TEMP%\mkshortcut_kcpp_ui.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\mkshortcut_kcpp_ui.vbs"
echo oLink.TargetPath = "http://localhost:5001" >> "%TEMP%\mkshortcut_kcpp_ui.vbs"
echo oLink.Description = "Open KoboldCPP Web UI" >> "%TEMP%\mkshortcut_kcpp_ui.vbs"
echo oLink.Save >> "%TEMP%\mkshortcut_kcpp_ui.vbs"
cscript //nologo "%TEMP%\mkshortcut_kcpp_ui.vbs"
del "%TEMP%\mkshortcut_kcpp_ui.vbs"
if %errorlevel% neq 0 goto :shortcut_error
echo [%date% %time%] OK - UI shortcut >> "%LOGFILE%"
echo.

:: -------------------------------
:: SUCCESS
:: -------------------------------
echo ========================================
echo        KOBOLDCPP INSTALL COMPLETE
echo ========================================
echo.
echo Installed to: %INSTALLDIR%\koboldcpp.exe
echo.
if %CUDA%==1 (
    echo Build: CUDA - full NVIDIA GPU acceleration
) else (
    echo Build: CPU + Vulkan
    echo        CPU works on any machine.
    echo        For GPU acceleration, select Vulkan
    echo        in the launcher GUI under Hardware.
    echo        Works with AMD, Intel, and most GPUs.
)
echo.
echo Desktop shortcuts created:
echo  - KoboldCPP      : Launches KoboldCPP
echo  - KoboldCPP UI   : Opens http://localhost:5001
echo.
echo NOTE: You still need a GGUF model file.
echo       Download models from https://huggingface.co
echo       and search for GGUF.
echo.
pause
exit /b 0

:: ================================================================
:: ERROR LABELS
:: ================================================================

:admin_error
echo.
echo ========================================
echo  ERROR: PLEASE RUN AS ADMINISTRATOR
echo ========================================
echo  Right-click the script and select
echo  "Run as administrator".
echo ========================================
pause
exit /b 1

:mkdir_error
echo ERROR: Failed to create install directory: %INSTALLDIR%
echo [%date% %time%] ERROR: mkdir failed >> "%LOGFILE%"
pause
exit /b 1

:curl_error
echo.
echo ========================================
echo  ERROR: CURL NOT FOUND
echo ========================================
echo  curl is required to download KoboldCPP.
echo  It is built into Windows 10/11.
echo  If missing, download manually from:
echo  https://github.com/LostRuins/koboldcpp/releases
echo  and place koboldcpp.exe in:
echo  %INSTALLDIR%
echo ========================================
echo [%date% %time%] ERROR: curl not found >> "%LOGFILE%"
pause
exit /b 1

:download_error
echo.
echo ========================================
echo  ERROR: DOWNLOAD FAILED
echo ========================================
echo  Could not download the KoboldCPP binary.
echo  Check your internet connection and retry.
echo  Or download manually from:
echo  https://github.com/LostRuins/koboldcpp/releases
echo  and place koboldcpp.exe in:
echo  %INSTALLDIR%
echo ========================================
echo [%date% %time%] ERROR: download failed >> "%LOGFILE%"
pause
exit /b 1

:shortcut_error
echo ERROR: Failed to create desktop shortcut
echo [%date% %time%] ERROR: shortcut failed >> "%LOGFILE%"
pause
exit /b 1
