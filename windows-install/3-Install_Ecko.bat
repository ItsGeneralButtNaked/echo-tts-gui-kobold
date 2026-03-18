@echo off
title ECKO Installer
color 0A
setlocal enabledelayedexpansion
set step=0
set total=11
set LOGFILE=%USERPROFILE%\ecko_install.log

echo ========================================
echo  ECKO Installer
echo ========================================
echo Log: %LOGFILE%
echo.

:: -------------------------------
:: CHECK WSL
:: -------------------------------
call :progress
echo Checking WSL...
wsl -l >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL not ready. Run configure_wsl.bat first.
    pause
    exit /b 1
)

:: -------------------------------
:: GPU DETECT
:: -------------------------------
call :progress
echo Detecting GPU...
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set CUDA=1
    echo CUDA detected
) else (
    set CUDA=0
    echo No CUDA GPU detected - will install CPU-only torch
)

:: -------------------------------
:: SUDO CACHE
:: Cache sudo for entire install session so no repeated password prompts.
:: You will be asked for your WSL password ONCE here.
:: The rule is removed automatically at the end of the script.
:: -------------------------------
call :progress
echo Setting up sudo cache ^(you may be prompted for your WSL password once^)...
wsl -e bash -c "echo \"$(whoami) ALL=(ALL) NOPASSWD:ALL\" | sudo tee /etc/sudoers.d/ecko-install > /dev/null"
call :check_error sudo-cache

:: -------------------------------
:: BASE INSTALL
:: -------------------------------
call :progress
echo Installing base packages...
wsl -e bash -c "sudo apt update -y && sudo apt upgrade -y"
call :check_error apt-update

wsl -e bash -c "sudo apt install -y git software-properties-common"
call :check_error apt-packages

wsl -e bash -c "sudo add-apt-repository ppa:deadsnakes/ppa -y"
call :check_error ppa

wsl -e bash -c "sudo apt update -y && sudo apt install -y python3.11 python3.11-venv"
call :check_error python

:: -------------------------------
:: REPO
:: Uses || instead of if/else to avoid URL mangling from mirrored networking
:: -------------------------------
call :progress
echo Cloning repo...
wsl -e bash -c "cd ~ && ([ -d ecko ] && echo 'ECKO repo already exists, skipping clone' || git clone https://github.com/ItsGeneralButtNaked/ecko)"
call :check_error git-clone

:: -------------------------------
:: VENV
:: -------------------------------
call :progress
echo Creating virtual environment...
wsl -e bash -c "cd ~/ecko && python3.11 -m venv venv"
call :check_error venv

:: -------------------------------
:: TORCH
:: -------------------------------
call :progress
echo Installing PyTorch...
if %CUDA%==1 (
    wsl -e bash -c "cd ~/ecko && source venv/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cu124"
) else (
    wsl -e bash -c "cd ~/ecko && source venv/bin/activate && pip install torch"
)
call :check_error torch

:: -------------------------------
:: REQUIREMENTS
:: -------------------------------
call :progress
echo Installing requirements...
wsl -e bash -c "cd ~/ecko && source venv/bin/activate && pip install -r requirements.txt"
call :check_error requirements

:: -------------------------------
:: FAISS
:: Multiline bash passed as single line to avoid cmd quoting issues.
:: If CUDA: ensure faiss-gpu-cu12 is installed (remove faiss-cpu if present).
:: If CPU: requirements.txt should already have faiss-cpu, just confirm.
:: -------------------------------
call :progress
if %CUDA%==1 (
    echo Installing FAISS GPU...
    wsl -e bash -c "cd ~/ecko && source venv/bin/activate && pip show faiss-gpu-cu12 > /dev/null 2>&1 && echo 'FAISS GPU already installed' || (pip uninstall -y faiss-cpu 2>/dev/null; pip install faiss-gpu-cu12)"
    call :check_error faiss-gpu
) else (
    echo Verifying FAISS CPU...
    wsl -e bash -c "cd ~/ecko && source venv/bin/activate && pip show faiss-cpu > /dev/null 2>&1 && echo 'FAISS CPU already installed' || pip install faiss-cpu"
    call :check_error faiss-cpu
)

:: -------------------------------
:: CLEANUP SUDO CACHE
:: -------------------------------
echo.
echo Removing temporary sudo rule...
wsl -e bash -c "sudo rm -f /etc/sudoers.d/ecko-install"

:: -------------------------------
:: CREATE DESKTOP SHORTCUTS
:: -------------------------------
call :progress
echo Creating desktop shortcuts...

set DESKTOP=%USERPROFILE%\Desktop

:: Ecko-Server shortcut - launches ecko_web.py in WSL
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\mkshortcut_server.vbs"
echo sLinkFile = "%DESKTOP%\Ecko-Server.lnk" >> "%TEMP%\mkshortcut_server.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\mkshortcut_server.vbs"
echo oLink.TargetPath = "wsl.exe" >> "%TEMP%\mkshortcut_server.vbs"
echo oLink.Arguments = "-e bash -c ""cd ~/ecko && source venv/bin/activate && python ecko_web.py; exec bash""" >> "%TEMP%\mkshortcut_server.vbs"
echo oLink.Description = "Start Ecko Server" >> "%TEMP%\mkshortcut_server.vbs"
echo oLink.WindowStyle = 1 >> "%TEMP%\mkshortcut_server.vbs"
echo oLink.Save >> "%TEMP%\mkshortcut_server.vbs"
cscript //nologo "%TEMP%\mkshortcut_server.vbs"
del "%TEMP%\mkshortcut_server.vbs"
call :check_error ecko-server-shortcut

:: Ecko shortcut - opens localhost:5050 in default browser
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\mkshortcut_ui.vbs"
echo sLinkFile = "%DESKTOP%\Ecko.lnk" >> "%TEMP%\mkshortcut_ui.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\mkshortcut_ui.vbs"
echo oLink.TargetPath = "https://localhost:5050" >> "%TEMP%\mkshortcut_ui.vbs"
echo oLink.Description = "Open Ecko Web UI" >> "%TEMP%\mkshortcut_ui.vbs"
echo oLink.Save >> "%TEMP%\mkshortcut_ui.vbs"
cscript //nologo "%TEMP%\mkshortcut_ui.vbs"
del "%TEMP%\mkshortcut_ui.vbs"
call :check_error ecko-ui-shortcut

echo Desktop shortcuts created.

:: -------------------------------
:: DONE
:: -------------------------------
echo.
echo ========================================
echo  INSTALL COMPLETE
echo ========================================
echo.
echo Desktop shortcuts created:
echo  - Ecko-Server  : Starts ecko_web.py in WSL
echo  - Ecko         : Opens https://localhost:5050
echo.
pause
exit /b 0

:: ================================
:: FUNCTIONS
:: ================================

:log
echo [%date% %time%] %* >> "%LOGFILE%"
exit /b

:check_error
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Step failed - %1
    echo Check log for details: %LOGFILE%
    call :log ERROR at %1
    echo Removing sudo cache after failed install...
    wsl -e bash -c "sudo rm -f /etc/sudoers.d/ecko-install" >nul 2>&1
    pause
    exit /b 1
)
call :log OK - %1
exit /b 0

:progress
set /a step+=1
echo.
echo [Step !step!/%total%]
echo ----------------------------------------
exit /b
