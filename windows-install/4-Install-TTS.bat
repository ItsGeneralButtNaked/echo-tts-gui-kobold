@echo off
title Echo-TTS Installer
color 0A

setlocal enabledelayedexpansion

set LOGFILE=%USERPROFILE%\ecko_tts_install.log

echo ========================================
echo        ECHO-TTS INSTALLER
echo ========================================
echo Log: %LOGFILE%
echo.

:: -------------------------------
:: CHECK WSL
:: -------------------------------
echo Checking WSL...
wsl -l >nul 2>&1
if %errorlevel% neq 0 goto :wsl_error
echo WSL OK
echo.

:: -------------------------------
:: CHECK CUDA
:: -------------------------------
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set CUDA=1
) else (
    set CUDA=0
)
if %CUDA%==0 goto :no_cuda_error
echo CUDA GPU detected
echo [%date% %time%] CUDA detected >> "%LOGFILE%"
echo.

:: -------------------------------
:: SUDO CACHE
:: -------------------------------
echo Setting up sudo cache ^(you may be prompted for your WSL password once^)...
wsl -e bash -c "echo \"$(whoami) ALL=(ALL) NOPASSWD:ALL\" | sudo tee /etc/sudoers.d/ecko-tts-install > /dev/null"
if %errorlevel% neq 0 goto :sudo_error
echo [%date% %time%] OK - sudo-cache >> "%LOGFILE%"
echo.

:: -------------------------------
:: CLONE REPO
:: -------------------------------
echo Cloning Echo-TTS repo...
wsl -e bash -c "cd ~ && ([ -d echo-tts-api ] && echo 'Echo-TTS repo already exists, skipping clone' || git clone https://github.com/KevinAHM/echo-tts-api)"
if %errorlevel% neq 0 goto :clone_error
echo [%date% %time%] OK - git-clone >> "%LOGFILE%"
echo.

:: -------------------------------
:: CREATE VENV
:: -------------------------------
echo Setting up virtual environment...
wsl -e bash -c "cd ~/echo-tts-api && ([ -f venv/bin/activate ] && echo 'Venv already exists' || (rm -rf venv && python3.11 -m venv venv))"
if %errorlevel% neq 0 goto :venv_error
echo [%date% %time%] OK - venv >> "%LOGFILE%"
echo.

:: -------------------------------
:: INSTALL TORCH (CUDA)
:: -------------------------------
echo Installing PyTorch (CUDA)...
wsl -e bash -c "cd ~/echo-tts-api && source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
if %errorlevel% neq 0 goto :torch_error
echo [%date% %time%] OK - torch >> "%LOGFILE%"
echo.

:: -------------------------------
:: INSTALL REQUIREMENTS
:: -------------------------------
echo Installing requirements...
wsl -e bash -c "cd ~/echo-tts-api && source venv/bin/activate && pip install -r requirements.txt"
if %errorlevel% neq 0 goto :requirements_error
echo [%date% %time%] OK - requirements >> "%LOGFILE%"
echo.

:: -------------------------------
:: SYSTEM DEPENDENCIES
:: -------------------------------
echo Installing system dependencies...
wsl -e bash -c "sudo apt update && sudo apt install -y ffmpeg libpython3.11 python3.11-dev && sudo ldconfig"
if %errorlevel% neq 0 goto :sysdeps_error
echo [%date% %time%] OK - system-deps >> "%LOGFILE%"
echo.

:: -------------------------------
:: CLEANUP SUDO CACHE (success path)
:: -------------------------------
echo Removing temporary sudo rule...
wsl -e bash -c "sudo rm -f /etc/sudoers.d/ecko-tts-install"
echo.

:: -------------------------------
:: CREATE DESKTOP SHORTCUT
:: -------------------------------
echo Creating Echo-TTS desktop shortcut...
set DESKTOP=%USERPROFILE%\Desktop
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\mkshortcut_tts.vbs"
echo sLinkFile = "%DESKTOP%\Echo-TTS-API.lnk" >> "%TEMP%\mkshortcut_tts.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\mkshortcut_tts.vbs"
echo oLink.TargetPath = "wsl.exe" >> "%TEMP%\mkshortcut_tts.vbs"
echo oLink.Arguments = "-e bash -c ""cd ~/echo-tts-api && source venv/bin/activate && ECHO_PERFORMANCE_PRESET=low_mid ECHO_FISH_DTYPE=bfloat16 python api_server.py; exec bash""" >> "%TEMP%\mkshortcut_tts.vbs"
echo oLink.Description = "Start Echo-TTS API Server" >> "%TEMP%\mkshortcut_tts.vbs"
echo oLink.WindowStyle = 1 >> "%TEMP%\mkshortcut_tts.vbs"
echo oLink.Save >> "%TEMP%\mkshortcut_tts.vbs"
cscript //nologo "%TEMP%\mkshortcut_tts.vbs"
del "%TEMP%\mkshortcut_tts.vbs"
if %errorlevel% neq 0 goto :shortcut_error
echo [%date% %time%] OK - desktop-shortcut >> "%LOGFILE%"
echo.

:: -------------------------------
:: SUCCESS
:: -------------------------------
echo ========================================
echo        ECHO-TTS INSTALL COMPLETE
echo ========================================
echo.
echo Desktop shortcut created:
echo  - Echo-TTS-API.lnk   : Launches the TTS API server
echo.
pause
exit /b 0

:: ================================================================
:: ERROR LABELS
:: All errors jump here - no call :log inside if-blocks anywhere
:: ================================================================

:wsl_error
echo.
echo ========================================
echo  ERROR: WSL NOT READY
echo ========================================
echo  Please run 1-Install-WSL.bat first.
echo ========================================
echo [%date% %time%] ERROR: WSL not ready >> "%LOGFILE%"
pause
exit /b 1

:no_cuda_error
echo.
echo ========================================
echo  NO CUDA GPU DETECTED
echo ========================================
echo  Echo-TTS requires an NVIDIA GPU.
echo  nvidia-smi was not found or failed.
echo.
echo  If you have an NVIDIA GPU, reinstall
echo  drivers from:
echo  https://www.nvidia.com/Download/index.aspx
echo.
echo  Alternatives:
echo   - ElevenLabs (cloud TTS, recommended)
echo   - CPU TTS (not supported by this script)
echo ========================================
echo [%date% %time%] ERROR: No CUDA GPU >> "%LOGFILE%"
pause
exit /b 1

:sudo_error
echo ERROR: Failed to set sudo cache
echo [%date% %time%] ERROR: sudo cache failed >> "%LOGFILE%"
pause
exit /b 1

:clone_error
echo ERROR: Git clone failed
echo [%date% %time%] ERROR: git clone failed >> "%LOGFILE%"
goto :cleanup_sudo_fail

:venv_error
echo ERROR: Venv creation failed
echo [%date% %time%] ERROR: venv failed >> "%LOGFILE%"
goto :cleanup_sudo_fail

:torch_error
echo ERROR: Torch install failed
echo [%date% %time%] ERROR: torch failed >> "%LOGFILE%"
goto :cleanup_sudo_fail

:requirements_error
echo ERROR: Requirements install failed
echo [%date% %time%] ERROR: requirements failed >> "%LOGFILE%"
goto :cleanup_sudo_fail

:sysdeps_error
echo ERROR: System package install failed
echo [%date% %time%] ERROR: system deps failed >> "%LOGFILE%"
goto :cleanup_sudo_fail

:shortcut_error
echo ERROR: Failed to create desktop shortcut
echo [%date% %time%] ERROR: shortcut failed >> "%LOGFILE%"
pause
exit /b 1

:cleanup_sudo_fail
echo.
echo Removing temporary sudo rule...
wsl -e bash -c "sudo rm -f /etc/sudoers.d/ecko-tts-install"
echo.
pause
exit /b 1
