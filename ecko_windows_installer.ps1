
# Ecko One‑Click Installer (Windows 10/11)

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/ItsGeneralButtNaked/Ecko.git"
$INSTALL_DIR = "$env:USERPROFILE\Ecko"
$LAUNCHER = "$INSTALL_DIR\launch_ecko.bat"
$DESKTOP = [Environment]::GetFolderPath("Desktop")
$SHORTCUT = "$DESKTOP\Ecko.lnk"

function Write-Info($msg){ Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-OK($msg){ Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }

function Test-Command($cmd){
    return $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path","Machine")
    $user = [Environment]::GetEnvironmentVariable("Path","User")
    $env:Path = "$machine;$user"
}

Write-Info "Checking winget..."

if (-not (Test-Command "winget")){
    Write-Warn "winget not found. Install 'App Installer' from Microsoft Store first."
    pause
    exit
}

# ------------------- PYTHON -------------------

Write-Info "Checking Python..."

if (-not (Test-Command "python")){
    Write-Info "Installing Python..."
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
}

Write-OK "Python ready"

# ------------------- GIT -------------------

Write-Info "Checking Git..."

if (-not (Test-Command "git")){
    Write-Info "Installing Git..."
    winget install -e --id Git.Git --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
}

Write-OK "Git ready"

# ------------------- FFMPEG -------------------

Write-Info "Checking FFmpeg..."

if (-not (Test-Command "ffmpeg")){
    Write-Info "Installing FFmpeg..."
    winget install -e --id Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
    Refresh-Path
}

Write-OK "FFmpeg ready"

# ------------------- CLONE REPO -------------------

Write-Info "Installing Ecko..."

if (!(Test-Path $INSTALL_DIR)){
    git clone $REPO_URL $INSTALL_DIR
}
else{
    Write-Warn "Install folder exists, skipping clone"
}

# ------------------- PYTHON PACKAGES -------------------

Write-Info "Installing Python dependencies..."

Set-Location $INSTALL_DIR

python -m ensurepip --upgrade
python -m pip install --upgrade pip

if (Test-Path "requirements.txt"){
    python -m pip install -r requirements.txt
}
else{
    Write-Warn "No requirements.txt found, installing basics"
    python -m pip install numpy pillow tqdm requests
}

# ------------------- LAUNCHER -------------------

Write-Info "Creating launcher..."

@"
@echo off
cd /d "$INSTALL_DIR"
python ecko_web.py
pause
"@ | Out-File $LAUNCHER -Encoding ascii

Write-OK "Launcher created"

# ------------------- DESKTOP SHORTCUT -------------------

Write-Info "Creating desktop shortcut..."

$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($SHORTCUT)
$Shortcut.TargetPath = $LAUNCHER
$Shortcut.WorkingDirectory = $INSTALL_DIR
$Shortcut.Save()

Write-OK "Desktop shortcut created"

Write-Host ""
Write-Host "Ecko installation complete." -ForegroundColor Green
Write-Host "Use the 'Ecko' shortcut on your desktop to launch." -ForegroundColor Green
pause
