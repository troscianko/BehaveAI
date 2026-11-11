<#
   BehaveAI - windows installer & launcher
  Windows_Launcher_ps.ps1 -- self-bootstrapping Ultralytics + venv launcher for Windows
  Usage:
    .\Windows_Launcher.bat                 # double-click or run from cmd
    powershell -ExecutionPolicy Bypass -NoProfile -File .\Windows_Launcher_ps.ps1 [script.py args...]
#>

param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $RemainingArgs
)

# Fail fast
$ErrorActionPreference = 'Stop'

# Logging/transcript
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$LogPath = Join-Path $ScriptDir "Windows_Launcher_ps.log"
if (Test-Path $LogPath) { Remove-Item $LogPath -ErrorAction SilentlyContinue }
Start-Transcript -Path $LogPath -Force

try {
    Write-Host "=== Windows_Launcher_ps.ps1 starting ==="

    # Config
    $VENV_DIR = Join-Path $env:USERPROFILE "ultralytics-venv"
    $PYTHON_CANDIDATES = @("py -3", "python", "python3")
    $MARKER = Join-Path $VENV_DIR ".ultralytics_ready"

    # -------------------------
    # Helper functions
    # -------------------------
    function Test-Command { param($cmd) try { & cmd /c "$cmd --version" > $null 2>&1; return $LASTEXITCODE -eq 0 } catch { return $false } }
    function Find-Python { foreach ($cmd in $PYTHON_CANDIDATES) { if (Test-Command $cmd) { return $cmd } }; return $null }

    function Ensure-Python {
        $found = Find-Python
        if ($found) { Write-Host "Found Python command: $found"; return $found }

        Write-Host ""
        $installChoice = Read-Host "Python 3 not found. Do you want to download & install Python 3.12 (64-bit)? (Y/N)"
        if ($installChoice.ToUpper() -ne 'Y') {
            Write-Host "User chose not to install Python. Aborting install."
            throw "Python missing"
        }

        Write-Host "Attempting to download and silently install Python 3.12 (64-bit)..." -ForegroundColor Yellow
        $pyVersion = "3.12.6"
        $url = "https://www.python.org/ftp/python/$pyVersion/python-$pyVersion-amd64.exe"
        $installer = Join-Path $env:TEMP "python-installer.exe"
        Write-Host "Downloading: $url"
        Invoke-WebRequest -Uri $url -OutFile $installer

        Write-Host "Running installer (silent, PrependPath=1). You may see a UAC prompt."
        $proc = Start-Process -FilePath $installer -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1" -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-Error "Python installer failed (exit code $($proc.ExitCode)). Please install Python manually."
            throw "Python installer failed"
        }

        Start-Sleep -Seconds 5
        $found = Find-Python
        if ($found) { Write-Host "Python installed and found as: $found"; return $found } else {
            Write-Warning "Python installed but not found in PATH. You may need to log out and back in."
            throw "Python not found after installation"
        }
    }

    function Venv-PythonExec { param([string[]]$Args) $p = Join-Path $VENV_DIR "Scripts\python.exe"; if (-not (Test-Path $p)) { throw "Venv python not found at $p" }; & $p @Args; return $LASTEXITCODE }

    function Detect-NvidiaGPU {
        try { $nvs = & nvidia-smi -L 2>$null; if ($LASTEXITCODE -eq 0 -and $nvs) { Write-Host "Detected NVIDIA GPU via nvidia-smi: $nvs"; return $true } } catch {}
        try { $adapters = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue; if ($adapters) { foreach ($a in $adapters) { if ($a.AdapterCompatibility -and $a.AdapterCompatibility -match "NVIDIA") { Write-Host "Detected NVIDIA GPU via WMI: $($a.Name)"; return $true } } } } catch {}
        return $false
    }

    function Choose-Torch-Wheel {
        param([bool]$nvidiaPresent)

        Write-Host ""
        Write-Host "PyTorch install options:"
        Write-Host "  1) CPU-only (recommended default)"
        Write-Host "  2) Auto-detect NVIDIA GPU and pick a compatible CUDA wheel (requires NVIDIA driver)"
        Write-Host "  3) Manually pick a CUDA wheel (advanced users)"

        $choice = Read-Host "Choose option - 1=CPU-only, 2=Auto-detect NVIDIA (CUDA), 3=Manual pick. Enter 1/2/3 [default=1]"
        if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

        switch ($choice) {
            "1" {
                return @{ indexUrl = "https://download.pytorch.org/whl/cpu"; label="cpu" }
            }
            "2" {
                if (-not $nvidiaPresent) {
                    Write-Warning "No NVIDIA GPU detected. Falling back to CPU-only."
                    return @{ indexUrl = "https://download.pytorch.org/whl/cpu"; label="cpu" }
                }

                Write-Host "Auto-detect: checking nvidia-smi for driver/CUDA info..."
                $ver = $null
                try {
                    $smi = & nvidia-smi 2>$null
                    if ($LASTEXITCODE -eq 0 -and $smi) {
                        $cudaLine = ($smi | Select-String -Pattern "CUDA Version" -SimpleMatch).ToString()
                        if ($cudaLine) {
                            $m = [regex]::Match($cudaLine, "CUDA Version:\s*([0-9]+\.[0-9]+)")
                            if ($m.Success) { $ver = $m.Groups[1].Value; Write-Host "nvidia-smi reports CUDA $ver" }
                        }
                    }
                } catch {
                    $ver = $null
                }

                if ($ver -and $ver.StartsWith("12")) {
                    return @{ indexUrl = "https://download.pytorch.org/whl/cu121"; label="cu121" }
                } elseif ($ver -and $ver.StartsWith("11.8")) {
                    return @{ indexUrl = "https://download.pytorch.org/whl/cu118"; label="cu118" }
                } elseif ($ver -and $ver.StartsWith("11.7")) {
                    return @{ indexUrl = "https://download.pytorch.org/whl/cu117"; label="cu117" }
                } else {
                    Write-Warning "Could not confidently detect a CUDA version from nvidia-smi. Please choose manually."
                    # fall through to manual selection
                }
            }
            "3" {
                Write-Host ""
                Write-Host "Manual CUDA wheel choices:"
                Write-Host "  a) cu121 (CUDA 12.1)"
                Write-Host "  b) cu118 (CUDA 11.8)"
                Write-Host "  c) cu117 (CUDA 11.7)"
                Write-Host "  d) cu116 (CUDA 11.6)"
                $pick = Read-Host "Pick (a/b/c/d) or press Enter for cu118"
                switch ($pick) {
                    "a" { return @{ indexUrl = "https://download.pytorch.org/whl/cu121"; label="cu121" } }
                    "b" { return @{ indexUrl = "https://download.pytorch.org/whl/cu118"; label="cu118" } }
                    "c" { return @{ indexUrl = "https://download.pytorch.org/whl/cu117"; label="cu117" } }
                    "d" { return @{ indexUrl = "https://download.pytorch.org/whl/cu116"; label="cu116" } }
                    default { return @{ indexUrl = "https://download.pytorch.org/whl/cu118"; label="cu118" } }
                }
            }
            default {
                Write-Host "Unknown choice; defaulting to CPU-only."
                return @{ indexUrl = "https://download.pytorch.org/whl/cpu"; label="cpu" }
            }
        }
    }

    function Is-Ready {
        if (Test-Path $MARKER) { return $true }
        $venvPython = Join-Path $VENV_DIR "Scripts\python.exe"
        if (Test-Path $venvPython) {
            try {
                & $venvPython -c "import ultralytics" > $null 2>&1; if ($LASTEXITCODE -ne 0) { return $false }
                & $venvPython -c "import cv2" > $null 2>&1; return ($LASTEXITCODE -eq 0)
            } catch { return $false }
        }
        return $false
    }

    # -------------------------
    # Bootstrap (create venv & install)
    # -------------------------
    function Bootstrap {
        Write-Host "== Ultralytics bootstrap for Windows: installing python packages into venv =="

        # find or install python
        $pyCmd = Ensure-Python

        Write-Host "Using Python launcher: $pyCmd"
        # Print the exact python version found
        try {
            $pyVer = & cmd /c "$pyCmd --version" 2>&1
            Write-Host "Python version: $pyVer"
        } catch {
            Write-Warning "Couldn't determine Python version with '$pyCmd --version'."
        }

        Write-Host "Creating virtualenv at $VENV_DIR (if missing)..."
        if (-not (Test-Path $VENV_DIR)) {
            & cmd /c "$pyCmd -m venv `"$VENV_DIR`""
            if ($LASTEXITCODE -ne 0) { throw "Failed to create virtualenv" }
        } else { Write-Host "Virtualenv already exists - reusing." }

        $venvPython = Join-Path $VENV_DIR "Scripts\python.exe"
        if (-not (Test-Path $venvPython)) { throw "Venv python not present after creation" }

        Write-Host "Virtualenv python path: $venvPython"
        try {
            $venvPyVer = & $venvPython --version 2>&1
            Write-Host "Virtualenv Python version: $venvPyVer"
        } catch {
            Write-Warning "Could not run venv python --version."
        }

        Write-Host "Upgrading pip, setuptools, wheel inside venv..."
        & $venvPython -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { Write-Warning "pip upgrade reported non-zero exit code" }

        # Ask CPU/GPU choice here
        Write-Host ""
        $installLibsChoice = Read-Host "Install ultralytics, torch and required Python packages now? (Y/N)"
        if ($installLibsChoice.ToUpper() -ne 'Y') { Write-Host "Skipping package installation per user request."; return }

        Write-Host "Checking for NVIDIA GPU..."
        $hasNvidia = Detect-NvidiaGPU
        $torchChoice = Choose-Torch-Wheel -nvidiaPresent:$hasNvidia
        Write-Host "Selected: $($torchChoice.label)"

        try {
            if ($torchChoice.label -eq "cpu") {
                Write-Host "Installing CPU-only PyTorch..."
                & $venvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
            } else {
                Write-Host "Installing CUDA-enabled PyTorch ($($torchChoice.label))."
                $idx = $torchChoice.indexUrl
                & $venvPython -m pip install --index-url $idx torch torchvision torchaudio
            }
            if ($LASTEXITCODE -ne 0) { Write-Warning "PyTorch install returned non-zero exit code; you may need to retry manually." }
        } catch {
            Write-Warning "PyTorch installation raised an error: $_"
        }

        Write-Host "Installing ultralytics and common dependencies inside venv..."
        # NOTE: deliberately NOT "ultralytics[export]" - that extra pulls in every export backend
        # (TFLite/litert-torch, ai-edge-quantizer, onnx, ...) whose pins don't resolve on newer
        # Python (e.g. 3.14), sending pip into endless backtracking. BehaveAI only exports to
        # NCNN, which needs just ncnn + pnnx.
        $packages = @("ultralytics", "numpy", "scipy", "pyyaml", "tqdm", "pillow", "opencv-python", "ncnn", "pnnx")
        & $venvPython -m pip install $packages
        if ($LASTEXITCODE -ne 0) { throw "pip install of core packages failed" }

        Write-Host ""
        Write-Host "Verifying important imports inside the venv (this may take a moment)..."

        # ultralytics
        try {
            & $venvPython -c "import ultralytics; print('ultralytics OK')" | Out-Host
            Write-Host "ultralytics import: SUCCESS"
        } catch {
            Write-Warning "ultralytics import: FAILED - check the log for details."
        }

        # torch
        try {
            & $venvPython -c "import torch; print('torch OK', torch.__version__)" | Out-Host
            Write-Host "torch import: SUCCESS"
        } catch {
            Write-Warning "torch import: FAILED - if you requested CUDA, confirm your NVIDIA driver and chosen CUDA wheel."
        }

        # cv2
        try {
            & $venvPython -c "import cv2; print('cv2 OK', cv2.__version__)" | Out-Host
            Write-Host "cv2 import: SUCCESS"
        } catch {
            Write-Warning "cv2 import: FAILED - consider installing opencv-contrib-python or check the log."
        }

        # marker
        if (-not (Test-Path $VENV_DIR)) { New-Item -ItemType Directory -Path $VENV_DIR | Out-Null }
        New-Item -ItemType File -Force -Path $MARKER | Out-Null
        Write-Host "Bootstrap complete."
    }

    # -------------------------
    # Main flow: ask user if need to install, unless env is already ready
    # -------------------------
    $envReady = Is-Ready
    if (-not $envReady) {
        Write-Host ""
        $installPrompt = Read-Host "Environment not ready. Install Python + libraries and create venv now? (Y/N)"
        if ($installPrompt.ToUpper() -eq 'Y') {
            Bootstrap
            $envReady = Is-Ready
            if (-not $envReady) { Write-Warning "Environment still not ready after bootstrap. Check the log: $LogPath" }
        } else {
            Write-Host "User chose not to install. If environment is not ready, the script will now exit."
            Stop-Transcript
            exit 2
        }
    } else {
        Write-Host "Environment already ready. Using existing venv at $VENV_DIR"
    }

    # -------------------------
    # Run requested script (or BehaveAI.py by default)
    # -------------------------
    $venvPythonExe = Join-Path $VENV_DIR "Scripts\python.exe"
    if ($RemainingArgs -and $RemainingArgs.Count -gt 0) {
        Write-Host "Running: $venvPythonExe $($RemainingArgs -join ' ')"
        & $venvPythonExe @RemainingArgs
        $exitCode = $LASTEXITCODE
        Write-Host "Script exited with code $exitCode"
        Stop-Transcript
        exit $exitCode
    } else {
        $default = Join-Path (Get-Location) "BehaveAI.py"
        if (Test-Path $default) {
            Write-Host "Running ./BehaveAI.py in $(Get-Location)"
            & $venvPythonExe $default
            $exitCode = $LASTEXITCODE
            Write-Host "BehaveAI.py exited with code $exitCode"
            Stop-Transcript
            exit $exitCode
        } else {
            Write-Host "No script provided and BehaveAI.py not found in $(Get-Location)." -ForegroundColor Yellow
            Write-Host "Usage: Windows_Launcher.bat path\to\script.py [args...]"
            Stop-Transcript
            exit 2
        }
    }
}
catch {
    Write-Error "Fatal error: $_"
    Write-Host "See the log at: $LogPath"
    Stop-Transcript
    exit 1
}
