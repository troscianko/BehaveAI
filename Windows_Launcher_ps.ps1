<#
   BehaveAI - Windows installer & launcher
  Windows_Launcher_ps.ps1 -- self-bootstrapping launcher for Windows
  Usage:
    .\Windows_Launcher.bat                 # double-click or run from cmd
    powershell -ExecutionPolicy Bypass -NoProfile -File .\Windows_Launcher_ps.ps1 [behaveai args...]
#>

param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $RemainingArgs
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$LogPath = Join-Path $ScriptDir "Windows_Launcher_ps.log"
if (Test-Path $LogPath) { Remove-Item $LogPath -ErrorAction SilentlyContinue }
Start-Transcript -Path $LogPath -Force

try {
    Write-Host "=== Windows_Launcher_ps.ps1 starting ==="

    $VENV_DIR = Join-Path $env:USERPROFILE "ultralytics-venv"
    $PYTHON_CANDIDATES = @("py -3", "python", "python3")
    $MARKER = Join-Path $VENV_DIR ".behaveai_ready"

    # -------------------------
    # Helper functions
    # -------------------------
    function Test-Command { param($cmd) try { & cmd /c "$cmd --version" > $null 2>&1; return $LASTEXITCODE -eq 0 } catch { return $false } }
    function Find-Python { foreach ($cmd in $PYTHON_CANDIDATES) { if (Test-Command $cmd) { return $cmd } }; return $null }

    function Ensure-Python {
        $found = Find-Python
        if ($found) { Write-Host "Found Python: $found"; return $found }

        $installChoice = Read-Host "Python 3 not found. Download & install Python 3.12 (64-bit)? (Y/N)"
        if ($installChoice.ToUpper() -ne 'Y') { throw "Python missing" }

        $pyVersion = "3.12.6"
        $url = "https://www.python.org/ftp/python/$pyVersion/python-$pyVersion-amd64.exe"
        $installer = Join-Path $env:TEMP "python-installer.exe"
        Write-Host "Downloading: $url"
        Invoke-WebRequest -Uri $url -OutFile $installer

        Write-Host "Running installer (silent, PrependPath=1). You may see a UAC prompt."
        $proc = Start-Process -FilePath $installer -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1" -Wait -PassThru
        if ($proc.ExitCode -ne 0) { throw "Python installer failed (exit code $($proc.ExitCode))" }

        Start-Sleep -Seconds 5
        $found = Find-Python
        if ($found) { Write-Host "Python installed and found as: $found"; return $found }
        throw "Python not found after installation. You may need to log out and back in."
    }

    function Detect-NvidiaGPU {
        try { $nvs = & nvidia-smi -L 2>$null; if ($LASTEXITCODE -eq 0 -and $nvs) { Write-Host "Detected NVIDIA GPU: $nvs"; return $true } } catch {}
        try {
            $adapters = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
            if ($adapters) { foreach ($a in $adapters) { if ($a.AdapterCompatibility -match "NVIDIA") { Write-Host "Detected NVIDIA GPU: $($a.Name)"; return $true } } }
        } catch {}
        return $false
    }

    function Choose-Torch-Wheel {
        param([bool]$nvidiaPresent)

        Write-Host ""
        Write-Host "PyTorch install options:"
        Write-Host "  1) CPU-only (recommended default)"
        Write-Host "  2) Auto-detect NVIDIA GPU and pick a compatible CUDA wheel"
        Write-Host "  3) Manually pick a CUDA wheel (advanced users)"

        $choice = Read-Host "Choose option - 1/2/3 [default=1]"
        if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

        switch ($choice) {
            "1" { return @{ indexUrl = "https://download.pytorch.org/whl/cpu"; label="cpu" } }
            "2" {
                if (-not $nvidiaPresent) {
                    Write-Warning "No NVIDIA GPU detected. Falling back to CPU-only."
                    return @{ indexUrl = "https://download.pytorch.org/whl/cpu"; label="cpu" }
                }
                $ver = $null
                try {
                    $smi = & nvidia-smi 2>$null
                    if ($LASTEXITCODE -eq 0 -and $smi) {
                        $cudaLine = ($smi | Select-String -Pattern "CUDA Version" -SimpleMatch).ToString()
                        $m = [regex]::Match($cudaLine, "CUDA Version:\s*([0-9]+\.[0-9]+)")
                        if ($m.Success) { $ver = $m.Groups[1].Value; Write-Host "nvidia-smi reports CUDA $ver" }
                    }
                } catch { $ver = $null }

                if     ($ver -and $ver.StartsWith("12.8")) { return @{ indexUrl = "https://download.pytorch.org/whl/cu128"; label="cu128" } }
                elseif ($ver -and $ver.StartsWith("12"))   { return @{ indexUrl = "https://download.pytorch.org/whl/cu124"; label="cu124" } }
                elseif ($ver -and $ver.StartsWith("11.8")) { return @{ indexUrl = "https://download.pytorch.org/whl/cu118"; label="cu118" } }
                else {
                    Write-Warning "Could not detect CUDA version. Please choose manually."
                    # fall through to manual
                    Write-Host "  a) cu128 (CUDA 12.8)  b) cu124 (CUDA 12.4)  c) cu118 (CUDA 11.8)"
                    $pick = Read-Host "Pick (a/b/c) [default=b]"
                    switch ($pick) {
                        "a" { return @{ indexUrl = "https://download.pytorch.org/whl/cu128"; label="cu128" } }
                        "c" { return @{ indexUrl = "https://download.pytorch.org/whl/cu118"; label="cu118" } }
                        default { return @{ indexUrl = "https://download.pytorch.org/whl/cu124"; label="cu124" } }
                    }
                }
            }
            "3" {
                Write-Host "  a) cu128 (CUDA 12.8)  b) cu124 (CUDA 12.4)  c) cu118 (CUDA 11.8)"
                $pick = Read-Host "Pick (a/b/c) [default=b]"
                switch ($pick) {
                    "a" { return @{ indexUrl = "https://download.pytorch.org/whl/cu128"; label="cu128" } }
                    "c" { return @{ indexUrl = "https://download.pytorch.org/whl/cu118"; label="cu118" } }
                    default { return @{ indexUrl = "https://download.pytorch.org/whl/cu124"; label="cu124" } }
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
            try { & $venvPython -c "import behaveai" > $null 2>&1; return ($LASTEXITCODE -eq 0) } catch { return $false }
        }
        return $false
    }

    function Bootstrap {
        Write-Host "== BehaveAI bootstrap: installing Python packages into venv =="

        $pyCmd = Ensure-Python
        Write-Host "Using Python: $pyCmd"

        if (-not (Test-Path $VENV_DIR)) {
            Write-Host "Creating virtualenv at $VENV_DIR..."
            & cmd /c "$pyCmd -m venv `"$VENV_DIR`""
            if ($LASTEXITCODE -ne 0) { throw "Failed to create virtualenv" }
        } else {
            Write-Host "Virtualenv already exists - reusing."
        }

        $venvPython = Join-Path $VENV_DIR "Scripts\python.exe"
        if (-not (Test-Path $venvPython)) { throw "Venv python not found after creation" }

        & $venvPython -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { Write-Warning "pip upgrade returned non-zero exit code" }

        $installChoice = Read-Host "Install BehaveAI and required Python packages now? (Y/N)"
        if ($installChoice.ToUpper() -ne 'Y') { Write-Host "Skipping package installation."; return }

        # Install torch with CUDA selection (needs a custom index, so done separately)
        Write-Host "Checking for NVIDIA GPU..."
        $hasNvidia = Detect-NvidiaGPU
        $torchChoice = Choose-Torch-Wheel -nvidiaPresent:$hasNvidia
        Write-Host "Selected PyTorch build: $($torchChoice.label)"

        try {
            if ($torchChoice.label -eq "cpu") {
                & $venvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
            } else {
                $idx = $torchChoice.indexUrl
                & $venvPython -m pip install --index-url $idx torch torchvision torchaudio
            }
            if ($LASTEXITCODE -ne 0) { Write-Warning "PyTorch install returned non-zero exit code." }
        } catch {
            Write-Warning "PyTorch installation raised an error: $_"
        }

        # Install BehaveAI from the extracted repo directory (where this script lives)
        Write-Host "Installing BehaveAI from $ScriptDir..."
        & $venvPython -m pip install $ScriptDir
        if ($LASTEXITCODE -ne 0) { throw "pip install of BehaveAI failed" }

        Write-Host ""
        Write-Host "Verifying install..."
        try { & $venvPython -c "import behaveai; print('behaveai OK')" | Out-Host } catch { Write-Warning "behaveai import failed - check the log." }
        try { & $venvPython -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())" | Out-Host } catch { Write-Warning "torch import failed." }
        try { & $venvPython -c "import cv2; print('cv2', cv2.__version__)" | Out-Host } catch { Write-Warning "cv2 import failed." }

        New-Item -ItemType File -Force -Path $MARKER | Out-Null
        Write-Host "Bootstrap complete."
    }

    # -------------------------
    # Main flow
    # -------------------------
    $envReady = Is-Ready
    if (-not $envReady) {
        $installPrompt = Read-Host "Environment not ready. Install now? (Y/N)"
        if ($installPrompt.ToUpper() -eq 'Y') {
            Bootstrap
            $envReady = Is-Ready
            if (-not $envReady) { Write-Warning "Environment still not ready after bootstrap. Check: $LogPath" }
        } else {
            Write-Host "User chose not to install. Exiting."
            Stop-Transcript
            exit 2
        }
    } else {
        Write-Host "Environment already ready. Using existing venv at $VENV_DIR"
    }

    # -------------------------
    # Launch behaveai entrypoint
    # -------------------------
    $behaveai = Join-Path $VENV_DIR "Scripts\behaveai.exe"
    if (-not (Test-Path $behaveai)) { throw "behaveai entrypoint not found at $behaveai — did installation complete?" }

    Write-Host "Launching: $behaveai $($RemainingArgs -join ' ')"
    & $behaveai @RemainingArgs
    $exitCode = $LASTEXITCODE
    Write-Host "behaveai exited with code $exitCode"
    Stop-Transcript
    exit $exitCode
}
catch {
    Write-Error "Fatal error: $_"
    Write-Host "See the log at: $LogPath"
    Stop-Transcript
    exit 1
}