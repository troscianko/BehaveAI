<#
  BehaveAI - Windows_Uninstaller_ps.ps1 -- safe interactive uninstaller
  Removes:
    - the virtualenv directory (default: %USERPROFILE%\ultralytics-venv)
    - the marker file inside the venv (.behaveai_ready)
    - the launcher and uninstaller logs
  Does NOT remove any source files, scripts, or system Python.
#>

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$log = Join-Path $scriptDir "Windows_Uninstaller.log"
if (Test-Path $log) { Remove-Item $log -Force -ErrorAction SilentlyContinue }
Start-Transcript -Path $log -Force

try {
    Write-Host "=== Windows_Uninstaller_ps.ps1 ==="
    Write-Host "This will remove the Python virtual environment, marker files, and logs."
    Write-Host "It will NOT remove source files/scripts or system Python."
    Write-Host ""

    $defaultVenv = Join-Path $env:USERPROFILE "ultralytics-venv"
    $venvPath = $defaultVenv

    if (-not (Test-Path $venvPath)) {
        Write-Host "Virtualenv not found at default location: $venvPath" -ForegroundColor Yellow
        $manual = Read-Host "Enter the full path to your venv (or press Enter to skip)"
        if ($manual) {
            if (Test-Path $manual) { $venvPath = $manual }
            else { Write-Host "Path does not exist. Aborting." -ForegroundColor Red; Stop-Transcript; exit 1 }
        }
    }

    $marker       = Join-Path $venvPath ".behaveai_ready"
    $launcherLog  = Join-Path $scriptDir "Windows_Launcher_ps.log"
    $installerLog = Join-Path $scriptDir "Windows_Uninstaller.log"

    Write-Host "Planned actions:"
    if (Test-Path $venvPath)     { Write-Host " - Virtualenv directory: $venvPath" }
    if (Test-Path $marker)       { Write-Host " - Marker file: $marker" }
    if (Test-Path $launcherLog)  { Write-Host " - Launcher log: $launcherLog" }

    Write-Host ""
    $proceed = Read-Host "Continue and remove the items above? (Y/N)"
    if ($proceed.ToUpper() -ne 'Y') { Write-Host "Aborting."; Stop-Transcript; exit 0 }

    # Check for running venv Python processes
    if (Test-Path $venvPath) {
        $venvPy = Join-Path $venvPath "Scripts\python.exe"
        if (Test-Path $venvPy) {
            try {
                $procs = Get-CimInstance Win32_Process | Where-Object {
                    $_.ExecutablePath -and ($_.ExecutablePath -ieq $venvPy)
                }
                if ($procs) {
                    Write-Host "Found $($procs.Count) running venv Python process(es)."
                    $kill = Read-Host "Kill them before removing the venv? (Y/N)"
                    if ($kill.ToUpper() -eq 'Y') {
                        foreach ($p in $procs) {
                            try { Stop-Process -Id $p.ProcessId -Force; Write-Host "Killed PID $($p.ProcessId)" }
                            catch { Write-Warning "Could not kill PID $($p.ProcessId): $_" }
                        }
                        Start-Sleep -Seconds 1
                    } else {
                        Write-Host "Close those processes and re-run the uninstaller."
                        Stop-Transcript; exit 1
                    }
                } else {
                    Write-Host "No running venv processes found."
                }
            } catch { Write-Warning "Error checking processes: $_" }
        }

        $delVenv = Read-Host "Remove virtualenv directory '$venvPath'? (Y/N)"
        if ($delVenv.ToUpper() -eq 'Y') {
            Write-Host "Removing virtualenv..."
            Remove-Item -LiteralPath $venvPath -Recurse -Force -ErrorAction Stop
            Write-Host "Virtualenv removed."
        } else {
            Write-Host "Skipped virtualenv removal."
        }
    } else {
        Write-Host "Virtualenv not present; nothing to remove."
    }

    if (Test-Path $marker) {
        try { Remove-Item -LiteralPath $marker -Force -ErrorAction SilentlyContinue; Write-Host "Marker removed." }
        catch { Write-Warning "Could not remove marker: $_" }
    }

    if (Test-Path $launcherLog) {
        $delLog = Read-Host "Remove launcher log '$launcherLog'? (Y/N)"
        if ($delLog.ToUpper() -eq 'Y') { Remove-Item -LiteralPath $launcherLog -Force -ErrorAction SilentlyContinue; Write-Host "Launcher log removed." }
    }

    Write-Host ""
    Write-Host "Uninstall complete. Log saved to: $log"
    Write-Host "Source files and scripts were not touched."
    Stop-Transcript
    exit 0
}
catch {
    Write-Error "Uninstall failed: $_"
    Write-Host "See log: $log"
    Stop-Transcript
    exit 1
}