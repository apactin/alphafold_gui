#requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "`n==> $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-Err ($msg) { Write-Host "[err]  $msg" -ForegroundColor Red }

Write-Info "Windows helper for AF3 WSL setup (Ubuntu 22.04 + CUDA + Docker GPU)"

# --- WSL presence ---
try {
  $wslVersion = wsl.exe --version 2>$null
  Write-Info "WSL detected."
} catch {
  Write-Warn "WSL not detected via 'wsl --version'. On Windows 10, this command may not exist."
  Write-Warn "If needed: enable WSL + Virtual Machine Platform in Windows Features, then reboot."
}

# --- Ubuntu presence ---
$distros = (wsl.exe -l -q 2>$null) | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
if (-not $distros -or ($distros -notcontains "Ubuntu-22.04")) {
  Write-Warn "Ubuntu-22.04 not found in: wsl -l"
  Write-Warn "Install it from the Microsoft Store:"
  Write-Warn "  https://apps.microsoft.com/detail/9pn20msr04dw"
} else {
  Write-Info "Ubuntu-22.04 is installed."
}

# --- VHDX expansion helper ---
Write-Info "VHDX expansion helper (optional)"
Write-Host "If you want to expand Ubuntu's virtual disk, this script can generate a DiskPart script."
Write-Host "You STILL must run DiskPart as admin."

# Default path based on your guide (user-specific!)
$defaultVhdx = "C:\Users\olive\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx"
$VhdxPath = $defaultVhdx

if (-not (Test-Path $VhdxPath)) {
  Write-Warn "Default VHDX path not found:"
  Write-Warn "  $VhdxPath"
  Write-Warn "Edit this script or pass your correct VHDX path."
}

$diskpartScriptPath = Join-Path $PSScriptRoot "diskpart_expand_vhdx.dps"

# 1,500,000 MB ~ 1.5 TB (your guide value). Adjust as needed.
$maxMB = 1500000

@"
select vdisk file="$VhdxPath"
expand vdisk maximum=$maxMB
exit
"@ | Set-Content -Path $diskpartScriptPath -Encoding ASCII

Write-Info "Generated DiskPart script:"
Write-Host "  $diskpartScriptPath"
Write-Host ""
Write-Host "To run (Admin PowerShell):"
Write-Host "  wsl --shutdown"
Write-Host "  diskpart /s `"$diskpartScriptPath`""
Write-Host "  wsl"
Write-Host "Then INSIDE Ubuntu:"
Write-Host "  sudo resize2fs /dev/sdX   (use: lsblk to find the right device)"
Write-Host "  df -h /"

Write-Info "Next: run the WSL installer script inside Ubuntu"
Write-Host "1) Open Ubuntu-22.04"
Write-Host "2) cd into your repo install folder"
Write-Host "3) chmod +x install/setup_wsl.sh"
Write-Host "4) ./install/setup_wsl.sh"
