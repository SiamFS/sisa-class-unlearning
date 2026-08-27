# Bootstraps the project's virtual environment (see setup_env.py).
# Usage: right-click "Run with PowerShell", or from a terminal:  .\setup.ps1
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 "$scriptDir\setup_env.py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python "$scriptDir\setup_env.py"
} else {
    Write-Error "Python was not found on PATH. Install Python 3.9+ from https://www.python.org/downloads/ and re-run this script."
    exit 1
}
