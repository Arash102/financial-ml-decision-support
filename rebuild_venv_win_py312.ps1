param()

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsFile = Join-Path $ProjectRoot "requirements-win-py312.lock.txt"

function Run-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host ">" $Executable ($Arguments -join " ")

    & $Executable @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path $RequirementsFile)) {
    throw "Missing lock file: $RequirementsFile"
}

Write-Host "Project root: $ProjectRoot"

if (Test-Path $VenvPath) {
    Write-Host "Removing old virtual environment..."
    Remove-Item -Recurse -Force $VenvPath
}

$PyLauncher = Get-Command "py" -ErrorAction SilentlyContinue

if ($null -eq $PyLauncher) {
    throw "The Windows Python launcher 'py' was not found."
}

Write-Host "Checking Python 3.12..."
Run-Checked "py" "-3.12" "-c" "import sys, struct; print(sys.version); print('bits=', struct.calcsize('P')*8); assert sys.version_info[:2] == (3, 12); assert struct.calcsize('P')*8 == 64"

Write-Host "Creating a clean .venv..."
Run-Checked "py" "-3.12" "-m" "venv" $VenvPath

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python was not created: $VenvPython"
}

Write-Host "Bootstrapping pip..."
Run-Checked $VenvPython "-m" "ensurepip" "--upgrade"

Write-Host "Installing stable pip tooling..."
Run-Checked $VenvPython "-m" "pip" "install" `
    "--disable-pip-version-check" `
    "--no-cache-dir" `
    "--index-url" "https://pypi.org/simple" `
    "pip==24.3.1" `
    "setuptools==75.6.0" `
    "wheel==0.45.1"

Write-Host "Installing the locked project environment..."
Run-Checked $VenvPython "-m" "pip" "install" `
    "--disable-pip-version-check" `
    "--no-cache-dir" `
    "--index-url" "https://pypi.org/simple" `
    "-r" $RequirementsFile

Write-Host "Checking dependency consistency..."
Run-Checked $VenvPython "-m" "pip" "check"

Write-Host "Registering the Jupyter kernel..."
Run-Checked $VenvPython "-m" "ipykernel" "install" `
    "--user" `
    "--name" "financial-ml-decision-support" `
    "--display-name" "Financial ML Decision Support (.venv)"

$VerifyPath = Join-Path $env:TEMP "verify_financial_ml_env.py"

$VerifyCode = @'
import sys
import numpy
import pandas
import yaml
import scipy
import sklearn
import xgboost
import optuna
import joblib
import statsmodels
import matplotlib
import typing_extensions
import ipykernel

print("Environment verification: PASSED")
print("Python:", sys.version)
print("Executable:", sys.executable)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("optuna:", optuna.__version__)
print("statsmodels:", statsmodels.__version__)
print("matplotlib:", matplotlib.__version__)
print("ipykernel:", ipykernel.__version__)
'@

Set-Content -Path $VerifyPath -Value $VerifyCode -Encoding UTF8

try {
    Write-Host "Verifying imports..."
    Run-Checked $VenvPython $VerifyPath
}
finally {
    Remove-Item $VerifyPath -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Setup completed successfully."
Write-Host "Select this kernel in VS Code:"
Write-Host "Financial ML Decision Support (.venv)"
