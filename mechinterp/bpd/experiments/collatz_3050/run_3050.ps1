<#
.SYNOPSIS
Prepare and run the BPD-COL-02 native Nanbeige gym on Windows.
.EXAMPLE
.\run_3050.ps1 -Action Setup
.\run_3050.ps1 -Action Run -Roots 1 -ForwardBudget 8 -Download
#>
[CmdletBinding()]
param(
    [ValidateSet("Setup", "Doctor", "Prepare", "Run", "Score", "Test")]
    [string]$Action = "Doctor",
    [string]$Python = "py",
    [string]$ModelDir = "",
    [string]$RunDir = "",
    [ValidateSet("trace", "arithmetic")]
    [string]$Tier = "trace",
    [ValidateRange(1, 32)]
    [int]$Roots = 1,
    [ValidateRange(8, 512)]
    [int]$ForwardBudget = 8,
    [ValidateRange(1, 86400)]
    [int]$MaxSeconds = 7200,
    [switch]$Download,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$GymRoot = $PSScriptRoot
$EnvironmentDir = Join-Path $GymRoot ".venv"
$GymPython = Join-Path $EnvironmentDir "Scripts\python.exe"
$EntryPoint = Join-Path $GymRoot "run_gym.py"
$Requirements = Join-Path $GymRoot "requirements.txt"

function Invoke-GymPython {
    param([string[]]$Arguments)
    & $GymPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Gym command exited with code $LASTEXITCODE. Its recorded results remain in the run folder."
    }
}

if ($Action -eq "Setup") {
    if (-not (Test-Path $GymPython)) {
        if ($Python -eq "py") {
            $HasPython311 = $false
            try {
                & $Python -3.11 -c "import sys" 2>$null
                $HasPython311 = ($LASTEXITCODE -eq 0)
            } catch {
                $HasPython311 = $false
            }
            if ($HasPython311) {
                & $Python -3.11 -m venv $EnvironmentDir
            } else {
                & $Python -3.12 -m venv $EnvironmentDir
            }
        } else {
            & $Python -m venv $EnvironmentDir
        }
        if ($LASTEXITCODE -ne 0) { throw "Python 3.11/3.12 environment creation failed." }
    }
    Invoke-GymPython -Arguments @("-c", "import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), 'Use Python 3.11 or 3.12'")
    Invoke-GymPython -Arguments @("-m", "pip", "install", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124")
    Invoke-GymPython -Arguments @("-m", "pip", "install", "-r", $Requirements)
    Write-Output "Environment ready: $GymPython"
    exit 0
}

if (-not (Test-Path $GymPython)) {
    throw "Run this launcher with -Action Setup first. It creates a separate Python environment."
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = Join-Path $GymRoot ("runs\" + $Tier + "-r" + $Roots)
}
$RunDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RunDir)
$SnapshotArguments = @()
if (-not [string]::IsNullOrWhiteSpace($ModelDir)) {
    $SnapshotArguments = @("--model-dir", $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ModelDir))
}
if ($Download) {
    if ($Action -notin @("Doctor", "Run")) {
        throw "Use -Download with -Action Doctor or Run."
    }
    Invoke-GymPython -Arguments (@($EntryPoint, "download") + $SnapshotArguments)
}

switch ($Action) {
    "Doctor" {
        Invoke-GymPython -Arguments (@($EntryPoint, "doctor") + $SnapshotArguments)
    }
    "Prepare" {
        Invoke-GymPython -Arguments @($EntryPoint, "prepare", "--tier", $Tier, "--roots", "$Roots", "--output", (Join-Path $RunDir "input_manifest.json"))
    }
    "Run" {
        # Resume preserves the original cases, runtime, source and tokenization.
        $Manifest = Join-Path $RunDir "input_manifest.json"
        Invoke-GymPython -Arguments @($EntryPoint, "prepare", "--tier", $Tier, "--roots", "$Roots", "--output", $Manifest)
        Invoke-GymPython -Arguments (@($EntryPoint, "doctor", "--output", (Join-Path $RunDir "doctor.json")) + $SnapshotArguments)
        $RunArguments = @($EntryPoint, "run", "--manifest", $Manifest, "--output", $RunDir,
                          "--forward-budget", "$ForwardBudget", "--max-seconds", "$MaxSeconds") + $SnapshotArguments
        if ($Resume) { $RunArguments += "--resume" }
        Invoke-GymPython -Arguments $RunArguments
    }
    "Score" {
        Invoke-GymPython -Arguments @($EntryPoint, "score", "--run", $RunDir)
    }
    "Test" {
        $SourceDir = Join-Path $GymRoot ".test_source"
        Invoke-GymPython -Arguments @($EntryPoint, "fetch-test-source", "--output", $SourceDir)
        $PreviousSourceDir = $env:BPD_COL3050_SOURCE_DIR
        try {
            $env:BPD_COL3050_SOURCE_DIR = $SourceDir
            Invoke-GymPython -Arguments @("-m", "unittest", "discover", "-s", (Join-Path $GymRoot "tests"), "-v")
        } finally {
            $env:BPD_COL3050_SOURCE_DIR = $PreviousSourceDir
        }
    }
}
