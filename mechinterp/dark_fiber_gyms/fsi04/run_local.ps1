<#
.SYNOPSIS
  Run the source-locked BPD FSI-04 development pilot on local TRM weights.
.DESCRIPTION
  Requires explicit local checkpoint paths, reads the fresh public development
  roster, and runs EvalMath resolution preflight. No weight downloads.
  The protected ARG-03 168-case test remains unopened.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$YagizCheckpoint,
  [Parameter(Mandatory=$true)][string]$GaoxinCheckpoint,
  [Parameter(Mandatory=$true)][string]$EvalMathRoot,
  [string]$PythonExe="python",
  [string]$Device="cuda:0",
  [string]$RosterDir="runs/fsi04/roster",
  [string]$OutputDir="runs/fsi04/native",
  [long]$ArtifactRunId=0,
  [switch]$ReuseExistingResults
)
$ErrorActionPreference="Stop"
$root=(Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Push-Location $root
try {
  $registeredFile=Join-Path $PSScriptRoot "REGISTRATION.json"
  $registered=Get-Content -Raw -LiteralPath $registeredFile | ConvertFrom-Json
  if ($registered.schema -ne "BPD-FSI04/native-causal-dose-matched-development-pilot/v1") { throw "FSI04 registration changed" }
  if ($Device -notmatch "^(cuda(:[0-9]+)?|cpu)$") { throw "Invalid computation device" }
  $roster=Join-Path $root $RosterDir
  $out=Join-Path $root $OutputDir
  New-Item -ItemType Directory -Force -Path $roster,$out | Out-Null
  $csv=Join-Path $roster "development.csv"
  $manifest=Join-Path $roster "development_manifest.json"
  if (!(Test-Path -LiteralPath $csv) -or !(Test-Path -LiteralPath $manifest)) {
    if ($ArtifactRunId -gt 0) {
      if (!(Get-Command gh -ErrorAction SilentlyContinue)) { throw "gh CLI unavailable; download the full-roster Actions artifact manually" }
      & gh run download $ArtifactRunId --repo MoralityLabAI/BPD --name bpd-fsi04-full-36-development-roots --dir $roster
      if ($LASTEXITCODE -ne 0) { throw "Artifact retrieval failed" }
    } else {
      & $PythonExe -m mechinterp.dark_fiber_gyms.fsi04.roster --out-dir $roster --max-attempts 100000
      if ($LASTEXITCODE -ne 0) { throw "Fresh roster generation failed" }
    }
  }
  if (!(Test-Path -LiteralPath $csv) -or !(Test-Path -LiteralPath $manifest)) { throw "Missing fresh unscored full roster" }
  $m=Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json
  $csvSha=(Get-FileHash -Algorithm SHA256 -LiteralPath $csv).Hash.ToLowerInvariant()
  $regSha=(Get-FileHash -Algorithm SHA256 -LiteralPath $registeredFile).Hash.ToLowerInvariant()
  if ($m.schema -ne "BPD-FSI04/development-roster/v1" -or $m.registration_sha256 -ne $regSha -or $m.csv_sha256 -ne $csvSha -or $m.counts.S4_stalled -ne 24 -or $m.counts.'WB-Prop' -ne 12 -or $m.ARG03_protected_test -ne "untouched") {
    throw "Development roster does not match preregistered SHA, counts, or test seal"
  }

  $paths=@{ "yagizdevre"=(Resolve-Path -LiteralPath $YagizCheckpoint).Path; "gaoxin492"=(Resolve-Path -LiteralPath $GaoxinCheckpoint).Path }
  foreach ($tag in @("yagizdevre","gaoxin492")) {
    $expected=($registered.models | Where-Object {$_.tag -eq $tag} | Select-Object -First 1).sha256
    $actual=(Get-FileHash -Algorithm SHA256 -LiteralPath $paths[$tag]).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Checkpoint SHA mismatch for $tag" }
    Write-Host "SHA locked $tag checkpoint verified"
  }

  foreach ($tag in @("yagizdevre","gaoxin492")) {
    $report=Join-Path $out "$tag-development.json"
    if ((Test-Path -LiteralPath $report) -and $ReuseExistingResults) { continue }
    & $PythonExe -m mechinterp.dark_fiber_gyms.fsi04.native_pilot --repo-root . --checkpoint-file $paths[$tag] --tag $tag --eval-csv $csv --roster-manifest $manifest --exclude-csv "mechinterp/loopscope02/results/benchmark_bundle.json" --out $report --device $Device --limit 0
    if ($LASTEXITCODE -ne 0) { throw "Native FSI04 checkpoint evaluation failed: $tag" }
  }

  $preflight=Join-Path $EvalMathRoot "skills/evalmath-mechinterp-power/scripts/preflight.py"
  if (!(Test-Path -LiteralPath $preflight)) { throw "EvalMath skill missing" }
  $env:EVALMATH_PREFLIGHT_FILE=(Resolve-Path -LiteralPath $preflight).Path
  $power=Join-Path $out "fsi04-development-power.json"
  & $PythonExe -m mechinterp.dark_fiber_gyms.fsi04.pilot_power --yagiz (Join-Path $out "yagizdevre-development.json") --gaoxin (Join-Path $out "gaoxin492-development.json") --roster-manifest $manifest --out $power
  if ($LASTEXITCODE -ne 0) { throw "FSI04 power audit rejected the native receipts" }
  $decision=(Get-Content -Raw -LiteralPath $power | ConvertFrom-Json).decision
  Write-Host "FSI04 development decision: $decision"
  Write-Host "Source-locked report: $power"
} finally {
  Pop-Location
}
