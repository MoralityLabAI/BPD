<#
BASIS-01-SCALE: local SHA-pinned TRM clean-input capture and exact-basis scoring.
Execute from any directory; all files are created under the chosen workspace.
Neither native checkpoints nor protected evaluation data are downloaded.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('gaoxin492','yagizdevre')]
    [string]$CheckpointKey,

    [Parameter(Mandatory=$true)]
    [string]$CheckpointFile,

    [string]$Workspace = 'runs/bpd_v3',

    [ValidateRange(1,4096)]
    [int]$ChunkSize = 256,

    [ValidateRange(0,183)]
    [int]$MaxBoards = 0,

    [ValidateSet('all','input','output')]
    [string]$Family = 'all',

    [ValidateSet('bf16','float32')]
    [string]$DType = 'bf16',

    [string]$Device = 'cuda:0'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not [System.IO.Path]::IsPathRooted($Workspace)) {
    $Workspace = Join-Path $repoRoot $Workspace
}
$Workspace = [System.IO.Path]::GetFullPath($Workspace)
New-Item -ItemType Directory -Path $Workspace -Force | Out-Null
$CheckpointFile = (Resolve-Path -LiteralPath $CheckpointFile).Path
$rosterFile = Join-Path $Workspace 'wb_stall_inputs.npz'
$rosterManifest = Join-Path $Workspace 'wb_stall_inputs.manifest.json'
$bundleFile = Join-Path $repoRoot 'mechinterp/loopscope02/results/benchmark_bundle.json'
$prepareScript = Join-Path $repoRoot 'mechinterp/bpd/v3/prepare_native_roster.py'
$captureScript = Join-Path $repoRoot 'mechinterp/bpd/v3/collect_basis_inputs.py'
$scoreScript = Join-Path $repoRoot 'mechinterp/bpd/v3/score_exported_basis.py'
$exportDir = Join-Path $Workspace ($CheckpointKey + '_basis_inputs')
$scoreDir = Join-Path $Workspace ($CheckpointKey + '_basis_scores')

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python exited $LASTEXITCODE when running $($Arguments[0])"
    }
}

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $rosterFile) -or -not (Test-Path -LiteralPath $rosterManifest)) {
        Invoke-CheckedPython -Arguments @(
            $prepareScript, '--bundle-json', $bundleFile,
            '--out', $rosterFile
        )
    }
    $roster = Get-Content -Raw -LiteralPath $rosterManifest | ConvertFrom-Json
    if ([int]$roster.n_boards -ne 183) {
        throw 'Expected exactly 183 frozen WB-Stall roots'
    }
    $rosterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $rosterFile).Hash.ToLowerInvariant()
    $bundleHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundleFile).Hash.ToLowerInvariant()
    if ($rosterHash -ne ([string]$roster.roster_npz_sha256).ToLowerInvariant()) {
        throw 'Roster NPZ SHA-256 mismatches its reviewed manifest'
    }
    if ($bundleHash -ne ([string]$roster.bundle_json_sha256).ToLowerInvariant()) {
        throw 'Benchmark bundle SHA-256 mismatches roster origin manifest'
    }

    $captureArgs = @(
        $captureScript, '--checkpoint-file', $CheckpointFile,
        '--checkpoint-key', $CheckpointKey,
        '--inputs-npz', $rosterFile,
        '--expected-roster-sha256', $rosterHash,
        '--out-dir', $exportDir,
        '--device', $Device, '--dtype', $DType,
        '--positions-per-call', '2'
    )
    if ($MaxBoards -gt 0) {
        $captureArgs += @('--max-boards', [string]$MaxBoards)
    }
    Invoke-CheckedPython -Arguments $captureArgs
    $captured = Get-Content -Raw -LiteralPath (Join-Path $exportDir 'manifest.json') | ConvertFrom-Json
    if ([string]$captured.checkpoint_key -ne $CheckpointKey) {
        throw 'Captured checkpoint label mismatch'
    }
    if ([string]$captured.roster_sha256 -ne $rosterHash) {
        throw 'Captured roster hash mismatch'
    }
    if ([int]$captured.n_passes -ne 63 -or [int]$captured.positions_per_call -ne 2) {
        throw 'Unexpected capture schedule'
    }
    if ($captured.sites.PSObject.Properties.Count -ne 4) {
        throw 'Expected four physical tied projection sites'
    }

    New-Item -ItemType Directory -Path $scoreDir -Force | Out-Null
    foreach ($site in $captured.sites.PSObject.Properties) {
        $name = $site.Name
        $metadata = $site.Value
        $npzPath = Join-Path $exportDir ($name.Replace('.', '_') + '.npz')
        $expectedHash = ([string]$metadata.npz_sha256).ToLowerInvariant()
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $npzPath).Hash.ToLowerInvariant()
        if ($expectedHash -ne $actualHash) {
            throw "Module export SHA mismatch: $name"
        }
        $output = Join-Path $scoreDir ($name.Replace('.', '_') + '_basis.json')
        Invoke-CheckedPython -Arguments @(
            $scoreScript, '--matrix-npz', $npzPath,
            '--expected-matrix-sha256', $expectedHash,
            '--out', $output, '--chunk-size', [string]$ChunkSize,
            '--family', $Family
        )
        Write-Host "Scored $name -> $output"
    }
    Write-Host "BASIS-01 module scores completed for $CheckpointKey."
    Write-Host "Evidence status: DEVELOPMENT_ONLY. Full recurrent digit-head tests remain pending."
}
finally {
    Pop-Location
}
