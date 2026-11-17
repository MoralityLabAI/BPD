# BPD-FSI04 full-roster and local native execution handoff

**Stage:** public unscored development-roster build plus a locally executable two-checkpoint runner. The immutable [FSI-04 registration](mechinterp/dark_fiber_gyms/fsi04/REGISTRATION.json) remains unchanged, and the ARG-03 protected 168-case test remains sealed.

## Hosted CPU construction

The [full-roster workflow](.github/workflows/bpd-fsi04-full-roster.yml) generates the registered 24 genuinely stalled Sudoku roots and 12 propagation controls from the fixed MRV generator seed. It checks unique source-board IDs, 36 scored-root denominator, solver uniqueness and full Sudoku isomorphism against the accessible earlier LOOPSCOPE-02 corpus. The historic 120 LOOPSCOPE-04 boards are available only as SHA-256 IDs; exact hash collisions can be checked, full-grid isomorphism against those unavailable source grids cannot be established.

The workflow publishes an artifact named `bpd-fsi04-full-36-development-roots` containing the **unscored** `development.csv` and `development_manifest.json`. The manifest carries the unchanged registration digest and CSV digest; any future GPU scoring rejects an altered roster or a changed checksum.

Check [GitHub Actions runs](https://github.com/MoralityLabAI/BPD/actions/workflows/bpd-fsi04-full-roster.yml) for the successful build and its run ID, then download the artifact with:

```bash
gh run download RUN_ID --repo MoralityLabAI/BPD \
  --name bpd-fsi04-full-36-development-roots --dir runs/fsi04/roster
```

Generating these puzzles takes CPU work and may fail closed if the accepted S4-stalled yield is too low for the frozen 100,000-attempt cap. A failure is a roster-generation/conditioning result, not evidence about causal neural effects. Never substitute old public evaluation boards to fill missing independent roots.

## Local Windows/GPU procedure

The supported one-command entrypoint is
[`mechinterp/dark_fiber_gyms/fsi04/run_local.ps1`](mechinterp/dark_fiber_gyms/fsi04/run_local.ps1).
Invoke from any folder using PowerShell 7 with explicit **existing** pinned checkpoint files:

```powershell
pwsh ./mechinterp/dark_fiber_gyms/fsi04/run_local.ps1 -YagizCheckpoint 'D:\models\yagizdevre\model.pt' -GaoxinCheckpoint 'D:\models\gaoxin492\step_21700' -EvalMathRoot 'D:\research\EvalMath' -PythonExe 'D:\envs\bluebeam\Scripts\python.exe' -ArtifactRunId RUN_ID -Device cuda:0
```

The command checks both model SHA-256 identities against the registration, validates the 24+12 roster counts and both file hashes, runs every arm on both models, and writes raw per-board native receipts and the independent-world power readout under `runs/fsi04/native/`. It refuses missing local weights and never downloads weights. Without `-ArtifactRunId`, it uses the same frozen roster generator locally, so no external GitHub CLI is required.

## Inference and evidence boundary

The primary test remains donor joint transplant **minus clean** correct Sudoku cells at horizon 32 on 24 S4 boards. The fully dose-matched random joint sham measures nonspecific corruption. The optional blank-cell score is a clearly recorded secondary metric; changing the primary to it requires a new prospective registration.

Even after fresh development scoring, the EvalMath preflight must label the 24-root result **development-only**, because its 95th-percentile/95%-confidence sham-tail criterion needs at least 59 independent calibration roots. The planned 256-root confirmation would require separate sealed sampling, fresh calibration and frozen decisions. No test-set generalization or mechanistic BPD victory can be claimed from a single root-matched pilot, and the old data already show an apparent donor-over-random advantage arising from less harmful donor corruption.

## Exploratory resolution estimate

The prior \`yagizdevre\` S4 root-wise *donor-minus-clean* correct-cell effect had standard deviation approximately **0.1659** across 40 already-published roots. Using EvalMath's normal-approximation planning formula with familywise alpha 0.05 across two comparisons, 80% power, and a target 0.15 effect exceeding a hypothetical 0.10 sham/scientific floor by 0.05 requires about **105 independent roots** if that historical variance persists. At only 24 roots, the corresponding approximate resolvable effect is **0.204**, above the registered target. At 256 roots the same calculation gives **0.132**. Those estimates are *sensitivity planning*, using previously selected mismatched-sham data; they have no coverage guarantee and cannot calibrate the new matched-sham null.

The more important physical test is whether the new displacement-matched random sham's 95th-percentile variation can be bounded below the target. The prescribed tolerance bound still requires at least 59 independent **new calibration** roots. A large first-successor cosine or readout L2 response does not override either gate.

## Existing compute environment

The current ChatGPT tool environment exposes CPU-only PyTorch and contains neither of the registered native 7M model checkpoints. It can generate and audit Sudoku roots, recompute historical FSI-03 probabilities, and source-check the launcher. True native continuation must execute on a machine with the required model weights or another explicitly configured model runtime.
