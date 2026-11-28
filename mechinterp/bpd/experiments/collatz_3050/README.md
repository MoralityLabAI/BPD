# BPD-COL-02: Nanbeige mechinterp gym for the RTX 3050

Run controlled state interventions through **Nanbeige4.2-3B's native two-pass
recurrence** on a 4-GiB GPU. The backend loads one physical decoder layer at a
time and reads embedding and output-head rows on demand. The initial experiment
asks whether prefix states that preserve the current answer readout can steer a
later answer toward an independently computed Collatz donor target.

**Evidence status:** source and end-to-end CPU tests use the released Nanbeige
architecture with tiny random weights. Pretrained task competence, RTX 3050
peak memory, and laptop latency await the hardware run. Every scientific report
is labelled `PILOT_ONLY`.

The exact oracle comes from [BPD-COL-01](../collatz_observability/README.md).
[Protocol](PROTOCOL.md), [scoring](SCORING.md) and [source pins](SOURCES.md)
describe the experiment and its evidence scope.
The [source-validation receipt](data/source_validation.json) records 31 passing
tests under Torch 2.6.0 CPU with zero skips. The [model inventory](data/model_profile.json)
records the exact parameter counts and weight sizes used for the memory plan.

## Windows quick start

From your BPD checkout, open PowerShell with Python 3.11 and a CUDA-capable
NVIDIA driver installed:

```powershell
Set-Location mechinterp/bpd/experiments/collatz_3050
.\run_3050.ps1 -Action Setup
.\run_3050.ps1 -Action Run -Tier trace -Roots 1 -ForwardBudget 8
```

`Setup` creates a dedicated `.venv` and installs Torch 2.6.0 with CUDA 12.4,
plus the pinned packages in `requirements.txt`. An explicit Python 3.11 or
3.12 executable can be supplied with `-Python C:\path\to\python.exe`.

The launcher looks for the pinned checkpoint in the Hugging Face cache. To use
an existing snapshot at a different location:

```powershell
.\run_3050.ps1 -Action Run -Tier trace -Roots 1 -ForwardBudget 8 `
  -ModelDir 'C:\path\to\Nanbeige4.2-3B\snapshots\451ed48c3273ecef7ea8faaa43c31ce529763bb1'
```

If the snapshot is absent, add `-Download` to the `Run` or `Doctor` command.
That option downloads the exact public revision and verifies its hashes.
Allow 10 GiB free disk for the checkpoint download, plus space for the Python
environment. The weights occupy approximately 7.77 GiB.

```powershell
.\run_3050.ps1 -Action Doctor -Download
.\run_3050.ps1 -Action Test
```

`Doctor` reports the GPU, available memory, dependency versions and checkpoint
inventory. `Test` fetches the pinned architecture source and generates tiny
random weights locally. Its native tests run on the CPU.

The first run creates `runs/trace-r1/`. Open `summary.json` for results and
`sessions/` for resource receipts. The `trace` tier supplies an exact trajectory
in the prompt, which makes this first run a routing/readout competence check.

## Progress through the gym

| Stage | Task | Suggested command after setup |
| --- | --- | --- |
| Hardware and hook smoke | One trace case, eight forwards | `./run_3050.ps1 -Action Run -Tier trace -Roots 1 -ForwardBudget 8` |
| Development controls | Four trace roots, 32 forwards | `./run_3050.ps1 -Action Run -Tier trace -Roots 4 -ForwardBudget 32` |
| Arithmetic transfer | Four arithmetic roots, 32 forwards | `./run_3050.ps1 -Action Run -Tier arithmetic -Roots 4 -ForwardBudget 32` |
| Larger public pilot | Up to 32 roots, eight forwards per case | Set `-Roots` and `-ForwardBudget` before the run |

Inspect the first session's elapsed time and memory receipt before expanding
the budget. Source-parameter roots are the replication unit. More prompt
tokens, recurrent passes or intervention arms leave that root count unchanged.
The generator's population is already public development material.

The PowerShell launcher defaults to one trace root and eight forwards. The
Python `prepare` command defaults to four trace roots; the committed
`data/development_cases.json` is the four-root **arithmetic** fixture. Each
manifest records its tier explicitly.

To freeze four cases while running only the first one initially:

```powershell
.\run_3050.ps1 -Action Run -Tier trace -Roots 4 -ForwardBudget 8
.\run_3050.ps1 -Action Run -Tier trace -Roots 4 -ForwardBudget 24 -Resume
```

Resume continues unattempted cases under the same frozen contract. Completed
cases, errors and skips remain recorded attempts. Use a fresh `-RunDir` for a
deliberate rerun. Changing the tier, case list, intervention dose, source,
runtime versions or rendered prompts requires a fresh run directory.

## What the experiment does

For each exact arithmetic root, the generator selects a recipient, an effect
donor with the opposite future branch label, and a sham donor with the same
future branch label. Their current arithmetic quotient and branch match.
Most selected triplets share a basin; `d=13` provides a separately reported
cross-basin calibration.

Nanbeige executes the prompt through 22 physical layers twice. At the first
native `model.norm` output, the gym captures the entire prompt state. During
intervention runs it edits the earlier token positions and preserves the final
prompt position bit for bit. The unchanged vocabulary head therefore has the
same current readout at that position. Attention in the second pass can use
the edited prefix.

Each case costs eight complete native forwards:

| Forward group | Purpose |
| --- | --- |
| Clean recipient and two clean donors | Capture natural donor states and measure baseline task competence |
| Identity patch | Check replay and instrumentation |
| Effect donor direction | Test steering toward the donor's opposite future label |
| Sham donor direction | Measure disruption along a same-target donor direction |
| Deterministic random direction | Measure generic state sensitivity at the same delivered norm |
| Full donor prefix | Record the larger raw donor substitution as a separate control |

The matched effect, sham and random doses are 10% of the smaller donor-prefix
distance. A bounded scalar search matches the actual displacement after BF16
rounding. Eligibility checks the measured norms within a 2% relative allowance
plus `1e-6` absolute. The full donor prefix has its own dose and stays outside
the matched primary comparison.

Both task tiers score the next token from declared continuations `" A"` and
`" B"`. Probabilities condition on these two answers. The pinned tokenizer
gives 96 tokens for the default trace prompts and 121 for arithmetic; all
matched triplets have equal token counts. The runner checks the complete
continuation boundary before executing a case.

The primary development statistic measures the increase in probability of the
effect donor's label, subtracting the larger absolute shift from the sham and
random controls. Cases enter that statistic only after all clean prompts are
correct and instrument checks pass. Root means receive equal weight. The
summary also shows competence failures, missing cases, arm-specific effects,
recipient correctness and the separate basin strata. [SCORING.md](SCORING.md)
gives the equations and denominators.

## Memory and I/O contract

| Setting | Registered profile |
| --- | ---: |
| Checkpoint | `Nanbeige/Nanbeige4.2-3B` at `451ed48c3273ecef7ea8faaa43c31ce529763bb1` |
| Actual total parameter count | 4,169,800,704 |
| Physical decoder layers / native passes | 22 / 2 |
| Resident decoder weights | One layer, approximately 273 MiB in BF16 |
| Prompt batch / length cap | 1 / 128 tokens |
| Torch CUDA allocation and reservation cap | 1.75 GiB |
| Added sampled process RSS cap | 2.75 GiB |
| Free memory requested by `Doctor` | 2 GiB GPU and 4 GiB system RAM |
| Decoder bytes streamed per two-pass forward | 12,596,035,584 bytes, approximately 11.73 GiB |
| Native forwards per case | 8 |

The low resident weight count entails substantial I/O. A one-case run requests
about 93.85 GiB of decoder tensor reads across its eight forwards; operating
system caching determines how much reaches storage. Startup also verifies both
complete checkpoint shards. Use the actual session timings to budget larger
runs.

The CUDA limit covers Torch allocator bytes. Driver and display allocations
sit outside that measurement. Host RSS is sampled at declared loading and
execution boundaries, so short-lived peaks between samples are outside its
receipt. This profile establishes explicit refusal conditions; the laptop run
must establish the observed device fit.

The wall-time budget is checked between forwards. A forward already executing
may finish after the threshold. The runner reserves an eight-forward allowance
before starting a case and records any interruption during that case.

## Saved evidence

| File | Contents |
| --- | --- |
| `run_contract.json` | Model revision, source hashes, runtime versions, seed, observer, dose and resource limits |
| `cases_manifest.json` | Exact oracle cases, trajectories, labels and selection policy |
| `tokenized_prompts.json` | Rendered prompts, input IDs and verified answer token IDs |
| `records.jsonl` | Every attempted case with control scores, patch checks and native layer schedules |
| `activations/*.safetensors` | Three clean first-pass states, clean final query vectors and each arm's final query |
| `sessions/*.json` | GPU identity, attempted forwards, elapsed time, memory and cleanup/error receipts |
| `summary.json` | Recomputed competence, eligibility, per-root results and pilot verdict |

Partial scores survive a later forward failure in the case's error record.
The `score` command checks frozen scorer hashes before replacing the summary.
Runtime outputs and model weights are ignored by Git.

```powershell
.\run_3050.ps1 -Action Score -Tier trace -Roots 1
```

## Python entry points

```bash
python run_gym.py prepare --tier trace --roots 1 --output runs/trace1/input.json
python run_gym.py doctor --model-dir /path/to/pinned/snapshot
python run_gym.py run --manifest runs/trace1/input.json --output runs/trace1 \
  --model-dir /path/to/pinned/snapshot --forward-budget 8
python run_gym.py score --run runs/trace1
```

Case preparation and scoring use the Python standard library. Native
architecture tests require the pinned runtime and source:

```bash
python run_gym.py fetch-test-source --output .test_source
BPD_COL3050_SOURCE_DIR=.test_source python -m unittest discover -s tests -v
```

The CI workflow installs Torch 2.6.0 CPU and fails if a native test is skipped.
It checks BF16 native/streamed parity, complete and selected vocabulary heads,
protected prefix patches, cleanup, root scoring and the full case runner.

## Relation to BPD

This gym supplies native recurrent state interventions and exact task labels
for BPD's observer program. Its immediate mechanism is prefix context carried
across the first loop boundary. The observer covers the answer position;
readouts at earlier positions may change during a patch. Donor states come
from ordinary prompt executions, while the interpolated combined states are
constructed interventions.

The package performs inference with frozen checkpoint weights. A future BPD
factor experiment can test which shared weight factors mediate an eligible
state effect, using pass-specific factor controls and fresh root planning.
The present receipts support task competence and causal state measurements
within this public pilot. Population-level mechanisms and a calibrated
95th-percentile sham floor require a separate, larger design.

The alternate Dory architecture needs its own recurrence and output-head
adapter. Nanbeige is the implemented model for this gym; [SOURCES.md](SOURCES.md)
records the audit behind that choice.
