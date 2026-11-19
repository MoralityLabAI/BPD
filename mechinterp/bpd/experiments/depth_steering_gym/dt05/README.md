# DT05: useful memory, autonomous routing, and valid Sudoku stopping

DT05 adds runnable development instruments beside the frozen DT04 campaign at
`1082496445cb99e85cf346597aa87d4cb28f7059`. It addresses the uploaded
`trm-SodokuGymReport.pdf` and its underlying source. The existing DT04 code,
execution lock, and results retain their original bytes.

Read [the report review](REVIEW_DT04.md) for the evidence and discrepancies.
[The development protocol](PROTOCOL_DT05.md) describes the allowed population
and comparison limits. GPU performance and new model accuracy still require
measurements on the target machine.

## What changed

| Technique | Behavior | Why it belongs here |
|---|---|---|
| First-valid stopping | Check digits, public clues, rows, columns, and boxes after every ACT step. Freeze valid answers and compact unfinished cases. | DT04's 7M bridge sometimes destroyed earlier correct answers. |
| Autonomous sparse factors | Select components using the compressed recurrence's current input. Keep only V/U and optional bias at replaced sites. | DT04 selected masks from a full teacher trajectory through the requested evaluation horizon. Its old component path also computed a dense residual before applying a zero residual mask. |
| Fixed and seeded-random controls | Reuse exactly the fitted factors with different component selection rules. | Measure whether current-input routing contributes beyond a fixed factor subset. |
| EMA | Keep a moving average after each optimizer update. Report final raw and EMA checkpoints. | A fixed final EMA provides a testable training variant without choosing a checkpoint on evaluation answers. |
| Thin head projection and JVP transport | Carry a small basis through one native H/L recurrence, report rank loss and leakage. | DT04 measured severe static-subspace drift. This implementation avoids a full state Jacobian or a square state Gramian. |
| Train-target RAM cache | Cache teacher logits and zL states at fit depths 2/4/6, keyed by weights, model recipe, and training tensors. | Repeated factor updates can reuse teacher work. Cache rows are independent copies; labels and masks from evaluation never enter it. |
| Measured microbatch selection | Warm up, repeat, synchronize CUDA, and choose the best measured examples/second within declared reserves. | Larger physical batches can use idle capacity while gradient accumulation preserves a fixed logical batch. |
| Current memory readings | Report current RSS separately from process peak, system availability, CUDA allocation/reservation, and device free bytes. | All 18 archived DT04 hardware RSS readings were zero. Missing required readings now stop admission. |

Online top-k still computes dense small-component matrix products. Twelve active
components out of 24 does not establish a 2x speedup or halve total memory.
The geometry helper is a local derivative instrument. It does not yet deliver
or establish an improved causal steering intervention.

## Setup and source checks

Use the existing BPD Python environment with `torch>=2.5`, NumPy, and `psutil`.
The supplied CI uses Python 3.12 and CPU Torch 2.5.1. CUDA training requires a
Torch build that supports your device. No command below downloads model weights.

```bash
python -m pip install psutil
python -m mechinterp.bpd.experiments.depth_steering_gym.dt04.verify_source --skip-runtime
python -m unittest discover -s mechinterp/bpd/experiments/depth_steering_gym/dt05 -t . -p 'test_*.py' -v
```

`--skip-runtime` verifies historical source bytes while allowing the DT05 test
environment to differ from DT04's original Torch 2.11.0+cu128 environment. The
DT04 execution runner and its own runtime requirement remain unchanged.

An excluded engineering run exercises native training, EMA, cached factor
distillation, every routing arm, compact stopping, and the transport instrument:

```bash
python -m mechinterp.bpd.experiments.depth_steering_gym.dt05.runner smoke --out runs/dt05/smoke --device cpu
```

Its synthetic fixture has eight cases and two optimizer updates. Its accuracy is
an implementation diagnostic. It creates zero scientific roots. Use a fresh
output directory for another smoke attempt.

## Use RAM and VRAM according to measured work

The default resource envelope preserves the DT04 process budget and adds reserves:

| Setting | Default |
|---|---:|
| Maximum process RSS | 12 GiB |
| Minimum available host memory | 4 GiB |
| Maximum CUDA reservation | 12.5 GiB |
| Minimum free CUDA memory | 1 GiB |
| Maximum owned teacher-cache tensors | 2 GiB |
| Logical training batch | 128 examples |
| Initial physical microbatch | 16 examples |

The cache grows only when requested training targets are reused. Small Sudoku4
datasets need far less than 2 GiB. Raising a cache cap alone does not allocate
RAM. Caller copies and tensor metadata sit outside its owned-byte counter;
process readings cover them. Host checks sample allocation boundaries and
cannot enforce an operating-system memory limit. CUDA additionally receives
an allocator fraction limit. Other processes can change available memory.

First measure the workload on the actual GPU. Training and inference have
separate commands because their live tensors differ:

```bash
python -m mechinterp.bpd.experiments.depth_steering_gym.dt05.runner calibrate --profile dt04 --stage training --device cuda:0 --precision bf16 --logical-batch 128 --candidates 8 16 32 64 128 --out runs/dt05/train-calibration.json
python -m mechinterp.bpd.experiments.depth_steering_gym.dt05.runner calibrate --profile full_7m --stage inference --device cuda:0 --precision bf16 --logical-batch 256 --candidates 16 32 64 128 256 --out runs/dt05/inference-calibration.json
```

`full_7m` has 6,828,034 parameters and 42 transformer-block executions per
outer ACT step. Inference calibration measures eight ACT steps. Training
calibration measures forward/backward at fit depths 2/4/6 and keeps Adam moment
storage plus an EMA shadow resident. Its timing excludes the optimizer and EMA
updates, input construction, and transfers. Both stages use random engineering
inputs, keep model weights unchanged, retain OOM/rejected trials, and restore RNG
state. A calibration with no admitted candidate has `selected_microbatch: null`.

Choose the reported microbatch when freezing the next plan. An effective batch
of 128 with microbatch 32 accumulates four chunks before one optimizer update.
The sampler draws all 128 examples first; blank/clue denominators come from
that entire batch. Changing the physical chunk size therefore preserves the
same sampled data and mathematical loss. Floating-point reduction order can
still change numerical trajectories. BF16 is an explicitly recorded training
variant; factor fitting and geometry use FP32.

Adjust available-memory reserves for the machine through
`--max-host-rss-gib`, `--min-host-available-gib`,
`--max-cuda-reserved-gib`, and `--min-cuda-free-gib`. Treat total installed RAM,
available host RAM, and free VRAM as different quantities. Calibrate while
other GPU jobs are idle if the goal is an uncontended throughput estimate.

## Freeze and run development

DT05 reserves model roots `2605000..2605058` for development with the exposed
Sudoku4 generator/catalog. Every root remains one independent model unit.
Plans record the actual source hashes and recipe before a run. Training uses
only the training split. Evaluation cases stay canonically disjoint within
each model root. DT04's prospective confirmation roots `2604100..2604355`
and every other historical seal remain outside this entry point.

```bash
python -m mechinterp.bpd.experiments.depth_steering_gym.dt05.runner plan --microbatch 32 --precision fp32 --out runs/dt05/plan.json
python -m mechinterp.bpd.experiments.depth_steering_gym.dt05.runner run --plan runs/dt05/plan.json --root 2605000 --device cuda:0 --out runs/dt05/campaign
```

The example microbatch 32 is illustrative. Substitute the measured training
choice. A new plan has a fixed update budget of 480 by default, matching the
old development recipe. `--steps`, `--ema-decay`, and `--factor-steps` expose
prospective training variants. Longer training is a new compute budget and
belongs in its own frozen plan. Both final raw and EMA models are scored;
EMA is the preselected factor teacher. The five evaluation arms are raw, EMA,
online factors, fixed-factor control, and seeded-random-factor control.

One invocation attempts one root. A campaign's exclusive attempt marker blocks
duplicate execution even after interruption or failure. Failed receipts remain
in place. There is no automatic retry or all-roots launch. Changing source after
plan creation invalidates the plan. A new campaign retains its own recipe and
attempt records; reusing a model seed across campaigns does not create an
additional independent root.

Each root directory contains the frozen plan, full split manifest, raw/EMA/
factor checkpoints with SHA-256 receipts, losses, memory readings, every case's
predictions at each observed depth, and the failure or completion receipt.
Factor checkpoints include their live routing mode. The primary development
comparison is online factors against the fixed-factor control at depth 8;
these measurements support development analysis until independent matched
controls and a new evidence contract establish a scientific mechanism test.

## Replay an existing local checkpoint

The replay loader accepts local files with an explicit SHA-256. Pure checkpoints
use the matching `mini`, `dt04`, or `full_7m` profile. Official 7M bridge weights
must match one of the two existing published bridge hashes. Loading uses
`torch.load(..., weights_only=True)` and has no download path.

Create a manifest using real digits, with 0 for blank clues:

```json
{
  "schema": "BPD-DT05/sudoku-cases/v1",
  "scope": "development",
  "cases": [
    {
      "id": "example-4x4",
      "givens": [[1,0,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]],
      "solution": [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]]
    }
  ]
}
```

Every puzzle must have a unique solution, checked by the exact Sudoku solver.
The evaluator receives gold labels for scoring; the stopping function accepts
only predictions and public givens. It reports raw fixed-depth outcomes and
the physically compacted policy separately, including casewise disagreements
caused by changed batch arithmetic. Its single-run wall times are diagnostics.
Use repeated calibration for a speed claim.

```bash
python -m mechinterp.bpd.experiments.depth_steering_gym.dt05.runner replay --checkpoint PATH_TO_MODEL --sha256 CHECKPOINT_SHA256 --checkpoint-format official --profile full_7m --cases LOCAL_DEVELOPMENT_CASES.json --depths 4 8 12 16 24 32 --microbatch 16 --device cuda:0 --out runs/dt05/bridge-stopping.json
```

For a saved DT05 factor model use `--checkpoint-format online_factors` with
the matching `--components` and `--top-k`. The smoke run also writes a pure
checkpoint receipt and `CASES.json` for a fully local replay check.

## Source map

| File | Responsibility |
|---|---|
| [runner.py](runner.py) | CLI, source/plan hashes, cohort admission, attempts and receipts |
| [training.py](training.py) | Effective-batch training, EMA, cached autonomous factor fitting |
| [techniques.py](techniques.py) | Validity checks, factor layers, thin projectors and JVP transport |
| [evaluation.py](evaluation.py) | Fixed and compact rollouts, case outcomes, geometry diagnostic |
| [resources.py](resources.py) | Memory readings, reserves, batch tuner and bounded CPU cache |
| [calibration.py](calibration.py) | Native random-input inference/training measurements |
| [evidence/REPRODUCTION.json](evidence/REPRODUCTION.json) | Reproduced DT04 EvalMath gate and missing-ledger limits |

Primary references for the implemented tools:
[PyTorch JVP](https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html),
[PyTorch performance guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
and the [official TRM repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
Their benchmark results belong to their own models and datasets.
