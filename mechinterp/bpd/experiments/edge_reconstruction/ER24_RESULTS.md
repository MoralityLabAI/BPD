# ER-24: empirical memory-budget-aware execution selector

## Design
Twelve new independent trained model instances (widths 64,128,256; roots 38,39 calibration and 40,41 test at each width). Same ER-23 four-pass trained tanh recurrence and SVD factor fault model. Two 12-probe episodes per root for 64/128; one for 256 to bound runtime. Six configurations at widths 64/128: eager batch 32/64, dense bound batch64, stream (batch,block)=(32,4),(64,8),(128,16). At width256 only eager64, dense64, stream64/8. Each configuration/root executed once in an isolated fresh process, order reversed for odd roots. SVD and training excluded, same benchmark kernel as ER-23. RSS is whole-process Linux ru_maxrss. Lookup fit uses only calibration rows: estimated time is median episode time and estimated RSS is maximum root peak plus 4096 KiB. Choose fastest predicted feasible mode; abstain if none. This is an offline held-out replay of the policy using premeasured candidate configurations, not an online scheduler or hard memory enforcement. Single timing repetition, exploratory.

## Calibration and holdout
|width|config|cal guarded peak MiB|cal median ms|test max RSS MiB|test median ms|
|---:|---|---:|---:|---:|---:|
|64|eager32|100.3|45.85|96.26|49.05|
|64|eager64|100.9|66.38|96.90|33.71|
|64|dense64|104.4|9.48|100.41|13.82|
|64|stream32/4|103.5|11.49|99.52|11.10|
|64|stream64/8|104.0|12.77|100.03|12.07|
|64|stream128/16|105.9|27.78|101.88|19.14|
|128|eager32|104.9|492.25|100.91|437.26|
|128|eager64|106.0|511.84|102.04|468.10|
|128|dense64|121.8|102.01|117.80|85.08|
|128|stream32/4|118.9|114.02|114.86|82.24|
|128|stream64/8|118.9|103.66|114.89|77.07|
|128|stream128/16|120.3|118.38|116.38|92.16|
|256|eager64|123.5|7173.35|119.49|5710.22|
|256|dense64|191.4|1164.25|187.41|1440.82|
|256|stream64/8|179.1|1638.62|175.09|1401.93|

All 54 test configuration-episode results match reference eager64 MAP and cost within 1e-10 (44 non-self comparisons); 10/10 distinct episodes recover hidden fault. Primary budgets 96,112,128,160,192,224 MiB: 60 episode-budget scenarios, 48 accepted and fit, 12 abstentions (none feasible among measured configurations). Median selected/fastest actually feasible time ratio 1.112; max 2.748; median eager64/selected time ratio 2.910 among accepted scenarios with eager64 feasible. The primary grid does not choose streaming. Secondary budget sensitivity at 120 and 184 MiB, chosen from calibration thresholds while test collection was ongoing: 20 scenarios, 18 accepted and fit, two false abstentions at width256/120 because the guard disallows eager64, which observed peak fits. Median regret 1.064. At width128/120 stream64/8 selected on all four episodes, peak max114.89 MiB; width256/184 stream64/8 selected on both, max175.09 MiB, speedups 2.532 and 9.128 relative to eager64. No observed accepted budget violations in either grid. Budgets reuse episodes and process measurements, so no independent binomial estimate over 48 scenarios. Only six independent held-out model instances; no general safety guarantee. Calibration prefers dense at generous budgets while holdout median fastest is stream at all widths; runtime regret exposes the cost-model error. Guard is arbitrary, selected without user direction; no measured hard cap, concurrency or hardware transfer.

## Reproduce
From experiment directory:
```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python er24.py prepare er24_worlds > er24_prepare.jsonl
python er24.py cal er24_worlds > er24_cal.jsonl
python er24.py test er24_worlds > er24_test.jsonl
python er24.py evaluate er24_cal.jsonl --test er24_test.jsonl > er24_policy.json
python -c 'import er24,json; from pathlib import Path; er24.BUDGETS=(120,184); print(json.dumps(er24.evaluate(er24.load(Path("er24_cal.jsonl")),er24.load(Path("er24_test.jsonl"))),indent=2))' > er24_sensitivity.json
```
Local executed source SHA256 3112b22e826d6a48b01a327d6d468f8b5573f49b96e88069576a4d11d24c2f0d; Git blob d75f300d8b2dddb03598f8db2bc712e557b3c72f verified remote. Receipts SHA256: cal 3d41b3b30e2d8371a599e4f74c869bb4e67716ecd38c69518d5bb6837bd8e0ca; test 968fd951d444fff04c617f556a21ea79d66645a980b7680eb145d032b64aeca7; policy 392807edb4256600e3af8bb7a678bc8493c5eb381fc4190c5a2628c19c56a0c2; sensitivity 74bb5b19069b6a63355c16d4ef09e548dea0e154925baecfd6e0421d4eb556bd (local file omits trailing newline). ER-23 commit CI green: https://github.com/MoralityLabAI/BPD/actions/runs/45861049377 and https://github.com/MoralityLabAI/BPD/actions/runs/45861049395 .

## Next gate
Use registered independent workload roots, memory-feasible prediction calibrated over sufficient samples, CPU time counters, measured operating-system containment (separate from RSS prediction) and online workload features such as lower-bound survival fraction. Evaluate native learned BPD factors and workload transfer. Exact MAP certificate still uses float64 tolerances, not outward-rounded arithmetic or Lean proof.
