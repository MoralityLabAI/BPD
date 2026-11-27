# CURATE-03B paired native pilot protocol (frozen before native measurement)

Evidence class: development only. Preserved: the CURATE-02 frozen campaign, `SOURCE_FREEZE.json`, its checkpoints and negative results, the failed selective-control admission (so no causal-auxiliary intervention), and EvalMath `REDESIGN_OR_PILOT_ONLY`. Nothing here can promote a training method. The DataGoblin PR #100 retrospective (full-screen abstention avoided 275 rejected attempts and missed 1 accepted update; early-12 kept 99.82% of recorded positive accepted gain with 9,040 acquisitions instead of 10,240; root-held-out adaptive thresholds kept 99.13%, below the 99.5% target) is spent-development evidence and motivates the gates. It is not evidence for them.

This file and the code it names were committed before any CURATE-03B engineering-root or fresh-root measurement. Earlier smoke runs used 4 CPU-s budgets and 1–3 windows on root 28000, only to exercise the code paths. They are not results.

## Source

`custody.py` (state isolation), `native_screen.py` (native feedback/candidate gradients, analytical scoring, acquisition order, gates, joint search, transactional acceptance), `pilot.py` (arms A–J), `engineering_03b.py` (admission), on top of `adamw_screen.py` and `batch_screen.py` from CURATE-03A. `transaction.probe_update` stays as the transactional reference implementation.

## Custody

`Custody.capture/restore` covers parameter values and existing gradients (restored in place, so the optimizer keeps its Parameter objects); every buffer slot, persistent or not, including buffers that were resized, replaced or added; every public module attribute, with deletion of attributes created during the call (`mode`, `schedule`, `calls`, `pass_offset`, `recorded_inputs`, `max_passes`, `training`, ...); AdamW per-parameter state and every param-group hyperparameter; Torch CPU, Python and NumPy global RNG; explicit torch/NumPy generators. Feedback gradients, candidate gradients and feedback re-evaluations always run in `isolated()`. The committed minibatch step runs in `transaction()`, and it is kept only on `commit()` with no exception. Feedback re-evaluation after a trial step replays the RNG point captured when the feedback gradient was taken (matched draws).

## Frozen numerics

* Acceptance: keep the step iff `cal_after <= cal_before + 1e-6`, where `cal_before` is the feedback loss recorded with the feedback gradient at the same parent (the CURATE-02 rule).
* Analytical vs transactional score: `|s_a - s_t| <= 1e-4 * (|s_t| + ||g_cal|| * ||delta||)` (float32).
* Aggregated mean raw per-edge gradient vs native concatenated-batch gradient: relative L2 error `<= 1e-5` and elementwise `|diff| <= 1e-5 * max(1, max|g_native|)`. Predicted joint score error `<= 1e-4 * (|s| + ||g_cal|| ||delta||)`. Float64 tolerances (`1e-12`, `1e-10`) apply to the CURATE-03A MLP fixtures only. `TiedComponentLinear` computes in float32 (`x.float() @ V`), so native ComponentTRM runs in float32 only. The final step always uses the natively recomputed concatenated gradient, so a parity failure flags the window and never changes which update is applied.

## Acquisition

* Order: within each edit slot, sort the window's 16 proposals by SHA-256 of `('BPD-TRM-CURATE-03B:acquisition-order:v1', split_seed, root, window, edge)`, then visit slots round-robin 0..3. Built only from semantic edge keys that exist before any acquisition.
* Stages 4, 8, 12, 16.
* Full-16 gate: acquire 16, abstain iff every predicted gain is `<= 0`.
* Early-12 gate: acquire 12, abstain iff all 12 are `<= 0`; otherwise acquire 16 and apply the full-16 gate.
* Shadow audit of abstained windows: known probability 0.25, from the SHA-256 draw `('BPD-TRM-CURATE-03B:audit:v1', root, arm, window)`. An audit acquires any remaining candidates, runs the arm's selection (including joint search) and a real acceptance test with commit forbidden, then records `would_accept` and realized gain. Audit CPU is charged to the arm. Unaudited counterfactuals stay missing and are never imputed as zero.
* The root-held-out adaptive threshold policy is a secondary exploratory arm, evaluated offline only (DataGoblin V2 export). It is not run live in this pilot.

## Arms (each from the identical warmup checkpoint, fresh AdamW lr 0.001, wd 1e-4, clip 1.0)

| Arm | Scoring | Objective | Gate | Joint rounds | Acceptance test | Scheduler mode |
|---|---|---|---|---|---|---|
| A_uniform_reinvest | none | — | — | 0 | no | uniform |
| B_transactional | `probe_update` per candidate | mechanistic | — | 0 | yes | mechanistic |
| C_analytic_ord | analytical | ordinary | — | 0 | yes | mechanistic |
| D_analytic_mech | analytical | mechanistic | — | 0 | yes | mechanistic |
| E_full16_gate | analytical | mechanistic | full-16 | 0 | yes | mechanistic |
| F_early12_gate | analytical | mechanistic | early-12 | 0 | yes | mechanistic |
| G_joint_full16 | analytical | mechanistic | full-16 | R | yes | mechanistic |
| H_joint_early12 | analytical | mechanistic | early-12 | R | yes | mechanistic |
| I_shuffled | analytical, scores shuffled across candidates | mechanistic | — | 0 | yes | shuffled |
| J_observer_only | analytical (observed only) | mechanistic | — | 0 | no | uniform |

The scheduler is CURATE-02 `select_candidates` with 4 selections, 20% exploration and pair cap 4. Joint search starts from the scheduler output, pins the exploration edge, and checks legality on the whole batch against the pre-window selection ledger. R = 1 unless the engineering rule picks 2: two rounds iff mean round-2 predicted improvement is at least 0.1 × the round-1 improvement **and** mean round-2 search CPU is at most 0.25 × mean candidate-acquisition CPU.

Exposure: each arm has its own ledgers. The selection ledger enforces the cap and resets only when fewer than 16 fit edges remain eligible; resets are counted. Separate proposal and committed ledgers are recorded. Nothing is shared or reset between arms.

## Budget

Primary budget: 100 process CPU seconds (`time.process_time`, all threads, 2 Torch threads) per arm per root. The check runs before each window; a window already started completes, and the overshoot is recorded, never truncated or rebated. Charged: proposal generation, batch materialization, cache construction, feedback gradients, candidate gradients or probes, scoring, gating, search, verification, acceptance steps, feedback re-evaluation, custody snapshot/restore, shadow audits, provenance hashing for scored arms, and controller overhead. Not charged: shared warmup, slot-projector discovery (identical inputs to every arm) and evaluation. MMS cache 16 MiB per arm. Process-tree hard cap 6 GiB (Win32 Job Object), with no GPU use. Wall-clock parallelism: at most four root processes at once.

Fixed-update comparisons run separately: on the engineering roots, B vs D and A vs J for 40 windows; on every fresh root, A vs J for 40 windows (observer-only invariance).

## Engineering admission (spent roots 28000, 28001; CURATE-02 phase-A warmups)

* AG1: `test_custody.py`, `test_native_screen.py` and the existing suite pass.
* AG2: fixed-update A vs J final state identical on both roots.
* AG3: at least 99% of analytical-vs-transactional comparisons within tolerance, and top-4 ranked set identical in at least 95% of windows (cold and warm optimizer × both objectives × clip 1.0/0.05/1e6 × 3 windows × 16 candidates).
* AG4: fixed-update B vs D selections identical in at least 95% of 40 windows on both roots.
* AG5: 100% of verified joint windows within the joint tolerance.
* AG6: median analytical per-candidate CPU at most 0.5 × median transactional probe CPU.
* AG7: peak process-tree commit under the 6 GiB job limit.
* AG8: acquisition order identical on recomputation for every window.

The fresh roster runs only if AG1–AG8 all pass. Failures are reported and none is waived.

## Fresh development roster (frozen)

Roots 28301–28308, never used before as trm_curation roots. Fresh warmup: CURATE-02 recipe, 6,000 updates, init seed `seed_for(root, 'init_TRM_w32_r29', 0)`. Every eight-root result is development evidence.

## Endpoints and analysis

Primary: ordinary delay-31 accuracy with steering disabled, EVAL partition, 512 cases per root. Secondary: delay-63 and delay-127 accuracy, all-distinct binding at delay 31 (256 cases), selective transport (oracle hybrid-displacement target hit at delay 31 on 48 stratified EVAL edges), and protected-state corruption (protected queries correct before the edit and wrong after).

Primary contrast family (Holm, fixed CPU): D−A, E−A, F−A, G−A, H−A. Secondary (unadjusted, descriptive): D−B, C−D, E−D, F−E, G−E, H−F, D−I, J−A. Statistics: per-root paired differences in correct counts, exact two-sided root sign-flip, seeded percentile bootstrap over roots (10,000 draws, seed 20261127). Scientific floor: +5 pp. EvalMath preflight (`repair_portfolio/evidence/preflight.py`) uses G−A as the paired effect and I−A as the sham. The `REDESIGN_OR_PILOT_ONLY` decision is preserved regardless of outcome. Changing it needs a separately registered experiment.
