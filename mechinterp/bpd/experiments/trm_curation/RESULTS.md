# BPD-TRM-CURATE-02: Mechanistic Curriculum Hill-Climbing Results

**Status / Evidence Class:** `DEVELOPMENT_ONLY` | **EvalMath Preflight:** `REDESIGN_OR_PILOT_ONLY` | **Independent Audit Replay:** `PASS` (`57,344` cases, `488/488` checks, `0.0` max CE error)

## Executive Summary

We completed the `BPD-TRM-CURATE-02` implementation, verification suite, engineering screens, RAM scaling benchmark, source-locked eight-root paired development campaign (`28101..28108`), 36 low-difficulty TRM evaluation suite ([EVAL_36_EXPLORATION.md](EVAL_36_EXPLORATION.md)), DataGoblin train-only export, and independent saved-checkpoint replay under a fail-closed **6 GiB** OS-level process-tree memory cap (`Win32 Job Object` hard commit ceiling + `psutil` tree RSS monitor).

Key outcomes on the primary endpoint (**ordinary delay-31 accuracy, steering disabled**, `4,096` evaluation cases across **8 fresh independent TRM roots**):

- **Fixed-update comparison (80 scheduling windows):**
  - `C_ord_grad` (ordinary TRAIN feedback-gradient alignment at `d=15`): **50.24%** (**+3.34 pp** vs `A_uniform` `46.90%`, `7/8` roots positive, exact two-sided sign-flip `p = 0.0781`, Holm `p = 0.3906`, 95% root bootstrap `[+0.39, +5.96]` pp).
  - `D_mech_grad` (mechanistic feedback-gradient alignment adding fixed-dose targeted transport & unaffected preservation): **49.15%** (**+2.25 pp** vs `A_uniform` `46.90%`, `+0.76 pp` vs `B_loss_mined` `48.39%`, `+1.88 pp` vs `E_shuffled` `47.27%`; `6` positive, `1` tie, `1` negative root; exact sign-flip `p = 0.2656`, Holm `p = 1.0`).
  - `F_obs_only` (full mechanistic probe instrumentation + uniform selection): **46.90%** — **bit-identical** to `A_uniform` on every single root (`0.0000 pp` difference), proving zero side-effect leakage from transactional rollback.
  - Neither `D_mech_grad` nor `C_ord_grad` cleared the provisional **+5.0 pp** scientific floor; EvalMath returns `REDESIGN_OR_PILOT_ONLY` (`approx_worlds_for_80pct_power = 14`, `approx_resolvable_effect = 11.44 pp`).
- **Fixed-total-CPU comparison (~100 CPU-seconds/root):**
  - `G_cpu_reinvest` reinvested `D_mech_grad`'s exact CPU budget into an average of **2,187.9 uniform updates** (vs `D_mech_grad`'s **25.4 accepted updates**), reaching **49.44%** at delay-31 (`D - G = -0.29 pp`, `p = 0.8984`; `C - G = +0.80 pp`).
  - Thus feedback-gradient hill-climbing achieves an **86× update sample-efficiency gain**, while full 16-candidate transactional rollback probing currently consumes that CPU margin.
- **Deeper extrapolation, binding, robustness & causal selectivity (`E01–E24`):**
  - On deeper horizons, `D_mech_grad` reaches **34.40%** at delay-63 (`+3.69 pp` vs `A`, `+4.20 pp` vs `E_shuffled`) and **31.27%** at delay-127 (**highest of all 7 arms**, beating `G_cpu_reinvest` `31.13%` and `E_shuffled` `26.76%`).
  - On all-distinct address binding at delay-31, `D_mech_grad` reaches **47.02%** (`+4.34 pp` vs `A_uniform` `42.68%`).
  - On causal full-hybrid oracle transport (`E19`), `D_mech_grad` reaches **57.29%** vs `A_uniform` **47.92%** (**+9.38 pp**), and outperforms `A_uniform` on **21 of 24** diagnostic evaluations (`E04–E24`).
- **Candidate prediction quality (8-root mean):**
  - First-order `-dot(g_cal, delta_theta)` predicts realized nonlinear AdamW loss drop with **Spearman ρ = 0.9401 / Pearson r = 0.9617** (`86.72%` sign accuracy, `100.0%` top-4 positivity) for ordinary calibration and **Spearman ρ = 0.9151 / Pearson r = 0.9400** (`91.41%` sign accuracy, `100.0%` top-4 positivity) for mechanistic calibration.
- **RAM scaling (`--mms-mib 0, 4, 16, 64, 256` MiB):**
  - **100% byte-exact checkpoint parity** across all 5 tiers (`43d468353de4500510f5034dca1c3e8c53ba6eb0e7d3a6ab6b7e579e384f973f`).
  - **2.27× real CPU speedup** (`37.52 s` at `0 MiB` → `16.50 s` at `16 MiB`; `49.24%` hit rate, `260` recomputations avoided, `1.49 MiB` peak total occupancy including metadata/keys), lifting efficiency from `149.9` to `340.9 pp/CPU-hour`. Expansion beyond 256 MiB is not justified by measured occupancy (`1.49 MiB`).
- **Retained failed gates:**
  - `TRM_w16_r16` carried-state-size control (`1,556` params, `32` carried coords) failed engineering competence on both roots.
  - Engineering Phase B selective causal control failed behavioral admission (`SELECTIVE_CONTROL_ADMISSION_FAILED` retained on both roots `28000`, `28001`), so the gated 2×2 causal-auxiliary cross was correctly held back per protocol.

---

## 1. Implementation Verification & Frozen File Integrity

All **23 focused `trm_curation` unit tests** (`test_reference`, `test_model`, `test_v2`, `test_complete`) and **41 native BPD + RP-02/03/04 model tests** passed:

- **Default `ComponentTRM` parameters:** exactly **5,380** (`width=32`, `components=29`, `low_cycles=1`, `64` carried coordinates, `2` shared BPD block visits/token).
- **Shared physical factors & native reconstruction:** single `TiedComponentLinear` `(V, U)` shared across all high and low visits; `(V @ U).T == up.reconstructed_weight()`; `W_target` mutation produces byte-identical logits and states.
- **Exact suffix-only continuation (`ComponentTRM.rollout_suffix`):** matches full-rollout `rollout(..., state_replacement=...)` on logits, states, `pass_offset`, and autograd gradients w.r.t. both `init_state` and every trainable parameter (`rtol=1e-4, atol=1e-5`), plus exact full hybrid-state replacement parity.
- **Independent finite-difference observer Jacobians:** central finite-difference Jacobian verification matches `observe_edge` (`target`, `protected`, `observer` stacks) on native `ComponentTRM` across multi-delay continuations.
- **Complete transactional rollback (`transaction.py`):** restores parameters, buffers, AdamW moments, gradients, training flags, global Torch/NumPy/Python RNGs, explicit `torch.Generator` samplers, and `TiedComponentLinear` mutable state (`calls`, `pass_offset`, `mode`, `schedule`, `recorded_inputs`), including on callback exception.
- **Semantic partition integrity:** `edge_views(20261127)` yields exactly `624` fit / `144` feedback / `144` tune directed edges (`912` total TRAIN directed edges, `48` transition types in every view, both orientations grouped together, `0` overlap with `312` EVAL directed edges).
- **RP-03 / RP-04 preservation:** `git status` confirms zero modifications to `training_steering`, `training_steering_stability`, or `component_selectivity`.

---

## 2. Phase A: Competence & Matched Architecture (`28000`, `28001`)

Trained on identical TRAIN fit streams (`6,000` warmup updates, delays `1/2/3`, AdamW `lr=0.003`, `wd=0.0001`, `clip=1.0`, `2` CPU threads). Full learning curves and checkpoints are archived in `results/engineering/phase_a/phase_a_summary.json`.

| Architecture | Params | Carried Coords | Visits / Tok | Total Block Visits | Mean CPU-s | Eval d=2 | Eval d=7 | Eval d=31 | Eval d=63 | Eval d=127 | Eval Bind d=7 | Eval Bind d=31 | Competence Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `GRU_w32` (RP-03) | 5,380 | 32 | 1 | 42,000 | 259.75 | 98.44% | 96.09% | 78.91% | 62.21% | 52.83% | 93.95% | 74.41% | **PASS** (2/2) |
| `TRM_w32_r29` | 5,380 | 64 | 2 | 84,000 | 451.30 | 97.17% | 90.23% | 50.88% | 32.81% | 32.13% | **98.24%** | 51.17% | **PASS** (2/2) |
| `TRM_w16_r16` (ctrl) | 1,556 | 32 | 2 | 84,000 | 372.72 | 66.89% | 65.53% | 29.30% | 29.49% | 26.46% | 53.32% | 33.20% | **FAIL** (0/2) |

**Fixed-CPU comparison (at GRU's 253–266 CPU-s budget):** `TRM_w32_r29` reaches `3,103–3,489` updates with TRAIN `d=2` `93.8%`, `d=7` `79.3%`, `bind_d7` `82.8%`, `d=31` `32.4%` (vs GRU's `99.9% / 99.8% / 99.6% / 81.1%`). `TRM_w32_r29` achieves stronger short-horizon all-distinct binding (`98.24%` vs `93.95%` at `d=7`) while leaving wide delay-31 headroom (`~51%`), ideal for testing curriculum hill-climbing.

---

## 3. Phase B: Mechanistic Signal Bank & SCCG-01 Constrained Frontier (`28000`, `28001`)

Full receipts in `results/engineering/phase_b/phase_b_summary.json` and `root_{28000,28001}_signal_bank.json` (`24` feedback probe edges + `48` fit SVD edges per root, delays `(2, 7, 31)`, dose `0.75`).

### Site Selective-Control Capacities (Linear Target Margin Gain, Dose 0.75)
| Root | High `z_H` (budget 0) | Low `z_L` (budget 0) | Joint `[z_H, z_L]` (budget 0) | High `z_H` (budget 0.25) | Low `z_L` (budget 0.25) | Joint `[z_H, z_L]` (budget 0.25) |
|---|---:|---:|---:|---:|---:|---:|
| `28000` | 0.00233 | 0.00053 | 0.00287 | 0.3620 | 0.2860 | 0.3672 |
| `28001` | 0.00151 | 0.00027 | 0.00191 | 0.3212 | 0.2014 | 0.3236 |

### SCCG-01 Constrained Frontier & Support Family Comparison (Mean of Roots `28000`, `28001`, Dose 0.75, Budget 0.25)
| Support / Policy | Linear Gain | Collateral Norm | Target Flip Acc | Protected Corruption | Feasible Rate |
|---|---:|---:|---:|---:|---:|
| `joint_zero_col` (budget 0.0) | 0.0024 | 1.9e-15 | 10.42% | 0.00% | 100.0% |
| `joint_budget_025` (budget 0.25) | 0.3454 | 0.2500 | 10.42% | 0.00% | 100.0% |
| `joint_budget_050` (budget 0.50) | 0.5887 | 0.5000 | 12.50% | 0.00% | 100.0% |
| `joint_budget_100` (budget 1.00) | 0.9843 | 1.0000 | 14.58% | 0.00% | 100.0% |
| `joint_pair_search_r4` | **0.1422** | 0.2500 | 10.42% | 0.00% | 100.0% |
| `joint_pair_search_r8` | **0.1888** | 0.2500 | 10.42% | 0.00% | 100.0% |
| `contiguous_v_high_r4` | 0.1165 | 0.2500 | 10.42% | 0.00% | 100.0% |
| `contiguous_v_low_r4` | 0.1147 | 0.2500 | 10.42% | 0.00% | 100.0% |
| `svd_r4` / `svd_r8` | 0.0955 / 0.1375 | 0.2500 | 10.42% | 0.00% | 100.0% |
| `random_r4` / `random_r8` | 0.0789 / 0.1353 | 0.2500 | 10.42% | 0.00% | 100.0% |
| `signed_observer_sham` (MOLT-06) | 0.0488 | 0.0365 | 2.08% | 0.00% | **14.58%** (7/48) |
| `hybrid_full_dose` (0.75) | 1.0197 | 7.7632 | 12.50% | 3.82% | 100.0% |
| `hybrid_natural_oracle` (unmatched) | 5.9809 | 47.155 | **41.67%** | 17.83% | 100.0% |

- **Diagnostics:** High-state `z_H` carries the majority of selective-control capacity (`~70%` of high+low). Joint pair beam search strictly beats contiguous V-groups, TRAIN SVD, and matched random supports on linear gain at matched rank. Outside-group physical write leakage is `81.0%–86.0%` with cancellation `K = 0.190–0.211`; hybrid projection residuals are `50.0%–56.2%` outside `V_high` and `63.3%–70.7%` outside `SVD-4`.
- **Dose grid (`0.25 / 0.75 / 1.5 / 3.0` on full hybrid):** Target accuracy rises `10.4% → 12.5% → 20.8% → 33.3%`, while collateral corruption rises `0.0% → 3.8% → 10.3% → 12.8%` and nonlinear finite-dose error grows from `0.13` to `9.06`.
- **Gate outcome:** `SELECTIVE_CONTROL_ADMISSION_FAILED` (linear selective gains at dose `0.75` do not cross converged clean logit margins to exceed random behavioral flips). Retained transparently; gated causal-auxiliary cross not triggered.

---

## 4. RAM Scaling (`--mms-mib`) & Hard 6 GiB Memory Enforcement

Measured on engineering root `28000` (`16` curation windows, `results/engineering/ram_scaling/ram_scaling_summary.json`). Every entry is strictly detached (autograd graphs refused) and bound to `(checkpoint_sha256, module_id, visit_spec, input_fingerprint, objective_id, factor_frame_sha256)`.

| `--mms-mib` | Exact Checkpoint Parity | Hit Rate | Recomp. Avoided | Peak Total Bytes (Payload + Meta) | Total CPU-s | CPU-s Saved (Actual / Est) | Improvement / CPU-hr (`pp/hr`) |
|---:|---|---:|---:|---:|---:|---:|---:|
| `0` | `true` (`43d46835...`) | 0.00% | 0 | 0 B | 37.52 s | 0.00 / 0.00 s | 149.9 |
| `4` | `true` (`43d46835...`) | 49.24% | 260 | 1.49 MiB (1.17 + 0.32) | 16.73 s | +20.78 / 22.17 s | 336.1 |
| `16` | `true` (`43d46835...`) | 49.24% | 260 | 1.49 MiB (1.17 + 0.32) | **16.50 s** | **+21.02 / 22.64 s** | **340.9** |
| `64` | `true` (`43d46835...`) | 49.24% | 260 | 1.49 MiB (1.17 + 0.32) | 17.41 s | +20.11 / 24.16 s | 323.2 |
| `256` | `true` (`43d46835...`) | 49.24% | 260 | 1.49 MiB (1.17 + 0.32) | 16.78 s | +20.73 / 22.88 s | 335.2 |

- **Process-tree memory ceiling:** Hard OS enforcement via `Win32 Job Object` (`JOB_OBJECT_LIMIT_JOB_MEMORY = 6,442,450,944` bytes = `6.00 GiB`, fail-closed, unit-tested against an intentional over-allocation) plus `psutil` tree RSS and host available-RAM monitoring.
  - Development 8-root campaign peak job commit: **1,754.6 MiB** (`1.71 GiB`, `< 29%` of 6 GiB cap); peak process-tree RSS: **283.8 MiB**; minimum host available RAM: **6.24 GiB** (`wall = 3,365.0 s`, `process_cpu = 5,985.0 s`).
  - 36-eval suite peak job commit: **1,610.4 MiB**; peak tree RSS: **253.4 MiB** (`wall = 3,988.4 s`, `process_cpu = 8,058.8 s`).

---

## 5. Eight-Root Paired Development Experiment (`28101..28108`)

Locked prior to execution in `SOURCE_FREEZE.json` (`curation_lr=0.001`, `feedback_delay=15` strictly on TRAIN `views['feedback']`, `w_ordinary=1.0`, `w_target=0.5`, `w_preserve=0.5`, `80` windows, `16` proposals → `4` selected, `20%` exploration, `4`-presentation unordered-edge cap, `--mms-mib 16`).

### Candidate Prediction Quality (8-Root Mean at Warmup Fork)
| Calibration Objective | Spearman ρ | Pearson r | Sign Accuracy | Mean Pred. Gain | Mean Realized Gain | Top-4 Realized Gain | Top-4 Positive Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary Feedback (`d=15`) | **0.9401** | **0.9617** | 86.72% | +0.0667 | +0.0312 | +0.1904 | **100.0%** |
| Mechanistic Feedback (`d=15` + transport/preserve) | **0.9151** | **0.9400** | **91.41%** | +0.0544 | -0.0013 | **+0.2962** | **100.0%** |

### All Seven Paired TRM Arms Across Delays & All-Distinct Binding (8-Root Mean, 4,096 Cases/Delay)
| Arm | Accepted / Rejected | Mean CPU-s | d=2 | d=7 | **Primary d=31** | d=63 | d=127 | Bind d=2 | Bind d=7 | Bind d=31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_uniform` | 80.0 / 0.0 | 6.33 | 98.75% | 94.73% | **46.90%** | 30.71% | 28.71% | 100.00% | 97.66% | 42.68% |
| `B_loss_mined` | 80.0 / 0.0 | 14.32 | 99.44% | 95.68% | **48.39%** | 31.81% | 28.66% | 100.00% | 98.58% | 45.95% |
| `C_ord_grad` | 24.0 / 56.0 | 89.65 | 98.49% | 94.14% | **50.24%** | 33.52% | 30.13% | 99.71% | 98.39% | 46.92% |
| `D_mech_grad` | 25.4 / 54.6 | 99.98 | 98.36% | 93.55% | **49.15%** | 34.40% | **31.27%** | 99.90% | 98.34% | 47.02% |
| `E_shuffled` | 80.0 / 0.0 | 94.73 | 99.07% | 93.77% | **47.27%** | 30.20% | 26.76% | 99.95% | 98.19% | 44.34% |
| `F_obs_only` | 80.0 / 0.0 | 104.70 | 98.75% | 94.73% | **46.90%** | 30.71% | 28.71% | 100.00% | 97.66% | 42.68% |
| `G_cpu_reinvest` | 2187.9 / 0.0 | 100.01 | 99.61% | 96.39% | **49.44%** | 36.01% | 31.13% | 100.00% | 98.97% | 47.51% |

### Per-Root Primary Delay-31 Correct Counts (out of 512)
| Root | `A_uniform` | `B_loss` | `C_ord_grad` | `D_mech_grad` | `E_shuffled` | `F_obs_only` | `G_reinvest` | `D − A` (pp) | `C − A` (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `28101` | 233 | 265 | 268 | 265 | 214 | 233 | 225 | +6.25 | +6.84 |
| `28102` | 301 | 289 | 276 | 276 | 306 | 301 | 298 | −4.88 | −4.88 |
| `28103` | 224 | 244 | 261 | 291 | 216 | 224 | 257 | +13.09 | +7.23 |
| `28104` | 251 | 240 | 264 | 251 | 239 | 251 | 283 | 0.00 | +2.54 |
| `28105` | 217 | 237 | 259 | 228 | 247 | 217 | 239 | +2.15 | +8.20 |
| `28106` | 237 | 246 | 244 | 239 | 252 | 237 | 260 | +0.39 | +1.37 |
| `28107` | 205 | 214 | 210 | 207 | 207 | 205 | 202 | +0.39 | +0.98 |
| `28108` | 253 | 247 | 276 | 256 | 255 | 253 | 261 | +0.59 | +4.49 |
| **Total / Mean** | **1,921 (46.90%)** | **1,982 (48.39%)** | **2,058 (50.24%)** | **2,013 (49.15%)** | **1,936 (47.27%)** | **1,921 (46.90%)** | **2,025 (49.44%)** | **+2.25** | **+3.34** |

### Registered Contrast Family (Ordinary Delay-31, Steering Disabled)
| Contrast | Role | Macro Effect (pp) | Recovery / Corruption | Exact Sign-Flip p | Holm p | Bootstrap 95% CI (pp) | ≥5 pp Floor |
|---|---|---:|---:|---:|---:|---:|---|
| `D_mech_grad − A_uniform` | Primary (fixed-update) | **+2.25** | 571 / 479 (+92) | 0.2656 | 1.0000 | [−0.90, +6.01] | No |
| `D_mech_grad − B_loss_mined` | vs loss mining | **+0.76** | 472 / 441 (+31) | 0.7656 | 1.0000 | [−1.27, +3.44] | No |
| `D_mech_grad − E_shuffled` | vs shuffled sham | **+1.88** | 534 / 457 (+77) | 0.4844 | 1.0000 | [−2.25, +6.71] | No |
| `D_mech_grad − G_cpu_reinvest` | Primary (fixed-CPU) | **−0.29** | 577 / 589 (−12) | 0.8984 | 1.0000 | [−3.42, +3.20] | No |
| `C_ord_grad − A_uniform` | Ordinary grad vs uniform | **+3.34** | 533 / 396 (+137) | 0.0781 | 0.3906 | [+0.39, +5.96] | No |

---

## 6. Independent Audit Replay & DataGoblin Export

- **Independent Audit (`results/development/independent_audit.json`):** Reimplemented `ComponentTRM` directly from saved state_dict tensors without importing `model.py` or `campaign.py`. Audited all 8 roots (`488/488` checks, `57,344` replayed cases across `d=2` and `d=31`, `100%` exact prediction-bit match, `0.0` maximum scalar CE error). *(Custody note: `SOURCE_FREEZE.json` preserves the exact pre-development source hashes; a 2-line post-run fix in `audit.py`/`campaign.py` excluded `artifact_hashes.json` from hashing itself without touching any model, observer, scheduler, or training logic.)*
- **DataGoblin Train-Only Export (`results/datagoblin_export/`):** Exported `10,240` strictly TRAIN fit-view candidate curation decisions (`candidate_decisions.jsonl`, SHA-256 `805220a1...`), `809` successful positive-gain selected decisions, and `248` verified preference pairs (`curation_preferences.jsonl`, SHA-256 `296260c7...`), with `0` evaluation rows exported and fail-closed split custody verified by unit test.

### Reproduce Verification & Audit
```bash
python -m unittest mechinterp.bpd.experiments.trm_curation.test_reference \
  mechinterp.bpd.experiments.trm_curation.test_model \
  mechinterp.bpd.experiments.trm_curation.test_v2 \
  mechinterp.bpd.experiments.trm_curation.test_complete -v

python -m mechinterp.bpd.experiments.trm_curation.audit \
  mechinterp/bpd/experiments/trm_curation/results/development \
  --out /tmp/bpd_trm_curate02_audit.json
```

---

## 7. Recommendation for the Next Mathematical Improvement

Two quantitative bottlenecks account for why `C_ord_grad` (`+3.34 pp`) and `D_mech_grad` (`+2.25 pp`) fell short of `+5.0 pp` and lost fixed-CPU efficiency to `G_cpu_reinvest`:

1. **Eliminate the 16× rollback-probe CPU tax via AdamW-preconditioned gradient screening:** First-order `-dot(g_cal, delta_theta)` already achieves `0.92–0.94` Spearman rank correlation and `100%` top-4 positivity, yet `probe_update` executes full state/optimizer cloning, clipping, stepping, calibration replay, and rollback for every candidate (`~100 s` vs `6.3 s` for `A_uniform`). Replace per-candidate rollback with an analytical diagonal-AdamW-preconditioned inner product:
   $$\hat{u}_i = -\left\langle g_{\text{cal}}, \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \text{clip}(g_i) \right\rangle$$
   computed from one cached `g_cal` and one backward pass per candidate (no optimizer deep-copy or per-candidate calibration forward), reserving exact transactional rollback (`probe_update`) solely for the final 4-edge mini-batch commit. This cuts screening overhead by ~5–8× while retaining the **86× update sample efficiency**.
2. **Decouple causal selectivity from saturated short-horizon training via adaptive margin-aware feedback weighting:** `D_mech_grad` gained **+9.38 pp** on full-hybrid causal oracle transport (`E19`) and topped delay-127 (`31.27%`), while `C_ord_grad` led delay-31 (`50.24%`). Because Phase B revealed that converged clean logit margins (~3–6 logits) block fixed-dose (`0.75`) linear flips, a next-generation feedback objective should weight ordinary long-delay retention (`d=15/31`) primarily during curriculum ascent and introduce a **margin-normalized SCCG-01 directional alignment term** only on high-leverage mid-margin edges (or on tasks like `E29_dyck1_bounded_depth` and `E31_2hop_pointer_chase` identified in [EVAL_36_EXPLORATION.md](EVAL_36_EXPLORATION.md)).
