# 36 Low-Difficulty TRM Evaluations Exploration (`E01–E36`)

**Receipt:** `results/eval_suite_36/eval_suite_36_results.json` | **Memory Cap:** `Win32 Job Object` 6.0 GiB (`peak_job_commit = 1,610.4 MiB`, `peak_tree_rss = 253.4 MiB`, `wall = 3,988.4 s`, `process_cpu = 8,058.8 s`) | **Sealed Cohorts Touched:** `0`

This suite explores **36 low-difficulty evaluations** specifically matched to compact weight-tied TRMs (`ComponentTRM`, 5,380 parameters):
- **Families 1–4 (`E01–E24`):** 24 diagnostic evaluations across all **8 fresh development roots (`28101..28108`)** comparing Arms `A_uniform`, `B_loss_mined`, `C_ord_grad`, `D_mech_grad`, `E_shuffled`, and `G_cpu_reinvest`.
- **Families 5–6 (`E25–E36`):** 12 fresh standalone low-difficulty algorithmic & iterative-refinement micro-benchmarks (`450` steps, 2 fresh roots `29000..29001`, matched 11-dim interface) comparing `GRU_w32`, `TRM_c1_uniform` (`low_cycles=1`), `TRM_c2_uniform` (`low_cycles=2`), and `TRM_c1_curated` (hard-example pool curation with 20% exploration).

---

## Families 1–4 (`E01–E24`): 8-Root Development TRM Diagnostics

| ID | Evaluation | Family | `A_uniform` | `B_loss` | `C_ord_grad` | `D_mech_grad` | `E_shuffled` | `G_reinvest` | `D − A` (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `E01` | `mem_delay_0` | F1 Horizon | 92.14% | 92.33% | 91.11% | 89.99% | 90.97% | **93.55%** | −2.15 |
| `E02` | `mem_delay_2` | F1 Horizon | 98.73% | 99.17% | 98.39% | 98.44% | 98.88% | **99.66%** | −0.29 |
| `E03` | `mem_delay_7` | F1 Horizon | 94.68% | 95.85% | 94.14% | 93.41% | 94.29% | **96.53%** | −1.27 |
| `E04` | `mem_delay_15` | F1 Horizon | 74.51% | 77.00% | **78.76%** | 76.46% | 76.37% | 77.34% | **+1.95** |
| `E05` | `mem_delay_31` | F1 Horizon | 45.75% | 45.36% | **49.46%** | 48.14% | 45.12% | 46.97% | **+2.39** |
| `E06` | `mem_delay_63_127_mean` | F1 Horizon | 31.45% | 31.76% | 33.11% | 32.86% | 30.91% | **33.62%** | **+1.42** |
| `E07` | `binding_all_distinct_d7` | F2 Binding | 97.75% | 98.24% | 98.24% | 98.14% | 98.05% | **99.02%** | **+0.39** |
| `E08` | `binding_all_distinct_d31` | F2 Binding | 45.70% | 44.63% | 47.07% | 46.88% | 43.46% | **50.10%** | **+1.17** |
| `E09` | `noise_clean_000_d31` | F2 Binding | 47.66% | 48.14% | **49.76%** | 49.56% | 48.73% | 49.12% | **+1.90** |
| `E10` | `noise_elevated_010_d31` | F2 Binding | 44.58% | 46.53% | **47.90%** | **47.90%** | 45.12% | 47.31% | **+3.32** |
| `E11` | `noise_high_020_d31` | F2 Binding | 41.16% | 41.85% | 42.97% | **43.65%** | 40.77% | 43.16% | **+2.49** |
| `E12` | `write_order_invariance` | F2 Binding | 91.33% | 93.30% | 94.56% | 94.83% | 95.39% | **96.38%** | **+3.50** |
| `E13` | `distractor_adv_same_addr` | F3 Stress | 36.82% | 38.18% | 39.89% | 38.13% | 38.38% | **41.94%** | **+1.32** |
| `E14` | `distractor_repeat_flood` | F3 Stress | 43.12% | 43.46% | **46.63%** | 44.19% | 43.75% | 46.19% | **+1.07** |
| `E15` | `distractor_zero_silent` | F3 Stress | 45.70% | 45.02% | 48.19% | **48.58%** | 45.02% | 46.39% | **+2.88** |
| `E16` | `slot_collision_3of4_d15` | F3 Stress | 72.10% | 76.66% | 76.02% | 75.09% | 75.77% | **81.44%** | **+2.99** |
| `E17` | `calibration_1_minus_ece` | F3 Stress | 53.38% | 53.34% | **57.68%** | 56.56% | 52.85% | 54.01% | **+3.18** |
| `E18` | `transition_shift_eval_d31` | F3 Stress | 47.14% | 48.18% | **53.13%** | 48.96% | 50.00% | 46.35% | **+1.82** |
| `E19` | `causal_full_hybrid_oracle` | F4 Causal | 47.92% | 53.13% | 52.08% | **57.29%** | 50.00% | 53.13% | **+9.38** |
| `E20` | `causal_high_vs_low_ratio` | F4 Causal | 70.43% | 71.01% | 71.37% | 71.33% | 71.13% | **71.48%** | **+0.90** |
| `E21` | `causal_sccg_zero_col_d31` | F4 Causal | 17.71% | **22.92%** | 17.71% | 21.88% | 15.63% | 15.63% | **+4.17** |
| `E22` | `causal_sccg_b025_net_sel` | F4 Causal | 17.03% | **22.92%** | 18.41% | 20.45% | 19.10% | 16.09% | **+3.42** |
| `E23` | `causal_lin_nonlin_fidelity` | F4 Causal | 76.88% | 73.55% | 77.01% | 77.34% | **78.82%** | 77.57% | **+0.46** |
| `E24` | `causal_non_cancel_1mK` | F4 Causal | 80.23% | 80.22% | **81.72%** | 80.39% | 80.99% | 80.53% | **+0.16** |

**Takeaway on `E01–E24`:** `D_mech_grad` beats `A_uniform` on **21 of 24** evaluations (all long-horizon, binding, noise, distractor, calibration, and causal metrics `E04–E24`), with its largest margin on **causal full-hybrid transport (`E19`: +9.38 pp)** and **high-noise / silent-distractor robustness (`E10`: +3.32 pp, `E15`: +2.88 pp)**. `C_ord_grad` leads pure ordinary extrapolation (`E04`, `E05`, `E09`, `E14`, `E17`, `E18`).

---

## Families 5–6 (`E25–E36`): 12 Standalone Low-Difficulty TRM Micro-Benchmarks

Each micro-benchmark trains matched 5,380-parameter models (`GRU_w32`, `TRM_c1_uniform`, `TRM_c2_uniform`, `TRM_c1_curated`) for `450` steps across 2 fresh roots (`29000`, `29001`).

| ID | Benchmark | Family | `GRU_w32` | `TRM_c1` | `TRM_c2` | `TRM_curated` | `c2 − c1` (pp) | `cur − c1` (pp) | TRM Suitability Tier |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `E25` | `parity_mod2` (len 8→16) | F5 Algo | 50.68% | 50.98% | 50.29% | 49.41% | −0.68 | −1.56 | **Too hard** (chance floor @ 450 steps) |
| `E26` | `mod4_accumulator` (8→16) | F5 Algo | 24.12% | 24.22% | 24.22% | 25.20% | 0.00 | +0.98 | **Too hard** (chance floor @ 450 steps) |
| `E27` | `copy_retrieve_2slot` (d 6→14) | F5 Algo | 51.76% | 46.48% | 45.12% | 23.54% | −1.37 | −22.95 | **Sweet spot** (reveals naive loss-mining shift overfitting!) |
| `E28` | `dynamic_overwrite_register` | F5 Algo | 44.53% | 41.60% | 41.31% | 38.18% | −0.29 | −3.42 | **Sweet spot** (clean state-update headroom) |
| `E29` | `dyck1_bounded_depth` (8→16) | F5 Algo | 100.00% | 41.80% | **47.85%** | **54.39%** | **+6.05** | **+12.60** | **Top TRM benchmark!** (strong cycle + curation gain) |
| `E30` | `majority_vote_4sym` (9→13) | F5 Algo | 99.22% | 97.56% | **98.83%** | 97.07% | +1.27 | −0.49 | Near-ceiling sanity check |
| `E31` | `2hop_pointer_chase` (disjoint) | F6 Iter | 71.00% | 62.40% | **65.82%** | 60.06% | **+3.42** | −2.34 | **Top TRM reasoning benchmark!** (clear 2-cycle refinement gain) |
| `E32` | `latin4_1blank` | F6 Iter | 100.00% | 100.00% | 100.00% | 100.00% | 0.00 | 0.00 | Saturated sanity check (100%) |
| `E33` | `latin4_2blank_coupled` | F6 Iter | 100.00% | 100.00% | 100.00% | 100.00% | 0.00 | 0.00 | Saturated sanity check (100%) |
| `E34` | `argmin4_index` | F6 Iter | 100.00% | 100.00% | 100.00% | 100.00% | 0.00 | 0.00 | Saturated sanity check (100%) |
| `E35` | `cellular_rule90_2step` | F6 Iter | 100.00% | 100.00% | 100.00% | 100.00% | 0.00 | 0.00 | Saturated sanity check (100%) |
| `E36` | `dfa_4state_extrap` (8→20) | F6 Iter | 24.22% | 25.00% | 25.39% | 25.88% | +0.39 | +0.88 | **Too hard** (chance floor @ 450 steps) |

### Key Conclusions for Choosing Low-Difficulty TRM Benchmarks
1. **Best low-difficulty TRM benchmarks with active headroom (recommended for future TRM curation & cycle-scaling):**
   - `E29_dyck1_bounded_depth` (`TRM_c1` reaches `99.8%` in-distribution but `41.8%` at 2× length; adding a 2nd low cycle yields **+6.05 pp** and curation yields **+12.60 pp**!)
   - `E31_2hop_pointer_chase` (disjoint mod-5 split; 2nd low cycle yields **+3.42 pp** directly from extra latent refinement!)
   - `E27_copy_retrieve_2slot` & `E28_dynamic_overwrite_register` (41–46% TRM eval headroom; `E27` is an especially sharp test against naive hard-example overfitting under distribution shift!)
   - Delayed-memory `E04–E19` (`d=15/31`, elevated noise `E10–E11`, silent distractors `E15`, causal oracle `E19`).
2. **Saturated low-difficulty controls (100% pass by 450 steps — use as quick unit/competence gates, not curation targets):** `E32` (`latin4_1blank`), `E33` (`latin4_2blank_coupled`), `E34` (`argmin4_index`), `E35` (`cellular_rule90_2step`), and `E30` (`majority_vote_4sym`).
3. **Hard algebraic extrapolation tasks (`E25` parity, `E26` mod-4 sum, `E36` 4-state DFA):** Remain at chance at 450 steps for both GRU and TRM; require length curricula or higher step budgets rather than low-difficulty short-budget probes.
