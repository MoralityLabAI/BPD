# BPD-TRM-GYM-01: Paired Multi-Task TRM Curriculum Hill-Climbing Campaign

**Evidence class:** `DEVELOPMENT_ONLY` (8 fresh independent roots `29201–29208` per task, strictly separate from CURATE-02 `28101–28108`, CURATE-03B pilot `28301–28308`, and exploratory `29000–29101`).
**Decision:** EvalMath `REDESIGN_OR_PILOT_ONLY` is preserved; no training method is promoted. All CURATE-02 frozen files, CURATE-03B frozen pilot artifacts, and the disabled selective causal-auxiliary intervention are preserved untouched.
**Hardware & custody:** CPU-only (`torch 2.15.0+cpu`, 2 Torch threads/worker, 3 parallel workers), hard **6 GiB process-tree commit cap enforced** via a single parent Win32 Job Object (`peak_job_commit_bytes = 1.30 GiB / 6.00 GiB`, wall time `1,260.7 s`).

## Experimental Design

- **Model:** Native weight-tied `ComponentTRM(width=32, components=29, low_cycles=1, input_dim=11, values=4)` (5,380 trainable parameters).
- **Tasks (5 diverse low-difficulty TRM benchmarks, strictly disjoint TRAIN/EVAL):**
  1. `E29_dyck1_bounded_depth` — length-extrapolation target (train len 8 → eval len 16; warmup 250 steps).
  2. `E31_2hop_pointer_chase` — iterative multi-hop reasoning target (`reg[reg[q]]`, 2 think tokens, mod-5 split; warmup 250 steps).
  3. `E28_dynamic_overwrite_register` — distribution-shift control (6 writes train → 10 writes eval; warmup 250 steps).
  4. `E27_copy_retrieve_2slot` — extrapolation-collapse control (delay 6 train → delay 14 eval, mod-3 split; warmup 250 steps).
  5. `G05_3slot_copy_retrieve` — 3-slot associative memory bridge (delay 5 train → delay 10 eval, mod-3 split; warmup 200 steps).
- **Paired arms (from identical shared warmup checkpoint per `(task, root)`, fresh AdamW `lr=0.001, wd=1e-4, clip=1.0`, matched `25.0` process CPU-s/arm; evaluation CPU excluded, all screening/custody/audit CPU charged):**
  - `A_uniform_reinvest`: 100% CPU reinvested into uniform updates.
  - `B_loss_pool_2x`: forward-only 2× pool hard-example selection (80% top-loss + 20% deterministic uniform exploration; zero extra backward passes).
  - `C_analytic_full16`: 16-candidate analytical AdamW screen (`adamw_screen.py`) + matched-RNG TRAIN feedback transactional acceptance.
  - `D_joint_early12`: prospective `early12` paid acquisition gate (`p=0.25` randomized shadow audits charged to arm) + 1-round joint minibatch hill climbing (`batch_screen.py`) + transactional acceptance.
  - `E_fast_analytic_k4`: low-overhead `K=4` analytical AdamW screen with TRAIN feedback gradient refreshed every 4 updates, selecting top-2 aligned micro-batches directly without transactional rollback tax (~4–5× more updates than `C`/`D`).
  - `F_shuffled_k4`: compute-matched sham control for `E` (pays identical `K=4` gradient + periodic feedback CPU, shuffles scores before picking top-2).

## Engineering & Custody Verification

- **Observer-only fixed-update invariance:** Byte-identical checkpoints and custody digests (`state_identical = true`, `custody_identical = true`) on **40 / 40** `(task, root)` pairs.
- **Native vs aggregated joint gradient parity:** **100%** within frozen float32 tolerance across every `C_analytic_full16` and `D_joint_early12` window (`max rel L2 = 2.94e-7 <= 1e-5`).
- **Budget discipline:** Max single-window overshoot `<= 1.11 s` (`D`), `<= 0.08 s` (`A`, `B`), `<= 0.31 s` (`E`, `F`).
- **Shadow audits (`D_joint_early12`, `p=0.25`):** 93 abstained windows across the campaign; 18 audited (0 would have been accepted), 75 unaudited remain explicitly `missing_unaudited`.

## Pooled Results Across All 5 Gym Tasks (8 Independent Roots)

| Arm | Pooled EVAL Acc (%) | Mean Δ vs A (pp) | +/− Roots | Exact Sign-Flip p | Holm p (4 contrasts) | 95% Root Bootstrap (pp) |
|---|---:|---:|---:|---:|---:|---|
| `A_uniform_reinvest` | **42.51** | — | — | — | — | — |
| `F_shuffled_k4` (sham) | 42.33 | −0.19 | — | — | — | — |
| `D_joint_early12` | 41.87 | −0.64 | 4/4 | 0.6094 | 1.0000 | [−2.94, +1.63] |
| `C_analytic_full16` | 41.85 | −0.67 | 4/4 | 0.6328 | 1.0000 | [−3.06, +1.74] |
| `E_fast_analytic_k4` | 41.46 | −1.05 | 5/3 | 0.3516 | 1.0000 | [−3.08, +0.69] |
| `B_loss_pool_2x` | 41.43 | −1.08 | 3/5 | 0.4609 | 1.0000 | [−3.71, +1.51] |

Pooled over all 5 tasks, no curriculum arm outperforms uniform CPU reinvestment under matched CPU, and none meets the `+5 pp` scientific floor.

## Per-Task Breakdown (8-Root Means @ 25.0 Process CPU-s / Arm)

| Task | Warmup EV | `A` Uniform | `B` Loss-2× | `C` Full-16 | `D` Joint-E12 | `E` Fast-K4 | `F` Shuf-K4 | `A` / `E` / `C` Commits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E31_2hop_pointer_chase` | 58.06 | **66.77** | 63.67 | 62.77 | 62.77 | 62.04 | 64.38 | 793 / 160 / 23 |
| `E28_dynamic_overwrite_register` | 40.36 | 40.43 | 36.43 | 40.75 | 41.31 | **42.77** | 41.33 | 860 / 142 / 32 |
| `G05_3slot_copy_retrieve` | 37.26 | 37.60 | 32.89 | **37.84** | 37.35 | 34.67 | 36.30 | 498 / 98 / 17 |
| `E27_copy_retrieve_2slot` | 29.83 | 35.16 | **41.48** | 36.11 | 36.16 | 36.38 | 37.99 | 494 / 98 / 10 |
| `E29_dyck1_bounded_depth` | 31.76 | 32.62 | **32.69** | 31.76 | 31.76 | 31.47 | 31.64 | 760 / 149 / 0 |

### Key Mechanistic & Statistical Findings

1. **Static TRAIN feedback alignment overfits on clean iterative reasoning (`E31_2hop_pointer_chase`, 8/8 roots competent & in headroom):**
   - Every curated arm significantly trailed `A_uniform_reinvest` (`66.77%`): `C` −4.00 pp (0/8 positive, Holm `p = 0.0312`), `D` −4.00 pp (0/8 positive, Holm `p = 0.0312`), `E` −4.74 pp (0/8 positive, Holm `p = 0.0312`), `B` −3.10 pp (2/6, Holm `p = 0.0469`).
   - Crucially, `E_fast_analytic_k4` (`62.04%`) trailed its **exact compute-matched shuffled sham `F_shuffled_k4` (`64.38%`) by −2.34 pp on 7/8 roots (`p = 0.0234`)**, even though warmup analytical screening achieved `Spearman = 0.851` with realized one-step TRAIN feedback gain. Greedily aligning updates to a static 128-example TRAIN feedback cohort harms out-of-sample generalization even at identical update count.
2. **Low-overhead `E_fast_analytic_k4` shows a genuine, sham-beating alignment signal under distribution shift (`E28_dynamic_overwrite_register`, 8/8 competent, 7/8 headroom) — though still below `+5 pp`:**
   - `E_fast_analytic_k4` reached `42.77%` (`+2.34 pp` vs `A`, 6/8 roots, 95% boot `[+0.10, +4.88]`, Holm `p = 0.3516`), beating both `C_analytic_full16` by **`+2.03 pp` on 8/8 roots (`p = 0.0078`)** and its compute-matched shuffled sham `F_shuffled_k4` by **`+1.44 pp` on 7/8 roots (`p = 0.0156`)**.
   - Forward-only hard-example mining `B_loss_pool_2x` degraded `E28` by −4.00 pp (6/8 negative, `p = 0.0469`).
3. **Near-saturated TRAIN feedback freezes transactional curation on length extrapolation (`E29_dyck1_bounded_depth`):**
   - By warmup step 250 (`train = 88.7%`, `eval = 31.8%`), `finite_dose_error` spiked to `0.1016` (10× higher than `E28`), analytical Spearman collapsed to `0.199` (`sign_acc = 0.477`), and **0.0 / 45.6** attempted updates passed the TRAIN feedback acceptance test across all 8 roots for both `C` and `D`.
4. **Warmup Mechanistic Signal Bank (8-root means per task):**

| Task | `pred_gain` Spearman | `grad_cos` Spearman | `observer_signed` Spearman | `target_transport` Spearman | `prot_interf` Spearman | Finite-Dose Error | Sign Acc | State PR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E31_2hop_pointer_chase` | 0.851 | 0.846 | 0.818 | 0.733 | 0.631 | 0.0151 | 0.820 | 4.02 |
| `G05_3slot_copy_retrieve` | 0.817 | 0.810 | 0.658 | 0.508 | 0.237 | 0.0132 | 0.844 | 2.26 |
| `E28_dynamic_overwrite_register` | 0.780 | 0.789 | 0.778 | 0.558 | 0.201 | 0.0070 | 0.898 | 3.43 |
| `E27_copy_retrieve_2slot` | 0.778 | 0.762 | 0.614 | 0.616 | 0.253 | 0.0161 | 0.773 | 1.98 |
| `E29_dyck1_bounded_depth` | 0.199 | −0.104 | 0.513 | 0.552 | 0.894 | 0.1016 | 0.477 | 1.57 |

## DataGoblin V3 Export & Replay

- Export directory: `results/curate03b/gym_campaign/datagoblin_export_v3/` (strictly TRAIN-only, 60,286 windows observed, 13,329 scored decision rows exported, 11-action vocabulary, separate `horizon_outcomes_gym_v3.jsonl`, total compressed directory `6.12 MiB`).
- DataGoblin root-held-out paid-acquisition policy evaluation and Inky Mini multi-task gym traces live in `MoralityLabAI/DataGoblin` PR #100 (`training/rlt_gym/hypergame_taste/bpd_trm_curate03b_gym/`).

```bash
python -m unittest mechinterp.bpd.experiments.trm_curation.test_gym_campaign -v
python -m mechinterp.bpd.experiments.trm_curation.gym_campaign --out mechinterp/bpd/experiments/trm_curation/results/curate03b/gym_campaign --budget-cpu 25.0 --threads 2 --workers 3
```
