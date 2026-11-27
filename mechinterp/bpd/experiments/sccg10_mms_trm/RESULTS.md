# SCCG-10 / MMS-TRM engineering results

Evidence class: `ENGINEERING_ONLY` (fresh roots 29200/29201; historical
CURATE-02 roots 28000/28001 used only for replay-style drift description).
No development root had been warmed or scored when these results were written.
The development protocol is frozen in `DEV_PROTOCOL.md` / `DEV_LOCK.json`;
development outcomes, if present, are in `results/development` and reported
separately below.

Source: branch `research/sccg10-mms-drift-20261127`, base CURATE-02
`af43e48` merged with BPD main `ab93e4d` (SCCG-01..09) and PR 175 `41a06e8`
(SCCG-10 reference). Transactional engineering pass on commit `dd2d5ba`;
batched pass and drift receipts record per-file LF-normalized source hashes.
Mathematics: `MATH_REVIEW.md`. Lean: MathCan `f0e6154`.

## Bottom line

1. **Mechanisms do not persist across committed updates on this model.** One
   sign-like AdamW step (‖Δθ‖ = 0.073) moves the full-suffix target derivative
   by about 47% of its norm and rotates the protected near-null space by about
   36°. After 8 updates the target drift exceeds 100%. A cross-checkpoint
   observer is a heuristic proposal at every age ≥ 1, never a certificate.
2. **What does persist: low-rank drift and supports.** ΔC carries 90% of its
   energy in 3–4 directions (of 27 rows). Cached k = 4 supports re-solved with
   exact live derivatives keep 66–91% of fresh-search gain at the collateral
   budget, while cached directions lose protection (collateral 0.28 → 0.52 against a
   0.25 budget).
3. **Explicit native verification is the economical tier.** It costs 0.13 s per
   candidate versus 1.8–2.3 s for refresh plus support search. No native
   uniform bound exists, so admission is always verification, never certification.
4. **The audit's dominant cost is candidate previews (63%), not the observer
   (20%).** A cross-checkpoint calibration atlas can remove only the minority
   share. The batched analytic AdamW preview removes most of the dominant
   share with selection-identical results (same fixed-update SHAs as
   CURATE-02's transactional probe on both roots, at 2.2–3.0× lower total arm
   CPU, and 4.4–5.1× against the full `probe_update`).
5. **The primary engineering contrast (D atlas vs A ordinary at equal CPU) has
   mixed signs on two roots in both passes.** Transactional: −1.8 and −6.6 pp.
   Batched: −0.8 and +2.5 pp. Holm p = 1.0. EvalMath returns
   `REDESIGN_OR_PILOT_ONLY`. No advantage is claimed.
6. **The limiting factor is alignment, not compute.** After warmup, training-delay
   (1–3) updates rarely improve the delay-15 feedback objective. Curated arms
   act mostly as a gate that rejects updates harmful to long-delay memory, and
   reinvesting compute in ordinary training *lowers* d31 on both roots
   (A: 80 updates 32.0% / 45.7% → about 1,780 updates 27.9–28.5% / 43.0–43.2%).

## Verification

| Check | Result |
|---|---|
| CURATE-02 focused tests (parity, suffix continuation, rollback) | 23/23 pass |
| SCCG-10 reference admission tests (PR 175) | 8/8 pass |
| SCCG-10/MMS-TRM tests (`test_mms.py`) | 15/15 pass |
| Instrument rows vs `observe_edge` signed observer | max abs diff 0.0 on all 4 drift roots |
| Forward-mode JVP vs reverse rows | < 1e-4 (test), exact vs `torch.func.jvp` |
| `preview_update` vs `probe_update` predicted gain | bitwise equal; full restore of model, optimizer, RNG |
| Transactional C (preview) vs C_probe (full probe), 80 windows | identical final SHA on both roots |
| Batched B, C vs transactional B, C, 80 windows | identical final SHAs on both roots |
| F_obs_only vs A_ordinary, 80 windows | identical final SHA on both roots (no observation leakage) |
| Independent replay (`replay.json`) | 64/64 saved economics checkpoints match state SHA and every delay/binding count; 8/8 deterministic reruns (A, C × 2 passes × 2 roots) reproduce receipt SHAs; 32/32 drift checkpoint SHAs match |
| Process-tree memory | Win32 Job Object 6 GiB cap enforced; drift peak commit 1.59–1.60 GiB |

D_atlas and E_shuffled are not bitwise replayable by re-execution: their
value-of-audit rule reads measured CPU time. Replay verifies their saved
checkpoints and outcome counts instead.

## Workstream B: checkpoint drift (pooled over 4 roots × 24 TRAIN panel edges)

See `MATH_REVIEW.md` section 8 for the full table. Key rows:

| updates | ‖Δθ‖ | state α/‖a₀‖ | state γ/‖C₀‖ | null-space sin | cached k4 gain / collateral | restricted k4 gain / collateral | live k4 gain / collateral |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 0.073 | 0.47 | 0.33 | 0.59 | 0.110 / 0.277 | 0.109 / 0.253 | 0.120 / 0.255 |
| 4 | 0.167 | 1.08 | 0.64 | 0.89 | 0.108 / 0.324 | 0.102 / 0.253 | 0.129 / 0.253 |
| 16 | 0.315 | 1.62 | 1.02 | 0.96 | 0.094 / 0.461 | 0.091 / 0.252 | 0.127 / 0.255 |
| 64 | 0.488 | 3.09 | 1.06 | 0.98 | 0.062 / 0.521 | 0.085 / 0.250 | 0.128 / 0.268 |

Carried-state edits at dose 0.75 with collateral budget 0.25 (protected
relative-logit ℓ₂ over 27 rows). Native factor writes give 0.04–0.07 gain. The
heuristic SCCG-10 admission never returned `CONDITIONAL_REUSE_CANDIDATE`
(1,344 cases). The privileged full hybrid at matched dose flips 5 of 78
eligible targets with 10 of 674 protected corruptions. The behavioral
selective-control gate fails again in kind, and causal auxiliary training
stays closed.

## Workstreams D/E: compute economics (engineering, 2 roots)

Fixed-update = 80 windows. Fixed-CPU = 100 process CPU-s per arm after
warmup, including setup, cache/atlas traffic, previews, rollbacks and discarded
updates. Accuracies are ordinary steering-disabled accuracies on 512 held
cases per delay (historical EVAL partition, exposed).

#### transactional previews
| Arm | budget | 29200 windows / accepted / d31 / d63 | 29201 windows / accepted / d31 / d63 |
|---|---|---|---|
| A_ordinary | cpu | 1781 / 1781 / 28.5% / 24.2% | 1584 / 1584 / 43.2% / 34.0% |
| B_ord_grad | cpu | 180 / 38 / 30.3% / 25.4% | 176 / 16 / 44.1% / 33.0% |
| C_mech_fresh | cpu | 152 / 32 / 32.6% / 24.8% | 154 / 79 / 43.8% / 35.2% |
| D_atlas_mms0 | cpu | 162 / 44 / 28.9% / 26.0% | 199 / 103 / 43.4% / 36.1% |
| D_atlas_mms4 | cpu | 210 / 71 / 27.7% / 24.8% | 273 / 166 / 34.0% / 32.6% |
| D_atlas_mms16 | cpu | 205 / 67 / 26.8% / 25.0% | 268 / 173 / 36.5% / 34.6% |
| D_atlas_mms64 | cpu | 217 / 70 / 27.5% / 26.0% | 278 / 176 / 29.7% / 32.4% |
| E_shuffled_mms16 | cpu | 209 / 68 / 28.3% / 25.6% | 212 / 66 / 43.0% / 35.2% |
| F_obs_only_mms16 | cpu | 294 / 294 / 27.9% / 23.2% | 332 / 332 / 40.6% / 34.8% |
| A_ordinary | updates | 80 / 80 / 32.0% / 22.7% | 80 / 80 / 45.7% / 37.3% |
| B_ord_grad | updates | 80 / 23 / 31.6% / 24.2% | 80 / 16 / 44.1% / 33.0% |
| C_mech_fresh | updates | 80 / 22 / 33.0% / 25.8% | 80 / 18 / 43.2% / 35.0% |
| C_probe | updates | 80 / 22 / 33.0% / 25.8% | 80 / 18 / 43.2% / 35.0% |
| D_atlas_mms16 | updates | 80 / 58 / 27.7% / 25.4% | 80 / 56 / 45.5% / 36.1% |
| E_shuffled_mms16 | updates | 80 / 5 / 29.5% / 23.6% | 80 / 41 / 42.2% / 34.0% |
| F_obs_only_mms16 | updates | 80 / 80 / 32.0% / 22.7% | 80 / 80 / 45.7% / 37.3% |

#### batched analytic previews
| Arm | budget | 29200 windows / accepted / d31 / d63 | 29201 windows / accepted / d31 / d63 |
|---|---|---|---|
| A_ordinary | cpu | 1790 / 1790 / 27.9% / 23.6% | 1786 / 1786 / 43.0% / 28.5% |
| B_ord_grad | cpu | 524 / 39 / 29.5% / 25.4% | 528 / 52 / 40.0% / 32.4% |
| C_mech_fresh | cpu | 433 / 46 / 33.8% / 25.6% | 427 / 104 / 41.4% / 35.4% |
| D_atlas_mms0 | cpu | 337 / 48 / 31.8% / 25.8% | 382 / 101 / 39.5% / 36.3% |
| D_atlas_mms4 | cpu | 497 / 64 / 26.4% / 25.2% | 567 / 177 / 45.5% / 36.3% |
| D_atlas_mms16 | cpu | 529 / 61 / 27.1% / 25.4% | 548 / 179 / 45.5% / 37.5% |
| D_atlas_mms64 | cpu | 534 / 61 / 26.4% / 25.8% | 530 / 148 / 42.4% / 37.1% |
| E_shuffled_mms16 | cpu | 552 / 101 / 27.5% / 25.2% | 528 / 38 / 41.4% / 34.2% |
| F_obs_only_mms16 | cpu | 663 / 663 / 27.1% / 23.4% | 706 / 706 / 35.7% / 26.4% |
| (fixed-update rows) | updates | identical to transactional for A, B, C, C_probe, F | D: 80/48/27.1%, 80/65/44.7%; E: 80/57/45.3% on 29201 |

Registered engineering contrasts (delay 31, fixed CPU, per root 29200 / 29201):

| Contrast | transactional | batched |
|---|---|---|
| D − A (primary) | −1.8 / −6.6 pp | −0.8 / +2.5 pp |
| D − C | −5.9 / −7.2 pp | −6.6 / +4.1 pp |
| D − B | −3.5 / −7.6 pp | −2.3 / +5.5 pp |
| D − E | −1.6 / −6.4 pp | −0.4 / +4.1 pp |

All Holm p = 1.0 (two roots cannot reach significance). EvalMath preflight:
`REDESIGN_OR_PILOT_ONLY` (heldout unsealed, sham quantile uncalibrated, too few
pilot worlds, pilot effect not above the variation floor). Sampling SE per arm
at 512 cases is about 2.1 pp.

Cost and policy diagnostics:
* Audited window (transactional): 16 previews 0.41 s, calibration gradient
  about 0.14 s, acceptance about 0.1 s; plain window about 0.06 s. Atlas-reuse
  windows skip only the calibration gradient.
* Atlas trace residual (predicted vs realized committed-step feedback change
  under a reused observer): median 0.06–0.30. Reused observers predict the
  committed step about as well as fresh ones, yet D commits more of its
  CPU to plain windows (value-of-audit rule) and so trains more on
  misaligned short-delay data.
* In 17–75% of audited windows even the best single candidate has positive
  predicted delay-15 feedback gain. Realized committed-step gains are positive
  in 17–51% of windows. Acceptance rates for C are 11–51%.
* One-step predicted vs realized committed gain: Spearman 0.84–0.99. Summed
  multi-window predicted gain tracks the realized 4/8-window feedback change
  better than the one-step score for C (0.86–0.99 vs 0.24–0.77). For C this is
  partly additivity by construction. It concerns the TRAIN feedback objective,
  not held accuracy.
* Atlas tiers 0/4/16/64 MiB: the atlas payload is 88.7 KB per entry, so tiers
  above 0 behave alike. The tier sweep tests reuse versus no reuse, not memory
  pressure.

## Workstream F: DataGoblin

`results/datagoblin_export/`: 37,408 TRAIN-only curriculum decisions (C and D,
both preview modes) and 13,344 intervention decisions (drift instrument).
Every record carries scope (`EXACT_CURRENT_OBSERVER`, `HEURISTIC_PROPOSAL`,
`VERIFIED_FINITE_ON_DECLARED_QUERIES`), estimated vs realized value,
provenance, dose, cost and verifier outcome. Historical-root rows are
marked `training_eligible: false`. No scheduler was trained, because the
exact and ordinary baselines established no curation headroom (item 5 above).
That gate stays closed.

## What would have to change

* Align the training stream with the target horizon, or score candidates on
  an objective whose gradient aligns with held long-delay accuracy (MATH_REVIEW
  section 6, open problem 4). Without alignment, faster audits only gate faster.
* Use batched analytic previews as the default exact economy (open problem 5).
* Treat the atlas as a *support* memory with restricted refresh, not a
  direction or gradient memory. It is worth more once a forward-mode
  implementation costs about one forward per JVP instead of about 20.
