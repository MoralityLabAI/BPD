# BPD-TRM-CURATE-03B results: analytical screening, paid acquisition and joint hill climbing

**Evidence classes.** Engineering admission uses spent roots 28000/28001. The paired pilot uses fresh development roots 28301–28308. The gym uses exploratory roots 29100/29101. The DataGoblin controller evaluation is an offline development replay. **Decision:** EvalMath `REDESIGN_OR_PILOT_ONLY` is preserved and nothing is promoted. The CURATE-02 freeze, checkpoints, negative results and failed selective-control admission are untouched; the causal-auxiliary intervention was not run. Protocol: [CURATE_03B_PILOT_PROTOCOL.md](CURATE_03B_PILOT_PROTOCOL.md), committed at b30de73 before any native run.

## Bottom line

* **Engineering works.** Analytical AdamW screening reproduces the transactional probe exactly on native ComponentTRM: max absolute predicted-gain error 0.0 over 1,152 comparisons, and identical fixed-update checkpoints. It costs 0.36× the per-candidate CPU. Custody, observer invariance, joint-gradient parity and cache-tier parity all hold.
* **The science does not.** At a matched 100 process CPU-s per arm, every curated arm is *worse* than uniform CPU reinvestment on delay-31 accuracy, by 3.3–4.7 pp, negative on 6–7 of 8 roots (Holm p = 0.156 for each primary contrast). Uniform reinvestment buys about 1,857 updates. Curated arms commit 18–37, and 64–84% of their attempted windows fail the TRAIN feedback acceptance test.
* **Paid acquisition is cheap but not enough.** Gates abstained on 39–66% of windows with almost no audited loss: E 0/171, F 2/257, G 0/115 and H 3/154 audited abstentions would have been accepted. A learned stage controller (DataGoblin) keeps 99.58% of positive committed gain at 0.751× estimated CPU, offline. None of this closes a 1,857-vs-30 update gap.

## Phase 0: sources, regression and CI

* PR heads pulled: BPD 8f95b45, DataGoblin 92ce901.
* Hosted `trm-curation` failed at 8f95b45 because CI installed only torch/numpy, while `signal_bank.py` needs scipy and `memory_cap.py` needs psutil. Fix: [requirements.txt](requirements.txt) plus the workflow change (6ab0dbb). The check passes on b30de73 and 4257731.
* Local suite: 40 tests OK. That covers the existing 25 (including the 48-fixture joint screen, max score error 1.73e-18), 6 custody tests and 9 native screen tests. The 96-fixture AdamW parity smoke is unchanged from CURATE-03A.
* DataGoblin retrospective (275 rejected attempts avoided with 1 accept missed; early-12 99.82% at 9,040/10,240; held-out thresholds 99.13%) remains spent-development motivation.

## Phase 1: transactional custody ([custody.py](custody.py), [test_custody.py](test_custody.py))

The snapshot covers:

* parameters (in place) and gradients;
* every buffer slot, including non-persistent, resized and added buffers;
* every public module attribute, with deletion of new ones;
* AdamW state and groups;
* Torch, Python and NumPy RNG, and explicit generators.

Tests confirm exact digest restoration after a successful callback, after an exception (isolated and transaction), after a rejected update, and across repeated execution. A committed transaction equals the plain step. Observer-only instrumentation reproduces the uninstrumented trajectory, RNG included, under a stochastic training callback. Stochastic feedback re-evaluation with matched RNG draws gives identical values. Native fixed-update checks are byte-identical: A vs J on 28000, 28001 and all 8 pilot roots, and B vs D on 28000/28001.

## Phase 2: analytical screening (spent roots; `results/curate03b/engineering/`)

* 1,152 candidate comparisons across cold/warm optimizer, ordinary/mechanistic objectives and clip 1.0/0.05/1e6. Clipping was active in 17–67% of candidates, depending on regime.
* 100% within tolerance, max error 0.0, top-4 sets identical in every window.
* Per-candidate CPU (median): 0.0625 s analytical vs 0.172 s transactional.
* Fixed 40 windows: D used 36–41 CPU-s, B used 87–132.
* Spearman of analytical prediction with realized nonlinear gain: 0.89–0.93 cold, 0.97–0.99 warm.

Custody snapshot/restore overhead is logged per stratum in `score_rows`/`custody_stats`, and per arm in each `result.json`.

## Phase 3: joint minibatch

* Aggregated mean raw per-edge gradient vs native concatenated gradient: within the frozen float32 tolerance in 100% of 7,281 pilot windows and all engineering windows (rel L2 ≤ 2.63e-7). Native ComponentTRM is float32-only; `TiedComponentLinear` casts its input to float32.
* Profiling (60 spent-root windows): round 1 changed the batch in 57% of windows, mean predicted +0.0103 and mean actual gain 0.0218 → 0.0352. Round 2 added +0.0021 predicted at 0.242 s search CPU, more than 0.25 × 0.753 s acquisition, so the frozen rule kept one round.
* Pilot: G swapped in 50% of windows. Predicted-vs-actual joint gain Spearman is 0.61 for G vs 0.92 for D.

## Phase 4: paid acquisition

* The prospective order (SHA-256 of semantic edge key within slot, round-robin) was deterministic in every window.
* Engineering replay re-audit of the order: early gates missed 1 accept on 28000 (98.97% retained) and none on 28001.
* Pilot audits (probability 0.25, known draws, CPU charged): see the bottom line. Horvitz–Thompson estimated missed accepts: F 8, H 12; E and G 0. 2,238 unaudited abstentions stay missing.

## Phase 5: paired native pilot (`results/curate03b/development/summary.json`)

Means over 8 roots, at 100 CPU-s per arm:

| Arm | d31 | d63 | d127 | bind d31 | transport | prot. corrupt | windows | committed | abstained | CPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A uniform reinvest | **47.24** | **34.79** | 28.47 | **48.68** | **45.8** | 29.0 | 1857 | 1857 | 0 | 100.0 |
| B transactional | 44.38 | 32.59 | 27.29 | 48.34 | 41.9 | 24.5 | 71 | 18.1 | 0 | 100.9 |
| C analytical ordinary | 46.48 | 34.03 | 28.03 | 48.44 | 44.8 | 29.1 | 163 | 37.0 | 0 | 100.2 |
| D analytical mechanistic | 43.60 | 33.72 | 28.42 | 47.02 | 43.5 | 23.4 | 159 | 27.4 | 0 | 100.4 |
| E + full-16 gate | 43.63 | 33.72 | **28.56** | 46.83 | 44.3 | **21.8** | 177 | 28.9 | 96.5 | 100.3 |
| F + early-12 gate | 43.95 | 33.30 | 28.15 | 47.71 | 43.8 | 22.7 | 207 | 24.1 | 136.5 | 100.3 |
| G joint + full-16 | 43.29 | 32.64 | 26.56 | 44.34 | 39.3 | 24.3 | 150 | 25.2 | 59.0 | 100.3 |
| H joint + early-12 | 42.53 | 32.57 | 27.08 | 43.95 | 40.1 | 25.3 | 165 | 26.2 | 74.9 | 100.4 |
| I shuffled | 43.31 | 31.54 | 27.00 | 45.75 | 44.0 | 25.0 | 181 | 28.6 | 0 | 100.4 |
| J observer-only | 43.51 | 31.79 | 28.30 | 47.12 | 44.3 | 27.5 | 162 | 162 | 0 | 100.4 |

Per-root delay-31 correct counts (of 512):

| Root | A | B | C | D | E | F | G | H | I | J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 28301 | 280 | 263 | 267 | 277 | 273 | 268 | 256 | 256 | 277 | 260 |
| 28302 | 254 | 261 | 281 | 234 | 226 | 251 | 252 | 249 | 277 | 239 |
| 28303 | 271 | 241 | 272 | 241 | 241 | 241 | 265 | 265 | 238 | 233 |
| 28304 | 247 | 258 | 252 | 258 | 258 | 258 | 253 | 253 | 228 | 232 |
| 28305 | 193 | 202 | 198 | 184 | 182 | 193 | 189 | 189 | 203 | 215 |
| 28306 | 194 | 158 | 189 | 168 | 170 | 164 | 132 | 132 | 137 | 158 |
| 28307 | 286 | 271 | 252 | 260 | 273 | 261 | 249 | 234 | 241 | 248 |
| 28308 | 210 | 164 | 193 | 164 | 164 | 164 | 177 | 164 | 173 | 197 |

Primary contrasts vs A (d31, Holm over 5; exact root sign-flip; 95% root bootstrap):

| Contrast | Mean (pp) | +/− roots | p | Holm p | Bootstrap 95% (pp) |
|---|---:|---:|---:|---:|---|
| D − A | −3.64 | 1/7 | 0.031 | 0.156 | [−5.83, −1.34] |
| E − A | −3.61 | 1/7 | 0.031 | 0.156 | [−5.83, −1.34] |
| F − A | −3.30 | 1/6 | 0.047 | 0.156 | [−5.74, −0.90] |
| G − A | −3.96 | 1/7 | 0.047 | 0.156 | [−7.03, −1.29] |
| H − A | −4.71 | 1/7 | 0.039 | 0.156 | [−8.08, −1.61] |

Secondary contrasts (unadjusted):

* D − B: −0.78 pp. Analytical screening ran 2.2× more windows than B and did no better.
* C − D: +2.88 pp [0.22, 5.57], 5/3 roots. The ordinary objective commits more often (22.7% vs 17.2% acceptance).
* E − D: +0.02. F − E: +0.32. G − E: −0.34. H − F: −1.42.
* D − I: +0.29, so ranking added nothing over shuffled scores at this budget.
* J − A: −3.74. Instrumentation alone costs about 90% of the update budget.

Selective transport: G −6.51 pp vs A (p 0.039, unadjusted). No arm reaches the +5 pp floor. EvalMath preflight on G − A with I − A as sham returns `REDESIGN_OR_PILOT_ONLY`: pilot effect −3.96 pp, worst sham +11.1 pp, about 10 roots for 80% power at a 10.4 pp resolvable effect.

CPU accounting (mean CPU-s per arm; full ledgers in each `result.json`):

* A: acceptance step 54.9, scheduler 17.1, materialization 9.8, proposal 7.5.
* B: transactional probes 88.0.
* D: candidate gradients 62.6, scoring 10.3, feedback re-evaluation 7.4, feedback gradient 5.0.
* G: candidate gradients 53.9, joint search 16.3.
* Shadow audits (inclusive): E 1.9, F 6.1, G 3.5.
* Budget overshoot: at most 1.81 s (B). Peak process-tree RSS: at most 400 MiB. 6 GiB job cap enforced.
* Host was saturated by other jobs, so process CPU time per unit of work carries contention noise. All arms within a root ran sequentially in one process.

Warmup SHA-256s and per-root `result.json` hashes are in `summary.json`. Source SHA-256s of the 14 run files are in every `result.json` and match commit b30de73 (LF bytes).

## Phase 6: exploratory gym (`results/curate03b/gym/gym_summary.json`)

* Competence after 450 uniform updates, TRAIN/EVAL on 29100 and 29101:
  * E31: 0.59/0.60 and 0.68/0.72.
  * E27: 0.55/0.52 and 0.60/0.52.
  * E28: 0.51/0.38 and 0.50/0.36.
  * E29: 1.00/0.34 and 0.92/0.48. E29 fails headroom on 29100 (EVAL below 0.35), so no hill climbing ran on it.
* Matched 30 CPU-s screen minus uniform, EVAL pp:
  * E31: −0.39 and −5.86.
  * E27: +1.37 and +31.45. On 29101 uniform reinvestment collapsed EVAL from 0.52 to 0.28, an extrapolation-delay failure, not a curation win.
  * E28: 0.00 and −12.70.
* Mechanistic signals, mean within-checkpoint Spearman with realized one-step gain:
  * predicted gain: E28 0.92, E31 0.90, E27 0.57, E29 0.37;
  * gradient cosine: 0.91 / 0.88 / 0.61 / 0.23;
  * signed observer Jacobian: 0.76 / 0.79 / 0.05 / 0.60;
  * target transport: 0.39 / 0.80 / 0.41 / 0.60;
  * protected interference: 0.41 / 0.20 / 0.63 / 0.44.
* Mean finite-dose error: 0.006–0.063. Prediction quality drops on the extrapolation task (E29).

## Phase 7: DataGoblin V2 export (`results/curate03b/datagoblin_export_v2/`)

* TRAIN-only, 26,328 windows, 183,520 candidate rows, 44,189 stage decisions.
* 9,136 measured individual labels (B probes only). The rest are explicit `missing_unaudited` or `not_acquired`.
* 697 audited and 2,238 unaudited abstentions. Horizon outcomes are in a separate file.
* Actions: ACQUIRE_4/8/12 11,470 each, ACQUIRE_16 9,779, ABSTAIN 2,935, SELECT_BATCH 8,535, SWAP_EDGE 712, VERIFY_BATCH 7,236, REINVEST_CPU 14,858.
* Controller results and Inky formats: DataGoblin PR #100, `training/rlt_gym/hypergame_taste/bpd_trm_curate03b/` (ca0a8f3).

## Phase 8: MMS (`results/curate03b/mms/mms_summary.json`)

* Root 28000, 40 fixed windows, D and G: final checkpoints and selections identical at 0/1/4/16/64/256 MiB.
* Hits: 0 at 0 MiB, 65–67 at 1 MiB, 83–85 from 4 MiB up. Peak occupancy 3.6 MB at every tier from 4 MiB up. CPU 42–44 s uncached vs 31–37 s cached.
* Utility saturates at 4 MiB, so 16 MiB is already more than needed.
* Raw per-edge gradient caching at an unchanged parent would serve 2.2–10.2% of pilot candidates, because proposals, nuisance variant and delay change every window. It was not implemented.

## Failed gates and limitations

* The scientific floor failed for every primary contrast, all in the wrong direction. Holm-adjusted p is not significant (0.156).
* E29 failed the gym headroom rule on root 29100.
* The learned controller missed the per-root 98% bar on held-out root 28301 (97.43%).
* Engineering on root 28000: the early-12/8/4 replay missed 1 accept.
* The CPU measurements ran on a contended host.
* The controller evaluation replays recorded parents and cannot see changed trajectories.
* Gym hill climbing uses 2 roots per task and is exploratory.

## Replay

```
python -m unittest discover -s mechinterp/bpd/experiments/trm_curation -t . -p 'test_*.py'
python -m mechinterp.bpd.experiments.trm_curation.engineering_03b --out <dir>/run_28000 --roots 28000
python -m mechinterp.bpd.experiments.trm_curation.pilot --roots 28301 --out <dir> --joint-rounds 1 --budget-cpu 100
python -m mechinterp.bpd.experiments.trm_curation.summarize_03b pilot --dev results/curate03b/development --roots 28301 ... 28308 --out summary.json
python -m mechinterp.bpd.experiments.trm_curation.export_datagoblin_v2 --dev results/curate03b/development --roots 28301 ... 28308 --out <export>
python -m mechinterp.bpd.experiments.trm_curation.gym_tasks --out <dir>
python -m mechinterp.bpd.experiments.trm_curation.mms_03b --out <dir> --root 28000
```

Large JSONL receipts are stored as deterministic `gzip -n`. Readers fall back to `.gz`, and `summary.json` reproduces byte-identically from the compressed files.
