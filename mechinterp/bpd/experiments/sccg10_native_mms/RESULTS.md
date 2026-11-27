# SCCG-10 / MMS-TRM: results

2026-11-27. Branch `research/sccg10-mms-trm-20261127`. It is built from
CURATE-02 at af43e48 (BPD #170), with main ab93e4d merged in
(SCCG-01..09, #161/#168/#171/#173) and the SCCG-10 design from #175 (41a06e8).
Evidence class: engineering receipts plus DEVELOPMENT_ONLY paired roots. No
confirmatory claim is made. ARG-03-N and reserved cohorts were not read.
Historical CURATE-02 roots were used for replay and smoke tests only.

## 0. Verdict

**Negative.** On 8 fresh development roots, the adaptive MMS-atlas arm (D)
learned *worse* than the strongest ordinary arm at equal CPU. On the primary
endpoint (ordinary steering-off final-development delay-31 accuracy, fixed
100 CPU-s per arm), D − A = −4.93 pp. Every one of the 8 roots was negative:
exact two-sided sign-flip p = 0.0078, Holm p = 0.031, root bootstrap 95% CI
[−6.89, −3.15] pp. The effect is the opposite of the hypothesis.

The required secondary contrasts were not significant:
* D − B: −1.71 pp (Holm 1.0);
* D − C: −1.44 pp (Holm 1.0);
* D − E: −0.81 pp (Holm 1.0).

The atlas does what it was built to do. It reuses 76% of mechanism directions
after native verification and gives D 3.1× the windows of fresh mechanism (C).
But two limits dominate:
1. The k8 support search still takes 81% of D's CPU.
2. The mechanistic calibration objective is misaligned with ordinary
   validation loss (ρ ≈ 0.15 against ρ ≈ 0.85 for ordinary scores).

The strongest ordinary baseline, uniform proposals with transactional
feedback-CE rollback, wins. That result is preserved. EvalMath preflight:
REDESIGN_OR_PILOT_ONLY.

## 1. Source custody and native parity (Phase 0)

* torch 2.15.0+cpu (CURATE-02's version), Python 3.12.10, 2 threads, CPU only.
* CURATE-02 focused tests: 23/23. BPD/RP/SCCG model tests: 67/67 plus 41
  subtests. SCCG-10 admission reference: 8/8. New native tests: 5/5. New
  MMS tests: 12/12.
* Historical bit-exact replay: re-running CURATE-02 root 28101 arms A and F
  from the committed `shared_warmup.pt` reproduced both final-state SHA-256
  values and the d31 correct counts (233/512).
* New exact instruments, parity-tested against CURATE-02 references:
  * the fast state Jacobian (4 backward passes per delay instead of 12)
    equals `observe_edge` to rtol 1e-5;
  * the forward-mode JVP equals J·u;
  * the analytic AdamW preview equals `probe_update`'s predicted gain, and
    the probe restores model state bit-exactly;
  * the write-site Jacobian equals a 4-point finite difference;
  * the replay's manual recurrence equals the native model with atol 0.
* Leakage check: arm F (full mechanism maintenance, uniform decisions)
  ended with a final state bit-identical to arm A at 80 windows on both
  engineering roots and on the CURATE-02 smoke root.
* Memory: every stage ran inside the fail-closed Win32 job object with a
  6 GiB commit limit. Peak job commit was 963 MiB for warmups and
  1.63 GiB for interleaved arms.

## 2. Fresh engineering roots 29200 and 29201

Roots were checked against every BPD root registry: used ranges are
26101–26108, 27101–27108, 28000–28001, 28101–28108 and 29000–29001. New draws
use the `SCCG10-MMS-TRM` seed domain over CURATE-02's TRAIN edge partition
(624 fit / 144 feedback / 144 tune directed edges).

| root | warmup CPU-s | tune d2 | tune d7 | tune d31 |
|---|---:|---:|---:|---:|
| 29200 | 362 | 99.8% | 97.7% | 62.4% |
| 29201 | 394 | 99.1% | 93.8% | 51.1% |

### 2.1 Checkpoint drift (Workstream B; `results/engineering/root_*/drift`)

The panel is 24 TRAIN-fit edges (6 per slot, distinct transitions), 4 queries
each, and delays 2/7/31. The update stream is uniform AdamW at lr 1e-3. Root
29201 means are shown; root 29200 has the same pattern
(`drift_summary.json`).

| updates | ‖Δθ‖ | a rel. drift | C rel. drift | row-space max sin | k8 cached/live gain | k8 cached protected | write-k8 cached/live |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.073 | 0.49 | 0.45 | 0.68 | 1.10 | 0.33 | 0.98 |
| 2 | 0.110 | 0.67 | 0.78 | 0.80 | 1.01 | 0.46 | 0.76 |
| 4 | 0.173 | 1.11 | 0.73 | 0.87 | 0.74 | 0.45 | 0.44 |
| 8 | 0.241 | 2.48 | 0.86 | 0.93 | 0.60 | 0.52 | 0.44 |
| 16 | 0.318 | 1.35 | 0.95 | 0.91 | 0.81 | 0.56 | 0.45 |
| 32 | 0.392 | 1.96 | 1.36 | 0.95 | 0.84 | 0.84 | 0.47 |
| 64 | 0.514 | 1.27 | 0.97 | 0.95 | 0.80 | 0.54 | 0.55 |

* **Derivatives drift fast and in few directions.** After one committed
  update the full-suffix target and protected maps change by about 45–50%,
  and by 16 updates they have changed completely. 95% of the drift energy
  lies in 4–5 of 27 singular directions.
* **Cached directions keep target value but lose protection.** Cached sparse
  directions keep 60–110% of the live target gain. Their protected change
  leaves the 0.25 budget within 2 updates, while live re-solved directions
  stay at 0.25–0.30.
* **Write space is more linear but loses value faster.** Write-space
  directions have finite error below 1e-3 but keep only about 45% of their
  value after 4 or more updates. The physical write metric drifts by 1% after
  one update and 9% after 64 (generalized eigenvalues 0.93–1.09). Factor-frame
  transport and physical reuse give the same gain.
* **Linear null directions are not safe in finite response.** Full-state
  directions violate protection: zero linear collateral, yet finite protected
  change of 0.7–1.1 at dose 0.75.
* **Costs.** Per edge and checkpoint: exact state observation 0.32–0.53 s;
  write observation 0.25–0.40 s; k8 support search about 1.0 s (state) and
  0.35 s (write); native verification of a cached direction 0.05–0.10 s;
  forward-mode JVP 0.35–0.56 s (no cheaper than the full Jacobian here).
* **SCCG-10 admission refused 24/24 edges at every checkpoint.** It was fed
  exact checkpoint-pair operator norms and sampled remainders. The uniform
  protected error γe is 19–43 against η = 0.25. MATH_REVIEW §1 explains why
  the certificate is sound but uninformative here.
* **Chronology** (`root_29201/chronology`). Per-step suffix Jacobian norms
  reach 7.6 (geometric mean 1.8–3.7). The operator-norm transport bound
  after 8 updates is 10^10–10^20; the actual drift is 0.3–11.4.

### 2.2 Behavioral selective-control gate

The gate was frozen in commit 1f090a5 before root 29200's drift data was
opened (`BEHAVIORAL_GATE.json`). Result on root 29200 at the warmup
checkpoint (`BEHAVIORAL_GATE_RESULT.json`): **FAIL**.
* SCCG-01 k8 directions at dose 0.75 flip 1/22 eligible targets (G1
  threshold 25%).
* Corruption is 0/186 (G2 pass). G3 was not evaluated.
* Clean target margins average −5.5 logits, while directions move them by
  +0.27.
* Privileged diagnostics: full hybrid replacement at natural dose (norm about
  4.3) flips 14/22; at matched dose 0.75 it flips 4/22.

CURATE-02's SELECTIVE_CONTROL_ADMISSION_FAILED therefore stays in force.
No causal auxiliary loss was trained in any arm.

### 2.3 Paired arms on engineering roots (TRAIN tune d31, 1152 sequences/root)

Fixed CPU, 100 CPU-s per arm, arms interleaved least-CPU-first in one process:

| arm | 29200 windows | 29200 d31 | 29201 windows | 29201 d31 | mean |
|---|---:|---:|---:|---:|---:|
| warmup | – | 62.4 | – | 51.1 | 56.8 |
| A uniform reinvest | 1540 | 55.5 | 1557 | 49.1 | 52.3 |
| G guarded ordinary (uniform + feedback-CE acceptance) | 1165 | 64.2 | 1153 | 58.9 | **61.6** |
| B ordinary-gradient curation, rate controller | 227 | 64.8 | 226 | 51.7 | 58.3 |
| B, always curated | 183 | 61.0 | 186 | 49.6 | 55.3 |
| C fresh mechanism | 12 | 59.4 | 12 | 49.2 | 54.3 |
| D MMS atlas + ladder | 32 | 63.1 | 28 | 49.2 | 56.2 |
| D, always curated | 14 | 58.8 | 13 | 51.5 | 55.1 |
| D, refresh cap 3 | 41 | 54.6 | 42 | 51.6 | 53.1 |
| E shuffled mechanism | 41 | 54.4 | 29 | 51.5 | 53.0 |
| F observation only | 9 | 57.2 | 17 | 46.0 | 51.6 |

G and the cap-3 variants ran in a separate interleaved process, E4. A repeat
of D (E3, tier 16 MiB) gave 60.1 on root 29200: fixed-CPU policies are
timing-dependent on a shared host.

Fixed update, 80 windows (`e2_fixed_update`):

| arm | 29200 CPU-s | 29200 d31 | 29201 CPU-s | 29201 d31 |
|---|---:|---:|---:|---:|
| A uniform | 5.2 | 60.2 | 4.8 | 48.4 |
| G guarded ordinary | 7.4 | 63.0 | 7.6 | 55.2 |
| B | 28.0 | 61.6 | 28.3 | 51.7 |
| C (safety stop at 61 / 64 windows) | 907.5 | 59.0 | 906.2 | 49.4 |
| D | 227.7 | 54.9 | 184.6 | 50.7 |
| E | 239.1 | 55.0 | 207.7 | 54.1 |
| F (bit-identical to A) | 265.8 | 60.2 | 289.1 | 48.4 |

MMS tiers (root 29200, D, fixed CPU): 0 MiB gave 13 windows, 60 refreshes,
60.5%. 4, 16, 64 and 256 MiB were identical: 41 windows, 56 refreshes,
60.1%. Peak atlas occupancy was 291 KB (12 entries: support, maps, direction,
null basis, source parameter snapshot), so tiers above 1 MiB cannot differ.
A 256 MiB tier has nothing to hold.

### 2.4 Where the mechanistic arms' CPU goes (Workstream D)

The ledger shows the limiting cost. In C/D/E/F, k8 beam support search takes
70–82 of every 100 CPU-s. The atlas works as designed:
* on root 29200, 226/288 D edge-uses reused a cached direction after native
  verification (78%), with 13 L3 support re-solves and 49 refreshes;
* mean entry age at reuse was 4.4 windows (max 19);
* D completed 2.3–2.7× more windows than C on the same budget.

That is not enough. A cold refresh of the 12 feedback edges costs 15–25
CPU-s, about 300 uniform windows. A steady-state curated window costs
0.6–1.0 s against 0.065 s for a uniform window.

The second limit is value, not cost. D's first-order scores predict its own
mechanistic calibration gain well (Spearman ρ = 0.91–0.94 over audited
windows) but predict the drop in *ordinary* feedback CE poorly (ρ = 0.14–0.21).
B's ordinary scores predict ordinary-CE drops at ρ = 0.83–0.87. The
mechanistic calibration objective is misaligned with ordinary validation loss.
This is the empirical failure of MATH_REVIEW Prop. 11's alignment condition.

Uniform training at train delays 1–3 *raises* feedback CE at delay 15. The
controller measured negative ordinary-block rates on every block. This is
why guarded ordinary training, which rejects 89–95% of uniform steps, is the
strongest ordinary competitor.

### 2.5 One-step versus multi-window value

The rate controller (multi-window EMA of realized ordinary-CE drop per CPU-s)
beat always-curated one-step greedy for ordinary curation on both roots
(+3.8 and +2.1 pp). For D the result was mixed (+4.3 and −2.3 pp). One-step
first-order scores are rank-valid for their own objective, but Prop. 10(c)
and the misalignment above show that rank validity of one-step scores is not
long-horizon validity.

## 3. Development roots 29301–29308 (prospectively frozen)

Frozen in `DEV_FREEZE.json`: source lock 0c4426bd over 29 files, verified by
`devrun.py` before every root. The plan is in DEV_PROTOCOL.md, committed in
96906a6 before any development root was trained.

Arm roles: U = uniform reinvestment (reported); A = guarded ordinary (the
primary comparator, selected on engineering TRAIN tune); B = ordinary
curation; C = fresh mechanism; D = MMS atlas; E = shuffled mechanism;
F = observation only. Warmups took 150–155 CPU-s each (6,000 updates; the host
was less loaded than during engineering).

### 3.1 Fixed total CPU, 100 CPU-s per arm (primary)

Final-development d31 correct counts out of 512 per root:

| root | U | A | B | C | D | E | F | D−A pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 29301 | 195 | 193 | 207 | 197 | 185 | 206 | 203 | −1.56 |
| 29302 | 244 | 232 | 239 | 201 | 221 | 227 | 206 | −2.15 |
| 29303 | 265 | 257 | 253 | 265 | 206 | 217 | 277 | −9.96 |
| 29304 | 224 | 173 | 184 | 150 | 162 | 155 | 140 | −2.15 |
| 29305 | 213 | 266 | 227 | 214 | 236 | 217 | 232 | −5.86 |
| 29306 | 211 | 241 | 191 | 239 | 209 | 241 | 222 | −6.25 |
| 29307 | 285 | 298 | 277 | 256 | 264 | 256 | 267 | −6.64 |
| 29308 | 209 | 239 | 189 | 234 | 214 | 211 | 218 | −4.88 |
| **mean d31 %** | 45.07 | **46.36** | 43.14 | 42.87 | **41.43** | 42.24 | 43.09 | **−4.93** |
| mean windows | 1632 | 1172 | 317 | 13.4 | 41.8 | 42.6 | 18.9 | |
| d63 % | 30.76 | 30.52 | 31.54 | 30.71 | 29.17 | 29.98 | 29.86 | |
| d127 % | 28.47 | 26.88 | 25.90 | 26.15 | 26.86 | 26.25 | 25.95 | |
| binding d31 % | 48.54 | 52.29 | 48.68 | 46.83 | 46.63 | 44.63 | 46.92 | |
| CE d31 | 4.625 | 3.888 | 4.540 | 4.495 | 4.448 | 4.367 | 4.450 | |

The Holm family over 8 roots (exact two-sided root sign-flip):

| contrast | mean pp | roots +/− | p | Holm p | bootstrap 95% |
|---|---:|---|---:|---:|---|
| D − A (primary) | −4.93 | 0/8 | 0.0078 | 0.031 | [−6.89, −3.15] |
| D − B | −1.71 | 3/5 | 0.336 | 1.0 | [−4.76, 1.37] |
| D − C | −1.44 | 4/4 | 0.547 | 1.0 | [−5.20, 1.90] |
| D − E | −0.81 | 4/4 | 0.516 | 1.0 | [−2.98, 1.25] |

No difference reaches the +5 pp practical floor in the hypothesized
direction.

Cost shares (mean over roots, fraction of each arm's CPU):
* A: data 0.27, train step 0.24, selection 0.20, acceptance 0.19.
* B: candidate previews 0.65.
* C: support search 0.90.
* D: support search 0.81, previews 0.06, observation 0.05.
* E: support search 0.81.
* F: support search 0.83.

Atlas ladder over D's 1,992 edge-uses: 1,511 verified reuses (75.9%),
43 L3 support re-solves, 434 refreshes and 4 ordinary fallbacks. L2
recorded NOT_APPLICABLE for every reuse check, as designed in exact mode.

Overruns: the last window crossed the cap by 0.1–7.3 CPU-s in 9 arm runs
(listed per arm in `arms_result.json`). C hit the 900 CPU-s safety stop in
the fixed-update block on 5 roots.

### 3.2 Fixed update, 80 windows (secondary)

| root | U | A | B | C | D | E | F | D−A pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 29301 | 223 | 200 | 224 | 191 | 225 | 216 | 223 | +4.88 |
| 29302 | 217 | 226 | 220 | 235 | 240 | 238 | 217 | +2.73 |
| 29303 | 230 | 251 | 249 | 247 | 246 | 284 | 230 | −0.98 |
| 29304 | 186 | 159 | 167 | 170 | 175 | 166 | 186 | +3.12 |
| 29305 | 255 | 272 | 255 | 292 | 231 | 254 | 255 | −8.01 |
| 29306 | 260 | 255 | 221 | 207 | 213 | 228 | 260 | −8.20 |
| 29307 | 270 | 276 | 267 | 267 | 251 | 271 | 270 | −4.88 |
| 29308 | 216 | 209 | 234 | 221 | 200 | 212 | 216 | −1.76 |
| **mean d31 %** | 45.34 | 45.12 | 44.85 | 44.68 | 43.48 | 45.63 | 45.34 | −1.64 |
| mean CPU-s | 5.1 | 7.0 | 22.3 | 787.8 | 186.1 | 166.2 | 283.2 | |

None of D−A, D−B, D−C, D−E is significant (Holm 0.41–0.94). At equal
updates, D spends 27× the CPU of A and 37× that of U and is not better. F's
final state is bit-identical to U's on all 8 roots: the full mechanism
maintenance leaves no trace in training.

### 3.3 Replay, preflight, memory
* Independent replay (`independent_replay.json`): PASS. It covered 16 run
  directories, 2,960 checks and 372,736 replayed cases: manual recurrence
  from raw tensors, regenerated batches, per-case correctness bits and
  binding counts, max CE error 0.0, and every artifact hash.
* EvalMath strict preflight (pinned `preflight.py`), with pilot D−A and sham
  E−A: REDESIGN_OR_PILOT_ONLY. Reasons: heldout_unsealed,
  sham_quantile_uncalibrated, and pilot effect not above the 5 pp variation
  floor (mean −4.93 pp, σ 2.87 pp, worst sham 9.6 pp).
* Peak job commit was 1.69 GiB per process (cap 6 GiB) and 4.3–4.9 k process
  CPU-s per two-root process.
* Raw `*_decisions.jsonl` and `*_mechanism.jsonl` are committed as
  deterministic `gzip -n` files. Their decompressed bytes match
  `artifact_hashes.json` (224/224 verified). Decompress before running
  `replay.py` on a fresh checkout.

### 3.4 Limiting cost and what would have to change

The ordinary arm wins for three measured reasons:
1. Combinatorial support search: 81% of D's CPU, even with 76% verified reuse.
2. Misaligned value: the mechanistic score ranks its own objective
   (ρ 0.91–0.94) but not ordinary validation CE (ρ 0.14–0.21).
3. Guarded rollback is nearly free: it rejects about 90% of uniform steps at
   about 0.085 s per window.

For D to break even, a curated window would have to be worth about 10–15
guarded windows. In engineering it was worth fewer than one.

Two changes would be needed before revisiting:
* a mechanism representation whose maintenance does not require
  combinatorial search;
* a calibration objective with measured alignment to ordinary held-out
  loss (MATH_REVIEW Prop. 11 and unresolved problem 4).

## 4. DataGoblin TRAIN-only export (Workstream F)

`results/engineering/datagoblin_export/` (gzip JSONL) holds:
* 26,960 candidate decisions, each with predicted gain, selection reason,
  window-level verifier outcome, realized calibration gain, cost and
  provenance URL;
* 8,952 mechanism-admission records, each with ladder level, reason, source
  and verified gain, physical dose and scope;
* a per-arm resource ledger.

Scopes are labelled: `NONE_HEURISTIC_PROPOSAL` for first-order curriculum
scores, and `NATIVE_FINITE_VERIFICATION_ON_FIXED_PANEL` for mechanism
directions. The latter is never a uniform certificate. The export reads only
TRAIN fit and feedback views and fails closed on any other split.

A learned scheduler (DataGoblin or a small recurrent model) was **not
trained**. The precondition was that exact and ordinary baselines show
mechanistic headroom, and engineering shows none: guarded ordinary and
ordinary curation beat every mechanistic arm at fixed CPU. A learned
scheduler would also add inference cost to an arm already limited by cost.

## 5. Mathematics (Workstream A)

See MATH_REVIEW.md. The Lean companions are in MathCan e0347e3
(`MMSTRMEconomics.lean`, 8 theorems): `lake build` succeeded, `leanchecker`
exited 0, there is no sorry, and the axioms are propext, Quot.sound and
Classical.choice.

## 6. What was not done or not established

* No confirmatory claim. The development cohort is unsealed and EvalMath
  says REDESIGN_OR_PILOT_ONLY.
* The causal auxiliary cross was not run, because the frozen behavioral gate
  failed.
* The DataGoblin or recurrent scheduler was not trained (no mechanistic
  headroom; §4).
* Native float32 suffix rounding has no rigorous bound (MATH_REVIEW Prop. 8).
  Native verification is exact up to float32 arithmetic, not interval
  certified.
* SCCG-10 conditional certificates never admitted a native reuse. The uniform
  premises are not available, and with exact checkpoint-pair operator norms
  they are 100× too loose. Heuristic mode (E5) with κ calibrated on root
  29201 refused all reuse and degenerated to fresh refresh.
* Fixed-CPU arms depend on host timing: CPU-timed controllers make different
  decisions under different load. Comparisons are valid within a root's
  interleaved process only, and runs are not bit-reproducible.
* The fixed-update engineering runs (E2) finished after the freeze. They
  informed no frozen value.
* CI: the `SCCG10 ... native` job failed once at ecf1af0 and passed at
  9ecfc0c with identical test code. Job logs need authentication, so the
  failing assertion is unknown.
* The write-site action space and k=4 supports were measured in the drift
  instrument only. The training arms use carried-state k=8 supports.
