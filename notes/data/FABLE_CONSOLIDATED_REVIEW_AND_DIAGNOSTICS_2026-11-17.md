# Consolidated Fable Review, Mutual Corrections, and B4 v2 Diagnostic Record (2026-11-17)

This file consolidates the complete Round 1 -> Round 2 (`v2`) Fable review and diagnostic exchange, every empirical receipt generated in `v2` (tied to pre-registration commit `180f1bcbbc633a16bfd887a9593252cefd7a6a65`), the post-hoc zero-fit / zero-CI diagnostics on the extracted `C = 512` `(V, U)` factors, and the distilled `v3` toy-first design requirements.

---

## 1. Executive Synthesis: What `v2` and the Zero-Fit Audits Establish

1. **Rotation-null geometry fails on sparse natural-scale features (`B2 v2` FAIL, `B3 v2` FAIL):**
   - At `K = 199` Haar read rotations (`u @ R`), restoring `core.{0..3}.{o_proj, down_proj}` (`8` factors) passes the `B2 v2` negative arm (`min p_rank = 0.030 > 0.00625`), and nested `state_only -> state_plus_clock` warm-starts eliminate all negative clock inversions in `B3 v2`.
   - However, on both `SettlingCoreLooper` (`B2 v2`) and `LoopedChase` (`B3 v2`), sweeping a sparse planted mechanism across `0.5x, 1.0x, 2.0x, 4.0x` of the largest natural raw KL fails `p_rank <= 0.01` (`LoopedChase` at `1x`: true plant `14.07` nats vs rotated read `38.76` nats, `p_rank = 0.99`). Once per-token softmax KL saturates on the active `1/16`–`1/32` support, dense RMS-matched rotations produce higher mean KL than the sparse true vector, and `argmax z_rot` actively anti-selects the SVD component aligned with the plant (`|cos| = 0.0002–0.02`).
2. **Full-rank (`C = 512` / `C = 256`) unconstrained `L_p` gates are unidentifiable (`~60–73%` active):**
   - Low reconstruction KL with `1,216–1,490 / 2,048` components active per pass proves basis span, not mechanistic decomposition (`B3 v2` `s1, s2 vs s3` gap collapses because `visible_only` CI reconstructs every checkpoint to `< 0.003` nats).
3. **Under-weighted faithfulness (`1e4` vs Goodfire `1e7`, `lr = 1e-3` vs `5e-5`, `15` vs `400` warmup steps) produced a `23%–34%` remainder `\Delta W` that the `40%` gated-off components actively cancel:**
   - Exact `||\Delta W||_F / ||W_target||_F` is `0.231–0.426` (mean `0.335`) on `gaoxin492` and `0.174–0.255` (mean `0.228`) on `yagizdevre`.
   - Turning **all `512` components ON while holding `\Delta W` at its pass-mean** (`comp_mask = 1, delta_mask = 0`) jumps Pass-20 KL from `0.006 -> 3.94` nats (`gaoxin492`) and `0.005 -> 12.18` nats (`yagizdevre`), whereas turning **off** 40% of components under the CI mask gave `0.079` / `0.059` at Pass 20 (`0.010` at Pass 62). Thus `gate = 0` meant *"off together with `\Delta W` to cancel `-(x - E_\tau[x])\Delta W^\top`"*, **not** *"ablatable"*. This directly explains the `36.7` / `102.1`-nat PGD blow-up when intermediate masks break the cancellation.
4. **Late-pass (`pass 62`) KL on `WB-Stall` sits below numerical reproducibility (`0.26–0.82` nats mean), and `BPDTrainer.evaluate()` steered chaotic boards using oracle clean-run states:**
   - Comparing exact `float32` vs native `bfloat16` target rollouts across the `16` eval boards yields `median KL = 0.0000` at Pass 62 (`11–13` settled boards stay at exact `0.0`), while `1–3` chaotic stall boards flip branches and jump to `4.0–11.1` nats (`mean = 0.26–0.72` nats, exceeding the registered `final_kl <= 0.15` gate).
   - `BPDTrainer.evaluate()` (`trainer.py` lines `571–576`) achieved `0.010` at Pass 62 only because it computed `ci_out = self.ci_net.forward_trajectory(clean_traj)` from the **oracle clean `bfloat16` trajectory at all 63 passes**, steering bifurcating boards back onto the clean branch.
   - At **Pass 20** (where all controls close at `0.003–0.008` nats), **untrained 60% SVD truncation** (`0.020` on `gaoxin492`, `0.006` on `yagizdevre`) beats trained BPD (`0.079` and `0.059`) by **`4x–10x`**.
5. **Both `pass_mean` ablation and `sphere_rescale` independently degrade clean SVD bases by `14x–68x` at Pass 20 (`31x–71x` combined):**
   - Plain zeroing (`mode="zero", sphere_rescale=False`) at 60% dynamic SVD density gives Pass-20 KL `0.0201` (`gaoxin492`) and `0.0062` (`yagizdevre`).
   - `sphere_rescale` alone worsens Pass-20 KL to `0.2748` (`13.7x`) and `0.4187` (`67.5x`); `pass_mean` alone worsens it to `0.4436` (`22.1x`) and `0.1623` (`26.2x`); both together reach `0.6216` (`31.0x`) and `0.4372` (`70.5x`).

---

## 2. Chronological Verbatim Fable Reviews & Diagnostic Exchange

### Pass A — Fable Verdict on Round 1 (`v1` Receipts at `MoralityLabAI/VPD` `e25b640`)

> **Verdict:** the receipts match every figure in the report, and both FAILs are real. The two PASSes do not show what their gates are named for, and the B4 write-up buries its own result: the random-component control beat the targeted edit.
>
> #### B4
> - **Every norm-matched perturbation raised final solve.** Of 183 boards at `s = 64`: clean `108`, targeted edit `121`, random learned components `132`, rotation control `114`. Selection by function is not shown to matter.
> - **None of the edited arms are resolved against each other.** Each control is a single draw, and the gaps are 0.8 to 1.25 unpaired standard errors. This needs per-board discordance counts and a distribution over at least 20 control draws.
> - **This restates v014.** `yagizdevre`'s deficit is stabilisation, and conditional refill already took its final solve to `0.754`.
> - **Criterion 2 could not discriminate.** Period-2 oscillation occurs on 1 board clean and 2 edited.
> - **The framing reads past the failure.** "3/5 passed" with a bolded `+7.10` pp leaves out that the failed criterion is the one testing the hypothesis.
> - **One of the four edited components is not sustaining in the target** (net `−0.12`); it was ranked by the reference's score.
> - **Functional similarity of `0.989–0.999` is uninformative** when 74 to 83 components are active on every pass. The similarity among unmatched pairs is the missing reference.
>
> #### Decomposition quality
> - **The decomposition is not adversarially faithful.** Shared PGD takes reconstruction loss from `0.23` to `23.3` nats; per-pass reaches `67.5` and is still rising at 160 steps. "Saturated" describes the curve, not the level.
> - **Target reconstruction is poor:** final KL `0.59` on `yagizdevre`, trained on 2 outer steps and 32 boards, then edited at 64 outer steps.
> - **The switch sweep sets the noise.** Loss runs `0.79`, `0.92`, `0.53` while the switch rate stays at `0.042–0.047`, so single runs differ by about ±0.2.
> - **The interventional path demo returned exactly `0.0`** for both direct and mediated effects, against a first-order prediction of `0.0206`. The report omits it, and exact zeros suggest the intervention is not wired.
>
> #### B3
> - **The CI-head refits are noisier than the margins.** State-plus-clock is worse than state-only in three of four checkpoints (`−0.18`, `−1.34`, `−2.03`), which cannot happen at an optimum. The two visibility passes rest on margins of `0.92` and `0.56`.
> - **The planted control used `sigma = 600` and scored z = `2790`.**
>
> #### B2
> - **The positive arm is 24 times the largest natural factor** (`1.415` against `0.060`). Neither positive control tests sensitivity near natural effect sizes; a sigma sweep down to that scale would.
> - **The negative arm's scope was narrowed.** The earlier registration covered `o_proj` and `down_proj`; `49c7207` covers `down_proj` only. The history file notes the rescope, but the registration carries no dated amendment and the summary says "untouched".
> - **Still 16 rotations and z-scores,** where I had asked for at least 100 and rank-based p-values.
>
> #### B1
> - **It passes on the floor alone.** Tied loss `8.98e-05` is 1,292 times the unrolled `6.95e-08`; the 1.5× clause would need `1.04e-07`.
> - **The budget-matched baseline cannot represent the model:** one component per pass in a looper built with four.
> - **The comparison is not like for like.** The tied loss sits at the hinge tolerance (zero hinged loss on every pass at the last step), while the unrolled baseline is unhinged.

---

### Pass B — Fable Review of Completed `B1–B3 v2` Receipts (`6949279`) & Mid-Run `B4 v2`

> **Short answer:** neither option as offered. Suspend the process instead of killing it, and Carbon is right that a third round of patches on this formulation would be wasted, though not for the reason it gives.
>
> #### The process
> Windows Resource Monitor can suspend `PID 275016` (CPU tab, right-click, Suspend Process). A suspended process stops touching its pages, so the compression thrash ends, and you can resume it when you are away from the machine. A CUDA context usually survives suspension, but not always, so that is a small risk against nine hours of fits.
>
> If you would rather decide between the two offered, the selector's verdict in B4 v2 is predictable, because it ranks components with the null that B2 and B3 just showed has no power. What is still worth 2.5 hours is the 39 random-draw distributions. They measure how much any norm-matched edit stabilises `yagizdevre`, which the dark-fibers paper can use whatever happens to BPD.
>
> #### Carbon's two diagnoses check out
> I read the v2 receipts on the branch.
> - **The rotation null is the wrong geometry.** On LoopedChase the exact planted factor costs `14.1` nats at natural scale and the rotated read costs `38.8`. Its rank p is `0.99`, so the true mechanism sits in the bottom tail. More rotations cannot fix that.
> - **The gates are not identified at full rank.** Between 1,300 and 1,490 of 2,048 components are active per pass and reconstruction loss is about zero for every CI head, so the gap statistics have nothing to measure.
>
> #### Where its account is off
> - **"What works" is overstated.** Low loss at full rank with two thirds of the dictionary switched on is reconstruction, not decomposition.
> - **The blind-recovery arm is unexplained.** The selected component's cosine with the planted read is `0.0002` to `0.02`, below chance. That is what you would see if the plant sits in one SVD-initialised component and the broken selector picked another. Reporting the best alignment over all components would settle it.
> - **The loop is what found the bottleneck.** The sigma sweep, `K = 199` rank p-values, restored `o_proj` and nested clock check were the last review's requests. The rotation null itself came from my tied-VPD design, so that part of the box is mine.
> - **The hours came from run order.** The full-rank TRM fits were launched after B2 v2 and B3 v2 had failed, against a registration that says to stop after each gate.
>
> #### What I would put in the next spec
> - **A null that permutes the component's own activations across tokens.** It preserves the activation distribution exactly, so saturation cannot favour either side. A Fisher-matched null would tie by construction in the linear regime; Fisher norm is better used for matching edit sizes.
> - **A gate-recovery test** on the routed toy from B1 v2: precision and recall of the active set against the planted routing.
> - **Both tests passing on toys, which takes minutes, before anything runs on TRM.**

---

### Pass C — Fable Audit of Debugger-Extracted `B4 v2` Stage `1a–4` Receipt (`6501d8d`)

> Pulling the extracted receipt at `6501d8d` to check the headline numbers against the file.
> The headline numbers match the file. Extracting through the debugger got more than resuming would have: the script never saves factors, so this is the only reason `(V, U)` exist on disk. The cost is that Stages `5a–8c` never ran and `B4 v2` has no verdict, which is the cheaper loss given the problems with those stages.
>
> #### What the receipt establishes
> - **Fidelity gate cleared, narrowly.** Final KL is `0.010` on both checkpoints (`v1` `yagizdevre` was `0.593`). That is under the one deterministic mask, on 16 boards, with 60% of components on per pass and 79–86% recruited at some pass. About 2% of live subcomponents are active per position in Goodfire's decomposition.
> - **The gates certify one mask, not a family.** Shared-source KL goes from `0.035` to `0.36` at the two steps used in training, `4.45` at five and `17.3` at twenty. Goodfire's language-model decomposition sits at `0.83` after 20 adversarial steps and `25.3` after 160, and they call that robust to roughly 20 steps and not to 160. This one holds to about two (different model and step size, so the comparison is rough). Per-pass sources are 2–7× stronger at every budget and plateau near 100 nats. No component yet has a supported "can be ablated" label.
> - **The model itself amplifies perturbations.** The hinge calibration perturbs one component's read by 10% per module. That alone gives KL `0.14, 0.49, 1.24` at the three readouts on `gaoxin492` and `0.12, 0.57, 1.59` on `yagizdevre`, a factor of 2.5–4.7 per outer step. This probably explains `v1`'s random edit moving the 64-step solve rate by 13 points. It argues for scoring edits on KL at passes 20, 41 and 62, not on solve rate at 64 steps.
> - **Gates do move across passes.** About 4.5% of the dictionary changes state per core call. But TRM's call schedule has period 7 (six `z_L` updates, then one `z_H` update), and the inner-matched losses show it: every seventh pass sits at `0.04–0.06` against about `0.07` for its neighbours. Call type has to be regressed out before any of this is read as state-dependent role switching.
>
> #### Two results that don't support what they appear to
> - **Depth extrapolation.** The "27× lower" figure compares two means over 8 boards. `gaoxin492` outer-only goes from `0.010` to `0.596` at 126 passes, and inner-matched stays at `0.022`. But `yagizdevre` outer-only also stays flat (`0.010` to `0.013`) without inner matching, so inner matching is not shown to be the cause. Inner matching also costs in-distribution: final KL `0.032` against `0.010`, and first-readout KL `0.31` against `0.08`. Per-board values would settle whether one or two boards carry the `0.596`.
> - **Switch sweep.** Switch rate moves from `0.0451` to `0.0431` across a 100× range of the coefficient, inside seed noise. At the default coefficient the term is about `9e-6` against a reconstruction loss near `0.02`, with factors frozen and 40 steps. It cannot detect over-merging; it does show the term is not distorting anything.

---

### Pass D — Fable Checks 1 & 3 on Extracted `(V, U)` Weights (`ede6949`, SHA-256 `65ecb2ab…`) + Mutual Code Corrections

#### Fable Checks 1 & 3
> **Check 1: the dictionary is not near its SVD start.** An SVD start has mutually orthogonal reads, mutually orthogonal writes, and zero residual. The trained factors have none of those:
>
> | Metric | `gaoxin492` | `yagizdevre` |
> | :--- | :--- | :--- |
> | Residual norm ÷ reconstructed norm | `0.33` | `0.23` |
> | Read-vector norm held by one shared direction | `6–15%` | `9–26%` |
> | Effective distinct read directions (of 512) | `40–144` | `13–63` |
> | Effective distinct write directions (of 512) | `68–257` | `12–72` |
> | `o_proj` singular values below `1e-3` of largest | `10–12` | `50–59` |
>
> - **The components don't sum to the weights.** Converted to a norm, `\Delta W` is about a third the size of the reconstruction on `gaoxin492`, up from zero at the start.
> - **Much of the movement is common-mode.** In `yagizdevre` `layers.1.o_proj`, 43% of reads and 50% of writes sit within cosine `0.5` of a single direction.
> - **Likely cause:** Adam at `1e-3` on entries of typical size `0.014–0.024` moves an entry by ~5–7% per step.
>
> **Check 3: the `bfloat16` concern mostly doesn't hold.**
> - `+0.15` on the four largest components: applied vs intended differs by `2–3%`; full ablation `0.2–0.4%`; norm matching ratio `1.000`. Single components outside top 64 by norm can differ by `10–60%`.
> - **Cross-checkpoint coordinates:** Reconstructed matrices have cosine `0.13–0.16` in three modules and `0.46` in `layers.1.o_proj` (vs `~0.002` random), concentrated in top 16 singular directions.

#### Mutual Code Corrections (Carbon <-> Fable)
1. **`evaluate()` uses `delta_value = 0.0` (Carbon corrected Fable; Fable confirmed):**
   - `BPDTrainer.evaluate()` (`trainer.py` line 566) holds `\Delta W` at its pass-mean (`delta_value = 0.0`), so `final_kl = 0.010` was measured with `\Delta W`'s input-dependent part **off**, and PGD's route is turning `s_delta` **on** (`0 -> 1`).
2. **Faithfulness normalization vs Goodfire hyperparameters (Fable corrected Carbon):**
   - Goodfire's reference code divides by element count just like `losses.py`; what caused the `0.33` residual was `coeff_faith = 1e4` (vs Goodfire `1e7`), `main_lr = 1e-3` (vs `5e-5`), and `15` warmup steps (vs `400`).
3. **Mask invariance along the shared direction (Fable corrected Carbon):**
   - Pointing at $\mathbb{E}[x]$ does not guarantee small activation variance; empirically, input variance along $\hat{d}_{\text{shared}}$ is `5.6x–27.6x` *above* typical isotropic variance, while `pass_mean` absorbs `76%–95%` of the second moment (`pass_cond_var / m2 = 0.05–0.24` inside the shared cluster vs `0.16–0.53` outside), making ablation `2x–5x` cheaper inside the cluster ("cheap, not free").

---

### Pass E — Fable Audit of Zero-Fit Separation Diagnostics (`eab20f9`) & Final Control Closure (`0b7e210`)

#### Fable Audit of `gate_b4_v2_nofit_diagnostics.json` (`eab20f9`)
> **The cancellation result holds, but only at the first readout.** The summary left out that the control arm fails after it, so every late-pass number in this file is uninterpretable for now.
>
> #### The control does not close
> All components on with the remainder on is the target model by construction, so its KL should be near zero at every readout. In the JSON it is:
> - `gaoxin492`: pass 20 `0.006`, pass 41 `0.39`, pass 62 `0.37`
> - `yagizdevre`: pass 20 `0.005`, pass 41 `0.04`, pass 62 `1.53`
>
> Two controls settle why: (1) all components on from the SVD start, and (2) target against target under a numerics change (`float32` vs `bfloat16`), with per-board KL. One candidate for BPD's `0.010` is that the gates are computed from the clean run and steer the masked rollout back onto it.
>
> #### What pass 20 supports
>
> | Arm, KL at pass 20 | `gaoxin492` | `yagizdevre` |
> | :--- | ---: | ---: |
> | All components + remainder (control) | `0.006` | `0.005` |
> | All components, remainder at its mean | `3.94` | `12.18` |
> | BPD 60% gate mask, remainder at its mean (receipt) | `0.079` | `0.059` |
> | SVD start, static rank 307, no training | `0.033` | `0.006` |
> | SVD start, top 60% by activation, no training | `0.020` | `0.006` |
> | Same, under pass-mean + sphere rescale | `0.62` | `0.44` |
>
> - **Cancellation is confirmed.** A gate of zero means "off together with the remainder", not "ablatable".
> - **Untrained truncation beats the trained decomposition at equal component count** (`2–4×` on `gaoxin492`, `~10×` on `yagizdevre`).
> - **The ablation mode does damage by itself** (`30–70×` worse than plain zeroing).

#### Resolution from `gate_b4_v2_controls_and_ablation_split.json` (`0b7e210`)
1. **Per-board numerical reproducibility floor (`float32` vs `bfloat16` target):**
   - At Pass 62, **`11–13` of `16` boards have `KL = 0.000000` (`median = 0.0` in every control arm)**, while **1 to 3 unstable stall boards** (`b11`, `b7` on `gaoxin492`; `b7`, `b10`, `b14`, `b13` on `yagizdevre`) flip search branches under `float32` vs `bfloat16` rounding and jump to `4.0–11.1` nats (`mean = 0.309` / `0.316` on `gaoxin492`; `0.264` / `0.718` on `yagizdevre`; SVD-start all-512 `mean = 0.054` / `0.819`).
2. **Oracle clean-run gate steering confirmed in `trainer.py` (`lines 571–576`):**
   - `BPDTrainer.evaluate()` feeds `clean_traj` (`bfloat16` target states at all `63` passes) into `self.ci_net.forward_trajectory(clean_traj)`, steering chaotic boards back onto the clean branch (`0.010` nats).
3. **`2×2` Factorial Split (`zero` vs `pass_mean`) × (`sphere_rescale` `False` vs `True`) on 60% Dynamic SVD at Pass 20:**
   - `zero`, `sphere_rescale = False`: **`0.0201`** (`gaoxin492`), **`0.0062`** (`yagizdevre`)
   - `zero`, `sphere_rescale = True`: **`0.2748`** (`13.7x` worse), **`0.4187`** (`67.5x` worse)
   - `pass_mean`, `sphere_rescale = False`: **`0.4436`** (`22.1x` worse), **`0.1623`** (`26.2x` worse)
   - `pass_mean`, `sphere_rescale = True`: **`0.6216`** (`31.0x` worse), **`0.4372`** (`70.5x` worse)
   - *(Code bug also documented: `components.py` line 369 gated `sphere_rescale` on `mode == "pass_mean"`, silently turning `zero + sphere_rescale` into a no-op unless `mean_comp_acts` is explicitly zeroed.)*

---

## 3. Checklist Locked for Next-Gen (`v3`) Toy-First Specification

1. **Toy-first hard stop:** Zero `TRM` compute until all toy gates pass in minutes:
   - **Activation-permutation null** (permuting component activations $a_c(b, p)$ across `(batch, token)` to preserve exact empirical marginal sparsity & saturation; Fisher metric reserved for edit norm-matching) across a `0.5x–4x` natural-scale `sigma` sweep.
   - **Gate-recovery precision/recall** on `StationaryRoutedLooper` (`B1`) + **planted constant-activation bias component** comparing `zero` vs `pass_mean` directly.
2. **Multiplicative `zero` masking (`sphere_rescale = False`) as default** (removing both `pass_mean` and `sphere_rescale` distortions).
3. **Goodfire-scale faithfulness regime:** `coeff_faith >= 1e7`, `main_lr ~ 5e-5` (scaled to entry RMS), `>= 400` faithfulness warmup steps, and hard relative-norm reporting `||\Delta W_m||_F / ||W_m||_F` alongside effective distinct read/write rank during training.
4. **Closed-loop CI evaluation & Pass-20 / per-board scoring:** Never feed oracle `clean_traj` states into `ci_net` at evaluation time; report per-board quantiles alongside means, and score `TRM` perturbations at Pass 20 (single outer step, where precision controls close at `< 0.008` nats) and regress out call type (`pass mod 7`) before claiming state-dependent switching.
