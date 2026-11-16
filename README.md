# `MoralityLabAI/BPD` — Behavioral Parameter Decomposition & Causal Dark Fibers for Looped Transformers

Parameter-space (**Behavioral Parameter Decomposition, BPD**) and state-space (**Causal Observable Quotients, `LOOPSCOPE-01..04` & `ARG-03-N`**) mechanistic interpretability for weight-tied, recurrent-depth transformers, companion to [`MoralityLabAI/VPD`](https://github.com/MoralityLabAI/VPD).

> **Held-Out Test Seal (`ARG-03-N`):** The 168-case `ARG-03-N` held-out split (`test.jsonl`, SHA-256 `ddd580212da5a5ea4f5fea740b515dd191b39e7a501d3e31b45944a666e7c8b5`) remains **strictly sealed (`0` model runs, excluded from this repository)** pending prospective completion of the fresh `final-R` development calibration gates in [`ARG03_FABLE_PREUNSEAL_AMENDMENT.md`](papers/shared_weights_dark_fibers/ARG03_FABLE_PREUNSEAL_AMENDMENT.md) and explicit human authorization.

---

## Repository Structure

### 1. Behavioral Parameter Decomposition ([`mechinterp/bpd/`](mechinterp/bpd/))
Extends Goodfire's single-pass [adVersarial Parameter Decomposition (VPD)](https://www.goodfire.ai/research/interpreting-lm-parameters) (`T = 1` four-term PyTorch parity verified) to shared recurrent matrices executed $T > 1$ times:
- **Core engine & Fable pre-run amendments (`fb7b1e9`):** One shared factor pair $(V_m, U_m)$ per physical matrix with pass-indexed causal importance $g_{b,p,\tau,c}$ (`state_only`, `state_plus_clock` Clock Gap, and `visible_only` Visible Gap), `max`-pooled temporal minimality, matched-pass fragility calibration, pass/batch-shared persistent PGD adversary, exact Monte Carlo rank-Holm rotation diagnostics (`K = 255`), item-contrast same-pass donor swaps (`contrast.py`), and Frobenius/rank-matched B4 checkpoint-edit sham controls.
- **Documentation & audit trail:**
  - Package guide: [`mechinterp/bpd/README.md`](mechinterp/bpd/README.md)
  - Formal equations & positioning: [`mechinterp/bpd/docs/design_and_positioning.md`](mechinterp/bpd/docs/design_and_positioning.md)
  - Initial pre-registration (`49c7207`): [`mechinterp/bpd/docs/registration.md`](mechinterp/bpd/docs/registration.md)
  - Fable pre-run amendments (`2026-11-15`): [`mechinterp/bpd/docs/amendments_2026-11-15_fable.md`](mechinterp/bpd/docs/amendments_2026-11-15_fable.md)
  - Positive-control run history & archived pre-amendment receipts: [`mechinterp/bpd/docs/positive_control_history.md`](mechinterp/bpd/docs/positive_control_history.md), [`mechinterp/bpd/results/historical_pre_amendment_49c7207/`](mechinterp/bpd/results/historical_pre_amendment_49c7207/)

### 2. Kernel-Checked Lean 4.19 Proof Curricula ([`experiments/bpd_lean_curriculum_v2/`](experiments/bpd_lean_curriculum_v2/))
- **`BPD-LG-02` (`experiments/bpd_lean_curriculum_v2/`):** Independently Lean-4.19-kernel-verified structural proof curriculum (`192` positive + `192` constructive axiom-free negative certificates and countermodels) for clean tactic-family mechanism separation.
- **`BPD-LG-01` (`experiments/bpd_lean_curriculum/`):** Preserved historical curriculum explicitly marked as a leaked/failed mechanism-separation benchmark for audit transparency.

### 3. Causal Dark Fibers Campaign: `LOOPSCOPE-04` & `ARG-03-N` ([`mechinterp/dark_fiber_gyms/`](mechinterp/dark_fiber_gyms/) & [`papers/shared_weights_dark_fibers/`](papers/shared_weights_dark_fibers/))
Tests whether recurrent transformers carry causally necessary reasoning state invisible to current readouts ($q \circ F_x = T_x \circ q$), how much is recoverable from visible traces ($C^{\text{trace}}$ vs. $C^{\text{now}}$), and where local delayed observability Gramians $G_{t,k}$ predict finite-angle sensitivity:
- **Unified `v014` Manuscript (unchanged):** [`papers/shared_weights_dark_fibers/PAPER.md`](papers/shared_weights_dark_fibers/PAPER.md) ([`main.pdf`](papers/shared_weights_dark_fibers/main.pdf))
- **Observable Quotient Theory Note:** [`papers/shared_weights_dark_fibers/CAUSAL_QUOTIENT_ARG03.md`](papers/shared_weights_dark_fibers/CAUSAL_QUOTIENT_ARG03.md)
- **Empirical Campaign Addendum (`LOOPSCOPE-04` 3A–3C & `ARG-03-N` `dev_full96`, Figures 6 & 7):** [`papers/shared_weights_dark_fibers/DARK_FIBERS_CAMPAIGN_ADDENDUM.md`](papers/shared_weights_dark_fibers/DARK_FIBERS_CAMPAIGN_ADDENDUM.md)
- **Frozen `ARG-03-N` Test Protocol & Fable Pre-Unseal Development Amendment:**
  - [`papers/shared_weights_dark_fibers/ARG03_FROZEN_TEST_PROTOCOL.md`](papers/shared_weights_dark_fibers/ARG03_FROZEN_TEST_PROTOCOL.md)
  - [`papers/shared_weights_dark_fibers/ARG03_FABLE_PREUNSEAL_AMENDMENT.md`](papers/shared_weights_dark_fibers/ARG03_FABLE_PREUNSEAL_AMENDMENT.md)
  - Split-safe CPU audit & fresh `48 + 48` `final-R` development generator: [`mechinterp/dark_fiber_gyms/arg03_preunseal/`](mechinterp/dark_fiber_gyms/arg03_preunseal/)
- **Provenance-Locked Receipts (For Independent Recomputation):**
  - Unified summary & dev calibration: [`papers/shared_weights_dark_fibers/campaign_20261115/`](papers/shared_weights_dark_fibers/campaign_20261115/)
  - 24 per-board `LOOPSCOPE-04` receipts + `rep_eval_strata.json` & 96-case `ARG-03-N` `per_case_diagnostics_dev96.json`: [`supplements/loopscope04_arg03_receipts/`](supplements/loopscope04_arg03_receipts/)
- **Foundational `LOOPSCOPE-01..03` Suites:** [`mechinterp/loopscope01/`](mechinterp/loopscope01/), [`mechinterp/loopscope02/`](mechinterp/loopscope02/), [`mechinterp/loopscope03/`](mechinterp/loopscope03/)

---

## Running the CPU Verification Suites

```powershell
$P = "$HOME/.venvs/bluebeam/Scripts/python.exe"

# Run all 4 deterministic CPU test suites (BPD core & Fable preflight, Dark Fiber gyms, ARG-03 pre-unseal audit, and BPD-LG-02 Lean generator)
& $P -m pytest -v

# Syntax & bytecode verification
& $P -m compileall -q mechinterp experiments
```

Provenance hashes linking back to `MoralityLabAI/LoopedTransformers` and `MoralityLabAI/VPD` are recorded in [`PROVENANCE.json`](PROVENANCE.json).
