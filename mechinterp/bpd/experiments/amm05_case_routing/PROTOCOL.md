# AMM-05: case-conditioned BPD routing under mixed donor edits

**Preregistered 2026-11-18, protocol first, before AMM-05 training/development scoring.** AMM-04 closed at development: memory+noisy-local improved over local by 3.61 pp with only 3/64 positive independent models, below its fixed 8 pp mechanistic floor. The AMM-03 and AMM-04 protected confirmation cohorts remain sealed. AMM-05 tests *within-model case-dependent* intervention choice rather than one component per model.

## Model and independent targets

Each independent root fits its own 4D rank-two **weight-tied linear physical teacher** and separately trained two-component BPD V,U factorization, with AMM-04's fixed optimization (200 teacher Adam steps LR=.035 batch 96; 140 factor steps LR=.035, target trajectory loss plus .5 weight MSE). Input (a,b) uniform [-.8,.8]; family wren has readouts q(h1)=a,q(h2)=b and basalt reverses roles. The donor's target y2 is determined from its input coordinate and the family *before model or BPD fitting* and is withheld from all decision policies. Model-specific V,U are full-product SVD-canonicalized in float64. Include all failed root fits.

## New source-disjoint mixed donor challenge

The recipient second-pass *target source* coordinate v ~ Uniform[-.2,.2] changes by an independent signed displacement in [.30,.45]. A distinct **nuisance coordinate** starts uniform [-.5,.5] and receives an independently signed displacement in [.15,.80]. This deliberate mixed edit adds a second physical direction so the preferred BPD component can change between cases within one root. External target y2 is the *donor target coordinate alone*. It is not a model output and cannot be computed by a decision policy from explicit target-source labels.

For every pair, evaluate the two **SVD-canonical component patch candidates**. First-pass full donor write D_h = F_BPD(h_d)-F_BPD(h_r), norm D=||D_h||_2. Candidate c is z_c=(((h_d-h_r)V)_c)U_c, projected into the current readout nullspace by setting output coordinate 0 to zero and normalized to ||z_c||_2=D. Zero-dose and vanishing-dark cases remain in the denominator and use a deterministic e1*D fallback. The policy picks exactly one patch after pass 1; continue one physically learned shared BPD pass. Independent task success is |q(h2_patched)-donor y2|<=0.18. A physically matched isotropic random-dark patch, zero sham and larger full-donor-state replacement are retained as controls. All donor cases, failures and negative effects are counted.

## Allowed fixed candidate causal-response features

An intervention policy may inspect the recipient and donor **first-pass** states, SVD-canonical factor geometry, the physical norm D and one candidate-patch continuation readout per candidate. It may NOT inspect the donor's second-pass continuation, the externally defined target y2, downstream success bits, or another candidate after making its one chosen patch.

The fixed eleven-feature *candidate* vector for both policies is:
1. constant 1;
2. current readout q(h1_r);
3. first-pass readout contrast q(h1_d)-q(h1_r);
4. dose D;
5. signed local delayed response [q(F(h1_r+patch_c))-q(F(h1_r))]/(D+1e-8);
6. its absolute value;
7. unnormalized signed local response;
8. first-pass alignment <patch_c, D_h>/(D²+1e-8);
9. weight-only signed delayed gain of component c under the target-coordinate row, with the source role known only for computing canonical geometry, not the label itself;
10. SVD singular-value gap;
11. basalt family indicator.

The same two candidate-patch probes used to compute features are charged to *every policy* as an identical **two unlabeled local forward evaluations per donor case**. Neither policy may execute F(h1_d) or use any labeled held-out outcomes. A separately labeled **audit-only donor continuation proxy** will score the potential shortcut that is prohibited by this budget; it never influences the trained policy.

## Memory, equally budgeted local policies and controls

Train global **ridge regression λ=10**, intercept unpenalized, on binary donor-success labels from both candidates in **64 independent AMM05 training roots × 128 donor pairs**. Root weights are equal; each training root provides two candidate records per pair, and no training root is reused in development. Freeze eleven ridge coefficients and globally best component before development. Clip predicted candidate probabilities to [0,1].

Each new root gets exactly **four independent calibration donor pairs**, with both component outcomes reported through independent p=.35 Bernoulli label flips. Both arms receive the same eight noisy labeled feedback bits and the same two causal-response probe features per pair. For a noisy bit y, the fixed unbiased response target is (y-.35)/.30; no clipping before the local fit.

- Primary **memory_calibrated**: fit an 11-feature root-specific ridge residual to (noise-corrected local outcomes minus frozen memory predictions), using λ=20 with the intercept penalized; predict frozen memory probability plus residual and choose higher score for each new case.
- Primary comparator **local_ridge**: fit the same 11-feature local ridge on the *same eight noisy labels* from scratch, λ=2 with unpenalized intercept; choose higher predicted per-case score. No training-world labels are available.
- Additional control **local_count**: choose the fixed component with more noisy calibration hits, ties favor larger signed weight-only delayed gain then component 0.
- **memory_only**: frozen cross-root ridge without calibration.
- **geometry_rule**: per case choose the candidate with larger *absolute* signed local delayed causal response.
- **pooled**: frozen globally training-best component.
- **random_component**: independent case-keyed hashed bit.
- **random_dark**, **random_dark_b**, **sham**, **full_donor**: physically matched-dose controls except labeled full-donor larger-dose ceiling.

All root-level decisions, case-level binary outcomes, and per-case actual intervention choice are retained. Local baseline and memory receive the same 8 feedback labels and the same 2 unlabeled intervention probes/test case. The **64 training-root corpus** is an additional disclosed amortized cost of memory.

## Phase separation, registration and uncertainty

SHA256 root namespace "AMM05/<phase>/<index>", independent of AMM01–04:
- 8 **excluded engineering** model roots, solely for source smoke, parity and feature tests.
- 64 **training** roots, each 128 independent donor pairs; both candidate success outcomes accessible only while learning memory.
- 64 **development** roots, each 4 calibration donor pairs with 8 noisy labels, followed by 128 heldout primary donor pairs AND 128 independent repeatability donor pairs.
- 128 **sealed confirmation** roots. Tests, experiments and invalid/incomplete preflight calls are forbidden from fitting or even generating a confirmation root, including indirectly through training. No intervention-policy revisions are permitted after development scoring.

**Primary effect:** average over independent development/confirmation model roots of memory_calibrated − local_ridge heldout success rate. Two secondary registered positive comparisons are memory_calibrated − local_count and memory_calibrated − geometry_rule. All three one-sided root-level exact sign tests are Holm-adjusted for familywise .05; report 2000 seeded paired model-root bootstrap 95% percentile CIs and signed root counts, plus per-case audit-only oracle ceilings. Outcomes sharing a model root are repeated observations, never independent worlds.

Effect threshold: **+0.08** absolute success probability, target +0.20, measurement resolution 1/128. Matched-sham variation: per independent development root evaluate the frozen local_ridge strategy on a *separate* 128-case repeatability donor set and use the absolute success-rate difference against its original 128-case set. Across 64 independent model roots, obtain the nonparametric sample-maximum upper 95th population quantile with >=95% confidence. Use the **corrected** EvalMath mechanistic-resolution preflight code pinned to merged commit 6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a (includes the zero-variance/zero-effect and discrete exact-sign floors), with independent root-mean effects and three comparisons. A REDESIGN_OR_PILOT_ONLY decision stops the study at development, regardless of attractive repeated-case totals.

Even with preflight READY, require the *development* rootwise test-label-only per-case oracle gap over local_ridge to reach at least 8pp to justify confirmation spending, and confirm that teacher/factor fit and dose parity gates hold. Do not use that oracle to tune the memory model or alter the registered protocol. Confirmation requires a **separately frozen manifest and exact source hash** and an explicit release decision; its own root population must not appear in any test or development receipt. Final claim requires Holm p<.05 and the primary 95% root bootstrap lower bound above max(8pp, sham floor, resolution), with no failed dose/parity gate.

## Engineering, provenance and evidence

Root independence includes separate physical and decomposition initialization and training streams; time/pass repeats are nested. Exact per-case success bits, chosen components, noisy local calibration bits, dose mismatch, current-readout drift, donor-proxy oracle label (audit only), teacher behavioral MSE, factor behavioral MSE and canonical singular gap are retained. On excluded roots, verify that both physical factors and both BPD factors were trained, SVD canonization is GL2 shear invariant for separated singular values, and native BPDModel/TiedComponentLinear PassMaskSchedule execution agrees with the independent implementation to <=1e-5.

Source and protocol commits precede model training; training memory is source-hash locked before development. The protected ARG-03-N split and AMM-03/04 confirmation populations remain strictly unaccessed. **Evidence class: synthetic trained linear recurrent BPD causal-policy evaluation.** It does not establish a nonlinear/TRM/LLM mechanism, real agent cognition, or Buzz relay memory transfer.
