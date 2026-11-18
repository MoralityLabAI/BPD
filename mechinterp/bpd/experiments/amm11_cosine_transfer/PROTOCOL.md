# BPD-AMM-11: prospective learned causal routing versus cosine alignment

**Prospectively registered on 2026-11-18, before any AMM11 source model or training.**

## Motivation and exposure

The source-frozen synthetic AMM09R1 development study established a +13.623pp registered gain of root-transfer memory over the strongest globally fixed admissible mask across 64 independently trained six-dimensional recurrent models. The subsequent [AMM10 exposed-data audit](../amm10_exposed_dissection/RESULTS.md) selected *cosine alignment with the donor displacement* as a particularly strong simple analytical competitor. Within eight-fold heldout evaluation of **already-exposed AMM09R1 TRAINING roots**, full ridge scored 40.14% and pure cosine 38.51% (+1.62pp). Removing alignment dropped ridge to 28.02%. Those observations are selection exposure. This prospective experiment uses **new SHA256 AMM11 phase namespaces** and treats cosine as the primary adversarial baseline. The selection-exposed AMM09R1 development and all AMM03–09R1 sealed confirmations remain entirely untouched.

The **primary question** is whether fitted strategy-memory contains a behaviorally meaningful advantage *beyond an explicit geometric alignment rule* once the latter receives the exact same physical candidate states and no restricted task targets.

## New synthetic learned recurrent source, physical endpoints

Independently train a 6D rank-three physical weight-tied three-pass recurrent transition \`F(h)=(h@I)@O\`, I[6,3],O[3,6], over source \`(a,b,c)~Uniform[-.8,.8]^3\`, h0=(0,0,0,a,b,c), q(h)=h[0], with family wren targets (a,b,c) and basalt targets (c,a,b), family by independent root index parity. Adam 400 steps at LR .035, batch 96, sum of three passwise target MSE and .0002 physical weight penalty. Freeze physical teacher, train a separate random-initialized BPD V[6,3],U[3,6] for 240 Adam steps LR .035, same batch and three teacher-state trajectory losses plus .5 mean squared weight reconstruction penalty. Canonicalize reconstructed VU via float64 rank-three SVD. Require actual nonlinear optimizer update of both stages and archive source-heldout three-pass teacher/factor task MSE, relative weight residual, singular gaps and hashes; **keep every poor-fit root in primary denominator**.

For each donor pair recipient source is Uniform[-.3,.3]^3; independently edit the nuisance first-pass target coordinate by signed Uniform[.15,.70] and the two delayed-target coordinates by signed Uniform[.25,.45]. Define donor target pair from **external source inputs before teacher/BPD fitting**. Six physical permitted masks: \`(0),(1),(2),(0,1),(0,2),(1,2)\`. For mask S, inject \`((h_don-h_rec)@V[:,S])@U[S,:]\` at recipient BPD first pass, zero current readout coordinate, and normalize the patch to the same physical full-donor first-pass L2 dose \`D=||F_BPD(hdon)-F_BPD(hrec)||\`. A degenerate projection uses e1*D; zero-dose and degenerate cases remain ITT. Continue two learned tied passes. Success iff *both* \`|q(h2patched)-y2donor|<=.18\` and \`|q(h3patched)-y3donor|<=.18\`. Reproduce native BPDModel/TiedComponentLinear three-pass execution parity, GL3 shear/recanonicalization, and equal physical dose and current-dark displacement on EXCLUDED engineering roots before training.

Every decision agent is given **exactly** the 18 AMM09R1 source-safe candidate causal-response features and three unpatched/patched two-horizon probes, totaling 14 unlabeled scalar candidate readouts per donor case (2 unpatched plus 12 patched). The hidden donor source-defined target, full donor future continuation and candidate/test true success labels are audit-only. The proxy cosine (feature column 14) is the normalized dot of the physically equal-dose current-dark candidate write with *the full donor first-pass state displacement*. That displacement is an allowed source-safe observation in this synthetic instrument, with explicit dependence on the donor first-pass state. No agent receives a direct donor second/third target.

## Register model populations, training, comparators and local budgets

Namespace \`AMM11/<phase>/<index>\` with **8 EXCLUDED engineering** roots for source/code tests; **64 TRAIN** roots, each 128 independent donor pairs and six candidate success labels; **64 independent DEVELOPMENT** model roots, each 16 disjoint local calibration donor cases, **512 primary test cases** and **512 independent repeatability cases**; **128 SEALED confirmation** model roots. Independent unit is one newly trained physical model and separate factorization. Nested donor cases, six masks and multiple loop passes are repeated observations, never additional independent roots.

Freeze training after fitting **two global memories to the same training corpus**, each action-labelled:
- \`full18\`: 18 features, ridge lambda 10 on slopes, intercept unpenalized, quantize 18 coefficients to 4 decimal places.
- \`compact10\`: feature indices \`[0,1,5,6,7,8,9,10,11,14]\`, identical ridge lambda/quantization rules.
- \`pooled\`: training-best fixed admissible mask using all six candidate labels from the same 64 roots. Freeze its chosen index and complete training case-bit digest.

**Primary competing policies**, each selecting one of the six masks per case by lowest-index tie:
1. \`full_memory\`: fit full18 from 64 train roots, clipped predicted success [0,1].
2. \`cosine\`: frozen, zero-training free geometric rule \`argmax_S x[S,14]\`, with exact source-feature identity tie handling.
3. \`compact_memory\`: the preregistered ten-column ridge score, clipped to [0,1].
4. \`pooled\`: training-selected strongest globally fixed allowed mask, independent of case.
5. \`local_ridge\`: single-model local 18-feature ridge lambda 2, free intercept, from the same **16 local calibration pairs x six candidates = 96 noisy binary reports** as every adaptive policy, noise flip p=.20, corrected response \`(reported-.20)/.60\`.
6. \`memory_plus_local\`: 18-dimensional ridge residual lambda 20 penalizing all coefficients, same 96 local reports, corrected targets minus frozen clipped full-memory prior. Choose from clipped updated scores.

Every arm has access to the **same** source-safe 18-feature candidate geometry, same donor first-pass state, and same 14 unlabeled delayed readout measurements per test case; it may ignore some signals. Cosine costs zero global label-training and zero local labels; global memory pays extra amortized 64-model supervision; local adaptation and local ridge share exactly the same 96 noisy local labels. Additional descriptive controls: \`max_cardinality\` (analytic, choose largest |S|), \`random_mask\` (per-case SHA256), equal-dose \`random_dark\` and \`random_dark_b\`, unpatched \`sham\`, and full donor state, an **equal-L2 but immediately visible** audit ceiling. No full triple mask is permitted in active action or per-case oracle.

## Primary effect and Holm family

Precommit **five** one-sided signed model-root contrasts, Holm familywise alpha=.05, paired 2,000-root bootstrap 95% percentile intervals and **all** ties:
1. **PRIMARY: full_memory minus cosine**.
2. compact_memory minus cosine.
3. full_memory minus compact_memory.
4. full_memory minus pooled.
5. memory_plus_local minus full_memory (local feedback value).

The meaningful mechanistic floor is **+0.08 absolute success probability**, target **+0.12** for power planning, and measurement resolution **1/512**. In the exposed AMM10 eightfold study the primary pilot was only +.0162; this argues that AMM11 may be a **negative meaningful-mechanism study**, and any raw significance below the registered floor cannot justify confirmation. A negative result is publishable without relabelling post hoc. A positive confirmation requires Holm-adjusted primary root-sign p<.05 and paired-root bootstrap lower bound above max(.08, registered matched-sham tolerance, 1/512). No alternative primary may be chosen after development.

To calibrate the physical sham variation, evaluate the **same deterministic cosine policy**, with frozen observational inputs but no local feedback, on each root's 512 primary donor cases and **512 independent repeatability** cases, retaining per-root paired absolute success-rate differences. Across 64 independent model roots the sample maximum is an upper tolerance bound for the population 95th percentile at >=95% confidence (95/95). Use corrected EvalMath mechanistic-resolution preflight pinned to \`MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a\`, with 64 independent root effects, all sham deviations, five registered comparisons, 128 planned confirmation roots, alpha .05 and target .12. \`REDESIGN_OR_PILOT_ONLY\` **blocks confirmation**.

Further fixed veto-only gates: at least **56/64** independent development models pass teacher and BPD source validation with three-pass task MSE <= 1e-4 and .02, relative weight error <=.10 and both singular gaps >.01; audit-only casewise **best among six masks** exceeds cosine by >=.08; all active equal-dose errors <=1e-5 and immediate readout shifts exactly 0; native BPD and gauge check in engineering. Report any failures, retaining their outcomes. No censoring or re-thresholding of roots is allowed.

The independent scientific confirmation would additionally require a **separately committed human-approved source/policy/EvalMath-bound release manifest**, and an additional no-reuse confirmation mechanism. **This project contains no confirmation runner** and rejects the \`confirmation\` namespace at source. All AMM03–09R1 confirmation and ARG03-N protected populations remain strictly sealed.

## Determinism and source provenance

Match the audited AMM09R1 CPU engineering recipe, pinned Python 3.12.12, PyTorch 2.10.0+cpu and NumPy 2.2.6, \`ATEN_CPU_CAPABILITY=default\`, \`MKL_CBWR=COMPATIBLE\`, single-thread OMP/MKL, fixed SHA256 per-root RNG, deterministic algorithms, and exact wheel hashes. Record source SHA256, runtime signature, physical model/factor hashes, all six candidate success bits, all 96 noisy calibration bits, all selected actions, rootwise shams, and entire original training/development raw JSON. Use a staged hosted workflow: engineering-only tests -> training policy -> **commit training lock before development** -> one source-locked development scoring -> corrected EvalMath -> lock development result. Full raw receipts are immutable and can be independently rescored from original action/bit traces. A source or numerical replay mismatch blocks scoring new roots.

## Evidence class

A prospective positive result on this task would show learned strategy-memory exceeding a **strong source-available analytical geometric rule** on *fresh synthetic 6D rank-three linear tied recurrent models*. It would not by itself establish nonlinear TRM/LLM mechinterp or agent memory transfer. AMM11 is explicitly a scientific discrimination experiment between learned routing and the strong hand-designed geometry proxy, with selection exposure disclosed.
