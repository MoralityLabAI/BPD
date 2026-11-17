# AMM-03: gauge-stable model-conditioned strategy memory

**Prospective registration: 2026-11-18. Protocol-first; before source or new results.**
AMM-02's family-index policy chose component zero for both families and equaled pooled strategy on all development roots. This new test asks whether cross-root memory of **trained causal geometry** adds benefit over cost-matched per-model calibration.

## Independent trained systems, labels, interventions

Each independent root trains a rank-2 4D weight-tied physical recurrence and a separately initialized rank-2 V/U BPD factorization, using AMM-02's physical task and optimization: teacher 200 Adam steps at LR .035 with batch 96 and L2 .0002, factorization 140 Adam steps at LR .035 with weight MSE coefficient .5. New SHA256 streams are keyed `AMM03/{phase}/{index}`, disjoint from AMM-02. External behavioral labels, defined before model fitting from source (a,b), are wren y1=a,y2=b and basalt y1=b,y2=a. The independent donor target is the donor's second-pass source coordinate.

Each pair changes only the second-pass target input by signed 0.30..0.45. The chosen action is one of two SVD-canonical BPD components, with raw write `((h_d-h_r)V_c)U_c` projected into current readout's nullspace and normalized to `D=||F_BPD(h_d)-F_BPD(h_r)||`. The active candidate and isotropic random-dark patches share exactly this one per-pair physical L2 dose (tolerance 1e-5) and displace the current readout by zero. Run a subsequent trained tied pass from patched recipient h1 and score `|q(h2)-donor external y2| <= .18` on **all generated pairs**. No-patch sham and full donor intermediate replacement (larger dose) are descriptive controls. Retain all failures and zero/degenerate doses.

## Predeclared geometry features

Canonicalize full VU by float64 SVD before computing features or component patches, then record singular gaps and GL(2)-shear/recanonicalization parity. For the changed input index j (wren 3, basalt 2), define dark_c by zeroing the first coordinate of U_c; `direction_c=sign(V[j,c])*dark_c/||dark_c||` (zero for degenerate vector). The delayed causal response is `r_c=q(F_BPD(direction_c))`. Fix the seven-dimensional feature vector of each action as:
`[1, r_c, |r_c|, r_c^2, f_c, (s0-s1)/(s0+s1+1e-8), 1_{basalt}]`, where `f_c=|V[j,c]|*||dark_c||/(||V[j,:]U||+1e-8)`.

Train a ridge regression (penalty .01 on slopes, unpenalized intercept) to predict candidate hit rate on **96 donor pairs per training root**. Training comprises 64 independent roots, balanced by family; fit both action rates per root, and freeze coefficients. The memory prior clips predicted candidate rates into [0,1]. No test donor labels enter features or training.

## Equal local calibration budget and frozen arms

Each new development/confirmation root receives **8 source-disjoint calibration donor pairs**, with both component actions measured, costing **16 labeled intervention executions**. The proposed `memory_calibrated` policy selects the component maximizing `(8*p_memory_c + calibration_hits_c)/16`; the `local_only` comparator selects the higher calibration hit count (tie -> component 0), using the **same eight paired local examples** and identical 16 labeled queries. Choices are frozen before 128 distinct test donor pairs. Both policies additionally have the same weight/geometry access. Memory incurs the disclosed *amortized* 64 training-world corpus cost.

Other prespecified arms: `memory_only` (ridge prior, zero local labels), `pooled` (single training-best action, zero local labels), `geometry_rule` (larger signed r_c, zero labels), `random_component` (single root-hashed component, zero labels), two independent equal-dose random-dark vectors, sham and full donor-state replacement. Every test arm is evaluated on the same 128 recipient/donor pairs; independent random-dark probes calibrate repeat variation.

## Independent worlds and preregistered decisions

- Engineering: 8 excluded models for source and native parity only.
- Training: 64 fresh independently trained models, 96 donor pairs each.
- Development: 64 independent models, each 8 calibration pairs + 128 test pairs.
- Confirmation: **128 sealed independent models** with the same per-root calibration/test counts. Test code and any invalid/absent preflight path are forbidden from creating or querying these models.

Primary independent-root effect: `memory_calibrated - local_only` accuracy. Secondary registered effects: `memory_calibrated - geometry_rule` and `memory_calibrated - pooled`. Three one-sided exact root-sign comparisons, Holm adjusted at familywise alpha .05; report 2000 root-cluster paired-bootstrap replicates and all root sign/tie counts. Meaningful science minimum +0.10 absolute accuracy, target +0.20, measurement resolution 1/128. Independent development matched-sham floor: sample maximum of the absolute per-root success-rate difference between the two independent random-dark arms over 64 worlds (95th population quantile at >=95% confidence). Preflight: pinned EvalMath `skills/evalmath-mechinterp-power/scripts/preflight.py --strict`, 64 development root-level signed primary differences and random-dark A/B sham contrasts, 3 comparisons, planned 128 confirmation roots. Variation floor = max(.10, 1/128, 95/95 sham tolerance). If strict preflight returns `REDESIGN_OR_PILOT_ONLY`, report development-only evidence and leave confirmation untouched. Passing preflight only authorizes a once-scored confirmation. Confirmatory success additionally needs a Holm-adjusted p<.05, primary bootstrap lower confidence limit exceeding the variation floor, and audited dose/source parity.

## Audit and evidence boundary

Require independent teacher/factor learning updates, teacher task MSE, factor reconstruction MSE, singular-gap/zero-dose diagnostics, full per-case binary traces, all policy choices and exact source/memory/dataset SHA256 receipts. Native BPDModel/TiedComponentLinear + PassMaskSchedule parity must reproduce excluded-root masked forward within 1e-5; GL2 full product invariant and recanonicalized component patches must close when gap >.01. Evidence class is **trained synthetic linear recurrent causal intervention**, never an assertion about nonlinear language models, deployed Goose/Codex agents, Buzz/Nostr relay memory, or unique mechanisms. Preserve AMM-01/02 and protected ARG-03-N splits unmodified.
