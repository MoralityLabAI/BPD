# BPD-CQ-07: reachability no-go and equal-clean-control result

**Research phase:** excluded-root engineering feasibility; **negative** causal discrimination result. The CQ-07 issue's goal was to repair CQ-06's off-range interventions with reachable recurrent states. This code and protocol were developed with these eight previously exposed model seeds. There is no CQ-07 prospective confirmation and **no** CQ-07 59-root sham-calibration result. The exact CQ-06 confirmation receipts and reported results remain unchanged.

## Structural rank certificate

For a legal first cue `(x,y,0,-1) Q`, the first hidden preactivation lies in an affine two-plane `p0+(x,y)A` in R^12. The two largest singular BPD output-write directions span `rowspan(B)` with B=U[0:2,:]. On all eight independently re-trained development seeds, the stacked 4x12 matrix `[A;B]` has rank **4**. Therefore its row subspaces intersect only at zero. Because the real tanh function is injective, **no nonzero rank-two preactivation patch restricted to this BPD block is exactly reachable from a legal first cue** in any of those learned model worlds. This obstruction is *conditional on each trained weight matrix's full-rank witness*, and the eight singular values provide numeric evidence for the rank premise.

| Seed | Minimum singular value of `[A;B]` | Median constrained patch preactivation residual | Patch passes head+dose constraints /128 | Clean-source match available /128 | Jointly feasible /128 | BPD donor-b hits | Matched clean-source hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| 997 | .01540 | 4.274 | 35 | 75 | 32 | 32 | 32 |
| 1009 | .05206 | 3.539 | 30 | 59 | 30 | 30 | 30 |
| 1013 | .02814 | 4.690 | 6 | 62 | 6 | 6 | 6 |
| 1021 | .02313 | .798 | 86 | 51 | 51 | 51 | 51 |
| 1033 | .00556 | 4.152 | 26 | 12 | 7 | 7 | 7 |
| 1039 | .03638 | 1.511 | 106 | 103 | 103 | 103 | 103 |
| 1049 | .01828 | 3.404 | 57 | 57 | 57 | 57 | 57 |
| 1051 | .00406 | 1.173 | 49 | 66 | 46 | 46 | 46 |
| **Total** | all rank 4 | | **395/1024** | **485/1024** | **332/1024** | **332/1024** | **332/1024** |

**State-box status:** zero violations for both constrained tanh BPD states and truly clean reservoir states. **Exact legal reachability of nonzero BPD states:** 0/1,024, both numerically (preactivation residual >1e-4) and by the full-row-rank conditional proposition. The in-range BPD states are mathematically different from naturally source-reachable states.

## Conditional matching collapses the apparent causal difference

The constrained-tanh BPD patch retains the first readout within ±.30 logit units and achieves an actual state dose within ±.15 of the full clean-donor displacement on **395/1,024** generated pairs. Independent reservoirs of 4,096 clean source inputs per world supply a first-parity/head/dose-matched alternative on **485/1,024** pairs. Their **intersection** contains 332 pairs, with all other generated pairs assigned failure in *both* intervention arms for the intention-to-treat comparison.

| Donor-target intervention (all 1,024 generated pairs) | Correct second-label output |
|---|---:|
| Constrained-tanh selected BPD patch, common-feasibility ITT | **332 (32.42%)** |
| Genuinely reachable clean reservoir state, common-feasibility ITT | **332 (32.42%)** |
| Second independently matched clean reservoir, common-both-feasible ITT | 331 (32.32%) |
| BPD restricted to same common-both-feasible subset | 331 (32.32%) |
| Constrained BPD scored when its own solver passes (descriptive) | 395 (38.57%) |
| Raw two-component tanh patch, no head/dose constraints | 933 (91.11%) |
| Full clean donor intermediate state | 1,024 (100%) |
| Sham | 0 |

Matched-source controls were selected without looking at donor delayed labels, donor x signs or second-visit cues. **Post-match**, all 332 jointly feasible first-reservoir controls carried the same x sign as the donor. Matching on the large physical donor displacement selected the sign-flipped branch of the same-parity fibre, which itself solves the delayed-label task. This explains why the clean-source control transfers as well as the BPD patch. It also shows why a matched-dose source control can become a strong rival rather than a mechanistically inert sham in this disconnected synthetic parity geometry.

The registered coarse physical dose bin is ±0.15, not exact per-pair equality. Reservoir absence means failure to find a match in 4,096 samples; it does not establish nonexistence of a match anywhere in the continuous source manifold. All 1,024 generated pairs and every unmatched/solver-failed case remain in the archived receipt. The raw two-component tanh arm's 933/1,024 score demonstrates why **unrestricted intervention success is not evidence for naturally reachable causal transfer** when the target effect is fully replicated by a true clean-source control under matched geometry.

## EvalMath stop decision

Official EvalMath preflight on the eight development root effects (all **0/128**) and two independent clean-reservoir sham effects (all **0/128**) returns **REDESIGN_OR_PILOT_ONLY**, strict exit code 2. Reasons: confirmation cohort unsealed, no nonparametric 95th/95%-confidence sham-tolerance bound (59 *fresh* calibration roots needed), observed target effect below the 15-point physical minimum, and impossible estimated power for that target. The 26-root estimate from CQ-06 is inapplicable once the stronger causal comparator yields zero development effect. With 32.4% common feasibility, an unconditional 15-point advantage is also severely constrained by the admissible population.

Consequently no new CQ-07 sham-calibration or confirmation roots were trained. No positive causal quotient or reachable rank-two patch claim is made. The conditional rank-and-tanh proposition remains a useful negative exact-geometry theorem; the numerical witnesses are eight development instances, not a statistical generalization to all possible models.

## Reproducibility

The frozen engineering source is [geometry.py](geometry.py) and the immutable per-pair full receipt is archived as gzip+base64 at [results/cq07_development_independent.json.gz.b64](results/cq07_development_independent.json.gz.b64). [results/LOCKED_SUMMARY.json](results/LOCKED_SUMMARY.json) records Git blob hashes for CQ-07's source/protocol/verifier and CQ-06's unchanged dependency, plus SHA-256 of the original uncompressed JSON. The [reference verifier](reference.py) replays every development root, checks all integer outcome counts and binary pair masks and validates native `BPDModel/TiedComponentLinear` continuation parity. The control-source geometry, dose bands, optimization costs and selection exposure are specified in [PROTOCOL.md](PROTOCOL.md).

The next appropriate experiment should use naturally reachable clean-state donor swaps as the intervention and measure **component-specific mediation at the next recurrence**, while providing task-irrelevant nuisance motions large enough to make clean control doses comparable without automatically crossing the donor-target sign branch. This requires a new prospective protocol and source family.
