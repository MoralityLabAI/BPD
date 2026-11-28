# BPD-NC-13: source context and delivered displacement at eight target gradients

**Prospective protocol, 18 November 2026.** Commit this document before NC-13 implementation, model initialization or data generation. This is a fixed sixteen-pair engineering diagnostic following the original NC-12 result. Its source-context intervention and displacement calibration are new choices motivated by exposed NC-12 outcomes.

## Question and registered population

Does restricting source discovery to the shared task context improve transfer, and does that result change when comparison arms match actual closed-loop write displacement? The receiving model's direct q=8 projector is retained under each calibration policy. Changes in its own loss must be shown alongside changes in the transfer gap.

Reserve pair IDs **2601300 through 2601315** inclusive, with A initialization equal to the pair ID and B initialization at pair ID +100000. Reserve excluded smoke **2601299**, with B initialization 2701299. The ordinary 1300 block belongs to the active SUB02 study. The smoke is excluded from every scientific aggregate. All sixteen development pairs are attempted once; failures remain in the archive and cannot be replaced. No confirmation population is registered.

Each model is the existing 20D `DenseLooper`, with four nonlinear recurrent steps sharing one dense physical `mix` matrix. Use independent A/B initializations and 2,048-case Gaussian training streams, split 1 for A and split 11 for B. Train each for 384 Adam updates, learning rate .005, batch 128, with the NC-12 optimizer and sampling recipe. A and B both predict sign(x0) when x4>0; the other context uses sign(x1) for A and sign(x2) for B. Labels enter model training and final evaluation only.

The accepted first-run recipe is CPU, one Torch thread, Torch 2.5.1+cpu, NumPy 2.3.5 and Python 3.12. Record full runtime versions and model-state hashes. Archive the trained weights so future numerical replay can start from the original models. Later executions of these IDs add no independent evidence.

## Data partitions and target access

Use the unchanged `sample_x(root, split, n)` generator, with separate streams:

| Role | Split and size | Rule |
|---|---|---|
| Source discovery pool | split 2, 4096 inputs | Mixed discovery is the first 1024 inputs. Shared discovery is the first 1024 inputs with x4>0, in original order. |
| Paired activation calibration | split 4, 512 inputs | First 128 positive-context inputs, preserving order. |
| Gradient calibration | First 8 of those 128 inputs | Exactly eight original-B gradient inputs for all construction. Each input supplies all four pass gradients. |
| Label-free B dose pool | split 6, 1024 inputs | Both contexts; use the same input pool under every calibration policy. |
| Final evaluation | split 3, 1024 inputs | Generate and score after all source choices, gains and shams are frozen. |

Require at least 1024 positive source candidates and 128 positive activation candidates. Archive selection indices, input hashes, overlap and unique source-gradient counts. A receives 1024 discovery gradient rows per source condition with overlap reused, plus its eight paired calibration gradients. Both source conditions use **raw** final-score gradients and an uncentered second moment. Changing source context leaves that weighting rule fixed.

The direct B projector uses the top rank-four eigenspace of the second moment of the **unit-normalized** eight B gradients. The 128 paired activation cases are available to every estimator, with direct B subspace estimation consuming the B gradients alone. There is no full-data B-gradient reference. A zero-head diagnostic uses up to eight inputs on a modified B model and is accounted for separately. Repeated B forward rollouts for displacement calibration are additional target access; the q=8 description refers only to original-model gradient calibration.

## Frozen alignment and physical interventions

Fit one per-pass activation Procrustes prior R_act on the 128 paired inputs. Normalize the eight A/B calibration gradient rows and fit

```
R_t = polar(G_A,t.T @ G_B,t / 8 + 0.05 * R_act,t)
P_transfer,c,t = R_t.T @ P_A,c,t @ R_t
```

Use that same alignment for mixed and shared source discovery. A one-row cyclic permutation of the eight B-gradient rows gives a shared-source descriptive alignment control. Record the full regularized singular spectrum, condition number, orthogonality and fit residual at each pass. Rank is four at each pass; require retained eigenvalue >1e-12 and lambda4/max(lambda5,1e-12)>=1.001. A zero gradient row or regularized singular value <=1e-12 blocks the corresponding construction.

Apply every projector through the actual `BPDModel/TiedComponentLinear` read override at all four recurrent passes, zero ablation, no sphere rescaling and zero residual bypass. Enforce a single scalar gain across passes. Model-specific physical factors remain in the receiving model. Save projectors and alignment matrices with hashes in the first-run archive.

## Common anchor and the two calibration policies

The **mixed-source transfer at gain 1** is the common reference for a root. Its retained clean-trajectory write RMS is E_star; its actual closed-loop write displacement RMS on the dose pool is D_star. This is an anchored diagnostic: the mixed-transfer arm is identical under the two policies. The B comparator within each policy is shared across source-context comparisons.

For clean row reads z and U=W_B.T, retained RMS is the square root of the mean of `(z_t P_t U)^2` over examples, passes and output coordinates. Clean-trajectory displacement instead uses `alpha*z_t P_t U - z_t U`. Actual closed-loop displacement uses the writes on the complete intervened trajectory minus the writes on the clean trajectory at the same input/pass. Record all these quantities with separate names, by pass and by context as well as globally.

**Policy E: retained energy.** For shared transfer, direct B and the permuted control, set alpha=E_star/E(P). Admit only finite gains in [.5,1.5]. A rejected gain remains rejected. It supplies no matched comparison through an alpha=1 fallback.

**Policy D: delivered closed-loop displacement.** Use B forward rollouts on the label-free dose pool to find a gain whose actual displacement matches D_star. Freeze this bounded search:

1. Evaluate all 17 gains on the uniform grid [.5,1.5], including 1. Record every gain and displacement.
2. If a grid value has relative error <=1e-4, select the accepted gain closest to 1, with lower gain breaking ties.
3. Otherwise find adjacent grid values bracketing D_star. Choose the bracket whose midpoint is closest to 1, then its lower endpoint to break ties. Perform exactly 24 bisection evaluations, updating the sign-changing bracket. The function need not be globally monotone.
4. Select among the queried gains meeting relative error <=1e-4 by distance from 1, then lower gain. If none qualify, return `NO_REGISTERED_BRACKET` or `MATCH_TOLERANCE_FAILED` as appropriate. A missing bracket means this search failed; it leaves global attainability unresolved.

Require D_star>1e-12. This search allows at most 41 full dose-pool rollouts per projector. Mixed transfer uses the exact anchor at gain 1. Shared transfer, B-only, the permuted control and each selected sham receive the same search budget and stopping rule. With three non-anchor main projectors and eight shams, there are at most **451 search rollouts** of 1024 B inputs, plus the explicitly counted clean/reference, parity and scoring calls. E calibration has access to the same dose inputs and records the resulting D. It uses no D-based gain selection. No labels, evaluation inputs or model gradients enter this search.

The two policies target different physical quantities. Their comparison measures the consequences of the registered recalibration. Closing a transfer gap can result from a worse B comparator, so every absolute task loss and gain remains visible.

## Sham controls and admission

Generate exactly 2048 local rank-four projector proposals around the **shared-source transferred** basis, using the NC06 local construction and angles .45,.65,.85,1.05 radians. Use generator seed `root*1009 + 13013`; cycle angles in that order. Project Gaussian columns off the shared basis, QR-orthonormalize, and construct E*cos(theta)+Q*sin(theta). Compute retained energy on the same clean dose inputs. Select the first eight proposals within +/-15% of the common E_star in fixed proposal order. Save all proposal IDs and retained energies, the accepted IDs and matrices, and rejection counts. Insufficient candidates blocks the sham-dependent inference.

These same eight projectors are used under both policies. Policy E retains their gain 1 and its +/-15% tolerance. Policy D applies the registered delivered-displacement search to each. A sham that fails D admission remains a failure; the sampler cannot substitute a later proposal. Matching displacement leaves downstream signed derivatives unconstrained. This control supports the registered engineering comparison and requires a stronger response-sensitive null for a later mechanistic identification claim.

Require clean task accuracy >=85% in both contexts for A and B, dense `mix` fraction >=90% at absolute weight>=1e-4, native full-mask logit error<=5e-5, KL<=1e-7 and relative physical residual<=1e-4. Check within-B factor-gauge covariance on the shared-source q8 projector at gain1, with physical-operator relative error<=1e-4 and closed-loop logit error<=5e-5. Restore B's head and verify its state hash after the zero-head diagnostic.

Keep an admission record per arm. An endpoint requires all sixteen pairs and all arms used by that endpoint to be admitted; otherwise its inference status is `INELIGIBLE`, with attempted measurements retained descriptively. Competence and global source failures apply to every endpoint. A failed sham calibration blocks the sham endpoint and any recovered-transfer verdict, while available context endpoints remain reportable under the full six-comparison family.

## Registered endpoints and family

Primary loss is B true-label NLL on x4>0. A pair contributes one mean, weighted equally with every other pair. Let L_M be the mixed-transfer reference, L_SE/L_SD the shared-source losses, and L_BE/L_BD the direct-target losses under the two policies. Six directional endpoints are registered:

| Endpoint | Root effect | Passing rule |
|---|---|---|
| Source context under E | L_M - L_SE | Lower interval >+.01 nats |
| Source context under D | L_M - L_SD | Lower interval >+.01 |
| Gap change with D | (L_SE-L_BE) - (L_SD-L_BD) | Lower interval >+.01 |
| Shared transfer against direct B under D | L_BD - L_SD | Lower interval >+.01 |
| Shared transfer against D shams | median eight D-sham NLL - L_SD | Lower interval >+.02 |
| Shared-transfer absolute fidelity under D | L_SD - clean B NLL | Upper interval <=+.05 |

Use 16,384 whole-pair percentile bootstrap resamples with NumPy PCG64 seed `26130013 + 101*j`, where j is the zero-based endpoint order above. Two-sided cell coverage is `1-.05/6`, with tails `.05/12` and `1-.05/12`. Preserve all six endpoints; technical ineligibility does not shrink the family. Report root values and nominal intervals as engineering uncertainty. Include the original B-only deficit under E and the permuted-control comparison descriptively. Private-context performance, correct-to-wrong/wrong-to-correct counts, per-pass KL and source-own-projector fidelity are diagnostics.

Output separate boolean flags for context-E, context-D and gap-change. `ENGINEERING_TRANSFER_RECOVERED` requires the last three endpoints to pass with all sixteen relevant pairs admitted. Otherwise output `ENGINEERING_NO_RECOVERY_OR_INELIGIBLE`. A positive context flag supports an effect of the registered source-population change. A positive gap-change flag requires inspection of absolute losses before attributing improvement to transfer. Neither a failed gate nor an interval crossing zero establishes equivalence.

## Evidence, resolution and execution order

The previous NC12 q8 effects are already exposed and concern a different estimator contrast. `DESIGN_PREFLIGHT.json` records an EvalMath call using that prior failure only as design context; it cannot estimate the new context effect or certify power for NC13. With sixteen new development pairs, the distribution-free 95th-percentile/95%-confidence matched-sham bound remains unavailable; its sample-maximum form requires at least59 independent roots from a fixed null population. Run a pinned post-development EvalMath check before interpreting a positive mechanism claim. Confirmation remains unregistered.

Commit the protocol first. Implement and test algebraic controls plus the excluded smoke next. Then commit an execution lock covering the runner, aggregate, protocol, source tests, every imported local helper and the runtime recipe. A full development run is enabled only from that lock. The first original sixteen-pair archive must retain model weights, frozen-choice manifests, source and runtime hashes, all gain queries, complete arm scores, failures and an aggregate. Follow-on documentation commits must not retrain the development roots automatically.

If code fails after a development root has been generated, preserve the partial archive and report a source failure. Any repair must identify the exposed population and keep the original failed run visible. A repaired execution cannot silently replace the first run. The ARG03-N, AMM, CQ and active SUB protected cohorts remain outside this protocol.
