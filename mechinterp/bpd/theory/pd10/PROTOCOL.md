# BPD-PD-10 | Cold-start smoothing against the 45-degree read-mixing trap

**Prospective registration, November 17, 2026.** [CONFIG.json](CONFIG.json) was committed first at \`0bd49d042c9aa3693e7a17b7d6a7580cbed03d70\`, before any PD-10 model training or source evaluation. This document fixes the hypotheses, newly reserved model seeds, optimizer budgets, source, controls, success rules and evidence scope.

## Specific mechanism question

[PD-09](../pd09/RESULTS_20261117.md) tested unsupervised smoothed hyperplane mass on an orthogonal four-dimensional read basis. Its annealed and fixed-width training each recovered zero complete mechanisms across 24 registered seeds, although a count-supervised positive arm recovered 24/24.

Posthoc [pair-rotation quadrature](../pd09/landscape_audit.py) found a specific optimization trap: Gaussian near-zero mass at smoothing bandwidth \(\sigma=.6\) rewards two read axes rotated near 45° into one another. By \(\sigma=.15\), the true coordinate pair has a better *global pairwise* objective, while the 45° mixture remains a local maximum. The original schedule started at \(\sigma=.6\), and its observed read directions remained in a mixed-pair configuration in every seed.

PD-10 tests whether **starting directly at the narrow smoothing width \(\sigma=.15\)** avoids that attraction and recovers the exact source factors better than the original \(0.6\to0.3\to0.15\) schedule. The test is a prospective comparison of **optimizer basins** under the unchanged model family, source distribution, initialization and binary gate evaluation.

## Frozen source, hidden supports and independent seeds

Use the *unchanged* PD-06 \`rotation_for_seed\` and \`sample_batch\` source code. For each model seed, independently sample a fixed orthogonal rotation Q through Gaussian QR with generator 900000+seed. Observed dense inputs are \(x=aQ^\top\), with sparse latent a supported on one, two or three coordinates, independent signed active magnitudes uniform in [0.75,1.25] and exactly zero inactive coordinates. All training inputs are sampled among four singletons, four adjacent pairs, and three triples. The diagonal pairs (0,2), (1,3), triple (0,2,3), and their four-pass transition remain withheld combinatorially.

New PD-10 seeds, with no overlap with previous PD-01 through PD-09 cohorts: \`1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621\`. Engineering-only excluded seed: \`2011\`.

Train using fresh generator 100000+seed, batch 32, one CPU thread, 360 Adam steps per trained arm at LR 0.025, weight decay zero. Each arm within a seed draws the **same training batches**. The final held-out evaluation uses 192 four-pass boards in each of the six PD-06 strata with generator 200000+seed+10000×family index; no held-out data may influence any parameter, threshold, or model selection.

## Exact unlabelled optimizer and six arms

Use the *same* orthogonal-QR parameterization and smooth-\(L_0\) objective as PD-09. A trainable full-rank A starts at the randomly initialized PD-06 BPD read factor V; define V=QR(A) with deterministic diagonal-sign convention and U=Vᵀ. Thus the all-components-on physical weight satisfies \(VU=I_4\) throughout orthogonal training, and no reconstruction shortcut can make the dictionary collapse. Optimize only the mean smoothed near-zero read activity:

\[
L_\sigma(V)=\mathbb E_{x,j}
 \big[1-\exp\big(-\tfrac12(xV_j/\sigma)^2\big)\big].
\]

The four **random-init, unlabelled** arms have identical initial A, data, optimizer and 360 updates, differing **only** in the preregistered sigma schedule.

| Arm | Updates 0–119 | 120–239 | 240–359 |
|---|---:|---:|---:|
| **cold015 (primary)** | 0.15 | 0.15 | 0.15 |
| cold030 | 0.30 | 0.30 | 0.30 |
| warm030_to015 | 0.30 | 0.15 | 0.15 |
| **anneal060_030_015 (paired comparator)** | 0.60 | 0.30 | 0.15 |

The two further controls are explicitly separated:

- **privileged_pairmix015:** initialize V from the *true* source rotation Q with coordinates 0 and 1 mixed by a fixed 42° rotation, then train under \(\sigma=.15\) for 360 updates. This is a **privileged, deliberately near-trap initialization stress arm**, not evidence of unlabelled factor discovery. It tests whether the narrow-width optimization is attracted toward 45° from a nearby perturbation.
- **count_supervised_positive:** repeat the original PD-06 true-count-supervised physical BPDModel training on the same fresh seeds for 360 updates. This is an explicitly labelled solvability control.

No primary or comparator training loss accesses Q, source support labels or cardinality labels; Q is used for source generation and after training for evaluation. All initial random factors and input training batches are matched within the seed. The privileged initialization arm is documented separately.

## Physical component verification and success gates

After an arm finishes, install its fitted orthogonal read factors V and tied U=Vᵀ into a real BPDModel/TiedComponentLinear physical identity teacher. Evaluate four tied passes with residual delta masked to zero and sphere rescaling disabled. Apply **exact binary component gate** \(D_j(x)=1[|xV_j/\|V_j\||\ge0.5]\) with no gate-training updates. The count-supervised positive control uses its independently learned physical V and U, evaluated under this identical fixed read gate.

Read-axis recovery uses one maximum-cosine assignment over all 4! permutations, shared across *all six* evaluation families; signs are ignored. A seed succeeds iff read-axis mean absolute cosine≥0.90, overall binary-forward MSE≤0.03, each seen-family MSE≤0.04 and four-pass exact support≥0.80, and each held-out-family MSE≤0.06 and exact four-pass support≥0.70. Every executed mask must be exactly binary, with nonzero optimizer first-step gradient and nonzero parameter updates for the trained arms.

The preregistered **primary hypothesis** is a seed-level superiority of cold015 over anneal060_030_015. It passes **only if** cold015 recovers at least **18/24** full mechanisms, wins at least **six net paired seeds** versus the original annealing schedule, and yields a two-sided exact McNemar **p≤0.05**. All three conditions must hold. Report each arm's seed success fraction and Wilson 95% interval, exact paired discordant counts, and source-specific basis alignment, MSE, false-off/false-on count, read-margin diagnostics and OOD support scores. The independent seed/rotation is the inferential unit.

## Analysis, reproducibility and boundary

The [PD-09 restricted pair-mixing calculation](../pd09/landscape_audit.py) is a posthoc motivation from **prior independent seeds**. The PDF-style source-law idealization distinguishes sharp zero mass from smoothing-induced near-cancellations. The new schedule may fail because initial narrow smoothing can yield small gradients or an existing local mixed-pair optimum. A negative outcome must remain a negative outcome with no changes to source, seeds, sigma, optimizer or success thresholds after unsealing.

Before PD-10 science, excluded-seed tests verify QR differentiability, exact physical factorization and hard masks, sigma schedule, identical random factor initialization, no privileged label access in the four unlabelled objectives, and a known 42° pairmixed control that uses Q only in its own initializer. Register the source commit SHA with a one-time EXECUTE_LOCK and archive the original complete 24×6 JSON receipt on GitHub Actions, regardless of scientific verdict.

A positive outcome would establish a narrow optimizer result under a four-dimensional orthogonally mixed linear source with bounded active amplitudes. It would **not** establish convergence to the exact expected-L0 global optimum, generic nonlinear mechanism recovery or a result on the pretrained 7M Sudoku TRM.
