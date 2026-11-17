# BPD-PD-08: staged read-basis discovery and label-free gate transfer

**Prospective scientific protocol dated 17 November 2026.** The frozen [CONFIG.json](CONFIG.json) was committed at 3dcb39180d53591a8c1e65b423cd7ac3e47967e3 before any PD-08 optimizer run. The original [PD-06](../pd06/RESULTS_20261117.md) and [PD-07](../pd07/RESULTS_20261117.md) scientific failures remain unchanged.

## Hypothesis and causal contrast

PD-06's count-supervised 4D models recovered the rotated read directions in 32/32 seeds, with poor exact routing of forbidden support combinations. PD-07 found that an exact normalized read gate executes all support combinations perfectly when read factors are planted and frozen (24/24), while joint unlabelled factor/gate learning aligned the read basis above cosine .9 in only four seeds and recovered the complete mechanism in 1/24.

PD-08 fixes the proposed interaction in two separated training stages: **stage one uses privileged cardinality labels to learn the physical read/write dictionary**, then both physical factors are **frozen bitwise**. Stage two refits **unlabelled** gates against the same dictionary. The comparison is between a gate using componentwise learned read coefficients and a raw-observation independent MLP trained on exactly the same factorization and held-out support task. Neither stage-two gate receives count or support labels.

This experiment tests **conditional compositional transfer after count-assisted dictionary discovery**. Any positive result does not establish unsupervised factor discovery, since stage one's count labels are privileged auxiliary supervision.

## Stage 1: one shared count-supervised factorization per seed

For each of 24 new seed–rotation pairs, train once using the original PD-06 model and count-supervised top-k surrogate with the real BPDModel/TiedComponentLinear physical shared linear I4 weight and four rank-one component factors. Initialise V and U randomly with init_svd=false; both factors and the original count/assignment GateMLP receive gradients during **all 360 Adam steps at LR .025**, with batch 32. The training source is the unchanged PD-06 independent rotation Q from Gaussian QR seed 900000+seed, and sparse latent vectors with k=1,2,3 sampled uniformly among the 11 training support sets.

Objective: output MSE + .30 matrix-faithfulness MSE + .05 actual binary gate activity + 1.0 cross-entropy against **true training support cardinality**. Only stage one sees the cardinality label. This repeats the mechanism that recovered the rotated read basis in PD-06, on new seeds.

At the end of the fixed stage-one budget, record the V and U state hash, read-basis similarity to Q (for sealed analysis only), and original count gate parameters. Then set requires_grad=false on both physical factors. **Every stage-two arm reuses those exact same frozen weights.** Their factor hashes must match before and after gate training, including the oracle and zero-fit controls.

## Stage 2: strictly unlabelled equal-budget gate refits

All stage-two training draws use a fresh generator at 400000+seed, with the same training support set and 32-board batches; no labels are used in any stage-two loss. For every trainable stage-two gate, reset gate parameters independently from the same initial seed 500000+seed, and train 180 Adam updates at LR .015. The physical factors and their read alignment remain fixed. Loss: masked output MSE + .05 actual binary gate fraction, executed through the real BPD component wrapper with residual delta off.

Prespecified arms:

1. **read_refit (primary):** threshold the absolute normalized learned read coefficient per component, with probability sigmoid(10(|x@V_j|/||V_j||−τ_j)) and τ_j=.25+.5 sigmoid(theta_j), theta initially 0. The gate's four threshold scalars train on stage-two reconstruction/sparsity only.
2. **raw_refit (comparator):** a shared raw-input GateMLP predicts four independent binary component masks. It receives the same frozen V,U, batch streams, budget and objective as read_refit, with fresh random gate weights matched across arm initialisation.
3. **read_nofit:** apply fixed 0.5 normalized read magnitude thresholds immediately after stage one; no stage-two training. This diagnoses whether read alignment alone suffices.
4. **read_shuffled_refit:** feed the normalized read activation of component j+1 (cyclically) into the gate for output component j, without adjusting the component writes. Thresholds train under the same budget. This destroys the correct read/write correspondence while keeping access to the same amplitudes.
5. **stage1_count_topk:** the stage-one count-supervised gate as trained, evaluated without any further updates. A repeat of PD-06's count-informed generalization problem on the same shared dictionary.
6. **oracle_support:** the true latent support activates the physical components in the registered source order. This is explicitly privileged evaluation and an achievable-output control.

The read and raw refit arms use identical source trajectories per seed but distinct architecture-constrained gate parameters. Their factor snapshots are identical. The latter four arms are secondary controls and never rewrite the primary comparison.

## Hidden support families and seal

Training, gate refitting and held-out evaluation all use the immutable PD-06 generator and the same per-seed Q, with no new supports in training. Active latent amplitude is signed and uniform on [.75,1.25]. Held-out includes 192 four-pass trajectories for each of six support families: four seen singletons, four seen adjacent pairs, **withheld pairs (0,2) and (1,3)**, three seen triples, **withheld triple (0,2,3)**, and **a withheld-support transition** [singleton, unseen pair, unseen triple, unseen pair]. Per-family evaluation RNG is 200000+seed+10000×family index.

No held-out family is accessed to choose stage-one or stage-two model parameters. The global component assignment is determined by factor-to-source-Q cosine alignment once per seed and held fixed over all six families. The unit of statistical replication is the model seed plus its independent rotation.

## Frozen acceptance criteria

A trainable PD-08 seed succeeds if its frozen dictionary has mean globally aligned read cosine ≥0.90, overall exact-binary output MSE ≤0.03, MSE ≤0.04 in each seen family and ≤0.06 in each withheld family, with exact complete four-pass planted support fraction ≥0.80 on each seen family and ≥0.70 on each withheld family. Every mask must be exactly binary; stage-one V,U first gradients and cumulative updates must be positive, and stage-two V,U must stay **bitwise identical** to the stage-one snapshot.

The **primary registered claim** read_refit beats raw_refit passes only if: **≥18/24** independent read_refit seed recoveries; **at least six net paired read-only seed wins**; **exact two-sided McNemar p≤0.05**. Report Wilson 95% seed intervals and all individual paired outcomes regardless of result, with per-family false-off/false-on event counts, learned thresholds, frozen factor angular errors, and comparison to zero-fit and shuffled-read controls.

## Scope and execution

The initial cardinality label is directly available from the exact sparse latent source. Thus the proposed benefit, if observed, is **a transfer property of gates after supervised basis learning**, not unsupervised discovery of a hidden mechanistic dictionary. If the pretrained factor basis itself fails to align, preserve the failed seed; never discard it or reinitialise after seeing outcomes.

A source-only engineering smoke on excluded seed 3001 verifies stage-one V/U gradient updates, shared seed-level Q, bitwise stage-two factor freezing, exact binary gates, read-shuffling correspondence, same stage-two initialisation and batch streams, and insensitivity of read/raw gate output to deliberately poisoned support labels. After CI passes, a one-time EXECUTE_LOCK pins the tested source and launches a 24×6 first-run CPU workflow, archiving the original receipt even on scientific FAIL. No full pretrained Sudoku TRM checkpoint is used.
