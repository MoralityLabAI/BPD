# HAD-03: complementary-sequence recurrent repair

Evidence class: **DEVELOPMENT_ONLY**. Initial CPU engineering run, November 26, 2026. This is a directly trained shared-factor recurrent proxy, not a native BPD decomposition, a TRM transformer, or an advance on an unresolved Hadamard order.

## Task and exact mechanics

Four sign sequences of odd length m. Define r[t] = sum_{s,i} x[s,i] x[s,(i+t) mod m], for t=1,...,(m-1)/2. E = sum_t r[t]^2. Flipping (s,i) changes r[t] by d[t] = -2 x[s,i] (x[s,i+t]+x[s,i-t]). The exact change is 2 r dot d + d dot d. For two distinct flips in the same sequence, add 4 x[s,i] x[s,j] at the unique inverse-shift class t = min(|i-j|,m-|i-j|). The oracle labels all first actions achieving minimum energy after one or two distinct flips. The verifier assembles the Goethals-Seidel array and checks all entries and the integer Gram identity.

The five candidate features are sign, normalized exact one-flip energy change, normalized squared residual displacement, normalized current energy and normalized sequence sum. Three passes share the V/U factor pair and global-mean context map. Width 24, 1,897 trainable parameters. No attention. This feature design makes greedy information available to every neural arm.

## Paired engineering recipe

Roots 7301-7304. Random signs at train lengths 5,7,9; 80 common warmup updates, batch 12, AdamW learning rate 0.003. Select one of six contiguous four-column U groups using 64 fresh training-length-7 calibration states and minimum intervened oracle cross-entropy. Four arms share warmup weights and 40 fresh paired batches, each with a fresh optimizer. The causal displacement is the detached negative task-loss gradient at the first-pass state projected into the chosen subspace, normalized to 0.75 per nondegenerate example. The sham is a fixed signed coordinate permutation of the fixed edit. The live arm recomputes the projector from the same indices. All arms compute ordinary and three auxiliary parameter gradients; non-task arms add the selected auxiliary gradient at 0.25 of the ordinary gradient L2 norm, then clip the combined gradient at 1. Ordinary inference is intervention-free. This is a label-informed training edit, not a demonstration of semantic component transport.

32 random starts at each unseen length 11 and 13 per root; 64 search iterations. Successful states are absorbing. Neural search adds Gumbel noise at scale 0.15. Greedy chooses minimum next energy, with random moves at non-improving states. Annealing samples one candidate per step with a simple linear cooling schedule. The evaluation matches iteration count and computed candidate-feature arrays, not FLOPs, wall time or classical search tuning. The annealer is an untuned sanity baseline.

## Executed local CPU result

PyTorch 2.10.0+cpu, one Torch thread, 7.435 seconds total on the local sandbox. 256 shared starts per arm, 1,536 arm-episodes. Every solved state passed exact Gram verification during execution. Each row below contains solved counts.

| arm | m=11 | m=13 | total / 256 | roots 7301..7304 | mean minimum energy |
|---|---:|---:|---:|---|---:|
| task |122|116|238|58,59,60,61|1.125|
| fixed |118|114|232|56,60,58,58|1.5|
| sham |122|111|233|56,61,58,58|1.4375|
| live |118|115|233|57,60,58,58|1.4375|
| greedy |128|121|249|64,61,62,62|0.4375|
| anneal |42|19|61|10,22,12,17|16.0|

Initial-state oracle top-1 set accuracy: task 0.8125, fixed/sham/live each 0.80859375. Fixed minus task = -6/256, fixed minus sham = -1/256. No steering advantage. Four roots and a near-ceiling greedy baseline make this a diagnostic, not confirmatory inference. The local receipt contains per-episode solve bits, group choices and calibration losses; successful sequences and checkpoints were not persisted by this pilot. Source verification tests one-step residual identities, the two-step label set against sequential recomputation of one-step energies, and an order-12 certificate.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python mechinterp/bpd/experiments/hadamard_repair/pilot.py --out had03_results.json
```

Requires numpy and torch; the source runs independently of the repository package. The repository's native BPDTrainer/TiedComponentLinear and its importance network have not yet been integrated.

## HAD-04 successor, proposed only

1. Use distinct engineering roots to select lengths and budgets with greedy success 20-80%; do not reuse pilot roots for confirmation. Compare one-flip greedy, exact two-flip lookahead, tabu, multi-restart annealing and learned proposals at matched candidate work and wall time. Include exact feasibility restrictions from row sums and PSD.
2. Cache r, per-action d and candidate energies with a configurable memory budget; measure incremental update correctness and actual bytes/RSS versus recomputation. Never materialize the full two-action tensor at conjecture-scale lengths.
3. Include an action-feature-only MLP, unrolled network and tied recurrence. Replace label-gradient guidance with training-only causally selected native BPD component interventions, evaluate both fixed and live groups, and record physical dose/gradient norms, projector drift, edit fidelity and protected-output damage.
4. Require independent verification of serialized successful sequence certificates and a frozen evaluation manifest. Primary contrasts at root level are fixed vs task and fixed vs sham, with a prospective minimum meaningful effect, multiplicity adjustment and enough independent roots. No claim about open orders until an independent exact certificate is produced.
