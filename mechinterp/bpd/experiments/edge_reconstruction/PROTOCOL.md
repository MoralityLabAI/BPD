# BPD-ER-01: Edge-deck identifiability and causal reconstruction

Source: [Cogentic edge reconstruction](https://github.com/google-research/google-research/tree/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/edge_reconstruction). Its reported theorem covers simple undirected graphs with m >= 1000n; no theorem transfer to neural graphs is assumed.

## Exact stage
Run oracle.py. Enumerate n=4..6 graphs with known n,m; independently canonicalize each card, preserve isolated vertices and multiplicities. Recover all compatible isomorphism classes from partial multiset S. Label ambiguous cases as sets, never force a unique answer. Split by canonical parent graph before sampling cards. Include triangle-plus-isolate versus claw as a negative control. Baseline: candidate extension of one card followed by exact deck containment; compare brute enumeration, WL prefilter plus exact matching, and learned ordering with the same verifier budget.

## Causal stage
Use sparse directed linear recurrent systems h[t+1]=A h[t]+Bx[t], with observed coordinates and known intervention semantics first. Choose A with spectral norm <=0.9; interventions zero one entry of A without renormalizing. With full state and excitation matrix H of full row rank, noiseless transition measurements give A from (Y-BX)H^+. Rank deficient H permits ambiguity. Compare direct least squares and sparse recovery before learned search. Later use independently permuted observation coordinates, partial readout C, then nonlinear BPD component gates. Hidden coordinate transformations and component drift can destroy identifiability; score only a declared equivalence class or observable predictions. A component gate may remove multiple edges.

## MMS arms
A: random feasible probes; B: greedy entropy reduction using exact candidate bank; C: learned BPD/MMS ordering; D: shuffled utility scores. Equal probe count, candidate verification count, model initialization and CPU thread budget; report wall time and peak RSS separately. Sweep candidate cache budgets 0, 16, 64, 256 MiB. Use uniform prior over canonical hypotheses for stage one, explicitly record posterior assumptions in later stages. Graph-deck observations are passive samples; active intervention selection belongs to the causal stage and requires a defined query interface. No oracle access to hidden edge identities.

## Evaluation
Primary: exact compatible-set recall and precision in graph stage, held-out intervention prediction error in causal stage. Secondary: exact reconstruction among identifiable instances, ambiguity calibration, edge F1 under fixed coordinate labels, probe count, verifier calls and RAM. Eight independent roots, paired arms, held-out graph families and sizes; report root-level uncertainty. Freeze seeds and budgets before evaluation. Require positive paired gain over random and shuffled arms and compare to analytic/greedy baselines. No neural training or causal results claimed yet.

## Handoff
LoopedTransformers: train recurrent candidate ranker. DataGoblin: generate verified partial-deck tasks and ambiguity certificates. HybridTRMLDT: branch over candidate edges. MathCan: prove candidate extension completeness and rank identifiability. Compactification-Theory: study sufficient summaries of hypothesis sets. Keep exact oracle separate from learning inputs.
