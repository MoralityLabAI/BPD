# ER-01 pilot receipt (2026-11-26)

Commands (Python, NetworkX 3.6.1, NumPy 2.3.5):
```
python oracle.py
python audit.py
OPENBLAS_NUM_THREADS=1 python causal.py
```

Executed locally. Source SHA256: oracle.py b03e8b9f30ed66795ee37295c8f48d31246102b9ace4751c7b27e137cff521af; audit.py 023f17e62914f95b8ec261105efd7e0386cdfaaf144217bf4b899b47bfbf907e (unused import removed after run); causal.py 62110724497e835015b8d7272f1f175f4c6928534878fc5d80070368a5cf1319.

Four oracle smoke checks passed. Atlas/VF2 checked every card for all 201 atlas graph classes on n=4,5,6; 1373 cards matched. This verifies deck construction independently against VF2 but does not exhaustively call the original labeled-graph enumeration API on n=6.

| n | graphs | cards | ambiguous full | minimum sufficient subdeck count histogram (size:frequency) |
|---|---:|---:|---:|---|
|4|11|33|4|0:4, 1:2, 3:1|
|5|34|170|4|0:4, 1:6, 2:14, 3:6|
|6|156|1170|4|0:4, 1:15, 2:115, 3:16, 4:2|

Known n,m and multiset containment; minimum is existential, zero means a unique graph at known n,m. Full deck ambiguities include two-edge adjacent/disjoint and triangle/claw with isolates. These tiny graphs are outside Cogentic's density theorem.

## Directed support sensing
32 independently seeded roots (seed 261126), 6 nodes, 30 directed edges excluding loops, exactly 3 active entries at weight 1/4, 4060 hypotheses, 64 independent Rademacher probe pairs, 12 measurements per arm. Measurements are known-scale c^T A x, evaluated via integer support sums. All policies preserve the true candidate. Cache 259840 raw bytes excluding support storage and temporaries.

Unique counts at each probe 1..12:
- random: 0,0,0,0,0,1,5,9,13,18,22,25
- adaptive exact entropy: 0,0,0,0,3,10,27,32,32,32,32,32
- static initial entropy: 0,0,0,0,0,4,7,12,17,24,27,29

Median remaining at probe 8: random 2, static 2, adaptive 1.
Adaptive root-level improvement in unique recovery at probe 8: 23/32 vs random, 20/32 vs static (adaptive all recovered).

Exploratory deterministic simulation, not preregistered; matched measurement budgets only. This is active linear sensing, not edge ablation or neural BPD. No learned arm, no timing/RSS comparison, no generalization claim, no completed Lean proof. Candidate predictions are indexed by hypothetical supports, with hidden support accessed only for environment observations and assertion. Next test: freeze roots, compare OMP/LASSO and random, static, adaptive and learned utility; add unequal weights, bounded noise, and partial observation. Include cache recomputation vs retained matrix at matched CPU and verifier budget.
