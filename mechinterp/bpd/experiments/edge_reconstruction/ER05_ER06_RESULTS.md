# ER-05/06: Factorial moments and bit-plane cache (2026-11-26)

## Exact theorem

For a posterior on k-edge supports and a binary-sign probe let K count selected positive edges. For j=0..k define mu_j=E[binom(K,j)], with mu_0=1. Since mu_j=sum_{r>=j} binom(r,j)p_r, binomial inversion gives p_r=sum_{j>=r}(-1)^(j-r)binom(j,r)mu_j. Thus moments up to k determine the full outcome law and its entropy. Each mu_j is the sum of j-way joint inclusion probabilities over the positive edge set. For k=3, in order r=0,1,2,3, p=(1-mu1+mu2-mu3,mu1-2mu2+3mu3,mu2-3mu3,mu3). Scope: fixed sparsity, equal known edge magnitudes, binary-sign noiseless sensing. No assertion for general neural BPD.

Pair insufficiency even at known equal posterior cardinality: take two disjoint cubes of three edge pairs each (12 edges). Each support selects one of each pair in a single cube. In P take the four even-parity choices from each cube; in Q take the four even choices in the first and the four odd choices in the second. Both banks contain eight distinct supports and have equal singleton and pair inclusion counts: within a cube each variable is unbiased and each pair of variables is independent, while across cubes no edges co-occur. Choose the first edge in each pair positive. K has distribution (0,3/4,0,1/4) under P and (1/8,3/8,3/8,1/8) under Q. Entropy differs by exactly one bit (0.811278 vs 1.811278). Any deterministic utility function of cardinality and singleton/pair marginals has worst-case absolute error >=1/2 bit on these banks. Probe realizable in n=6 bilinear family: all output signs positive and three input signs positive, three negative. Reachability of these banks from the uniform all-support prior is unproved. Both fixtures pass executable equalities; no Lean formalization.

Pair-only relaxation: mu3 belongs to [max(0,(2mu2-mu1)/3), min(mu2/3,1-mu1+mu2)]. Entropy is concave along this interval; the numerical max uses 28 bisection steps on the entropy derivative. This is an optimistic approximate score, not a certified numerical upper bound. The full posterior filter remains exact.

## Reproduction

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python moments.py > moments.json`

NumPy 2.3.5, CPU. Source SHA256 of local final file: 90b4faaa486ab1c2db8805426c54c2ed60badbc0d1f62b34471edee4739056f3. Raw local receipt SHA256 3cf43b6ba6409cb7a76ffc37bf799853bc876dec51c72cf1a8999742bd7bbf03. Benchmark executed first; equal-cardinality fixture added and executed separately. New probe roots 144..207 for n=6; shifted n=7 roots 208..231. Same seed namespace 261127, 64 probes, 12 measurements, uniform support bank; exact tie handling lowest unused index within 1e-12 of max.

| n | roots | candidates | random unique at 8/12 | pair maxent at 8/12 | dense / planes / moments at 8/12 |
|---|---:|---:|---|---|---|
|6|64|4060|31/63|59/64|63/64|
|7|24|11480|8/22|15/24|20/24|

All 88 roots reproduce dense probe choices and surviving counts exactly for both exact methods. 67584 factorial-moment distributions matched direct integer outcome counts (all 64 probes x 12 steps x 88 roots). True support retained by every policy.

| n | byte matrix payload | two-plane payload | median dense ms | median planes ms | median paired speedup | moments ms | pair-only ms |
|---|---:|---:|---:|---:|---:|---:|---:|
|6|259840|65024|3.810|3.596|1.079|4.385|11.011|
|7|734720|183680|9.972|8.759|1.249|9.538|16.154|

Raw bit planes encode K using low and high bits; intersections with alive mask give low/high/both counts, from which all four bins are exactly recovered. Cache construction and filtering charged to time; moment direct-audit overhead excluded. Timing order rotated by root. Shared support enumeration and world creation excluded. All modes currently allocate the byte matrix transiently. Pair and moment modes also retain pair tables; third-order single-mask payload (32512/91840 bytes) is not total RAM. Python objects, LUT, index arrays and temporary allocations are excluded from reported raw payload; RSS unmeasured. Speedups are exploratory single-run CPU measurements, not preregistered. Pair-only failed speed objective; matched probe budgets, not CPU budgets.

## Next gate

Use bounded/noisy unequal weights, explicit posterior probability updates and calibrated uncertainty; exact two-bit scheme applies only when outcome alphabet has four symbols. Benchmark direct packed construction without dense temporary and RSS on larger banks; compare compiled bit-plane intersections against the dense baseline. In BPD, estimate higher-order structure only for intervention-relevant groups and include an exact/approximate uncertainty check. Actual BPD intervention adapters and support-disjoint learned generalization remain pending.
