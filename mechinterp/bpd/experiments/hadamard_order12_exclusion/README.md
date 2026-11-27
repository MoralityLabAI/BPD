# HAD-11: computational exclusion of the remaining order-12 multiplier family at order 2396

2026-11-27. Exact finite CPU computation. No new Hadamard order, Lean proof, learned-model experiment or Fable consultation. Novelty relative to the literature is unverified.

## Statement and proof boundary

There is no quadruple of binary sequences on Z598 with summed periodic autocorrelation -4 at every nonzero shift invariant under the common multiplier group G={1,139,183,185,231,277,321,367,413,415,459,597}. This is a computational exclusion of this particular cyclic width-four construction family.

In CRT coordinates Z2 x Z13 x Z23, the group acts as H6 on Z13 and inversion on Z23. Split parity halves into U,V, complementary signed ternary channels with summed PAF targets 599 delta - 2J and 599 delta on Z299. Each of the three Z13 orbit columns A,B,C has weight 1,6,6, and is symmetric on Z23.

Compression onto Z13 yields the complete HAD07 catalogue: 12 canonical U sectors and 10067 labeled sign-normalized V partners. Original source and quotient receipt replay byte-for-byte. Normalize central margins independently by U/V row signs and sort by the two central margins, retaining tied row labels. These operations preserve the relaxation.

Compression onto Z23 requires S=A+6D with D=B+C and sum S^2 equal to 599 delta -26J for U and 599 delta for V. Thus the A constraints modulo12 are necessary. The central support theorem enumerates all 2^24 binary parameter pairs in the characteristic-two symmetric group ring, with first-support orientation normalized by common multipliers. Match signs modulo3 independently on complementary supports. Full original roster: 1093 central margin profiles, 248 positive, 57745 normalized central assignments, 635195 after restoring orientation. Independent fast root-table existence solver, multiplicity counter, export and scalar validation agree (2656270 PAF and 461960 margin checks).

## Local margin and mass enumeration

D is zero, twice a signed coordinate axis or mixed with both coordinates ±1. E=B-C is respectively a doubled signed axis, zero or one of two opposite mixed vectors. These choices are equivalent to B,C being signed unit axes. Local U/V costs are 3d^2+a*d; both budgets are <=49, sum 90. With A_V mass m, budgets are (481+m)/12 and (599-m)/12. Enforce B and C singleton parities, exact D row sums and costs. Enumerate the D multisets within equal A signed-axis groups of inversion pairs.

For z zero-D pairs, p mixed equal-sign pairs, q mixed opposite-sign pairs, and v2 doubled V pairs, the E half-pair margin is (x+y,-x+y)+2(l_u,l_v), where x=-p,-p+2,...,p and y=-q,-q+2,...,q. The zero-D part is feasible iff |l_u|+|l_v|<=z and z-|l_u|-|l_v| is even. Its V degree k ranges from |l_v| to z-|l_u| by two. Combined B+C V mass is singleton mass+2(p+q+2v2+2k). Each row record carries all attainable mass bits and D permutation multiplicity. Join exact channel costs and total B+C mass (599-m)/6.

Incremental HAD10 roster: 582 quotient pairs, 3096 histogram and 91880 witness combinations. Filter retains 463 pairs, 2108 histograms and 38184 witness combinations. Direct route from original catalogue bypasses prior local support parity, combined energy and central histogram filters. 3045 pairs have central witnesses; 22558 histogram tests represent 1305092 witness combinations. Filter retains 987 pairs, 6252 histograms, 278948 witness combinations. Independent group-composition enumeration with explicit E DP covers all 4622 row classes without the prior D-margin pruning, agrees on all 84252 allowed row-cost entries and all survivor histograms. Independent audit validates every emitted pattern, singleton option, cost, margin and multiplicity: 385963 direct patterns, 23863317 expanded row-class records.

## Complete matching

At fixed A, the equation sum(A_f D_f+3D_f^2)=R=(target-sum A_f^2)/12 gives a linear convolution syndrome mod3. Expand D multisets into every placement within each signed-axis A group; match both channel syndromes at complementary exact costs, verify mass and full integer S PAF. Row signs, permutations, per-row B/C swaps and a common Z23 multiplier preserve this relaxation. The raw-mask independent canonicalizer verifies all 278948 direct equivalence assignments to 102617 representatives. No mixed CRT equations are needed because this larger necessary relaxation is empty.

A second matcher uses an additive uint64 fingerprint of the full integer PAF computed by a quadratic form. Equal integer PAFs necessarily have equal fingerprints, so collisions only add candidates. Exact integer verification handles any hit. 10000 scalar fingerprint cross-checks pass. Both matchers share the independently audited D row banks and energy partition.

| Campaign | Jobs | Pair operations | Expanded D rows | Signature matches | Mass matches | Exact S matches |
|---|---:|---:|---:|---:|---:|---:|
| Incremental syndrome | 38184 | 1466869804 | 285470496 | 32 | 32 | 0 |
| Direct syndrome | 102617 | 2369246705 | 424478607 | 14 | 14 | 0 |
| Direct integer fingerprint | 102617 | 2369246705 | 424478607 | 0 | 0 | 0 |

Final cap hits zero; all jobs accounted for. First incremental low cap deferred 22768 jobs, which subsequently completed. Initial independent local audit printed and wrote its receipt before shell timeout; independent row enumeration also recompiled and replayed bytewise. The included-main warning was corrected by explicit return 0. No floating-point exclusion. The reproducibility archive HAD11_order12_exclusion.zip includes the complete sources, direct and incremental receipts, prior HAD10 inputs and HAD07 quotient source, audit scripts and reproduction shell wrapper assembled after the individual runs. 297 entries, SHA256 7c71d26dd295f5e7312b7b36f1eeb69c853b76cbcc8d35d12f804f17763b8429. This repository subset publishes the syndrome kernel and audit summary; the package is required for the full direct replay and independent matcher. Executed syndrome.cpp Git blob: 75728e2892488d2fe2de67d213a25e27ac539981. Independent integer_match.cpp blob: a88973a8c90be9ba9f79949e1bc3e99d0236d9a1. Independent local enumeration blob: e6ee5023323435f15d9bdb5fdd15a6b5efabf0b8.

## Next step

With earlier exact obstructions, 20/32 common multiplier groups are excluded. All groups of order >=12 are excluded. Largest unresolved group order8, 96 orbits per row, 384 binary signs, {1,47,183,229,369,415,551,597}. Use the same approach with four Z13 orbit columns of weights 1,4,4,4 and derive new central congruences and energy bounds. Wider borders and other multiplier groups remain unresolved. Preserve the exclusion as a deterministic LDT rule; causal training requires EvalMath preflight and preservation of all sealed cohorts.
