# SCCG-06: dynamic protected-response quotient atlas
2026-11-27. Constructed development, after SCCG-01 through -05 merged.
The new fixture measures exact support/kernel maintenance, not finite
task scores, neural weight updates or training speed. The benchmark was
executed locally before publication; no prospective population inference.

## Exact invariant quotient
Work over Q (or R for the mathematical statements). Let C_t=C_0+UV_t,
with U full column rank r and fixed. Choose L of full row rank with
ker L=im U. Then D=LC_t=LC_0. For support S let Z_S be a basis
matrix of ker D_S. Every vector killed by C_t,S is killed by D_S,
hence uniquely Z_S w. Conversely C_t,S Z_S w=0 implies Z_S w is a
valid null vector. Therefore ker C_t,S = Z_S ker(C_t,S Z_S).
Moreover LC_t,S Z_S=0, so im(C_t,S Z_S) is contained in im U and the
reduced matrix has rank at most r.

Choose a row map M with MU invertible; [L;M] is invertible, so we may
store D and solve only E_t,S Z_S w=0 with E_t=M C_t. The rank-nullity
identity is nullity(C_t,S)=dim ker D_S-rank(E_t,S Z_S). If D_S is
injective then S can be discarded for the entire declared update class.
This is stronger than caching active circuits: inactive candidates
must remain available because changes in E can create circuits.
At fixed D and support budget, if E changes on columns J, only
candidate supports intersecting J need refresh. An inverted index
implements this rule; it is sufficient, not necessarily minimal
(E changes can also annihilate Z_S). Changed D requires rebuild.
Multiple mutable rows and rank changes are allowed within this split.

No claim that ordinary SGD has this exact structure. A projection
of a measured gradient onto a low-rank update class is insufficient
to certify exact membership. If an external update direction changes,
verify the actual invariant D or invalidate the atlas. Full source
matrix changes can erase the entire reuse advantage. The implementation
takes fixed/mutable integer row matrices; arbitrary rational row-map
output would require clearing denominators before modular screening.
The non-coordinate U fixture uses an exact integer annihilator.

## Physical metric and task readout
For a fixed physical dictionary W with independent selected columns,
cache H_S=Z_S^T W_S^T W_S Z_S. Let K_t,S span the reduced kernel;
then the current metric is K_t,S^T H_S K_t,S, and the target and
clean-state projection terms can be compressed in the same coordinates.
Changing W invalidates these metric caches even if D stays constant.
For zero baseline protected scores this is a valid structural screen for
SCCG-05 finite control. Nonzero protected baseline scores need a
different affine consistency test (SCCG-03); this kernel-only screen
cannot be used to reject all variable-radius edits in that case.
No finite-gain optimizer or metric-cache update is executed in SCCG-06.

## Work and memory
Compile by screening each size-k support once on D; full rank modulo
a prime proves full rational column rank, deficiency alone does not.
Retain the exact rational kernel of each remaining dependent support.
For q updates the rebuild cost is sum of q full support scans; reuse
pays one scan and reduced refreshes. With comparable per-epoch costs
B (rebuild), R (refresh), one-time compilation A, break-even is
q(B-R)>A when B>R, ignoring other shared costs. Costs of verifying
invariance, indexes, rank fallback and any changed physical dictionary
must enter a real training budget. The program checks invariant D on
every update and does not estimate a GPU or training-time speedup.

## Constructed benchmark
Eight roots 41000..41007, 14 columns, choose 4, 8 fixed protected rows,
one mutable row, 12 epochs per root. Six roots have two disjoint
Vandermonde cancellation groups from the SCCG-05 construction; two
roots are unplanted. Dense and sparse changes open and close
cancellation supports. All candidate enumeration ignores the planted
group identities. The direct full rebuild is the comparator. Both
cached paths must match the direct support and nullity at every epoch;
each returned basis must annihilate the current exact matrix and
have independent columns. Edge fixtures include higher nullities,
changing mutable row count, modular false deficiency, non-coordinate
U, and stale quotient rejection.

Local Python 3.13.5:
- 96,096 support-epoch decisions; 96 epochs on 8 roots.
- 12 quotient candidates across roots; 36 active-support births
  including the initial state, 30 deaths.
- Both cache paths agree exactly with the direct rebuild throughout.
- Rebuild: 96,096 modular screens, 54 full rational kernels;
  0.911555439 s total.
- Dense refresh: 8,008 modular screens, 12 full rational kernels,
  144 reduced kernels; compile 0.07495088 s, update 0.002553594 s,
  total 0.077504474 s.
- Sparse refresh: 8,008 modular screens, 12 full rational kernels,
  77 reduced kernels, 67 reuses; compile 0.075652007 s,
  update 0.001849536 s, total 0.077501543 s.
- Maximum serialized base payload per root 68 bytes; excludes
  dictionaries, inverted index, matrix inputs, active states, allocator
  overhead and process memory. No measured RAM/VRAM peak.

Timing is a single local sequential run, with arm order reversed
on alternating roots. Verification and common fixture generation
are outside these policy timings. Both cached totals are about 11.76x
faster on this small constructed workload. Sparse refresh cuts the
reduced solves but shows no material total-time advantage at this scale.

## Reproduction and evidence
Run python dynamic_atlas.py --output receipt.json.
Tested source Git blob 02e111f0b0141d0287ccf00516c2d306826611ad,
SHA256 706dea293e1d2d35eb493bad094a6fa42ac2ae46aebca21a7e09e5ab421303a1.
Local receipt SHA256 96ee74f65fb12507b0464ad824d3356f2c9a8ffa0fb22cd3313066f6e3f8fb8f.
The receipt includes canonical per-epoch support/nullity digests.
CI produces a fresh receipt (timings may differ). Verification shares
the exact Fraction kernel routine, but directly computes full kernels
and separately tests lifted basis validity and independence. No
independent Lean kernel proof or trained model results.

Next: quantify exact and approximate update structure on actual
training checkpoints before adopting this cache. Measure retained
memory, cache admission/break-even and invalidation costs versus
task-only reinvestment. Derive robust finite-collateral bounds for
a fixed structural atlas under nonzero update residual; a small
residual does not imply exact nullspace preservation. Preserve
existing sealed roots and require prospective EvalMath preflight.
