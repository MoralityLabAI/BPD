# SCCG-05: certified rare-support search and reusable structural atlas
Date 2026-11-27. Constructed development only. This is a restricted
zero-baseline reader problem, not native neural steering or general
SCCG-03 search. No trained ranker, reserved roots, or paid compute.
The first local experiment preceded publication; rank-cache and atlas
comparators were added after inspecting the first outcome. Both runs
are development evidence, not a prospective statistical comparison.

## A. Monotone finite-gain upper bound
Let delta belong to T_U = im(W_U) intersect ker C_phys, with ||delta||<=e.
Assume C_phys x=0 and a_phys^T x=0, m>0 and rho0²=||x||²+m.
Exact protected normalized scores equal zero iff C_phys delta=0.
Let P_U project onto T_U. The scaled score gain is
rho0 a_phys^T delta / sqrt(||x+delta||²+m).
Cauchy-Schwarz bounds the numerator by rho0 e ||P_U a_phys||.
The denominator is at least sqrt(m+||(I-P_U)x||²), since the orthogonal
component of x is fixed. Hence for all descendants:
V_U² <= rho0² e² ||P_U a_phys||²/(rho0²-||P_U x||²).
For nested spans, projected target norm decreases and denominator
increases, so the bound is monotone. Strictly below a positive threshold
prunes the entire branch. Equal bounds are retained.

For independent physical columns W, use G=W^T W, h=W^T x and
a=W^T a_phys. Let Z contain a rational basis of ker(C_phys W_U).
Then the projection norm squares are exactly
(Z^T a_U)^T (Z^T G_U Z)^-1 (Z^T a_U)
and the analogous expression for h_U. All arithmetic is rational;
no numerical rank cutoff. For integer protected columns, full column
rank modulo a prime proves full column rank over Q (a nonzero minor).
Modular rank deficiency alone proves nothing. The implementation
only prunes on full rank and confirms remaining leaf nullspaces over Q.

## B. Exact one-dimensional finite decision
For physical null line w, choose its sign so A=a_phys^T w>=0.
Write G0=||w||²>0, h0=x^T w. Along delta=t w, t>=0,
scaled gain squared is rho0² A² t²/(rho0²+2 h0 t+G0 t²).
Its derivative has sign rho0²+h0 t for t>0 when A!=0.
If e<rho0, Cauchy-Schwarz implies rho0²+h0 t>0 on the allowed ray.
Thus the maximum occurs at t=e/sqrt(G0); the opposite ray cannot
produce positive gain. Threshold g>0 is attained iff
L sqrt(G0)>=B, with
L=rho0² A² e²/G0 - g²(rho0²+e²), B=2g² h0 e.
Compare signs before squaring, reversing the squared inequality when
both sides are negative. The benchmark guarantees leaf nullity at most
one; the implementation refuses a larger nullity rather than quietly
using an inapplicable oracle. General SCCG-03 remains the fallback design.

## C. Constructed rare circuits with independent ground truth
n=14, k=4, protection rank 8. Base columns are (1,t,...,t^7) for
distinct integers t. Every set of <=8 base columns is independent by
the Vandermonde determinant. On 12 of 16 roots two additional columns
are negative sums of disjoint triples of base columns. Expand any
linear relation supported on <=4 columns into a relation of at most
8 base columns. If it contains one composite, all three members of
its triple must be present and coefficients are proportional to all
ones. With both composites, disjointness would require six base columns
in addition to the composites, exceeding four. Thus exactly two
dependent four-supports per planted root, each nullity one, and none
on the four unplanted roots. Columns are randomly permuted.

Physical map is 2I plus a signed shift, with norm <=3 and smallest
singular value >=1. A spare side coordinate has clean value 3;
nonzero editable clean coordinates ensure x is not orthogonal to
the editable span. Any rational protected and target coefficient
rows in the generator can be extended to physical query rows:
solve their editable restrictions via W inverse and set the side
query coefficient to cancel their clean numerator. Normalization
mass m=1, e=2, threshold g=4 (scaled score units).
On each circuit the target sum is 32 (high) or 1 (low). The physical
null direction with four unit coefficients has norm in [2,6].
rho0>=sqrt(10), and the augmented normalization radius lies between
rho0-e and rho0+e. The low circuit gain is <=rho0/(rho0-2)<4.
The high circuit has gain >=(64/6)rho0/(rho0+2)>4 at positive full dose.
This gives exact ground truth independent of the search.
Each planted root has two solvable queries, one infeasible query;
unplanted roots have three infeasible queries.

## D. Local development outcomes and cost
Seeds 31000..31015. Three target queries per root, fixed structural
geometry within a root. The exhaustive audit of query zero checked
16,016 leaves, detecting exactly 24 dependent supports. All 48 query
decisions agree with the construction for every arm. All comparisons
are same-root; no independent population inference.
First local run: lex 0.477s, rank 0.359s, geometry cold 1.562s,
64 KiB 1.514s, 512 KiB 0.852s. Post-result added rank cache and atlas,
then replayed the original arms. Second run:

| Policy | Modular rank calls | Rational nullspaces | Finite queries | Peak retained cache estimate bytes | Seconds |
|---|---:|---:|---:|---:|---:|
| Lex | 34849 | 60 | 60 | 0 | .513232 |
| Rank tree | 21310 | 60 | 60 | 0 | .385037 |
| Rank tree, 512 KiB | 9231 | 24 | 60 | 412984 | .248335 |
| Geometry cold | 14978 | 1462 | 39 | 0 | 1.568907 |
| Geometry 64 KiB | 14825 | 1338 | 39 | 65536 | 1.488166 |
| Geometry 512 KiB | 7401 | 617 | 39 | 492112 | .764599 |
| Structural atlas | 16040 | 24 | 60 | 2600 | .140789 |

Atlas scans all 16,016 supports once, saves the 24 dependent supports
and their exact structural data, then answers changing target queries.
The recorded rank count includes the repeated 24 deficient-screen
calls. The geometric bound avoids 21 finite queries yet spends much
more on rational nullspaces and Gram inverses. With 512 KiB,
geometry reuses 7,577 cache entries. A smaller cache thrashes
(13,696 evictions); bigger cache helps this implementation, but the
cheap structural atlas remains faster and uses far less retained data.
Atlas cache estimate excludes list storage, container allocator and
the common model/geometry inputs. Estimates are recursive Python
object sizes plus fixed allowance, not measured RSS or a full memory
budget. Times are descriptive single-process sequential observations,
not randomized interleaved timing or hardware throughput evidence.

## E. Reuse theorem and limits
For zero protected baseline and independent W, every support with
positive finite gain must have nonzero ker(C_phys W_S). Thus
target-independent enumeration of all dependent size-k supports is
complete for every target with zero clean numerator, positive threshold
and any positive dose (smaller supports are included by padding).
The structural atlas is exact if its downstream finite solver is exact.
It is not a polynomial-time algorithm for unrestricted n,k. Enumerating
only minimal circuits may miss the optimum of larger supports combining
several circuits. C, W changes invalidate structural data; x changes
invalidate cached projected clean norm, but leave nullspace/Gram
unchanged. A nonzero baseline protected score permits variable-radius
solutions with no nullspace and needs the general SCCG-03 treatment.
Learned proposals were deferred after SCCG-04 and this strong exact
control; no neural or compute-amortization claim for a general family.
Next: measure break-even across repeated changing queries and matrix
updates, including exact low-rank cache updates; then introduce nonzero
protected scores and full-trajectory native response with prospective
EvalMath preflight. Hardness results and evidence seals unchanged.

## Reproduction
python circuit_search.py --output receipt.json
Local executed source SHA256 e396eb64d80d9609dd8eceae388f879076014089cdf43e1d9cc03ef0d6a282d7
Git blob a694c5d9ccf8d99f268323cf16062212253ae0ed.
Original local first receipt SHA256 987482ba48c28548008499776f76608a77c5c6b70575b9cacf7200ff8f5f4c55.
Second receipt SHA256 0ff858ff253f6b069e070b87d5919e1aa641859cfdf621d7d580b4ce13fe46bd.
A hosted replay must compare non-timing rows; Python object-size
counts can also be platform dependent.
