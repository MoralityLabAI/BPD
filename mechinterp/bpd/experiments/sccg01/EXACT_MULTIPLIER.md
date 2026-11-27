# SCCG-08: complete algebraic multiplier audit
Date 2026-11-27. Additive source instrument on SCCG-07.
The 192 roots are the already exposed SCCG-07 constructed fixtures.
The original 65-point rational search and exact Monte Carlo default are
unchanged. This is a complete exact decision on the stated rational
mathematical inputs, not a trained-model or measured training result.

## 1. Polynomial reduction
Let M(lambda)=A+lambda B be an n-dimensional real symmetric affine pencil,
with rational entries in A,B. On a rational closed interval [l,h],
consider det(z I+M(lambda))=z^n+c_1(lambda)z^(n-1)+...+c_n(lambda).
M is PSD iff every c_j>=0. Necessity follows by multiplying the
factors z+mu_i with nonnegative eigenvalues. Conversely any negative
eigenvalue gives a positive real root, impossible for a monic polynomial
with all nonnegative coefficients. Each c_j has degree <=j.
For SCCG-07 use the cap eta² rho2/e² as h and zero as l.
The S-lemma premises from ROBUST_ATLAS.md remain mandatory.

The square-free least common multiple of the nonconstant coefficient
polynomials and both interval boundaries has degree <=n(n+1)/2+2.
Isolate every real root over QQ; refine to strictly separated rational
intervals. Check one rational point in every intervening open cell within
[l,h], then every isolated root. At a root of the product, a constraint
vanishes iff its gcd with the product has a root in the isolator.
Otherwise the constraint has constant nonzero sign there. This is complete,
including endpoint/singleton feasibility. Zero and constant polynomials
are handled separately. The feasible set is a closed convex interval
possibly empty or singleton; the sign-cell method does not require
interior feasibility. No complexity claim for large coefficient heights.

The independent positive-witness check uses every principal minor instead
of characteristic coefficients; it refines an algebraic isolator when
minor roots not belonging to the characteristic product could obscure
a sign. The optional principal-minor decision comparator uses the same
root-isolation machinery with different polynomial constraints. SymPy
1.14.0 exact arithmetic and root isolation are trusted dependencies.
No Lean kernel certificate or independently implemented root isolation.

## 2. Counterexamples to multiplier grids
Take G=1,H=1,h=1/2,rho2=3,e=eta=1. Then
M(lambda)=[[lambda-2,1/2],[1/2,3-lambda]].
Its determinant is -(lambda-5/2)^2; at lambda=5/2 it is PSD.
That unique feasible multiplier is absent from the 65-point grid
lambda=3i/64. This is a genuine SCCG-07 case, with positive offset.
Any policy treating a grid miss as proven unsafe is unsound.

For a generic affine-pencil edge case, take block diagonal matrices
[[lambda,1],[1,lambda/2]] and [[2,lambda],[lambda,1]].
Their PSD intersection on [0,2] is exactly lambda=sqrt(2).
No rational multiplier grid, however fine, finds it. The solver returns
a polynomial and isolating interval [41/29,17/12]; the principal-minor
checker confirms the algebraic witness. This generic pencil example
is not asserted to have SCCG-07's special block form.
Also check zero pencil, singular PSD pencil, point interval, indefinite
constant pencil and the irrational singleton excluded by either bound.

## 3. Exposed cohort replay
Local Python/SymPy 1.14.0 replay of 96+96 previous fixtures:
orthogonal 41 safe/55 unsafe; varying clean coordinates 36 safe/60 unsafe.
All 77 positive witnesses pass principal-minor replay; all 96 orthogonal
decisions match the independent closed-form matrix condition. The same
12,288 rational sample edits were replayed. The new solver admits no
additional historical roots. Both formerly unresolved cases 61035 and
61045 have empty multiplier-feasibility sets by both polynomial methods.
Explicit rational dose-feasible violating edits, independently checked:
61035: (-41679/65536,13893/32768,13893/32768,41679/65536);
61045: (58617/65536,-58617/65536).
These were located after the original campaign by a deterministic small
integer-direction search with rational dose bisection; the regression
replays the resulting vectors. No population inference.

All 192 local historical rows match seed, regime, trace, grid witness
and sample breach fields in the preserved SCCG-07 receipt. The local
fixture was an equivalent minimal copy of the original rational class;
hosted replay imports the unchanged full robust_atlas.py. New source
Git blobs:
exact_multiplier.py 24787db6239fb9dba8c7f9133adc463cfac8b3f3
multiplier_benchmark.py 918ae33e55fd429fca727872ca6cbb978ef1b2df.
Local second receipt SHA256
d6bac715d0a543035288c679d0054df43edf80566d6d58ebbfc335bf8ce5fc04.
Local first receipt SHA256
0e14356cc968fb87f1197ad7b5e7bb804c345f57704933771d65d95613aaa435.
Elapsed ~7.13 seconds includes all exact decisions, old grid, sample
replay and principal-minor checks. No performance comparison.

## 4. Operational scope
Suggested admission hierarchy: cheap sufficient trace or rational PSD
certificate; on failure use exact multiplier audit if affordable;
otherwise refresh/fall back. A mathematically unsafe subspace for uniform
reuse can still contain individual acceptable interventions. Never
promote approximate reuse into an exact structural exclusion.
No native checkpoint drift or full recurrent continuation. For the next
stage, derive a valid finite collateral certificate for nonzero baseline
protected scores, then verify costs on fresh roots before any training
admission. Exact input uncertainty and rounding remain open.
