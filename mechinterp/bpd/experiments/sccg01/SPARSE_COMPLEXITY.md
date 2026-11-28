# SCCG-01B: sparse selective-control complexity and cancellation circuits

Addendum to THEORY.md; exact mathematical proof and finite fixtures.
No claim of priority: related sparse-linear-system hardness is classical
(Natarajan, 1995, https://doi.org/10.1137/S0097539792240406), and kernel
circuits are the elementary vectors of a representable matroid.
The following result is tailored to the protected-control objective.

## Decision theorem
Given rational C, a and integer k, decide whether some real u satisfies
||u||_2 <= 1, Cu=0, |supp u| <= k, a^T u > 0.
This is NP-complete even for C with entries in {-1,0,1}, a a single
coordinate selector, and Euclidean physical dose.

Membership in NP: a subset S of at most k column indices is a certificate.
Test rank([C_S; a_S^T]) > rank(C_S) by polynomial-time exact rational
elimination. This is equivalent to a_S being nonzero on ker C_S.
A null vector with nonzero target can be sign-flipped and scaled to
meet the unit physical dose. No encoding of irrational coefficients is
needed in the certificate.

Hardness: reduce Exact Cover by 3-Sets (X3C), a classical NP-complete
problem (Garey and Johnson, Computers and Intractability, 1979, SP2).
Given 3q elements and m triples, let D be the 0/1 element-triple
incidence matrix, C=[D|-1], a=e_(m+1), and k=q+1. Write u=(x,t).
Then positive target requires t>0 and D x=t*1. At most q triple
columns may be used. Every element must be covered by their union,
since an uncovered row would give 0=t. Each triple covers exactly
three elements, so at least q are needed. Equality forces q pairwise
disjoint triples covering the entire universe. On that support each
nonzero coefficient x_j=t. Conversely an exact cover permits all q
selected x_j=t>0. Normalize with t=1/sqrt(q+1).

This proves a gap stronger than mere feasibility: the exact optimum
under this reduction is either zero or 1/sqrt(q+1). Every feasible
positive target requires a cover and has dose sqrt(q+1)*t; a cover
attains the displayed optimum. Any polynomial algorithm returning a
feasible positive-factor multiplicative approximation on all instances
would decide X3C in polynomial time under the usual exact/rational
decision model (or distinguish the polynomially separated objective
gap using adequate certified precision). This hardness says nothing
by itself about typical learned BPD dictionaries.

## Rank-bounded circuit theorem
Fix C and a over a field of characteristic zero, and write r=rank C.
Some support S of size <=k permits nonzero a^T u in ker C_S iff some
column-matroid circuit T of C of size <=min(k,r+1) has a nonzero target
on its unique (up to scale) null relation.

Proof: the reverse direction is immediate. In the forward direction
choose inclusion-minimal T subset S whose restricted target is nonzero
on ker C_T. Choose u with a^T u != 0. If ker C_T has dimension >=2,
choose a nonzero w in ker C_T intersect ker a_T^T. Take a coordinate
where w is nonzero and subtract a multiple of w from u to cancel
that coordinate, preserving target. This contradicts minimality.
Thus dim ker C_T=1. Every nonzero null vector has full support,
otherwise a proper subset is dependent and its one-dimensional null
relation equals the full-support relation, a contradiction. So T is a
circuit, rank C_T=|T|-1, and |T|<=r+1. Rank-zero matrices include
singleton zero-column circuits.

Consequently, positive selective gain can be decided by enumerating
at most sum_(j=1)^min(k,r+1) binomial(n,j) supports and checking
the augmented rank. For fixed protection rank this is polynomial in
n (an XP bound in r, no FPT claim). If k>=r+1, checking a outside
row(C) suffices. This theorem is for existence of positive gain.
Optimal finite gain can require a combination of several circuits and
a different support search. Small numerical singular values also do
not supply exact rank certificates.

## Implication
A zero-collateral BPD policy should first estimate the effective
protection rank and identify minimally dependent supports with target
gain, then assess finite nonlinear behavior and budget. For large rank,
use bounded support portfolios or exact search only where affordable;
the NP-completeness result prevents a general greedy guarantee.
This is a precise reason to study Goodfire-style communities and
MMS-resident reader response maps in recurrent models.

The independent Python fixture enumerates small exact-cover families
and tests the general rank/circuit statement on small integer matrices.
No neural outcome, Lean kernel verification, or protected split access.
