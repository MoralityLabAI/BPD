# Finite-horizon observable equivalence of BPD optimal-basis masks

## Scope

The 2026-11-17 BPD optimum-basis result chooses maximum-weight hyperplane-mass bases in the ideal full-rank unregularized expected-L0 model. The relative optimum change `R=M M_*^-1` need not be monomial. Its full-representation mask invariance requires `R^-1 D R=D` when the write is injective and every input direction is reached. This note proves the strictly weaker *readout-relative* version for a linear recurrence, including limited reachable input spans.

Let a source representation be `x=M_* a`; `U` is its write map into recurrent state, `T` a square recurrent linear map, and `C` the current linear readout. For masks `D=diag(d_1,...,d_d)`, compare the anchor edit `U D M_* a` with the reparameterized edit `U R^-1 D R M_* a` on input span `im S`. Their relative error is `U (R^-1 D R-D) M_* S`.

## Theorem 1: observable equivalence and Gramian

Put `E_D=(R^-1 D R-D) M_* S`, `A_j=C T^j U` and `G_k=sum_{j=0}^k A_j^T A_j`. The following are equivalent over the reals for every finite `k`:

1. `A_j E_D=0` for every `j<=k`;
2. `E_D^T G_k E_D=0`;
3. every column of `E_D` lies in `intersection_{j=0}^k ker(A_j)`.

Proof: `E_D^T G_k E_D=sum (A_j E_D)^T (A_j E_D)`, a sum of positive semidefinite matrices; each diagonal entry is the sum of squared real coordinates. Thus the sum vanishes exactly when every `A_j E_D` vanishes. Exact-Fraction arithmetic implements this identity without floating-tolerance decisions.

Consequently, the set of observationally invariant masks decreases as k grows. For `T` of state dimension `n`, Cayley-Hamilton expresses `T^n` as a combination of `I,T,...,T^(n-1)`. Multiplying the resulting identity by successive powers of `T` shows checking through `k=n-1` establishes zero deviation at **all** later visits. The dimension bound uses the recurrent-state dimension, not the BPD factor rank.

**Limit:** for nonlinear dynamics, `A_j` is the Jacobian of the native readout after `j` visits at a fixed anchor. First-order invisibility gives a candidate tangent quotient, while an actual finite patch needs nonlinearity/margin controls. SCM-14-like Hessian bounds would supply a sufficient local error budget only after architecture/source matching.

## Theorem 2: delayed break of an optimum-mask symmetry

Assume an absolutely continuous full-dimensional Gaussian law for the source activation. Every nonzero read direction has hyperplane mass 0, so every invertible read basis is expected-L0 optimal. Choose

```
M_*=U=I_2, R=[[1,t],[0,1]], D=diag(1,0), t != 0,
C=[0,1], T=[[0,1],[1,0]], S=I_2.
```

Then `R^-1 D R-D=[[0,t],[0,0]]`. At the present readout `C E_D=0`, so the singleton mask appears invariant. After one recurrence `C T E_D=[0,t]`, which is nonzero. Both the zero and full masks commute with every R and remain invariant at all horizons.

This witness has **four instantaneous invariant masks and two one-step invariant masks**, although both bases are equally sparse in the ideal L0 objective. The delayed-observable quotient is strictly finer than the present-readout quotient.

## Theorem 3: a truncated-observer mask family may not be a block algebra

Now take `d=4`, `R=I+e_1(e_2+e_3+e_4)^T`, `M_*=U=I`, `S=(0,-1,-1,1)^T`, `C=e_2^T` and `T` swapping coordinates 1 and 2. The difference `E_D` is supported on `e_1` with coefficient `-d_1+d_2+d_3-d_4`. Thus every mask is currently invisible, while one-step equivalence holds exactly when

`-d_1+d_2+d_3-d_4=0`.

Six of the sixteen binary masks satisfy this condition. In particular `(1,1,0,0)` and `(1,0,1,0)` are one-step invariant, while their coordinate intersection `(1,0,0,0)` fails. The one-step invariant masks are **not closed under meet**, so they are not a Boolean algebra of coordinate blocks. The full common-centralizer masks from BPD's injective/full-span theorem remain a Boolean block algebra.

The failure comes from a lower-dimensional reachable input span and observational cancellation. Accordingly, BPD's invariant-mask graph must only be interpreted as a finest block partition under its stronger full-representation commutation premise. This exact observation corrects any attempt to infer universal component blocks from incomplete output readouts.

## Proposition 4: present-readout top-1 capacity versus delayed observability

For `W=I_2`, `C=e_2^T`, the same swap `T`, and observable visits `j=0,1`, the stacked observability matrix is a permutation of `I_2`, hence rank 2. For the mixed input `a=(1,1)`, no diagonal 0/1 mask keeping at most one component can preserve both observed readouts. For the phase-disjoint input `a=(1,0)`, the top-1 mask `(1,0)` does preserve them. This supplies a readout-relative capacity explanation for why BPD-PD-03 top-1 gates can succeed on phase-disjoint inputs yet fail on mixed-source controls.

The proposition is an exact linear capacity witness; PD-03's measured routing statistics remain unchanged. Learned gating performance, basis discovery, rotated recurrent states and nonlinear head-normalization require independently registered experiments.
