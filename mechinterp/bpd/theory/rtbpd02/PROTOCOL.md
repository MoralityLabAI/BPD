# RT-BPD-02: reached-state / reader-weighted low-rank theory (source-only registration)

**Registered 2026-11-23.** Scope: exact finite-dimensional mathematics, deterministic NumPy CPU fixtures, and a schema for a future, explicitly authorized CAL-only Nanbeige archive replay. The Nanbeige pretrained research goal remains paused; no model loading, new neural forward passes, training, GPU reservation, or protected-root access is authorized by this protocol. The 168-case ARG-03-N confirmation cohort stays sealed.

## Physical questions and estimands

A shared operator `W : R^n -> R^m` is applied repeatedly to inputs with pass-specific reached-state second moments `Sigma_t = E[h_t h_t^T]`. For paired counterfactual worlds at the same occurrence, define `Delta h_t = h_t^A - h_t^B` and `Sigma_t^Delta = E[Delta h_t Delta h_t^T]`. The root is the independent unit; worlds, tokens, module occurrences and passes are nested. Keep a fixed pass, layer, module, token selector, checkpoint, precision and factor rank in every comparison.

For positive-semidefinite `Sigma`, fixed linear downstream reader `A` (or identity), and rank bound `r`, minimize

```
    E_A(K) = tr(A (W-K) Sigma (W-K)^T A^T)
           = || A (W-K) Sigma^(1/2) ||_F^2
    subject to rank(K) <= r.
```

Freeze three candidates: (i) Euclidean rank-r truncated SVD of `W`; (ii) input-moment-optimal `K_r`; (iii) fixed-reader-and-input-moment-optimal `K_{r,A}`. Compare their realized physical operators, empirical squared write-error and downstream reader-error on the same source-only vectors. A data-dependent Jacobian `A(h)` has no claimed single-SVD solution; any such optimization requires a separately specified numerical method.

## Exact mathematical claims to check

Let `S=Sigma^(1/2)`, `S^+` its Moore-Penrose pseudoinverse, and `B=AWS`. With `M_r` denoting an SVD rank-r truncation:

1. `K_r=(WS)_r S^+` attains the minimum, including for singular PSD `Sigma`, with objective `sum_{j>r} sigma_j(WS)^2`.
2. `K_{r,A}=A^+ (AWS)_r S^+` attains the minimum for a fixed `A`, with objective `sum_{j>r} sigma_j(AWS)^2`. The result follows because `AWS` has columns in `im(A)` and rows in the row space of `S`.
3. Solutions are generally nonunique on `ker(S)` and `ker(A)`. Pseudoinverses choose a minimal-norm representative, with SVD degeneracy handled by comparing operators' physical responses and objectives rather than arbitrary atom labels.
4. For an orthogonal input coordinate change `Q`, `W' = W Q^T`, `Sigma'=Q Sigma Q^T`, the objective and optimal physical map transform covariantly. In degenerate singular-value spaces, compare objectives and projector subspaces rather than expecting an identical SVD basis.
5. When `A=I` and `Sigma=I`, the formula reduces to conventional truncated SVD.

These are weighted Eckart-Young facts; novelty claimed here is the recurrent-state and downstream-reader application, not the linear-algebra theorem.

## Locked synthetic checks

- `W = diag(4,3,2)`, `Sigma = diag(0.01,1,10)`, rank 2. Standard truncated SVD has weighted error `40`; input-weighted optimum has error `0.16`. Compare both against the tail singular-value oracle, with absolute error <=1e-9.
- Same `W`, `Sigma`, rank 1 and reader `A=[0,1,0]`. Reader-specific optimum must have reader error <=1e-9; input-weighted rank-1 candidate has reader error `9`.
- Positive-semidefinite singular `Sigma`, with a zero eigenvalue. Match the SVD tail oracle and verify no spurious demand on the unobserved direction.
- Identity-input/identity-reader reduction; orthogonal input-gauge covariance; nonsquare rank-deficient reader; null reader; zero moment; invalid indefinite/asymmetric/nonfinite input rejection; paired-contrast input checks.
- All tests run on CPU using only Python stdlib and NumPy. No Torch, HuggingFace, device allocation, model assets or external services.

Source-only acceptance means each theorem-derived identity agrees with its independent numerical oracle and the locked fixtures pass. No native Nanbeige correctness or behavioral mediation claim follows from this gate.

## Future archive replay (separately authorized)

Accept CAL-only archived, actually reached *physical* pre-down_proj inputs, using an explicit schema containing repository/commit/checkpoint digest, backend and quantization, physical layer, pass, token selector, root ID, world ID, input vector, and paired source-identification policy. First build root-balanced `Sigma_t` and `Sigma_t^Delta` on **CAL roots only**; freeze bases before comparing on disjoint DEV roots. Log post-rounding physical interventions and independently scored oracle outcomes as separate later stages. The currently public Nanbeige report gives aggregate cosines and fitted-operator comparisons; it does not publish the raw per-root reached-state vectors required to construct these covariances. This source-only stage therefore makes no archive-fit accuracy assertion.

Primary reference: `MoralityLabAI/LoopedTransformers` Nanbeige synthesis at `c074e35` (`reports/nanbeige_bpd_gym/NANBEIGE_RESEARCH_SYNTHESIS_20261123.md`), and `BPD` NC-06 gauge/weighting work. Maintain the research pause and the protected ARG-03-N seal.
