# RT-BPD-02: reached-state and fixed-reader-weighted low rank

**Status:** exact source-only formulation and CPU fixture; no Nanbeige pretrained evaluation. The original 168-case ARG-03-N cohort remains sealed, and the Nanbeige research goal remains paused.

[Source-only registration](PROTOCOL.md) · [NumPy implementation](weighted_rank.py) · [CPU tests](test_weighted_rank.py)

## Why this sits alongside BPD

The `c074e35` Nanbeige synthesis in `MoralityLabAI/LoopedTransformers` reports almost equal learned VPD/BPD *physical operators*, together with large NF4/BF16 differences in reached inputs and native donor writes. Its REFIT49 symmetric accounting assigns most observed cross-precision pass-1 write disagreement to the reached input. These are development observations, not a native success for the new method.

Ordinary SVD minimizes `||W-K||_F^2`. The recurrent question instead asks how accurately the reduced operator acts on the input distribution at each **physical occurrence**. Let `Sigma=E[h h^T]`, or use `E[(h^A-h^B)(h^A-h^B)^T]` for source contrasts. For a fixed reader `A` chosen in advance, use:

```
min_{rank K <= r} || A (W-K) Sigma^(1/2) ||_F^2.
```

With `S=Sigma^(1/2)`, the closed-form solutions are:

```
K_input  = (W S)_r S^+
K_reader = A^+ (A W S)_r S^+
```

Pseudoinverses cover singular positive-semidefinite moments and rank-deficient readers. The exact optimal loss is the squared tail of the singular values of `W S` or `A W S`, respectively. Input directions in `ker(S)` and write directions in `ker(A)` are unidentifiable under this local criterion, so arbitrary factor labels have no causal interpretation. These results are weighted low-rank algebra, rather than new theorems about neural behavior.

For a nonlinear continuation, an input-dependent Jacobian `A(h)` changes the objective; the closed forms only cover a frozen linear `A`. A future native study must use a registered physical reader, rounded dose and oracle-scored continuation before claiming causal benefit.

## CPU reference and result

From the BPD repository root, with NumPy installed:

```bash
python -m unittest discover -s mechinterp/bpd/theory/rtbpd02 -p 'test_weighted_rank.py' -v
```

The pre-registered three-dimensional fixture has `W=diag(4,3,2)`, `Sigma=diag(0.01,1,10)`. For rank 2, conventional SVD drops the third coordinate and suffers weighted squared error **40**, whereas reached-state SVD drops the first and attains **0.16**. With rank 1 and reader `[0,1,0]`, reader-weighted fitting gets zero reader error whereas input-only fitting gets **9**. These are predetermined synthetic identities. The tests also check singular moments, rank-deficient readers, input-coordinate rotations, the `Sigma=I, A=I` reduction, paired contrasts and rejection of malformed inputs.

## Archive bridge requirements

For a numerical test on Nanbeige, source a CAL-only bundle of *actual reached* input vectors indexed by physical layer, module, pass, token support, precision, independently grouped root and world, with checkpoint and source digests. The published aggregate currently reports only physical operator and vector cosines, not the covariance matrices or raw reached vectors needed for this fit. No native model or sealed confirmation cohort is accessed by this package. The next action after this source gate would require a separate explicit resume decision.
