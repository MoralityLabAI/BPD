# INK-BPD-04: Goodfire writer alignment versus the physical reader tangent

**22 November 2026. Engineering and exact source-only mathematics.** This
study extends INK-BPD-01/03 in this directory. Existing development receipts,
the historical synthetic trained checkpoint, and all sealed BPD confirmation
populations remain unchanged. No model-weight training or independent task
cohort has been scored for INK-BPD-04.

## Motivation and evidence status

[Lee et al., *Interpreting Inkling 2 Mini*, Goodfire, 20 Nov 2026](https://www.goodfire.com/research/interpreting-inkling-2-mini/)
report high absolute cosine alignment between millions of MoE output
columns and reader weights, concentrated near block boundaries. They also
identify coordinate 2027 with RMS-normalized activation approximately 32.4
and 46% of the squared norm, which contributes to several next-loop
reader-score gaps. Their findings motivate a causal candidate, not a
demonstration that removing the coordinate harms the real model.

BPD's existing source-only `routing_math.py` and synthetic native
`TinyCrossLoopRouter` measure reader gaps, matched physical L2/RMS
interventions, and selector/content paths. INK-BPD-03 showed that a
single-pass gate inversion could be repaired by recurrence on development
data. This extension audits a mathematical confound in translating a
large writer/reader cosine or coordinate score share into physical
necessity: **the raw-state RMS normalizer changes under a write.**

## Exact derivative

For a frozen reader `u`, two prior block states `v2, v3`, and
`r3 = ||v3||_2`, the idealized Inkling reader computes

```text
s3 = dot(u, RMSNorm(v3))/sqrt(d) = dot(u, v3)/r3
gap = s3 - s2
beta3 = sigmoid(gap)
```

The exact Jacobian with respect to the *physical raw block-3 state* is

```text
grad_v3(s3) = u/r3 - dot(u,v3)*v3/(r3**3)
grad_v3(beta3) = beta3*(1-beta3)*grad_v3(s3)
dot(grad_v3(s3), v3) = 0
```

The final identity says a purely radial writer can have high cosine with
`u` and a large frozen coordinate attribution while having **zero
first-order reader-gate effect**. A transverse direction with lower
weight-space alignment can move the gate more. This is an exact
normalization-geometric fact, not an equivalence between a local
gradient and downstream task importance.

For each writer column `w`, `writer_scan` outputs both absolute
Goodfire-style raw cosine and the projection of the reader's *tangent
gradient* along the unit physical write direction. For a raw physical
patch `v3 -> v3'`, `physical_patch_audit` recomputes the entire
two-source reader after normalization, then compares actual gate
movement to the local first-order prediction. Its residual measures
curvature of this local reader response.

## Executable checks and scope

- `test_reader_tangent.py`: exact coordinate-score additivity,
  norm-energy shares, radial-null identity, centered finite differences,
  a deliberately radial high-cosine counterexample, and input rejection.
- `test_reader_tangent_native.py`: extracts actual raw second-slot
  writer states and query reader weights from the native PyTorch toy
  recurrence at pass index 1; checks that the analytic source reader
  reproduces its clean and physically patched softmax beta to float32
  tolerance; checks per-sample delivered dose and RMS conservation.
- Both use synthetic public objects. The native test uses an untrained
  model specifically to audit interface semantics, so it is neither a
  replicated learned mechanism nor a Goodfire/Inkling model result.

Run from repository root:

```sh
python -m unittest discover -s mechinterp/bpd/experiments/inkling_routing -p 'test_reader_tangent.py' -v
python -m unittest discover -s mechinterp/bpd/experiments/inkling_routing -p 'test_reader_tangent_native.py' -v
```

The native check requires the existing optional `torch` dependency.
The first check is pure Python and runs without model weights or GPU.

## Next causal experiment (requires a separate prospective registration)

A BPD factor attributed to a block-boundary reader or Inkling
coordinate 2027 should be source-selected separately by
(i) raw writer-reader cosine, (ii) block-normalized coordinate gap
and energy share, and (iii) physical tangent-projected gate sensitivity.
Preserve the complete selection family. On a *frozen, verified* model,
register actual writer hooks with full downstream normalization,
cross-loop Q/K/V/MoE readers, keep/gain gates and the output head.
Use single-pass and all-pass physical interventions, norm/L2-matched
random and disjoint writer shams, and a direct-gate positive control.
Report intermediate corruption and recovery with final verifier-backed
task outcomes. Include complete joint admission, head parity, physical
displacement, and source hashes.

The independent unit, causal task effect threshold, measured 59-root
sham-tail calibration, multiplicity, checkpoint sample, and a **fresh
sealed** confirmation population must be locked under EvalMath before
making a native causal claim. Prior NC-13/AMM/INK-BPD development data
cannot supply a new confirmation cohort. This source-only implementation
does not run a preflight or license one.

## Distinct evidence classes

- Goodfire observations: reported external, observational Inkling data.
- Reader tangent and radial-null: exact algebra for the frozen two-slot
  RMS-normalized reader.
- Native parity: synthetic PyTorch interface-level reproducibility.
- Pretrained Inkling writer necessity, BPD component transfer, or learned
  causal control of task success: **open**.
