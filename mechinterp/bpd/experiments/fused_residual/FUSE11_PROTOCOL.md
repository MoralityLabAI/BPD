# FUSE-11 | Retained adversarial PGD graph
Prospective protocol. Base da81dab14226c413cd5d0a71f61c4fe8143bba6a; FUSE-10 native gate passed.

Final adversarial rollout is followed by autograd.grad(loss_adv, leaf_sources,
retain_graph=True), then joint total_loss.backward(). Every residual request
in final rollout gets its own _Primal adjoint; use the same original detached
residual storage as earlier selected sites. The source-leaf gradient query
should prune factor-adjoint execution because leaf sources are independent of
V/U. Assert zero factor backward nodes before and after that query. Explicitly
seal the pool after the final forward, verify cached parameter identities and
versions, prohibit further forward admission, and keep independent adjoints
live until the native joint backward. No changes to the PGD update, optimizer
order, faithfulness, CI loss, or mask-only inner graphs. Close the pool before
optimizer mutation and on exceptions. Default off, step_exact only. Scope is
sequential first-order CPU FP32 with native adapters; no concurrent or external
raw-storage writes, checkpointing or higher-order gradients.

For T passes, K inner PGD iterations, M sites, k admitted sites, and both
loss branches enabled, eligible native reconstructions = M*T*(K+2).
FUSE09 with inner reuse = M*T*(K+2)-k*((K+1)*T-1).
FUSE11 = M*T*(K+2)-k*((K+2)*T-1). Incremental savings k*T with same
admission. Four sites, K=2, T=12: 192 native, 122 FUSE09 / 98 FUSE11
at binding two sites. Calculation: FUSE09 192-2*35=122;
FUSE11 192-2*47=98. Full cache: 52 to 4. Counts exclude clean mean
capture, faithfulness and evaluation. No timing extrapolation.

Gates:
- 96 stress fixtures, fresh roots 4011001..8, 4 scales, ordinary/exact/near
  residuals, stochastic graph plus discarded inner plus retained final;
  compare source query then joint V/U and input/mask gradients exactly,
  storage cap, 0 source-query factor nodes, 6 joint nodes per case,
  sealed-pool reentry rejection and cleanup.
- 9 native campaigns, two paired updates each, fresh seeds derived from
  4111001, width and passes. Matched FUSE09 versus FUSE11, optional equal
  FUSE10 on both arms. Full/binding/zero caps; K=0/2; stochastic or
  adversarial loss disabled; inner cache disabled; clean-window control.
  Check actual ascent gradients, raw preclip joint gradients, states,
  optimizer moments, metrics, tolerances, RNG, selected-site counts and
  cache lifetimes; exact finite tensors and scalars.
- Report descriptive paired timings; wide manually via --wide --width
  512/1024 --passes 6/12 after small native gate. No automatic expensive
  matrix expansion. GPU, mixed precision, real pretrained checkpoints and
  total memory remain HOLD.
