# AMM-05 pre-source clarification: unlabeled readout budget

Registered before source freeze and before training/development data (2026-11-18).

The candidate feature's signed local delayed response uses an **unpatched
reference continuation readout** q(F(h1_r)) in addition to the two
component-patched continuation readouts. Each agent therefore receives
exactly **three unlabeled continuation readout measurements** per donor
case: one common baseline plus one for each candidate. Both policies
receive the exact same readouts, the same 8 noisy local calibration labels
per new model root, and the same SVD-canonical weight-derived geometry.

The external donor target and full-donor continuation remain audit-only.
Computing them to construct labels or report ceiling diagnostics does
not grant them to the policy. The frozen 11-feature vector, model counts,
noise levels, root splits and evaluation thresholds are unchanged.
