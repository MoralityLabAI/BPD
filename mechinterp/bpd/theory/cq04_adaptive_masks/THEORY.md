# CQ-04: adaptive mask cost and basis gauge

For a learned transition \`F(h)=(hV)U\` with two components, define one visit with a hard mask \`m ∈ {0,1}²\` by \`F_m(h)=((hV)⊙m)U\`. A two-visit input-conditioned execution plan is \`π(h₀)=(m₁,m₂)\` and produces \`h₁=F_m₁(h₀)\`, \`h₂=F_m₂(h₁)\`. The active-slot cost is \`k(π,h₀)=||m₁||₀+||m₂||₀\`, between zero and four.

The independent target is \`(a,b)\` from \`h₀=(0,0,a,b)\`. The per-input objective on development examples is to minimize \`k\` subject to \`max(|h₁[0]-a|, |h₂[0]-b|) ≤ ε=0.12\`. A trained imitation policy observes the original source coordinates and predicts hard slot bits. The target-guided oracle is a *training-supervision upper bound*. It is never exposed to held-out labels at inference.

Let \`A_m\` denote the fixed two-visit plan's probability of success on independent calibration cases, and \`k_m\` its cost. The input-blind comparison solves the simple linear program \`max Σ_m p_m A_m\` subject to \`Σ_m p_m k_m ≤ κ\`, \`Σ_m p_m=1\`, \`p_m≥0\`, where \`κ\` is the controller's calibration mean active-slot budget. With the budget and normalization constraints, an extreme solution uses at most two nonzero \`p_m\`. Enumerating pairs gives the exact calibration optimum among input-blind stochastic constant masks.

A component-based donor patch at the intermediate state is \`δ=((h_d-h_r)V ⊙ m₁)U\` and the future readout uses the recipient's fixed \`m₂\`. Norm-matched current-dark controls remove the immediate visible coordinate before matching the patch length. Only the donor's independently specified b enters the causal success criterion. The larger-budget full intermediate replacement is a ceiling.

For invertible \`R∈GL(2)\`, \`(VR)(R⁻¹U)=VU\` exactly. With component masks inserted between those factors, \`((hVR)⊙m)R⁻¹U\` generally differs from \`((hV)⊙m)U\`. Therefore hard mask identity is coordinate-dependent; retaining the same four bits after a shear measures gauge fragility. A complete BPD invariant claim requires block-level or gauge-invariant routing, outside CQ-04.

The eight model seeds are independent roots for this model-initialization family. The many inputs within one root support conditional performance estimates, not an eightfold larger population of models. The sham-tail bound under EvalMath requires a separate larger pilot calibration.
