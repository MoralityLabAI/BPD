# EFD-01 conditional rank certificate

The Lean file EnergyFiberRank.lean formally checks a dimension-loss inequality: stacking the gradient of one scalar potential onto a readout Jacobian can remove at most one dimension, provided the stacked matrix has rank no more than the original readout rank plus one.

The proof models rank values as natural-number fields of a ledger; its oneScalarRow and futureBound premises explicitly represent linear-algebraic facts. It does not prove matrix rank subadditivity from definitions or verify an actual learned TRM Jacobian. The EFD-01 NumPy source-only tests measure concrete matrices numerically, while LoopedTransformers native tests use full float64 SVD.

The separate CI uses Lean 4.19.0 with #print axioms, no Mathlib dependency, no sorry, no pretrained model input and no protected test source.

A fully formal physical-space result still requires a Mathlib linear-map proof for the dimension of the kernel of a stacked pair of current-readout and scalar-energy derivatives. This later strengthening remains a distinct proof obligation.
