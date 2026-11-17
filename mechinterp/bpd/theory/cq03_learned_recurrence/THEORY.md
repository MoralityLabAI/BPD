# Learned recurrent observability and BPD component gauge

Let `F(h)=Wh` be a learned two-step weight-tied linear recurrent transition, `q(h)=Ch` its scalar readout, and `W_hat=(V U)^T` the reconstructed BPD physical operator. The training labels enforce `CWh0=a` and `CW²h0=b` for `h0=(0,0,a,b)`. Training W from these behavioral labels does not force a particular matrix of channel swaps. The decomposition is trained afterward using the separate W and state targets.

A component intervention at the first visit changes the intermediate state by

`δ_c(h_d,h_r) = ((h_d-h_r)^T V_c) U_c^T`,

with the actual future readout change `C W_hat δ_c` and instantaneous perturbation `C δ_c`. The donor behavioral target `b_d` is a property of the independently generated input, not of `δ_c`. For a current-dark projected intervention, set `P_kerC δ_c` and normalize its norm to the native dose. This supports a matched comparison against `P_kerC` random directions, with a separate test of current visible equality.

A calibrated singleton component is basis-dependent. For any invertible 2x2 `R`, the transformation `V'=V R`, `U'=R^-1 U` leaves `V'U'=VU` and therefore the full two-component physical W invariant. In general, the individual `c` contribution and its donor-transfer effect change. A centralizer-invariant block is a different object from a singleton feature chosen in one arbitrary trained gauge. The CQ-01 finite-horizon quotient uses `C W^j U` to test detectability; a nonzero delayed singular value does not, by itself, guarantee an outcome change above task tolerance.

This experiment uses **both BPD components active** at each pass. It tests the trained factor geometry and intervention reliability, while separate PD-03 experiments test gate cardinality and false-ons. A good full reconstruction can coexist with poor singleton transfer. In the present two-output task, state reconstruction and behavioral prediction are both independently measured.

Run on independent seeds and report all outcomes; keep native parity, patch norm, input overlap and readout disturbance explicit. The two-stage synthetic linear model is distinct from a trained transformer and from the sealed ARG-03-N tests.
