# Direct dense reference, 26 November 2026

The first COG-04 run used the full SVD factorization as its floating-point
reference. Its relative weight error was 7.85e-8, while two recorded ordinary
readout decisions changed the usefulness class against direct dense arithmetic.
Tied or nearly tied scores make the discrete action sensitive to roundoff.

The experiment retains that arm as `rank12_32bit`. A new
`full_rank_float32` control uses V = I and U = W transpose, so its transmitted
activations are the normalized input features themselves. This control uses
the same BPD component API, byte format, nonlinear readout and public action
support. It supplies the direct dense reference named in the original protocol.
All previous rank/bit arms remain in the report. The resulting family contains
23 arms and 22 comparisons to the reference.

This change concerns arithmetic parity and transport accounting. Model training,
rank choices, bit depths, training seed and verifier budget stay as registered.
The public readout has now been inspected; the report keeps an exploratory
engineering evidence label, and records the eight roots exposed to the earlier
four-root smoke. The source contains no selected best arm or readout-dependent
threshold.
