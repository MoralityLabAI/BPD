# AMM-07 engineering amendment: numeric replay of global ridge memory

Registered **after the first training-only run, before any development root**.
The original source-only CI trained 64 roots and committed memory lock
`3c7322f`. An independent hosted replay used the same source hashes and
regenerated exactly the same **candidate hit counts** (3088,1787) and the
same per-root 128-case binary candidate-bit digest
`5826fbe0eda0b3c33f77874420b447702164fe8f3b8d5efdfaeab4452d5d913d`.
The raw floating-point model features varied by roughly 1e-6 between
CPU runs, creating approximately 1e-7 variation in the 11 fitted
ridge coefficients. As the original training-memory SHA covered every
unrounded float (and all training-row metrics), its whole-JSON digest
differed across CPU workers.

**No AMM-07 development or confirmation roots were trained or scored.**
The old failing and passing training receipts remain in the source-locked
CI archives. This is a numeric engineering correction, with the original
scientific hypothesis, root partitions, controls, training data,
calibration budgets, outcome labels and EvalMath gates unchanged.

Before any development generation, apply the following frozen repair:
1. Quantize **only the eleven fitted global ridge coefficients** to
   exactly four decimal places using Python `round(float,4)`. This
   quantization is part of the trained *policy* and must be applied
   identically on every replay.
2. Define the immutable memory ID as SHA256 of the compact training
   policy/metadata, source SHA256 and the exact train-root **candidate
   bit digest**. Archive all raw training feature floats separately,
   without requiring bit-exact cross-worker floating-point metrics.
3. Reject a memory whose full training source rows do not regenerate
   the locked binary outcomes/roots. Retain the entire unfiltered
   original training corpus. A new immutable policy SHA and source SHA
   must be committed before *first development* scoring.
4. The development receipt replay compares source, independent roots,
   rootwise candidate/policy/duplicate-label binary records, and root
   classifier/fidelity decisions. Full floating-point raw JSON receipts
   are archived with their own original-run checksums and may vary at
   final machine precision on other CPU hardware.

Four decimal places are conservative relative to the observed
maximum coefficient discrepancy (~1.5e-7). This amendment is based
only on repeated **training** evidence. It cannot authorize any
change to a development or confirmation endpoint, success threshold
or root population. If the repaired policy still fails independent
training replay, the next study remains blocked.
