# BPD-SF05B Stage 1: a nonprivileged recurrent correction policy

[Protocol](PROTOCOL.md) · [SF05B source-family benchmark](../sf05b_source_families/PROTOCOL_STAGE0.md) · [Issue #48](https://github.com/MoralityLabAI/BPD/issues/48)

The new 32D shared recurrent teacher processes three-user source-tagged scalar writes. Its first coordinate is the actual visible readout. The teacher and a separate 32-component physical BPD factor pair train exclusively on F_train histories of 2–4 writes.

The correction model receives only the prequery hidden state, a proposed physical BPD write, the known public query and current readout. Its training labels come from the privileged SF05-A2 finite Newton oracle **on F_train**. Its inference method has no access to any teacher forward call, future public readout, derivative, donor answer or oracle label. Unit tests poison those forbidden fields and verify unchanged output.

After the Stage 1 source lock, the excluded engineering roots 901/902 are evaluated on separate F_train records and structurally heldout 5–6-write histories. The full physical write, output-SVD rank 24, teacher physical matrix, exact dark donor state, unchanged state, oracle-corrected arms, learned corrections and same-policy random/signed-gradient controls are recorded separately. A trained teacher that fails the heldout source-binding competence gate is reported as such.

Run from BPD root with PyTorch installed:

\`\`\`bash
python -m unittest discover -s mechinterp/bpd/theory/sf05b_stage1/tests -v
python -m mechinterp.bpd.theory.sf05b_stage1.native_parity
python -m mechinterp.bpd.theory.sf05b_stage1.evaluate --roots 901 902 --out /tmp/sf05b_stage1.json
\`\`\`

This is an excluded-root engineering and OOD diagnostics campaign. The 59 new calibration roots 6101–6159 and 32 confirmation roots 7101–7132 remain **reserved and unexecuted**. Source-family disjointness means history length/templates, not an unseen source-ID alphabet, nor language-model privacy. ARG-03-N's 168-case test remains sealed.
