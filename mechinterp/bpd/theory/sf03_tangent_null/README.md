# BPD-SF03: tangent-matched dark-fiber controls

[Protocol](PROTOCOL.md) | [SF02 parent result](../sf02_secret_circuit/RESULTS.md) | [Spectral Faithfulness](https://github.com/idolum-ai/spectral-faithfulness)

SF02 showed strong source-conditioned transfer from a six-factor BPD donor write, while a physical factor fit failed on one root. SF03 asks the stricter question: does the factor direction contribute to **future output** beyond its local Jacobian magnitude and sign?

The signed-gradient sham randomizes a dark-state intervention's Jacobian-orthogonal component while keeping its norm and exact first-order signed output effect fixed. Unsigned-gradient controls randomize the sensitivity sign. The experiment also measures the sensitivity of each successive readout to the authorized and other user's private input.

Use excluded engineering roots only for source validation:

```bash
python -m unittest discover -s mechinterp/bpd/theory/sf03_tangent_null/tests -v
python -m mechinterp.bpd.theory.sf03_tangent_null.experiment --phase excluded
```

Prospective calibration: 59 new roots 4101..4159, followed by a one-time threshold lock from `G_null` values. Confirmation: 32 new roots 5101..5132. This maintains separate source roots while using the same synthetic task family.

The signed-gradient null deliberately uses the privileged BPD tangent effect. Matching this control does **not** demonstrate a new secret-bearing BPD mechanism. All results and failures will be reported by independent teacher root, including the original SF02 factor reconstruction bound.
