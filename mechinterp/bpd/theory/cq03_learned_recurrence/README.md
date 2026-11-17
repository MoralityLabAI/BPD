# BPD-CQ-03: learned recurrence and causal observable masks

A controlled two-stage CPU study of delayed readout in a **trained physical recurrence**. An independently supervised rank-two transition learns `h0=(0,0,a,b) -> y1=a -> y2=b`; native BPD read/write factors reconstruct its two-step execution with a shared physical pair. The intervention question is whether a selected component can transfer donor b while holding recipient a fixed at the current readout. The correct donor b is defined by the mathematical task, not the component decomposition.

The `PROTOCOL.md` is committed before the prospective cohort. The code has a no-repository pure Torch backend for independent replay and an optional native `BPDModel/TiedComponentLinear` backend used by dedicated GitHub Actions. The native core uses a learned W, both hard component masks on at each recurrent pass, and `delta_mask=0`. Thus it tests learned physical-factor causal directions and the gauge problem, **not** trainable pass gates or a sparse routing win.

From the BPD checkout root:

```sh
python -m unittest discover -s mechinterp/bpd/theory/cq03_learned_recurrence/tests -v
python -m mechinterp.bpd.theory.cq03_learned_recurrence.experiment --backend independent --out mechinterp/bpd/theory/cq03_learned_recurrence/results/cq03_independent.json
python -m mechinterp.bpd.theory.cq03_learned_recurrence.experiment --backend native --out mechinterp/bpd/theory/cq03_learned_recurrence/results/cq03_native.json
```

The independent backend can also be invoked as `python path/to/experiment.py` in a clean sandbox without the BPD package. The unit tests load it by file path and skip native parity only when the BPD package cannot be imported.

The full-basis gauge is exact at the source algebra level; individual component interventions can vary under a shear. Report native outcome, current-dark projection, random dark, other component and full-state separately, including norm matching and immediate-readout disturbance. A positive would be limited to the specifically trained linear synthetic task.
