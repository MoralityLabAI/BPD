# BPD-CQ-02: trained-factor delayed observability

The pre-data source-locked protocol is in [PROTOCOL.md](PROTOCOL.md); [THEORY.md](THEORY.md) distinguishes linear observability from masked BPD behavior. [RESULTS.md](RESULTS.md) is the post-registration CPU result. This is an **engineered linear 4D positive control** testing the causal-observation instrument with trainable shared sparse factors. The transition T is an explicit fixed channel swap. It is not evidence of delayed reasoning in a trained Transformer.

After fitting 2 components with exact hard top-one gate across 4 visits, measure how the observer C=(0,1,0,0) sees a native donor component swap before and after the fixed channel-swapping transport. Both the native patch and a random patch in the same 3D instantaneous readout fiber are tested at *identical per-pair norms*; identity transport and mixed-source inputs are negative controls. The two-axis observability matrix is evaluated on the learned BPD write factors.

From `MoralityLabAI/BPD` root:

```bash
python -m unittest discover -s mechinterp/bpd/theory/cq02_trained_observer/tests -v
python -m mechinterp.bpd.theory.cq02_trained_observer.pilot --backend independent --out /tmp/cq02-independent.json
python -m mechinterp.bpd.theory.cq02_trained_observer.pilot --backend native --out /tmp/cq02-native.json
```

The second command requires PyTorch; the third additionally executes **the actual repository `BPDModel` and `TiedComponentLinear`**, checking native versus explicit component-output parity on independent calibration vectors. The native result is a separately generated receipt; its observations must be reported separately until hosted CI finishes. Both backends share the frozen source/seed/optimizer contract. No sealed ARG-03-N inputs, private SCM weights, or remote model calls are used. A completed independent reference is committed to `results/cq02_cpu_reference.json`.

The preregistered mask-routing success is intentionally easier than BPD-PD-03's 24-seed superiority test: the input law has two disjoint modes with exact one-hot support. It validates the causal crosswalk before the harder native recurrent model assay.
