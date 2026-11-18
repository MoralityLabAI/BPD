# BPD-DT01: Sudoku / ARC-1 / hidden-ending storyworld depth steering

[Protocol](PROTOCOL.md) · [Task generators](worlds.py) · [Protected causal subspace evaluator](steering.py) · [Training and native TRM bridge](training.py) · [Source audit](audit.py)

A usable substrate for BPD steering experiments motivated by [CGC-02](https://github.com/MoralityLabAI/Compactification-Theory/pull/247), where a task-selected physical subspace recovered donor behavior at supervised depth and failed badly at a deeper unseen loop. Training uses 2/4/6 loops and a **separate** evaluation uses 8/12. Native TRM's H/L commits remain observable through the existing [`TRMBPDAdapter`](../../adapters/trm.py).

CPU smoke:

```bash
python -m unittest discover -s mechinterp/bpd/experiments/depth_steering_gym -p 'test_*.py' -v
python -m mechinterp.bpd.experiments.depth_steering_gym.audit --seed 301
```

Task interfaces: deterministic exact 4x4 Sudoku oracle, 9x9 legality/solve verification, local ARC-AGI-1 JSON parsing with test answers **omitted** from the model view, explicitly **synthetic** ARC-style transformation rules on disjoint train/heldout program families, and an exact short storyworld with hidden terminal-only endings. The host can grade gold labels after inference. Official ARC-1 files and pretrained TRM checkpoints are supplied separately; the smoke test does not download or score them.

The steering code uses a physical metric and an actual full normalized observer to select candidate causal subspaces on development states. It provides the privileged signed task-Jacobian-and-dose matched null and a strict protected-readout checker, which are necessary controls for the failure seen in CGC-02. It is **not a claimed trained BPD steering success**. Source-only tests exercise a rotating recurrent mechanism that makes an early-depth direction useless at later depth.
