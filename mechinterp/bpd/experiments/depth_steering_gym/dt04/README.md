# BPD-DT04: 59-Root Competent Depth-Transport & Protected Steering Gym

[Prospective Protocol](PROTOCOL_DT04.md) · [Design Preflight](DESIGN_PREFLIGHT.json) · [Tasks](tasks.py) · [Causal & BPD Factors](causal.py) · [7M Hardware Audit](hardware_7m_audit.py) · [Runner](runner.py) · [Auditor & Aggregator](aggregate.py) · [Read-only Viewer](ui_server.py)

Additive Stage-1 study extending [`depth_steering_gym`](../README.md) while keeping every historical DT01, DT02-A, DT03 and UI file untouched.

## Quick Verification & Viewer

From the BPD repository root:

```powershell
# 1. Verify DT04 execution lock + historical DT01/DT02/DT03 locks
python -m mechinterp.bpd.experiments.depth_steering_gym.dt04.verify_source

# 2. Run deterministic contract and viewer tests (no training or ARG-03-N access)
python -m unittest mechinterp.bpd.experiments.depth_steering_gym.dt04.test_dt04 `
                   mechinterp.bpd.experiments.depth_steering_gym.dt04.test_dt04_ui -v

# 3. Launch read-only DT04 + DT02-A comparison dashboard on loopback (default 127.0.0.1:8766)
python -m mechinterp.bpd.experiments.depth_steering_gym.dt04.ui_server --open
```

## Registered Overnight Execution Pipeline

```powershell
# Excluded engineering smoke (root 2603999)
python -m mechinterp.bpd.experiments.depth_steering_gym.dt04.runner --mode smoke --out-dir mechinterp/bpd/experiments/depth_steering_gym/dt04/results_smoke

# 59-root calibrated development run (roots 2604000..2604058)
python -m mechinterp.bpd.experiments.depth_steering_gym.dt04.runner --mode dev --out-dir mechinterp/bpd/experiments/depth_steering_gym/dt04/results

# Empirical RTX 5080 memory envelope & offline SHA-verified 7M TRM horizon/dark-kick audit
python -m mechinterp.bpd.experiments.depth_steering_gym.dt04.runner --mode hardware-7m --out-dir mechinterp/bpd/experiments/depth_steering_gym/dt04/results

# Independent receipt audit, 16,384 whole-root bootstrap, and pinned EvalMath 95/95 preflight
python -m mechinterp.bpd.experiments.depth_steering_gym.dt04.aggregate --out-dir mechinterp/bpd/experiments/depth_steering_gym/dt04/results
```
