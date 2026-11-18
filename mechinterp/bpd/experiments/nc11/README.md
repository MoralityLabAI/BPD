# NC-11: causal-Jacobian cross-model projector alignment

**Prospective first commit:** [PROTOCOL.md](PROTOCOL.md), `60b65642c03cb3064912aa3929582479aa21dd20` before implementation or any NC-11 model training.

NC-10's ordinary activation Procrustes gave a **negative** first-eight-pair causal transfer result. This study tests whether using **model-output Jacobians of a limited 128-unlabeled-case B calibration sample** yields a more useful A-to-B alignment. The additional target gradient queries are counted and a **B-only rank-four gradient projector fitted to the identical 128 cases** is an equal-target-budget control.

Two independent dense 20D four-pass nonlinear recurrent models learn related externally defined tasks. The shared branch depends on `x0` when `x4>0`; private branches depend on `x1` or `x2` when `x4<=0`. Each model trains separately for 384 updates on 2048 separate inputs. Source A projector discovery gets 1024 unlabeled model-score gradient examples. The source-only alignment uses the first 128 positive-context inputs from an independent 512-case pool; eight local physical retained-write-RMS-matched sham projectors are admitted from 2048 fixed proposals on an **additional dose-only input split**. True task labels are accessed only on a disjoint 1024-case final evaluation split.

The four frozen directional gates compare **Jacobian transport** to activation transport, permuted Jacobian transport, the matched-budget B-only projector and local physical shams. The noninferiority criterion is upper paired-root interval ≤0.05 nat versus B's 128-gradient projector. The other thresholds are positive lower endpoints of 0.01, 0.01 and 0.02 nats, respectively, using **98.75% familywise root-cluster bootstrap intervals** across four comparisons.

Engineering model pairs: **roots 1100–1107**, distinct from all earlier NC series. Excluded source smoke root: **1099**. Every original negative or ineligible root is retained with its original causal outcomes, calibration diagnostics, source lock and zero-head/gauge controls.

Run from repository root:

```bash
python -m pytest -q mechinterp/bpd/experiments/nc11/test_experiment.py
python -m mechinterp.bpd.experiments.nc11.experiment --root 1099 --steps 384 --discovery-size 256 --calibration-pool-size 512 --dose-size 256 --evaluation-size 256 --proposals 256 --out /tmp/nc11_smoke.json
```

For registered independent development, run `--root 1100` through `--root 1107` each with 1024 discovery/dose/evaluation cases, fixed 512-case calibration pool and 2048 shams, then:

```bash
python -m mechinterp.bpd.experiments.nc11.aggregate --directory /tmp/nc11_development --out /tmp/nc11_development/NC11_ORIGINAL_SUMMARY.json
```

The GitHub Actions workflow checks `EXECUTE_LOCK.json` before the source smoke and all 16 trained development models. Only a passed source gate permits the eight-pair job. Green CI does not change the meaning of a scientific `ENGINEERING_NULL_OR_INELIGIBLE` outcome. **Confirmation remains sealed**, as do protected unrelated datasets.
