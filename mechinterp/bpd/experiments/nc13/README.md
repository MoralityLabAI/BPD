# NC13: source context and delivered displacement

The [protocol](PROTOCOL.md) was committed first at `069b355e8f37ec780dc6c46b82176b6f32f8e39b`, before implementation and new model data. It reserves sixteen paired roots at eight target gradient inputs, comparing mixed/shared source discovery under retained-energy and actual closed-loop-displacement calibration.

The mixed-source transfer at gain1 supplies one common reference per pair. Source conditioning preserves the 1024-row discovery budget and raw-gradient convention. Displacement matching can use up to451 additional B dose-pool rollouts per pair; all target forward access is recorded separately from gradient access. Its bounded search leaves failed matches visible.

`DESIGN_PREFLIGHT.json` uses exposed NC12 effects only as prior design context. Its `REDESIGN_OR_PILOT_ONLY` output permits this engineering diagnostic and supplies no confirmation readiness for NC13. The [cross-thread bridge](CROSS_THREAD_BRIDGE.md) links the study to DT03 and the existing LOOPSCOPE12 response controls.

## Source checks

```sh
python -m mechinterp.bpd.experiments.nc13.verify_source
python -m pytest mechinterp/bpd/experiments/nc13 -q
python -m mechinterp.bpd.experiments.nc13.experiment --root 2601299 --outdir /tmp/nc13-excluded-smoke
```

Development roots require the complete execution lock and accepted runtime. Each output root is reserved before training, and existing reservations prevent replacement. The workflow enables the original development job only for the initial pull-request opening. Later documentation updates run source checks while retaining the first development archive.
