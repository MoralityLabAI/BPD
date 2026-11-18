# CQ-08 EvalMath mechanistic-resolution preflight: before 59 sham roots

The original [EvalMath mechanistic-resolution skill](https://github.com/MoralityLabAI/EvalMath/tree/main/skills/evalmath-mechinterp-power) requires the root (not individual samples) to be the independent statistical unit and a 95th-percentile/95%-confidence nonparametric sham threshold from independently generated calibration worlds. The pre-data design is frozen in `PROTOCOL.md`. Local `power.py` implements the same exact 59-root binomial order-statistic tolerance rule with separate 8-root development effect worlds, avoiding contamination of the effect pilot by 59 zero-effect sham rows.

**Current status: `REDESIGN_OR_PILOT_ONLY`.** The eight development roots cover 1,024 repeated donor pairs, with 1003 source-matched; their observed mean selected-vs-random1 damage difference is +0.244140625 (root SD about 0.04937). Their 8 matched random-vs-random sham effects span 0 to 0.078125. Eight worlds cannot give a calibrated 95th-percentile/95%-confidence sham tolerance bound. The required 59 new sham worlds are separately preregistered and untouched at the time this note was written. The planned confirmation contains 32 new model roots and must remain unopened.

The scientific variation floor is at least +0.15 in correct-cell damage probability; the planning effect is +0.30; two comparisons form the registered Holm-adjusted family. Root-level power is calculated from development SD and the independent 59-world sham upper tolerance floor, taking the maximum of that floor, scientific minimum, and 1/128 measurement resolution. A READY result can authorize a future explicit release decision but is itself only a power-planning artifact.

Run before sham (expected exit code 2):

```
python -m mechinterp.bpd.theory.cq08_nuisance_mediation.power --development /tmp/cq08_development_independent.json --strict
```

After a write-once 59-root sham receipt exists, pass `--sham` and archive the preflight JSON. No confirmation code should run merely because the `power.py` decision becomes `READY_FOR_CONFIRMATION`.
