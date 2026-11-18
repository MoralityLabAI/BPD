# BPD-AMM-11: transferred causal memory against cosine alignment

[PROTOCOL.md](PROTOCOL.md) registered **before new AMM11 training or model source**, on November 18, 2026. It discloses the exposed AMM10 result, in which a simple physical alignment rule achieved 38.51% success across exposed root-heldout training cases, approaching the 18-feature ridge's 40.14%.

AMM11 tests whether the learned global BPD strategy gains a **mechanistically meaningful +8pp** absolute success probability against this analytical rule on **64 newly trained** six-dimensional rank-three three-pass recurrent models. The target planning effect is +12pp. The source-defined two-delay donor repair endpoint and six allowed canonical component masks are held fixed from the prior task. A proposed experimental mechanism is rejected if its measured root-level effect lies inside the 95/95 independent-root matched cosine-repeatability floor or below the scientific minimum, regardless of nominal sign-test significance.

The same fresh 64-TRAIN-model ×128-case/root corpus trains both full18 and compact10 global ridge memories, and selects the strongest fixed admissible pooled mask. New DEVELOPMENT models each have 16 noisy calibration donor cases (96 candidate labels), 512 primary donor pairs, and 512 disjoint repeatability donor pairs. The policy arms are full_memory, pure cosine, compact_memory, pooled, local_ridge, memory_plus_local, max_cardinality and hashed random mask. Equal physical-dose random-dark shams, no-patch and full donor equal-norm visible ceiling are retained.

**The primary registered comparison is full_memory minus cosine**. Five one-sided independent-root signed contrasts receive Holm adjustment; 2,000 root-cluster bootstrap intervals are descriptive or prospective based on population status. The exact CPU backend, runtime package/binary SHAs, optimizer and root namespace are pinned, and the source checks original training-policy identity before every development run.

## Hosted CPU reproduction

Run within the accepted Python 3.12.12 / Torch 2.10.0+cpu / NumPy 2.2.6 runtime with the `RUNTIME.json` environment values set *before Python starts*:

```bash
python -m unittest mechinterp.bpd.experiments.amm11_cosine_transfer.test_experiment -v

python -m mechinterp.bpd.experiments.amm11_cosine_transfer.experiment \
  --phase training --out /tmp/amm11_training.json

# Commit results/TRAINING_LOCK.json from compact(training) BEFORE development
python -m mechinterp.bpd.experiments.amm11_cosine_transfer.experiment \
  --phase development --memory /tmp/amm11_training.json \
  --out /tmp/amm11_development.json
```

The dedicated [staged workflow](../../../../.github/workflows/bpd-amm11-cosine-transfer.yml) performs engineering checks, 64-root training and a policy lock. A missing or drifting training lock forbids DEVELOPMENT generation. Subsequent independent-root development is source-locked and evaluated using the corrected pinned EvalMath preflight at `MoralityLabAI/EvalMath@6d9c28dbe8e4c42b3fba76ae7adb84aed9f0921a`, with a 95th-quantile/95%-confidence sham tolerance calculation. The full original per-case JSON is archived for an independent replay before any report.

AMM11 contains **no confirmation runner**. The 128 AMM11 reserved confirmation identities and every earlier AMM03–09R1 or ARG-03-N protected population remain sealed. The evidence is synthetic rank-three linear recurrent BPD; no inference about nonlinear TRMs, model-generated reasoning or deployed agent memory follows directly.
