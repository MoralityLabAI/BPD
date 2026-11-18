# BPD-PD-14: Gaussian width correction versus orthogonal restart search

**Prospectively registered November 18, 2026.** The source roster and complete decision protocol were committed before any training on PD-14 scientific seeds.

PD-13 failed its strict clustered 16D mechanism-recovery gate (2/32 priority2048 and 2/32 lex2048). Its separately frozen post-hoc result showed that the original strong-margin Gaussian near-zero objective favors an explicit 45-degree mixed basis over the planted basis at sigma=0.15. PD-14 tests whether a narrower width, selected before fitting on new seeds, improves strict physical component identification.

## What the 160 prospective arm receipts will measure

A 32-seed, paired two-factor experiment uses 16 strong- and 16 weak-margin *new* rotations, the unchanged PD-12 clustered source, exactly 2048 training trajectories per seed, eight physical BPD held-out support families and a planted-Q oracle.

| Arm | Strong / weak σ | Training search | Candidate score evaluations |
|---|---|---|---:|
| **wide_single** | .15 / .05 | 9 sweeps, single original QR | 39,960 |
| **narrow_single** | .05 / .02 | 9 sweeps, same QR | 39,960 |
| wide_restart | .15 / .05 | 3 independent QR starts × 3 sweeps, best train mass | 39,960 |
| narrow_restart | .05 / .02 | same starts and selection, narrower σ | 39,960 |
| planted_Q_oracle | privileged | no search, physical positive control | 0 |

All learned arms see the same 8192 observed training vectors per seed. Pair order is lexicographic and every rotation uses the original global 37-angle grid. Restart selection uses the observed training Gaussian mass only. Held-out source Q, supports and correct component counts are withheld from all search arms.

A separate independent 1024-trajectory diagnostic draws fresh x from each source and compares Q against an explicitly defined Q-frame four-pair 45-degree remix at both registered widths. It is a privileged **objective consistency measurement**, never an optimizer arm.

## Frozen scientific decision

The single primary paired comparison concerns the **16 strong-margin** new sources: narrow_single versus wide_single at equal training samples, initial QR and 39,960 candidate-angle evaluations. Complete mechanism recovery retains the PD-12/13 mean read cosine, MSE and exact four-pass support thresholds.

The primary mechanism gate requires at least **8/16** narrow_single recoveries, at least **six net** narrow-only paired wins and **exact two-sided McNemar p ≤ 0.05**. An additional predeclared objective-consistency gate requires the original wide sigma to favor the fixed remix, and the narrow sigma to favor Q, on at least **15/16** independent strong-margin diagnostic pools. Study scientific PASS requires both gates; reports will always display them separately.

Restart effects and both weak-margin comparisons are secondary descriptive outcomes, with no extra retroactive significance claims.

## Files and evidence

- [CONFIG.json](CONFIG.json) fixes the 32 seeds, widths, budgets and decision thresholds.
- [PROTOCOL.md](PROTOCOL.md) fixes source, optimizer, restart tie-breaking, physical BPD evaluation and provenance.
- [experiment.py](experiment.py) implements four unlabelled searches, the physical evaluator, diagnostics and frozen analysis.
- [test_experiment.py](test_experiment.py) uses only engineering seed 4001 and synthetic summaries for preflight.
- [PairRestartBudget.lean](PairRestartBudget.lean) certifies 9 sweeps = 3×3 sweeps = 39,960 scores = 327,352,320 candidate-by-vector evaluations.
- The registered first-run workflow executes only upon committing EXECUTE_LOCK.json **after source and CI gates are green**. It archives full evidence independently of scientific PASS/FAIL.

The original PD-13 first-run evidence, scoring, and post-hoc diagnostic remain unchanged.
