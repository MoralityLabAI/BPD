# AMM-01: agent-memory transfer on a tied recurrent causal-probe task

**Registered 2026-11-18. Status: source-only prospective protocol.**
The independent unit is a freshly drawn linear recurrent world, including its
latent mechanism sign and coefficient. No ARG-03-N data, pretrained model, LLM,
trained BPD decomposition, or external relay is involved.

## Question and claim boundary

Do training-only records of which *probe horizon* worked for an opaque
task family improve independent agents' ability to identify the sign of a
future causal effect, given exactly one equal-dose noisy causal probe per case?
The agents are deterministic policy functions, not language models. This is a
controlled *strategy-memory* positive instrument, not evidence of BPD discovery
in real networks, real-world agent learning, or multi-agent coordination.

## Frozen world and observation model

Each independent world has a 4D shared transition `h <- W h`, current
readout `q(h)=h[0]`, and physically fixed probe
`d=(0,+1/sqrt(2),-1/sqrt(2),0)`, so `q(d)=0`, `||d||_2=1`.
The future effect `q(W^t d)` is measured at either t=1 or t=2,
with additive independent Gaussian noise N(0, 0.55^2). Querying either
horizon uses the exact same dose and one readout. The binary ground truth
is the independent hidden sign s in {-1,+1}, with world-specific gain
g ~ Uniform[0.45,0.85]. No label or W matrix is visible to the policy.

The generator has two equiprobable opaque family keys:
- `wren`: W[0,1]=s*g, W[0,2]=-s*g; all other W entries zero.
  t=1 has the signal and t=2 is uninformative.
- `basalt`: W[0,1]=s*0.035, W[0,2]=-s*0.035,
  W[0,3]=1, W[3,1]=s*g, W[3,2]=-s*g; all other W entries zero.
  t=2 has the strong signal, t=1 has a weak signal.

Each W has rank <=2 and is weight-tied across passes. Policy sees only
the opaque family key and training-derived routing records. It never sees
s, g, W, the ground-truth effect, or target test outcomes. Correctness means
the sign of the one noisy observed effect equals s; exactly zero votes -1.
Use the same pre-generated t=1/t=2 noise for arms choosing the same horizon.

## Fixed partitions and budgets

Deterministic phase-separated SHA256-seeded PRNG worlds
`AMM01/{phase}/{index}`: training indices 0..63;
development indices 0..63; confirmation indices 0..511. For each world,
exactly 256 independent noisy one-query *cases* are evaluated. Thus
64 independent training worlds, 64 development/sham worlds and 512
confirmation worlds; within-root cases are repeated measurements.
Family assignment is index parity, 50/50 in each partition; latent sign,
gain, and Gaussian observations come from phase-isolated RNG streams.
All outcomes, including failures, remain in raw receipts.

Training worlds may be queried at both horizons to learn a family-keyed
routing table. Freeze one best horizon per family by training accuracy;
ties select t=1. The pooled baseline learns one globally best horizon from
the *same training queries*, also with t=1 tie-break. It sees family keys
but does not have family-keyed memory. Two other preregistered controls
are the exact swapped family table (misaddressed memory) and a
deterministically seeded 50/50 horizon-randomized policy. A same-horizon
pooled-policy noise resample per development world calibrates repeated-
measurement sham variation. All five arms use one probe per scored case.

## Locked comparisons and resolution

Primary effect = mean over independent confirmation roots of the
family-keyed memory policy's accuracy minus pooled-policy accuracy.
Two secondary comparisons in the single multiplicity family are
memory minus swapped-family memory and memory minus random horizon.
Use one-sided exact sign tests on nonzero root-level paired effects,
Holm adjustment across all three comparisons, and 2000 seeded paired
root bootstrap replicates for each mean effect's 95% percentile CI.
Use *all* 512 roots, including zero effects and failures.

Scientific effect floor = +0.15 accuracy units; forecast target effect
+0.25; measurement resolution = 1/256 root accuracy; alpha=.05,
three comparisons. On independent development worlds, run the exact
EvalMath `skills/evalmath-mechinterp-power/scripts/preflight.py --strict`
using per-root memory-minus-pooled effects and absolute paired
noise-resampling sham differences, q=.95 and confidence=.95. 64
independent worlds suffice for the sample-maximum 95/95 tolerance rule.
If strict preflight fails, confirmation is blocked and the project stays
development-only. Require the primary bootstrap lower confidence
bound to exceed max(0.15, the calibrated 95/95 sham floor) and Holm p<.05
to label a prospective synthetic transfer result.

## Source ordering, anti-leakage and reporting

Commit this protocol *before* generating results, commit the implementation
and tests *before* opening confirmation, then run development preflight
and only if READY_FOR_CONFIRMATION score fresh confirmation worlds.
Freeze the learned table using training only. Source-digest all code,
protocol, and memory records; retain all scored roots and binary
per-case outcomes. A train/dev/confirmation source ID collision is a
hard error. Never use confirmation results for model, policy or
hyperparameter selection. Report unit, case count, matched dose,
minimum effect, calibrated sham floor, planned comparisons, CI and
p-values, and negative controls. Store any executed result as synthetic
evidence with the exact source commit.

## Buzz / Nostr handoff

Export a *small, unsigned* NIP-29 kind-9 channel-message template that
contains experiment ID, GitHub commit/PR URL, receipt digest, evidence
class and schema version, tagged with a caller-supplied channel UUID.
It contains no truth labels, private chat, keys or full source memories.
An authenticated Buzz client must add Nostr id/pubkey/sig and publish it.
No real Buzz relay is provisioned or contacted by this protocol.
