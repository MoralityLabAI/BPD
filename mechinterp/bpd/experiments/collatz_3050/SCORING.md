# How the gym scores a run

The endpoint measures whether a prefix intervention steers the final answer
toward the effect donor's answer. Each run reports `PILOT_ONLY`. The cases come
from public development parameters, and eligibility depends on the checkpoint's
clean task performance. [Case construction](cases.py) and [the scorer](scoring.py)
define the executable contract.

## Answer probabilities and task competence

The runner scores two declared answer tokens, A and B, at the final prompt
position after the second native pass. Given their logits, it reports

\[
p(A)=\frac{\exp z_A}{\exp z_A+\exp z_B},\qquad p(B)=1-p(A).
\]

These probabilities condition on choosing one of the two answers. They describe
the relative preference between those tokens; the vocabulary-wide probability
mass assigned to the pair remains outside this measure. An exact tie fails the
competence check.

Every case contains a recipient, an effect donor with the opposite exact future
branch, and a sham donor with the recipient's future branch. All share the
parameter and current arithmetic quotient/branch. The default case at `d=13`
serves as a calibration across basins. The remaining default cases use donors
from the same basin, including the exact trace sham at `d=73`. The summary
retains these groups in `basin_strata`.

All three clean answers must be correct for the case to enter the endpoint.
`clean_task` counts each role among completed records. `competence_coverage`
reports how many completed cases meet that joint requirement and gives both the
completed-case and planned-case denominators. Errors retain partial scores in
the raw records for diagnosis. An incomplete case stays outside the estimate.

The arithmetic tier supplies the rule, seed, and horizon, making this a bounded
computation task. The trace tier supplies exact trajectory evidence and measures
trace readout. Each tier has its own manifest.

## Current readout and intervention admission

[The intervention](interventions.py) edits normalized prefix positions after the
first native pass. It preserves the final-position vector exactly. With the
same vocabulary head, equality of that vector implies equality of every current
head logit. The runner also compares the patched first-pass vector against the
clean recipient. `protected_readout_max_abs=0` records this structural
certificate; the scorer checks the certificate supplied by the runner. The
second pass can then use the edited prefix to change its final answer.

A completed case requires identity, effect, sham, random, and full-donor-prefix
arms. Every arm must have explicit admission and a preserved current readout.
The identity dose is zero, with final answer probabilities agreeing within
`1e-6`. Effect, sham, and random use positive delivered L2 doses. Their largest
and smallest measured norms must satisfy

\[
D_{\max}-D_{\min}\leq 0.02D_{\max}+10^{-6}.
\]

This prospective engineering allowance uses the displacement after native-dtype
rounding. The intervention aims each delivered norm within 1% of the common
requested dose, so the scorer allows 2% between arms plus the absolute allowance.
The full-donor-prefix arm has its own raw dose and serves as a separate reference.
All edited boundaries are constructed states; natural reachability needs
separate evidence.

## Paired effect and parameter roots

Let \(y_E\) be the effect donor's answer, and let \(p_R(y_E)\) be the clean
recipient's conditional answer probability. For arm \(a\), define

\[
s_a=p_a(y_E)-p_R(y_E),\qquad
C=s_{\mathrm{effect}}-
\max\{|s_{\mathrm{sham}}|,|s_{\mathrm{random}}|\}.
\]

The summary keeps every shift, the observed control envelope, and \(C\).
`donor_recovery` records whether an arm selects the effect donor's answer.
`recipient_task_correct` checks that same arm against the recipient's original
answer. Since these answers differ, successful donor steering can make the
recipient answer incorrect.

A root is the source parameter \(d\). Repeated horizons and cases share this
root. Every planned case in a root must pass competence and instrument checks
before that root contributes. The scorer averages cases within a complete root,
then gives each complete root equal weight. `n_units` counts all planned roots;
`n_eligible_units` counts roots admitted to that mean. Each failure or unattempted
case remains in the summary with its reason.

## Development limits

The declared scientific effect scale is `0.05` conditional probability units,
or five percentage points. The output reports it alongside the measured effect.
The observed sham/random envelope describes the controls in this run. A
calibrated sham floor remains `null`.

For a sample maximum to cover a population's 95th percentile with 95% confidence,
the elementary calculation is \(1-0.95^n\geq0.95\), requiring \(n\geq59\)
independent draws from the intended root-level null distribution. The current
deterministic development selection provides paired clusters for inspection.
Calibrating that distribution requires a separately designed pilot. Resolution,
power, and confirmation on fresh roots remain open. The report's scope stays
conditional on the tested checkpoint, prompts, intervention site, and declared
controls.
