# AMM-09R reproducibility engineering results

The registered pinned32 recipe passed exact replay on two independent hosted
workers reporting AMD EPYC 9V74 and AMD EPYC 7763 processors. Each worker ran
two separate processes. Every recorded pinned32 numerical payload matched,
including initial parameters, trained weights, canonical factors, assay arrays,
candidate bits, and the fitted engineering policy.

All four historical replays also reproduced the complete original AMM09
64-root training JSON payload, including every recorded weight hash, all
49,152 six-mask training labels, and the original policy hash. The prior failed
replay remains an archived observation; this run establishes exact reproduction
on the two recorded workers and leaves its precise historical cause unisolated.

## Registered runs

| Item | Receipt |
| --- | --- |
| Engineering registration | `ffc7143e3b597d840c5000a94a45b45d964931cf` |
| Executed source | `6fecc60dbb083230847819ee04b442b9aa9015c1` |
| Hosted run | [44291185512](https://github.com/MoralityLabAI/BPD/actions/runs/44291185512) |
| Python / torch / numpy | 3.12.12 / 2.10.0+cpu / 2.2.6 |
| Independent engineering roots | 8, with 128 training-fixture and 512 separate assay pairs each |
| Worker/process coverage | 2 workers × 2 processes for each registered configuration |
| Complete original training replays | 4, each containing the same 64 already exposed roots |
| Confirmatory roots accessed | 0 |

[ENGINEERING_ACCEPTANCE.json](results/ENGINEERING_ACCEPTANCE.json) records all
nine passing acceptance gates. [RECEIPTS.json](results/RECEIPTS.json) binds each
of the twelve raw receipts by SHA256, together with the hosted artifact IDs.
[DIAGNOSTICS.json](results/DIAGNOSTICS.json) retains the observed runtime details
and the comparison between the two registered configurations.

Both baseline32 and pinned32 were internally exact across the observed workers.
The engineering comparison therefore did not reproduce the earlier uncontrolled
host failure. The predeclared pinned32 recipe was admitted by exact equality
and physical checks, with no scientific success-rate selection.

## What the runtime change affected

Comparing baseline32 with pinned32 on the same engineering roots changed all
four initialization tensors on every root. The largest element difference was
5.364418029785156e-7. The hashed source minibatch streams stayed identical.
Six of 6,144 six-mask training-fixture labels and 15 of 24,576 separate assay
labels changed. These are comparisons between numerical recipes; the fixed
pinned32 recipe had zero changes across workers or process repeats.

The initialization difference supports the dispatch mechanism described in
[NUMERICS_NOTES.md](NUMERICS_NOTES.md). Several runtime settings changed together,
so this observation does not identify a single cause of the historical incident.

## Physical checks

All eight engineering roots passed every physical gate. Maximum active-dose
error was 2.384185791015625e-7, below the registered 1e-5 bound. Immediate readout
displacement and native BPD wrapper parity error were both zero. Maximum full
GL(3) operator error was 4.440892098500626e-16; maximum recanonicalized component
error was 1.7763568394002505e-15.

The full donor-state control has the same L2 displacement D and can change the
immediate readout. Its role is an unrestricted visible control. The original
AMM09 protocol's larger-dose wording is preserved in that historical record;
the new audit records the physical quantities directly.

## Scientific scope

These are excluded engineering observations. Eight model roots cannot calibrate
a 95/95 scientific sham envelope, and repeated processes add no independent
scientific units. No scientific effect or p-value is inferred from this fixture.
The original AMM09 development audit and pinned EvalMath output were independently
rerun from the archived files and matched their committed results exactly.

A separate AMM09R1 protocol registers fresh training and development streams
under the accepted numerical recipe. Its training policy must be committed
before development. Every historical protected cohort stays sealed, and the
fresh study reserves a separate sealed confirmation population.
