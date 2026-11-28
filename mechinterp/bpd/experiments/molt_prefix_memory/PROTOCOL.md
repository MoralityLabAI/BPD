# MOLT-BPD-01: causal selection of prefix memory components

Development protocol, 2026-11-26. Motivation: [Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923).
This is an independent experiment design. Native BPD integration, trained-model measurements and
confirmation remain future work. Companion repositories use branch `research/molt-goblin-01-20261126`.
All existing NC09, AMM, ARG-03 and FSI-04 cohort restrictions remain in force.

## Question and first experiment

Can a frozen component selector recover useful information from earlier prefix states and improve
later verified answers at a controlled physical intervention dose? The first experiment uses a
competent frozen checkpoint, a fixed task population and a value-space intervention. Successful
development would support a later study of whether training with these interventions helps ordinary
inference when steering is disabled.

Begin with the smallest task that contains an auditable delayed dependency. A recipient encounters
a damaged intermediate result; an earlier, source-valid donor contains the missing information.
The verifier can check the final answer and the local repair independently. Vary delay and irrelevant
prefix distractors during TRAIN. Retain a clean copy of each episode to measure damage introduced
by the same intervention policy. Establish learned competence against declared trivial baselines
and retain an exact-policy ceiling as a separate structural reference. The corrupted episode
must leave repair headroom for the learned checkpoint before the mechanism can be measured.

## Physical implementation and prefix contract

For a frozen checkpoint `theta` and each allowed prefix position `j < i`, reconstruct the native
projected value with its complete BPD factorization:

```text
V_j = sum_c V[j,c] + V[j,residual]
V'_ij = sum_c g[i,j,c] V[j,c] + V[j,residual]
a_ij = softmax_over_allowed_prefix_and_null(q_i K_j / sqrt(d))
z'_i = sum_j a_ij V'_ij
```

The fixed null slot makes the first position well-defined. The intact baseline uses `g=1`.
Preserve projection biases and every undecomposed term through the native output write. Measure
parity at the complete head output and again at the recipient state after the write. Record dtype,
factorization hash, reconstruction error and the tolerance used to accept parity. Component sign
or permutation changes that represent the same factorization must preserve the intervention after
the gates are transformed consistently.

Value gating changes the memory content while preserving attention weights. A key intervention
changes the attention distribution and receives its own arm and dose calibration. Track the
physical recipient displacement `delta_h = h_intervened - h_intact`; an equal value-space norm
alone gives insufficient control over the post-attention write.
Here `h_intact` means the native execution of that same clean or corrupted case before adding
the memory intervention, so the corruption and intervention doses remain separately recorded.

The gate may read fixed inputs, fixed epoch state, public receipts and states in a declared strict
prefix dependency superset. Include its selection scores and the attention denominator in that
dependency audit. A gate that selects one donor after inspecting the whole prefix still depends on
that whole prefix. Current-loop computation inside a position can be local to that position; any
dependence on its previous-loop state belongs to the separate residual-recurrence arm.

Freeze the checkpoint, factorization, retrieval policy and stochastic choices within an evaluated
epoch. Pass-varying dropout, a mutable cache eviction policy, a loop counter or a changing global
receipt can change the transition map. Such changes require an explicit new epoch or a separately
analyzed recurrent system. The finite-depth theorem applies to the complete dependency graph.

## Component-bank record

Each bank entry needs enough provenance to reject an incompatible reuse:

| Field | Meaning and check |
| --- | --- |
| `checkpoint_digest`, `factorization_digest` | Native weights and component coordinates used to build the entry |
| `frame_id`, `problem_version`, `epoch_id`, `public_state_digest` | Fixed premises and external context for the forward calculation |
| `root_id`, `source_case`, `input_digest`, `prefix_position` | Source lineage and allowed causal location |
| `component_ids`, `residual_digest`, `dtype` | Reconstruction and numerical convention |
| `dependency_digest`, `verifier_digest`, `checker_version`, `verifier_receipt_id` | Inputs supporting the stored state and the source-validity decision |
| `bytes`, `build_time`, `last_used` | Measured resource cost; `last_used` cannot change scientific gate behavior within an epoch |

These names belong to the proposed adapter envelope. Content digests and human-readable
version labels have separate roles. Retracted premises require a new environment-owned frame
and revalidation of prior certificates; frame revisions retain the same root lineage and split.

Training roots supply component alignment, donor-selection features and gate fitting. DEVELOPMENT
roots supply independent sham calibration and resolution planning. Any policy choice influenced by
DEVELOPMENT must be disclosed as development selection before a fresh future confirmation population
is registered. Ordinary test, OOD and protected confirmation labels never enter fitting or the bank.
Store only public qualification receipts in DataGoblin-facing records.

## Comparison family

Use staged comparisons so each arm answers a defined question. Record every attempted arm,
including failed matches and failed source-fidelity checks.

| Stage | Arms | Decision supported |
| --- | --- | --- |
| Instrument | Native intact, full reconstruction, exact no-op | Whether the decomposition preserves the model being studied |
| Causal development | Frozen BPD gate, random equal-sparsity gate, equal physical displacement sham, signed future-Jacobian sham, source-shuffled donor, memory disabled | Whether donor-specific content helps beyond location, dose and local output sensitivity |
| Architecture | Prefix KV and residual recurrence with declared compute matching | Whether the memory architecture changes the result |
| Learning follow-up | Ordinary task training, task plus targeted steering, task plus matched sham steering | Whether any gain survives ordinary evaluation with steering disabled |

For a signed future-Jacobian sham, define the recipient write location and delayed readout first.
Let `J` be the Jacobian of that entire prespecified future output with respect to the recipient state.
Match `J delta_h`, including signs, within a frozen tolerance, as well as the actual write norm,
allowed intervention support and state radius. Preserve the complete output contract when building
the match. Record the matched residual and feasibility result for each candidate. A reduced-rank
or magnitude-only match is a separately named weaker control. An infeasible strong match blocks
the corresponding mechanistic claim and remains in the receipt.
The sham must also be a distinct feasible intervention. A match that reproduces the target
direction provides no donor-specific discrimination, even when its residual is zero.

Sham construction may use the frozen model's derivatives and declared public inputs. Its optimizer
must have the same search allowance across arms, with no access to protected answers. Pair donor
and sham cases within a root and retain their shared input and corruption identifiers.

## Outcomes and independent units

Define eligible clean cases using the intact source model and the independent verifier before
comparing intervention arms. Create a fixed paired corruption for each eligible case. Keep source
failures and unmatched interventions in the overall outcome ledger; report their frequency and the
prespecified treatment in the primary score. Conditional source-valid efficacy is a secondary view.

For root `b` and method `m`, measure:

```text
R_b(m) = repaired corrupted cases / declared corrupted-case denominator
C_b(m) = clean cases made incorrect / declared clean-case denominator
S_b(m) = (R_b(m) + 1 - C_b(m)) / 2
Delta_b = S_b(BPD) - S_b(registered strongest sham)
```

Construct equal-sized clean and corrupted strata or freeze equivalent stratum weights. `S` is
balanced verified success. This places the paired root contrast in `[-1,1]` for the FSI-01 bounded
effect contract. Also report the unscaled recovery-minus-corruption contrast, which equals
`2 Delta_b`. The original proposed +5 percentage point net gain therefore corresponds to
`delta_mech = 0.025` on the balanced score. These values are planning proposals; development
calibration must determine whether the instrument can resolve them before registration.
Choose the primary sham on DEVELOPMENT and freeze its identity before confirmation. Any claim
against a full sham family must cover every registered member with its declared multiplicity
rule; selecting a new comparator after reading confirmation outcomes changes the analysis.

The principal replication unit for transfer across learned systems is an independently trained
model root with independently assigned task worlds. Visits, masks, donor alternatives and repeated
cases remain measurements inside that root. A fixed-checkpoint claim about new task worlds uses
world roots instead. Shared training or evaluation worlds across model seeds introduce crossed
dependence; state that structure and use a prospectively chosen analysis that respects it.

Use paired binary FSI-01 only where its case independence assumptions hold. Grouped outcomes use
root-level paired inference, with the equal-root macro effect as the primary estimate. A descriptive
cluster bootstrap carries its approximate label. Adjust for the complete frozen comparison family.
Keep raw recovery and corruption rates beside the combined score so a favorable aggregate cannot
conceal a damaged clean-task stratum.

## Gates before additional compute

| Gate | Required receipt | Consequence of failure |
| --- | --- | --- |
| Reconstruction | Full-head and recipient-state no-op parity, complete residual coverage | Repair the instrument |
| Task competence | Frozen baseline and source fidelity, with all attempted roots accounted for | Redesign task or training recipe |
| Dose validity | Actual displacement, state radius, support and match residual per case | Restrict the claim to controls that passed |
| Development resolution | Independent root effects and matched shams passed to pinned EvalMath strict preflight | Retain development-only status and redesign the contrast |
| Prospective release | Frozen test family, selection history, sufficient independent roots and separate authorized sealed population | Keep confirmation closed |

Run `python skills/evalmath-mechinterp-power/scripts/preflight.py --pilot pilot.json --strict`
in EvalMath after collecting independent development receipts. This document supplies no fabricated
pilot inputs and makes no readiness claim. Eight trained roots can reveal variance during development.
The sample-maximum 95th-percentile sham bound at 95% confidence requires at least 59 independent
sham worlds. Confirmation size must also satisfy the chosen effect floor, comparison family and
FSI-01 evidence policy; a single convenient seed count cannot establish all of these requirements.

## RAM study and learning handoff

The proposed RAM sweep uses 2, 4, 8 and 16 GiB total process-tree memory caps while holding enforced
CPU quota, thread count and allowed model/verifier work fixed. Each cell remains subject to its
existing operator cap and requires the registered path for any increase. Count model weights, optimizer state where present, component
maps, both live memory banks, child checkers and the donor bank within the cap. A runner must enforce the process-tree cap and record peak
resident memory before these runs can be treated as resource comparisons.

Measure verified solutions per wall-clock hour, bank hit and miss counts, bytes retained, rebuild
work and rejected stale entries. Use the same task order across paired arms. Changes to global
weights invalidate weight-dependent entries. Versioned detached summaries can be an explicit
approximation arm; live autograd state follows the declared carry and supervision clocks.

After the causal instrument passes, compare ordinary training against targeted and sham-steered
training with all evaluations performed through the ordinary model path. Log actual parameter
gradients, optimizer steps, processed tokens and total block work. DataGoblin receives public
receipts containing dose, effect interval, source validity and measured costs. It may propose the
next experiment array. HybridTRMLDT's independent membrane handles admissible public-state updates;
the operator's runner owns execution and resource caps.

## Deliverables and evidence status

The first implementation should produce a frozen factorization adapter, parity checks and a paired
episode runner. Its output ledger must include `unit`, `n_units`, `n_cases`, `primary_contrast`,
`physical_dose`, `scientific_floor`, `sham_floor_and_source`, `planned_and_actual_comparisons`,
`estimated_resolution`, `selection_exposure`, `causal_controls`, `verdict` and source hashes.

Current evidence class: development design. The protocol defines the instrument and the decisions
its receipts would support. Native intervention code and measured causal effects remain pending.

### Source contracts

- [BPD research instructions](../../../../AGENTS.md)
- [EvalMath mechanistic-resolution skill](https://github.com/MoralityLabAI/EvalMath/blob/main/skills/evalmath-mechinterp-power/SKILL.md)
- [FSI-01 finite-sample contract](https://github.com/MoralityLabAI/EvalMath/blob/main/paper/finite_sample_inference.md)
- [Hybrid prefix-memory design](https://github.com/MoralityLabAI/HybridTRMLDT/blob/research/molt-goblin-01-20261126/docs/molt_prefix_memory_hybrid_v1.md)
