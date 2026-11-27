# RP04 Amendment 01: independent replay arithmetic and fresh draw

Date: 2026-11-26. Evidence class: **DEVELOPMENT_ONLY**.

## Scope and original failure

The source-first RP04 eight-root development attempt generated a complete set of
frozen-checkpoint outcomes. Its independent replay stopped on a scalar difference
of 3.05176e-5 at model root 27102, above the predeclared 2e-5 limit. Preserve
the original first-attempt FAIL verdict. The original attempt's raw files and
local supplemental replay were held in a previous, nonpersistent workspace and
are not present in this published branch. Numbers from that unarchived attempt
must not be presented as independently published confirmation.

The arithmetic discrepancy arose because the independent auditor evaluated a
pathway condition as receiver-write plus selected components. The native hook
evaluates the write at the patched carry state, then subtracts excluded writes.
These are algebraically equal in exact arithmetic, with different float32
rounding. The auditor now evaluates precisely the native operation sequence.
The threshold remains 2e-5. A focused test compares the intermediate recurrent
write at the patched step against the native run.

## Replacement development population

The eight *model roots* remain 27101 through 27108 and are already exposed.
The intervention inventory, 0.75 L2 dose, 256 base pairs per root, query
expansion, outcome family and Holm adjustment remain as frozen in PROTOCOL.md.
The development sampler now uses the disjoint
`slot-selectivity-v2-audit-arithmetic` SHA-256 seed domain. Smoke retains its
existing v1 domain, allowing exact comparison with historical engineering
inputs. Generate the replacement only from a separately published new source
lock and preserve the original source commit as part of the audit lineage.

## Publication and replay requirements

An amended run must archive all raw per-root checkpoint predictions, input and
checkpoint hashes, root-level statistical receipts, diagnostic summary, the
original and corrected code revisions, and an independent PASS/FAIL audit. CI
keeps the replacement outputs as a named GitHub Actions artifact for review.
The published source lock and its run manifest define the reproducible source
of truth. Every outcome is descriptive development evidence; the eight exposed
roots cannot independently calibrate the population sham tail.

This amendment does not alter RP03 source or checkpoints and authorizes no
confirmation-cohort release or merge.
