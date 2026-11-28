# BPD-NBR-03 pre-run amendment: disjoint seed intervals

The original protocol at f12944c records consecutive fresh root seeds.
Review of the inherited CQ-06 source before any NBR-03 execution found
that its generator uses seed + offset + training_step, so consecutive
root seeds can reuse underlying random-number streams across different
rotated worlds. The completed NBR-02 development study used that
inherited scheme and remains unchanged; it does not establish independent
population inference.

Before any new root is generated, replace the NBR-03 root roster by
202631250301 + 2000000*i for i=0..7 and use excluded toy root
202629250300. Every specified seed offset is smaller than 1000000,
so distinct root intervals are separated, including from the earlier
NBR-02 roster. This is a pre-execution source hygiene correction,
with all architectures, arms, cases, update budgets, scores, access
accounting and evidence scope unchanged. No new model result existed
when this amendment was written.
