# RP03 engineering: address binding emerges after longer task training

The fixed two-root run completed 2,000 task-only updates per model in 21.00
seconds. All six prescribed checkpoints are retained. The source recipe stayed
unchanged. Every diagnostic uses the TRAIN semantic partition and has
engineering scope.

The table pools the equally sized diagnostics from roots 27000 and 27001. Each
ordinary entry contains 1,024 cases. Address-binding entries also contain 1,024
cases, arising from 256 base sequences queried at every address.

| Updates | Delay | Ordinary correct | Ordinary accuracy | Address-binding correct | Address-binding accuracy |
|---|---:|---:|---:|---:|---:|
| 200 | 2 | 538/1,024 | 52.54% | 256/1,024 | 25.00% |
| 200 | 7 | 504/1,024 | 49.22% | 256/1,024 | 25.00% |
| 800 | 2 | 562/1,024 | 54.88% | 262/1,024 | 25.59% |
| 800 | 7 | 528/1,024 | 51.56% | 282/1,024 | 27.54% |
| 2,000 | 2 | 638/1,024 | 62.30% | 485/1,024 | 47.36% |
| 2,000 | 7 | 605/1,024 | 59.08% | 458/1,024 | 44.73% |

The modal-value baseline scores exactly 25% on every address-binding cohort.
At 2,000 updates, root 27000 reaches 49.61% binding accuracy at both delays;
root 27001 reaches 45.12% at delay 2 and 39.84% at delay 7. This demonstrates that
the architecture can learn some address-specific behavior with longer ordinary
training. The 200-update checkpoint sits at the query-blind baseline on this
diagnostic.

## The selected component intervention still transports few requested values

The fixed RP02 group scan was repeated on separate training calibration pairs
at each checkpoint. Its complete eight-group score matrices are retained.
At 2,000 updates, targeted counterfactual accuracy under the L2-0.75 write edit
is 13.28% and 14.06% for root 27000 at delays 2 and 7. Root 27001 records 12.11%
and 16.80%. The corresponding unaffected-query accuracies span 57.81% to 59.38%.

The model's ordinary learning improves before this particular component edit
becomes a reliable mechanism for transporting the requested value. Both facts
belong to the steering redesign decision.

## Evidence

[The fixed plan](PLAN.md) specifies the source recipe and complete roster.
[Summary rows](results/summary.json) retain every prescribed endpoint.
[The completion receipt](results/completion.json) binds both result files.
Each root directory includes the three model/optimizer checkpoints, all
diagnostic predictions and the cross-entropy trace sampled every 50 updates.
The run manifest records source hashes, runtime versions and the prior
`REDESIGN_OR_PILOT_ONLY` planning inputs.

These are fresh diagnostic draws from training tuples on two exposed model
roots. They supply no held-out population estimate. The separate extension
plan preserves this run and makes any later budget choice explicit.
