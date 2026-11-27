# ER-02/03 exploratory receipt

CPU: PyTorch 2.10.0+cpu, NumPy 2.3.5. Commands:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python learned.py > learned.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bitset.py > bitset.json
```
Seed 261127; 64 train roots (0..63), 16 development roots (64..79), 64 held-out roots (80..143). Shared candidate family across roots, so this is probe/instance generalization, not graph-family or support-disjoint generalization. 63879 training examples, 14994 development examples, 4225 parameters, dev MSE 0.0755382255 bits squared. Best development checkpoint across 12 epochs. One model seed, exploratory, no preregistration or neural ablation claim.

Unique recovery counts at probes 1..12:
- random: 0,0,0,0,2,6,19,30,43,50,53,57
- static: 0,0,0,1,1,7,18,26,37,46,52,57
- marginal heuristic: 0,0,0,0,1,6,24,33,45,53,57,59
- learned: 0,0,0,1,5,11,28,43,55,56,57,60
- exact: 0,0,0,0,1,20,46,63,64,64,64,64

At probe 8 learned vs random: 21 wins, 8 losses; vs marginal heuristic: 20 wins, 10 losses; vs exact: 1 win, 21 losses. Median cumulative scoring times learned 2.906 ms, exact 1.716 ms; learned fails speed objective. The variance-labeled arm is a balanced sign-marginal heuristic, not exact predictive variance. Static initialization excluded from scoring timer. Probe budget matched; CPU time not matched. The loop model is a tied residual MLP, not a full TRM architecture.

Cache replay: identical actions and surviving counts for 64/64 roots. Raw prediction matrix 259840 bytes. Median build+score+filter time 3.090 ms cached, 4.087 ms recomputed; median paired speedup 1.296. Timing excludes world generation and miscellaneous overhead, no peak RSS measured.

Packed partition masks: 64/64 action and count trajectory matches against full prediction matrix. Raw packed payload 130048 bytes (excludes Python int/container overhead and construction temporaries). Median times matrix 3.128 ms, packed 4.046 ms, paired speedup matrix/packed 0.771. Packing overhead defeats end-to-end speed at this size. Timing scopes differ slightly. Build transiently allocates original matrix. No measured RSS reduction claim.

## Structural obstruction for marginal-only utility
Let five distinct candidate edges be a,b,c,d,e. Compare uniform posterior P on {a,b,e}, {c,d,e} and Q on {a,c,e}, {b,d,e}. All edge marginals match. A probe with signs +,+,-,-,+ produces outcomes 3,-1 for P and 1,1 for Q, so H_P=1 bit and H_Q=0. Such signs are realizable in the six-node bilinear sensing family by choosing edges with appropriate source/destination signs. This is a counterexample for general posterior banks; reachability of both particular banks under the gym's prior and probe history has not been established. Marginals alone cannot be a universally sufficient compression of arbitrary banks. A single deterministic estimator of this common summary has worst-case absolute error at least 0.5 bit (MSE >=0.25 averaged over the two banks).

Next: covariance features, exact third-order moment sufficient for four outcomes, adaptive switch between cached exact and learned estimates; vectorized/native partition representation and repeated-query amortization; test held-out supports and graph sizes. Noise and actual BPD gates remain future experiments.
