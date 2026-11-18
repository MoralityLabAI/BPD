# TRM Depth Lab: Local Research UI

**Status:** additive read-only browser interface for the existing BPD-DT01, DT02-A and DT03 gym. Archived negative results and experiment source locks remain intact.

## Launch

From the root of the MoralityLabAI/BPD repository, in a Python environment with PyTorch installed:

```bash
python -m mechinterp.bpd.experiments.depth_steering_gym.ui_server --open
# Or browse manually after starting:
python -m mechinterp.bpd.experiments.depth_steering_gym.ui_server --port 8765
```

Open http://127.0.0.1:8765/ in your browser. The server binds loopback only and refuses remote hosts. It implements read-only GET endpoints, has no authentication, and must not be exposed through a public proxy.

## Panels

- **Overview:** original CPU-trained 14,082-parameter mini-TRM results for engineering roots 901 and 902, depth 8/12 exact task outcomes, token/action proxies, and training losses.
- **Task explorer:** deterministic UI-only Sudoku4 puzzles with optional host solution; demo-conditioned synthetic ARC-style examples with a host answer reveal; interactive hidden-ending storyworld with five actions, an optional public teacher suggestion and terminal-only host grading.
- **Causal diagnostics:** the historical no-edit, full protected-dark and attention-SVD rank8 arms at depths 4/8/12, including donor token agreement, logit MSE and edit norms, labeled as engineering comparisons without Jacobian-matched controls.
- **Compute planning:** basic H/L state footprint, conceptual FP32 full-output dense Jacobian size, and BF16 parameter storage. These are arithmetic lower bounds, not GPU benchmarks.

## Evidence and safety

The viewer verifies the exact archived SHA-256 before serving experimental metrics:

```text
9d4f9463c0853c6cc2491711ebf3cb6c37021a61861f3947962dff67fa76b30b
```

Source: [DT02-A PR #103](https://github.com/MoralityLabAI/BPD/pull/103), [frozen first CI run #44277919962](https://github.com/MoralityLabAI/BPD/actions/runs/44277919962). Repairs: [merged DT03 PR #113](https://github.com/MoralityLabAI/BPD/pull/113) and [DT03_REVIEW.md](DT03_REVIEW.md). The historical README, PROTOCOL, LOCK_STAGE0, LOCK_CI and first-run JSON remain unchanged.

The app never trains, loads model weights, contacts external model hosts, or accesses the sealed 168-case ARG-03-N population. Its synthetic UI fixtures have domain-separated seeds and are not scientific evaluation roots. Public model inputs and human-inspectable host gold are separate objects. Storyworld host endings are returned only after the terminal step.

## Tests

```bash
python -m unittest mechinterp.bpd.experiments.depth_steering_gym.test_ui -v
python -m unittest mechinterp.bpd.experiments.depth_steering_gym.test_dt03_contract -v
```

The UI test suite checks immutable archive digests, task field separation, deterministic fixtures, local HTTP handling, invalid action requests, and page source. A separate PR would be required for native inference or a training interface, including new checkpoint locks and a registered evaluation protocol.
