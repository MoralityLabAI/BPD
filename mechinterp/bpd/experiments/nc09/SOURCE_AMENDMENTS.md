# NC-09 pre-development engineering amendment (2026-11-18)

The source-only protocol at `e85724a42b97d9fa41f82cf76d537e71832250a7` remains unchanged.

The first [hosted source audit #44268471316](https://github.com/MoralityLabAI/BPD/actions/runs/44268471316) verified all original source/blob hashes but reported **7/10 unit tests passing**. Three source tests failed because `[N,4,20] @ [4,20,20]` attempted to broadcast the sample and pass axes. The development job `development-eight-pairs` was **skipped**, so **zero roots 900–907** were trained or scored in that run.

Before development, the implementation changed only the shape-correct replay from `source @ R` to `einsum("nti,tij->ntj",source,R)`; the independently generated synthetic unit-test replay received the same correction. No task definitions, ranks, eigengaps, alignment samples, evaluator, physical sham selection, confidence rules, source splits, or root population changed.

The two updated Git blob SHAs are recorded in `EXECUTE_LOCK.json`:
- `experiment.py`: `3ddfbb92a1ec9a0b92bb64575191846c6cd338db`
- `test_experiment.py`: `ddace0e5f50b8e688410a52eb99786ade2e50088`

Retain the first source-lock and failed-CI history. Any scientific output from the subsequently amended source remains **engineering development**; the confirmation cohort is sealed.
