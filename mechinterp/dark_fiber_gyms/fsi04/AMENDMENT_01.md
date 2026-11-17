# FSI-04 pre-native amendment 01: score missing cells separately

**Date: 2026-11-17.** This amendment was made after the engineering-only
fresh-roster smoke check and before any FSI-04 native checkpoint inference.
The original immutable `REGISTRATION.json` remains the canonical source
for the **primary** 10-percentage-point change in correct cells across all
81 Sudoku positions, donor joint transplant versus clean at horizon 32.

The development pilot also records a *secondary* outcome over originally
blank puzzle cells. Each record contains the number of missing cells, the
number now correct, wrong-to-correct gains, correct-to-wrong losses, and net
improvement versus clean. A root's missing-cell effect is that net divided
by its own count of missing cells. Evaluators should report root-macro means
so puzzles with different clue counts receive equal weight.

The missing-cell measure addresses clue saturation and exposes whether
behavioral changes concern inference on the puzzle's unknown cells. It is
**secondary and exploratory** for FSI-04. Any future confirmation using it
as the primary outcome needs a distinct prospective registration, its own
minimum mechanistically meaningful effect, matched sham variation floor,
power calculation and untouched roots. Historical FSI-03 summaries contain
only aggregate 81-cell correct counts and cannot supply a valid missing-cell
sham calibration.

The current CI tests only this scoring arithmetic with synthetic toy
states. No ARG-03 sealed test material and no actual model checkpoint
weights were accessed by this amendment.
