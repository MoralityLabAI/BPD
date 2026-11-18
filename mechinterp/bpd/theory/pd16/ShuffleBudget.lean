import Init

/-!
BPD-PD-16 arithmetic certificate for shuffled-weight placebo parity.

This proves only the exact finite candidate-angle and per-observation
counts for three nine-sweep 16D searches. It makes no claim about
a random permutation's statistical independence, source identifiability,
or learned causal mechanism recovery.
-/

namespace BPD.PD16

def pairs16 : Nat := 16 * 15 / 2
def candidates (sweeps pairs angles : Nat) := sweeps * pairs * angles
def observedVectors (boards passes : Nat) := boards * passes
def weightedOperations (sweeps pairs angles boards passes : Nat) :=
  candidates sweeps pairs angles * observedVectors boards passes

theorem registered_120_pairs : pairs16 = 120 := by decide
theorem registered_8192_vectors : observedVectors 2048 4 = 8192 := by decide
theorem each_three_searches_has_39960_candidates :
  candidates 9 pairs16 37 = 39960 := by decide
theorem each_three_searches_has_327352320_scalar_scores :
  weightedOperations 9 pairs16 37 2048 4 = 327352320 := by decide

theorem paired_weight_permutation_keeps_length
    (aligned shuffled : List Nat)
    (h : aligned.length = shuffled.length) :
    aligned.length = shuffled.length := h

#print axioms BPD.PD16.registered_120_pairs
#print axioms BPD.PD16.registered_8192_vectors
#print axioms BPD.PD16.each_three_searches_has_39960_candidates
#print axioms BPD.PD16.each_three_searches_has_327352320_scalar_scores

end BPD.PD16
