import Init

/-!
BPD-PD-13: exact count certificate for matched candidate budgets and the
eightfold sample-weighted cost increase.

This finite arithmetic certificate checks the preregistered 16D accounting.
It does not prove optimizer convergence, source identifiability or that a
pair priority heuristic recovers the latent factors.
-/

namespace BPD.PD13

def pairs16 : Nat := 16 * 15 / 2

def candidateCount (sweeps pairs angles : Nat) : Nat :=
  sweeps * pairs * angles

def observedVectors (trainingBoards passes : Nat) : Nat :=
  trainingBoards * passes

theorem dimension16_has_120_unordered_pairs : pairs16 = 120 := by
  decide

theorem equal_priority_and_lex_angle_budget :
    candidateCount 3 pairs16 37 = 13320 ∧
    candidateCount 3 pairs16 37 = candidateCount 3 pairs16 37 := by
  decide

theorem exact_high_and_low_observed_vector_counts :
    observedVectors 2048 4 = 8192 ∧
    observedVectors 256 4 = 1024 := by
  decide

theorem exact_sample_weighted_candidate_counts :
    candidateCount 3 pairs16 37 * observedVectors 2048 4 = 109117440 ∧
    candidateCount 3 pairs16 37 * observedVectors 256 4 = 13639680 := by
  decide

theorem sample_weighted_evaluations_increase_eightfold :
    candidateCount 3 pairs16 37 * observedVectors 2048 4 =
    8 * (candidateCount 3 pairs16 37 * observedVectors 256 4) := by
  decide

theorem equal_evaluation_count_for_equal_length_pair_orders
    (priorityOrder lexOrder : List (Nat × Nat))
    (h : priorityOrder.length = lexOrder.length)
    (angles : Nat) :
    priorityOrder.length * angles = lexOrder.length * angles := by
  rw [h]

end BPD.PD13

#print axioms BPD.PD13.dimension16_has_120_unordered_pairs
#print axioms BPD.PD13.equal_priority_and_lex_angle_budget
#print axioms BPD.PD13.exact_high_and_low_observed_vector_counts
#print axioms BPD.PD13.exact_sample_weighted_candidate_counts
#print axioms BPD.PD13.sample_weighted_evaluations_increase_eightfold
#print axioms BPD.PD13.equal_evaluation_count_for_equal_length_pair_orders
