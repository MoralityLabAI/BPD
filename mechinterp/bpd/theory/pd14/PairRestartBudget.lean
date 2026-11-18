import Init

/-!
BPD-PD-14: finite budget certificate for a 9-sweep single start and
three independent three-sweep restarts, on the exact 16D 37-angle grid.

This theorem checks candidate and vector score accounting only. It
proves nothing about Gaussian source identifiability, optimizer
convergence, statistical independence, or held-out mechanism recovery.
-/

namespace BPD.PD14

def unorderedPairs : Nat := 16 * 15 / 2
def angles : Nat := 37
def passes : Nat := 4
def trainingBoards : Nat := 2048
def totalCandidateScores (starts sweeps : Nat) : Nat :=
  starts * sweeps * unorderedPairs * angles
def scalarScores (starts sweeps : Nat) : Nat :=
  totalCandidateScores starts sweeps * (trainingBoards * passes)

theorem nine_sweeps_equals_three_by_three :
    totalCandidateScores 1 9 = totalCandidateScores 3 3 := by
  decide

theorem equal_exact_candidate_budget :
    totalCandidateScores 1 9 = 39960 ∧
    totalCandidateScores 3 3 = 39960 := by
  decide

theorem equal_exact_sample_weighted_budget :
    scalarScores 1 9 = 327352320 ∧
    scalarScores 3 3 = 327352320 := by
  decide

theorem both_modes_use_same_8192_vectors :
    trainingBoards * passes = 8192 := by
  decide

#print axioms BPD.PD14.nine_sweeps_equals_three_by_three
#print axioms BPD.PD14.equal_exact_candidate_budget
#print axioms BPD.PD14.equal_exact_sample_weighted_budget
#print axioms BPD.PD14.both_modes_use_same_8192_vectors

end BPD.PD14
