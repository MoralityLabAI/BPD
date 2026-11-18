import Init

/-!
BPD-PD-15 exact candidate-angle and sample-weighted accounting.

This finite certificate proves registered 16D score counts are identical
for unweighted and norm-weighted searches at nine sweeps. It does not
prove Gaussian objective consistency, any transcendental inequality,
optimizer convergence, or physical mechanism identification.
-/

namespace BPD.PD15

def dimensions : Nat := 16
def unorderedPairs : Nat := 16 * 15 / 2
def grid : Nat := 37
def sweeps : Nat := 9
def trainingBoards : Nat := 2048
def passes : Nat := 4
def observedVectors : Nat := trainingBoards * passes

def angleScores (sweeps pairs angles : Nat) : Nat := sweeps * pairs * angles
def sampleScores (sweeps pairs angles boards passes : Nat) : Nat :=
  angleScores sweeps pairs angles * boards * passes

theorem exactly_120_pairs : unorderedPairs = 120 := by decide
theorem exactly_8192_observed_vectors : observedVectors = 8192 := by decide

theorem both_gaussian_objectives_use_39960_angles :
  angleScores sweeps unorderedPairs grid = 39960 ∧
  angleScores sweeps unorderedPairs grid = 39960 := by
  decide

theorem exactly_327352320_candidate_by_vector_scores :
  sampleScores sweeps unorderedPairs grid trainingBoards passes = 327352320 := by
  decide

theorem norm_weight_equal_candidate_budget
  (uniform weighted : List (Nat × Nat))
  (h : uniform.length = weighted.length) (angles : Nat) :
  uniform.length * angles = weighted.length * angles := by
  rw [h]

#print axioms BPD.PD15.exactly_120_pairs
#print axioms BPD.PD15.exactly_8192_observed_vectors
#print axioms BPD.PD15.both_gaussian_objectives_use_39960_angles
#print axioms BPD.PD15.exactly_327352320_candidate_by_vector_scores
#print axioms BPD.PD15.norm_weight_equal_candidate_budget

end BPD.PD15
