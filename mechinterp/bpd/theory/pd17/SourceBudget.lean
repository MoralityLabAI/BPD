import Init

/-!
BPD-PD-17 exact finite source/cohort accounting certificate.

The support-matching exact-zero mass theorem is proved in THEORY.md via
the determinant matching argument and audited with exact rationals.
These Lean calculations certify only registered integer constants.
-/

namespace BPD.PD17

def dim : Nat := 16
def supportCardinalityChoices : Nat := 4
def roots : Nat := 32
def sourceLaws : Nat := 2
def arms : Nat := 4
def sweeps : Nat := 9
def pairs : Nat := dim * (dim - 1) / 2
def angles : Nat := 37
def boards : Nat := 2048
def steps : Nat := 4

theorem exact_singleton_probability_denominator :
    supportCardinalityChoices * dim = 64 := by decide
theorem exact_normalized_one_mixed_column_denominator :
    (supportCardinalityChoices * dim) * dim = 1024 := by decide
theorem exactly_120_orthogonal_pairs : pairs = 120 := by decide
theorem exact_matched_givens_angle_budget :
    sweeps * pairs * angles = 39960 := by decide
theorem exact_candidate_by_vector_budget :
    sweeps * pairs * angles * boards * steps = 327352320 := by decide
theorem exact_original_paired_cohort_size :
    roots * sourceLaws * arms = 256 := by decide

#print axioms BPD.PD17.exact_singleton_probability_denominator
#print axioms BPD.PD17.exact_normalized_one_mixed_column_denominator
#print axioms BPD.PD17.exact_matched_givens_angle_budget
#print axioms BPD.PD17.exact_candidate_by_vector_budget
#print axioms BPD.PD17.exact_original_paired_cohort_size

end BPD.PD17
