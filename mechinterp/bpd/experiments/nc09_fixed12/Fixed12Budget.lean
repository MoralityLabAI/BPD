import Init

/-!
NC09 Fixed12: finite source rank-allocation and dimension budget arithmetic.
This certifies integer invariants only. It does not prove the DP optimum,
physical causal fidelity, statistical independence or any experimental claim.
-/

namespace BPD.NC09Fixed12

def recurrentPasses : Nat := 4
def allowedRanks : List Nat := [2, 3, 4, 5, 6]
def registeredRoots : Nat := 16
def trainedWidth : Nat := 20
def mainBudget : Nat := 12
def strongerReference : Nat := recurrentPasses * 4
def fullCandidateGrid : Nat := allowedRanks.length ^ recurrentPasses

theorem fixed_DP_budget : mainBudget = 12 := by decide
theorem flat_three_budget : recurrentPasses * 3 = mainBudget := by decide
theorem flat_four_budget : strongerReference = 16 := by decide
theorem fixed12_saves_four_slots : strongerReference - mainBudget = 4 := by decide
theorem finite_pass_rank_tuple_count : fullCandidateGrid = 625 := by decide
theorem sixteen_independent_development_models : registeredRoots = 16 := by decide
theorem all_ranks_finite : ∀ r ∈ allowedRanks, r ≤ trainedWidth := by decide

#print axioms BPD.NC09Fixed12.flat_three_budget
#print axioms BPD.NC09Fixed12.flat_four_budget
#print axioms BPD.NC09Fixed12.finite_pass_rank_tuple_count

end BPD.NC09Fixed12
