import Init

/-!
NC08 finite physical-rank slot-budget certificate. This checks only arithmetic
and registered fixed candidate counts, not model-score optimality, source
availability, recurrent causal sufficiency, or task behavior.
-/

namespace BPD.NC08

def passes : Nat := 4
def rootCount : Nat := 8
def minimumPassRank : Nat := 2
def maximumPassRank : Nat := 6
def referencePerPassRank : Nat := 4
def referenceTotalSlots : Nat := referencePerPassRank * passes
def registeredBudgets : List Nat := [8, 10, 12, 14]

theorem flat_four_uses_sixteen_slots :
    referenceTotalSlots = 16 := by decide

theorem all_frozen_candidate_budgets_below_flat_four :
    ∀ b ∈ registeredBudgets, b < referenceTotalSlots := by
  decide

theorem rank_twelve_saves_four_physical_slots :
    referenceTotalSlots - 12 = 4 := by decide

theorem admissible_four_pass_minimum_is_eight :
    minimumPassRank * passes = 8 := by decide

theorem maximum_single_pass_rank_is_six :
    maximumPassRank = 6 := by decide

theorem eight_roots_five_slot_options_four_passes :
    rootCount * 5 * passes = 160 := by decide

#print axioms BPD.NC08.flat_four_uses_sixteen_slots
#print axioms BPD.NC08.all_frozen_candidate_budgets_below_flat_four
#print axioms BPD.NC08.rank_twelve_saves_four_physical_slots

end BPD.NC08
