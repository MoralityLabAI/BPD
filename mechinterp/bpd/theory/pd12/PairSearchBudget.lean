import Init

/-!
BPD-PD-12 exact candidate-angle accounting for lexicographic Givens search.

The registered dimensional budgets count each evaluated angle at each
unordered read-column pair over three sweeps. These propositions certify
the finite counts and equal candidate-evaluation budgets for global/local
search. They are arithmetic bookkeeping, not a proof of recovery, runtime,
or correctness of nonlinear neural mechanisms.
-/

namespace BPD.PD12

def pairCount (d : Nat) : Nat := d * (d - 1) / 2

def candidates (d sweeps angles : Nat) : Nat :=
  sweeps * pairCount d * angles

theorem dimension8_pairs : pairCount 8 = 28 := by decide

theorem dimension16_pairs : pairCount 16 = 120 := by decide

theorem matched_8D_global_local_candidate_budget :
    candidates 8 3 37 = 3108 ∧ candidates 8 3 37 = candidates 8 3 37 := by
  decide

theorem matched_16D_global_local_candidate_budget :
    candidates 16 3 37 = 13320 ∧ candidates 16 3 37 = candidates 16 3 37 := by
  decide

theorem low_budget_costs :
    candidates 8 3 13 = 1092 ∧ candidates 16 3 13 = 4680 := by
  decide

theorem dimension_costs_grow_and_low_budget_is_cheaper :
    candidates 8 3 13 < candidates 8 3 37 ∧
    candidates 16 3 13 < candidates 16 3 37 ∧
    candidates 8 3 37 < candidates 16 3 37 := by
  decide

end BPD.PD12

#print axioms BPD.PD12.dimension8_pairs
#print axioms BPD.PD12.dimension16_pairs
#print axioms BPD.PD12.matched_8D_global_local_candidate_budget
#print axioms BPD.PD12.matched_16D_global_local_candidate_budget
#print axioms BPD.PD12.low_budget_costs
#print axioms BPD.PD12.dimension_costs_grow_and_low_budget_is_cheaper
