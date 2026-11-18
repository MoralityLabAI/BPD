import Init

/-!
FC04 finite dimension and original source computation bookkeeping.
No scientific/causal claim follows from these arithmetic certificates.
-/
namespace BPD.NC09FC04

def retainedRanks : List Nat := [4,3,3,2]
def physicalWidth : Nat := 20
def originalRankBudget : Nat := retainedRanks.foldl (· + ·) 0
def fullTangentParameters : Nat :=
  retainedRanks.foldl (fun s r => s + (physicalWidth-r)*r) 0
def registeredRootCount : Nat := 16
def registeredChartCount : Nat := 32
def maxStepsEachChart : Nat := 64
def originalShamBudget : Nat := 8

theorem source_rank_budget_twelve : originalRankBudget = 12 := by decide
theorem complement_parameter_budget : fullTangentParameters = 202 := by decide
theorem original_chart_evaluation_budget :
    registeredChartCount * maxStepsEachChart = 2048 := by decide
theorem prospectively_independent_roots : registeredRootCount = 16 := by decide
theorem original_shams_per_root : originalShamBudget = 8 := by decide

#print axioms BPD.NC09FC04.source_rank_budget_twelve
#print axioms BPD.NC09FC04.complement_parameter_budget
#print axioms BPD.NC09FC04.original_chart_evaluation_budget

end BPD.NC09FC04
