import Init

/-!
NC09-FC03: finite dimension bookkeeping for the fixed-rank constrained chart.
This file proves only integer rank/parameter/experiment invariants. Analytic
principal-angle and numerical constrained-feasibility claims are documented
in THEORY.md and tested against actual IEEE float64 projectors in CI.
-/

namespace BPD.NC09FC03

def passRanks : List Nat := [4, 3, 3, 2]
def totalRetainedRank : Nat := passRanks.foldl (· + ·) 0
def chartParameterCount : Nat := 4 * 2
def archivedProjectorFloatCount : Nat := 4 * 20 * 20
def independentModelRoots : Nat := 16
def perRootCharts : Nat := 32
def maxGradientSteps : Nat := 64

theorem registered_source_budget :
    totalRetainedRank = 12 := by decide

theorem exactly_eight_chart_parameters :
    chartParameterCount = 8 := by decide

theorem projector_archive_size :
    archivedProjectorFloatCount = 1600 := by decide

theorem first_engineering_cohort :
    independentModelRoots = 16 := by decide

theorem source_chart_optimizer_budget :
    perRootCharts * maxGradientSteps = 2048 := by decide

#print axioms BPD.NC09FC03.registered_source_budget
#print axioms BPD.NC09FC03.exactly_eight_chart_parameters
#print axioms BPD.NC09FC03.source_chart_optimizer_budget

end BPD.NC09FC03
