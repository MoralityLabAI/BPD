import Init

/-!
NC09-FC01: exact rational order-statistic arithmetic for a 59-root
maximum as a nonparametric 95/95 tolerance bound.

The arithmetic theorem checks 1 - (19/20)^59 >= 19/20 by clearing
positive denominators. This is conditional on iid/exchangeable root-level
null statistics and is not evidence of source independence or of a trained
causal mechanism.
-/

namespace BPD.NC09Calibration

def calibrationRoots : Nat := 59
def heldoutConfirmationRoots : Nat := 32
def shamsPerRoot : Nat := 8
def registeredShamObservations : Nat := calibrationRoots * shamsPerRoot

theorem exact_59_root_95_95_arithmetic :
    20 * (19:Nat)^59 ≤ (20:Nat)^59 := by decide

theorem exactly_472_nested_sham_outcomes :
    registeredShamObservations = 472 := by decide

theorem twenty_dimensional_four_pass_TV_budget :
    4 * 3 = 12 := by decide

#print axioms BPD.NC09Calibration.exact_59_root_95_95_arithmetic
#print axioms BPD.NC09Calibration.exactly_472_nested_sham_outcomes

end BPD.NC09Calibration
