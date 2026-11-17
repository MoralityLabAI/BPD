import Init

/-!
BPD-PD-06: exact finite support-mass certificate for the source design.

Under independent continuous nonzero amplitudes, a read supported on
coordinates S has structural zero mass equal to the total source
probability of supports disjoint from S. The probability argument remains
in the paper note. This Lean file proves all combinatorial weights with
their common denominator 36.

Training: four singletons each weight 3/36, four cycle pairs each 3/36,
three triples each 4/36. The support split is fixed before any PD06 data.
-/

namespace BPD.PD06

def weightedSupports : List (List Nat × Nat) :=
  [([0],3),([1],3),([2],3),([3],3),
   ([0,1],3),([1,2],3),([2,3],3),([0,3],3),
   ([0,1,2],4),([0,1,3],4),([1,2,3],4)]

def structuralZeroMass (readSupport : List Nat) : Nat :=
  weightedSupports.foldl
    (fun acc entry =>
      if readSupport.any (fun i => entry.1.contains i) then acc
      else acc + entry.2) 0

theorem support_total_weight :
    (weightedSupports.map Prod.snd).sum = 36 := by
  decide

theorem rotated_coordinate_hyperplane_masses :
    structuralZeroMass [0] = 19 ∧
    structuralZeroMass [1] = 15 ∧
    structuralZeroMass [2] = 19 ∧
    structuralZeroMass [3] = 19 := by
  decide

theorem rotated_four_coordinate_expected_l0 :
    4 * 36 - (19 + 15 + 19 + 19) = 2 * 36 := by
  decide

def allSixPairsStrict : Bool :=
  ([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)] : List (Nat × Nat)).all
    (fun p =>
      decide (structuralZeroMass [p.1,p.2] < structuralZeroMass [p.1] ∧
              structuralZeroMass [p.1,p.2] < structuralZeroMass [p.2]))

theorem noncoordinate_two_support_strictly_lower_mass :
    allSixPairsStrict = true := by
  decide

-- These two held-out pair supports are absent from the training support list.
def seenPairSupports : List (List Nat) :=
  [[0,1],[1,2],[2,3],[0,3]]

def unseenPairSupports : List (List Nat) :=
  [[0,2],[1,3]]

theorem heldout_pairs_distinct_from_seen :
    (unseenPairSupports.all fun x => !(seenPairSupports.contains x)) = true := by
  decide

def seenTripleSupports : List (List Nat) :=
  [[0,1,2],[0,1,3],[1,2,3]]

theorem heldout_triple_distinct :
    (seenTripleSupports.contains [0,2,3]) = false := by
  decide

end BPD.PD06

#print axioms BPD.PD06.support_total_weight
#print axioms BPD.PD06.rotated_coordinate_hyperplane_masses
#print axioms BPD.PD06.rotated_four_coordinate_expected_l0
#print axioms BPD.PD06.noncoordinate_two_support_strictly_lower_mass
#print axioms BPD.PD06.heldout_pairs_distinct_from_seen
#print axioms BPD.PD06.heldout_triple_distinct
