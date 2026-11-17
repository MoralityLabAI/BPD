import Lean

/-!
BPD-PD-05 source-law cardinality separability, quantized magnitude witness.

The generated source uses floating magnitudes in [0.75, 1.25] on active
coordinates and exact zero on inactive ones. This exact Nat certificate
uses integer scaled versions [75,125], with threshold 138 corresponding
to magnitude sum 1.38, which lies strictly between 1.25 and 1.50.

The real-interval version follows by inequalities in the research note;
these Nat results alone do not certify continuous PyTorch floating-point
training, assignment correctness, or neural mechanism identification.
-/

namespace BPD.PD05

def countFromMagnitudes (a b : Nat) : Nat :=
  if a + b < 138 then 1 else 2

theorem one_active_first (a : Nat) (hmax : a ≤ 125) :
    countFromMagnitudes a 0 = 1 := by
  have h : a + 0 < 138 := by omega
  simp [countFromMagnitudes, h]

theorem one_active_second (b : Nat) (hmax : b ≤ 125) :
    countFromMagnitudes 0 b = 1 := by
  have h : 0 + b < 138 := by omega
  simp [countFromMagnitudes, h]

theorem two_active (a b : Nat)
    (ha : 75 ≤ a) (hb : 75 ≤ b) :
    countFromMagnitudes a b = 2 := by
  have h : ¬ (a + b < 138) := by omega
  simp [countFromMagnitudes, h]

theorem quantized_one_two_gap
    (a b : Nat)
    (haLow : 75 ≤ a) (haHigh : a ≤ 125)
    (hbLow : 75 ≤ b) (hbHigh : b ≤ 125) :
    countFromMagnitudes a 0 = 1 ∧
    countFromMagnitudes 0 b = 1 ∧
    countFromMagnitudes a b = 2 := by
  exact ⟨one_active_first a haHigh,
    one_active_second b hbHigh,
    two_active a b haLow hbLow⟩

/-- Both cardinality classes remain disjoint under source bounds. -/
theorem gap_between_one_and_two_active
    (single doubleA doubleB : Nat)
    (hs : single ≤ 125)
    (ha : 75 ≤ doubleA) (hb : 75 ≤ doubleB) :
    single < doubleA + doubleB := by
  omega

end BPD.PD05

#print axioms BPD.PD05.one_active_first
#print axioms BPD.PD05.one_active_second
#print axioms BPD.PD05.two_active
#print axioms BPD.PD05.quantized_one_two_gap
#print axioms BPD.PD05.gap_between_one_and_two_active
