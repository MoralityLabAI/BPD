import Init

/-!
BPD-PD-07 planted read-conditioned gate: exact quantized margin witness.

If a learned read basis is *exactly* the source latent orthogonal basis, the
normalized read coefficient of inactive feature j is zero, while active
magnitudes are in [75,125] after scaling by 100. The threshold 50
reconstructs a support mask for **every one of the 2^4 subsets**, including
combinations absent during training.

This is a finite theorem over Nat and arbitrary Bool support functions.
It does not establish that the PD07 optimizer learns the true rotated basis
or that learned threshold parameters preserve a robust margin after training.
-/

namespace BPD.PD07

def thresholdGate (x : Nat) : Bool :=
  decide (50 ≤ x)

theorem planted_active_gate (x : Nat) (h : 75 ≤ x) :
    thresholdGate x = true := by
  simp only [thresholdGate, decide_eq_true_eq]
  omega

theorem planted_inactive_gate :
    thresholdGate 0 = false := by
  decide

theorem read_gate_exact_on_any_coordinate (active : Bool)
    (magnitude : Nat) (h : 75 ≤ magnitude) :
    thresholdGate (if active then magnitude else 0) = active := by
  cases active with
  | false => exact planted_inactive_gate
  | true => exact planted_active_gate magnitude h

theorem read_gate_exact_on_any_four_coordinate_support
    (active : Fin 4 → Bool) (magnitude : Fin 4 → Nat)
    (h : ∀ i, 75 ≤ magnitude i) :
    ∀ i, thresholdGate (if active i then magnitude i else 0) = active i := by
  intro i
  exact read_gate_exact_on_any_coordinate (active i) (magnitude i) (h i)

theorem arbitrary_four_coordinate_subset_including_heldout
    (s0 s1 s2 s3 : Bool)
    (a b c d : Nat)
    (ha : 75 ≤ a) (hb : 75 ≤ b)
    (hc : 75 ≤ c) (hd : 75 ≤ d) :
    thresholdGate (if s0 then a else 0) = s0 ∧
    thresholdGate (if s1 then b else 0) = s1 ∧
    thresholdGate (if s2 then c else 0) = s2 ∧
    thresholdGate (if s3 then d else 0) = s3 := by
  exact ⟨read_gate_exact_on_any_coordinate s0 a ha,
    read_gate_exact_on_any_coordinate s1 b hb,
    read_gate_exact_on_any_coordinate s2 c hc,
    read_gate_exact_on_any_coordinate s3 d hd⟩

end BPD.PD07

#print axioms BPD.PD07.planted_active_gate
#print axioms BPD.PD07.planted_inactive_gate
#print axioms BPD.PD07.read_gate_exact_on_any_coordinate
#print axioms BPD.PD07.read_gate_exact_on_any_four_coordinate_support
#print axioms BPD.PD07.arbitrary_four_coordinate_subset_including_heldout
