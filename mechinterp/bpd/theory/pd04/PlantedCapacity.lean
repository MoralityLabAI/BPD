import Init

/-!
# PD04 planted-coordinate capacity certificates

This concerns the explicit 2D identity teacher with two planted coordinate
rank-one writes. It is a scope-limited exact integer certificate, not a
theorem about a trained overcomplete dictionary or arbitrary nonlinear gate.
-/

namespace BPD.PD04

def coordinateWrite (enable0 enable1 : Bool) (a b : Int) : Int × Int :=
  (if enable0 then a else 0, if enable1 then b else 0)

def top1Write (choose0 : Bool) (a b : Int) : Int × Int :=
  if choose0 then coordinateWrite true false a b
  else coordinateWrite false true a b

/-- An overlap with two nonzero coordinates cannot be represented by either
planted one-hot coordinate write. -/
theorem top1_cannot_reconstruct_overlap
    (a b : Int) (ha : a ≠ 0) (hb : b ≠ 0)
    (choose0 : Bool) :
    top1Write choose0 a b ≠ (a, b) := by
  cases choose0 with
  | false =>
      intro h
      have hzero : (0 : Int) = a := congrArg Prod.fst h
      exact ha hzero.symm
  | true =>
      intro h
      have hzero : (0 : Int) = b := congrArg Prod.snd h
      exact hb hzero.symm

/-- Both planted factors recover a two-coordinate teacher exactly. -/
theorem two_writes_reconstruct_overlap (a b : Int) :
    coordinateWrite true true a b = (a, b) := by
  rfl

/-- On a single-support input, the inactive write can be switched on without
affecting the output. Consequently exact reconstruction is insufficient to
identify the minimum support pattern. -/
theorem redundant_second_gate_on_single_support (a : Int) :
    coordinateWrite true true a 0 =
    coordinateWrite true false a 0 := by
  rfl

theorem redundant_first_gate_on_single_support (b : Int) :
    coordinateWrite true true 0 b =
    coordinateWrite false true 0 b := by
  rfl

/-- The mixed cardinality nested case requires one coordinate write in the
first phase and both writes in the overlapping second phase. -/
theorem nested_support_reconstructs (a b c d : Int) :
    coordinateWrite true false a 0 = (a, 0) ∧
    coordinateWrite true false b 0 = (b, 0) ∧
    coordinateWrite true true c d = (c, d) := by
  exact ⟨rfl, rfl, rfl⟩

end BPD.PD04

#print axioms BPD.PD04.top1_cannot_reconstruct_overlap
#print axioms BPD.PD04.two_writes_reconstruct_overlap
#print axioms BPD.PD04.redundant_second_gate_on_single_support
#print axioms BPD.PD04.redundant_first_gate_on_single_support
#print axioms BPD.PD04.nested_support_reconstructs
