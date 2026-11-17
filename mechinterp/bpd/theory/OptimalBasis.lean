import Init

/-!
BPD idealized-L0 identifiability: exact finite atomic examples, invariant
intervention masks, and a geometric recurrence certificate.

Lean 4.19.0, Init only. The finite examples below are exhaustive over five
specified projective directions. The accompanying note gives the separate
analytic proof for all real projective directions, and explicitly labels the
vector-matroid greedy theorem and real-norm bound as paper proofs rather than
Lean-kernel-certified general theorems.
-/

set_option maxRecDepth 4096
set_option maxHeartbeats 1000000

namespace BPD.OptimalBasis

structure Vec2 where
  x : Int
  y : Int
  deriving Repr, DecidableEq

def dot (r a : Vec2) : Int := r.x * a.x + r.y * a.y

def det (r s : Vec2) : Int := r.x * s.y - r.y * s.x

-- Five pairwise distinct projective read directions.
def read (i : Fin 5) : Vec2 :=
  match i.val with
  | 0 => ⟨1, 0⟩
  | 1 => ⟨0, 1⟩
  | 2 => ⟨1, 1⟩
  | 3 => ⟨1, -1⟩
  | _ => ⟨1, 2⟩

def fullRank (i j : Fin 5) : Prop := det (read i) (read j) ≠ 0

-- z multiplied by two for the distribution {(1,1),(1,-1)}.
def twoAtomMass (i : Fin 5) : Nat :=
  (if dot (read i) ⟨1, 1⟩ = 0 then 1 else 0) +
  (if dot (read i) ⟨1, -1⟩ = 0 then 1 else 0)

-- Exactly one unordered optimum: the rotated (1,1),(1,-1) basis.
-- Explicit exhaustive candidate enumeration avoids a classical
-- Decidable instance for quantified propositions over Fin 5.
def candidateIndices : List (Fin 5) := [0, 1, 2, 3, 4]

def rotIsOptimal (i j : Fin 5) : Bool :=
  (det (read i) (read j) != 0) && (twoAtomMass i + twoAtomMass j == 2)

def rotExpected (i j : Fin 5) : Bool :=
  ((i == 2) && (j == 3)) || ((i == 3) && (j == 2))

theorem rotated_unique_optimal :
    candidateIndices.all (fun i =>
      candidateIndices.all (fun j =>
        rotIsOptimal i j == rotExpected i j)) = true := by
  decide

theorem rotated_reference_has_zero_mass :
    twoAtomMass 0 + twoAtomMass 1 = 0 := by
  decide

theorem rotated_optimum_has_two_mass :
    twoAtomMass 2 + twoAtomMass 3 = 2 := by
  decide

-- z multiplied by three for {(1,0),(0,1),(1,1)}.
def threeAtomMass (i : Fin 5) : Nat :=
  (if dot (read i) ⟨1, 0⟩ = 0 then 1 else 0) +
  (if dot (read i) ⟨0, 1⟩ = 0 then 1 else 0) +
  (if dot (read i) ⟨1, 1⟩ = 0 then 1 else 0)

-- Exactly three unordered optimum bases, thus six ordered choices.
def threeIsOptimal (i j : Fin 5) : Bool :=
  (det (read i) (read j) != 0) && (threeAtomMass i + threeAtomMass j == 2)

def threeExpected (i j : Fin 5) : Bool :=
  ((i == 0) && (j == 1)) || ((i == 1) && (j == 0)) ||
  ((i == 0) && (j == 3)) || ((i == 3) && (j == 0)) ||
  ((i == 1) && (j == 3)) || ((i == 3) && (j == 1))

theorem three_atom_exact_optimal_set :
    candidateIndices.all (fun i =>
      candidateIndices.all (fun j =>
        threeIsOptimal i j == threeExpected i j)) = true := by
  decide

theorem atomic_cancellation_zero_mass_for_extra_reads :
    threeAtomMass 2 = 0 ∧ threeAtomMass 4 = 0 := by
  decide

/-! For arbitrary types, relative reparameterization by an invertible r
preserves a mask exactly when the mask commutes with r. -/

theorem conjugate_mask_iff_commute {S : Type}
    (r rInv d : S → S)
    (hleft : ∀ s, rInv (r s) = s)
    (hright : ∀ s, r (rInv s) = s) :
    (∀ s, rInv (d (r s)) = d s) ↔
    (∀ s, d (r s) = r (d s)) := by
  constructor
  · intro h s
    have hs := congrArg r (h s)
    simpa only [hright] using hs
  · intro h s
    rw [h s, hleft]

/-! The surjectivity/injectivity assumptions are essential. The source
activation domain must reach the entire latent coordinate space, and the
write map must distinguish every latent vector. -/

theorem observed_edit_invariance_iff_commute
    {X S Y : Type}
    (encode : X → S) (decode : S → Y) (r rInv d : S → S)
    (hsurj : ∀ s, ∃ x, encode x = s)
    (hinj : ∀ a b, decode a = decode b → a = b)
    (hleft : ∀ s, rInv (r s) = s)
    (hright : ∀ s, r (rInv s) = s) :
    (∀ x, decode (rInv (d (r (encode x)))) = decode (d (encode x))) ↔
    (∀ s, d (r s) = r (d s)) := by
  constructor
  · intro h
    apply (conjugate_mask_iff_commute r rInv d hleft hright).mp
    intro s
    obtain ⟨x, hx⟩ := hsurj s
    rw [← hx]
    exact hinj _ _ (h x)
  · intro h x
    have hc : ∀ s, rInv (d (r s)) = d s :=
      (conjugate_mask_iff_commute r rInv d hleft hright).mpr h
    exact congrArg decode (hc (encode x))

/-! Exact 2-component shear: full-block masks are invariant, singleton
mask changes represented intervention. -/

def shear (p : Int × Int) : Int × Int := (p.1 + p.2, p.2)
def unshear (p : Int × Int) : Int × Int := (p.1 - p.2, p.2)
def keepAll (p : Int × Int) : Int × Int := p
def keepNone (_ : Int × Int) : Int × Int := (0, 0)
def keepFirst (p : Int × Int) : Int × Int := (p.1, 0)

theorem unshear_shear (p : Int × Int) : unshear (shear p) = p := by
  cases p with
  | mk x y => simp [unshear, shear]

theorem shear_unshear (p : Int × Int) : shear (unshear p) = p := by
  cases p with
  | mk x y => simp [unshear, shear]

theorem whole_block_mask_invariant (p : Int × Int) :
    unshear (keepAll (shear p)) = keepAll p := by
  exact unshear_shear p

theorem zero_block_mask_invariant (p : Int × Int) :
    unshear (keepNone (shear p)) = keepNone p := by
  simp [unshear, keepNone]

theorem singleton_mask_noninvariant :
    unshear (keepFirst (shear (0, 1))) ≠ keepFirst (0, 1) := by
  decide

/-! Algebraic geometric recurrence only. The real Lipschitz hypotheses are
spelled out in the note; this theorem formalizes the induction skeleton. -/

def geom (L : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => 1 + L * geom L n

theorem recurrence_bound (e : Nat → Nat) (L eps : Nat)
    (h0 : e 0 = 0)
    (hstep : ∀ n, e (n + 1) ≤ eps + L * e n) :
    ∀ n, e n ≤ eps * geom L n := by
  intro n
  induction n with
  | zero => simp [h0, geom]
  | succ n ih =>
    calc
      e (n + 1) ≤ eps + L * e n := hstep n
      _ ≤ eps + L * (eps * geom L n) :=
        Nat.add_le_add_left (Nat.mul_le_mul_left L ih) eps
      _ = eps * geom L (n + 1) := by
        simp [geom, Nat.mul_add, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]

end BPD.OptimalBasis

#print axioms BPD.OptimalBasis.rotated_unique_optimal
#print axioms BPD.OptimalBasis.three_atom_exact_optimal_set
#print axioms BPD.OptimalBasis.conjugate_mask_iff_commute
#print axioms BPD.OptimalBasis.observed_edit_invariance_iff_commute
#print axioms BPD.OptimalBasis.unshear_shear
#print axioms BPD.OptimalBasis.shear_unshear
#print axioms BPD.OptimalBasis.singleton_mask_noninvariant
#print axioms BPD.OptimalBasis.recurrence_bound
