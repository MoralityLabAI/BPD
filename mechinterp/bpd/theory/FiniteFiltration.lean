import Lean

/-!
# Finite maximum-weight basis filtration, conditional on the greedy witness

Any finite vector matroid has a greedy basis that simultaneously attains
the maximum rank at each hyperplane-mass threshold. That matroid existence
theorem remains a separate proof obligation for the linear independence
predicate. The result below kernel-checks the entire general numeric
optimality iff threshold-saturation equivalence once the witness exists.

The levelRanks vector stores ranks of all positive integer mass thresholds.
A profile stores the count of rows of a basis at each threshold. Summing
the profile gives the integer-scaled maximum-weight basis score.

The greedy-witness premise is explicit, so the general vector-matroid
independence theorem is not silently claimed as Lean-verified.
-/

namespace BPD.FiniteFiltration

def Bounded : List Nat → List Nat → Prop
  | [], [] => True
  | x :: xs, y :: ys => x ≤ y ∧ Bounded xs ys
  | _, _ => False

theorem bounded_sum_le {counts ranks : List Nat}
    (h : Bounded counts ranks) : counts.sum ≤ ranks.sum := by
  induction counts generalizing ranks with
  | nil =>
    cases ranks with
    | nil => simp
    | cons y ys => simp [Bounded] at h
  | cons x xs ih =>
    cases ranks with
    | nil => simp [Bounded] at h
    | cons y ys =>
      have hxy : x ≤ y := h.1
      have htail : Bounded xs ys := h.2
      have ht := ih htail
      simp only [List.sum_cons]
      omega

theorem bounded_equal_of_sum_equal {counts ranks : List Nat}
    (h : Bounded counts ranks)
    (he : counts.sum = ranks.sum) : counts = ranks := by
  induction counts generalizing ranks with
  | nil =>
    cases ranks with
    | nil => rfl
    | cons y ys => simp [Bounded] at h
  | cons x xs ih =>
    cases ranks with
    | nil => simp [Bounded] at h
    | cons y ys =>
      have hxy : x ≤ y := h.1
      have htail : Bounded xs ys := h.2
      have ht := bounded_sum_le htail
      have he0 : x + xs.sum = y + ys.sum := by simpa using he
      have heHead : x = y := by omega
      have heTail : xs.sum = ys.sum := by omega
      have hxs : xs = ys := ih htail heTail
      simp [heHead, hxs]

structure RankedCandidates (β : Type) where
  bases : List β
  levelRanks : List Nat
  profile : β → List Nat
  upperBound : ∀ b, b ∈ bases → Bounded (profile b) levelRanks
  greedy : β
  greedyMember : greedy ∈ bases
  greedySaturates : profile greedy = levelRanks

def RankedCandidates.score {β : Type} (F : RankedCandidates β) (b : β) : Nat :=
  (F.profile b).sum

/--
If a finite basis family has a simultaneous rank-saturating greedy basis,
a basis maximizes its integer weight score iff it saturates every level.
No dimension-specific enumeration is used.
-/
theorem optimal_iff_saturates {β : Type} (F : RankedCandidates β)
    (b : β) (hb : b ∈ F.bases) :
    (∀ c, c ∈ F.bases → F.score c ≤ F.score b) ↔
      F.profile b = F.levelRanks := by
  constructor
  · intro hoptimal
    have hgreedy := hoptimal F.greedy F.greedyMember
    have hmax : F.levelRanks.sum ≤ (F.profile b).sum := by
      simpa only [RankedCandidates.score, F.greedySaturates] using hgreedy
    have hup := bounded_sum_le (F.upperBound b hb)
    exact bounded_equal_of_sum_equal (F.upperBound b hb)
      (Nat.le_antisymm hup hmax)
  · intro hprofile c hc
    have hle := bounded_sum_le (F.upperBound c hc)
    simpa only [RankedCandidates.score, hprofile] using hle

theorem best_score_is_rank_integral {β : Type}
    (F : RankedCandidates β) :
    F.score F.greedy = F.levelRanks.sum := by
  simp [RankedCandidates.score, F.greedySaturates]

/-- Every maximizing basis has the same threshold signature. -/
theorem equal_profiles_of_optimal {β : Type} (F : RankedCandidates β)
    (a b : β) (ha : a ∈ F.bases) (hb : b ∈ F.bases)
    (haMax : ∀ c, c ∈ F.bases → F.score c ≤ F.score a)
    (hbMax : ∀ c, c ∈ F.bases → F.score c ≤ F.score b) :
    F.profile a = F.profile b := by
  have hae := (optimal_iff_saturates F a ha).mp haMax
  have hbe := (optimal_iff_saturates F b hb).mp hbMax
  exact hae.trans hbe.symm

end BPD.FiniteFiltration

#print axioms BPD.FiniteFiltration.bounded_sum_le
#print axioms BPD.FiniteFiltration.bounded_equal_of_sum_equal
#print axioms BPD.FiniteFiltration.optimal_iff_saturates
#print axioms BPD.FiniteFiltration.equal_profiles_of_optimal
