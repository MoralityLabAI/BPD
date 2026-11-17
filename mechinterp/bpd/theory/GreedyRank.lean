import Lean

/-!
Greedy rank preservation for an abstract finite rank oracle.

The four assumptions are the familiar vector-matroid rank laws in a form
suited to a sequential scan: monotonicity, an increment of at most one for
one added candidate, and closure compatibility when a chosen list lies
inside the already processed list and has the same rank.

This file kernel-proves that greedy chooses an independent subset which
has exactly the rank of each processed prefix. For a list sorted by the
hyperplane-mass weight, these prefixes are the threshold sets; combine this
with FiniteFiltration.lean for optimality. Deriving the abstract rank laws
from a concrete matrix over R is an additional formalization obligation.
-/

namespace BPD.GreedyRank

def Included {α : Type} (xs ys : List α) : Prop :=
  ∀ x, x ∈ xs → x ∈ ys

def choose {α : Type} (rank : List α → Nat)
    (selected : List α) (x : α) : List α :=
  if rank (x :: selected) > rank selected then x :: selected else selected

def scan {α : Type} (rank : List α → Nat)
    (selected processed : List α) : List α → List α × List α
  | [] => (selected, processed)
  | x :: xs => scan rank (choose rank selected x) (x :: processed) xs

theorem choose_preserves_rank_and_independence {α : Type}
    (rank : List α → Nat)
    (hmono : ∀ (x : α) S, rank S ≤ rank (x :: S))
    (hunit : ∀ (x : α) S, rank (x :: S) ≤ rank S + 1)
    (hclosure : ∀ (S P : List α) (x : α),
      Included S P → rank S = rank P →
      rank (x :: S) = rank (x :: P))
    (S P : List α) (x : α)
    (heq : rank S = rank P)
    (hind : S.length = rank S)
    (hsub : Included S P) :
    rank (choose rank S x) = rank (x :: P) ∧
    (choose rank S x).length = rank (choose rank S x) ∧
    Included (choose rank S x) (x :: P) := by
  have hc := hclosure S P x hsub heq
  by_cases h : rank (x :: S) > rank S
  · have hlen : (x :: S).length = rank (x :: S) := by
      have hup := hunit x S
      simp only [List.length_cons]
      omega
    have hsub' : Included (x :: S) (x :: P) := by
      intro a ha
      simp only [List.mem_cons] at ha ⊢
      rcases ha with rfl | has
      · exact Or.inl rfl
      · exact Or.inr (hsub a has)
    simpa [choose, h] using And.intro hc (And.intro hlen hsub')
  · have hmon := hmono x S
    have hsame : rank (x :: S) = rank S := by omega
    have heq' : rank S = rank (x :: P) := by omega
    have hsub' : Included S (x :: P) := by
      intro a ha
      simp only [List.mem_cons]
      exact Or.inr (hsub a ha)
    simpa [choose, h] using And.intro heq' (And.intro hind hsub')

theorem scan_preserves_rank_and_independence {α : Type}
    (rank : List α → Nat)
    (hmono : ∀ (x : α) S, rank S ≤ rank (x :: S))
    (hunit : ∀ (x : α) S, rank (x :: S) ≤ rank S + 1)
    (hclosure : ∀ (S P : List α) (x : α),
      Included S P → rank S = rank P →
      rank (x :: S) = rank (x :: P))
    (selected processed remaining : List α)
    (heq : rank selected = rank processed)
    (hind : selected.length = rank selected)
    (hsub : Included selected processed) :
    let result := scan rank selected processed remaining
    rank result.1 = rank result.2 ∧
    result.1.length = rank result.1 ∧
    Included result.1 result.2 := by
  induction remaining generalizing selected processed with
  | nil =>
    simpa [scan] using And.intro heq (And.intro hind hsub)
  | cons x xs ih =>
    have hnext := choose_preserves_rank_and_independence
      rank hmono hunit hclosure selected processed x heq hind hsub
    simpa [scan] using ih (choose rank selected x) (x :: processed)
      hnext.1 hnext.2.1 hnext.2.2

/-- Finite greedy scan selects an independent basis of the processed prefix. -/
theorem greedy_prefix_rank {α : Type}
    (rank : List α → Nat)
    (hzero : rank [] = 0)
    (hmono : ∀ (x : α) S, rank S ≤ rank (x :: S))
    (hunit : ∀ (x : α) S, rank (x :: S) ≤ rank S + 1)
    (hclosure : ∀ (S P : List α) (x : α),
      Included S P → rank S = rank P →
      rank (x :: S) = rank (x :: P))
    (candidates : List α) :
    let result := scan rank [] [] candidates
    rank result.1 = rank result.2 ∧
    result.1.length = rank result.1 ∧
    Included result.1 result.2 := by
  apply scan_preserves_rank_and_independence
    rank hmono hunit hclosure [] [] candidates
  · exact rfl
  · simpa [hzero]
  · intro a ha
    simp at ha

end BPD.GreedyRank

#print axioms BPD.GreedyRank.choose_preserves_rank_and_independence
#print axioms BPD.GreedyRank.scan_preserves_rank_and_independence
#print axioms BPD.GreedyRank.greedy_prefix_rank
