import Lean

/-!
# BPD EFD-01: rank-budget certificate for an energy-tangent dark fiber

This file verifies the *rank arithmetic* of the physical energy/head tangent
intersection. The linear-algebra fact that stacking ONE scalar-energy row onto
a present-readout matrix increases rank by at most one is an explicit premise.
No Mathlib vector-space/nullspace theorem is silently claimed.

Let d be the physical state width, r the present observable Jacobian rank,
and s the stacked present-observable/energy-gradient rank. Under s <= r+1,
dim K = d-s and dim ker(Dp)=d-r, provided s<=d and r<=d.

Source paper: BPD-EFD-01 / LOOPSCOPE-10.
-/

namespace BPD.EnergyFiberRank

structure RankLedger where
  dim : Nat
  presentRank : Nat
  jointRank : Nat
  futureRestrictedRank : Nat
  presentBound : presentRank ≤ dim
  jointBound : jointRank ≤ dim
  oneScalarRow : jointRank ≤ presentRank + 1
  futureBound : futureRestrictedRank ≤ dim - jointRank

def presentDarkDimension (L : RankLedger) : Nat :=
  L.dim - L.presentRank

def energyDarkDimension (L : RankLedger) : Nat :=
  L.dim - L.jointRank

/-- Adding one scalar energy constraint loses at most one dark dimension. -/
theorem energy_fiber_loses_at_most_one_dimension (L : RankLedger) :
    L.presentDarkDimension ≤ L.energyDarkDimension + 1 := by
  rcases L with ⟨dim, presentRank, jointRank, futureRestrictedRank,
    hPresent, hJoint, hRow, hFuture⟩
  simp only [presentDarkDimension, energyDarkDimension]
  omega

/-- Equivalent lower bound for the present-and-energy-dark joint kernel. -/
theorem energy_dark_dimension_lower_bound (L : RankLedger) :
    L.presentDarkDimension - 1 ≤ L.energyDarkDimension := by
  have h := energy_fiber_loses_at_most_one_dimension L
  omega

/-- The delayed observable rank cannot exceed the physical fiber dimension.
This uses the explicit futureRestrictedRank bound recorded in the ledger. -/
theorem future_rank_le_energy_dark_dimension (L : RankLedger) :
    L.futureRestrictedRank ≤ L.energyDarkDimension := by
  exact L.futureBound

/-- A strictly positive delayed rank implies the joint kernel has capacity. -/
theorem delayed_rank_positive_implies_nonempty_fiber
    (L : RankLedger) (h : 0 < L.futureRestrictedRank) :
    0 < L.energyDarkDimension := by
  have hrank := future_rank_le_energy_dark_dimension L
  omega

/-- A positive delayed rank requires at least one physical state degree
outside the joint constraint rank. -/
theorem positive_delayed_rank_requires_spare_dimension
    (L : RankLedger) (h : 0 < L.futureRestrictedRank) :
    L.jointRank < L.dim := by
  have hd := delayed_rank_positive_implies_nonempty_fiber L h
  dsimp [energyDarkDimension] at hd
  omega

end BPD.EnergyFiberRank

#print axioms BPD.EnergyFiberRank.energy_fiber_loses_at_most_one_dimension
#print axioms BPD.EnergyFiberRank.energy_dark_dimension_lower_bound
#print axioms BPD.EnergyFiberRank.future_rank_le_energy_dark_dimension
#print axioms BPD.EnergyFiberRank.delayed_rank_positive_implies_nonempty_fiber
#print axioms BPD.EnergyFiberRank.positive_delayed_rank_requires_spare_dimension
