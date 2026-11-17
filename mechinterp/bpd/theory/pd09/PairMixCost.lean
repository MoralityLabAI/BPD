import Init

/-!
PD09 source-level exact-L0 distinction between canonical coordinate reads and
two-feature diagonal read mixtures.

The finite structural support weights come from the unchanged PD06 source.
When independent active amplitudes have continuous densities, the mixed
direction e_i +/- e_j is exactly zero only when both latent coordinates are
inactive. This conditional measure argument is stated in the written note;
the kernel checks below certify the finite support combinatorics.

The numerical local maxima of the SMOOTHED Gaussian near-zero kernel are
checked separately by deterministic Gaussian quadrature in landscape_audit.py.
-/

namespace BPD.PD09

def weightedSupports : List (List Nat × Nat) :=
  [([0],3),([1],3),([2],3),([3],3),
   ([0,1],3),([1,2],3),([2,3],3),([0,3],3),
   ([0,1,2],4),([0,1,3],4),([1,2,3],4)]

def structuralZeroMass (readSupport : List Nat) : Nat :=
  weightedSupports.foldl
    (fun acc entry =>
      if readSupport.any (fun i => entry.1.contains i)
      then acc else acc + entry.2) 0

theorem original_source_mass_total :
    (weightedSupports.map Prod.snd).sum = 36 := by
  decide

theorem adjacent_01_sharp_masses :
    structuralZeroMass [0] = 19 ∧
    structuralZeroMass [1] = 15 ∧
    structuralZeroMass [0,1] = 9 := by
  decide

theorem adjacent_12_sharp_masses :
    structuralZeroMass [1] = 15 ∧
    structuralZeroMass [2] = 19 ∧
    structuralZeroMass [1,2] = 9 := by
  decide

theorem canonical_01_has_sixteen_more_zero_mass_units :
    structuralZeroMass [0] + structuralZeroMass [1] =
      2 * structuralZeroMass [0,1] + 16 := by
  decide

theorem canonical_12_has_sixteen_more_zero_mass_units :
    structuralZeroMass [1] + structuralZeroMass [2] =
      2 * structuralZeroMass [1,2] + 16 := by
  decide

theorem sharp_expected_l0_prefers_latent_axes_01 :
    2 * structuralZeroMass [0,1] <
      structuralZeroMass [0] + structuralZeroMass [1] := by
  decide

theorem sharp_expected_l0_prefers_latent_axes_12 :
    2 * structuralZeroMass [1,2] <
      structuralZeroMass [1] + structuralZeroMass [2] := by
  decide

end BPD.PD09

#print axioms BPD.PD09.adjacent_01_sharp_masses
#print axioms BPD.PD09.adjacent_12_sharp_masses
#print axioms BPD.PD09.canonical_01_has_sixteen_more_zero_mass_units
#print axioms BPD.PD09.canonical_12_has_sixteen_more_zero_mass_units
#print axioms BPD.PD09.sharp_expected_l0_prefers_latent_axes_01
