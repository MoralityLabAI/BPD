"""Independent posthoc numerical checks for PD09 smoothed read landscape."""
from __future__ import annotations

import unittest

from mechinterp.bpd.theory.pd09.landscape_audit import (
    WEIGHTED_SUPPORTS, exact_support_weights,
    kernel_pair_mass, difference, crossing_bandwidth, audit,
)


class TestPD09Landscape(unittest.TestCase):
    def test_exact_source_support_frequencies(self):
        self.assertEqual(sum(w for _,w in WEIGHTED_SUPPORTS),36)
        for pair in ((0,1),(1,2)):
            m=exact_support_weights(pair)
            self.assertEqual(m["p00"],9)
            self.assertEqual(m["p11"],11)
            self.assertEqual(m["z_i"]+m["z_j"],34)
            self.assertEqual(m["z_diagonal"],9)
            self.assertEqual(m["sharp_coordinate_advantage_numerator"],16)

    def test_smoothing_prefers_pair_mix_initially_but_coordinates_at_end(self):
        for pair in ((0,1),(1,2)):
            self.assertAlmostEqual(difference(pair,.6),.0439879533,places=5)
            self.assertAlmostEqual(difference(pair,.3),-.1098542630,places=5)
            self.assertAlmostEqual(difference(pair,.15),-.2284493031,places=5)
            self.assertTrue(.429 < crossing_bandwidth(pair) < .432)

    def test_diagonal_remains_a_local_near_zero_mass_maximum(self):
        for pair in ((0,1),(1,2)):
            for sigma in (.6,.3,.15):
                central=kernel_pair_mass(pair,45,sigma)
                self.assertGreater(central,kernel_pair_mass(pair,44,sigma))
                self.assertGreater(central,kernel_pair_mass(pair,46,sigma))
            # The initial bandwidth prefers the diagonal, while the final
            # bandwidth has the exact coordinate direction as a better
            # solution. Local diagonal criticality nevertheless survives.
            self.assertGreater(difference(pair,.6),0)
            self.assertLess(difference(pair,.15),0)

    def test_audit_retains_posthoc_scope(self):
        r=audit()
        self.assertEqual(r["study_id"],"BPD-PD-09")
        self.assertIn("posthoc",r["scope"])
        self.assertEqual(len(r["cases"]),2)
        self.assertEqual(r["quadrature_n"],128)


if __name__=="__main__":
    unittest.main()
