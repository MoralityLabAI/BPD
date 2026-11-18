import unittest
try:
    from mechinterp.bpd.theory.cq08_nuisance_mediation.confirmation_release import binom_sign_p
except ModuleNotFoundError:
    from cq08_nuisance_mediation.confirmation_release import binom_sign_p

class ConfirmationReleaseAnalysisTests(unittest.TestCase):
    def test_boundary_exact_p_values(self):
        self.assertAlmostEqual(binom_sign_p(32,32),2**-32)
        self.assertAlmostEqual(binom_sign_p(32,23),.010030803503468633)
        self.assertAlmostEqual(binom_sign_p(32,16),.5699749670457095)
    def test_out_of_range_sign_counts(self):
        for k in (0,16,23,32):
            self.assertGreaterEqual(binom_sign_p(32,k),0.)
            self.assertLessEqual(binom_sign_p(32,k),1.)
