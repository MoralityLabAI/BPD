"""Source-only engineering tests on excluded seed 901."""
import unittest
import torch
from mechinterp.bpd.theory.sf04_source_binding.experiment import (
    COMPONENTS, H, IN, Factors, Teacher, sample, paired_source_swap,
    component_writes, intervene, tangent_matched, rng,
)


class SF04SourceRules(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_source_last_write_wins(self):
        first, second, query, target, data = sample(91127, 1024)
        s1, s2 = data["source1"], data["source2"]
        v1, v2 = data["value1"], data["value2"]
        role = data["role"]
        for i in range(len(role)):
            r = int(role[i])
            desired = data["public"][i] if r == 2 else (
                v2[i] if r == int(s2[i]) else
                (v1[i] if r == int(s1[i]) else v1.new_zeros(()))
            )
            self.assertAlmostEqual(float(target[i]), float(desired), places=7)

    def test_donor_flips_only_second_message_source(self):
        record, donor, donor_target = paired_source_swap(90123, 100)
        first, second, query, target, meta = record
        self.assertTrue(torch.equal(donor[:, 2:], second[:, 2:]))
        self.assertTrue(torch.equal(donor[:, :2], 1-second[:, :2]))
        public = meta["role"] == 2
        self.assertTrue(torch.equal(donor_target[public], target[public]))

    def test_full_component_write_matches_physical_factor_product(self):
        teacher, factor = Teacher(901), Factors(901)
        record, donor, _ = paired_source_swap(90195, 32)
        h0 = torch.zeros(32, H)
        h1 = teacher.step(h0, record[0])
        writes = component_writes(teacher, factor, h1, record[1], donor)
        expected = (teacher.features(h1, donor) -
                    teacher.features(h1, record[1])) @ factor.weight()
        self.assertTrue(torch.allclose(writes.sum(1), expected, atol=1e-6))
        self.assertEqual(tuple(writes.shape), (32, COMPONENTS, H))

    def test_dark_projection_keeps_current_readout_exact(self):
        teacher, factor = Teacher(901), Factors(901)
        record, donor, _ = paired_source_swap(90219, 50)
        h0 = torch.zeros(50, H)
        h1 = teacher.step(h0, record[0])
        h2 = teacher.step(h1, record[1])
        writes = component_writes(teacher, factor, h1, record[1], donor)
        out, delta = intervene(teacher, h2, record[2], writes, [0, 1, 5])
        self.assertTrue(torch.equal((h2+delta)[:, 0], h2[:, 0]))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_signed_tangent_sham_preserves_norm_and_response(self):
        g = rng(90127)
        delta = torch.randn(96, H, generator=g)
        grad = torch.randn(96, H, generator=g)
        delta[:, 0] = 0
        sham, ne, te = tangent_matched(delta, grad, g)
        self.assertLess(ne, 1e-5)
        self.assertLess(te, 1e-5)
        self.assertTrue(torch.equal(sham[:, 0], torch.zeros(96)))

    def test_full_operator_gauge_invariant(self):
        factor = Factors(901)
        R = torch.eye(COMPONENTS)
        R[0,1] = .7
        a = factor.weight()
        b = (factor.V@R)@torch.linalg.solve(R,factor.U)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))


if __name__ == "__main__":
    unittest.main()
