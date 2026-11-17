"""Source-level and independent generator tests for AMM-01."""
import math
import unittest
from . import experiment as e
from .bridge import buzz_message


class AMM01Tests(unittest.TestCase):
    def test_rank2_current_dark_same_physical_dose(self):
        d = (0., 1/math.sqrt(2), -1/math.sqrt(2), 0.)
        self.assertAlmostEqual(sum(x*x for x in d), 1.0, places=14)
        self.assertEqual(d[0], 0.0)
        for i in range(20):
            w = e.world("train", i)
            m = e.operator(w)
            self.assertEqual(len(m), 4)
            self.assertTrue(all(x == 0.0 for x in m[1]))
            self.assertTrue(all(x == 0.0 for x in m[2]))
            self.assertAlmostEqual(e.exact_probe(w, 1), e.matvec(m, d)[0], places=14)

    def test_prespecified_late_observability(self):
        for i in range(30):
            w = e.world("train", i)
            if w.family == "wren":
                self.assertAlmostEqual(e.exact_probe(w, 1), math.sqrt(2)*w.sign*w.gain)
                self.assertEqual(e.exact_probe(w, 2), 0.0)
            else:
                self.assertAlmostEqual(e.exact_probe(w, 1), math.sqrt(2)*w.sign*e.EPSILON)
                self.assertAlmostEqual(e.exact_probe(w, 2), math.sqrt(2)*w.sign*w.gain)

    def test_train_dev_test_disjoint(self):
        sets = [{e.world(phase, i).root for i in range(n)} for phase,n in
                (("train", e.TRAIN_ROOTS), ("development", e.DEV_ROOTS), ("confirmation", e.CONFIRM_ROOTS))]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(len(sets[2]), 512)

    def test_policy_cannot_see_world_label(self):
        m = e.learn_memory()
        self.assertEqual(m["source_partition"], "train")
        self.assertEqual(m["sha256"], e.digest({k:v for k,v in m.items() if k != "sha256"}))
        self.assertEqual(set(m["family_horizon"]), set(e.FAMILIES))
        self.assertTrue(all(h in (1,2) for h in m["family_horizon"].values()))
        for phase in ("train", "development", "confirmation"):
            w = e.world(phase, 0)
            self.assertEqual(e.choose_horizon("memory", w.family, m, w.root, 1),
                             e.choose_horizon("memory", w.family, m, w.root, 2))
        self.assertNotIn("latent_sign", m)

    def test_one_query_each_arm_each_case(self):
        w = e.world("development", 0)
        r = e.evaluate_root(w, e.learn_memory())
        for a in ("memory","pooled","swapped","random","sham"):
            self.assertEqual(sum(r["horizon_counts"][a]), e.CASES)
            self.assertEqual(len(r["case_bits"][a]), e.CASES)
            self.assertEqual(r["hits"][a], r["case_bits"][a].count("1"))
        self.assertEqual(e.observations(w, 4), e.observations(w, 4))

    def test_evalmath_root_preflight_shape(self):
        data = e.cohort("development")
        spec = e.preflight_spec(data)
        self.assertEqual(spec["planned_comparisons"], 3)
        self.assertEqual(spec["confirm_units_planned"], 512)
        self.assertEqual(len(spec["pilot"]), 64)
        self.assertEqual(len({r["unit_id"] for r in spec["pilot"]}), 64)
        self.assertEqual(spec["scientific_min_effect"], "0.15")

    def test_one_sided_exact_and_holm(self):
        self.assertEqual(e.sign_p_one_sided([0,0]), 1)
        self.assertAlmostEqual(e.sign_p_one_sided([1,1,1,-1]), 5/16)
        adj = e.holm({"a":0.01, "b":0.02, "c":0.5})
        self.assertEqual(adj, {"a":0.03, "b":0.04, "c":0.5})

    def test_confirmation_must_have_preflight(self):
        with self.assertRaises(ValueError):
            e.bundle("confirmation")
        with self.assertRaises(ValueError):
            e.analyze({"phase":"development"}, {"decision":"READY_FOR_CONFIRMATION"})

    def test_buzz_requires_uuid_and_confirm_receipt(self):
        dev = e.cohort("development")
        with self.assertRaises(ValueError):
            buzz_message(dev,"invalid","https://github.com/MoralityLabAI/BPD/pull/1")
        with self.assertRaises(ValueError):
            buzz_message(dev,"00000000-0000-4000-8000-000000000001","https://github.com/MoralityLabAI/BPD/pull/1")
        receipt = {"experiment":e.ID,"phase":"confirmation","independent_units":512,
                   "analysis":{"evidence_class":"synthetic", "primary_gate_pass":True,
                               "contrasts":{"memory_vs_pooled":{"mean_gain":0.17}}}}
        event = buzz_message(receipt,"00000000-0000-4000-8000-000000000001", "https://github.com/MoralityLabAI/BPD/pull/1")
        self.assertEqual(set(event), {"kind","tags","content"})
        self.assertEqual(event["kind"],9)
        self.assertEqual(event["tags"][0][0],"h")
        self.assertNotIn("latent_sign",event["content"])
        self.assertNotIn("sig",event)


if __name__ == "__main__":
    unittest.main()
