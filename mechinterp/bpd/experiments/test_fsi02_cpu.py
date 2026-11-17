"""FSI-02 controlled finite-basis source and prospective world-replay checks."""
import json,os,unittest
from fractions import Fraction as Q
from pathlib import Path
from mechinterp.bpd.experiments.fsi02_run import (
    DIRECTIONS,BASES,law,true_l0,exact_minimizers,score_world,run)
ROOT=Path(__file__).resolve().parents[3]
REG=ROOT/'mechinterp/bpd/experiments/fsi02_registration_v1.json'
ARCHIVE=ROOT/'mechinterp/bpd/experiments/fsi02_cpu_world_receipt.json'

class FSI02MathTest(unittest.TestCase):
    def test_rotated_unique_optimum(self):
        for seed in (2026120101,2026120107,2026120116):
            self.assertEqual(exact_minimizers(law(seed,'rotated')),((2,3),))
    def test_cancellation_has_exactly_three_optima(self):
        for seed in (2026120117,2026120125,2026120132):
            self.assertEqual(set(exact_minimizers(law(seed,'cancellation'))),{(0,1),(0,3),(1,3)})
    def test_all_basis_candidates_are_independent(self):
        self.assertEqual(len(BASES),10)
    def test_mask_whole_block_and_singleton_witness(self):
        row=score_world(2026120121,'cancellation',64,256)
        self.assertTrue(row['mask']['singleton_not_invariant'])
        self.assertTrue(row['mask']['full_block_invariant'])

@unittest.skipUnless(os.getenv('EVALMATH_FSI_FILE') and Path(os.environ.get('EVALMATH_FSI_FILE','')).is_file(),
                     'EvalMath pinned-source CI supplies the FSI module')
class FSI02ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected=json.loads(ARCHIVE.read_text())
        cls.actual=run(json.loads(REG.read_text()))
    def test_exact_worlds_replay(self):
        self.assertEqual(self.actual['worlds'],self.expected['worlds'])
        self.assertEqual(len(self.actual['worlds']),32)
    def test_inference_replays(self):
        self.assertEqual(json.loads(json.dumps(self.actual['p1'])),self.expected['p1'])
        self.assertEqual(self.actual['p2'],self.expected['p2'])
        self.assertEqual(self.actual['mask'],self.expected['mask'])
    def test_optimality_and_scope(self):
        self.assertEqual(self.actual['p2']['exact_optimal_worlds'],32)
        self.assertEqual(self.actual['mask']['cancellation_worlds_with_singleton_counterexample'],16)
        self.assertIn('ARG03 protected test untouched',self.actual['scope'])
    def test_world_count_and_significance_conditional(self):
        self.assertEqual(self.actual['world_count'],32)
        self.assertEqual(Q(self.actual['p1']['effect']),Q(1773,4096))
        self.assertEqual(Q(self.actual['p1']['signflip_p']),Q(1,20001))

if __name__=='__main__':unittest.main()
