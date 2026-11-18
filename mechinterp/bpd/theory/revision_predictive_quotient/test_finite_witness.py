"""Source-only independent witness and delayed source quotient tests."""
import sys
import unittest
from pathlib import Path
from itertools import combinations
sys.path.insert(0,str(Path(__file__).resolve().parent))
from finite_witness import audit,signature,old_posterior,revised_posterior,witness


class BPDRevisionWitnessTests(unittest.TestCase):
    def test_four_bits_all_future_classes(self):
        out=audit(4,2)
        self.assertEqual(out["old_histories"],16)
        self.assertEqual(out["present_labels"],5)
        self.assertEqual(out["future_classes"],16)
        self.assertGreater(out["checked_present_collision_pairs"],0)
        self.assertFalse(out["neural_mechanism_tested"])

    def test_complete_small_family(self):
        for n in range(2,6):
            for j in range(1,n):
                self.assertEqual(len({signature(s,n,j) for s in range(1<<n)}),1<<n)
                for a,b in combinations(range(1<<n),2):
                    if a.bit_count()!=b.bit_count():continue
                    self.assertEqual(old_posterior(a.bit_count(),n,j),
                                     old_posterior(b.bit_count(),n,j))
                    event=witness(a,b,n,j)
                    self.assertEqual(len(event),j)
                    self.assertNotEqual(revised_posterior(a,n,event),
                                        revised_posterior(b,n,event))

    def test_current_equivalence_delayed_failure(self):
        a,b=0b0011,0b0101
        self.assertEqual(a.bit_count(),b.bit_count())
        e=witness(a,b,4,2)
        self.assertNotEqual(signature(a,4,2),signature(b,4,2))
        self.assertNotEqual(revised_posterior(a,4,e),revised_posterior(b,4,e))

    def test_invalid(self):
        for fn in (lambda:witness(1,1,4,2),lambda:witness(1,2,4,0),
                   lambda:witness(1,3,4,2),lambda:signature(-1,4,2),
                   lambda:old_posterior(1,4,4)):
            with self.assertRaises(ValueError):fn()


if __name__=="__main__":unittest.main()
