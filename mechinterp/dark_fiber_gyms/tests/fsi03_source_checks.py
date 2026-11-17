"""Synthetic source-contract tests for the published native TRM receipt evaluator."""
from __future__ import annotations
import json,os,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fsi03_native_audit import audit,FILES,ANGLES,case_value


def records():
    ids=[f'board-{i:03}' for i in range(120)]
    roster={'n_eval':120,'rows':[{'case_id':x,'stratum':'WB-Prop'} for x in ids]}
    out={'roster':roster}
    for model in ('gaoxin','yagiz'):
        rows=[]
        for i,case_id in enumerate(ids):
            tests={}
            for ang in ANGLES:
                def direction(kind):
                    good=(kind=='high_eig' and i<10) or (kind=='random_dir' and 10<=i<14)
                    return {'exact_solve_at_h12':bool(good),'board_flips_vs_clean_at_h12':int(good),
                      'lag12_readout_l2':float(i+(3 if kind=='high_eig' else 0))}
                tests['angle_'+ang]={'directions':{a:direction(a) for a in ('high_eig','low_eig','random_dir')}}
            rows.append({'case_id':case_id,'causal_direction_tests':tests})
        out[model]={'mode':'gramian','checkpoint_sha256':'a'*64,'rows':rows}
    return out


class TestFSI03(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        for name,path in FILES.items():
            dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(records()[name]))
        self.root=root

    def test_root_level_family_and_exact_discordances(self):
        r=audit(self.root)
        self.assertEqual(r['independent_units_declared'],120)
        self.assertEqual(r['family']['size'],16)
        self.assertTrue(r['repeat_checkpoints_are_not_extra_roots'])
        x=r['binary_comparisons'][0]
        self.assertEqual((x['wins'],x['losses'],x['ties']),(10,4,106))
        self.assertEqual(x['effect_pp'],5.0)
        self.assertEqual(r['recorded_checkpoint_contrasts']['gaoxin']['angles']['0.04']['high_minus_random_mean_lag12'],3.0)
        self.assertFalse(r['family']['any_beat_10pp'])

    def test_scope_and_provenance(self):
        r=audit(self.root)
        self.assertEqual(set(r['source_sha256']),{'roster','gaoxin','yagiz'})
        self.assertIn('not read',r['heldout_seal'])
        self.assertTrue(all('holm_exact' in x for x in r['binary_comparisons']))

    def test_mismatched_root_ids_fail(self):
        file=self.root/FILES['gaoxin']
        doc=json.loads(file.read_text())
        doc['rows'][3]['case_id']='alien'
        file.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,'roster mismatch'):
            audit(self.root)

    def test_exact_vs_any_flip_are_distinct_metrics(self):
        d={'exact_solve_at_h12':False,'board_flips_vs_clean_at_h12':10}
        self.assertEqual(case_value(d,'exact_solve_at_h12'),0)
        self.assertEqual(case_value(d,'board_any_flip'),1)

if __name__=='__main__':unittest.main()
