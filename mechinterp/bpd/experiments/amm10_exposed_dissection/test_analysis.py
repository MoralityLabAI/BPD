"""Source-only tests for the AMM10 exposed-root analyzer.

These fixtures never access a scientific root, a development feature matrix,
or any confirmation source. No physical models are trained.
"""
from __future__ import annotations
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np
import analysis as a


class AMM10Tests(unittest.TestCase):
    def test_immutable_six_mask_space(self):
        self.assertEqual(a.CANDS,tuple(f"mask_{i}" for i in range(6)))
        self.assertEqual(a.N_FEATURES if hasattr(a,"N_FEATURES") else 18,18)
        self.assertNotIn("mask_6",a.CANDS)
        self.assertEqual(len(a.VARIANTS),9)
        self.assertEqual(len(a.HEURISTICS),6)

    def test_protected_root_guard_and_membership(self):
        with self.assertRaises(ValueError):
            a.members([{"root":"confirmation/0000","family":"wren"}],"development",64)
        with self.assertRaises(ValueError):
            a.members([{"root":"development/0000","family":"wren"}]*64,"development",64)
        rows=[{"root":f"development/{i:04d}","family":a.FAMILIES[i%2]} for i in range(64)]
        self.assertIs(a.members(rows,"development",64),rows)
        rows[31]["family"]="wren"
        with self.assertRaises(ValueError):a.members(rows,"development",64)

    def test_raw_hash_integrity(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"fixture.json";p.write_text("{}")
            with self.assertRaises(ValueError):a.raw_json(p,a.TRAIN_SHA)

    def test_frozen_feature_variant_columns(self):
        self.assertEqual(a.VARIANTS["full18"],tuple(range(18)))
        self.assertNotIn(17,a.VARIANTS["no_family"])
        self.assertNotIn(14,a.VARIANTS["no_alignment"])
        self.assertTrue(all(k in a.VARIANTS["compact_causal"] for k in (0,1,5,6,7,8,9,10,11,14)))

    def test_analytic_rank_identity_candidate_constant(self):
        # Source-like feature matrix, candidate-independent base readouts.
        rng=np.random.default_rng(1515)
        x=np.zeros((3,6,18),dtype=np.float64)
        x[:,:,0]=1
        x[:,:,1]=np.array([1,1,1,2,2,2])/3
        for i in (2,3,4,15,16,17):
            x[:,:,i]=rng.normal(size=(3,1))
        d2=rng.normal(size=(3,6))
        d3=rng.normal(size=(3,6))
        base2=rng.normal(size=(3,1))
        base3=rng.normal(size=(3,1))
        x[:,:,5:7]=np.stack((d2,d3),axis=2)
        x[:,:,7]=abs(d2);x[:,:,8]=abs(d3)
        x[:,:,9]=d2*d2;x[:,:,10]=d3*d3;x[:,:,11]=d2*d3
        x[:,:,12]=base2+d2;x[:,:,13]=base3+d3
        x[:,:,14]=rng.uniform(-1,1,size=(3,6))
        beta=np.linspace(-.1,.25,18)
        result=a.score_identity(x,beta)
        self.assertEqual(result["preclip_action_mismatch_count"],0)
        self.assertEqual(result["postclip_action_mismatch_count"],0)
        self.assertLess(result["max_preclip_score_error"],1e-12)
        self.assertEqual(result["max_candidate_invariant_spread"],0.)

    def test_ridge_and_tie_behavior(self):
        rng=np.random.default_rng(20261118)
        x=rng.normal(size=(768,18))
        x[:,0]=1.
        y=rng.integers(0,2,size=(768,),dtype=np.int8)
        for name,cols in a.VARIANTS.items():
            w=a.ridge(x,y,cols)
            self.assertEqual(len(w),len(cols))
            self.assertTrue(bool(np.isfinite(w).all()),name)
        x=np.zeros((10,6,18),dtype=np.float64)
        self.assertTrue(all(c==0 for c in a.heuristic(x,"max_cosine")))
        self.assertTrue(all(c==0 for c in a.heuristic(x,"max_cardinality")))

    def test_mathematically_identical_mask_ties_across_float_ulp(self):
        # For a cardinality-only policy, masks 3,4,5 are exactly the same
        # feature row. A vectorized dot product can differ by one ulp.
        x=np.zeros((2,6,18),dtype=np.float64)
        x[:,:,0]=1.
        x[:,:,1]=np.array([1,1,1,2,2,2])/3.
        s=np.array([[.07,.07,.07,.21834327270398918,
                     .2183432727039892,.2183432727039892]]*2)
        # Ordinary argmax chooses 4 even though 3 has identical inputs.
        self.assertEqual(np.argmax(s,axis=1).tolist(),[4,4])
        fixed=a.identical_feature_ties(x,(0,1),s)
        self.assertEqual(np.argmax(fixed,axis=1).tolist(),[3,3])
        self.assertTrue(bool(np.array_equal(fixed[:,3],fixed[:,4])))
        # Distinct feature rows remain distinguishable; exact tie only.
        self.assertEqual(fixed[0,0],s[0,0])
        self.assertEqual(fixed[0,3],s[0,3])

    def test_root_effect_bootstrap_reproducible(self):
        x=[4]*31+[-3]*24+[0]*9
        y=[0]*64
        r1=a.effect(x,y,128,"unit-test")
        r2=a.effect(x,y,128,"unit-test")
        self.assertEqual(r1,r2)
        self.assertEqual(r1["positive_roots"],31)
        self.assertEqual(r1["negative_roots"],24)
        self.assertEqual(r1["tied_roots"],9)
        self.assertEqual(r1["delta_hits"],31*4-24*3)
        self.assertEqual(r1["descriptive_cluster_bootstrap_95"][0]<=r1["delta_rate"],True)

if __name__=="__main__":unittest.main()
