import unittest
import numpy as np
from .reference import graph_summary, semantic_edges, selective_direction, joint_support_search, predicted_gain


class ReferenceTests(unittest.TestCase):
    def test_graph(self):
        t, e = graph_summary('train'), graph_summary('eval')
        self.assertEqual(t, dict(vertices=153, directed_edges=912, slot_edges=[228]*4,
                                 components=[153], transition_types=48))
        self.assertEqual(e, dict(vertices=103, directed_edges=312, slot_edges=[78]*4,
                                 components=[5,14,84], transition_types=24))
        train = set(semantic_edges('train'))
        self.assertTrue(all((h,r,s) in train and sum(x!=y for x,y in zip(r,h))==1 for r,h,s in train))
        self.assertFalse({x for r,h,_ in train for x in (r,h)} &
                         {x for r,h,_ in semantic_edges('eval') for x in (r,h)})

    def test_joint_support(self):
        a = np.array([1.,1.,.01]); C = np.array([[1.,-1.,0.]]); B = np.eye(3)
        ids, r, _ = joint_support_search(a,C,B,1.,rank=2,beam=1)
        self.assertEqual(ids,(0,1)); self.assertAlmostEqual(r['gain'],np.sqrt(2))
        self.assertLess(r['collateral'],1e-12)

    def test_physical_gauge(self):
        a = np.array([1.,2.,3.]); C = np.array([[1.,0.,0.]])
        x = selective_direction(a,C,np.eye(3),.75)
        y = selective_direction(a,C,np.diag([.1,-5.,2.]),.75)
        np.testing.assert_allclose(x['delta'], y['delta'], atol=1e-12)
        self.assertAlmostEqual(x['dose'],.75)
        self.assertLess(x['collateral'],1e-12)

    def test_no_null(self):
        x = selective_direction(np.ones(2),np.eye(2),np.eye(2),1.)
        self.assertEqual(x['null_rank'],0); self.assertEqual(x['gain'],0.)
        y = selective_direction(np.ones(2),np.zeros((0,2)),np.zeros((2,0)),1.)
        self.assertEqual(y['gain'],0.)

    def test_conditional_gain(self):
        g=np.array([2.,-1.]); d=np.array([-.1,.2]); L=3.
        pred,lower=predicted_gain(g,d,L)
        self.assertAlmostEqual(pred,.4); self.assertAlmostEqual(lower,.325)
        self.assertEqual(predicted_gain(g,d)[1],None)

if __name__=='__main__': unittest.main()
