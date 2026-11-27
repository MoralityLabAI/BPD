"""Semantic and pairing contracts for all-address RP04 readouts."""
import unittest
from dataclasses import replace
import torch
from .common import Recipe
from .data import make_batch, seed_for
from ..training_steering.data import partition_mask,root_encoding

class DataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.recipe=replace(Recipe(),smoke_base_pairs=16,delay=3)
        self.base,self.batch,self.metadata=make_batch(27010,'smoke',self.recipe)
    def test_all_queries_share_prefix_and_noise(self):
        x=self.batch.receiver.reshape(16,4,*self.batch.receiver.shape[1:])
        self.assertTrue(torch.equal(x[:,:,:-1],x[:,0:1,:-1].expand(-1,4,-1,-1)))
        self.assertEqual(self.batch.query.tolist(),list(range(4))*16)
        for name in ('receiver','donor','hybrid'):
            f=getattr(self.batch,name)
            self.assertTrue(torch.equal(f[:,4:],self.batch.receiver[:,4:]))
    def test_hybrid_changes_only_requested_slot(self):
        for i in range(16):
            changed=(self.base.values_h[i]!=self.base.values_r[i]).nonzero().flatten().tolist()
            self.assertEqual(changed,[int(self.base.edit_slot[i])])
        self.assertEqual(sum(self.metadata['targeted']),16)
        self.assertEqual(len(self.metadata['targeted']),64)
        for values in (self.base.values_r,self.base.values_d,self.base.values_h):
            self.assertTrue(bool(partition_mask(values,'eval').all()))
    def test_decode_every_query_and_value_label(self):
        address,_=root_encoding(27010)
        inverse=torch.argsort(address)
        decoded=inverse[self.batch.receiver[:,-1,3:7].argmax(-1)]
        self.assertTrue(torch.equal(decoded,self.batch.query))
        for suffix in ('r','d','h'):
            expected=getattr(self.batch,'values_'+suffix).gather(1,self.batch.query[:,None]).squeeze(1)
            self.assertTrue(torch.equal(expected,getattr(self.batch,'labels_'+suffix)))
    def test_purpose_seed_is_separate_and_reproducible(self):
        self.assertNotEqual(seed_for(27010,'smoke'),seed_for(27010,'development'))
        _,b,m=make_batch(27010,'smoke',self.recipe)
        self.assertEqual(m,self.metadata)
        self.assertEqual(b.fingerprint(),self.batch.fingerprint())
        with self.assertRaises(ValueError):seed_for(27010,'confirmation')

if __name__=='__main__':unittest.main()
