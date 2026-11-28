import unittest
import torch
from .reference import semantic_edges
from .data import edge_batch
from .model import ComponentTRM


class ModelTests(unittest.TestCase):
    def test_data(self):
        edges = semantic_edges('train')[:8]
        b = edge_batch(edges,root=123,seed=456,delay=7)
        self.assertEqual(b.query.tolist(),list(range(4))*8)
        self.assertTrue(torch.equal(b.donor,b.hybrid))
        self.assertTrue(torch.equal(b.receiver[:,4:],b.hybrid[:,4:]))
        self.assertTrue(torch.equal((b.values_r!=b.values_h).sum(1),torch.ones(32,dtype=torch.long)))
        self.assertTrue(torch.equal(b.labels_r!=b.labels_h,b.query==b.edit_slot))
        self.assertTrue(torch.equal((b.receiver!=b.hybrid).any(-1).sum(-1),torch.ones(32,dtype=torch.long)))
        self.assertEqual(b.fingerprint(),edge_batch(edges,root=123,seed=456,delay=7).fingerprint())
        with self.assertRaises(ValueError):
            edge_batch(semantic_edges('eval')[:1],root=123,seed=456,delay=7)

    def test_native_model(self):
        torch.manual_seed(123)
        m=ComponentTRM()
        self.assertEqual(sum(p.numel() for p in m.parameters()),5380)
        b=edge_batch(semantic_edges('train')[:3],root=123,seed=456,delay=7)
        x=b.receiver
        o=m.rollout(x)
        self.assertEqual(o['block_calls'],2*x.shape[1])
        self.assertEqual(tuple(o['states'].shape),(12,12,64))
        clean_prefix=m.prefix_state(x)
        h=m.rollout(x,state_replacement=clean_prefix)
        torch.testing.assert_close(o['logits'],h['logits'])
        hybrid=m.rollout(b.hybrid)
        patched=m.rollout(x,state_replacement=m.prefix_state(b.hybrid))
        torch.testing.assert_close(hybrid['logits'],patched['logits'])
        torch.nn.functional.cross_entropy(o['logits'],b.labels_r).backward()
        self.assertTrue(torch.isfinite(m.up.V.grad).all())
        self.assertTrue(torch.isfinite(m.up.U.grad).all())
        self.assertIsNone(m.up.schedule)

if __name__=='__main__': unittest.main()
