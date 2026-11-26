import unittest
import random
from types import SimpleNamespace
import numpy as np
import torch
from .reference import semantic_edges
from .scheduler import edge_views, pair_key, Candidate, select_candidates
from .transaction import probe_update
from .observer import observe_edge


class Tests(unittest.TestCase):
    def test_views(self):
        views=edge_views(17)
        self.assertEqual(set().union(*map(set,views.values())),set(semantic_edges('train')))
        keys=[{pair_key(e) for e in rows} for rows in views.values()]
        self.assertEqual(sum(map(len,keys)),len(set.union(*keys)))
        for rows in views.values():
            self.assertEqual(len({(s,r[s],h[s]) for r,h,s in rows}),48)
        self.assertEqual(views,edge_views(17))

    def test_selection(self):
        pool=edge_views(17)['fit'][:20]
        rows=[Candidate(e, float(i), -1.,1.,'x') for i,e in enumerate(pool)]
        args=dict(fit_edges=pool,epoch='x',mode='mechanistic',count=10,seed=2,window=0,cap=1)
        got,ledger=select_candidates(rows,**args)
        self.assertEqual(got,select_candidates(list(reversed(rows)),**args)[0])
        self.assertEqual(len(got),10)
        self.assertTrue(all(reason!='ranked' for _,reason in got))
        self.assertTrue(all(n<=1 for n in ledger.values()))
        with self.assertRaises(ValueError): select_candidates(rows,**dict(args,epoch='stale'))
        with self.assertRaises(ValueError): select_candidates(rows,**dict(args,fit_edges=semantic_edges('eval')))
        with self.assertRaises(ValueError): select_candidates(rows+[rows[0]],**args)

    def test_probe(self):
        torch.manual_seed(5)
        m=torch.nn.Linear(2,1); o=torch.optim.AdamW(m.parameters(),lr=.01)
        x=torch.tensor([[1.,2.]])
        loss=lambda: (m(x)-1).square().mean()
        # Populate Adam moments before measuring a transaction.
        o.zero_grad(); loss().backward(); o.step()
        weights={k:v.clone() for k,v in m.state_dict().items()}
        step=[s['step'].clone() for s in o.state.values()]
        rng=torch.get_rng_state().clone()
        def draw():
            _=torch.rand(1); _=random.random(); _=np.random.rand()
            return loss()
        p=probe_update(m,o,draw,draw)
        self.assertGreater(p['realized_gain'],0)
        self.assertGreater(p['predicted_gain'],0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in m.state_dict().items(): torch.testing.assert_close(v,weights[k],rtol=0,atol=0)
        for s,v in zip(o.state.values(),step): torch.testing.assert_close(s['step'],v,rtol=0,atol=0)
        def fail(): raise RuntimeError('sentinel')
        with self.assertRaises(RuntimeError): probe_update(m,o,fail,loss)
        for k,v in m.state_dict().items(): torch.testing.assert_close(v,weights[k],rtol=0,atol=0)

    def test_observer(self):
        class Linear(torch.nn.Module):
            def prefix_state(self,x): return x[:,0,:]
            def rollout(self,x,state_replacement,patch_step):
                return {'logits':state_replacement*(x.shape[1]-4)}
        x=torch.zeros(4,7,4); h=x.clone(); h[:,0,1]=1
        b=SimpleNamespace(receiver=x,hybrid=h,query=torch.arange(4),edit_slot=torch.zeros(4,dtype=torch.long),
                          labels_r=torch.zeros(4,dtype=torch.long),labels_h=torch.tensor([1,0,0,0]))
        r=observe_edge(Linear(),b,[0,2])
        self.assertEqual(tuple(r['observer'].shape),(24,4))
        self.assertEqual(tuple(r['protected'].shape),(18,4))
        self.assertAlmostEqual(r['target_linear'],2.25)
        self.assertAlmostEqual(r['target_actual'],2.25)
