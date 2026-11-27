import copy
import unittest
import torch
from .batch_screen import mean_gradient, search_batch
from .adamw_screen import screen_adamw

def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

class TestJointScreen(unittest.TestCase):
    def test_native_adamw(self):
        torch.set_num_threads(1)
        count=0
        max_error=0.
        max_score_error=0.
        for seed in range(12):
            for warm in (0,3):
                for clip in (.05,1.):
                    torch.manual_seed(seed)
                    model=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.Tanh(),torch.nn.Linear(8,3)).double()
                    params=tuple(model.parameters())
                    opt=torch.optim.AdamW(params,lr=.003,weight_decay=.01,foreach=False)
                    x,y=torch.randn(16,4,5,dtype=torch.float64),torch.randn(16,4,3,dtype=torch.float64)
                    def loss(i):return (model(x[list(i)])-y[list(i)]).square().mean()
                    for _ in range(warm):
                        opt.zero_grad();loss((0,1,2,3)).backward();opt.step()
                    cal=torch.autograd.grad(loss((12,13,14,15)),params)
                    rows=[torch.autograd.grad(loss((i,)),params) for i in range(16)]
                    parent=copy.deepcopy(model.state_dict());oparent=copy.deepcopy(opt.state_dict())
                    state=torch.get_rng_state().clone()
                    grads=copy.deepcopy([p.grad for p in params])
                    result=search_batch(opt,rows,cal,(0,1,2,3),pinned=(0,),allowed=lambda b:15 not in b,clip=clip)
                    self.assertIn(0,result['batch']);self.assertNotIn(15,result['batch']);self.assertLessEqual(result['scored_batches'],37)
                    self.assertGreaterEqual(result['predicted_gain'],result['initial_gain'])
                    self.assertTrue(all(torch.equal(v,parent[k]) for k,v in model.state_dict().items()))
                    self.assertTrue(equal_tree(opt.state_dict(),oparent));self.assertTrue(torch.equal(state,torch.get_rng_state()))
                    self.assertTrue(equal_tree([p.grad for p in params],grads))
                    batch=result['batch'];g=mean_gradient([rows[i] for i in batch]);joint=torch.autograd.grad(loss(batch),params)
                    for a,b in zip(g,joint):
                        self.assertTrue(torch.allclose(a,b,rtol=1e-12,atol=1e-12))
                        max_error=max(max_error,float((a-b).abs().max()))
                    pred=screen_adamw(opt,joint,cal,clip)
                    max_score_error=max(max_score_error,abs(pred["predicted_gain"]-result["predicted_gain"]))
                    self.assertAlmostEqual(pred["predicted_gain"],result["predicted_gain"],places=12)
                    opt.zero_grad();loss(batch).backward();torch.nn.utils.clip_grad_norm_(params,clip);opt.step()
                    for p,old,d in zip(params,parent.values(),pred['deltas']):
                        self.assertTrue(torch.equal(p.detach()-old,d))
                    count+=1
        print(f'{count} dense CPU fixtures; max mean-gradient error {max_error:.3g}; max score error {max_score_error:.3g}')
    def test_missing(self):
        a=torch.tensor([2.]);b=torch.tensor([4.]);r=mean_gradient([(a,None,None),(None,b,None)])
        self.assertEqual(r[0].item(),1.);self.assertEqual(r[1].item(),2.);self.assertIsNone(r[2])

if __name__=='__main__':unittest.main()
