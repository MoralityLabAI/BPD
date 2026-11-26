import copy, itertools, unittest
import numpy as np
import torch
import torch.nn.functional as F
import pilot

class NativeBoundaryGym(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(962699)
        self.m=pilot.ComponentStateMemory(width=16,input_dim=8,values=6,components=16)
        self.b=pilot.batch(962699,71,8,2)
        self.p=pilot.projectors(self.m,torch.arange(8).reshape(2,4))
        self.q=torch.eye(16).roll(5,1)

    def test_exact_oracle_and_cycle_edit(self):
        for bits in itertools.product(range(2),repeat=6):
            for c,b1,b2 in (bits[:3],bits[3:]):
                edges=(c,c^b1,c^b1^b2)
                self.assertEqual(edges[0]^edges[1],b1)
                self.assertEqual(edges[1]^edges[2],b2)
                solutions=[x for x in itertools.product(range(2),repeat=3)
                           if x[0]==c and x[0]^x[1]==b1 and x[1]^x[2]==b2]
                self.assertEqual(solutions,[edges])
        changed=(self.b['y']!=self.b['donor_y']).numpy()
        mask=np.repeat(np.arange(2)[None,:]==self.b['slot'].numpy()[:,None],3,axis=1)
        np.testing.assert_array_equal(changed,mask)

    def test_native_full_component_dense_parity(self):
        out=self.m.rollout(self.b['x'])
        expected=out['incoming_states']@self.m.recurrent.V@self.m.recurrent.U
        torch.testing.assert_close(out['recurrent_writes'],expected,atol=2e-7,rtol=2e-6)
        pred=out['logits'].detach().clone()
        with torch.no_grad():self.m.recurrent.W_target.fill_(99.)
        torch.testing.assert_close(self.m(self.b['x']),pred,atol=0,rtol=0)

    def test_dose_and_projector(self):
        r=self.m.rollout(self.b['x'])['states'][:,1]
        d=self.m.rollout(self.b['donor_x'])['states'][:,1]
        delta,meta=pilot.state_patch(r,d,self.p,self.b['slot'],.75)
        torch.testing.assert_close(meta['delivered_norm'],torch.full((8,),.75),atol=2e-7,rtol=0)
        torch.testing.assert_close(delta.norm(dim=1),(delta@self.q).norm(dim=1),atol=2e-7,rtol=0)
        torch.testing.assert_close(self.p@self.p,self.p,atol=4e-7,rtol=1e-6)

    def test_causal_gradient_changes_training(self):
        models=[copy.deepcopy(self.m) for _ in range(3)]
        grads,_=pilot.loss_gradients(self.m,self.b,self.p,self.q)
        for i,m in enumerate(models):
            opt=torch.optim.SGD(m.parameters(),lr=.01)
            pilot.update(m,opt,grads,i)
        plain=copy.deepcopy(self.m)
        opt=torch.optim.SGD(plain.parameters(),lr=.01)
        loss=.5*(F.binary_cross_entropy_with_logits(plain(self.b['x']),self.b['y'])+
                 F.binary_cross_entropy_with_logits(plain(self.b['donor_x']),self.b['donor_y']))
        loss.backward();torch.nn.utils.clip_grad_norm_(pilot.params(plain),1.);opt.step()
        for p,q in zip(pilot.params(models[0]),pilot.params(plain)):
            torch.testing.assert_close(p,q,atol=0,rtol=0)
        self.assertGreater(float(pilot.norm([p-q for p,q in zip(pilot.params(models[0]),pilot.params(models[1]))])),0.)
        self.assertGreater(float(pilot.norm([p-q for p,q in zip(pilot.params(models[1]),pilot.params(models[2]))])),0.)

    def test_normalized_patch_gradient_finite_difference(self):
        r=torch.randn(4,16,dtype=torch.float64,requires_grad=True)
        d=torch.randn_like(r); p=self.p.double(); slots=torch.tensor([0,1,0,1])
        weights=torch.randn_like(r)
        def fn(x):return (pilot.state_patch(x,d,p,slots,.75)[0]*weights).sum()
        analytic=torch.autograd.grad(fn(r),r)[0]
        eps=1e-5
        for row,col in [(0,0),(1,3),(2,8),(3,15)]:
            hi=r.detach().clone();lo=hi.clone();hi[row,col]+=eps;lo[row,col]-=eps
            numerical=(fn(hi)-fn(lo))/(2*eps)
            self.assertAlmostEqual(float(analytic[row,col]),float(numerical),places=7)

if __name__=='__main__':unittest.main()
