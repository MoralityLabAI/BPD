import unittest

import torch

from .adamw_screen import screen_adamw
from .gym02_campaign import _train_step_raw, make_model, CLIP, LR_CURATE, WD
from .gym02_feedback import TASKS, sample_task


class TestGym02NativeParent(unittest.TestCase):
    def test_parent_score_all_five(self):
        torch.set_num_threads(1)
        for task in TASKS:
            model = make_model(29300, task)
            params = tuple(model.parameters())
            opt = torch.optim.AdamW(params, lr=LR_CURATE, weight_decay=WD)
            x,y,_=sample_task(task, 17, 8, 'train')
            a,b,_=sample_task(task, 29, 8, 'train')
            _train_step_raw(model,opt,x,y)
            cal = torch.autograd.grad(torch.nn.functional.cross_entropy(model(a),b),params)
            raw = torch.autograd.grad(torch.nn.functional.cross_entropy(model(x),y),params)
            expected = screen_adamw(opt,raw,cal,CLIP)
            parent = [p.detach().clone() for p in params]
            _,native,score = _train_step_raw(model,opt,x,y,calibration=cal)
            self.assertAlmostEqual(expected['predicted_gain'],score,places=7)
            for g,h,p,old,d in zip(raw,native,params,parent,expected['deltas'],strict=True):
                self.assertTrue(torch.allclose(g,h,rtol=1e-5,atol=1e-7))
                self.assertTrue(torch.allclose(p.detach()-old,d,rtol=1e-5,atol=1e-7))


if __name__=='__main__':
    unittest.main()
