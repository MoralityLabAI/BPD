"""CPU source and shallow training assertions for DT02-A, no heldout science roots."""
import unittest
import torch
from mechinterp.bpd.experiments.depth_steering_gym.ci_pilot import (
    FIT, TEST, FAMILIES, cases, expert_action, story_encoding,
    init_model, forward_depths, steering, train,
)
from mechinterp.bpd.experiments.depth_steering_gym.worlds import sudoku4, sudoku_legal

class CIContract(unittest.TestCase):
    def test_depth_frozen(self):
        self.assertEqual(FIT,(2,4,6))
        self.assertEqual(TEST,(8,12))
        self.assertGreater(min(TEST),max(FIT))

    def test_exact_sudoku_host_gold(self):
        x,y=cases('sudoku4',901,4,'train')
        self.assertEqual(tuple(x.shape),(4,16))
        self.assertTrue((y>=0).all() and (y<4).all())
        self.assertTrue(sudoku_legal(tuple(tuple(int(v+1) for v in r)
                  for r in y[0].reshape(4,4)),require_complete=True))

    def test_arc_synthetic_family_split(self):
        x,y=cases('arc_style',901,8,'train')
        a,b=cases('arc_style',901,8,'eval')
        self.assertEqual(x.shape,y.shape)
        self.assertFalse(torch.equal(y,b))
        self.assertEqual(a.shape,x.shape)

    def test_storyworld_hidden_gate_not_in_input(self):
        a={'day':3,'trust':2,'evidence':1,'stability':2,
           'token':False,'terminal':False,'actions':('listen','research')}
        self.assertEqual(expert_action(a),'research')
        self.assertEqual(len(story_encoding(a)),16)
        self.assertTrue(all(1<=x<=11 for x in story_encoding(a)))

    def test_native_recurrence_and_grad(self):
        torch.set_num_threads(1)
        model=init_model(901)
        x,_=cases('sudoku4',910,2,'train')
        output=forward_depths(model,x,(2,4))
        self.assertEqual(tuple(output[2].shape),(2,16,9))
        loss=output[4].square().mean();loss.backward()
        self.assertTrue(torch.isfinite(model.layers[0].o_proj.weight.grad).all())

    def test_one_training_update_exercises_real_native_trm(self):
        torch.set_num_threads(1)
        model,loss=train(901,'sudoku4',updates=1,batch=2)
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        audit=steering(model,901,n=2)
        self.assertLessEqual(audit['protected_all_vocab_max_error'],1e-6)
        self.assertEqual(set(audit['depths']),{'4','8','12'})

if __name__=='__main__':unittest.main()
