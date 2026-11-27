import unittest
import torch

from .gym02_feedback import E27, E28, E29, E31, G05, TASKS, support, sample_task, exact_e31, feedback_panel


class TestGym02Feedback(unittest.TestCase):
    def test_support(self):
        for task, expected in ((E27, (10, 6)), (E31, (153, 103)), (G05, (42, 22))):
            tr, ev = set(support(task, 'train')), set(support(task, 'eval'))
            self.assertEqual((len(tr), len(ev)), expected)
            self.assertFalse(tr & ev)
            for split, roster in (('train', tr), ('eval', ev)):
                x, y, _ = sample_task(task, 19, 2048, split)
                w = len(next(iter(roster)))
                maps = x[:, :w, 7:11].argmax(-1)
                self.assertTrue(set(map(tuple, maps.tolist())) <= roster)
                qstep = w if task == E31 else x.shape[1]-1
                q = x[:, qstep, 3:7].argmax(-1)
                row = torch.arange(len(y))
                target = maps[row, maps[row, q]] if task == E31 else maps[row, q]
                self.assertTrue(torch.equal(target, y))
    def test_panels(self):
        for task in TASKS:
            for mode in ('iid', 'balanced', 'population_or_large'):
                x, y, _ = feedback_panel(task, 81, mode=mode)
                x2, y2, _ = feedback_panel(task, 81, mode=mode)
                self.assertTrue(torch.equal(x,x2) and torch.equal(y,y2))
                self.assertEqual(len(y), (612 if task == E31 else 512)
                                 if mode == 'population_or_large' else 128)
                if mode == 'balanced' and task != E31:
                    self.assertEqual(torch.bincount(y,minlength=4).tolist(), [32]*4)
        x,y,_ = exact_e31()
        maps=x[:,:4,7:11].argmax(-1)
        q=x[:,4,3:7].argmax(-1)
        self.assertEqual(len(set(zip(map(tuple,maps.tolist()),q.tolist()))),612)
        self.assertTrue(torch.equal(maps[torch.arange(len(y)),maps[torch.arange(len(y)),q]],y))
    def test_invalid_split(self):
        with self.assertRaises(ValueError):
            sample_task(E31,1,5,'test')


if __name__ == '__main__':
    unittest.main()
