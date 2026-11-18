import json
import tempfile
import unittest
from pathlib import Path
import torch
from mechinterp.bpd.experiments.depth_steering_gym.worlds import (
    sudoku4, sudoku_legal, count_solutions, arc_analogue, load_arc1_task,
    HiddenEndingWorld, grid,
)
from mechinterp.bpd.experiments.depth_steering_gym.steering import (
    DepthContract, nullspace_basis, select_task_block, evaluate_patch,
    signed_task_matched_sham,
)

class TaskTests(unittest.TestCase):
    def test_sudoku_exact_oracle_and_unique_solution(self):
        for seed in range(5):
            case=sudoku4(seed,"train",6)
            self.assertTrue(sudoku_legal(case.solution,require_complete=True))
            n,sol=count_solutions(case.givens)
            self.assertEqual(n,1)
            self.assertEqual(sol,case.solution)
            self.assertTrue(case.grade(case.solution))
            self.assertFalse(case.grade(case.givens))
    def test_arc_style_heldout_program_family(self):
        a=arc_analogue(1,"train")
        b=arc_analogue(1,"heldout")
        self.assertNotEqual(a.source,"ARC-AGI-1 local JSON")
        self.assertTrue(a.grade(a.answer))
        self.assertTrue(b.grade(b.answer))
        self.assertEqual(a.model_view()["test"][0].keys(),{"input"})
    def test_official_arc1_no_gold_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"f00.json"
            path.write_text(json.dumps({"train":[{"input":[[1,0]],"output":[[2,0]]}],
                                        "test":[{"input":[[1,1]]}]}))
            task=load_arc1_task(path)
            self.assertIsNone(task.answer)
            with self.assertRaises(ValueError):task.grade(grid([[2,2]]))
            self.assertNotIn("output",task.model_view()["test"][0])
    def test_hidden_endings_only_terminal(self):
        w=HiddenEndingWorld(7,"test",6)
        self.assertNotIn("secret_cause",w.observe())
        self.assertNotIn("hidden_ending",w.observe())
        with self.assertRaises(ValueError):w.ending()
        for act in ("research","research","listen","listen","share","stabilize"):
            w.step(act)
        self.assertIn(w.ending(),HiddenEndingWorld.ENDINGS)
        with self.assertRaises(ValueError):w.step("listen")
        self.assertEqual(w.commitment(),HiddenEndingWorld(7,"test",6).commitment())

class DepthTests(unittest.TestCase):
    def test_disjoint_depths(self):
        self.assertEqual(DepthContract().test,(8,12))
        with self.assertRaises(ValueError):DepthContract(fit=(2,4),test=(4,8))
    def test_full_observer_nullspace(self):
        p=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        q=nullspace_basis(p)
        self.assertEqual(q.shape,(3,1))
        self.assertLess(float((p@q).abs().max()),1e-7)
    def test_selected_basis_fails_out_of_depth_under_rotation(self):
        # In a 3D state, h[0] is the protected scalar. The task reads h[1].
        # A single shared 90-degree rotation on h[1:3] makes task gradients
        # alternate between orthogonal hidden directions across depths.
        M=torch.tensor([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        def step(h,context):return h@M.T
        def task(h):return h[:,1:2]
        q=torch.tensor([[0.,0.],[1.,0.],[0.,1.]])
        h=torch.randn(32,3,generator=torch.Generator().manual_seed(42))
        basis,_=select_task_block(step,task,h,q,DepthContract(fit=(1,),test=(2,),rank=1))
        self.assertLess(float(abs(basis[0,0])),1e-7)
        J1=torch.autograd.functional.jacobian(lambda x:task(step(x,None)),h[:1])
        # The depth-1 selected direction has zero first-order task effect at depth-2.
        delta=(torch.ones(1,3)@basis)@basis.T
        self.assertLess(float(task(step(step(delta,None),None)).abs().max()),1e-6)
    def test_signed_jacobian_sham(self):
        g=torch.Generator().manual_seed(11)
        j=torch.randn(30,1,3,generator=g)
        b=torch.eye(3)
        d=torch.randn(30,3,generator=g)
        out,m=signed_task_matched_sham(j,d,b,g)
        self.assertLess(m["signed_full_task_jacobian_max_error"],1e-5)
        self.assertLess(m["l2_max_error"],1e-5)
    def test_protected_observer_gate(self):
        def step(h,ctx):return h
        def task(h):return h[:,1:]
        def observer(h):return torch.softmax(torch.stack((h[:,0],torch.zeros_like(h[:,0])),1),1)
        recipient=torch.zeros(10,3)
        donor=recipient.clone();donor[:,1]=1.
        safe=torch.zeros_like(recipient);safe[:,1]=1.
        good=evaluate_patch(step,task,observer,recipient,donor,safe,8)
        self.assertEqual(good["donor_recovery"],1.0)
        bad=safe.clone();bad[:,0]=.1
        with self.assertRaises(AssertionError):evaluate_patch(step,task,observer,recipient,donor,bad,8)

if __name__=="__main__":unittest.main()

class TrainingTests(unittest.TestCase):
    def test_training_only_fit_depths(self):
        from mechinterp.bpd.experiments.depth_steering_gym.training import TrainingBatch,supervised_depth_loss,audit_depth_test
        class Probe:
            def __init__(self):self.calls=[]
            def output_at_depth(self,x,depth):
                self.calls.append(depth)
                return torch.stack((x+x*depth,torch.zeros_like(x)),dim=-1)
        model=Probe();batch=TrainingBatch(torch.ones(4,3),torch.zeros(4,3,dtype=torch.long),torch.ones(4,3),"sudoku4","root1")
        loss=supervised_depth_loss(model,batch,DepthContract())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.calls,[2,4,6])
        report=audit_depth_test(model,batch,DepthContract())
        self.assertEqual(set(report),{"8","12"})
        self.assertEqual(model.calls,[2,4,6,8,12])
    def test_source_only_receipt_explicit(self):
        from mechinterp.bpd.experiments.depth_steering_gym.audit import audit
        r=audit(301)
        self.assertTrue(r["sudoku4_unique"])
        self.assertTrue(r["story_preterminal_guard"])
        self.assertFalse(r["native_trm_run"])
        self.assertFalse(r["official_arc1_scored"])
