"""DT03 review-contract regression checks; public deterministic fixtures only."""
from __future__ import annotations
import unittest
import torch

from .dt03_causal import (
    flat_observer_jacobian, physical_task_gramian,
    select_full_coordinate_block, whole_case_recovery,
    validate_delivered_sham, output_weight_basis, DualState,
)
from .dt03_tasks import (
    sudoku4_canonical,audit_puzzle_split,blank_weighted_ce,
    blank_accuracy,arc_demo_tokens,native_arc_query_positions,
    public_reachable_policy,rollout_reachable,
)
from .worlds import sudoku4,arc_analogue,HiddenEndingWorld


class CausalReviewTests(unittest.TestCase):
    def test_softmax_full_coordinates_keep_nonzero_jacobian(self):
        x=torch.tensor([[0.4,-0.1,0.6]],dtype=torch.float64)
        def visible(z):
            return torch.softmax(torch.stack(
                (z[:,0],z[:,1],-z[:,0]),dim=-1),dim=-1).unsqueeze(1)
        J=flat_observer_jacobian(visible,x)
        self.assertEqual(tuple(J.shape),(1,3,3))
        self.assertGreater(float(J.abs().max()),.01)
        self.assertLess(float(J.sum(1).abs().max()),1e-9)
        gram=physical_task_gramian(J)
        self.assertGreater(float(torch.trace(gram)),0.01)

    def test_jacobian_has_every_cell_and_class(self):
        x=torch.tensor([[.2,.4],[.5,-.3]],dtype=torch.float64)
        def out(s):
            return torch.stack((torch.stack((s[:,0],s[:,1]),dim=-1),
                                torch.stack((s[:,1],-s[:,0]),dim=-1)),dim=1)
        J=flat_observer_jacobian(out,x)
        self.assertEqual(tuple(J.shape),(2,4,2))
        self.assertAlmostEqual(float(J[0,3,0]),-1.0,places=9)
        self.assertAlmostEqual(float(J[0,2,1]),1.0,places=9)

    def test_cross_sample_dependency_rejected(self):
        with self.assertRaisesRegex(ValueError,"batch examples coupled"):
            flat_observer_jacobian(lambda s:s-s.mean(0,keepdim=True),torch.ones(3,2))

    def test_select_physical_block_tensor_observer(self):
        g=torch.Generator().manual_seed(55)
        h=torch.randn(4,3,generator=g)
        def f(x,d):return x*float(d)
        def task(z):return torch.stack([z[:,:2],z[:,1:]],dim=1)
        block,metadata=select_full_coordinate_block(f,task,h,torch.eye(3),
                                                     (2,4),2)
        self.assertEqual(tuple(block.shape),(3,2))
        self.assertEqual(metadata["full_output_coordinates"],4)
        self.assertEqual(metadata["full_state_coordinates"],3)
        self.assertGreater(metadata["eigenvalues"][0],0)

    def test_whole_case_recovery_rejects_partial_case(self):
        target=torch.zeros(2,2,2)
        predicted=target.clone()
        predicted[0,0,0]=2.
        result=whole_case_recovery(predicted,target,.01)
        self.assertEqual(result["whole_case_recovery"],.5)
        self.assertEqual(result["coordinate_recovery"],.875)
        self.assertEqual(result["per_case_recovered"],[False,True])

    def test_sham_measures_original_full_dose_and_signed_task(self):
        edit=torch.tensor([[1.,1.,0.]])
        ctrl=torch.tensor([[1.,-1.,0.]])
        J=torch.tensor([[[1.,0.,0.]]])
        r=validate_delivered_sham(edit,ctrl,torch.eye(3),J)
        self.assertTrue(r["distinct_contrast_available"])
        self.assertEqual(r["delivered_l2_error"],0.)
        self.assertEqual(r["full_signed_jacobian_error"],0.)
        with self.assertRaises(AssertionError):
            validate_delivered_sham(edit,ctrl*0.5,torch.eye(3),J)

    def test_output_svd_physically_output_side(self):
        W=torch.tensor([[2.,0.],[0.,1.],[0.,0.]])
        B=output_weight_basis(W,2)
        self.assertEqual(tuple(B.shape),(3,2))
        self.assertAlmostEqual(float((B.T@B-torch.eye(2)).abs().max()),0,places=5)

    def test_complete_hl_pack_and_unpack(self):
        h=torch.randn(2,4,3,generator=torch.Generator().manual_seed(8))
        l=torch.randn(2,4,3,generator=torch.Generator().manual_seed(9))
        packed=DualState(h,l).flatten()
        restored=DualState.unflatten(packed,(4,3))
        self.assertEqual(tuple(packed.shape),(2,24))
        self.assertTrue(torch.equal(restored.z_h,h))
        self.assertTrue(torch.equal(restored.z_l,l))

    def test_nonfinite_observer_rejected(self):
        with self.assertRaisesRegex(ValueError,"finite"):
            flat_observer_jacobian(lambda x:x*float("nan"),torch.ones(2,3))


class TaskReviewTests(unittest.TestCase):
    def test_sudoku_symmetries_have_same_canonical(self):
        c=sudoku4(83,"train",blanks=6)
        base=c.givens
        renamed=tuple(tuple(0 if x==0 else 5-x for x in row) for row in base)
        swapped=(base[1],base[0],base[2],base[3])
        transposed=tuple(tuple(base[j][i] for j in range(4)) for i in range(4))
        self.assertEqual(sudoku4_canonical(base),sudoku4_canonical(renamed))
        self.assertEqual(sudoku4_canonical(base),sudoku4_canonical(swapped))
        self.assertEqual(sudoku4_canonical(base),sudoku4_canonical(transposed))
        report=audit_puzzle_split([c],[c])
        self.assertEqual(report["cross_split_isomorphic_items"],1)
        self.assertFalse(report["admissible"])

    def test_blank_loss_excludes_known_cells(self):
        inputs=torch.tensor([[1,2,1,4]])
        gold=torch.tensor([[0,1,2,3]])
        logits=torch.zeros(1,4,4)
        logits[0,0,0]=5
        logits[0,2,2]=5
        logits[0,1,:]=-50
        output=blank_weighted_ce(logits,gold,inputs)
        self.assertLess(float(output),.1)
        accuracy=blank_accuracy(logits.argmax(-1),gold,inputs)
        self.assertEqual(accuracy["blank_accuracy"],1.)
        self.assertEqual(accuracy["clue_preservation"],0.)

    def test_arc_same_test_input_distinct_demonstrations(self):
        a=arc_analogue(44,"train")
        b=type(a)(a.id,a.train,a.test_input,a.answer,a.split,a.source)
        altered_demos=tuple((inp,tuple(tuple((x+1)%3 for x in row)
                        for row in out)) for inp,out in b.train)
        b=type(a)(b.id,altered_demos,b.test_input,b.answer,b.split,b.source)
        at=arc_demo_tokens(a)
        bt=arc_demo_tokens(b)
        self.assertNotEqual(at,bt)
        self.assertEqual(tuple(at[i] for i in native_arc_query_positions()),
                         tuple(bt[i] for i in native_arc_query_positions()))
        self.assertEqual(len(at),127)
        self.assertEqual(max(at),11)

    def test_public_story_teacher_ends_for_every_secret_gate(self):
        gates=set()
        for root in range(60):
            for horizon in (7,8,12):
                x=rollout_reachable(root,horizon)
                self.assertTrue(x["terminal_success"])
                self.assertNotEqual(x["ending"],"unfinished")
                self.assertFalse(x["hidden_gate_accessed_by_policy"])
                if horizon==7:gates.add(x["ending"])
        self.assertEqual(gates,set(HiddenEndingWorld.ENDINGS)-{"unfinished"})

    def test_unreachable_terminal_deadline_is_explicit(self):
        with self.assertRaisesRegex(ValueError,"unreachable"):
            public_reachable_policy({"day":6,"horizon":7,"trust":0,
                                     "evidence":0,"stability":2,
                                     "token":False,"terminal":False})


if __name__=="__main__":unittest.main()
