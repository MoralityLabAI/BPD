"""Engineering parity checks on synthetic tensors, without a search benchmark."""

import copy

import pytest
import torch

from mechinterp.bpd.components import PassMaskSchedule, TiedComponentLinear
from mechinterp.bpd.experiments.hadamard_population.had05 import (
    F,
    LDT,
    Recurrent,
    evaluate,
    loss,
    proj,
    update,
)
from mechinterp.bpd.experiments.hadamard_population.native_adapter import (
    NativeRecurrent,
    native_from_recurrent,
)


@pytest.fixture(autouse=True)
def deterministic_synthetic_fixture():
    """These seeds select test tensors; they create no experimental roots."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        yield


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def parameter_pairs(source, native):
    original = dict(source.named_parameters())
    converted = dict(native.named_parameters())
    assert set(converted) == {
        "enc.weight", "enc.bias", "ctx.weight", "out.weight", "out.bias", "core.V", "core.U"
    }
    for name, parameter in original.items():
        if name == "V":
            yield name, parameter, converted["core.V"], False
        elif name == "U":
            yield name, parameter, converted["core.U"], True
        else:
            yield name, parameter, converted[name], False


def reference_masked(source, x, gates):
    """Apply explicit component gates to the historical recurrence equation."""
    x = source.prepare(x)
    hidden = x.new_zeros((*x.shape[:2], source.w))
    before_final = None
    for index in range(x.shape[2]):
        before_final = hidden
        hidden = torch.tanh(
            source.enc(x[:, :, index])
            + ((hidden @ source.V) * gates[:, index]) @ source.U.T
            + source.ctx(hidden.mean(1, keepdim=True))
        )
    return source.out(hidden).squeeze(-1), x[:, :, -1], before_final


def test_conversion_copies_factor_orientation_and_preserves_rng():
    source = Recurrent(w=8)
    state = {k: v.detach().clone() for k, v in source.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    native = native_from_recurrent(source)
    assert isinstance(native.core, TiedComponentLinear)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(native.core.V, source.V)
    assert torch.equal(native.core.U, source.U.T)
    assert torch.equal(native.U, source.U)
    assert_close(native.core.reconstructed_weight(), (source.V @ source.U.T).T)
    for name, original, converted, transposed in parameter_pairs(source, native):
        assert original.data_ptr() != converted.data_ptr(), name
        assert torch.equal(original, converted.T if transposed else converted), name
    for name, value in source.state_dict().items():
        assert torch.equal(value, state[name]), name


@pytest.mark.parametrize("use_ldt", [True, False])
@pytest.mark.parametrize("history", [1, 4, 7])
@pytest.mark.parametrize("width", [8, 24])
def test_forward_input_and_parameter_gradient_parity(use_ldt, history, width):
    source = Recurrent(w=width, use_ldt=use_ldt)
    native = native_from_recurrent(source)
    x = torch.randn(2, 5, history, F, requires_grad=True)
    xn = x.detach().clone().requires_grad_()
    target = torch.randn(2, 5)
    expected = source(x)
    actual = native(xn)
    assert_close(actual, expected)
    loss(expected, target).backward()
    loss(actual, target).backward()
    assert_close(xn.grad, x.grad)
    if not use_ldt:
        assert torch.count_nonzero(xn.grad[..., LDT]) == 0
    for name, original, converted, transposed in parameter_pairs(source, native):
        assert original.grad is not None, name
        assert converted.grad is not None, name
        assert_close(converted.grad.T if transposed else converted.grad, original.grad)
    assert native.core.calls == history
    assert native.core.pass_offset == 0
    assert torch.all(native.core.schedule.comp_mask == 1)
    assert torch.all(native.core.schedule.delta_mask == 0)


def test_repeated_forward_resets_local_and_global_passes():
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    seen = []
    handle = native.core.register_forward_pre_hook(
        lambda module, args: seen.append(module.pass_offset + module.calls)
    )
    try:
        for history, offset in [(4, 9), (1, 0), (7, 2), (4, 0)]:
            seen.clear()
            x = torch.randn(2, 3, history, F)
            assert_close(native(x, pass_offset=offset), source(x))
            assert seen == list(range(offset, offset + history))
            assert native.core.calls == history
    finally:
        handle.remove()


def test_default_zero_delta_preserves_trainable_path_after_target_changes():
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    x = torch.randn(2, 3, 4, F)
    with torch.no_grad():
        native.core.W_target.fill_(100)
        # Zero ablation also excludes any accidental pass-mean replacement.
        native.core.calibrated_passes.fill_(True)
        native.core.mean_delta_out.fill_(50)
    assert_close(native(x), source(x))
    loss(native(x), torch.randn(2, 3)).backward()
    assert native.core.V.grad is not None
    assert float(native.core.V.grad.norm()) > 0


def test_pass_specific_masks_and_mask_gradients_match_factor_equation():
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    x = torch.randn(2, 3, 5, F)
    gates = torch.rand(2, 5, 3, 8, requires_grad=True)
    native_gates = gates.detach().clone().requires_grad_()
    expected, _, _ = reference_masked(source, x, gates)
    actual = native(x, schedule=PassMaskSchedule(native_gates), pass_offset=13)
    assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert_close(native_gates.grad, gates.grad)
    for _, original, converted, transposed in parameter_pairs(source, native):
        assert_close(converted.grad.T if transposed else converted.grad, original.grad)
    assert native.core.pass_offset == 13
    assert torch.all(native.core.schedule.delta_mask == 0)


def test_repeated_legacy_finishes_replay_the_same_global_pass():
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    x = torch.randn(2, 3, 4, F)
    last, hidden = source.mid(x)
    native_last, native_hidden = native.mid(x, pass_offset=11)
    assert_close(native_last, last)
    assert_close(native_hidden, hidden)
    assert native.core.calls == 3
    for scale in [0.0, 0.2, -0.5, 0.0]:
        edit = scale * torch.randn_like(hidden)
        expected = source.finish(last, hidden + edit)
        actual = native.finish(native_last, native_hidden + edit)
        assert_close(actual, expected)
        assert_close(
            torch.autograd.grad(actual.square().sum(), native_hidden, retain_graph=True)[0],
            torch.autograd.grad(expected.square().sum(), hidden, retain_graph=True)[0],
        )
        assert native.core.calls == 1
        assert native.core.pass_offset == 14
        assert native.core.schedule.comp_mask.shape[1] == 1


def test_explicit_boundaries_preserve_interleaved_masks_and_offsets():
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    records = []
    for history, offset in [(3, 5), (6, 20), (1, 2)]:
        x = torch.randn(2, 3, history, F)
        gates = torch.rand(1, history, 1, 8)
        expected, last, hidden = reference_masked(source, x, gates)
        boundary = native.boundary(x, schedule=PassMaskSchedule(gates), pass_offset=offset)
        assert_close(boundary.last, last)
        assert_close(boundary.hidden, hidden)
        assert native.core.calls == history - 1
        records.append((boundary, expected, gates[:, -1]))
    # Each replay restores its own global final pass and one-pass mask window.
    for index in [0, 2, 1, 0, 1]:
        boundary, expected, final_gate = records[index]
        assert_close(native.resume(boundary), expected)
        assert native.core.pass_offset == boundary.pass_index
        assert native.core.calls == 1
        assert torch.equal(native.core.schedule.comp_mask[:, 0], final_gate)
        edit = torch.randn_like(boundary.hidden) * 0.1
        h = boundary.hidden + edit
        edited = torch.tanh(
            source.enc(boundary.last)
            + ((h @ source.V) * final_gate) @ source.U.T
            + source.ctx(h.mean(1, keepdim=True))
        )
        assert_close(native.resume(boundary, h), source.out(edited).squeeze(-1))


def test_legacy_boundary_detects_stale_context_and_survives_independent_forward():
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    x = torch.randn(2, 3, 4, F)
    first_last, first_hidden = native.mid(x)
    native(torch.randn(2, 3, 7, F), pass_offset=30)
    assert_close(native.finish(first_last, first_hidden), source(x))
    native.mid(torch.randn(2, 3, 2, F))
    with pytest.raises(ValueError, match="boundary/resume"):
        native.finish(first_last, first_hidden)


@pytest.mark.parametrize("arm", [0, 1, 2, 3])
def test_historical_gradient_update_helper_accepts_native_factors(arm):
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    x, y = torch.randn(3, 4, 4, F), torch.randn(3, 4)
    group = [0, 1, 2, 3]
    projection = proj(source, group)
    assert_close(proj(native, group), projection)
    rotation = torch.eye(8).roll(1, dims=1)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=0.003)
    native_optimizer = torch.optim.AdamW(native.parameters(), lr=0.003)
    source_receipt = update(source, source_optimizer, x, y, arm, projection, group, rotation)
    native_receipt = update(native, native_optimizer, x, y, arm, projection, group, rotation)
    assert_close(torch.tensor(native_receipt), torch.tensor(source_receipt))
    for _, original, converted, transposed in parameter_pairs(source, native):
        assert_close(converted.T if transposed else converted, original)
    assert_close(native(x), source(x))


def test_state_dict_and_copy_preserve_parameter_tying():
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    x = torch.randn(2, 3, 4, F)
    last, hidden = native.mid(x)
    native.finish(last, hidden)
    cloned = copy.deepcopy(native)
    loaded = NativeRecurrent(Recurrent(w=8))
    loaded.load_state_dict(native.state_dict())
    for model in [cloned, loaded]:
        assert_close(model(x), source(x))
        assert len([module for module in model.modules() if isinstance(module, TiedComponentLinear)]) == 1
        assert len(list(model.parameters())) == len(list(source.parameters()))
        with pytest.raises(ValueError, match="another model"):
            model.resume(native.boundary(x))


def test_checkpoint_save_load_roundtrip(tmp_path):
    source = Recurrent(w=8, use_ldt=False).eval()
    native = native_from_recurrent(source)
    # A changed factor must survive independently of the frozen target buffer.
    with torch.no_grad():
        native.core.V.add_(0.01)
    x = torch.randn(2, 3, 4, F)
    expected = native(x)
    checkpoint_path = tmp_path / "native_scheduler.pt"
    torch.save(
        {"width": native.w, "use_ldt": native.use_ldt, "model": native.state_dict()},
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = native_from_recurrent(
        Recurrent(w=checkpoint["width"], use_ldt=checkpoint["use_ldt"])
    ).eval()
    restored.load_state_dict(checkpoint["model"])
    assert not restored.use_ldt
    assert_close(restored(x), expected)
    for name, value in native.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name


def test_bounded_allocator_trace_matches_historical_scheduler():
    """One fixed fixture, capped at eight chunks, checks execution wiring."""
    source = Recurrent(w=24).eval()
    native = native_from_recurrent(source)
    arguments = dict(
        seed=314159,
        m=15,
        p=4,
        chunk=2,
        rounds=8,
        name="native_parity_fixture",
        cache_mib=1,
    )
    expected = evaluate(**arguments, model=source)
    actual = evaluate(**arguments, model=native)
    assert 1 <= expected["rounds"] <= 8
    assert expected["candidate_scores"] <= 8 * 2 * 4 * 15
    assert actual["model_calls"] == expected["model_calls"] == expected["rounds"]
    # Clock time is outside the deterministic trace contract.
    assert {k: v for k, v in actual.items() if k != "seconds"} == {
        k: v for k, v in expected.items() if k != "seconds"
    }


def test_invalid_input_and_schedule_are_rejected():
    source = Recurrent(w=8)
    native = native_from_recurrent(source)
    x = torch.randn(2, 3, 4, F)
    with pytest.raises(ValueError, match="float32"):
        native(x.double())
    with pytest.raises(ValueError, match="positive"):
        native(x[:, :, :0])
    with pytest.raises(ValueError, match="nonnegative"):
        native(x, pass_offset=-1)
    with pytest.raises(ValueError, match="shape"):
        native(x, schedule=PassMaskSchedule(torch.ones(1, 3, 1, 8)))
    with pytest.raises(ValueError, match="finite"):
        native(x, schedule=PassMaskSchedule(torch.full((1, 4, 1, 8), float("nan"))))
