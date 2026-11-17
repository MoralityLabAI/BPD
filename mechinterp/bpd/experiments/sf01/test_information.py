"""Source-only mathematical sanity checks. No trained BPD result is claimed."""

from math import isclose, isinf, log2

import pytest

from mechinterp.bpd.experiments.sf01.information import (
    conditional_mutual_information,
    finite_channel_information,
    public_channel_ledger,
    tied_two_pass,
)


def test_perfect_secret_is_one_bit_and_independent_is_zero():
    rows = [{"s": s, "copy": s, "independent": n} for s in (0, 1) for n in (0, 1)]
    assert isclose(conditional_mutual_information(rows, secret="s", signal="copy"), 1.0)
    assert isclose(conditional_mutual_information(rows, secret="s", signal="independent"), 0.0)


def test_conditioned_public_context_removes_vacuous_information():
    rows = [{"s": s, "x": s} for s in (0, 1)]
    assert isclose(conditional_mutual_information(rows, secret="s", signal="x"), 1.0)
    assert isclose(conditional_mutual_information(rows, secret="s", signal="x", given="x"), 0.0)


def test_chat_echoes_public_bet_zero_incremental_information():
    rows = [{"s": s, "bet": s, "chat": s} for s in (0, 1)]
    l = public_channel_ledger(rows, secret="s", action="bet", talk="chat")
    assert isclose(l["action_bits"], 1.0)
    assert isclose(l["additional_talk_bits"], 0.0)
    assert isclose(l["total_bits"], 1.0)


def test_chat_reveals_new_bit_conditional_on_bet():
    rows = [{"s": (a, b), "bet": a, "chat": b} for a in (0, 1) for b in (0, 1)]
    l = public_channel_ledger(rows, secret="s", action="bet", talk="chat")
    assert isclose(l["action_bits"], 1.0)
    assert isclose(l["additional_talk_bits"], 1.0)
    assert isclose(l["total_bits"], 2.0)


def test_source_routing_makes_cross_user_secret_independent():
    rows = [
        {"s_a": a, "s_b": b, "public": c, "recipient": recipient,
         "output": a if recipient == "A" else b}
        for a in (0, 1) for b in (0, 1) for c in (0, 1) for recipient in ("A", "B")
    ]
    for recipient, correct, other in (("A", "s_a", "s_b"), ("B", "s_b", "s_a")):
        role_rows = [r for r in rows if r["recipient"] == recipient]
        assert isclose(conditional_mutual_information(role_rows, secret=correct, signal="output", given="public"), 1.0)
        assert isclose(conditional_mutual_information(role_rows, secret=other, signal="output", given=("public",correct)), 0.0)


def test_bernoulli_channel_pairwise_kl_is_upper_bound():
    out = finite_channel_information({0: {0: 0.75, 1: 0.25}, 1: {0: 0.25, 1: 0.75}}, {0: 0.5, 1: 0.5})
    expected = 1 + 0.25 * log2(0.25) + 0.75 * log2(0.75)
    assert isclose(out["mi_bits"], expected)
    assert out["pairwise_kl_upper_bits"] >= out["mi_bits"]


def test_pairwise_kl_infinite_on_support_mismatch():
    out = finite_channel_information({0: {0: 1.0}, 1: {1: 1.0}}, {0: 0.5, 1: 0.5})
    assert isclose(out["mi_bits"], 1)
    assert isinf(out["pairwise_kl_upper_bits"])


def test_matched_current_readout_but_delayed_authorized_transfer():
    # q(h) = h[2]. For both source secrets, q(h_0)=q(h_1)=0 at
    # t=0 and t=1. After two shared recurrent applications, private
    # q differs by the donor secret; the unauthorized q is unchanged.
    for a in (0.0, 1.0):
        for s in (0.0, 1.0):
            h0 = (s, 0.0, 0.0)
            h1 = (0.0, s, 0.0)
            assert h0[2] == h1[2] == 0
            assert tied_two_pass(s, a)[2] == a * s
    assert tied_two_pass(1, 0)[2] == 0
    assert tied_two_pass(1, 0, leak=0.5)[2] == 0.5


def test_invalid_weights_and_undefined_records_fail():
    with pytest.raises(ValueError):
        conditional_mutual_information([], secret="s", signal="y")
    with pytest.raises(ValueError):
        conditional_mutual_information([{"s": 0, "y": 1, "w": -1}], secret="s", signal="y", weight="w")
    with pytest.raises(ValueError):
        finite_channel_information({0: {0: 0.4}}, {0: 1.0})


def test_xor_synergy_hides_secret_from_each_single_factor():
    # Two independently random bits K and C=S XOR K each reveal zero
    # information about S individually, while their joint state reveals S.
    rows = [{"s": s, "key": k, "cipher": s ^ k}
            for s in (0, 1) for k in (0, 1)]
    assert isclose(conditional_mutual_information(rows, secret="s", signal="key"), 0.0)
    assert isclose(conditional_mutual_information(rows, secret="s", signal="cipher"), 0.0)
    assert isclose(conditional_mutual_information(rows, secret="s", signal=("key", "cipher")), 1.0)
