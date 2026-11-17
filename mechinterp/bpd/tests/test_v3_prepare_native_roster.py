"""Source-only roster encoder tests, with a fake public Sudoku bundle."""
import pytest
import numpy as np
from mechinterp.bpd.v3.prepare_native_roster import encode_stall


def test_stall_filtered_and_encodes():
    rows=[{'item_id':'a','question':'.'*81,'prop_solved':False},
          {'item_id':'b','question':'1'+'.'*80,'prop_solved':True},
          {'item_id':'c','question':'9'+'.'*80,'prop_solved':False}]
    tokens,ids,h=encode_stall({'items':rows},expected_n=2)
    assert tokens.shape==(2,81)
    assert tokens.tolist()[0]==[1]*81
    assert tokens[1,0]==10
    assert ids==['a','c']
    assert len(h[0])==64


def test_roster_rejects_bad_inputs():
    with pytest.raises(ValueError,match='requires 183'):
        encode_stall({'items':[]})
    with pytest.raises(ValueError,match='invalid'):
        encode_stall({'items':[{'item_id':'a','question':'bad','prop_solved':False}]},expected_n=1)
    with pytest.raises(ValueError,match='duplicated'):
        encode_stall({'items':[{'item_id':'a','question':'.'*81,'prop_solved':False}]*2},expected_n=2)
