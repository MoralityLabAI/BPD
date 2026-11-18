"""Finite, exact BPD-SUB-07 six-token oracle and disjoint sequence partitions.

Pure-Python label law independent of torch, trained source and experimental effects.
All 4**6 ordered sequences are enumerated. Train, discovery, dose and heldout
partitioning is frozen by NumPy PCG64 seed 246810.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import log2
from typing import Callable

import numpy as np

STUDY = "BPD-SUB-07"
TOKENS = ("X", "S", "C", "R")
SEQ_LEN = 6
VOCAB = len(TOKENS)
DATA_SEED = 246810
PARTITIONS = (2048,512,512,1024)
INITIAL_BITS = (0,1)


def transition(bits:tuple[int,int], token:int)->tuple[int,int]:
    """Reversible affine transformations over F_2^2."""
    a,b=bits
    if token==0: return (a^1,b)       # X
    if token==1: return (b,a)         # swap
    if token==2: return (a,b^a)       # control a, target b
    if token==3: return (a^b,b)       # control b, target a
    raise ValueError("Token outside registered oracle alphabet")


def oracle(sequence:tuple[int,...])->int:
    if len(sequence)!=SEQ_LEN:
        raise ValueError("Source sequence requires exactly six tokens")
    bits=INITIAL_BITS
    for symbol in sequence:
        bits=transition(bits,int(symbol))
    return bits[1]


@lru_cache(maxsize=1)
def universe():
    sequences=tuple(product(range(VOCAB),repeat=SEQ_LEN))
    labels=tuple(oracle(s) for s in sequences)
    return sequences,labels


def split_indices():
    seq,labels=universe()
    perm=np.random.default_rng(DATA_SEED).permutation(len(seq))
    a,b,c,d=PARTITIONS
    assert a+b+c+d==len(seq)==4096
    result={
        "train":perm[:a].copy(),
        "discovery":perm[a:a+b].copy(),
        "dose":perm[a+b:a+b+c].copy(),
        "heldout":perm[a+b+c:].copy(),
    }
    assert len(set(int(i) for ar in result.values() for i in ar))==4096
    return result


def partition(name:str):
    sets=split_indices()
    if name not in sets:
        raise ValueError("Unknown partition")
    sequences, labels=universe()
    keys=sets[name]
    return (np.asarray([sequences[int(i)] for i in keys],dtype=np.int64),
            np.asarray([labels[int(i)] for i in keys],dtype=np.float32),
            [int(i) for i in keys])


def binary_entropy(p:float)->float:
    return 0. if p in (0.,1.) else -p*log2(p)-(1.-p)*log2(1.-p)


def conditional_certificate(feature:Callable[[tuple[int,...]],object])->dict:
    """Exact population Bayes accuracy and conditional entropy, no model fitting."""
    seq,labels=universe()
    grouped=defaultdict(list)
    for x,y in zip(seq,labels):
        grouped[feature(x)].append(y)
    n=len(labels)
    correct=sum(max(sum(v),len(v)-sum(v)) for v in grouped.values())
    conditional=sum(len(v)*binary_entropy(sum(v)/len(v))
                    for v in grouped.values())/n
    return {
        "bayes_correct":correct,
        "bayes_accuracy":correct/n,
        "conditional_entropy_bits":conditional,
        "distinct_features":len(grouped),
    }


def exact_task_certificate():
    seq,labels=universe()
    positives=sum(labels)
    counts=lambda s:tuple(s.count(token) for token in range(VOCAB))
    stats={
        "study":STUDY,
        "oracle_sequences":len(seq),
        "positives":positives,
        "positive_rate":positives/len(seq),
        "label_entropy_bits":binary_entropy(positives/len(seq)),
        "prefix2":conditional_certificate(lambda s:s[:2]),
        "prefix4":conditional_certificate(lambda s:s[:4]),
        "last_token":conditional_certificate(lambda s:s[-1]),
        "token_counts":conditional_certificate(counts),
        "named_order_witness":{
            "a":(0,0,1,1,3,2),
            "b":(0,0,1,1,2,3),
            "a_label":oracle((0,0,1,1,3,2)),
            "b_label":oracle((0,0,1,1,2,3)),
        },
        "data_split_seed":DATA_SEED,
        "split_sizes":dict(zip(
            ("train","discovery","dose","heldout"),PARTITIONS)),
    }
    validate_certificate(stats)
    return stats


def validate_certificate(stats:dict)->None:
    assert stats["oracle_sequences"]==4096
    assert stats["positives"]==2291
    assert stats["prefix2"]["bayes_correct"]==2399
    assert stats["prefix4"]["bayes_correct"]==2735
    assert stats["token_counts"]["bayes_correct"]==2335
    assert stats["last_token"]["bayes_correct"]==2291
    assert abs(stats["label_entropy_bits"]-.9898206326381731)<1e-12
    assert abs(stats["prefix2"]["conditional_entropy_bits"]
               -.9694865132073917)<1e-12
    assert abs(stats["prefix4"]["conditional_entropy_bits"]
               -.8710851970360994)<1e-12
    assert abs(stats["token_counts"]["conditional_entropy_bits"]
               -.9581228215762154)<1e-12
    w=stats["named_order_witness"]
    assert sorted(w["a"])==sorted(w["b"]) and w["a_label"]!=w["b_label"]
    assert max(stats[k]["bayes_accuracy"]
               for k in ("prefix2","prefix4","last_token","token_counts"))<.70
    split_indices()
