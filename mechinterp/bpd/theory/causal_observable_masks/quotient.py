"""Exact rational causal-observability audit of equivalent BPD basis masks.

Standalone mathematical instrument. It does not access neural checkpoints or
ARG-03-N sealed examples. Rows of R mix an optimal anchor read basis; M is
that anchor read matrix; U is its write matrix; S spans reached inputs.
"""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import product
from typing import Sequence

Matrix = tuple[tuple[Q, ...], ...]


def matrix(rows: Sequence[Sequence[int | str | Q]]) -> Matrix:
    output = tuple(tuple(Q(x) for x in row) for row in rows)
    if not output or not output[0] or any(len(row) != len(output[0]) for row in output):
        raise ValueError("nonempty rectangular matrix required")
    return output


def eye(n: int) -> Matrix:
    if n < 1:
        raise ValueError("positive dimension required")
    return matrix([[int(i == j) for j in range(n)] for i in range(n)])


def size(a: Matrix) -> tuple[int, int]:
    return len(a), len(a[0])


def mul(a: Matrix, b: Matrix) -> Matrix:
    if size(a)[1] != size(b)[0]:
        raise ValueError("matrix dimensions do not match")
    return matrix([[sum(a[i][k] * b[k][j] for k in range(size(b)[0]))
                    for j in range(size(b)[1])] for i in range(size(a)[0])])


def subtract(a: Matrix, b: Matrix) -> Matrix:
    if size(a) != size(b):
        raise ValueError("matrix dimensions do not match")
    return matrix([[x-y for x,y in zip(a[i], b[i])] for i in range(len(a))])


def plus(a: Matrix, b: Matrix) -> Matrix:
    if size(a) != size(b):
        raise ValueError("matrix dimensions do not match")
    return matrix([[x+y for x,y in zip(a[i], b[i])] for i in range(len(a))])


def transpose(a: Matrix) -> Matrix:
    return matrix(list(zip(*a)))


def inverse(a: Matrix) -> Matrix:
    n, m = size(a)
    if n != m:
        raise ValueError("inverse requires a square matrix")
    augmented = [list(a[i]) + list(eye(n)[i]) for i in range(n)]
    for col in range(n):
        pivot = next((i for i in range(col,n) if augmented[i][col]), None)
        if pivot is None:
            raise ValueError("singular basis")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        factor = augmented[col][col]
        augmented[col] = [v/factor for v in augmented[col]]
        for i in range(n):
            if i != col:
                factor = augmented[i][col]
                augmented[i] = [x-factor*y for x,y in zip(augmented[i],augmented[col])]
    return matrix([row[n:] for row in augmented])


def rank(a: Matrix) -> int:
    x = [list(r) for r in a]
    n,m = size(a)
    row = 0
    for col in range(m):
        pivot = next((i for i in range(row,n) if x[i][col]),None)
        if pivot is None:
            continue
        x[row],x[pivot] = x[pivot],x[row]
        div = x[row][col]
        x[row] = [v/div for v in x[row]]
        for i in range(n):
            if i != row:
                f=x[i][col]
                x[i]=[u-f*v for u,v in zip(x[i],x[row])]
        row+=1
        if row==n:break
    return row


def zero(a: Matrix) -> bool:
    return all(x==0 for row in a for x in row)


def mask_matrix(bits: Sequence[int]) -> Matrix:
    if not bits or any(type(v) is not int or v not in (0,1) for v in bits):
        raise ValueError("binary mask required")
    return matrix([[bits[i] if i==j else 0 for j in range(len(bits))] for i in range(len(bits))])


def commutator_change(r: Matrix, bits: Sequence[int]) -> Matrix:
    """R^-1 D R - D, the optimal-basis coordinate-change error."""
    if size(r)!=(len(bits),len(bits)):
        raise ValueError("mask and basis dimensions differ")
    d=mask_matrix(bits)
    return subtract(mul(mul(inverse(r),d),r),d)


def powers_times_readout(c: Matrix, t: Matrix, u: Matrix, horizon: int) -> list[Matrix]:
    """A_j=C*T^j*U, including the present readout at j=0."""
    d=size(t)[0]
    if horizon<0 or size(t)!=(d,d) or size(c)[1]!=d or size(u)[0]!=d:
        raise ValueError("invalid recurrence/readout horizon")
    p=eye(d)
    result=[]
    for j in range(horizon+1):
        result.append(mul(mul(c,p),u))
        p=mul(p,t)
    return result


def response_matrices(r: Matrix, bits: Sequence[int], m: Matrix, u: Matrix,
                      t: Matrix, c: Matrix, s: Matrix, horizon: int) -> list[Matrix]:
    """Exact observed difference on the reached input-span columns of S."""
    e=mul(mul(commutator_change(r,bits),m),s)
    return [mul(a,e) for a in powers_times_readout(c,t,u,horizon)]


def observable_gramian(t: Matrix, c: Matrix, u: Matrix, horizon: int) -> Matrix:
    d=size(u)[1]
    g=matrix([[0]*d for _ in range(d)])
    for a in powers_times_readout(c,t,u,horizon):
        g=plus(g,mul(transpose(a),a))
    return g


def invariant_to_horizon(r: Matrix, bits: Sequence[int], m: Matrix, u: Matrix,
                         t: Matrix, c: Matrix, s: Matrix, horizon: int) -> bool:
    v=mul(mul(commutator_change(r,bits),m),s)
    g=observable_gramian(t,c,u,horizon)
    # Exact rational Gramian equivalence: v^T G v=sum ||C T^j U v||^2.
    return zero(mul(mul(transpose(v),g),v))


def first_detected(r: Matrix, bits: Sequence[int], m: Matrix, u: Matrix,
                   t: Matrix, c: Matrix, s: Matrix) -> int | None:
    """First visible horizon. None means invisible for every future linear step.

    Cayley-Hamilton implies horizons 0..d-1 suffice for a d-dimensional T.
    """
    for j,a in enumerate(response_matrices(r,bits,m,u,t,c,s,size(t)[0]-1)):
        if not zero(a):
            return j
    return None


def all_masks(n: int) -> tuple[tuple[int,...],...]:
    if not (1 <= n <= 10):
        raise ValueError("CPU audit limited to 10 components")
    return tuple(product((0,1),repeat=n))


def invariant_masks(r: Matrix,m: Matrix,u: Matrix,t: Matrix,c: Matrix,s: Matrix,horizon: int) -> tuple[tuple[int,...],...]:
    return tuple(bits for bits in all_masks(size(r)[0])
                 if invariant_to_horizon(r,bits,m,u,t,c,s,horizon))


def closed_under_intersection(masks: Sequence[tuple[int,...]]) -> bool:
    members=set(masks)
    return all(tuple(a&b for a,b in zip(x,y)) in members for x in members for y in members)


def delayed_shear(twist: int):
    """2D analytic Gaussian optimal-basis orbit: singleton dark now, visible at j=1."""
    if twist==0:raise ValueError("nonzero shear needed")
    return dict(r=matrix([[1,twist],[0,1]]),m=eye(2),u=eye(2),
                t=matrix([[0,1],[1,0]]),c=matrix([[0,1]]),s=eye(2))


def nonlattice_four(twist: int):
    """4D truncated-observer mask cancellation on a 1D reached input span."""
    if twist==0:raise ValueError("nonzero shear needed")
    return dict(r=matrix([[1,twist,twist,twist],[0,1,0,0],
                          [0,0,1,0],[0,0,0,1]]),m=eye(4),u=eye(4),
                t=matrix([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]),
                c=matrix([[0,1,0,0]]),s=matrix([[0],[-1],[-1],[1]]))


def capacity_witness():
    """Top-1 can reconstruct phase-disjoint but not two-coordinate mixed state."""
    c=matrix([[0,1]])
    t=matrix([[0,1],[1,0]])
    u=eye(2)
    x_disjoint=matrix([[1],[0]])
    x_mixed=matrix([[1],[1]])
    a=powers_times_readout(c,t,u,1)
    results={}
    for kind,x in (("phase_disjoint",x_disjoint),("mixed",x_mixed)):
        valid=[]
        for bits in all_masks(2):
            if sum(bits)>1:continue
            d=mask_matrix(bits)
            valid.append((bits,all(mul(z,mul(d,x))==mul(z,x) for z in a)))
        results[kind]={'successful_top1_masks':[list(b) for b,good in valid if good],
                       'observability_rank':rank(matrix([row for z in a for row in z]))}
    return results
