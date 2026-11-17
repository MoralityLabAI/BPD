"""BASIS-01-SCALE: memory-bounded module-output certificates on exported TRM vectors.

Standalone NumPy CLI. No model access, gradients, gates, or checkpoint download.
Historical floor_and_basis.py remains the independent scalar reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BUDGETS = (0.01, 0.03, 0.10)


def softmax(a: np.ndarray) -> np.ndarray:
    y = np.asarray(a, dtype=np.float64)
    y = y - y.max(axis=-1, keepdims=True)
    z = np.exp(y)
    return z / z.sum(axis=-1, keepdims=True)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if np.any(q <= 0) or np.any(p < 0):
        raise ValueError('probabilities must be finite with q > 0')
    return np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300) / q), 0), axis=-1)


def make_basis(w: np.ndarray, family: str, kind: str, *, seed: int = 0,
               svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None):
    """Exact orthogonal complete basis, (V @ U).T=W, without dense solves."""
    w = np.asarray(w, np.float64)
    if w.ndim != 2 or min(w.shape) == 0 or not np.isfinite(w).all():
        raise ValueError('W must be finite nonempty 2D')
    dout, din = w.shape
    rng = np.random.default_rng(seed)
    if kind == 'svd':
        left, _, right_t = svd if svd is not None else np.linalg.svd(w, full_matrices=True)
        if left.shape != (dout, dout) or right_t.shape != (din, din):
            raise ValueError('full SVD must include every input and output direction')
    if family == 'input':
        if kind == 'neuron':
            v = np.eye(din)
        elif kind == 'svd':
            v = right_t.T
        elif kind == 'random':
            v = np.linalg.qr(rng.normal(size=(din, din)))[0]
        else:
            raise ValueError('unknown input basis')
        u = v.T @ w.T
    elif family == 'output':
        if kind == 'coordinate':
            u = np.eye(dout)
        elif kind == 'svd':
            u = left.T
        elif kind == 'random':
            u = np.linalg.qr(rng.normal(size=(dout, dout)))[0].T
        else:
            raise ValueError('unknown output basis')
        v = w.T @ u.T
    else:
        raise ValueError('family must be input or output')
    return v, u


def score_basis(w: np.ndarray, inputs: np.ndarray, v: np.ndarray, u: np.ndarray,
                *, budgets: tuple[float, ...] = BUDGETS, chunk_size: int = 256,
                gramian: np.ndarray | None = None, call_type: np.ndarray | None = None,
                board_idx: np.ndarray | None = None) -> dict:
    """Exact zero-ablation module bound, streamed by activation rows.

    The numerical-null write classification is separately recorded. It is a
    property of W and the selected basis, not evidence of semantic sparsity.
    """
    w, v, u = (np.asarray(z, np.float64) for z in (w, v, u))
    inputs = np.asarray(inputs)
    if w.ndim != 2 or inputs.ndim != 2 or v.ndim != 2 or u.ndim != 2:
        raise ValueError('expected 2D W, inputs, V, U')
    dout, din = w.shape
    n, feat = inputs.shape
    if n < 1 or feat != din or v.shape[0] != din or u.shape != (v.shape[1], dout):
        raise ValueError('matrix and vector shapes do not match')
    if chunk_size < 1 or not (np.isfinite(w).all() and np.isfinite(v).all() and np.isfinite(u).all()):
        raise ValueError('invalid chunk size or nonfinite factorization')
    if not budgets or any((not np.isfinite(b) or b < 0) for b in budgets):
        raise ValueError('budget must be nonnegative and finite')
    if call_type is not None:
        call_type = np.asarray(call_type)
        if call_type.shape != (n,) or not np.issubdtype(call_type.dtype, np.integer):
            raise ValueError('call_type must have an integer label for every vector')
    if board_idx is not None:
        board_idx = np.asarray(board_idx)
        if board_idx.shape != (n,) or not np.issubdtype(board_idx.dtype, np.integer):
            raise ValueError('board_idx must have an integer label for every vector')
    if gramian is None:
        def norm(z):
            return np.linalg.norm(z, axis=-1)
        metric = 'euclidean'
    else:
        g = np.asarray(gramian, np.float64)
        if g.shape != (dout, dout) or not np.isfinite(g).all() or not np.allclose(g, g.T, rtol=0, atol=1e-10):
            raise ValueError('G must be finite, symmetric and positive semidefinite')
        eigval, eigvec = np.linalg.eigh((g + g.T) / 2)
        if eigval.min() < -1e-9:
            raise ValueError('G must be positive semidefinite')
        root = eigvec * np.sqrt(np.maximum(eigval, 0))
        def norm(z):
            return np.linalg.norm(z @ root, axis=-1)
        metric = 'psd_seminorm'
    c = u.shape[0]
    write_norms = norm(u)
    null_tol = max(float(write_norms.max()) * 1e-10, 1e-14)
    near_null = write_norms <= null_tol
    n_nonnull = c - int(near_null.sum())
    residual = float(np.linalg.norm((v @ u).T - w) / max(np.linalg.norm(w), 1e-300))
    if residual > 1e-10:
        raise ValueError(f'full-on factorization error too large: {residual}')
    accum = {str(beta): dict(off_sum=0, nonnull_active_sum=0, max_error=-np.inf,
                             max_budget=-np.inf, actual_sum=0.0, cert_sum=0.0,
                             softmax_sum=0.0, phases={}) for beta in budgets}
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        x = np.asarray(inputs[lo:hi], np.float64)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite activation in export')
        full = x @ w.T
        activ = x @ v
        costs = np.abs(activ) * write_norms[None, :]
        output_norm = norm(full)
        order = np.argsort(costs, axis=1, kind='stable')
        sorted_cost = np.take_along_axis(costs, order, axis=1)
        cumulative = np.cumsum(sorted_cost, axis=1)
        ref_probs = softmax(full)
        for beta in budgets:
            item = accum[str(beta)]
            lengths = np.sum(cumulative <= beta * output_norm[:, None] + 1e-14, axis=1)
            selected = np.arange(c)[None, :] < lengths[:, None]
            off = np.zeros_like(selected)
            np.put_along_axis(off, order, selected, axis=1)
            cert = np.where(lengths > 0, cumulative[np.arange(hi-lo), np.maximum(lengths-1, 0)], 0.0)
            drop = (activ * off) @ u
            actual = norm(drop)
            item['off_sum'] += int(lengths.sum())
            item['nonnull_active_sum'] += int(np.count_nonzero((~off) & (~near_null)[None, :]))
            item['max_error'] = max(item['max_error'], float(np.max(actual-cert)))
            item['max_budget'] = max(item['max_budget'], float(np.max(cert-beta*output_norm)))
            denom = np.maximum(output_norm, 1e-12)
            item['actual_sum'] += float(np.sum(actual/denom))
            item['cert_sum'] += float(np.sum(cert/denom))
            item['softmax_sum'] += float(np.sum(kl(ref_probs, softmax(full-drop))))
            if call_type is not None:
                phase = call_type[lo:hi]
                for tag in np.unique(phase):
                    selected_phase = phase == tag
                    z = item['phases'].setdefault(str(int(tag)), dict(n=0, off=0))
                    z['n'] += int(selected_phase.sum())
                    z['off'] += int(lengths[selected_phase].sum())
    reports = {}
    for beta in budgets:
        item = accum[str(beta)]
        reports[str(beta)] = {
            'mean_off': item['off_sum']/n,
            'mean_active': c-item['off_sum']/n,
            'mean_active_fraction': 1-item['off_sum']/(n*c),
            'mean_active_nonnull_writes': item['nonnull_active_sum']/n,
            'mean_active_fraction_nonnull_writes': (item['nonnull_active_sum']/(n*n_nonnull) if n_nonnull else 0.0),
            'max_actual_minus_certificate': item['max_error'],
            'max_certificate_minus_budget': item['max_budget'],
            'mean_relative_actual': item['actual_sum']/n,
            'mean_relative_certificate': item['cert_sum']/n,
            'mean_logit_kl': item['softmax_sum']/n,
            'per_call_type': {tag: {'n_vectors': z['n'], 'mean_active': c-z['off']/z['n']}
                              for tag, z in sorted(item['phases'].items(), key=lambda pair: int(pair[0]))},
        }
    return {
        'family_components': c, 'near_null_write_directions': int(near_null.sum()),
        'non_null_write_directions': n_nonnull, 'near_null_write_threshold': null_tol,
        'relative_full_on_residual': residual, 'budgets': reports,
        'n_vectors': n, 'chunk_size': chunk_size, 'metric': metric,
        'n_independent_board_roots': (int(np.unique(board_idx).size) if board_idx is not None else None),
        'evidence_type': 'MODULE_OUTPUT_CERTIFICATE_ONLY',
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def score_npz(path: Path, *, chunk_size: int = 256, max_boards: int | None = None,
              expected_sha256: str | None = None, family: str = 'all') -> dict:
    digest = sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError('matrix NPZ SHA-256 differs from reviewed manifest')
    with np.load(path, allow_pickle=False) as z:
        w, x = z['W'], z['inputs']
        boards = z['board_idx'] if 'board_idx' in z.files else None
        phase = z['call_type'] if 'call_type' in z.files else None
        pass_idx = z['pass_idx'] if 'pass_idx' in z.files else None
        ckpt_sha = str(z['checkpoint_sha256'].item()) if 'checkpoint_sha256' in z.files else None
        roster_sha = str(z['roster_sha256'].item()) if 'roster_sha256' in z.files else None
    if phase is not None and pass_idx is not None:
        if phase.shape != pass_idx.shape or not np.array_equal(phase, pass_idx % 7):
            raise ValueError('phase tags do not match registered period-7 clock')
    if max_boards is not None:
        if max_boards < 1 or boards is None:
            raise ValueError('max_boards requires positive N and board labels')
        roots = np.unique(boards)
        if max_boards > len(roots):
            raise ValueError('max_boards exceeds the number of roots')
        choose = np.isin(boards, roots[:max_boards])
        x = x[choose]
        boards = boards[choose]
        if phase is not None:
            phase = phase[choose]
    if family not in ('input', 'output', 'all'):
        raise ValueError('family must be input, output or all')
    # Full SVD needs to be computed exactly once per matrix for the two bases.
    svd = np.linalg.svd(np.asarray(w, np.float64), full_matrices=True)
    report = {'_provenance': {
        'module_npz_sha256': digest, 'checkpoint_sha256': ckpt_sha,
        'roster_sha256': roster_sha, 'board_root_subset': max_boards,
        'status': ('DEVELOPMENT_ONLY_REUSED_TRM_ROOTS' if boards is not None else 'UNLABELED_MODULE_INPUTS'),
        'evidence': 'MODULE_OUTPUT_CERTIFICATE_NOT_DOWNSTREAM_DIGIT_HEAD_KL',
    }}
    for fam, kinds in [('input', ('neuron', 'svd', 'random')),
                       ('output', ('coordinate', 'svd', 'random'))]:
        if family not in (fam, 'all'):
            continue
        for kind in kinds:
            v, u = make_basis(w, fam, kind, svd=svd)
            report[f'{fam}/{kind}'] = score_basis(w, x, v, u, chunk_size=chunk_size,
                                                 call_type=phase, board_idx=boards)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix-npz', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--chunk-size', type=int, default=256)
    p.add_argument('--max-boards', type=int, help='development-only board-level subset')
    p.add_argument('--expected-matrix-sha256')
    p.add_argument('--family', choices=('all', 'input', 'output'), default='all')
    args = p.parse_args()
    result = score_npz(args.matrix_npz, chunk_size=args.chunk_size,
                       max_boards=args.max_boards,
                       expected_sha256=args.expected_matrix_sha256,
                       family=args.family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'source_sha': result['_provenance']['module_npz_sha256'],
                      'arms': [k for k in result if not k.startswith('_')]}, indent=2))


if __name__ == '__main__':
    main()
