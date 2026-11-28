"""Synthetic campaign and lightweight controller for protected low-rank updates (CPU, NumPy).

Run from the repository root:
    python -m mechinterp.bpd.experiments.nullspace_update.campaign --out mechinterp/bpd/experiments/nullspace_update/results/campaign.json
Evidence class: synthetic_cpu. Nothing here is evidence about any hosted model.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import nullspace as ns

RIDGE_REL = 1e-2  # Tikhonov strength in null coordinates / unprotected coordinates (relative to sigma_max^2)
CANDIDATES = ("ordinary", "ridge_lam10", "ridge_lam1000", "ns_pinv", "ns_full", "ns_r2", "ns_r4", "ns_r8", "ns_r16",
              "tsvd_r4", "tsvd_r16", "approx_e90")
KEEP_WEIGHT = 2.0


def peak_rss_mib() -> float:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                   [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError
        return pmc.PeakWorkingSetSize / 2 ** 20
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sample_params(seed: int) -> dict:
    """Jointly random scenario parameters; no grid of separable levels."""
    r = np.random.default_rng(10_000 + seed)
    d_in = int(r.integers(48, 257))
    full = r.random() < 0.08
    k = d_in if full else max(2, int(d_in * r.uniform(0.05, 0.98)))
    return dict(seed=seed, d_in=d_in, d_out=int(r.integers(16, 49)), k=k,
                keep_rows=k + int(r.integers(0, k // 2 + 2)),
                log_cond=float(r.uniform(0, 8)), conflict=float(r.uniform(0, 1)),
                m_new=int(min(640, max(8, d_in * 10 ** r.uniform(-0.7, 0.7)))),
                noise=float(10 ** r.uniform(-3, -0.3)), target_rank=int(r.integers(1, 17)),
                offspan=float(r.uniform(0, 0.5)), dtype="float32" if r.random() < 0.4 else "float64")


def _unit_rows(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-300)


def make(p: dict) -> dict:
    r = np.random.default_rng(p["seed"])
    d, k = p["d_in"], p["k"]
    Q, _ = np.linalg.qr(r.standard_normal((d, d)))
    B, F = Q[:, :k], Q[:, k:]
    s = np.geomspace(1.0, 10 ** -p["log_cond"], k) if k > 1 else np.ones(k)
    A, _ = np.linalg.qr(r.standard_normal((p["keep_rows"], k)))
    Xk = (A * s) @ B.T
    Xk *= np.sqrt(p["keep_rows"]) / np.linalg.norm(Xk)

    def inspan(n):
        return _unit_rows((r.standard_normal((n, k)) * np.sqrt(s)) @ B.T)

    def offspan(n):
        return _unit_rows(r.standard_normal((n, d - k)) @ F.T) if d > k else np.zeros((n, d))

    def new(n):
        return np.sqrt(p["conflict"]) * inspan(n) + np.sqrt(1 - p["conflict"]) * offspan(n)

    def keep_test(n):
        return np.sqrt(1 - p["offspan"]) * inspan(n) + np.sqrt(p["offspan"]) * offspan(n)

    W = r.standard_normal((p["d_out"], d)) / np.sqrt(d)
    E = r.standard_normal((p["d_out"], p["target_rank"])) @ r.standard_normal((p["target_rank"], d))
    Xn, Xn_t, Xk_t = new(p["m_new"]), new(1024), keep_test(1024)
    scale = 1.0 / np.sqrt(d)  # target RMS equals the RMS of the protected layer's own outputs
    E *= scale / np.sqrt(np.mean((Xn_t @ E.T) ** 2))
    R = Xn @ E.T
    R = R + p["noise"] * scale * r.standard_normal(R.shape)
    dt = np.dtype(p["dtype"])
    return dict(W=W, E=E, Xn_t=Xn_t, Xk_t=Xk_t, Xk=Xk.astype(dt), Xn=Xn.astype(dt), R=R.astype(dt))


def features(d: dict, dtype: str) -> np.ndarray:
    """Computable before solving: needs only X_keep, X_new and the dtype."""
    Xk, Xn = d["Xk"], d["Xn"]
    _, S, Vt = ns.keep_spectrum(Xk)
    cut = ns.numerical_rank_tol(Xk) * S.max()
    r = int((S > cut).sum())
    dim = Xk.shape[1]
    Xn64 = np.asarray(Xn, np.float64)
    inside = float(np.linalg.norm(Xn64 @ Vt[:r].T) ** 2 / max(np.linalg.norm(Xn64) ** 2, 1e-300)) if r else 0.0
    cond = float(np.log10(S[0] / S[r - 1])) if r else 0.0
    nd = dim - r
    return np.array([inside, np.log10(Xn.shape[0] / dim), cond, r / dim,
                     np.log10(Xn.shape[0] / max(nd, 1)), 1.0 if dtype == "float32" else 0.0])


FEATURE_NAMES = ("inside_frac", "log10_m_over_d", "log10_cond_keep", "rank_over_d",
                 "log10_m_over_nulldim", "is_float32")


def solve_all(d: dict) -> dict:
    Xn, Xk, R = d["Xn"], d["Xk"], d["R"]
    N, _ = ns.null_basis(Xk)
    fam = ns.nullspace_update_family(Xn, Xk, R, (2, 4, 8, 16), N=N, ridge_rel=RIDGE_REL)
    tf = ns.truncated_svd_family(Xn, R, (4, 16), ridge_rel=RIDGE_REL)
    out = {"ordinary": ns.ordinary_ridge(Xn, R), "ridge_lam10": ns.soft_protected_ridge(Xn, Xk, R, 10.0),
           "ridge_lam1000": ns.soft_protected_ridge(Xn, Xk, R, 1000.0), "ns_pinv": ns.nullspace_update(Xn, Xk, R, N=N), "ns_full": fam[None],
           "ns_r2": fam[2], "ns_r4": fam[4], "ns_r8": fam[8], "ns_r16": fam[16],
           "tsvd_r4": tf[4], "tsvd_r16": tf[16]}
    out["approx_e90"], _ = ns.approx_protected_update(Xn, Xk, R, energy=0.9)
    return out


def metrics(d: dict, D) -> dict:
    D = np.asarray(D, np.float64)
    Xk64 = np.asarray(d["Xk"], np.float64)
    tgt = d["Xn_t"] @ d["E"].T
    ko = d["Xk_t"] @ d["W"].T
    r2 = 1 - np.linalg.norm(d["Xn_t"] @ D - tgt) ** 2 / np.linalg.norm(tgt) ** 2
    pe = np.linalg.norm(d["Xk_t"] @ D) / np.linalg.norm(ko)
    return {"r2": float(r2), "prot_err": float(pe), "utility": float(r2 - KEEP_WEIGHT * pe),
            "leak": ns.keep_leak(Xk64, D), "fro": float(np.linalg.norm(D))}


def N_dim(d):
    return ns.null_basis(d["Xk"])[0].shape[1]


def run_scenario(p: dict) -> dict:
    d = make(p)
    sols = solve_all(d)
    res = {k: metrics(d, v) for k, v in sols.items()}
    rr = np.random.default_rng(p["seed"] + 7)
    ref = sols["ns_r8"]
    res["random_r8"] = metrics(d, ns.random_matched(ref.shape, 8, float(np.linalg.norm(ref)), rr))
    lit, U = ns.literal_projector_update(d["Xn"], d["Xk"], d["R"])
    res["literal_pinv"] = metrics(d, lit)
    res["literal_pinv"]["U_norm_over_D_stable"] = float(
        np.linalg.norm(U) / max(np.linalg.norm(sols["ns_pinv"]), 1e-300))
    return {"p": p, "f": features(d, p["dtype"]).tolist(), "res": res, "null_dim": int(N_dim(d))}


# ---------------- controllers (NumPy, written here) ----------------
class Standardizer:
    def fit(self, X):
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        return self

    def __call__(self, X):
        return (X - self.m) / self.s


class Centroid:
    def fit(self, X, y):
        self.z = Standardizer().fit(X)
        self.cls = sorted(set(y))
        Z = self.z(X)
        self.C = np.array([Z[[l == c for l in y]].mean(0) for c in self.cls])
        return self

    def predict(self, X):
        Z = self.z(X)
        return [self.cls[i] for i in np.argmin(((Z[:, None] - self.C[None]) ** 2).sum(-1), axis=1)]


class Tree:
    def __init__(self, depth=3, min_leaf=10):
        self.depth, self.min_leaf = depth, min_leaf

    def fit(self, X, y):
        self.root = self._grow(X, np.array(y), self.depth)
        return self

    @staticmethod
    def _gini(y):
        _, c = np.unique(y, return_counts=True)
        return 1 - ((c / c.sum()) ** 2).sum()

    def _grow(self, X, y, depth):
        vals, counts = np.unique(y, return_counts=True)
        leaf = vals[np.argmax(counts)]
        if depth == 0 or len(vals) == 1 or len(y) < 2 * self.min_leaf:
            return leaf
        best = None
        for j in range(X.shape[1]):
            for t in np.unique(np.quantile(X[:, j], np.linspace(0.1, 0.9, 9))):
                m = X[:, j] <= t
                if m.sum() < self.min_leaf or (~m).sum() < self.min_leaf:
                    continue
                g = (m.sum() * self._gini(y[m]) + (~m).sum() * self._gini(y[~m])) / len(y)
                if best is None or g < best[0]:
                    best = (g, j, t, m)
        if best is None:
            return leaf
        _, j, t, m = best
        return (j, t, self._grow(X[m], y[m], depth - 1), self._grow(X[~m], y[~m], depth - 1))

    def predict(self, X):
        def one(x, n):
            while isinstance(n, tuple):
                n = n[2] if x[n[0]] <= n[1] else n[3]
            return n
        return [one(x, self.root) for x in X]


class UtilityRegressor:
    """Per-candidate ridge regression of utility on [1, z, z^2]; pick the argmax."""
    def __init__(self, names, alpha=1.0):
        self.names, self.alpha = names, alpha

    def _phi(self, X):
        Z = self.z(X)
        return np.hstack([np.ones((len(Z), 1)), Z, Z ** 2])

    def fit(self, X, U):
        self.z = Standardizer().fit(X)
        P = self._phi(X)
        self.W = np.linalg.solve(P.T @ P + self.alpha * np.eye(P.shape[1]), P.T @ U)
        return self

    def predict(self, X):
        return [self.names[i] for i in np.argmax(self._phi(X) @ self.W, axis=1)]


def evaluate(train, test, names):
    Xtr, Xte = np.array([t["f"] for t in train]), np.array([t["f"] for t in test])
    Utr = np.array([[t["res"][n]["utility"] for n in names] for t in train])
    Ute = np.array([[t["res"][n]["utility"] for n in names] for t in test])
    ytr = [names[i] for i in Utr.argmax(1)]
    idx = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(123)
    out = {"n_train": len(train), "n_test": len(test),
           "train_label_counts": {n: ytr.count(n) for n in names if n in ytr}}
    scores = {f"fixed_{n}": float(Ute[:, i].mean()) for i, n in enumerate(names)}
    scores["oracle"] = float(Ute.max(1).mean())
    scores["random"] = float(Ute[np.arange(len(Ute)), rng.integers(len(names), size=len(Ute))].mean())
    best_fixed_train = names[int(Utr.mean(0).argmax())]
    scores["fixed_best_on_train"] = float(Ute[:, idx[best_fixed_train]].mean())
    picks = {}
    for cname, model in (("centroid", Centroid().fit(Xtr, ytr)), ("tree3", Tree(3).fit(Xtr, ytr)),
                         ("utility_regressor", UtilityRegressor(names).fit(Xtr, Utr))):
        pk = model.predict(Xte)
        scores[f"controller_{cname}"] = float(np.mean([Ute[i, idx[c]] for i, c in enumerate(pk)]))
        picks[cname] = {n: pk.count(n) for n in names if n in pk}
    out.update(mean_utility=scores, best_fixed_train=best_fixed_train, controller_picks=picks,
               oracle_pick_counts={n: int((Ute.argmax(1) == i).sum()) for i, n in enumerate(names)})
    return out


def adaptive_study(pool, groups=100, budget=16):
    rows = []
    usable = [t for t in pool if t["null_dim"] >= 16]
    for g in range(groups):
        mods = usable[4 * g:4 * g + 4]
        if len(mods) < 4:
            break
        ds = [make(t["p"]) for t in mods]
        spectra = [ns.null_coordinate_spectrum(d["Xn"], d["Xk"], d["R"], ridge_rel=RIDGE_REL) for d in ds]
        res = {}
        for name, alloc in (("adaptive", ns.adaptive_rank_allocation(spectra, budget)),
                            ("uniform", ns.uniform_rank_allocation(spectra, budget))):
            sse = tot = 0.0
            for d, a in zip(ds, alloc):
                D = ns.nullspace_update(d["Xn"], d["Xk"], d["R"], rank=a, ridge_rel=RIDGE_REL).astype(np.float64)
                tgt = d["Xn_t"] @ d["E"].T
                sse += np.linalg.norm(d["Xn_t"] @ D - tgt) ** 2
                tot += np.linalg.norm(tgt) ** 2
            res[name] = 1 - sse / tot
        rows.append(res)
    a, u = np.array([r["adaptive"] for r in rows]), np.array([r["uniform"] for r in rows])
    return {"groups": len(rows), "budget": budget, "adaptive_pooled_r2": float(a.mean()),
            "uniform_pooled_r2": float(u.mean()), "adaptive_wins": int((a > u + 1e-12).sum()),
            "uniform_wins": int((u > a + 1e-12).sum())}


def method_table(pool):
    names = list(CANDIDATES) + ["random_r8", "literal_pinv"]
    rows = {}
    for n in names:
        def g(k):
            return np.array([t["res"][n][k] for t in pool])
        wins = sum(max(CANDIDATES, key=lambda c: t["res"][c]["utility"]) == n for t in pool)
        rows[n] = {"utility": float(g("utility").mean()), "r2": float(g("r2").mean()),
                   "prot_err": float(g("prot_err").mean()), "leak_median": float(np.median(g("leak"))),
                   "leak_max": float(g("leak").max()), "oracle_wins": wins}
    lit = np.array([t["res"]["literal_pinv"]["U_norm_over_D_stable"] for t in pool])
    rows["literal_pinv"]["U_over_stable_D_median"] = float(np.median(lit))
    rows["literal_pinv"]["U_over_stable_D_max"] = float(lit.max())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    pool = [run_scenario(sample_params(i)) for i in range(a.n)]
    names = list(CANDIDATES)
    cut = a.n // 2
    split_a = evaluate(pool[:cut], pool[cut:cut + a.n // 4], names)
    tr = [t for t in pool if t["p"]["log_cond"] < 3 and t["p"]["conflict"] < 0.6]
    te = [t for t in pool if t["p"]["log_cond"] >= 3 and t["p"]["conflict"] >= 0.6]
    split_b = evaluate(tr, te, names)
    rep = {"evidence_class": "synthetic_cpu", "keep_weight": KEEP_WEIGHT, "n_scenarios": a.n,
           "feature_names": FEATURE_NAMES, "method_table_all": method_table(pool),
           "method_table_float32": method_table([t for t in pool if t["p"]["dtype"] == "float32"]),
           "method_table_float64": method_table([t for t in pool if t["p"]["dtype"] == "float64"]),
           "heldout_seeds": split_a, "heldout_property_combo": split_b,
           "adaptive_rank": adaptive_study(pool), "empty_null_scenarios": sum(t["null_dim"] == 0 for t in pool),
           "seconds": time.time() - t0, "peak_rss_mib": peak_rss_mib()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rep[k] for k in ("seconds", "peak_rss_mib", "empty_null_scenarios")}))


if __name__ == "__main__":
    main()
