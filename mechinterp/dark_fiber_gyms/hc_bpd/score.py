"""Score frozen HC-BPD panels; standard library only."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean

ARMS = ("same_value", "donor", "action_sae", "prompt_sae", "pca", "haar", "sham")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def upper_binomial(k, n):
    """One-sided 95% Clopper-Pearson upper limit."""
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        cdf = sum(math.comb(n, j) * p**j * (1-p)**(n-j)
                  for j in range(k+1))
        if cdf > 0.05:
            lo = p
        else:
            hi = p
    return hi


def score(roster, receipts):
    # Roster is an externally frozen list of root/case pairs.
    expected = {(x["root"], x["case"]): x for x in roster}
    require(len(expected) == len(roster), "duplicate roster case")
    roots = sorted({r for r, _ in expected})
    require(len(roots) == 64, "need 64 assigned test roots")
    for root in roots:
        cases = [x for x in roster if x["root"] == root]
        require(len(cases) == 4 and
                {(x["context"], x["expected"]) for x in cases} ==
                {(c, a) for c in ("initial", "challenge")
                 for a in ("deploy", "hold")}, "incomplete paired roster")
    cells = {}
    for x in receipts:
        key = (x["root"], x["case"])
        require(key in expected and x["arm"] in ARMS, "unassigned receipt")
        require(x["expected"] == expected[key]["expected"], "action mismatch")
        cell = (*key, x["arm"])
        require(cell not in cells, "duplicate receipt")
        require(x["loops"] == 2 and x["site"] == 95, "wrong primary endpoint")
        require(x["status"] == "ok", "assigned trial failed; panel blocked")
        for field in ("before", "after", "dose", "radius_error"):
            require(type(x[field]) in (int, float) and math.isfinite(x[field]),
                    "invalid numeric receipt")
        for field in ("clean_head", "head"):
            require(isinstance(x[field], str) and bool(x[field]), "missing head digest")
        if x["arm"] == "same_value":
            require(x["head"] == x["clean_head"] and x["before"] == x["after"]
                    and x["dose"] == 0, "same-value parity failure")
        else:
            require(abs(x["dose"] - .05) <= 1e-6 and
                    0 <= x["radius_error"] <= 1e-6, "physical dose mismatch")
        cells[cell] = x
    require(set(cells) == {(*k, a) for k in expected for a in ARMS},
            "missing assigned trial")
    effects, shams = [], []
    for root in roots:
        values, controls = [], []
        for key in sorted(k for k in expected if k[0] == root):
            panel = [cells[(*key, a)] for a in ARMS]
            require(len({(x["before"], x["clean_head"]) for x in panel}) == 1,
                    "clean baseline mismatch")
            a, s = (cells[(*key, arm)] for arm in ("action_sae", "sham"))
            values.append(.5 * (math.tanh(a["after"]/4)-math.tanh(s["after"]/4)))
            controls.append(abs(.5 * (math.tanh(s["after"]/4)-
                                       math.tanh(s["before"]/4))))
        effects.append(mean(values))
        shams.append(max(controls))
    rng = random.Random(19001)
    draws = sorted(mean(rng.choices(effects, k=len(effects))) for _ in range(10000))
    k = sum(x > .05 for x in shams)
    upper = upper_binomial(k, len(roots))
    return {"roots": len(roots), "rows": len(cells), "mean": mean(effects),
            "bootstrap_95": [draws[250], draws[9749]],
            "sham_exceedances": k, "sham_upper_95": upper,
            "fixed_prefix_statistical_gate": draws[250] > .05 and upper <= .05,
            "evidence_scope": "arithmetic only; custody and native audit required",
            "root_effects": dict(zip(roots, effects))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roster", type=Path)
    p.add_argument("receipts", type=Path)
    p.add_argument("--roster-sha256", required=True)
    args = p.parse_args()
    require(digest(args.roster) == args.roster_sha256, "roster hash mismatch")
    roster = json.loads(args.roster.read_text())
    receipts = [json.loads(x) for x in args.receipts.read_text().splitlines()]
    print(json.dumps(score(roster, receipts), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
