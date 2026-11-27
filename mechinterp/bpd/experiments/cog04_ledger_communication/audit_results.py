"""Independently replay COG-04 decisions and exact communication accounting.

This audit uses the standard library and the source-pinned exact Horn checker.
It also checks every stored artifact hash and the two actual setup streams.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cog04_audit_ledger", HERE / "vendor/ledger.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder):
    folder = Path(folder)
    hashes = json.loads((folder / "file_hashes.json").read_text())
    for name, expected in hashes.items():
        if sha(folder / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    summary = json.loads((folder / "summary.json").read_text())
    if sha(HERE / "vendor/ledger.py") != summary["source_module_sha256"]:
        raise ValueError("upstream checker source changed")
    arms = {a["name"]: a for a in json.loads((folder / "arms.json").read_text())}
    setups = json.loads((folder / "setup_streams.json").read_text())
    for name, arm in arms.items():
        sender = bytes.fromhex(setups[name]["sender_hex"])
        receiver = bytes.fromhex(setups[name]["receiver_hex"])
        if len(sender) + len(receiver) != arm["setup_bytes"]:
            raise ValueError("setup byte count mismatch")
        if hashlib.sha256(sender).hexdigest() != arm["sender_setup_sha256"]:
            raise ValueError("sender setup hash mismatch")
        if hashlib.sha256(receiver).hexdigest() != arm["receiver_setup_sha256"]:
            raise ValueError("receiver setup hash mismatch")
    tasks = {t["root_id"]: t for t in ledger.generate_tasks(summary["per_split"], seed=20261126)}
    records = [json.loads(s) for s in (folder / "per_root.jsonl").read_text().splitlines()]
    by_key = defaultdict(list)
    with gzip.open(folder / "decisions.jsonl.gz", "rt") as stream:
        for line in stream:
            d = json.loads(line)
            by_key[(d["root_id"], d["arm"])].append(d)
    total = Counter()
    for row in records:
        key = row["root_id"], row["arm"]
        task = tasks[row["root_id"]]
        if row["split"] != task["split"] or row["task_id"] != task["task_id"]:
            raise ValueError("row source binding mismatch")
        known, rejected, stats = ledger.initial_state(task), set(), Counter()
        decisions = by_key.pop(key)
        for expected_step, d in enumerate(decisions):
            available = [i for i, r in enumerate(task["world"]["rules"])
                         if r["conclusion"] not in known and i not in rejected]
            if d["step"] != expected_step or d["known_before_sha256"] != ledger.digest(known):
                raise ValueError("history binding mismatch")
            if d["action"] not in available or d["candidate_count"] != len(available):
                raise ValueError("policy acted outside public support")
            receipt = ledger.check_step(task, known, d["action"])
            useful = receipt["new_fact"] is not None and receipt["new_fact"] in ledger.useful_atoms(task)
            for field in ("accepted", "new_fact", "solved", "code", "verifier_calls"):
                if d[field] != receipt[field]:
                    raise ValueError(f"checker outcome mismatch: {field}")
            if d["useful"] != useful:
                raise ValueError("usefulness label mismatch")
            name = d["arm"]
            if name in arms:
                arm = arms[name]
                forward = 56 + 8 * math.ceil(len(available) * arm["rank"] * arm["bits_per_component"] / 8)
                if d["forward_bits"] != forward or d["reply_bits"] != 16:
                    raise ValueError("message byte accounting mismatch")
                stats["offline_audit_verifier_calls"] += 1
                ref = ledger.check_step(task, known, d["full_action"])
                full_useful = ref["new_fact"] is not None and ref["new_fact"] in ledger.useful_atoms(task)
                stats["correct_to_wrong"] += int(full_useful and not useful)
                stats["wrong_to_correct"] += int(useful and not full_useful)
                stats["full_useful"] += int(full_useful)
                stats["argmax_certified"] += int(d["argmax_certified"])
                if d["argmax_certified"] and d["action"] != d["full_action"]:
                    raise ValueError("reported argmax certificate failed")
            stats["verifier_calls"] += 1
            stats["accepted"] += int(receipt["accepted"])
            stats["useful"] += int(useful)
            stats["new_lemmas"] += int(receipt["new_fact"] is not None)
            stats["forward_bits"] += d["forward_bits"]
            stats["reply_bits"] += d["reply_bits"]
            if receipt["new_fact"] is not None:
                rejected.clear()
            elif not receipt["accepted"]:
                rejected.add(d["action"])
            known = receipt["next_known"]
            if receipt["solved"] and expected_step + 1 != len(decisions):
                raise ValueError("calls after stopping condition")
        for name, val in stats.items():
            if row.get(name, 0) != val:
                raise ValueError(f"root summary mismatch: {name}")
        if len(decisions) > row["budget"] or row["known_final_sha256"] != ledger.digest(known):
            raise ValueError("budget or final state mismatch")
        if row["solved"] != (task["world"]["goal"] in known):
            raise ValueError("root solved flag mismatch")
        total.update(stats)
    if by_key:
        raise ValueError("unattached decision rows")
    for item in summary["readouts"]:
        selected = [r for r in records if r["split"] == item["split"] and r["arm"] == item["arm"]]
        if len(selected) != item["roots"] or sum(r["solved"] for r in selected) != item["solved"]:
            raise ValueError("aggregate solved count mismatch")
        for field in ("verifier_calls", "forward_bits", "reply_bits", "correct_to_wrong", "wrong_to_correct"):
            if sum(r.get(field, 0) for r in selected) != item[field]:
                raise ValueError(f"aggregate mismatch: {field}")
    return {"passed": True, "root_arm_rollouts_replayed": len(records),
            "unique_readout_roots": len({r["root_id"] for r in records}),
            "decision_rows_replayed": total["verifier_calls"],
            "setup_streams_verified": 2 * len(arms), "artifact_hashes_verified": len(hashes),
            "totals": dict(total)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=HERE / "results")
    a = p.parse_args()
    print(json.dumps(audit(a.results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
