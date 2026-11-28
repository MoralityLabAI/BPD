"""Launch only a locked native batch under an externally held GPU lease."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from score import digest, require


def run(lock, output):
    require(lock["loops"] in (1, 2), "native reader supports only two budgets")
    require(os.environ.get("HC_BPD_GPU_LEASE_ID") == lock["lease_id"]
            and bool(lock["lease_id"]), "external lease acknowledgement missing")
    for name in ("reader", "model", "cases", "runtime_manifest"):
        entry = lock[name]
        require(digest(entry["path"]) == entry["sha256"], name + " hash mismatch")
    runtime = json.loads(Path(lock["runtime_manifest"]["path"]).read_text())
    for entry in runtime["files"]:
        require(digest(entry["path"]) == entry["sha256"], "runtime file mismatch")
    require(bool(runtime["files"]), "empty runtime manifest")
    rows = [json.loads(line) for line in Path(lock["cases"]["path"]).read_text().splitlines()]
    ids = [row["id"] for row in rows]
    require(bool(ids) and len(set(ids)) == len(ids), "invalid input IDs")
    # Native tool performs token capacity and intervention site checks.
    require(not output.exists(), "refusing to overwrite output")
    tmp = output.with_suffix(output.suffix + ".partial")
    with tmp.open("x") as f:
        subprocess.run([str(Path(lock["reader"]["path"]).resolve()),
                        str(Path(lock["model"]["path"]).resolve()),
                        str(lock["loops"]), str(Path(lock["cases"]["path"]).resolve())],
                       stdout=f, check=True, env=dict(os.environ, GGML_CUDA_DISABLE_GRAPHS="1"))
    got = [json.loads(line) for line in tmp.read_text().splitlines()]
    require([x["id"] for x in got] == ids and
            all(x["loops"] == lock["loops"] for x in got), "native coverage mismatch")
    # Exclusive publication; retain partial on failure for audit.
    os.link(tmp, output)
    tmp.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lock", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--lock-sha256", required=True)
    a = p.parse_args()
    require(digest(a.lock) == a.lock_sha256, "lock digest mismatch")
    run(json.loads(a.lock.read_text()), a.output)
