"""Read-only, dependency-free local dashboard for the BPD TRM depth gym.

Exposes archived engineering receipts and reproducible *new UI fixtures* only.
Never reads protected ARG-03-N data, downloads model weights, or runs training.
Default listener is loopback. No network calls originate from the application.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import webbrowser
from urllib.parse import parse_qs, urlsplit

from .dt03_tasks import (
    arc_demo_tokens, audit_puzzle_split, blank_accuracy,
    native_arc_query_positions, public_reachable_policy, sudoku4_canonical,
)
from .worlds import HiddenEndingWorld, arc_analogue, count_solutions, sudoku4

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results" / "dt02a_native_excluded.json"
INDEX = HERE / "ui" / "index.html"
EXPECTED_ARCHIVE_SHA256 = (
    "9d4f9463c0853c6cc2491711ebf3cb6c37021a61861f3947962dff67fa76b30b"
)
ARCHIVED_RUN = "https://github.com/MoralityLabAI/BPD/actions/runs/44277919962"
MAX_SEED = 1_000_000
MIN_HORIZON = 7
MAX_HORIZON = 16


def load_archive(path: Path = ARCHIVE) -> dict:
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("archived first-run JSON SHA-256 mismatch")
    report = json.loads(raw)
    if (
        report.get("experiment") != "BPD-DT02-A"
        or report.get("supervised_depths") != [2, 4, 6]
        or report.get("heldout_depths") != [8, 12]
        or [row["seed"] for row in report["roots"]] != [901, 902]
        or report.get("official_arc1_evaluated") is not False
    ):
        raise ValueError("unexpected experiment identity or source-evidence scope")
    return report


def overview(report: dict) -> dict:
    roots = []
    for row in report["roots"]:
        families = {}
        for name, item in row["family"].items():
            families[name] = {
                "parameters": item["parameter_count"],
                "train_loss": item["final_loss"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "depths": item["eval"],
            }
            if name == "sudoku4":
                families[name]["steering"] = item["protected_steering"]
        roots.append({"seed": row["seed"], "families": families})
    return {
        "schema": "bpd-dt03-ui-overview-v1",
        "evidence": "SOURCE_LOCKED_ENGINEERING_ONLY",
        "status": "task_ability_gate_failed",
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "archived_run": ARCHIVED_RUN,
        "fit_depths": report["supervised_depths"],
        "heldout_depths": report["heldout_depths"],
        "roots": roots,
        "warnings": [
            "Historical DT02 exact Sudoku/ARC task solve rates are zero.",
            "Storyworld action labels are proxy outcomes; terminal success is zero.",
            "These are 14,082-parameter mini-TRMs, not a 7M checkpoint.",
            "The DT02 steering arms lack a full signed-Jacobian-matched sham.",
            "No official ARC-AGI-1 performance or mechanism confirmation is claimed.",
        ],
    }


def fixture_seed(public_seed: int, family: str) -> int:
    if not isinstance(public_seed, int) or not 0 <= public_seed <= MAX_SEED:
        raise ValueError("fixture seed must be an integer in 0..1000000")
    return int.from_bytes(sha256(
        f"bpd-dt03-ui-fixture-v1/{family}/{public_seed}".encode()
    ).digest()[:8], "big") % (2**31)


def task_fixture(task: str, seed: int, *, split: str = "train",
                 horizon: int = 12, actions: tuple[str, ...] = ()) -> dict:
    if task not in {"sudoku", "arc", "story"}:
        raise ValueError("unknown demo task")
    derived = fixture_seed(seed, task)
    base = {"schema": "bpd-dt03-ui-fixture-v1", "task": task, "seed": seed,
            "source": "synthetic_ui_only", "registered_evaluation_root": False}
    if task == "sudoku":
        c = sudoku4(derived, split="ui", blanks=6)
        count, _ = count_solutions(c.givens)
        return {**base, "public": {"givens": c.givens},
                "host_gold": {"solution": c.solution},
                "metadata": {"blanks": sum(v == 0 for row in c.givens for v in row),
                             "unique_solution": count == 1,
                             "canonical_group": sha256(
                                 repr(sudoku4_canonical(c.givens)).encode()
                             ).hexdigest()[:16]}}
    if task == "arc":
        if split not in {"train", "heldout"}:
            raise ValueError("ARC split must be train or heldout")
        case = arc_analogue(derived, split)
        tokens = arc_demo_tokens(case)
        return {**base, "public": case.model_view(),
                "host_gold": {"answer": case.answer},
                "metadata": {"synthetic_only": True, "input_token_length": len(tokens),
                             "query_position_count": len(native_arc_query_positions()),
                             "program_split": split}}
    if not isinstance(horizon, int) or not MIN_HORIZON <= horizon <= MAX_HORIZON:
        raise ValueError("story horizon must be 7..16")
    if len(actions) > horizon or any(action not in HiddenEndingWorld.ACTIONS for action in actions):
        raise ValueError("invalid or excessive action history")
    world = HiddenEndingWorld(derived, split="ui", horizon=horizon)
    trajectory = []
    for action in actions:
        if world.state.closed:
            raise ValueError("action issued after terminal state")
        world.step(action)
        trajectory.append({"step": len(trajectory) + 1, "action": action})
    state = world.observe()
    public = {k: state[k] for k in (
        "day", "trust", "evidence", "stability", "token", "terminal"
    )}
    policy = None
    if not state["terminal"]:
        try:
            policy = public_reachable_policy({**state, "horizon": horizon})
        except ValueError:
            pass
    result = {**base, "public": public,
              "metadata": {"horizon": horizon, "actions": trajectory,
                           "available_actions": list(HiddenEndingWorld.ACTIONS),
                           "public_teacher_suggestion": policy,
                           "host_commitment": world.commitment()}}
    if state["terminal"]:
        result["host_gold"] = {"ending": world.ending(),
                               "success": world.ending() != "unfinished"}
    return result


def _query_int(query: dict, name: str, default: int) -> int:
    vals = query.get(name, [])
    if len(vals) > 1:
        raise ValueError("duplicate parameter: " + name)
    try:
        return int(vals[0]) if vals else default
    except (ValueError, TypeError) as err:
        raise ValueError("invalid integer parameter: " + name) from err


def _query_one(query: dict, name: str, default: str) -> str:
    vals = query.get(name, [])
    if len(vals) > 1:
        raise ValueError("duplicate parameter: " + name)
    return vals[0] if vals else default


def api_response(path: str, query: dict, report: dict) -> dict:
    if path == "/api/overview":
        return overview(report)
    if path == "/api/fixture":
        task = _query_one(query, "task", "sudoku")
        seed = _query_int(query, "seed", 42)
        split = _query_one(query, "split", "train")
        horizon = _query_int(query, "horizon", 12)
        actions_raw = _query_one(query, "actions", "")
        actions = tuple(actions_raw.split(",")) if actions_raw else ()
        return task_fixture(task, seed, split=split, horizon=horizon, actions=actions)
    raise KeyError("endpoint not found")


def handler_for(report: dict, index_html: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path in ("/", "/index.html"):
                data, status, content_type = index_html, 200, "text/html; charset=utf-8"
            else:
                try:
                    payload = api_response(parsed.path, parse_qs(parsed.query), report)
                    status = 200
                except ValueError as e:
                    payload = {"error": str(e)}
                    status = 400
                except KeyError:
                    payload = {"error": "endpoint not found"}
                    status = 404
                data = json.dumps(payload, sort_keys=True).encode()
                content_type = "application/json; charset=utf-8"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy",
                             "default-src 'self' 'unsafe-inline'; connect-src 'self'; "
                             "img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            return

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="Open browser on start")
    args = p.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        p.error("The research viewer binds to localhost only")
    if not 1 <= args.port <= 65535:
        p.error("port must be in 1..65535")
    report = load_archive()
    index = INDEX.read_bytes()
    with ThreadingHTTPServer((args.host, args.port), handler_for(report, index)) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print("DT03 gym research viewer:", url, flush=True)
        print("Read-only archived evidence and synthetic UI fixtures. No training.",flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
