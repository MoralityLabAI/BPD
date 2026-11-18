"""Read-only localhost viewer comparing historical DT02-A and prospective DT04 results.

Binds strictly to loopback (`127.0.0.1` / `localhost` / `::1`), default port `8766`
so it never collides with the immutable DT03 viewer on `8765`. Verifies the
historical DT02-A SHA-256 (`9d4f9463...`) and serves either completed DT04
summary/hardware artifacts or live root-by-root progress from `dt04/results`.
Never initiates training, downloads weights, or accesses ARG-03-N.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import webbrowser

from ..ui_server import load_archive, overview as dt02_overview, task_fixture
from . import DEV_ROOTS, STUDY_ID
from .verify_source import sha256_file, verify_historical_locks

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results"
INDEX_HTML = HERE / "ui" / "index.html"


def build_dt04_payload(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict:
    """Load completed DT04 summary + 7M hardware audit, or live per-root receipts."""
    verify_historical_locks()
    dt02 = dt02_overview(load_archive())

    summary_path = results_dir / "DT04_ORIGINAL_SUMMARY.json"
    audit_path = results_dir / "DT04_ORIGINAL_AUDIT.json"
    hw_path = results_dir / "DT04_HARDWARE_AND_7M_AUDIT.json"

    summary = json.loads(summary_path.read_bytes()) if summary_path.is_file() else None
    audit = json.loads(audit_path.read_bytes()) if audit_path.is_file() else None
    hw = json.loads(hw_path.read_bytes()) if hw_path.is_file() else None

    roots_dir = results_dir / "roots"
    completed_roots = 0
    if roots_dir.is_dir():
        completed_roots = sum(1 for r in DEV_ROOTS if (roots_dir / f"root_{r}.json").is_file())

    return {
        "schema": "bpd-dt04-ui-bundle-v1",
        "study": STUDY_ID,
        "readonly": True,
        "arg03_n_sealed": True,
        "historical_dt02": dt02,
        "dt04_progress": {
            "expected_roots": len(DEV_ROOTS),
            "completed_roots_on_disk": completed_roots,
            "summary_available": summary is not None,
            "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
            "hardware_7m_available": hw is not None,
        },
        "dt04_summary": summary,
        "dt04_audit": {
            "all_receipts_passed_integrity": audit.get("all_receipts_passed_integrity"),
            "admitted_roots_count": audit.get("admitted_roots_count"),
            "total_verified_sham_certificates": audit.get("total_verified_sham_certificates"),
        } if audit else None,
        "dt04_hardware_7m": hw,
    }


def api_dispatch(path: str, query: dict, results_dir: Path) -> dict:
    if path == "/api/dt04":
        return build_dt04_payload(results_dir)
    if path == "/api/fixture":
        task = (query.get("task") or ["sudoku"])[0]
        seed_raw = (query.get("seed") or ["42"])[0]
        split = (query.get("split") or ["train"])[0]
        horizon_raw = (query.get("horizon") or ["12"])[0]
        actions_raw = (query.get("actions") or [""])[0]
        try:
            seed = int(seed_raw)
            horizon = int(horizon_raw)
        except ValueError as err:
            raise ValueError("seed and horizon must be integers") from err
        actions = tuple(actions_raw.split(",")) if actions_raw else ()
        return task_fixture(task, seed, split=split, horizon=horizon, actions=actions)
    raise KeyError("endpoint not found")


def make_handler(results_dir: Path, html_bytes: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path in ("/", "/index.html"):
                body, code, ctype = html_bytes, 200, "text/html; charset=utf-8"
            else:
                try:
                    payload = api_dispatch(parsed.path, parse_qs(parsed.query), results_dir)
                    code = 200
                except ValueError as exc:
                    payload = {"error": str(exc)}
                    code = 400
                except KeyError:
                    payload = {"error": "endpoint not found"}
                    code = 404
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        ap.error("DT04 research viewer binds strictly to localhost")
    if not 1 <= args.port <= 65535:
        ap.error("port must be in 1..65535")
    verify_historical_locks()
    html_bytes = INDEX_HTML.read_bytes()
    with ThreadingHTTPServer((args.host, args.port), make_handler(args.results_dir, html_bytes)) as srv:
        url = f"http://127.0.0.1:{srv.server_port}/"
        print("BPD-DT04 research viewer:", url, flush=True)
        print("Read-only archived + DT04 receipts. No training or ARG-03-N access.", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
