"""Read-only HTTP and security tests for the additive DT04 research viewer."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from .ui_server import INDEX_HTML, ThreadingHTTPServer, build_dt04_payload, make_handler


class DT04ViewerTests(unittest.TestCase):
    def test_html_contains_no_external_assets_or_storage(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("BPD-DT04", src)
        self.assertNotIn('src="https://', src)
        self.assertNotIn("window.localStorage", src)

    def test_http_endpoints_and_headers(self):
        with TemporaryDirectory() as tmp:
            srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Path(tmp), INDEX_HTML.read_bytes()))
            t = Thread(target=srv.serve_forever, daemon=True)
            t.start()
            base = f"http://127.0.0.1:{srv.server_port}"
            try:
                with urlopen(base + "/", timeout=8) as r:
                    self.assertEqual(r.status, 200)
                    self.assertEqual(r.headers["X-Frame-Options"], "DENY")
                    self.assertIn(b"TRM Depth Lab", r.read())
                with urlopen(base + "/api/dt04", timeout=8) as r:
                    data = json.load(r)
                    self.assertEqual(data["schema"], "bpd-dt04-ui-bundle-v1")
                    self.assertTrue(data["arg03_n_sealed"])
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(base + "/api/unknown", timeout=8)
                self.assertEqual(ctx.exception.code, 404)
            finally:
                srv.shutdown()
                srv.server_close()
                t.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
