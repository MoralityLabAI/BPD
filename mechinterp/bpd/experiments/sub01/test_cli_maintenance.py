"""Post-run display regression: aggregate CLI never indexes smoke-only fields."""
import json
import sys

from mechinterp.bpd.experiments.sub01 import experiment as sub01


def test_aggregate_cli_reports_without_retraining(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sub01, "evaluate_root", lambda root: {"root": root})
    monkeypatch.setattr(
        sub01, "aggregate",
        lambda roots: {"study": "BPD-SUB-01", "verdict": "synthetic-test-only"},
    )
    out = tmp_path / "fake.json"
    monkeypatch.setattr(
        sys, "argv",
        ["sub01", "--phase", "development", "--ack-development", "--output", str(out)],
    )
    sub01.main()
    printed = json.loads(capsys.readouterr().out)["summary"]
    assert printed == {"study": "BPD-SUB-01", "verdict": "synthetic-test-only"}
    assert json.loads(out.read_text())["roots"] == [
        {"root": seed} for seed in range(1200, 1208)
    ]
