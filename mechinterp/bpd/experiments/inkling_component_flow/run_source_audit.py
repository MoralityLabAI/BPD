"""Generate or replay the constructed INK-BPD-06 engineering receipt.

The fixture uses a native BPD wrapper with explicitly assigned parameters.
It performs zero training updates and scores zero task outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import torch
from torch import nn

from mechinterp.bpd.components import TiedComponentLinear
from mechinterp.bpd.experiments.inkling_component_flow.factor_flow import (
    ExecutionSite, ReaderSpec, clean_target_writes, component_direction_patch,
    component_reader_flow, patch_reader_audit, to_jsonable,
)


ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path(__file__).resolve().parent
SOURCES = (
    "mechinterp/bpd/components.py",
    "mechinterp/bpd/config.py",
    "mechinterp/bpd/experiments/inkling_component_flow/factor_flow.py",
    "mechinterp/bpd/experiments/inkling_component_flow/test_factor_flow.py",
    "mechinterp/bpd/experiments/inkling_component_flow/run_source_audit.py",
)


def fixture_receipt() -> dict[str, Any]:
    # All weights and inputs that influence this fixture are assigned below.
    # Module initialization consumes RNG but never supplies an observed weight.
    layer = nn.Linear(2, 4, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([
            [1., .25], [0., .5], [.25, 0.], [0., -.25],
        ]))
        layer.bias.copy_(torch.tensor([.125, -.25, .25, .125]))
    model = TiedComponentLinear(layer, 2, init_svd=False)
    with torch.no_grad():
        model.V.copy_(torch.eye(2))
        model.U.copy_(torch.tensor([[1., 0., 0., 0.], [0., .25, 0., .25]]))
    queries = torch.tensor([
        [1., 0., 0., 0.], [0., 1., 0., .5],
        [.5, .5, 1., 0.], [-.5, .25, 0., 1.],
    ])
    block2 = torch.tensor([.25, -.5, 1., .5], dtype=torch.float64)
    other = torch.tensor([.25, 1., -.25, .5], dtype=torch.float64)
    cases = []
    for loop, x in enumerate((torch.tensor([1., .5]), torch.tensor([-1., -.5]))):
        writer = ExecutionSite(
            "constructed-fixture", "explicit-weights-v1", "fixture-example",
            0, loop, 18, "core.down", "depth",
        )
        receiver = ExecutionSite(
            writer.root_id, writer.checkpoint_id, writer.example_id,
            0, loop + 1, 13, "message_input", "depth",
        )
        spec = ReaderSpec(receiver, ("Q", "K", "V", "MoE"), queries,
                          .015625, torch.tensor([1., .75, 1.25, .5]))
        with torch.no_grad():
            observed = model(x)
        native_calls = model.calls
        ledger = clean_target_writes(model, x, observed, writer)
        if model.calls != native_calls:
            raise RuntimeError("observer changed the native invocation count")
        block3 = observed.double() + other
        flow = component_reader_flow(ledger, spec, block2, block3, other)
        delta = component_direction_patch(ledger, 0, .125)
        patch = patch_reader_audit(spec, block2, block3, delta,
                                   max_dose=.125, source_site=writer)
        if patch["decomposition_max_abs_error"] > 1e-12:
            raise RuntimeError("finite message decomposition failed")
        cases.append(to_jsonable({"input": x, "observed_output": observed,
                                  "flow": flow, "patch": patch}))
    roles = [case["flow"]["gap_derivative_per_write"][0][0] for case in cases]
    if not roles[0] > 0 > roles[1]:
        raise RuntimeError("constructed opposing component roles were lost")
    return {
        "schema": "bpd/ink-bpd06-source-fixture/v1",
        "evidence_class": "SOURCE_INSTRUMENT_ONLY",
        "scope": "Constructed native BPD linear writes and local two-block readers",
        "training_updates": 0,
        "independent_trained_model_roots": 0,
        "task_outcomes_scored": 0,
        "constructed_invocations": len(cases),
        "full_transformer_continuation": False,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "diagnostic_dtype": "float64"},
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in SOURCES},
        "model_parameters": to_jsonable({
            "V": model.V, "U": model.U, "W_target": model.W_target, "bias": model.bias,
        }),
        "fixture_note": "Explicit inputs supply the opposing signs; this is an engineering property check.",
        "cases": cases,
    }


def compare(expected: Any, actual: Any, path: str = "root") -> None:
    """Check all fields; retain platform differences in the current receipt."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"receipt keys differ at {path}")
        for key in expected:
            if path == "root" and key == "runtime":
                continue
            compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"receipt array differs at {path}")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{path}[{i}]")
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (float, int))
                or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f"receipt scalar differs at {path}: {expected} / {actual}")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"receipt value differs at {path}: {expected!r} / {actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write the current source-fixture receipt")
    parser.add_argument("--check", type=Path, help="Compare against a retained receipt")
    args = parser.parse_args()
    receipt = fixture_receipt()
    if args.check:
        compare(json.loads(args.check.read_text()), receipt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.check or args.out:
        print(json.dumps({"status": "PASS", "evidence_class": receipt["evidence_class"],
                          "constructed_invocations": receipt["constructed_invocations"],
                          "task_outcomes_scored": 0,
                          "retained_receipt_checked": bool(args.check)}, sort_keys=True))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
