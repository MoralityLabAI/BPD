"""Verify the committed compressed HAD-05 replay without rerunning training."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from audit import receipt
from archive_audit import run as archive_receipt, positive
from receipt_checks import validate


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    packaging = json.loads((directory / "PACKAGING.json").read_text())
    if manifest.get("complete") is not True:
        raise ValueError("Replay manifest is incomplete")
    expected = {"allocation": "results.json", "exact": "archive_results.json",
        "near": "near_results.json"}
    if set(packaging["receipts"]) != set(expected.values()):
        raise ValueError("Compressed receipt roster is incomplete")
    output = {"scope": "SAVED_DEVELOPMENT_RECEIPT_AUDIT", "families": {}}
    with tempfile.TemporaryDirectory(prefix="had05_receipt_") as tmp:
        for family, name in expected.items():
            item = packaging["receipts"][name]
            if item["stored_file"] != name + ".gz":
                raise ValueError("Unexpected compressed receipt filename")
            compressed = (directory / item["stored_file"]).read_bytes()
            if hashlib.sha256(compressed).hexdigest() != item["compressed_sha256"]:
                raise ValueError(f"Compressed hash mismatch: {name}")
            raw = gzip.decompress(compressed)
            digest = hashlib.sha256(raw).hexdigest()
            if digest != item["uncompressed_sha256"] or digest != manifest["output_sha256"][name]:
                raise ValueError(f"Uncompressed hash mismatch: {name}")
            if len(raw) != item["uncompressed_bytes"] or len(compressed) != item["compressed_bytes"]:
                raise ValueError(f"Size mismatch: {name}")
            path = Path(tmp) / name
            path.write_bytes(raw)
            integrity = validate(path, family)
            certificates = receipt(path) if family == "allocation" else archive_receipt(path, near=family == "near")
            output["families"][family] = {
                "sha256": digest, "integrity": integrity,
                "exact_certificate_audit": certificates,
            }
    output["disjoint_source_positive_control"] = positive()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path,
        default=Path(__file__).parent / "revision_20261126" / "replay")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.dir)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
