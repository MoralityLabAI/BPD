"""Verify the BPD checkout against its benchmark receipt locks and source audit."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from verify_dory_bench_sae import audit, require


ROOT = Path(__file__).resolve().parent
VERIFIER = "verify_dory_bench_sae.py"


def check_digest(content, digest, name):
    require(sha256(content).hexdigest() == digest, f"SHA-256 mismatch for {name}")


def check_release(root=ROOT, source_bundle=None):
    receipts = root / "receipts"
    manifest = json.loads((receipts / "manifest.json").read_bytes())
    require(manifest["schema"] == "bpd-dory-benchmark-sae-receipt-release-v1", "Wrong release schema")
    locks = manifest["receipt_sha256"]
    require(set(locks) == {item.name for item in receipts.iterdir() if item.name != "manifest.json"},
            "Receipt directory and manifest differ")
    files = {name: (receipts / name).read_bytes() for name in locks}
    for name, data in files.items():
        check_digest(data, locks[name], name)
    saved_verifier = (root / VERIFIER).read_bytes()
    check_digest(saved_verifier, manifest["source_verifier_sha256"], VERIFIER)
    if source_bundle is not None:
        source_bytes = source_bundle.read_bytes()
        check_digest(source_bytes, manifest["source_bundle_sha256"], "original experiment bundle")
        with ZipFile(BytesIO(source_bytes)) as archive:
            for name, data in {**files, VERIFIER: saved_verifier}.items():
                require(archive.read(name) == data, f"Bundle member changed for {name}")
            for name, digest in manifest["source_bundle_provenance_sha256"].items():
                check_digest(archive.read(name), digest, name)
    with BytesIO() as archive_bytes:
        with ZipFile(archive_bytes, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return audit(archive_bytes)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_release(source_bundle=args.source_bundle), indent=2, sort_keys=True))
