"""Privacy-minimized Buzz/NIP-29 export of AMM-01 result provenance.

Produces an UNSIGNED kind:9 draft. Signing, membership, and delivery are the
responsibility of a separate authorized Buzz Nostr client.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from .experiment import ID, canonical, digest


def buzz_message(receipt: dict, channel: str, github_url: str) -> dict:
    try:
        channel_id = str(uuid.UUID(channel))
    except (ValueError, AttributeError) as e:
        raise ValueError("NIP-29 requires a Buzz channel UUID") from e
    if github_url.startswith("https://github.com/MoralityLabAI/BPD/") is False:
        raise ValueError("require a public GitHub BPD repository artifact URL")
    if receipt.get("experiment") != ID:
        raise ValueError("unexpected experiment")
    if receipt.get("phase") != "confirm_fresh" or "analysis" not in receipt:
        raise ValueError("only completed confirmation receipts may be exported")
    a = receipt["analysis"]
    payload = {"schema": "BPD-AMM-01/buzz-summary/v1", "experiment": ID,
               "artifact_url": github_url, "receipt_canonical_sha256": digest(receipt),
               "root_units": receipt["independent_units"],
               "evidence_class": a["evidence_class"],
               "primary_gate_pass": a["primary_gate_pass"],
               "primary_mean_gain": a["contrasts"]["memory_vs_pooled"]["mean_gain"]}
    return {"kind": 9, "tags": [["h", channel_id], ["t", "bpd-amm01"]],
            "content": canonical(payload)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--channel", required=True)
    p.add_argument("--github-url", required=True)
    args = p.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    print(json.dumps(buzz_message(receipt, args.channel, args.github_url), indent=2))


if __name__ == "__main__":
    main()
