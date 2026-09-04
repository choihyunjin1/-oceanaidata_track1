"""Record a local, credential-free receipt after the user accepts the license online."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "tabpfn26" / "user-license-receipt.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-user-accepted", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.confirm_user_accepted:
        parser.error(
            "run only after the user personally accepts the TabPFN-2.6 license "
            "at https://platform.priorlabs.ai"
        )
    path = args.output.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": "ocean.tabpfn26.user_license_receipt.v1",
        "license_accepted_by_user": True,
        "model_version": "v2.6",
        "synthetic_only_provenance_reviewed": True,
        "competition_use_terms_reviewed": True,
        "accepted_at_utc": datetime.now(UTC).isoformat(),
        "source_url": "https://platform.priorlabs.ai",
        "credentials_stored": False,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
