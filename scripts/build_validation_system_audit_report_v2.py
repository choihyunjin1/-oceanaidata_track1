"""Build the append-only r2 semantic correction to the validation-system report.

The r2 generation preserves the complete validated r1 artifact and changes only
the P3 target label in the scope-definition block, plus the generation identity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_SCRIPT = Path(__file__).with_name("build_validation_system_audit_report_v1.py")
SPEC = importlib.util.spec_from_file_location("build_validation_system_audit_report_v1", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise RuntimeError(f"Could not import base report builder: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

REPORT_ID = "validation-system-audit-2026-08-22-r2"
DEFAULT_OUTPUT = Path("reports/generated/validation_system_audit_2026-08-22_r2/artifact.json")
OLD_SCOPE_LABEL = "P3는 water-level RMSE(m)"
NEW_SCOPE_LABEL = "P3는 유의파고(hs) RMSE(m)"


def build_artifact(evidence: dict[str, Any], *, generated_at: str) -> dict[str, Any]:
    """Apply the one-field semantic correction to the complete r1 artifact."""

    artifact = base.build_artifact(evidence, generated_at=generated_at)
    scope_blocks = [
        block for block in artifact["manifest"]["blocks"] if block.get("id") == "scope_definitions"
    ]
    base._require(len(scope_blocks) == 1, "scope_definitions block cardinality drifted")
    body = scope_blocks[0]["body"]
    base._require(body.count(OLD_SCOPE_LABEL) == 1, "old P3 scope label cardinality drifted")
    base._require(NEW_SCOPE_LABEL not in body, "P3 scope correction was already applied upstream")
    scope_blocks[0]["body"] = body.replace(OLD_SCOPE_LABEL, NEW_SCOPE_LABEL, 1)
    artifact["package_info"]["originUrl"] = f"artifact://{REPORT_ID}"
    _validate_correction(artifact)
    return artifact


def _validate_correction(artifact: dict[str, Any]) -> None:
    scope = next(
        block["body"]
        for block in artifact["manifest"]["blocks"]
        if block.get("id") == "scope_definitions"
    )
    base._require(NEW_SCOPE_LABEL in scope, "corrected P3 scope label missing")
    base._require(OLD_SCOPE_LABEL not in scope, "superseded P3 scope label remains")
    base._require(
        artifact["package_info"]["originUrl"] == f"artifact://{REPORT_ID}",
        "r2 package identity drifted",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate pinned inputs and the complete corrected artifact without writing output.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    expected = (root / DEFAULT_OUTPUT).resolve()
    base._require(output.resolve() == expected, f"output is frozen at {DEFAULT_OUTPUT.as_posix()}")
    evidence = base.collect_evidence(root)
    generated_at = args.generated_at or datetime.now(base.KST).isoformat()
    artifact = build_artifact(evidence, generated_at=generated_at)
    if args.check_only:
        print("PASS: validated append-only r2 target-label correction")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"PASS: wrote corrected canonical report artifact to {DEFAULT_OUTPUT.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
