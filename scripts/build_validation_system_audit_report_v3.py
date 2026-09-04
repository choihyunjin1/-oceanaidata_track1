"""Build the append-only r3 dependent-note correction.

The r3 generation preserves r2 and corrects the remaining P3 target name in
the technical synthesis source note, plus the generation identity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_SCRIPT = Path(__file__).with_name("build_validation_system_audit_report_v2.py")
SPEC = importlib.util.spec_from_file_location("build_validation_system_audit_report_v2", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise RuntimeError(f"Could not import r2 report builder: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

REPORT_ID = "validation-system-audit-2026-08-22-r3"
DEFAULT_OUTPUT = Path("reports/generated/validation_system_audit_2026-08-22_r3/artifact.json")
OLD_NOTE_LABEL = "P3 uses water-level RMSE/case sampling"
NEW_NOTE_LABEL = "P3 uses significant-wave-height (hs) RMSE/case sampling"


def build_artifact(evidence: dict[str, Any], *, generated_at: str) -> dict[str, Any]:
    """Apply the directly dependent source-note correction to complete r2."""

    artifact = base.build_artifact(evidence, generated_at=generated_at)
    method_sources = [
        source for source in artifact["manifest"]["sources"] if source.get("id") == "method_note"
    ]
    base.base._require(len(method_sources) == 1, "method_note source cardinality drifted")
    note = method_sources[0]["note"]
    base.base._require(note.count(OLD_NOTE_LABEL) == 1, "old P3 note label cardinality drifted")
    base.base._require(NEW_NOTE_LABEL not in note, "P3 note correction already applied upstream")
    method_sources[0]["note"] = note.replace(OLD_NOTE_LABEL, NEW_NOTE_LABEL, 1)
    artifact["package_info"]["originUrl"] = f"artifact://{REPORT_ID}"
    _validate_correction(artifact)
    return artifact


def _validate_correction(artifact: dict[str, Any]) -> None:
    note = next(
        source["note"]
        for source in artifact["manifest"]["sources"]
        if source.get("id") == "method_note"
    )
    base.base._require(NEW_NOTE_LABEL in note, "corrected P3 method-note label missing")
    base.base._require(OLD_NOTE_LABEL not in note, "superseded P3 method-note label remains")
    serialized = json.dumps(artifact, ensure_ascii=False)
    base.base._require("water-level RMSE" not in serialized, "water-level target label remains")
    base.base._require(
        artifact["package_info"]["originUrl"] == f"artifact://{REPORT_ID}",
        "r3 package identity drifted",
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
    base.base._require(
        output.resolve() == expected, f"output is frozen at {DEFAULT_OUTPUT.as_posix()}"
    )
    evidence = base.base.collect_evidence(root)
    generated_at = args.generated_at or datetime.now(base.base.KST).isoformat()
    artifact = build_artifact(evidence, generated_at=generated_at)
    if args.check_only:
        print("PASS: validated append-only r3 P3 target-name correction")
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
