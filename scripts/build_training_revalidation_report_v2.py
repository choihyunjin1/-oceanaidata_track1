"""Build the append-only r2 table-sort schema correction.

The r1 canonical artifact is preserved after the official packager rejected
both tables because ``defaultSort.field`` referenced the non-declared helper
field ``sequence``.  This correction changes only those two sort fields and
the append-only generation identity; all evidence, narrative, visuals,
datasets, source pins, and the sealed registry remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

BASE_SCRIPT = Path(__file__).with_name("build_training_revalidation_report_v1.py")
SPEC = importlib.util.spec_from_file_location("build_training_revalidation_report_v1", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise RuntimeError(f"Could not import r1 report builder: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

REPORT_ID = "training-revalidation-2026-08-22-r2"
DEFAULT_OUTPUT = Path("reports/generated/training_revalidation_2026-08-22_r2/artifact.json")
R1_ARTIFACT = Path("reports/generated/training_revalidation_2026-08-22_r1/artifact.json")
EXPECTED_R1_ARTIFACT_SHA256 = "be6ec56240fd3650ff39c9b7468b639b393362448e5765af95958eba00c80ee9"
EXPECTED_REGISTRY_SHA256 = "907c9f5b2df2a4ae70799ef1fadd04737fb619d0d9bc3c30f7009e1201f19117"


def _sha256_bytes(payload: dict[str, Any]) -> str:
    return hashlib.sha256(base._canonical_bytes(payload)).hexdigest()


def collect_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the original aggregate pins, sealed registry, and exact r1 artifact."""

    evidence = base.collect_evidence(root)
    registry_path = root / base.DEFAULT_REGISTRY
    base._require(registry_path.is_file(), "sealed training registry is missing")
    registry_actual = base._sha256(registry_path)
    base._require(
        registry_actual == EXPECTED_REGISTRY_SHA256,
        f"training registry SHA drifted: {registry_actual}",
    )
    registry = base._read_json(registry_path)
    base._validate_registry(registry)
    base._require(
        _sha256_bytes(registry) == EXPECTED_REGISTRY_SHA256,
        "training registry canonical serialization drifted",
    )

    r1_path = root / R1_ARTIFACT
    base._require(r1_path.is_file(), "r1 artifact is missing")
    r1_actual = base._sha256(r1_path)
    base._require(
        r1_actual == EXPECTED_R1_ARTIFACT_SHA256,
        f"r1 artifact SHA drifted: {r1_actual}",
    )
    r1 = base._read_json(r1_path)
    return evidence, registry, r1


def build_artifact(
    evidence: dict[str, Any], registry: dict[str, Any], r1: dict[str, Any]
) -> dict[str, Any]:
    """Apply only the two declared-column sort fixes and r2 identity."""

    generated_at = registry["generated_at_kst"]
    artifact = base.build_artifact(evidence, registry, generated_at=generated_at)
    base._require(artifact == r1, "reconstructed r1 artifact drifted")
    for table in artifact["manifest"]["tables"]:
        base._require(table["defaultSort"]["field"] == "sequence", "r1 sort field drifted")
        declared = {column["field"] for column in table["columns"]}
        base._require("problem" in declared, f"problem column missing: {table['id']}")
        table["defaultSort"]["field"] = "problem"
    artifact["package_info"]["originUrl"] = f"artifact://{REPORT_ID}"
    _validate_correction(artifact, r1)
    return artifact


def _validate_correction(artifact: dict[str, Any], r1: dict[str, Any]) -> None:
    for table in artifact["manifest"]["tables"]:
        declared = {column["field"] for column in table["columns"]}
        base._require(
            table["defaultSort"]["field"] in declared, f"sort field undeclared: {table['id']}"
        )
        base._require(
            table["defaultSort"]["field"] == "problem", f"sort fix drifted: {table['id']}"
        )
    normalized = copy.deepcopy(artifact)
    for table in normalized["manifest"]["tables"]:
        table["defaultSort"]["field"] = "sequence"
    normalized["package_info"]["originUrl"] = r1["package_info"]["originUrl"]
    base._require(normalized == r1, "r2 changes exceed the three-field allowlist")
    base._validate_artifact(artifact, registry_sha=EXPECTED_REGISTRY_SHA256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source pins and the complete three-field correction without writing.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    base._require(output.resolve() == (root / DEFAULT_OUTPUT).resolve(), "r2 output path is frozen")
    evidence, registry, r1 = collect_inputs(root)
    artifact = build_artifact(evidence, registry, r1)
    if args.check_only:
        print("PASS: validated r2 declared-column table-sort correction")
        return 0
    base._write_new(output, artifact)
    print(f"PASS: wrote corrected canonical artifact to {DEFAULT_OUTPUT.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
