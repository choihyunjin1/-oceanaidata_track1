"""Append-only r2 correction adding required visual blocks to the full-cycle report.

The r1 canonical artifact passed its internal analytical checks but the official
portable packager failed closed because the manifest declared charts and a
table without corresponding visible blocks.  This correction adds exactly one
table block and three chart blocks, then updates only the artifact generation
identity.  Registry, evidence, datasets, visuals, narrative, and source pins
remain byte-semantically unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

BASE_SCRIPT = Path(__file__).with_name("build_full_improvement_cycle_report_v1.py")
SPEC = importlib.util.spec_from_file_location("build_full_improvement_cycle_report_v1", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise RuntimeError(f"Could not import r1 builder: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

REPORT_ID = "full-improvement-cycle-2026-08-22-r2"
DEFAULT_OUTPUT = Path("reports/generated/full_improvement_cycle_2026-08-22_r2/artifact.json")
R1_ARTIFACT = Path("reports/generated/full_improvement_cycle_2026-08-22_r1/artifact.json")
EXPECTED_R1_ARTIFACT_SHA256 = "0102012c5ea109d9551fede360abc4d0ddc65cdd3b7c237c56b9853aefd7b90b"
EXPECTED_REGISTRY_SHA256 = "69fd2896a3df8626df6c80996683309236ad2953aa422beb7f9b0a99a157327c"

ADDED_BLOCKS = [
    {"id": "full_cycle_exact_table", "type": "table", "tableId": "full_cycle_exact_registry"},
    {"id": "p1_signed_chart", "type": "chart", "chartId": "p1_signed_validation_delta"},
    {"id": "p2_signed_chart", "type": "chart", "chartId": "p2_signed_validation_delta"},
    {"id": "p3_signed_chart", "type": "chart", "chartId": "p3_signed_validation_delta"},
]


def collect_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence = base.collect_evidence(root)
    registry_path = root / base.DEFAULT_REGISTRY
    base._require(registry_path.is_file(), "sealed full-cycle registry is missing")
    registry_actual = base._sha256(registry_path)
    base._require(
        registry_actual == EXPECTED_REGISTRY_SHA256,
        f"full-cycle registry SHA drifted: {registry_actual}",
    )
    registry = base._read_json(registry_path)
    base._validate_registry(registry)
    base._require(
        hashlib.sha256(base._canonical_bytes(registry)).hexdigest() == EXPECTED_REGISTRY_SHA256,
        "full-cycle registry serialization drifted",
    )

    r1_path = root / R1_ARTIFACT
    base._require(r1_path.is_file(), "r1 artifact is missing")
    r1_actual = base._sha256(r1_path)
    base._require(r1_actual == EXPECTED_R1_ARTIFACT_SHA256, f"r1 artifact SHA drifted: {r1_actual}")
    r1 = base._read_json(r1_path)
    base._validate_artifact(r1, registry_sha=EXPECTED_REGISTRY_SHA256)
    return evidence, registry, r1


def build_artifact(r1: dict[str, Any]) -> dict[str, Any]:
    artifact = copy.deepcopy(r1)
    blocks = artifact["manifest"]["blocks"]
    key_findings_index = next(
        index for index, block in enumerate(blocks) if block["id"] == "key_findings"
    )
    blocks[key_findings_index + 1 : key_findings_index + 1] = copy.deepcopy(ADDED_BLOCKS)
    artifact["package_info"]["originUrl"] = f"artifact://{REPORT_ID}"
    _validate_correction(artifact, r1)
    return artifact


def _validate_correction(artifact: dict[str, Any], r1: dict[str, Any]) -> None:
    blocks = artifact["manifest"]["blocks"]
    block_ids = [block["id"] for block in blocks]
    base._require(len(block_ids) == len(set(block_ids)), "duplicate block IDs")
    chart_block_ids = [block["chartId"] for block in blocks if block["type"] == "chart"]
    declared_chart_ids = [chart["id"] for chart in artifact["manifest"]["charts"]]
    base._require(
        chart_block_ids == declared_chart_ids, "chart blocks do not match declared charts"
    )
    table_block_ids = [block["tableId"] for block in blocks if block["type"] == "table"]
    declared_table_ids = [table["id"] for table in artifact["manifest"]["tables"]]
    base._require(
        table_block_ids == declared_table_ids, "table blocks do not match declared tables"
    )

    normalized = copy.deepcopy(artifact)
    added_ids = {block["id"] for block in ADDED_BLOCKS}
    normalized["manifest"]["blocks"] = [
        block for block in normalized["manifest"]["blocks"] if block["id"] not in added_ids
    ]
    normalized["package_info"]["originUrl"] = r1["package_info"]["originUrl"]
    base._require(normalized == r1, "r2 changes exceed visual-block and origin identity allowlist")
    base._validate_artifact(artifact, registry_sha=EXPECTED_REGISTRY_SHA256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    base._require(output.resolve() == (root / DEFAULT_OUTPUT).resolve(), "r2 output path is frozen")
    _, _, r1 = collect_inputs(root)
    artifact = build_artifact(r1)
    artifact_sha = hashlib.sha256(base._canonical_bytes(artifact)).hexdigest()
    if args.check_only:
        print(f"PASS: validated r2 visual-block correction ({artifact_sha}); no files written")
        return 0
    base._write_new(output, artifact)
    print(
        f"PASS: wrote corrected canonical artifact to {DEFAULT_OUTPUT.as_posix()} ({artifact_sha})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
