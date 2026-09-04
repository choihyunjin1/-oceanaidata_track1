#!/usr/bin/env python
"""Independent read-only QA for the P2 nested-surrogate conformance artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    PROJECT_ROOT / "artifacts/p2_authoritative_nested_surrogate_conformance_20260825_v1"
)
EXPECTED_CONFIG_SHA256 = "7f84b707bf7059e947a9145f7df4fbbab762739db1b1e7d2d95feaede14a28b9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(artifact_dir: Path = DEFAULT_ARTIFACT) -> dict[str, Any]:
    directory = artifact_dir.resolve(strict=True)
    manifest_path = directory / "manifest.json"
    manifest = _read(manifest_path)
    _require(manifest["config"]["sha256"] == EXPECTED_CONFIG_SHA256, "config pin changed")
    _require(
        _sha256(PROJECT_ROOT / manifest["config"]["path"]) == EXPECTED_CONFIG_SHA256,
        "config bytes changed",
    )
    for filename, expected in manifest["outputs"].items():
        path = (directory / filename).resolve(strict=True)
        _require(path.parent == directory, "manifest output escaped artifact")
        _require(_sha256(path) == expected["sha256"], f"output hash changed: {filename}")
        _require(path.stat().st_size == int(expected["bytes"]), f"output size changed: {filename}")
    _require(
        _sha256(PROJECT_ROOT / manifest["conformance_source"]["path"])
        == manifest["conformance_source"]["sha256"],
        "conformance implementation changed after artifact",
    )
    _require(
        _sha256(PROJECT_ROOT / manifest["runner"]["path"])
        == manifest["runner"]["sha256"],
        "runner changed after artifact",
    )

    qa = _read(directory / "qa.json")
    result = _read(directory / "conformance_result.json")
    plan = _read(directory / "execution_plan.json")
    synthetic = _read(directory / "synthetic_conformance_receipts.json")
    metadata = _read(directory / "train_metadata_receipt.json")
    seeds = _read(directory / "child_seed_receipt.json")
    resource = _read(directory / "resource_estimate.json")

    _require(qa["pending_dimension_count"] == qa["pending_dimensions_passed"] == 3, "dimension QA failed")
    _require(len(result["dimensions"]) == 3, "result dimension count changed")
    _require(all(value.startswith("PASS_") for value in result["dimensions"].values()), "a dimension failed")
    _require(result["technical_go"] is True, "technical GO absent")
    _require(result["actual_45_cell_execution"] == "BLOCKED_AUTHORIZATION_ONLY", "fit gate changed")
    _require(plan["outer_prefix_cell_count"] == 15, "outer-prefix count changed")
    _require(plan["seeded_cell_count"] == 45, "seeded-cell count changed")
    _require(len({item["cell_id"] for item in plan["seeded_cells"]}) == 45, "duplicate cells")
    _require(not any(item["fit_authorized"] for item in plan["seeded_cells"]), "a fit is authorized")
    for prefix in plan["prefix_plans"]:
        _require(
            prefix["prefix_time_count"] == prefix["prefix_count_rule_expected"],
            "prefix count rule failed",
        )
        _require(len(prefix["inner_folds"]) == 3, "inner fold count changed")
        _require(all(item["strict_embargo_pass"] for item in prefix["inner_folds"]), "embargo failed")

    mask = synthetic["joint_target_mask"]
    _require(mask["temp_rows_masked"] == mask["psal_rows_masked"], "TEMP/PSAL mask differs")
    _require(mask["public_rows_changed"] == 0, "public layer changed")
    ledger = synthetic["component_oof_ledger"]
    _require(ledger["component_count"] == 5, "component count changed")
    _require(ledger["same_ordered_key_and_truth_across_components"], "OOF keys differ")
    meta = synthetic["prefix_local_meta_refit"]
    _require(not meta["frozen_stack_reused"] and not meta["frozen_gate_reused"], "frozen meta reused")
    for weights in meta["stack_weights"].values():
        _require(all(value >= 0 for value in weights.values()), "negative stack weight")
        _require(abs(sum(weights.values()) - 1.0) <= 1e-10, "stack is not sum-one")
    for receipt in synthetic["epoch_full_refit"].values():
        _require(receipt["full_prefix_refit"] and not receipt["frozen_epoch_reused"], "epoch refit failed")

    _require(metadata["columns_read"] == ["station", "layer", "time"], "metadata scope changed")
    _require(metadata["value_columns_read"] == [], "train values were read")
    _require(metadata["files_opened"] == ["observations.csv"], "unexpected data file opened")
    _require(seeds["child_seed_count"] == seeds["unique_child_seed_count"] == 900, "seed fan-out failed")
    _require(resource["fit_count_if_separately_authorized"] == {"deep": 720, "derivation": "45 seeded cells x (3 inner + 1 full) x (4 deep + 1 router)", "router": 180, "total": 900}, "resource fit counts changed")
    zero_keys = (
        "official_test_reads",
        "sample_submission_reads",
        "submission_candidate_reads",
        "submission_files_generated",
        "new_model_fits",
        "new_predictions",
        "uploads",
        "p3_era5_process_mutations",
    )
    _require(all(int(qa[key]) == 0 for key in zero_keys), "forbidden action count is nonzero")
    _require(not qa["official_public_score_used_for_selection"], "Public selected the recipe")
    _require(not qa["official_public_score_used_for_tuning"], "Public tuned the recipe")
    _require(not any(path.suffix.lower() in {".csv", ".parquet", ".pt"} for path in directory.iterdir()), "prediction/checkpoint artifact found")
    return {
        "schema_version": "p2_authoritative_nested_surrogate_conformance_independent_qa.v1",
        "status": "PASS_INDEPENDENT_READ_ONLY_QA",
        "manifest_sha256": _sha256(manifest_path),
        "verified_manifest_outputs": len(manifest["outputs"]),
        "pending_dimensions_passed": 3,
        "outer_prefix_cells": 15,
        "seeded_cells": 45,
        "unique_child_seeds": 900,
        "actual_model_fits": 0,
        "official_test_sample_submission_reads": 0,
        "public_score_selection_or_tuning": False,
        "p3_process_mutations": 0,
        "decision": "GO_TECHNICAL_CONFORMANCE_BLOCKED_AUTHORIZATION_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    print(json.dumps(validate(args.artifact_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
