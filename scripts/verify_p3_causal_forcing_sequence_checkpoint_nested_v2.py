"""Independent post-execution QA for the P3 nested-checkpoint v2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


CANONICAL_ARTIFACT_RELATIVE = (
    "artifacts/p3_causal_forcing_sequence_checkpoint_nested_20260827_v2"
)
CANONICAL_QA_RELATIVE = (
    "artifacts/p3_causal_forcing_sequence_checkpoint_nested_20260827_v2_QA/"
    "independent_qa.json"
)
PREFIXES = (1.00,)
KEYS = ["fold", "anchor_id", "station", "lead_h"]


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(content).hexdigest()


def verify(*, root: Path, artifact: Path) -> dict[str, Any]:
    canonical = (root / CANONICAL_ARTIFACT_RELATIVE).resolve(strict=True)
    if artifact.resolve(strict=True) != canonical:
        raise PermissionError("only the canonical nested-checkpoint artifact may be verified")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    registry = json.loads((artifact / "registry.json").read_text(encoding="utf-8"))
    blind_commit = json.loads((artifact / "blind_outer_commit.json").read_text(encoding="utf-8"))
    recorded_manifest_sha = (artifact / "manifest.sha256").read_text(encoding="ascii").split()[0]

    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(_sha256(artifact / "manifest.json") == recorded_manifest_sha, "manifest SHA differs")
    output_files = manifest.get("output_files", {})
    for relative, receipt in output_files.items():
        path = artifact / relative
        require(path.is_file(), f"manifest output missing: {relative}")
        if path.is_file():
            require(int(path.stat().st_size) == int(receipt["bytes"]), f"byte size differs: {relative}")
            require(_sha256(path) == receipt["sha256"], f"SHA differs: {relative}")

    inner_checkpoint_paths = [
        relative
        for relative in output_files
        if relative.startswith("inner_checkpoints/") and relative.endswith(".pt")
    ]
    outer_model_paths = [
        relative
        for relative in output_files
        if relative.startswith("outer_models/") and relative.endswith("model.pt")
    ]
    blind_paths = [
        relative
        for relative in output_files
        if relative.startswith("blind_predictions/") and relative.endswith(".npy")
    ]
    require(len(inner_checkpoint_paths) == 72, "inner checkpoint count is not 72")
    require(len(outer_model_paths) == 9, "outer model count is not 9")
    require(len(blind_paths) == 9, "blind prediction count is not 9")

    selection_receipts = metrics.get("inner_selection_receipts", [])
    require(len(selection_receipts) == 3, "selection receipt count is not 3")
    earliest_minimum_ok = True
    selected_epochs: list[int] = []
    for receipt in selection_receipts:
        history = np.asarray(receipt.get("rmse_by_epoch_m", []), dtype=float)
        selected = int(receipt.get("selected_epoch", 0))
        selected_epochs.append(selected)
        if history.shape != (8,) or not np.isfinite(history).all():
            earliest_minimum_ok = False
            continue
        if selected != int(np.argmin(history)) + 1:
            earliest_minimum_ok = False
    require(earliest_minimum_ok, "selected epoch is not the earliest exact inner minimum")
    require(all(1 <= epoch <= 8 for epoch in selected_epochs), "selected epoch is outside 1..8")

    outer_receipts = metrics.get("outer_refit_receipts", [])
    require(len(outer_receipts) == 9, "outer refit receipt count is not 9")
    require(
        all(row.get("outer_reference_values_opened") is False for row in outer_receipts),
        "an outer receipt reports reference access before blind commit",
    )
    require(
        blind_commit.get("prediction_file_count") == 9
        and blind_commit.get("outer_reference_values_opened_before_commit") is False,
        "blind outer commit contract differs",
    )
    commit_paths = {
        str(row["blind_prediction_relative_path"]): str(row["blind_prediction_sha256"])
        for row in blind_commit.get("prediction_files", [])
    }
    require(set(commit_paths) == set(blind_paths), "blind commit path set differs")
    for relative, sha in commit_paths.items():
        require(output_files.get(relative, {}).get("sha256") == sha, f"blind commit SHA differs: {relative}")

    oof = pd.read_parquet(artifact / "oof/learning_curve_oof.parquet")
    required_oof = {
        *KEYS,
        "prefix_fraction",
        "target_hs",
        "incumbent_prediction",
        "fixed8_prediction",
        "checkpoint_nested_prediction",
    }
    require(required_oof.issubset(oof.columns), "OOF columns differ")
    require(len(oof) == 1086, "OOF row count differs")
    require(not oof.duplicated(["prefix_fraction", *KEYS]).any(), "OOF keys are duplicated")
    for prefix in PREFIXES:
        current = oof.loc[np.isclose(oof["prefix_fraction"].to_numpy(float), prefix)]
        require(len(current) == 1086, f"OOF prefix row count differs: {prefix}")
        blocks = current.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
            lambda values: tuple(sorted(values.astype(int)))
        )
        require(
            blocks.map(lambda value: value == (3, 6, 9, 12, 18, 24)).all(),
            f"OOF six-lead block differs: {prefix}",
        )
    numeric = oof[
        ["target_hs", "incumbent_prediction", "fixed8_prediction", "checkpoint_nested_prediction"]
    ].to_numpy(float)
    require(np.isfinite(numeric).all(), "OOF contains a non-finite value")
    require(
        np.all((oof["checkpoint_nested_prediction"] >= 0.0) & (oof["checkpoint_nested_prediction"] <= 30.0)),
        "nested checkpoint prediction is outside 0..30m",
    )

    access = metrics.get("access_counters", {})
    zero_access_keys = (
        "outer_reference_parsed_reads_before_blind_commit",
        "anonymous_evaluation_value_reads",
        "hidden_target_reads",
        "submission_artifact_reads",
        "submission_artifact_writes",
        "upload_attempts",
        "era5_artifact_or_process_reads",
        "era5_artifact_or_process_writes",
    )
    require(all(int(access.get(key, -1)) == 0 for key in zero_access_keys), "forbidden access counter is nonzero")
    require(int(access.get("outer_reference_parsed_reads_after_blind_commit", -1)) == 1, "post-commit reference read counter differs")
    require(metrics.get("candidate_created") is False, "metrics reports a candidate")
    require(metrics.get("candidate_uploaded") is False, "metrics reports an upload")
    require(registry.get("candidate_created") is False, "registry reports a candidate")
    require(registry.get("candidate_uploaded") is False, "registry reports an upload")
    require(not list(artifact.rglob("*.csv")), "artifact unexpectedly contains a CSV")

    result = {
        "schema_version": "p3_checkpoint_nested_independent_qa.v1",
        "created_at": _now(),
        "artifact": CANONICAL_ARTIFACT_RELATIVE,
        "status": "PASS" if not failures else "FAIL",
        "checks": {
            "manifest_sha_and_output_hashes": not any("SHA" in value or "byte" in value for value in failures),
            "inner_checkpoint_count": len(inner_checkpoint_paths),
            "outer_model_count": len(outer_model_paths),
            "blind_prediction_count": len(blind_paths),
            "earliest_exact_minimum": earliest_minimum_ok,
            "selected_epoch_min": min(selected_epochs) if selected_epochs else None,
            "selected_epoch_max": max(selected_epochs) if selected_epochs else None,
            "oof_rows": int(len(oof)),
            "forbidden_access_counters_zero": all(
                int(access.get(key, -1)) == 0 for key in zero_access_keys
            ),
            "candidate_absent": metrics.get("candidate_created") is False,
            "upload_absent": metrics.get("candidate_uploaded") is False,
        },
        "failures": failures,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact", type=Path, default=Path(CANONICAL_ARTIFACT_RELATIVE))
    parser.add_argument("--output", type=Path, default=Path(CANONICAL_QA_RELATIVE))
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    artifact = args.artifact if args.artifact.is_absolute() else root / args.artifact
    output = args.output if args.output.is_absolute() else root / args.output
    result = verify(root=root, artifact=artifact)
    result["qa_sha256"] = _write_exclusive(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
