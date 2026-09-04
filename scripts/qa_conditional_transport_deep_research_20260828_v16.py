"""Independent QA for the P1/P2/P3 Deep Research v16 one-shot cycle."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/conditional_transport_deep_research_20260828_v16"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    artifact_dirs = {
        "p1": ROOT
        / "artifacts/p1_async_latent_state_gp_subset_scan_anchor_union_20260828_v1",
        "p2": ROOT
        / "artifacts/p2_alpha50_supervised_rank1_trainonly_regime_veto_20260828_v1",
        "p3": ROOT
        / "artifacts/p3_era5_wave_directional_energy_memory_20260828_v1",
    }
    p1 = read_json(artifact_dirs["p1"] / "result.json")
    p1_manifest = read_json(artifact_dirs["p1"] / "manifest.json")
    p1_commitment = read_json(artifact_dirs["p1"] / "prediction_commitment.json")
    p2 = read_json(artifact_dirs["p2"] / "result.json")
    p3 = read_json(artifact_dirs["p3"] / "result.json")
    p3_manifest = read_json(artifact_dirs["p3"] / "manifest.json")

    p1_result_hash = sha256_file(artifact_dirs["p1"] / "result.json")
    p1_seal_hash = sha256_file(artifact_dirs["p1"] / "sealed_predictions.npz")
    p3_result_hash = sha256_file(artifact_dirs["p3"] / "result.json")
    p3_artifact_hashes = {
        name: sha256_file(path)
        for name, path in {
            "source_base_model": artifact_dirs["p3"] / "models/source_base.cbm",
            "source_enriched_model": artifact_dirs["p3"] / "models/source_enriched.cbm",
            "source_prediction_receipt": artifact_dirs["p3"]
            / "source_prediction_seal.json",
            "source_prediction_seal": artifact_dirs["p3"]
            / "source_predictions_sealed.parquet",
        }.items()
    }

    checks = {
        "p1_terminal_support_no_go": p1["decision"]
        == "NO_GO_SUPPORT_EXACT_E150_NO_OP",
        "p1_one_shot_zero_fit": (
            int(p1["scientific_execution_count"]) == 1
            and int(p1["fixed_hyperparameter_model_fit_count"]) == 0
        ),
        "p1_exact_no_anchor_deletion": int(p1["anchor_deletions"]) == 0,
        "p1_truth_not_opened": int(p1["truth_rows_read_after_commitment"]) == 0,
        "p1_result_hash_matches": p1_result_hash
        == p1_manifest["artifact_hashes"]["result.json"],
        "p1_prediction_seal_matches": p1_seal_hash
        == p1_commitment["sealed_file_sha256"]
        == p1_manifest["artifact_hashes"]["sealed_predictions.npz"],
        "p2_terminal_structural_no_go": (
            p2["decision"] == "NO_GO_IMPLEMENTATION_PREFLIGHT"
            and p2["terminal"] is True
            and p2["failed_stage"] == "inner_oof_alpha50_reference_support"
        ),
        "p2_no_commitment_or_outer_truth": (
            p2["prediction_commitment_written"] is False
            and int(p2["outer_validation_truth_rows_read"]) == 0
            and p2["outer_metrics_computed"] is False
            and int(p2["prediction_files_written"]) == 0
        ),
        "p2_no_retry": p2["retry_performed"] is False,
        "p3_terminal_source_no_go": p3["status"] == "NO_GO_SOURCE_GATE",
        "p3_source_gate_failed": p3["source_gate"]["passed"] is False,
        "p3_source_delta_is_regression": float(
            p3["source_gate"]["overall"]["delta_m"]
        )
        > 0.0,
        "p3_shadow_truth_not_opened": (
            int(p3["preflight"]["fresh_shadow_support"]["outcome_values_read"])
            == 0
            and p3["shadow_gate"] is None
        ),
        "p3_result_hash_matches": p3_result_hash == p3_manifest["result_sha256"],
        "p3_artifact_hashes_match": p3_artifact_hashes
        == p3_manifest["artifacts"]
        == p3["artifact_hashes"],
        "official_rows_zero_all": all(
            int(result["official_test_sample_submission_rows_read"]) == 0
            for result in (p1, p2, p3)
        ),
        "submission_generation_zero_all": all(
            result["submission_generated_or_uploaded"] is False
            for result in (p1, p2, p3)
        ),
        "no_csv_in_cycle_artifacts": not any(
            path.suffix.lower() == ".csv"
            for directory in artifact_dirs.values()
            for path in directory.rglob("*")
            if path.is_file()
        ),
    }

    inventory = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for directory in artifact_dirs.values()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    qa = {
        "schema_version": "deep_research.independent_qa.20260828.v16",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": {
            "p1_decision": p1["decision"],
            "p2_decision": p2["decision"],
            "p3_decision": p3["status"],
            "official_uploads": 0,
            "submission_candidates": 0,
        },
        "artifact_inventory": inventory,
    }
    atomic_text(
        REPORT_DIR / "independent-qa.json",
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )

    manifest_files = [
        REPORT_DIR / "report-source.md",
        REPORT_DIR / "technical-report.md",
        REPORT_DIR / "gap-matrix.md",
        REPORT_DIR / "claim-source-ledger.md",
        REPORT_DIR / "independent-qa.json",
        ROOT / "scripts/qa_conditional_transport_deep_research_20260828_v16.py",
        artifact_dirs["p1"] / "result.json",
        artifact_dirs["p1"] / "manifest.json",
        artifact_dirs["p2"] / "result.json",
        artifact_dirs["p3"] / "result.json",
        artifact_dirs["p3"] / "manifest.json",
    ]
    manifest = "".join(
        f"{sha256_file(path)}  {str(path.relative_to(ROOT)).replace(chr(92), '/')}\n"
        for path in manifest_files
    )
    atomic_text(REPORT_DIR / "manifest.sha256", manifest)
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
