"""Independent QA for the deadline parallel Deep Research v17 cycle."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/deadline_parallel_deep_research_20260828_v17"


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


def atomic_text(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    dirs = {
        "p1": ROOT
        / "artifacts/p1_station_pooled_hierarchical_residual_subset_scan_anchor_union_20260828_v1",
        "p2": ROOT
        / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2",
        "p3": ROOT
        / "artifacts/p3_era5_joint_wave_state_multitask_transfer_20260828_v1",
    }
    p1 = read_json(dirs["p1"] / "result.json")
    p1_manifest = read_json(dirs["p1"] / "manifest.json")
    p2 = read_json(dirs["p2"] / "result.json")
    p2_qa = read_json(dirs["p2"] / "independent_qa.json")
    p3 = read_json(dirs["p3"] / "result.json")
    p3_manifest = read_json(dirs["p3"] / "manifest.json")

    checks = {
        "p1_terminal_support_no_go": p1["decision"]
        == "NO_GO_SUPPORT_EXACT_E150_NO_OP",
        "p1_one_shot": int(p1["scientific_execution_count"]) == 1,
        "p1_no_active_fold_or_deletion": not p1["active_folds"]
        and int(p1["anchor_deletions"]) == 0,
        "p1_truth_and_official_zero": (
            int(p1["truth_rows_read_after_commitment"]) == 0
            and int(p1["official_test_sample_submission_rows_read"]) == 0
        ),
        "p1_result_hash_matches": sha256_file(dirs["p1"] / "result.json")
        == p1_manifest["artifact_hashes"]["result.json"],
        "p2_terminal_no_go": p2["decision"] == "NO_GO_EXACT_NO_OUTPUT",
        "p2_one_shot_no_prediction_rerun": (
            int(p2["execution_count"]) == 1 and p2["model_or_prediction_rerun"] is False
        ),
        "p2_nov_dec_exact_noop": abs(
            float(p2["metrics"]["by_fold"]["2025_nov_dec"]["delta_rmse"])
        )
        == 0.0,
        "p2_required_benefit_gate_failed": (
            p2["gate_checks"]["pooled_delta"] is False
            and p2["gate_checks"]["2024_sep_oct"] is False
        ),
        "p2_independent_qa_pass": p2_qa["status"] == "PASS",
        "p3_terminal_source_no_go": p3["status"] == "NO_GO_SOURCE_GATE",
        "p3_one_source_pair_only": int(p3["fit_count"]) == 2,
        "p3_ci_and_transport_gate_failed": (
            p3["source_gate"]["checks"]["bootstrap_ci90_upper_below_zero"] is False
            and p3["source_gate"]["checks"]["all_three_years_non_degrading"] is False
        ),
        "p3_shadow_truth_zero": (
            int(p3["preflight"]["fresh_shadow_support"]["outcome_values_read"]) == 0
            and p3["shadow_gate"] is None
        ),
        "p3_result_hash_matches": sha256_file(dirs["p3"] / "result.json")
        == p3_manifest["result_sha256"],
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
            for directory in dirs.values()
            for path in directory.rglob("*")
            if path.is_file()
        ),
    }
    inventory = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for directory in dirs.values()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    result = {
        "schema_version": "deep_research.independent_qa.20260828.v17",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": {
            "p1_decision": p1["decision"],
            "p2_decision": p2["decision"],
            "p3_decision": p3["status"],
            "submission_candidates": 0,
            "official_uploads": 0,
        },
        "artifact_inventory": inventory,
    }
    atomic_text(
        REPORT_DIR / "independent-qa.json",
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    manifest_files = [
        REPORT_DIR / "report-source.md",
        REPORT_DIR / "technical-report.md",
        REPORT_DIR / "gap-matrix.md",
        REPORT_DIR / "claim-source-ledger.md",
        REPORT_DIR / "independent-qa.json",
        ROOT / "scripts/qa_deadline_parallel_deep_research_20260828_v17.py",
        dirs["p1"] / "result.json",
        dirs["p1"] / "manifest.json",
        dirs["p2"] / "result.json",
        dirs["p2"] / "independent_qa.json",
        dirs["p3"] / "result.json",
        dirs["p3"] / "manifest.json",
    ]
    manifest = "".join(
        f"{sha256_file(path)}  {str(path.relative_to(ROOT)).replace(chr(92), '/')}\n"
        for path in manifest_files
    )
    atomic_text(REPORT_DIR / "manifest.sha256", manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
