"""Independent artifact-only QA for the one-shot P1 Sobol HPO run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_mstcn_sobol_hpo_20260829_v1"
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID


class VerificationError(RuntimeError):
    """Raised when an execution artifact contradicts the registered contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(*, root: Path = ROOT) -> dict[str, Any]:
    artifact_dir = root / "artifacts" / EXPERIMENT_ID
    config_path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    runner_path = root / "scripts" / f"run_{EXPERIMENT_ID}.py"
    required = [
        config_path,
        runner_path,
        artifact_dir / "preflight.json",
        artifact_dir / "sealed_design.json",
        artifact_dir / "sealed_design_receipt.json",
        artifact_dir / "q2_discovery_blind_receipt.json",
        artifact_dir / "q2_top2_three_seed_blind_receipt.json",
        artifact_dir / "q2_preconfirm_gate.json",
        artifact_dir / "aggregate.json",
        artifact_dir / "terminal_result.json",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise VerificationError(f"required artifacts are absent: {missing}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight = json.loads((artifact_dir / "preflight.json").read_text(encoding="utf-8"))
    design = json.loads((artifact_dir / "sealed_design.json").read_text(encoding="utf-8"))
    design_receipt = json.loads(
        (artifact_dir / "sealed_design_receipt.json").read_text(encoding="utf-8")
    )
    discovery_receipt = json.loads(
        (artifact_dir / "q2_discovery_blind_receipt.json").read_text(encoding="utf-8")
    )
    top2_receipt = json.loads(
        (artifact_dir / "q2_top2_three_seed_blind_receipt.json").read_text(encoding="utf-8")
    )
    gate = json.loads((artifact_dir / "q2_preconfirm_gate.json").read_text(encoding="utf-8"))
    aggregate = json.loads((artifact_dir / "aggregate.json").read_text(encoding="utf-8"))
    terminal = json.loads((artifact_dir / "terminal_result.json").read_text(encoding="utf-8"))
    discovery_score = artifact_dir / discovery_receipt["score_path"]
    top2_score = artifact_dir / top2_receipt["score_path"]
    all_fit_receipts = discovery_receipt["fit_receipts"] + top2_receipt["fit_receipts"]
    checks = {
        "experiment_identity": all(
            value.get("experiment_id") == EXPERIMENT_ID
            for value in (config, preflight, design, aggregate, terminal)
        ),
        "config_hash_matches_preflight": _sha256(config_path) == preflight["config_sha256"],
        "runner_hash_matches_preflight": _sha256(runner_path) == preflight["runner_sha256"],
        "design_has_32_points": len(design["points"]) == 32,
        "design_hash_matches_receipt": _sha256(artifact_dir / "sealed_design.json")
        == design_receipt["design_sha256"],
        "design_sealed_before_fit": design_receipt["sealed_before_first_fit"] is True,
        "discovery_blind_hash_matches_receipt": discovery_score.is_file()
        and _sha256(discovery_score) == discovery_receipt["score_sha256"]
        and discovery_score.stat().st_size == discovery_receipt["score_bytes"],
        "top2_blind_hash_matches_receipt": top2_score.is_file()
        and _sha256(top2_score) == top2_receipt["score_sha256"]
        and top2_score.stat().st_size == top2_receipt["score_bytes"],
        "discovery_fit_count_32": aggregate["discovery_fits"] == 32,
        "top2_additional_seed_fit_count_4": aggregate["top2_additional_seed_fits"] == 4,
        "receipt_fit_counts_32_plus_4": len(discovery_receipt["fit_receipts"]) == 32
        and len(top2_receipt["fit_receipts"]) == 4,
        "all_fits_reached_epoch_150": all(
            receipt["epochs"] == 150 for receipt in all_fit_receipts
        ),
        "all_fits_nonfinite_zero": all(
            receipt["nonfinite_count_total"] == 0 for receipt in all_fit_receipts
        ),
        "all_fit_checkpoints_absent": all(
            receipt["checkpoint_persisted"] is False for receipt in all_fit_receipts
        ),
        "blind_receipts_precede_truth": discovery_receipt[
            "same_fold_holdout_truth_columns_opened_before_receipt"
        ]
        == 0
        and top2_receipt["same_fold_holdout_truth_columns_opened_before_receipt"] == 0,
        "receipt_official_rows_zero": discovery_receipt["official_interface_rows_read"] == 0
        and top2_receipt["official_interface_rows_read"] == 0,
        "torch_cpu_thread_cap": preflight["torch_threads"] == {"intraop": 2, "interop": 1},
        "selected_epoch_is_150": aggregate["selected_recipe"]["epoch"] == 150,
        "selected_recipe_matches_gate": aggregate["selected_recipe"] == gate["selected_recipe"],
        "preconfirm_decision_consistent": gate["decision"]
        == (
            "PASS_TO_CONFIRMATION"
            if all(gate["checks"].values())
            else "STOP_BEFORE_CONFIRMATION"
        ),
        "official_rows_zero": aggregate["official_interface_rows_read"] == 0,
        "csv_not_created": aggregate["csv_created"] is False
        and not any(artifact_dir.rglob("*.csv")),
        "upload_not_performed": aggregate["upload_performed"] is False,
        "checkpoint_not_persisted": not any(artifact_dir.rglob("*.pt")),
        "terminal_matches_aggregate": terminal == aggregate,
        "no_result_based_rerun": aggregate["result_based_rerun_authorized"] is False,
    }
    if gate["decision"] == "PASS_TO_CONFIRMATION":
        for phase in ("q3", "q4"):
            checks[f"{phase}_blind_receipt_exists"] = (
                artifact_dir / f"{phase}_confirmatory_blind_receipt.json"
            ).is_file()
        checks["confirmatory_metrics_exist"] = (artifact_dir / "confirmatory_metrics.json").is_file()
        checks["confirmatory_fit_count_6"] = aggregate.get("confirmatory_fits") == 6
    else:
        checks["confirmation_not_started"] = aggregate.get("q3_q4_training_started") is False
        checks["confirmatory_artifacts_absent"] = not any(
            artifact_dir.glob("q[34]_confirmatory_blind*")
        )
    return {
        "schema_version": "p1.mstcn_sobol_hpo.independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = verify()
    if args.write:
        path = ARTIFACT_DIR / "independent_qa.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
