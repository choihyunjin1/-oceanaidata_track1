"""Independent aggregate/key/provenance QA for the completed P1 cycle."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from p1_qc.improvement_cycle import (
    PROJECT_ROOT,
    _aligned_oof,
    _evaluate_branch,
    load_cycle_config,
    sha256_file,
    write_json_exclusive,
)


def main() -> int:
    config_path, config = load_cycle_config("configs/experiments/p1_full_improvement_cycle_v1.json")
    root = PROJECT_ROOT / config["paths"]["artifact_root"]
    qa_dir = root / "qa"
    if qa_dir.exists():
        raise FileExistsError(qa_dir)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for relative, pin in manifest["artifacts"].items():
        path = PROJECT_ROOT / relative
        if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
            mismatches.append(relative)

    stored = pd.read_parquet(root / "winner_oof.parquet")
    incumbent_source = pd.read_parquet(PROJECT_ROOT / config["paths"]["incumbent_oof"])
    keys = config["key_columns"]
    exact_outer_keys = stored[keys].equals(incumbent_source[keys])
    exact_outer_truth = stored["label"].equals(incumbent_source["label"])
    truth = stored["label"].to_numpy(dtype=np.int8)
    incumbent = stored["incumbent_prediction"].to_numpy(dtype=np.int8)
    candidate = stored["candidate_prediction"].to_numpy(dtype=np.int8)
    independent_incumbent_f1 = float(f1_score(truth, incumbent))
    independent_candidate_f1 = float(f1_score(truth, candidate))

    frame, aligned_truth, x_pred, c_pred, px, pc = _aligned_oof(config)
    reconstructed, selection_report, _ = _evaluate_branch(
        "causal_event_rescue_walk_forward",
        config,
        frame,
        aligned_truth,
        x_pred,
        c_pred,
        px,
        pc,
    )
    exact_selection_reconstruction = bool(np.array_equal(candidate, reconstructed))
    selections = selection_report["walk_forward_selections"]
    split_contract = {
        "q2_exact_incumbent_fallback": selections["2025_q2"]["rule"] == "exact_incumbent_fallback",
        "q3_calibrated_only_on_q2": selections["2025_q3"]["calibration_folds"] == ["2025_q2"],
        "q4_calibrated_only_on_q2_q3": selections["2025_q4"]["calibration_folds"]
        == ["2025_q2", "2025_q3"],
    }

    test = pd.read_csv(PROJECT_ROOT / config["paths"]["test_csv"], low_memory=False)
    candidate_path = root / "candidate/P1_IMPROVED_ENSEMBLE_V1.csv"
    reproduced_path = root / "candidate/reproduced.csv"
    submission = pd.read_csv(candidate_path, keep_default_na=False, low_memory=False)
    candidate_checks = {
        "rows": len(submission),
        "key_order_exact": submission[keys].equals(test[keys]),
        "duplicate_keys": int(submission.duplicated(keys).sum()),
        "null_key_cells": int(submission[keys].isna().sum().sum()),
        "binary_label": bool(submission["label"].isin([0, 1]).all()),
        "integer_label": bool(pd.api.types.is_integer_dtype(submission["label"])),
        "candidate_reproduction_byte_identical": candidate_path.read_bytes()
        == reproduced_path.read_bytes(),
    }
    frozen = pd.read_csv(PROJECT_ROOT / config["paths"]["frozen_submission"], keep_default_na=False)
    candidate_checks["changed_vs_frozen"] = int((submission["label"] != frozen["label"]).sum())
    candidate_checks["added_vs_frozen"] = int(
        ((submission["label"] == 1) & (frozen["label"] == 0)).sum()
    )
    candidate_checks["removed_vs_frozen"] = int(
        ((submission["label"] == 0) & (frozen["label"] == 1)).sum()
    )

    ensemble_path = root / "resume_models/p1_improved_ensemble_corrected.joblib"
    ensemble = joblib.load(ensemble_path)
    model_checks = {
        "winner_branch": ensemble.winner_branch,
        "incumbent_backend": ensemble.incumbent_model.backend,
        "incumbent_feature_mode": ensemble.incumbent_model.feature_mode,
        "incumbent_feature_count": len(ensemble.incumbent_model.encoder.feature_columns),
        "causal_backend": ensemble.causal_model.backend,
        "causal_feature_mode": ensemble.causal_model.feature_mode,
        "causal_feature_count": len(ensemble.causal_model.encoder.feature_columns),
        "causal_future_feature_count": len(
            [
                name
                for name in ensemble.causal_model.encoder.feature_columns
                if "next" in name or "center" in name or "full_length" in name
            ]
        ),
        "deployment_parameters": ensemble.deployment_parameters,
    }

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    numerical_checks = {
        "incumbent_f1_matches_metrics": independent_incumbent_f1
        == metrics["guard"]["incumbent"]["f1"],
        "candidate_f1_matches_metrics": independent_candidate_f1
        == metrics["guard"]["candidate"]["f1"],
        "result_delta_matches": result["outer_f1_delta"]
        == independent_candidate_f1 - independent_incumbent_f1,
        "candidate_strictly_improves": independent_candidate_f1 > independent_incumbent_f1,
        "all_fold_deltas_nonnegative": min(metrics["guard"]["by_fold_f1_delta"].values()) >= 0,
        "all_station_deltas_nonnegative": min(metrics["guard"]["by_station_f1_delta"].values())
        >= 0,
    }
    protected_checks = {
        name: sha256_file(PROJECT_ROOT / config["paths"][name]) == expected
        for name, expected in config["expected_sha256"].items()
    }
    checks = {
        "manifest_pin_mismatch_count": len(mismatches),
        "outer_rows": len(stored),
        "outer_duplicate_keys": int(stored.duplicated(keys).sum()),
        "exact_outer_keys": exact_outer_keys,
        "exact_outer_truth": exact_outer_truth,
        "exact_selection_reconstruction": exact_selection_reconstruction,
        "split_contract": split_contract,
        "candidate": candidate_checks,
        "models": model_checks,
        "numerical": numerical_checks,
        "protected_sha": protected_checks,
    }
    passed = (
        not mismatches
        and len(stored) == 421032
        and not stored.duplicated(keys).any()
        and exact_outer_keys
        and exact_outer_truth
        and exact_selection_reconstruction
        and all(split_contract.values())
        and candidate_checks["rows"] == 169011
        and candidate_checks["key_order_exact"]
        and candidate_checks["duplicate_keys"] == 0
        and candidate_checks["null_key_cells"] == 0
        and candidate_checks["binary_label"]
        and candidate_checks["integer_label"]
        and candidate_checks["candidate_reproduction_byte_identical"]
        and model_checks["incumbent_feature_mode"] == "offline"
        and model_checks["incumbent_feature_count"] == 80
        and model_checks["causal_feature_mode"] == "causal"
        and model_checks["causal_feature_count"] == 76
        and model_checks["causal_future_feature_count"] == 0
        and all(numerical_checks.values())
        and all(protected_checks.values())
    )
    receipt = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "validated_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "decision": "QA_PASS" if passed else "QA_FAIL",
        "P0_finding_count": 0 if passed else 1,
        "P1_finding_count": 0,
        "checks": checks,
        "independent_metrics": {
            "incumbent_micro_f1": independent_incumbent_f1,
            "candidate_micro_f1": independent_candidate_f1,
            "delta": independent_candidate_f1 - independent_incumbent_f1,
        },
        "selection_reconstruction": selection_report,
        "artifact_sha256": {
            "config": sha256_file(config_path),
            "result": sha256_file(root / "result.json"),
            "metrics": sha256_file(root / "metrics.json"),
            "manifest": sha256_file(root / "manifest.json"),
            "winner_oof": sha256_file(root / "winner_oof.parquet"),
            "candidate": sha256_file(candidate_path),
            "corrected_ensemble": sha256_file(ensemble_path),
            "validator": sha256_file(Path(__file__)),
        },
        "caveat": "The exact walk-forward point estimate improves, but the preregistered KST-day bootstrap 90% CI includes zero; this is local OOF evidence, not a hidden-test guarantee.",
        "test_label_reads": 0,
        "submission_uploads": 0,
    }
    qa_dir.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(qa_dir / "independent_validation.json", receipt)
    if not passed:
        raise RuntimeError("P1 full-cycle independent QA failed")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
