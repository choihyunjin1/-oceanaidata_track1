"""Independent aggregate reconciliation for the P3 breakthrough reconnaissance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.validation import rmse

LEADS = {3, 6, 9, 12, 18, 24}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: float, right: float, tolerance: float = 1e-12) -> None:
    if not np.isclose(left, right, rtol=0.0, atol=tolerance):
        raise ValueError(f"metric mismatch: {left!r} != {right!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="artifacts/p3/final_ensemble_validation/oof.parquet")
    parser.add_argument("--failure", default="artifacts/p3/failure_recon/diagnostics.json")
    parser.add_argument("--bias", default="artifacts/p3/bias_correction_probe/metrics.json")
    parser.add_argument("--bias-oof", default="artifacts/p3/bias_correction_probe/oof.parquet")
    parser.add_argument("--router", default="artifacts/p3/trajectory_router_probe/metrics.json")
    parser.add_argument(
        "--sea-state-validation", default="artifacts/p3/sea_state_probe/paired_validation.json"
    )
    parser.add_argument(
        "--amplitude-validation",
        default="artifacts/p3/amplitude_weight_probe/paired_validation.json",
    )
    parser.add_argument("--output", default="artifacts/p3/failure_recon/validation.json")
    args = parser.parse_args()
    paths = {name: Path(value) for name, value in vars(args).items()}
    required = [
        paths["oof"],
        paths["failure"],
        paths["bias"],
        paths["bias_oof"],
        paths["sea_state_validation"],
        paths["amplitude_validation"],
    ]
    if paths["router"].exists():
        required.append(paths["router"])
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    oof = pd.read_parquet(paths["oof"])
    key = ["fold", "anchor_id", "station", "lead_h"]
    if len(oof) != 1_092 or oof["anchor_id"].nunique() != 182:
        raise ValueError("expected exactly 182 cases and 1,092 forecast rows")
    if oof.duplicated(key).any() or set(oof["lead_h"].astype(int)) != LEADS:
        raise ValueError("OOF key or lead contract failed")
    per_case = oof.groupby(["fold", "anchor_id"], sort=False)["lead_h"].agg(["size", "nunique"])
    if not per_case.eq(6).all().all():
        raise ValueError("each case must contain six distinct leads")
    prediction_columns = [
        "target_hs",
        "single_prediction",
        "multi_prediction",
        "persistence",
        "prediction",
    ]
    if not np.isfinite(oof[prediction_columns].to_numpy(float)).all():
        raise ValueError("OOF contains non-finite truth or predictions")

    failure = _read(paths["failure"])
    recomputed = {
        "ensemble": rmse(oof["target_hs"], oof["prediction"]),
        "single": rmse(oof["target_hs"], oof["single_prediction"]),
        "multi": rmse(oof["target_hs"], oof["multi_prediction"]),
        "persistence": rmse(oof["target_hs"], oof["persistence"]),
    }
    for name, value in recomputed.items():
        _close(value, float(failure["overall"][name]["rmse"]))
    cut_checks: dict[str, object] = {}
    for cut, rows in failure["cuts"].items():
        total_rows = sum(int(row["rows"]) for row in rows)
        share = sum(float(row["squared_error_share"]) for row in rows)
        if total_rows != len(oof):
            raise ValueError(f"{cut} rows do not reconcile")
        _close(share, 1.0)
        cut_checks[cut] = {"rows": total_rows, "squared_error_share_sum": share}

    bias = _read(paths["bias"])
    bias_oof = pd.read_parquet(paths["bias_oof"])
    if bias_oof[key].duplicated().any() or len(bias_oof) != len(oof):
        raise ValueError("bias OOF key contract failed")
    frozen_error = float(
        np.max(
            np.abs(
                bias_oof.sort_values(key)["frozen_prediction"].to_numpy(float)
                - oof.sort_values(key)["prediction"].to_numpy(float)
            )
        )
    )
    _close(frozen_error, 0.0)
    bias_recomputed = {
        "corrected": rmse(bias_oof["target_hs"], bias_oof["prediction"]),
        "frozen_ensemble": rmse(bias_oof["target_hs"], bias_oof["frozen_prediction"]),
    }
    for name, value in bias_recomputed.items():
        _close(value, float(bias["metrics"][name]["rmse"]))

    router_check: dict[str, object] | None = None
    if paths["router"].exists():
        router = _read(paths["router"])
        class_count = sum(int(value) for value in router["overall"]["class_counts"].values())
        confusion_count = sum(
            sum(int(value) for value in row) for row in router["overall"]["confusion"]
        )
        fold_count = sum(int(row["validation_cases"]) for row in router["folds"].values())
        if class_count != 182 or confusion_count != 182 or fold_count != 182:
            raise ValueError("router aggregate counts do not reconcile")
        router_check = {
            "cases": class_count,
            "confusion_total": confusion_count,
            "fold_case_total": fold_count,
        }

    matched_checks: dict[str, object] = {}
    for name in ("sea_state_validation", "amplitude_validation"):
        path = paths[name]
        probe = _read(path)
        grain = probe["grain"]
        if (
            probe["status"] != "passed"
            or int(grain["cases"]) != 182
            or int(grain["rows_per_arm"]) != 1_092
            or int(grain["leads_per_case"]) != 6
        ):
            raise ValueError(f"{name} grain or status contract failed")
        matched_checks[name] = {
            "delta_rmse": float(probe["paired_case_bootstrap"]["delta_rmse_candidate_minus_base"]),
            "ci90": [float(value) for value in probe["paired_case_bootstrap"]["ci90"]],
            "probability_improved": float(
                probe["paired_case_bootstrap"]["probability_candidate_improved"]
            ),
        }

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "passed",
        "confidence": "ready_to_share_with_caveats",
        "grain": {
            "cases": int(oof["anchor_id"].nunique()),
            "rows": int(len(oof)),
            "leads": sorted(LEADS),
            "duplicate_keys": 0,
        },
        "recomputed_rmse": recomputed,
        "cut_reconciliation": cut_checks,
        "bias_probe": {
            "frozen_prediction_max_absolute_difference": frozen_error,
            "recomputed_rmse": bias_recomputed,
        },
        "router_probe": router_check,
        "matched_probes": matched_checks,
        "caveats": [
            "Local validation is not the hidden official score.",
            "Future-trajectory cuts are research-only outcome diagnostics.",
            "Validation-to-test domain AUC diagnoses covariate shift but does not identify target shift.",
        ],
        "sha256": {
            name: _sha256(path)
            for name, path in paths.items()
            if name != "output" and path.exists()
        },
        "raw_rows_written": 0,
        "external_observations_used": 0,
    }
    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
