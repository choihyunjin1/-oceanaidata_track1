"""Independent, train-only QA for the sealed P3 Chronos-2 v2 result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.chronos2_transfer import rmse, sha256_file, write_json

ROOT = Path("artifacts/p3_chronos2_full_nested_20260828_v2")
CONFIG = Path("configs/experiments/p3_chronos2_full_nested_20260828_v2.json")


def _key_digest(frame: pd.DataFrame) -> str:
    ordered = frame[["anchor_id", "lead_h"]].sort_values(["anchor_id", "lead_h"])
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result_path = ROOT / "full_nested_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seal_path = ROOT / "PREDICTION_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    blind_path = ROOT / "sealed_predictions" / "all_outer_predictions.parquet"
    blind = pd.read_parquet(blind_path)
    if "target_hs" in blind.columns:
        raise RuntimeError("blind prediction file contains outer truth")
    if sha256_file(blind_path) != seal["combined_sha256"]:
        raise RuntimeError("combined blind prediction hash mismatch")
    if not all(sha256_file(item["path"]) == item["sha256"] for item in seal["fold_files"]):
        raise RuntimeError("fold blind prediction hash mismatch")
    anchor_path = Path(config["bindings"]["train_anchors"]["path"])
    target_columns = ["anchor_id", *[f"target_{lead}" for lead in (3, 6, 9, 12, 18, 24)]]
    targets = pd.read_parquet(anchor_path, columns=target_columns).set_index("anchor_id")
    full = blind.copy()
    for lead in (3, 6, 9, 12, 18, 24):
        mask = full["lead_h"].eq(lead)
        ids = full.loc[mask, "anchor_id"].to_numpy(dtype=np.int64)
        full.loc[mask, "target_hs"] = targets.loc[ids, f"target_{lead}"].to_numpy(dtype=float)
    full["persistence_prediction"] = full["current_hs"]
    incumbent = pd.read_parquet(config["bindings"]["frozen_incumbent_oof"]["path"])
    incumbent = incumbent.loc[
        incumbent["prefix_fraction"].eq(1.0),
        ["anchor_id", "lead_h", "incumbent_prediction"],
    ]
    candidate_keys = full[["anchor_id", "lead_h"]]
    incumbent_keys = incumbent[["anchor_id", "lead_h"]]
    exact = full.merge(incumbent, on=["anchor_id", "lead_h"], validate="one_to_one")
    full_candidate_rmse = rmse(full["target_hs"], full["prediction"])
    full_persistence_rmse = rmse(full["target_hs"], full["persistence_prediction"])
    exact_candidate_rmse = rmse(exact["target_hs"], exact["prediction"])
    exact_persistence_rmse = rmse(exact["target_hs"], exact["persistence_prediction"])
    exact_incumbent_rmse = rmse(exact["target_hs"], exact["incumbent_prediction"])
    declared = result["metrics"]
    checks = {
        "seal_hash": sha256_file(seal_path) == result["prediction_seal_sha256"],
        "combined_hash": sha256_file(blind_path) == result["combined_blind_prediction_sha256"],
        "seal_precedes_truth_record": seal["sealed_at_utc"]
        < result["truth_first_read_after_seal_at_utc"],
        "blind_has_no_truth_column": "target_hs" not in blind.columns,
        "exact_candidate_metric": abs(
            exact_candidate_rmse - declared["candidate"]["pooled_rmse_m"]
        )
        < 1e-12,
        "exact_persistence_metric": abs(
            exact_persistence_rmse - declared["persistence"]["pooled_rmse_m"]
        )
        < 1e-12,
        "exact_incumbent_metric": abs(
            exact_incumbent_rmse - declared["incumbent"]["pooled_rmse_m"]
        )
        < 1e-12,
        "all_predictions_finite": bool(np.isfinite(blind["prediction"]).all()),
        "terminal_no_go_consistent": result["status"] == "TERMINAL_NO_GO"
        and exact_candidate_rmse > exact_incumbent_rmse,
    }
    if not all(checks.values()):
        raise RuntimeError(f"independent QA failed: {checks}")
    missing_from_incumbent = candidate_keys.merge(
        incumbent_keys, on=["anchor_id", "lead_h"], how="left", indicator=True
    )
    missing_from_candidate = incumbent_keys.merge(
        candidate_keys, on=["anchor_id", "lead_h"], how="left", indicator=True
    )
    missing_from_incumbent = missing_from_incumbent.loc[
        missing_from_incumbent["_merge"].eq("left_only"), ["anchor_id", "lead_h"]
    ]
    missing_from_candidate = missing_from_candidate.loc[
        missing_from_candidate["_merge"].eq("left_only"), ["anchor_id", "lead_h"]
    ]
    qa = {
        "status": "PASS",
        "checks": checks,
        "sealed_full_universe": {
            "rows": int(len(full)),
            "cases": int(full["anchor_id"].nunique()),
            "candidate_rmse_m": full_candidate_rmse,
            "persistence_rmse_m": full_persistence_rmse,
            "delta_candidate_minus_persistence_m": full_candidate_rmse
            - full_persistence_rmse,
            "key_sha256": _key_digest(candidate_keys),
        },
        "exact_three_way_comparator_universe": {
            "rows": int(len(exact)),
            "cases": int(exact["anchor_id"].nunique()),
            "candidate_rmse_m": exact_candidate_rmse,
            "persistence_rmse_m": exact_persistence_rmse,
            "incumbent_rmse_m": exact_incumbent_rmse,
            "delta_candidate_minus_incumbent_m": exact_candidate_rmse - exact_incumbent_rmse,
            "key_sha256": _key_digest(exact),
        },
        "key_mismatch_audit": {
            "sealed_rows_absent_from_incumbent": int(len(missing_from_incumbent)),
            "sealed_cases_absent_from_incumbent": int(
                missing_from_incumbent["anchor_id"].nunique()
            ),
            "incumbent_rows_absent_from_sealed": int(len(missing_from_candidate)),
            "incumbent_cases_absent_from_sealed": int(
                missing_from_candidate["anchor_id"].nunique()
            ),
            "sealed_only_key_sha256": _key_digest(missing_from_incumbent),
            "incumbent_only_key_sha256": _key_digest(missing_from_candidate),
        },
        "result_sha256": sha256_file(result_path),
        "seal_sha256": sha256_file(seal_path),
        "combined_blind_sha256": sha256_file(blind_path),
    }
    destination = ROOT / "independent_qa.json"
    write_json(destination, qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
