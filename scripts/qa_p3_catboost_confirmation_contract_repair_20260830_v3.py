#!/usr/bin/env python3
"""Independent historical QA for the terminal P3 confirmation-only repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from p3_wave.catboost_confirmation_repair_v3 import EXPERIMENT_ID, sha256_file  # noqa: E402
from p3_wave.catboost_ordered_hpo import (  # noqa: E402
    evaluate_confirmation_gate,
    metric_deltas,
    paired_case_bootstrap,
)

CONFIG_PATH = ROOT / "configs/experiments/p3_catboost_confirmation_contract_repair_20260830_v3.json"
SOURCE_CONFIG_PATH = ROOT / "configs/experiments/p3_catboost_ordered_hpo_20260829_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/p3_catboost_confirmation_contract_repair_20260830_v3/independent-qa.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_qa(output: Path) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))
    outputs = {
        name: ROOT / relative
        for name, relative in config["outputs"].items()
        if name != "artifact_dir"
    }
    result = json.loads(outputs["result"].read_text(encoding="utf-8"))
    seal = json.loads(outputs["confirmation_seal"].read_text(encoding="utf-8"))
    lock = json.loads(outputs["attempt_lock"].read_text(encoding="utf-8"))
    blind = pd.read_parquet(outputs["confirmation_blind_predictions"])

    expected_columns = [
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "control_prediction",
        "challenger_prediction",
    ]
    _require(result["experiment_id"] == EXPERIMENT_ID, "result experiment id changed")
    _require(result["status"] == "CONFIRMATION_GATE_FAIL_HPO_CLOSED", "terminal status changed")
    _require(list(blind.columns) == expected_columns, "blind prediction schema changed")
    _require(not blind.duplicated(["fold", "anchor_id", "station", "lead_h"]).any(), "duplicate blind keys")
    _require(len(blind) == 1092, "blind row count changed")
    _require(blind[["fold", "anchor_id"]].drop_duplicates().shape[0] == 182, "blind case count changed")
    _require(not any("target" in column.lower() or "truth" in column.lower() for column in blind), "truth leaked into blind predictions")
    prediction_values = blind[["control_prediction", "challenger_prediction"]].to_numpy(np.float64)
    _require(np.isfinite(prediction_values).all(), "blind prediction contains non-finite values")
    _require((prediction_values >= 0.0).all() and (prediction_values <= 30.0).all(), "blind prediction outside bounds")
    observed_prediction_hash = sha256_file(outputs["confirmation_blind_predictions"])
    _require(observed_prediction_hash == seal["prediction_sha256"], "sealed prediction hash mismatch")
    _require(observed_prediction_hash == result["confirmation"]["blind_prediction_sha256"], "result prediction hash mismatch")
    _require(seal["truth_columns_present"] is False, "seal truth marker changed")

    anchors_path = ROOT / source["inputs"]["train_anchors"]["path"]
    _require(sha256_file(anchors_path) == source["inputs"]["train_anchors"]["sha256"], "anchor hash changed")
    anchors = pd.read_parquet(anchors_path)
    anchor_lookup = anchors.set_index("anchor_id")
    truth_blocks: list[pd.DataFrame] = []
    for lead in [3, 6, 9, 12, 18, 24]:
        block = blind.loc[
            blind["lead_h"].eq(lead), ["fold", "anchor_id", "station", "lead_h"]
        ].copy()
        block["target_hs"] = anchor_lookup.loc[block["anchor_id"], f"target_{lead}"].to_numpy()
        truth_blocks.append(block)
    truth = pd.concat(truth_blocks, ignore_index=True)
    evaluated = blind.merge(
        truth, on=["fold", "anchor_id", "station", "lead_h"], validate="one_to_one"
    )
    recomputed_metrics = metric_deltas(evaluated)
    recomputed_bootstrap = paired_case_bootstrap(
        evaluated,
        replicates=config["confirmation"]["bootstrap"]["replicates"],
        seed=config["confirmation"]["bootstrap"]["seed"],
    )
    recomputed_gate = evaluate_confirmation_gate(
        recomputed_metrics, recomputed_bootstrap, config["confirmation"]["gate"]
    )
    _require(recomputed_metrics == result["confirmation"]["metrics"], "metric recomputation differs")
    _require(recomputed_bootstrap == result["confirmation"]["paired_case_bootstrap"], "bootstrap recomputation differs")
    _require(recomputed_gate == result["confirmation"]["gate"], "gate recomputation differs")
    _require(recomputed_gate["pass"] is False, "failed gate unexpectedly passed")
    _require(all(value is False for value in recomputed_gate["checks"].values()), "not every gate check failed")

    boundary_checks = {
        "selection_search_fit_count_zero": result["frozen_selection"]["selection_search_fit_count"] == 0,
        "confirmation_fit_count_three": result["confirmation"]["fit_count"] == 3,
        "full_refit_fit_count_zero": result["full_refit_fit_count"] == 0,
        "official_rows_read_zero": result["official_rows_read"] == 0,
        "csv_files_written_zero": result["csv_files_written"] == 0,
        "submission_or_upload_false": result["submission_or_upload_attempted"] is False,
        "lock_forbids_rerun": lock["rerun_forbidden"] is True,
        "lock_selection_search_false": lock["selection_search_rerun"] is False,
        "lock_config_hash_matches": lock["config_sha256"] == sha256_file(CONFIG_PATH),
    }
    _require(all(boundary_checks.values()), "execution boundary QA failed")

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_TERMINAL_SCIENTIFIC_NO_GO",
        "checks": {
            "blind_schema_and_keys": True,
            "blind_prediction_sealed_before_truth": True,
            "prediction_hash_matches": True,
            "historical_truth_recomputed_from_frozen_anchors": True,
            "metrics_exact_match": True,
            "bootstrap_exact_match": True,
            "gate_exact_match": True,
            "all_promotion_checks_failed": True,
            **boundary_checks,
        },
        "observed": {
            "rows": int(len(blind)),
            "cases": int(blind[["fold", "anchor_id"]].drop_duplicates().shape[0]),
            "control_rmse_m": recomputed_metrics["control_rmse_m"],
            "challenger_rmse_m": recomputed_metrics["challenger_rmse_m"],
            "delta_rmse_m": recomputed_metrics["delta_rmse_m"],
            "bootstrap_ci90_lower_m": recomputed_bootstrap["ci90_lower_m"],
            "bootstrap_ci90_upper_m": recomputed_bootstrap["ci90_upper_m"],
            "result_sha256": sha256_file(outputs["result"]),
            "blind_prediction_sha256": observed_prediction_hash,
            "confirmation_seal_sha256": sha256_file(outputs["confirmation_seal"]),
            "attempt_lock_sha256": sha256_file(outputs["attempt_lock"]),
        },
        "conclusion": "The v3 schema repair succeeded, but frozen challenger_21 is scientifically closed after deterioration on every confirmation fold, station, and lead.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_qa(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
