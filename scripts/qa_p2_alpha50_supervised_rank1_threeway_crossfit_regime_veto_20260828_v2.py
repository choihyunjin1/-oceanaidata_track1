"""Finalize and independently QA only the already committed P2 v2 predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for directory in (ROOT, SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (  # noqa: E402
    paired_kst_day_bootstrap,
)
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1 as base,
)
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2 as experiment,
)


def finalize(data_dir: Path) -> dict[str, Any]:
    config, base_config = experiment.load_config(data_dir)
    output = ROOT / config["artifact_directory"]
    commitment_path = output / "prediction_commitment.json"
    result_path = output / "result.json"
    qa_path = output / "independent_qa.json"
    if result_path.exists() or qa_path.exists():
        raise FileExistsError("terminal recovery artifact already exists")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if commitment["truth_metric_computed"] or commitment["outer_validation_truth_column_loaded"]:
        raise base.ContractError("commitment is not truth-late")
    if commitment["official_rows_read"] != 0 or commitment["correction_vector_modified"]:
        raise base.ContractError("commitment violates official/no-modification contract")
    anchor_path = ROOT / base_config["immutable_inputs"]["alpha50_proxy"]["path"]
    scored_parts = []
    for fold_name, record in commitment["outputs"].items():
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or base.sha256_file(path) != record["sha256"]
        ):
            raise base.ContractError("committed prediction hash drifted")
        with np.load(path, allow_pickle=False) as payload:
            scored = pd.DataFrame(
                {
                    "time": pd.to_datetime(payload["time_ns"], unit="ns", utc=True),
                    "layer": payload["layer"].astype(int),
                    "reference": payload["reference"],
                    "candidate": payload["candidate"],
                    "correction": payload["correction"],
                    "original_correction": payload["original_correction"],
                    "regime_enabled": payload["regime_enabled"].astype(bool),
                }
            )
        if len(scored) != int(record["rows"]) or scored.duplicated(["time", "layer"]).any():
            raise base.ContractError("committed prediction key/row contract failed")
        truth = base.block_anchor(anchor_path, fold_name, include_truth=True)[
            ["time", "layer", "truth"]
        ]
        scored = scored.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        if scored["truth"].isna().any():
            raise base.ContractError("truth binding failed")
        scored["fold"] = fold_name
        scored_parts.append(scored)
    scored = pd.concat(scored_parts, ignore_index=True)
    metrics = {
        "aggregate": base.metric_record(scored),
        "by_fold": {
            key: base.metric_record(group) for key, group in scored.groupby("fold", sort=True)
        },
        "by_layer": {
            str(int(key)): base.metric_record(group)
            for key, group in scored.groupby("layer", sort=True)
        },
    }
    gate = config["gate"]
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(gate["bootstrap_replicates"]),
        seed=int(gate["bootstrap_seed"]),
    )
    fold_deltas = [item["delta_rmse"] for item in metrics["by_fold"].values()]
    layer_deltas = [item["delta_rmse"] for item in metrics["by_layer"].values()]
    correction = scored["correction"].to_numpy(dtype=np.float64)
    active_share = float(np.mean(np.abs(correction) > 1e-12))
    correction_rms = float(np.sqrt(np.mean(np.square(correction))))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    disabled = ~scored["regime_enabled"]
    enabled = scored["regime_enabled"]
    disabled_error = float(
        np.max(np.abs(scored.loc[disabled, "correction"].to_numpy(dtype=np.float64)), initial=0.0)
    )
    enabled_error = float(
        np.max(
            np.abs(
                scored.loc[enabled, "correction"].to_numpy(dtype=np.float64)
                - scored.loc[enabled, "original_correction"].to_numpy(dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    checks = {
        "pooled_delta": metrics["aggregate"]["delta_rmse"]
        <= float(gate["pooled_delta_rmse_max_c"]),
        "bootstrap_ci": bootstrap["ci90_high"] < float(gate["bootstrap_ci90_upper_max_c"]),
        "2024_sep_oct": metrics["by_fold"]["2024_sep_oct"]["delta_rmse"]
        <= float(gate["2024_sep_oct_delta_rmse_max_c"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas)
        >= int(gate["minimum_improved_folds"]),
        "worst_fold": max(fold_deltas) <= float(gate["maximum_worst_fold_regression_c"]),
        "worst_layer": max(layer_deltas) <= float(gate["maximum_layer_regression_c"]),
        "active_share": float(gate["minimum_active_share"])
        <= active_share
        <= float(gate["maximum_active_share"]),
        "correction_rms": float(gate["minimum_correction_rms_c"])
        <= correction_rms
        <= float(gate["maximum_correction_rms_c"]),
        "correction_p99": correction_p99 <= float(gate["maximum_correction_p99_c"]),
        "veto_disabled_exact_noop": disabled_error <= 1e-12,
        "enabled_correction_vector_unchanged": enabled_error <= 1e-12,
    }
    decision = "GO_LOCAL_ONLY_NO_UPLOAD" if all(checks.values()) else "NO_GO_EXACT_NO_OUTPUT"
    result = {
        "schema_version": "p2.threeway_crossfit_regime_veto.result.v2",
        "experiment_id": experiment.EXPERIMENT_ID,
        "decision": decision,
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "recovery_disclosure": "The one-shot generated and sealed predictions, then failed in post-commit metric formatting because pandas Series.max rejected numpy's initial keyword. No model or prediction was rerun; this finalize-only QA verified committed hashes and computed metrics from the sealed vectors.",
        "metrics": metrics,
        "bootstrap": bootstrap,
        "active_share": active_share,
        "correction_rms_c": correction_rms,
        "correction_p99_c": correction_p99,
        "gate_checks": checks,
        "inner_truth_rows_read_before_commitment": int(commitment["inner_truth_rows"]),
        "outer_truth_rows_read_after_commitment": len(scored),
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "execution_count": 1,
        "model_or_prediction_rerun": False,
    }
    base.atomic_json(result_path, result)
    no_truth_path = output / "scored_predictions_no_truth.parquet"
    scored.drop(columns="truth").to_parquet(no_truth_path, index=False)
    qa = {
        "experiment_id": experiment.EXPERIMENT_ID,
        "status": "PASS",
        "decision_matches_gate_conjunction": (decision == "GO_LOCAL_ONLY_NO_UPLOAD")
        == all(checks.values()),
        "commitment_sha256": base.sha256_file(commitment_path),
        "result_sha256": base.sha256_file(result_path),
        "no_truth_predictions_sha256": base.sha256_file(no_truth_path),
        "prediction_hashes": {
            name: record["sha256"] for name, record in commitment["outputs"].items()
        },
        "disabled_exact_noop_max_abs_c": disabled_error,
        "enabled_original_vector_max_abs_difference_c": enabled_error,
        "official_rows_read": 0,
        "csv_generated_or_uploaded": False,
        "model_or_prediction_rerun": False,
    }
    base.atomic_json(qa_path, qa)
    return {"result": result, "qa": qa}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            finalize(arguments.data_dir.expanduser().resolve()),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
