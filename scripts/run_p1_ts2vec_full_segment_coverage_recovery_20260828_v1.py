"""Run the one-shot P1 frozen-encoder full-segment coverage recovery audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc.full_segment_coverage_recovery import (  # noqa: E402
    infer_complete_segments,
    vectorized_conditional_prototype_scores,
)
from p1_qc.ts2vec_conditional_normal_prototype import (  # noqa: E402
    HierarchicalContrastiveEncoder,
    binary_metrics,
    day_block_bootstrap_probability,
    decode_components,
    finite_normal_tail_threshold,
    fit_conditional_prototype,
    robust_fit_transform,
    sha256_file,
)

CONFIG = ROOT / "configs/experiments/p1_ts2vec_full_segment_coverage_recovery_20260828_v1.json"
MODULE = ROOT / "src/p1_qc/full_segment_coverage_recovery.py"
RUNNER = Path(__file__).resolve()
EXPERIMENT_ID = "p1_ts2vec_full_segment_coverage_recovery_20260828_v1"
FORBIDDEN = ("test.csv", "test_context", "sample_submission", "submission.csv")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _segment_ids(keys: pd.DataFrame) -> np.ndarray:
    time_values = pd.to_datetime(keys["time"], utc=True)
    boundary = (
        keys["station"].astype(str).ne(keys["station"].astype(str).shift())
        | keys["year"].ne(keys["year"].shift())
        | keys["layer"].ne(keys["layer"].shift())
        | time_values.diff().ne(pd.Timedelta(minutes=10))
    )
    boundary.iloc[0] = True
    return boundary.cumsum().to_numpy(dtype=np.int64)


def _event_count(truth: np.ndarray, prediction: np.ndarray, segments: np.ndarray) -> int:
    found = 0
    start = 0
    while start < len(truth):
        if truth[start] != 1:
            start += 1
            continue
        stop = start + 1
        while stop < len(truth) and truth[stop] == 1 and segments[stop] == segments[start]:
            stop += 1
        if stop - start >= 19 and prediction[start:stop].any():
            found += 1
        start = stop
    return found


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("experiment ID drifted")
    policy = config["execution_policy"]
    required_false = (
        "parent_experiment_rerun",
        "new_encoder_training",
        "q2_truth_read_authorized",
        "q3_q4_truth_read_authorized",
        "official_input_read_authorized",
        "submission_generation_authorized",
        "official_upload_authorized",
        "result_based_threshold_or_architecture_rerun",
    )
    if any(policy.get(name) is not False for name in required_false):
        raise RuntimeError("execution policy opened a prohibited boundary")
    checked: dict[str, Any] = {}
    specs = dict(config["immutable_inputs"])
    parent = config["parent_experiment"]
    specs.update(
        {
            "parent_encoder": {
                "path": parent["encoder_path"],
                "bytes": parent["encoder_bytes"],
                "sha256": parent["encoder_sha256"],
            },
            "parent_result": {
                "path": parent["result_path"],
                "sha256": parent["result_sha256"],
            },
            "parent_final_qa": {
                "path": parent["final_qa_path"],
                "sha256": parent["final_qa_sha256"],
            },
        }
    )
    for name, spec in specs.items():
        path = ROOT / spec["path"]
        lowered = str(path).lower()
        if any(token in lowered for token in FORBIDDEN):
            raise RuntimeError(f"forbidden path in contract: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if "bytes" in spec and path.stat().st_size != int(spec["bytes"]):
            raise RuntimeError(f"immutable size drifted: {name}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(f"immutable hash drifted: {name}")
        checked[name] = {"bytes": path.stat().st_size, "sha256": digest}
    return {
        "status": "PASS",
        "checked": checked,
        "q2_truth_rows_read": 0,
        "official_rows_read": 0,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    receipt = validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    if output.exists():
        raise FileExistsError("one-shot recovery artifact already exists")
    output.mkdir(parents=True, exist_ok=False)

    encoder_features = config["encoder_features"]
    calendar_features = config["calendar_features"]
    feature_path = ROOT / config["immutable_inputs"]["feature_cache"]["path"]
    key_path = ROOT / config["immutable_inputs"]["feature_key_sidecar"]["path"]
    label_path = ROOT / config["immutable_inputs"]["training_labels"]["path"]
    frame = pd.read_parquet(feature_path, columns=encoder_features + calendar_features)
    keys = pd.read_parquet(key_path)
    if len(frame) != len(keys):
        raise RuntimeError("feature and key rows no longer align")
    time_utc = pd.to_datetime(keys["time"], utc=True)
    stop_kst = config["historical_scope"]["exclusive_stop_kst"]
    stop_utc = pd.Timestamp(stop_kst).tz_convert("UTC")
    historical = time_utc.lt(stop_utc).to_numpy()
    historical_keys = keys.loc[historical].reset_index(drop=True)
    historical_frame = frame.loc[historical].reset_index(drop=True)
    labels = pd.read_parquet(
        label_path,
        columns=["station", "year", "layer", "time", "label", "anomaly_type"],
        filters=[("time", "<", stop_kst)],
    ).reset_index(drop=True)
    ordered = ["station", "year", "layer", "time"]
    if len(labels) != len(historical_keys) or not historical_keys[ordered].astype(str).equals(
        labels[ordered].astype(str)
    ):
        raise RuntimeError("historical predicate rows do not align")

    historical_time = pd.to_datetime(historical_keys["time"], utc=True)
    fit_stop = pd.Timestamp(config["representation"]["encoder_fit_stop_kst"]).tz_convert("UTC")
    fit_mask = historical_time.lt(fit_stop).to_numpy()
    scaled, median, mad = robust_fit_transform(
        historical_frame.loc[fit_mask, encoder_features].to_numpy(),
        historical_frame[encoder_features].to_numpy(),
    )
    segments = _segment_ids(historical_keys)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation = config["representation"]
    model = HierarchicalContrastiveEncoder(
        len(encoder_features),
        int(representation["hidden_width"]),
        int(representation["embedding_width"]),
        tuple(int(value) for value in representation["dilations"]),
        float(representation["dropout"]),
    ).to(device)
    checkpoint_path = ROOT / config["parent_experiment"]["encoder_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    embeddings, covered, coverage_audit = infer_complete_segments(
        model,
        scaled,
        segments,
        np.ones(len(scaled), dtype=bool),
        device,
    )
    if coverage_audit["coverage"] < float(config["historical_gate"]["coverage_min"]):
        result = {
            "experiment_id": EXPERIMENT_ID,
            "status": "NO_GO_COVERAGE_RECOVERY_FAILED",
            "coverage_audit": coverage_audit,
            "parent_checkpoint_epoch": int(checkpoint["epoch"]),
            "new_model_fit_count": 0,
            "historical_label_rows_read": int(len(labels)),
            "q2_truth_rows_read": 0,
            "official_rows_read": 0,
        }
        _atomic_json(output / "result.json", result)
        return result

    truth = labels["label"].to_numpy(dtype=np.int8)
    cells = historical_keys["station"].astype(str).str.cat(
        historical_keys["layer"].astype(str), sep="/"
    ).to_numpy()
    normal_stop = pd.Timestamp(config["historical_scope"]["normal_reference_stop_kst"]).tz_convert("UTC")
    reference = historical_time.lt(normal_stop).to_numpy() & covered & (truth == 0)
    scoring = config["scoring"]
    prototype = fit_conditional_prototype(
        embeddings,
        historical_frame["day_sin"].to_numpy(),
        historical_frame["day_cos"].to_numpy(),
        cells,
        reference,
        shrinkage_rows=int(scoring["cell_shrinkage_rows"]),
        scale_floor=float(scoring["scale_floor"]),
        bank_per_cell=int(scoring["normal_knn_bank_per_cell"]),
    )
    scores = vectorized_conditional_prototype_scores(
        prototype,
        embeddings,
        historical_frame["day_sin"].to_numpy(),
        historical_frame["day_cos"].to_numpy(),
        cells,
        knn_k=int(scoring["normal_knn_k"]),
        prototype_weight=float(scoring["prototype_weight"]),
        device=device,
        batch_rows=int(scoring["gpu_distance_batch_rows"]),
    )
    cal_start, cal_stop = [
        pd.Timestamp(value).tz_convert("UTC")
        for value in config["historical_scope"]["calibration"]
    ]
    qual_start, qual_stop = [
        pd.Timestamp(value).tz_convert("UTC")
        for value in config["historical_scope"]["qualification"]
    ]
    calibration = historical_time.ge(cal_start).to_numpy() & historical_time.le(cal_stop).to_numpy()
    qualification = historical_time.ge(qual_start).to_numpy() & historical_time.le(qual_stop).to_numpy()
    threshold = finite_normal_tail_threshold(
        scores[calibration & (truth == 0)], float(scoring["normal_tail_alpha"])
    )
    additions = decode_components(
        scores,
        segments,
        threshold,
        minimum_rows=int(scoring["minimum_component_rows"]),
        bridge_rows=int(scoring["bridge_rows"]),
    )
    zero = np.zeros(len(truth), dtype=np.int8)
    calibration_metrics = binary_metrics(truth[calibration], additions[calibration])
    qualification_metrics = binary_metrics(truth[qualification], additions[qualification])
    qualification_days = max(1, pd.Series(historical_time[qualification]).dt.floor("D").nunique())
    qualification_fp_per_day = qualification_metrics["fp"] / qualification_days
    recovered_events = _event_count(truth[qualification], additions[qualification], segments[qualification])
    direction_cells = sum(
        binary_metrics(truth[qualification & (cells == cell)], additions[qualification & (cells == cell)])["f1"] > 0
        for cell in sorted(set(cells[qualification]))
    )
    day_ids = (
        pd.Series(historical_time[qualification]).dt.floor("D").astype("int64").to_numpy()
        // int(pd.Timedelta(days=1).value)
    )
    bootstrap_probability = day_block_bootstrap_probability(
        truth[qualification],
        additions[qualification],
        zero[qualification],
        day_ids,
        block_days=int(config["historical_gate"]["block_bootstrap_days"]),
        replicates=int(config["historical_gate"]["block_bootstrap_replicates"]),
        seed=int(config["historical_gate"]["block_bootstrap_seed"]),
    )
    gate = {
        "coverage_complete": coverage_audit["coverage"] >= float(config["historical_gate"]["coverage_min"]),
        "calibration_f1_gt_zero": calibration_metrics["f1"] > 0,
        "qualification_f1_gt_zero": qualification_metrics["f1"] > 0,
        "qualification_precision_gt_zero": qualification_metrics["precision"] > 0,
        "qualification_new_eligible_long_events_min": recovered_events >= int(config["historical_gate"]["qualification_new_eligible_long_events_min"]),
        "direction_consistent_cells_min": direction_cells >= int(config["historical_gate"]["direction_consistent_cells_min"]),
        "false_positive_per_day_cap": qualification_fp_per_day <= float(config["historical_gate"]["false_positive_per_day_cap"]),
        "block_bootstrap_probability_min": bootstrap_probability >= float(config["historical_gate"]["probability_delta_f1_gt_zero_min"]),
    }
    gate_pass = all(gate.values())
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_FOR_SEPARATE_Q2_OUTER_RESEARCH_ONLY" if gate_pass else "NO_GO_HISTORICAL_GATE",
        "interpretation_scope": "post_failure_technical_recovery_not_fresh_family_evidence",
        "device": str(device),
        "parent_checkpoint_epoch": int(checkpoint["epoch"]),
        "parent_model_fit_count": int(config["parent_experiment"]["parent_model_fit_count"]),
        "new_model_fit_count": 0,
        "coverage_audit": coverage_audit,
        "threshold": float(threshold),
        "calibration": calibration_metrics,
        "qualification": qualification_metrics,
        "qualification_fp_per_day": float(qualification_fp_per_day),
        "qualification_recovered_long_events": int(recovered_events),
        "direction_consistent_cells": int(direction_cells),
        "block_bootstrap_probability_delta_f1_gt_zero": float(bootstrap_probability),
        "gate": gate,
        "anchor_deletions": 0,
        "historical_no_op_sha256": hashlib.sha256(zero.tobytes()).hexdigest(),
        "historical_label_rows_read": int(len(labels)),
        "q2_truth_rows_read": 0,
        "q3_q4_truth_rows_read": 0,
        "official_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "scaler_aggregate": {
            "feature_count": int(len(median)),
            "finite_median_count": int(np.isfinite(median).sum()),
            "finite_scale_count": int(np.isfinite(mad).sum()),
        },
    }
    _atomic_json(output / "result.json", result)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "module_sha256": sha256_file(MODULE),
        "runner_sha256": sha256_file(RUNNER),
        "parent_encoder_sha256": sha256_file(checkpoint_path),
        "result_sha256": sha256_file(output / "result.json"),
        "preflight": receipt,
        "raw_values_persisted": False,
        "embeddings_persisted": False,
        "keys_persisted": False,
        "submission_generated_or_uploaded": False,
    }
    _atomic_json(output / "manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.check == arguments.execute:
        raise SystemExit("choose exactly one of --check or --execute")
    config = _json(CONFIG)
    result = validate_contract(config) if arguments.check else run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
