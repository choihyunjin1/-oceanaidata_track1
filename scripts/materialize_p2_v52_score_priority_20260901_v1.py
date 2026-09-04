"""Materialize the user-approved immutable P2 v52 candidate exactly once."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_v52_score_priority_third_moment_input_gradient_20260901_v1 as v52  # noqa: E402

from p2_restore.data import KEYS, P2Data  # noqa: E402
from p2_restore.features import build_test_features, build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)
from p2_restore.submission import build_submission, validate_submission  # noqa: E402

EXPERIMENT_ID = "p2_v52_score_priority_deployment_20260901_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)


class ContractError(RuntimeError):
    """Raised when the immutable deployment contract drifts."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = config["frozen_candidate"]
    data = config["data_contract"]
    output = config["output"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_USER_APPROVED_NOT_EXECUTED"
        or frozen["required_status"]
        != "SCORE_PRIORITY_PASS_EXPLICIT_STABILITY_RISK"
        or frozen["required_qa_status"] != "PASS"
        or frozen["architecture"] != "v50_masked_third_central_moment_pooling"
        or frozen["input_gradient_coefficient"] != 0.01
        or frozen["input_gradient_token_channel"] != 0
        or frozen["seeds"] != [20260901, 20260902, 20260903]
        or frozen["epochs"] != 60
        or frozen["blend"] != {"anchor_weight": 0.8, "model_weight": 0.2}
        or frozen["model_minus_anchor_clip_C"] != 2.5
        or frozen["maximum_action_C"] != 0.5
        or frozen["full_history_fit_count"] != 3
        or frozen["tuning"] != 0
        or frozen["automatic_retry_count"] != 0
        or data["anchor_name"] != "P2_1_RANK1_BIN17_ONLY"
        or data["hidden_truth_allowed"]
        or data["score_file_allowed"]
        or data["sample_submission_temp_values_allowed"]
        or data["external_data_allowed"]
        or data["pretrained_weights_allowed"]
        or config["training_contract"]["row_deletion"]
        or config["training_contract"]["outer_result_tuning"]
        or config["training_contract"]["official_feedback_tuning"]
        or output["expected_rows"] != 26061
        or output["upload_count"] != 0
    ):
        raise ContractError("v52 deployment contract drift")
    for name in ("config", "runner", "result", "independent_qa"):
        path = ROOT / frozen[name]
        if not path.is_file() or v12.sha256_file(path) != frozen[f"{name}_sha256"]:
            raise ContractError(f"frozen v52 {name} hash drift")
    result = json.loads((ROOT / frozen["result"]).read_text(encoding="utf-8"))
    qa = json.loads(
        (ROOT / frozen["independent_qa"]).read_text(encoding="utf-8")
    )
    if (
        result["status"] != frozen["required_status"]
        or not result["score_priority_gate"]["pass"]
        or result["stability_diagnostic"]["pass"]
        or qa["status"] != frozen["required_qa_status"]
    ):
        raise ContractError("v52 result/QA authorization drift")
    return config


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    raw = os.environ.get(config["data_contract"]["environment_variable"])
    if not raw:
        raise ContractError("P2_DATA_DIR is required")
    data_dir = Path(raw).resolve()
    paths = {
        "observations": data_dir / "observations.csv",
        "test_index": data_dir / "test_index.csv",
        "sample_submission": data_dir / "sample_submission.csv",
        "baseline_interp": data_dir / "baseline_interp.csv",
        "anchor": Path(config["data_contract"]["anchor_path"]),
    }
    for name, path in paths.items():
        expected = config["data_contract"][f"{name}_sha256"]
        if not path.is_file() or v12.sha256_file(path) != expected:
            raise ContractError(f"v52 deployment source hash drift: {name}")
    return paths


def _load_frames(paths: dict[str, Path]) -> tuple[pd.DataFrame, ...]:
    dtype = {"station": "string", "time": "string"}
    observations = pd.read_csv(paths["observations"], dtype=dtype)
    test_index = pd.read_csv(paths["test_index"], dtype=dtype)
    baseline = pd.read_csv(paths["baseline_interp"], dtype=dtype)
    anchor = pd.read_csv(paths["anchor"], dtype=dtype)
    sample_columns = list(pd.read_csv(paths["sample_submission"], nrows=0).columns)
    sample_keys = pd.read_csv(
        paths["sample_submission"], usecols=KEYS, dtype=dtype
    )
    if (
        list(test_index.columns) != KEYS + ["nominal_depth"]
        or list(baseline.columns) != KEYS + ["temp"]
        or list(anchor.columns) != KEYS + ["temp"]
        or sample_columns != KEYS + ["temp"]
        or len(test_index) != 26061
        or not test_index[KEYS].equals(baseline[KEYS])
        or not test_index[KEYS].equals(anchor[KEYS])
        or not test_index[KEYS].equals(sample_keys[KEYS])
        or test_index.duplicated(KEYS).any()
        or sample_keys.duplicated(KEYS).any()
    ):
        raise ContractError("official schema/key/order contract failed")
    if not (
        np.isfinite(pd.to_numeric(baseline["temp"], errors="coerce")).all()
        and np.isfinite(pd.to_numeric(anchor["temp"], errors="coerce")).all()
    ):
        raise ContractError("baseline or clean anchor is non-finite")
    return observations, test_index, baseline, anchor, sample_keys


def preflight() -> dict[str, Any]:
    config = load_config()
    paths = _paths(config)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_USER_APPROVED_EXACT_V52_MATERIALIZATION",
        "source_hashes": {name: v12.sha256_file(path) for name, path in paths.items()},
        "frozen_v52_config_sha256": config["frozen_candidate"]["config_sha256"],
        "frozen_v52_runner_sha256": config["frozen_candidate"]["runner_sha256"],
        "frozen_v52_result_sha256": config["frozen_candidate"]["result_sha256"],
        "frozen_v52_qa_sha256": config["frozen_candidate"][
            "independent_qa_sha256"
        ],
        "seeds": config["frozen_candidate"]["seeds"],
        "epochs": config["frozen_candidate"]["epochs"],
        "planned_fits": 3,
        "tuning": 0,
        "automatic_retries": 0,
        "hidden_truth_rows_read": 0,
        "score_file_rows_read": 0,
        "sample_submission_temp_values_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
    }
    payload["preflight_sha256"] = v12.sha256_json(payload)
    return payload


def execute() -> dict[str, Any]:
    started = time.perf_counter()
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("v52 deployment exactly-once namespace already exists")
    config = load_config()
    paths = _paths(config)
    output_dir = Path(config["output"]["directory"])
    if output_dir.exists():
        raise FileExistsError("v52 deployment output directory already exists")
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    v12.atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": v12.sha256_file(CONFIG),
            "runner_sha256": v12.sha256_file(RUNNER),
            "fit_plan": "full_history_3_frozen_seeds",
            "automatic_retry_count": 0,
            "hidden_truth_access": 0,
            "score_access": 0,
            "external_access": 0,
            "pretrained_weights_loaded": 0,
        },
    )
    observations, test_index, baseline, anchor, sample_keys = _load_frames(paths)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    train_table = build_training_features(observations)
    train_design = build_normalized_curvature_design(train_table.frame)
    train_tokens, train_mask, train_context = v12.build_arrays(train_table.frame)
    local = train_design.keys["time"].dt.tz_convert("Asia/Seoul")
    start = pd.Timestamp(config["training_contract"]["training_start_inclusive_kst"])
    selected = local >= start
    weights, weight_receipt = v52.v13.domain_balanced_weights(
        train_design.keys.loc[selected, "layer"].to_numpy(int), local[selected]
    )

    dummy_sample = test_index.loc[:, KEYS].copy()
    dummy_sample["temp"] = 0.0
    data = P2Data(
        observations=observations,
        test_index=test_index,
        sample_submission=dummy_sample,
        baseline=baseline,
    )
    query_table = build_test_features(data)
    query_frame = query_table.frame.copy()
    query_frame["target"] = pd.to_numeric(query_frame["baseline"], errors="raise")
    query_design = build_normalized_curvature_design(query_frame)
    query_tokens, query_mask, query_context = v12.build_arrays(query_frame)

    source_config = v52.load_config()
    v12.VerticalDeepSet = v52.v50.MaskedThirdCentralMomentProfileVerticalDeepSet
    predictions: list[np.ndarray] = []
    fit_receipts: list[dict[str, Any]] = []
    for seed in config["frozen_candidate"]["seeds"]:
        prediction, receipt = v52.train_predict_seed(
            train_tokens[selected],
            train_mask[selected],
            train_context[selected],
            train_design.normalized_target[selected],
            weights,
            query_tokens,
            query_mask,
            query_context,
            source_config,
            int(seed),
        )
        predictions.append(prediction)
        fit_receipts.append(receipt)
    if len(fit_receipts) != 3:
        raise ContractError("v52 deployment fit count drift")
    mean_normalized = np.mean(np.vstack(predictions), axis=0)
    absolute_model = query_design.baseline + mean_normalized * query_design.profile_scale
    anchor_values = pd.to_numeric(anchor["temp"], errors="raise").to_numpy(float)
    clip = float(config["frozen_candidate"]["model_minus_anchor_clip_C"])
    clipped = np.clip(absolute_model - anchor_values, -clip, clip)
    candidate_values = anchor_values + float(
        config["frozen_candidate"]["blend"]["model_weight"]
    ) * clipped
    maximum_action = float(np.max(np.abs(candidate_values - anchor_values)))
    if maximum_action > float(config["frozen_candidate"]["maximum_action_C"]) + 1e-12:
        raise ContractError("v52 deployment action cap drift")
    candidate = build_submission(test_index, candidate_values)
    output_dir.mkdir(parents=True)
    output = output_dir / config["output"]["filename"]
    candidate.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    validation = validate_submission(output, test_index)
    roundtrip = pd.read_csv(output, dtype={"station": "string", "time": "string"})
    roundtrip_values = pd.to_numeric(roundtrip["temp"], errors="coerce").to_numpy(float)
    roundtrip_error = float(np.max(np.abs(roundtrip_values - candidate_values)))
    if roundtrip_error > 1e-12:
        raise ContractError("v52 deployment numeric CSV round-trip drift")
    action = candidate_values - anchor_values
    absolute_action = np.abs(action)
    result = {
        "schema_version": "p2.v52_score_priority_deployment.result.20260901.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "MATERIALIZED_READY_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 3,
        "training_rows": int(selected.sum()),
        "training_time_min_kst": local[selected].min().isoformat(),
        "training_time_max_kst": local[selected].max().isoformat(),
        "weight_receipt": weight_receipt,
        "fit_receipts": fit_receipts,
        "candidate": {
            "name": "P2_V52_SCORE_PRIORITY_FULL_HISTORY_BLEND020",
            "path": str(output.resolve()),
            "rows": len(candidate),
            "bytes": output.stat().st_size,
            "sha256": v12.sha256_file(output),
            "validation": validation,
            "schema_exact": list(candidate.columns) == KEYS + ["temp"],
            "sample_schema_exact": list(roundtrip.columns) == KEYS + ["temp"],
            "key_order_exact": bool(candidate[KEYS].equals(test_index[KEYS])),
            "sample_key_order_exact": bool(candidate[KEYS].equals(sample_keys[KEYS])),
            "duplicate_keys": int(candidate.duplicated(KEYS).sum()),
            "finite": bool(np.isfinite(candidate_values).all()),
            "submission_domain": bool(
                ((candidate_values >= -5.0) & (candidate_values <= 45.0)).all()
            ),
            "numeric_roundtrip_max_abs_error_C": roundtrip_error,
            "changed_rows_vs_anchor": int(np.count_nonzero(absolute_action > 1e-12)),
            "active_share_vs_anchor": float(np.mean(absolute_action > 1e-12)),
            "abs_action_p50_C": float(np.quantile(absolute_action, 0.50)),
            "abs_action_p90_C": float(np.quantile(absolute_action, 0.90)),
            "abs_action_p99_C": float(np.quantile(absolute_action, 0.99)),
            "abs_action_max_C": maximum_action,
            "action_rms_C": float(np.sqrt(np.mean(np.square(action)))),
            "minimum_C": float(candidate_values.min()),
            "maximum_C": float(candidate_values.max()),
        },
        "anchor": {
            "name": config["data_contract"]["anchor_name"],
            "path": str(paths["anchor"]),
            "sha256": config["data_contract"]["anchor_sha256"],
            "modified": False,
        },
        "internal_evidence": {
            "pooled_delta_rmse_C": -0.052651613292153066,
            "canonical_nominal_expected_points_delta": 0.660647755248464,
            "canonical_transport_adjusted_expected_points_delta": 0.5389656636384578,
            "v23_internal_margin_rmse_C": -0.0007591467204575159,
            "fold_layer_non_harm_cells": 7,
            "fold_layer_total_cells": 9,
            "caveat": "Internal score-priority evidence is not an official guarantee; unchanged stability diagnostic failed 7/9.",
        },
        "submission_metadata": {
            "title": config["output"]["title"],
            "summary": config["output"]["summary"],
        },
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "official_test_index_rows_read": int(len(test_index)),
            "official_baseline_rows_read": int(len(baseline)),
            "official_sample_key_rows_read": int(len(sample_keys)),
            "sample_submission_temp_values_read": 0,
            "anchor_rows_read": int(len(anchor)),
            "score_file_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "external_rows_read": 0,
            "pretrained_weight_files_loaded": 0,
            "submission_csv_created": 1,
            "uploads": 0,
            "automatic_retries": 0,
            "tuning": 0,
        },
        "hashes": {
            "config": v12.sha256_file(CONFIG),
            "runner": v12.sha256_file(RUNNER),
            "observations": v12.sha256_file(paths["observations"]),
            "test_index": v12.sha256_file(paths["test_index"]),
            "sample_submission": v12.sha256_file(paths["sample_submission"]),
            "baseline_interp": v12.sha256_file(paths["baseline_interp"]),
            "anchor": v12.sha256_file(paths["anchor"]),
            "source_v52_config": config["frozen_candidate"]["config_sha256"],
            "source_v52_runner": config["frozen_candidate"]["runner_sha256"],
            "source_v52_result": config["frozen_candidate"]["result_sha256"],
            "source_v52_independent_qa": config["frozen_candidate"][
                "independent_qa_sha256"
            ],
        },
    }
    v12.atomic_json(ARTIFACT / "result.json", result)
    v12.atomic_json(REPORT / "result.json", result)
    v12.atomic_json(output_dir / "manifest.json", result)
    (output_dir / "upload-note.md").write_text(
        f"# {config['output']['title']}\n\n"
        f"{config['output']['summary']}\n\n"
        f"CSV: `{output.resolve()}`\n\n"
        f"SHA-256: `{result['candidate']['sha256']}`\n\n"
        f"Rows: `{len(candidate)}`; status: `READY_NOT_UPLOADED`.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    value = preflight() if args.preflight else execute()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
