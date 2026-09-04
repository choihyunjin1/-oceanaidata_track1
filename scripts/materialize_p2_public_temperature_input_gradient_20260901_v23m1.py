"""Materialize the user-approved immutable P2 v23 candidate once."""

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
import run_p2_public_temperature_input_gradient_regularized_deepset_20260901_v23 as v23  # noqa: E402

from p2_restore.data import KEYS, P2Data  # noqa: E402
from p2_restore.features import build_test_features, build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)
from p2_restore.submission import build_submission, validate_submission  # noqa: E402

EXPERIMENT_ID = "p2_public_temperature_input_gradient_deployment_20260901_v23m1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)


class ContractError(RuntimeError):
    """Raised when the immutable deployment contract drifts."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_APPROVED_NOT_EXECUTED"
        or config["frozen_candidate"]["fit_count"] != 3
        or config["frozen_candidate"]["seeds"] != [20260901, 20260902, 20260903]
        or config["frozen_candidate"]["epochs"] != 60
        or config["frozen_candidate"]["input_gradient_coefficient"] != 0.01
        or config["frozen_candidate"]["blend"]
        != {"anchor_weight": 0.8, "model_weight": 0.2}
        or config["frozen_candidate"]["model_minus_anchor_clip_C"] != 2.5
        or config["frozen_candidate"]["maximum_action_C"] != 0.5
        or config["frozen_candidate"]["tuning"] != 0
        or config["frozen_candidate"]["automatic_retry_count"] != 0
        or config["data_contract"]["hidden_truth_allowed"]
        or config["data_contract"]["score_file_allowed"]
        or config["data_contract"]["sample_submission_values_allowed"]
        or config["training_contract"]["row_deletion"]
        or config["training_contract"]["outer_result_tuning"]
        or config["output"]["expected_rows"] != 26061
        or config["output"]["upload_count"] != 0
    ):
        raise ContractError("v23m1 materialization contract drift")
    for name in ("config", "runner", "result", "independent_qa"):
        path = ROOT / config["frozen_candidate"][name]
        if not path.is_file() or v12.sha256_file(path) != config["frozen_candidate"][
            f"{name}_sha256"
        ]:
            raise ContractError(f"frozen v23 {name} hash drift")
    result = json.loads(
        (ROOT / config["frozen_candidate"]["result"]).read_text(encoding="utf-8")
    )
    qa = json.loads(
        (ROOT / config["frozen_candidate"]["independent_qa"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        result["status"] != config["frozen_candidate"]["required_status"]
        or qa["status"] != config["frozen_candidate"]["required_qa_status"]
    ):
        raise ContractError("v23 result/QA authorization drift")
    return config


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    raw = os.environ.get(config["data_contract"]["environment_variable"])
    if not raw:
        raise ContractError("P2_DATA_DIR is required")
    data = Path(raw).resolve()
    paths = {
        "observations": data / "observations.csv",
        "test_index": data / "test_index.csv",
        "baseline_interp": data / "baseline_interp.csv",
        "anchor": Path(config["data_contract"]["anchor_path"]),
    }
    for name, path in paths.items():
        expected = config["data_contract"][f"{name}_sha256"]
        if not path.is_file() or v12.sha256_file(path) != expected:
            raise ContractError(f"{name} hash drift")
    return paths


def _load_frames(paths: dict[str, Path]) -> tuple[pd.DataFrame, ...]:
    dtype = {"station": "string", "time": "string"}
    observations = pd.read_csv(paths["observations"], dtype=dtype)
    test_index = pd.read_csv(paths["test_index"], dtype=dtype)
    baseline = pd.read_csv(paths["baseline_interp"], dtype=dtype)
    anchor = pd.read_csv(paths["anchor"], dtype=dtype)
    if (
        list(test_index.columns) != KEYS + ["nominal_depth"]
        or list(baseline.columns) != KEYS + ["temp"]
        or list(anchor.columns) != KEYS + ["temp"]
        or len(test_index) != 26061
        or not test_index[KEYS].equals(baseline[KEYS])
        or not test_index[KEYS].equals(anchor[KEYS])
        or test_index.duplicated(KEYS).any()
    ):
        raise ContractError("official schema/key/order contract failed")
    if not (
        np.isfinite(pd.to_numeric(baseline["temp"], errors="coerce")).all()
        and np.isfinite(pd.to_numeric(anchor["temp"], errors="coerce")).all()
    ):
        raise ContractError("baseline/anchor is non-finite")
    return observations, test_index, baseline, anchor


def preflight() -> dict[str, Any]:
    config = load_config()
    paths = _paths(config)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_USER_APPROVED_EXACT_V23_MATERIALIZATION",
        "source_hashes": {name: v12.sha256_file(path) for name, path in paths.items()},
        "frozen_v23_config_sha256": config["frozen_candidate"]["config_sha256"],
        "frozen_v23_runner_sha256": config["frozen_candidate"]["runner_sha256"],
        "frozen_v23_result_sha256": config["frozen_candidate"]["result_sha256"],
        "seeds": config["frozen_candidate"]["seeds"],
        "epochs": config["frozen_candidate"]["epochs"],
        "planned_fits": 3,
        "tuning": 0,
        "automatic_retries": 0,
        "hidden_truth_rows_read": 0,
        "score_file_rows_read": 0,
        "sample_submission_rows_read": 0,
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
        raise FileExistsError("v23m1 exactly-once namespace already exists")
    config = load_config()
    paths = _paths(config)
    output_dir = Path(config["output"]["directory"])
    if output_dir.exists():
        raise FileExistsError("v23m1 output directory already exists")
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
        },
    )
    observations, test_index, baseline, anchor = _load_frames(paths)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    train_table = build_training_features(observations)
    train_design = build_normalized_curvature_design(train_table.frame)
    train_tokens, train_mask, train_context = v12.build_arrays(train_table.frame)
    local = train_design.keys["time"].dt.tz_convert("Asia/Seoul")
    start = pd.Timestamp(config["training_contract"]["training_start_inclusive_kst"])
    selected = local >= start
    weights, weight_receipt = v23.v13.domain_balanced_weights(
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

    source_config = json.loads(
        (ROOT / config["frozen_candidate"]["config"]).read_text(encoding="utf-8")
    )
    predictions = []
    fit_receipts = []
    for seed in config["frozen_candidate"]["seeds"]:
        prediction, receipt = v23.train_predict_seed(
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
        raise ContractError("v23m1 fit count drift")
    mean_normalized = np.mean(np.vstack(predictions), axis=0)
    absolute_model = query_design.baseline + mean_normalized * query_design.profile_scale
    anchor_values = pd.to_numeric(anchor["temp"], errors="raise").to_numpy(float)
    clipped = np.clip(
        absolute_model - anchor_values,
        -float(config["frozen_candidate"]["model_minus_anchor_clip_C"]),
        float(config["frozen_candidate"]["model_minus_anchor_clip_C"]),
    )
    candidate_values = anchor_values + float(
        config["frozen_candidate"]["blend"]["model_weight"]
    ) * clipped
    maximum_action = float(np.max(np.abs(candidate_values - anchor_values)))
    if maximum_action > float(config["frozen_candidate"]["maximum_action_C"]) + 1e-12:
        raise ContractError("v23m1 action cap drift")
    candidate = build_submission(test_index, candidate_values)
    output_dir.mkdir(parents=True)
    output = output_dir / config["output"]["filename"]
    candidate.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    validation = validate_submission(output, test_index)
    roundtrip = pd.read_csv(output, dtype={"station": "string", "time": "string"})
    roundtrip_values = pd.to_numeric(roundtrip["temp"], errors="coerce").to_numpy(float)
    if not np.array_equal(roundtrip_values, candidate_values):
        raise ContractError("numeric CSV round-trip drift")
    action = candidate_values - anchor_values
    abs_action = np.abs(action)
    result = {
        "schema_version": "p2.public_temperature_input_gradient_deployment.result.20260901.v23m1",
        "experiment_id": EXPERIMENT_ID,
        "status": "MATERIALIZED_READY_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 3,
        "training_rows": int(selected.sum()),
        "training_time_min_kst": local[selected].min().isoformat(),
        "training_time_max_kst": local[selected].max().isoformat(),
        "training_months": sorted(set(local[selected].dt.strftime("%Y-%m"))),
        "weight_receipt": weight_receipt,
        "fit_receipts": fit_receipts,
        "candidate": {
            "name": "P2_V23_PUBLIC_TEMP_INPUT_GRADIENT_FULL_HISTORY_BLEND020",
            "path": str(output),
            "rows": len(candidate),
            "sha256": v12.sha256_file(output),
            "validation": validation,
            "duplicate_keys": int(candidate.duplicated(KEYS).sum()),
            "finite": bool(np.isfinite(candidate_values).all()),
            "key_order_exact": bool(candidate[KEYS].equals(test_index[KEYS])),
            "changed_rows_vs_anchor": int(np.count_nonzero(abs_action > 1e-12)),
            "active_share_vs_anchor": float(np.mean(abs_action > 1e-12)),
            "abs_action_p50_C": float(np.quantile(abs_action, 0.50)),
            "abs_action_p90_C": float(np.quantile(abs_action, 0.90)),
            "abs_action_p99_C": float(np.quantile(abs_action, 0.99)),
            "abs_action_max_C": maximum_action,
            "action_rms_C": float(np.sqrt(np.mean(np.square(action)))),
            "minimum_C": float(candidate_values.min()),
            "maximum_C": float(candidate_values.max()),
        },
        "anchor": {
            "name": "P2_1_RANK1_BIN17_ONLY",
            "path": str(paths["anchor"]),
            "sha256": config["data_contract"]["anchor_sha256"],
            "public_rmse_C": config["data_contract"]["anchor_public_rmse_C"],
            "public_points": config["data_contract"]["anchor_public_points"],
            "modified": False,
        },
        "internal_evidence": {
            "historical_delta_rmse_C": -0.05189246657169555,
            "canonical_nominal_expected_points_delta": 0.6511223381640603,
            "canonical_transport_adjusted_expected_points_delta": 0.5294402465540541,
            "caveat": "Exploratory repeatedly exposed historical surface; not an official score guarantee. v26a retrospective audit found only 6/9 fold-layer cells non-harm.",
        },
        "submission_metadata": {
            "title": config["output"]["title"],
            "summary": config["output"]["summary"],
        },
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "official_test_index_rows_read": int(len(test_index)),
            "official_baseline_rows_read": int(len(baseline)),
            "official_anchor_rows_read": int(len(anchor)),
            "sample_submission_rows_read": 0,
            "score_file_rows_read": 0,
            "hidden_truth_rows_read": 0,
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
            "baseline_interp": v12.sha256_file(paths["baseline_interp"]),
            "anchor": v12.sha256_file(paths["anchor"]),
            "source_v23_config": config["frozen_candidate"]["config_sha256"],
            "source_v23_runner": config["frozen_candidate"]["runner_sha256"],
            "source_v23_result": config["frozen_candidate"]["result_sha256"],
        },
    }
    v12.atomic_json(ARTIFACT / "result.json", result)
    v12.atomic_json(REPORT / "result.json", result)
    v12.atomic_json(output_dir / "manifest.json", result)
    (output_dir / "upload-note.md").write_text(
        f"# {config['output']['title']}\n\n"
        f"{config['output']['summary']}\n\n"
        f"CSV SHA-256: `{result['candidate']['sha256']}`\n"
        f"Rows: `{len(candidate)}`\n"
        "Status: `READY_NOT_UPLOADED`\n",
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
    result = preflight() if args.preflight else execute()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
