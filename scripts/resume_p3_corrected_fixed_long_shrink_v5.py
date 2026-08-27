"""Load-only continuation of the consumed P3 v5 full-refit attempt."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import scripts.run_p3_corrected_fixed_long_shrink_v4 as v4
    import scripts.run_p3_corrected_fixed_long_shrink_v5_full_refit as v5
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    import run_p3_corrected_fixed_long_shrink_v4 as v4
    import run_p3_corrected_fixed_long_shrink_v5_full_refit as v5

PROTOCOL_RELATIVE = "configs/experiments/p3_corrected_fixed_long_shrink_v5_resume_protocol.json"
PROTOCOL_SHA256 = "4cec179a854fd67138a32ec30f0edd5fd2c0de2ad45562be1aa7ce71a995fda2"
STAGE_RELATIVE = "tmp/p3_corrected_shrink_v5_refit_8o788k5g"
MODEL_SHA256 = {
    "single.cbm": "ee32813e5b1867b272dc7b0b2c15cb285af14a733574b777ffb35c051b2a525e",
    "multi.cbm": "dbb0c3497f16c1fb684857c1e7e4389d3e4532ab62f05d28c00c61784c590313",
    "router.joblib": "897caa8e357b58db59f3ce8746a01fa85750f9379fc56d8ea00986cf5fed4bca",
    "calibrator.joblib": "ff0ef734d1bd7dbdf0e0365be7c70d9465c31140a9baf7f03d8e3263b0270df9",
    "feature_columns.json": "49f1e03b43cce691f046027a4c59aac1bcb6846970c32ebf94c4b02382d4fa7a",
}


def _model_paths(root: Path) -> dict[str, Path]:
    directory = root / STAGE_RELATIVE / "models/full"
    return {name: directory / name for name in MODEL_SHA256}


def _resume_inputs(root: Path, data_dir: Path) -> dict[str, tuple[Path, str]]:
    inputs = v5._expected_inputs(root, data_dir)
    inputs.update(
        {
            "resume_protocol": (root / PROTOCOL_RELATIVE, PROTOCOL_SHA256),
            "attempt_lock": (
                root / v5.LOCK_RELATIVE,
                "49a264391dfa95f7cfe2a14332a4e11a21654b4eb96d7883e85d7ddb9a5bb616",
            ),
            "failed_runner": (
                root / "scripts/run_p3_corrected_fixed_long_shrink_v5_full_refit.py",
                "7372a3455af619b8f04cbccbf48949f134d07acf7755669289a0c0c3ed5815ef",
            ),
        }
    )
    inputs.update(
        {
            f"fresh_model/{name}": (path, MODEL_SHA256[name])
            for name, path in _model_paths(root).items()
        }
    )
    return inputs


def _verify_inputs(inputs: dict[str, tuple[Path, str]]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, (path, expected) in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"required resume input is missing: {name}")
        digest = v4.sha256_file(path)
        if digest != expected:
            raise ValueError(f"resume input SHA differs: {name}")
        observed[name] = digest
    return observed


def _infer_fresh_models(
    *, root: Path, data_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model_paths = _model_paths(root)
    feature_payload = json.loads(model_paths["feature_columns.json"].read_text(encoding="utf-8"))
    test_features = pd.read_parquet(root / v5.CACHE_RELATIVE / "test_features.parquet")
    test_index = pd.read_csv(data_dir / "test_index.csv")
    persistence = pd.read_csv(data_dir / "baseline_persistence.csv")
    if not persistence[v4.KEYS].equals(test_index[v4.KEYS]):
        raise ValueError("persistence keys/order differ from test_index")
    routed, current = v4._saved_base_inference(
        test_features=test_features,
        test_index=test_index,
        feature_columns=feature_payload["columns"],
        single_path=model_paths["single.cbm"],
        multi_path=model_paths["multi.cbm"],
        router_path=model_paths["router.joblib"],
    )
    anchor = persistence["hs_pred"].to_numpy(float)
    if not np.allclose(current, anchor, atol=0.0, rtol=0.0):
        raise ValueError("same-case hs_current differs from persistence")
    calibrator = joblib.load(model_paths["calibrator.joblib"])
    prediction = calibrator.predict(routed, anchor, test_index["lead_h"].to_numpy(int))
    candidate = v4.build_submission(test_index, prediction)
    prior = pd.read_csv(root / v5.V4_RELATIVE / "candidate/submission.csv")
    if not prior[v4.KEYS].equals(test_index[v4.KEYS]):
        raise ValueError("prior winner candidate keys/order differ")
    prior_values = prior["hs_pred"].to_numpy(float)
    return (
        candidate,
        test_index,
        {
            "minimum_m": float(np.min(prediction)),
            "median_m": float(np.median(prediction)),
            "maximum_m": float(np.max(prediction)),
            "mean_m": float(np.mean(prediction)),
            "fresh_refit_minus_prior_winner_prediction_rmse_m": v4._rmse(prediction, prior_values),
            "fresh_refit_minus_prior_winner_prediction_max_abs_m": float(
                np.max(np.abs(prediction - prior_values))
            ),
        },
    )


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    snapshot = _verify_inputs(_resume_inputs(root, data_dir))
    protocol = json.loads((root / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))
    if protocol["resume_contract"]["refit_count_during_resume"] != 0:
        raise ValueError("resume protocol permits no refit")
    if protocol["resume_contract"]["persistence_weight"] != 0.25:
        raise ValueError("resume winner coefficient differs")
    stage = root / STAGE_RELATIVE
    if (stage / "candidate").exists() or (stage / "metrics.json").exists():
        raise FileExistsError("preserved stage already contains resume outputs")
    if (root / v5.OUTPUT_RELATIVE).exists():
        raise FileExistsError("canonical full-refit output already exists")
    _, evaluation = v4.evaluate_identical_oof(root)
    if not evaluation["gate"]["passed"]:
        raise RuntimeError("unchanged corrected OOF winner gate no longer passes")
    return {
        "status": "SAME_ATTEMPT_LOAD_ONLY_RESUME_CHECK_PASS",
        "pinned_input_count": len(snapshot),
        "protocol_sha256": snapshot["resume_protocol"],
        "preserved_model_count": len(MODEL_SHA256),
        "candidate_absent": True,
        "output_absent": True,
        "resume_fit_count": 0,
        "winner_rmse_m": evaluation["candidate"]["rmse_m"],
    }


def resume_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    check_only(root=root, data_dir=data_dir)
    inputs = _resume_inputs(root, data_dir)
    before = _verify_inputs(inputs)
    started = time.perf_counter()
    stage = root / STAGE_RELATIVE
    first, test_index, first_summary = _infer_fresh_models(root=root, data_dir=data_dir)
    candidate_path = stage / "candidate/submission.csv"
    v4._write_submission_exclusive(first, test_index, candidate_path)
    second, second_index, second_summary = _infer_fresh_models(root=root, data_dir=data_dir)
    reproduced_path = stage / "candidate/reproduced_submission.csv"
    v4._write_submission_exclusive(second, second_index, reproduced_path)
    if candidate_path.read_bytes() != reproduced_path.read_bytes():
        raise RuntimeError("fresh-refit saved-model inference is not byte-identical")
    _, evaluation = v4.evaluate_identical_oof(root)
    training_receipt = {
        "original_attempt_actual_fit_counts": {
            "single_catboost": 1,
            "multi_catboost": 1,
            "router": 1,
        },
        "resume_fit_counts": {"single_catboost": 0, "multi_catboost": 0, "router": 0},
        "full_training_anchor_count": 24_360,
        "single_training_rows": 146_160,
        "feature_count": 591,
        "seed": 20260817,
        "fresh_model_sha256": MODEL_SHA256,
        "fresh_models_differ_from_prior_full_fit": {
            "single": MODEL_SHA256["single.cbm"]
            != "b59972b88932a860ae38318f626e13f70e9429dd71234c2b4d52fc7369eedeac",
            "multi": MODEL_SHA256["multi.cbm"]
            != "9dffba79e9ccf33482fd7212b3eda5ec3a98606d5b14a2e34e156988b87280cc",
            "router": MODEL_SHA256["router.joblib"]
            != "897caa8e357b58db59f3ce8746a01fa85750f9379fc56d8ea00986cf5fed4bca",
        },
    }
    metrics = {
        "created_at": v4._now(),
        "experiment_id": "p3_corrected_fixed_long_shrink_v5_full_refit",
        "status": "ACTUAL_FULL_REFIT_SAME_ATTEMPT_RESUMED_CANDIDATE_CREATED_NOT_UPLOADED",
        "interruption_disclosure": {
            "original_runner_sha256": before["failed_runner"],
            "failure": "post-fit prior-model 1e-12 prediction equality guard",
            "fit_completed_before_failure": True,
            "candidate_created_before_failure": False,
            "same_attempt_resume": True,
            "resume_load_only": True,
            "winner_structure_or_weight_changed": False,
        },
        "evaluation": evaluation,
        "training_receipt": training_receipt,
        "candidate_validation": {
            "rows": len(first),
            "cases": int(first["case_id"].nunique()),
            "key_order_exact": True,
            "finite": bool(np.isfinite(first["hs_pred"]).all()),
            "range_0_to_30_m": bool(first["hs_pred"].between(0.0, 30.0).all()),
            "fresh_saved_model_reproduction_byte_identical": True,
            "first_inference_summary": first_summary,
            "second_inference_summary": second_summary,
            "candidate_sha256": v4.sha256_file(candidate_path),
            "reproduced_candidate_sha256": v4.sha256_file(reproduced_path),
        },
        "access_counters_total_attempt": {
            "test_feature_cache_reads": 3,
            "test_index_reads": 3,
            "test_context_reads": 0,
            "test_target_or_hidden_label_reads": 0,
            "absolute_test_timestamp_recovery_attempts": 0,
            "current_or_frozen_writes": 0,
            "upload_attempts": 0,
        },
        "elapsed_seconds_resume": time.perf_counter() - started,
    }
    v4._atomic_json(stage / "metrics.json", metrics)
    v4._atomic_json(
        stage / "resume_completion_status.json",
        {
            "created_at": v4._now(),
            "status": "SAME_ATTEMPT_RESUME_COMPLETE",
            "attempt_lock_sha256": before["attempt_lock"],
            "resume_protocol_sha256": before["resume_protocol"],
            "load_only": True,
            "fit_count": 0,
            "candidate_sha256": v4.sha256_file(candidate_path),
            "candidate_reproduction_byte_identical": True,
            "uploaded": False,
        },
    )
    after = _verify_inputs(inputs)
    if before != after:
        raise RuntimeError("pinned resume inputs changed during continuation")
    implementation = {
        "original_config": root / v5.CONFIG_RELATIVE,
        "resume_protocol": root / PROTOCOL_RELATIVE,
        "failed_runner": root / "scripts/run_p3_corrected_fixed_long_shrink_v5_full_refit.py",
        "resume_runner": Path(__file__).resolve(),
        "calibrator": root / "src/p3_wave/corrected_fixed_long_shrink.py",
        "test": root / "tests/test_resume_p3_corrected_fixed_long_shrink_v5.py",
    }
    manifest = {
        "created_at": v4._now(),
        "experiment_id": "p3_corrected_fixed_long_shrink_v5_full_refit",
        "status": metrics["status"],
        "append_only_generation": True,
        "same_attempt_resume": True,
        "attempt_lock_sha256": before["attempt_lock"],
        "resume_protocol_sha256": before["resume_protocol"],
        "input_sha256_before": before,
        "input_sha256_after": after,
        "input_unchanged": True,
        "implementation_sha256": {
            name: v4.sha256_file(path) for name, path in implementation.items()
        },
        "output_files": v4._artifact_hashes(stage),
        "actual_fit_counts_original_attempt": training_receipt[
            "original_attempt_actual_fit_counts"
        ],
        "fit_counts_resume": training_receipt["resume_fit_counts"],
        "candidate_created": True,
        "candidate_uploaded": False,
        "test_target_or_hidden_labels_used": 0,
        "absolute_test_timestamp_recovered": False,
        "current_or_frozen_mutated": False,
        "no_raw_values_in_manifest": True,
    }
    v4._atomic_json(stage / "manifest.json", manifest)
    manifest_sha = v4.sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n", encoding="ascii")
    output = root / v5.OUTPUT_RELATIVE
    if output.exists():
        raise FileExistsError("canonical output appeared before resume finalize")
    stage.replace(output)
    result = {
        "status": metrics["status"],
        "artifact_dir": v5.OUTPUT_RELATIVE,
        "candidate_sha256": v4.sha256_file(output / "candidate/submission.csv"),
        "reproduced_candidate_sha256": v4.sha256_file(
            output / "candidate/reproduced_submission.csv"
        ),
        "metrics_sha256": v4.sha256_file(output / "metrics.json"),
        "completion_sha256": v4.sha256_file(output / "resume_completion_status.json"),
        "manifest_sha256": v4.sha256_file(output / "manifest.json"),
        "corrected_oof_rmse_m": evaluation["candidate"]["rmse_m"],
        "delta_m": evaluation["delta_candidate_minus_incumbent_m"],
        "original_fit_counts": training_receipt["original_attempt_actual_fit_counts"],
        "resume_fit_counts": training_receipt["resume_fit_counts"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", choices=("check-only", "resume"), default="check-only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).expanduser().resolve(strict=True)
    if args.mode == "check-only":
        print(json.dumps(check_only(root=root, data_dir=data_dir), ensure_ascii=False, indent=2))
        return 0
    resume_experiment(root=root, data_dir=data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
