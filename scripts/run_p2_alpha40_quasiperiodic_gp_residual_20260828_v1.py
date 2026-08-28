"""Run the bounded P2 alpha40 quasi-periodic residual pilot in two stages.

``predict`` writes and hashes prediction-only NPZ files without loading the
validation truth column from the frozen OOF anchor.  ``score`` verifies those
hashes first, then loads truth and computes the preregistered metrics.  No
official test/sample/submission path is resolved or read by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
import sklearn

from p2_restore.depth_registered_cmfpca import build_layer_identity_panel
from p2_restore.dynamic_sigmoid_profile import build_public_features
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (
    TARGET_LAYERS,
    FittedResidualLayer,
    bounded_profile_correction,
    evaluate_gate,
    paired_kst_day_bootstrap,
    predict_forward_seasonal_oas,
    rmse,
)
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_alpha40_quasiperiodic_gp_residual_20260828_v1"
DEFAULT_CONFIG = REPO / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
DEFAULT_ARTIFACT = REPO / "artifacts" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def resolve_data_dir(argument: Path | None) -> Path:
    raw = argument if argument is not None else os.environ.get("P2_DATA_DIR")
    if not raw:
        raise FileNotFoundError("set P2_DATA_DIR or pass --data-dir")
    root = Path(raw).expanduser().resolve()
    require((root / "observations.csv").is_file(), "observations.csv is missing")
    return root


def git_snapshot() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def read_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    expected = [
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    ]
    require(list(frame.columns) == expected, "observations schema changed")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    require(not frame.duplicated(["station", "time", "layer"]).any(), "observation keys duplicate")
    return frame


def block_anchor(
    anchor_path: Path,
    block: str,
    *,
    include_truth: bool,
) -> pd.DataFrame:
    columns = ["time", "layer", "current_blend50", "block"]
    if include_truth:
        columns.insert(2, "truth")
    frame = pd.read_parquet(anchor_path, columns=columns, filters=[("block", "==", block)])
    require(len(frame) > 0, f"anchor lacks block {block}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    require(not frame.duplicated(["time", "layer"]).any(), f"anchor keys duplicate: {block}")
    return frame.sort_values(["time", "layer"]).reset_index(drop=True)


def add_metadata(frame: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    metadata = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["station", "time", "layer", "nominal_depth"],
    ].copy()
    require(not metadata.duplicated(["time", "layer"]).any(), "target metadata duplicate")
    result = frame.merge(metadata, on=["time", "layer"], how="left", validate="one_to_one")
    require(result[["station", "nominal_depth"]].notna().all().all(), "target metadata missing")
    return result


def feature_rows(public_features: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    times = pd.DatetimeIndex(rows["time"])
    require(times.isin(public_features.index).all(), "public feature time missing")
    return public_features.loc[times].reset_index(drop=True)


def alpha40_reference(
    *,
    panel: pd.DataFrame,
    endpoints: pd.DataFrame,
    query: pd.DataFrame,
    train_stop: pd.Timestamp,
    config: dict[str, object],
    exclude: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    model = config["model"]
    prediction, receipts = predict_forward_seasonal_oas(
        panel,
        query,
        train_stop=train_stop,
        exclude_start=None if exclude is None else exclude[0],
        exclude_stop=None if exclude is None else exclude[1],
        season_bin_days=int(model["season_bin_days"]),
        season_window_days=float(model["season_window_days"]),
        minimum_season_rows=int(model["minimum_season_rows"]),
        fallback_nearest_complete_rows=int(model["fallback_nearest_complete_rows"]),
    )
    base = query[config["validation_anchor"]["prediction_column"]].to_numpy(np.float64)
    alpha = float(model["reference_alpha"])
    blended = base + alpha * (prediction - base)
    projection = project_profiles_vectorized(query, blended, endpoints)
    return projection.prediction, receipts


def fit_predict_fold(
    *,
    fold_name: str,
    fold_spec: dict[str, object],
    config: dict[str, object],
    observations: pd.DataFrame,
    public_features: pd.DataFrame,
    anchor_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    start, stop = utc(fold_spec["start"]), utc(fold_spec["stop"])
    masked = observations.copy()
    validation = (
        masked["time"].ge(start)
        & masked["time"].lt(stop)
        & masked["layer"].isin(TARGET_LAYERS)
    )
    masked.loc[validation, ["temp", "psal"]] = np.nan
    require(masked.loc[validation, ["temp", "psal"]].isna().all().all(), "joint mask failed")
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)

    query = block_anchor(anchor_path, fold_name, include_truth=False)
    query = add_metadata(query, observations)
    require(query["time"].ge(start).all() and query["time"].lt(stop).all(), "fold bounds differ")
    reference, reference_receipts = alpha40_reference(
        panel=panel,
        endpoints=endpoints,
        query=query,
        train_stop=start,
        config=config,
    )
    query["reference"] = reference

    training_parts: list[pd.DataFrame] = []
    inner_receipts: dict[str, object] = {}
    for training_block in fold_spec["residual_training_blocks"]:
        training = block_anchor(anchor_path, training_block, include_truth=True)
        training = add_metadata(training, observations)
        bounds = config["block_bounds"][training_block]
        inner_start, inner_stop = utc(bounds[0]), utc(bounds[1])
        require(training["time"].lt(start).all(), "residual training crossed outer fold start")
        inner_reference, receipts = alpha40_reference(
            panel=panel,
            endpoints=endpoints,
            query=training,
            train_stop=start,
            config=config,
            exclude=(inner_start, inner_stop),
        )
        training["reference"] = inner_reference
        training["residual"] = training["truth"].to_numpy(np.float64) - inner_reference
        training_parts.append(training)
        inner_receipts[training_block] = receipts
    training = pd.concat(training_parts, ignore_index=True)
    require(training["time"].lt(start).all(), "prefix label firewall failed")

    train_features = feature_rows(public_features, training)
    query_features = feature_rows(public_features, query)
    model_config = config["model"]
    raw_mean = np.zeros(len(query), dtype=np.float64)
    uncertainty = np.full(len(query), np.inf, dtype=np.float64)
    row_gate = np.zeros(len(query), dtype=bool)
    layer_receipts: dict[str, object] = {}
    for layer in TARGET_LAYERS:
        train_rows = training["layer"].to_numpy(int) == layer
        query_rows = query["layer"].to_numpy(int) == layer
        fitted = FittedResidualLayer.fit(
            train_features.loc[train_rows].reset_index(drop=True),
            training.loc[train_rows, "residual"].to_numpy(np.float64),
            gamma=float(model_config["state_rff_gamma"]),
            components=int(model_config["state_rff_components"]),
            seed=int(model_config["rff_seed"]) + int(layer),
            uncertainty_quantile=float(model_config["uncertainty_training_quantile"]),
            max_iter=int(model_config["bayesian_ridge_max_iter"]),
            tolerance=float(model_config["bayesian_ridge_tolerance"]),
        )
        layer_features = query_features.loc[query_rows].reset_index(drop=True)
        mean, standard_deviation, maximum_absolute = fitted.predict(layer_features)
        support = (
            layer_features["public_temp_count"].to_numpy(float)
            >= int(model_config["minimum_public_temperature_count"])
        ) & (
            layer_features["public_depth_span"].to_numpy(float)
            >= float(model_config["minimum_public_depth_span_m"])
        ) & (
            maximum_absolute <= float(model_config["maximum_standardized_state_absolute"])
        ) & (standard_deviation <= fitted.uncertainty_threshold)
        raw_mean[query_rows] = mean
        uncertainty[query_rows] = standard_deviation
        row_gate[query_rows] = support
        layer_receipts[str(layer)] = fitted.receipt()

    gate_frame = pd.DataFrame({"time": query["time"], "row_gate": row_gate})
    profile_gate = gate_frame.groupby("time", sort=False)["row_gate"].transform("all").to_numpy(bool)
    raw_correction = float(model_config["residual_blend"]) * raw_mean
    correction, correction_receipt = bounded_profile_correction(
        raw_correction,
        profile_gate,
        rms_cap=float(model_config["correction_rms_cap_c"]),
        p99_cap=float(model_config["correction_p99_cap_c"]),
    )
    candidate = reference + correction
    require(np.array_equal(candidate[~profile_gate], reference[~profile_gate]), "fallback is not exact")
    output = query.loc[:, ["time", "layer"]].copy()
    output["reference"] = reference
    output["candidate"] = candidate
    output["correction"] = correction
    output["uncertainty"] = uncertainty
    output["enabled"] = profile_gate
    receipt = {
        "fold": fold_name,
        "outer_train_label_stop_exclusive_utc": start.isoformat(),
        "validation_start_utc": start.isoformat(),
        "validation_stop_utc": stop.isoformat(),
        "validation_truth_column_loaded": False,
        "validation_target_temp_psal_masked_together": True,
        "residual_training_blocks": list(fold_spec["residual_training_blocks"]),
        "residual_training_rows": int(len(training)),
        "reference_oas": reference_receipts,
        "inner_reference_oas": inner_receipts,
        "layers": layer_receipts,
        "correction": correction_receipt,
    }
    return output, receipt


def write_prediction_npz(path: Path, frame: pd.DataFrame) -> None:
    times = pd.DatetimeIndex(frame["time"])
    np.savez_compressed(
        path,
        time_ns=times.as_unit("ns").asi8.astype(np.int64),
        layer=frame["layer"].to_numpy(np.int16),
        reference=frame["reference"].to_numpy(np.float64),
        candidate=frame["candidate"].to_numpy(np.float64),
        correction=frame["correction"].to_numpy(np.float64),
        uncertainty=frame["uncertainty"].to_numpy(np.float64),
        enabled=frame["enabled"].to_numpy(bool),
    )


def decode_committed_time(values: np.ndarray) -> pd.DatetimeIndex:
    """Decode committed integers across pandas source timestamp resolutions.

    The already-hashed pilot files were written from parquet ``datetime64[us]``
    values before this explicit conversion was added.  Their magnitude makes
    the storage unit unambiguous and no prediction value is rewritten.
    """

    current = np.asarray(values, dtype=np.int64)
    unit = "ns" if int(np.max(np.abs(current))) >= 10**17 else "us"
    decoded = pd.DatetimeIndex(pd.to_datetime(current, unit=unit, utc=True))
    if decoded.min().year < 2024 or decoded.max().year > 2025:
        raise RuntimeError("committed time integer unit is invalid")
    return decoded


def prediction_stage(
    *,
    config_path: Path,
    artifact_dir: Path,
    data_dir: Path,
    config: dict[str, object],
) -> None:
    if artifact_dir.exists():
        existing_files = [path for path in artifact_dir.rglob("*") if path.is_file()]
        require(
            not existing_files and not (artifact_dir / "prediction_commitment.json").exists(),
            f"nonempty append-only artifact exists: {artifact_dir}",
        )
    observations_path = data_dir / "observations.csv"
    require(sha256(observations_path) == config["input_pins"]["observations_sha256"], "observation pin changed")
    anchor_path = REPO / config["validation_anchor"]["path"]
    require(sha256(anchor_path) == config["validation_anchor"]["sha256"], "anchor pin changed")
    observations = read_observations(observations_path)
    public_features = build_public_features(observations)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = artifact_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    fold_receipts: dict[str, object] = {}
    outputs: dict[str, object] = {}
    for fold_name, fold_spec in config["folds"].items():
        prediction, receipt = fit_predict_fold(
            fold_name=fold_name,
            fold_spec=fold_spec,
            config=config,
            observations=observations,
            public_features=public_features,
            anchor_path=anchor_path,
        )
        path = predictions_dir / f"{EXPERIMENT_ID}_{fold_name}.npz"
        write_prediction_npz(path, prediction)
        outputs[fold_name] = {
            "path": str(path.relative_to(REPO)),
            "rows": int(len(prediction)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        fold_receipts[fold_name] = receipt
    commitment = {
        "schema_version": "p2.alpha40_quasiperiodic_gp_residual.prediction_commitment.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_count": 1,
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "truth_metric_computed": False,
        "validation_truth_column_loaded": False,
        "precommit_recovery": {
            "performed": True,
            "reason": "Feasibility runs stopped before any prediction file or commitment: the earliest seasonal bin lacked nearby complete rows, then the 2024 Jul-Aug outer fold proved non-cross-fittable because excluding its sole prior target block left zero labels.",
            "truth_metric_observed_before_recovery": False,
            "recovery_change": "Added a deterministic nearest-1000 complete-prefix boundary fallback and replaced the structurally non-cross-fittable 2024 Jul-Aug fold with the preregistered 2025 Nov-Dec block. Final committed folds are 2024 Sep-Oct, 2025 Jul-Aug, and 2025 Nov-Dec.",
        },
        "hyperparameter_search_count": int(config["model"]["hyperparameter_search_count"]),
        "official_test_sample_submission_paths_read": False,
        "prediction_outputs": outputs,
        "fold_receipts": fold_receipts,
        "inputs": {
            "config": {"path": str(config_path.relative_to(REPO)), "sha256": sha256(config_path)},
            "observations": {"path": str(observations_path), "sha256": sha256(observations_path)},
            "frozen_oof_anchor": {"path": str(anchor_path.relative_to(REPO)), "sha256": sha256(anchor_path)},
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "git": git_snapshot(),
        },
    }
    commitment_path = artifact_dir / "prediction_commitment.json"
    commitment_path.write_text(json.dumps(commitment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "predict", "commitment": str(commitment_path), "outputs": outputs}, ensure_ascii=False, indent=2))


def metric_group(frame: pd.DataFrame, label: str) -> dict[str, object]:
    reference = rmse(frame["truth"].to_numpy(), frame["reference"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    return {
        "group": label,
        "rows": int(len(frame)),
        "alpha40_reference_rmse": reference,
        "quasiperiodic_gp_candidate_rmse": candidate,
        "delta_rmse": candidate - reference,
    }


def score_stage(
    *,
    config_path: Path,
    artifact_dir: Path,
    config: dict[str, object],
) -> None:
    commitment_path = artifact_dir / "prediction_commitment.json"
    result_path = artifact_dir / "result.json"
    require(commitment_path.is_file(), "prediction commitment is absent")
    require(not result_path.exists(), "result already exists; one-shot score cannot rerun")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    require(commitment["truth_metric_computed"] is False, "commitment was not pre-truth")
    require(commitment["validation_truth_column_loaded"] is False, "truth was loaded before hash")
    anchor_path = REPO / config["validation_anchor"]["path"]
    fold_frames: list[pd.DataFrame] = []
    for fold_name, output in commitment["prediction_outputs"].items():
        prediction_path = REPO / output["path"]
        require(sha256(prediction_path) == output["sha256"], f"prediction hash changed: {fold_name}")
        with np.load(prediction_path, allow_pickle=False) as payload:
            frame = pd.DataFrame(
                {
                    "time": decode_committed_time(payload["time_ns"]),
                    "layer": payload["layer"].astype(int),
                    "reference": payload["reference"].astype(float),
                    "candidate": payload["candidate"].astype(float),
                    "correction": payload["correction"].astype(float),
                    "uncertainty": payload["uncertainty"].astype(float),
                    "enabled": payload["enabled"].astype(bool),
                }
            )
        truth = block_anchor(anchor_path, fold_name, include_truth=True).loc[:, ["time", "layer", "truth"]]
        scored = frame.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        require(len(scored) == len(frame) and scored["truth"].notna().all(), "truth alignment failed")
        scored["fold"] = fold_name
        fold_frames.append(scored)
    scored = pd.concat(fold_frames, ignore_index=True)
    aggregate = metric_group(scored, "aggregate")
    by_fold = {
        str(name): metric_group(group, str(name))
        for name, group in scored.groupby("fold", sort=True)
    }
    by_layer = {
        str(int(name)): metric_group(group, str(int(name)))
        for name, group in scored.groupby("layer", sort=True)
    }
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(config["model"]["bootstrap_replicates"]),
        seed=int(config["model"]["bootstrap_seed"]),
    )
    correction = scored["correction"].to_numpy(np.float64)
    enabled = scored["enabled"].to_numpy(bool)
    correction_rms = float(np.sqrt(np.mean(correction**2)))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    fallback_max = float(np.max(np.abs(correction[~enabled]))) if (~enabled).any() else 0.0
    require(fallback_max == 0.0, "committed fallback is not exact")
    gate = evaluate_gate(
        aggregate_delta=float(aggregate["delta_rmse"]),
        ci90_high=float(bootstrap["ci90_high"]),
        fold_deltas={name: float(value["delta_rmse"]) for name, value in by_fold.items()},
        layer_deltas={name: float(value["delta_rmse"]) for name, value in by_layer.items()},
        correction_rms=correction_rms,
        correction_p99=correction_p99,
        thresholds=config["gate"],
    )
    decision = "PASS_BOUNDED_LOCAL_GATE_NO_CSV_NO_UPLOAD" if gate["passed"] else "FAIL_GATE_STOP_NO_CSV_NO_RESEARCH_LOOP"
    result = {
        "schema_version": "p2.alpha40_quasiperiodic_gp_residual.result.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_count": 1,
        "decision": decision,
        "historical_exposure": config["historical_exposure"],
        "split_claim": "architecture-fresh committed forward folds; not exact untouched holdout",
        "reference": "Frozen OOF current_blend50 anchor plus prefix-fit seasonal OAS alpha=0.40 and endpoint/PAVA projection.",
        "metrics": {"aggregate": aggregate, "by_fold": by_fold, "by_layer": by_layer},
        "paired_kst_day_bootstrap": bootstrap,
        "correction": {
            "rms_c": correction_rms,
            "p99_absolute_c": correction_p99,
            "maximum_absolute_c": float(np.max(np.abs(correction))),
            "enabled_rows": int(enabled.sum()),
            "enabled_fraction": float(enabled.mean()),
            "fallback_maximum_absolute_c": fallback_max,
        },
        "gate": gate,
        "prediction_commitment": {
            "path": str(commitment_path.relative_to(REPO)),
            "sha256": sha256(commitment_path),
            "verified_before_truth_load": True,
        },
        "leakage_audit": {
            "official_test_sample_submission_paths_read": False,
            "official_answer_or_mirror_read": False,
            "candidate_csv_generated": False,
            "official_upload_performed": False,
            "validation_target_temp_psal_masked_together_during_prediction": True,
            "validation_truth_loaded_only_after_prediction_hash_verification": True,
            "post_result_parameter_search_performed": False,
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    sources = [
        config_path,
        REPO / "src" / "p2_restore" / f"{EXPERIMENT_ID}.py",
        Path(__file__).resolve(),
        REPO / "scripts" / f"qa_{EXPERIMENT_ID}.py",
        REPO / "tests" / f"test_{EXPERIMENT_ID}.py",
    ]
    manifest = {
        "schema_version": "p2.alpha40_quasiperiodic_gp_residual.manifest.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "sources": {str(path.relative_to(REPO)): sha256(path) for path in sources},
        "inputs": commitment["inputs"],
        "outputs": {
            "prediction_commitment": {"path": str(commitment_path.relative_to(REPO)), "sha256": sha256(commitment_path)},
            "predictions": commitment["prediction_outputs"],
            "result": {"path": str(result_path.relative_to(REPO)), "sha256": sha256(result_path)},
            "candidate_csv": None,
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "score", "decision": decision, "metrics": result["metrics"], "gate": gate}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    require(args.execute, "bounded pilot requires --execute")
    config_path = args.config.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["experiment_id"] == EXPERIMENT_ID, "config experiment differs")
    require(int(config["model"]["hyperparameter_search_count"]) == 0, "search count differs")
    require(config["leakage_contract"]["official_test_sample_submission_paths_read"] is False, "official path flag")
    require(config["leakage_contract"]["candidate_csv_generation_authorized"] is False, "CSV flag")
    if args.stage == "predict":
        prediction_stage(
            config_path=config_path,
            artifact_dir=artifact_dir,
            data_dir=resolve_data_dir(args.data_dir),
            config=config,
        )
    else:
        score_stage(config_path=config_path, artifact_dir=artifact_dir, config=config)


if __name__ == "__main__":
    main()
