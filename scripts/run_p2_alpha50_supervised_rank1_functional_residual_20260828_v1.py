"""Run the sealed P2 supervised rank-one functional residual diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p2_restore.depth_registered_cmfpca import (  # noqa: E402
    build_layer_identity_panel,
)
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (  # noqa: E402
    bounded_profile_correction,
    paired_kst_day_bootstrap,
    predict_forward_seasonal_oas,
    rmse,
)
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.supervised_rank1_functional_residual import (  # noqa: E402
    TARGET_LAYERS,
    SupervisedRank1Residual,
    build_public_functional_features,
    orthogonal_share,
    vector_cosine,
)

EXPERIMENT_ID = "p2_alpha50_supervised_rank1_functional_residual_20260828_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"


class ContractError(RuntimeError):
    """Raised when an immutable or truth-late contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_config(data_dir: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID drifted")
    policy = config["execution_policy"]
    if any(
        (
            policy["official_test_sample_submission_read_allowed"],
            policy["submission_csv_generation_allowed"],
            policy["official_upload_authorized"],
            policy["result_based_retry"],
        )
    ):
        raise ContractError("forbidden official access or retry was enabled")
    for record in config["immutable_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ContractError(f"immutable input changed: {path}")
    observations = data_dir / config["source_observations"]["filename"]
    if not observations.is_file() or sha256_file(observations) != config["source_observations"]["sha256"]:
        raise ContractError("observations.csv pin changed")
    return config


def utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def read_observations(path: Path) -> pd.DataFrame:
    expected = ["station", "year", "layer", "time", "temp", "psal", "depth", "nominal_depth"]
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    if list(frame.columns) != expected:
        raise ContractError("observations schema changed")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    if frame.duplicated(["station", "time", "layer"]).any():
        raise ContractError("observation keys duplicate")
    return frame


def block_anchor(path: Path, block: str, *, include_truth: bool) -> pd.DataFrame:
    columns = ["time", "layer", "current_blend50", "block"]
    if include_truth:
        columns.insert(2, "truth")
    frame = pd.read_parquet(path, columns=columns, filters=[("block", "==", block)])
    if frame.empty:
        raise ContractError(f"anchor lacks block {block}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["layer"] = frame["layer"].astype(int)
    if frame.duplicated(["time", "layer"]).any():
        raise ContractError(f"anchor keys duplicate: {block}")
    return frame.sort_values(["time", "layer"]).reset_index(drop=True)


def add_metadata(frame: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    metadata = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["station", "time", "layer", "nominal_depth"],
    ].copy()
    if metadata.duplicated(["time", "layer"]).any():
        raise ContractError("target metadata duplicate")
    result = frame.merge(metadata, on=["time", "layer"], how="left", validate="one_to_one")
    if result[["station", "nominal_depth"]].isna().any().any():
        raise ContractError("target metadata missing")
    return result


def alpha50_reference(
    *,
    panel: pd.DataFrame,
    endpoints: pd.DataFrame,
    query: pd.DataFrame,
    train_stop: pd.Timestamp,
    config: dict[str, Any],
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
    base = query["current_blend50"].to_numpy(dtype=np.float64)
    blended = base + float(model["reference_alpha"]) * (prediction - base)
    return project_profiles_vectorized(query, blended, endpoints).prediction, receipts


def profile_response(training: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    pivot = training.pivot(index="time", columns="layer", values="residual").reindex(columns=TARGET_LAYERS)
    complete = pivot.notna().all(axis=1)
    return pd.DatetimeIndex(pivot.index[complete]), pivot.loc[complete].to_numpy(dtype=np.float64)


def align_features(features: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    if not times.isin(features.index).all():
        raise ContractError("public functional features lack target times")
    return features.loc[times].reset_index(drop=True)


def fit_predict_fold(
    *,
    fold_name: str,
    fold_spec: dict[str, Any],
    config: dict[str, Any],
    observations: pd.DataFrame,
    functional_features: pd.DataFrame,
    anchor_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start, stop = utc(fold_spec["start"]), utc(fold_spec["stop"])
    masked = observations.copy()
    validation_mask = (
        masked["time"].ge(start)
        & masked["time"].lt(stop)
        & masked["layer"].isin(TARGET_LAYERS)
    )
    masked.loc[validation_mask, ["temp", "psal"]] = np.nan
    if not masked.loc[validation_mask, ["temp", "psal"]].isna().all().all():
        raise ContractError("joint validation target mask failed")
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)
    query = add_metadata(block_anchor(anchor_path, fold_name, include_truth=False), observations)
    reference, reference_receipts = alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=query,
        train_stop=start,
        config=config,
    )
    query["reference"] = reference

    training_parts: list[pd.DataFrame] = []
    training_reference_receipts: dict[str, Any] = {}
    for training_block in fold_spec["training_blocks"]:
        training = add_metadata(block_anchor(anchor_path, training_block, include_truth=True), observations)
        if not training["time"].lt(start).all():
            raise ContractError("training labels cross outer fold boundary")
        bounds = config["block_bounds"][training_block]
        inner_reference, receipts = alpha50_reference(
            panel=panel,
            endpoints=endpoints,
            query=training,
            train_stop=start,
            config=config,
            exclude=(utc(bounds[0]), utc(bounds[1])),
        )
        training["reference"] = inner_reference
        training["residual"] = training["truth"].to_numpy(dtype=np.float64) - inner_reference
        training_parts.append(training)
        training_reference_receipts[training_block] = receipts
    training = pd.concat(training_parts, ignore_index=True)
    train_times, response = profile_response(training)
    train_features = align_features(functional_features, train_times)
    valid_train = train_features["public_profile_valid"].to_numpy(dtype=bool)
    train_features = train_features.loc[valid_train].reset_index(drop=True)
    response = response[valid_train]
    train_times = train_times[valid_train]
    fitted = SupervisedRank1Residual.fit(train_features, response, train_times)

    query_times = pd.DatetimeIndex(sorted(query["time"].unique()))
    query_features = align_features(functional_features, query_times)
    profile_prediction, profile_enabled, leverage = fitted.predict(query_features, query_times)
    layer_to_column = {layer: index for index, layer in enumerate(TARGET_LAYERS)}
    time_to_row = {timestamp: index for index, timestamp in enumerate(query_times)}
    raw_correction = np.zeros(len(query), dtype=np.float64)
    enabled = np.zeros(len(query), dtype=bool)
    row_leverage = np.full(len(query), np.inf, dtype=np.float64)
    for row, values in enumerate(query.itertuples(index=False)):
        profile_row = time_to_row[pd.Timestamp(values.time)]
        column = layer_to_column[int(values.layer)]
        raw_correction[row] = profile_prediction[profile_row, column]
        enabled[row] = bool(profile_enabled[profile_row])
        row_leverage[row] = leverage[profile_row]
    correction, cap_receipt = bounded_profile_correction(
        raw_correction,
        enabled,
        rms_cap=float(config["model"]["correction_rms_cap_c"]),
        p99_cap=float(config["model"]["correction_absolute_cap_c"]),
    )
    unprojected = reference + correction
    projected = project_profiles_vectorized(query, unprojected, endpoints).prediction
    actual_correction = projected - reference
    if np.max(np.abs(actual_correction[~enabled]), initial=0.0) > 1e-12:
        raise ContractError("unsupported profile changed after projection")
    output = query[["time", "layer", "current_blend50"]].copy()
    output["reference"] = reference
    output["candidate"] = projected
    output["correction"] = actual_correction
    output["enabled"] = enabled
    output["leverage"] = row_leverage
    receipt = {
        "fold": fold_name,
        "validation_truth_column_loaded": False,
        "validation_target_temp_psal_masked_together": True,
        "training_blocks": list(fold_spec["training_blocks"]),
        "training_profile_rows": int(len(train_times)),
        "model": fitted.receipt(),
        "reference_oas": reference_receipts,
        "training_reference_oas": training_reference_receipts,
        "preprojection_cap": cap_receipt,
        "postprojection_correction_rms": float(np.sqrt(np.mean(np.square(actual_correction)))),
        "postprojection_correction_p99": float(np.quantile(np.abs(actual_correction), 0.99)),
    }
    return output, receipt


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    np.savez_compressed(
        path,
        time_ns=pd.DatetimeIndex(frame["time"]).as_unit("ns").asi8,
        layer=frame["layer"].to_numpy(dtype=np.int16),
        current_blend50=frame["current_blend50"].to_numpy(dtype=np.float64),
        reference=frame["reference"].to_numpy(dtype=np.float64),
        candidate=frame["candidate"].to_numpy(dtype=np.float64),
        correction=frame["correction"].to_numpy(dtype=np.float64),
        enabled=frame["enabled"].to_numpy(dtype=bool),
        leverage=frame["leverage"].to_numpy(dtype=np.float64),
    )


def metric_record(frame: pd.DataFrame) -> dict[str, float | int]:
    baseline = rmse(frame["truth"], frame["reference"])
    candidate = rmse(frame["truth"], frame["candidate"])
    return {"rows": int(len(frame)), "reference_rmse": baseline, "candidate_rmse": candidate, "delta_rmse": candidate - baseline}


def run(config: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_directory = ROOT / config["artifact_directory"]
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    observations = read_observations(data_dir / config["source_observations"]["filename"])
    functional_features = build_public_functional_features(
        observations,
        ridge=float(config["model"]["spline_ridge"]),
        change_hours=tuple(map(int, config["model"]["change_hours"])),
    )
    anchor_path = ROOT / config["immutable_inputs"]["alpha50_proxy"]["path"]
    predictions_directory = output_directory / "predictions"
    predictions_directory.mkdir()
    outputs: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    for fold_name, fold_spec in config["folds"].items():
        prediction, receipt = fit_predict_fold(
            fold_name=fold_name,
            fold_spec=fold_spec,
            config=config,
            observations=observations,
            functional_features=functional_features,
            anchor_path=anchor_path,
        )
        path = predictions_directory / f"{fold_name}.npz"
        write_prediction(path, prediction)
        outputs[fold_name] = {"path": str(path.relative_to(ROOT)), "rows": int(len(prediction)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        receipts[fold_name] = receipt
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "comparator": "INCUMBENT_PROXY_VALIDATION",
        "comparator_disclosure": config["comparator_disclosure"],
        "truth_metric_computed": False,
        "validation_truth_column_loaded": False,
        "official_rows_read": 0,
        "outputs": outputs,
        "receipts": receipts,
        "observations_sha256": config["source_observations"]["sha256"],
        "anchor_sha256": config["immutable_inputs"]["alpha50_proxy"]["sha256"],
        "config_sha256": sha256_file(CONFIG),
    }
    atomic_json(output_directory / "prediction_commitment.json", commitment)

    scored_frames: list[pd.DataFrame] = []
    for fold_name, record in outputs.items():
        path = ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ContractError("committed prediction changed")
        with np.load(path, allow_pickle=False) as payload:
            prediction = pd.DataFrame(
                {
                    "time": pd.to_datetime(payload["time_ns"], unit="ns", utc=True),
                    "layer": payload["layer"].astype(int),
                    "current_blend50": payload["current_blend50"],
                    "reference": payload["reference"],
                    "candidate": payload["candidate"],
                    "correction": payload["correction"],
                    "enabled": payload["enabled"],
                    "leverage": payload["leverage"],
                }
            )
        truth = block_anchor(anchor_path, fold_name, include_truth=True)[["time", "layer", "truth"]]
        scored = prediction.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        if scored["truth"].isna().any() or len(scored) != int(record["rows"]):
            raise ContractError("truth binding failed")
        scored["fold"] = fold_name
        scored_frames.append(scored)
    scored = pd.concat(scored_frames, ignore_index=True)
    metrics = {
        "aggregate": metric_record(scored),
        "by_fold": {str(key): metric_record(group) for key, group in scored.groupby("fold", sort=True)},
        "by_layer": {str(int(key)): metric_record(group) for key, group in scored.groupby("layer", sort=True)},
    }
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(config["gate"]["bootstrap_replicates"]),
        seed=int(config["gate"]["bootstrap_seed"]),
    )
    oas_axis = scored["reference"].to_numpy() - scored["current_blend50"].to_numpy()
    correction = scored["candidate"].to_numpy() - scored["reference"].to_numpy()
    strongest = pd.read_parquet(
        ROOT / config["immutable_inputs"]["strongest_common_oof"]["path"],
        columns=["time", "layer", "block", "baseline", "prediction"],
    )
    strongest["time"] = pd.to_datetime(strongest["time"], utc=True)
    aligned = scored[["time", "layer", "fold"]].merge(
        strongest,
        left_on=["time", "layer", "fold"],
        right_on=["time", "layer", "block"],
        how="left",
        validate="one_to_one",
    )
    if aligned["prediction"].isna().any():
        raise ContractError("strongest common OOF does not cover the three-fold surface")
    historical_axis = aligned["prediction"].to_numpy() - aligned["baseline"].to_numpy()
    axis = {
        "oas_cosine": vector_cosine(correction, oas_axis),
        "oas_orthogonal_share": orthogonal_share(correction, oas_axis),
        "historical_cosine": vector_cosine(correction, historical_axis),
        "historical_orthogonal_share": orthogonal_share(correction, historical_axis),
    }
    correction_rms = float(np.sqrt(np.mean(np.square(correction))))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    active_share = float(np.mean(np.abs(correction) > 1e-12))
    gate = config["gate"]
    fold_deltas = [value["delta_rmse"] for value in metrics["by_fold"].values()]
    layer_deltas = [value["delta_rmse"] for value in metrics["by_layer"].values()]
    checks = {
        "pooled_delta": metrics["aggregate"]["delta_rmse"] <= float(gate["pooled_delta_rmse_max_c"]),
        "bootstrap_ci": bootstrap["ci90_high"] < float(gate["bootstrap_ci90_upper_max_c"]),
        "2024_sep_oct": metrics["by_fold"]["2024_sep_oct"]["delta_rmse"] <= float(gate["2024_sep_oct_delta_rmse_max_c"]),
        "improved_folds": sum(value < 0.0 for value in fold_deltas) >= int(gate["minimum_improved_folds"]),
        "worst_fold": max(fold_deltas) <= float(gate["maximum_worst_fold_regression_c"]),
        "worst_layer": max(layer_deltas) <= float(gate["maximum_layer_regression_c"]),
        "active_share": float(gate["minimum_active_share"]) <= active_share <= float(gate["maximum_active_share"]),
        "correction_rms": float(gate["minimum_correction_rms_c"]) <= correction_rms <= float(gate["maximum_correction_rms_c"]) + 1e-12,
        "correction_p99": correction_p99 <= float(gate["maximum_correction_p99_c"]) + 1e-12,
        "oas_independence": abs(axis["oas_cosine"]) <= float(gate["maximum_absolute_axis_cosine"]) and axis["oas_orthogonal_share"] >= float(gate["minimum_axis_orthogonal_share"]),
        "historical_independence": abs(axis["historical_cosine"]) <= float(gate["maximum_absolute_axis_cosine"]) and axis["historical_orthogonal_share"] >= float(gate["minimum_axis_orthogonal_share"]),
    }
    result = {
        "schema_version": "p2.supervised_rank1_functional_residual.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "GO_LOCAL_ONLY_NO_UPLOAD" if all(checks.values()) else "NO_GO_EXACT_NO_OUTPUT",
        "comparator": "INCUMBENT_PROXY_VALIDATION",
        "comparator_disclosure": config["comparator_disclosure"],
        "metrics": metrics,
        "bootstrap": bootstrap,
        "axis_diagnostics": axis,
        "active_share": active_share,
        "correction_rms_c": correction_rms,
        "correction_p99_c": correction_p99,
        "gate_checks": checks,
        "fit_count": len(config["folds"]),
        "truth_rows_read_after_commitment": int(len(scored)),
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "runtime": {"elapsed_seconds": time.perf_counter() - started, "python": platform.python_version()},
    }
    atomic_json(output_directory / "result.json", result)
    scored.drop(columns="truth").to_parquet(output_directory / "scored_predictions_no_truth.parquet", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.check == args.execute:
        raise SystemExit("choose exactly one of --check or --execute")
    data_dir = args.data_dir.expanduser().resolve()
    config = load_config(data_dir)
    if args.check:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "PASS", "official_rows_read": 0}, ensure_ascii=False, indent=2))
        return
    print(json.dumps(run(config, data_dir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
