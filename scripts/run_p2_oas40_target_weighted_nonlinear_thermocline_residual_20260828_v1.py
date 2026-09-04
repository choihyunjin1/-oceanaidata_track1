"""Run the sealed P2 target-weighted nonlinear thermocline pilot.

The runner only reads historical observations and a frozen historical OOF
anchor.  It never resolves official test/sample/submission files and never
creates a candidate CSV.  Prediction files are committed before truth metrics.
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
from p2_restore.dynamic_sigmoid_profile import (
    SigmoidSpec,
    build_public_features,
    feature_columns,
    fit_latent_ridge,
    fit_parameter_catalog,
    fit_public_profile,
    public_profile_arrays,
    stable_parameter_mask,
)
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (
    predict_forward_seasonal_oas,
)
from p2_restore.p2_oas40_target_weighted_nonlinear_thermocline_residual_20260828_v1 import (
    bounded_fixed_correction,
    evaluate_gate,
    fit_covariate_shift_weights,
    latent_standardized_distance,
    paired_kst_day_bootstrap,
    rmse,
    vector_cosine,
    weighted_quantile,
)
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_oas40_target_weighted_nonlinear_thermocline_residual_20260828_v1"
DEFAULT_CONFIG = REPO / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
DEFAULT_ARTIFACT = REPO / "artifacts" / EXPERIMENT_ID
TARGET_LAYERS = (2, 3, 4)
HIDDEN_START = pd.Timestamp("2025-09-01T00:00:00+09:00").tz_convert("UTC")
HIDDEN_STOP = pd.Timestamp("2025-11-01T00:00:00+09:00").tz_convert("UTC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "predict", "score"), required=True)
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
    return {"commit": commit, "dirty": bool(status), "dirty_entry_count": len(status)}


def read_historical_observations(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    expected = ["station", "year", "layer", "time", "temp", "psal", "depth", "nominal_depth"]
    chunks: list[pd.DataFrame] = []
    hidden_discarded = 0
    total = 0
    for chunk in pd.read_csv(path, dtype={"station": "string", "time": "string"}, chunksize=100_000):
        require(list(chunk.columns) == expected, "observations schema changed")
        total += len(chunk)
        parsed = pd.to_datetime(chunk["time"], utc=True)
        hidden = np.asarray((parsed >= HIDDEN_START) & (parsed < HIDDEN_STOP), dtype=bool)
        hidden_discarded += int(hidden.sum())
        kept = chunk.loc[~hidden].copy()
        kept["time"] = parsed[~hidden]
        chunks.append(kept)
    frame = pd.concat(chunks, ignore_index=True)
    require(not frame.duplicated(["station", "time", "layer"]).any(), "observation keys duplicate")
    require(not np.asarray((frame["time"] >= HIDDEN_START) & (frame["time"] < HIDDEN_STOP)).any(), "hidden interval retained")
    return frame, {
        "source_rows": int(total),
        "historical_rows_retained": int(len(frame)),
        "hidden_interval_rows_discarded_before_feature_or_model_use": int(hidden_discarded),
        "hidden_interval_rows_used": 0,
    }


def block_anchor(anchor_path: Path, block: str, *, include_truth: bool) -> pd.DataFrame:
    columns = ["time", "layer", "current_blend50", "block"]
    if include_truth:
        columns.insert(2, "truth")
    frame = pd.read_parquet(anchor_path, columns=columns, filters=[("block", "==", block)])
    require(len(frame) > 0, f"anchor lacks block {block}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    require(not frame.duplicated(["time", "layer"]).any(), f"anchor keys duplicate: {block}")
    return frame.sort_values(["time", "layer"]).reset_index(drop=True)


def add_target_depth(frame: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    metadata = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["station", "time", "layer", "depth", "nominal_depth"],
    ].copy()
    actual = metadata["depth"].to_numpy(np.float64)
    nominal = metadata["nominal_depth"].to_numpy(np.float64)
    metadata["target_depth"] = np.where(np.isfinite(actual) & (actual > 0.0), actual, nominal)
    metadata = metadata.loc[:, ["station", "time", "layer", "target_depth"]]
    require(not metadata.duplicated(["time", "layer"]).any(), "target metadata duplicate")
    result = frame.merge(metadata, on=["time", "layer"], how="left", validate="one_to_one")
    require(result[["station", "target_depth"]].notna().all().all(), "target metadata missing")
    return result


def mask_validation(observations: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    result = observations.copy()
    selected = result["time"].ge(start) & result["time"].lt(stop) & result["layer"].isin(TARGET_LAYERS)
    result.loc[selected, ["temp", "psal"]] = np.nan
    require(result.loc[selected, ["temp", "psal"]].isna().all().all(), "joint target mask failed")
    return result


def sigmoid_spec(config: dict[str, object]) -> SigmoidSpec:
    values = config["thermocline"]
    return SigmoidSpec(
        center_bounds_m=tuple(map(float, values["center_bounds_m"])),
        width_bounds_m=tuple(map(float, values["width_bounds_m"])),
        center_start_fractions=tuple(map(float, values["deterministic_start_center_fractions"])),
        width_starts_m=tuple(map(float, values["deterministic_start_width_m"])),
        max_nfev=int(values["least_squares_max_nfev"]),
        ftol=float(values["least_squares_ftol"]),
        xtol=float(values["least_squares_xtol"]),
        gtol=float(values["least_squares_gtol"]),
        boundary_fraction=float(values["boundary_fraction"]),
        target_depths_m=tuple(map(float, values["target_nominal_depth_m"])),
    )


def alpha_references(
    *,
    panel: pd.DataFrame,
    endpoints: pd.DataFrame,
    query: pd.DataFrame,
    train_stop: pd.Timestamp,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    anchor = config["oas_anchor"]
    oas, receipts = predict_forward_seasonal_oas(
        panel,
        query,
        train_stop=train_stop,
        season_bin_days=int(anchor["season_bin_days"]),
        season_window_days=float(anchor["season_window_days"]),
        minimum_season_rows=int(anchor["minimum_season_rows"]),
        fallback_nearest_complete_rows=int(anchor["fallback_nearest_complete_rows"]),
    )
    base = query[anchor["base_prediction_column"]].to_numpy(np.float64)
    alpha20 = base + float(anchor["direction_alpha"]) * (oas - base)
    alpha40 = base + float(anchor["reference_alpha"]) * (oas - base)
    alpha20 = project_profiles_vectorized(query, alpha20, endpoints).prediction
    alpha40 = project_profiles_vectorized(query, alpha40, endpoints).prediction
    require(np.isfinite(alpha20).all() and np.isfinite(alpha40).all(), "OAS references non-finite")
    return alpha20, alpha40, receipts


def training_time_mask(
    index: pd.DatetimeIndex,
    block_names: list[str],
    config: dict[str, object],
    train_stop: pd.Timestamp,
) -> np.ndarray:
    allowed = np.zeros(len(index), dtype=bool)
    for block_name in block_names:
        bounds = config["block_bounds"][block_name]
        allowed |= np.asarray((index >= utc(bounds[0])) & (index < utc(bounds[1])), dtype=bool)
    return allowed & np.asarray(index < train_stop, dtype=bool)


def shift_columns(features: pd.DataFrame) -> tuple[str, ...]:
    excluded = {"annual_sin", "annual_cos", "m2_sin", "m2_cos"}
    return tuple(column for column in feature_columns(features) if column not in excluded)


def fit_predict_fold(
    *,
    fold_name: str,
    fold_spec: dict[str, object],
    config: dict[str, object],
    observations: pd.DataFrame,
    anchor_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    start, stop = utc(fold_spec["start"]), utc(fold_spec["stop"])
    purge = pd.Timedelta(days=int(config["validation"]["purge_days"]))
    train_stop = start - purge
    masked = mask_validation(observations, start, stop)
    public_features = build_public_features(
        masked, change_hours=tuple(map(int, config["thermocline"]["recent_change_hours"]))
    )
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)

    query = add_target_depth(block_anchor(anchor_path, fold_name, include_truth=False), observations)
    require(query["time"].ge(start).all() and query["time"].lt(stop).all(), "fold bounds differ")
    alpha20, reference, oas_receipts = alpha_references(
        panel=panel,
        endpoints=endpoints,
        query=query,
        train_stop=train_stop,
        config=config,
    )

    spec = sigmoid_spec(config)
    def allowed(index: pd.Index) -> np.ndarray:
        return training_time_mask(
            pd.DatetimeIndex(index),
            list(fold_spec["parameter_training_blocks"]),
            config,
            train_stop,
        )
    catalog = fit_parameter_catalog(
        masked,
        spec=spec,
        allowed_time=allowed,
        stride_minutes=int(config["thermocline"]["profile_fit_stride_minutes"]),
        minimum_points=int(config["thermocline"]["minimum_full_profile_points"]),
    )
    thermo = config["thermocline"]
    stable = stable_parameter_mask(
        catalog,
        minimum_abs_amplitude_c=float(thermo["minimum_abs_amplitude_c"]),
        minimum_r2=float(thermo["minimum_profile_r2"]),
        maximum_condition=float(thermo["maximum_scaled_jacobian_condition"]),
        maximum_spread_c=float(thermo["maximum_multistart_target_spread_c"]),
    )
    source_catalog = catalog.loc[stable].reset_index(drop=True)
    require(len(source_catalog) >= int(thermo["minimum_latent_training_rows"]), "too few stable profiles")
    source_time = pd.DatetimeIndex(source_catalog["time"])
    query_times = pd.DatetimeIndex(query["time"].drop_duplicates().sort_values())
    require(source_time.isin(public_features.index).all(), "source public features missing")
    require(query_times.isin(public_features.index).all(), "query public features missing")
    source_features = public_features.loc[source_time].reset_index(drop=True)
    query_features = public_features.loc[query_times].reset_index(drop=True)
    ratio_config = config["target_shift"]
    weights = fit_covariate_shift_weights(
        source_features,
        query_features,
        source_time=source_time,
        query_time=query_times,
        columns=shift_columns(public_features),
        logistic_c=float(ratio_config["logistic_c"]),
        max_iter=int(ratio_config["logistic_max_iter"]),
        seed=int(ratio_config["seed"]),
        weight_clip=tuple(map(float, ratio_config["weight_clip"])),
        minimum_effective_sample_fraction=float(ratio_config["minimum_effective_sample_fraction"]),
        maximum_auc=float(ratio_config["maximum_cross_day_discriminator_auc"]),
    )

    manifold = reference.copy()
    row_support = np.zeros(len(query), dtype=bool)
    latent_receipt: dict[str, object] = {"fit_performed": False}
    if weights.overlap_passed:
        weighted_catalog = source_catalog.copy()
        weighted_catalog["sample_weight"] = (
            weighted_catalog["sample_weight"].to_numpy(np.float64) * weights.source_weights
        )
        model_columns = feature_columns(public_features)
        latent = fit_latent_ridge(
            public_features,
            weighted_catalog,
            columns=model_columns,
            alpha=float(thermo["latent_ridge_alpha"]),
            minimum_feature_coverage=float(thermo["minimum_feature_coverage"]),
            minimum_rows=int(thermo["minimum_latent_training_rows"]),
            center_bounds_m=tuple(map(float, thermo["center_bounds_m"])),
            width_bounds_m=tuple(map(float, thermo["width_bounds_m"])),
        )
        latent_source_features = public_features.loc[source_time].reset_index(drop=True)
        source_distance = latent_standardized_distance(latent, latent_source_features)
        query_distance = latent_standardized_distance(latent, query_features)
        distance_threshold = weighted_quantile(
            source_distance,
            weights.source_weights,
            float(thermo["latent_support_weighted_quantile"]),
        )
        latent_prediction = latent.predict(query_features)
        feature_lookup = {time: position for position, time in enumerate(query_times)}
        for _, group in query.groupby("time", sort=False):
            if len(group) != 3 or set(group["layer"].astype(int)) != set(TARGET_LAYERS):
                continue
            timestamp = pd.Timestamp(group["time"].iloc[0])
            position = feature_lookup[timestamp]
            if query_distance[position] > distance_threshold:
                continue
            feature_row = query_features.iloc[position]
            depth, temperature = public_profile_arrays(feature_row)
            ordered = group.sort_values("layer")
            fit = fit_public_profile(
                depth,
                temperature,
                ordered["target_depth"].to_numpy(np.float64),
                center_m=float(latent_prediction[position, 0]),
                log_width=float(latent_prediction[position, 1]),
                minimum_points=int(thermo["minimum_public_points"]),
                minimum_depth_span_m=float(thermo["minimum_public_depth_span_m"]),
                center_step_m=float(thermo["finite_difference_center_step_m"]),
                log_width_step=float(thermo["finite_difference_log_width_step"]),
                condition_max=float(thermo["maximum_scaled_jacobian_condition"]),
            )
            if not fit.supported or not fit.observable or abs(fit.amplitude_c) < float(thermo["minimum_abs_amplitude_c"]):
                continue
            rows = ordered.index.to_numpy(int)
            if not np.isfinite(fit.target_prediction).all():
                continue
            manifold[rows] = fit.target_prediction
            row_support[rows] = True
        latent_receipt = {
            "fit_performed": True,
            "selected_feature_count": len(latent.feature_columns),
            "stable_catalog_rows": int(len(source_catalog)),
            "catalog_rows": int(len(catalog)),
            "latent_distance_threshold": float(distance_threshold),
            "query_distance_median": float(np.median(query_distance)),
            "query_distance_p90": float(np.quantile(query_distance, 0.90)),
        }

    correction_config = config["correction"]
    raw = float(correction_config["fixed_blend"]) * (manifold - reference)
    gate_frame = pd.DataFrame({"time": query["time"], "supported": row_support, "raw": raw})
    profile_support = (
        gate_frame.groupby("time", sort=False)["supported"]
        .transform("all")
        .to_numpy(bool)
        .copy()
    )
    profile_max = gate_frame.groupby("time", sort=False)["raw"].transform(lambda values: float(np.max(np.abs(values)))).to_numpy(float)
    profile_support &= profile_max >= float(correction_config["minimum_profile_raw_max_absolute_c"])
    profile_support &= profile_max <= float(correction_config["maximum_profile_raw_max_absolute_c"])
    proposed = reference + raw
    projected = project_profiles_vectorized(query, proposed, endpoints)
    envelope_ok = np.isclose(projected.prediction, proposed, rtol=0.0, atol=1e-12)
    envelope_frame = pd.DataFrame({"time": query["time"], "ok": envelope_ok})
    profile_support &= envelope_frame.groupby("time", sort=False)["ok"].transform("all").to_numpy(bool)
    correction, correction_receipt = bounded_fixed_correction(
        raw,
        profile_support,
        rms_cap=float(correction_config["maximum_rms_c"]),
        p99_cap=float(correction_config["maximum_p99_absolute_c"]),
    )
    candidate = reference + correction
    require(np.array_equal(candidate[~profile_support], reference[~profile_support]), "fallback differs")
    output = query.loc[:, ["time", "layer"]].copy()
    output["alpha20"] = alpha20
    output["reference"] = reference
    output["candidate"] = candidate
    output["correction"] = correction
    output["enabled"] = profile_support
    receipt = {
        "fold": fold_name,
        "validation_start_utc": start.isoformat(),
        "validation_stop_utc": stop.isoformat(),
        "training_stop_with_7d_purge_utc": train_stop.isoformat(),
        "validation_truth_column_loaded": False,
        "target_temp_psal_masked_together": True,
        "hidden_interval_values_used": False,
        "parameter_training_blocks": list(fold_spec["parameter_training_blocks"]),
        "oas": oas_receipts,
        "shift": weights.receipt(),
        "latent": latent_receipt,
        "correction": correction_receipt,
    }
    return output, receipt


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    times = pd.DatetimeIndex(frame["time"])
    np.savez_compressed(
        path,
        time_ns=times.as_unit("ns").asi8.astype(np.int64),
        layer=frame["layer"].to_numpy(np.int16),
        alpha20=frame["alpha20"].to_numpy(np.float64),
        reference=frame["reference"].to_numpy(np.float64),
        candidate=frame["candidate"].to_numpy(np.float64),
        correction=frame["correction"].to_numpy(np.float64),
        enabled=frame["enabled"].to_numpy(bool),
    )


def decode_time(values: np.ndarray) -> pd.DatetimeIndex:
    decoded = pd.DatetimeIndex(pd.to_datetime(np.asarray(values, dtype=np.int64), unit="ns", utc=True))
    require(decoded.min().year >= 2024 and decoded.max().year <= 2025, "committed time invalid")
    return decoded


def verify_pins(config_path: Path, data_dir: Path, config: dict[str, object]) -> dict[str, object]:
    observations = data_dir / "observations.csv"
    pins = config["input_pins"]
    checks = {
        "observations": sha256(observations) == pins["observations_sha256"],
        "dynamic_sigmoid_dependency": sha256(REPO / "src/p2_restore/dynamic_sigmoid_profile.py") == pins["dynamic_sigmoid_dependency_sha256"],
        "oas_dependency": sha256(REPO / "src/p2_restore/p2_alpha40_quasiperiodic_gp_residual_20260828_v1.py") == pins["oas_dependency_sha256"],
        "profile_projection_dependency": sha256(REPO / "src/p2_restore/profile_projection.py") == pins["profile_projection_dependency_sha256"],
        "validation_anchor": sha256(REPO / config["oas_anchor"]["validation_anchor_path"]) == config["oas_anchor"]["validation_anchor_sha256"],
    }
    require(all(checks.values()), f"input pin failure: {checks}")
    return {"passed": True, "checks": checks, "config_sha256": sha256(config_path)}


def preflight_stage(config_path: Path, data_dir: Path, config: dict[str, object]) -> None:
    pins = verify_pins(config_path, data_dir, config)
    require(config["experiment_id"] == EXPERIMENT_ID, "experiment ID differs")
    require(config["authorization"]["execution_count"] == 1, "execution count differs")
    require(config["authorization"]["hyperparameter_search_count"] == 0, "search count differs")
    leakage = config["leakage_contract"]
    require(leakage["official_test_index_sample_submission_paths_read"] is False, "official access flag")
    require(leakage["candidate_csv_generation_authorized"] is False, "CSV authorization flag")
    anchor = REPO / config["oas_anchor"]["validation_anchor_path"]
    schema = pd.read_parquet(anchor, columns=["time", "layer", "current_blend50", "block"]).columns.tolist()
    print(json.dumps({"stage": "preflight", "passed": True, "pins": pins, "anchor_columns": schema}, indent=2))


def prediction_stage(
    *, config_path: Path, artifact_dir: Path, data_dir: Path, config: dict[str, object]
) -> None:
    require(not any(artifact_dir.rglob("*")) if artifact_dir.exists() else True, "artifact is not empty")
    pins = verify_pins(config_path, data_dir, config)
    observations, observation_audit = read_historical_observations(data_dir / "observations.csv")
    anchor_path = REPO / config["oas_anchor"]["validation_anchor_path"]
    artifact_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir = artifact_dir / "predictions"
    predictions_dir.mkdir()
    outputs: dict[str, object] = {}
    receipts: dict[str, object] = {}
    for fold_name, fold_spec in config["folds"].items():
        prediction, receipt = fit_predict_fold(
            fold_name=fold_name,
            fold_spec=fold_spec,
            config=config,
            observations=observations,
            anchor_path=anchor_path,
        )
        output_path = predictions_dir / f"{fold_name}.npz"
        write_prediction(output_path, prediction)
        outputs[fold_name] = {
            "path": str(output_path.relative_to(REPO)),
            "rows": int(len(prediction)),
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        }
        receipts[fold_name] = receipt
    commitment = {
        "schema_version": "p2.oas40_target_weighted_nonlinear_thermocline_residual.prediction_commitment.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_count": 1,
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "truth_metric_computed": False,
        "validation_truth_column_loaded": False,
        "official_paths_read": False,
        "candidate_csv_generated": False,
        "hidden_interval_values_used": False,
        "hyperparameter_search_count": 0,
        "preflight": pins,
        "observation_audit": observation_audit,
        "fold_receipts": receipts,
        "prediction_outputs": outputs,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "cpu_thread_cap": 2,
            "git": git_snapshot(),
        },
    }
    path = artifact_dir / "prediction_commitment.json"
    path.write_text(json.dumps(commitment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "predict", "commitment": str(path.relative_to(REPO)), "outputs": outputs}, indent=2))


def metric_group(frame: pd.DataFrame) -> dict[str, object]:
    reference = rmse(frame["truth"].to_numpy(), frame["reference"].to_numpy())
    candidate = rmse(frame["truth"].to_numpy(), frame["candidate"].to_numpy())
    return {"rows": int(len(frame)), "alpha40_reference_rmse_c": reference, "candidate_rmse_c": candidate, "delta_rmse_c": candidate - reference}


def report_markdown(result: dict[str, object]) -> str:
    aggregate = result["metrics"]["aggregate"]
    correction = result["correction"]
    bootstrap = result["paired_kst_day_bootstrap"]
    return "\n".join(
        [
            f"# {EXPERIMENT_ID}",
            "",
            f"- decision: `{result['decision']}`",
            f"- historical rows: `{aggregate['rows']}`",
            f"- alpha40 local proxy RMSE: `{aggregate['alpha40_reference_rmse_c']:.10f}` C",
            f"- candidate RMSE: `{aggregate['candidate_rmse_c']:.10f}` C",
            f"- delta RMSE: `{aggregate['delta_rmse_c']:+.10f}` C",
            f"- KST-day bootstrap CI90: `[{bootstrap['ci90_low_c']:+.10f}, {bootstrap['ci90_high_c']:+.10f}]` C",
            f"- active share: `{correction['enabled_fraction']:.6%}`",
            f"- correction RMS / p99: `{correction['rms_c']:.10f}` / `{correction['p99_absolute_c']:.10f}` C",
            f"- |cosine(new direction, alpha20->alpha40)|: `{abs(result['alpha_direction_cosine']):.10f}`",
            "- official/test_index/sample/submission/hidden-period values used: `0`",
            "- candidate CSV/upload: `0`",
            "- validation claim: exposed historical three-block bounded pilot; not untouched holdout",
            "",
        ]
    )


def score_stage(config_path: Path, artifact_dir: Path, config: dict[str, object]) -> None:
    commitment_path = artifact_dir / "prediction_commitment.json"
    result_path = artifact_dir / "result.json"
    require(commitment_path.is_file(), "prediction commitment missing")
    require(not result_path.exists(), "one-shot result already exists")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    require(commitment["truth_metric_computed"] is False, "truth already computed")
    require(commitment["validation_truth_column_loaded"] is False, "truth loaded before seal")
    anchor_path = REPO / config["oas_anchor"]["validation_anchor_path"]
    scored_parts: list[pd.DataFrame] = []
    for fold_name, output in commitment["prediction_outputs"].items():
        prediction_path = REPO / output["path"]
        require(sha256(prediction_path) == output["sha256"], f"prediction hash changed: {fold_name}")
        with np.load(prediction_path, allow_pickle=False) as payload:
            prediction = pd.DataFrame(
                {
                    "time": decode_time(payload["time_ns"]),
                    "layer": payload["layer"].astype(int),
                    "alpha20": payload["alpha20"].astype(float),
                    "reference": payload["reference"].astype(float),
                    "candidate": payload["candidate"].astype(float),
                    "correction": payload["correction"].astype(float),
                    "enabled": payload["enabled"].astype(bool),
                }
            )
        truth = block_anchor(anchor_path, fold_name, include_truth=True).loc[:, ["time", "layer", "truth"]]
        scored = prediction.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        require(len(scored) == len(prediction) and scored["truth"].notna().all(), "truth alignment failed")
        scored["fold"] = fold_name
        scored_parts.append(scored)
    scored = pd.concat(scored_parts, ignore_index=True)
    aggregate = metric_group(scored)
    by_fold = {str(name): metric_group(group) for name, group in scored.groupby("fold", sort=True)}
    by_layer = {str(int(name)): metric_group(group) for name, group in scored.groupby("layer", sort=True)}
    gate_config = config["gate"]
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(gate_config["bootstrap_replicates"]),
        seed=int(gate_config["bootstrap_seed"]),
    )
    correction = scored["correction"].to_numpy(np.float64)
    enabled = scored["enabled"].to_numpy(bool)
    correction_metrics = {
        "enabled_rows": int(enabled.sum()),
        "enabled_fraction": float(enabled.mean()),
        "rms_c": float(np.sqrt(np.mean(np.square(correction)))),
        "p99_absolute_c": float(np.quantile(np.abs(correction), 0.99)),
        "maximum_absolute_c": float(np.max(np.abs(correction))),
        "fallback_maximum_absolute_c": float(np.max(np.abs(correction[~enabled]))) if (~enabled).any() else 0.0,
    }
    require(correction_metrics["fallback_maximum_absolute_c"] == 0.0, "fallback changed")
    cosine = vector_cosine(correction, scored["reference"].to_numpy() - scored["alpha20"].to_numpy())
    all_shift_passed = all(
        bool(value["shift"]["overlap_passed"]) for value in commitment["fold_receipts"].values()
    )
    gate = evaluate_gate(
        aggregate_delta=float(aggregate["delta_rmse_c"]),
        ci90_high=float(bootstrap["ci90_high_c"]),
        fold_deltas={name: float(value["delta_rmse_c"]) for name, value in by_fold.items()},
        layer_deltas={name: float(value["delta_rmse_c"]) for name, value in by_layer.items()},
        active_share=float(correction_metrics["enabled_fraction"]),
        correction_rms=float(correction_metrics["rms_c"]),
        correction_p99=float(correction_metrics["p99_absolute_c"]),
        cosine=cosine,
        all_shift_folds_passed=all_shift_passed,
        thresholds=gate_config,
    )
    decision = "ORTHOGONAL_PROBE_READY_NO_CSV_NO_UPLOAD" if gate["passed"] else "NO_GO_KEEP_OAS40_NO_CSV_NO_RESEARCH_LOOP"
    result = {
        "schema_version": "p2.oas40_target_weighted_nonlinear_thermocline_residual.result.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "execution_count": 1,
        "decision": decision,
        "historical_exposure": config["historical_exposure"],
        "reference": "prefix-fit seasonal OAS alpha40 local proxy with frozen current_blend50 base and existing endpoint/PAVA projection",
        "metrics": {"aggregate": aggregate, "by_fold": by_fold, "by_layer": by_layer},
        "paired_kst_day_bootstrap": bootstrap,
        "correction": correction_metrics,
        "alpha_direction_cosine": cosine,
        "shift_by_fold": {name: value["shift"] for name, value in commitment["fold_receipts"].items()},
        "gate": gate,
        "prediction_commitment": {"path": str(commitment_path.relative_to(REPO)), "sha256": sha256(commitment_path), "verified_before_truth_load": True},
        "leakage_audit": {
            "official_test_index_sample_submission_paths_read": False,
            "official_answer_or_mirror_read": False,
            "hidden_interval_values_used": False,
            "candidate_csv_generated": False,
            "official_upload_performed": False,
            "target_temp_psal_masked_together": True,
            "target_temp_psal_used_as_input_or_state_update": False,
            "truth_loaded_only_after_prediction_hash_verification": True,
            "post_result_parameter_search_performed": False,
        },
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = artifact_dir / "report.md"
    report_path.write_text(report_markdown(result), encoding="utf-8")
    source_paths = [
        config_path,
        REPO / "src/p2_restore" / f"{EXPERIMENT_ID}.py",
        Path(__file__).resolve(),
        REPO / "tests" / f"test_{EXPERIMENT_ID}.py",
        REPO / "scripts" / f"qa_{EXPERIMENT_ID}.py",
    ]
    manifest = {
        "schema_version": "p2.oas40_target_weighted_nonlinear_thermocline_residual.manifest.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "sources": {str(path.relative_to(REPO)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in source_paths},
        "inputs": {
            "observations": {"source": "P2_DATA_DIR/observations.csv", "sha256": config["input_pins"]["observations_sha256"]},
            "validation_anchor": {"path": config["oas_anchor"]["validation_anchor_path"], "sha256": config["oas_anchor"]["validation_anchor_sha256"]},
        },
        "outputs": {
            "prediction_commitment": {"path": str(commitment_path.relative_to(REPO)), "sha256": sha256(commitment_path)},
            "predictions": commitment["prediction_outputs"],
            "result": {"path": str(result_path.relative_to(REPO)), "sha256": sha256(result_path)},
            "report": {"path": str(report_path.relative_to(REPO)), "sha256": sha256(report_path)},
            "candidate_csv": None,
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "score", "decision": decision, "metrics": result["metrics"], "gate": gate}, indent=2))


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["experiment_id"] == EXPERIMENT_ID, "config experiment differs")
    require(config["authorization"]["hyperparameter_search_count"] == 0, "search is forbidden")
    if args.stage == "preflight":
        preflight_stage(config_path, resolve_data_dir(args.data_dir), config)
        return
    require(args.execute, "predict/score requires --execute")
    if args.stage == "predict":
        prediction_stage(
            config_path=config_path,
            artifact_dir=artifact_dir,
            data_dir=resolve_data_dir(args.data_dir),
            config=config,
        )
    else:
        score_stage(config_path, artifact_dir, config)


if __name__ == "__main__":
    main()
