"""Run the one-shot leakage-safe nested PLS capacity grid for P2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Set the process-wide ceiling before importing NumPy/SciPy-backed libraries.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _directory in (ROOT, SRC):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from p2_restore.depth_registered_cmfpca import build_layer_identity_panel  # noqa: E402
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (  # noqa: E402
    bounded_profile_correction,
    paired_kst_day_bootstrap,
)
from p2_restore.p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2 import (  # noqa: E402
    contiguous_time_groups,
    time_group_sha256,
)
from p2_restore.p2_nested_pls_capacity_grid_20260829_v1 import (  # noqa: E402
    CapacityPoint,
    FittedPLSResidual,
    capacity_grid,
    select_inner_point,
)
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.supervised_rank1_functional_residual import (  # noqa: E402
    TARGET_LAYERS,
    build_public_functional_features,
    vector_cosine,
)
from p2_restore.trainonly_regime_veto import season_bin  # noqa: E402
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1 as base,
)

EXPERIMENT_ID = "p2_nested_pls_capacity_grid_20260829_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reference_config(config: dict[str, Any], *, spline_ridge: float) -> dict[str, Any]:
    reference = config["reference"]
    return {
        "model": {
            "reference_alpha": float(reference["alpha"]),
            "season_bin_days": int(reference["season_bin_days"]),
            "season_window_days": float(reference["season_window_days"]),
            "minimum_season_rows": int(reference["minimum_season_rows"]),
            "fallback_nearest_complete_rows": int(
                reference["fallback_nearest_complete_rows"]
            ),
            "spline_ridge": float(spline_ridge),
            "change_hours": list(reference["change_hours"]),
        }
    }


def load_config(data_dir: Path) -> tuple[dict[str, Any], tuple[CapacityPoint, ...]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise base.ContractError("experiment ID drifted")
    policy = config["execution_policy"]
    if any(
        (
            policy["official_hidden_gap_values_read_allowed"],
            policy["official_test_sample_submission_read_allowed"],
            policy["submission_csv_generation_allowed"],
            policy["official_upload_authorized"],
            policy["result_based_retry"],
        )
    ) or int(policy["maximum_executions"]) != 1:
        raise base.ContractError("one-shot or forbidden-access policy drifted")
    resource = config["resource_contract"]
    if int(resource["logical_cpus_available"]) != 8:
        raise base.ContractError("logical CPU contract drifted")
    if int(resource["maximum_total_threads"]) > 4 or int(resource["blas_threads"]) > 4:
        raise base.ContractError("thread ceiling exceeds four")
    if int(resource["joblib_processes"]) != 1:
        raise base.ContractError("joblib process count must remain one")
    if config["nested_selection"]["rotations"] != [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        raise base.ContractError("three-way rotation drifted")
    if config["nested_selection"]["enabled_season_bins"] != [17, 18]:
        raise base.ContractError("enabled season bins drifted")
    if float(config["reference"]["alpha"]) != 0.5:
        raise base.ContractError("alpha50 anchor drifted")
    if float(config["nested_selection"]["correction_p99_cap_c"]) != 0.2:
        raise base.ContractError("p99 cap drifted")
    points = capacity_grid(config["capacity_grid"])
    if len(config["folds"]) != 3:
        raise base.ContractError("outer fold count drifted")
    for record in config["immutable_inputs"].values():
        path = ROOT / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or base.sha256_file(path) != record["sha256"]
        ):
            raise base.ContractError(f"immutable input changed: {path}")
    observations = data_dir / config["source_observations"]["filename"]
    if (
        not observations.is_file()
        or base.sha256_file(observations) != config["source_observations"]["sha256"]
    ):
        raise base.ContractError("observations.csv pin changed")
    return config, points


def _training_anchor_keys(anchor_path: Path, blocks: list[str]) -> pd.DataFrame:
    frame = pd.concat(
        [base.block_anchor(anchor_path, block, include_truth=False)[["time", "layer"]] for block in blocks],
        ignore_index=True,
    )
    counts = frame.groupby("time", sort=True)["layer"].nunique()
    complete = pd.DatetimeIndex(counts[counts == len(TARGET_LAYERS)].index)
    if complete.empty:
        raise base.ContractError("outer training has no complete target profiles")
    return frame.loc[frame["time"].isin(complete)].sort_values(["time", "layer"])


def _group_anchor(
    anchor_path: Path,
    blocks: list[str],
    group_times: pd.DatetimeIndex,
    *,
    include_truth: bool,
) -> pd.DataFrame:
    frame = pd.concat(
        [base.block_anchor(anchor_path, block, include_truth=include_truth) for block in blocks],
        ignore_index=True,
    )
    result = frame.loc[frame["time"].isin(group_times)].sort_values(["time", "layer"])
    if result.empty or result["time"].nunique() != len(group_times):
        raise base.ContractError("cross-fit group anchor coverage failed")
    return result.reset_index(drop=True)


def _prepare_inner_rotation(
    *,
    outer_name: str,
    outer_spec: dict[str, Any],
    held_group: int,
    fit_group: int,
    support_group: int,
    groups: tuple[pd.DatetimeIndex, ...],
    config: dict[str, Any],
    observations: pd.DataFrame,
    anchor_path: Path,
) -> dict[str, Any]:
    outer_start = base.utc(outer_spec["start"])
    blocks = list(outer_spec["training_blocks"])
    held_times, fit_times, support_times = (
        groups[held_group],
        groups[fit_group],
        groups[support_group],
    )
    if (
        set(held_times.asi8) & set(fit_times.asi8)
        or set(held_times.asi8) & set(support_times.asi8)
        or set(fit_times.asi8) & set(support_times.asi8)
    ):
        raise base.ContractError("inner roles overlap")
    masked = observations.copy()
    target = masked["layer"].isin(TARGET_LAYERS)
    support = masked["time"].isin(support_times)
    masked.loc[target & ~support, ["temp", "psal"]] = np.nan
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)
    model_config = _reference_config(config, spline_ridge=0.001)

    held = base.add_metadata(
        _group_anchor(anchor_path, blocks, held_times, include_truth=False), observations
    )
    held_reference, _ = base.alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=held,
        train_stop=outer_start,
        config=model_config,
    )
    held["reference"] = held_reference
    training = base.add_metadata(
        _group_anchor(anchor_path, blocks, fit_times, include_truth=True), observations
    )
    if not training["time"].lt(outer_start).all():
        raise base.ContractError("inner correction labels cross outer boundary")
    training_reference, _ = base.alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=training,
        train_stop=outer_start,
        config=model_config,
    )
    training["residual"] = training["truth"].to_numpy(dtype=np.float64) - training_reference
    train_times, response = base.profile_response(training)

    # H labels are bound only after reference and correction-fit inputs are frozen.
    truth = _group_anchor(anchor_path, blocks, held_times, include_truth=True)[
        ["time", "layer", "truth"]
    ]
    held = held.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
    if held["truth"].isna().any():
        raise base.ContractError("inner held truth binding failed")
    return {
        "outer_name": outer_name,
        "held_group": held_group,
        "fit_group": fit_group,
        "support_group": support_group,
        "held": held,
        "endpoints": endpoints,
        "train_times": train_times,
        "response": response,
    }


def _row_raw_prediction(
    *,
    fitted: FittedPLSResidual,
    query: pd.DataFrame,
    query_features: pd.DataFrame,
    query_times: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile_prediction, leverage = fitted.predict_raw(query_features, query_times)
    profile_valid = query_features["public_profile_valid"].to_numpy(dtype=bool)
    layer_to_column = {layer: index for index, layer in enumerate(TARGET_LAYERS)}
    time_to_row = {timestamp: index for index, timestamp in enumerate(query_times)}
    raw = np.zeros(len(query), dtype=np.float64)
    row_leverage = np.full(len(query), np.inf, dtype=np.float64)
    row_valid = np.zeros(len(query), dtype=bool)
    for row, values in enumerate(query.itertuples(index=False)):
        profile_row = time_to_row[pd.Timestamp(values.time)]
        raw[row] = profile_prediction[profile_row, layer_to_column[int(values.layer)]]
        row_leverage[row] = leverage[profile_row]
        row_valid[row] = profile_valid[profile_row]
    return raw, row_leverage, row_valid


def _empty_inner_stats(points: tuple[CapacityPoint, ...]) -> dict[str, dict[str, Any]]:
    return {
        point.point_id: {
            "point": asdict(point),
            "rows": 0,
            "candidate_sse": 0.0,
            "reference_sse": 0.0,
            "groups": {},
            "layers": {},
            "correction_max_abs_c": 0.0,
        }
        for point in points
    }


def _update_inner_stats(
    stats: dict[str, Any],
    *,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    correction: np.ndarray,
    held_group: int,
    layers: np.ndarray,
) -> None:
    candidate_error = np.square(candidate - truth)
    reference_error = np.square(reference - truth)
    stats["rows"] += len(truth)
    stats["candidate_sse"] += float(candidate_error.sum())
    stats["reference_sse"] += float(reference_error.sum())
    group = stats["groups"].setdefault(str(held_group), [0, 0.0, 0.0])
    group[0] += len(truth)
    group[1] += float(candidate_error.sum())
    group[2] += float(reference_error.sum())
    for layer in TARGET_LAYERS:
        keep = layers == layer
        record = stats["layers"].setdefault(str(layer), [0, 0.0, 0.0])
        record[0] += int(keep.sum())
        record[1] += float(candidate_error[keep].sum())
        record[2] += float(reference_error[keep].sum())
    stats["correction_max_abs_c"] = max(
        float(stats["correction_max_abs_c"]),
        float(np.max(np.abs(correction), initial=0.0)),
    )


def _delta_from_accumulator(record: list[float]) -> float:
    rows, candidate_sse, reference_sse = record
    return float(np.sqrt(candidate_sse / rows) - np.sqrt(reference_sse / rows))


def _finalize_inner_records(stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for point_id, value in stats.items():
        rows = int(value["rows"])
        candidate_rmse = float(np.sqrt(value["candidate_sse"] / rows))
        reference_rmse = float(np.sqrt(value["reference_sse"] / rows))
        records.append(
            {
                "point_id": point_id,
                "point": value["point"],
                "rows": rows,
                "reference_rmse": reference_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": candidate_rmse - reference_rmse,
                "worst_group_delta_rmse": max(
                    _delta_from_accumulator(record) for record in value["groups"].values()
                ),
                "worst_layer_delta_rmse": max(
                    _delta_from_accumulator(record) for record in value["layers"].values()
                ),
                # A maximum-absolute bound is deliberately supplied to the inner p99 gate.
                "correction_p99_c": float(value["correction_max_abs_c"]),
                "inner_correction_bound_is_max_abs": True,
            }
        )
    return records


def _nested_select_outer(
    *,
    outer_name: str,
    outer_spec: dict[str, Any],
    config: dict[str, Any],
    points: tuple[CapacityPoint, ...],
    observations: pd.DataFrame,
    feature_cache: dict[float, pd.DataFrame],
    anchor_path: Path,
    fit_counter: dict[str, int],
    evaluation_counter: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    keys = _training_anchor_keys(anchor_path, list(outer_spec["training_blocks"]))
    nested = config["nested_selection"]
    groups = contiguous_time_groups(
        keys["time"],
        groups=int(nested["time_groups"]),
        minimum_profiles=int(nested["minimum_group_profiles"]),
    )
    stats = _empty_inner_stats(points)
    group_receipts = [
        {
            "group": index,
            "profiles": len(group),
            "start": group[0].isoformat(),
            "stop_inclusive": group[-1].isoformat(),
            "time_sha256": time_group_sha256(group),
        }
        for index, group in enumerate(groups)
    ]
    for held_group, fit_group, support_group in nested["rotations"]:
        rotation = _prepare_inner_rotation(
            outer_name=outer_name,
            outer_spec=outer_spec,
            held_group=int(held_group),
            fit_group=int(fit_group),
            support_group=int(support_group),
            groups=groups,
            config=config,
            observations=observations,
            anchor_path=anchor_path,
        )
        query = rotation["held"]
        query_times = pd.DatetimeIndex(sorted(query["time"].unique()))
        reference = query["reference"].to_numpy(dtype=np.float64)
        truth = query["truth"].to_numpy(dtype=np.float64)
        layers = query["layer"].to_numpy(dtype=int)
        in_enabled_bin = np.isin(
            season_bin(query["time"], int(config["reference"]["season_bin_days"])),
            np.asarray(nested["enabled_season_bins"], dtype=int),
        )
        for ridge in map(float, config["capacity_grid"]["spline_ridge"]):
            train_features = base.align_features(feature_cache[ridge], rotation["train_times"])
            valid_train = train_features["public_profile_valid"].to_numpy(dtype=bool)
            train_features = train_features.loc[valid_train].reset_index(drop=True)
            response = rotation["response"][valid_train]
            train_times = rotation["train_times"][valid_train]
            query_features = base.align_features(feature_cache[ridge], query_times)
            for rank in map(int, config["capacity_grid"]["rank"]):
                fitted = FittedPLSResidual.fit(
                    train_features,
                    response,
                    train_times,
                    rank=rank,
                )
                fit_counter["inner_pls_fits"] += 1
                raw, row_leverage, row_valid = _row_raw_prediction(
                    fitted=fitted,
                    query=query,
                    query_features=query_features,
                    query_times=query_times,
                )
                for quantile in map(float, config["capacity_grid"]["leverage_quantile"]):
                    enabled = (
                        row_valid
                        & np.isfinite(row_leverage)
                        & (row_leverage <= fitted.leverage_limit(quantile))
                        & in_enabled_bin
                    )
                    for cap in map(float, config["capacity_grid"]["rms_cap_c"]):
                        bounded, _ = bounded_profile_correction(
                            raw,
                            enabled,
                            rms_cap=cap,
                            p99_cap=float(nested["correction_p99_cap_c"]),
                        )
                        for strength in map(float, config["capacity_grid"]["strength"]):
                            point = CapacityPoint(rank, ridge, quantile, cap, strength)
                            projected = project_profiles_vectorized(
                                query,
                                reference + bounded * strength,
                                rotation["endpoints"],
                            ).prediction
                            correction = projected - reference
                            if np.max(np.abs(correction[~enabled]), initial=0.0) > 1e-12:
                                raise base.ContractError("inner disabled profile changed")
                            _update_inner_stats(
                                stats[point.point_id],
                                truth=truth,
                                reference=reference,
                                candidate=projected,
                                correction=correction,
                                held_group=int(held_group),
                                layers=layers,
                            )
                            evaluation_counter["rotation_point_evaluations"] += 1
    records = _finalize_inner_records(stats)
    selected = select_inner_point(records)
    evaluation_counter["outer_grid_points_evaluated"] += len(records)
    receipt = {
        "groups": group_receipts,
        "rotations": nested["rotations"],
        "grid_points": len(records),
        "records_sha256": _canonical_sha256(records),
        "outer_truth_rows_read_before_selection": 0,
    }
    return selected, records, receipt


def _fit_outer_prediction(
    *,
    outer_name: str,
    outer_spec: dict[str, Any],
    point: CapacityPoint,
    config: dict[str, Any],
    observations: pd.DataFrame,
    functional_features: pd.DataFrame,
    anchor_path: Path,
    fit_counter: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start, stop = base.utc(outer_spec["start"]), base.utc(outer_spec["stop"])
    masked = observations.copy()
    validation_mask = (
        masked["time"].ge(start)
        & masked["time"].lt(stop)
        & masked["layer"].isin(TARGET_LAYERS)
    )
    masked.loc[validation_mask, ["temp", "psal"]] = np.nan
    panel, _, _ = build_layer_identity_panel(masked)
    endpoints = public_endpoint_frame(masked)
    query = base.add_metadata(
        base.block_anchor(anchor_path, outer_name, include_truth=False), observations
    )
    model_config = _reference_config(config, spline_ridge=point.spline_ridge)
    reference, reference_receipts = base.alpha50_reference(
        panel=panel,
        endpoints=endpoints,
        query=query,
        train_stop=start,
        config=model_config,
    )
    training_parts: list[pd.DataFrame] = []
    training_reference_receipts: dict[str, Any] = {}
    for training_block in outer_spec["training_blocks"]:
        training = base.add_metadata(
            base.block_anchor(anchor_path, training_block, include_truth=True), observations
        )
        if not training["time"].lt(start).all():
            raise base.ContractError("outer correction labels cross boundary")
        bounds = config["block_bounds"][training_block]
        training_reference, receipts = base.alpha50_reference(
            panel=panel,
            endpoints=endpoints,
            query=training,
            train_stop=start,
            config=model_config,
            exclude=(base.utc(bounds[0]), base.utc(bounds[1])),
        )
        training["residual"] = (
            training["truth"].to_numpy(dtype=np.float64) - training_reference
        )
        training_parts.append(training)
        training_reference_receipts[training_block] = receipts
    training = pd.concat(training_parts, ignore_index=True)
    train_times, response = base.profile_response(training)
    train_features = base.align_features(functional_features, train_times)
    valid_train = train_features["public_profile_valid"].to_numpy(dtype=bool)
    train_features = train_features.loc[valid_train].reset_index(drop=True)
    response = response[valid_train]
    train_times = train_times[valid_train]
    fitted = FittedPLSResidual.fit(train_features, response, train_times, rank=point.rank)
    fit_counter["outer_pls_fits"] += 1

    query_times = pd.DatetimeIndex(sorted(query["time"].unique()))
    query_features = base.align_features(functional_features, query_times)
    raw, row_leverage, row_valid = _row_raw_prediction(
        fitted=fitted,
        query=query,
        query_features=query_features,
        query_times=query_times,
    )
    in_enabled_bin = np.isin(
        season_bin(query["time"], int(config["reference"]["season_bin_days"])),
        np.asarray(config["nested_selection"]["enabled_season_bins"], dtype=int),
    )
    enabled = (
        row_valid
        & np.isfinite(row_leverage)
        & (row_leverage <= fitted.leverage_limit(point.leverage_quantile))
        & in_enabled_bin
    )
    bounded, cap_receipt = bounded_profile_correction(
        raw,
        enabled,
        rms_cap=point.rms_cap_c,
        p99_cap=float(config["nested_selection"]["correction_p99_cap_c"]),
    )
    projected = project_profiles_vectorized(
        query,
        reference + bounded * point.strength,
        endpoints,
    ).prediction
    correction = projected - reference
    if np.max(np.abs(correction[~enabled]), initial=0.0) > 1e-12:
        raise base.ContractError("outer disabled profile changed")
    output = query[["time", "layer", "current_blend50"]].copy()
    output["reference"] = reference
    output["candidate"] = projected
    output["correction"] = correction
    output["enabled"] = enabled
    output["leverage"] = row_leverage
    receipt = {
        "fold": outer_name,
        "outer_validation_truth_column_loaded": False,
        "selected_point": asdict(point),
        "selected_point_id": point.point_id,
        "training_blocks": list(outer_spec["training_blocks"]),
        "training_profiles": len(train_times),
        "model": fitted.receipt(point.leverage_quantile),
        "reference_oas": reference_receipts,
        "training_reference_oas": training_reference_receipts,
        "preprojection_cap": cap_receipt,
        "postprojection_correction_rms_c": float(np.sqrt(np.mean(np.square(correction)))),
        "postprojection_correction_p99_c": float(np.quantile(np.abs(correction), 0.99)),
    }
    return output, receipt


def _write_prediction(path: Path, frame: pd.DataFrame) -> None:
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


def _make_report(result: dict[str, Any], commitment: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Pooled delta RMSE: `{metrics['aggregate']['delta_rmse']:.9f}` C",
        f"- Day-bootstrap upper q95: `{result['bootstrap']['ci90_high']:.9f}` C",
        f"- PLS fits: `{result['fit_counts']['total_pls_fits']}`; grid points: `243 x 3 outer folds`",
        f"- Runtime: `{result['runtime']['elapsed_seconds']:.3f}` seconds",
        "- Official hidden/test/sample/submission values read: `0 rows`; CSV/upload: `false`.",
        "",
        "## Outer selections",
        "",
    ]
    for fold, record in commitment["selections"].items():
        lines.append(
            f"- `{fold}`: `{record['point_id']}`; inner delta "
            f"`{record['inner_delta_rmse']:.9f}`; eligible `{record['inner_selection_eligible']}`"
        )
    lines.extend(["", "## Gate checks", ""])
    lines.extend(f"- `{name}`: `{passed}`" for name, passed in result["gate_checks"].items())
    lines.extend(
        [
            "",
            "The three outer blocks are sealed against their own labels but historically exposed; this is local proxy evidence only.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    config: dict[str, Any],
    points: tuple[CapacityPoint, ...],
    data_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = ROOT / config["artifact_directory"]
    report_directory = ROOT / config["report_directory"]
    if output.exists() or report_directory.exists():
        raise FileExistsError("one-shot output already exists")
    output.mkdir(parents=True)
    predictions_directory = output / "predictions"
    predictions_directory.mkdir()
    observations = base.read_observations(data_dir / config["source_observations"]["filename"])
    feature_cache: dict[float, pd.DataFrame] = {}
    for ridge in map(float, config["capacity_grid"]["spline_ridge"]):
        feature_cache[ridge] = build_public_functional_features(
            observations,
            ridge=ridge,
            change_hours=tuple(map(int, config["reference"]["change_hours"])),
        )
    anchor_path = ROOT / config["immutable_inputs"]["alpha50_proxy"]["path"]
    fit_counter = {"inner_pls_fits": 0, "outer_pls_fits": 0}
    evaluation_counter = {
        "outer_grid_points_evaluated": 0,
        "rotation_point_evaluations": 0,
    }
    outputs: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    outer_receipts: dict[str, Any] = {}
    grid_summaries: dict[str, Any] = {}

    with threadpool_limits(limits=int(config["resource_contract"]["blas_threads"])):
        for outer_name, outer_spec in config["folds"].items():
            selected, records, nested_receipt = _nested_select_outer(
                outer_name=outer_name,
                outer_spec=outer_spec,
                config=config,
                points=points,
                observations=observations,
                feature_cache=feature_cache,
                anchor_path=anchor_path,
                fit_counter=fit_counter,
                evaluation_counter=evaluation_counter,
            )
            point = CapacityPoint(**selected["point"])
            prediction, outer_receipt = _fit_outer_prediction(
                outer_name=outer_name,
                outer_spec=outer_spec,
                point=point,
                config=config,
                observations=observations,
                functional_features=feature_cache[point.spline_ridge],
                anchor_path=anchor_path,
                fit_counter=fit_counter,
            )
            path = predictions_directory / f"{outer_name}.npz"
            _write_prediction(path, prediction)
            outputs[outer_name] = {
                "path": str(path.relative_to(ROOT)),
                "rows": len(prediction),
                "bytes": path.stat().st_size,
                "sha256": base.sha256_file(path),
            }
            selections[outer_name] = {
                "point_id": point.point_id,
                "point": asdict(point),
                "inner_selection_eligible": bool(selected["inner_selection_eligible"]),
                "inner_rows": int(selected["rows"]),
                "inner_reference_rmse": float(selected["reference_rmse"]),
                "inner_candidate_rmse": float(selected["candidate_rmse"]),
                "inner_delta_rmse": float(selected["delta_rmse"]),
                "inner_worst_group_delta_rmse": float(selected["worst_group_delta_rmse"]),
                "inner_worst_layer_delta_rmse": float(selected["worst_layer_delta_rmse"]),
            }
            outer_receipts[outer_name] = {"nested": nested_receipt, "outer_fit": outer_receipt}
            grid_summaries[outer_name] = records

    resource = config["resource_contract"]
    total_fits = fit_counter["inner_pls_fits"] + fit_counter["outer_pls_fits"]
    if fit_counter["inner_pls_fits"] != int(resource["expected_inner_pls_fits"]):
        raise base.ContractError("inner PLS fit count drifted")
    if fit_counter["outer_pls_fits"] != int(resource["expected_outer_pls_fits"]):
        raise base.ContractError("outer PLS fit count drifted")
    if total_fits != int(resource["expected_total_pls_fits"]):
        raise base.ContractError("total PLS fit count drifted")
    if evaluation_counter["outer_grid_points_evaluated"] != int(
        resource["expected_candidate_evaluations"]
    ):
        raise base.ContractError("grid evaluation count drifted")
    if evaluation_counter["rotation_point_evaluations"] != int(
        resource["expected_rotation_point_evaluations"]
    ):
        raise base.ContractError("rotation-point evaluation count drifted")

    grid_summary = {
        "experiment_id": EXPERIMENT_ID,
        "grid_points_per_outer": len(points),
        "grid_sha256": _canonical_sha256([asdict(point) for point in points]),
        "outer": grid_summaries,
        "official_rows_read": 0,
    }
    base.atomic_json(output / "inner_grid_summary.json", grid_summary)
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "truth_metric_computed": False,
        "outer_validation_truth_columns_loaded": False,
        "outer_truth_rows_read_before_commitment": 0,
        "official_hidden_gap_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_csv_generated": False,
        "official_upload_performed": False,
        "config_sha256": base.sha256_file(CONFIG),
        "grid_summary_sha256": base.sha256_file(output / "inner_grid_summary.json"),
        "grid_sha256": grid_summary["grid_sha256"],
        "selections": selections,
        "outputs": outputs,
        "outer_receipts": outer_receipts,
        "fit_counts": {**fit_counter, "total_pls_fits": total_fits},
        "evaluation_counts": evaluation_counter,
        "thread_contract": config["resource_contract"],
    }
    base.atomic_json(output / "prediction_commitment.json", commitment)

    scored_parts: list[pd.DataFrame] = []
    for outer_name, record in outputs.items():
        path = ROOT / record["path"]
        if base.sha256_file(path) != record["sha256"]:
            raise base.ContractError("committed outer prediction changed")
        with np.load(path, allow_pickle=False) as payload:
            scored = pd.DataFrame(
                {
                    "time": pd.to_datetime(payload["time_ns"], unit="ns", utc=True),
                    "layer": payload["layer"].astype(int),
                    "current_blend50": payload["current_blend50"],
                    "reference": payload["reference"],
                    "candidate": payload["candidate"],
                    "correction": payload["correction"],
                    "enabled": payload["enabled"],
                }
            )
        truth = base.block_anchor(anchor_path, outer_name, include_truth=True)[
            ["time", "layer", "truth"]
        ]
        scored = scored.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        if scored["truth"].isna().any() or len(scored) != int(record["rows"]):
            raise base.ContractError("outer truth binding failed")
        scored["fold"] = outer_name
        scored_parts.append(scored)
    scored = pd.concat(scored_parts, ignore_index=True)
    metrics = {
        "aggregate": base.metric_record(scored),
        "by_fold": {
            str(key): base.metric_record(group) for key, group in scored.groupby("fold", sort=True)
        },
        "by_layer": {
            str(int(key)): base.metric_record(group)
            for key, group in scored.groupby("layer", sort=True)
        },
    }
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(config["gate"]["bootstrap_replicates"]),
        seed=int(config["gate"]["bootstrap_seed"]),
    )
    correction = scored["correction"].to_numpy(dtype=np.float64)
    oas_axis = scored["reference"].to_numpy(dtype=np.float64) - scored[
        "current_blend50"
    ].to_numpy(dtype=np.float64)
    axis = {"oas_cosine": vector_cosine(correction, oas_axis)}
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
        raise base.ContractError("historical axis coverage failed")
    historical_axis = aligned["prediction"].to_numpy(dtype=np.float64) - aligned[
        "baseline"
    ].to_numpy(dtype=np.float64)
    axis["historical_cosine"] = vector_cosine(correction, historical_axis)
    correction_rms = float(np.sqrt(np.mean(np.square(correction))))
    correction_p99 = float(np.quantile(np.abs(correction), 0.99))
    fold_deltas = [record["delta_rmse"] for record in metrics["by_fold"].values()]
    layer_deltas = [record["delta_rmse"] for record in metrics["by_layer"].values()]
    gate = config["gate"]
    checks = {
        "all_inner_selections_eligible": all(
            record["inner_selection_eligible"] for record in selections.values()
        ),
        "pooled_delta": metrics["aggregate"]["delta_rmse"]
        <= float(gate["pooled_delta_rmse_max_c"]),
        "2024_sep_oct": metrics["by_fold"]["2024_sep_oct"]["delta_rmse"]
        <= float(gate["2024_sep_oct_delta_rmse_max_c"]),
        "two_of_three_folds": sum(delta < 0.0 for delta in fold_deltas)
        >= int(gate["minimum_improved_folds"]),
        "worst_fold": max(fold_deltas) <= float(gate["maximum_worst_fold_regression_c"]),
        "all_layers": max(layer_deltas) <= float(gate["maximum_layer_regression_c"]),
        "day_bootstrap_upper": bootstrap["ci90_high"]
        < float(gate["bootstrap_upper_max_c"]),
        "oas_cosine": abs(axis["oas_cosine"])
        <= float(gate["maximum_absolute_axis_cosine"]),
        "historical_cosine": abs(axis["historical_cosine"])
        <= float(gate["maximum_absolute_axis_cosine"]),
        "correction_p99": correction_p99 <= float(gate["maximum_correction_p99_c"]),
        "correction_rms": correction_rms <= float(gate["maximum_correction_rms_c"]),
    }
    result = {
        "schema_version": "p2.nested_pls_capacity_grid.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "GO_LOCAL_ONLY_NO_UPLOAD" if all(checks.values()) else "NO_GO_CLOSE_FAMILY",
        "comparator": config["comparator"],
        "comparator_disclosure": config["comparator_disclosure"],
        "metrics": metrics,
        "bootstrap": bootstrap,
        "axis_diagnostics": axis,
        "correction_rms_c": correction_rms,
        "correction_p99_c": correction_p99,
        "gate_checks": checks,
        "selections": selections,
        "fit_counts": {**fit_counter, "total_pls_fits": total_fits},
        "evaluation_counts": evaluation_counter,
        "outer_truth_rows_read_after_commitment": len(scored),
        "official_hidden_gap_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "threadpool_info": threadpool_info(),
            "maximum_total_threads": int(config["resource_contract"]["maximum_total_threads"]),
        },
    }
    base.atomic_json(output / "result.json", result)
    report_directory.mkdir(parents=True)
    report_path = report_directory / "report-source.md"
    report_path.write_text(_make_report(result, commitment), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.check == arguments.execute:
        raise SystemExit("choose exactly one of --check or --execute")
    data_dir = arguments.data_dir.expanduser().resolve()
    config, points = load_config(data_dir)
    if arguments.check:
        output = ROOT / config["artifact_directory"]
        report = ROOT / config["report_directory"]
        if output.exists() or report.exists():
            raise SystemExit("one-shot output path already exists")
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "PASS_PREFLIGHT",
                    "grid_points": len(points),
                    "expected_total_pls_fits": config["resource_contract"][
                        "expected_total_pls_fits"
                    ],
                    "maximum_total_threads": config["resource_contract"][
                        "maximum_total_threads"
                    ],
                    "official_rows_read": 0,
                },
                indent=2,
            )
        )
        return
    print(json.dumps(run(config, points, data_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
