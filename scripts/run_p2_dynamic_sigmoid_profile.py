"""Run the dry contract and fold-local precheck for P2 dynamic sigmoid profiles.

This generation cannot create outer predictions, read test-index values, infer the
hidden interval, create a submission, or upload anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.dynamic_sigmoid_profile import (
    TARGET_LAYERS,
    SigmoidSpec,
    TimeBlock,
    build_public_features,
    closed_form_convex_alpha,
    feature_columns,
    fit_latent_ridge,
    fit_parameter_catalog,
    fit_public_profile,
    joint_mask_target_intervals,
    public_profile_arrays,
    rmse,
    stable_parameter_mask,
    target_depth_frame,
)

EXPERIMENT_ID = "p2_dynamic_sigmoid_profile_v1"
OBSERVATION_COLUMNS = [
    "station",
    "year",
    "layer",
    "time",
    "temp",
    "psal",
    "depth",
    "nominal_depth",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(current) for key, current in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(current) for current in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(current) for current in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        current = float(value)
        return current if np.isfinite(current) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _status(
    path: Path,
    *,
    started: float,
    progress: float,
    phase: str,
    detail: str,
    state: str = "running",
) -> None:
    elapsed = max(time.perf_counter() - started, 0.001)
    remaining = elapsed * max(100.0 - progress, 0.0) / max(progress, 1.0)
    _write_json(
        path,
        {
            "title": "P2 dynamic sigmoid profile precheck",
            "experiment_id": EXPERIMENT_ID,
            "status": state,
            "progress": float(np.clip(progress, 0.0, 100.0)),
            "phase": phase,
            "detail": detail,
            "elapsed_seconds": elapsed,
            "eta": (datetime.now().astimezone() + timedelta(seconds=remaining)).isoformat(),
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected experiment id")
    if value.get("status") != "preregistered_precheck_only":
        raise ValueError("contract is not precheck-only")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("runner must remain local research-only")
    problem = value["problem_contract"]
    forbidden = {
        "hidden_target_values_read": False,
        "test_index_values_read": False,
        "outer_truth_scoring_allowed": False,
        "submission_generation_allowed": False,
    }
    if any(problem.get(key) is not expected for key, expected in forbidden.items()):
        raise ValueError("a prohibited data/action gate is not fail-closed")
    if value.get("submission_created") is not False:
        raise ValueError("contract unexpectedly permits a submission")
    return value


def _blocks(contract: Mapping[str, Any]) -> tuple[TimeBlock, ...]:
    result = tuple(
        TimeBlock.from_strings(name, values)
        for name, values in contract["validation"]["blocks"].items()
    )
    expected_days = int(contract["validation"]["all_blocks_exact_days"])
    if len(result) != 3 or any(block.days != expected_days for block in result):
        raise ValueError("validation blocks are not three exact 61-day intervals")
    return result


def _sigmoid_spec(contract: Mapping[str, Any]) -> SigmoidSpec:
    value = contract["sigmoid"]
    targets = contract["problem_contract"]["target_nominal_depth_m"]
    return SigmoidSpec(
        tuple(float(current) for current in value["center_bounds_m"]),
        tuple(float(current) for current in value["width_bounds_m"]),
        tuple(float(current) for current in value["deterministic_start_center_fractions"]),
        tuple(float(current) for current in value["deterministic_start_width_m"]),
        int(value["least_squares_max_nfev"]),
        float(value["least_squares_ftol"]),
        float(value["least_squares_xtol"]),
        float(value["least_squares_gtol"]),
        float(value["boundary_fraction"]),
        tuple(float(targets[str(layer)]) for layer in TARGET_LAYERS),
    )


def _resolve_data_dir(cli_path: str | None) -> Path:
    raw = cli_path or os.environ.get("P2_DATA_DIR")
    if not raw:
        raise FileNotFoundError("set P2_DATA_DIR or pass --data-dir")
    root = Path(raw).expanduser().resolve()
    for name in ("observations.csv", "README.md"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"P2 data directory is missing {name}")
    return root


def _read_observations(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        data_dir / "observations.csv",
        dtype={"station": "string", "time": "string"},
    )
    if list(frame.columns) != OBSERVATION_COLUMNS:
        raise ValueError("unexpected observations schema")
    if frame.duplicated(["station", "year", "layer", "time"]).any():
        raise ValueError("duplicate observation keys")
    if set(frame["station"].dropna().unique()) != {"S-ORS"}:
        raise ValueError("P2 observations contain an unexpected station")
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError("observation time contains missing values")
    return frame


def _hidden_mask(observations: pd.DataFrame, contract: Mapping[str, Any]) -> np.ndarray:
    start, stop = contract["problem_contract"]["hidden_interval_kst"]
    block = TimeBlock.from_strings("hidden", (start, stop))
    return (
        block.mask(pd.to_datetime(observations["time"], utc=True))
        & observations["layer"].isin(TARGET_LAYERS).to_numpy()
    )


def _audit_hidden(observations: pd.DataFrame, contract: Mapping[str, Any]) -> dict[str, int]:
    selected = _hidden_mask(observations, contract)
    if not observations.loc[selected, ["temp", "psal"]].isna().all().all():
        raise ValueError("distributed hidden target temp/psal are unexpectedly populated")
    return {
        "hidden_target_rows": int(selected.sum()),
        "hidden_temp_nonmissing": int(observations.loc[selected, "temp"].notna().sum()),
        "hidden_psal_nonmissing": int(observations.loc[selected, "psal"].notna().sum()),
    }


def _git_provenance(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    return {"git_commit": commit or None, "dirty": bool(status), "dirty_entry_count": len(status)}


def _source_hashes(root: Path, config_path: Path, observation_path: Path) -> dict[str, object]:
    paths = {
        "config": config_path,
        "helper": root / "src/p2_restore/dynamic_sigmoid_profile.py",
        "runner": root / "scripts/run_p2_dynamic_sigmoid_profile.py",
        "tests": root / "tests/test_p2_dynamic_sigmoid_profile.py",
        "observations": observation_path,
    }
    return {
        name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }


def _base_receipt(
    *,
    root: Path,
    contract: Mapping[str, Any],
    config_path: Path,
    observation_path: Path,
    mode: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "aggregate_only": True,
        "external_values_used": False,
        "hidden_target_values_read": False,
        "test_index_values_read": False,
        "outer_predictions_created": False,
        "outer_truth_scored": False,
        "submission_created": False,
        "upload_attempted": False,
        "blocks": {
            block.name: {
                "start_kst": block.start.isoformat(),
                "stop_kst": block.stop.isoformat(),
                "days": block.days,
            }
            for block in _blocks(contract)
        },
        "provenance": {
            **_git_provenance(root),
            "python": sys.version.split()[0],
            "files": _source_hashes(root, config_path, observation_path),
            "literature": contract["provenance"]["primary_literature"],
        },
    }


def _dry_run(
    *,
    root: Path,
    contract: Mapping[str, Any],
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    status_path: Path,
    started: float,
) -> dict[str, object]:
    _status(
        status_path,
        started=started,
        progress=15.0,
        phase="dry-run",
        detail="observations schema and fail-closed contract",
    )
    observations = _read_observations(data_dir)
    hidden = _audit_hidden(observations, contract)
    result = {
        **_base_receipt(
            root=root,
            contract=contract,
            config_path=config_path,
            observation_path=data_dir / "observations.csv",
            mode="dry-run",
        ),
        "status": "dry_run_pass",
        "data_contract": {
            "observation_rows": len(observations),
            "observation_columns": list(observations.columns),
            "duplicate_keys": 0,
            **hidden,
        },
        "forbidden_file_reads": {
            "test_index.csv": 0,
            "sample_submission.csv": 0,
            "baseline_interp.csv": 0,
            "hidden_answer_or_mirror": 0,
        },
    }
    _write_json(output_dir / "dry_run.json", result)
    _status(
        status_path,
        started=started,
        progress=100.0,
        phase="complete",
        detail="dry-run PASS; no target score or prediction generated",
        state="complete",
    )
    return result


def _block_public_frame(features: pd.DataFrame, block: TimeBlock) -> pd.DataFrame:
    return features.loc[block.mask(features.index)]


def _outer_allowed_time(
    index: pd.DatetimeIndex,
    *,
    outer: TimeBlock,
    hidden: TimeBlock,
    purge_days: int,
) -> np.ndarray:
    return ~outer.expanded_mask(index, purge_days=purge_days) & ~hidden.mask(index)


def _precheck_fold(
    *,
    observations: pd.DataFrame,
    public_features: pd.DataFrame,
    outer: TimeBlock,
    hidden: TimeBlock,
    contract: Mapping[str, Any],
    spec: SigmoidSpec,
    status_path: Path,
    started: float,
    progress_start: float,
    progress_stop: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    validation = contract["validation"]
    sigmoid = contract["sigmoid"]
    feature_config = contract["public_features"]
    gate_config = contract["gates"]
    purge_days = int(validation["purge_days"])
    masked = joint_mask_target_intervals(observations, (outer,))
    times = pd.to_datetime(masked["time"], utc=True)
    current_target = outer.mask(times) & masked["layer"].isin(TARGET_LAYERS).to_numpy()
    if not masked.loc[current_target, ["temp", "psal"]].isna().all().all():
        raise AssertionError("current outer target mask is incomplete")

    validation_public = _block_public_frame(public_features, outer)
    support = (
        validation_public["public_temp_count"].to_numpy(float)
        >= int(sigmoid["minimum_public_points"])
    ) & (
        validation_public["public_depth_span"].to_numpy(float)
        >= float(sigmoid["minimum_public_depth_span_m"])
    )
    support_share = float(np.mean(support)) if len(support) else 0.0
    gate_1 = {
        "validation_times": len(validation_public),
        "supported_times": int(support.sum()),
        "support_share": support_share,
        "threshold": float(gate_config["gate_1_public_support"]["minimum_share"]),
        "pass": bool(support_share >= float(gate_config["gate_1_public_support"]["minimum_share"])),
    }

    def progress(done: int, total: int) -> None:
        if done % 500 and done != total:
            return
        fraction = done / max(total, 1)
        value = progress_start + 0.55 * (progress_stop - progress_start) * fraction
        _status(
            status_path,
            started=started,
            progress=value,
            phase=f"precheck {outer.name}",
            detail=f"variable-projection profiles {done}/{total}",
        )

    catalog = fit_parameter_catalog(
        masked,
        spec=spec,
        allowed_time=lambda index: _outer_allowed_time(
            index, outer=outer, hidden=hidden, purge_days=purge_days
        ),
        stride_minutes=int(sigmoid["profile_fit_stride_minutes"]),
        minimum_points=int(sigmoid["minimum_full_profile_points"]),
        progress=progress,
    )
    stratified = catalog["success"].astype(bool).to_numpy() & (
        np.abs(catalog["amplitude_c"].to_numpy(float))
        >= float(feature_config["parameter_training_filter"]["minimum_abs_amplitude_c"])
    )
    stratified_catalog = catalog.loc[stratified]
    spread_p95 = (
        float(np.quantile(stratified_catalog["multistart_target_spread_c"], 0.95))
        if len(stratified_catalog)
        else float("inf")
    )
    gate_2_config = gate_config["gate_2_multistart_stability"]
    gate_2 = {
        "stratified_profiles": len(stratified_catalog),
        "multistart_target_spread_p95_c": spread_p95,
        "minimum_profiles": int(gate_2_config["minimum_stratified_profiles"]),
        "maximum_p95_c": float(gate_2_config["maximum_p95_target_spread_c"]),
        "pass": bool(
            len(stratified_catalog) >= int(gate_2_config["minimum_stratified_profiles"])
            and spread_p95 <= float(gate_2_config["maximum_p95_target_spread_c"])
        ),
    }
    gate_3_config = gate_config["gate_3_full_profile_fit"]
    r2_share = (
        float(
            np.mean(stratified_catalog["r2"].to_numpy(float) >= float(gate_3_config["minimum_r2"]))
        )
        if len(stratified_catalog)
        else 0.0
    )
    boundary_share = (
        float(stratified_catalog["boundary_saturated"].astype(bool).mean())
        if len(stratified_catalog)
        else 1.0
    )
    gate_3 = {
        "minimum_r2": float(gate_3_config["minimum_r2"]),
        "r2_pass_share": r2_share,
        "minimum_r2_pass_share": float(gate_3_config["minimum_share_at_or_above_r2"]),
        "boundary_saturation_share": boundary_share,
        "maximum_boundary_saturation_share": float(
            gate_3_config["maximum_boundary_saturation_share"]
        ),
        "r2_quantiles": {
            str(quantile): float(np.nanquantile(stratified_catalog["r2"], quantile))
            if len(stratified_catalog)
            else None
            for quantile in (0.1, 0.5, 0.9)
        },
        "pass": bool(
            r2_share >= float(gate_3_config["minimum_share_at_or_above_r2"])
            and boundary_share <= float(gate_3_config["maximum_boundary_saturation_share"])
        ),
    }

    training_filter = feature_config["parameter_training_filter"]
    stable = stable_parameter_mask(
        catalog,
        minimum_abs_amplitude_c=float(training_filter["minimum_abs_amplitude_c"]),
        minimum_r2=float(training_filter["minimum_r2"]),
        maximum_condition=float(training_filter["maximum_scaled_jacobian_condition"]),
        maximum_spread_c=float(training_filter["maximum_multistart_target_spread_c"]),
    )
    catalog = catalog.copy()
    catalog["stable_for_ridge"] = stable
    ridge_error: str | None = None
    public_diagnostics: list[dict[str, object]] = []
    try:
        ridge = fit_latent_ridge(
            public_features,
            catalog.loc[stable],
            columns=feature_columns(public_features),
            alpha=float(feature_config["ridge_alpha"]),
            minimum_feature_coverage=float(feature_config["minimum_feature_coverage"]),
            minimum_rows=int(feature_config["minimum_ridge_rows"]),
            center_bounds_m=spec.center_bounds_m,
            width_bounds_m=spec.width_bounds_m,
        )
        latent = ridge.predict(validation_public)
        gate_4_config = gate_config["gate_4_public_observability"]
        for row_number, (_, row) in enumerate(validation_public.iterrows()):
            depth, temperature = public_profile_arrays(row)
            current = fit_public_profile(
                depth,
                temperature,
                np.asarray(spec.target_depths_m),
                center_m=float(latent[row_number, 0]),
                log_width=float(latent[row_number, 1]),
                minimum_points=int(sigmoid["minimum_public_points"]),
                minimum_depth_span_m=float(sigmoid["minimum_public_depth_span_m"]),
                center_step_m=float(sigmoid["finite_difference_center_step_m"]),
                log_width_step=float(sigmoid["finite_difference_log_width_step"]),
                condition_max=float(gate_4_config["maximum_profiled_jacobian_condition"]),
            )
            public_diagnostics.append(
                {
                    "supported": current.supported,
                    "amplitude_c": current.amplitude_c,
                    "condition": current.profiled_jacobian_condition,
                    "observable": current.observable,
                }
            )
    except (ValueError, np.linalg.LinAlgError) as exc:
        ridge_error = f"{type(exc).__name__}: {exc}"

    diagnostic = pd.DataFrame(public_diagnostics)
    minimum_amplitude = float(training_filter["minimum_abs_amplitude_c"])
    if diagnostic.empty:
        stratified_supported = np.zeros(0, dtype=bool)
        observable_share = 0.0
        observable_count = 0
    else:
        stratified_supported = diagnostic["supported"].astype(bool).to_numpy() & (
            np.abs(diagnostic["amplitude_c"].to_numpy(float)) >= minimum_amplitude
        )
        observable_count = int(
            diagnostic.loc[stratified_supported, "observable"].astype(bool).sum()
        )
        observable_share = (
            float(observable_count / stratified_supported.sum())
            if stratified_supported.sum()
            else 0.0
        )
    gate_4_config = gate_config["gate_4_public_observability"]
    gate_4 = {
        "ridge_training_rows": int(stable.sum()),
        "ridge_error": ridge_error,
        "stratified_supported_times": int(stratified_supported.sum()),
        "observable_times": observable_count,
        "observable_share": observable_share,
        "minimum_stratified_supported_times": int(
            gate_4_config["minimum_stratified_supported_times"]
        ),
        "minimum_observable_share": float(
            gate_4_config["minimum_observable_share_of_stratified_supported"]
        ),
        "condition_quantiles": {
            str(quantile): float(
                np.nanquantile(
                    diagnostic.loc[stratified_supported, "condition"].replace(
                        [np.inf, -np.inf], np.nan
                    ),
                    quantile,
                )
            )
            if int(stratified_supported.sum())
            and diagnostic.loc[stratified_supported, "condition"]
            .replace([np.inf, -np.inf], np.nan)
            .notna()
            .any()
            else None
            for quantile in (0.5, 0.9, 0.95)
        },
        "pass": bool(
            ridge_error is None
            and int(stratified_supported.sum())
            >= int(gate_4_config["minimum_stratified_supported_times"])
            and observable_share
            >= float(gate_4_config["minimum_observable_share_of_stratified_supported"])
        ),
    }
    gates = {"gate_1": gate_1, "gate_2": gate_2, "gate_3": gate_3, "gate_4": gate_4}
    return (
        {
            "outer_block": outer.name,
            "current_outer_target_rows_masked": int(current_target.sum()),
            "current_outer_truth_rows_scored": 0,
            "catalog_rows": len(catalog),
            "catalog_success_rows": int(catalog["success"].astype(bool).sum()),
            "gates": gates,
            "pass_gates_1_to_4": bool(all(current["pass"] for current in gates.values())),
        },
        catalog,
    )


def _load_incumbent(root: Path, contract: Mapping[str, Any]) -> pd.DataFrame:
    value = contract["incumbent"]
    path = (root / value["path"]).resolve()
    if _sha256(path) != value["sha256"]:
        raise ValueError("frozen incumbent OOF SHA differs")
    allowed = list(value["columns_allowed"])
    if "truth" in allowed or value.get("truth_column_read") is not False:
        raise ValueError("incumbent truth access is not fail-closed")
    frame = pd.read_parquet(path, columns=allowed)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    if frame.duplicated(["time", "layer"]).any() or not np.isfinite(frame["prediction"]).all():
        raise ValueError("incumbent allowed columns fail key/value contract")
    return frame


def _inner_truth(observations: pd.DataFrame, *, inner: TimeBlock, outer: TimeBlock) -> pd.DataFrame:
    times = pd.to_datetime(observations["time"], utc=True)
    selected = (
        inner.mask(times)
        & observations["layer"].isin(TARGET_LAYERS).to_numpy()
        & np.isfinite(observations["temp"].to_numpy(float))
    )
    if np.any(selected & outer.mask(times)):
        raise AssertionError("current outer truth entered an inner score")
    frame = observations.loc[selected, ["time", "layer", "temp", "depth", "nominal_depth"]].copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    depths = target_depth_frame(observations).rename(columns={"target_depth": "_target_depth"})
    frame = frame.merge(depths, on=["time", "layer"], how="left", validate="one_to_one")
    frame = frame.rename(columns={"temp": "truth", "_target_depth": "target_depth"})
    frame = frame.loc[np.isfinite(frame["target_depth"])].drop(columns=["depth", "nominal_depth"])
    return frame


def _predict_inner_sigmoid(
    *,
    truth: pd.DataFrame,
    features: pd.DataFrame,
    ridge: Any,
    incumbent: pd.DataFrame,
    contract: Mapping[str, Any],
    spec: SigmoidSpec,
) -> pd.DataFrame:
    sigmoid = contract["sigmoid"]
    filter_config = contract["public_features"]["parameter_training_filter"]
    gate_4 = contract["gates"]["gate_4_public_observability"]
    unique_times = pd.DatetimeIndex(sorted(truth["time"].unique()))
    current_features = features.reindex(unique_times)
    latent = ridge.predict(current_features)
    records: list[dict[str, object]] = []
    depth_by_time = truth.groupby("time", sort=False)
    for row_number, current_time in enumerate(unique_times):
        row = current_features.loc[current_time]
        depth, temperature = public_profile_arrays(row)
        target_rows = depth_by_time.get_group(current_time).sort_values("layer")
        target_depth = target_rows["target_depth"].to_numpy(float)
        fitted = fit_public_profile(
            depth,
            temperature,
            target_depth,
            center_m=float(latent[row_number, 0]),
            log_width=float(latent[row_number, 1]),
            minimum_points=int(sigmoid["minimum_public_points"]),
            minimum_depth_span_m=float(sigmoid["minimum_public_depth_span_m"]),
            center_step_m=float(sigmoid["finite_difference_center_step_m"]),
            log_width_step=float(sigmoid["finite_difference_log_width_step"]),
            condition_max=float(gate_4["maximum_profiled_jacobian_condition"]),
        )
        active = bool(
            fitted.supported
            and fitted.observable
            and abs(fitted.amplitude_c) >= float(filter_config["minimum_abs_amplitude_c"])
            and np.isfinite(fitted.target_prediction).all()
        )
        for layer, prediction in zip(
            target_rows["layer"].to_numpy(int), fitted.target_prediction, strict=True
        ):
            records.append(
                {
                    "time": current_time,
                    "layer": int(layer),
                    "sigmoid_prediction": float(prediction) if active else np.nan,
                    "active": active,
                }
            )
    prediction = pd.DataFrame.from_records(records)
    result = truth.merge(prediction, on=["time", "layer"], how="left", validate="one_to_one")
    result = result.merge(
        incumbent.rename(columns={"prediction": "incumbent"}),
        on=["time", "layer"],
        how="inner",
        validate="one_to_one",
    )
    result["challenger"] = np.where(
        result["active"].fillna(False), result["sigmoid_prediction"], result["incumbent"]
    )
    if not np.isfinite(result[["truth", "incumbent", "challenger"]]).all().all():
        raise ValueError("inner comparison contains non-finite values")
    return result


def _gate_5(
    *,
    observations: pd.DataFrame,
    public_features: pd.DataFrame,
    catalogs: Mapping[str, pd.DataFrame],
    blocks: Sequence[TimeBlock],
    contract: Mapping[str, Any],
    spec: SigmoidSpec,
    incumbent: pd.DataFrame,
) -> dict[str, object]:
    feature_config = contract["public_features"]
    training_filter = feature_config["parameter_training_filter"]
    gate = contract["gates"]["gate_5_inner_predictive"]
    purge_days = int(contract["validation"]["purge_days"])
    output: dict[str, object] = {}
    for outer in blocks:
        catalog = catalogs[outer.name]
        inner_frames: list[pd.DataFrame] = []
        for inner in blocks:
            if inner.name == outer.name:
                continue
            inner_excluded = inner.expanded_mask(
                pd.DatetimeIndex(pd.to_datetime(catalog["time"], utc=True)),
                purge_days=purge_days,
            )
            stable = stable_parameter_mask(
                catalog,
                minimum_abs_amplitude_c=float(training_filter["minimum_abs_amplitude_c"]),
                minimum_r2=float(training_filter["minimum_r2"]),
                maximum_condition=float(training_filter["maximum_scaled_jacobian_condition"]),
                maximum_spread_c=float(training_filter["maximum_multistart_target_spread_c"]),
            )
            training = stable & ~inner_excluded
            ridge = fit_latent_ridge(
                public_features,
                catalog.loc[training],
                columns=feature_columns(public_features),
                alpha=float(feature_config["ridge_alpha"]),
                minimum_feature_coverage=float(feature_config["minimum_feature_coverage"]),
                minimum_rows=int(feature_config["minimum_ridge_rows"]),
                center_bounds_m=spec.center_bounds_m,
                width_bounds_m=spec.width_bounds_m,
            )
            truth = _inner_truth(observations, inner=inner, outer=outer)
            current = _predict_inner_sigmoid(
                truth=truth,
                features=public_features,
                ridge=ridge,
                incumbent=incumbent,
                contract=contract,
                spec=spec,
            )
            current["inner_block"] = inner.name
            current["ridge_training_rows"] = int(training.sum())
            inner_frames.append(current)
        pooled = pd.concat(inner_frames, ignore_index=True)
        alpha = closed_form_convex_alpha(
            pooled["truth"].to_numpy(float),
            pooled["incumbent"].to_numpy(float),
            pooled["challenger"].to_numpy(float),
            bounds=tuple(float(current) for current in gate["alpha_bounds"]),
        )
        pooled["blended"] = pooled["incumbent"] + alpha * (
            pooled["challenger"] - pooled["incumbent"]
        )
        pooled_delta = rmse(pooled["truth"], pooled["blended"]) - rmse(
            pooled["truth"], pooled["incumbent"]
        )
        by_block: dict[str, object] = {}
        for name, current in pooled.groupby("inner_block", sort=False):
            by_block[str(name)] = {
                "rows": len(current),
                "active_share": float(current["active"].astype(bool).mean()),
                "ridge_training_rows": int(current["ridge_training_rows"].iloc[0]),
                "incumbent_rmse": rmse(current["truth"], current["incumbent"]),
                "blended_rmse": rmse(current["truth"], current["blended"]),
                "delta_rmse": rmse(current["truth"], current["blended"])
                - rmse(current["truth"], current["incumbent"]),
            }
        by_layer: dict[str, object] = {}
        for layer, current in pooled.groupby("layer", sort=True):
            by_layer[str(int(layer))] = {
                "rows": len(current),
                "incumbent_rmse": rmse(current["truth"], current["incumbent"]),
                "blended_rmse": rmse(current["truth"], current["blended"]),
                "delta_rmse": rmse(current["truth"], current["blended"])
                - rmse(current["truth"], current["incumbent"]),
            }
        block_delta = [float(current["delta_rmse"]) for current in by_block.values()]
        layer_delta = [float(current["delta_rmse"]) for current in by_layer.values()]
        passed = bool(
            alpha > 0.0
            and pooled_delta <= -float(gate["minimum_pooled_improvement_c"])
            and max(block_delta) <= float(gate["maximum_each_inner_block_delta_rmse_c"])
            and max(layer_delta) <= float(gate["maximum_worst_layer_delta_rmse_c"])
        )
        output[outer.name] = {
            "outer_truth_rows_scored": 0,
            "inner_blocks": [block.name for block in blocks if block.name != outer.name],
            "rows": len(pooled),
            "alpha": alpha,
            "pooled_incumbent_rmse": rmse(pooled["truth"], pooled["incumbent"]),
            "pooled_blended_rmse": rmse(pooled["truth"], pooled["blended"]),
            "pooled_delta_rmse": pooled_delta,
            "by_inner_block": by_block,
            "by_layer": by_layer,
            "pass": passed,
            "failure_policy_applied": "none" if passed else "exact_incumbent_no_op",
        }
    return {
        "executed": True,
        "outer_truth_scored": False,
        "folds": output,
        "all_outer_inner_gates_pass": bool(all(value["pass"] for value in output.values())),
    }


def _precheck(
    *,
    root: Path,
    contract: Mapping[str, Any],
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    status_path: Path,
    started: float,
) -> dict[str, object]:
    observations = _read_observations(data_dir)
    hidden_audit = _audit_hidden(observations, contract)
    blocks = _blocks(contract)
    hidden = TimeBlock.from_strings("hidden", contract["problem_contract"]["hidden_interval_kst"])
    spec = _sigmoid_spec(contract)
    feature_config = contract["public_features"]
    _status(
        status_path,
        started=started,
        progress=5.0,
        phase="features",
        detail="building fixed public-only feature state",
    )
    public_features = build_public_features(
        observations,
        public_layers=tuple(
            int(current) for current in feature_config["temperature_and_salinity_layers"]
        ),
        gradient_pairs=feature_config["adjacent_gradient_pairs"],
        change_hours=tuple(int(current) for current in feature_config["causal_change_hours"]),
    )
    fold_results: dict[str, object] = {}
    catalogs: dict[str, pd.DataFrame] = {}
    for number, outer in enumerate(blocks):
        start = 8.0 + number * 25.0
        stop = start + 25.0
        result, catalog = _precheck_fold(
            observations=observations,
            public_features=public_features,
            outer=outer,
            hidden=hidden,
            contract=contract,
            spec=spec,
            status_path=status_path,
            started=started,
            progress_start=start,
            progress_stop=stop,
        )
        fold_results[outer.name] = result
        catalogs[outer.name] = catalog
        _status(
            status_path,
            started=started,
            progress=stop,
            phase=f"precheck {outer.name}",
            detail=f"gates 1-4 {'PASS' if result['pass_gates_1_to_4'] else 'FAIL'}",
        )
    pass_1_to_4 = bool(all(value["pass_gates_1_to_4"] for value in fold_results.values()))
    if pass_1_to_4:
        _status(
            status_path,
            started=started,
            progress=84.0,
            phase="inner gate",
            detail="all observability gates passed; evaluating fold-local inner OOF only",
        )
        incumbent = _load_incumbent(root, contract)
        gate_5 = _gate_5(
            observations=observations,
            public_features=public_features,
            catalogs=catalogs,
            blocks=blocks,
            contract=contract,
            spec=spec,
            incumbent=incumbent,
        )
    else:
        gate_5 = {
            "executed": False,
            "reason": "at least one outer fold failed gates 1-4",
            "outer_truth_scored": False,
        }
    receipt = {
        **_base_receipt(
            root=root,
            contract=contract,
            config_path=config_path,
            observation_path=data_dir / "observations.csv",
            mode="precheck",
        ),
        "status": "inner_gate_evaluated" if gate_5["executed"] else "precheck_failed",
        "data_contract": {
            "observation_rows": len(observations),
            **hidden_audit,
            "public_feature_rows": len(public_features),
            "public_model_feature_count": len(feature_columns(public_features)),
        },
        "gates_1_to_4": {
            "pass": pass_1_to_4,
            "folds": fold_results,
        },
        "gate_5_inner_only": gate_5,
        "forbidden_operations": {
            "test_index_value_reads": 0,
            "sample_submission_reads": 0,
            "hidden_target_value_reads": 0,
            "outer_truth_scores": 0,
            "outer_prediction_rows": 0,
            "submission_rows": 0,
            "upload_attempts": 0,
        },
    }
    _write_json(output_dir / "precheck.json", receipt)
    final_state = "complete" if gate_5.get("executed") else "stopped"
    _status(
        status_path,
        started=started,
        progress=100.0,
        phase="complete",
        detail=(
            "gate 5 evaluated; no outer score generated"
            if gate_5.get("executed")
            else "fail-fast at observability gates 1-4; gate 5 not run"
        ),
        state=final_state,
    )
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/p2_dynamic_sigmoid_profile_v1.json"),
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--mode", choices=("dry-run", "precheck"), required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    contract = _load_contract(config_path)
    data_dir = _resolve_data_dir(args.data_dir)
    output_dir = (
        (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    configured_status = Path(contract["outputs"]["status_path"])
    status_path = args.status_path or configured_status
    status_path = (root / status_path).resolve() if not status_path.is_absolute() else status_path
    started = time.perf_counter()
    try:
        if args.mode == "dry-run":
            result = _dry_run(
                root=root,
                contract=contract,
                config_path=config_path,
                data_dir=data_dir,
                output_dir=output_dir,
                status_path=status_path,
                started=started,
            )
        else:
            result = _precheck(
                root=root,
                contract=contract,
                config_path=config_path,
                data_dir=data_dir,
                output_dir=output_dir,
                status_path=status_path,
                started=started,
            )
    except Exception as exc:
        _status(
            status_path,
            started=started,
            progress=100.0,
            phase="failed",
            detail=f"{type(exc).__name__}: {exc}",
            state="failed",
        )
        raise
    print(json.dumps(_json_safe(result), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
