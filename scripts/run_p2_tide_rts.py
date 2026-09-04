"""Run the preregistered P2 public-only observability/RTS one-shot.

The recoverability precheck is intentionally terminal.  Target layers 2--4
temperature and salinity are simultaneously hidden for each 61-day outer
block, with a seven-day purge.  A conditional RTS model is evaluated only if
all structural gates pass without changing this file or its configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_tide_rts import (
    PUBLIC_LAYERS,
    TARGET_LAYERS,
    PublicFactorEncoder,
    ResidualRegressor,
    TideRTSConfig,
    actual_depth_interpolation,
    build_tide_panel,
    cadence_segments,
    exact_fallback,
    fold_target_depths,
    kalman_rts_smoother,
    m2_relationship_diagnostics,
    observability_diagnostics,
    outer_split,
    posterior_target_sd,
    residual_skill_r2,
    support_diagnostics,
)

EXPECTED_EXPERIMENT = "p2_tide_rts_v1"
EXPECTED_ROWS = 69_850
OUTPUT_LABELS = ("temp_l2", "temp_l3", "temp_l4", "psal_l2", "psal_l3", "psal_l4")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _status(
    path: Path,
    *,
    progress: float,
    phase: str,
    detail: str,
    started: float,
    status: str = "running",
    full_model_executed: bool = False,
) -> None:
    elapsed = max(time.perf_counter() - started, 0.001)
    remaining = elapsed * max(100.0 - progress, 0.0) / max(progress, 1.0)
    eta = datetime.now().astimezone() + timedelta(seconds=remaining)
    _write_json(
        path,
        {
            "title": "P2 tide-aware recoverability one-shot",
            "experiment_id": EXPECTED_EXPERIMENT,
            "status": status,
            "progress": float(progress),
            "progress_fraction": float(progress / 100.0),
            "phase": phase,
            "detail": detail,
            "elapsed_seconds": elapsed,
            "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST"),
            "full_model_executed": bool(full_model_executed),
            "submission_created": False,
            "external_values_used": False,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != EXPECTED_EXPERIMENT:
        raise ValueError("unexpected P2 tide RTS experiment id")
    if contract.get("status") != "preregistered_local_one_shot":
        raise ValueError("P2 tide RTS contract is not preregistered")
    if contract.get("research_only") is not True or contract.get("upload_allowed") is not False:
        raise ValueError("P2 tide RTS must remain local-only")
    if contract.get("external_values_used") is not False:
        raise ValueError("external values are forbidden")
    if contract["outputs"].get("submission_created") is not False:
        raise ValueError("this runner must never create a submission")
    if contract["validation"].get("selection_policy") is None:
        raise ValueError("fixed one-shot selection policy is absent")
    problem = contract["problem_contract"]
    if tuple(problem["public_layers"]) != PUBLIC_LAYERS:
        raise ValueError("public layer contract changed")
    if tuple(problem["target_layers"]) != TARGET_LAYERS:
        raise ValueError("target layer contract changed")
    if set(problem["simultaneous_mask_variables"]) != {"temp", "psal"}:
        raise ValueError("temperature/salinity must be masked together")
    return contract


def _model_config(contract: dict[str, Any]) -> TideRTSConfig:
    value = contract["precheck"]
    return TideRTSConfig(
        factors=int(value["factors"]),
        ridge_alpha=float(value["ridge_alpha"]),
        minimum_feature_coverage=float(value["minimum_feature_coverage"]),
        slow_phi_min=float(value["slow_phi_min"]),
        slow_phi_max=float(value["slow_phi_max"]),
        resonator_damping=float(value["resonator_damping"]),
        resonator_process_variance=float(value["resonator_process_variance"]),
        gramian_horizon_steps=int(value["gramian_horizon_steps"]),
        gramian_rank_tolerance=float(value["gramian_rank_tolerance"]),
        gramian_condition_max=float(value["gramian_condition_max"]),
        support_quantile=float(value["support_quantile"]),
        minimum_support_share=float(value["minimum_support_share"]),
        minimum_two_temp_coverage=float(value["minimum_two_public_temp_coverage"]),
        posterior_sd_scale_max=float(contract["conditional_model"]["posterior_sd_scale_max"]),
    )


def _validate_frozen(
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    hashes: dict[str, dict[str, Any]] = {}
    for name, item in contract["frozen_inputs"].items():
        path = Path(item["path"])
        digest = _sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"frozen {name} SHA256 changed")
        hashes[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}

    adaptive = pd.read_parquet(contract["frozen_inputs"]["adaptive_oof"]["path"])
    physical = pd.read_parquet(contract["frozen_inputs"]["physical_oof"]["path"])
    required = {"time", "layer", "truth", "block", "prediction"}
    for name, frame in (("adaptive", adaptive), ("physical", physical)):
        if missing := required.difference(frame.columns):
            raise ValueError(f"frozen {name} OOF misses {sorted(missing)}")
        if len(frame) != EXPECTED_ROWS or frame.duplicated(["time", "layer"]).any():
            raise ValueError(f"frozen {name} OOF grain changed")
        if not np.isfinite(frame[["truth", "prediction"]].to_numpy(float)).all():
            raise ValueError(f"frozen {name} OOF contains non-finite values")

    keys = ["time", "layer"]
    adaptive = adaptive.sort_values(keys).reset_index(drop=True)
    physical = physical.sort_values(keys).reset_index(drop=True)
    if not adaptive[keys].equals(physical[keys]):
        raise ValueError("adaptive and physical OOF keys differ")
    if not np.array_equal(adaptive["block"].to_numpy(), physical["block"].to_numpy()):
        raise ValueError("adaptive and physical OOF blocks differ")
    if not np.allclose(
        adaptive["truth"].to_numpy(float), physical["truth"].to_numpy(float), rtol=0, atol=0
    ):
        raise ValueError("adaptive and physical OOF truth differs")

    truth = adaptive["truth"].to_numpy(float)
    adaptive_rmse = _rmse(truth, adaptive["prediction"].to_numpy(float))
    physical_rmse = _rmse(truth, physical["prediction"].to_numpy(float))
    expected_adaptive = float(contract["frozen_inputs"]["adaptive_oof"]["expected_rmse"])
    expected_physical = float(contract["frozen_inputs"]["physical_oof"]["expected_rmse"])
    if not np.isclose(adaptive_rmse, expected_adaptive, rtol=0, atol=1e-12):
        raise ValueError("adaptive proxy RMSE changed")
    if not np.isclose(physical_rmse, expected_physical, rtol=0, atol=1e-12):
        raise ValueError("physical proxy RMSE changed")
    adaptive = adaptive.rename(columns={"prediction": "adaptive_prediction"})
    adaptive["physical_prediction"] = physical["prediction"].to_numpy(float)
    return adaptive, {
        **hashes,
        "metrics": {
            "rows": len(adaptive),
            "adaptive_proxy_rmse": adaptive_rmse,
            "physical_projection_rmse": physical_rmse,
        },
    }


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    current = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(actual) & np.isfinite(current)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(actual[valid] - current[valid]))))


def _map_oof(panel_times: pd.DatetimeIndex, oof: pd.DataFrame) -> dict[str, np.ndarray]:
    rows = panel_times.get_indexer(pd.DatetimeIndex(pd.to_datetime(oof["time"], utc=True)))
    if np.any(rows < 0):
        raise ValueError("OOF timestamp absent from observation panel")
    columns = oof["layer"].map({2: 0, 3: 1, 4: 2}).to_numpy()
    if pd.isna(columns).any():
        raise ValueError("OOF contains a non-target layer")
    columns = columns.astype(int)
    shape = (len(panel_times), len(TARGET_LAYERS))
    result: dict[str, np.ndarray] = {
        "truth": np.full(shape, np.nan),
        "adaptive": np.full(shape, np.nan),
        "physical": np.full(shape, np.nan),
        "block": np.full(shape, "", dtype=object),
    }
    result["truth"][rows, columns] = oof["truth"].to_numpy(float)
    result["adaptive"][rows, columns] = oof["adaptive_prediction"].to_numpy(float)
    result["physical"][rows, columns] = oof["physical_prediction"].to_numpy(float)
    result["block"][rows, columns] = oof["block"].to_numpy(str)
    return result


def _pooled_skill(
    truth: np.ndarray, prediction: np.ndarray, scale: np.ndarray, columns: slice
) -> float:
    current_truth = truth[:, columns] / scale[columns]
    current_prediction = prediction[:, columns] / scale[columns]
    return residual_skill_r2(current_truth.ravel(), current_prediction.ravel())


def _precheck_gate(
    fold_results: dict[str, dict[str, Any]],
    aggregate_r2: dict[str, float],
    contract: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    gates = contract["precheck"]["gates"]
    fold_skill = all(
        float(result["residual_skill"]["pooled_temperature_r2"])
        > float(gates["each_fold_pooled_temperature_residual_r2_strictly_above"])
        for result in fold_results.values()
    )
    layer_skill = all(
        aggregate_r2[f"temp_l{layer}"]
        > float(gates["aggregate_each_temperature_layer_residual_r2_strictly_above"])
        for layer in TARGET_LAYERS
    )
    rank = all(
        result["observability"]["rank"] == result["observability"]["state_dimension"]
        for result in fold_results.values()
    )
    condition = all(
        result["observability"]["condition"] <= float(gates["observability_condition_max"])
        for result in fold_results.values()
    )
    support = all(
        result["support"]["validation_supported_share"] >= float(gates["support_share_min"])
        for result in fold_results.values()
    )
    coverage = all(
        result["coverage"]["two_public_temperature_share"]
        >= float(gates["two_public_temp_coverage_min"])
        for result in fold_results.values()
    )
    checks = {
        "each_fold_pooled_temperature_residual_r2_positive": fold_skill,
        "aggregate_each_temperature_layer_residual_r2_positive": layer_skill,
        "observability_full_rank": rank,
        "observability_condition_stable": condition,
        "support_share": support,
        "public_temperature_coverage": coverage,
    }
    return all(checks.values()), checks


def _conditional_rts(
    panel: Any,
    oof: pd.DataFrame,
    mapped: dict[str, np.ndarray],
    fold_cache: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    config: TideRTSConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate = np.full((len(panel.times), len(TARGET_LAYERS)), np.nan)
    support_matrix = np.zeros_like(candidate, dtype=bool)
    uncertainty = np.full_like(candidate, np.nan)
    diagnostics: dict[str, Any] = {}
    for name, cached in fold_cache.items():
        split = cached["split"]
        encoder: PublicFactorEncoder = cached["encoder"]
        regressor: ResidualRegressor = cached["regressor"]
        labels = cached["labels"]
        state, observed_share = encoder.transform(panel.public_values, panel.times)
        public_standardized = encoder.standardized_public(panel.public_values)
        public_mask = np.isfinite(panel.public_values[:, encoder.selected])
        target_observation = regressor.normalized_observations(labels)
        target_mask = np.isfinite(labels) & split.training[:, None]
        observations = np.column_stack((public_standardized, target_observation))
        observed = np.column_stack((public_mask, target_mask))
        loading = np.vstack((encoder.public_loading, regressor.normalized_loading()))
        noise = np.r_[encoder.public_noise, np.square(regressor.residual_scale)]

        smooth_mean = np.full_like(state, np.nan)
        smooth_covariance = np.full(
            (len(panel.times), encoder.state_dimension, encoder.state_dimension), np.nan
        )
        for start, stop in cadence_segments(panel.times):
            result = kalman_rts_smoother(
                observations[start:stop],
                observed[start:stop],
                loading,
                noise,
                encoder.transition,
                encoder.process_covariance,
            )
            smooth_mean[start:stop] = result.mean
            smooth_covariance[start:stop] = result.covariance
        correction = regressor.predict(smooth_mean)[:, :3]
        posterior = posterior_target_sd(
            smooth_covariance,
            regressor.normalized_loading()[:3],
            regressor.residual_scale[:3],
            regressor.label_scale[:3],
        )
        public_count = np.isfinite(panel.public_temp).sum(axis=1)
        supported = (
            split.validation[:, None]
            & (public_count >= 2)[:, None]
            & (observed_share >= config.minimum_feature_coverage)[:, None]
            & (posterior <= config.posterior_sd_scale_max * regressor.label_scale[:3])
        )
        base = mapped["adaptive"]
        current = exact_fallback(base, correction, supported & np.isfinite(base))
        selected = split.validation[:, None] & np.isfinite(base)
        candidate[selected] = current[selected]
        support_matrix[selected] = supported[selected]
        uncertainty[selected] = posterior[selected]
        diagnostics[name] = {
            "scored_rows": int(selected.sum()),
            "supported_rows": int((supported & selected).sum()),
            "supported_share": float((supported & selected).sum() / max(selected.sum(), 1)),
            "posterior_sd_median": float(np.nanmedian(posterior[selected])),
            "posterior_sd_p95": float(np.nanquantile(posterior[selected], 0.95)),
        }

    rows = panel.times.get_indexer(pd.DatetimeIndex(pd.to_datetime(oof["time"], utc=True)))
    columns = oof["layer"].map({2: 0, 3: 1, 4: 2}).to_numpy(int)
    current = candidate[rows, columns]
    if not np.isfinite(current).all():
        raise AssertionError("conditional RTS did not cover every frozen OOF row")
    frame = oof.copy()
    frame["candidate_unprojected"] = current
    frame["supported"] = support_matrix[rows, columns]
    frame["posterior_sd"] = uncertainty[rows, columns]

    endpoints = public_endpoint_frame(panel_to_observations(panel))
    projected = project_profiles_vectorized(frame, current, endpoints)
    projected_values = np.where(frame["supported"].to_numpy(bool), projected.prediction, current)
    frame["candidate"] = projected_values
    diagnostics["projection"] = projected.diagnostics()
    return frame, diagnostics


def panel_to_observations(panel: Any) -> pd.DataFrame:
    """Minimal public endpoint long frame used only by the label-blind projection."""

    parts: list[pd.DataFrame] = []
    for column, layer in enumerate(PUBLIC_LAYERS):
        parts.append(
            pd.DataFrame(
                {"time": panel.times, "layer": layer, "temp": panel.public_temp[:, column]}
            )
        )
    return pd.concat(parts, ignore_index=True)


def _candidate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    def cut(selected: np.ndarray) -> dict[str, float | int]:
        truth = frame.loc[selected, "truth"].to_numpy(float)
        frozen = frame.loc[selected, "adaptive_prediction"].to_numpy(float)
        candidate = frame.loc[selected, "candidate"].to_numpy(float)
        frozen_rmse = _rmse(truth, frozen)
        candidate_rmse = _rmse(truth, candidate)
        return {
            "rows": int(selected.sum()),
            "frozen_rmse": frozen_rmse,
            "candidate_rmse": candidate_rmse,
            "delta_rmse": candidate_rmse - frozen_rmse,
        }

    all_rows = np.ones(len(frame), dtype=bool)
    return {
        **cut(all_rows),
        "by_block": {
            name: cut(frame["block"].eq(name).to_numpy())
            for name in frame["block"].drop_duplicates()
        },
        "by_layer": {
            str(layer): cut(frame["layer"].eq(layer).to_numpy()) for layer in TARGET_LAYERS
        },
    }


def _paired_day_bootstrap(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, float | int]:
    truth = frame["truth"].to_numpy(float)
    frozen = frame["adaptive_prediction"].to_numpy(float)
    candidate = frame["candidate"].to_numpy(float)
    day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    unique = np.unique(day)
    groups = [np.flatnonzero(day == value) for value in unique]
    generator = np.random.default_rng(seed)
    delta = np.empty(replicates)
    for replicate in range(replicates):
        selected = generator.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[index] for index in selected])
        delta[replicate] = _rmse(truth[rows], candidate[rows]) - _rmse(truth[rows], frozen[rows])
    return {
        "replicates": int(replicates),
        "kst_days": int(len(unique)),
        "delta_rmse": _rmse(truth, candidate) - _rmse(truth, frozen),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0)),
    }


def _promotion(
    metrics: dict[str, Any], bootstrap: dict[str, Any], contract: dict[str, Any]
) -> dict[str, bool]:
    gate = contract["promotion_gate"]
    by_block = metrics["by_block"]
    checks = {
        "candidate_rmse": metrics["candidate_rmse"] <= float(gate["candidate_rmse_max"]),
        "paired_ci90_high": bootstrap["ci90_high"] < float(gate["paired_ci90_high_max"]),
        "improved_blocks": sum(value["delta_rmse"] < 0 for value in by_block.values())
        >= int(gate["minimum_improved_blocks"]),
        "worst_block": max(value["delta_rmse"] for value in by_block.values())
        <= float(gate["worst_block_delta_rmse_max"]),
        "layer4_non_degrade": metrics["by_layer"]["4"]["delta_rmse"]
        <= float(gate["layer4_delta_rmse_max"]),
    }
    return checks


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = _load_contract(args.config)
    config = _model_config(contract)
    output_dir = Path(contract["outputs"]["directory"])
    status_path = Path(contract["outputs"]["status"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _status(
        status_path,
        progress=2,
        phase="contract",
        detail="고정 계약과 frozen SHA 검증 중",
        started=started,
    )
    data_dir = resolve_data_dir(args.data_dir)
    oof, frozen = _validate_frozen(contract)
    _status(
        status_path,
        progress=10,
        phase="data",
        detail="원본 P2 계약 검사와 public-only panel 구성 중",
        started=started,
    )
    data = load_p2_data(data_dir)
    panel = build_tide_panel(data.observations)
    mapped = _map_oof(panel.times, oof)
    observed_truth = panel.target_temp
    valid_truth = np.isfinite(mapped["truth"])
    if not np.allclose(
        observed_truth[valid_truth], mapped["truth"][valid_truth], rtol=0, atol=1e-12
    ):
        raise ValueError("OOF labels do not reconcile to immutable observations")

    fold_results: dict[str, dict[str, Any]] = {}
    fold_cache: dict[str, dict[str, Any]] = {}
    aggregate_truth: dict[str, list[np.ndarray]] = {label: [] for label in OUTPUT_LABELS}
    aggregate_prediction: dict[str, list[np.ndarray]] = {label: [] for label in OUTPUT_LABELS}
    blocks = contract["validation"]["blocks"]
    for number, (name, interval) in enumerate(blocks.items(), start=1):
        progress = 15 + (number - 1) * 20
        _status(
            status_path,
            progress=progress,
            phase=f"precheck_{name}",
            detail=f"{name}: 61일 temp+psal 동시 mask, ±7일 purge precheck",
            started=started,
        )
        split = outer_split(
            panel.times,
            interval[0],
            interval[1],
            purge_days=int(contract["problem_contract"]["purge_days"]),
        )
        expected_block = mapped["block"] == name
        if np.any(expected_block & ~split.validation[:, None]):
            raise ValueError(f"{name} frozen OOF rows escape the preregistered mask")
        train_public = split.training & (np.isfinite(panel.public_temp).sum(axis=1) >= 2)
        encoder = PublicFactorEncoder.fit(
            panel.public_values,
            panel.public_feature_names,
            panel.times,
            train_public,
            config=config,
        )
        state, observed_share = encoder.transform(panel.public_values, panel.times)
        query_depth = fold_target_depths(panel, split.training)
        psal_baseline = actual_depth_interpolation(
            panel.public_psal, panel.public_depth, query_depth
        )
        temp_residual = mapped["truth"] - mapped["adaptive"]
        psal_residual = panel.target_psal - psal_baseline
        labels = np.column_stack((temp_residual, psal_residual))
        if np.isfinite(labels[split.validation, :3]).any():
            # Labels remain in a separate array for scoring, but training must never select them.
            if np.any(split.training & split.validation):
                raise AssertionError("outer label leakage")
        regressor = ResidualRegressor.fit(state, labels, split.training, alpha=config.ridge_alpha)
        prediction = regressor.predict(state)

        per_output: dict[str, dict[str, float | int]] = {}
        for column, label in enumerate(OUTPUT_LABELS):
            if column < 3:
                selected = expected_block[:, column]
            else:
                selected = split.validation & np.isfinite(labels[:, column])
            truth = labels[selected, column]
            current = prediction[selected, column]
            skill = residual_skill_r2(truth, current)
            per_output[label] = {"rows": int(selected.sum()), "residual_r2": skill}
            aggregate_truth[label].append(truth)
            aggregate_prediction[label].append(current)

        validation_temp = expected_block
        pooled_temp_r2 = residual_skill_r2(
            (labels[:, :3] / regressor.label_scale[:3])[validation_temp],
            (prediction[:, :3] / regressor.label_scale[:3])[validation_temp],
        )
        validation_all = np.column_stack(
            (
                validation_temp,
                np.broadcast_to(split.validation[:, None], (len(panel.times), 3))
                & np.isfinite(labels[:, 3:]),
            )
        )
        pooled_joint_r2 = residual_skill_r2(
            (labels / regressor.label_scale)[validation_all],
            (prediction / regressor.label_scale)[validation_all],
        )
        observability = observability_diagnostics(
            encoder.transition,
            encoder.public_loading,
            encoder.public_noise,
            horizon_steps=config.gramian_horizon_steps,
            rank_tolerance=config.gramian_rank_tolerance,
        )
        support = support_diagnostics(
            state,
            train_public,
            split.validation,
            quantile=config.support_quantile,
        )
        public_count = np.isfinite(panel.public_temp[split.validation]).sum(axis=1)
        coverage = {
            "validation_rows": int(split.validation.sum()),
            "two_public_temperature_share": float(np.mean(public_count >= 2)),
            "selected_feature_observed_share_median": float(
                np.median(observed_share[split.validation])
            ),
            "selected_feature_observed_share_p05": float(
                np.quantile(observed_share[split.validation], 0.05)
            ),
        }
        m2 = m2_relationship_diagnostics(
            panel.times,
            panel.public_temp,
            labels,
            split.training,
            window_days=tuple(int(value) for value in contract["precheck"]["m2_windows_days"]),
        )
        fold_results[name] = {
            "interval_kst": interval,
            "purge_days": int(contract["problem_contract"]["purge_days"]),
            "training_rows": int(split.training.sum()),
            "validation_rows": int(split.validation.sum()),
            "simultaneous_target_mask": {
                "layers": list(TARGET_LAYERS),
                "variables": ["temp", "psal"],
            },
            "residual_skill": {
                "pooled_temperature_r2": pooled_temp_r2,
                "pooled_joint_standardized_r2": pooled_joint_r2,
                "by_output": per_output,
            },
            "observability": observability,
            "m2_relationship": m2,
            "support": support,
            "coverage": coverage,
            "target_depth_medians_from_outer_train": query_depth[0].tolist(),
        }
        fold_cache[name] = {
            "split": split,
            "encoder": encoder,
            "regressor": regressor,
            "labels": labels,
        }

    aggregate_r2 = {
        label: residual_skill_r2(
            np.concatenate(aggregate_truth[label]), np.concatenate(aggregate_prediction[label])
        )
        for label in OUTPUT_LABELS
    }
    precheck_passed, precheck_checks = _precheck_gate(fold_results, aggregate_r2, contract)
    _status(
        status_path,
        progress=78,
        phase="precheck_decision",
        detail=(
            "구조 gate 통과; conditional RTS 실행"
            if precheck_passed
            else "구조 gate 실패; preregistered NO-GO로 RTS 실행 중단"
        ),
        started=started,
    )
    result: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": EXPECTED_EXPERIMENT,
        "research_only": True,
        "adaptive_after_prior_outer_exposure": True,
        "fresh_holdout_claimed": False,
        "external_values_used": False,
        "uploaded": False,
        "submission_created": False,
        "frozen_inputs": frozen,
        "precheck": {
            "passed": precheck_passed,
            "checks": precheck_checks,
            "aggregate_residual_r2": aggregate_r2,
            "by_block": fold_results,
        },
        "reference_metrics": {
            "adaptive_proxy_rmse": float(
                contract["frozen_inputs"]["adaptive_oof"]["expected_rmse"]
            ),
            "physical_projection_rmse": float(
                contract["frozen_inputs"]["physical_oof"]["expected_rmse"]
            ),
        },
        "full_model_executed": False,
        "decision": "NO_GO_PRECHECK",
    }
    if precheck_passed:
        _status(
            status_path,
            progress=82,
            phase="conditional_rts",
            detail="고정 joint T/S + M2 Kalman/RTS 한 설정 실행",
            started=started,
            full_model_executed=True,
        )
        candidate, rts_diagnostics = _conditional_rts(
            panel, oof, mapped, fold_cache, contract, config
        )
        metrics = _candidate_metrics(candidate)
        bootstrap = _paired_day_bootstrap(
            candidate,
            replicates=int(contract["bootstrap"]["replicates"]),
            seed=int(contract["bootstrap"]["seed"]),
        )
        promotion = _promotion(metrics, bootstrap, contract)
        result.update(
            {
                "full_model_executed": True,
                "conditional_rts": rts_diagnostics,
                "metrics": metrics,
                "paired_kst_day_bootstrap": bootstrap,
                "promotion_checks": promotion,
                "decision": (
                    "PROMOTE_LOCAL_RESEARCH_CANDIDATE_NO_UPLOAD"
                    if all(promotion.values())
                    else "REJECT_CONDITIONAL_RTS_KEEP_FROZEN"
                ),
            }
        )
        candidate_path = output_dir / "oof.parquet"
        candidate.loc[
            :,
            [
                "time",
                "layer",
                "truth",
                "block",
                "physical_prediction",
                "adaptive_prediction",
                "candidate",
                "supported",
                "posterior_sd",
            ],
        ].to_parquet(candidate_path, index=False)
        result["artifacts"] = {
            "oof": {
                "path": str(candidate_path),
                "sha256": _sha256(candidate_path),
                "rows": len(candidate),
            }
        }

    result["elapsed_seconds"] = time.perf_counter() - started
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    manifest = {
        "experiment_id": EXPECTED_EXPERIMENT,
        "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        "config": {"path": str(args.config), "sha256": _sha256(args.config)},
        "full_model_executed": result["full_model_executed"],
        "submission_created": False,
        "external_values_used": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _status(
        status_path,
        progress=100,
        phase="complete",
        detail=result["decision"],
        started=started,
        status="complete",
        full_model_executed=bool(result["full_model_executed"]),
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/p2_tide_rts_v1.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
