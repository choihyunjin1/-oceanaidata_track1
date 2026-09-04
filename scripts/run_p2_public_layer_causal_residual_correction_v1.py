"""Run the frozen P2 public-layer causal residual correction experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from p2_restore.data import KEYS, load_p2_data, resolve_data_dir
from p2_restore.dynamic_sigmoid_profile import effective_depth
from p2_restore.profile_projection import public_endpoint_frame
from p2_restore.public_layer_causal_residual import (
    CausalResidualSpec,
    apply_correction_and_projection,
    build_public_residual_state,
    correction_for_rows,
)
from p2_restore.submission import build_submission, validate_submission

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = ROOT / "configs/experiments/p2_public_layer_causal_residual_correction_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/p2_public_layer_causal_residual_correction_v1"
KST = ZoneInfo("Asia/Seoul")
TARGET_LAYERS = (2, 3, 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("configured path escapes repository") from error
    return path


def _load_config(path: Path) -> dict[str, Any]:
    if path.resolve() != CANONICAL_CONFIG.resolve():
        raise ValueError("only the canonical frozen config is accepted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "p2_public_layer_causal_residual_correction.v1":
        raise ValueError("unexpected config schema")
    if value.get("upload_allowed") is not False or value.get("official_score_reads") != 0:
        raise ValueError("experiment must remain local-only and official-score blind")
    return value


def _verify_pin(record: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_repo(record["path"])
    actual = _sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"input pin mismatch: {record['path']}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"input byte-size mismatch: {record['path']}")
    return {"path": record["path"], "sha256": actual, "bytes": path.stat().st_size}


def _verify_inputs(config: dict[str, Any]) -> dict[str, Any]:
    pins: dict[str, Any] = {
        "current_incumbent": _verify_pin(config["current_incumbent"]),
        "round_a_candidate": _verify_pin(
            {
                "path": config["round_a"]["candidate_path"],
                "sha256": config["round_a"]["candidate_sha256"],
            }
        ),
        "round_a_reproduction": _verify_pin(
            {
                "path": config["round_a"]["reproduction_path"],
                "sha256": config["round_a"]["reproduction_sha256"],
            }
        ),
        "round_a_stack_model": _verify_pin(
            {
                "path": config["round_a"]["saved_stack_model_path"],
                "sha256": config["round_a"]["saved_stack_model_sha256"],
            }
        ),
    }
    pins["reference_oof"] = {
        fraction: _verify_pin(record) for fraction, record in config["reference_oof"].items()
    }
    candidate = _resolve_repo(config["round_a"]["candidate_path"])
    reproduction = _resolve_repo(config["round_a"]["reproduction_path"])
    if candidate.read_bytes() != reproduction.read_bytes():
        raise ValueError("Round A candidate is not byte-identical to saved-model reproduction")
    model = joblib.load(_resolve_repo(config["round_a"]["saved_stack_model_path"]))
    pins["round_a_saved_model_load"] = {
        "status": "PASS",
        "type": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    return pins


def _spec(config: dict[str, Any]) -> CausalResidualSpec:
    value = config["correction"]
    return CausalResidualSpec(
        public_layers=tuple(int(layer) for layer in value["public_layers"]),
        rolling_hours=int(value["rolling_window_hours"]),
        cadence_minutes=int(value["cadence_minutes"]),
        minimum_samples=int(value["minimum_samples"]),
        residual_clip_c=float(value["residual_clip_c"]),
        minimum_anchors=int(value["minimum_anchors"]),
        ridge_slope_lambda=float(value["ridge_slope_lambda"]),
        correction_scale=float(value["correction_scale"]),
        correction_clip_c=float(value["correction_clip_c"]),
        maximum_anchor_span_c=float(value["maximum_anchor_span_c"]),
        depth_scale_m=float(value["depth_scale_m"]),
    )


def _target_lookup(observations: pd.DataFrame) -> pd.DataFrame:
    lookup = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["station", "layer", "time", "temp", "depth", "nominal_depth"],
    ].copy()
    lookup["_time_key"] = pd.to_datetime(lookup["time"], utc=True)
    lookup = lookup.drop(columns="time")
    if lookup.duplicated(["station", "layer", "_time_key"]).any():
        raise ValueError("target observation lookup keys are not unique")
    return lookup


def _attach_validation(frame: pd.DataFrame, lookup: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    keyed = frame.copy()
    keyed["_time_key"] = pd.to_datetime(keyed["time"], utc=True)
    merged = keyed.merge(
        lookup,
        on=["station", "layer", "_time_key"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(frame) or merged["temp"].isna().any():
        raise ValueError("architecture-matched OOF failed truth alignment")
    kst = merged["_time_key"].dt.tz_convert("Asia/Seoul")
    hidden = kst.ge(pd.Timestamp("2025-09-01", tz="Asia/Seoul")) & kst.lt(
        pd.Timestamp("2025-11-01", tz="Asia/Seoul")
    )
    if hidden.any():
        raise ValueError("validation OOF overlaps the hidden target interval")
    target_depth = effective_depth(
        merged["depth"].to_numpy(float), merged["nominal_depth"].to_numpy(float)
    )
    if not np.isfinite(target_depth).all():
        raise ValueError("validation target depth is incomplete")
    merged["truth"] = merged.pop("temp")
    return merged, target_depth


def _test_depth(test_index: pd.DataFrame, lookup: pd.DataFrame) -> np.ndarray:
    keyed = test_index.loc[:, KEYS + ["nominal_depth"]].copy()
    keyed["_time_key"] = pd.to_datetime(keyed["time"], utc=True)
    depth_lookup = lookup.drop(columns=["temp", "nominal_depth"])
    merged = keyed.merge(
        depth_lookup,
        on=["station", "layer", "_time_key"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(test_index):
        raise ValueError("test depth alignment changed row count")
    target_depth = effective_depth(
        merged["depth"].to_numpy(float), merged["nominal_depth"].to_numpy(float)
    )
    if not np.isfinite(target_depth).all():
        raise ValueError("test target depth is incomplete")
    return target_depth


def _metric_components(
    frame: pd.DataFrame, prediction: np.ndarray, layer_counts: dict[int, int]
) -> dict[str, Any]:
    values = np.asarray(prediction, dtype=np.float64)
    truth = frame["truth"].to_numpy(dtype=np.float64)
    error2 = (values - truth) ** 2
    fold_mse: dict[str, float] = {}
    fold_rmse: dict[str, float] = {}
    layer_fold_mse: dict[int, list[float]] = {layer: [] for layer in layer_counts}
    total_weight = float(sum(layer_counts.values()))
    for fold, part in frame.assign(_error2=error2).groupby("fold", sort=True):
        weighted_mse = 0.0
        for layer, count in layer_counts.items():
            selected = part["layer"].astype(int).eq(layer)
            if not selected.any():
                raise ValueError(f"fold {fold} has no layer {layer}")
            mse = float(part.loc[selected, "_error2"].mean())
            layer_fold_mse[layer].append(mse)
            weighted_mse += count / total_weight * mse
        fold_mse[str(fold)] = weighted_mse
        fold_rmse[str(fold)] = float(np.sqrt(weighted_mse))
    primary = float(np.sqrt(np.mean(list(fold_mse.values()))))
    by_layer = {
        str(layer): float(np.sqrt(np.mean(mse_values)))
        for layer, mse_values in layer_fold_mse.items()
    }
    return {"primary_rmse_c": primary, "by_fold_rmse_c": fold_rmse, "by_layer_rmse_c": by_layer}


def _paired_day_bootstrap(
    frame: pd.DataFrame,
    reference: np.ndarray,
    candidate: np.ndarray,
    layer_counts: dict[int, int],
    *,
    replicates: int,
    seed: int,
    interval: float,
) -> dict[str, Any]:
    work = frame.loc[:, ["fold", "layer", "_time_key", "truth"]].copy()
    work["day"] = work["_time_key"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    truth = work["truth"].to_numpy(float)
    work["reference_sse"] = (np.asarray(reference, dtype=float) - truth) ** 2
    work["candidate_sse"] = (np.asarray(candidate, dtype=float) - truth) ** 2
    rng = np.random.default_rng(seed)
    total_weight = float(sum(layer_counts.values()))
    reference_fold_mse: list[np.ndarray] = []
    candidate_fold_mse: list[np.ndarray] = []
    for _, fold_frame in work.groupby("fold", sort=True):
        days = sorted(fold_frame["day"].unique())
        day_position = {day: position for position, day in enumerate(days)}
        sample = rng.integers(0, len(days), size=(replicates, len(days)))
        fold_reference = np.zeros(replicates, dtype=np.float64)
        fold_candidate = np.zeros(replicates, dtype=np.float64)
        for layer, weight_count in layer_counts.items():
            part = fold_frame.loc[fold_frame["layer"].astype(int).eq(layer)]
            reference_sse = np.zeros(len(days), dtype=np.float64)
            candidate_sse = np.zeros(len(days), dtype=np.float64)
            counts = np.zeros(len(days), dtype=np.int64)
            grouped = part.groupby("day", sort=False).agg(
                reference_sse=("reference_sse", "sum"),
                candidate_sse=("candidate_sse", "sum"),
                rows=("reference_sse", "size"),
            )
            for day, row in grouped.iterrows():
                position = day_position[str(day)]
                reference_sse[position] = float(row["reference_sse"])
                candidate_sse[position] = float(row["candidate_sse"])
                counts[position] = int(row["rows"])
            sampled_count = counts[sample].sum(axis=1)
            if np.any(sampled_count == 0):
                raise ValueError("bootstrap sampled an empty layer")
            weight = weight_count / total_weight
            fold_reference += weight * reference_sse[sample].sum(axis=1) / sampled_count
            fold_candidate += weight * candidate_sse[sample].sum(axis=1) / sampled_count
        reference_fold_mse.append(fold_reference)
        candidate_fold_mse.append(fold_candidate)
    reference_rmse = np.sqrt(np.mean(np.vstack(reference_fold_mse), axis=0))
    candidate_rmse = np.sqrt(np.mean(np.vstack(candidate_fold_mse), axis=0))
    delta = candidate_rmse - reference_rmse
    tail = (1.0 - interval) / 2.0
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "unit": "KST calendar day sampled within fold",
        "interval_mass": float(interval),
        "delta_rmse_c": float(delta.mean()),
        "delta_interval_c": [float(np.quantile(delta, tail)), float(np.quantile(delta, 1 - tail))],
        "probability_candidate_improves": float(np.mean(delta < 0.0)),
    }


def _evaluate(
    config: dict[str, Any],
    observations: pd.DataFrame,
    state: pd.DataFrame,
    spec: CausalResidualSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lookup = _target_lookup(observations)
    endpoints = public_endpoint_frame(observations)
    layer_counts = {int(layer): int(count) for layer, count in config["validation"]["official_layer_counts"].items()}
    metrics: dict[str, Any] = {}
    corrections: dict[str, Any] = {}
    expected_curve = json.loads(
        (ROOT / "artifacts/p2_architecture_matched_reference_v3/reference_curve_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    expected_by_fraction = {
        f"{int(round(float(point['fraction']) * 100)):03d}": float(point["prediction_mean_metric"])
        for point in expected_curve["points"]
    }
    for index, (fraction, record) in enumerate(config["reference_oof"].items()):
        raw = pd.read_csv(
            _resolve_repo(record["path"]), dtype={"station": "string", "time": "string"}
        )
        frame, target_depth = _attach_validation(raw, lookup)
        reference = frame["prediction_mean"].to_numpy(float)
        correction = correction_for_rows(frame, target_depth, state, spec)
        projected = apply_correction_and_projection(
            frame, reference, correction.correction, endpoints
        )
        reference_metric = _metric_components(frame, reference, layer_counts)
        candidate_metric = _metric_components(frame, projected.prediction, layer_counts)
        expected = expected_by_fraction[fraction]
        if abs(reference_metric["primary_rmse_c"] - expected) > 1e-10:
            raise ValueError(f"reference metric reproduction failed for fraction {fraction}")
        by_fold_delta = {
            fold: candidate_metric["by_fold_rmse_c"][fold] - value
            for fold, value in reference_metric["by_fold_rmse_c"].items()
        }
        by_layer_delta = {
            layer: candidate_metric["by_layer_rmse_c"][layer] - value
            for layer, value in reference_metric["by_layer_rmse_c"].items()
        }
        bootstrap = _paired_day_bootstrap(
            frame,
            reference,
            projected.prediction,
            layer_counts,
            replicates=int(config["validation"]["bootstrap_replicates"]),
            seed=int(config["validation"]["bootstrap_seed"]) + index,
            interval=float(config["validation"]["bootstrap_interval"]),
        )
        metrics[fraction] = {
            "rows": int(len(frame)),
            "reference": reference_metric,
            "candidate": candidate_metric,
            "delta_rmse_c": candidate_metric["primary_rmse_c"] - reference_metric["primary_rmse_c"],
            "by_fold_delta_rmse_c": by_fold_delta,
            "by_layer_delta_rmse_c": by_layer_delta,
            "improved_fold_count": int(sum(value < 0.0 for value in by_fold_delta.values())),
            "bootstrap": bootstrap,
            "projection": projected.diagnostics(),
        }
        corrections[fraction] = correction.diagnostics
    return metrics, corrections


def _safety_gate(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    gate = config["local_safety_gate"]
    full = metrics["100"]
    checks = {
        "late_fraction_deltas_all_negative": all(
            metrics[f"{int(round(float(fraction) * 100)):03d}"]["delta_rmse_c"] < 0.0
            for fraction in gate["late_fraction_deltas_all_negative"]
        ),
        "full_fraction_material_delta": full["delta_rmse_c"]
        <= float(gate["full_fraction_delta_rmse_c_at_most"]),
        "full_fraction_ci90_upper_below_zero": full["bootstrap"]["delta_interval_c"][1]
        < float(gate["full_fraction_ci90_upper_below"]),
        "minimum_improved_full_fraction_folds": full["improved_fold_count"]
        >= int(gate["minimum_improved_full_fraction_folds"]),
        "maximum_full_fraction_layer_regression": max(full["by_layer_delta_rmse_c"].values())
        <= float(gate["maximum_full_fraction_layer_regression_c"]),
        "maximum_2024_sep_oct_regression": full["by_fold_delta_rmse_c"]["outer_2024_sep_oct"]
        <= float(gate["maximum_2024_sep_oct_regression_c"]),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _candidate_frames(
    config: dict[str, Any],
    data,
    state: pd.DataFrame,
    spec: CausalResidualSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    current_path = _resolve_repo(config["current_incumbent"]["path"])
    round_a_path = _resolve_repo(config["round_a"]["candidate_path"])
    validate_submission(current_path, data.test_index)
    validate_submission(round_a_path, data.test_index)
    current = pd.read_csv(current_path, dtype={"station": "string", "time": "string"})
    round_a = pd.read_csv(round_a_path, dtype={"station": "string", "time": "string"})
    if not current[KEYS].equals(round_a[KEYS]):
        raise ValueError("current and Round A keys differ")
    lookup = _target_lookup(data.observations)
    target_depth = _test_depth(data.test_index, lookup)
    correction = correction_for_rows(data.test_index, target_depth, state, spec)
    projected = apply_correction_and_projection(
        data.test_index,
        current["temp"].to_numpy(float),
        correction.correction,
        public_endpoint_frame(data.observations),
    )
    round_b = build_submission(data.test_index, projected.prediction)
    fallback = build_submission(
        data.test_index,
        0.5 * current["temp"].to_numpy(float) + 0.5 * round_a["temp"].to_numpy(float),
    )
    diagnostics = {
        "correction": correction.diagnostics,
        "projection": projected.diagnostics(),
        "round_b_changed_rows_vs_current": int(
            (~np.isclose(projected.prediction, current["temp"].to_numpy(float), rtol=0.0, atol=1e-12)).sum()
        ),
        "round_b_max_abs_change_vs_current_c": float(
            np.max(np.abs(projected.prediction - current["temp"].to_numpy(float)))
        ),
        "fallback_changed_rows_vs_current": int(
            (~np.isclose(fallback["temp"], current["temp"], rtol=0.0, atol=1e-12)).sum()
        ),
    }
    return round_b, fallback, diagnostics


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run(config: dict[str, Any], data_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"append-only output already exists: {output}")
    input_pins = _verify_inputs(config)
    data = load_p2_data(data_dir)
    source_pins = {
        name: {"sha256": _sha256(data_dir / name), "bytes": (data_dir / name).stat().st_size}
        for name in ("observations.csv", "test_index.csv", "sample_submission.csv", "baseline_interp.csv", "README.md")
    }
    spec = _spec(config)
    state = build_public_residual_state(data.observations, spec)
    metrics, validation_corrections = _evaluate(config, data.observations, state, spec)
    gate = _safety_gate(config, metrics)
    round_b, fallback, deployment = _candidate_frames(config, data, state, spec)
    round_b_validation = validate_submission(round_b, data.test_index)
    fallback_validation = validate_submission(fallback, data.test_index)
    structural_checks = {
        "state_time_unique": not state.duplicated("time").any(),
        "state_time_monotonic": bool(state["time"].is_monotonic_increasing),
        "validation_cell_count_15": sum(
            len(value["reference"]["by_fold_rmse_c"]) for value in metrics.values()
        )
        == 15,
        "round_b_schema": round_b_validation["rows"] == 26061,
        "fallback_schema": fallback_validation["rows"] == 26061,
        "bounded_deployment_correction": deployment["correction"]["maximum_absolute_correction_c"]
        <= spec.correction_clip_c + 1e-12,
        "hidden_target_temperature_values_not_accessed": True,
        "hidden_target_salinity_values_not_accessed": True,
    }
    structural_pass = all(structural_checks.values())
    selected = "round_b" if structural_pass and gate["passed"] else "fallback_blend50"

    output.mkdir(parents=True, exist_ok=False)
    candidate_dir = output / "candidate"
    candidate_dir.mkdir()
    round_b_path = candidate_dir / config["output"]["round_b_candidate"].split("/")[-1]
    fallback_path = candidate_dir / config["output"]["fallback_candidate"].split("/")[-1]
    round_b.to_csv(round_b_path, index=False, lineterminator="\n")
    fallback.to_csv(fallback_path, index=False, lineterminator="\n")
    round_b_file_validation = validate_submission(round_b_path, data.test_index)
    fallback_file_validation = validate_submission(fallback_path, data.test_index)
    if round_b_file_validation != round_b_validation or fallback_file_validation != fallback_validation:
        raise AssertionError("CSV roundtrip validation changed")

    round_a_qa = {
        "status": "PASS_BYTE_IDENTICAL_SAVED_MODEL_REPRODUCTION",
        "candidate": input_pins["round_a_candidate"],
        "reproduction": input_pins["round_a_reproduction"],
        "saved_model": input_pins["round_a_stack_model"],
        "saved_model_load": input_pins["round_a_saved_model_load"],
        "byte_identical": True,
    }
    selection = {
        "selected": selected,
        "reason": "LOCAL_AND_STRUCTURAL_GATES_PASS" if selected == "round_b" else "PREREGISTERED_FALLBACK_AFTER_GATE_FAIL",
        "selected_path": (
            round_b_path.relative_to(ROOT).as_posix()
            if selected == "round_b"
            else fallback_path.relative_to(ROOT).as_posix()
        ),
        "selected_sha256": _sha256(round_b_path if selected == "round_b" else fallback_path),
        "local_safety_gate": gate,
        "structural_qa": {"passed": structural_pass, "checks": structural_checks},
        "upload_performed": False,
    }
    metrics_document = {
        "schema_version": "p2_public_layer_causal_residual_correction.metrics.v1",
        "comparison_mode": config["validation"]["comparison_mode"],
        "exact_official_incumbent_comparison": False,
        "cells": metrics,
        "validation_correction_diagnostics": validation_corrections,
        "local_safety_gate": gate,
    }
    _write_json(output / "round_a_qa.json", round_a_qa)
    _write_json(output / "metrics.json", metrics_document)
    _write_json(output / "selection.json", selection)
    manifest = {
        "schema_version": "p2_public_layer_causal_residual_correction.manifest.v1",
        "experiment_id": config["experiment_id"],
        "completed_at_kst": datetime.now(KST).isoformat(),
        "config": {"path": CANONICAL_CONFIG.relative_to(ROOT).as_posix(), "sha256": _sha256(CANONICAL_CONFIG)},
        "input_pins": input_pins,
        "source_pins": source_pins,
        "data_policy": {
            "source_mutated": False,
            "hidden_target_temperature_values_accessed": 0,
            "hidden_target_salinity_values_accessed": 0,
            "official_score_reads": 0,
            "upload_performed": False,
        },
        "architecture_matched_cell_count": 15,
        "round_a_qa": round_a_qa,
        "round_b": {
            "path": round_b_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(round_b_path),
            "bytes": round_b_path.stat().st_size,
            "validation": round_b_file_validation,
            "deployment_diagnostics": deployment,
        },
        "fallback": {
            "path": fallback_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(fallback_path),
            "bytes": fallback_path.stat().st_size,
            "validation": fallback_file_validation,
            "formula": config["fallback"]["formula"],
        },
        "selection": selection,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    _write_json(output / "manifest.json", manifest)
    result = {
        "status": "COMPLETE_LOCAL_ONLY",
        "selected": selected,
        "selected_path": selection["selected_path"],
        "selected_sha256": selection["selected_sha256"],
        "round_b_sha256": manifest["round_b"]["sha256"],
        "fallback_sha256": manifest["fallback"]["sha256"],
        "full_fraction_delta_rmse_c": metrics["100"]["delta_rmse_c"],
        "full_fraction_ci90_c": metrics["100"]["bootstrap"]["delta_interval_c"],
        "local_safety_gate_passed": gate["passed"],
        "structural_qa_passed": structural_pass,
        "official_score_reads": 0,
        "upload_performed": False,
    }
    _write_json(output / "result.json", result)
    result["manifest_sha256"] = _sha256(output / "manifest.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    _verify_inputs(config)
    if not args.execute:
        print(json.dumps({"status": "CHECK_ONLY_PASS", "config_sha256": _sha256(args.config)}))
        return 0
    data_dir = resolve_data_dir(args.data_dir)
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("output must remain under the repository root") from error
    print(json.dumps(_run(config, data_dir, output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
