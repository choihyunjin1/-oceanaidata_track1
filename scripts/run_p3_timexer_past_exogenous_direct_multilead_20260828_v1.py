"""Guarded local-only runner for the P3 past-exogenous direct TimeXer v1."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from p3_wave.data import LEADS, select_independent_validation
from p3_wave.timexer_direct_multilead import (
    DirectTimeXerConfig,
    PastExogenousDirectTimeXer,
    fit_hourly_statistics,
    persistence_additive_prediction,
    promotion_gates,
    sha256_file,
)

EXPERIMENT_ID = "p3_timexer_past_exogenous_direct_multilead_20260828_v1"
DEFAULT_CONFIG = Path(
    "configs/experiments/p3_timexer_past_exogenous_direct_multilead_20260828_v1.json"
)
PAIR_KEYS = ["fold", "anchor_id", "station", "lead_h"]
STRUCTURAL_COLUMNS = [
    "anchor_id",
    "station",
    "anchor_time",
    "grid_position",
    "current_hs",
]
PROTECTED_FILES = (
    "src/p3_wave/deep.py",
    "src/p3_wave/revin_patch.py",
    "src/p3_wave/chronos2_transfer.py",
    "src/p3_wave/era5_context_transfer.py",
    "src/p3_wave/era5_safe_advantage_router.py",
    "src/p3_wave/tsmixer_residual.py",
    "configs/experiments/p3_chronos2_transfer_20260828_v1.json",
    "configs/experiments/p3_chronos2_full_nested_20260828_v2.json",
    "configs/experiments/p3_era5_context_transfer_dependency_recovery_20260828_v2.json",
    "configs/experiments/p3_era5_incumbent_safe_advantage_router_20260828_v1.json",
    "configs/experiments/p3_tsmixer_observed_residual_20260828_v1.json",
    "reports/p3_era5_context_transfer_terminal_20260828.json",
)


def _load_base_runner() -> ModuleType:
    path = Path(__file__).with_name("run_p3_tsmixer_observed_residual_20260828_v1.py")
    spec = importlib.util.spec_from_file_location("_p3_tsmixer_runner_reuse", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen nested runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ObservedResidualTSMixer = PastExogenousDirectTimeXer
    module.TSMixerConfig = DirectTimeXerConfig
    module.fit_hourly_statistics = fit_hourly_statistics
    return module


BASE = _load_base_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resource-clearance", default="")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protected_hashes() -> dict[str, str]:
    return {path: sha256_file(path) for path in PROTECTED_FILES if Path(path).is_file()}


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != (
        "p3.timexer_past_exogenous_direct_multilead.preregistration.v1"
    ):
        raise ValueError("unexpected config schema")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected experiment id")
    if config.get("status") != "LOCAL_EXECUTION_AUTHORIZED_PENDING_GPU_RESOURCE_COORDINATION":
        raise ValueError("local execution authorization is missing")
    authorization = config["authorization"]
    if authorization != {
        "approved_at_kst": "2026-08-28T11:42:39.4927006+09:00",
        "scope": "single_bounded_local_only_nested_run",
        "full_gpu_launch_requires_resource_coordination": True,
        "official_context_or_submission_access_authorized": False,
    }:
        raise ValueError("authorization contract changed")
    inputs = config["inputs"]
    if inputs["future_weather_allowed"] is not False:
        raise ValueError("future weather must remain forbidden")
    if inputs["absolute_time_allowed"] is not False:
        raise ValueError("absolute time must remain forbidden")
    if inputs["cross_case_or_future_station_context_allowed"] is not False:
        raise ValueError("cross-case context must remain forbidden")
    if inputs["era5_286_feature_experiment_mutation_allowed"] is not False:
        raise ValueError("frozen ERA5 lineage must remain read-only")

    expected_model = {
        "family": "TimeXer_style_asymmetric_endogenous_exogenous_encoder",
        "endogenous_attention": "patch_self_attention",
        "exogenous_attention": "variate_cross_attention",
        "hourly_points": 49,
        "derived_channels": 12,
        "endogenous_channels": ["hs"],
        "exogenous_channels": [
            "tp",
            "hmax",
            "wspd",
            "gust",
            "relh",
            "caph",
            "airt",
            "wvdir_sin",
            "wvdir_cos",
            "wdir_sin",
            "wdir_cos",
        ],
        "patch_length": 7,
        "patch_stride": 7,
        "patch_count": 7,
        "d_model": 64,
        "attention_heads": 4,
        "encoder_layers": 2,
        "feedforward_width": 128,
        "dropout": 0.1,
        "station_embedding": 8,
        "output_leads_h": [3, 6, 9, 12, 18, 24],
        "output_mode": "direct_joint_residual",
        "skip": "additive_persistence_current_hs_only",
        "router_allowed": False,
        "posthoc_lead_selection_allowed": False,
        "seeds": [142857, 271828, 314159],
    }
    if config["model"] != expected_model:
        raise ValueError("model contract changed")
    training = config["training"]
    if training != {
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0002,
        "batch_size": 256,
        "maximum_epochs": 80,
        "patience": 10,
        "minimum_inner_improvement_m": 0.0001,
        "minimum_inner_persistence_improvement_m": 0.0001,
        "minimum_inner_improved_seeds_per_fold": 2,
        "precision": "bf16_amp_on_cuda",
        "gradient_clip_norm": 1.0,
        "loss": "uniform_six_lead_residual_mse",
        "maximum_single_fold_seed_seconds": 1200,
        "maximum_total_seconds": 10800,
        "result_based_hyperparameter_changes": False,
    }:
        raise ValueError("training contract changed")
    validation = config["validation"]
    if validation["outer_folds"] != 3 or validation["outer_embargo_hours"] != 78:
        raise ValueError("outer validation contract changed")
    if validation["inner_window_days"] != 60 or validation["inner_gap_hours"] != 78:
        raise ValueError("inner validation contract changed")
    if validation["outer_prediction_sealed_before_truth_attach"] is not True:
        raise ValueError("truth-late seal is required")
    if config["promotion_gate"] != {
        "pooled_delta_rmse_m_max": -0.005,
        "improved_folds_min": 2,
        "fold_count": 3,
        "case_bootstrap_ci90_upper_lt": 0.0,
        "worst_station_regression_m_max": 0.01,
        "lead_12_18_24_non_regression_required": True,
    }:
        raise ValueError("promotion gate changed")
    policy = config["execution_policy"]
    if policy["local_execution_authorized"] is not True:
        raise ValueError("local execution must be explicitly authorized")
    if policy["full_gpu_launch_requires_resource_coordination"] is not True:
        raise ValueError("GPU resource coordination guard changed")
    if any(
        policy[key] is not False
        for key in (
            "official_input_access_authorized",
            "official_candidate_csv_authorized",
            "official_upload_authorized",
        )
    ):
        raise ValueError("official access must remain unauthorized")
    if config["artifact_dir"] != f"artifacts/{EXPERIMENT_ID}":
        raise ValueError("artifact namespace changed")


def _validate_bindings(config: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, binding in config["bindings"].items():
        path = Path(binding["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing binding {name}: {path}")
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise RuntimeError(f"binding hash mismatch for {name}: {actual}")
        resolved[name] = path
        hashes[name] = actual
    sequence = np.load(resolved["train_sequences"], mmap_mode="r")
    station = np.load(resolved["train_station"], mmap_mode="r")
    anchors = pq.ParquetFile(resolved["train_anchors"]).metadata
    if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (289, 10):
        raise RuntimeError("train sequence shape changed")
    if station.shape != (sequence.shape[0],):
        raise RuntimeError("station code shape changed")
    if anchors.num_rows != sequence.shape[0]:
        raise RuntimeError("anchor row count differs from sequence count")
    return {
        "hashes": hashes,
        "train_cases": int(sequence.shape[0]),
        "sequence_shape": list(map(int, sequence.shape)),
        "station_shape": list(map(int, station.shape)),
        "anchor_rows": int(anchors.num_rows),
    }


def check_only(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    binding = _validate_bindings(config)
    artifact = Path(config["artifact_dir"])
    if artifact.exists():
        raise FileExistsError("experiment artifact already exists")
    model_config = DirectTimeXerConfig()
    model_config.validate()
    empty = np.zeros(12, dtype=np.float32)
    unit = np.ones(12, dtype=np.float32)
    model = PastExogenousDirectTimeXer(empty, unit, model_config)
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "CHECK_ONLY_PASS",
        "config_sha256": sha256_file(config_path),
        "bindings": binding,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "artifact_exists": False,
        "writes": 0,
        "official_inputs_read": False,
        "protected_lineage_hashes": _protected_hashes(),
    }


def smoke(config_path: Path, config: dict[str, Any], device_name: str) -> dict[str, Any]:
    check = check_only(config_path, config)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(20260828)
    device = torch.device(device_name)
    sequence_path = Path(config["bindings"]["train_sequences"]["path"])
    station_path = Path(config["bindings"]["train_station"]["path"])
    raw = np.asarray(
        np.load(sequence_path, mmap_mode="r")[:4], dtype=np.float32
    ).copy()
    station_values = np.asarray(
        np.load(station_path, mmap_mode="r")[:4], dtype=np.int64
    ).copy()
    center, scale = fit_hourly_statistics(raw)
    model = PastExogenousDirectTimeXer(center, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0002)
    values = torch.from_numpy(raw).to(device)
    station = torch.from_numpy(station_values).to(device)
    target = torch.zeros((4, len(LEADS)), dtype=torch.float32, device=device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(values, station)
    loss = torch.mean(torch.square(prediction - target))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    elapsed = time.perf_counter() - started
    if prediction.shape != (4, len(LEADS)) or not torch.isfinite(prediction).all():
        raise RuntimeError("smoke produced an invalid prediction")
    return {
        **check,
        "status": "CPU_SMOKE_PASS" if device.type == "cpu" else "CUDA_SMOKE_PASS",
        "smoke_device": str(device),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "runtime_seconds": float(elapsed),
        "input_source": "read_only_train_sequence_cache_first_four_without_targets",
        "writes": 0,
    }


def _blind_seed_frame(
    anchors: pd.DataFrame,
    ids: np.ndarray,
    absolute_prediction: np.ndarray,
    *,
    fold: str,
    seed: int,
) -> pd.DataFrame:
    lookup = anchors.set_index("anchor_id")
    rows: list[dict[str, Any]] = []
    for row_number, anchor_id in enumerate(ids):
        anchor = lookup.loc[int(anchor_id)]
        for lead_number, lead in enumerate(LEADS):
            rows.append(
                {
                    "fold": fold,
                    "seed": int(seed),
                    "anchor_id": int(anchor_id),
                    "anchor_time": pd.Timestamp(anchor["anchor_time"]),
                    "station": str(anchor["station"]),
                    "lead_h": int(lead),
                    "current_hs": float(anchor["current_hs"]),
                    "timexer_prediction": float(
                        absolute_prediction[row_number, lead_number]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(PAIR_KEYS + ["seed"]).any():
        raise RuntimeError("duplicate blind prediction keys")
    if any("target" in column or "truth" in column for column in frame):
        raise RuntimeError("blind prediction frame exposes truth")
    return frame


def _terminal_inner_no_go(
    *,
    config_path: Path,
    config: dict[str, Any],
    output: Path,
    attempt_path: Path,
    vault: Any,
    fold: str,
    records: list[dict[str, Any]],
    protected_before: dict[str, str],
    started: float,
) -> dict[str, Any]:
    protected_after = _protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("protected P3 lineage changed")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "TERMINAL_NO_GO_INNER_PERSISTENCE",
        "failed_fold": fold,
        "fold_seed_records": records,
        "config_sha256": sha256_file(config_path),
        "attempt_lock_sha256": sha256_file(attempt_path),
        "outer_truth_open_count": int(vault.outer_open_count),
        "target_access_log": vault.access_log,
        "official_inputs_read": False,
        "candidate_csv_created": False,
        "uploaded": False,
        "protected_lineage_hashes_before_after_identical": True,
        "protected_lineage_hashes": protected_after,
        "runtime_seconds_total": float(time.perf_counter() - started),
    }
    result_path = output / "result.json"
    BASE._atomic_json(result_path, result)
    BASE._atomic_json(
        output / "manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "status": result["status"],
            "config_sha256": sha256_file(config_path),
            "result_sha256": sha256_file(result_path),
            "official_inputs_read": False,
            "uploaded": False,
        },
    )
    return result


def execute(config_path: Path, config: dict[str, Any], resource_clearance: str) -> dict[str, Any]:
    started = time.perf_counter()
    _validate_config(config)
    if resource_clearance != "P1_IDLE_CONFIRMED":
        raise PermissionError("full GPU launch requires --resource-clearance P1_IDLE_CONFIRMED")
    if not torch.cuda.is_available():
        raise RuntimeError("bounded nested execution requires CUDA")
    binding_check = _validate_bindings(config)
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    protected_before = _protected_hashes()
    output = Path(config["artifact_dir"])
    if output.exists():
        raise FileExistsError("experiment namespace exists; one-shot rerun forbidden")
    output.mkdir(parents=True, exist_ok=False)
    attempt_path = output / "ATTEMPT_LOCK.json"
    BASE._atomic_json(
        attempt_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "started_at_utc": BASE._now(),
            "config_sha256": sha256_file(config_path),
            "authorization": config["authorization"],
            "resource_clearance": resource_clearance,
            "official_inputs_read": False,
            "outer_truth_read": False,
        },
    )

    paths = {name: Path(item["path"]) for name, item in config["bindings"].items()}
    anchors = pd.read_parquet(paths["train_anchors"], columns=STRUCTURAL_COLUMNS)
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True)
    values = np.load(paths["train_sequences"], mmap_mode="r")
    station = np.load(paths["train_station"], mmap_mode="r")
    incumbent = pd.read_parquet(
        paths["frozen_incumbent_oof"],
        columns=[
            "prefix_fraction",
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "incumbent_prediction",
        ],
    )
    incumbent = incumbent.loc[incumbent["prefix_fraction"].eq(1.0)].drop(
        columns="prefix_fraction"
    )
    if incumbent.duplicated(PAIR_KEYS).any():
        raise RuntimeError("incumbent comparator keys are duplicated")
    validation_ids_by_fold = {
        str(name): np.sort(group["anchor_id"].unique().astype(np.int64))
        for name, group in incumbent.groupby("fold", observed=True)
    }
    windows = [tuple(item) for item in config["validation"]["outer_windows"]]
    expected_folds = [name for name, _, _ in windows]
    if set(validation_ids_by_fold) != set(expected_folds):
        raise RuntimeError("outer fold membership changed")
    all_validation_ids = np.unique(
        np.concatenate(list(validation_ids_by_fold.values()))
    )
    vault = BASE.TargetVault(paths["train_anchors"])
    residual_target = np.full((len(anchors), len(LEADS)), np.nan, dtype=np.float32)
    fold_seed_records: list[dict[str, Any]] = []
    all_blind: list[pd.DataFrame] = []
    training = config["training"]

    for fold_index, (fold_name, start_text, _) in enumerate(windows):
        if time.perf_counter() - started > float(training["maximum_total_seconds"]):
            raise TimeoutError("total experiment exceeded the preregistered bound")
        validation_ids = validation_ids_by_fold[fold_name]
        validation_start = pd.Timestamp(start_text, tz="UTC")
        train_end = validation_start - pd.Timedelta(
            hours=int(config["validation"]["outer_embargo_hours"])
        )
        train_ids = anchors.loc[
            anchors["anchor_time"].lt(train_end), "anchor_id"
        ].to_numpy(dtype=np.int64)
        current_or_future = np.concatenate(
            [
                validation_ids_by_fold[name]
                for name in expected_folds[fold_index:]
            ]
        )
        if np.intersect1d(train_ids, current_or_future).size:
            raise RuntimeError("outer train intersects current or future validation")
        target_frame = vault.read_outer_train(
            train_ids, forbidden_validation_ids=current_or_future, fold=fold_name
        )
        residual_target[train_ids] = BASE._residual_matrix(
            target_frame, anchors, train_ids
        )
        outer_train = anchors.set_index("anchor_id").loc[train_ids]
        inner_end = outer_train["anchor_time"].max() + pd.Timedelta(minutes=20)
        inner_start = inner_end - pd.Timedelta(
            days=int(config["validation"]["inner_window_days"])
        )
        inner_ids = select_independent_validation(
            anchors,
            start=inner_start,
            end=inner_end,
            gap_hours=int(config["validation"]["inner_gap_hours"]),
        )
        inner_ids = np.intersect1d(inner_ids, train_ids)
        fit_end = inner_start - pd.Timedelta(
            hours=int(config["validation"]["inner_gap_hours"])
        )
        fit_ids = outer_train.loc[
            outer_train["anchor_time"].lt(fit_end)
        ].index.to_numpy(dtype=np.int64)
        if not len(fit_ids) or len(inner_ids) < 6:
            raise RuntimeError(f"insufficient nested cases for {fold_name}")
        persistence_inner_rmse = BASE._rmse(
            residual_target[inner_ids], np.zeros_like(residual_target[inner_ids])
        )
        selections: list[tuple[int, dict[str, Any], float]] = []
        for seed in config["model"]["seeds"]:
            seed_started = time.perf_counter()
            inner_path = output / "inner_checkpoints" / fold_name / f"seed_{seed}.pt"
            inner = BASE._select_epoch(
                values=values,
                station=station,
                residual_target=residual_target,
                fit_ids=fit_ids,
                validation_ids=inner_ids,
                config=config,
                seed=int(seed),
                device=device,
                checkpoint_path=inner_path,
                seed_started=seed_started,
            )
            selections.append((int(seed), inner, seed_started))
        improved = sum(
            inner["inner_best_rmse_m"]
            <= persistence_inner_rmse
            - float(training["minimum_inner_persistence_improvement_m"])
            for _, inner, _ in selections
        )
        if improved < int(training["minimum_inner_improved_seeds_per_fold"]):
            for seed, inner, _ in selections:
                fold_seed_records.append(
                    {
                        "fold": fold_name,
                        "seed": seed,
                        "outer_train_cases": int(len(train_ids)),
                        "inner_fit_cases": int(len(fit_ids)),
                        "inner_validation_cases": int(len(inner_ids)),
                        "inner_persistence_rmse_m": float(persistence_inner_rmse),
                        "inner": inner,
                        "outer_prediction_created": False,
                    }
                )
            return _terminal_inner_no_go(
                config_path=config_path,
                config=config,
                output=output,
                attempt_path=attempt_path,
                vault=vault,
                fold=fold_name,
                records=fold_seed_records,
                protected_before=protected_before,
                started=started,
            )
        for seed, inner, seed_started in selections:
            outer_path = output / "outer_checkpoints" / fold_name / f"seed_{seed}.pt"
            residual, outer = BASE._refit_predict(
                values=values,
                station=station,
                residual_target=residual_target,
                train_ids=train_ids,
                validation_ids=validation_ids,
                selected_epoch=int(inner["selected_epoch"]),
                config=config,
                seed=seed,
                device=device,
                checkpoint_path=outer_path,
                seed_started=seed_started,
            )
            current = anchors.set_index("anchor_id").loc[
                validation_ids, "current_hs"
            ].to_numpy(dtype=float)
            absolute = persistence_additive_prediction(current, residual)
            blind = _blind_seed_frame(
                anchors, validation_ids, absolute, fold=fold_name, seed=seed
            )
            blind_path = output / "sealed_seed_predictions" / fold_name / f"seed_{seed}.parquet"
            BASE._atomic_parquet(blind_path, blind)
            record = {
                "fold": fold_name,
                "seed": seed,
                "outer_train_cases": int(len(train_ids)),
                "inner_fit_cases": int(len(fit_ids)),
                "inner_validation_cases": int(len(inner_ids)),
                "outer_validation_cases": int(len(validation_ids)),
                "inner_persistence_rmse_m": float(persistence_inner_rmse),
                "inner": inner,
                "outer": outer,
                "blind_prediction_path": str(blind_path.resolve()),
                "blind_prediction_sha256": sha256_file(blind_path),
                "runtime_seconds": float(time.perf_counter() - seed_started),
            }
            fold_seed_records.append(record)
            all_blind.append(blind)
            BASE._atomic_json(
                output / "fold_seed_records" / f"{fold_name}_seed_{seed}.json",
                record,
            )

    seed_frame = pd.concat(all_blind, ignore_index=True)
    seed_path = output / "sealed_seed_predictions" / "all_fold_seed_predictions.parquet"
    BASE._atomic_parquet(seed_path, seed_frame)
    ensemble = (
        seed_frame.groupby(
            PAIR_KEYS + ["anchor_time", "current_hs"], as_index=False, observed=True
        )["timexer_prediction"]
        .mean()
        .sort_values(PAIR_KEYS)
        .reset_index(drop=True)
    )
    final_blind = ensemble.merge(
        incumbent[PAIR_KEYS + ["incumbent_prediction"]],
        on=PAIR_KEYS,
        validate="one_to_one",
    )
    if len(final_blind) != len(incumbent):
        raise RuntimeError("candidate and incumbent keys differ")
    final_blind["candidate_prediction"] = final_blind["timexer_prediction"]
    combined_path = output / "sealed_predictions" / "all_outer_predictions.parquet"
    BASE._atomic_parquet(combined_path, final_blind)
    seal_path = output / "PREDICTION_SEAL.json"
    seal = {
        "experiment_id": EXPERIMENT_ID,
        "sealed_at_utc": BASE._now(),
        "outer_truth_read_before_seal": False,
        "official_inputs_read": False,
        "direct_all_six_leads_without_router": True,
        "fold_seed_prediction_files": [
            {
                "fold": record["fold"],
                "seed": record["seed"],
                "path": record["blind_prediction_path"],
                "sha256": record["blind_prediction_sha256"],
                "inner_checkpoint_sha256": record["inner"]["checkpoint_sha256"],
                "outer_checkpoint_sha256": record["outer"]["checkpoint_sha256"],
            }
            for record in fold_seed_records
        ],
        "seed_predictions_sha256": sha256_file(seed_path),
        "combined_prediction_path": str(combined_path.resolve()),
        "combined_prediction_sha256": sha256_file(combined_path),
        "rows": int(len(final_blind)),
        "cases": int(final_blind["anchor_id"].nunique()),
    }
    BASE._atomic_json(seal_path, seal)

    outer_targets = vault.open_outer_once(
        all_validation_ids,
        seal_path=seal_path,
        combined_prediction_path=combined_path,
    )
    evaluated = final_blind.copy()
    target_lookup = outer_targets.set_index("anchor_id")
    evaluated["target_hs"] = np.nan
    for lead in LEADS:
        mask = evaluated["lead_h"].eq(lead)
        ids = evaluated.loc[mask, "anchor_id"].to_numpy(dtype=np.int64)
        evaluated.loc[mask, "target_hs"] = target_lookup.loc[
            ids, f"target_{lead}"
        ].to_numpy(dtype=float)
    evaluated["persistence_prediction"] = evaluated["current_hs"]
    evaluated_path = output / "evaluated_outer_predictions.parquet"
    BASE._atomic_parquet(evaluated_path, evaluated)

    metrics = {
        "candidate": BASE._metric_slices(evaluated, "candidate_prediction"),
        "incumbent": BASE._metric_slices(evaluated, "incumbent_prediction"),
        "persistence": BASE._metric_slices(evaluated, "persistence_prediction"),
    }
    metrics["delta_vs_incumbent"] = BASE._metric_delta(
        metrics["candidate"], metrics["incumbent"]
    )
    metrics["bootstrap_vs_incumbent"] = BASE._case_bootstrap(
        evaluated,
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["validation"]["bootstrap_seed"]),
    )
    seed_rmse: dict[str, float] = {}
    truth_keys = evaluated[PAIR_KEYS + ["target_hs"]]
    for seed in config["model"]["seeds"]:
        one = seed_frame.loc[seed_frame["seed"].eq(seed)].drop(columns="seed")
        one = one.merge(truth_keys, on=PAIR_KEYS, validate="one_to_one")
        seed_rmse[str(seed)] = BASE._rmse(
            one["target_hs"], one["timexer_prediction"]
        )
    metrics["candidate_rmse_by_seed_m"] = seed_rmse
    delta = metrics["delta_vs_incumbent"]
    bootstrap = metrics["bootstrap_vs_incumbent"]
    gates = promotion_gates(
        pooled_delta_m=float(delta["pooled_delta_m"]),
        fold_deltas_m=delta["by_fold_delta_m"],
        station_deltas_m=delta["by_station_delta_m"],
        lead_deltas_m=delta["by_lead_h_delta_m"],
        bootstrap_ci90_upper_m=float(bootstrap["ci90_m"][1]),
    )
    protected_after = _protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("protected P3 lineage changed")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "TERMINAL_LOCAL_GO" if gates["local_promotion_go"] else "TERMINAL_LOCAL_NO_GO",
        "config_sha256": sha256_file(config_path),
        "attempt_lock_sha256": sha256_file(attempt_path),
        "prediction_seal_sha256": sha256_file(seal_path),
        "combined_prediction_sha256": sha256_file(combined_path),
        "evaluated_outer_sha256": sha256_file(evaluated_path),
        "truth_first_read_after_prediction_seal": True,
        "outer_truth_open_count": int(vault.outer_open_count),
        "target_access_log": vault.access_log,
        "fold_seed_records": fold_seed_records,
        "metrics": metrics,
        "seed_rmse_spread_m": float(max(seed_rmse.values()) - min(seed_rmse.values())),
        "gates": gates,
        "official_inputs_read": False,
        "candidate_csv_created": False,
        "uploaded": False,
        "binding_preflight": binding_check,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "protected_lineage_hashes_before_after_identical": True,
        "protected_lineage_hashes": protected_after,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    result_path = output / "result.json"
    BASE._atomic_json(result_path, result)
    BASE._atomic_json(
        output / "manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "status": result["status"],
            "config_sha256": sha256_file(config_path),
            "result_sha256": sha256_file(result_path),
            "prediction_seal_sha256": sha256_file(seal_path),
            "evaluated_outer_sha256": sha256_file(evaluated_path),
            "checkpoint_count": int(len(fold_seed_records) * 2),
            "official_inputs_read": False,
            "candidate_csv_created": False,
            "uploaded": False,
        },
    )
    return result


def main() -> int:
    args = parse_args()
    config = _read_json(args.config)
    if args.check_only:
        result = check_only(args.config, config)
    elif args.smoke:
        result = smoke(args.config, config, args.smoke_device)
    else:
        result = execute(args.config, config, args.resource_clearance)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
