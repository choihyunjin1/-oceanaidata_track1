"""Exactly-once sealed P3 observed-weather TSMixer residual experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, TensorDataset

from p3_wave.data import LEADS, load_p3_data, select_independent_validation
from p3_wave.sequences import build_test_sequences
from p3_wave.tsmixer_residual import (
    ObservedResidualTSMixer,
    TSMixerConfig,
    decision_gates,
    fit_hourly_statistics,
    incumbent_preserving_blend,
    sha256_file,
)

STRUCTURAL_COLUMNS = ["anchor_id", "station", "anchor_time", "grid_position", "current_hs"]
TARGET_COLUMNS = ["anchor_id", *[f"target_{lead}" for lead in LEADS]]
PAIR_KEYS = ["fold", "anchor_id", "station", "lead_h"]
PROTECTED_FILES = (
    "src/p3_wave/deep.py",
    "src/p3_wave/revin_patch.py",
    "src/p3_wave/chronos2_transfer.py",
    "configs/experiments/p3_revin_patch_v1.json",
    "configs/experiments/p3_chronos2_transfer_20260828_v1.json",
    "configs/experiments/p3_chronos2_full_nested_20260828_v2.json",
    "reports/p3_era5_context_transfer_terminal_20260828.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/p3_tsmixer_observed_residual_20260828_v1.json",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validate_binding(binding: dict[str, str], name: str) -> Path:
    path = Path(binding["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise RuntimeError(f"{name} hash mismatch: {actual}")
    return path


def _protected_hashes() -> dict[str, str]:
    return {path: sha256_file(path) for path in PROTECTED_FILES if Path(path).is_file()}


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "1.0":
        raise ValueError("unexpected config schema")
    if config.get("experiment_id") != "p3_tsmixer_observed_residual_20260828_v1":
        raise ValueError("unexpected experiment id")
    if config.get("status") != "PREREGISTERED_BEFORE_ANY_FIT_OR_OUTER_TARGET_READ":
        raise ValueError("experiment is not in its preregistered state")
    if config["input_contract"] != {
        "context_hours": 48,
        "native_rows_10min": 289,
        "hourly_rows_inclusive": 49,
        "hourly_sampling_stride_native_rows": 6,
        "observed_channels": [
            "hs",
            "tp",
            "hmax",
            "wspd",
            "gust",
            "caph",
            "airt",
            "relh",
            "wvdir_sin",
            "wvdir_cos",
            "wdir_sin",
            "wdir_cos",
        ],
        "append_missing_mask_per_channel": True,
        "append_relative_time": True,
        "future_weather": False,
        "absolute_time_feature": False,
        "cross_case_context": False,
    }:
        raise ValueError("input contract changed")
    architecture = config["architecture"]
    expected_architecture = {
        "name": "ObservedResidualTSMixer",
        "width": 64,
        "blocks": 4,
        "time_hidden": 128,
        "feature_hidden": 128,
        "dropout": 0.1,
        "station_embedding": 8,
        "output": "six_lead_future_hs_minus_current_hs",
    }
    if architecture != expected_architecture:
        raise ValueError("architecture contract changed")
    training = config["training"]
    expected_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0002,
        "batch_size": 256,
        "maximum_epochs": 80,
        "patience": 10,
        "minimum_inner_improvement_m": 0.0001,
        "precision": "bf16_amp_on_cuda",
        "gradient_clip_norm": 1.0,
        "seeds": [20260828, 20260829, 20260830],
        "loss": "uniform_six_lead_mse",
        "result_based_hyperparameter_changes": False,
    }
    if training != expected_training:
        raise ValueError("training contract changed")
    if config["blend"] != {
        "protected_incumbent_leads_h": [3, 6, 9],
        "active_leads_h": [12, 18, 24],
        "incumbent_weight": 0.8,
        "tsmixer_weight": 0.2,
    }:
        raise ValueError("blend contract changed")
    if config["prohibitions"]["official_context_index_sample_read_before_information_go"] is not True:
        raise ValueError("official-input gate must fail closed")


class TargetVault:
    """Filtered local-target access with a single post-seal outer opening."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.access_log: list[dict[str, Any]] = []
        self.outer_open_count = 0

    def _read(self, ids: np.ndarray) -> pd.DataFrame:
        requested = np.asarray(ids, dtype=np.int64)
        if not len(requested) or len(np.unique(requested)) != len(requested):
            raise ValueError("target request IDs must be non-empty and unique")
        dataset = ds.dataset(self.path, format="parquet")
        condition = ds.field("anchor_id").isin(requested.tolist())
        frame = dataset.to_table(columns=TARGET_COLUMNS, filter=condition).to_pandas()
        if len(frame) != len(requested) or set(frame["anchor_id"]) != set(map(int, requested)):
            raise RuntimeError("filtered target read is incomplete")
        frame = frame.set_index("anchor_id").loc[requested].reset_index()
        if not np.isfinite(frame.drop(columns="anchor_id").to_numpy(dtype=float)).all():
            raise RuntimeError("target read contains non-finite values")
        return frame

    def read_outer_train(
        self, ids: np.ndarray, *, forbidden_validation_ids: np.ndarray, fold: str
    ) -> pd.DataFrame:
        if np.intersect1d(ids, forbidden_validation_ids).size:
            raise PermissionError("outer validation truth requested before prediction seal")
        frame = self._read(ids)
        self.access_log.append(
            {"purpose": "outer_train_and_inner_selection", "fold": fold, "rows": len(frame)}
        )
        return frame

    def open_outer_once(
        self, ids: np.ndarray, *, seal_path: Path, combined_prediction_path: Path
    ) -> pd.DataFrame:
        if self.outer_open_count:
            raise PermissionError("outer truth may be opened exactly once")
        seal = _read_json(seal_path)
        if seal.get("outer_truth_read_before_seal") is not False:
            raise PermissionError("prediction seal does not prove truth-late execution")
        if seal.get("combined_prediction_sha256") != sha256_file(combined_prediction_path):
            raise PermissionError("sealed combined prediction changed")
        frame = self._read(ids)
        self.outer_open_count += 1
        self.access_log.append(
            {
                "purpose": "outer_evaluation_after_all_predictions_and_hashes",
                "rows": len(frame),
                "seal_sha256": sha256_file(seal_path),
            }
        )
        return frame


def _target_matrix(frame: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    lookup = frame.set_index("anchor_id")
    return np.column_stack(
        [lookup.loc[ids, f"target_{lead}"].to_numpy(dtype=np.float32) for lead in LEADS]
    )


def _residual_matrix(frame: pd.DataFrame, anchors: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    absolute = _target_matrix(frame, ids)
    current = anchors.set_index("anchor_id").loc[ids, "current_hs"].to_numpy(dtype=np.float32)
    return absolute - current[:, None]


def _loader(
    values: np.ndarray,
    station: np.ndarray,
    targets: np.ndarray,
    ids: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(values[ids], dtype=np.float32)),
        torch.from_numpy(np.asarray(station[ids], dtype=np.int64)),
        torch.from_numpy(np.asarray(targets[ids], dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    values: np.ndarray,
    station: np.ndarray,
    ids: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        raw = torch.from_numpy(np.asarray(values[batch_ids], dtype=np.float32)).to(
            device, non_blocking=True
        )
        station_batch = torch.from_numpy(np.asarray(station[batch_ids], dtype=np.int64)).to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(raw, station_batch)
        outputs.append(output.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> None:
    model.train()
    for raw, station, target in loader:
        raw = raw.to(device, non_blocking=True)
        station = station.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(raw, station)
            loss = torch.mean(torch.square(prediction - target))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _rmse(truth: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    truth_array = np.asarray(truth, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(np.square(prediction_array - truth_array))))


def _select_epoch(
    *,
    values: np.ndarray,
    station: np.ndarray,
    residual_target: np.ndarray,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    seed_started: float,
) -> dict[str, Any]:
    training = config["training"]
    _set_seed(seed)
    center, scale = fit_hourly_statistics(values[fit_ids])
    model = ObservedResidualTSMixer(center, scale, TSMixerConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader = _loader(
        values,
        station,
        residual_target,
        fit_ids,
        batch_size=int(training["batch_size"]),
        shuffle=True,
    )
    history: list[float] = []
    best_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        _one_epoch(model, loader, optimizer, device=device)
        prediction = _predict(
            model,
            values,
            station,
            validation_ids,
            batch_size=int(training["batch_size"]),
            device=device,
        )
        score = _rmse(residual_target[validation_ids], prediction)
        history.append(score)
        if score < best_rmse - float(training["minimum_inner_improvement_m"]):
            best_rmse = score
            best_epoch = epoch
            best_state = deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if time.perf_counter() - seed_started > 1200.0:
            raise TimeoutError("single fold-seed exceeded the preregistered 20-minute limit")
        if stale >= int(training["patience"]):
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("inner checkpoint selection failed")
    _atomic_checkpoint(
        checkpoint_path,
        {
            "experiment_id": config["experiment_id"],
            "stage": "inner_best",
            "seed": seed,
            "selected_epoch": best_epoch,
            "inner_best_rmse_m": best_rmse,
            "epochs_ran": len(history),
            "history_rmse_m": history,
            "center": center,
            "scale": scale,
            "state_dict": best_state,
        },
    )
    return {
        "selected_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "inner_best_rmse_m": float(best_rmse),
        "history_rmse_m": history,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _refit_predict(
    *,
    values: np.ndarray,
    station: np.ndarray,
    residual_target: np.ndarray,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    selected_epoch: int,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    seed_started: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    training = config["training"]
    _set_seed(seed)
    center, scale = fit_hourly_statistics(values[train_ids])
    model = ObservedResidualTSMixer(center, scale, TSMixerConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader = _loader(
        values,
        station,
        residual_target,
        train_ids,
        batch_size=int(training["batch_size"]),
        shuffle=True,
    )
    for _ in range(selected_epoch):
        _one_epoch(model, loader, optimizer, device=device)
        if time.perf_counter() - seed_started > 1200.0:
            raise TimeoutError("single fold-seed exceeded the preregistered 20-minute limit")
    _atomic_checkpoint(
        checkpoint_path,
        {
            "experiment_id": config["experiment_id"],
            "stage": "outer_refit",
            "seed": seed,
            "selected_epoch": selected_epoch,
            "center": center,
            "scale": scale,
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        },
    )
    prediction = _predict(
        model,
        values,
        station,
        validation_ids,
        batch_size=int(training["batch_size"]),
        device=device,
    )
    return prediction, {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_epoch": int(selected_epoch),
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
                    "tsmixer_prediction": float(absolute_prediction[row_number, lead_number]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(PAIR_KEYS + ["seed"]).any():
        raise RuntimeError("duplicate blind seed prediction keys")
    forbidden = [column for column in frame if "target" in column or "truth" in column]
    if forbidden:
        raise RuntimeError("blind seed file exposes truth")
    return frame


def _metric_slices(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {"pooled_rmse_m": _rmse(frame["target_hs"], frame[prediction_column])}
    for dimension in ("fold", "station", "lead_h"):
        result[f"by_{dimension}_rmse_m"] = {
            str(key): _rmse(group["target_hs"], group[prediction_column])
            for key, group in frame.groupby(dimension, observed=True)
        }
    return result


def _metric_delta(candidate: dict[str, Any], comparator: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled_delta_m": candidate["pooled_rmse_m"] - comparator["pooled_rmse_m"]
    }
    for dimension in ("fold", "station", "lead_h"):
        source = f"by_{dimension}_rmse_m"
        result[f"by_{dimension}_delta_m"] = {
            key: candidate[source][key] - comparator[source][key] for key in candidate[source]
        }
    return result


def _case_bootstrap(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, Any]:
    work = frame[["anchor_id"]].copy()
    work["candidate_sq"] = np.square(frame["candidate_prediction"] - frame["target_hs"])
    work["incumbent_sq"] = np.square(frame["incumbent_prediction"] - frame["target_hs"])
    grouped = work.groupby("anchor_id", observed=True).agg(
        candidate_sq=("candidate_sq", "sum"),
        incumbent_sq=("incumbent_sq", "sum"),
        rows=("candidate_sq", "size"),
    )
    values = grouped.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for number in range(replicates):
        sampled = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        deltas[number] = np.sqrt(sampled[0] / sampled[2]) - np.sqrt(sampled[1] / sampled[2])
    return {
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "ci90_m": [float(np.quantile(deltas, 0.05)), float(np.quantile(deltas, 0.95))],
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def _full_train_and_candidate(
    *,
    config: dict[str, Any],
    values: np.ndarray,
    station: np.ndarray,
    anchors: pd.DataFrame,
    target_vault: TargetVault,
    selected_epochs: dict[int, int],
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Guarded official-input branch; caller must have proved information GO."""

    data_dir_text = os.environ.get("P3_DATA_DIR")
    if not data_dir_text:
        raise RuntimeError("P3_DATA_DIR is required only after official-information GO")
    data_dir = Path(data_dir_text).expanduser().resolve()
    for filename, expected in config["official_source_sha256_read_only_after_information_go"].items():
        path = data_dir / filename
        if sha256_file(path) != expected:
            raise RuntimeError(f"official source hash changed: {filename}")
    incumbent_binding = config["bindings"][
        "official_incumbent_submission_read_only_after_information_go"
    ]
    incumbent_path = _validate_binding(incumbent_binding, "official_incumbent_submission")

    all_ids = anchors["anchor_id"].to_numpy(dtype=np.int64)
    all_target_frame = target_vault._read(all_ids)
    residual_target = np.full((len(anchors), len(LEADS)), np.nan, dtype=np.float32)
    residual_target[all_ids] = _residual_matrix(all_target_frame, anchors, all_ids)
    test_data = load_p3_data(data_dir)
    test_sequences = build_test_sequences(test_data)
    test_ids = np.arange(len(test_sequences.values), dtype=np.int64)
    current = test_sequences.values[:, -1, 0].astype(np.float64)
    final_predictions: list[np.ndarray] = []
    final_records: list[dict[str, Any]] = []
    training = config["training"]
    for seed in training["seeds"]:
        _set_seed(int(seed))
        center, scale = fit_hourly_statistics(values[all_ids])
        model = ObservedResidualTSMixer(center, scale, TSMixerConfig()).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        loader = _loader(
            values,
            station,
            residual_target,
            all_ids,
            batch_size=int(training["batch_size"]),
            shuffle=True,
        )
        for _ in range(int(selected_epochs[int(seed)])):
            _one_epoch(model, loader, optimizer, device=device)
        checkpoint_path = output / "final_models" / f"seed_{seed}.pt"
        _atomic_checkpoint(
            checkpoint_path,
            {
                "experiment_id": config["experiment_id"],
                "stage": "full_train_after_information_go",
                "seed": int(seed),
                "selected_epoch_median": int(selected_epochs[int(seed)]),
                "center": center,
                "scale": scale,
                "state_dict": {
                    name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                },
            },
        )
        delta = _predict(
            model,
            test_sequences.values,
            test_sequences.station_code,
            test_ids,
            batch_size=int(training["batch_size"]),
            device=device,
        )
        final_predictions.append(np.clip(current[:, None] + delta, 0.0, 30.0))
        final_records.append(
            {
                "seed": int(seed),
                "epoch": int(selected_epochs[int(seed)]),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
    model_mean = np.mean(final_predictions, axis=0)
    cases = test_data.test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    model_rows: list[dict[str, Any]] = []
    for case_number, case in enumerate(cases.itertuples(index=False)):
        for lead_number, lead in enumerate(LEADS):
            model_rows.append(
                {
                    "case_id": str(case.case_id),
                    "station": str(case.station),
                    "lead_h": int(lead),
                    "tsmixer_prediction": float(model_mean[case_number, lead_number]),
                }
            )
    model_frame = pd.DataFrame(model_rows)
    incumbent = pd.read_csv(incumbent_path)
    expected_columns = ["case_id", "station", "lead_h", "hs_pred"]
    if list(incumbent.columns) != expected_columns:
        raise RuntimeError("official incumbent schema changed")
    merged = test_data.test_index.merge(
        incumbent.rename(columns={"hs_pred": "incumbent_prediction"}),
        on=["case_id", "station", "lead_h"],
        validate="one_to_one",
    ).merge(model_frame, on=["case_id", "station", "lead_h"], validate="one_to_one")
    incumbent_matrix = merged["incumbent_prediction"].to_numpy(dtype=float).reshape(-1, 6)
    tsmixer_matrix = merged["tsmixer_prediction"].to_numpy(dtype=float).reshape(-1, 6)
    merged["hs_pred"] = incumbent_preserving_blend(incumbent_matrix, tsmixer_matrix).reshape(-1)
    candidate = merged[expected_columns]
    submission_dir = Path("submissions") / config["experiment_id"]
    submission_path = submission_dir / "submission.csv"
    _atomic_parquet(output / "official_candidate_audit.parquet", merged)
    submission_dir.mkdir(parents=True, exist_ok=True)
    temporary = submission_path.with_suffix(".csv.tmp")
    candidate.to_csv(temporary, index=False)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, submission_path)
    receipt_path = submission_dir / "validator_receipt.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/validate_p3_submission.py",
            str(submission_path),
            "--data-dir",
            str(data_dir),
            "--receipt",
            str(receipt_path),
        ],
        check=True,
    )
    return {
        "created": True,
        "uploaded": False,
        "official_inputs_first_read_after_information_go": True,
        "same_anonymous_case_only": True,
        "absolute_time_reconstructed": False,
        "selected_epochs_by_seed": {str(key): int(value) for key, value in selected_epochs.items()},
        "final_models": final_records,
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": sha256_file(submission_path),
        "validator_receipt_path": str(receipt_path.resolve()),
        "validator_receipt_sha256": sha256_file(receipt_path),
    }


def execute(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    _validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("bounded execution requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    bindings = config["bindings"]
    sequence_path = _validate_binding(bindings["train_sequences"], "train_sequences")
    station_path = _validate_binding(bindings["train_station"], "train_station")
    anchor_path = _validate_binding(bindings["train_anchors"], "train_anchors")
    incumbent_path = _validate_binding(bindings["frozen_incumbent_oof"], "incumbent_oof")
    protected_before = _protected_hashes()

    output = Path(config["artifact_dir"])
    terminal_path = output / "result.json"
    if output.exists() or terminal_path.exists():
        raise FileExistsError("experiment namespace already exists; exactly-once rerun forbidden")
    output.mkdir(parents=True, exist_ok=False)
    attempt_path = output / "ATTEMPT_LOCK.json"
    _atomic_json(
        attempt_path,
        {
            "experiment_id": config["experiment_id"],
            "started_at_utc": _now(),
            "config_sha256": sha256_file(config_path),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "official_inputs_read": False,
            "outer_truth_read": False,
        },
    )

    anchors = pd.read_parquet(anchor_path, columns=STRUCTURAL_COLUMNS)
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True)
    values = np.load(sequence_path, mmap_mode="r")
    station = np.load(station_path, mmap_mode="r")
    if values.shape != (len(anchors), 289, 10) or station.shape != (len(anchors),):
        raise RuntimeError("sequence cache shape changed")
    incumbent_keys = pd.read_parquet(
        incumbent_path,
        columns=["prefix_fraction", "fold", "anchor_id", "station", "lead_h", "incumbent_prediction"],
    )
    incumbent_keys = incumbent_keys.loc[incumbent_keys["prefix_fraction"].eq(1.0)].drop(
        columns="prefix_fraction"
    )
    if incumbent_keys.duplicated(PAIR_KEYS).any():
        raise RuntimeError("frozen incumbent exact comparator keys are duplicated")
    validation_ids_by_fold = {
        str(name): np.sort(group["anchor_id"].unique().astype(np.int64))
        for name, group in incumbent_keys.groupby("fold", observed=True)
    }
    windows = [tuple(item) for item in config["outer_validation"]["windows"]]
    expected_names = [name for name, _, _ in windows]
    if set(validation_ids_by_fold) != set(expected_names):
        raise RuntimeError("incumbent fold membership changed")

    all_validation_ids = np.unique(np.concatenate(list(validation_ids_by_fold.values())))
    vault = TargetVault(anchor_path)
    training = config["training"]
    residual_target = np.full((len(anchors), len(LEADS)), np.nan, dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    seed_blind_frames: list[pd.DataFrame] = []
    for fold_name, start_text, _ in windows:
        validation_ids = validation_ids_by_fold[fold_name]
        validation_start = pd.Timestamp(start_text, tz="UTC")
        train_end = validation_start - pd.Timedelta(
            hours=int(config["outer_validation"]["embargo_hours"])
        )
        chronological_train_ids = anchors.loc[
            anchors["anchor_time"].lt(train_end), "anchor_id"
        ].to_numpy(dtype=np.int64)
        # Rolling-origin semantics: a later prefix may use observations belonging to
        # an earlier, already-predicted outer window.  The current and future outer
        # memberships remain unavailable because they occur after ``train_end``.
        # This matches the frozen Chronos nested split and preserves the preregistered
        # 60-day independent inner lattice (19/23/46 cases).
        train_ids = chronological_train_ids
        current_or_future_validation_ids = np.concatenate(
            [
                ids
                for name, ids in validation_ids_by_fold.items()
                if expected_names.index(name) >= expected_names.index(fold_name)
            ]
        )
        if np.intersect1d(train_ids, current_or_future_validation_ids).size:
            raise RuntimeError("outer train intersects current or future validation membership")
        target_frame = vault.read_outer_train(
            train_ids,
            forbidden_validation_ids=current_or_future_validation_ids,
            fold=fold_name,
        )
        residual_target[train_ids] = _residual_matrix(target_frame, anchors, train_ids)
        outer_train = anchors.set_index("anchor_id").loc[train_ids]
        inner_end = outer_train["anchor_time"].max() + pd.Timedelta(minutes=20)
        inner_start = inner_end - pd.Timedelta(days=int(config["inner_selection"]["window_days"]))
        inner_ids = select_independent_validation(
            anchors,
            start=inner_start,
            end=inner_end,
            gap_hours=int(config["inner_selection"]["gap_hours"]),
        )
        inner_ids = np.intersect1d(inner_ids, train_ids)
        fit_end = inner_start - pd.Timedelta(hours=int(config["inner_selection"]["gap_hours"]))
        fit_ids = outer_train.loc[outer_train["anchor_time"].lt(fit_end)].index.to_numpy(
            dtype=np.int64
        )
        if len(fit_ids) == 0 or len(inner_ids) < 6:
            raise RuntimeError(f"insufficient nested cases for {fold_name}")
        for seed in training["seeds"]:
            if time.perf_counter() - started > 10800.0:
                raise TimeoutError("total experiment exceeded the preregistered 3-hour limit")
            seed_started = time.perf_counter()
            inner_checkpoint = output / "inner_checkpoints" / fold_name / f"seed_{seed}.pt"
            inner = _select_epoch(
                values=values,
                station=station,
                residual_target=residual_target,
                fit_ids=fit_ids,
                validation_ids=inner_ids,
                config=config,
                seed=int(seed),
                device=device,
                checkpoint_path=inner_checkpoint,
                seed_started=seed_started,
            )
            outer_checkpoint = output / "outer_checkpoints" / fold_name / f"seed_{seed}.pt"
            delta, outer_checkpoint_record = _refit_predict(
                values=values,
                station=station,
                residual_target=residual_target,
                train_ids=train_ids,
                validation_ids=validation_ids,
                selected_epoch=int(inner["selected_epoch"]),
                config=config,
                seed=int(seed),
                device=device,
                checkpoint_path=outer_checkpoint,
                seed_started=seed_started,
            )
            current = anchors.set_index("anchor_id").loc[
                validation_ids, "current_hs"
            ].to_numpy(dtype=float)
            absolute = np.clip(current[:, None] + delta, 0.0, 30.0)
            blind = _blind_seed_frame(
                anchors,
                validation_ids,
                absolute,
                fold=fold_name,
                seed=int(seed),
            )
            blind_path = output / "sealed_seed_predictions" / fold_name / f"seed_{seed}.parquet"
            _atomic_parquet(blind_path, blind)
            seed_seconds = time.perf_counter() - seed_started
            record = {
                "fold": fold_name,
                "seed": int(seed),
                "outer_train_cases": int(len(train_ids)),
                "inner_fit_cases": int(len(fit_ids)),
                "inner_validation_cases": int(len(inner_ids)),
                "outer_validation_cases": int(len(validation_ids)),
                "rolling_origin_prior_outer_cases_in_train": int(
                    np.intersect1d(train_ids, all_validation_ids).size
                ),
                "inner": inner,
                "outer": outer_checkpoint_record,
                "blind_prediction_path": str(blind_path.resolve()),
                "blind_prediction_sha256": sha256_file(blind_path),
                "runtime_seconds": float(seed_seconds),
            }
            fold_records.append(record)
            seed_blind_frames.append(blind)
            _atomic_json(output / "fold_seed_records" / f"{fold_name}_seed_{seed}.json", record)

    all_seed_blind = pd.concat(seed_blind_frames, ignore_index=True)
    seed_combined_path = output / "sealed_seed_predictions" / "all_fold_seed_predictions.parquet"
    _atomic_parquet(seed_combined_path, all_seed_blind)
    ensemble = (
        all_seed_blind.groupby(PAIR_KEYS + ["anchor_time", "current_hs"], as_index=False, observed=True)[
            "tsmixer_prediction"
        ]
        .mean()
        .sort_values(PAIR_KEYS)
        .reset_index(drop=True)
    )
    comparator = incumbent_keys[PAIR_KEYS + ["incumbent_prediction"]]
    final_blind = ensemble.merge(comparator, on=PAIR_KEYS, validate="one_to_one")
    if len(final_blind) != len(comparator):
        raise RuntimeError("TSMixer and incumbent exact comparator keys differ")
    incumbent_matrix = final_blind["incumbent_prediction"].to_numpy(dtype=float).reshape(-1, 6)
    tsmixer_matrix = final_blind["tsmixer_prediction"].to_numpy(dtype=float).reshape(-1, 6)
    final_blind["candidate_prediction"] = incumbent_preserving_blend(
        incumbent_matrix, tsmixer_matrix, long_model_weight=0.2
    ).reshape(-1)
    if not np.array_equal(
        final_blind.loc[final_blind["lead_h"].isin((3, 6, 9)), "candidate_prediction"].to_numpy(),
        final_blind.loc[final_blind["lead_h"].isin((3, 6, 9)), "incumbent_prediction"].to_numpy(),
    ):
        raise RuntimeError("protected early leads are not bit-exact incumbent values")
    combined_path = output / "sealed_predictions" / "all_outer_predictions.parquet"
    _atomic_parquet(combined_path, final_blind)
    seal_path = output / "PREDICTION_SEAL.json"
    seal = {
        "experiment_id": config["experiment_id"],
        "sealed_at_utc": _now(),
        "outer_truth_read_before_seal": False,
        "official_inputs_read_before_gate": False,
        "fold_seed_prediction_files": [
            {
                "fold": record["fold"],
                "seed": record["seed"],
                "path": record["blind_prediction_path"],
                "sha256": record["blind_prediction_sha256"],
                "inner_checkpoint_sha256": record["inner"]["checkpoint_sha256"],
                "outer_checkpoint_sha256": record["outer"]["checkpoint_sha256"],
            }
            for record in fold_records
        ],
        "seed_combined_sha256": sha256_file(seed_combined_path),
        "combined_prediction_path": str(combined_path.resolve()),
        "combined_prediction_sha256": sha256_file(combined_path),
        "rows": int(len(final_blind)),
        "cases": int(final_blind["anchor_id"].nunique()),
    }
    _atomic_json(seal_path, seal)

    outer_targets = vault.open_outer_once(
        all_validation_ids, seal_path=seal_path, combined_prediction_path=combined_path
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
    _atomic_parquet(evaluated_path, evaluated)

    metrics = {
        "candidate": _metric_slices(evaluated, "candidate_prediction"),
        "incumbent": _metric_slices(evaluated, "incumbent_prediction"),
        "tsmixer": _metric_slices(evaluated, "tsmixer_prediction"),
        "persistence": _metric_slices(evaluated, "persistence_prediction"),
    }
    metrics["delta_vs_incumbent"] = _metric_delta(metrics["candidate"], metrics["incumbent"])
    metrics["bootstrap_vs_incumbent"] = _case_bootstrap(
        evaluated,
        replicates=int(config["evaluation"]["bootstrap_replicates"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    metrics["prediction_novelty_rms_m"] = _rmse(
        evaluated["incumbent_prediction"], evaluated["candidate_prediction"]
    )

    seed_metrics: dict[str, float] = {}
    for seed in training["seeds"]:
        seed_frame = all_seed_blind.loc[all_seed_blind["seed"].eq(seed)].drop(columns="seed")
        seed_evaluated = seed_frame.merge(comparator, on=PAIR_KEYS, validate="one_to_one")
        seed_evaluated = seed_evaluated.merge(
            evaluated[PAIR_KEYS + ["target_hs"]], on=PAIR_KEYS, validate="one_to_one"
        )
        incumbent_seed = seed_evaluated["incumbent_prediction"].to_numpy(dtype=float).reshape(-1, 6)
        tsmixer_seed = seed_evaluated["tsmixer_prediction"].to_numpy(dtype=float).reshape(-1, 6)
        seed_evaluated["candidate_prediction"] = incumbent_preserving_blend(
            incumbent_seed, tsmixer_seed, long_model_weight=0.2
        ).reshape(-1)
        seed_metrics[str(seed)] = _rmse(
            seed_evaluated["target_hs"], seed_evaluated["candidate_prediction"]
        )
    metrics["candidate_rmse_by_seed_m"] = seed_metrics
    seed_spread = max(seed_metrics.values()) - min(seed_metrics.values())
    runtime_before_final = time.perf_counter() - started
    delta = metrics["delta_vs_incumbent"]
    bootstrap = metrics["bootstrap_vs_incumbent"]
    gates = decision_gates(
        pooled_delta_m=float(delta["pooled_delta_m"]),
        fold_deltas_m=delta["by_fold_delta_m"],
        station_deltas_m=delta["by_station_delta_m"],
        lead_deltas_m=delta["by_lead_h_delta_m"],
        bootstrap_ci90_upper_m=float(bootstrap["ci90_m"][1]),
        probability_improved=float(bootstrap["probability_improved"]),
        novelty_rms_m=float(metrics["prediction_novelty_rms_m"]),
        seed_rmse_spread_m=float(seed_spread),
        runtime_seconds=float(runtime_before_final),
        maximum_seed_seconds=max(record["runtime_seconds"] for record in fold_records),
    )

    selected_epochs_by_seed = {
        int(seed): int(
            np.median(
                [
                    record["inner"]["selected_epoch"]
                    for record in fold_records
                    if record["seed"] == int(seed)
                ]
            )
        )
        for seed in training["seeds"]
    }
    if gates["official_info_go"]:
        final_candidate = _full_train_and_candidate(
            config=config,
            values=values,
            station=station,
            anchors=anchors,
            target_vault=vault,
            selected_epochs=selected_epochs_by_seed,
            output=output,
            device=device,
        )
    else:
        final_candidate = {
            "created": False,
            "uploaded": False,
            "reason": "official_information_gate_failed",
            "official_inputs_read": False,
        }

    protected_after = _protected_hashes()
    if protected_after != protected_before:
        raise RuntimeError("a protected P3 lineage file changed during the experiment")
    result = {
        "experiment_id": config["experiment_id"],
        "status": "TERMINAL_OFFICIAL_INFO_GO" if gates["official_info_go"] else "TERMINAL_NO_GO",
        "config_sha256": sha256_file(config_path),
        "attempt_lock_sha256": sha256_file(attempt_path),
        "prediction_seal_sha256": sha256_file(seal_path),
        "combined_prediction_sha256": sha256_file(combined_path),
        "evaluated_outer_sha256": sha256_file(evaluated_path),
        "truth_first_read_after_prediction_seal": True,
        "outer_truth_open_count": vault.outer_open_count,
        "target_access_log": vault.access_log,
        "official_inputs_read_before_information_gate": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "fold_seed_records": fold_records,
        "selected_epochs_by_seed_for_final": {
            str(key): value for key, value in selected_epochs_by_seed.items()
        },
        "metrics": metrics,
        "seed_rmse_spread_m": float(seed_spread),
        "gates": gates,
        "final_candidate": final_candidate,
        "protected_lineage_hashes_before_after_identical": True,
        "protected_lineage_hashes": protected_after,
        "runtime_seconds_before_final_fit": float(runtime_before_final),
        "runtime_seconds_total": float(time.perf_counter() - started),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_worktree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "uploaded": False,
    }
    _atomic_json(terminal_path, result)
    manifest = {
        "experiment_id": config["experiment_id"],
        "created_at_utc": _now(),
        "terminal_result": {
            "path": str(terminal_path.resolve()),
            "sha256": sha256_file(terminal_path),
        },
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "prediction_seal": {"path": str(seal_path.resolve()), "sha256": sha256_file(seal_path)},
        "evaluated_outer": {
            "path": str(evaluated_path.resolve()),
            "sha256": sha256_file(evaluated_path),
        },
        "checkpoint_count": int(len(fold_records) * 2 + len(final_candidate.get("final_models", []))),
        "official_candidate": final_candidate,
        "uploaded": False,
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return result


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise SystemExit("Pass --execute to consume the exactly-once bounded GPU experiment")
    config_path = Path(args.config)
    config = _read_json(config_path)
    result = execute(config_path, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
