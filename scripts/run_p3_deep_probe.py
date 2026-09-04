"""Honest inner-stopped GRU/TCN structural probe for P3."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from p3_wave.data import LEADS, select_independent_validation
from p3_wave.deep import build_model, fit_channel_statistics
from p3_wave.models import threshold_case_weights
from p3_wave.validation import build_forecast_folds, metric_slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--sequence-cache", default="artifacts/p3/sequences_all20_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/deep_probe")
    parser.add_argument("--architectures", nargs="+", default=["gru", "tcn"])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fixed-epochs", type=int, default=0)
    return parser.parse_args()


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _targets(anchors: pd.DataFrame) -> np.ndarray:
    current = anchors["current_hs"].to_numpy(dtype=np.float32)
    return np.column_stack(
        [anchors[f"target_{lead}"].to_numpy(dtype=np.float32) - current for lead in LEADS]
    )


def _loader(
    values: np.ndarray,
    station: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(values[ids], dtype=np.float32)),
        torch.from_numpy(np.asarray(station[ids], dtype=np.int64)),
        torch.from_numpy(np.asarray(target[ids], dtype=np.float32)),
        torch.from_numpy(np.asarray(weights[ids], dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _train_epochs(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    model.train()
    for _ in range(epochs):
        for raw, station, target, weight in loader:
            raw = raw.to(device, non_blocking=True)
            station = station.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(raw, station)
                loss = torch.mean(weight[:, None] * torch.square(prediction - target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    values: np.ndarray,
    station: np.ndarray,
    ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        raw = torch.from_numpy(np.asarray(values[batch_ids], dtype=np.float32)).to(device)
        station_batch = torch.from_numpy(np.asarray(station[batch_ids], dtype=np.int64)).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(raw, station_batch)
        output.append(prediction.float().cpu().numpy())
    return np.concatenate(output)


def _select_epoch(
    architecture: str,
    values: np.ndarray,
    station: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    fit_ids: np.ndarray,
    calibration_ids: np.ndarray,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[int, list[float]]:
    _seed(seed)
    center, scale = fit_channel_statistics(values[fit_ids])
    model = build_model(architecture, center, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    loader = _loader(values, station, target, weights, fit_ids, batch_size, True)
    history: list[float] = []
    best_epoch = 1
    best_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for raw, station_batch, target_batch, weight_batch in loader:
            raw = raw.to(device, non_blocking=True)
            station_batch = station_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(raw, station_batch)
                loss = torch.mean(weight_batch[:, None] * torch.square(prediction - target_batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        calibration_prediction = _predict(
            model, values, station, calibration_ids, batch_size, device
        )
        score = float(np.sqrt(np.mean(np.square(calibration_prediction - target[calibration_ids]))))
        history.append(score)
        if score < best_rmse - 1e-4:
            best_rmse = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("deep early stopping failed")
    return best_epoch, history


def _fold_metadata(anchors: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    lookup = anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    for lead in LEADS:
        blocks.append(
            pd.DataFrame(
                {
                    "anchor_id": ids,
                    "station": lookup.loc[ids, "station"].astype(str).to_numpy(),
                    "lead_h": lead,
                    "current_hs": lookup.loc[ids, "current_hs"].to_numpy(dtype=float),
                    "target_hs": lookup.loc[ids, f"target_{lead}"].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def main() -> int:
    args = parse_args()
    feature_cache = Path(args.feature_cache)
    sequence_cache = Path(args.sequence_cache)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_parquet(feature_cache / "train_anchors.parquet")
    values = np.load(sequence_cache / "train_values.npy", mmap_mode="r")
    station = np.load(sequence_cache / "train_station.npy", mmap_mode="r")
    target = _targets(anchors)
    weights = threshold_case_weights(anchors["current_hs"].to_numpy()).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    rows: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {}
    for architecture in args.architectures:
        diagnostics[architecture] = {}
        for fold_number, fold in enumerate(build_forecast_folds(anchors)):
            outer_train = anchors.set_index("anchor_id").loc[fold.train_ids]
            inner_end = outer_train["anchor_time"].max() + pd.Timedelta(minutes=20)
            inner_start = inner_end - pd.Timedelta(days=60)
            calibration_ids = select_independent_validation(
                anchors, start=inner_start, end=inner_end, gap_hours=78
            )
            calibration_ids = np.intersect1d(calibration_ids, fold.train_ids)
            fit_end = inner_start - pd.Timedelta(hours=78)
            fit_ids = outer_train.loc[outer_train["anchor_time"].lt(fit_end)].index.to_numpy(
                dtype=np.int64
            )
            if len(fit_ids) == 0 or len(calibration_ids) < 6:
                raise ValueError(f"insufficient inner rows for {fold.name}")
            if args.fixed_epochs > 0:
                best_epoch = args.fixed_epochs
                history: list[float] = []
            else:
                best_epoch, history = _select_epoch(
                    architecture,
                    values,
                    station,
                    target,
                    weights,
                    fit_ids,
                    calibration_ids,
                    args.max_epochs,
                    args.patience,
                    args.batch_size,
                    20260816 + fold_number,
                    device,
                )
            _seed(20260916 + fold_number)
            center, scale = fit_channel_statistics(values[fold.train_ids])
            model = build_model(architecture, center, scale).to(device)
            outer_loader = _loader(
                values,
                station,
                target,
                weights,
                fold.train_ids,
                args.batch_size,
                True,
            )
            _train_epochs(model, outer_loader, best_epoch, device)
            delta = _predict(model, values, station, fold.validation_ids, args.batch_size, device)
            current = (
                anchors.set_index("anchor_id")
                .loc[fold.validation_ids, "current_hs"]
                .to_numpy(dtype=float)
            )
            absolute = np.clip(delta + current[:, None], 0.0, 30.0)
            frame = _fold_metadata(anchors, fold.validation_ids)
            frame["architecture"] = architecture
            frame["fold"] = fold.name
            frame["prediction"] = absolute.T.reshape(-1)
            rows.append(frame)
            diagnostics[architecture][fold.name] = {
                "best_epoch": best_epoch,
                "epochs_ran": len(history),
                "inner_best_rmse": min(history) if history else None,
                "fit_rows": int(len(fit_ids)),
                "calibration_cases": int(len(calibration_ids)),
            }
    oof = pd.concat(rows, ignore_index=True)
    metrics = {
        name: metric_slices(group, group["prediction"].to_numpy())
        for name, group in oof.groupby("architecture", observed=True)
    }
    base = oof.loc[oof["architecture"].eq(args.architectures[0])].copy()
    metrics["persistence"] = metric_slices(base, base["current_hs"].to_numpy())
    oof_path = output / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_deep_probe",
        "device": str(device),
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "fixed_epochs": args.fixed_epochs,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "oof_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
