"""Run one bounded NCAD-inspired P1 synthetic-context experiment on historical Q2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from p1_qc.synthetic_context_tcn import (
    SyntheticContextTCN,
    continuous_segments,
    decode_long_components,
    fit_robust_scale,
    inject_synthetic_event,
    transform_robust,
    union_diagnostics,
    window_rows,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_ncad_synthetic_long_event_20260828_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
MODULE_PATH = ROOT / "src" / "p1_qc" / "synthetic_context_tcn.py"
KEY_COLUMNS = ("station", "year", "layer", "time")


class ContractError(RuntimeError):
    """Raised when the frozen experiment contract no longer holds."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    if config["status"] != "PREREGISTERED_SINGLE_BOUNDED_ATTEMPT":
        raise ContractError("experiment is not preregistered")
    if config["training"]["result_based_retry"] is not False:
        raise ContractError("result-based retry was enabled")
    if config["prohibitions"]["q3_q4_read"] is not True:
        raise ContractError("Q3/Q4 read prohibition changed")
    if int(config["windowing"]["minimum_component_rows"]) != 19:
        raise ContractError("long-event component contract changed")
    return config


def verify_input(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing immutable input: {path}")
    observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if "bytes" in record and int(record["bytes"]) != observed["bytes"]:
        raise ContractError(f"input byte size changed: {path}")
    if record.get("sha256") != observed["sha256"]:
        raise ContractError(f"input hash changed: {path}")
    return observed


def load_q2(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    source = load_json(ROOT / config["source_config"])
    immutable = source["immutable_inputs"]
    truth_record = immutable["frozen_truth_and_folds"]
    key_record = immutable["feature_key_sidecar"]
    feature_record = immutable["feature_cache"]
    truth_path = ROOT / truth_record["path"]
    key_path = ROOT / key_record["path"]
    feature_path = ROOT / feature_record["path"]
    verified = {
        "truth": verify_input(truth_path, truth_record),
        "keys": verify_input(key_path, key_record),
        "features": verify_input(feature_path, feature_record),
    }

    scanner = ds.dataset(truth_path, format="parquet").scanner(
        columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
        filter=ds.field("fold") == config["split"]["fold"],
        use_threads=True,
    )
    membership = scanner.to_table().to_pandas().reset_index(drop=True)
    if membership.empty or membership.duplicated(list(KEY_COLUMNS)).any():
        raise ContractError("Q2 membership is empty or duplicated")
    all_keys = pd.read_parquet(key_path, columns=["ordinal", *KEY_COLUMNS])
    lookup = pd.MultiIndex.from_frame(all_keys.loc[:, KEY_COLUMNS].astype(str))
    requested = pd.MultiIndex.from_frame(membership.loc[:, KEY_COLUMNS].astype(str))
    ordinals = lookup.get_indexer(requested)
    if np.any(ordinals < 0) or len(np.unique(ordinals)) != len(ordinals):
        raise ContractError("Q2 feature binding is not one-to-one")
    membership["time"] = pd.to_datetime(membership["time"], utc=True, format="mixed")
    features = pd.read_parquet(feature_path, columns=config["features"])
    numeric = features.iloc[ordinals].to_numpy(dtype=np.float32, copy=True)

    anchor_path = ROOT / config["anchor_generator"]["path"]
    anchor_record = {
        "sha256": config["anchor_generator"]["sha256"],
    }
    verified["anchor_generator"] = verify_input(anchor_path, anchor_record)
    with np.load(anchor_path, allow_pickle=False) as archive:
        anchor = archive[config["anchor_generator"]["anchor_array"]].astype(np.int8, copy=True)
    if len(anchor) != len(membership) or not np.isin(anchor, [0, 1]).all():
        raise ContractError("anchor is unaligned or non-binary")
    return membership, numeric, anchor, verified


def split_rows(keys: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    unique = np.sort(keys["time"].unique())
    outer1, outer2 = config["split"]["outer"]
    boundary1 = pd.Timestamp(unique[int(np.floor(float(outer1) * len(unique)))])
    boundary2 = pd.Timestamp(unique[int(np.floor(float(outer2) * len(unique)))])
    outer_purge = pd.Timedelta(days=int(config["split"]["outer_purge_days"]))
    development_end = boundary1 - outer_purge
    calibration_end = boundary2 - outer_purge
    times = keys["time"]
    development = np.flatnonzero((times < development_end).to_numpy())
    calibration = np.flatnonzero(((times >= boundary1) & (times < calibration_end)).to_numpy())
    qualification = np.flatnonzero((times >= boundary2).to_numpy())

    dev_unique = np.sort(times.iloc[development].unique())
    fraction = float(config["split"]["fit_fraction_of_development"])
    inner_boundary = pd.Timestamp(dev_unique[int(np.floor(fraction * len(dev_unique)))])
    inner_purge = pd.Timedelta(days=int(config["split"]["inner_purge_days"]))
    fit = np.flatnonzero((times < inner_boundary - inner_purge).to_numpy())
    selection = np.flatnonzero(((times >= inner_boundary) & (times < development_end)).to_numpy())
    result = {
        "fit": fit,
        "selection": selection,
        "development": development,
        "calibration": calibration,
        "qualification": qualification,
    }
    if any(len(rows) == 0 for rows in result.values()):
        raise ContractError("one or more chronological splits are empty")
    return result


class MixedWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        windows: Sequence[np.ndarray],
        *,
        seed: int,
        synthetic_probability: float,
        event_min_rows: int,
        event_max_rows: int,
        primary_channels: Sequence[int],
        difference_channels: Sequence[int],
    ) -> None:
        self.values = values
        self.labels = labels
        self.windows = tuple(np.asarray(rows, dtype=np.int64) for rows in windows)
        self.seed = seed
        self.synthetic_probability = synthetic_probability
        self.event_min_rows = event_min_rows
        self.event_max_rows = event_max_rows
        self.primary_channels = tuple(primary_channels)
        self.difference_channels = tuple(difference_channels)
        self.epoch = 0
        self.normal = tuple(
            index for index, rows in enumerate(self.windows) if int(np.sum(self.labels[rows])) == 0
        )
        if not self.normal:
            raise ContractError("no normal windows are available for synthetic exposure")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.windows[index]
        x = self.values[rows].copy()
        y = self.labels[rows].astype(np.float32, copy=True)
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 9_176)
        if int(y.sum()) == 0 and rng.random() < self.synthetic_probability:
            donor_index = int(rng.choice(np.asarray(self.normal)))
            donor = self.values[self.windows[donor_index]]
            x, y, _ = inject_synthetic_event(
                x,
                rng,
                event_min_rows=self.event_min_rows,
                event_max_rows=self.event_max_rows,
                primary_channels=self.primary_channels,
                difference_channels=self.difference_channels,
                donor=donor,
            )
        return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def infer_scores(
    model: nn.Module,
    values: np.ndarray,
    windows: Sequence[np.ndarray],
    rows: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    totals = np.zeros(len(values), dtype=np.float64)
    counts = np.zeros(len(values), dtype=np.int32)
    for start in range(0, len(windows), batch_size):
        batch_rows = windows[start : start + batch_size]
        batch = torch.from_numpy(
            np.stack([values[indices] for indices in batch_rows]).astype(np.float32)
        ).to(device)
        probability = torch.sigmoid(model(batch)).float().cpu().numpy()
        for indices, score in zip(batch_rows, probability, strict=True):
            np.add.at(totals, indices, score)
            np.add.at(counts, indices, 1)
    selected = np.asarray(rows, dtype=np.int64)
    if np.any(counts[selected] == 0):
        raise ContractError("inference windows did not cover every selected row")
    output = np.zeros(len(values), dtype=np.float32)
    covered = counts > 0
    output[covered] = (totals[covered] / counts[covered]).astype(np.float32)
    return output


def threshold_search(
    scores: np.ndarray,
    keys: pd.DataFrame,
    labels: np.ndarray,
    anchor: np.ndarray,
    rows: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    quantiles = config["threshold"]["inner_score_quantiles"]
    finite = scores[rows][np.isfinite(scores[rows])]
    candidates = sorted({float(np.quantile(finite, float(q))) for q in quantiles}, reverse=True)
    best_threshold = float("inf")
    best = union_diagnostics(labels, anchor, np.zeros(len(labels), np.int8), rows)
    best["threshold"] = best_threshold
    floor = float(config["threshold"]["added_precision_floor"])
    for threshold in candidates:
        additions = decode_long_components(
            scores,
            keys,
            rows,
            threshold=threshold,
            minimum_rows=int(config["windowing"]["minimum_component_rows"]),
            bridge_rows=int(config["windowing"]["bridge_rows"]),
        )
        metrics = union_diagnostics(labels, anchor, additions, rows)
        metrics["threshold"] = threshold
        if metrics["added_rows"] and metrics["added_precision"] < floor:
            continue
        key = (float(metrics["candidate_f1"]), threshold)
        best_key = (float(best["candidate_f1"]), float(best["threshold"]))
        if key > best_key:
            best_threshold, best = threshold, metrics
    return best_threshold, best


def execute() -> dict[str, Any]:
    config = load_config()
    artifact_dir = ROOT / config["artifact_dir"]
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"append-only artifact already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    keys, numeric, anchor, verified = load_q2(config)
    rows = split_rows(keys, config)
    labels = keys["label"].to_numpy(dtype=np.int8)

    scale = fit_robust_scale(numeric[rows["fit"]])
    values = transform_robust(numeric, scale)
    del numeric
    segments = {name: continuous_segments(keys, indices) for name, indices in rows.items()}
    window_size = int(config["windowing"]["rows"])
    train_stride = int(config["windowing"]["training_stride_rows"])
    inference_stride = int(config["windowing"]["inference_stride_rows"])
    train_windows = window_rows(segments["fit"], window_size, train_stride)
    inference_windows = {
        name: window_rows(parts, window_size, inference_stride, pad_short=True)
        for name, parts in segments.items()
        if name in {"selection", "calibration", "qualification"}
    }
    feature_names = list(config["features"])
    primary = [
        index
        for index, name in enumerate(feature_names)
        if name == "temp_raw"
        or "resid" in name
        or name in {"temp_peer_residual", "temp_abs_peer_residual"}
    ]
    difference = [
        index
        for index, name in enumerate(feature_names)
        if "diff" in name or "acceleration" in name
    ]
    dataset = MixedWindowDataset(
        values,
        labels,
        train_windows,
        seed=seed,
        synthetic_probability=float(config["synthetic"]["probability_on_normal_window"]),
        event_min_rows=int(config["synthetic"]["event_min_rows"]),
        event_max_rows=int(config["synthetic"]["event_max_rows"]),
        primary_channels=primary,
        difference_channels=difference,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyntheticContextTCN(
        len(feature_names),
        width=int(config["model"]["width"]),
        dilations=tuple(config["model"]["dilations"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    pos_weight = torch.tensor(float(config["training"]["positive_weight"]), device=device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_threshold = float("inf")
    best_metrics: dict[str, Any] | None = None
    stale = 0
    for epoch in range(1, int(config["training"]["maximum_epochs"]) + 1):
        dataset.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for batch, target in loader:
            batch = batch.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        selection_scores = infer_scores(
            model,
            values,
            inference_windows["selection"],
            rows["selection"],
            batch_size=int(config["training"]["batch_size"]),
            device=device,
        )
        threshold, metrics = threshold_search(
            selection_scores, keys, labels, anchor, rows["selection"], config
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "threshold": None if not np.isfinite(threshold) else threshold,
            **metrics,
        }
        if not np.isfinite(record["threshold"] or np.nan):
            record["threshold"] = None
        history.append(record)
        current = (float(metrics["candidate_f1"]), threshold)
        previous = (
            (-np.inf, -np.inf)
            if best_metrics is None
            else (float(best_metrics["candidate_f1"]), best_threshold)
        )
        if current > previous:
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
        if epoch >= int(config["training"]["minimum_epochs"]) and stale >= int(
            config["training"]["patience"]
        ):
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "features": feature_names,
            "scale_center": scale.center,
            "scale_scale": scale.scale,
        },
        checkpoint_path,
    )

    split_scores: dict[str, np.ndarray] = {}
    split_additions: dict[str, np.ndarray] = {}
    split_metrics: dict[str, dict[str, Any]] = {}
    for name in ("calibration", "qualification"):
        score = infer_scores(
            model,
            values,
            inference_windows[name],
            rows[name],
            batch_size=int(config["training"]["batch_size"]),
            device=device,
        )
        additions = (
            np.zeros(len(labels), dtype=np.int8)
            if not np.isfinite(best_threshold)
            else decode_long_components(
                score,
                keys,
                rows[name],
                threshold=best_threshold,
                minimum_rows=int(config["windowing"]["minimum_component_rows"]),
                bridge_rows=int(config["windowing"]["bridge_rows"]),
            )
        )
        split_scores[name] = score
        split_additions[name] = additions
        split_metrics[name] = union_diagnostics(labels, anchor, additions, rows[name])

    calibration = split_metrics["calibration"]
    calibration_pass = float(calibration["delta_f1"]) >= float(
        config["calibration_safety"]["minimum_delta_f1"]
    ) and int(calibration["added_fp"]) <= int(config["calibration_safety"]["maximum_added_fp"])
    qualification = split_metrics["qualification"]
    qualification_pass = (
        calibration_pass
        and float(qualification["delta_f1"])
        > float(config["qualification_gate"]["delta_f1_strictly_above"])
        and float(qualification["added_precision"]) > float(qualification["anchor_f1_over_2"])
        and int(qualification["added_tp"]) >= int(config["qualification_gate"]["minimum_added_tp"])
    )
    status = (
        "GO_LOCAL_DIRECTIONAL"
        if qualification_pass
        else ("NO_GO_CALIBRATION_SAFETY" if not calibration_pass else "NO_GO_QUALIFICATION")
    )

    prediction_path = artifact_dir / "sealed_split_predictions.npz"
    np.savez_compressed(
        prediction_path,
        calibration_score=split_scores["calibration"].astype(np.float16),
        calibration_addition=split_additions["calibration"],
        qualification_score=split_scores["qualification"].astype(np.float16),
        qualification_addition=split_additions["qualification"],
    )
    result = {
        "schema_version": "p1.ncad_synthetic_long_event.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_scope": config["claim"],
        "historical_surface": config["historical_surface"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "training": {
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_threshold": None if not np.isfinite(best_threshold) else best_threshold,
            "fit_windows": len(train_windows),
            "normal_fit_windows": len(dataset.normal),
            "history": history,
        },
        "split_rows": {name: int(len(value)) for name, value in rows.items()},
        "split_positive_rows": {name: int(labels[value].sum()) for name, value in rows.items()},
        "inner_selection": best_metrics,
        "calibration": calibration,
        "qualification": qualification,
        "gates": {
            "calibration_pass": calibration_pass,
            "qualification_pass": qualification_pass,
            "submission_value": "NONE; historical local directional evidence only",
        },
        "access": {
            "q2_rows_read": int(len(keys)),
            "q3_q4_rows_read": 0,
            "official_test_rows_read": 0,
            "sample_submission_rows_read": 0,
            "submission_generated_or_uploaded": False,
        },
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "module_sha256": sha256_file(MODULE_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "prediction_sha256": sha256_file(prediction_path),
        },
        "verified_inputs": verified,
    }
    atomic_json(artifact_dir / "result.json", result)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(artifact_dir.iterdir())
            if path.is_file()
        },
    }
    atomic_json(artifact_dir / "manifest.json", manifest)
    return result


def check_only() -> dict[str, Any]:
    config = load_config()
    source = load_json(ROOT / config["source_config"])
    immutable = source["immutable_inputs"]
    verified = {
        name: verify_input(ROOT / immutable[name]["path"], immutable[name])
        for name in ("frozen_truth_and_folds", "feature_key_sidecar", "feature_cache")
    }
    verify_input(
        ROOT / config["anchor_generator"]["path"],
        {"sha256": config["anchor_generator"]["sha256"]},
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY",
        "verified": verified,
        "q3_q4_rows_read": 0,
        "official_test_rows_read": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = execute() if args.execute else check_only()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
