"""Check, smoke, or execute the sealed P1 exact degradation-mask pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc.p1_anomalybert_exact_degradation_mask_anchor_union_20260828_v1 import (  # noqa: E402
    ExactDegradationMaskTransformer,
    anchor_union,
    build_features,
    contiguous_full_windows,
    coverage_windows,
    day_block_bootstrap_probability,
    decode_components,
    fit_cell_scale,
    inject_exact_degradation,
    mask_loss,
    seed_everything,
    segment_ids,
    sha256_file,
    smoke_step,
    synthetic_family_metrics,
    transform_cell_scale,
    union_metrics,
)

EXPERIMENT_ID = "p1_anomalybert_exact_degradation_mask_anchor_union_20260828_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
MODULE = ROOT / "src/p1_qc" / f"{EXPERIMENT_ID}.py"
RUNNER = Path(__file__).resolve()
KEY_COLUMNS = ("station", "year", "layer", "time")
FORBIDDEN_TOKENS = ("test.csv", "sample_submission", "submission.csv", "q3", "q4")


class ContractError(RuntimeError):
    """Raised when a sealed experiment contract is no longer exact."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".partial"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".partial"
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".partial") as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _ordered_key_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, KEY_COLUMNS].astype(str).itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _multiindex(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, KEY_COLUMNS].astype(str))


def _map_ordinals(all_keys: pd.DataFrame, requested: pd.DataFrame) -> np.ndarray:
    lookup = _multiindex(all_keys)
    if not lookup.is_unique:
        raise ContractError("training keys are not unique")
    ordinals = lookup.get_indexer(_multiindex(requested))
    if np.any(ordinals < 0) or len(np.unique(ordinals)) != len(ordinals):
        raise ContractError("requested keys do not bind one-to-one")
    return ordinals.astype(np.int64)


def validate_contract(config: dict[str, Any], *, open_q2: bool = False) -> dict[str, Any]:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID drifted")
    if int(config["split"]["purge_days"]) != 15:
        raise ContractError("15-day purge changed")
    if int(config["windowing"]["window_rows"]) != 1024:
        raise ContractError("sealed 1024-row window changed")
    if int(config["training"]["maximum_epochs"]) > 30:
        raise ContractError("epoch budget exceeds authorization")
    if int(config["training"]["maximum_optimizer_steps"]) > 10000:
        raise ContractError("optimizer-step budget exceeds authorization")
    if int(config["training"]["fit_count"]) != 1:
        raise ContractError("fit count changed")
    if config["training"]["result_based_retry"] is not False:
        raise ContractError("result-driven retry is prohibited")
    decoder = config["decoder"]
    prohibited = (
        decoder["score_smoothing_rows"] != 0
        or decoder["bridge_rows"] != 0
        or decoder["point_adjustment"] is not False
        or decoder["truth_fill"] is not False
        or decoder["positive_prevalence_targeting"] is not False
        or decoder["anchor_deletions_allowed"] is not False
    )
    if prohibited:
        raise ContractError("raw point decoding contract changed")
    policy = config["execution_policy"]
    if (
        policy["official_upload_authorized"] is not False
        or policy["submission_csv_generation_allowed"] is not False
        or policy["official_test_sample_submission_read_allowed"] is not False
        or policy["q3_q4_read_allowed"] is not False
    ):
        raise ContractError("forbidden access was authorized")
    checked: dict[str, Any] = {}
    for name, record in config["immutable_inputs"].items():
        deferred = bool(record.get("read_only_after_historical_gate")) and not open_q2
        if deferred:
            checked[name] = {"deferred": True, "reason": "historical_gate_not_opened"}
            continue
        path = ROOT / record["path"]
        lowered = str(path).lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS) and name != "q2_truth_and_keys":
            raise ContractError(f"forbidden input path: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed["bytes"] != int(record["bytes"]) or observed["sha256"] != record["sha256"]:
            raise ContractError(f"immutable input changed: {name}")
        checked[name] = observed
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "config_sha256": sha256_file(CONFIG),
        "inputs": checked,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
    }


def check_only() -> dict[str, Any]:
    config = _json(CONFIG)
    receipt = validate_contract(config)
    return {
        **receipt,
        "mode": "CHECK_ONLY",
        "model_fit_count": 0,
        "scientific_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    receipt = validate_contract(config)
    started = time.perf_counter()
    cpu = smoke_step("cpu", int(config["training"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu = smoke_step("cuda", int(config["training"]["seed"]))
        gpu["device_name"] = torch.cuda.get_device_name(0)
        gpu["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
    else:
        gpu = {"skipped": True, "reason": "cuda_unavailable"}
    checks = {
        "cpu_finite": cpu["finite"] is True,
        "cpu_exact_shape": cpu["output_shape"] == [2, 1024],
        "gpu_finite_or_skipped": gpu.get("finite") is True or gpu.get("skipped") is True,
        "q2_deferred": receipt["inputs"]["q2_frozen_anchor"]["deferred"] is True,
    }
    result = {
        "schema_version": "p1.anomalybert_exact_degradation_mask.smoke.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cpu": cpu,
        "gpu": gpu,
        "elapsed_seconds": time.perf_counter() - started,
        "model_fit_count": 0,
        "scientific_rows_read": 0,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }
    output = ROOT / config["artifacts"]["smoke_directory"] / "smoke.json"
    if output.exists():
        raise FileExistsError("sealed smoke output already exists")
    _atomic_json(output, result)
    return result


def _load_base(config: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = ROOT / config["immutable_inputs"]["training_rows"]["path"]
    columns = [*KEY_COLUMNS, "temp", "psal", "depth"]
    frame = pd.read_parquet(path, columns=columns)
    raw = frame[["temp", "psal", "depth"]].to_numpy(dtype=np.float32, copy=True)
    missing = ~np.isfinite(raw)
    return frame.loc[:, KEY_COLUMNS].copy(), raw, missing


def _scan_labels(
    config: dict[str, Any],
    *,
    start_kst: str | None,
    stop_kst: str,
) -> pd.DataFrame:
    path = ROOT / config["immutable_inputs"]["training_rows"]["path"]
    stop = pd.Timestamp(stop_kst).tz_convert("Asia/Seoul").isoformat()
    condition = ds.field("time") < stop
    if start_kst is not None:
        start = pd.Timestamp(start_kst).tz_convert("Asia/Seoul").isoformat()
        condition = condition & (ds.field("time") >= start)
    scanner = ds.dataset(path, format="parquet").scanner(
        columns=[*KEY_COLUMNS, "label", "anomaly_type"],
        filter=condition,
        use_threads=True,
    )
    return scanner.to_table().to_pandas().reset_index(drop=True)


def _model(config: dict[str, Any], input_width: int) -> ExactDegradationMaskTransformer:
    spec = config["model"]
    return ExactDegradationMaskTransformer(
        input_width=input_width,
        window_rows=int(config["windowing"]["window_rows"]),
        patch_rows=int(spec["patch_rows"]),
        d_model=int(spec["d_model"]),
        heads=int(spec["heads"]),
        layers=int(spec["layers"]),
        feedforward_width=int(spec["feedforward_width"]),
        dropout=float(spec["dropout"]),
    )


def _synthetic_batch(
    features: np.ndarray,
    windows: np.ndarray,
    indices: np.ndarray,
    *,
    epoch: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    synthetic = config["synthetic"]
    seed = int(config["training"]["seed"])
    families = tuple(str(value) for value in synthetic["families"])
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in indices:
        start, stop = windows[int(index)]
        clean = features[start:stop]
        rng = np.random.default_rng(seed + epoch * 1_000_003 + int(index) * 1009)
        if rng.random() < float(synthetic["clean_no_op_probability"]):
            inputs.append(clean.copy())
            targets.append(np.zeros(len(clean), dtype=np.float32))
        else:
            kind = families[int(rng.integers(0, len(families)))]
            corrupted, mask, _ = inject_exact_degradation(
                clean,
                kind,
                rng,
                synthetic["duration_rows"],
                synthetic["standardized_amplitude"],
            )
            inputs.append(corrupted)
            targets.append(mask)
    return np.stack(inputs), np.stack(targets)


def _synthetic_validation(
    model: ExactDegradationMaskTransformer,
    features: np.ndarray,
    windows: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    threshold = float(config["training"]["fixed_synthetic_probability_threshold"])
    synthetic = config["synthetic"]
    batch_size = int(config["training"]["batch_size"])
    seed = int(config["training"]["seed"])
    truth: dict[str, list[np.ndarray]] = {family: [] for family in synthetic["families"]}
    predictions: dict[str, list[np.ndarray]] = {family: [] for family in synthetic["families"]}
    model.eval()
    examples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for window_index, (start, stop) in enumerate(windows):
        clean = features[start:stop]
        for family_index, family in enumerate(synthetic["families"]):
            rng = np.random.default_rng(seed + 90_000_001 + window_index * 101 + family_index)
            corrupted, mask, _ = inject_exact_degradation(
                clean,
                str(family),
                rng,
                synthetic["duration_rows"],
                synthetic["standardized_amplitude"],
            )
            examples.append((str(family), corrupted, mask))
    with torch.no_grad():
        for offset in range(0, len(examples), batch_size):
            batch = examples[offset : offset + batch_size]
            tensor = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
            probability = torch.sigmoid(model(tensor)).cpu().numpy()
            for row, (family, _, mask) in enumerate(batch):
                truth[family].append(mask.astype(np.int8))
                predictions[family].append((probability[row] >= threshold).astype(np.int8))
    return synthetic_family_metrics(truth, predictions)


def _train(
    features: np.ndarray,
    training_windows: np.ndarray,
    validation_windows: np.ndarray,
    config: dict[str, Any],
    output: Path,
) -> tuple[ExactDegradationMaskTransformer, dict[str, Any]]:
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    torch.set_num_threads(int(training["cpu_threads"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(config, features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    checkpoint = output / "best_checkpoint.pt"
    best_macro = -1.0
    best_epoch = 0
    stale = 0
    steps = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        model.train()
        losses: list[float] = []
        order = rng.permutation(len(training_windows))
        for offset in range(0, len(order), int(training["batch_size"])):
            if steps >= int(training["maximum_optimizer_steps"]):
                break
            selected = order[offset : offset + int(training["batch_size"])]
            inputs, masks = _synthetic_batch(
                features, training_windows, selected, epoch=epoch, config=config
            )
            tensor = torch.from_numpy(inputs).to(device)
            target = torch.from_numpy(masks).to(device)
            logits = model(tensor)
            loss = mask_loss(
                logits,
                target,
                positive_weight=float(training["positive_weight"]),
                dice_weight=float(training["dice_loss_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
        validation = _synthetic_validation(
            model, features, validation_windows, config, device
        )
        macro = float(np.mean([float(row["f1"]) for row in validation.values()]))
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": steps,
                "train_loss": float(np.mean(losses)),
                "synthetic_validation_macro_raw_f1": macro,
                "primary_family_raw_f1": {
                    family: float(validation[family]["f1"])
                    for family in config["synthetic"]["primary_long_gate_families"]
                },
            }
        )
        print(
            json.dumps(
                {"epoch": epoch, "steps": steps, "loss": history[-1]["train_loss"], "macro_f1": macro},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if macro > best_macro + 1e-8:
            best_macro = macro
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "synthetic_validation_macro_raw_f1": macro,
                },
                checkpoint,
            )
        else:
            stale += 1
        if (
            epoch >= int(training["minimum_epochs"])
            and stale >= int(training["patience"])
        ) or steps >= int(training["maximum_optimizer_steps"]):
            break
    if not checkpoint.is_file():
        raise RuntimeError("training did not create a checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    final_validation = _synthetic_validation(
        model, features, validation_windows, config, device
    )
    return model, {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "model_fit_count": 1,
        "training_windows": int(len(training_windows)),
        "synthetic_validation_windows": int(len(validation_windows)),
        "epochs_ran": len(history),
        "optimizer_steps": steps,
        "best_epoch": int(best_epoch),
        "best_synthetic_validation_macro_raw_f1": float(best_macro),
        "history": history,
        "final_synthetic_validation": final_validation,
    }


def _infer_scores(
    model: ExactDegradationMaskTransformer,
    features: np.ndarray,
    windows: list[Any],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(len(features), dtype=np.float64)
    counts = np.zeros(len(features), dtype=np.int32)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(windows), batch_size):
            batch_windows = windows[offset : offset + batch_size]
            tensor = torch.from_numpy(
                np.stack([features[window.rows] for window in batch_windows])
            ).to(device)
            probability = torch.sigmoid(model(tensor)).cpu().numpy()
            for row, window in enumerate(batch_windows):
                indices = window.rows[: window.valid_rows]
                scores[indices] += probability[row, : window.valid_rows]
                counts[indices] += 1
    covered = counts > 0
    scores[covered] /= counts[covered]
    scores[~covered] = np.nan
    return scores.astype(np.float32), covered


def _event_count(
    keys: pd.DataFrame,
    truth: np.ndarray,
    anomaly_type: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    eligible: np.ndarray,
    families: set[str],
    minimum_rows: int,
) -> tuple[int, int]:
    segments = segment_ids(keys)
    target = np.asarray(truth, dtype=np.int8)
    types = np.asarray(anomaly_type, dtype=str)
    base = np.asarray(anchor, dtype=np.int8)
    updated = np.asarray(candidate, dtype=np.int8)
    mask = np.asarray(eligible, dtype=bool)
    events = 0
    cells: set[str] = set()
    cursor = 0
    while cursor < len(target):
        if not mask[cursor] or target[cursor] != 1:
            cursor += 1
            continue
        stop = cursor + 1
        while (
            stop < len(target)
            and mask[stop]
            and target[stop] == 1
            and segments[stop] == segments[cursor]
        ):
            stop += 1
        tokens: set[str] = set()
        for value in types[cursor:stop]:
            tokens.update(item.strip() for item in value.split("+") if item.strip())
        was_missed = not base[cursor:stop].any()
        now_hit = updated[cursor:stop].any()
        if stop - cursor >= minimum_rows and tokens & families and was_missed and now_hit:
            events += 1
            cells.add(f"{keys.iloc[cursor]['station']}/{keys.iloc[cursor]['layer']}")
        cursor = stop
    return events, len(cells)


def _surface_diagnostics(
    keys: pd.DataFrame,
    truth: np.ndarray,
    anomaly_type: np.ndarray,
    anchor: np.ndarray,
    additions: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = np.flatnonzero(mask)
    candidate = anchor_union(anchor, additions)
    metrics = union_metrics(truth[rows], anchor[rows], additions[rows])
    times = pd.to_datetime(keys.loc[mask, "time"], utc=True, format="mixed")
    days = max(1, times.dt.floor("D").nunique())
    recovered, recovered_cells = _event_count(
        keys,
        truth,
        anomaly_type,
        anchor,
        candidate,
        mask,
        set(config["synthetic"]["primary_long_gate_families"]),
        int(config["decoder"]["minimum_component_rows"]),
    )
    cells = keys["station"].astype(str).str.cat(keys["layer"].astype(str), sep="/").to_numpy()
    directions: list[float] = []
    by_cell: dict[str, dict[str, float | int]] = {}
    for cell in sorted(set(cells[mask])):
        selected = mask & (cells == cell)
        cell_metrics = union_metrics(truth[selected], anchor[selected], additions[selected])
        by_cell[cell] = cell_metrics
        directions.append(float(cell_metrics["delta_f1"]))
    day_ordinal = times.dt.floor("D").astype("int64").to_numpy() // int(pd.Timedelta(days=1).value)
    gate = config["historical_gate"]
    bootstrap = day_block_bootstrap_probability(
        truth[rows],
        candidate[rows],
        anchor[rows],
        day_ordinal,
        block_days=int(gate["bootstrap_block_days"]),
        replicates=int(gate["bootstrap_replicates"]),
        seed=int(gate["bootstrap_seed"]),
    )
    return {
        **metrics,
        "rows": int(len(rows)),
        "false_positive_rows_per_day": int(metrics["added_fp"]) / days,
        "recovered_primary_long_events": recovered,
        "recovered_primary_long_event_cells": recovered_cells,
        "direction_consistent_cells": int(sum(value > 0 for value in directions)),
        "worst_cell_delta_f1": float(min(directions, default=0.0)),
        "block_bootstrap_probability_delta_f1_gt_zero": bootstrap,
        "by_cell": by_cell,
    }


def _calibration_threshold(
    scores: np.ndarray,
    truth: np.ndarray,
    anchor: np.ndarray,
    mask: np.ndarray,
    segments: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, np.ndarray, dict[str, Any]]:
    best_threshold = float(config["decoder"]["calibration_threshold_grid"][0])
    best_additions = np.zeros(len(truth), dtype=np.int8)
    best_metrics = union_metrics(truth[mask], anchor[mask], best_additions[mask])
    grid_records: list[dict[str, float | int]] = []
    for threshold in config["decoder"]["calibration_threshold_grid"]:
        additions = decode_components(
            scores,
            segments,
            mask,
            threshold=float(threshold),
            minimum_rows=int(config["decoder"]["minimum_component_rows"]),
            maximum_rows=int(config["decoder"]["maximum_component_rows"]),
        )
        metrics = union_metrics(truth[mask], anchor[mask], additions[mask])
        grid_records.append(
            {
                "threshold": float(threshold),
                "candidate_f1": float(metrics["candidate_f1"]),
                "delta_f1": float(metrics["delta_f1"]),
                "added_rows": int(metrics["added_rows"]),
            }
        )
        key = (float(metrics["candidate_f1"]), float(threshold))
        best_key = (float(best_metrics["candidate_f1"]), best_threshold)
        if key > best_key:
            best_threshold = float(threshold)
            best_additions = additions
            best_metrics = metrics
    return best_threshold, best_additions, {"selection_grid": grid_records, **best_metrics}


def _historical_gate_checks(metrics: dict[str, Any], config: dict[str, Any], *, qualification: bool) -> dict[str, bool]:
    gate = config["historical_gate"]
    checks = {
        "delta_f1_gt_zero": float(metrics["delta_f1"]) > 0.0,
        "added_precision_gt_anchor_f1_over_2": float(metrics["added_precision"])
        > float(metrics["anchor_f1_over_2"]),
        "minimum_recovered_primary_long_events": int(metrics["recovered_primary_long_events"])
        >= int(gate["minimum_recovered_primary_long_events"]),
        "minimum_recovered_primary_long_event_cells": int(metrics["recovered_primary_long_event_cells"])
        >= int(gate["minimum_direction_consistent_cells"]),
        "false_positive_rows_per_day_cap": float(metrics["false_positive_rows_per_day"])
        <= float(gate["maximum_false_positive_rows_per_day"]),
        "anchor_positive_removed_rows_zero": int(metrics["anchor_positive_removed_rows"]) == 0,
    }
    if qualification:
        checks.update(
            {
                "minimum_worst_cell_delta_f1": float(metrics["worst_cell_delta_f1"])
                >= float(gate["minimum_worst_cell_delta_f1"]),
                "minimum_direction_consistent_cells": int(metrics["direction_consistent_cells"])
                >= int(gate["minimum_direction_consistent_cells"]),
                "bootstrap_probability_min": float(
                    metrics["block_bootstrap_probability_delta_f1_gt_zero"]
                )
                >= float(gate["minimum_probability_delta_f1_gt_zero"]),
            }
        )
    return checks


def _q2_outer(
    model: ExactDegradationMaskTransformer,
    features: np.ndarray,
    all_keys: pd.DataFrame,
    segments: np.ndarray,
    threshold: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_contract(config, open_q2=True)
    record = config["immutable_inputs"]["q2_truth_and_keys"]
    scanner = ds.dataset(ROOT / record["path"], format="parquet").scanner(
        columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
        filter=ds.field("fold") == str(record["fold"]),
        use_threads=True,
    )
    q2 = scanner.to_table().to_pandas().reset_index(drop=True)
    ordinals = _map_ordinals(all_keys, q2)
    q2_mask = np.zeros(len(all_keys), dtype=bool)
    q2_mask[ordinals] = True
    windows = coverage_windows(
        segments,
        q2_mask,
        int(config["windowing"]["window_rows"]),
        int(config["windowing"]["inference_stride_rows"]),
    )
    scores, covered = _infer_scores(
        model, features, windows, int(config["training"]["batch_size"])
    )
    if not covered[q2_mask].all():
        raise ContractError("Q2 inference coverage is incomplete")
    additions = decode_components(
        scores,
        segments,
        q2_mask,
        threshold=threshold,
        minimum_rows=int(config["decoder"]["minimum_component_rows"]),
        maximum_rows=int(config["decoder"]["maximum_component_rows"]),
    )[ordinals]
    anchor_record = config["immutable_inputs"]["q2_frozen_anchor"]
    with np.load(ROOT / anchor_record["path"], allow_pickle=False) as archive:
        anchor = archive[anchor_record["array"]].astype(np.int8, copy=True)
    receipt_record = config["immutable_inputs"]["q2_frozen_anchor_receipt"]
    receipt = _json(ROOT / receipt_record["path"])
    if _ordered_key_sha(q2) != receipt["ordered_key_sha256"]:
        raise ContractError("Q2 ordered keys differ from frozen anchor receipt")
    if len(anchor) != len(q2) or not np.isin(anchor, [0, 1]).all():
        raise ContractError("Q2 frozen anchor is unaligned or non-binary")
    truth = q2["label"].to_numpy(dtype=np.int8)
    candidate = anchor_union(anchor, additions)
    metrics = union_metrics(truth, anchor, additions)
    checks = {
        "delta_f1_gt_zero": float(metrics["delta_f1"]) > 0.0,
        "added_precision_gt_anchor_f1_over_2": float(metrics["added_precision"])
        > float(metrics["anchor_f1_over_2"]),
        "anchor_positive_removed_rows_zero": int(metrics["anchor_positive_removed_rows"]) == 0,
    }
    return (
        {
            "historical_surface": "Q2 historically exposed; directional outer evidence only",
            **metrics,
            "checks": checks,
            "gate_pass": all(checks.values()),
            "rows_read": int(len(q2)),
            "ordered_key_sha256": _ordered_key_sha(q2),
        },
        {
            "q2_score": scores[ordinals].astype(np.float16),
            "q2_addition": additions,
            "q2_anchor": anchor,
            "q2_candidate": candidate,
        },
    )


def _write_report(config: dict[str, Any], result: dict[str, Any]) -> None:
    report = ROOT / config["artifacts"]["aggregate_only_report"]
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        f"- 결론: `{result['status']}`",
        f"- 모델 fit 수: {result['training']['model_fit_count']}",
        f"- best epoch: {result['training']['best_epoch']}",
        f"- synthetic macro raw F1: {result['training']['best_synthetic_validation_macro_raw_f1']:.6f}",
        f"- Q2 truth rows read: {result['access']['q2_truth_rows_read']}",
        "- Q3/Q4 및 공식 test/sample/submission 접근: 0",
        "- point adjustment, smoothing, truth fill, anchor deletion: 모두 0",
    ]
    if "calibration" in result:
        lines.extend(
            [
                f"- calibration delta F1: {result['calibration']['delta_f1']:.6f}",
                f"- calibration added precision: {result['calibration']['added_precision']:.6f}",
            ]
        )
    if "qualification" in result:
        lines.extend(
            [
                f"- qualification delta F1: {result['qualification']['delta_f1']:.6f}",
                f"- qualification added precision: {result['qualification']['added_precision']:.6f}",
            ]
        )
    if "q2_outer" in result:
        lines.append(f"- Q2 outer delta F1: {result['q2_outer']['delta_f1']:.6f}")
    lines.extend(
        [
            "",
            "이 보고서는 집계치만 포함하며 원 관측값, 키, 제출 CSV를 포함하지 않는다.",
        ]
    )
    _atomic_text(report, "\n".join(lines) + "\n")


def _write_manifest(
    config: dict[str, Any],
    output: Path,
    receipt: dict[str, Any],
    report_path: Path,
) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    files[str(report_path.relative_to(ROOT))] = {
        "bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
    }
    manifest = {
        "schema_version": "p1.anomalybert_exact_degradation_mask.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "module_sha256": sha256_file(MODULE),
        "runner_sha256": sha256_file(RUNNER),
        "preexecution_contract": receipt,
        "files": files,
        "raw_values_persisted": False,
        "keys_persisted": False,
        "submission_csv_generated": False,
        "upload_performed": False,
    }
    _atomic_json(output / "manifest.json", manifest)


def execute() -> dict[str, Any]:
    config = _json(CONFIG)
    receipt = validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    report_path = ROOT / config["artifacts"]["aggregate_only_report"]
    if output.exists() or report_path.exists():
        raise FileExistsError("one-shot output already exists; result-driven reruns are prohibited")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    all_keys, raw, raw_missing = _load_base(config)
    times = pd.to_datetime(all_keys["time"], utc=True, format="mixed")
    cells = all_keys["station"].astype(str).str.cat(all_keys["layer"].astype(str), sep="/").to_numpy()
    groups = segment_ids(all_keys)
    fit_labels_frame = _scan_labels(
        config,
        start_kst=None,
        stop_kst=config["split"]["fit_stop_exclusive_kst"],
    )
    fit_ordinals = _map_ordinals(all_keys, fit_labels_frame)
    fit_label = np.full(len(all_keys), -1, dtype=np.int8)
    fit_label[fit_ordinals] = fit_labels_frame["label"].to_numpy(dtype=np.int8)
    fit_stop = pd.Timestamp(config["split"]["fit_stop_exclusive_kst"]).tz_convert("UTC")
    fit_mask = times.lt(fit_stop).to_numpy()
    fit_normal = fit_mask & (fit_label == 0)
    scale = fit_cell_scale(raw, cells, fit_normal)
    scaled = transform_cell_scale(raw, cells, scale)
    features = build_features(scaled, raw_missing, times)
    boundaries = np.r_[True, groups[1:] != groups[:-1]]
    features[boundaries, 5:7] = 0.0
    split = config["split"]
    training_mask = fit_normal & times.lt(
        pd.Timestamp(split["synthetic_training_stop_exclusive_kst"]).tz_convert("UTC")
    ).to_numpy()
    validation_mask = (
        fit_normal
        & times.ge(pd.Timestamp(split["synthetic_validation_start_kst"]).tz_convert("UTC")).to_numpy()
        & times.lt(
            pd.Timestamp(split["synthetic_validation_stop_exclusive_kst"]).tz_convert("UTC")
        ).to_numpy()
    )
    windowing = config["windowing"]
    training_windows = contiguous_full_windows(
        groups,
        training_mask,
        int(windowing["window_rows"]),
        int(windowing["training_stride_rows"]),
    )
    validation_windows = contiguous_full_windows(
        groups,
        validation_mask,
        int(windowing["window_rows"]),
        int(windowing["training_stride_rows"]),
    )
    preflight_checks = {
        "training_windows_minimum": len(training_windows) >= 32,
        "synthetic_validation_windows_minimum": len(validation_windows) >= 10,
        "fit_scaler_finite": bool(
            np.isfinite(scale.center).all()
            and np.isfinite(scale.scale).all()
            and np.isfinite(features).all()
        ),
        "feature_width_exact": features.shape[1] == len(config["input_features"]),
        "q2_inputs_deferred": receipt["inputs"]["q2_frozen_anchor"]["deferred"] is True,
    }
    preflight = {
        "schema_version": "p1.anomalybert_exact_degradation_mask.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(preflight_checks.values()) else "FAIL",
        "checks": preflight_checks,
        "fit_rows_read": int(len(fit_labels_frame)),
        "fit_normal_rows": int(fit_normal.sum()),
        "training_windows": int(len(training_windows)),
        "synthetic_validation_windows": int(len(validation_windows)),
        "feature_count": int(features.shape[1]),
        "model_fit_count": 0,
        "calibration_truth_rows_read": 0,
        "qualification_truth_rows_read": 0,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
    }
    _atomic_json(output / "preflight.json", preflight)
    if not all(preflight_checks.values()):
        raise ContractError("bounded preflight failed before model fit")
    model, training = _train(
        features, training_windows, validation_windows, config, output
    )
    fidelity = training["final_synthetic_validation"]
    fidelity_gate = config["synthetic_fidelity_gate"]
    fidelity_checks: dict[str, bool] = {}
    for family in config["synthetic"]["primary_long_gate_families"]:
        fidelity_checks[f"{family}_raw_f1"] = float(fidelity[family]["f1"]) >= float(
            fidelity_gate["minimum_raw_f1_each_primary_family"]
        )
        fidelity_checks[f"{family}_boundary_mae"] = float(
            fidelity[family]["boundary_mae_rows"]
        ) <= float(fidelity_gate["maximum_boundary_mae_rows_each_primary_family"])
    synthetic_record = {
        "metrics": fidelity,
        "checks": fidelity_checks,
        "gate_pass": all(fidelity_checks.values()),
        "fixed_probability_threshold": config["training"][
            "fixed_synthetic_probability_threshold"
        ],
    }
    _atomic_json(output / "synthetic_fidelity.json", synthetic_record)
    access = {
        "fit_label_rows_read": int(len(fit_labels_frame)),
        "calibration_truth_rows_read": 0,
        "qualification_truth_rows_read": 0,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }
    result: dict[str, Any] = {
        "schema_version": "p1.anomalybert_exact_degradation_mask.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "NO_GO_SYNTHETIC_FIDELITY",
        "training": training,
        "synthetic_fidelity": synthetic_record,
        "anchor_deletions": 0,
        "point_adjustment": False,
        "score_smoothing": False,
        "truth_fill": False,
        "result_based_rerun": False,
        "access": access,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if all(fidelity_checks.values()):
        calibration_labels = _scan_labels(
            config,
            start_kst=split["calibration_start_kst"],
            stop_kst=split["calibration_stop_exclusive_kst"],
        )
        calibration_ordinals = _map_ordinals(all_keys, calibration_labels)
        calibration_mask = np.zeros(len(all_keys), dtype=bool)
        calibration_mask[calibration_ordinals] = True
        access["calibration_truth_rows_read"] = int(len(calibration_labels))
        calibration_windows = coverage_windows(
            groups,
            calibration_mask,
            int(windowing["window_rows"]),
            int(windowing["inference_stride_rows"]),
        )
        calibration_scores, calibration_covered = _infer_scores(
            model,
            features,
            calibration_windows,
            int(config["training"]["batch_size"]),
        )
        calibration_coverage = float(calibration_covered[calibration_mask].mean())
        if calibration_coverage != float(windowing["required_historical_coverage"]):
            raise ContractError("calibration coverage is not exactly complete")
        historical_truth = np.zeros(len(all_keys), dtype=np.int8)
        historical_types = np.full(len(all_keys), "", dtype=object)
        historical_truth[calibration_ordinals] = calibration_labels["label"].to_numpy(dtype=np.int8)
        historical_types[calibration_ordinals] = calibration_labels["anomaly_type"].fillna("").to_numpy()
        zero_anchor = np.zeros(len(all_keys), dtype=np.int8)
        threshold, calibration_additions, threshold_record = _calibration_threshold(
            calibration_scores,
            historical_truth,
            zero_anchor,
            calibration_mask,
            groups,
            config,
        )
        calibration = _surface_diagnostics(
            all_keys,
            historical_truth,
            historical_types,
            zero_anchor,
            calibration_additions,
            calibration_mask,
            config,
        )
        calibration["coverage"] = calibration_coverage
        calibration["selected_threshold"] = threshold
        calibration["threshold_selection"] = threshold_record["selection_grid"]
        calibration_checks = _historical_gate_checks(calibration, config, qualification=False)
        calibration["checks"] = calibration_checks
        calibration["gate_pass"] = all(calibration_checks.values())
        result["calibration"] = calibration
        prediction_arrays: dict[str, np.ndarray] = {
            "calibration_score": calibration_scores[calibration_ordinals].astype(np.float16),
            "calibration_addition": calibration_additions[calibration_ordinals],
        }
        if calibration["gate_pass"]:
            qualification_labels = _scan_labels(
                config,
                start_kst=split["qualification_start_kst"],
                stop_kst=split["qualification_stop_exclusive_kst"],
            )
            qualification_ordinals = _map_ordinals(all_keys, qualification_labels)
            qualification_mask = np.zeros(len(all_keys), dtype=bool)
            qualification_mask[qualification_ordinals] = True
            access["qualification_truth_rows_read"] = int(len(qualification_labels))
            historical_truth[qualification_ordinals] = qualification_labels["label"].to_numpy(dtype=np.int8)
            historical_types[qualification_ordinals] = qualification_labels["anomaly_type"].fillna("").to_numpy()
            qualification_windows = coverage_windows(
                groups,
                qualification_mask,
                int(windowing["window_rows"]),
                int(windowing["inference_stride_rows"]),
            )
            qualification_scores, qualification_covered = _infer_scores(
                model,
                features,
                qualification_windows,
                int(config["training"]["batch_size"]),
            )
            qualification_coverage = float(qualification_covered[qualification_mask].mean())
            if qualification_coverage != float(windowing["required_historical_coverage"]):
                raise ContractError("qualification coverage is not exactly complete")
            qualification_additions = decode_components(
                qualification_scores,
                groups,
                qualification_mask,
                threshold=threshold,
                minimum_rows=int(config["decoder"]["minimum_component_rows"]),
                maximum_rows=int(config["decoder"]["maximum_component_rows"]),
            )
            qualification = _surface_diagnostics(
                all_keys,
                historical_truth,
                historical_types,
                zero_anchor,
                qualification_additions,
                qualification_mask,
                config,
            )
            qualification["coverage"] = qualification_coverage
            qualification["selected_threshold"] = threshold
            qualification_checks = _historical_gate_checks(
                qualification, config, qualification=True
            )
            qualification["checks"] = qualification_checks
            qualification["gate_pass"] = all(qualification_checks.values())
            result["qualification"] = qualification
            prediction_arrays.update(
                {
                    "qualification_score": qualification_scores[qualification_ordinals].astype(np.float16),
                    "qualification_addition": qualification_additions[qualification_ordinals],
                }
            )
            if qualification["gate_pass"]:
                q2, q2_arrays = _q2_outer(
                    model, features, all_keys, groups, threshold, config
                )
                result["q2_outer"] = q2
                prediction_arrays.update(q2_arrays)
                access["q2_truth_rows_read"] = int(q2["rows_read"])
                result["status"] = "GO_Q2_OUTER" if q2["gate_pass"] else "NO_GO_Q2_OUTER"
            else:
                result["status"] = "NO_GO_QUALIFICATION"
        else:
            result["status"] = "NO_GO_CALIBRATION"
        _atomic_npz(output / "sealed_predictions.npz", **prediction_arrays)
    result["access"] = access
    result["elapsed_seconds"] = time.perf_counter() - started
    _atomic_json(output / "result.json", result)
    _write_report(config, result)
    _write_manifest(config, output, receipt, report_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    config = _json(CONFIG)
    if arguments.check:
        result = check_only()
    elif arguments.smoke:
        result = run_smoke(config)
    else:
        result = execute()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
