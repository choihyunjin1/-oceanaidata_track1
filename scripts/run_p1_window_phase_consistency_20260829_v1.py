"""One-shot P1 alternate-tiling audit and gated paired-view warm-start.

The runner is historical-train/OOF only.  It never resolves an official test,
sample, submission, or output CSV.  Q2 probabilities are sealed before the Q2
truth reader is called; Q3 and Q4 are both sealed before either score is read.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_window_phase_consistency_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
AGGREGATE_PATH = ROOT / "reports" / EXPERIMENT_ID / "aggregate.json"
SOURCE_RUNNER = ROOT / "scripts" / "run_p1_incumbent_preserving_mstcn_asrf_v2.py"
SOURCE_ARTIFACT_DIR = ROOT / "artifacts" / "p1_incumbent_preserving_mstcn_asrf_v2"
CHECKPOINT_DIR = ROOT / "artifacts" / "p1_mstcn_checkpoint_diagnostic_20260827_v2"


class ContractError(RuntimeError):
    """Raised when a preregistered scientific or integrity condition changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_json(path: Path, value: Any) -> None:
    _exclusive_bytes(path, _json_bytes(value))


def _atomic_json(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _exclusive_npz(path: Path, **arrays: Any) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        with path.open("xb") as output, temporary.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return _sha256(path)


def _load_source() -> Any:
    spec = importlib.util.spec_from_file_location("p1_mstcn_source_v2", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the pinned P1 MS-TCN source runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict[str, Any]:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if data.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    recipe = data["fixed_recipe"]
    if recipe != {
        "width": 512,
        "epoch": 150,
        "threshold": 0.8,
        "seeds": [20260827, 20260839, 20260863],
        "window_rows": 2048,
        "stride_rows": 512,
        "alternate_phase_rows": 256,
        "prediction_representation": "raw_three_seed_ensemble_mean",
        "inference_combination": (
            "arithmetic_mean_of_default_and_plus256_views_before_fixed_decoder"
        ),
    }:
        raise ContractError("fixed recipe changed")
    warm = data["paired_view_warm_start"]
    required = {
        "run_count": 1,
        "epochs": 5,
        "learning_rate": 1.0e-5,
        "consistency_weight": 1.0,
        "micro_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 64,
    }
    if any(warm.get(name) != value for name, value in required.items()):
        raise ContractError("paired-view warm-start contract changed")
    return data


def _checkpoint_identities() -> dict[str, dict[str, Any]]:
    manifest_path = CHECKPOINT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(row["path"]): row for row in manifest["files"]}


def check_only() -> dict[str, Any]:
    config = _load_config()
    identities: dict[str, Any] = {}
    for name, pin in config["source_pins"].items():
        path = (ROOT / pin["path"]).resolve()
        if not path.is_file():
            raise ContractError(f"pinned source is absent: {name}")
        observed = _sha256(path)
        if observed != pin["sha256"]:
            raise ContractError(f"pinned source hash changed: {name}")
        identities[name] = {"path": pin["path"], "bytes": path.stat().st_size, "sha256": observed}
    checkpoint_files = _checkpoint_identities()
    for phase in ("q3", "q4"):
        for seed in config["fixed_recipe"]["seeds"]:
            name = f"{phase}_width_512_seed_{seed}_epoch_150_state.pt"
            path = CHECKPOINT_DIR / name
            expected = checkpoint_files.get(name)
            if expected is None or not path.is_file():
                raise ContractError(f"frozen e150 checkpoint is absent: {name}")
            if int(path.stat().st_size) != int(expected["bytes"]) or _sha256(path) != expected[
                "sha256"
            ]:
                raise ContractError(f"frozen e150 checkpoint changed: {name}")
    source = _load_source()
    source_config = source._canonical_config()
    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    if not torch.cuda.is_available():
        raise ContractError("CUDA is required for exact bf16 replay")
    runtime = source.verify_runtime_identity(source_config)
    return {
        "schema_version": "p1.window_phase_consistency.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(CONFIG_PATH),
        "source_identities": identities,
        "frozen_checkpoint_count": 6,
        "runtime": runtime,
        "official_interface_reads": 0,
        "result": "PASS",
    }


@dataclass(frozen=True)
class PhaseWindow:
    segment_id: int
    start: int
    window_size: int
    row_ids: Any
    left_pad: int

    @property
    def valid_length(self) -> int:
        return int(len(self.row_ids))


def build_phase_windows(
    layout: Any, *, window_size: int, stride: int, phase_rows: int
) -> tuple[PhaseWindow, ...]:
    """Build a complete tiling after prepending ``phase_rows`` invalid rows."""

    if not (window_size >= stride >= 1) or not (0 <= phase_rows < stride):
        raise ValueError("invalid alternate-tiling geometry")
    windows: list[PhaseWindow] = []
    for segment in layout.segments:
        start = -phase_rows
        while start < segment.size:
            local_start = max(0, start)
            local_stop = min(segment.size, start + window_size)
            if local_stop > local_start:
                rows = segment.row_ids[local_start:local_stop].copy()
                windows.append(
                    PhaseWindow(
                        int(segment.segment_id),
                        int(start),
                        int(window_size),
                        rows,
                        int(local_start - start),
                    )
                )
            start += stride
    if not windows:
        raise ValueError("phase tiling produced no windows")
    return tuple(windows)


def paired_phase_windows(windows: Sequence[Any], layout: Any, *, phase_rows: int) -> tuple[PhaseWindow, ...]:
    segments = {int(segment.segment_id): segment for segment in layout.segments}
    result: list[PhaseWindow] = []
    for window in windows:
        segment = segments[int(window.segment_id)]
        start = int(window.start) - int(phase_rows)
        local_start = max(0, start)
        local_stop = min(segment.size, start + int(window.window_size))
        rows = segment.row_ids[local_start:local_stop].copy()
        result.append(
            PhaseWindow(
                int(window.segment_id),
                start,
                int(window.window_size),
                rows,
                int(local_start - start),
            )
        )
    return tuple(result)


def _materialize_phase_features(features: Any, windows: Sequence[PhaseWindow]) -> tuple[Any, Any]:
    import numpy as np

    size = int(windows[0].window_size)
    result = np.zeros((len(windows), size, features.shape[1]), dtype=np.float32)
    valid = np.zeros((len(windows), size), dtype=np.float32)
    for index, window in enumerate(windows):
        stop = window.left_pad + window.valid_length
        result[index, window.left_pad : stop] = features[window.row_ids]
        valid[index, window.left_pad : stop] = 1.0
    return result, valid


def _materialize_phase_targets(targets: Any, windows: Sequence[PhaseWindow]) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    size = int(windows[0].window_size)
    event = np.zeros((len(windows), size), dtype=np.float32)
    boundary = np.zeros((len(windows), size, 2), dtype=np.float32)
    kinds = np.zeros((len(windows), size, 5), dtype=np.float32)
    valid = np.zeros((len(windows), size), dtype=bool)
    for index, window in enumerate(windows):
        start = int(window.left_pad)
        stop = start + window.valid_length
        rows = window.row_ids
        event[index, start:stop] = targets.row_label[rows]
        boundary[index, start:stop, 0] = targets.start_boundary[rows]
        boundary[index, start:stop, 1] = targets.end_boundary[rows]
        kinds[index, start:stop] = targets.anomaly_type[rows]
        valid[index, start:stop] = True
    return event, boundary, kinds, valid


def _center_weights(window_size: int) -> Any:
    import numpy as np

    positions = np.arange(window_size, dtype=np.float64)
    center = (window_size - 1) / 2.0
    return np.maximum(1.0 - np.abs(positions - center) / (center + 1.0), np.finfo(float).eps)


def stitch_phase_predictions(predictions: Any, windows: Sequence[PhaseWindow], *, n_rows: int) -> Any:
    import numpy as np

    values = np.asarray(predictions)
    tail = values.shape[2:]
    accumulator = np.zeros((n_rows, *tail), dtype=np.float64)
    denominator = np.zeros(n_rows, dtype=np.float64)
    weights = _center_weights(int(windows[0].window_size))
    for index, window in enumerate(windows):
        start = int(window.left_pad)
        stop = start + window.valid_length
        current = weights[start:stop]
        reshape = (window.valid_length,) + (1,) * len(tail)
        accumulator[window.row_ids] += values[index, start:stop] * current.reshape(reshape)
        denominator[window.row_ids] += current
    if np.any(denominator <= 0.0):
        raise ContractError("alternate tiling did not cover every aligned row")
    reshape = (n_rows,) + (1,) * len(tail)
    return (accumulator / denominator.reshape(reshape)).astype(values.dtype, copy=False)


def _batches(values: Sequence[Any], size: int, *, order: Sequence[int] | None = None) -> Iterable[tuple[Any, ...]]:
    indices = list(range(len(values))) if order is None else list(order)
    for start in range(0, len(indices), size):
        yield tuple(values[index] for index in indices[start : start + size])


def predict_phase(source: Any, model: Any, encoded: Any, windows: Sequence[PhaseWindow], *, batch_size: int, device: Any) -> Any:
    import numpy as np

    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    row_parts: list[Any] = []
    boundary_parts: list[Any] = []
    type_parts: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch in _batches(windows, batch_size):
            values, valid = _materialize_phase_features(encoded.features, batch)
            with source._autocast(device):
                output = model(
                    torch.from_numpy(values).to(device, non_blocking=True),
                    valid_mask=torch.from_numpy(valid).to(device, dtype=torch.bool, non_blocking=True),
                )
            row_parts.append(torch.sigmoid(output.final_logits).float().cpu().numpy())
            boundary_parts.append(torch.sigmoid(output.boundary_logits).float().cpu().numpy())
            type_parts.append(torch.sigmoid(output.type_logits).float().cpu().numpy())
    return source.PredictionBundle(
        stitch_phase_predictions(np.concatenate(row_parts), windows, n_rows=encoded.surface.rows),
        stitch_phase_predictions(
            np.concatenate(boundary_parts), windows, n_rows=encoded.surface.rows
        ),
        stitch_phase_predictions(np.concatenate(type_parts), windows, n_rows=encoded.surface.rows),
    )


def _phase_windows(encoded: Any, config: dict[str, Any], phase_rows: int) -> tuple[PhaseWindow, ...]:
    recipe = config["fixed_recipe"]
    return build_phase_windows(
        encoded.layout,
        window_size=int(recipe["window_rows"]),
        stride=int(recipe["stride_rows"]),
        phase_rows=int(phase_rows),
    )


def _replay_q2_e150(
    source: Any,
    training: Any,
    holdout: Any,
    *,
    source_config: dict[str, Any],
    config: dict[str, Any],
    device: Any,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    import numpy as np

    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    rows = holdout.surface.rows
    sums = {
        "p0_row": np.zeros(rows, dtype=np.float32),
        "p0_boundary": np.zeros((rows, 2), dtype=np.float32),
        "p0_type": np.zeros((rows, len(source.TYPE_NAMES)), dtype=np.float32),
        "p256_row": np.zeros(rows, dtype=np.float32),
        "p256_boundary": np.zeros((rows, 2), dtype=np.float32),
        "p256_type": np.zeros((rows, len(source.TYPE_NAMES)), dtype=np.float32),
    }
    receipts: list[dict[str, Any]] = []
    phase256 = _phase_windows(holdout, config, int(config["fixed_recipe"]["alternate_phase_rows"]))
    for seed in config["fixed_recipe"]["seeds"]:
        capacity = source._config_for_capacity(source_config, width=512, seed=int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        model = source._new_model(training.features.shape[1], capacity, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(capacity["training"]["learning_rate"]),
            weight_decay=float(capacity["training"]["weight_decay"]),
        )
        windows = source._selected_windows(training, capacity)
        positive_weight = source._positive_weight(training.surface.labels)
        _steps, total_steps, _warmup = source._schedule_geometry(capacity, window_count=len(windows))
        global_step = 0
        history: list[dict[str, Any]] = []
        for epoch in range(1, 151):
            started = time.perf_counter()
            telemetry, global_step, learning_rate = source._train_epoch(
                model,
                optimizer,
                training,
                windows,
                config=capacity,
                positive_weight=positive_weight,
                device=device,
                epoch=epoch,
                global_step=global_step,
                total_steps=total_steps,
            )
            history.append(
                source._history_record(
                    epoch=epoch,
                    telemetry=telemetry,
                    global_step=global_step,
                    learning_rate=learning_rate,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            if epoch == 1 or epoch % 10 == 0:
                print(f"Q2 replay seed={seed} epoch={epoch}/150", flush=True)
        default = source.predict_encoded(
            model,
            holdout,
            source._all_windows(holdout, capacity),
            batch_size=64,
            device=device,
        )
        alternate = predict_phase(source, model, holdout, phase256, batch_size=64, device=device)
        for prefix, bundle in (("p0", default), ("p256", alternate)):
            sums[f"{prefix}_row"] += bundle.row_probability.astype(np.float32, copy=False)
            sums[f"{prefix}_boundary"] += bundle.boundary_probability.astype(
                np.float32, copy=False
            )
            sums[f"{prefix}_type"] += bundle.type_probability.astype(np.float32, copy=False)
        history_path = ARTIFACT_DIR / f"q2_replay_seed_{seed}_history.json"
        _exclusive_json(history_path, history)
        receipts.append(
            {
                "seed": int(seed),
                "epochs": 150,
                "optimizer_steps": int(global_step),
                "history": {
                    "path": history_path.name,
                    "bytes": history_path.stat().st_size,
                    "sha256": _sha256(history_path),
                },
            }
        )
        del optimizer, model, default, alternate
        gc.collect()
        torch.cuda.empty_cache()
    denominator = float(len(config["fixed_recipe"]["seeds"]))
    default_bundle = source.PredictionBundle(
        sums["p0_row"] / denominator,
        sums["p0_boundary"] / denominator,
        sums["p0_type"] / denominator,
    )
    alternate_bundle = source.PredictionBundle(
        sums["p256_row"] / denominator,
        sums["p256_boundary"] / denominator,
        sums["p256_type"] / denominator,
    )
    return default_bundle, alternate_bundle, receipts


def bernoulli_symmetric_js(left_logits: Any, right_logits: Any) -> Any:
    import torch

    left = torch.sigmoid(left_logits).clamp(1.0e-6, 1.0 - 1.0e-6)
    right = torch.sigmoid(right_logits).clamp(1.0e-6, 1.0 - 1.0e-6)
    middle = 0.5 * (left + right)

    def kl(probability: Any, reference: Any) -> Any:
        return probability * torch.log(probability / reference) + (1.0 - probability) * torch.log(
            (1.0 - probability) / (1.0 - reference)
        )

    return 0.5 * (kl(left, middle) + kl(right, middle)).mean()


def _paired_overlap_js(output0: Any, output256: Any, pairs: Sequence[tuple[Any, PhaseWindow]]) -> Any:
    import numpy as np
    import torch

    losses: list[Any] = []
    for index, (default, alternate) in enumerate(pairs):
        _common, left, right = np.intersect1d(
            default.row_ids, alternate.row_ids, assume_unique=True, return_indices=True
        )
        if not len(left):
            continue
        left_index = torch.as_tensor(left, device=output0.final_logits.device, dtype=torch.long)
        right_index = torch.as_tensor(
            right + int(alternate.left_pad), device=output256.final_logits.device, dtype=torch.long
        )
        losses.append(
            bernoulli_symmetric_js(
                output0.final_logits[index].index_select(0, left_index),
                output256.final_logits[index].index_select(0, right_index),
            )
        )
    if not losses:
        raise ContractError("paired views have no common real rows")
    return torch.stack(losses).mean()


def _move_targets(torch: Any, batch: tuple[Any, Any, Any, Any], device: Any) -> tuple[Any, ...]:
    event, boundary, kinds, valid = batch
    return (
        torch.from_numpy(event).to(device, non_blocking=True),
        torch.from_numpy(boundary).to(device, non_blocking=True),
        torch.from_numpy(kinds).to(device, non_blocking=True),
        torch.from_numpy(valid).to(device, dtype=torch.bool, non_blocking=True),
    )


def warm_start_seed(
    source: Any,
    training: Any,
    *,
    source_config: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    seed: int,
    device: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    _np, _pd, torch, model_api, data_api = source._load_scientific()
    capacity = source._config_for_capacity(source_config, width=512, seed=int(seed))
    model = source._new_model(training.features.shape[1], capacity, device)
    checkpoint_name = f"{phase}_width_512_seed_{seed}_epoch_150_state.pt"
    checkpoint = torch.load(CHECKPOINT_DIR / checkpoint_name, map_location="cpu", weights_only=True)
    if checkpoint.get("phase") != phase or int(checkpoint.get("seed", -1)) != int(seed):
        raise ContractError("frozen warm-start checkpoint metadata changed")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    warm = config["paired_view_warm_start"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(warm["learning_rate"]),
        weight_decay=float(warm["weight_decay"]),
    )
    default_windows = source._selected_windows(training, capacity)
    alternate_windows = paired_phase_windows(
        default_windows, training.layout, phase_rows=int(config["fixed_recipe"]["alternate_phase_rows"])
    )
    positive_weight = source._positive_weight(training.surface.labels)
    loss_config = source._loss_config(capacity, positive_weight)
    micro = int(warm["micro_batch_size"])
    accumulation = int(warm["gradient_accumulation_steps"])
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    for epoch in range(1, int(warm["epochs"]) + 1):
        order = list(range(len(default_windows)))
        random.Random(int(seed) + 150 + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total_supervised = 0.0
        total_js = 0.0
        observed = 0
        gradient_clips = 0
        started = time.perf_counter()
        batches = list(_batches(order, micro))
        for batch_index, ids_tuple in enumerate(batches):
            ids = tuple(int(value) for value in ids_tuple)
            default = tuple(default_windows[index] for index in ids)
            alternate = tuple(alternate_windows[index] for index in ids)
            values0, valid0 = data_api.materialize_windows(training.features, default)
            values256, valid256 = _materialize_phase_features(training.features, alternate)
            targets0 = source._materialize_target_batch(training.targets, default)
            targets256 = _materialize_phase_targets(training.targets, alternate)
            event0, boundary0, kinds0, mask0 = _move_targets(torch, targets0, device)
            event256, boundary256, kinds256, mask256 = _move_targets(
                torch, targets256, device
            )
            feature_mask0 = torch.from_numpy(valid0).to(device, dtype=torch.bool, non_blocking=True)
            feature_mask256 = torch.from_numpy(valid256).to(
                device, dtype=torch.bool, non_blocking=True
            )
            if not torch.equal(feature_mask0, mask0) or not torch.equal(feature_mask256, mask256):
                raise ContractError("paired feature/target masks differ")
            with source._autocast(device):
                output0 = model(
                    torch.from_numpy(values0).to(device, non_blocking=True),
                    valid_mask=feature_mask0,
                )
                output256 = model(
                    torch.from_numpy(values256).to(device, non_blocking=True),
                    valid_mask=feature_mask256,
                )
                loss0 = model_api.compute_ms_tcn_asrf_loss(
                    output0, event0, boundary0, kinds0, valid_mask=mask0, config=loss_config
                )
                loss256 = model_api.compute_ms_tcn_asrf_loss(
                    output256,
                    event256,
                    boundary256,
                    kinds256,
                    valid_mask=mask256,
                    config=loss_config,
                )
                supervised = 0.5 * (loss0.total + loss256.total)
                js = _paired_overlap_js(output0, output256, tuple(zip(default, alternate, strict=True)))
                loss = (supervised + float(warm["consistency_weight"]) * js) / accumulation
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite paired-view loss")
            loss.backward()
            total_supervised += float(supervised.detach().float().cpu()) * len(ids)
            total_js += float(js.detach().float().cpu()) * len(ids)
            observed += len(ids)
            boundary_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches)
            if boundary_step:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(warm["gradient_clip_norm"])
                )
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("non-finite paired-view gradient")
                if float(norm.detach().float().cpu()) > float(warm["gradient_clip_norm"]):
                    gradient_clips += 1
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        history.append(
            {
                "epoch": epoch,
                "supervised_loss": total_supervised / observed,
                "symmetric_js": total_js / observed,
                "optimizer_steps_cumulative": optimizer_steps,
                "gradient_clip_count": gradient_clips,
                "epoch_wall_seconds": time.perf_counter() - started,
            }
        )
        print(f"warm-start phase={phase} seed={seed} epoch={epoch}/5", flush=True)
    return model, history


def _decode_average(source: Any, source_config: dict[str, Any], config: dict[str, Any], holdout: Any, default: Any, alternate: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    score0 = source._decoder_row_probability(default, source_config)
    score256 = source._decoder_row_probability(alternate, source_config)
    score = 0.5 * (score0 + score256)
    boundary = 0.5 * (default.boundary_probability + alternate.boundary_probability)
    proposal = source.decode_long_event_segments(
        score,
        boundary,
        holdout.layout,
        high_threshold=float(config["fixed_recipe"]["threshold"]),
        snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=source._maximum_segment_rows(source_config),
    )
    if holdout.surface.anchor is None:
        raise ContractError("current-Router anchor is absent")
    candidate = source.anchor_preserving_union(holdout.surface.anchor, proposal)
    return np.asarray(score, dtype=np.float32), np.asarray(proposal, dtype=np.int8), candidate


def _seal_arrays(path: Path, receipt_path: Path, arrays: dict[str, Any], *, phase: str, key_sha: str) -> dict[str, Any]:
    sha = _exclusive_npz(path, **arrays)
    receipt = {
        "schema_version": "p1.window_phase_consistency.blind.v1",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "score_path": path.name,
        "score_bytes": path.stat().st_size,
        "score_sha256": sha,
        "ordered_holdout_key_sha256": key_sha,
        "array_inventory": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "same_fold_truth_columns_opened_before_receipt": 0,
        "official_interface_reads": 0,
    }
    _exclusive_json(receipt_path, receipt)
    return receipt


def _verify_receipt(receipt_path: Path) -> dict[str, Any]:
    import numpy as np

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    path = (receipt_path.parent / str(receipt["score_path"])).resolve()
    if path.parent != receipt_path.parent.resolve() or not path.is_file():
        raise ContractError("sealed score path escapes or is absent")
    if int(path.stat().st_size) != int(receipt["score_bytes"]) or _sha256(path) != receipt[
        "score_sha256"
    ]:
        raise ContractError("sealed score identity changed")
    with np.load(path, allow_pickle=False) as archive:
        observed = {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in archive.files
        }
    if observed != receipt["array_inventory"]:
        raise ContractError("sealed array inventory changed")
    if receipt.get("same_fold_truth_columns_opened_before_receipt") != 0:
        raise ContractError("truth-firewall attestation changed")
    return receipt


def _binary_metrics(source: Any, truth: Any, prediction: Any) -> dict[str, Any]:
    return source.binary_metrics(truth, prediction)


def _evaluate_confirmatory(source: Any, config: dict[str, Any], truths: dict[str, Any], holdouts: dict[str, Any], candidates: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    folds: dict[str, Any] = {}
    truth_parts: list[Any] = []
    anchor_parts: list[Any] = []
    candidate_parts: list[Any] = []
    for phase in ("q3", "q4"):
        truth = truths[phase]["label"].to_numpy(dtype=np.int8)
        anchor = np.asarray(holdouts[phase].surface.anchor, dtype=np.int8)
        candidate = np.asarray(candidates[phase], dtype=np.int8)
        anchor_score = _binary_metrics(source, truth, anchor)
        candidate_score = _binary_metrics(source, truth, candidate)
        folds[phase] = {
            "rows": len(truth),
            "anchor": anchor_score,
            "candidate": candidate_score,
            "delta_f1": float(candidate_score["f1"] - anchor_score["f1"]),
        }
        truth_parts.append(truth)
        anchor_parts.append(anchor)
        candidate_parts.append(candidate)
    truth = np.concatenate(truth_parts)
    anchor = np.concatenate(anchor_parts)
    candidate = np.concatenate(candidate_parts)
    anchor_score = _binary_metrics(source, truth, anchor)
    candidate_score = _binary_metrics(source, truth, candidate)
    removed = int(np.sum((anchor == 1) & (candidate == 0)))
    delta = float(candidate_score["f1"] - anchor_score["f1"])
    checks = {
        "q3_delta_f1_strictly_positive": folds["q3"]["delta_f1"] > 0.0,
        "q4_delta_f1_strictly_positive": folds["q4"]["delta_f1"] > 0.0,
        "pooled_delta_f1_min": delta >= float(config["confirmatory_gate"]["pooled_delta_f1_min"]),
        "anchor_positive_removed_rows": removed
        == int(config["confirmatory_gate"]["anchor_positive_removed_rows"]),
    }
    return {
        "folds": folds,
        "pooled": {
            "rows": len(truth),
            "anchor": anchor_score,
            "candidate": candidate_score,
            "delta_f1": delta,
            "anchor_positive_removed_rows": removed,
        },
        "gate_checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }


def _write_terminal(value: dict[str, Any]) -> dict[str, Any]:
    _exclusive_json(ARTIFACT_DIR / "terminal_result.json", value)
    _atomic_json(AGGREGATE_PATH, value)
    return value


def execute() -> dict[str, Any]:
    import numpy as np

    if ATTEMPT_LOCK.exists() or ARTIFACT_DIR.exists() or AGGREGATE_PATH.exists():
        raise FileExistsError("one-shot attempt namespace already exists; rerun is forbidden")
    preflight = check_only()
    config = _load_config()
    source = _load_source()
    source_config = source._canonical_config()
    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    device = torch.device("cuda")
    # Materialise every read-only dependency needed for the Q2 attempt before
    # consuming the one-shot namespace.  No stochastic fit or fold truth read
    # occurs in this block.
    surfaces = source.load_blind_surfaces(source_config, root=ROOT)
    q2_encoder, q2_train, q2, q2_split = source._prepare_phase_surfaces(
        surfaces, source_config, "q2", root=ROOT
    )
    _exclusive_json(
        ATTEMPT_LOCK,
        {
            "schema_version": "p1.window_phase_consistency.attempt.v1",
            "experiment_id": EXPERIMENT_ID,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config_sha256": preflight["config_sha256"],
            "one_shot": True,
        },
    )
    ARTIFACT_DIR.mkdir(parents=False, exist_ok=False)
    _exclusive_json(ARTIFACT_DIR / "preflight.json", preflight)
    started = datetime.now(UTC)
    _exclusive_json(ARTIFACT_DIR / "q2_split.json", q2_split)
    _exclusive_json(ARTIFACT_DIR / "q2_encoder.json", source._encoder_receipt(q2_encoder))
    default, alternate, replay_receipts = _replay_q2_e150(
        source,
        q2_train,
        q2,
        source_config=source_config,
        config=config,
        device=device,
    )
    q2_receipt_path = SOURCE_ARTIFACT_DIR / "q2_qualification_grid_receipt.json"
    grid = source.load_sealed_q2_grid(q2_receipt_path)
    recipe = config["fixed_recipe"]
    row_index = np.flatnonzero((grid.widths == 512) & (grid.epochs == 150))
    threshold_index = np.flatnonzero(np.isclose(grid.thresholds, 0.8, rtol=0.0, atol=0.0))
    if len(row_index) != 1 or len(threshold_index) != 1:
        raise ContractError("fixed Q2 e150 cell is absent")
    row_id = int(row_index[0])
    threshold_id = int(threshold_index[0])
    default_score = source._decoder_row_probability(default, source_config)
    alternate_score = source._decoder_row_probability(alternate, source_config)
    default_proposal = source.decode_long_event_segments(
        default_score,
        default.boundary_probability,
        q2.layout,
        high_threshold=0.8,
        snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=source._maximum_segment_rows(source_config),
    )
    replay_identity = {
        "decoder_probability_bitwise_equal": bool(
            np.array_equal(default_score, grid.row_probability[row_id])
        ),
        "boundary_probability_bitwise_equal": bool(
            np.array_equal(default.boundary_probability, grid.boundary_probability[row_id])
        ),
        "proposal_bitwise_equal": bool(
            np.array_equal(default_proposal, grid.proposal[row_id, threshold_id])
        ),
    }
    average_score = np.asarray(0.5 * (default_score + alternate_score), dtype=np.float32)
    average_boundary = np.asarray(
        0.5 * (default.boundary_probability + alternate.boundary_probability), dtype=np.float32
    )
    alternate_proposal = source.decode_long_event_segments(
        alternate_score,
        alternate.boundary_probability,
        q2.layout,
        high_threshold=0.8,
        snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=source._maximum_segment_rows(source_config),
    )
    average_proposal = source.decode_long_event_segments(
        average_score,
        average_boundary,
        q2.layout,
        high_threshold=0.8,
        snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=source._maximum_segment_rows(source_config),
    )
    average_candidate = source.anchor_preserving_union(q2.surface.anchor, average_proposal)
    q99 = float(np.quantile(np.abs(default_score - alternate_score), 0.99))
    xor_rows = int(np.sum(default_proposal != alternate_proposal))
    arrays = {
        "default_probability": np.asarray(default_score, dtype=np.float32),
        "plus256_probability": np.asarray(alternate_score, dtype=np.float32),
        "average_probability": average_score,
        "default_boundary_probability": np.asarray(
            default.boundary_probability, dtype=np.float32
        ),
        "plus256_boundary_probability": np.asarray(
            alternate.boundary_probability, dtype=np.float32
        ),
        "default_proposal": np.asarray(default_proposal, dtype=np.int8),
        "plus256_proposal": np.asarray(alternate_proposal, dtype=np.int8),
        "average_proposal": np.asarray(average_proposal, dtype=np.int8),
        "average_candidate": np.asarray(average_candidate, dtype=np.int8),
    }
    blind_receipt_path = ARTIFACT_DIR / "q2_alternate_tiling_blind_receipt.json"
    blind_receipt = _seal_arrays(
        ARTIFACT_DIR / "q2_alternate_tiling_blind.npz",
        blind_receipt_path,
        arrays,
        phase="q2",
        key_sha=surfaces.membership_sha256["2025_q2"],
    )
    _verify_receipt(blind_receipt_path)
    q2_truth_frame = source.load_fold_truth_after_receipts(
        source_config, q2.surface, [q2_receipt_path], fold="2025_q2", root=ROOT
    )
    q2_truth = q2_truth_frame["label"].to_numpy(dtype=np.int8)
    anchor_score = source.binary_metrics(q2_truth, q2.surface.anchor)
    candidate_score = source.binary_metrics(q2_truth, average_candidate)
    delta = float(candidate_score["f1"] - anchor_score["f1"])
    kill = config["q2_preflight"]["kill_if_any"]
    checks = {
        "default_replay_bitwise_identical": all(replay_identity.values()),
        "q99_absolute_probability_difference": q99
        >= float(kill["q99_absolute_probability_difference_below"]),
        "proposal_xor_rows": xor_rows >= int(kill["proposal_xor_rows_below"]),
        "fixed_average_anchor_union_delta_f1": delta
        >= float(kill["fixed_average_anchor_union_delta_f1_below"]),
    }
    q2_metrics = {
        "schema_version": "p1.window_phase_consistency.q2_metrics.v1",
        "blind_receipt": {
            "path": blind_receipt_path.name,
            "sha256": _sha256(blind_receipt_path),
            "score_sha256": blind_receipt["score_sha256"],
        },
        "replay_receipts": replay_receipts,
        "replay_identity": replay_identity,
        "q99_absolute_probability_difference": q99,
        "proposal_xor_rows": xor_rows,
        "anchor": anchor_score,
        "fixed_average_candidate": candidate_score,
        "fixed_average_anchor_union_delta_f1": delta,
        "gate_checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }
    _exclusive_json(ARTIFACT_DIR / "q2_preflight_metrics.json", q2_metrics)
    if not all(checks.values()):
        return _write_terminal(
            {
                "schema_version": "p1.window_phase_consistency.aggregate.v1",
                "experiment_id": EXPERIMENT_ID,
                "status": "NO_GO_Q2_PREFLIGHT",
                "started_at_utc": started.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "q2_preflight": q2_metrics,
                "paired_view_training_started": False,
                "paired_view_training_run_count": 0,
                "official_interface_reads": 0,
                "prediction_csv_created": False,
                "upload_performed": False,
            }
        )

    phase_receipts: dict[str, Path] = {}
    holdouts: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    for phase in ("q3", "q4"):
        encoder, training, holdout, split = source._prepare_phase_surfaces(
            surfaces, source_config, phase, root=ROOT
        )
        _exclusive_json(ARTIFACT_DIR / f"{phase}_split.json", split)
        _exclusive_json(ARTIFACT_DIR / f"{phase}_encoder.json", source._encoder_receipt(encoder))
        sums: dict[str, Any] = {
            "default_row": np.zeros(holdout.surface.rows, dtype=np.float32),
            "default_boundary": np.zeros((holdout.surface.rows, 2), dtype=np.float32),
            "default_type": np.zeros((holdout.surface.rows, len(source.TYPE_NAMES)), dtype=np.float32),
            "alternate_row": np.zeros(holdout.surface.rows, dtype=np.float32),
            "alternate_boundary": np.zeros((holdout.surface.rows, 2), dtype=np.float32),
            "alternate_type": np.zeros((holdout.surface.rows, len(source.TYPE_NAMES)), dtype=np.float32),
        }
        for seed in recipe["seeds"]:
            model, history = warm_start_seed(
                source,
                training,
                source_config=source_config,
                config=config,
                phase=phase,
                seed=int(seed),
                device=device,
            )
            history_path = ARTIFACT_DIR / f"{phase}_seed_{seed}_paired_view_history.json"
            _exclusive_json(history_path, history)
            default_blind = source.predict_encoded(
                model,
                holdout,
                source._all_windows(holdout, source_config),
                batch_size=64,
                device=device,
            )
            alternate_blind = predict_phase(
                source,
                model,
                holdout,
                _phase_windows(holdout, config, int(recipe["alternate_phase_rows"])),
                batch_size=64,
                device=device,
            )
            for prefix, bundle in (("default", default_blind), ("alternate", alternate_blind)):
                sums[f"{prefix}_row"] += bundle.row_probability.astype(np.float32, copy=False)
                sums[f"{prefix}_boundary"] += bundle.boundary_probability.astype(
                    np.float32, copy=False
                )
                sums[f"{prefix}_type"] += bundle.type_probability.astype(np.float32, copy=False)
            del model, default_blind, alternate_blind
            gc.collect()
            torch.cuda.empty_cache()
        count = float(len(recipe["seeds"]))
        default_blind = source.PredictionBundle(
            sums["default_row"] / count,
            sums["default_boundary"] / count,
            sums["default_type"] / count,
        )
        alternate_blind = source.PredictionBundle(
            sums["alternate_row"] / count,
            sums["alternate_boundary"] / count,
            sums["alternate_type"] / count,
        )
        probability, proposal, candidate = _decode_average(
            source, source_config, config, holdout, default_blind, alternate_blind
        )
        arrays = {
            "average_probability": probability,
            "proposal": np.asarray(proposal, dtype=np.int8),
            "candidate": np.asarray(candidate, dtype=np.int8),
        }
        receipt_path = ARTIFACT_DIR / f"{phase}_paired_view_blind_receipt.json"
        _seal_arrays(
            ARTIFACT_DIR / f"{phase}_paired_view_blind.npz",
            receipt_path,
            arrays,
            phase=phase,
            key_sha=surfaces.membership_sha256[source_config["phase_protocols"][phase]["fold"]],
        )
        phase_receipts[phase] = receipt_path
        holdouts[phase] = holdout
        candidates[phase] = candidate
    _verify_receipt(phase_receipts["q3"])
    _verify_receipt(phase_receipts["q4"])
    base_receipts = {
        phase: SOURCE_ARTIFACT_DIR / f"{phase}_confirmatory_blind_receipt.json"
        for phase in ("q3", "q4")
    }
    truths = {
        phase: source.load_fold_truth_after_receipts(
            source_config,
            holdouts[phase].surface,
            [base_receipts[phase]],
            fold=source_config["phase_protocols"][phase]["fold"],
            root=ROOT,
        )
        for phase in ("q3", "q4")
    }
    confirmatory = _evaluate_confirmatory(source, config, truths, holdouts, candidates)
    return _write_terminal(
        {
            "schema_version": "p1.window_phase_consistency.aggregate.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": (
                "PASS_CONFIRMATORY_LOCAL_ONLY"
                if confirmatory["result"] == "PASS"
                else "NO_GO_CONFIRMATORY"
            ),
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "q2_preflight": q2_metrics,
            "confirmatory": confirmatory,
            "paired_view_training_started": True,
            "paired_view_training_run_count": 1,
            "official_interface_reads": 0,
            "prediction_csv_created": False,
            "upload_performed": False,
        }
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = check_only() if args.check_only else execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
