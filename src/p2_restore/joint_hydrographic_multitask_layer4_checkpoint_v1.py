"""Append-only P2 joint-hydrographic checkpoint-selection experiment.

The module never reads P2 test, sample-submission, baseline-submission, or
scoring files.  Active outer-fold target scalars remain byte-routed and
undecoded until all 45 outer prediction arrays have been written with O_EXCL
and bound by an aggregate commitment.

This is a retrospective, research-only comparison.  Its three outer windows
were used by earlier experiments, so even a positive result is not an honest
fresh-holdout promotion result.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

CONFIG_RELATIVE = (
    "configs/experiments/"
    "p2_joint_hydrographic_multitask_layer4_checkpoint_v1.json"
)
SCHEMA_VERSION = "p2_joint_hydrographic_multitask_layer4.checkpoint_v1"
KST = ZoneInfo("Asia/Seoul")
TARGET_LAYERS = (2, 3, 4)
KEYS = ("fold", "station", "layer", "time")
FRACTION_TOKENS = {0.4: "040", 0.55: "055", 0.7: "070", 0.85: "085", 1.0: "100"}
Progress = Callable[[dict[str, Any]], None]


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_bytes(path: Path, payload: bytes) -> None:
    """Write immutable bytes exactly once."""

    if not isinstance(payload, bytes):
        raise TypeError("exclusive payload must be bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("exclusive write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    exclusive_bytes(path, canonical_json_bytes(value) + b"\n")


def _strict_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    return value


def _workspace_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    workspace = root.resolve(strict=True)
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts:
        raise ValueError("workspace path must be portable and relative")
    candidate = (workspace / child).resolve(strict=must_exist)
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError("workspace path escaped the repository")
    return candidate


def _pin(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_pin(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if observed != {"sha256": expected["sha256"], "bytes": int(expected["bytes"])}:
        raise PermissionError(f"pinned file changed: {path.name}")
    return {"path": str(expected.get("path", path.name)), **observed}


def load_config(root: Path, config_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    canonical = _workspace_path(workspace, CONFIG_RELATIVE)
    requested = (config_path or canonical).resolve(strict=True)
    if requested != canonical:
        raise PermissionError("only the canonical checkpoint_v1 config is accepted")
    config = _strict_json(canonical)
    validate_config(config)
    return canonical, config


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed on any change to the registered scientific surface."""

    if config.get("schema_version") != SCHEMA_VERSION or config.get("problem") != "P2":
        raise ValueError("checkpoint_v1 identity changed")
    for flag in (
        "research_only",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
        "exact_official_incumbent_comparison",
    ):
        expected = flag == "research_only"
        if config.get(flag) is not expected:
            raise PermissionError(f"research firewall changed: {flag}")
    fractions = [float(value) for value in config.get("prefix_fractions", [])]
    seeds = [int(value) for value in config.get("seed_ids", [])]
    folds = config.get("folds", [])
    if fractions != list(FRACTION_TOKENS) or seeds != [20260823, 20260824, 20260825]:
        raise ValueError("registered fractions or seeds changed")
    if [item.get("name") for item in folds] != [
        "outer_2024_sep_oct",
        "outer_2025_may_jun",
        "outer_2025_jul_aug",
    ]:
        raise ValueError("outer folds changed")
    if len(config.get("prefix_pins", [])) != 15:
        raise ValueError("prefix pin count changed")
    protocol = config.get("checkpoint_protocol", {})
    if (
        float(protocol.get("inner_train_fraction", -1)) != 0.75
        or int(protocol.get("inner_embargo_days", -1)) != 7
        or int(protocol.get("max_epochs", -1)) != 120
        or int(protocol.get("patience_epochs", -1)) != 30
        or int(protocol.get("evaluation_interval", -1)) != 1
        or protocol.get("selection_rule") != "EXACT_MINIMUM_THEN_EARLIEST_EPOCH"
        or protocol.get("seed_epoch_aggregation") != "MEDIAN_OF_THREE"
        or protocol.get("full_prefix_refit") is not True
        or protocol.get("outer_truth_available_to_checkpoint_selection") is not False
    ):
        raise ValueError("checkpoint protocol changed")
    recipe = config.get("model_and_training", {})
    expected_recipe = {
        "input_channels": 54,
        "hidden_width": 160,
        "dilations": [1, 2, 4, 8, 16, 32],
        "dropout": 0.05,
        "learning_rate": 0.0003,
        "weight_decay": 0.001,
        "chunk_length": 512,
        "chunk_stride": 384,
        "batch_size": 12,
        "gradient_clip_norm": 1.0,
        "vertical_difference_weight": 0.25,
        "parameter_count": 1021602,
        "layer4_clip_c": [-5.0, 45.0],
        "cuda_bfloat16": True,
    }
    if dict(recipe) != expected_recipe:
        raise ValueError("r3 architecture or optimizer recipe changed")
    source = config.get("source_boundary", {})
    if set(source.get("allowed_files", {})) != {"README.md", "observations.csv"}:
        raise PermissionError("allowed source surface changed")
    if set(source.get("forbidden_semantic_reads", ())) != {
        "test_index.csv",
        "sample_submission.csv",
        "baseline_interp.csv",
        "score.py",
    }:
        raise PermissionError("forbidden source surface changed")
    if set(config.get("reference_predictions", {})) != {str(value) for value in fractions}:
        raise ValueError("reference OOF map changed")
    for group in (
        config.get("pinned_implementation", {}).values(),
        config.get("checkpoint_implementation", {}).values(),
        config.get("reference_predictions", {}).values(),
        [config.get("r3_final_comparator", {})],
    ):
        for item in group:
            path = Path(str(item.get("path", "")))
            digest = str(item.get("sha256", ""))
            if path.is_absolute() or ".." in path.parts or len(digest) != 64:
                raise ValueError("invalid portable pin")
    output = Path(str(config.get("output", {}).get("path", "")))
    if output.is_absolute() or ".." in output.parts or not output.parts:
        raise ValueError("output path is not portable")


def build_execution_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    recipe = config["model_and_training"]
    chunks = [int(item["training_chunks"]) for item in config["prefix_pins"]]
    full_steps_per_epoch = sum(
        math.ceil(count / int(recipe["batch_size"])) * len(config["seed_ids"])
        for count in chunks
    )
    approximate_inner_steps_per_epoch = sum(
        math.ceil((count * 0.75) / int(recipe["batch_size"])) * len(config["seed_ids"])
        for count in chunks
    )
    maximum_epochs = int(config["checkpoint_protocol"]["max_epochs"])
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.checkpoint_plan.v1",
        "inner_fits": 45,
        "full_prefix_refits": 45,
        "total_fits": 90,
        "outer_prediction_arrays": 45,
        "max_epochs": maximum_epochs,
        "patience_epochs": int(config["checkpoint_protocol"]["patience_epochs"]),
        "full_steps_per_epoch_all_cells": full_steps_per_epoch,
        "approximate_inner_steps_per_epoch_all_cells": approximate_inner_steps_per_epoch,
        "worst_case_optimizer_steps": maximum_epochs
        * (full_steps_per_epoch + approximate_inner_steps_per_epoch),
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }


def _runtime_snapshot() -> tuple[SimpleNamespace, dict[str, Any]]:
    import numpy as np
    import pandas as pd
    import torch

    from p2_restore import deep_data
    from p2_restore import joint_hydrographic_multitask as model
    from p2_restore import joint_hydrographic_multitask_layer4_execution_r3 as r3

    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint_v1 requires the registered CUDA runtime")
    numerical = SimpleNamespace(np=np, pd=pd, torch=torch, deep_data=deep_data, model=model, r3=r3)
    snapshot = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    return numerical, snapshot


def preflight(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Read only registered source bytes and local aggregate artifacts; write nothing."""

    workspace = root.resolve(strict=True)
    resolved_data = data_dir.resolve(strict=True)
    config_file, config = load_config(workspace, config_path)
    output = _workspace_path(workspace, str(config["output"]["path"]), must_exist=False)
    if output.exists():
        raise FileExistsError(f"append-only output already exists: {output}")

    data_pins: dict[str, Any] = {}
    for name, expected in config["source_boundary"]["allowed_files"].items():
        path = (resolved_data / name).resolve(strict=True)
        if path.parent != resolved_data:
            raise PermissionError("source path escaped P2_DATA_DIR")
        data_pins[name] = _verify_pin(path, expected)

    implementation_pins = {
        **{
            f"base:{role}": _verify_pin(_workspace_path(workspace, pin["path"]), pin)
            for role, pin in config["pinned_implementation"].items()
        },
        **{
            f"checkpoint:{role}": _verify_pin(_workspace_path(workspace, pin["path"]), pin)
            for role, pin in config["checkpoint_implementation"].items()
        },
    }
    reference_pins = {
        role: _verify_pin(_workspace_path(workspace, pin["path"]), pin)
        for role, pin in config["reference_predictions"].items()
    }
    comparator = config["r3_final_comparator"]
    comparator_pin = _verify_pin(_workspace_path(workspace, comparator["path"]), comparator)
    _numerical, runtime = _runtime_snapshot()
    del _numerical
    result = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.preflight.v1",
        "status": "READY_FOR_EXPLICIT_EXECUTE",
        "checked_at_kst": _now_kst(),
        "config": _pin(config_file, workspace),
        "data_pins": data_pins,
        "implementation_pins": implementation_pins,
        "reference_pins": reference_pins,
        "r3_comparator_pin_integrity_only": comparator_pin,
        "r3_comparator_semantically_read": False,
        "forbidden_source_files_semantically_read": 0,
        "runtime": runtime,
        "plan": build_execution_plan(config),
        "output_absent": True,
        "writes": 0,
        "model_fits": 0,
        "outer_truth_scalars_decoded": 0,
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }
    result["summary_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def split_inner_times(
    prefix: Any,
    *,
    train_fraction: float,
    embargo_days: int,
    pd_module: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return prefix-only train/calibration timestamps with a strict embargo."""

    pd = pd_module
    times = pd.DatetimeIndex(prefix).sort_values().unique()
    if len(times) < 8 or times.has_duplicates:
        raise ValueError("prefix is too small or duplicated")
    boundary = int(math.floor(len(times) * float(train_fraction)))
    if boundary < 1 or boundary >= len(times):
        raise ValueError("inner split boundary is invalid")
    calibration = pd.DatetimeIndex(times[boundary:])
    calibration_start = calibration[0]
    cutoff = calibration_start - pd.Timedelta(days=int(embargo_days))
    train = pd.DatetimeIndex(times[times < cutoff])
    if not len(train) or not len(calibration):
        raise ValueError("inner split produced an empty side")
    if train.max() >= cutoff or calibration.min() != calibration_start:
        raise PermissionError("inner embargo failed")
    if len(train.intersection(calibration)):
        raise PermissionError("inner train/calibration overlap")
    audit = {
        "prefix_timestamps": int(len(times)),
        "nominal_boundary_index": boundary,
        "inner_train_timestamps": int(len(train)),
        "inner_calibration_timestamps": int(len(calibration)),
        "embargo_days": int(embargo_days),
        "calibration_start_utc": calibration_start.isoformat(),
        "training_cutoff_utc_exclusive": cutoff.isoformat(),
        "outer_truth_used": False,
    }
    return train, calibration, audit


def exact_best_epoch(history: Sequence[Mapping[str, Any]]) -> tuple[int, float]:
    if not history:
        raise ValueError("checkpoint history is empty")
    pairs: list[tuple[float, int]] = []
    for item in history:
        score = float(item["validation_rmse_c"])
        epoch = int(item["epoch"])
        if not math.isfinite(score) or epoch < 1:
            raise ValueError("checkpoint history contains an invalid point")
        pairs.append((score, epoch))
    score, epoch = min(pairs)
    return epoch, score


def median_epoch(values: Sequence[int]) -> int:
    epochs = sorted(int(value) for value in values)
    if len(epochs) != 3 or epochs[0] < 1:
        raise ValueError("median checkpoint requires exactly three positive epochs")
    return epochs[1]


def _timestamp_sha256(times: Any, np_module: Any) -> str:
    values = times.to_numpy(dtype="datetime64[ns]").astype("<i8", copy=False)
    return hashlib.sha256(np_module.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _prefix_pin(config: Mapping[str, Any], fold: str, fraction: float) -> Mapping[str, Any]:
    matches = [
        item
        for item in config["prefix_pins"]
        if item["fold"] == fold and float(item["fraction"]) == float(fraction)
    ]
    if len(matches) != 1:
        raise ValueError("prefix pin is absent or duplicated")
    return matches[0]


def _set_deterministic(numerical: SimpleNamespace, seed: int) -> None:
    np = numerical.np
    torch = numerical.torch
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def _new_model(numerical: SimpleNamespace, config: Mapping[str, Any], device: Any) -> Any:
    recipe = config["model_and_training"]
    model = numerical.model.JointHydrographicTCN(
        int(recipe["input_channels"]),
        hidden=int(recipe["hidden_width"]),
        dilations=tuple(int(value) for value in recipe["dilations"]),
        dropout=float(recipe["dropout"]),
    ).to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != int(recipe["parameter_count"]):
        raise PermissionError("model parameter count changed")
    return model


def _cpu_state(model: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _torch_bytes(torch: Any, payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def _validation_layer4_rmse(
    model: Any,
    panel: Any,
    normalizer: Any,
    calibration_times: Any,
    *,
    config: Mapping[str, Any],
    numerical: SimpleNamespace,
    device: Any,
) -> tuple[float, int]:
    np = numerical.np
    torch = numerical.torch
    recipe = config["model_and_training"]
    selected = np.asarray(panel.times.isin(calibration_times), dtype=bool)
    selected &= panel.reference_target_mask[:, 2]
    selected &= np.isfinite(panel.target_temperature[:, 2])
    selected &= np.isfinite(panel.temperature_baseline[:, 2])
    minimum = int(config["checkpoint_protocol"]["minimum_inner_calibration_layer4_rows"])
    if int(selected.sum()) < minimum:
        raise RuntimeError("inner calibration has too few finite Layer-4 rows")
    bounds = tuple(
        bound
        for bound in numerical.deep_data.make_chunk_bounds(
            panel.segment_ids,
            length=int(recipe["chunk_length"]),
            stride=int(recipe["chunk_stride"]),
        )
        if selected[bound[0] : bound[1]].any()
    )
    inputs = normalizer.transform_inputs(panel.inputs)
    sums = np.zeros(len(panel.times), dtype=np.float64)
    counts = np.zeros(len(panel.times), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(bounds), int(recipe["batch_size"])):
            current = bounds[begin : begin + int(recipe["batch_size"])]
            batch = np.zeros(
                (len(current), int(recipe["chunk_length"]), inputs.shape[1]),
                dtype=np.float32,
            )
            for offset, (start, stop) in enumerate(current):
                batch[offset, : stop - start] = inputs[start:stop]
            tensor = torch.from_numpy(batch).to(device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                normalized = model(tensor)[..., 2, 0]
            values = normalized.float().cpu().numpy()
            for offset, (start, stop) in enumerate(current):
                width = stop - start
                sums[start:stop] += values[offset, :width]
                counts[start:stop] += 1.0
    if (counts[selected] <= 0).any():
        raise RuntimeError("inner validation overlap-add omitted selected rows")
    normalized = sums[selected] / counts[selected]
    physical = (
        panel.temperature_baseline[selected, 2]
        + normalized * normalizer.target_scale[2, 0]
        + normalizer.target_center[2, 0]
    )
    lower, upper = (float(value) for value in recipe["layer4_clip_c"])
    physical = np.clip(physical, lower, upper)
    truth = panel.target_temperature[selected, 2]
    return float(np.sqrt(np.mean((physical - truth) ** 2))), int(selected.sum())


def _fit_inner(
    panel: Any,
    train_times: Any,
    calibration_times: Any,
    *,
    seed: int,
    config: Mapping[str, Any],
    numerical: SimpleNamespace,
    progress: Progress | None,
    progress_context: Mapping[str, Any],
) -> dict[str, Any]:
    np = numerical.np
    torch = numerical.torch
    recipe = config["model_and_training"]
    protocol = config["checkpoint_protocol"]
    selected = np.asarray(panel.times.isin(train_times), dtype=bool)
    normalizer = numerical.model.JointHydrographicNormalizer.fit(panel, selected)
    chunk_x, chunk_y, chunk_mask, bounds = numerical.model.materialize_joint_chunks(
        panel,
        normalizer,
        selected,
        length=int(recipe["chunk_length"]),
        stride=int(recipe["chunk_stride"]),
        minimum_joint_values=24,
    )
    _set_deterministic(numerical, seed)
    device = torch.device("cuda")
    model = _new_model(numerical, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_key = (float("inf"), 2**31 - 1)
    stale = 0
    steps = 0
    for epoch in range(1, int(protocol["max_epochs"]) + 1):
        order = torch.randperm(len(chunk_x), generator=generator)
        loss_sum = 0.0
        batches = 0
        model.train()
        for start in range(0, len(order), int(recipe["batch_size"])):
            indices = order[start : start + int(recipe["batch_size"])]
            inputs = chunk_x[indices].to(device=device)
            targets = chunk_y[indices].to(device=device)
            mask = chunk_mask[indices].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model.training_loss(
                    inputs,
                    targets,
                    mask,
                    vertical_difference_weight=float(recipe["vertical_difference_weight"]),
                )
            if not torch.isfinite(loss):
                raise RuntimeError("inner training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip_norm"]))
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
            steps += 1
        score, rows = _validation_layer4_rmse(
            model,
            panel,
            normalizer,
            calibration_times,
            config=config,
            numerical=numerical,
            device=device,
        )
        point = {
            "epoch": epoch,
            "mean_training_loss": loss_sum / max(batches, 1),
            "validation_rmse_c": score,
            "validation_layer4_rows": rows,
        }
        history.append(point)
        key = (score, epoch)
        if key < best_key:
            best_key = key
            best_state = _cpu_state(model)
            stale = 0
        else:
            stale += 1
        if progress is not None and (epoch == 1 or epoch % 10 == 0):
            progress({**dict(progress_context), "event": "p2_checkpoint_inner_epoch", **point})
        if stale >= int(protocol["patience_epochs"]):
            break
    best_epoch, best_score = exact_best_epoch(history)
    if best_state is None or (best_score, best_epoch) != best_key:
        raise RuntimeError("inner checkpoint selection failed")
    payload = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.inner_checkpoint.v1",
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_rmse_c": float(best_score),
        "model_state": best_state,
        "input_center": torch.from_numpy(normalizer.input_center.copy()),
        "input_scale": torch.from_numpy(normalizer.input_scale.copy()),
        "target_center": torch.from_numpy(normalizer.target_center.copy()),
        "target_scale": torch.from_numpy(normalizer.target_scale.copy()),
    }
    del model
    torch.cuda.empty_cache()
    return {
        "best_epoch": int(best_epoch),
        "best_validation_rmse_c": float(best_score),
        "history": history,
        "optimizer_steps": int(steps),
        "training_chunks": int(len(bounds)),
        "checkpoint_bytes": _torch_bytes(torch, payload),
    }


def _fit_full_predict(
    panel: Any,
    full_times: Any,
    reference_fold: Any,
    *,
    selected_epoch: int,
    seed: int,
    config: Mapping[str, Any],
    numerical: SimpleNamespace,
) -> tuple[Any, dict[str, Any], bytes]:
    np = numerical.np
    pd = numerical.pd
    torch = numerical.torch
    recipe = config["model_and_training"]
    selected = np.asarray(panel.times.isin(full_times), dtype=bool)
    normalizer = numerical.model.JointHydrographicNormalizer.fit(panel, selected)
    chunk_x, chunk_y, chunk_mask, bounds = numerical.model.materialize_joint_chunks(
        panel,
        normalizer,
        selected,
        length=int(recipe["chunk_length"]),
        stride=int(recipe["chunk_stride"]),
        minimum_joint_values=24,
    )
    _set_deterministic(numerical, seed)
    device = torch.device("cuda")
    model = _new_model(numerical, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    steps = 0
    final_loss = float("nan")
    model.train()
    for _epoch in range(int(selected_epoch)):
        order = torch.randperm(len(chunk_x), generator=generator)
        for start in range(0, len(order), int(recipe["batch_size"])):
            indices = order[start : start + int(recipe["batch_size"])]
            inputs = chunk_x[indices].to(device=device)
            targets = chunk_y[indices].to(device=device)
            mask = chunk_mask[indices].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model.training_loss(
                    inputs,
                    targets,
                    mask,
                    vertical_difference_weight=float(recipe["vertical_difference_weight"]),
                )
            if not torch.isfinite(loss):
                raise RuntimeError("full-prefix refit loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip_norm"]))
            optimizer.step()
            final_loss = float(loss.detach().cpu())
            steps += 1

    positions = panel.times.get_indexer(pd.to_datetime(reference_fold["time"], utc=True))
    layers = reference_fold["layer"].to_numpy(dtype=int)
    if (positions < 0).any() or not set(layers).issubset(TARGET_LAYERS):
        raise ValueError("outer reference keys are absent from the fold-blind panel")
    layer4 = layers == 4
    temperature, domain_audit = numerical.r3._predict_panel_temperature(
        model,
        panel,
        normalizer,
        config={"model_and_training": recipe, "curve_protocol": {"batch_size": recipe["batch_size"]}},
        numerical=numerical,
        device=device,
        required_layer4_positions=positions[layer4],
    )
    reference = reference_fold[f"seed_{seed}"].to_numpy(dtype=np.float64)
    prediction = reference.copy()
    lower, upper = (float(value) for value in recipe["layer4_clip_c"])
    prediction[layer4] = temperature[positions[layer4], 2]
    prediction[layer4] = numerical.r3._csv_float_roundtrip(
        np.clip(prediction[layer4], lower, upper),
        pd_module=pd,
        np_module=np,
    )
    prediction = numerical.r3._validate_assembled_layer4_prediction(
        prediction,
        reference,
        layer4,
        clip_bounds=(lower, upper),
        np_module=np,
    )
    model.cpu()
    model_payload = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.full_refit.v1",
        "seed": int(seed),
        "selected_epoch": int(selected_epoch),
        "model_state": model.state_dict(),
        "input_center": torch.from_numpy(normalizer.input_center.copy()),
        "input_scale": torch.from_numpy(normalizer.input_scale.copy()),
        "target_center": torch.from_numpy(normalizer.target_center.copy()),
        "target_scale": torch.from_numpy(normalizer.target_scale.copy()),
    }
    model_bytes = _torch_bytes(torch, model_payload)
    del model
    torch.cuda.empty_cache()
    receipt = {
        "selected_epoch": int(selected_epoch),
        "optimizer_steps": int(steps),
        "training_chunks": int(len(bounds)),
        "final_training_loss": float(final_loss),
        "outer_rows": int(len(prediction)),
        "outer_layer4_rows": int(layer4.sum()),
        "outer_truth_used_for_fit_or_checkpoint_selection": False,
        "layer2_and_layer3_exact_stage_a_seed_values": True,
        "only_layer4_temperature_replaced": True,
        "physical_prediction_domain_audit": domain_audit,
    }
    return prediction, receipt, model_bytes


def _load_reference(
    workspace: Path,
    config: Mapping[str, Any],
    fraction: float,
    numerical: SimpleNamespace,
) -> Any:
    pd = numerical.pd
    np = numerical.np
    pin = config["reference_predictions"][str(float(fraction))]
    path = _workspace_path(workspace, pin["path"])
    _verify_pin(path, pin)
    frame = pd.read_csv(path, dtype={"fold": "string", "station": "string", "time": "string"})
    seeds = [int(value) for value in config["seed_ids"]]
    expected = [*KEYS, *(f"seed_{seed}" for seed in seeds), "prediction_mean"]
    if list(frame.columns) != expected or "truth" in frame.columns:
        raise PermissionError("Stage-A blind reference schema changed")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="mixed")
    if frame.duplicated(list(KEYS)).any():
        raise ValueError("Stage-A reference keys are duplicated")
    values = frame[[f"seed_{seed}" for seed in seeds]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Stage-A reference predictions are non-finite")
    return frame


def _load_truth_after_commitment(
    observations_path: Path,
    *,
    config: Mapping[str, Any],
    numerical: SimpleNamespace,
) -> tuple[Any, dict[str, Any]]:
    """Decode registered outer T/S only; caller must already verify commitment."""

    pd = numerical.pd
    np = numerical.np
    r3 = numerical.r3
    expected_columns = [
        "station", "year", "layer", "time", "temp", "psal", "depth", "nominal_depth"
    ]
    windows = [
        (
            str(fold["name"]),
            datetime.fromisoformat(str(fold["start_kst"])),
            datetime.fromisoformat(str(fold["stop_kst"])),
        )
        for fold in config["folds"]
    ]
    selected: list[tuple[str, str, int, str, float]] = []
    converted = 0
    with observations_path.open("rb") as stream:
        line, spans = r3._csv_field_spans(stream.readline(), expected_fields=len(expected_columns))
        if [r3._decode_csv_field(line, span) for span in spans] != expected_columns:
            raise ValueError("observations schema changed before truth phase")
        for row_number, raw_row in enumerate(stream, 2):
            try:
                line, spans = r3._csv_field_spans(raw_row, expected_fields=len(expected_columns))
                station = r3._decode_csv_field(line, spans[0])
                layer = int(r3._decode_csv_field(line, spans[2]))
                time_text = r3._decode_csv_field(line, spans[3])
                keyed_time = datetime.fromisoformat(time_text)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"truth key routing failed at row {row_number}") from exc
            fold_name = next(
                (name for name, start, stop in windows if start <= keyed_time < stop),
                None,
            )
            if fold_name is None or layer not in TARGET_LAYERS:
                continue
            temp_text = r3._decode_csv_field(line, spans[4])
            psal_text = r3._decode_csv_field(line, spans[5])
            temp = float(temp_text) if temp_text else float("nan")
            psal = float(psal_text) if psal_text else float("nan")
            converted += 2
            if np.isfinite(temp) and np.isfinite(psal):
                selected.append((fold_name, station, layer, time_text, temp))
    truth = pd.DataFrame.from_records(selected, columns=[*KEYS, "truth"])
    truth["time"] = pd.to_datetime(truth["time"], utc=True, format="mixed")
    if truth.empty or truth.duplicated(list(KEYS)).any():
        raise ValueError("outer truth surface is empty or duplicated")
    return truth, {
        "outer_truth_rows": int(len(truth)),
        "outer_temp_psal_scalars_decoded_after_commitment": int(converted),
        "non_outer_target_scalars_decoded": 0,
        "forbidden_source_files_semantically_read": 0,
    }


def _official_weighted_rmse(
    truth: Any,
    prediction: Any,
    layers: Any,
    counts: Mapping[str, int],
    np: Any,
) -> float:
    weighted = 0.0
    total = 0
    for layer in TARGET_LAYERS:
        keep = layers == layer
        if not keep.any():
            raise ValueError(f"metric lacks layer {layer}")
        weight = int(counts[str(layer)])
        weighted += weight * float(np.mean((prediction[keep] - truth[keep]) ** 2))
        total += weight
    return float(np.sqrt(weighted / total))


def _curve_metric(frame: Any, column: str, counts: Mapping[str, int], np: Any) -> dict[str, Any]:
    by_fold: dict[str, float] = {}
    fold_mse: list[float] = []
    for fold, current in frame.groupby("fold", sort=False):
        score = _official_weighted_rmse(
            current["truth"].to_numpy(float),
            current[column].to_numpy(float),
            current["layer"].to_numpy(int),
            counts,
            np,
        )
        by_fold[str(fold)] = score
        fold_mse.append(score**2)
    by_layer = {
        str(layer): float(
            np.sqrt(
                np.mean(
                    (
                        frame.loc[frame["layer"].eq(layer), column].to_numpy(float)
                        - frame.loc[frame["layer"].eq(layer), "truth"].to_numpy(float)
                    )
                    ** 2
                )
            )
        )
        for layer in TARGET_LAYERS
    }
    return {"aggregate": float(np.sqrt(np.mean(fold_mse))), "by_fold": by_fold, "by_layer": by_layer}


def _csv_bytes(frame: Any) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _create_directories(output: Path, config: Mapping[str, Any]) -> None:
    output.mkdir(parents=False, exist_ok=False)
    (output / "cells").mkdir()
    (output / "folds").mkdir()
    for fold in config["folds"]:
        fold_name = str(fold["name"])
        (output / "cells" / fold_name).mkdir()
        (output / "folds" / fold_name).mkdir()
        for fraction in config["prefix_fractions"]:
            token = FRACTION_TOKENS[float(fraction)]
            (output / "cells" / fold_name / f"fraction_{token}").mkdir()
            for seed in config["seed_ids"]:
                (output / "cells" / fold_name / f"fraction_{token}" / f"seed_{seed}").mkdir()


def _verify_commitment_pin(workspace: Path, pin: Mapping[str, Any]) -> None:
    path = _workspace_path(workspace, str(pin["path"]))
    observed = _pin(path, workspace)
    if observed != dict(pin):
        raise PermissionError("prediction commitment bytes changed")


def _git_snapshot(workspace: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--short"], cwd=workspace, capture_output=True, text=True, check=False
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "UNKNOWN",
        "dirty": bool(status.stdout.strip()),
        "status_line_count": len(status.stdout.splitlines()),
    }


def execute(
    *,
    root: Path,
    data_dir: Path,
    config_path: Path | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Execute once in a fresh append-only namespace."""

    workspace = root.resolve(strict=True)
    resolved_data = data_dir.resolve(strict=True)
    ready = preflight(root=workspace, data_dir=resolved_data, config_path=config_path)
    config_file, config = load_config(workspace, config_path)
    numerical, runtime = _runtime_snapshot()
    output = _workspace_path(workspace, str(config["output"]["path"]), must_exist=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    _create_directories(output, config)
    attempt_lock = output / str(config["output"]["attempt_lock"])
    exclusive_json(
        attempt_lock,
        {
            "schema_version": "p2_joint_hydrographic_multitask_layer4.attempt_lock.v1",
            "created_at_kst": _now_kst(),
            "config": _pin(config_file, workspace),
            "preflight_summary_sha256": ready["summary_sha256"],
            "execute": True,
        },
    )
    started = _now_kst()
    observations_path = (resolved_data / "observations.csv").resolve(strict=True)
    references = {
        float(fraction): _load_reference(workspace, config, float(fraction), numerical)
        for fraction in config["prefix_fractions"]
    }
    predictions: dict[tuple[str, float, int], Any] = {}
    selected_epochs: dict[str, int] = {}
    cell_receipts: list[dict[str, Any]] = []
    fold_commitment_pins: list[dict[str, Any]] = []
    inner_steps = 0
    full_steps = 0

    for fold_index, fold in enumerate(config["folds"]):
        for pin in fold_commitment_pins:
            _verify_commitment_pin(workspace, pin)
        fold_name = str(fold["name"])
        observations, fold_audit = numerical.r3._load_fold_blind_observations(
            observations_path,
            fold=fold,
            embargo_days=int(config["checkpoint_protocol"]["inner_embargo_days"]),
            verified_prior_fold_commitments=len(fold_commitment_pins),
            expected_prior_fold_commitments=fold_index,
            pd_module=numerical.pd,
            np_module=numerical.np,
        )
        panel = numerical.model.build_joint_hydrographic_panel(observations)
        if panel.inputs.shape[1] != int(config["model_and_training"]["input_channels"]):
            raise PermissionError("public-only input width changed")
        reference_by_fraction = {
            fraction: frame.loc[frame["fold"].eq(fold_name)].reset_index(drop=True)
            for fraction, frame in references.items()
        }
        for raw_fraction in config["prefix_fractions"]:
            fraction = float(raw_fraction)
            token = FRACTION_TOKENS[fraction]
            prefix = numerical.model.stage_a_prefix_times(
                panel,
                outer_start=numerical.pd.Timestamp(fold["start_kst"]).tz_convert("UTC"),
                embargo_days=int(config["checkpoint_protocol"]["inner_embargo_days"]),
                fraction=fraction,
            )
            expected_prefix = _prefix_pin(config, fold_name, fraction)
            if (
                len(prefix) != int(expected_prefix["timestamps"])
                or _timestamp_sha256(prefix, numerical.np)
                != expected_prefix["timestamp_order_sha256"]
            ):
                raise PermissionError("runtime prefix changed from r3")
            train_times, calibration_times, split_audit = split_inner_times(
                prefix,
                train_fraction=float(config["checkpoint_protocol"]["inner_train_fraction"]),
                embargo_days=int(config["checkpoint_protocol"]["inner_embargo_days"]),
                pd_module=numerical.pd,
            )
            inner_results: dict[int, dict[str, Any]] = {}
            for seed in config["seed_ids"]:
                seed = int(seed)
                context = {"fold": fold_name, "fraction": fraction, "seed": seed}
                result = _fit_inner(
                    panel,
                    train_times,
                    calibration_times,
                    seed=seed,
                    config=config,
                    numerical=numerical,
                    progress=progress,
                    progress_context=context,
                )
                inner_results[seed] = result
                inner_steps += int(result["optimizer_steps"])
                cell_dir = output / "cells" / fold_name / f"fraction_{token}" / f"seed_{seed}"
                exclusive_bytes(cell_dir / "inner_best.pt", result["checkpoint_bytes"])
                exclusive_json(
                    cell_dir / "inner_history.json",
                    {
                        "schema_version": "p2_joint_hydrographic_multitask_layer4.inner_history.v1",
                        **context,
                        "split_audit": split_audit,
                        "best_epoch": result["best_epoch"],
                        "best_validation_rmse_c": result["best_validation_rmse_c"],
                        "optimizer_steps": result["optimizer_steps"],
                        "training_chunks": result["training_chunks"],
                        "history": result["history"],
                        "outer_truth_used": False,
                    },
                )
            selected_epoch = median_epoch(
                [inner_results[int(seed)]["best_epoch"] for seed in config["seed_ids"]]
            )
            selection_key = f"{fold_name}|{fraction}"
            selected_epochs[selection_key] = selected_epoch
            fraction_dir = output / "cells" / fold_name / f"fraction_{token}"
            exclusive_json(
                fraction_dir / "checkpoint_selection.json",
                {
                    "schema_version": "p2_joint_hydrographic_multitask_layer4.selection.v1",
                    "fold": fold_name,
                    "fraction": fraction,
                    "seed_best_epochs": {
                        str(seed): int(inner_results[int(seed)]["best_epoch"])
                        for seed in config["seed_ids"]
                    },
                    "selected_common_epoch": selected_epoch,
                    "aggregation": "MEDIAN_OF_THREE",
                    "outer_truth_used": False,
                },
            )
            for seed in config["seed_ids"]:
                seed = int(seed)
                prediction, receipt, model_bytes = _fit_full_predict(
                    panel,
                    prefix,
                    reference_by_fraction[fraction],
                    selected_epoch=selected_epoch,
                    seed=seed,
                    config=config,
                    numerical=numerical,
                )
                full_steps += int(receipt["optimizer_steps"])
                key = (fold_name, fraction, seed)
                predictions[key] = prediction
                cell_dir = output / "cells" / fold_name / f"fraction_{token}" / f"seed_{seed}"
                exclusive_bytes(cell_dir / "full_refit.pt", model_bytes)
                prediction_buffer = io.BytesIO()
                numerical.np.save(
                    prediction_buffer,
                    prediction.astype("<f8", copy=False),
                    allow_pickle=False,
                )
                exclusive_bytes(cell_dir / "outer_prediction.npy", prediction_buffer.getvalue())
                complete = {
                    "schema_version": "p2_joint_hydrographic_multitask_layer4.cell_receipt.v1",
                    "fold": fold_name,
                    "fraction": fraction,
                    "seed": seed,
                    "split_audit": split_audit,
                    "seed_inner_best_epoch": int(inner_results[seed]["best_epoch"]),
                    "selected_common_epoch": selected_epoch,
                    **receipt,
                    "prediction_values_sha256": hashlib.sha256(
                        prediction.astype("<f8", copy=False).tobytes()
                    ).hexdigest(),
                    "official_promotion_allowed": False,
                    "candidate_or_test_prediction": False,
                    "upload_performed": False,
                }
                exclusive_json(cell_dir / "receipt.json", complete)
                cell_receipts.append(complete)
                if progress is not None:
                    progress(
                        {
                            "event": "p2_checkpoint_outer_prediction_committed",
                            "fold": fold_name,
                            "fraction": fraction,
                            "seed": seed,
                            "selected_epoch": selected_epoch,
                        }
                    )
        fold_files = sorted(
            path
            for path in (output / "cells" / fold_name).rglob("*")
            if path.is_file()
        )
        fold_payload = {
            "schema_version": "p2_joint_hydrographic_multitask_layer4.fold_commitment.v1",
            "fold": fold_name,
            "prediction_arrays": 15,
            "active_outer_target_scalars_decoded_before_commitment": 0,
            "fold_blind_audit": fold_audit,
            "artifacts": {
                path.relative_to(output).as_posix(): _pin(path, workspace)
                for path in fold_files
            },
        }
        fold_path = output / "folds" / fold_name / "fold_commitment.json"
        exclusive_json(fold_path, fold_payload)
        fold_pin = _pin(fold_path, workspace)
        fold_commitment_pins.append(fold_pin)
        del panel, observations

    aggregate_payload = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.prediction_commitment.v1",
        "created_at_kst": _now_kst(),
        "fold_commitments": fold_commitment_pins,
        "outer_prediction_arrays": len(predictions),
        "expected_outer_prediction_arrays": 45,
        "active_outer_target_scalars_decoded_before_commitment": 0,
        "selected_epochs": selected_epochs,
    }
    if len(predictions) != 45 or len(fold_commitment_pins) != 3:
        raise RuntimeError("aggregate commitment is incomplete")
    aggregate_path = output / str(config["output"]["prediction_commitment"])
    exclusive_json(aggregate_path, aggregate_payload)
    aggregate_pin = _pin(aggregate_path, workspace)
    _verify_commitment_pin(workspace, aggregate_pin)
    for pin in fold_commitment_pins:
        _verify_commitment_pin(workspace, pin)

    # First semantic target/comparator access after the immutable 45-array commitment.
    truth, truth_audit = _load_truth_after_commitment(
        observations_path,
        config=config,
        numerical=numerical,
    )
    comparator_pin = config["r3_final_comparator"]
    comparator_path = _workspace_path(workspace, comparator_pin["path"])
    _verify_pin(comparator_path, comparator_pin)
    r3_oof = numerical.pd.read_csv(
        comparator_path,
        dtype={"fold": "string", "station": "string", "time": "string"},
    )
    r3_oof["time"] = numerical.pd.to_datetime(r3_oof["time"], utc=True, format="mixed")
    required_r3 = {"fraction", *KEYS, "challenger_mean", "truth"}
    if not required_r3.issubset(r3_oof.columns) or r3_oof.duplicated(["fraction", *KEYS]).any():
        raise ValueError("r3 comparator schema or key uniqueness changed")

    points: list[dict[str, Any]] = []
    oof_parts: list[Any] = []
    counts = config["metric_contract"]["official_layer_counts"]
    for raw_fraction in config["prefix_fractions"]:
        fraction = float(raw_fraction)
        parts = []
        for fold in config["folds"]:
            fold_name = str(fold["name"])
            current = references[fraction].loc[
                references[fraction]["fold"].eq(fold_name)
            ].reset_index(drop=True).copy()
            for seed in config["seed_ids"]:
                current[f"checkpoint_seed_{seed}"] = predictions[(fold_name, fraction, int(seed))]
            parts.append(current)
        frame = numerical.pd.concat(parts, ignore_index=True)
        frame["reference_mean"] = frame["prediction_mean"].to_numpy(float)
        seed_columns = [f"checkpoint_seed_{seed}" for seed in config["seed_ids"]]
        frame["checkpoint_mean"] = frame[seed_columns].to_numpy(float).mean(axis=1)
        frame = frame.merge(truth, on=list(KEYS), how="inner", validate="one_to_one")
        old = r3_oof.loc[numerical.np.isclose(r3_oof["fraction"].to_numpy(float), fraction)]
        old = old.loc[:, [*KEYS, "challenger_mean", "truth"]].rename(
            columns={"challenger_mean": "r3_final_mean", "truth": "r3_truth"}
        )
        frame = frame.merge(old, on=list(KEYS), how="inner", validate="one_to_one")
        if len(frame) != len(truth):
            raise ValueError("checkpoint/reference/r3/truth key intersection is incomplete")
        if not numerical.np.allclose(
            frame["truth"].to_numpy(float), frame["r3_truth"].to_numpy(float), rtol=0, atol=1e-12
        ):
            raise PermissionError("r3 stored truth differs from freshly decoded outer truth")
        reference_metric = _curve_metric(frame, "reference_mean", counts, numerical.np)
        r3_metric = _curve_metric(frame, "r3_final_mean", counts, numerical.np)
        checkpoint_metric = _curve_metric(frame, "checkpoint_mean", counts, numerical.np)
        points.append(
            {
                "fraction": fraction,
                "architecture_matched_stage_a": reference_metric,
                "r3_final_epoch": r3_metric,
                "checkpoint_v1": checkpoint_metric,
                "delta_checkpoint_minus_r3_final_c": checkpoint_metric["aggregate"]
                - r3_metric["aggregate"],
                "delta_checkpoint_minus_stage_a_c": checkpoint_metric["aggregate"]
                - reference_metric["aggregate"],
            }
        )
        frame.insert(0, "fraction", fraction)
        oof_parts.append(
            frame.loc[
                :,
                [
                    "fraction", *KEYS, *seed_columns, "checkpoint_mean", "r3_final_mean",
                    "reference_mean", "truth",
                ],
            ]
        )
    metrics = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.metrics.v1",
        "comparison_mode": "CHECKPOINT_POLICY_RETROSPECTIVE_DIAGNOSTIC",
        "exact_official_incumbent_comparison": False,
        "outer_windows_previously_exposed": True,
        "official_promotion_allowed": False,
        "points": points,
        "full_fraction_checkpoint_minus_r3_final_c": points[-1][
            "delta_checkpoint_minus_r3_final_c"
        ],
        "full_fraction_checkpoint_minus_stage_a_c": points[-1][
            "delta_checkpoint_minus_stage_a_c"
        ],
    }
    metrics_path = output / str(config["output"]["metrics"])
    oof_path = output / str(config["output"]["oof"])
    receipt_path = output / str(config["output"]["training_receipt"])
    exclusive_json(metrics_path, metrics)
    exclusive_bytes(oof_path, _csv_bytes(numerical.pd.concat(oof_parts, ignore_index=True)))
    receipt = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.training_receipt.v1",
        "started_at_kst": started,
        "completed_at_kst": _now_kst(),
        "config": _pin(config_file, workspace),
        "preflight_summary_sha256": ready["summary_sha256"],
        "runtime": runtime,
        "git": _git_snapshot(workspace),
        "plan": build_execution_plan(config),
        "inner_fits": 45,
        "full_prefix_refits": 45,
        "inner_optimizer_steps": int(inner_steps),
        "full_refit_optimizer_steps": int(full_steps),
        "selected_common_epochs": selected_epochs,
        "prediction_commitment": aggregate_pin,
        "fold_commitments": fold_commitment_pins,
        "truth_access_audit": truth_audit,
        "cell_receipt_count": len(cell_receipts),
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }
    exclusive_json(receipt_path, receipt)

    manifest_path = output / str(config["output"]["manifest"])
    sidecar_path = output / str(config["output"]["manifest_sidecar"])
    seal_path = output / str(config["output"]["seal"])
    artifacts = {
        path.relative_to(output).as_posix(): _pin(path, workspace)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path not in {manifest_path, sidecar_path, seal_path}
    }
    manifest = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.manifest.v1",
        "created_at_kst": _now_kst(),
        "append_only": True,
        "research_only": True,
        "config": _pin(config_file, workspace),
        "preflight_summary_sha256": ready["summary_sha256"],
        "prediction_commitment": aggregate_pin,
        "artifacts": artifacts,
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }
    exclusive_json(manifest_path, manifest)
    exclusive_bytes(
        sidecar_path,
        f"{sha256_file(manifest_path)}  manifest.json\n".encode("ascii"),
    )
    seal = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.seal.v1",
        "complete": True,
        "status": "RETROSPECTIVE_DIAGNOSTIC_COMPLETE",
        "manifest": _pin(manifest_path, workspace),
        "manifest_sidecar": _pin(sidecar_path, workspace),
        "prediction_commitment": aggregate_pin,
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }
    exclusive_json(seal_path, seal)
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.execution_result.v1",
        "status": seal["status"],
        "output": output.relative_to(workspace).as_posix(),
        "metrics": metrics,
        "manifest_sha256": sha256_file(manifest_path),
        "seal_sha256": sha256_file(seal_path),
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }


__all__ = [
    "CONFIG_RELATIVE",
    "build_execution_plan",
    "exact_best_epoch",
    "exclusive_bytes",
    "execute",
    "load_config",
    "median_epoch",
    "preflight",
    "split_inner_times",
    "validate_config",
]
