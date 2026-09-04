"""Executable, research-only P2 joint-hydrographic Layer-4 learning curve.

No numerical package is imported at module import time.  The only public run
entry requires the live identity-bound capability minted by the contract after
fresh QA, authorization, and consumed-lock verification.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import math
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

from p2_restore import joint_hydrographic_multitask_layer4_contract_r2 as guard

Progress = Callable[[dict[str, Any]], None]
KST = ZoneInfo("Asia/Seoul")
TARGET_LAYERS = (2, 3, 4)
PUBLIC_LAYERS = (1, 5, 6, 7, 8)
KEYS = ("fold", "station", "layer", "time")


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _emit(progress: Progress | None, **payload: Any) -> None:
    if progress is not None:
        progress(payload)


def _fraction_token(fraction: float) -> str:
    try:
        return {0.4: "040", 0.55: "055", 0.7: "070", 0.85: "085", 1.0: "100"}[
            float(fraction)
        ]
    except KeyError as exc:
        raise ValueError("unregistered Layer-4 prefix fraction") from exc


def build_execution_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = config["curve_protocol"]
    chunks = [int(item["training_chunks"]) for item in config["prefix_pins"]]
    derived_steps = sum(
        int(protocol["epochs_per_fit"])
        * math.ceil(count / int(protocol["batch_size"]))
        * len(protocol["seed_ids"])
        for count in chunks
    )
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.execution_plan.r2",
        "stage": guard.STAGE,
        "problem": "P2",
        "comparison_mode": guard.MODE,
        "exact_official_incumbent_comparison": False,
        "fold_major_order": list(protocol["fold_major_order"]),
        "prefix_fractions": [float(value) for value in protocol["prefix_fractions"]],
        "pipeline_seeds": [int(value) for value in protocol["seed_ids"]],
        "cell_order": "FOLD_THEN_FRACTION_THEN_SEED",
        "fit_cells": 45,
        "blind_prediction_arrays": 45,
        "fold_commitments": 3,
        "epochs_per_fit": 28,
        "optimizer_steps": derived_steps,
        "bootstrap_replicates_per_fraction": 5000,
        "implementation_correction": dict(config["implementation_correction"]),
        "prior_v1_optimizer_steps_not_reused": 56,
        "fresh_r2_optimizer_steps": derived_steps,
        "full_fit_jobs": 0,
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }


def _workspace_pin(path: Path, workspace: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(workspace).as_posix(),
        "sha256": guard.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _output_pin(path: Path, output: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output).as_posix(),
        "sha256": guard.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_loaded_module(
    module: Any,
    *,
    workspace: Path,
    expected: Mapping[str, Any],
    role: str,
) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise PermissionError(f"late numerical dependency has no source file: {role}")
    path = Path(module_file).resolve(strict=True)
    canonical = (workspace / str(expected["path"])).resolve(strict=True)
    if path != canonical or _workspace_pin(path, workspace) != dict(expected):
        raise PermissionError(f"late numerical dependency pin failed: {role}")


def _load_numerical_stack(workspace: Path, config: Mapping[str, Any]) -> SimpleNamespace:
    modules = {
        "PACKAGE_INIT": importlib.import_module("p2_restore"),
        "PURE_MODEL": importlib.import_module("p2_restore.joint_hydrographic_multitask"),
        "DEEP_DATA": importlib.import_module("p2_restore.deep_data"),
        "FEATURES": importlib.import_module("p2_restore.features"),
        "DATA": importlib.import_module("p2_restore.data"),
        "CURVE_GATE": importlib.import_module("p2_restore.meaningful_learning_curve"),
    }
    expected = {
        "PACKAGE_INIT": config["source_pins"]["PACKAGE_INIT"],
        "PURE_MODEL": config["scientific_surface"]["pure_model"],
        "DEEP_DATA": config["source_pins"]["DEEP_DATA"],
        "FEATURES": config["source_pins"]["FEATURES"],
        "DATA": config["source_pins"]["DATA"],
        "CURVE_GATE": config["source_pins"]["CURVE_GATE"],
    }
    for role, module in modules.items():
        _verify_loaded_module(module, workspace=workspace, expected=expected[role], role=role)
    model = modules["PURE_MODEL"]
    return SimpleNamespace(
        model=model,
        deep_data=modules["DEEP_DATA"],
        curve_gate=modules["CURVE_GATE"],
        np=model.np,
        pd=model.pd,
        torch=model.torch,
    )


def _set_deterministic_runtime(numerical: SimpleNamespace, seed: int) -> dict[str, Any]:
    np = numerical.np
    torch = numerical.torch
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("registered Layer-4 CUDA runtime is unavailable")
    return {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "numpy": np.__version__,
        "pandas": numerical.pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _csv_field_spans(
    raw_line: bytes,
    *,
    expected_fields: int,
) -> tuple[bytes, list[tuple[int, int]]]:
    """Locate RFC-4180 fields without decoding unselected target scalars."""

    line = raw_line[:-2] if raw_line.endswith(b"\r\n") else raw_line.rstrip(b"\n")
    if b"\r" in line or b"\n" in line:
        raise ValueError("observations CSV contains an embedded newline")
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    quoted = False
    while index < len(line):
        value = line[index]
        if value == 34:
            if quoted and index + 1 < len(line) and line[index + 1] == 34:
                index += 2
                continue
            quoted = not quoted
        elif value == 44 and not quoted:
            spans.append((start, index))
            start = index + 1
        index += 1
    if quoted:
        raise ValueError("observations CSV contains an unterminated quote")
    spans.append((start, len(line)))
    if len(spans) != int(expected_fields):
        raise ValueError("observations CSV row width changed")
    return line, spans


def _decode_csv_field(raw_line: bytes, span: tuple[int, int]) -> str:
    token_bytes = raw_line[span[0] : span[1]]
    if not token_bytes:
        return ""
    token = token_bytes.decode("utf-8")
    parsed = next(csv.reader([token]))
    if len(parsed) != 1:
        raise ValueError("selected CSV span did not decode to one field")
    return parsed[0]


def _numeric(text: str) -> float:
    return float(text) if text else float("nan")


def _csv_float_roundtrip(values: Any, *, pd_module: Any, np_module: Any) -> Any:
    """Apply the frozen in-memory pandas CSV numeric boundary."""

    buffer = io.StringIO(newline="")
    pd_module.DataFrame(
        {"value": np_module.asarray(values, dtype=np_module.float64)}
    ).to_csv(buffer, index=False, lineterminator="\n")
    buffer.seek(0)
    return pd_module.read_csv(buffer)["value"].to_numpy(dtype=np_module.float64)


def _load_fold_blind_observations(
    observations_path: Path,
    *,
    fold: Mapping[str, Any],
    embargo_days: int,
    verified_prior_fold_commitments: int,
    expected_prior_fold_commitments: int,
    pd_module: Any,
    np_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """Decode public values and only target history strictly before cutoff."""

    if verified_prior_fold_commitments != expected_prior_fold_commitments:
        raise PermissionError("previous fold commitment was not verified before target history load")
    pd = pd_module
    np = np_module
    expected_columns = [
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    ]
    start = datetime.fromisoformat(str(fold["start_kst"]))
    stop = datetime.fromisoformat(str(fold["stop_kst"]))
    cutoff = (pd.Timestamp(start).tz_convert("Asia/Seoul") - pd.Timedelta(days=embargo_days)).to_pydatetime()
    rows: list[tuple[Any, ...]] = []
    decoded_training_rows = 0
    withheld_target_rows = 0
    active_fold_target_rows = 0
    with observations_path.open("rb") as stream:
        raw_header = stream.readline()
        header_line, header_spans = _csv_field_spans(
            raw_header,
            expected_fields=len(expected_columns),
        )
        header = [_decode_csv_field(header_line, span) for span in header_spans]
        if header != expected_columns:
            raise ValueError("fold-blind observations schema changed")
        for row_number, raw_row in enumerate(stream, 2):
            try:
                line, spans = _csv_field_spans(raw_row, expected_fields=len(expected_columns))
                station = _decode_csv_field(line, spans[0])
                year = int(_decode_csv_field(line, spans[1]))
                layer = int(_decode_csv_field(line, spans[2]))
                time_text = _decode_csv_field(line, spans[3])
                keyed_time = datetime.fromisoformat(time_text)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"fold-blind key routing failed at row {row_number}") from exc
            if keyed_time.tzinfo is None or not time_text.endswith("+09:00"):
                raise ValueError("fold-blind timestamp lost the KST offset")
            if layer not in (*PUBLIC_LAYERS, *TARGET_LAYERS):
                raise ValueError("observations layer escaped the registered 1..8 surface")
            public = layer in PUBLIC_LAYERS
            time_safe_target = layer in TARGET_LAYERS and keyed_time < cutoff
            if public or time_safe_target:
                # Only these explicitly authorized fields are decoded.
                temp = _numeric(_decode_csv_field(line, spans[4]))
                psal = _numeric(_decode_csv_field(line, spans[5]))
                if time_safe_target:
                    decoded_training_rows += 1
            else:
                # Spans 4 and 5 remain raw bytes: no decode and no conversion.
                temp = float("nan")
                psal = float("nan")
                withheld_target_rows += 1
                if start <= keyed_time < stop:
                    active_fold_target_rows += 1
            depth = _numeric(_decode_csv_field(line, spans[6]))
            nominal_depth = _numeric(_decode_csv_field(line, spans[7]))
            rows.append((station, year, layer, time_text, temp, psal, depth, nominal_depth))
    frame = pd.DataFrame.from_records(rows, columns=expected_columns)
    keyed = pd.to_datetime(frame["time"], utc=True, format="mixed")
    cutoff_utc = pd.Timestamp(cutoff).tz_convert("UTC")
    withheld = frame["layer"].isin(TARGET_LAYERS) & keyed.ge(cutoff_utc)
    if not frame.loc[withheld, ["temp", "psal"]].isna().all().all():
        raise AssertionError("withheld target scalar entered the fold-blind frame")
    if not np.isfinite(frame.loc[frame["layer"].isin(PUBLIC_LAYERS), "temp"]).any():
        raise ValueError("fold-blind frame lacks finite public temperature")
    return frame, {
        "fold": str(fold["name"]),
        "rows": int(len(frame)),
        "cutoff_kst_exclusive": pd.Timestamp(cutoff).isoformat(),
        "verified_prior_fold_commitments": int(verified_prior_fold_commitments),
        "decoded_time_safe_training_target_rows": int(decoded_training_rows),
        "withheld_target_rows": int(withheld_target_rows),
        "active_fold_target_rows": int(active_fold_target_rows),
        "active_fold_target_temp_psal_scalar_fields_decoded_or_converted": 0,
        "withheld_target_temp_psal_scalar_fields_decoded_or_converted": 0,
        "anomaly_or_hidden_target_proxy_reads": 0,
        "raw_records_streamed_for_station_layer_time_routing": int(len(frame)),
        "raw_full_file_read_for_sha256_integrity_only": True,
        "public_layers_loaded_at_all_times": list(PUBLIC_LAYERS),
        "target_layers_loaded_only_before_cutoff": list(TARGET_LAYERS),
    }


def _load_metric_truth_after_commitment(
    capability: guard.ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
    observations_path: Path,
    pd_module: Any,
    np_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """The capability transition is performed before the first target decode."""

    guard.claim_truth_and_score_phase(capability, root=root, config=config)
    pd = pd_module
    np = np_module
    expected_columns = [
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    ]
    windows = [
        (
            str(fold["name"]),
            datetime.fromisoformat(str(fold["start_kst"])),
            datetime.fromisoformat(str(fold["stop_kst"])),
        )
        for fold in config["curve_protocol"]["folds"]
    ]
    selected: list[tuple[str, str, int, str, float]] = []
    converted = 0
    with observations_path.open("rb") as stream:
        line, spans = _csv_field_spans(stream.readline(), expected_fields=len(expected_columns))
        if [_decode_csv_field(line, span) for span in spans] != expected_columns:
            raise ValueError("metric-truth observations schema changed")
        for row_number, raw_row in enumerate(stream, 2):
            try:
                line, spans = _csv_field_spans(raw_row, expected_fields=len(expected_columns))
                station = _decode_csv_field(line, spans[0])
                layer = int(_decode_csv_field(line, spans[2]))
                time_text = _decode_csv_field(line, spans[3])
                keyed_time = datetime.fromisoformat(time_text)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"metric-truth key routing failed at row {row_number}") from exc
            fold_name = next(
                (name for name, start, stop in windows if start <= keyed_time < stop),
                None,
            )
            if fold_name is None or layer not in TARGET_LAYERS:
                continue
            temp = _numeric(_decode_csv_field(line, spans[4]))
            psal = _numeric(_decode_csv_field(line, spans[5]))
            converted += 2
            if np.isfinite(temp) and np.isfinite(psal):
                selected.append((fold_name, station, layer, time_text, temp))
    truth = pd.DataFrame.from_records(
        selected,
        columns=["fold", "station", "layer", "time", "truth"],
    )
    truth["time"] = pd.to_datetime(truth["time"], utc=True, format="mixed")
    if truth.empty or truth.duplicated(list(KEYS)).any():
        raise ValueError("registered metric truth is empty or duplicated")
    return truth, {
        "validation_truth_rows": int(len(truth)),
        "validation_temp_psal_scalars_converted_after_aggregate_commitment": int(converted),
        "nonvalidation_target_scalars_converted": 0,
        "hidden_test_target_scalars_converted": 0,
        "test_index_or_sample_submission_semantic_reads": 0,
    }


def _load_reference_fraction(
    workspace: Path,
    config: Mapping[str, Any],
    fraction: float,
    numerical: SimpleNamespace,
) -> Any:
    pd = numerical.pd
    np = numerical.np
    role = guard.FRACTION_ROLES[str(float(fraction))]
    pin = config["stage_a_reference"]["artifacts"][role]
    path = (workspace / str(pin["path"])).resolve(strict=True)
    if _workspace_pin(path, workspace) != dict(pin):
        raise PermissionError(f"Stage-A OOF pin changed: {role}")
    frame = pd.read_csv(path, dtype={"fold": "string", "station": "string", "time": "string"})
    seeds = [int(value) for value in config["curve_protocol"]["seed_ids"]]
    expected_columns = [*KEYS, *(f"seed_{seed}" for seed in seeds), "prediction_mean"]
    if list(frame.columns) != expected_columns:
        raise ValueError("sealed Stage-A OOF columns changed")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="mixed")
    if frame.duplicated(list(KEYS)).any() or set(frame["layer"].astype(int)) != set(TARGET_LAYERS):
        raise ValueError("sealed Stage-A OOF keys changed")
    values = frame[[f"seed_{seed}" for seed in seeds]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("sealed Stage-A OOF prediction is non-finite")
    if not np.allclose(values.mean(axis=1), frame["prediction_mean"].to_numpy(float), rtol=0, atol=5e-13):
        raise ValueError("sealed Stage-A OOF mean changed")
    if "truth" in frame.columns:
        raise AssertionError("blind Stage-A OOF unexpectedly contains truth")
    return frame


def _prefix_pin(config: Mapping[str, Any], fold: str, fraction: float) -> Mapping[str, Any]:
    matches = [
        item
        for item in config["prefix_pins"]
        if item["fold"] == fold and float(item["fraction"]) == float(fraction)
    ]
    if len(matches) != 1:
        raise ValueError("registered prefix pin is absent or duplicated")
    return matches[0]


def _timestamp_order_sha256(prefix: Any, *, np_module: Any) -> str:
    # Frozen v2 static preflight canonicalized timestamps as little-endian ns.
    values = prefix.to_numpy(dtype="datetime64[ns]").astype("<i8", copy=False)
    return hashlib.sha256(np_module.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _reference_key_sha256(frame: Any) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, list(KEYS)].itertuples(index=False, name=None):
        fold, station, layer, time = row
        digest.update(
            f"{fold}\0{station}\0{int(layer)}\0{time.isoformat()}\n".encode()
        )
    return digest.hexdigest()


def _model_state_sha256(model: Any, *, np_module: Any) -> str:
    np = np_module
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(guard.canonical_json_bytes({"shape": list(array.shape)}))
        digest.update(np.asarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _validate_required_layer4_physical_prediction(
    physical: Any,
    *,
    panel_rows: int,
    required_layer4_positions: Any,
    np_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """Validate only values that can enter the registered Layer-4 OOF array."""

    np = np_module
    values = np.asarray(physical, dtype=np.float64)
    raw_positions = np.asarray(required_layer4_positions)
    if values.shape != (int(panel_rows), 3):
        raise ValueError("Layer-4 physical prediction shape is invalid")
    if (
        raw_positions.ndim != 1
        or raw_positions.dtype.kind not in "iu"
        or len(raw_positions) == 0
    ):
        raise ValueError("registered Layer-4 OOF positions are invalid")
    positions = raw_positions.astype(np.int64, copy=False)
    if (
        (positions < 0).any()
        or (positions >= int(panel_rows)).any()
        or len(np.unique(positions)) != len(positions)
    ):
        raise ValueError("registered Layer-4 OOF positions are invalid")
    required = values[positions, 2]
    if not np.isfinite(required).all():
        raise ValueError("required Layer-4 OOF physical prediction is non-finite")
    required_mask = np.zeros(values.shape, dtype=bool)
    required_mask[positions, 2] = True
    nonrequired_nonfinite = int((~np.isfinite(values) & ~required_mask).sum())
    return values, {
        "validation_domain": "REGISTERED_STAGE_A_OOF_LAYER4_POSITIONS_ONLY",
        "panel_rows": int(panel_rows),
        "panel_physical_values": int(values.size),
        "required_layer4_positions": int(len(positions)),
        "required_layer4_values_finite": True,
        "nonrequired_nonfinite_physical_values": nonrequired_nonfinite,
        "global_full_panel_finiteness_required": False,
        "validated_before_any_persistence": True,
    }


def _predict_panel_temperature(
    model: Any,
    panel: Any,
    normalizer: Any,
    *,
    config: Mapping[str, Any],
    numerical: SimpleNamespace,
    device: Any,
    required_layer4_positions: Any,
) -> tuple[Any, dict[str, Any]]:
    np = numerical.np
    torch = numerical.torch
    recipe = config["model_and_training"]
    length = int(recipe["chunk_length"])
    bounds = numerical.deep_data.make_chunk_bounds(
        panel.segment_ids,
        length=length,
        stride=int(recipe["chunk_stride"]),
    )
    inputs = normalizer.transform_inputs(panel.inputs)
    sums = np.zeros((len(panel.times), 3), dtype=np.float64)
    counts = np.zeros((len(panel.times), 1), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(bounds), int(config["curve_protocol"]["batch_size"])):
            current = bounds[begin : begin + int(config["curve_protocol"]["batch_size"])]
            batch = np.zeros((len(current), length, inputs.shape[1]), dtype=np.float32)
            for offset, (start, stop) in enumerate(current):
                batch[offset, : stop - start] = inputs[start:stop]
            tensor = torch.from_numpy(batch).to(device=device, non_blocking=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                prediction = model(tensor)[..., 0]
            values = prediction.float().cpu().numpy()
            for offset, (start, stop) in enumerate(current):
                width = stop - start
                sums[start:stop] += values[offset, :width]
                counts[start:stop] += 1.0
    if (counts <= 0).any():
        raise RuntimeError("overlap-add inference omitted one or more timestamps")
    normalized = sums / counts
    physical = normalizer.inverse_temperature(panel, normalized)
    return _validate_required_layer4_physical_prediction(
        physical,
        panel_rows=len(panel.times),
        required_layer4_positions=required_layer4_positions,
        np_module=np,
    )


def _cell_directory(output: Path, fold: str, fraction: float, seed: int) -> Path:
    return output / "cells" / fold / f"fraction_{_fraction_token(fraction)}" / f"seed_{seed}"


def _fit_predict_cell(
    capability: guard.ExecutionCapability | object,
    *,
    workspace: Path,
    output: Path,
    config: Mapping[str, Any],
    fold: Mapping[str, Any],
    fraction: float,
    seed: int,
    panel: Any,
    reference_fold: Any,
    fold_audit: Mapping[str, Any],
    numerical: SimpleNamespace,
    progress: Progress | None,
) -> tuple[Any, dict[str, Any]]:
    """One and only one capability-ordered fit/prediction cell."""

    fold_name = str(fold["name"])
    guard.claim_cell(
        capability,
        root=workspace,
        config=config,
        fold=fold_name,
        fraction=fraction,
        seed=seed,
    )
    np = numerical.np
    pd = numerical.pd
    torch = numerical.torch
    recipe = config["model_and_training"]
    protocol = config["curve_protocol"]
    outer_start = pd.Timestamp(fold["start_kst"]).tz_convert("UTC")
    prefix = numerical.model.stage_a_prefix_times(
        panel,
        outer_start=outer_start,
        embargo_days=int(protocol["embargo_days"]),
        fraction=float(fraction),
    )
    registered = _prefix_pin(config, fold_name, fraction)
    prefix_digest = _timestamp_order_sha256(prefix, np_module=np)
    if (
        len(prefix) != int(registered["timestamps"])
        or prefix_digest != registered["timestamp_order_sha256"]
        or prefix.max() >= outer_start - pd.Timedelta(days=int(protocol["embargo_days"]))
    ):
        raise PermissionError("runtime prefix differs from the frozen Stage-A timestamp pin")
    selected = np.asarray(panel.times.isin(prefix), dtype=bool)
    normalizer = numerical.model.JointHydrographicNormalizer.fit(panel, selected)
    chunk_x, chunk_y, chunk_mask, bounds = numerical.model.materialize_joint_chunks(
        panel,
        normalizer,
        selected,
        length=int(recipe["chunk_length"]),
        stride=int(recipe["chunk_stride"]),
        minimum_joint_values=24,
    )
    if len(bounds) != int(registered["training_chunks"]):
        raise PermissionError("training chunk count differs from frozen static preflight")
    _set_deterministic_runtime(numerical, int(seed))
    device = torch.device("cuda")
    model = numerical.model.JointHydrographicTCN(
        int(recipe["input_channels"]),
        hidden=int(recipe["hidden_width"]),
        dilations=tuple(int(value) for value in recipe["dilations"]),
        dropout=float(recipe["dropout"]),
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != int(recipe["parameter_count"]):
        raise PermissionError("trainable parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    steps = 0
    last_loss = float("nan")
    model.train()
    for _epoch in range(int(protocol["epochs_per_fit"])):
        order = torch.randperm(len(chunk_x), generator=generator)
        for start in range(0, len(order), int(protocol["batch_size"])):
            indices = order[start : start + int(protocol["batch_size"])]
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
                raise RuntimeError("joint-hydrographic training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip_norm"]))
            optimizer.step()
            steps += 1
            last_loss = float(loss.detach().cpu())
    expected_steps = int(protocol["epochs_per_fit"]) * math.ceil(
        len(bounds) / int(protocol["batch_size"])
    )
    if steps != expected_steps:
        raise AssertionError("cell optimizer-step count changed")
    positions = panel.times.get_indexer(pd.to_datetime(reference_fold["time"], utc=True))
    layers = reference_fold["layer"].to_numpy(dtype=int)
    if (positions < 0).any() or not set(layers).issubset(TARGET_LAYERS):
        raise ValueError("reference validation keys are absent from fold-blind panel")
    layer4 = layers == 4
    if int(layer4.sum()) != guard.REQUIRED_LAYER4_ROWS_BY_FOLD[fold_name]:
        raise ValueError("reference validation Layer-4 row count changed")
    temperature, physical_domain_audit = _predict_panel_temperature(
        model,
        panel,
        normalizer,
        config=config,
        numerical=numerical,
        device=device,
        required_layer4_positions=positions[layer4],
    )
    reference = reference_fold[f"seed_{seed}"].to_numpy(dtype=np.float64)
    prediction = reference.copy()
    prediction[layer4] = temperature[positions[layer4], 2]
    prediction[layer4] = _csv_float_roundtrip(
        np.clip(prediction[layer4], -5.0, 45.0),
        pd_module=pd,
        np_module=np,
    )
    prediction = np.asarray(prediction, dtype=np.float64)
    if (
        not np.array_equal(prediction[~layer4], reference[~layer4])
        or not np.isfinite(prediction).all()
        or prediction.min() < -5.0
        or prediction.max() > 45.0
    ):
        raise AssertionError("Layer-4-only ablation or physical clip failed")

    cell_dir = _cell_directory(output, fold_name, fraction, seed)
    model_path = cell_dir / config["output_contract"]["per_cell"]["model_bundle"]
    prediction_path = cell_dir / config["output_contract"]["per_cell"]["blind_prediction_array"]
    receipt_path = cell_dir / config["output_contract"]["per_cell"]["cell_receipt"]
    state_sha = _model_state_sha256(model, np_module=np)
    model.cpu()
    model_buffer = io.BytesIO()
    torch.save(
        {
            "schema_version": "p2_joint_hydrographic_multitask_layer4.model_bundle.r2",
            "fold": fold_name,
            "fraction": float(fraction),
            "seed": int(seed),
            "model_state": model.state_dict(),
            "input_center": torch.from_numpy(normalizer.input_center.copy()),
            "input_scale": torch.from_numpy(normalizer.input_scale.copy()),
            "target_center": torch.from_numpy(normalizer.target_center.copy()),
            "target_scale": torch.from_numpy(normalizer.target_scale.copy()),
            "prefix_timestamp_order_sha256": prefix_digest,
        },
        model_buffer,
    )
    prediction_buffer = io.BytesIO()
    np.save(prediction_buffer, prediction.astype("<f8", copy=False), allow_pickle=False)
    guard.exclusive_bytes(model_path, model_buffer.getvalue())
    guard.exclusive_bytes(prediction_path, prediction_buffer.getvalue())
    model_pin = _workspace_pin(model_path, workspace)
    prediction_pin = _workspace_pin(prediction_path, workspace)
    receipt = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.cell_receipt.r2",
        "fold": fold_name,
        "fraction": float(fraction),
        "seed": int(seed),
        "prefix_timestamps": int(len(prefix)),
        "prefix_timestamp_order_sha256": prefix_digest,
        "training_chunks": int(len(bounds)),
        "epochs": int(protocol["epochs_per_fit"]),
        "optimizer_steps": int(steps),
        "final_training_loss": last_loss,
        "trainable_parameters": int(parameters),
        "model_state_sha256": state_sha,
        "validation_rows": int(len(prediction)),
        "validation_key_order_sha256": _reference_key_sha256(reference_fold),
        "prediction_values_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
        "model_bundle": model_pin,
        "blind_prediction_array": prediction_pin,
        "physical_prediction_domain_audit": physical_domain_audit,
        "blindness": {
            "active_fold_target_temp_psal_scalar_fields_decoded_or_converted": fold_audit[
                "active_fold_target_temp_psal_scalar_fields_decoded_or_converted"
            ],
            "outer_truth_used_for_fit_or_epoch_selection": False,
            "future_target_truth_used_for_fit": False,
            "layer2_and_layer3_exact_stage_a_seed_values": True,
            "only_layer4_temperature_replaced": True,
            "prediction_persisted_before_active_fold_truth_decode": True,
        },
        "official_promotion_allowed": False,
        "candidate_or_test_prediction": False,
        "upload_performed": False,
    }
    guard.exclusive_json(receipt_path, receipt)
    receipt_pin = _workspace_pin(receipt_path, workspace)
    artifacts = {
        "model_bundle": model_pin,
        "blind_prediction_array": prediction_pin,
        "cell_receipt": receipt_pin,
    }
    guard.complete_cell(
        capability,
        root=workspace,
        config=config,
        fold=fold_name,
        fraction=fraction,
        seed=seed,
        artifact_pins=artifacts,
    )
    del optimizer, model, chunk_x, chunk_y, chunk_mask
    torch.cuda.empty_cache()
    _emit(
        progress,
        event="p2_joint_layer4_blind_cell_complete",
        fold=fold_name,
        fraction=float(fraction),
        seed=int(seed),
        optimizer_steps=int(steps),
    )
    return prediction, {**receipt, "cell_receipt": receipt_pin}


def _create_stage_tree(output: Path, config: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output)
    os.mkdir(output / "cells")
    os.mkdir(output / "folds")
    for fold in config["curve_protocol"]["fold_major_order"]:
        fold_cells = output / "cells" / fold
        os.mkdir(fold_cells)
        fold_commitment = output / "folds" / fold
        os.mkdir(fold_commitment)
        for fraction in config["curve_protocol"]["prefix_fractions"]:
            fraction_dir = fold_cells / f"fraction_{_fraction_token(float(fraction))}"
            os.mkdir(fraction_dir)
            for seed in config["curve_protocol"]["seed_ids"]:
                os.mkdir(fraction_dir / f"seed_{seed}")


def _fold_commitment_payload(
    *,
    workspace: Path,
    config: Mapping[str, Any],
    fold: str,
    reference_by_fraction: Mapping[float, Any],
    cells: Sequence[Mapping[str, Any]],
    prior_fold_commitments: Sequence[Mapping[str, Any]],
    fold_audit: Mapping[str, Any],
) -> dict[str, Any]:
    fold_index = config["curve_protocol"]["fold_major_order"].index(fold)
    expected_fractions = {
        float(value) for value in config["curve_protocol"]["prefix_fractions"]
    }
    if set(reference_by_fraction) != expected_fractions:
        raise ValueError("fold commitment reference fractions changed")
    audit_checks = (
        fold_audit.get("fold") == fold,
        fold_audit.get("verified_prior_fold_commitments") == fold_index,
        fold_audit.get(
            "active_fold_target_temp_psal_scalar_fields_decoded_or_converted"
        )
        == 0,
        fold_audit.get(
            "withheld_target_temp_psal_scalar_fields_decoded_or_converted"
        )
        == 0,
        fold_audit.get("anomaly_or_hidden_target_proxy_reads") == 0,
        len(prior_fold_commitments) == fold_index,
    )
    if not all(audit_checks):
        raise PermissionError("fold commitment blindness audit failed")
    expected_order = [
        (float(fraction), int(seed))
        for fraction in config["curve_protocol"]["prefix_fractions"]
        for seed in config["curve_protocol"]["seed_ids"]
    ]
    observed_order = [(float(cell["fraction"]), int(cell["seed"])) for cell in cells]
    if (
        len(cells) != 15
        or observed_order != expected_order
        or any(str(cell.get("fold")) != fold for cell in cells)
    ):
        raise ValueError("fold commitment does not bind the exact 15-cell order")
    for cell in cells:
        blindness = cell.get("blindness")
        if not isinstance(blindness, Mapping) or (
            blindness.get(
                "active_fold_target_temp_psal_scalar_fields_decoded_or_converted"
            )
            != 0
            or blindness.get("outer_truth_used_for_fit_or_epoch_selection") is not False
            or blindness.get("future_target_truth_used_for_fit") is not False
            or blindness.get("layer2_and_layer3_exact_stage_a_seed_values") is not True
            or blindness.get("only_layer4_temperature_replaced") is not True
            or blindness.get("prediction_persisted_before_active_fold_truth_decode")
            is not True
        ):
            raise PermissionError("cell receipt blindness evidence failed")
    keys = {_reference_key_sha256(frame) for frame in reference_by_fraction.values()}
    rows = {len(frame) for frame in reference_by_fraction.values()}
    if len(keys) != 1 or len(rows) != 1:
        raise ValueError("Stage-A validation keys differ across prefix fractions")
    projection = [
        {
            "fraction": float(cell["fraction"]),
            "seed": int(cell["seed"]),
            "model_bundle": cell["model_bundle"],
            "blind_prediction_array": cell["blind_prediction_array"],
            "cell_receipt": cell["cell_receipt"],
            "prediction_values_sha256": cell["prediction_values_sha256"],
            "model_state_sha256": cell["model_state_sha256"],
            "optimizer_steps": int(cell["optimizer_steps"]),
        }
        for cell in cells
    ]
    cells_sha = hashlib.sha256(guard.canonical_json_bytes({"cells": projection})).hexdigest()
    combined = hashlib.sha256(
        bytes.fromhex(next(iter(keys))) + bytes.fromhex(cells_sha)
    ).hexdigest()
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.fold_commitment.r2",
        "stage": "ACTIVE_FOLD_15_BLIND_PREDICTIONS_COMMITTED",
        "config": {"path": guard.CONFIG_RELATIVE, "sha256": guard.CONFIG_SHA256},
        "fold": fold,
        "fold_order_index": fold_index,
        "cell_order": "FRACTION_THEN_SEED",
        "cell_prediction_count": 15,
        "validation_rows": int(next(iter(rows))),
        "validation_key_order_sha256": next(iter(keys)),
        "cells_sha256": cells_sha,
        "combined_fold_commitment_sha256": combined,
        "cells": projection,
        "prior_fold_commitments": list(prior_fold_commitments),
        "blind_input_audit": dict(fold_audit),
        "active_fold_target_temp_psal_scalar_decodes_before_commitment": 0,
        "truth_columns_present_in_predictions": False,
        "official_promotion_allowed": False,
        "candidate_or_test_prediction": False,
        "upload_performed": False,
    }


def _write_fold_commitment(
    capability: guard.ExecutionCapability | object,
    *,
    workspace: Path,
    output: Path,
    config: Mapping[str, Any],
    fold: str,
    reference_by_fraction: Mapping[float, Any],
    cells: Sequence[Mapping[str, Any]],
    prior_fold_commitments: Sequence[Mapping[str, Any]],
    fold_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    guard.claim_fold_commitment(capability, root=workspace, config=config, fold=fold)
    payload = _fold_commitment_payload(
        workspace=workspace,
        config=config,
        fold=fold,
        reference_by_fraction=reference_by_fraction,
        cells=cells,
        prior_fold_commitments=prior_fold_commitments,
        fold_audit=fold_audit,
    )
    path = output / "folds" / fold / config["output_contract"]["per_fold_commitment"]
    guard.exclusive_json(path, payload)
    if guard.strict_json_object(path) != payload:
        raise RuntimeError("fold commitment failed O_EXCL deep-equality reload")
    pin = _workspace_pin(path, workspace)
    guard.complete_fold_commitment(
        capability,
        root=workspace,
        config=config,
        fold=fold,
        commitment_pin=pin,
    )
    return payload, pin


def _reverify_fold_commitment(
    *,
    workspace: Path,
    config: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
    expected_pin: Mapping[str, Any],
) -> None:
    path = workspace / str(expected_pin["path"])
    if _workspace_pin(path, workspace) != dict(expected_pin):
        raise PermissionError("prior fold commitment bytes changed")
    observed = guard.strict_json_object(path)
    if observed != dict(expected_payload):
        raise PermissionError("prior fold commitment deep payload changed")
    cells = observed.get("cells")
    if not isinstance(cells, list) or len(cells) != 15:
        raise PermissionError("prior fold commitment cell projection changed")
    for cell in cells:
        for role in ("model_bundle", "blind_prediction_array", "cell_receipt"):
            pin = cell.get(role)
            if not isinstance(pin, Mapping):
                raise PermissionError(f"prior fold commitment lacks {role}")
            artifact = workspace / str(pin.get("path"))
            if not artifact.is_file() or _workspace_pin(artifact, workspace) != dict(pin):
                raise PermissionError(f"prior fold committed artifact changed: {role}")
    cells_sha = hashlib.sha256(guard.canonical_json_bytes({"cells": cells})).hexdigest()
    combined = hashlib.sha256(
        bytes.fromhex(str(observed["validation_key_order_sha256"]))
        + bytes.fromhex(cells_sha)
    ).hexdigest()
    if (
        observed.get("cell_prediction_count") != 15
        or observed.get("active_fold_target_temp_psal_scalar_decodes_before_commitment") != 0
        or observed.get("cells_sha256") != cells_sha
        or observed.get("combined_fold_commitment_sha256") != combined
    ):
        raise PermissionError("prior fold commitment is not a valid blind boundary")


def _aggregate_commitment_payload(
    *,
    workspace: Path,
    config: Mapping[str, Any],
    fold_commitments: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    blind_predictions: Mapping[tuple[str, float, int], Any],
    numerical: SimpleNamespace,
) -> dict[str, Any]:
    np = numerical.np
    expected_order = [
        (str(fold), float(fraction), int(seed))
        for fold in config["curve_protocol"]["fold_major_order"]
        for fraction in config["curve_protocol"]["prefix_fractions"]
        for seed in config["curve_protocol"]["seed_ids"]
    ]
    observed_order = [(str(cell["fold"]), float(cell["fraction"]), int(cell["seed"])) for cell in cells]
    if observed_order != expected_order or list(blind_predictions) != expected_order:
        raise ValueError("aggregate commitment cell order changed")
    if len(fold_commitments) != 3:
        raise ValueError("aggregate commitment lacks three fold commitments")
    values_hasher = hashlib.sha256()
    rows_by_cell: list[int] = []
    for key, cell in zip(expected_order, cells, strict=True):
        values = np.asarray(blind_predictions[key], dtype="<f8")
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("aggregate blind prediction is invalid")
        value_sha = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        if value_sha != cell["prediction_values_sha256"]:
            raise PermissionError("cell prediction values differ from its receipt digest")
        values_hasher.update(f"{key[0]}|{key[1]}|{key[2]}\0".encode("ascii"))
        values_hasher.update(values.tobytes(order="C"))
        rows_by_cell.append(int(len(values)))
    cell_projection = [
        {
            "fold": cell["fold"],
            "fraction": float(cell["fraction"]),
            "seed": int(cell["seed"]),
            "blind_prediction_array": cell["blind_prediction_array"],
            "cell_receipt": cell["cell_receipt"],
            "model_bundle": cell["model_bundle"],
            "prediction_values_sha256": cell["prediction_values_sha256"],
        }
        for cell in cells
    ]
    cells_sha = hashlib.sha256(guard.canonical_json_bytes({"cells": cell_projection})).hexdigest()
    folds_sha = hashlib.sha256(
        guard.canonical_json_bytes({"fold_commitments": list(fold_commitments)})
    ).hexdigest()
    combined = hashlib.sha256(
        bytes.fromhex(values_hasher.hexdigest())
        + bytes.fromhex(cells_sha)
        + bytes.fromhex(folds_sha)
    ).hexdigest()
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.prediction_commitment.r2",
        "stage": "ALL_45_BLIND_PREDICTIONS_AND_THREE_FOLDS_COMMITTED_BEFORE_METRIC_TRUTH",
        "config": {"path": guard.CONFIG_RELATIVE, "sha256": guard.CONFIG_SHA256},
        "implementation_pins": guard.implementation_pins(workspace),
        "stage_a_seal": config["stage_a_reference"]["artifacts"]["SEAL"],
        "cell_order": "FOLD_THEN_FRACTION_THEN_SEED",
        "cell_prediction_count": 45,
        "fold_commitment_count": 3,
        "rows_by_cell_in_order": rows_by_cell,
        "prediction_values_sha256": values_hasher.hexdigest(),
        "cell_artifacts_sha256": cells_sha,
        "fold_commitments_sha256": folds_sha,
        "combined_prediction_commitment_sha256": combined,
        "cells": cell_projection,
        "fold_commitments": list(fold_commitments),
        "validation_truth_scalar_decodes_before_commitment": 0,
        "truth_columns_present": False,
        "all_prediction_and_model_artifacts_o_excl_o_binary": True,
        "official_promotion_allowed": False,
        "candidate_or_test_prediction": False,
        "upload_performed": False,
    }


def _write_aggregate_commitment(
    capability: guard.ExecutionCapability | object,
    *,
    workspace: Path,
    output: Path,
    config: Mapping[str, Any],
    fold_commitments: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    blind_predictions: Mapping[tuple[str, float, int], Any],
    numerical: SimpleNamespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    guard.claim_aggregate_commitment(capability, root=workspace, config=config)
    payload = _aggregate_commitment_payload(
        workspace=workspace,
        config=config,
        fold_commitments=fold_commitments,
        cells=cells,
        blind_predictions=blind_predictions,
        numerical=numerical,
    )
    path = output / config["output_contract"]["aggregate"]["prediction_commitment"]
    guard.exclusive_json(path, payload)
    if guard.strict_json_object(path) != payload:
        raise RuntimeError("aggregate prediction commitment failed deep reload")
    pin = _workspace_pin(path, workspace)
    guard.complete_aggregate_commitment(
        capability,
        root=workspace,
        config=config,
        commitment_pin=pin,
    )
    return payload, pin


def _reverify_aggregate_commitment(
    *,
    workspace: Path,
    config: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
    expected_pin: Mapping[str, Any],
    fold_commitments: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    blind_predictions: Mapping[tuple[str, float, int], Any],
    numerical: SimpleNamespace,
) -> None:
    path = workspace / str(expected_pin["path"])
    if _workspace_pin(path, workspace) != dict(expected_pin):
        raise PermissionError("aggregate prediction commitment bytes changed")
    for pin in fold_commitments:
        artifact = workspace / str(pin["path"])
        if not artifact.is_file() or _workspace_pin(artifact, workspace) != dict(pin):
            raise PermissionError("aggregate fold commitment pin changed")
    for cell in cells:
        for role in ("model_bundle", "blind_prediction_array", "cell_receipt"):
            pin = cell[role]
            artifact = workspace / str(pin["path"])
            if not artifact.is_file() or _workspace_pin(artifact, workspace) != dict(pin):
                raise PermissionError(f"aggregate cell artifact changed: {role}")
    recomputed = _aggregate_commitment_payload(
        workspace=workspace,
        config=config,
        fold_commitments=fold_commitments,
        cells=cells,
        blind_predictions=blind_predictions,
        numerical=numerical,
    )
    if guard.strict_json_object(path) != dict(expected_payload) or recomputed != dict(expected_payload):
        raise PermissionError("in-memory blind curve differs from aggregate commitment")


def _official_weighted_rmse(
    truth: Any,
    prediction: Any,
    layer: Any,
    counts: Mapping[str, int],
    *,
    np_module: Any,
) -> float:
    np = np_module
    target = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    layers = np.asarray(layer, dtype=int)
    if target.shape != estimate.shape or target.shape != layers.shape or not len(target):
        raise ValueError("metric vectors are empty or misaligned")
    weighted = 0.0
    total = 0
    for current in TARGET_LAYERS:
        keep = layers == current
        if not keep.any():
            raise ValueError(f"metric has no rows for layer {current}")
        weight = int(counts[str(current)])
        weighted += weight * float(np.mean((estimate[keep] - target[keep]) ** 2))
        total += weight
    return float(np.sqrt(weighted / total))


def _curve_metric(
    frame: Any,
    prediction_column: str,
    counts: Mapping[str, int],
    *,
    np_module: Any,
) -> tuple[float, dict[str, float], dict[str, float]]:
    np = np_module
    fold_mse: list[float] = []
    by_fold: dict[str, float] = {}
    for fold, current in frame.groupby("fold", sort=False):
        score = _official_weighted_rmse(
            current["truth"].to_numpy(float),
            current[prediction_column].to_numpy(float),
            current["layer"].to_numpy(int),
            counts,
            np_module=np,
        )
        by_fold[str(fold)] = score
        fold_mse.append(score**2)
    aggregate = float(np.sqrt(np.mean(fold_mse)))
    by_layer = {
        str(layer): float(
            np.sqrt(
                np.mean(
                    (
                        frame.loc[frame["layer"].eq(layer), prediction_column].to_numpy(float)
                        - frame.loc[frame["layer"].eq(layer), "truth"].to_numpy(float)
                    )
                    ** 2
                )
            )
        )
        for layer in TARGET_LAYERS
    }
    return aggregate, by_fold, by_layer


def _paired_kst_day_bootstrap(
    frame: Any,
    *,
    reference_column: str,
    challenger_column: str,
    counts: Mapping[str, int],
    fold_order: Sequence[str],
    replicates: int,
    seed: int,
    pd_module: Any,
    np_module: Any,
) -> tuple[list[float], dict[str, Any]]:
    pd = pd_module
    np = np_module
    work = frame.loc[:, ["fold", "layer", "time", "truth", reference_column, challenger_column]].copy()
    work["kst_day"] = pd.to_datetime(work["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    work["reference_se"] = (work[reference_column].to_numpy(float) - work["truth"].to_numpy(float)) ** 2
    work["challenger_se"] = (work[challenger_column].to_numpy(float) - work["truth"].to_numpy(float)) ** 2
    rng = np.random.default_rng(int(seed))
    reference_fold: list[Any] = []
    challenger_fold: list[Any] = []
    clusters: dict[str, int] = {}
    if set(work["fold"].astype(str)) != set(fold_order):
        raise ValueError("bootstrap fold identity changed")
    for fold in fold_order:
        fold_frame = work.loc[work["fold"].astype(str).eq(str(fold))]
        days = sorted(fold_frame["kst_day"].unique())
        draws = rng.multinomial(len(days), np.full(len(days), 1.0 / len(days)), size=int(replicates)).astype(np.float64)
        reference_weighted = np.zeros(int(replicates), dtype=np.float64)
        challenger_weighted = np.zeros(int(replicates), dtype=np.float64)
        total_weight = 0
        for layer in TARGET_LAYERS:
            grouped = (
                fold_frame.loc[fold_frame["layer"].eq(layer)]
                .groupby("kst_day", sort=False)
                .agg(reference_se=("reference_se", "sum"), challenger_se=("challenger_se", "sum"), rows=("truth", "size"))
                .reindex(days, fill_value=0)
            )
            denominator = draws @ grouped["rows"].to_numpy(float)
            if np.any(denominator <= 0):
                raise ValueError("bootstrap omitted an entire layer")
            weight = int(counts[str(layer)])
            reference_weighted += weight * (draws @ grouped["reference_se"].to_numpy(float) / denominator)
            challenger_weighted += weight * (draws @ grouped["challenger_se"].to_numpy(float) / denominator)
            total_weight += weight
        reference_fold.append(reference_weighted / total_weight)
        challenger_fold.append(challenger_weighted / total_weight)
        clusters[str(fold)] = len(days)
    delta = np.sqrt(np.mean(np.stack(challenger_fold), axis=0)) - np.sqrt(
        np.mean(np.stack(reference_fold), axis=0)
    )
    if len(delta) != int(replicates) or not np.isfinite(delta).all():
        raise ValueError("paired KST-day bootstrap is invalid")
    interval = np.quantile(delta, [0.05, 0.95], method="linear")
    return [float(interval[0]), float(interval[1])], {
        "replicates": int(replicates),
        "cluster": "KST_day",
        "seed": int(seed),
        "fold_cluster_counts": clusters,
        "paired_reference_and_challenger": True,
        "delta_definition": "challenger_minus_reference_rmse_c",
    }


def _attach_blind_seed_predictions(
    frame: Any,
    *,
    fraction: float,
    fold_order: Sequence[str],
    seeds: Sequence[int],
    blind_predictions: Mapping[tuple[str, float, int], Any],
    np_module: Any,
) -> Any:
    """Attach each blind array to its own fold without assuming OOF block order."""

    np = np_module
    observed_folds = set(frame["fold"].astype(str))
    if observed_folds != set(fold_order):
        raise ValueError("Stage-A OOF fold identity changed")
    result = frame.copy()
    for seed in seeds:
        column = f"challenger_seed_{int(seed)}"
        result[column] = np.nan
        for fold in fold_order:
            selected = result["fold"].astype(str).eq(str(fold))
            values = np.asarray(
                blind_predictions[(str(fold), float(fraction), int(seed))],
                dtype=np.float64,
            )
            if values.ndim != 1 or len(values) != int(selected.sum()):
                raise ValueError("blind prediction rows do not match their registered fold")
            result.loc[selected, column] = values
        if not np.isfinite(result[column].to_numpy(dtype=np.float64)).all():
            raise ValueError("blind prediction fold placement is incomplete or non-finite")
    return result


def _evaluate_after_commitment(
    *,
    workspace: Path,
    output: Path,
    config: Mapping[str, Any],
    references: Mapping[float, Any],
    blind_predictions: Mapping[tuple[str, float, int], Any],
    truth: Any,
    truth_audit: Mapping[str, Any],
    fold_audits: Mapping[str, Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    commitment_pin: Mapping[str, Any],
    numerical: SimpleNamespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    np = numerical.np
    pd = numerical.pd
    protocol = config["curve_protocol"]
    counts = config["metric_and_gates"]["official_layer_counts"]
    seeds = [int(value) for value in protocol["seed_ids"]]
    points: list[dict[str, Any]] = []
    bootstrap_points: list[dict[str, Any]] = []
    oof_frames: list[Any] = []
    full_reference_fold: dict[str, float] = {}
    full_challenger_fold: dict[str, float] = {}
    full_reference_layer: dict[str, float] = {}
    full_challenger_layer: dict[str, float] = {}
    sealed_metrics = guard.strict_json_object(
        workspace / config["stage_a_reference"]["artifacts"]["CURVE_METRICS"]["path"]
    )
    for registered_fraction in protocol["prefix_fractions"]:
        fraction = float(registered_fraction)
        frame = _attach_blind_seed_predictions(
            references[fraction],
            fraction=fraction,
            fold_order=[str(fold) for fold in protocol["fold_major_order"]],
            seeds=seeds,
            blind_predictions=blind_predictions,
            np_module=np,
        )
        challenger_seed_values = frame[
            [f"challenger_seed_{seed}" for seed in seeds]
        ].to_numpy(float)
        challenger_mean = frame["prediction_mean"].to_numpy(dtype=np.float64).copy()
        layer4 = frame["layer"].to_numpy(dtype=int) == 4
        challenger_mean[layer4] = _csv_float_roundtrip(
            np.clip(challenger_seed_values[layer4].mean(axis=1), -5.0, 45.0),
            pd_module=pd,
            np_module=np,
        )
        if not np.array_equal(
            challenger_mean[~layer4],
            frame.loc[~layer4, "prediction_mean"].to_numpy(dtype=np.float64),
        ):
            raise AssertionError("mean deployment changed sealed Stage-A layers 2/3")
        frame["challenger_mean"] = challenger_mean
        frame = frame.merge(truth, on=list(KEYS), how="left", validate="one_to_one")
        if not np.isfinite(frame["truth"].to_numpy(float)).all():
            raise ValueError("Stage-A validation key lacks jointly observed truth")
        reference_metric, reference_fold, reference_layer = _curve_metric(
            frame,
            "prediction_mean",
            counts,
            np_module=np,
        )
        challenger_metric, challenger_fold, challenger_layer = _curve_metric(
            frame,
            "challenger_mean",
            counts,
            np_module=np,
        )
        sealed_point = next(
            item for item in sealed_metrics["points"] if float(item["fraction"]) == fraction
        )
        reference_seed_metrics = [
            _curve_metric(frame, f"seed_{seed}", counts, np_module=np)[0] for seed in seeds
        ]
        reference_checks = [
            np.isclose(reference_metric, float(sealed_point["prediction_mean_metric"]), rtol=0, atol=2e-12),
            np.allclose(reference_seed_metrics, sealed_point["seed_metrics"], rtol=0, atol=2e-12),
            int(sealed_point["rows"]) == len(frame),
            all(
                np.isclose(reference_fold[name], float(sealed_point["fold_metrics"][name]), rtol=0, atol=2e-12)
                for name in reference_fold
            ),
            all(
                np.isclose(reference_layer[name], float(sealed_point["layer_metrics"][name]), rtol=0, atol=2e-12)
                for name in reference_layer
            ),
        ]
        if not all(reference_checks):
            raise PermissionError("recomputed Stage-A metric differs from sealed receipt")
        bootstrap_seed = int(protocol["bootstrap_seed"])
        ci90, bootstrap = _paired_kst_day_bootstrap(
            frame,
            reference_column="prediction_mean",
            challenger_column="challenger_mean",
            counts=counts,
            fold_order=[str(fold) for fold in protocol["fold_major_order"]],
            replicates=int(protocol["bootstrap_replicates"]),
            seed=bootstrap_seed,
            pd_module=pd,
            np_module=np,
        )
        points.append(
            {
                "fraction": fraction,
                "incumbent": float(reference_metric),
                "challenger": float(challenger_metric),
                "delta": float(challenger_metric - reference_metric),
                "delta_ci90": ci90,
                "reference_by_fold": reference_fold,
                "challenger_by_fold": challenger_fold,
                "reference_by_layer": reference_layer,
                "challenger_by_layer": challenger_layer,
            }
        )
        bootstrap_points.append({"fraction": fraction, "delta_ci90": ci90, **bootstrap})
        export = frame.loc[
            :,
            [
                *KEYS,
                *(f"seed_{seed}" for seed in seeds),
                "prediction_mean",
                *(f"challenger_seed_{seed}" for seed in seeds),
                "challenger_mean",
                "truth",
            ],
        ].copy()
        export.insert(0, "fraction", fraction)
        oof_frames.append(export)
        if fraction == 1.0:
            full_reference_fold = reference_fold
            full_challenger_fold = challenger_fold
            full_reference_layer = reference_layer
            full_challenger_layer = challenger_layer
    fold_order = list(protocol["fold_major_order"])
    fold_deltas = [full_challenger_fold[name] - full_reference_fold[name] for name in fold_order]
    slice_deltas = {
        **{
            f"layer_{layer}": full_challenger_layer[str(layer)] - full_reference_layer[str(layer)]
            for layer in TARGET_LAYERS
        },
        "2024_sep_oct": full_challenger_fold["outer_2024_sep_oct"]
        - full_reference_fold["outer_2024_sep_oct"],
    }
    local_gates = numerical.curve_gate.numeric_curve_gate(
        points,
        fold_deltas=fold_deltas,
        slice_deltas=slice_deltas,
        maximum_slice_regression_c=float(
            config["metric_and_gates"]["maximum_layer_or_same_season_regression_c"]
        ),
        full_effect_c=float(
            config["metric_and_gates"]["full_delta_challenger_minus_reference_at_most_c"]
        ),
    )
    local_qualification = all(local_gates.values())
    metrics = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.metrics.r2",
        "comparison_mode": guard.MODE,
        "exact_official_incumbent_comparison": False,
        "points": points,
        "fold_delta_order": fold_order,
        "fold_deltas_challenger_minus_reference": fold_deltas,
        "slice_deltas_challenger_minus_reference": slice_deltas,
        "official_promotion_allowed": False,
    }
    bootstrap_receipt = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.bootstrap_receipt.r2",
        "prediction_commitment": dict(commitment_pin),
        "points": bootstrap_points,
        "all_five_fractions_exactly_5000_paired_kst_day_replicates": all(
            point["replicates"] == 5000 and point["cluster"] == "KST_day"
            for point in bootstrap_points
        ),
        "candidate_or_test_prediction": False,
        "upload_performed": False,
    }
    evidence = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.learning_curve_evidence.r2",
        "problem": "P2",
        "comparison_mode": guard.MODE,
        "exact_official_incumbent_comparison": False,
        "local_qualification_only": True,
        "prediction_commitment": dict(commitment_pin),
        "points": points,
        "fold_deltas_candidate_minus_incumbent": fold_deltas,
        "slice_deltas_candidate_minus_incumbent": slice_deltas,
        "local_numeric_gates": local_gates,
        "local_qualification": local_qualification,
        "leakage_checks": {
            "active_fold_target_temp_psal_scalar_decodes_before_each_fold_commitment_zero": all(
                audit["active_fold_target_temp_psal_scalar_fields_decoded_or_converted"] == 0
                for audit in fold_audits.values()
            ),
            "previous_fold_target_history_requires_verified_commitment": all(
                audit["verified_prior_fold_commitments"] == index
                for index, audit in enumerate(fold_audits.values())
            ),
            "all_45_blind_predictions_before_metric_truth": len(cells) == 45,
            "three_fold_commitments_before_metric_truth": True,
            "aggregate_commitment_reverified_after_truth_load_before_merge": True,
            "target_layer_scalars_absent_from_model_inputs": True,
            "outer_truth_not_used_for_fit_or_epoch_selection": True,
            "hidden_test_target_or_proxy_reads_zero": truth_audit["hidden_test_target_scalars_converted"] == 0,
            "test_index_or_sample_submission_semantic_reads_zero": truth_audit["test_index_or_sample_submission_semantic_reads"] == 0,
        },
        "reproducibility_checks": {
            "frozen_design_and_pure_model_exact_pins_verified": True,
            "full_transitive_source_pins_verified": True,
            "stage_a_v3_reference_exact_pins_verified": True,
            "all_15_prefix_counts_and_timestamp_digests_verified_before_fit": True,
            "three_fixed_distinct_seeds": len(seeds) == len(set(seeds)) == 3,
            "exact_45_fits": len(cells) == 45,
            "exact_6132_optimizer_steps": sum(int(cell["optimizer_steps"]) for cell in cells) == 6132,
            "layer2_and_layer3_exact_stage_a_seed_values": True,
            "only_layer4_temperature_replaced": True,
            "three_seed_prediction_mean_then_metric": True,
            "paired_kst_day_bootstrap_exactly_5000": all(
                point["replicates"] == 5000 for point in bootstrap_points
            ),
            "all_binary_model_and_prediction_writes_o_excl_o_binary": True,
            "no_resume_or_duplicate_write": True,
        },
        "output_firewall": {
            "research_only": True,
            "official_promotion_allowed": False,
            "full_fit_performed": False,
            "candidate_generated": False,
            "test_prediction_generated": False,
            "upload_performed": False,
        },
    }
    if not all(evidence["leakage_checks"].values()) or not all(
        evidence["reproducibility_checks"].values()
    ):
        raise PermissionError("Layer-4 leakage or reproducibility evidence failed")
    decision = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.gate_decision.r2",
        "status": (
            "LOCAL_QUALIFIED_RESEARCH_ONLY_PENDING_EXACT_OFFICIAL_PAIRED_AB"
            if local_qualification
            else "RESEARCH_ONLY_NO_LOCAL_QUALIFICATION"
        ),
        "local_numeric_gates": local_gates,
        "local_qualification": local_qualification,
        "passed": False,
        "official_promotion": False,
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "upload_performed": False,
    }
    return metrics, bootstrap_receipt, evidence, decision, pd.concat(oof_frames, ignore_index=True)


def _csv_bytes(frame: Any) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def execute_layer4_curve(
    *,
    capability: guard.ExecutionCapability | object,
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the exact 45-cell curve; caller reports and hashes are not trusted."""

    del preflight
    workspace = root.resolve(strict=True)
    resolved_data = data_dir.resolve(strict=True)
    canonical = guard.load_canonical_config(workspace, supplied_config=config)
    fresh_preflight = guard.static_preflight(workspace, resolved_data, supplied_config=canonical)
    guard.begin_execution(
        capability,
        root=workspace,
        config=canonical,
        preflight=fresh_preflight,
    )
    try:
        numerical = _load_numerical_stack(workspace, canonical)
        runtime = _set_deterministic_runtime(
            numerical,
            int(canonical["curve_protocol"]["seed_ids"][0]),
        )
        frozen_runtime = fresh_preflight["operational_snapshot"]["runtime"]
        for key in (
            "python",
            "numpy",
            "pandas",
            "torch",
            "torch_cuda",
            "cuda_available",
            "gpu_name",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
        ):
            if runtime[key] != frozen_runtime[key]:
                raise PermissionError(f"in-process runtime differs from preflight: {key}")
        output = guard.stage_paths(workspace, canonical)["output"]
        _create_stage_tree(output, canonical)
        started = _now_kst()
        observations_path = (resolved_data / "observations.csv").resolve(strict=True)
        if observations_path.parent != resolved_data:
            raise PermissionError("observations path escaped runtime data directory")

        references = {
            float(fraction): _load_reference_fraction(
                workspace,
                canonical,
                float(fraction),
                numerical,
            )
            for fraction in canonical["curve_protocol"]["prefix_fractions"]
        }
        blind_predictions: dict[tuple[str, float, int], Any] = {}
        all_cells: list[dict[str, Any]] = []
        fold_commitment_payloads: list[dict[str, Any]] = []
        fold_commitment_pins: list[dict[str, Any]] = []
        fold_audits: dict[str, dict[str, Any]] = {}
        for fold_index, fold in enumerate(canonical["curve_protocol"]["folds"]):
            fold_name = str(fold["name"])
            for prior_payload, prior_pin in zip(
                fold_commitment_payloads,
                fold_commitment_pins,
                strict=True,
            ):
                _reverify_fold_commitment(
                    workspace=workspace,
                    config=canonical,
                    expected_payload=prior_payload,
                    expected_pin=prior_pin,
                )
            observations, audit = _load_fold_blind_observations(
                observations_path,
                fold=fold,
                embargo_days=int(canonical["curve_protocol"]["embargo_days"]),
                verified_prior_fold_commitments=len(fold_commitment_pins),
                expected_prior_fold_commitments=fold_index,
                pd_module=numerical.pd,
                np_module=numerical.np,
            )
            panel = numerical.model.build_joint_hydrographic_panel(observations)
            if panel.inputs.shape[1] != int(canonical["model_and_training"]["input_channels"]):
                raise PermissionError("public input channel count changed")
            fold_audits[fold_name] = audit
            reference_by_fraction = {
                fraction: frame.loc[frame["fold"].eq(fold_name)].reset_index(drop=True)
                for fraction, frame in references.items()
            }
            fold_cells: list[dict[str, Any]] = []
            for fraction in canonical["curve_protocol"]["prefix_fractions"]:
                fraction = float(fraction)
                for seed in canonical["curve_protocol"]["seed_ids"]:
                    seed = int(seed)
                    prediction, receipt = _fit_predict_cell(
                        capability,
                        workspace=workspace,
                        output=output,
                        config=canonical,
                        fold=fold,
                        fraction=fraction,
                        seed=seed,
                        panel=panel,
                        reference_fold=reference_by_fraction[fraction],
                        fold_audit=audit,
                        numerical=numerical,
                        progress=progress,
                    )
                    key = (fold_name, fraction, seed)
                    blind_predictions[key] = prediction
                    fold_cells.append(receipt)
                    all_cells.append(receipt)
            fold_payload, fold_pin = _write_fold_commitment(
                capability,
                workspace=workspace,
                output=output,
                config=canonical,
                fold=fold_name,
                reference_by_fraction=reference_by_fraction,
                cells=fold_cells,
                prior_fold_commitments=fold_commitment_pins,
                fold_audit=audit,
            )
            fold_commitment_payloads.append(fold_payload)
            fold_commitment_pins.append(fold_pin)
            _emit(progress, event="p2_joint_layer4_fold_committed", fold=fold_name, cells=15)
            del panel, observations

        aggregate_payload, aggregate_pin = _write_aggregate_commitment(
            capability,
            workspace=workspace,
            output=output,
            config=canonical,
            fold_commitments=fold_commitment_pins,
            cells=all_cells,
            blind_predictions=blind_predictions,
            numerical=numerical,
        )
        _reverify_aggregate_commitment(
            workspace=workspace,
            config=canonical,
            expected_payload=aggregate_payload,
            expected_pin=aggregate_pin,
            fold_commitments=fold_commitment_pins,
            cells=all_cells,
            blind_predictions=blind_predictions,
            numerical=numerical,
        )
        truth, truth_audit = _load_metric_truth_after_commitment(
            capability,
            root=workspace,
            config=canonical,
            observations_path=observations_path,
            pd_module=numerical.pd,
            np_module=numerical.np,
        )
        _reverify_aggregate_commitment(
            workspace=workspace,
            config=canonical,
            expected_payload=aggregate_payload,
            expected_pin=aggregate_pin,
            fold_commitments=fold_commitment_pins,
            cells=all_cells,
            blind_predictions=blind_predictions,
            numerical=numerical,
        )
        metrics, bootstrap, evidence, decision, oof = _evaluate_after_commitment(
            workspace=workspace,
            output=output,
            config=canonical,
            references=references,
            blind_predictions=blind_predictions,
            truth=truth,
            truth_audit=truth_audit,
            fold_audits=fold_audits,
            cells=all_cells,
            commitment_pin=aggregate_pin,
            numerical=numerical,
        )
        aggregate = canonical["output_contract"]["aggregate"]
        oof_path = output / aggregate["learning_curve_oof"]
        metrics_path = output / aggregate["metrics"]
        bootstrap_path = output / aggregate["bootstrap_receipt"]
        evidence_path = output / aggregate["learning_curve_evidence"]
        decision_path = output / aggregate["gate_decision"]
        receipt_path = output / aggregate["training_receipt"]
        guard.exclusive_bytes(oof_path, _csv_bytes(oof))
        guard.exclusive_json(metrics_path, metrics)
        guard.exclusive_json(bootstrap_path, bootstrap)
        guard.exclusive_json(evidence_path, evidence)
        guard.exclusive_json(decision_path, decision)
        training_receipt = {
            "schema_version": "p2_joint_hydrographic_multitask_layer4.training_receipt.r2",
            "started_at_kst": started,
            "completed_at_kst": _now_kst(),
            "config": {"path": guard.CONFIG_RELATIVE, "sha256": guard.CONFIG_SHA256},
            "v1_failure_evidence_pins": fresh_preflight["operational_snapshot"][
                "v1_failure_evidence_pins"
            ],
            "implementation_correction": dict(canonical["implementation_correction"]),
            "plan": build_execution_plan(canonical),
            "runtime": runtime,
            "prediction_commitment": aggregate_pin,
            "fold_commitments": fold_commitment_pins,
            "fold_blind_input_audits": fold_audits,
            "truth_access_audit": truth_audit,
            "model_fits": len(all_cells),
            "blind_prediction_arrays": len(blind_predictions),
            "optimizer_steps": sum(int(cell["optimizer_steps"]) for cell in all_cells),
            "candidate_predictions": 0,
            "test_predictions": 0,
            "uploads": 0,
        }
        guard.exclusive_json(receipt_path, training_receipt)

        # Recheck every long-run pin and control byte before manifest creation.
        _lock, _lock_sha, end_preflight = guard.verify_consumed_attempt_lock(
            workspace,
            resolved_data,
            canonical,
        )
        if end_preflight["summary_sha256"] != fresh_preflight["summary_sha256"]:
            raise PermissionError("operational snapshot changed during Layer-4 execution")
        _reverify_aggregate_commitment(
            workspace=workspace,
            config=canonical,
            expected_payload=aggregate_payload,
            expected_pin=aggregate_pin,
            fold_commitments=fold_commitment_pins,
            cells=all_cells,
            blind_predictions=blind_predictions,
            numerical=numerical,
        )
        for payload, pin in zip(fold_commitment_payloads, fold_commitment_pins, strict=True):
            _reverify_fold_commitment(
                workspace=workspace,
                config=canonical,
                expected_payload=payload,
                expected_pin=pin,
            )

        manifest_path = output / aggregate["manifest"]
        sidecar_path = output / aggregate["manifest_sidecar"]
        seal_path = output / aggregate["seal"]
        pre_manifest_files = guard.expected_output_files(canonical) - {
            aggregate["manifest"],
            aggregate["manifest_sidecar"],
            aggregate["seal"],
        }
        artifact_pins = {
            relative: _output_pin(guard.contained_path(output, relative, must_exist=True), output)
            for relative in sorted(pre_manifest_files)
        }
        manifest = {
            "schema_version": "p2_joint_hydrographic_multitask_layer4.manifest.r2",
            "created_at_kst": _now_kst(),
            "append_only": True,
            "problem": "P2",
            "comparison_mode": guard.MODE,
            "exact_official_incumbent_comparison": False,
            "config": {"path": guard.CONFIG_RELATIVE, "sha256": guard.CONFIG_SHA256},
            "scientific_surface_pins": end_preflight["operational_snapshot"]["scientific_surface_pins"],
            "v1_failure_evidence_pins": end_preflight["operational_snapshot"][
                "v1_failure_evidence_pins"
            ],
            "implementation_correction": dict(canonical["implementation_correction"]),
            "implementation_pins": end_preflight["operational_snapshot"]["implementation_pins"],
            "source_pins": canonical["source_pins"],
            "stage_a_reference_pins": canonical["stage_a_reference"]["artifacts"],
            "data_pins": end_preflight["operational_snapshot"]["data_pins"],
            "preflight_summary_sha256": end_preflight["summary_sha256"],
            "prediction_commitment": _output_pin(output / aggregate["prediction_commitment"], output),
            "artifacts": artifact_pins,
            "local_qualification": bool(decision["local_qualification"]),
            "official_promotion_allowed": False,
            "full_fit_performed": False,
            "candidate_generated": False,
            "test_prediction_generated": False,
            "uploads": 0,
        }
        guard.exclusive_json(manifest_path, manifest)
        guard.exclusive_bytes(
            sidecar_path,
            f"{guard.sha256_file(manifest_path)}  manifest.json\n".encode("ascii"),
        )
        seal = {
            "schema_version": "p2_joint_hydrographic_multitask_layer4.seal.r2",
            "complete": True,
            "status": decision["status"],
            "comparison_mode": guard.MODE,
            "exact_official_incumbent_comparison": False,
            "local_qualification": bool(decision["local_qualification"]),
            "official_promotion_allowed": False,
            "config": {"path": guard.CONFIG_RELATIVE, "sha256": guard.CONFIG_SHA256},
            "v1_failure_tombstone": canonical["v1_failure_evidence"][
                "failure_tombstone"
            ],
            "prediction_commitment": _output_pin(output / aggregate["prediction_commitment"], output),
            "manifest": _output_pin(manifest_path, output),
            "manifest_sidecar": _output_pin(sidecar_path, output),
            "candidate_generated": False,
            "test_prediction_generated": False,
            "upload_count": 0,
        }
        guard.exclusive_json(seal_path, seal)
        guard.complete_execution_phase(capability, root=workspace, config=canonical)
        verified = guard.verify_seal(workspace, canonical)
        return {
            "schema_version": "p2_joint_hydrographic_multitask_layer4.execution_result.r2",
            "status": decision["status"],
            "local_qualification": bool(decision["local_qualification"]),
            "official_promotion_allowed": False,
            "output": output.relative_to(workspace).as_posix(),
            "model_fits": 45,
            "blind_prediction_arrays": 45,
            "optimizer_steps": 6132,
            "prediction_commitment_sha256": aggregate_pin["sha256"],
            "manifest_sha256": verified["manifest_sha256"],
            "seal_sha256": verified["seal_sha256"],
            "candidate_generated": False,
            "test_prediction_generated": False,
            "uploads": 0,
        }
    finally:
        try:
            guard.revoke_execution_capability(capability)
        except PermissionError:
            pass


__all__ = [
    "_aggregate_commitment_payload",
    "_csv_field_spans",
    "_decode_csv_field",
    "_fit_predict_cell",
    "_load_fold_blind_observations",
    "_load_metric_truth_after_commitment",
    "_paired_kst_day_bootstrap",
    "_timestamp_order_sha256",
    "build_execution_plan",
    "execute_layer4_curve",
]
