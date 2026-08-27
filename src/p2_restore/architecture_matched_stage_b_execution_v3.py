"""Executable P2 architecture-matched Stage-B challenger v3.

The module deliberately imports no numerical package at import time.  Every
direct execution starts with the canonical static preflight, which verifies
the completed Stage-A v3 seal and all preregistered bytes before the numerical
stack is imported.  The run writes only aggregate learning-curve evidence;
it never fits on the full history, predicts the hidden test set, creates a
candidate CSV, or uploads anything.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

from p2_restore.architecture_matched_stage_b_contract_v3 import (
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    FRACTION_ROLES,
    MODE,
    canonical_json_bytes,
    contained_path,
    exclusive_json,
    implementation_pins,
    load_canonical_config,
    sha256_file,
    stage_paths,
    static_preflight,
    strict_json_object,
    verify_consumed_attempt_lock,
    verify_execution_authorization,
    verify_pre_execution_qa,
    verify_stage_a_reference,
)

Progress = Callable[[dict[str, Any]], None]
TARGET_LAYERS = (2, 3, 4)
KEYS = ("fold", "station", "layer", "time")
KST = ZoneInfo("Asia/Seoul")


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _emit(progress: Progress | None, **payload: Any) -> None:
    if progress is not None:
        progress(payload)


def _derived_seed(base_seed: int, *labels: object) -> int:
    text = "|".join((str(base_seed), *(str(label) for label in labels)))
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value % 2_147_483_646 + 1


def _fraction_token(fraction: float) -> str:
    try:
        return {
            0.4: "040",
            0.55: "055",
            0.7: "070",
            0.85: "085",
            1.0: "100",
        }[float(fraction)]
    except KeyError as exc:
        raise ValueError("unregistered Stage-B prefix fraction") from exc


def build_execution_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = config["curve_protocol"]
    folds = [str(fold["name"]) for fold in protocol["outer_folds"]]
    fractions = [float(value) for value in protocol["prefix_fractions"]]
    seeds = [int(value) for value in protocol["seed_ids"]]
    cells = len(folds) * len(fractions)
    return {
        "schema_version": "p2_architecture_matched_stage_b_execution.plan.v3",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "hypothesis_id": config["hypothesis"]["id"],
        "folds": folds,
        "prefix_fractions": fractions,
        "complete_pipeline_seeds": seeds,
        "outer_prefix_cells": cells,
        "challenger_fits": cells * len(seeds),
        "challenger_outer_predictions": cells * len(seeds),
        "paired_kst_day_bootstrap_replicates_per_fraction": int(protocol["bootstrap_replicates"]),
        "full_fit_jobs": 0,
        "test_predictions": 0,
        "candidate_files": 0,
        "uploads": 0,
    }


def _pin(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
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
        raise PermissionError(f"late-imported Stage-B dependency has no file: {role}")
    observed_path = Path(module_file).resolve(strict=True)
    canonical_path = (workspace / str(expected["path"])).resolve(strict=True)
    if observed_path != canonical_path or _pin(observed_path, workspace) != dict(expected):
        raise PermissionError(f"late-imported Stage-B dependency fails exact pin: {role}")


def _load_numerical_stack(
    workspace: Path,
    config: Mapping[str, Any],
) -> SimpleNamespace:
    """Late-import and byte-check every numerical module used by Stage B."""

    modules = {
        "STAGE_A_V2_ENGINE": importlib.import_module(
            "p2_restore.architecture_matched_stage_a_execution_v2"
        ),
        "DEEP_DATA": importlib.import_module("p2_restore.deep_data"),
        "PROFILE_PROJECTION": importlib.import_module("p2_restore.profile_projection"),
        "FINAL_INFERENCE": importlib.import_module("p2_restore.final_inference"),
        "DATA": importlib.import_module("p2_restore.data"),
        "CURVE_GATE": importlib.import_module("p2_restore.meaningful_learning_curve"),
        "MODEL_MODULE": importlib.import_module("p2_restore.model"),
        "PACKAGE_INIT": importlib.import_module("p2_restore"),
        "CENTRAL_V3_EVALUATOR": importlib.import_module("ocean_goal.meaningful_score_v3"),
    }
    for role, module in modules.items():
        _verify_loaded_module(
            module,
            workspace=workspace,
            expected=config["source_pins"][role],
            role=role,
        )
    engine = modules["STAGE_A_V2_ENGINE"]
    return SimpleNamespace(
        engine=engine,
        np=engine.np,
        pd=engine.pd,
        torch=engine.torch,
        projection=modules["PROFILE_PROJECTION"],
        final_inference=modules["FINAL_INFERENCE"],
        curve_gate=modules["CURVE_GATE"],
        central=modules["CENTRAL_V3_EVALUATOR"],
    )


def _robust_transform(
    train: Any,
    query: Any,
    *,
    clip: float,
    np_module: Any,
) -> tuple[Any, Any]:
    np = np_module
    train_values = np.asarray(train, dtype=np.float64)
    query_values = np.asarray(query, dtype=np.float64)
    if train_values.ndim != 2 or query_values.ndim != 2:
        raise ValueError("conditional-analog features must be two dimensional")
    if train_values.shape[1] != query_values.shape[1] or not len(train_values):
        raise ValueError("conditional-analog train/query feature shapes differ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmedian(train_values, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.nanmedian(np.abs(train_values - center), axis=0) * 1.4826
        fallback = np.nanstd(train_values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

    def transform(values: Any) -> Any:
        standardized = (values - center) / scale
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        return np.clip(standardized, -clip, clip).astype(np.float32)

    return transform(train_values), transform(query_values)


def _weighted_rank2_projection(
    neighbor_targets: Any,
    weights: Any,
    anchors: Any,
    *,
    torch_module: Any,
) -> Any:
    """Project anchors onto batched weighted local rank-2 affine manifolds."""

    torch = torch_module
    if (
        neighbor_targets.ndim != 3
        or neighbor_targets.shape[2] != 3
        or weights.shape != neighbor_targets.shape[:2]
        or anchors.shape != (neighbor_targets.shape[0], 3)
    ):
        raise ValueError("rank-2 projection tensors are misaligned")
    mean = torch.einsum("bk,bkj->bj", weights, neighbor_targets)
    centered = neighbor_targets - mean[:, None, :]
    covariance = torch.einsum("bk,bki,bkj->bij", weights, centered, centered)
    _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, :, -2:]
    offset = anchors - mean
    coordinates = torch.einsum("bij,bi->bj", basis, offset)
    return mean + torch.einsum("bij,bj->bi", basis, coordinates)


def _fit_predict_rank2(
    train_features: Any,
    train_targets: Any,
    query_features: Any,
    anchor_profiles: Any,
    *,
    seed: int,
    projection_dimensions: int,
    nearest_neighbors: int,
    batch_size: int,
    standardized_clip: float,
    np_module: Any,
    torch_module: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fit one seeded public-covariate analog model and predict one outer fold."""

    np = np_module
    torch = torch_module
    targets = np.asarray(train_targets, dtype=np.float64)
    anchors = np.asarray(anchor_profiles, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
        raise ValueError("rank-2 training profiles must be finite three-layer targets")
    if anchors.ndim != 2 or anchors.shape[1] != 3 or not np.isfinite(anchors).all():
        raise ValueError("rank-2 anchors must be finite three-layer profiles")
    train_scaled, query_scaled = _robust_transform(
        train_features,
        query_features,
        clip=standardized_clip,
        np_module=np,
    )
    if len(query_scaled) != len(anchors):
        raise ValueError("query features and Stage-A anchors are misaligned")
    dimension = min(int(projection_dimensions), train_scaled.shape[1])
    if dimension < 2:
        raise ValueError("seeded analog projection dimension is degenerate")
    rng = np.random.default_rng(int(seed))
    gaussian = rng.standard_normal((train_scaled.shape[1], dimension))
    projection, _ = np.linalg.qr(gaussian, mode="reduced")
    projection = np.asarray(projection[:, :dimension], dtype=np.float32)
    train_projected = np.asarray(train_scaled @ projection, dtype=np.float32)
    query_projected = np.asarray(query_scaled @ projection, dtype=np.float32)
    neighbors = min(int(nearest_neighbors), len(train_projected))
    if neighbors < 3:
        raise ValueError("fewer than three complete prefix profiles are available")

    train_x = torch.as_tensor(train_projected, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    query_x = torch.as_tensor(query_projected, dtype=torch.float32, device=device)
    anchor_y = torch.as_tensor(anchors, dtype=torch.float32, device=device)
    output: list[Any] = []
    with torch.no_grad():
        for start in range(0, len(query_projected), int(batch_size)):
            stop = min(len(query_projected), start + int(batch_size))
            distances = torch.cdist(query_x[start:stop], train_x)
            values, indices = torch.topk(
                distances,
                k=neighbors,
                dim=1,
                largest=False,
                sorted=True,
            )
            scale = values[:, -1].clamp_min(1e-6)
            weights = torch.softmax(-values / scale[:, None], dim=1)
            neighbor_targets = train_y[indices]
            projected = _weighted_rank2_projection(
                neighbor_targets,
                weights,
                anchor_y[start:stop],
                torch_module=torch,
            )
            output.append(projected.detach().cpu())
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)
    prediction = torch.cat(output, dim=0).numpy().astype(np.float64)
    if prediction.shape != anchors.shape or not np.isfinite(prediction).all():
        raise ValueError("rank-2 conditional-analog prediction is invalid")
    return prediction, {
        "training_complete_profile_count": int(len(train_projected)),
        "query_complete_profile_count": int(len(query_projected)),
        "projection_dimensions": int(dimension),
        "nearest_neighbors": int(neighbors),
        "local_manifold_rank": 2,
        "projection_matrix_sha256": hashlib.sha256(
            np.asarray(projection, dtype="<f4").tobytes()
        ).hexdigest(),
    }


def _complete_profile_rows(frame: Any, *, pd_module: Any, np_module: Any) -> Any:
    pd = pd_module
    np = np_module
    keyed = frame.loc[:, ["station", "time", "layer"]].copy()
    keyed["time"] = pd.to_datetime(keyed["time"], utc=True)
    keyed["_row"] = np.arange(len(keyed), dtype=int)
    group = keyed.groupby(["station", "time"], sort=False, dropna=False)["layer"]
    complete = keyed.loc[
        group.transform("size").eq(3)
        & group.transform("nunique").eq(3)
        & keyed["layer"].isin(TARGET_LAYERS)
    ].sort_values(["station", "time", "layer"])
    if complete.empty:
        return np.empty((0, 3), dtype=int)
    valid = complete.groupby(["station", "time"], sort=False, dropna=False)["layer"].transform(
        lambda values: set(values.astype(int)) == set(TARGET_LAYERS)
    )
    complete = complete.loc[valid]
    if len(complete) % 3:
        raise ValueError("complete outer profiles cannot be reshaped to three layers")
    rows = complete["_row"].to_numpy(dtype=int).reshape(-1, 3)
    if not np.array_equal(
        frame.iloc[rows.ravel()]["layer"].to_numpy(int), np.tile(TARGET_LAYERS, len(rows))
    ):
        raise ValueError("complete profile rows are not ordered by target layer")
    return rows


def _prediction_sha256(values: Any, *, np_module: Any) -> str:
    payload = np_module.asarray(values, dtype="<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def _csv_field_spans(
    raw_line: bytes, *, expected_fields: int
) -> tuple[bytes, list[tuple[int, int]]]:
    """Locate single-record RFC-4180 fields without decoding their contents."""

    line = raw_line[:-2] if raw_line.endswith(b"\r\n") else raw_line.rstrip(b"\n")
    if b"\r" in line or b"\n" in line:
        raise ValueError("P2 CSV contains an unsupported embedded newline")
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    in_quotes = False
    while index < len(line):
        value = line[index]
        if value == 34:
            if in_quotes and index + 1 < len(line) and line[index + 1] == 34:
                index += 2
                continue
            in_quotes = not in_quotes
        elif value == 44 and not in_quotes:
            spans.append((start, index))
            start = index + 1
        index += 1
    if in_quotes:
        raise ValueError("P2 CSV contains an unterminated quoted field")
    spans.append((start, len(line)))
    if len(spans) != int(expected_fields):
        raise ValueError("P2 CSV row width changed")
    return line, spans


def _decode_csv_field(raw_line: bytes, span: tuple[int, int]) -> str:
    """Decode exactly one explicitly selected field and no neighboring field."""

    start, stop = span
    token = raw_line[start:stop].decode("utf-8")
    parsed = next(csv.reader([token]))
    if len(parsed) != 1:
        raise ValueError("selected P2 CSV field did not decode to one scalar")
    return parsed[0]


def _challenger_fold_seed(
    *,
    reference_fold: Any,
    reference_column: str,
    panel: Any,
    endpoints: Any,
    prefix: Any,
    model: Mapping[str, Any],
    derived_seed: int,
    numerical: SimpleNamespace,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    np = numerical.np
    pd = numerical.pd
    rows = _complete_profile_rows(reference_fold, pd_module=pd, np_module=np)
    output = reference_fold[reference_column].to_numpy(dtype=np.float64).copy()
    train_mask = np.asarray(panel.times.isin(prefix), dtype=bool) & panel.target_mask.all(axis=1)
    if not train_mask.any():
        raise ValueError("fold-prefix has no jointly complete target profiles")
    if len(rows):
        group_times = pd.to_datetime(reference_fold.iloc[rows[:, 0]]["time"], utc=True)
        positions = panel.times.get_indexer(group_times)
        if (positions < 0).any():
            raise ValueError("outer reference time is absent from P2Panel")
        projected, diagnostics = _fit_predict_rank2(
            panel.inputs[train_mask],
            panel.target[train_mask],
            panel.inputs[positions],
            output[rows],
            seed=derived_seed,
            projection_dimensions=int(model["projection_dimensions"]),
            nearest_neighbors=int(model["nearest_neighbors"]),
            batch_size=int(model["query_batch_size"]),
            standardized_clip=float(model.get("standardized_clip", 12.0)),
            np_module=np,
            torch_module=numerical.torch,
            device=device,
        )
        output[rows.ravel()] = projected.ravel()
    else:
        diagnostics = {
            "training_complete_profile_count": int(train_mask.sum()),
            "query_complete_profile_count": 0,
            "projection_dimensions": int(model["projection_dimensions"]),
            "nearest_neighbors": min(int(model["nearest_neighbors"]), int(train_mask.sum())),
            "local_manifold_rank": 2,
            "projection_matrix_sha256": "0" * 64,
        }
    physical = numerical.projection.project_profiles_vectorized(
        reference_fold,
        output,
        endpoints,
    )
    final = numerical.final_inference.csv_float_roundtrip(physical.prediction)
    if not np.isfinite(final).all():
        raise ValueError("Stage-B postprocessed challenger is non-finite")
    incomplete_rows = len(reference_fold) - int(rows.size)
    return final, {
        **diagnostics,
        "outer_rows": int(len(reference_fold)),
        "complete_profile_rows": int(rows.size),
        "incomplete_profile_rows_identity_before_common_postprocess": int(incomplete_rows),
        "public_endpoint_projection_eligible_rows": int(physical.eligible_mask.sum()),
        "public_endpoint_projection_active_rows": int(physical.active_mask.sum()),
        "prediction_sha256": _prediction_sha256(final, np_module=np),
    }


def _load_fold_blind_observations(
    observations_path: Path,
    *,
    outer_start: Any,
    embargo_days: int,
    pd_module: Any,
    np_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load public inputs and only this fold's time-safe training targets.

    Target-layer temperature/salinity fields at or after the embargo cutoff
    are deliberately never decoded or converted.  They become NaN in the fold-local
    frame, so neither panel construction nor the challenger can observe outer
    validation truth before the aggregate prediction commitment is written.
    """

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
    cutoff = outer_start.tz_convert("Asia/Seoul") - pd.Timedelta(days=int(embargo_days))
    cutoff_datetime = cutoff.to_pydatetime()
    cutoff_text = cutoff.isoformat()
    rows: list[tuple[Any, ...]] = []
    allowed_target_rows = 0
    withheld_target_rows = 0
    latest_allowed_target_time = ""

    def numeric(value: str) -> float:
        return float(value) if value else float("nan")

    with observations_path.open("rb") as stream:
        raw_header = stream.readline()
        if not raw_header:
            raise ValueError("observations.csv is empty")
        header_line, header_spans = _csv_field_spans(
            raw_header,
            expected_fields=len(expected_columns),
        )
        header = [_decode_csv_field(header_line, span) for span in header_spans]
        if header != expected_columns:
            raise ValueError("blind observations schema changed")
        for row_number, raw_row in enumerate(stream, 2):
            try:
                row_line, spans = _csv_field_spans(
                    raw_row,
                    expected_fields=len(expected_columns),
                )
            except ValueError as exc:
                raise ValueError(f"observations row changed at line {row_number}") from exc
            station, year_text, layer_text, time_text = (
                _decode_csv_field(row_line, spans[index]) for index in range(4)
            )
            if not time_text.endswith("+09:00"):
                raise ValueError("blind observation timestamp lost its KST offset")
            try:
                keyed_time = datetime.fromisoformat(time_text)
            except ValueError as exc:
                raise ValueError(
                    f"blind observation timestamp is invalid at line {row_number}"
                ) from exc
            if keyed_time.utcoffset() != KST.utcoffset(keyed_time):
                raise ValueError("blind observation timestamp lost its KST offset")
            layer = int(layer_text)
            if layer not in (1, 2, 3, 4, 5):
                raise ValueError("blind observation layer is outside 1..5")
            public_layer = layer in (1, 5)
            time_safe_target = layer in TARGET_LAYERS and keyed_time < cutoff_datetime
            if public_layer or time_safe_target:
                temp = numeric(_decode_csv_field(row_line, spans[4]))
                psal = numeric(_decode_csv_field(row_line, spans[5]))
                if time_safe_target:
                    allowed_target_rows += 1
                    latest_allowed_target_time = max(latest_allowed_target_time, time_text)
            else:
                # Do not decode, convert, or otherwise inspect fields 4 and 5.
                temp = float("nan")
                psal = float("nan")
                withheld_target_rows += 1
            rows.append(
                (
                    station,
                    int(year_text),
                    layer,
                    time_text,
                    temp,
                    psal,
                    numeric(_decode_csv_field(row_line, spans[6])),
                    numeric(_decode_csv_field(row_line, spans[7])),
                )
            )
    frame = pd.DataFrame.from_records(rows, columns=expected_columns)
    keyed_time = pd.to_datetime(frame["time"], utc=True, format="mixed")
    cutoff_utc = cutoff.tz_convert("UTC")
    withheld = frame["layer"].isin(TARGET_LAYERS) & keyed_time.ge(cutoff_utc)
    if not frame.loc[withheld, ["temp", "psal"]].isna().all().all():
        raise AssertionError("fold-local blind frame retained validation target truth")
    allowed = frame["layer"].isin(TARGET_LAYERS) & keyed_time.lt(cutoff_utc)
    if int(allowed.sum()) != allowed_target_rows or int(withheld.sum()) != withheld_target_rows:
        raise AssertionError("blind target-row accounting changed")
    if not len(frame) or not np.isfinite(frame.loc[frame["layer"].isin((1, 5)), "temp"]).any():
        raise ValueError("blind observations lack public temperature inputs")
    return frame, {
        "rows": int(len(frame)),
        "cutoff_kst_exclusive": cutoff_text,
        "allowed_training_target_rows": int(allowed_target_rows),
        "latest_allowed_training_target_time_kst": latest_allowed_target_time,
        "withheld_target_rows": int(withheld_target_rows),
        "withheld_target_scalar_fields_decoded_or_converted": 0,
        "validation_target_temp_psal_strings_converted": 0,
        "validation_truth_columns_read_by_challenger": 0,
        "raw_source_records_streamed_for_key_routing": int(len(frame)),
        "raw_source_bytes_preflight_hashed_for_integrity_only": True,
        "public_layers_loaded_at_all_times": [1, 5],
        "target_layers_loaded_only_before_cutoff": [2, 3, 4],
    }


def _load_validation_truth_after_commitment(
    observations_path: Path,
    *,
    config: Mapping[str, Any],
    pd_module: Any,
    np_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load only registered outer-fold truths after the blind commitment."""

    pd = pd_module
    np = np_module
    windows = [
        (
            datetime.fromisoformat(str(fold["start_kst"])),
            datetime.fromisoformat(str(fold["stop_kst"])),
        )
        for fold in config["curve_protocol"]["outer_folds"]
    ]
    selected: list[tuple[str, int, str, float]] = []
    converted_scalar_count = 0
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
    with observations_path.open("rb") as stream:
        raw_header = stream.readline()
        if not raw_header:
            raise ValueError("observations.csv is empty during truth attachment")
        header_line, header_spans = _csv_field_spans(
            raw_header,
            expected_fields=len(expected_columns),
        )
        header = [_decode_csv_field(header_line, span) for span in header_spans]
        if header != expected_columns:
            raise ValueError("validation-truth observations schema changed")
        for row_number, raw_row in enumerate(stream, 2):
            try:
                row_line, spans = _csv_field_spans(
                    raw_row,
                    expected_fields=len(expected_columns),
                )
            except ValueError as exc:
                raise ValueError(f"observations row changed at line {row_number}") from exc
            station = _decode_csv_field(row_line, spans[0])
            layer = int(_decode_csv_field(row_line, spans[2]))
            time_text = _decode_csv_field(row_line, spans[3])
            if not time_text.endswith("+09:00"):
                raise ValueError("validation-truth timestamp lost its KST offset")
            try:
                keyed_time = datetime.fromisoformat(time_text)
            except ValueError as exc:
                raise ValueError(
                    f"validation-truth timestamp is invalid at line {row_number}"
                ) from exc
            if keyed_time.utcoffset() != KST.utcoffset(keyed_time):
                raise ValueError("validation-truth timestamp lost its KST offset")
            in_registered_outer = any(start <= keyed_time < stop for start, stop in windows)
            if layer not in TARGET_LAYERS or not in_registered_outer:
                # Target scalars outside the three fixed validation folds are
                # deliberately not decoded, converted, or retained.
                continue
            temp_text = _decode_csv_field(row_line, spans[4])
            psal_text = _decode_csv_field(row_line, spans[5])
            temp = float(temp_text) if temp_text else float("nan")
            psal = float(psal_text) if psal_text else float("nan")
            converted_scalar_count += 2
            if np.isfinite(temp) and np.isfinite(psal):
                selected.append((station, layer, time_text, temp))
    truth = pd.DataFrame.from_records(
        selected,
        columns=["station", "layer", "time", "truth"],
    )
    truth["time"] = pd.to_datetime(truth["time"], utc=True, format="mixed")
    if truth.empty or truth.duplicated(["station", "layer", "time"]).any():
        raise ValueError("joint target truth keys are duplicated")
    return truth, {
        "validation_truth_rows": int(len(truth)),
        "validation_target_scalars_converted_after_commitment": int(converted_scalar_count),
        "registered_outer_windows_kst": [
            [start.isoformat(), stop.isoformat()] for start, stop in windows
        ],
        "nonvalidation_target_scalars_converted": 0,
        "hidden_test_target_scalars_converted": 0,
    }


def _load_reference_fraction_blind(
    *,
    workspace: Path,
    config: Mapping[str, Any],
    fraction: float,
    numerical: SimpleNamespace,
) -> Any:
    pd = numerical.pd
    np = numerical.np
    role = FRACTION_ROLES[str(float(fraction))]
    pin = config["stage_a_reference"]["artifacts"][role]
    path = (workspace / str(pin["path"])).resolve(strict=True)
    frame = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    seeds = config["curve_protocol"]["seed_ids"]
    expected_columns = [
        *KEYS,
        *(f"seed_{seed}" for seed in seeds),
        "prediction_mean",
    ]
    if list(frame.columns) != expected_columns:
        raise ValueError("sealed Stage-A OOF columns changed")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    if frame.duplicated(list(KEYS)).any() or not set(frame["layer"]).issubset(TARGET_LAYERS):
        raise ValueError("sealed Stage-A OOF keys are invalid")
    expected_folds = [fold["name"] for fold in config["curve_protocol"]["outer_folds"]]
    observed_folds = list(frame["fold"].drop_duplicates())
    if len(observed_folds) != len(expected_folds) or set(observed_folds) != set(expected_folds):
        raise ValueError("sealed Stage-A OOF fold identities changed")
    numeric_columns = [*(f"seed_{seed}" for seed in seeds), "prediction_mean"]
    if not np.isfinite(frame[numeric_columns].to_numpy(float)).all():
        raise ValueError("sealed Stage-A OOF prediction is non-finite")
    recalculated = frame[[f"seed_{seed}" for seed in seeds]].to_numpy(float).mean(axis=1)
    if not np.allclose(
        recalculated,
        frame["prediction_mean"].to_numpy(float),
        rtol=0.0,
        atol=5e-13,
    ):
        raise ValueError("sealed Stage-A OOF prediction mean is not the three-seed mean")
    for fold in config["curve_protocol"]["outer_folds"]:
        current = frame.loc[frame["fold"].eq(fold["name"]), "time"]
        start = pd.Timestamp(fold["start_kst"]).tz_convert("UTC")
        stop = pd.Timestamp(fold["stop_kst"]).tz_convert("UTC")
        if current.empty or not (current.ge(start) & current.lt(stop)).all():
            raise ValueError(f"sealed Stage-A OOF escaped outer fold: {fold['name']}")
    if "truth" in frame.columns:
        raise AssertionError("blind Stage-A reference unexpectedly contains truth")
    return frame


def _attach_validation_truth(blind: Any, truth: Any, *, np_module: Any) -> Any:
    merged = blind.merge(
        truth,
        on=["station", "layer", "time"],
        how="left",
        validate="one_to_one",
    )
    if not np_module.isfinite(merged["truth"].to_numpy(float)).all():
        raise ValueError("sealed Stage-A OOF key lacks jointly observed validation truth")
    return merged


def _assert_reference_metrics(
    frame: Any,
    *,
    fraction: float,
    config: Mapping[str, Any],
    sealed_metrics: Mapping[str, Any],
    numerical: SimpleNamespace,
) -> tuple[float, list[float], dict[str, float], dict[str, float]]:
    np = numerical.np
    engine = numerical.engine
    counts = config["metric_and_gates"]["official_layer_counts"]
    seeds = config["curve_protocol"]["seed_ids"]
    point = next(
        item for item in sealed_metrics["points"] if float(item["fraction"]) == float(fraction)
    )
    seed_metrics = []
    for seed in seeds:
        metric, _fold, _layer = engine._curve_metric(frame, f"seed_{seed}", counts)
        seed_metrics.append(float(metric))
    metric, by_fold, by_layer = engine._curve_metric(frame, "prediction_mean", counts)
    expected = [float(value) for value in point["seed_metrics"]]
    comparisons = [
        np.isclose(metric, float(point["prediction_mean_metric"]), rtol=0.0, atol=2e-12),
        np.allclose(seed_metrics, expected, rtol=0.0, atol=2e-12),
        int(point["rows"]) == len(frame),
        all(
            np.isclose(by_fold[name], float(point["fold_metrics"][name]), rtol=0.0, atol=2e-12)
            for name in by_fold
        ),
        all(
            np.isclose(by_layer[name], float(point["layer_metrics"][name]), rtol=0.0, atol=2e-12)
            for name in by_layer
        ),
    ]
    if not all(comparisons):
        raise ValueError(f"sealed Stage-A aggregate metric mismatch at fraction {fraction}")
    return float(metric), seed_metrics, by_fold, by_layer


def _paired_kst_day_bootstrap(
    frame: Any,
    *,
    reference_column: str,
    challenger_column: str,
    counts: Mapping[str, int],
    replicates: int,
    seed: int,
    pd_module: Any,
    np_module: Any,
) -> tuple[list[float], dict[str, Any]]:
    """Paired cluster bootstrap, resampling KST days independently by fold."""

    pd = pd_module
    np = np_module
    work = frame.loc[
        :, ["fold", "layer", "time", "truth", reference_column, challenger_column]
    ].copy()
    work["kst_day"] = (
        pd.to_datetime(work["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    )
    work["reference_se"] = (
        work[reference_column].to_numpy(float) - work["truth"].to_numpy(float)
    ) ** 2
    work["challenger_se"] = (
        work[challenger_column].to_numpy(float) - work["truth"].to_numpy(float)
    ) ** 2
    rng = np.random.default_rng(int(seed))
    reference_fold_mse: list[Any] = []
    challenger_fold_mse: list[Any] = []
    cluster_counts: dict[str, int] = {}
    for fold, fold_frame in work.groupby("fold", sort=False):
        days = sorted(fold_frame["kst_day"].unique())
        if not days:
            raise ValueError("bootstrap fold has no KST-day clusters")
        draws = rng.multinomial(
            len(days),
            np.full(len(days), 1.0 / len(days)),
            size=int(replicates),
        ).astype(np.float64)
        weighted_reference = np.zeros(int(replicates), dtype=np.float64)
        weighted_challenger = np.zeros(int(replicates), dtype=np.float64)
        total_weight = 0
        for layer in TARGET_LAYERS:
            current = fold_frame.loc[fold_frame["layer"].eq(layer)]
            grouped = (
                current.groupby("kst_day", sort=False)
                .agg(
                    reference_se=("reference_se", "sum"),
                    challenger_se=("challenger_se", "sum"),
                    rows=("truth", "size"),
                )
                .reindex(days, fill_value=0)
            )
            denominator = draws @ grouped["rows"].to_numpy(float)
            if np.any(denominator <= 0):
                raise ValueError("bootstrap resample omitted an entire target layer")
            layer_reference = draws @ grouped["reference_se"].to_numpy(float) / denominator
            layer_challenger = draws @ grouped["challenger_se"].to_numpy(float) / denominator
            weight = int(counts[str(layer)])
            weighted_reference += weight * layer_reference
            weighted_challenger += weight * layer_challenger
            total_weight += weight
        reference_fold_mse.append(weighted_reference / total_weight)
        challenger_fold_mse.append(weighted_challenger / total_weight)
        cluster_counts[str(fold)] = len(days)
    reference_metric = np.sqrt(np.mean(np.stack(reference_fold_mse), axis=0))
    challenger_metric = np.sqrt(np.mean(np.stack(challenger_fold_mse), axis=0))
    delta = challenger_metric - reference_metric
    if len(delta) != int(replicates) or not np.isfinite(delta).all():
        raise ValueError("paired KST-day bootstrap produced invalid deltas")
    interval = np.quantile(delta, [0.05, 0.95], method="linear")
    return [float(interval[0]), float(interval[1])], {
        "replicates": int(replicates),
        "cluster": "KST_day",
        "fold_cluster_counts": cluster_counts,
        "paired_reference_and_challenger": True,
        "delta_definition": "challenger_minus_reference_rmse_c",
    }


def _blind_predictions(
    *,
    workspace: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    numerical: SimpleNamespace,
    progress: Progress | None,
) -> tuple[dict[float, Any], list[dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    """Generate all 45 outer predictions without loading validation truth."""

    np = numerical.np
    pd = numerical.pd
    engine = numerical.engine
    protocol = config["curve_protocol"]
    model = config["hypothesis"]["model"]
    seeds = [int(value) for value in protocol["seed_ids"]]
    device = numerical.torch.device("cuda")
    cell_receipts: list[dict[str, Any]] = []
    fold_inputs: dict[str, tuple[Any, Any]] = {}
    fold_audits: dict[str, dict[str, Any]] = {}
    input_names: set[str] = set()
    observations_path = (data_dir / "observations.csv").resolve(strict=True)
    if observations_path.parent != data_dir:
        raise PermissionError("blind observations path escaped the pinned data directory")
    for fold in protocol["outer_folds"]:
        fold_name = str(fold["name"])
        outer_start = pd.Timestamp(fold["start_kst"]).tz_convert("UTC")
        observations, audit = _load_fold_blind_observations(
            observations_path,
            outer_start=outer_start,
            embargo_days=int(protocol["embargo_days"]),
            pd_module=pd,
            np_module=np,
        )
        panel = engine._joint_masked_panel(observations)
        endpoints = numerical.projection.public_endpoint_frame(observations)
        fold_inputs[fold_name] = (panel, endpoints)
        fold_audits[fold_name] = audit
        input_names.update(panel.input_names)

    blind_by_fraction: dict[float, Any] = {}

    for fraction in protocol["prefix_fractions"]:
        fraction = float(fraction)
        frame = _load_reference_fraction_blind(
            workspace=workspace,
            config=config,
            fraction=fraction,
            numerical=numerical,
        )
        if "truth" in frame.columns or len(frame) != 78156:
            raise AssertionError("blind reference keys or truth boundary changed")
        challenger_columns: list[str] = []
        for seed in seeds:
            column = f"challenger_seed_{seed}"
            challenger_columns.append(column)
            frame[column] = np.nan
            for fold in protocol["outer_folds"]:
                fold_name = str(fold["name"])
                selected = frame["fold"].eq(fold_name)
                reference_fold_frame = frame.loc[selected].reset_index(drop=True)
                outer_start = pd.Timestamp(fold["start_kst"]).tz_convert("UTC")
                panel, endpoints = fold_inputs[fold_name]
                prefix = engine._prefix_times(
                    panel,
                    outer_start=outer_start,
                    embargo_days=int(protocol["embargo_days"]),
                    fraction=fraction,
                )
                if not len(prefix) or prefix.max() >= outer_start - pd.Timedelta(
                    days=int(protocol["embargo_days"])
                ):
                    raise ValueError("fold-prefix violates the preregistered embargo")
                derived_seed = _derived_seed(
                    seed,
                    fold_name,
                    _fraction_token(fraction),
                    "conditional_analog_rank2",
                )
                prediction, diagnostics = _challenger_fold_seed(
                    reference_fold=reference_fold_frame,
                    reference_column=f"seed_{seed}",
                    panel=panel,
                    endpoints=endpoints,
                    prefix=prefix,
                    model=model,
                    derived_seed=derived_seed,
                    numerical=numerical,
                    device=device,
                )
                frame.loc[selected, column] = prediction
                cell_receipts.append(
                    {
                        "fold": fold_name,
                        "fraction": fraction,
                        "pipeline_seed": seed,
                        "derived_model_seed": derived_seed,
                        "prefix_timestamp_count": int(len(prefix)),
                        "prefix_last_time_utc": prefix.max().isoformat(),
                        "outer_start_utc": outer_start.isoformat(),
                        "guards": {
                            "prefix_before_outer_embargo": True,
                            "joint_temp_psal_complete_profile_training": True,
                            "neighbor_selection_uses_public_covariates_only": True,
                            "validation_truth_values_loaded_or_converted": False,
                            "outer_target_labels_used_for_fit": False,
                            "future_target_labels_used_for_fit": False,
                            "stage_a_seed_anchor_is_not_a_training_label": True,
                            "no_scalar_blend_or_alpha_search": True,
                            "catboost_used": False,
                        },
                        "diagnostics": diagnostics,
                    }
                )
                _emit(
                    progress,
                    event="stage_b_fold_prefix_seed_complete",
                    fold=fold_name,
                    fraction=fraction,
                    pipeline_seed=seed,
                )
        if not np.isfinite(frame[challenger_columns].to_numpy(float)).all():
            raise ValueError("one or more Stage-B fold predictions are missing")
        frame["challenger_mean"] = frame[challenger_columns].to_numpy(float).mean(axis=1)
        blind_by_fraction[fraction] = frame

    if len(cell_receipts) != 45 or set(blind_by_fraction) != {
        0.4,
        0.55,
        0.7,
        0.85,
        1.0,
    }:
        raise AssertionError("blind Stage-B prediction plan did not complete all 45 cells")
    if any(
        audit["validation_target_temp_psal_strings_converted"] != 0
        or audit["validation_truth_columns_read_by_challenger"] != 0
        for audit in fold_audits.values()
    ):
        raise AssertionError("validation truth was accessed during blind prediction")
    return blind_by_fraction, cell_receipts, fold_audits, input_names


def _prediction_commitment_payload(
    *,
    workspace: Path,
    config: Mapping[str, Any],
    blind_by_fraction: Mapping[float, Any],
    cell_receipts: Sequence[Mapping[str, Any]],
    numerical: SimpleNamespace,
) -> dict[str, Any]:
    np = numerical.np
    seeds = [int(value) for value in config["curve_protocol"]["seed_ids"]]
    prediction_columns = [*(f"challenger_seed_{seed}" for seed in seeds), "challenger_mean"]
    expected_columns = [
        *KEYS,
        *(f"seed_{seed}" for seed in seeds),
        "prediction_mean",
        *prediction_columns,
    ]
    expected_cells = {
        (float(fraction), str(fold["name"]), seed)
        for fraction in config["curve_protocol"]["prefix_fractions"]
        for fold in config["curve_protocol"]["outer_folds"]
        for seed in seeds
    }
    observed_cells = {
        (float(cell["fraction"]), str(cell["fold"]), int(cell["pipeline_seed"]))
        for cell in cell_receipts
    }
    if len(cell_receipts) != 45 or observed_cells != expected_cells:
        raise ValueError("blind commitment does not contain each registered cell exactly once")
    key_hasher = hashlib.sha256()
    prediction_hasher = hashlib.sha256()
    rows_by_fraction: dict[str, int] = {}
    for registered in config["curve_protocol"]["prefix_fractions"]:
        fraction = float(registered)
        frame = blind_by_fraction[fraction]
        if list(frame.columns) != expected_columns:
            raise AssertionError("blind commitment frame columns changed or contain truth")
        if frame.duplicated(list(KEYS)).any():
            raise ValueError("blind commitment keys are duplicated")
        recalculated_mean = (
            frame[[f"challenger_seed_{seed}" for seed in seeds]].to_numpy(float).mean(axis=1)
        )
        if not np.array_equal(
            recalculated_mean,
            frame["challenger_mean"].to_numpy(float),
        ):
            raise ValueError("blind challenger mean is not the exact three-seed mean")
        token = _fraction_token(fraction)
        key_frame = frame.loc[:, list(KEYS)].copy()
        key_frame.insert(0, "fraction", fraction)
        key_frame["time"] = key_frame["time"].map(lambda value: value.isoformat())
        buffer = io.StringIO(newline="")
        key_frame.to_csv(buffer, index=False, header=False, lineterminator="\n")
        key_hasher.update(token.encode("ascii") + b"\0")
        key_hasher.update(buffer.getvalue().encode("utf-8"))
        values = frame.loc[:, prediction_columns].to_numpy(dtype="<f8")
        if not np.isfinite(values).all():
            raise ValueError("blind prediction commitment contains non-finite values")
        prediction_hasher.update(token.encode("ascii") + b"\0")
        prediction_hasher.update(values.tobytes(order="C"))
        rows_by_fraction[str(fraction)] = int(len(frame))
    cell_projection = [
        {
            "fraction": float(cell["fraction"]),
            "fold": str(cell["fold"]),
            "pipeline_seed": int(cell["pipeline_seed"]),
            "derived_model_seed": int(cell["derived_model_seed"]),
            "outer_rows": int(cell["diagnostics"]["outer_rows"]),
            "prediction_sha256": str(cell["diagnostics"]["prediction_sha256"]),
        }
        for cell in cell_receipts
    ]
    cell_sha = hashlib.sha256(canonical_json_bytes({"cells": cell_projection})).hexdigest()
    combined = hashlib.sha256(
        bytes.fromhex(key_hasher.hexdigest())
        + bytes.fromhex(prediction_hasher.hexdigest())
        + bytes.fromhex(cell_sha)
    ).hexdigest()
    return {
        "schema_version": "p2_architecture_matched_stage_b.prediction_commitment.v3",
        "stage": "ALL_45_BLIND_PREDICTIONS_BEFORE_VALIDATION_TRUTH",
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation_pins": implementation_pins(workspace),
        "stage_a_seal": config["stage_a_reference"]["artifacts"]["SEAL"],
        "aggregate_only": True,
        "row_level_predictions_persisted": False,
        "truth_columns_present": False,
        "validation_truth_read_or_merged_before_commitment": False,
        "prefix_fractions_in_order": [
            float(value) for value in config["curve_protocol"]["prefix_fractions"]
        ],
        "key_columns_in_order": ["fraction", *KEYS],
        "prediction_columns_in_order": prediction_columns,
        "rows_by_fraction": rows_by_fraction,
        "total_rows": int(sum(rows_by_fraction.values())),
        "cell_prediction_count": int(len(cell_projection)),
        "key_order_sha256": key_hasher.hexdigest(),
        "prediction_values_sha256": prediction_hasher.hexdigest(),
        "cell_prediction_sha256": cell_sha,
        "combined_prediction_commitment_sha256": combined,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }


def _write_prediction_commitment(
    *,
    workspace: Path,
    output: Path,
    config: Mapping[str, Any],
    blind_by_fraction: Mapping[float, Any],
    cell_receipts: Sequence[Mapping[str, Any]],
    numerical: SimpleNamespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _prediction_commitment_payload(
        workspace=workspace,
        config=config,
        blind_by_fraction=blind_by_fraction,
        cell_receipts=cell_receipts,
        numerical=numerical,
    )
    relative = config["output_contract"]["artifacts"]["prediction_commitment"]
    path = contained_path(output, relative)
    exclusive_json(path, payload)
    reloaded = strict_json_object(path)
    if reloaded != payload:
        raise RuntimeError("O_EXCL prediction commitment failed deep-equality reload")
    return reloaded, _pin(path, output)


def _reverify_prediction_commitment(
    *,
    workspace: Path,
    output: Path,
    config: Mapping[str, Any],
    blind_by_fraction: Mapping[float, Any],
    cell_receipts: Sequence[Mapping[str, Any]],
    numerical: SimpleNamespace,
    expected_pin: Mapping[str, Any],
) -> dict[str, Any]:
    path = contained_path(
        output,
        config["output_contract"]["artifacts"]["prediction_commitment"],
    )
    if _pin(path, output) != dict(expected_pin):
        raise RuntimeError("prediction commitment bytes changed before truth attachment")
    observed = strict_json_object(path)
    recomputed = _prediction_commitment_payload(
        workspace=workspace,
        config=config,
        blind_by_fraction=blind_by_fraction,
        cell_receipts=cell_receipts,
        numerical=numerical,
    )
    if observed != recomputed:
        raise RuntimeError("blind predictions differ from the persisted aggregate commitment")
    return observed


def _evaluate_curve_after_commitment(
    *,
    workspace: Path,
    config: Mapping[str, Any],
    numerical: SimpleNamespace,
    blind_by_fraction: Mapping[float, Any],
    truth: Any,
    cell_receipts: Sequence[Mapping[str, Any]],
    fold_blind_audits: Mapping[str, Mapping[str, Any]],
    truth_access_audit: Mapping[str, Any],
    input_names: set[str],
    commitment_pin: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach validation truth only after the blind commitment and evaluate."""

    np = numerical.np
    pd = numerical.pd
    engine = numerical.engine
    protocol = config["curve_protocol"]
    counts = config["metric_and_gates"]["official_layer_counts"]
    sealed_metrics_pin = config["stage_a_reference"]["artifacts"]["CURVE_METRICS"]
    sealed_metrics = strict_json_object(workspace / sealed_metrics_pin["path"])
    seeds = [int(value) for value in protocol["seed_ids"]]
    points: list[dict[str, Any]] = []
    full_reference_fold: dict[str, float] = {}
    full_reference_layer: dict[str, float] = {}
    full_challenger_fold: dict[str, float] = {}
    full_challenger_layer: dict[str, float] = {}

    for registered in protocol["prefix_fractions"]:
        fraction = float(registered)
        frame = _attach_validation_truth(
            blind_by_fraction[fraction],
            truth,
            np_module=np,
        )
        reference_metric, reference_seed_metrics, reference_fold, reference_layer = (
            _assert_reference_metrics(
                frame,
                fraction=fraction,
                config=config,
                sealed_metrics=sealed_metrics,
                numerical=numerical,
            )
        )
        challenger_columns = [f"challenger_seed_{seed}" for seed in seeds]
        challenger_seed_metrics = [
            float(engine._curve_metric(frame, column, counts)[0]) for column in challenger_columns
        ]
        challenger_metric, challenger_fold, challenger_layer = engine._curve_metric(
            frame,
            "challenger_mean",
            counts,
        )
        ci90, bootstrap_receipt = _paired_kst_day_bootstrap(
            frame,
            reference_column="prediction_mean",
            challenger_column="challenger_mean",
            counts=counts,
            replicates=int(protocol["bootstrap_replicates"]),
            seed=_derived_seed(
                int(protocol["bootstrap_seed"]),
                _fraction_token(fraction),
                "paired_kst_day_bootstrap",
            ),
            pd_module=pd,
            np_module=np,
        )
        points.append(
            {
                "fraction": fraction,
                "incumbent": float(reference_metric),
                "challenger": float(challenger_metric),
                "delta_ci90": ci90,
                "incumbent_seed_metrics": reference_seed_metrics,
                "challenger_seed_metrics": challenger_seed_metrics,
                "bootstrap": bootstrap_receipt,
            }
        )
        if fraction == 1.0:
            full_reference_fold = reference_fold
            full_reference_layer = reference_layer
            full_challenger_fold = challenger_fold
            full_challenger_layer = challenger_layer

    if not full_reference_fold:
        raise AssertionError("100% Stage-B point was not evaluated")
    fold_order = [str(fold["name"]) for fold in protocol["outer_folds"]]
    fold_deltas = [
        float(full_challenger_fold[name] - full_reference_fold[name]) for name in fold_order
    ]
    slice_deltas = {
        **{
            f"layer_{layer}": float(
                full_challenger_layer[str(layer)] - full_reference_layer[str(layer)]
            )
            for layer in TARGET_LAYERS
        },
        "2024_sep_oct": float(
            full_challenger_fold["outer_2024_sep_oct"] - full_reference_fold["outer_2024_sep_oct"]
        ),
    }
    stage_a = config["stage_a_reference"]["artifacts"]
    reference_binding = {
        "stage_a_config": stage_a["CONFIG"],
        "deployed_graph_manifest": stage_a["ARCHITECTURE_MANIFEST"],
        "training_recipe": stage_a["TRAINING_RECIPE"],
        "reference_oof_100": stage_a["OOF_100"],
        "reference_seal": stage_a["SEAL"],
    }
    forbidden_inputs = {
        *(f"temp_{layer}" for layer in TARGET_LAYERS),
        *(f"psal_{layer}" for layer in TARGET_LAYERS),
        *(f"public_temp_{layer}" for layer in TARGET_LAYERS),
        *(f"public_psal_{layer}" for layer in TARGET_LAYERS),
    }
    evidence = {
        "schema_version": "p2_architecture_matched_stage_b.learning_curve_evidence.v3",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "baseline_identity": {
            "comparison_mode": MODE,
            "explicitly_not_exact_official_incumbent": True,
            "training_recipe_origin": "NEW_PREREGISTERED_TIME_SAFE_RECIPE",
            "immutable_csv_used_only_for_official_paired_ab": True,
        },
        "reference_binding": reference_binding,
        "prediction_commitment": dict(commitment_pin),
        "validation_truth_access_audit": dict(truth_access_audit),
        "preregistration": {
            "generation_id": config["preregistration"]["generation_id"],
            "config_path": CONFIG_RELATIVE,
            "config_sha256": CONFIG_SHA256,
            "created_before_first_fit": True,
            "hypothesis_count": 1,
            "score_derived_tuning": False,
        },
        "curve_protocol": {
            "comparison_mode": MODE,
            "prefix_fractions": [float(value) for value in protocol["prefix_fractions"]],
            "seed_ids": seeds,
            "seed_aggregation": protocol["seed_aggregation"],
            "bootstrap_replicates": int(protocol["bootstrap_replicates"]),
            "bootstrap_cluster": protocol["bootstrap_cluster"],
            "incumbent_fresh_refit_each_prefix": True,
            "architecture_matched_reference_fresh_refit_each_prefix": True,
            "challenger_fresh_refit_each_prefix": True,
            "same_fold_keys_metric_postprocess": True,
            "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
            "deployed_inference_graph_sha_pinned": True,
            "nested_chronological_component_oof": True,
            "prefix_local_epoch_selection": True,
            "three_complete_pipeline_seeds": True,
            "reference_100_percent_oof_sealed_before_challenger_scoring": True,
            "all_45_blind_predictions_committed_before_validation_truth_read_or_merge": True,
        },
        "points": points,
        "fold_deltas_candidate_minus_incumbent": fold_deltas,
        "fold_delta_order": fold_order,
        "slice_deltas_candidate_minus_incumbent": slice_deltas,
        "leakage_checks": {
            "stage_a_all_prefix_oof_sealed_before_challenger_scoring": True,
            "joint_temp_psal_mask_applied_before_every_label_use": True,
            "target_layer_values_absent_from_public_features": not bool(
                input_names.intersection(forbidden_inputs)
            ),
            "neighbor_selection_uses_public_covariates_only": True,
            "outer_and_future_target_labels_excluded_from_fit": True,
            "prefix_embargo_enforced_for_all_45_fits": len(cell_receipts) == 45,
            "validation_truth_unread_during_all_45_predictions": all(
                audit["validation_truth_columns_read_by_challenger"] == 0
                and audit["validation_target_temp_psal_strings_converted"] == 0
                and audit["withheld_target_scalar_fields_decoded_or_converted"] == 0
                for audit in fold_blind_audits.values()
            ),
            "precommitment_raw_bytes_used_only_for_integrity_and_key_routing": all(
                audit["raw_source_bytes_preflight_hashed_for_integrity_only"] is True
                for audit in fold_blind_audits.values()
            ),
            "withheld_current_fold_target_scalars_never_decoded_or_converted": all(
                audit["withheld_target_scalar_fields_decoded_or_converted"] == 0
                for audit in fold_blind_audits.values()
            ),
            "aggregate_prediction_commitment_written_before_truth_attachment": True,
            "hidden_test_target_values_not_used": truth_access_audit[
                "hidden_test_target_scalars_converted"
            ]
            == 0,
        },
        "reproducibility_checks": {
            "canonical_preregistration_sha_verified": True,
            "stage_a_v3_reference_exact_pins_verified": True,
            "three_fixed_distinct_seed_ids": len(seeds) == len(set(seeds)) == 3,
            "seeded_projection_rule_fixed_before_fit": True,
            "same_fold_keys_metric_and_postprocess": True,
            "sealed_reference_metrics_byte_recomputed": True,
            "paired_kst_day_bootstrap_exactly_5000": int(protocol["bootstrap_replicates"]) == 5000,
            "cuda_deterministic_algorithms_enabled": bool(
                numerical.torch.are_deterministic_algorithms_enabled()
            ),
            "no_hyperparameter_search": config["hypothesis"]["hyperparameter_searches"] == 0,
            "aggregate_only_no_row_predictions_written": True,
            "prediction_commitment_reloaded_and_deep_equal_before_truth_attachment": True,
            "prediction_commitment_reverified_after_truth_load_before_merge": True,
        },
        "output_firewall": {
            "aggregate_only": True,
            "full_fit_performed": False,
            "candidate_generated": False,
            "test_prediction_generated": False,
            "upload_performed": False,
            "official_submission_count": 0,
        },
    }
    if not all(evidence["leakage_checks"].values()) or not all(
        evidence["reproducibility_checks"].values()
    ):
        raise PermissionError("Stage-B leakage or reproducibility guard failed")
    return evidence


def execute_stage_b(
    *,
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
    attempt_lock: Path,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Execute and seal the aggregate-only challenger learning curve."""

    del preflight
    workspace = root.resolve(strict=True)
    resolved_data_dir = data_dir.resolve(strict=True)

    # This is intentionally the first execution action.  It deep-verifies the
    # sealed Stage-A v3 reference, config, source bytes, runtime probe, and data
    # schema/keys before any numerical dependency is imported or fit begins.
    fresh_preflight = static_preflight(
        workspace,
        resolved_data_dir,
        supplied_config=config,
    )
    if fresh_preflight.get("status") != "PASS_STATIC_IMPLEMENTATION_ONLY_STAGE_A_SEALED":
        raise PermissionError("fresh canonical Stage-B preflight did not pass")
    canonical = load_canonical_config(workspace, supplied_config=config)
    expected_implementation_pins = fresh_preflight["implementation_pins"]
    if expected_implementation_pins != implementation_pins(workspace):
        raise PermissionError("Stage-B implementation changed after fresh preflight")
    verified_reference = verify_stage_a_reference(workspace, canonical)

    numerical = _load_numerical_stack(workspace, canonical)
    numerical.engine.set_deterministic_seed(canonical["curve_protocol"]["seed_ids"][0])
    numerical.torch.use_deterministic_algorithms(True)
    numerical.torch.backends.cudnn.benchmark = False
    numerical.torch.backends.cudnn.deterministic = True
    numerical.torch.backends.cuda.matmul.allow_tf32 = False
    runtime = numerical.engine._verify_runtime(canonical)
    if runtime != fresh_preflight["runtime_probe"]["runtime"]:
        raise RuntimeError("isolated and Stage-B execution runtime reports differ")
    data_pins = numerical.engine._verify_data_pins(resolved_data_dir, canonical)

    paths = stage_paths(workspace, canonical)
    received_lock = attempt_lock.resolve(strict=True)
    canonical_lock = paths["attempt_lock"].resolve(strict=True)
    if received_lock != canonical_lock:
        raise PermissionError("engine did not receive the canonical consumed Stage-B lock")
    _qa, qa_sha256 = verify_pre_execution_qa(workspace, canonical)
    _authorization, authorization_sha256 = verify_execution_authorization(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        require_unconsumed=False,
    )
    verify_consumed_attempt_lock(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    if paths["output"].exists():
        raise FileExistsError("append-only Stage-B output already exists")

    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output)
    started = _now_kst()
    blind_by_fraction, cells, fold_blind_audits, input_names = _blind_predictions(
        workspace=workspace,
        data_dir=resolved_data_dir,
        config=canonical,
        numerical=numerical,
        progress=progress,
    )
    commitment, commitment_pin = _write_prediction_commitment(
        workspace=workspace,
        output=output,
        config=canonical,
        blind_by_fraction=blind_by_fraction,
        cell_receipts=cells,
        numerical=numerical,
    )
    if commitment["cell_prediction_count"] != 45:
        raise AssertionError("aggregate blind commitment did not bind all 45 predictions")

    # This is the first validation-truth access.  Immediately after loading
    # only the three registered outer windows, re-hash the still-blind in-memory
    # predictions and deep-equal them to the O_EXCL commitment before merging.
    truth, truth_access_audit = _load_validation_truth_after_commitment(
        (resolved_data_dir / "observations.csv").resolve(strict=True),
        config=canonical,
        pd_module=numerical.pd,
        np_module=numerical.np,
    )
    _reverify_prediction_commitment(
        workspace=workspace,
        output=output,
        config=canonical,
        blind_by_fraction=blind_by_fraction,
        cell_receipts=cells,
        numerical=numerical,
        expected_pin=commitment_pin,
    )
    evidence = _evaluate_curve_after_commitment(
        workspace=workspace,
        config=canonical,
        numerical=numerical,
        blind_by_fraction=blind_by_fraction,
        truth=truth,
        cell_receipts=cells,
        fold_blind_audits=fold_blind_audits,
        truth_access_audit=truth_access_audit,
        input_names=input_names,
        commitment_pin=commitment_pin,
    )
    central_contract = numerical.central.load_contract(workspace)
    decision = numerical.central.evaluate_learning_curve(central_contract, evidence)
    if (
        decision.get("passed") is not False
        or decision.get("exact_official_incumbent_comparison") is not False
    ):
        raise PermissionError("architecture-matched Stage-B cannot self-promote")
    local_numeric = numerical.curve_gate.numeric_curve_gate(
        evidence["points"],
        fold_deltas=evidence["fold_deltas_candidate_minus_incumbent"],
        slice_deltas=evidence["slice_deltas_candidate_minus_incumbent"],
        maximum_slice_regression_c=float(
            canonical["metric_and_gates"]["maximum_each_critical_slice_regression_c"]
        ),
        full_effect_c=float(
            canonical["metric_and_gates"]["full_delta_candidate_minus_reference_at_most_c"]
        ),
    )
    for key, value in local_numeric.items():
        if decision["gates"].get(key) is not value:
            raise AssertionError(f"central and P2 numeric gates differ: {key}")
    local_qualification = bool(decision["local_qualification"])
    gate_decision = {
        **decision,
        "schema_version": "p2_architecture_matched_stage_b.gate_decision.v3",
        "status": (
            "LOCAL_QUALIFIED_PENDING_SEPARATE_APPROVAL"
            if local_qualification
            else "RESEARCH_ONLY_NO_LOCAL_QUALIFICATION"
        ),
        "local_numeric_gate_crosscheck": local_numeric,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "upload_performed": False,
    }
    plan = build_execution_plan(canonical)
    receipt = {
        "schema_version": "p2_architecture_matched_stage_b.training_receipt.v3",
        "started_at_kst": started,
        "completed_at_kst": _now_kst(),
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "hypothesis": canonical["hypothesis"],
        "plan": plan,
        "runtime": runtime,
        "stage_a_reference": verified_reference,
        "prediction_commitment": {
            **commitment_pin,
            "combined_prediction_commitment_sha256": commitment[
                "combined_prediction_commitment_sha256"
            ],
            "persisted_before_validation_truth_access": True,
            "reverified_after_truth_load_before_merge": True,
        },
        "fold_blind_input_audits": fold_blind_audits,
        "validation_truth_access_audit": truth_access_audit,
        "cells": cells,
        "guard_summary": {
            "fresh_canonical_preflight_rerun_by_engine": True,
            "caller_preflight_trusted": False,
            "stage_a_v3_seal_verified_before_numerical_import_and_fit": True,
            "all_source_runtime_data_pins_before_output_and_fit": True,
            "joint_temp_psal_mask_applied_before_all_label_use": True,
            "all_45_predictions_before_validation_truth_access": True,
            "withheld_current_fold_target_scalars_never_decoded_or_converted": all(
                audit["withheld_target_scalar_fields_decoded_or_converted"] == 0
                for audit in fold_blind_audits.values()
            ),
            "raw_source_bytes_before_commitment_limited_to_integrity_and_key_routing": True,
            "aggregate_prediction_commitment_o_excl": True,
            "row_level_predictions_persisted": False,
            "commitment_reverified_after_truth_load_before_merge": True,
            "challenger_fit_count": len(cells),
            "challenger_outer_prediction_count": len(cells),
            "full_fit_count": 0,
            "candidate_count": 0,
            "test_prediction_count": 0,
            "upload_count": 0,
            "aggregate_only": True,
        },
    }
    artifacts = canonical["output_contract"]["artifacts"]
    commitment_path = contained_path(output, artifacts["prediction_commitment"])
    evidence_path = contained_path(output, artifacts["learning_curve_evidence"])
    decision_path = contained_path(output, artifacts["gate_decision"])
    receipt_path = contained_path(output, artifacts["training_receipt"])
    exclusive_json(evidence_path, evidence)
    exclusive_json(decision_path, gate_decision)
    exclusive_json(receipt_path, receipt)

    # Seal only after revalidating every long-run dependency and control byte.
    end_preflight = static_preflight(workspace, resolved_data_dir)
    if end_preflight["implementation_pins"] != expected_implementation_pins:
        raise PermissionError("Stage-B implementation changed during execution")
    if end_preflight["source_pins"] != fresh_preflight["source_pins"]:
        raise PermissionError("Stage-B source pin changed during execution")
    if end_preflight["runtime_probe"]["runtime"] != runtime:
        raise PermissionError("Stage-B runtime changed during execution")
    if numerical.engine._verify_runtime(canonical) != runtime:
        raise PermissionError("Stage-B in-process runtime changed during execution")
    if numerical.engine._verify_data_pins(resolved_data_dir, canonical) != data_pins:
        raise PermissionError("Stage-B data bytes changed during execution")
    verify_stage_a_reference(workspace, canonical)
    _qa_end, qa_sha256_end = verify_pre_execution_qa(workspace, canonical)
    if qa_sha256_end != qa_sha256:
        raise PermissionError("Stage-B QA receipt changed during execution")
    _authorization_end, authorization_sha256_end = verify_execution_authorization(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        require_unconsumed=False,
        require_output_absent=False,
    )
    if authorization_sha256_end != authorization_sha256:
        raise PermissionError("Stage-B authorization changed during execution")
    verify_consumed_attempt_lock(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )

    artifact_pins = {
        "prediction_commitment": _pin(commitment_path, output),
        "learning_curve_evidence": _pin(evidence_path, output),
        "gate_decision": _pin(decision_path, output),
        "training_receipt": _pin(receipt_path, output),
    }
    manifest = {
        "schema_version": "p2_architecture_matched_stage_b.manifest.v3",
        "created_at_kst": _now_kst(),
        "append_only": True,
        "aggregate_only": True,
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation_pins": expected_implementation_pins,
        "source_pins": end_preflight["source_pins"],
        "stage_a_reference_pins": canonical["stage_a_reference"]["artifacts"],
        "prediction_commitment": artifact_pins["prediction_commitment"],
        "runtime": runtime,
        "data_source_pins": data_pins,
        "artifacts": artifact_pins,
        "local_qualification": local_qualification,
        "official_promotion_allowed": False,
        "full_fit_performed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }
    manifest_path = contained_path(output, artifacts["manifest"])
    exclusive_json(manifest_path, manifest)
    seal = {
        "schema_version": "p2_architecture_matched_stage_b.seal.v3",
        "complete": True,
        "status": gate_decision["status"],
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "local_qualification": local_qualification,
        "official_promotion_allowed": False,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "stage_a_seal": canonical["stage_a_reference"]["artifacts"]["SEAL"],
        "prediction_commitment": artifact_pins["prediction_commitment"],
        "manifest": _pin(manifest_path, output),
        "candidate_generated": False,
        "test_prediction_generated": False,
        "upload_count": 0,
    }
    seal_path = contained_path(output, artifacts["seal"])
    exclusive_json(seal_path, seal)
    return {
        "schema_version": "p2_architecture_matched_stage_b_execution.result.v3",
        "status": gate_decision["status"],
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "local_qualification": local_qualification,
        "official_promotion_allowed": False,
        "output": output.relative_to(workspace).as_posix(),
        "prediction_commitment_sha256": sha256_file(commitment_path),
        "evidence_sha256": sha256_file(evidence_path),
        "decision_sha256": sha256_file(decision_path),
        "receipt_sha256": sha256_file(receipt_path),
        "manifest_sha256": sha256_file(manifest_path),
        "seal_sha256": sha256_file(seal_path),
        "challenger_fits": len(cells),
        "full_fit_performed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }


__all__ = [
    "_fit_predict_rank2",
    "_paired_kst_day_bootstrap",
    "_weighted_rank2_projection",
    "build_execution_plan",
    "execute_stage_b",
]
