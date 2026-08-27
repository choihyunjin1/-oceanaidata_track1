"""Append-only P1 MS-TCN checkpoint diagnostic.

The Q2-only plateau recipe is sealed before either Q3 or Q4 holdout target is
opened.  Fresh Q3/Q4 fits preserve a small blind prediction curve.  Epoch 150
is scored and its scientific decision is sealed first; only then may the other
registered epochs be scored in an immutable same-truth oracle section that
cannot mutate the recipe or count as promotion evidence.  The already exposed
historical folds remain retrospective diagnostics.  This runner does not create
a deployable prediction file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_mstcn_checkpoint_diagnostic_20260827_v1"
SOURCE_EXPERIMENT_ID = "p1_incumbent_preserving_mstcn_asrf_v2"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
EXPECTED_CONFIG_SHA256 = "d19ce4c44c46796eefabc22e926a4da075ba2ac5e88522622100b6558ca54af1"
SOURCE_RUNNER_PATH = ROOT / "scripts" / f"run_{SOURCE_EXPERIMENT_ID}.py"
KEY_COLUMNS = ("station", "year", "layer", "time")


class ContractError(RuntimeError):
    """Raised when an append-only, truth-firewall, or fixed-recipe contract fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_npz(path: Path, **arrays: Any) -> str:
    import numpy as np

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _atomic_torch_save(path: Path, value: Any, torch: Any) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _file_identity(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    shown = (
        resolved.relative_to(root.resolve()).as_posix()
        if root is not None and resolved.is_relative_to(root.resolve())
        else resolved.name
    )
    return {
        "path": shown,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _array_inventory(arrays: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


def _canonical_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if _sha256(path) != EXPECTED_CONFIG_SHA256:
        raise ContractError("diagnostic config digest changed")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("diagnostic experiment identity changed")
    recipe = config.get("fixed_recipe", {})
    exact_recipe = {
        "width": 512,
        "epoch": 150,
        "threshold": 0.8,
        "representation": "raw_three_seed_ensemble_mean",
        "seeds": [20260827, 20260839, 20260863],
        "blind_prediction_epochs": [120, 125, 130, 145, 150],
        "saved_state_epochs": [145, 150],
    }
    if recipe != exact_recipe:
        raise ContractError("fixed Q2 plateau recipe changed")
    training = config.get("training_contract", {})
    if not (
        training.get("source_schedule_horizon_epochs") == 300
        and training.get("stop_epoch") == 150
        and training.get("schedule_change_from_source") is False
    ):
        raise ContractError("300-horizon epoch-150 training contract changed")
    evaluation = config.get("evaluation_contract", {})
    if not (
        evaluation.get("scientific_metric_epoch") == 150
        and evaluation.get("truth_scored_epochs") == [150]
        and evaluation.get("same_truth_oracle_diagnostic_epochs") == [120, 125, 130, 145]
        and evaluation.get("same_truth_oracle_promotion_evidence") is False
        and evaluation.get("same_truth_oracle_recipe_mutation_allowed") is False
    ):
        raise ContractError("fixed-only Q3/Q4 scoring contract changed")
    if not all(config.get("prohibitions", {}).values()):
        raise ContractError("a diagnostic prohibition was disabled")
    return config


def _verify_source_pins(config: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, expected in config["source_pins"].items():
        relative = Path(str(expected["path"]))
        path = (root / relative).resolve()
        if not path.is_file() or not path.is_relative_to(root.resolve()):
            raise ContractError(f"source pin path is absent or escapes root: {name}")
        identity = {
            "path": relative.as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        if identity != expected:
            raise ContractError(f"source pin changed: {name}")
        observed[name] = identity
    return observed


def _load_source_runner(*, root: Path = ROOT) -> Any:
    path = root / "scripts" / f"run_{SOURCE_EXPERIMENT_ID}.py"
    module_name = f"{EXPERIMENT_ID}_source"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ContractError("cannot load pinned source runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def check_only(*, root: Path = ROOT) -> dict[str, Any]:
    config_path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config = _canonical_config(config_path)
    pins = _verify_source_pins(config, root=root)
    source = _load_source_runner(root=root)
    source_preflight = source.check_only(root=root)
    source.verify_blind_receipt(
        root / "artifacts" / SOURCE_EXPERIMENT_ID / "q2_qualification_grid_receipt.json"
    )
    return {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__)),
        "source_pins": pins,
        "source_runtime_identity": source_preflight["runtime_identity"],
        "source_immutable_input_count": len(source_preflight["immutable_inputs"]),
        "source_q2_blind_receipt_verified": True,
        "q3_q4_truth_columns_read": 0,
        "official_interface_reads": 0,
        "artifact_namespace_available": not (
            (root / "artifacts" / EXPERIMENT_ID).exists()
            or (root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json").exists()
        ),
        "result": "PASS",
    }


def _binary_metrics(source: Any, truth: Any, prediction: Any) -> dict[str, Any]:
    return source.binary_metrics(truth, prediction)


def _candidate_delta_matrix(
    source: Any,
    truth: Any,
    anchor: Any,
    candidate: Any,
) -> Any:
    import numpy as np

    anchor_score = float(_binary_metrics(source, truth, anchor)["f1"])
    deltas = np.empty(candidate.shape[:2], dtype=np.float64)
    for capacity_index in range(candidate.shape[0]):
        for threshold_index in range(candidate.shape[1]):
            score = _binary_metrics(source, truth, candidate[capacity_index, threshold_index])
            deltas[capacity_index, threshold_index] = float(score["f1"]) - anchor_score
    return deltas


def select_plateau_recipe(
    *,
    widths: Any,
    epochs: Any,
    thresholds: Any,
    deltas: Any,
    neighbor_offset: int,
) -> dict[str, Any]:
    """Select the registered radius-one plateau without any outer-fold target."""

    import numpy as np

    best_key: tuple[float, float, float, int, float, int] | None = None
    best: dict[str, Any] | None = None
    for center_index, (width, epoch) in enumerate(
        zip(widths.tolist(), epochs.tolist(), strict=True)
    ):
        neighbor_indices: list[int] = []
        neighbor_epochs = (
            int(epoch) - neighbor_offset,
            int(epoch),
            int(epoch) + neighbor_offset,
        )
        for neighbor_epoch in neighbor_epochs:
            hit = np.flatnonzero((widths == int(width)) & (epochs == int(neighbor_epoch)))
            if len(hit) != 1:
                neighbor_indices = []
                break
            neighbor_indices.append(int(hit[0]))
        if not neighbor_indices:
            continue
        for threshold_index, threshold in enumerate(thresholds.tolist()):
            neighbor_delta = deltas[neighbor_indices, threshold_index]
            key = (
                float(neighbor_delta.min()),
                float(neighbor_delta.mean()),
                float(deltas[center_index, threshold_index]),
                -int(width),
                float(threshold),
                -int(epoch),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "width": int(width),
                    "epoch": int(epoch),
                    "threshold": float(threshold),
                    "neighbor_epochs": list(neighbor_epochs),
                    "neighbor_delta_f1": [float(value) for value in neighbor_delta.tolist()],
                    "worst_neighbor_delta_f1": float(neighbor_delta.min()),
                    "mean_neighbor_delta_f1": float(neighbor_delta.mean()),
                    "center_delta_f1": float(deltas[center_index, threshold_index]),
                }
    if best is None:
        raise ContractError("Q2 plateau grid has no complete radius-one candidate")
    return best


def _month_text(keys: Any) -> Any:
    import pandas as pd

    return (
        pd.to_datetime(keys["time"], utc=True, format="mixed")
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m")
        .to_numpy()
    )


def _fixed_monthly_metrics(
    source: Any,
    *,
    truth: Any,
    anchor: Any,
    candidate: Any,
    months: Any,
    registered_months: Sequence[str],
) -> dict[str, Any]:
    import numpy as np

    result: dict[str, Any] = {}
    y = np.asarray(truth, dtype=np.int8)
    a = np.asarray(anchor, dtype=np.int8)
    p = np.asarray(candidate, dtype=np.int8)
    for month in registered_months:
        mask = months == month
        if not mask.any():
            raise ContractError(f"registered Q2 month is empty: {month}")
        anchor_score = _binary_metrics(source, y[mask], a[mask])
        candidate_score = _binary_metrics(source, y[mask], p[mask])
        added = (p == 1) & (a == 0) & mask
        added_rows = int(added.sum())
        added_precision = float(y[added].mean()) if added_rows else 1.0
        delta = float(candidate_score["f1"] - anchor_score["f1"])
        result[month] = {
            "rows": int(mask.sum()),
            "anchor_f1": float(anchor_score["f1"]),
            "candidate_f1": float(candidate_score["f1"]),
            "delta_f1": delta,
            "added_rows": added_rows,
            "added_precision": added_precision,
            "required_added_precision_strictly_above": float(anchor_score["f1"] / 2.0),
            "delta_positive": delta > 0.0,
            "added_precision_gate": added_precision > float(anchor_score["f1"] / 2.0),
        }
    return result


def _assert_expected_q2_revalidation(
    config: dict[str, Any], plateau: dict[str, Any], monthly: dict[str, Any]
) -> None:
    expected = config["q2_revalidation"]["expected_plateau_winner"]
    tolerance = float(config["q2_revalidation"]["absolute_tolerance"])
    for field in ("width", "epoch", "threshold", "neighbor_epochs"):
        if plateau[field] != expected[field]:
            raise ContractError(f"Q2 plateau winner changed: {field}")
    if any(
        not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
        for left, right in zip(
            plateau["neighbor_delta_f1"],
            expected["neighbor_delta_f1"],
            strict=True,
        )
    ):
        raise ContractError("Q2 plateau neighbor deltas changed")
    expected_monthly = config["q2_revalidation"]["expected_monthly_metrics"]
    if set(monthly) != set(expected_monthly):
        raise ContractError("Q2 monthly development inventory changed")
    for month, expected_row in expected_monthly.items():
        observed = monthly[month]
        if observed["rows"] != expected_row["rows"]:
            raise ContractError(f"Q2 month row count changed: {month}")
        for field in ("anchor_f1", "candidate_f1", "delta_f1", "added_precision"):
            if not math.isclose(
                float(observed[field]),
                float(expected_row[field]),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ContractError(f"Q2 monthly metric changed: {month}/{field}")
        if not (observed["delta_positive"] and observed["added_precision_gate"]):
            raise ContractError(f"fixed recipe failed Q2 monthly gate: {month}")


def revalidate_q2_recipe(
    source: Any,
    source_config: dict[str, Any],
    surfaces: Any,
    config: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    import numpy as np

    receipt_path = root / "artifacts" / SOURCE_EXPERIMENT_ID / "q2_qualification_grid_receipt.json"
    grid = source.load_sealed_q2_grid(receipt_path)
    truth_frame = source.load_fold_truth_after_receipts(
        source_config,
        surfaces.q2,
        [receipt_path],
        fold="2025_q2",
        root=root,
    )
    truth = truth_frame["label"].to_numpy(dtype=np.int8)
    anchor = np.asarray(surfaces.q2.anchor, dtype=np.int8)
    deltas = _candidate_delta_matrix(source, truth, anchor, grid.candidate)
    plateau = select_plateau_recipe(
        widths=grid.widths,
        epochs=grid.epochs,
        thresholds=grid.thresholds,
        deltas=deltas,
        neighbor_offset=int(config["q2_revalidation"]["plateau_neighbor_offset_epochs"]),
    )
    capacity = np.flatnonzero(
        (grid.widths == int(config["fixed_recipe"]["width"]))
        & (grid.epochs == int(config["fixed_recipe"]["epoch"]))
    )
    threshold = np.flatnonzero(
        np.isclose(
            grid.thresholds,
            float(config["fixed_recipe"]["threshold"]),
            rtol=0.0,
            atol=0.0,
        )
    )
    if len(capacity) != 1 or len(threshold) != 1:
        raise ContractError("fixed Q2 plateau recipe is absent from the sealed grid")
    selected_candidate = grid.candidate[int(capacity[0]), int(threshold[0])]
    monthly = _fixed_monthly_metrics(
        source,
        truth=truth,
        anchor=anchor,
        candidate=selected_candidate,
        months=_month_text(surfaces.q2.keys),
        registered_months=config["q2_revalidation"]["development_windows_kst"],
    )
    _assert_expected_q2_revalidation(config, plateau, monthly)
    return {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.q2_revalidation.v1",
        "role": "development_only_not_promotion_evidence",
        "source_grid_receipt": _file_identity(receipt_path, root=root),
        "plateau_selection": plateau,
        "monthly_development_metrics": monthly,
        "q3_q4_truth_columns_read": 0,
        "result": "PASS",
    }


def _seal_recipe(
    config: dict[str, Any],
    q2_revalidation_path: Path,
    *,
    artifact_dir: Path,
) -> tuple[dict[str, Any], Path]:
    recipe = {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.fixed_recipe.v1",
        **config["fixed_recipe"],
        "source": "sealed Q2 grid radius-one plateau plus three Q2 KST-month gates",
        "source_revalidation": _file_identity(q2_revalidation_path),
        "source_schedule_horizon_epochs": 300,
        "fresh_refit_stop_epoch": 150,
        "sealed_before_q3_q4_training": True,
        "sealed_before_q3_q4_truth_access": True,
        "q3_q4_truth_columns_read": 0,
        "q3_q4_result_driven_changes_authorized": False,
    }
    path = artifact_dir / "selected_recipe.json"
    _exclusive_json(path, recipe)
    return recipe, path


def _fit_seed_checkpoint_curve(
    source: Any,
    training: Any,
    holdout: Any,
    *,
    source_config: dict[str, Any],
    phase: str,
    seed: int,
    device: Any,
    artifact_dir: Path,
    checkpoint_epochs: Sequence[int],
    state_epochs: Sequence[int],
) -> tuple[dict[int, Any], dict[str, Any]]:
    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    capacity = source._config_for_capacity(source_config, width=512, seed=int(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    source._reset_cuda_peak_memory(torch, device)
    model = source._new_model(training.features.shape[1], capacity, device)
    expected_parameters = int(
        source_config["architecture"]["exact_parameter_count_by_width_at_input_165"]["512"]
    )
    if int(model.trainable_parameter_count) != expected_parameters:
        raise ContractError("width-512 parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(capacity["training"]["learning_rate"]),
        weight_decay=float(capacity["training"]["weight_decay"]),
    )
    windows = source._selected_windows(training, capacity)
    holdout_windows = source._all_windows(holdout, capacity)
    positive_weight = source._positive_weight(training.surface.labels)
    _steps, total_steps, _warmup = source._schedule_geometry(capacity, window_count=len(windows))
    if int(capacity["training"]["maximum_epochs"]) != 300:
        raise ContractError("source schedule horizon changed")
    checkpoint_set = set(int(value) for value in checkpoint_epochs)
    state_set = set(int(value) for value in state_epochs)
    if not state_set.issubset(checkpoint_set):
        raise ContractError("saved state epochs must be blind prediction epochs")
    stop_epoch = max(checkpoint_set)
    if stop_epoch != 150:
        raise ContractError("fresh refit stop epoch changed")
    global_step = 0
    history: list[dict[str, Any]] = []
    predictions: dict[int, Any] = {}
    state_artifacts: list[dict[str, Any]] = []
    for epoch in range(1, stop_epoch + 1):
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
        record = source._history_record(
            epoch=epoch,
            telemetry=telemetry,
            global_step=global_step,
            learning_rate=learning_rate,
            elapsed_seconds=time.perf_counter() - started,
        )
        if epoch in checkpoint_set:
            predictions[epoch] = source.predict_encoded(
                model,
                holdout,
                holdout_windows,
                batch_size=int(capacity["training"]["batch_size"]),
                device=device,
            )
            record["blind_checkpoint_captured"] = True
        if epoch in state_set:
            state_path = artifact_dir / f"{phase}_width_512_seed_{seed}_epoch_{epoch}_state.pt"
            state_sha = _atomic_torch_save(
                state_path,
                {
                    "schema_version": "p1.mstcn_checkpoint_diagnostic.state.v1",
                    "experiment_id": EXPERIMENT_ID,
                    "phase": phase,
                    "width": 512,
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "source_schedule_horizon_epochs": 300,
                    "input_features": int(training.features.shape[1]),
                    "parameter_count": int(model.trainable_parameter_count),
                    "state_dict": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                },
                torch,
            )
            state_artifacts.append(
                {
                    "path": state_path.name,
                    "bytes": int(state_path.stat().st_size),
                    "sha256": state_sha,
                    "epoch": int(epoch),
                }
            )
            record["state_saved"] = True
        history.append(record)
    if tuple(sorted(predictions)) != tuple(sorted(checkpoint_set)):
        raise ContractError("a registered blind prediction checkpoint is absent")
    if [row["epoch"] for row in state_artifacts] != sorted(state_set):
        raise ContractError("a registered adjacent/final state is absent")
    history_path = artifact_dir / f"{phase}_width_512_seed_{seed}_training_history.json"
    _exclusive_json(history_path, history)
    receipt = {
        "phase": phase,
        "width": 512,
        "seed": int(seed),
        "fresh_refit": True,
        "epochs_trained": stop_epoch,
        "source_schedule_horizon_epochs": 300,
        "checkpoint_epochs": sorted(checkpoint_set),
        "saved_state_epochs": sorted(state_set),
        "optimizer_steps": int(global_step),
        "batch_size": int(capacity["training"]["batch_size"]),
        "parameter_count": int(model.trainable_parameter_count),
        "positive_weight": float(positive_weight),
        "history_artifact": _file_identity(history_path),
        "state_artifacts": state_artifacts,
        "nonfinite_count_total": int(sum(int(row["nonfinite_count"]) for row in history)),
        "gradient_clip_count_total": int(sum(int(row["gradient_clip_count"]) for row in history)),
        **source._cuda_peak_memory_receipt(torch, device),
    }
    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, receipt


def _fit_and_seal_phase(
    source: Any,
    training: Any,
    holdout: Any,
    *,
    source_config: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    device: Any,
    recipe_path: Path,
    key_sha256: str,
    artifact_dir: Path,
) -> Path:
    import numpy as np

    epochs = tuple(int(value) for value in config["fixed_recipe"]["blind_prediction_epochs"])
    seeds = tuple(int(value) for value in config["fixed_recipe"]["seeds"])
    row_sum = {epoch: np.zeros(holdout.surface.rows, dtype=np.float32) for epoch in epochs}
    boundary_sum = {
        epoch: np.zeros((holdout.surface.rows, 2), dtype=np.float32) for epoch in epochs
    }
    type_sum = {
        epoch: np.zeros((holdout.surface.rows, len(source.TYPE_NAMES)), dtype=np.float32)
        for epoch in epochs
    }
    fit_receipts: list[dict[str, Any]] = []
    for seed in seeds:
        predictions, receipt = _fit_seed_checkpoint_curve(
            source,
            training,
            holdout,
            source_config=source_config,
            phase=phase,
            seed=seed,
            device=device,
            artifact_dir=artifact_dir,
            checkpoint_epochs=epochs,
            state_epochs=config["fixed_recipe"]["saved_state_epochs"],
        )
        fit_receipts.append(receipt)
        for epoch in epochs:
            row_sum[epoch] += predictions[epoch].row_probability.astype(np.float32, copy=False)
            boundary_sum[epoch] += predictions[epoch].boundary_probability.astype(
                np.float32, copy=False
            )
            type_sum[epoch] += predictions[epoch].type_probability.astype(np.float32, copy=False)
        del predictions
        gc.collect()
    row_parts: list[Any] = []
    boundary_parts: list[Any] = []
    type_parts: list[Any] = []
    proposal_parts: list[Any] = []
    candidate_parts: list[Any] = []
    threshold = float(config["fixed_recipe"]["threshold"])
    for epoch in epochs:
        bundle = source.PredictionBundle(
            row_sum[epoch] / float(len(seeds)),
            boundary_sum[epoch] / float(len(seeds)),
            type_sum[epoch] / float(len(seeds)),
        )
        proposal = source.decode_long_event_segments(
            source._decoder_row_probability(bundle, source_config),
            bundle.boundary_probability,
            holdout.layout,
            high_threshold=threshold,
            snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
            minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
            maximum_rows=source._maximum_segment_rows(source_config),
        )
        if holdout.surface.anchor is None:
            raise ContractError(f"{phase} current-Router anchor is absent")
        candidate = source.anchor_preserving_union(holdout.surface.anchor, proposal)
        row_parts.append(bundle.row_probability.astype(np.float32, copy=False))
        boundary_parts.append(bundle.boundary_probability.astype(np.float32, copy=False))
        type_parts.append(bundle.type_probability.astype(np.float32, copy=False))
        proposal_parts.append(np.asarray(proposal, dtype=np.int8))
        candidate_parts.append(np.asarray(candidate, dtype=np.int8))
    arrays = {
        "epochs": np.asarray(epochs, dtype=np.int16),
        "row_probability": np.stack(row_parts, axis=0),
        "boundary_probability": np.stack(boundary_parts, axis=0),
        "type_probability": np.stack(type_parts, axis=0),
        "proposal": np.stack(proposal_parts, axis=0),
        "candidate": np.stack(candidate_parts, axis=0),
    }
    score_path = artifact_dir / f"{phase}_blind_checkpoint_curve.npz"
    score_sha = _atomic_npz(score_path, **arrays)
    receipt = {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.phase_blind.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": phase,
        "fold": source_config["phase_protocols"][phase]["fold"],
        "score_path": score_path.name,
        "score_bytes": int(score_path.stat().st_size),
        "score_sha256": score_sha,
        "config_sha256": _sha256(CONFIG_PATH),
        "recipe_sha256": _sha256(recipe_path),
        "ordered_holdout_key_sha256": key_sha256,
        "holdout_rows": int(holdout.surface.rows),
        "array_inventory": _array_inventory(arrays),
        "checkpoint_epochs": list(epochs),
        "scientific_metric_epoch": 150,
        "same_truth_oracle_diagnostic_epochs": [120, 125, 130, 145],
        "same_truth_oracle_promotion_evidence": False,
        "same_truth_oracle_recipe_mutation_allowed": False,
        "fit_receipts": fit_receipts,
        "same_fold_holdout_truth_columns_opened_before_receipt": 0,
        "prior_fold_metrics_computed_before_both_phase_seals": False,
        "official_interface_reads": 0,
    }
    receipt_path = artifact_dir / f"{phase}_blind_checkpoint_curve_receipt.json"
    _exclusive_json(receipt_path, receipt)
    return receipt_path


def _verify_artifact_identity(parent: Path, identity: dict[str, Any]) -> None:
    path = (parent / str(identity["path"])).resolve()
    if path.parent != parent.resolve() or not path.is_file():
        raise ContractError("fit artifact escapes or is absent from run namespace")
    if int(path.stat().st_size) != int(identity["bytes"]) or _sha256(path) != str(
        identity["sha256"]
    ):
        raise ContractError("fit artifact identity changed")


def _verify_phase_receipt(
    source: Any,
    source_config: dict[str, Any],
    config: dict[str, Any],
    receipt_path: Path,
    holdout: Any,
    recipe_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    phase = str(receipt.get("phase"))
    if phase not in {"q3", "q4"}:
        raise ContractError("phase blind receipt has an unknown phase")
    expected_epochs = config["fixed_recipe"]["blind_prediction_epochs"]
    if not (
        receipt.get("schema_version") == "p1.mstcn_checkpoint_diagnostic.phase_blind.v1"
        and receipt.get("experiment_id") == EXPERIMENT_ID
        and receipt.get("config_sha256") == _sha256(CONFIG_PATH)
        and receipt.get("recipe_sha256") == _sha256(recipe_path)
        and receipt.get("checkpoint_epochs") == expected_epochs
        and receipt.get("scientific_metric_epoch") == 150
        and receipt.get("same_truth_oracle_diagnostic_epochs") == [120, 125, 130, 145]
        and receipt.get("same_truth_oracle_promotion_evidence") is False
        and receipt.get("same_truth_oracle_recipe_mutation_allowed") is False
        and receipt.get("same_fold_holdout_truth_columns_opened_before_receipt") == 0
        and receipt.get("prior_fold_metrics_computed_before_both_phase_seals") is False
    ):
        raise ContractError("phase blind receipt contract changed")
    if receipt.get("ordered_holdout_key_sha256") != source._ordered_key_sha(holdout.surface.keys):
        raise ContractError("phase blind receipt key identity changed")
    score_path = (receipt_path.parent / str(receipt["score_path"])).resolve()
    if score_path.parent != receipt_path.parent.resolve():
        raise ContractError("phase blind score path escapes run namespace")
    if not score_path.is_file() or {
        "bytes": int(score_path.stat().st_size),
        "sha256": _sha256(score_path),
    } != {
        "bytes": int(receipt["score_bytes"]),
        "sha256": str(receipt["score_sha256"]),
    }:
        raise ContractError("phase blind score identity changed")
    with np.load(score_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected_names = {
        "epochs",
        "row_probability",
        "boundary_probability",
        "type_probability",
        "proposal",
        "candidate",
    }
    if set(arrays) != expected_names or _array_inventory(arrays) != receipt.get("array_inventory"):
        raise ContractError("phase blind array inventory changed")
    rows = int(holdout.surface.rows)
    checkpoint_count = len(expected_epochs)
    expected_shapes = {
        "epochs": (checkpoint_count,),
        "row_probability": (checkpoint_count, rows),
        "boundary_probability": (checkpoint_count, rows, 2),
        "type_probability": (checkpoint_count, rows, len(source.TYPE_NAMES)),
        "proposal": (checkpoint_count, rows),
        "candidate": (checkpoint_count, rows),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise ContractError("phase blind array shape changed")
    if arrays["epochs"].tolist() != expected_epochs:
        raise ContractError("phase blind checkpoint order changed")
    if not all(
        np.isfinite(arrays[name]).all() and ((arrays[name] >= 0.0) & (arrays[name] <= 1.0)).all()
        for name in ("row_probability", "boundary_probability", "type_probability")
    ) or not all(np.isin(arrays[name], [0, 1]).all() for name in ("proposal", "candidate")):
        raise ContractError("phase blind arrays are invalid")
    threshold = float(config["fixed_recipe"]["threshold"])
    for index, _epoch in enumerate(expected_epochs):
        bundle = source.PredictionBundle(
            arrays["row_probability"][index],
            arrays["boundary_probability"][index],
            arrays["type_probability"][index],
        )
        proposal = source.decode_long_event_segments(
            source._decoder_row_probability(bundle, source_config),
            bundle.boundary_probability,
            holdout.layout,
            high_threshold=threshold,
            snap_radius=int(source_config["decoder"]["boundary_peak_snap_radius_rows"]),
            minimum_rows=int(source_config["decoder"]["minimum_added_segment_rows"]),
            maximum_rows=source._maximum_segment_rows(source_config),
        )
        candidate = source.anchor_preserving_union(holdout.surface.anchor, proposal)
        if not np.array_equal(proposal, arrays["proposal"][index]) or not np.array_equal(
            candidate, arrays["candidate"][index]
        ):
            raise ContractError("phase blind semantic replay changed")
    if len(receipt.get("fit_receipts", [])) != 3:
        raise ContractError("phase seed fit count changed")
    for fit in receipt["fit_receipts"]:
        _verify_artifact_identity(receipt_path.parent, fit["history_artifact"])
        if [row["epoch"] for row in fit["state_artifacts"]] != [145, 150]:
            raise ContractError("saved state epoch inventory changed")
        for identity in fit["state_artifacts"]:
            _verify_artifact_identity(receipt_path.parent, identity)
    replay = {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.semantic_replay.v1",
        "phase": phase,
        "checkpoint_epochs": expected_epochs,
        "all_checkpoint_decoder_replays": True,
        "all_checkpoint_anchor_union_replays": True,
        "truth_columns_read": 0,
        "result": "PASS",
    }
    return receipt, arrays, replay


def _load_truth_after_both_phase_receipts(
    source: Any,
    source_config: dict[str, Any],
    verified: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    holdouts: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    import pyarrow.dataset as dataset

    if set(verified) != {"q3", "q4"}:
        raise ContractError("both phase receipts must verify before truth access")
    truths: dict[str, Any] = {}
    with source._verified_immutable_read(
        source_config, "frozen_truth_and_folds", root=root
    ) as oof_path:
        source_data = dataset.dataset(oof_path, format="parquet")
        for phase in ("q3", "q4"):
            fold = source_config["phase_protocols"][phase]["fold"]
            receipt = verified[phase][0]
            if receipt["ordered_holdout_key_sha256"] != source._ordered_key_sha(
                holdouts[phase].surface.keys
            ):
                raise ContractError("verified receipt/holdout keys changed before truth")
            scanner = source_data.scanner(
                columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
                filter=dataset.field("fold") == fold,
                use_threads=True,
            )
            truth = scanner.to_table().to_pandas().reset_index(drop=True)
            truth, _membership = source._validate_registered_holdout_membership(
                truth, source_config, fold=fold
            )
            if not source._keys_equal(holdouts[phase].surface.keys, truth):
                raise ContractError("opened historical truth differs from blind keys")
            truths[phase] = truth
    return truths


def _evaluate_fixed_epoch_only(
    source: Any,
    source_config: dict[str, Any],
    truths: dict[str, Any],
    holdouts: dict[str, Any],
    verified: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    import numpy as np

    metric_epoch = 150
    truth_parts: list[Any] = []
    anchor_parts: list[Any] = []
    candidate_parts: list[Any] = []
    station_parts: list[Any] = []
    bootstrap_folds: list[tuple[Any, Any, Any, Any]] = []
    fold_metrics: dict[str, Any] = {}
    for phase in ("q3", "q4"):
        arrays = verified[phase][1]
        hit = np.flatnonzero(arrays["epochs"] == metric_epoch)
        if len(hit) != 1:
            raise ContractError("fixed metric epoch is absent from blind curve")
        candidate = np.asarray(arrays["candidate"][int(hit[0])], dtype=np.int8)
        truth = truths[phase]["label"].to_numpy(dtype=np.int8)
        anchor = np.asarray(holdouts[phase].surface.anchor, dtype=np.int8)
        anchor_score = source.binary_metrics(truth, anchor)
        candidate_score = source.binary_metrics(truth, candidate)
        fold_metrics[phase] = {
            "anchor": anchor_score,
            "candidate": candidate_score,
            "delta_f1": float(candidate_score["f1"] - anchor_score["f1"]),
        }
        truth_parts.append(truth)
        anchor_parts.append(anchor)
        candidate_parts.append(candidate)
        station_parts.append(holdouts[phase].surface.keys["station"].astype(str).to_numpy())
        bootstrap_folds.append((holdouts[phase].surface.keys, truth, anchor, candidate))
    truth = np.concatenate(truth_parts)
    anchor = np.concatenate(anchor_parts)
    candidate = np.concatenate(candidate_parts)
    stations = np.concatenate(station_parts)
    anchor_score = source.binary_metrics(truth, anchor)
    candidate_score = source.binary_metrics(truth, candidate)
    expected = source_config["confirmatory_gate"]["expected_q3_q4_current_router_counts"]
    if any(int(anchor_score[name]) != int(expected[name]) for name in ("tp", "fp", "fn")):
        raise ContractError("fixed diagnostic anchor identity changed")
    added = (candidate == 1) & (anchor == 0)
    removed = int(np.sum((anchor == 1) & (candidate == 0)))
    by_station: dict[str, Any] = {}
    for station in sorted(set(stations.tolist())):
        mask = stations == station
        station_anchor = source.binary_metrics(truth[mask], anchor[mask])
        station_candidate = source.binary_metrics(truth[mask], candidate[mask])
        by_station[station] = {
            "anchor": station_anchor,
            "candidate": station_candidate,
            "delta_f1": float(station_candidate["f1"] - station_anchor["f1"]),
        }
    gate = source_config["confirmatory_gate"]
    bootstrap = source._paired_day_block_bootstrap(
        bootstrap_folds,
        replicates=int(gate["bootstrap_replicates"]),
        block_days=int(gate["bootstrap_block_kst_days"]),
        seed=int(gate["bootstrap_seed"]),
    )
    delta = float(candidate_score["f1"] - anchor_score["f1"])
    return {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.fixed_metrics.v1",
        "role": "retrospective diagnostic on already exposed historical folds; not fresh promotion evidence",
        "scientific_metric_epoch": metric_epoch,
        "truth_scored_epochs": [metric_epoch],
        "same_truth_oracle_epochs_pending": [120, 125, 130, 145],
        "folds": fold_metrics,
        "pooled": {
            "rows": int(len(truth)),
            "anchor": anchor_score,
            "candidate": candidate_score,
            "delta_f1": delta,
            "added_rows": int(added.sum()),
            "added_row_precision": float(truth[added].mean()) if added.any() else 0.0,
            "anchor_positive_removed_rows": removed,
        },
        "by_station": by_station,
        "bootstrap": bootstrap,
        "fixed_recipe_improved_pooled": delta > 0.0,
        "both_fold_deltas_positive": all(row["delta_f1"] > 0.0 for row in fold_metrics.values()),
        "official_probe_authorized": False,
        "three_official_points_claimed": False,
    }


def _evaluate_same_truth_oracle_diagnostic(
    source: Any,
    truths: dict[str, Any],
    holdouts: dict[str, Any],
    verified: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    *,
    fixed_decision_path: Path,
    recipe_path: Path,
    oracle_epochs: Sequence[int],
) -> dict[str, Any]:
    """Score pre-sealed checkpoints only after the fixed e150 decision is immutable."""

    import numpy as np

    if not fixed_decision_path.is_file():
        raise ContractError("fixed epoch-150 decision must be sealed before oracle scoring")
    fixed_decision_identity = _file_identity(fixed_decision_path)
    recipe_sha_before = _sha256(recipe_path)
    rows: list[dict[str, Any]] = []
    for epoch in oracle_epochs:
        truth_parts: list[Any] = []
        anchor_parts: list[Any] = []
        candidate_parts: list[Any] = []
        fold_metrics: dict[str, Any] = {}
        for phase in ("q3", "q4"):
            arrays = verified[phase][1]
            hit = np.flatnonzero(arrays["epochs"] == int(epoch))
            if len(hit) != 1:
                raise ContractError(f"oracle epoch is absent from blind curve: {epoch}")
            truth = truths[phase]["label"].to_numpy(dtype=np.int8)
            anchor = np.asarray(holdouts[phase].surface.anchor, dtype=np.int8)
            candidate = np.asarray(arrays["candidate"][int(hit[0])], dtype=np.int8)
            anchor_score = source.binary_metrics(truth, anchor)
            candidate_score = source.binary_metrics(truth, candidate)
            fold_metrics[phase] = {
                "anchor_f1": float(anchor_score["f1"]),
                "candidate_f1": float(candidate_score["f1"]),
                "delta_f1": float(candidate_score["f1"] - anchor_score["f1"]),
            }
            truth_parts.append(truth)
            anchor_parts.append(anchor)
            candidate_parts.append(candidate)
        truth = np.concatenate(truth_parts)
        anchor = np.concatenate(anchor_parts)
        candidate = np.concatenate(candidate_parts)
        anchor_score = source.binary_metrics(truth, anchor)
        candidate_score = source.binary_metrics(truth, candidate)
        added = (candidate == 1) & (anchor == 0)
        rows.append(
            {
                "epoch": int(epoch),
                "folds": fold_metrics,
                "pooled_anchor_f1": float(anchor_score["f1"]),
                "pooled_candidate_f1": float(candidate_score["f1"]),
                "pooled_delta_f1": float(candidate_score["f1"] - anchor_score["f1"]),
                "added_rows": int(added.sum()),
                "added_precision": (float(truth[added].mean()) if added.any() else 0.0),
                "anchor_positive_removed_rows": int(np.sum((anchor == 1) & (candidate == 0))),
            }
        )
    recipe_sha_after = _sha256(recipe_path)
    if recipe_sha_after != recipe_sha_before:
        raise ContractError("fixed recipe changed during oracle scoring")
    best = max(rows, key=lambda row: (row["pooled_delta_f1"], -row["epoch"]))
    return {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.same_truth_oracle.v1",
        "role": "same-truth checkpoint trajectory oracle diagnostic only",
        "fixed_scientific_decision": fixed_decision_identity,
        "fixed_recipe_sha256_before": recipe_sha_before,
        "fixed_recipe_sha256_after": recipe_sha_after,
        "recipe_mutated": False,
        "scientific_decision_mutated": False,
        "promotion_evidence": False,
        "official_probe_authorized": False,
        "oracle_epochs": [int(value) for value in oracle_epochs],
        "rows": rows,
        "same_truth_oracle_best": {
            "epoch": int(best["epoch"]),
            "pooled_delta_f1": float(best["pooled_delta_f1"]),
            "interpretation": "diagnostic ceiling on already opened truth; not a selectable recipe",
        },
    }


def _acquire_namespace(preflight: dict[str, Any], *, root: Path = ROOT) -> Path:
    artifact_dir = root / "artifacts" / EXPERIMENT_ID
    lock_path = root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    if artifact_dir.exists() or lock_path.exists():
        raise FileExistsError("append-only diagnostic namespace already exists")
    lock = {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.attempt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "config_sha256": preflight["config_sha256"],
        "runner_sha256": preflight["runner_sha256"],
        "one_shot": True,
        "automatic_retry": False,
    }
    payload = _json_bytes(lock)
    _exclusive_json(lock_path, lock)
    try:
        artifact_dir.mkdir(parents=False, exist_ok=False)
    except BaseException:
        if lock_path.is_file() and lock_path.read_bytes() == payload:
            lock_path.unlink()
        raise
    return artifact_dir


def _write_manifest(artifact_dir: Path) -> Path:
    entries = []
    for path in sorted(artifact_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            entries.append(_file_identity(path))
    manifest = {
        "schema_version": "p1.mstcn_checkpoint_diagnostic.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_count_excluding_manifest": len(entries),
        "files": entries,
        "official_interface_reads": 0,
        "submission_created": False,
        "upload_performed": False,
    }
    path = artifact_dir / "manifest.json"
    _exclusive_json(path, manifest)
    return path


def execute(
    *,
    expected_runner_sha256: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    observed_runner_sha = _sha256(Path(__file__))
    if expected_runner_sha256.casefold() != observed_runner_sha:
        raise ContractError("--expected-runner-sha256 must match reviewed runner bytes")
    preflight = check_only(root=root)
    if not preflight["artifact_namespace_available"]:
        raise FileExistsError("append-only diagnostic namespace already exists")
    config = _canonical_config(root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json")
    source = _load_source_runner(root=root)
    source_config = source._canonical_config(
        root / "configs" / "experiments" / f"{SOURCE_EXPERIMENT_ID}.json"
    )
    _np, _pd, torch, _model_api, _data_api = source._load_scientific()
    if not torch.cuda.is_available():
        raise ContractError("fixed bf16 diagnostic requires CUDA")
    device = torch.device("cuda")
    surfaces = source.load_blind_surfaces(source_config, root=root)
    artifact_dir = _acquire_namespace(preflight, root=root)
    started = datetime.now(UTC)
    terminal_path = artifact_dir / "terminal_result.json"
    try:
        _exclusive_json(artifact_dir / "preflight.json", preflight)
        q2_revalidation = revalidate_q2_recipe(source, source_config, surfaces, config, root=root)
        q2_path = artifact_dir / "q2_plateau_revalidation.json"
        _exclusive_json(q2_path, q2_revalidation)
        selected_recipe, recipe_path = _seal_recipe(config, q2_path, artifact_dir=artifact_dir)
        holdouts: dict[str, Any] = {}
        receipt_paths: dict[str, Path] = {}
        for phase in ("q3", "q4"):
            encoder, training, holdout, split = source._prepare_phase_surfaces(
                surfaces, source_config, phase, root=root
            )
            _exclusive_json(artifact_dir / f"{phase}_split.json", split)
            _exclusive_json(
                artifact_dir / f"{phase}_encoder.json", source._encoder_receipt(encoder)
            )
            receipt_paths[phase] = _fit_and_seal_phase(
                source,
                training,
                holdout,
                source_config=source_config,
                config=config,
                phase=phase,
                device=device,
                recipe_path=recipe_path,
                key_sha256=surfaces.membership_sha256[
                    source_config["phase_protocols"][phase]["fold"]
                ],
                artifact_dir=artifact_dir,
            )
            holdouts[phase] = holdout
        verified = {
            phase: _verify_phase_receipt(
                source,
                source_config,
                config,
                receipt_paths[phase],
                holdouts[phase],
                recipe_path,
            )
            for phase in ("q3", "q4")
        }
        _exclusive_json(
            artifact_dir / "blind_semantic_replays.json",
            {phase: verified[phase][2] for phase in ("q3", "q4")},
        )
        truths = _load_truth_after_both_phase_receipts(
            source, source_config, verified, holdouts, root=root
        )
        metrics = _evaluate_fixed_epoch_only(source, source_config, truths, holdouts, verified)
        fixed_metrics_path = artifact_dir / "fixed_epoch_150_metrics.json"
        _exclusive_json(fixed_metrics_path, metrics)
        status = (
            "RETROSPECTIVE_FIXED_E150_IMPROVED"
            if metrics["fixed_recipe_improved_pooled"]
            else "RETROSPECTIVE_FIXED_E150_NOT_IMPROVED"
        )
        fixed_decision_path = artifact_dir / "fixed_epoch_150_decision.json"
        _exclusive_json(
            fixed_decision_path,
            {
                "schema_version": "p1.mstcn_checkpoint_diagnostic.fixed_decision.v1",
                "experiment_id": EXPERIMENT_ID,
                "status": status,
                "scientific_metric_epoch": 150,
                "selected_recipe_sha256": _sha256(recipe_path),
                "fixed_metrics": _file_identity(fixed_metrics_path),
                "same_truth_oracle_computed_before_decision": False,
                "same_truth_oracle_may_mutate_decision": False,
                "fresh_promotion_evidence": False,
                "official_probe_authorized": False,
            },
        )
        oracle = _evaluate_same_truth_oracle_diagnostic(
            source,
            truths,
            holdouts,
            verified,
            fixed_decision_path=fixed_decision_path,
            recipe_path=recipe_path,
            oracle_epochs=config["evaluation_contract"]["same_truth_oracle_diagnostic_epochs"],
        )
        oracle_path = artifact_dir / "same_truth_oracle_diagnostic.json"
        _exclusive_json(oracle_path, oracle)
        terminal = {
            "schema_version": "p1.mstcn_checkpoint_diagnostic.terminal.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "device": torch.cuda.get_device_name(device),
            "selected_recipe": selected_recipe,
            "scientific_metric_epoch": 150,
            "truth_scored_epochs": [150],
            "same_truth_oracle_scored_epochs": [120, 125, 130, 145],
            "same_truth_oracle": _file_identity(oracle_path),
            "same_truth_oracle_mutated_fixed_decision": False,
            "same_truth_oracle_promotion_evidence": False,
            "fresh_promotion_evidence": False,
            "official_probe_authorized": False,
            "submission_created": False,
            "upload_performed": False,
        }
        _exclusive_json(terminal_path, terminal)
        manifest_path = _write_manifest(artifact_dir)
        return {
            **terminal,
            "manifest": _file_identity(manifest_path),
            "fixed_metrics": metrics,
        }
    except BaseException as error:
        if not terminal_path.exists():
            _exclusive_json(
                terminal_path,
                {
                    "schema_version": "p1.mstcn_checkpoint_diagnostic.terminal.v1",
                    "experiment_id": EXPERIMENT_ID,
                    "status": "FAILED_NO_AUTOMATIC_RETRY",
                    "started_at_utc": started.isoformat(),
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "submission_created": False,
                    "upload_performed": False,
                },
            )
        if not (artifact_dir / "manifest.json").exists():
            _write_manifest(artifact_dir)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-runner-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check_only:
        result = check_only()
    else:
        if not args.expected_runner_sha256:
            raise ContractError("--execute requires --expected-runner-sha256")
        result = execute(expected_runner_sha256=args.expected_runner_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
