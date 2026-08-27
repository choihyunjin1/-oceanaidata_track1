#!/usr/bin/env python3
"""Run one sealed matched-budget P2 continuous-depth T/S screen.

The script reads only ``observations.csv``.  It never opens P2 test indices,
sample submissions, baselines, prior submissions, or any external mirror.
Its two arms have identical model initialization, batches, optimizer, epochs,
and inference.  The sole treatment is a fixed density/N2-proxy loss weight.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from p2_restore.features import TARGET_LAYERS  # noqa: E402
from p2_restore.ts_continuous_depth_challenger_20260827_v1 import (  # noqa: E402
    ContinuousDepthNormalizer,
    ContinuousDepthPanel,
    TSContinuousDepthTCN,
    build_continuous_depth_panel,
    density_n2_proxy,
    materialize_training_chunks,
    predict_panel,
)


RUN_ID = "p2_ts_continuous_depth_challenger_20260827_v1"
SEED = 20260827
EPOCHS = 18
BATCH_SIZE = 16
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-3
GRADIENT_CLIP = 1.0
HIDDEN = 96
DILATIONS = (1, 2, 4, 8, 16)
DROPOUT = 0.05
CHUNK_LENGTH = 512
CHUNK_STRIDE = 384
VERTICAL_DIFFERENCE_WEIGHT = 0.25
CONTROL_PHYSICS_WEIGHT = 0.0
CHALLENGER_PHYSICS_WEIGHT = 0.10
PURGE_DAYS = 7
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260828


@dataclass(frozen=True)
class Fold:
    name: str
    start: str
    stop: str


FOLDS = (
    Fold("2024_sep_oct", "2024-09-01", "2024-11-01"),
    Fold("2025_jul_aug", "2025-07-01", "2025-09-01"),
    Fold("2025_nov_dec", "2025-11-01", "2026-01-01"),
)


PROMOTION_RULE = {
    "promote": {
        "overall_temperature_rmse_delta_max": -0.001,
        "paired_day_bootstrap_90ci_upper_max": 0.0,
        "minimum_improved_blocks": 2,
        "minimum_improved_layers": 2,
        "worst_block_delta_max": 0.0025,
        "worst_layer_delta_max": 0.0025,
    },
    "reject": {
        "paired_day_bootstrap_90ci_lower_min": 0.0,
        "or_overall_delta_min_with_at_most_one_improved_block": 0.003,
        "or_worst_block_delta_min": 0.01,
        "or_worst_layer_delta_min": 0.01,
    },
    "otherwise": "INCONCLUSIVE",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ["P2_DATA_DIR"]) if os.environ.get("P2_DATA_DIR") else None,
        help="Directory containing observations.csv. May also be supplied by P2_DATA_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "structural_challenger_20260827_v1" / "p2",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required to execute the preregistered one-shot screen.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _json_number(value: float) -> float | None:
    current = float(value)
    return current if math.isfinite(current) else None


def _rmse(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    keep = np.asarray(mask, dtype=bool) & np.isfinite(truth) & np.isfinite(prediction)
    count = int(keep.sum())
    if count == 0:
        return {"rows": 0, "rmse": None}
    value = float(np.sqrt(np.mean(np.square(prediction[keep] - truth[keep]))))
    return {"rows": count, "rmse": value}


def _delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left["rmse"] is None or right["rmse"] is None:
        return None
    return float(left["rmse"] - right["rmse"])


def _kst_bounds(fold: Fold) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(fold.start, tz="Asia/Seoul").tz_convert("UTC"),
        pd.Timestamp(fold.stop, tz="Asia/Seoul").tz_convert("UTC"),
    )


def _fold_mask(times: pd.DatetimeIndex, fold: Fold) -> np.ndarray:
    start, stop = _kst_bounds(fold)
    return np.asarray((times >= start) & (times < stop), dtype=bool)


def _set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _train_arm(
    panel: ContinuousDepthPanel,
    normalizer: ContinuousDepthNormalizer,
    selected_times: np.ndarray,
    *,
    physics_weight: float,
    device: torch.device,
) -> tuple[TSContinuousDepthTCN, dict[str, Any]]:
    chunks = materialize_training_chunks(
        panel,
        normalizer,
        selected_times,
        length=CHUNK_LENGTH,
        stride=CHUNK_STRIDE,
    )
    _set_deterministic(SEED)
    model = TSContinuousDepthTCN(
        panel.inputs.shape[1],
        hidden=HIDDEN,
        dilations=DILATIONS,
        dropout=DROPOUT,
    ).to(device)
    initial_hash = _state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    target_center = torch.as_tensor(
        normalizer.target_center, dtype=torch.float32, device=device
    )
    target_scale = torch.as_tensor(
        normalizer.target_scale, dtype=torch.float32, device=device
    )
    n2_scale = torch.as_tensor(normalizer.n2_scale, dtype=torch.float32, device=device)
    updates = 0
    final_components = {"direct": 0.0, "vertical": 0.0, "density_n2": 0.0, "total": 0.0}
    for epoch in range(EPOCHS):
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(chunks.bounds))
        component_sums = {key: 0.0 for key in final_components}
        batches = 0
        for batch_start in range(0, len(order), BATCH_SIZE):
            indices = torch.as_tensor(order[batch_start : batch_start + BATCH_SIZE], dtype=torch.long)
            inputs = chunks.inputs.index_select(0, indices).to(device)
            query_depths = chunks.query_depths.index_select(0, indices).to(device)
            baselines = chunks.baselines.index_select(0, indices).to(device)
            targets = chunks.targets.index_select(0, indices).to(device)
            mask = chunks.mask.index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            direct, vertical, density = model.loss_components(
                inputs,
                query_depths,
                baselines,
                targets,
                mask,
                target_center=target_center,
                target_scale=target_scale,
                n2_scale=n2_scale,
            )
            loss = (
                direct
                + VERTICAL_DIFFERENCE_WEIGHT * vertical
                + float(physics_weight) * density
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            component_sums["direct"] += float(direct.detach().cpu())
            component_sums["vertical"] += float(vertical.detach().cpu())
            component_sums["density_n2"] += float(density.detach().cpu())
            component_sums["total"] += float(loss.detach().cpu())
            batches += 1
            updates += 1
        if epoch == EPOCHS - 1:
            final_components = {
                key: value / max(batches, 1) for key, value in component_sums.items()
            }
    metadata = {
        "initial_state_sha256": initial_hash,
        "final_state_sha256": _state_hash(model),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_chunks": len(chunks.bounds),
        "optimizer_updates": updates,
        "epochs": EPOCHS,
        "physics_weight": float(physics_weight),
        "final_epoch_mean_losses": final_components,
    }
    return model, metadata


def _paired_day_bootstrap(
    times: pd.DatetimeIndex,
    truth: np.ndarray,
    control: np.ndarray,
    physics: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    row_times = np.repeat(times.to_numpy(), len(TARGET_LAYERS))
    row_truth = truth.reshape(-1)
    row_control = control.reshape(-1)
    row_physics = physics.reshape(-1)
    keep = mask.reshape(-1) & np.isfinite(row_truth) & np.isfinite(row_control) & np.isfinite(
        row_physics
    )
    dates = pd.DatetimeIndex(row_times[keep]).tz_convert("Asia/Seoul").normalize()
    day_codes, unique_dates = pd.factorize(dates, sort=True)
    count = np.bincount(day_codes, minlength=len(unique_dates)).astype(np.float64)
    control_sse = np.bincount(
        day_codes,
        weights=np.square(row_control[keep] - row_truth[keep]),
        minlength=len(unique_dates),
    )
    physics_sse = np.bincount(
        day_codes,
        weights=np.square(row_physics[keep] - row_truth[keep]),
        minlength=len(unique_dates),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(unique_dates), size=len(unique_dates))
        denominator = count[sampled].sum()
        control_rmse = math.sqrt(float(control_sse[sampled].sum() / denominator))
        physics_rmse = math.sqrt(float(physics_sse[sampled].sum() / denominator))
        deltas[replicate] = physics_rmse - control_rmse
    low, median, high = np.quantile(deltas, (0.05, 0.50, 0.95))
    return {
        "unit": "KST_calendar_day",
        "days": int(len(unique_dates)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "delta_definition": "physics_rmse_minus_control_rmse",
        "ci90": [_json_number(low), _json_number(high)],
        "median": _json_number(median),
    }


def _decide(
    overall_delta: float,
    bootstrap: dict[str, Any],
    block_deltas: list[float],
    layer_deltas: list[float],
) -> tuple[str, list[str]]:
    improved_blocks = sum(value < 0.0 for value in block_deltas)
    improved_layers = sum(value < 0.0 for value in layer_deltas)
    low, high = bootstrap["ci90"]
    worst_block = max(block_deltas)
    worst_layer = max(layer_deltas)
    promote = (
        overall_delta <= -0.001
        and high < 0.0
        and improved_blocks >= 2
        and improved_layers >= 2
        and worst_block <= 0.0025
        and worst_layer <= 0.0025
    )
    if promote:
        return "PROMOTE", ["all preregistered promotion guards passed"]
    rejection_reasons: list[str] = []
    if low > 0.0:
        rejection_reasons.append("paired-day 90% CI is strictly harmful")
    if overall_delta >= 0.003 and improved_blocks <= 1:
        rejection_reasons.append("material overall harm with at most one improved block")
    if worst_block >= 0.01:
        rejection_reasons.append("at least one block worsened by >=0.01 RMSE")
    if worst_layer >= 0.01:
        rejection_reasons.append("at least one layer worsened by >=0.01 RMSE")
    if rejection_reasons:
        return "REJECT", rejection_reasons
    return "INCONCLUSIVE", ["promotion and rejection guards were both incomplete"]


def _sealed_config() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "hypothesis": (
            "A differentiable target-profile density-gradient loss improves the same "
            "continuous-depth joint T/S model over its no-physics matched control."
        ),
        "folds": [asdict(fold) for fold in FOLDS],
        "timezone": "Asia/Seoul",
        "target_layers": list(TARGET_LAYERS),
        "purge_days_each_side": PURGE_DAYS,
        "seed": SEED,
        "arms": {
            "control": {"physics_weight": CONTROL_PHYSICS_WEIGHT},
            "physics": {"physics_weight": CHALLENGER_PHYSICS_WEIGHT},
        },
        "shared_budget": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "hidden": HIDDEN,
            "dilations": list(DILATIONS),
            "dropout": DROPOUT,
            "chunk_length": CHUNK_LENGTH,
            "chunk_stride": CHUNK_STRIDE,
            "vertical_difference_weight": VERTICAL_DIFFERENCE_WEIGHT,
            "early_stopping": False,
            "seeds_per_arm": 1,
        },
        "metric": "row-pooled temperature RMSE over all finite target rows",
        "uncertainty": "paired KST-calendar-day bootstrap",
        "promotion_rule": PROMOTION_RULE,
        "no_result_driven_retuning": True,
    }


def _execute(args: argparse.Namespace) -> None:
    if args.data_dir is None:
        raise SystemExit("--data-dir or P2_DATA_DIR is required")
    observations_path = args.data_dir.resolve() / "observations.csv"
    if not observations_path.is_file():
        raise FileNotFoundError(observations_path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(
        observations_path,
        usecols=["layer", "time", "temp", "psal", "depth", "nominal_depth"],
    )
    observations["layer"] = observations["layer"].astype(int)
    parsed_time = pd.to_datetime(observations["time"], utc=True)
    sealed_panel = build_continuous_depth_panel(observations)
    if not sealed_panel.times.equals(pd.DatetimeIndex(parsed_time.drop_duplicates()).sort_values()):
        raise AssertionError("panel timestamps do not match the observation surface")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_control = np.full((len(sealed_panel.times), len(TARGET_LAYERS), 2), np.nan)
    oof_physics = np.full_like(oof_control, np.nan)
    fold_training: dict[str, Any] = {}
    fold_masks: dict[str, np.ndarray] = {}
    blind_audit: dict[str, Any] = {}
    for fold_index, fold in enumerate(FOLDS):
        validation_mask = _fold_mask(sealed_panel.times, fold)
        if not validation_mask.any():
            raise RuntimeError(f"fold has no timestamps: {fold.name}")
        fold_masks[fold.name] = validation_mask
        start, stop = _kst_bounds(fold)
        row_validation = np.asarray((parsed_time >= start) & (parsed_time < stop), dtype=bool)
        target_rows = observations["layer"].isin(TARGET_LAYERS).to_numpy()
        withheld_rows = row_validation & target_rows
        fold_blind = observations.copy()
        fold_blind.loc[withheld_rows, ["temp", "psal"]] = np.nan
        blind_panel = build_continuous_depth_panel(fold_blind)
        if blind_panel.input_names != sealed_panel.input_names:
            raise AssertionError("fold masking changed public feature names")
        if not np.array_equal(blind_panel.inputs, sealed_panel.inputs, equal_nan=True):
            raise AssertionError("fold masking changed public model inputs")
        if blind_panel.temperature_mask[validation_mask].any():
            raise AssertionError("validation target temperature survived simultaneous masking")
        if blind_panel.salinity_mask[validation_mask].any():
            raise AssertionError("validation target salinity survived simultaneous masking")

        purge_start = start - pd.Timedelta(days=PURGE_DAYS)
        purge_stop = stop + pd.Timedelta(days=PURGE_DAYS)
        train_selected = np.asarray(
            (blind_panel.times < purge_start) | (blind_panel.times >= purge_stop), dtype=bool
        )
        normalizer = ContinuousDepthNormalizer.fit(blind_panel, train_selected)
        blind_audit[fold.name] = {
            "validation_timestamps": int(validation_mask.sum()),
            "target_rows_simultaneously_masked": int(withheld_rows.sum()),
            "public_input_identity_after_masking": True,
            "validation_temperature_labels_in_training_panel": int(
                blind_panel.temperature_mask[validation_mask].sum()
            ),
            "validation_salinity_labels_in_training_panel": int(
                blind_panel.salinity_mask[validation_mask].sum()
            ),
            "purged_training_timestamps": int((~train_selected).sum()),
        }

        arm_predictions: dict[str, np.ndarray] = {}
        arm_metadata: dict[str, Any] = {}
        arm_order = ("control", "physics") if fold_index % 2 == 0 else ("physics", "control")
        for arm in arm_order:
            physics_weight = (
                CONTROL_PHYSICS_WEIGHT if arm == "control" else CHALLENGER_PHYSICS_WEIGHT
            )
            model, training_metadata = _train_arm(
                blind_panel,
                normalizer,
                train_selected,
                physics_weight=physics_weight,
                device=device,
            )
            arm_predictions[arm] = predict_panel(
                model,
                blind_panel,
                normalizer,
                device=device,
                length=CHUNK_LENGTH,
                stride=CHUNK_STRIDE,
                batch_size=BATCH_SIZE,
            )
            arm_metadata[arm] = training_metadata
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if arm_metadata["control"]["initial_state_sha256"] != arm_metadata["physics"][
            "initial_state_sha256"
        ]:
            raise AssertionError("matched arms did not start from identical parameters")
        for key in ("training_chunks", "optimizer_updates", "epochs", "parameter_count"):
            if arm_metadata["control"][key] != arm_metadata["physics"][key]:
                raise AssertionError(f"matched arm budget differs for {key}")
        oof_control[validation_mask] = arm_predictions["control"][validation_mask]
        oof_physics[validation_mask] = arm_predictions["physics"][validation_mask]
        fold_training[fold.name] = {
            "arm_execution_order": list(arm_order),
            "control": arm_metadata["control"],
            "physics": arm_metadata["physics"],
            "matched_initial_state": True,
            "matched_optimizer_updates": True,
        }

    validation_union = np.zeros(len(sealed_panel.times), dtype=bool)
    for mask in fold_masks.values():
        validation_union |= mask
    temperature_eval_mask = sealed_panel.temperature_mask & validation_union[:, None]
    salinity_eval_mask = sealed_panel.salinity_mask & validation_union[:, None]
    baseline = np.stack(
        (sealed_panel.temperature_baseline, sealed_panel.salinity_baseline), axis=-1
    )
    truth = np.stack((sealed_panel.target_temperature, sealed_panel.target_salinity), axis=-1)

    overall_control = _rmse(
        truth[:, :, 0], oof_control[:, :, 0], temperature_eval_mask
    )
    overall_physics = _rmse(
        truth[:, :, 0], oof_physics[:, :, 0], temperature_eval_mask
    )
    overall_baseline = _rmse(
        truth[:, :, 0], baseline[:, :, 0], temperature_eval_mask
    )
    overall_delta = _delta(overall_physics, overall_control)
    if overall_delta is None:
        raise RuntimeError("primary temperature metric is unavailable")

    by_block: dict[str, Any] = {}
    block_deltas: list[float] = []
    for fold in FOLDS:
        time_mask = fold_masks[fold.name]
        temp_mask = sealed_panel.temperature_mask & time_mask[:, None]
        sal_mask = sealed_panel.salinity_mask & time_mask[:, None]
        control_temp = _rmse(truth[:, :, 0], oof_control[:, :, 0], temp_mask)
        physics_temp = _rmse(truth[:, :, 0], oof_physics[:, :, 0], temp_mask)
        delta = _delta(physics_temp, control_temp)
        if delta is None:
            raise RuntimeError(f"temperature block metric unavailable: {fold.name}")
        block_deltas.append(delta)
        by_block[fold.name] = {
            "temperature": {
                "baseline": _rmse(truth[:, :, 0], baseline[:, :, 0], temp_mask),
                "control": control_temp,
                "physics": physics_temp,
                "physics_minus_control_rmse": delta,
            },
            "salinity_secondary": {
                "baseline": _rmse(truth[:, :, 1], baseline[:, :, 1], sal_mask),
                "control": _rmse(truth[:, :, 1], oof_control[:, :, 1], sal_mask),
                "physics": _rmse(truth[:, :, 1], oof_physics[:, :, 1], sal_mask),
            },
        }

    by_layer: dict[str, Any] = {}
    layer_deltas: list[float] = []
    for offset, layer in enumerate(TARGET_LAYERS):
        temp_mask = temperature_eval_mask[:, offset]
        sal_mask = salinity_eval_mask[:, offset]
        control_temp = _rmse(truth[:, offset, 0], oof_control[:, offset, 0], temp_mask)
        physics_temp = _rmse(truth[:, offset, 0], oof_physics[:, offset, 0], temp_mask)
        delta = _delta(physics_temp, control_temp)
        if delta is None:
            raise RuntimeError(f"temperature layer metric unavailable: {layer}")
        layer_deltas.append(delta)
        by_layer[str(layer)] = {
            "temperature": {
                "baseline": _rmse(truth[:, offset, 0], baseline[:, offset, 0], temp_mask),
                "control": control_temp,
                "physics": physics_temp,
                "physics_minus_control_rmse": delta,
            },
            "salinity_secondary": {
                "baseline": _rmse(truth[:, offset, 1], baseline[:, offset, 1], sal_mask),
                "control": _rmse(truth[:, offset, 1], oof_control[:, offset, 1], sal_mask),
                "physics": _rmse(truth[:, offset, 1], oof_physics[:, offset, 1], sal_mask),
            },
        }

    bootstrap = _paired_day_bootstrap(
        sealed_panel.times,
        truth[:, :, 0],
        oof_control[:, :, 0],
        oof_physics[:, :, 0],
        temperature_eval_mask,
    )
    decision, decision_reasons = _decide(
        overall_delta,
        bootstrap,
        block_deltas,
        layer_deltas,
    )

    true_n2 = density_n2_proxy(
        truth[:, :, 0], truth[:, :, 1], sealed_panel.query_depths
    )
    control_n2 = density_n2_proxy(
        oof_control[:, :, 0], oof_control[:, :, 1], sealed_panel.query_depths
    )
    physics_n2 = density_n2_proxy(
        oof_physics[:, :, 0], oof_physics[:, :, 1], sealed_panel.query_depths
    )
    joint_eval = sealed_panel.joint_mask & validation_union[:, None]
    n2_by_pair: dict[str, Any] = {}
    for pair in range(len(TARGET_LAYERS) - 1):
        mask = joint_eval[:, pair] & joint_eval[:, pair + 1]
        control_metric = _rmse(true_n2[:, pair], control_n2[:, pair], mask)
        physics_metric = _rmse(true_n2[:, pair], physics_n2[:, pair], mask)
        n2_by_pair[f"{TARGET_LAYERS[pair]}-{TARGET_LAYERS[pair + 1]}"] = {
            "control": control_metric,
            "physics": physics_metric,
            "physics_minus_control_rmse": _delta(physics_metric, control_metric),
        }

    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "sealed_config": _sealed_config(),
        "primary_temperature": {
            "baseline": overall_baseline,
            "control": overall_control,
            "physics": overall_physics,
            "physics_minus_control_rmse": overall_delta,
            "paired_day_bootstrap": bootstrap,
        },
        "by_block": by_block,
        "by_layer": by_layer,
        "density_n2_secondary_by_adjacent_pair": n2_by_pair,
        "masking_audit": blind_audit,
        "matched_training_audit": fold_training,
        "nonduplication_audit": {
            "fixed_depth_deeponet": "prior decoder used a fixed three-depth buffer and T-only output",
            "joint_hydrographic_multitask": (
                "prior joint model used fixed depths and direct/vertical T/S losses but no output-density loss"
            ),
            "teos_analog": (
                "prior analog used a simplified public-state density proxy for neighbour similarity, "
                "not a predicted target T/S density-gradient loss"
            ),
            "profile_projection": "prior projection imposed temperature-only monotonic PAVA/envelopes",
            "soft_gates": "prior soft gates blended completed predictors without joint physical training",
        },
        "limitations": [
            "The density relation is an explicitly linearized differentiable proxy, not full TEOS-10.",
            "Only one preregistered seed and one fixed loss weight were screened.",
            "The three local blocks may not reproduce the hidden distribution.",
            "The experiment isolates the physics loss; it does not compare model-class capacity to the incumbent.",
            "No result-driven rerun or hyperparameter tuning is permitted for this screen.",
        ],
        "data_access_audit": {
            "files_opened_by_runner": ["observations.csv"],
            "official_test_opened": False,
            "sample_submission_opened": False,
            "submission_created": False,
            "raw_observation_values_written": False,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    script_path = Path(__file__).resolve()
    module_path = SOURCE_ROOT / "p2_restore" / "ts_continuous_depth_challenger_20260827_v1.py"
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sanitized_command": (
            ".venv-p1/Scripts/python.exe scripts/run_p2_ts_continuous_depth_challenger_20260827_v1.py "
            "--data-dir $P2_DATA_DIR --execute"
        ),
        "source_sha256": {
            script_path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256_file(script_path),
            module_path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256_file(module_path),
        },
        "input": {
            "filename": observations_path.name,
            "sha256": _sha256_file(observations_path),
            "rows": int(len(observations)),
        },
        "result_sha256": _sha256_file(result_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "decision": decision,
                "temperature_control_rmse": overall_control["rmse"],
                "temperature_physics_rmse": overall_physics["rmse"],
                "physics_minus_control_rmse": overall_delta,
                "paired_day_bootstrap_ci90": bootstrap["ci90"],
                "result": str(result_path.relative_to(REPOSITORY_ROOT)),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = _parse_args()
    if not args.execute:
        print(json.dumps(_sealed_config(), indent=2, sort_keys=True))
        return
    _execute(args)


if __name__ == "__main__":
    main()
