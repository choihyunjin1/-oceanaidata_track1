"""Sealed three-window nested local-only LoRA evaluation for P3 Chronos-2."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from p3_wave.chronos2_transfer import (
    CONTEXT_STEPS_20MIN,
    PREDICTION_STEPS_20MIN,
    point_predictions,
    prepare_context_inputs,
    prepare_training_episodes,
    rmse,
    sha256_file,
    write_json,
)
from p3_wave.data import LEADS, select_independent_validation
from p3_wave.validation import build_forecast_folds

STRUCTURAL_COLUMNS = ["anchor_id", "station", "anchor_time", "grid_position", "current_hs"]
TARGET_COLUMNS = ["anchor_id", *[f"target_{lead}" for lead in LEADS]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/p3_chronos2_full_nested_20260828_v2.json",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_pipeline(config: dict[str, Any]):
    import torch
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(
        config["upstream"]["model_id"],
        revision=config["upstream"]["model_revision"],
        device_map="cuda",
        dtype=torch.bfloat16,
    )


def _validate_binding(binding: dict[str, str], name: str) -> Path:
    path = Path(binding["path"])
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise RuntimeError(f"{name} hash mismatch: {actual}")
    return path


def _read_filtered_targets(path: Path, anchor_ids: np.ndarray) -> pd.DataFrame:
    """Read only requested train target rows; never call before the prediction seal for outer IDs."""

    dataset = ds.dataset(path, format="parquet")
    condition = ds.field("anchor_id").isin(np.asarray(anchor_ids, dtype=np.int64).tolist())
    frame = dataset.to_table(columns=TARGET_COLUMNS, filter=condition).to_pandas()
    if len(frame) != len(anchor_ids) or set(frame["anchor_id"]) != set(map(int, anchor_ids)):
        raise RuntimeError("filtered anchor target read is incomplete")
    return frame.sort_values("anchor_id").reset_index(drop=True)


def _p3_train_wave_path() -> Path:
    root = os.environ.get("P3_DATA_DIR")
    if not root:
        raise RuntimeError("P3_DATA_DIR is required and must point to the immutable P3 source")
    path = Path(root).expanduser().resolve() / "train_wave.csv"
    if not path.is_file():
        raise FileNotFoundError("P3_DATA_DIR/train_wave.csv is missing")
    return path


def _stratified_tail_ids(
    anchors: pd.DataFrame, ids: np.ndarray, max_cases_per_station: int
) -> np.ndarray:
    subset = anchors.set_index("anchor_id").loc[ids].reset_index()
    selected = (
        subset.sort_values("anchor_time")
        .groupby("station", observed=True, group_keys=False)
        .tail(max_cases_per_station)
    )
    return np.sort(selected["anchor_id"].to_numpy(dtype=np.int64))


def _truth_matrix(targets: pd.DataFrame, anchor_ids: np.ndarray) -> np.ndarray:
    lookup = targets.set_index("anchor_id")
    return np.column_stack(
        [lookup.loc[anchor_ids, f"target_{lead}"].to_numpy(dtype=float) for lead in LEADS]
    )


def _adapter_hashes(checkpoint: Path) -> dict[str, str]:
    config_path = checkpoint / "adapter_config.json"
    weights_path = checkpoint / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError("LoRA adapter checkpoint is incomplete")
    return {
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weights_sha256": sha256_file(weights_path),
    }


def _fit_lora(
    base_config: dict[str, Any],
    nested_config: dict[str, Any],
    episodes: list[dict[str, object]],
    *,
    steps: int,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    selection = nested_config["inner_selection"]
    _set_seed(int(selection["seed"]))
    pipeline = _load_pipeline(base_config)
    started = time.perf_counter()
    tuned = pipeline.fit(
        episodes,
        prediction_length=PREDICTION_STEPS_20MIN,
        context_length=CONTEXT_STEPS_20MIN,
        min_past=CONTEXT_STEPS_20MIN,
        finetune_mode="lora",
        num_steps=int(steps),
        batch_size=int(selection["batch_size_series"]),
        learning_rate=float(selection["learning_rate"]),
        output_dir=output_dir,
        save_strategy="no",
        eval_strategy="no",
    )
    elapsed = time.perf_counter() - started
    model_class = type(tuned.model).__name__
    if "Peft" not in model_class:
        raise RuntimeError(f"LoRA silently fell back to full fine-tuning: {model_class}")
    checkpoint = output_dir / "finetuned-ckpt"
    metadata = {
        "steps": int(steps),
        "runtime_seconds": elapsed,
        "model_class": model_class,
        "checkpoint_dir": str(checkpoint.resolve()),
        **_adapter_hashes(checkpoint),
    }
    return tuned, metadata


def _blind_frame(
    anchors: pd.DataFrame,
    ids: np.ndarray,
    prediction: np.ndarray,
    fold_name: str,
) -> pd.DataFrame:
    lookup = anchors.set_index("anchor_id")
    rows: list[dict[str, object]] = []
    for row_number, anchor_id in enumerate(ids):
        row = lookup.loc[int(anchor_id)]
        for lead_number, lead in enumerate(LEADS):
            rows.append(
                {
                    "fold": fold_name,
                    "anchor_id": int(anchor_id),
                    "anchor_time": pd.Timestamp(row["anchor_time"]),
                    "station": str(row["station"]),
                    "lead_h": int(lead),
                    "current_hs": float(row["current_hs"]),
                    "prediction": float(prediction[row_number, lead_number]),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    if len(frame) != 6 * len(ids) or not np.isfinite(frame["prediction"]).all():
        raise RuntimeError("invalid blind outer prediction frame")
    return frame


def _metric_slices(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled_rmse_m": rmse(frame["target_hs"], frame[prediction_column]),
        "by_fold_rmse_m": {},
        "by_station_rmse_m": {},
        "by_lead_rmse_m": {},
    }
    for dimension, output_key in (
        ("fold", "by_fold_rmse_m"),
        ("station", "by_station_rmse_m"),
        ("lead_h", "by_lead_rmse_m"),
    ):
        result[output_key] = {
            str(key): rmse(group["target_hs"], group[prediction_column])
            for key, group in frame.groupby(dimension, observed=True)
        }
    return result


def _slice_deltas(
    candidate: dict[str, Any], comparator: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled_delta_m": candidate["pooled_rmse_m"] - comparator["pooled_rmse_m"]
    }
    for key in ("by_fold_rmse_m", "by_station_rmse_m", "by_lead_rmse_m"):
        result[key.replace("rmse", "delta")] = {
            name: candidate[key][name] - comparator[key][name] for name in candidate[key]
        }
    return result


def _cluster_bootstrap(
    frame: pd.DataFrame,
    comparator_column: str,
    cluster_column: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    candidate_error = np.square(frame["prediction"].to_numpy() - frame["target_hs"].to_numpy())
    comparator_error = np.square(
        frame[comparator_column].to_numpy() - frame["target_hs"].to_numpy()
    )
    working = frame[[cluster_column]].copy()
    working["candidate_sq"] = candidate_error
    working["comparator_sq"] = comparator_error
    grouped = working.groupby(cluster_column, observed=True).agg(
        candidate_sq=("candidate_sq", "sum"),
        comparator_sq=("comparator_sq", "sum"),
        rows=("candidate_sq", "size"),
    )
    values = grouped[["candidate_sq", "comparator_sq", "rows"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(values), size=len(values))
        sums = values[sample].sum(axis=0)
        deltas[index] = np.sqrt(sums[0] / sums[2]) - np.sqrt(sums[1] / sums[2])
    observed = rmse(frame["target_hs"], frame["prediction"]) - rmse(
        frame["target_hs"], frame[comparator_column]
    )
    return {
        "cluster": cluster_column,
        "clusters": int(len(values)),
        "replicates": int(replicates),
        "observed_delta_m": observed,
        "ci90_m": [float(np.quantile(deltas, 0.05)), float(np.quantile(deltas, 0.95))],
        "probability_delta_below_zero": float(np.mean(deltas < 0.0)),
    }


def _environment_gate(config: dict[str, Any]) -> None:
    parent = config["parent_experiment"]
    if sha256_file(parent["path"]) != parent["sha256"]:
        raise RuntimeError("parent experiment changed")
    preflight = config["environment_preflight"]
    if sha256_file(preflight["path"]) != preflight["sha256"]:
        raise RuntimeError("environment preflight changed")
    if _read_json(preflight["path"])["status"] != preflight["required_status"]:
        raise RuntimeError("environment preflight is not PASS")


def execute(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    started_all = time.perf_counter()
    _environment_gate(config)
    bindings = config["bindings"]
    sequence_path = _validate_binding(bindings["train_sequences"], "train_sequences")
    anchor_path = _validate_binding(bindings["train_anchors"], "train_anchors")
    incumbent_path = _validate_binding(bindings["frozen_incumbent_oof"], "incumbent_oof")
    output = Path(config["artifact_dir"])
    terminal_result = output / "full_nested_result.json"
    if terminal_result.exists():
        raise RuntimeError("terminal v2 result already exists; append-only rerun forbidden")
    output.mkdir(parents=True, exist_ok=True)
    attempt_lock = output / "ATTEMPT_LOCK.json"
    if not attempt_lock.exists():
        write_json(
            attempt_lock,
            {
                "experiment_id": config["experiment_id"],
                "started_at_utc": _now_utc(),
                "config_sha256": sha256_file(config_path),
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            },
        )
    anchors = pd.read_parquet(anchor_path, columns=STRUCTURAL_COLUMNS)
    raw_values = np.load(sequence_path, mmap_mode="r")
    folds = build_forecast_folds(
        anchors,
        windows=tuple(tuple(item) for item in config["outer_validation"]["windows"]),
        embargo_hours=int(config["outer_validation"]["embargo_hours"]),
    )
    selection = config["inner_selection"]
    train_wave_path = _p3_train_wave_path()
    blind_frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        fold_started = time.perf_counter()
        outer_train = anchors.set_index("anchor_id").loc[fold.train_ids]
        inner_end = outer_train["anchor_time"].max() + pd.Timedelta(minutes=20)
        inner_start = inner_end - pd.Timedelta(days=int(selection["window_days"]))
        inner_ids = select_independent_validation(
            anchors,
            start=inner_start,
            end=inner_end,
            gap_hours=int(selection["gap_hours"]),
        )
        inner_ids = np.intersect1d(inner_ids, fold.train_ids)
        fit_end = inner_start - pd.Timedelta(hours=int(selection["gap_hours"]))
        inner_fit_ids = outer_train.loc[
            outer_train["anchor_time"].lt(fit_end)
        ].index.to_numpy(dtype=np.int64)
        inner_fit_ids = _stratified_tail_ids(
            anchors, inner_fit_ids, int(selection["fit_max_cases_per_station"])
        )
        inner_episodes, kept_inner_fit = prepare_training_episodes(
            raw_values, anchors, inner_fit_ids, train_wave_path
        )
        inner_targets = _read_filtered_targets(anchor_path, inner_ids)
        inner_truth = _truth_matrix(inner_targets, inner_ids)
        inner_inputs = prepare_context_inputs(raw_values, inner_ids)
        candidates: list[dict[str, Any]] = []
        for steps in selection["step_budgets"]:
            candidate_dir = output / "inner_models" / fold.name / f"steps_{int(steps):04d}"
            tuned, metadata = _fit_lora(
                _read_json(config["parent_experiment"]["path"]),
                config,
                inner_episodes,
                steps=int(steps),
                output_dir=candidate_dir,
            )
            prediction = point_predictions(tuned, inner_inputs, batch_size=120)
            metadata["inner_validation_rmse_m"] = rmse(inner_truth, prediction)
            candidates.append(metadata)
        candidates.sort(key=lambda item: (item["inner_validation_rmse_m"], item["steps"]))
        selected = candidates[0]
        outer_fit_ids = _stratified_tail_ids(
            anchors,
            fold.train_ids,
            int(selection["fit_max_cases_per_station"]),
        )
        outer_episodes, kept_outer_fit = prepare_training_episodes(
            raw_values, anchors, outer_fit_ids, train_wave_path
        )
        final_dir = output / "outer_models" / fold.name / f"steps_{selected['steps']:04d}"
        final_model, final_metadata = _fit_lora(
            _read_json(config["parent_experiment"]["path"]),
            config,
            outer_episodes,
            steps=int(selected["steps"]),
            output_dir=final_dir,
        )
        outer_inputs = prepare_context_inputs(raw_values, fold.validation_ids)
        outer_prediction = point_predictions(final_model, outer_inputs, batch_size=120)
        blind = _blind_frame(anchors, fold.validation_ids, outer_prediction, fold.name)
        blind_path = output / "sealed_predictions" / f"{fold.name}.parquet"
        blind_path.parent.mkdir(parents=True, exist_ok=True)
        blind.to_parquet(blind_path, index=False)
        blind_sha = sha256_file(blind_path)
        blind_frames.append(blind)
        record = {
            "fold": fold.name,
            "inner_fit_cases_requested": int(len(inner_fit_ids)),
            "inner_fit_cases_complete": int(len(kept_inner_fit)),
            "inner_validation_cases": int(len(inner_ids)),
            "inner_candidates": candidates,
            "selected_steps": int(selected["steps"]),
            "outer_fit_cases_requested": int(len(outer_fit_ids)),
            "outer_fit_cases_complete": int(len(kept_outer_fit)),
            "outer_validation_cases": int(len(fold.validation_ids)),
            "outer_model": final_metadata,
            "blind_prediction_path": str(blind_path.resolve()),
            "blind_prediction_sha256": blind_sha,
            "runtime_seconds": time.perf_counter() - fold_started,
        }
        fold_records.append(record)
        write_json(output / "fold_records" / f"{fold.name}.json", record)
    blind_all = pd.concat(blind_frames, ignore_index=True)
    combined_path = output / "sealed_predictions" / "all_outer_predictions.parquet"
    blind_all.to_parquet(combined_path, index=False)
    combined_sha = sha256_file(combined_path)
    seal_path = output / "PREDICTION_SEAL.json"
    seal = {
        "experiment_id": config["experiment_id"],
        "sealed_at_utc": _now_utc(),
        "outer_truth_read_before_seal": False,
        "rows": int(len(blind_all)),
        "cases": int(blind_all["anchor_id"].nunique()),
        "fold_files": [
            {
                "fold": record["fold"],
                "path": record["blind_prediction_path"],
                "sha256": record["blind_prediction_sha256"],
            }
            for record in fold_records
        ],
        "combined_path": str(combined_path.resolve()),
        "combined_sha256": combined_sha,
    }
    write_json(seal_path, seal)
    seal_sha = sha256_file(seal_path)

    # The first outer-target read occurs only after all predictions and the seal exist.
    outer_ids = np.sort(blind_all["anchor_id"].unique().astype(np.int64))
    outer_targets = _read_filtered_targets(anchor_path, outer_ids)
    target_lookup = outer_targets.set_index("anchor_id")
    evaluated = blind_all.copy()
    for lead in LEADS:
        mask = evaluated["lead_h"].eq(lead)
        evaluated.loc[mask, "target_hs"] = target_lookup.loc[
            evaluated.loc[mask, "anchor_id"].to_numpy(dtype=np.int64), f"target_{lead}"
        ].to_numpy(dtype=float)
    incumbent = pd.read_parquet(incumbent_path)
    incumbent = incumbent.loc[
        incumbent["prefix_fraction"].eq(1.0),
        ["anchor_id", "lead_h", "incumbent_prediction"],
    ]
    evaluated = evaluated.merge(incumbent, on=["anchor_id", "lead_h"], validate="one_to_one")
    evaluated["persistence_prediction"] = evaluated["current_hs"]
    evaluated["anchor_day_utc"] = pd.to_datetime(
        evaluated["anchor_time"], utc=True
    ).dt.strftime("%Y-%m-%d")
    evaluated_path = output / "evaluated_outer_predictions.parquet"
    evaluated.to_parquet(evaluated_path, index=False)
    metrics = {
        "candidate": _metric_slices(evaluated, "prediction"),
        "persistence": _metric_slices(evaluated, "persistence_prediction"),
        "incumbent": _metric_slices(evaluated, "incumbent_prediction"),
    }
    metrics["delta_vs_persistence"] = _slice_deltas(
        metrics["candidate"], metrics["persistence"]
    )
    metrics["delta_vs_incumbent"] = _slice_deltas(
        metrics["candidate"], metrics["incumbent"]
    )
    bootstrap_reps = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    metrics["bootstrap"] = {
        comparator: {
            cluster: _cluster_bootstrap(
                evaluated,
                column,
                cluster,
                replicates=bootstrap_reps,
                seed=bootstrap_seed + offset,
            )
            for offset, cluster in enumerate(("anchor_id", "anchor_day_utc"))
        }
        for comparator, column in (
            ("persistence", "persistence_prediction"),
            ("incumbent", "incumbent_prediction"),
        )
    }
    fold_deltas = metrics["delta_vs_incumbent"]["by_fold_delta_m"]
    research_go = (
        np.isfinite(evaluated["prediction"]).all()
        and len(fold_records) == 3
        and metrics["candidate"]["pooled_rmse_m"]
        < metrics["persistence"]["pooled_rmse_m"]
    )
    official_value_go = (
        metrics["candidate"]["pooled_rmse_m"]
        < metrics["incumbent"]["pooled_rmse_m"]
        and sum(delta < 0.0 for delta in fold_deltas.values()) >= 2
        and metrics["bootstrap"]["incumbent"]["anchor_id"]["ci90_m"][1] < 0.0
    )
    result = {
        "experiment_id": config["experiment_id"],
        "status": "TERMINAL_PASS" if research_go else "TERMINAL_NO_GO",
        "research_go": bool(research_go),
        "official_value_go": bool(official_value_go),
        "checkpoint_provenance_gate": config["upstream"]["checkpoint_provenance_gate"],
        "official_submission_authorized": False,
        "config_sha256": sha256_file(config_path),
        "attempt_lock_sha256": sha256_file(attempt_lock),
        "prediction_seal_sha256": seal_sha,
        "combined_blind_prediction_sha256": combined_sha,
        "evaluated_outer_sha256": sha256_file(evaluated_path),
        "truth_first_read_after_seal_at_utc": _now_utc(),
        "fold_records": fold_records,
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started_all,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_worktree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }
    write_json(terminal_result, result)
    return result


def main() -> int:
    args = parse_args()
    if not args.execute:
        raise SystemExit("Pass --execute to consume this new v2 local-only attempt")
    config_path = Path(args.config)
    config = _read_json(config_path)
    result = execute(config_path, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
