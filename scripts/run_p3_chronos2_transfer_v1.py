"""Run the isolated, train-only P3 Chronos-2 environment and suitability probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from p3_wave.chronos2_transfer import (
    CONTEXT_STEPS_20MIN,
    PREDICTION_STEPS_20MIN,
    point_predictions,
    prediction_frame,
    prepare_context_inputs,
    prepare_training_episodes,
    rmse,
    score_frame,
    sha256_file,
    write_json,
)
from p3_wave.data import select_independent_validation
from p3_wave.validation import build_forecast_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/p3_chronos2_transfer_20260828_v1.json"
    )
    parser.add_argument(
        "--stage", choices=("preflight", "zero-shot", "lora-smoke"), required=True
    )
    parser.add_argument("--gpu-smoke", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pipeline(config: dict[str, object]):
    import torch
    from chronos import Chronos2Pipeline

    upstream = config["upstream"]
    return Chronos2Pipeline.from_pretrained(
        upstream["model_id"],
        revision=upstream["model_revision"],
        device_map="cuda",
        dtype=torch.bfloat16,
    )


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def direct_url_commit() -> str | None:
    distribution = importlib.metadata.distribution("chronos-forecasting")
    text = distribution.read_text("direct_url.json")
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def preflight(config: dict[str, object], output: Path, gpu_smoke: bool) -> dict[str, object]:
    import accelerate
    import chronos
    import peft
    import torch
    import transformers
    from chronos import Chronos2Pipeline
    from transformers.utils.import_utils import is_peft_available

    upstream = config["upstream"]
    signature = inspect.signature(Chronos2Pipeline.fit)
    commit = direct_url_commit()
    checks = {
        "cuda_available": bool(torch.cuda.is_available()),
        "fit_has_finetune_mode": "finetune_mode" in signature.parameters,
        "fit_has_lora_config": "lora_config" in signature.parameters,
        "peft_importable": bool(is_peft_available()),
        "official_commit_exact": commit == upstream["commit_sha"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Chronos environment fail-closed: {checks}")
    report: dict[str, object] = {
        "status": "PASS",
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "versions": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "chronos": chronos.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "chronos_direct_url_commit": commit,
        "fit_signature": str(signature),
        "checks": checks,
        "gpu_smoke": None,
    }
    if gpu_smoke:
        set_seed(int(config["validation"]["seed"]))
        pipeline = load_pipeline(config)
        synthetic = [
            np.linspace(1.0, 2.0, CONTEXT_STEPS_20MIN + PREDICTION_STEPS_20MIN).astype(
                np.float32
            ),
            np.linspace(2.0, 1.0, CONTEXT_STEPS_20MIN + PREDICTION_STEPS_20MIN).astype(
                np.float32
            ),
        ]
        started = time.perf_counter()
        forecast = pipeline.predict(
            [row[:CONTEXT_STEPS_20MIN] for row in synthetic],
            prediction_length=PREDICTION_STEPS_20MIN,
            batch_size=2,
        )
        predict_seconds = time.perf_counter() - started
        smoke_dir = output / "synthetic_lora_smoke"
        started = time.perf_counter()
        tuned = pipeline.fit(
            synthetic,
            prediction_length=PREDICTION_STEPS_20MIN,
            context_length=CONTEXT_STEPS_20MIN,
            min_past=CONTEXT_STEPS_20MIN,
            finetune_mode="lora",
            num_steps=1,
            batch_size=2,
            learning_rate=1e-5,
            output_dir=smoke_dir,
            save_strategy="no",
            eval_strategy="no",
        )
        fit_seconds = time.perf_counter() - started
        model_class = type(tuned.model).__name__
        if "Peft" not in model_class:
            raise RuntimeError(f"LoRA silently fell back to full fine-tuning: {model_class}")
        trainable = sum(parameter.numel() for parameter in tuned.model.parameters() if parameter.requires_grad)
        report["gpu_smoke"] = {
            "prediction_seconds": predict_seconds,
            "prediction_shape": list(forecast[0].shape),
            "prediction_finite": bool(torch.isfinite(forecast[0]).all()),
            "one_step_lora_seconds": fit_seconds,
            "model_class": model_class,
            "trainable_parameters": int(trainable),
            "checkpoint_dir": str((smoke_dir / "finetuned-ckpt").resolve()),
        }
    return report


def _load_train_bindings(config: dict[str, object]):
    bindings = config["bindings"]
    for key in ("train_sequences", "train_station", "train_anchors"):
        path = Path(bindings[key]["path"])
        actual = sha256_file(path)
        if actual != bindings[key]["sha256"]:
            raise RuntimeError(f"input hash mismatch for {key}: {actual}")
    anchors = pd.read_parquet(bindings["train_anchors"]["path"])
    raw_values = np.load(bindings["train_sequences"]["path"], mmap_mode="r")
    return anchors, raw_values


def zero_shot(config: dict[str, object], output: Path) -> dict[str, object]:
    anchors, raw_values = _load_train_bindings(config)
    pipeline = load_pipeline(config)
    frames: list[pd.DataFrame] = []
    runtime: dict[str, float] = {}
    for fold in build_forecast_folds(anchors):
        inputs = prepare_context_inputs(raw_values, fold.validation_ids)
        started = time.perf_counter()
        prediction = point_predictions(pipeline, inputs, batch_size=120)
        runtime[fold.name] = time.perf_counter() - started
        frames.append(
            prediction_frame(
                anchors, fold.validation_ids, prediction, fold=fold.name
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    incumbent = pd.read_parquet(config["bindings"]["frozen_incumbent_oof"])
    incumbent = incumbent.loc[
        incumbent["prefix_fraction"].eq(1.0),
        ["anchor_id", "lead_h", "incumbent_prediction"],
    ]
    frame = frame.merge(incumbent, on=["anchor_id", "lead_h"], validate="one_to_one")
    distance = rmse(frame["incumbent_prediction"], frame["prediction"])
    metrics = score_frame(frame)
    metrics["incumbent_rmse_m"] = rmse(frame["target_hs"], frame["incumbent_prediction"])
    metrics["delta_vs_incumbent_m"] = metrics["rmse_m"] - metrics["incumbent_rmse_m"]
    metrics["rms_distance_from_incumbent_m"] = distance
    metrics["runtime_seconds_by_fold"] = runtime
    frame.to_parquet(output / "zero_shot_oof.parquet", index=False)
    return metrics


def _p3_train_wave_path() -> Path:
    root = os.environ.get("P3_DATA_DIR")
    if not root:
        raise RuntimeError("P3_DATA_DIR must be set for train-only LoRA smoke")
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


def lora_smoke(config: dict[str, object], output: Path) -> dict[str, object]:
    anchors, raw_values = _load_train_bindings(config)
    validation = config["validation"]
    fold = build_forecast_folds(anchors)[0]
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
    fit_ids = _stratified_tail_ids(
        anchors, fit_ids, int(validation["lora_fit_max_cases_per_station"])
    )
    episodes, kept_ids = prepare_training_episodes(
        raw_values, anchors, fit_ids, _p3_train_wave_path()
    )
    if len(episodes) < 30 or len(calibration_ids) < 6:
        raise RuntimeError("insufficient train-only episodes for LoRA smoke")
    calibration_inputs = prepare_context_inputs(raw_values, calibration_ids)
    target_lookup = anchors.set_index("anchor_id")
    truth = np.column_stack(
        [
            target_lookup.loc[calibration_ids, f"target_{lead}"].to_numpy(dtype=float)
            for lead in (3, 6, 9, 12, 18, 24)
        ]
    )
    current = target_lookup.loc[calibration_ids, "current_hs"].to_numpy(dtype=float)
    baseline_pipeline = load_pipeline(config)
    zero_shot_prediction = point_predictions(
        baseline_pipeline, calibration_inputs, batch_size=120
    )
    zero_shot_rmse = rmse(truth, zero_shot_prediction)
    candidates: list[dict[str, object]] = []
    for steps in validation["lora_step_budgets"]:
        set_seed(int(validation["seed"]))
        pipeline = load_pipeline(config)
        checkpoint_root = output / "local_lora_smoke" / f"steps_{int(steps):04d}"
        started = time.perf_counter()
        tuned = pipeline.fit(
            episodes,
            prediction_length=PREDICTION_STEPS_20MIN,
            context_length=CONTEXT_STEPS_20MIN,
            min_past=CONTEXT_STEPS_20MIN,
            finetune_mode="lora",
            num_steps=int(steps),
            batch_size=int(validation["lora_batch_size_series"]),
            learning_rate=float(validation["lora_learning_rate"]),
            output_dir=checkpoint_root,
            save_strategy="no",
            eval_strategy="no",
        )
        fit_seconds = time.perf_counter() - started
        model_class = type(tuned.model).__name__
        if "Peft" not in model_class:
            raise RuntimeError(f"LoRA silently fell back to full fine-tuning: {model_class}")
        prediction = point_predictions(tuned, calibration_inputs, batch_size=120)
        checkpoint = checkpoint_root / "finetuned-ckpt"
        adapter_config = checkpoint / "adapter_config.json"
        adapter_weights = checkpoint / "adapter_model.safetensors"
        if not adapter_config.is_file() or not adapter_weights.is_file():
            raise RuntimeError("LoRA checkpoint is missing adapter config or weights")
        candidates.append(
            {
                "steps": int(steps),
                "validation_rmse_m": rmse(truth, prediction),
                "persistence_rmse_m": rmse(truth, np.repeat(current[:, None], 6, axis=1)),
                "fit_seconds": fit_seconds,
                "model_class": model_class,
                "checkpoint_dir": str(checkpoint.resolve()),
                "adapter_config_sha256": sha256_file(adapter_config),
                "adapter_weights_sha256": sha256_file(adapter_weights),
            }
        )
    best = min(candidates, key=lambda item: item["validation_rmse_m"])
    return {
        "status": "PASS",
        "fold": fold.name,
        "fit_cases_requested": int(len(fit_ids)),
        "fit_cases_complete_dense_future": int(len(kept_ids)),
        "inner_validation_cases": int(len(calibration_ids)),
        "inner_zero_shot_rmse_m": zero_shot_rmse,
        "inner_persistence_rmse_m": rmse(
            truth, np.repeat(current[:, None], 6, axis=1)
        ),
        "best_delta_vs_inner_zero_shot_m": best["validation_rmse_m"] - zero_shot_rmse,
        "candidates": candidates,
        "best_validation_rmse_checkpoint": best,
        "note": "Minimal first-fold feasibility probe only; not a full nested model or submission candidate.",
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output = Path(config["artifact_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        payload = preflight(config, output, args.gpu_smoke)
        destination = output / "environment_preflight.json"
    elif args.stage == "zero-shot":
        payload = zero_shot(config, output)
        destination = output / "zero_shot_metrics.json"
    else:
        payload = lora_smoke(config, output)
        destination = output / "local_lora_smoke_metrics.json"
    payload["experiment_id"] = config["experiment_id"]
    payload["config_sha256"] = sha256_file(config_path)
    payload["git_head"] = os.popen("git rev-parse HEAD").read().strip()
    payload["dirty_worktree"] = bool(os.popen("git status --porcelain").read().strip())
    write_json(destination, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
