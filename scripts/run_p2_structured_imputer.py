"""Train and evaluate the P2 target-aware long-gap structured imputer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.deep_data import build_panel
from p2_restore.profile_projection import project_profiles, public_endpoint_frame
from p2_restore.structured_imputer import (
    StructuredMaskConfig,
    build_hourly_panel,
    interpolate_hourly_prediction,
    time_mask,
    train_structured_model,
)
from p2_restore.submission import build_submission, validate_submission

BLOCKS = {
    "2024_sep_oct": ("2024-09-01", "2024-11-01"),
    "2025_jul_aug": ("2025-07-01", "2025-09-01"),
    "2025_nov_dec": ("2025-11-01", "2026-01-01"),
}
DEV_BLOCK = ("2024-07-01", "2024-09-01")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


class Progress:
    def __init__(self, path: Path, total_units: int) -> None:
        self.path = path
        self.total_units = total_units
        self.completed = 0.0
        self.started = time.perf_counter()

    def update(self, phase: str, detail: str, fraction: float = 0.0) -> None:
        progress = min(99.0, 100.0 * (self.completed + fraction) / self.total_units)
        elapsed = max(time.perf_counter() - self.started, 0.01)
        remaining = elapsed * max(100.0 - progress, 0.0) / max(progress, 0.5)
        eta = datetime.now().astimezone() + timedelta(seconds=remaining)
        _write_json(
            self.path,
            {
                "title": "P2 장기 구조 마스킹 복원기",
                "status": "running",
                "progress": progress,
                "phase": phase,
                "detail": detail,
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST"),
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )

    def finish_unit(self) -> None:
        self.completed += 1.0

    def complete(self, detail: str) -> None:
        _write_json(
            self.path,
            {
                "title": "P2 장기 구조 마스킹 복원기",
                "status": "complete",
                "progress": 100.0,
                "phase": "완료",
                "detail": detail,
                "eta": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S KST"),
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )


def _validate_contract(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "p2_structured_mask_imputer_v1":
        raise ValueError("unexpected structured-imputer experiment id")
    if value.get("status") != "authorized_local_score_optimization":
        raise ValueError("structured-imputer experiment is not locally authorized")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("structured imputer must remain local-only")
    contract = value["target_input_contract"]
    if contract.get("all_three_layers_masked_together") is not True:
        raise ValueError("simultaneous target-layer masking is mandatory")
    if contract.get("target_salinity") is not False or contract.get("external_values") is not False:
        raise ValueError("target salinity and external observations are forbidden")
    return value


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def _hourly_from_long(
    frame: pd.DataFrame, column: str
) -> dict[int, tuple[pd.DatetimeIndex, np.ndarray]]:
    result: dict[int, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    keyed = frame.assign(
        time=pd.to_datetime(frame["time"], utc=True), _hour=lambda x: x.time.dt.floor("h")
    )
    for layer in (2, 3, 4):
        current = keyed.loc[keyed["layer"] == layer].groupby("_hour", sort=True)[column].median()
        result[layer] = (pd.DatetimeIndex(current.index), current.to_numpy(float))
    return result


def _slow_correction(
    frame: pd.DataFrame,
    hourly_times: pd.DatetimeIndex,
    hourly_prediction: np.ndarray,
    base_column: str,
) -> np.ndarray:
    base_hourly = _hourly_from_long(frame, base_column)
    correction = np.empty(len(frame), dtype=np.float64)
    row_time = pd.to_datetime(frame["time"], utc=True)
    for offset, layer in enumerate((2, 3, 4)):
        selected = frame["layer"].to_numpy(int) == layer
        target_times = row_time[selected]
        finite = np.isfinite(hourly_prediction[:, offset])
        if finite.sum() < 2:
            raise RuntimeError(f"structured hourly layer {layer} has insufficient coverage")
        imputed = interpolate_hourly_prediction(
            hourly_times[finite],
            hourly_prediction[finite, offset : offset + 1].repeat(3, axis=1),
            target_times,
        )[:, 0]
        source_time, source_value = base_hourly[layer]
        base_slow = np.interp(
            pd.DatetimeIndex(target_times).as_unit("ns").asi8.astype(float),
            source_time.as_unit("ns").asi8.astype(float),
            source_value,
        )
        correction[selected] = imputed - base_slow
    if not np.isfinite(correction).all():
        raise RuntimeError("structured slow correction is not finite")
    return correction


def _select_alpha(
    frame: pd.DataFrame,
    base: np.ndarray,
    correction: np.ndarray,
    endpoints: pd.DataFrame,
    train_mask: np.ndarray,
    grid: list[float],
) -> tuple[float, float]:
    truth = frame["truth"].to_numpy(float)
    best = (float("inf"), 0.0)
    for alpha in grid:
        projected = project_profiles(frame, base + alpha * correction, endpoints).prediction
        score = _rmse(truth[train_mask], projected[train_mask])
        candidate = (score, float(alpha))
        if candidate < best:
            best = candidate
    return best[1], best[0]


def _metrics(frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    truth = frame["truth"].to_numpy(float)

    def cut(mask: np.ndarray) -> dict[str, float | int]:
        baseline = _rmse(truth[mask], base[mask])
        current = _rmse(truth[mask], candidate[mask])
        return {
            "rows": int(mask.sum()),
            "baseline_rmse": baseline,
            "candidate_rmse": current,
            "delta_rmse": current - baseline,
        }

    return {
        **cut(np.ones(len(frame), dtype=bool)),
        "by_block": {
            str(block): cut(frame["block"].eq(block).to_numpy())
            for block in frame["block"].unique()
        },
        "by_layer": {str(layer): cut(frame["layer"].eq(layer).to_numpy()) for layer in (2, 3, 4)},
    }


def _bootstrap(
    frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray, replicates: int = 2000
) -> dict[str, object]:
    truth = frame["truth"].to_numpy(float)
    days = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    unique = np.unique(days)
    blocks = [np.flatnonzero(days == day) for day in unique]
    rng = np.random.default_rng(20260816)
    delta = np.empty(replicates)
    for number in range(replicates):
        rows = np.concatenate(
            [blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]
        )
        delta[number] = _rmse(truth[rows], candidate[rows]) - _rmse(truth[rows], base[rows])
    return {
        "replicates": replicates,
        "kst_days": len(unique),
        "delta_rmse": _rmse(truth, candidate) - _rmse(truth, base),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_structured_mask_imputer_v1.json"),
    )
    parser.add_argument(
        "--base-oof",
        type=Path,
        default=Path("artifacts/p2_physical_profile_projection_v1/oof.parquet"),
    )
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=Path("submissions/p2/P2_PHYSICAL_PROFILE_PROJECTION_V1.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_structured_mask_imputer_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/p2_structured_mask_imputer.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    contract = _validate_contract(args.preregistration)
    progress = Progress(args.status_file, total_units=9)
    progress.update("데이터", "10분 관측을 누출 없는 시간 패널로 집계", 0.1)
    data = load_p2_data(resolve_data_dir(args.data_dir))
    panel = build_hourly_panel(build_panel(data.observations))
    endpoints = public_endpoint_frame(data.observations)
    progress.finish_unit()

    model_contract = contract["model"]
    base_config = StructuredMaskConfig(
        window_hours=int(model_contract["window_hours"]),
        context_hours=int(model_contract["context_hours"]),
        mask_hours=tuple(int(value) for value in model_contract["structured_mask_hours"]),
        hidden=int(model_contract["hidden"]),
        blocks=int(model_contract["blocks"]),
        dropout=float(model_contract["dropout"]),
        samples_per_epoch=int(model_contract["samples_per_epoch"]),
        weight_decay=float(model_contract["weight_decay"]),
        max_epochs=int(model_contract["max_epochs"]),
        patience=int(model_contract["patience"]),
        seed=int(model_contract["seed"]),
    )
    dev = time_mask(panel, *DEV_BLOCK)
    screens: list[dict[str, object]] = []
    selected_result = None
    for number, learning_rate in enumerate(model_contract["learning_rates"]):
        config = replace(
            base_config, learning_rate=float(learning_rate), seed=base_config.seed + number
        )

        def screen_callback(
            state: dict[str, object],
            lr: float = float(learning_rate),
            limit: int = config.max_epochs,
        ) -> None:
            score = state.get("validation_rmse")
            detail = f"lr={lr:g} · epoch {state['epoch']}/{limit}"
            if score is not None:
                detail += f" · dev RMSE {float(score):.5f}"
            progress.update("개발 블록 탐색", detail, float(state["epoch"]) / limit)

        result = train_structured_model(
            panel,
            train_hours=~dev,
            config=config,
            evaluation_block=dev,
            select_best=True,
            progress=screen_callback,
        )
        screens.append(
            {
                "learning_rate": float(learning_rate),
                "best_epoch": result.best_epoch,
                "best_rmse": result.best_rmse,
                "history": result.history,
            }
        )
        if selected_result is None or float(result.best_rmse) < float(selected_result.best_rmse):
            selected_result = result
        progress.finish_unit()
        gc.collect()
        torch.cuda.empty_cache()
    assert selected_result is not None
    selected_lr = selected_result.config.learning_rate
    selected_epochs = selected_result.best_epoch

    block_predictions: dict[str, np.ndarray] = {}
    fold_summaries: dict[str, object] = {}
    for number, (name, dates) in enumerate(BLOCKS.items()):
        held = time_mask(panel, *dates)
        config = replace(
            base_config,
            learning_rate=selected_lr,
            max_epochs=selected_epochs,
            patience=selected_epochs,
            seed=base_config.seed + 100 + number,
        )

        def fold_callback(state: dict[str, object], block_name: str = name) -> None:
            progress.update(
                "고정 outer 학습",
                f"{block_name} · epoch {state['epoch']}/{selected_epochs}",
                float(state["epoch"]) / selected_epochs,
            )

        result = train_structured_model(
            panel,
            train_hours=~held,
            config=config,
            evaluation_block=held,
            select_best=False,
            progress=fold_callback,
        )
        block_predictions[name] = result.hourly_prediction
        valid = held[:, None] & panel.target_mask & np.isfinite(result.hourly_prediction)
        fold_summaries[name] = {
            "epochs": selected_epochs,
            "standalone_hourly_rmse": _rmse(panel.target[valid], result.hourly_prediction[valid]),
            "history": result.history,
        }
        progress.finish_unit()
        gc.collect()
        torch.cuda.empty_cache()

    base_oof = pd.read_parquet(args.base_oof)
    required = {"time", "layer", "truth", "block", "prediction"}
    if missing := required.difference(base_oof.columns):
        raise ValueError(f"base OOF is missing columns: {sorted(missing)}")
    if len(base_oof) != 69_850 or base_oof.duplicated(["time", "layer"]).any():
        raise ValueError("base OOF grain changed")
    base = base_oof["prediction"].to_numpy(float)
    correction = np.empty(len(base_oof), dtype=float)
    for name, prediction in block_predictions.items():
        selected = base_oof["block"].eq(name).to_numpy()
        correction[selected] = _slow_correction(
            base_oof.loc[selected], panel.times, prediction, "prediction"
        )
    grid = [float(value) for value in contract["blend"]["alpha_grid"]]
    lobo = np.empty(len(base_oof), dtype=float)
    alphas: dict[str, float] = {}
    for held in BLOCKS:
        train = ~base_oof["block"].eq(held).to_numpy()
        test = ~train
        alpha, _ = _select_alpha(base_oof, base, correction, endpoints, train, grid)
        alphas[held] = alpha
        projected = project_profiles(base_oof, base + alpha * correction, endpoints).prediction
        lobo[test] = projected[test]
    all_alpha, all_fit_rmse = _select_alpha(
        base_oof, base, correction, endpoints, np.ones(len(base_oof), dtype=bool), grid
    )
    fitted = project_profiles(base_oof, base + all_alpha * correction, endpoints).prediction
    metrics = _metrics(base_oof, base, lobo)
    fitted_metrics = _metrics(base_oof, base, fitted)
    bootstrap = _bootstrap(base_oof, base, lobo)
    progress.finish_unit()

    hidden = time_mask(panel, "2025-09-01", "2025-11-01")
    full_config = replace(
        base_config,
        learning_rate=selected_lr,
        max_epochs=selected_epochs,
        patience=selected_epochs,
        seed=base_config.seed + 999,
    )

    def full_callback(state: dict[str, object]) -> None:
        progress.update(
            "전체 학습",
            f"hidden 61일 복원 · epoch {state['epoch']}/{selected_epochs}",
            float(state["epoch"]) / selected_epochs,
        )

    full = train_structured_model(
        panel,
        train_hours=~hidden,
        config=full_config,
        evaluation_block=hidden,
        select_best=False,
        progress=full_callback,
    )
    progress.finish_unit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "full_model.pt"
    torch.save(
        {
            "config": asdict(full.config),
            "epochs": full.epochs,
            "normalizer": asdict(full.normalizer),
            "state_dict": full.state_dict,
        },
        checkpoint,
    )

    incumbent = pd.read_csv(args.base_submission)
    if len(incumbent) != 26_061 or not all(
        np.array_equal(incumbent[column].astype(str), data.test_index[column].astype(str))
        for column in ("station", "layer", "time")
    ):
        raise ValueError("base submission keys changed")
    test_frame = incumbent.copy()
    test_frame["prediction"] = incumbent["temp"].to_numpy(float)
    test_correction = _slow_correction(
        test_frame, panel.times, full.hourly_prediction, "prediction"
    )
    test_prediction = project_profiles(
        test_frame, incumbent["temp"].to_numpy(float) + all_alpha * test_correction, endpoints
    ).prediction
    output = Path(contract["submission"]["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    build_submission(data.test_index, test_prediction).to_csv(
        output, index=False, encoding="utf-8", lineterminator="\n"
    )
    validation = validate_submission(output, data.test_index)

    oof_path = args.output_dir / "oof.parquet"
    saved = base_oof[["time", "layer", "truth", "block", "prediction"]].rename(
        columns={"prediction": "baseline"}
    )
    saved["correction"] = correction
    saved["prediction"] = lobo
    saved["fitted_prediction"] = fitted
    saved.to_parquet(oof_path, index=False, compression="zstd")
    decision = (
        "PROMOTE_STRUCTURED_MASK_CHALLENGER_NO_UPLOAD"
        if metrics["delta_rmse"] < 0 and bootstrap["ci90_high"] < 0
        else "REJECT_STRUCTURED_MASK_KEEP_PHYSICAL_PROJECTION_NO_UPLOAD"
    )
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "adaptive_after_prior_outer_exposure": True,
        "fresh_holdout_claimed": False,
        "external_values_used": False,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "development_screens": screens,
        "selected_learning_rate": selected_lr,
        "selected_epochs": selected_epochs,
        "fold_training": fold_summaries,
        "lobo_alphas": alphas,
        "all_oof_alpha_for_hidden": all_alpha,
        "all_oof_fit_rmse": all_fit_rmse,
        "metrics": metrics,
        "fitted_metrics": fitted_metrics,
        "paired_kst_day_bootstrap": bootstrap,
        "decision": decision,
        "artifacts": {
            "oof": {"path": oof_path.as_posix(), "sha256": _sha256(oof_path)},
            "checkpoint": {"path": checkpoint.as_posix(), "sha256": _sha256(checkpoint)},
            "submission": {"path": output.as_posix(), "sha256": _sha256(output), **validation},
        },
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "preregistration_sha256": _sha256(args.preregistration),
            "base_oof_sha256": _sha256(args.base_oof),
            "base_submission_sha256": _sha256(args.base_submission),
            "result_sha256": _sha256(result_path),
            "oof_sha256": _sha256(oof_path),
            "checkpoint_sha256": _sha256(checkpoint),
            "submission_sha256": _sha256(output),
            "uploaded": False,
        },
    )
    progress.complete(f"{decision} · LOBO ΔRMSE {metrics['delta_rmse']:+.6f}℃ · 업로드 없음")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
