"""Multi-seed finalist evaluation, full fit, and local P2 submission freezing."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.deep_data import build_panel
from p2_restore.deep_models import MODEL_SPECS
from p2_restore.deep_training import TrainingConfig, save_checkpoint, train_fold, train_full_model
from p2_restore.model import VALIDATION_BLOCKS
from p2_restore.submission import build_submission, validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(self, path: Path, jobs: int) -> None:
        self.path = path
        self.jobs = jobs
        self.done = 0
        self.started = time.perf_counter()

    def update(
        self,
        phase: str,
        detail: str,
        fraction: float = 0.0,
        *,
        status: str = "running",
    ) -> None:
        position = self.done + min(max(fraction, 0.0), 1.0)
        progress = 5 + 92 * position / max(self.jobs, 1)
        elapsed = time.perf_counter() - self.started
        eta_seconds = elapsed * (self.jobs - position) / position if position > 0 else 0
        eta = datetime.now().astimezone() + timedelta(seconds=max(eta_seconds, 0))
        _write_json(
            self.path,
            {
                "title": "P2 결선 모델·제출 동결",
                "status": status,
                "progress": 100.0 if status == "complete" else progress,
                "phase": phase,
                "detail": detail,
                "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST") if position else "속도 측정 중",
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )

    def finish(self) -> None:
        self.done += 1


def _validate_contract(path: Path, tournament: dict[str, object]) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "p2_deep_finalists_v1":
        raise ValueError("unexpected P2 finalist experiment id")
    if value.get("status") != "authorized_local_finalist_run":
        raise ValueError("P2 finalist experiment is not authorized")
    if value.get("upload_allowed") is not False or value.get("research_only") is not True:
        raise ValueError("P2 finalist experiment must remain local-only")
    if value.get("finalists") != ["lsti_style", "timemixerpp_style"]:
        raise ValueError("P2 finalist list changed")
    if value.get("seeds") != [20260816, 20260817, 20260818]:
        raise ValueError("P2 finalist seeds changed")
    ranked = [row["model"] for row in tournament["ranking"][:2]]
    if set(ranked) != set(value["finalists"]):
        raise ValueError(f"contract finalists are not the tournament top two: {ranked}")
    return value


def _aligned_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    keys = ["time", "layer", "block"]
    base = frames[0].copy()
    base["time"] = pd.to_datetime(base["time"], utc=True)
    base = base.sort_values(keys).reset_index(drop=True)
    values = [base["prediction"].to_numpy(float)]
    for frame in frames[1:]:
        current = frame.copy()
        current["time"] = pd.to_datetime(current["time"], utc=True)
        current = current.sort_values(keys).reset_index(drop=True)
        if not base[[*keys, "truth"]].equals(current[[*keys, "truth"]]):
            raise ValueError("multi-seed OOF grain or truth differs")
        values.append(current["prediction"].to_numpy(float))
    base["prediction"] = np.mean(values, axis=0)
    return base


def _fit_weights(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, float]:
    inputs = frame[columns].to_numpy(float)
    truth = frame["truth"].to_numpy(float)

    def objective(weights: np.ndarray) -> float:
        return float(np.mean((inputs @ weights - truth) ** 2))

    result = minimize(
        objective,
        np.full(len(columns), 1.0 / len(columns)),
        bounds=[(0.0, 1.0)] * len(columns),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success or not np.isclose(result.x.sum(), 1.0, atol=1e-7):
        raise RuntimeError(f"convex stack optimization failed: {result.message}")
    return result.x, float(objective(result.x) ** 0.5)


def _stack_predictions(
    frame: pd.DataFrame, columns: list[str], weights: dict[str, dict[str, float]]
) -> np.ndarray:
    prediction = np.empty(len(frame), dtype=np.float64)
    for layer in (2, 3, 4):
        selected = frame["layer"].to_numpy(int) == layer
        vector = np.array([weights[str(layer)][column] for column in columns])
        prediction[selected] = frame.loc[selected, columns].to_numpy(float) @ vector
    return prediction


def _paired_day_bootstrap(
    frame: pd.DataFrame, candidate: np.ndarray, *, replicates: int = 2000
) -> dict[str, object]:
    time = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul")
    day = time.dt.strftime("%Y-%m-%d").to_numpy()
    unique = np.unique(day)
    blocks = [np.flatnonzero(day == value) for value in unique]
    truth = frame["truth"].to_numpy(float)
    incumbent = frame["router_400"].to_numpy(float)
    rng = np.random.default_rng(20260816)
    deltas = np.empty(replicates)
    for number in range(replicates):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])
        candidate_rmse = np.mean((candidate[indices] - truth[indices]) ** 2) ** 0.5
        incumbent_rmse = np.mean((incumbent[indices] - truth[indices]) ** 2) ** 0.5
        deltas[number] = candidate_rmse - incumbent_rmse
    return {
        "replicates": replicates,
        "kst_days": len(unique),
        "delta_rmse": float(
            np.mean((candidate - truth) ** 2) ** 0.5 - np.mean((incumbent - truth) ** 2) ** 0.5
        ),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0)),
    }


def _panel_test_prediction(
    data_index: pd.DataFrame, panel_times: pd.DatetimeIndex, prediction: np.ndarray
) -> np.ndarray:
    time = pd.to_datetime(data_index["time"], utc=True)
    positions = panel_times.get_indexer(time)
    if (positions < 0).any():
        raise ValueError("test_index time is absent from the deep panel")
    layers = data_index["layer"].to_numpy(int) - 2
    return prediction[positions, layers]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("configs/experiments/p2_deep_finalists_v1.json"),
    )
    parser.add_argument(
        "--tournament-result",
        type=Path,
        default=Path("artifacts/p2_deep_model_tournament_v1/result.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/p2_deep_finalists_v1"))
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/p2_model_tournament.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    tournament = json.loads(args.tournament_result.read_text(encoding="utf-8"))
    contract = _validate_contract(args.preregistration, tournament)
    data_dir = resolve_data_dir(args.data_dir)
    data = load_p2_data(data_dir)
    panel = build_panel(data.observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = {spec.name: spec for spec in MODEL_SPECS}
    extra_seeds = contract["seeds"][1:]
    jobs = len(contract["finalists"]) * len(extra_seeds) * 3 + 8
    progress = Progress(args.status_file, jobs)

    finalist_oof: dict[str, pd.DataFrame] = {}
    epoch_by_model_seed: dict[str, dict[str, list[int]]] = {}
    for model_name in contract["finalists"]:
        model_root = args.output_dir / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        seed_frames = [
            pd.read_parquet(
                Path("artifacts/p2_deep_model_tournament_v1/models") / model_name / "oof.parquet"
            )
        ]
        original_epochs = list(tournament["models"][model_name]["best_epochs"].values())
        epoch_by_model_seed[model_name] = {str(contract["seeds"][0]): original_epochs}
        spec = specs[model_name]
        learning_rate = float(tournament["selected_learning_rates"][model_name])
        for seed in extra_seeds:
            fold_frames: list[pd.DataFrame] = []
            fold_epochs: list[int] = []
            for block_number, (block, (start, stop)) in enumerate(VALIDATION_BLOCKS.items()):
                summary_path = model_root / f"seed{seed}_{block}.json"
                oof_path = model_root / f"seed{seed}_{block}_oof.parquet"
                if summary_path.exists() and oof_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    fold_frames.append(pd.read_parquet(oof_path))
                    fold_epochs.append(int(summary["best_epoch"]))
                    progress.finish()
                    progress.update(
                        "multi-seed OOF", f"재사용 · {model_name} · seed {seed} · {block}"
                    )
                    continue
                config = TrainingConfig(
                    model=model_name,
                    learning_rate=learning_rate,
                    weight_decay=spec.weight_decay,
                    max_epochs=spec.max_epochs,
                    patience=spec.patience,
                    seed=seed + block_number,
                    evaluation_interval=2,
                    diffusion_samples=1,
                )

                def callback(
                    state: dict[str, object],
                    name: str = model_name,
                    current_seed: int = seed,
                    current_block: str = block,
                    limit: int = spec.max_epochs,
                ) -> None:
                    progress.update(
                        "multi-seed OOF",
                        f"{name} · seed {current_seed} · {current_block} · epoch {state['epoch']}/{limit} · RMSE {float(state['rmse']):.5f}",
                        int(state["epoch"]) / limit,
                    )

                result = train_fold(
                    panel,
                    block=block,
                    start=start,
                    stop=stop,
                    config=config,
                    progress=callback,
                )
                summary = result.summary()
                summary["checkpoint_sha256"] = save_checkpoint(
                    model_root / f"seed{seed}_{block}.pt", result
                )
                result.oof.to_parquet(oof_path, index=False, compression="zstd")
                _write_json(summary_path, summary)
                fold_frames.append(result.oof)
                fold_epochs.append(result.best_epoch)
                progress.finish()
                del result
                gc.collect()
                torch.cuda.empty_cache()
            seed_oof = pd.concat(fold_frames, ignore_index=True)
            seed_oof.to_parquet(
                model_root / f"seed{seed}_oof.parquet", index=False, compression="zstd"
            )
            seed_frames.append(seed_oof)
            epoch_by_model_seed[model_name][str(seed)] = fold_epochs
        finalist_oof[model_name] = _aligned_mean(seed_frames)
        finalist_oof[model_name].to_parquet(
            model_root / "ensemble_oof.parquet", index=False, compression="zstd"
        )

    incumbent = pd.read_parquet("artifacts/p2_max_round_convergence_v1/oof.parquet")[
        ["time", "layer", "truth", "block", "router_400"]
    ].copy()
    incumbent["time"] = pd.to_datetime(incumbent["time"], utc=True)
    stack = incumbent
    model_oof: dict[str, pd.DataFrame] = {
        "depth_query_bitcn": pd.read_parquet(
            "artifacts/p2_deep_model_tournament_v1/models/depth_query_bitcn/oof.parquet"
        ),
        "moment_units_scratch": pd.read_parquet(
            "artifacts/p2_deep_model_tournament_v1/models/moment_units_scratch/oof.parquet"
        ),
        **finalist_oof,
    }
    for name, frame in model_oof.items():
        current = frame[["time", "layer", "block", "prediction"]].copy()
        current["time"] = pd.to_datetime(current["time"], utc=True)
        stack = stack.merge(
            current.rename(columns={"prediction": name}),
            on=["time", "layer", "block"],
            validate="one_to_one",
        )
    columns = contract["stacking"]["contributors"]
    weights: dict[str, dict[str, float]] = {}
    layer_rmse: dict[str, float] = {}
    for layer in (2, 3, 4):
        selected = stack["layer"] == layer
        fitted, score = _fit_weights(stack.loc[selected], columns)
        weights[str(layer)] = {
            column: float(weight) for column, weight in zip(columns, fitted, strict=True)
        }
        layer_rmse[str(layer)] = score
    stacked_oof = _stack_predictions(stack, columns, weights)
    truth = stack["truth"].to_numpy(float)
    incumbent_prediction = stack["router_400"].to_numpy(float)
    stacked_rmse = float(np.mean((stacked_oof - truth) ** 2) ** 0.5)
    incumbent_rmse = float(np.mean((incumbent_prediction - truth) ** 2) ** 0.5)

    lobo_prediction = np.empty(len(stack))
    lobo_weights: dict[str, object] = {}
    for held in VALIDATION_BLOCKS:
        lobo_weights[held] = {}
        for layer in (2, 3, 4):
            train = (stack["block"] != held) & (stack["layer"] == layer)
            test = (stack["block"] == held) & (stack["layer"] == layer)
            fitted, _ = _fit_weights(stack.loc[train], columns)
            lobo_prediction[test] = stack.loc[test, columns].to_numpy(float) @ fitted
            lobo_weights[held][str(layer)] = {
                column: float(weight) for column, weight in zip(columns, fitted, strict=True)
            }
    lobo_rmse = float(np.mean((lobo_prediction - truth) ** 2) ** 0.5)
    bootstrap = _paired_day_bootstrap(stack, stacked_oof)

    full_predictions: dict[str, np.ndarray] = {}
    full_manifests: dict[str, object] = {}
    for model_name in contract["single_seed_contributors"] + contract["finalists"]:
        spec = specs[model_name]
        seeds = contract["seeds"] if model_name in contract["finalists"] else [contract["seeds"][0]]
        model_predictions: list[np.ndarray] = []
        model_entries: list[dict[str, object]] = []
        for seed in seeds:
            seed_epochs = epoch_by_model_seed.get(model_name, {}).get(str(seed), [])
            epochs = (
                int(np.median(seed_epochs))
                if seed_epochs
                else int(np.median(list(tournament["models"][model_name]["best_epochs"].values())))
            )
            config = TrainingConfig(
                model=model_name,
                learning_rate=float(tournament["selected_learning_rates"][model_name]),
                weight_decay=spec.weight_decay,
                max_epochs=epochs,
                patience=epochs,
                seed=seed,
                evaluation_interval=max(1, epochs),
                diffusion_samples=1,
            )

            def full_callback(
                state: dict[str, object],
                name: str = model_name,
                current_seed: int = seed,
                limit: int = epochs,
            ) -> None:
                progress.update(
                    "full fit",
                    f"{name} · seed {current_seed} · epoch {state['epoch']}/{limit} · train MSE {float(state['train_mse_c']):.5f}",
                    int(state["epoch"]) / limit,
                )

            trained = train_full_model(panel, config, epochs=epochs, progress=full_callback)
            model_predictions.append(
                _panel_test_prediction(data.test_index, panel.times, trained.prediction)
            )
            checkpoint_path = args.output_dir / "full" / f"{model_name}_seed{seed}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model_name,
                    "config": asdict(config),
                    "epochs": epochs,
                    "input_center": trained.normalizer.input_center,
                    "input_scale": trained.normalizer.input_scale,
                    "residual_center": trained.normalizer.residual_center,
                    "residual_scale": trained.normalizer.residual_scale,
                    "state_dict": trained.state_dict,
                },
                checkpoint_path,
            )
            model_entries.append(
                {
                    "seed": seed,
                    "epochs": epochs,
                    "checkpoint": checkpoint_path.as_posix(),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "final_train_mse_c": trained.final_train_mse_c,
                }
            )
            progress.finish()
            del trained
            gc.collect()
            torch.cuda.empty_cache()
        full_predictions[model_name] = np.mean(model_predictions, axis=0)
        full_manifests[model_name] = model_entries

    incumbent_file = Path("submissions/p2/P2_SCORE_ROUTER_ROUND400.csv")
    if (
        _sha256(incumbent_file)
        != "069b782588ccad2a1c74d68586769268b104d686f9dc443f8a8ba136afb192b5"
    ):
        raise ValueError("frozen P2 incumbent submission hash changed")
    test_stack = pd.DataFrame(
        {
            "layer": data.test_index["layer"].to_numpy(int),
            "router_400": pd.read_csv(incumbent_file)["temp"].to_numpy(float),
            **full_predictions,
        }
    )
    test_prediction = _stack_predictions(test_stack, columns, weights)
    submission_path = Path(contract["submission"]["path"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    build_submission(data.test_index, test_prediction).to_csv(
        submission_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    submission_validation = validate_submission(submission_path, data.test_index)

    stacked_frame = stack[["time", "layer", "truth", "block", "router_400"]].copy()
    stacked_frame["prediction"] = stacked_oof
    stacked_frame["lobo_prediction"] = lobo_prediction
    stacked_oof_path = args.output_dir / "stacked_oof.parquet"
    stacked_frame.to_parquet(stacked_oof_path, index=False, compression="zstd")
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": contract["experiment_id"],
        "research_only": True,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "tournament_result_sha256": _sha256(args.tournament_result),
        "preregistration_sha256": _sha256(args.preregistration),
        "incumbent_rmse": incumbent_rmse,
        "stacked_oof_rmse": stacked_rmse,
        "lobo_stacked_rmse": lobo_rmse,
        "delta_vs_incumbent": stacked_rmse - incumbent_rmse,
        "weights_by_layer": weights,
        "layer_rmse": layer_rmse,
        "lobo_weights": lobo_weights,
        "bootstrap": bootstrap,
        "epoch_by_model_seed": epoch_by_model_seed,
        "full_models": full_manifests,
        "stacked_oof_path": stacked_oof_path.as_posix(),
        "stacked_oof_sha256": _sha256(stacked_oof_path),
        "submission": {
            "path": submission_path.as_posix(),
            "sha256": _sha256(submission_path),
            **submission_validation,
        },
        "decision": "FREEZE_DEEP_STACK_CANDIDATE_NO_UPLOAD",
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "result_sha256": _sha256(result_path),
            "stacked_oof_sha256": _sha256(stacked_oof_path),
            "submission_sha256": _sha256(submission_path),
            "uploaded": False,
        },
    )
    progress.update(
        "complete",
        f"완료 · OOF RMSE {stacked_rmse:.6f} · LOBO {lobo_rmse:.6f} · 제출 후보 검증 PASS · 업로드 없음",
        1.0,
        status="complete",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
