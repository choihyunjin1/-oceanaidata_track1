"""Run the preregistered P3 event-balanced dense-trajectory one-shot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p3_trajectory import (
    ClosedFormTrajectoryRegressor,
    attach_official_targets,
    build_blind_prediction_frame,
    build_trajectory_dataset,
    event_balanced_weights,
    metric_slices,
    paired_event_bootstrap,
    select_lattice_phase,
)
from p3_wave.data import audit_p3_data, load_p3_data
from p3_wave.validation import build_forecast_folds

DEFAULT_CONFIG = "configs/experiments/p3_event_balanced_trajectory_v1.json"
DEFAULT_STATUS = "artifacts/status/p3_event_balanced_trajectory_v1.json"
PAIR_KEYS = ["fold", "anchor_id", "station", "lead_h"]


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _status(
    path: Path,
    *,
    state: str,
    phase: str,
    progress: float,
    detail: str,
    eta: str,
) -> None:
    _atomic_json(
        path,
        {
            "title": "P3 event-balanced 72-step trajectory one-shot",
            "status": state,
            "phase": phase,
            "progress": float(progress),
            "detail": detail,
            "eta": eta,
            "updated_at": _now(),
        },
    )


def _git_state(root: Path) -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"sha": sha, "dirty": bool(status), "changed_path_count": len(status)}


def _frozen_hashes(root: Path, config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative in config["frozen_inputs"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen input is missing: {relative}")
        result[name] = _sha256(path)
    return result


def _compatibility_check(root: Path, dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
    cache_path = root / config["frozen_inputs"]["anchor_compatibility_cache"]
    cache = pd.read_parquet(cache_path, columns=["anchor_id", "station", "anchor_time"])
    cache["anchor_time"] = pd.to_datetime(cache["anchor_time"], utc=True)
    current = dataset.anchors[["anchor_id", "station", "anchor_time"]].copy()
    current["anchor_time"] = pd.to_datetime(current["anchor_time"], utc=True)
    exact = cache.reset_index(drop=True).equals(current.reset_index(drop=True))
    if not exact:
        raise ValueError("trajectory anchor ids/times do not match the frozen all20 cache")
    return {"rows": int(len(cache)), "anchor_id_station_time_exact": True}


def _episode_summary(dataset: Any) -> dict[str, Any]:
    complete = dataset.anchors.loc[dataset.complete_path].copy()
    event_size = complete.groupby(["station", "episode_id"], observed=True).size()
    return {
        "official_anchors": int(len(dataset.anchors)),
        "complete_path_anchors": int(dataset.complete_path.sum()),
        "complete_path_fraction": float(dataset.complete_path.mean()),
        "episodes": int(len(event_size)),
        "event_anchor_count_quantiles": {
            "p50": float(event_size.quantile(0.50)),
            "p90": float(event_size.quantile(0.90)),
            "p99": float(event_size.quantile(0.99)),
            "max": int(event_size.max()),
        },
        "complete_by_station": {
            str(station): int(count)
            for station, count in complete.groupby("station", observed=True).size().items()
        },
    }


def _load_and_audit(data_dir: str | Path | None) -> tuple[Any, Any, dict[str, Any]]:
    data = load_p3_data(data_dir)
    public_audit = audit_p3_data(data)
    trajectory = build_trajectory_dataset(data.wave)
    return data, trajectory, public_audit


def _run_smoke(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    output: Path,
    status_path: Path,
    data_dir: str | Path | None,
) -> dict[str, Any]:
    _status(
        status_path,
        state="running",
        phase="smoke_data_audit",
        progress=10,
        detail="원본 구조·20분 anchor·frozen 호환성을 검사 중",
        eta="약 2~5분",
    )
    data, dataset, public_audit = _load_and_audit(data_dir)
    compatibility = _compatibility_check(root, dataset, config)
    frozen = _frozen_hashes(root, config)
    source_root = Path(data_dir).resolve() if data_dir else None
    if source_root is None:
        raise ValueError("--data-dir or P3_DATA_DIR must be explicit for the one-shot")
    payload = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "mode": "smoke_only_no_validation_labels_opened",
        "config_sha256": _sha256(config_path),
        "input": {
            "train_wave_filename": "train_wave.csv",
            "train_wave_sha256": _sha256(source_root / "train_wave.csv"),
        },
        "public_audit": public_audit,
        "trajectory_audit": _episode_summary(dataset),
        "compatibility": compatibility,
        "frozen_sha256": frozen,
        "invariants": {
            "source_rows_mutated": 0,
            "external_observations_used": 0,
            "test_context_used_for_training": False,
            "validation_targets_opened": False,
            "hyperparameter_search_run": False,
            "submission_written_or_uploaded": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    smoke_path = output / "smoke.json"
    _atomic_json(smoke_path, payload)
    _status(
        status_path,
        state="smoke_passed",
        phase="ready_for_one_full_evaluation",
        progress=20,
        detail=(
            f"smoke PASS · complete path {payload['trajectory_audit']['complete_path_anchors']:,}/"
            f"{payload['trajectory_audit']['official_anchors']:,}"
        ),
        eta="full 평가 약 5~15분",
    )
    return payload


def _model_from_config(variant: str, config: dict[str, Any]) -> ClosedFormTrajectoryRegressor:
    return ClosedFormTrajectoryRegressor(
        variant=variant,
        alpha=float(config["models"]["ridge_alpha"]),
        trend_window=int(config["models"]["dlinear_trend_window_steps"]),
        auxiliary_weight=float(config["loss"]["auxiliary_path_step_weight"]),
    )


def _metric_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    candidate = metric_slices(frame, "prediction")
    incumbent = metric_slices(frame, "incumbent_prediction")
    return {
        "candidate": candidate,
        "incumbent": incumbent,
        "delta_rmse": float(candidate["rmse"] - incumbent["rmse"]),
    }


def _gate(
    paired: pd.DataFrame,
    bootstrap: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["gate"]
    comparison = _metric_comparison(paired)
    candidate = comparison["candidate"]
    incumbent = comparison["incumbent"]
    threshold_check = candidate["rmse"] <= float(specification["maximum_candidate_rmse"])
    bootstrap_check = float(bootstrap["ci90"][1]) < 0.0
    lead_checks = {
        str(lead): candidate["by_lead"][str(lead)] <= incumbent["by_lead"][str(lead)]
        for lead in (18, 24)
    }
    station_limit = float(specification["maximum_station_rmse_degradation"])
    station_checks = {
        station: candidate["by_station"][station]
        <= incumbent["by_station"][station] + station_limit
        for station in candidate["by_station"]
    }
    checks = {
        "candidate_rmse_at_most_0p7701609198910191": bool(threshold_check),
        "paired_event_bootstrap_ci90_upper_below_zero": bool(bootstrap_check),
        "lead_18_non_degrading": bool(lead_checks["18"]),
        "lead_24_non_degrading": bool(lead_checks["24"]),
        "all_station_degradation_at_most_0p010": bool(all(station_checks.values())),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "lead_checks": lead_checks,
        "station_checks": station_checks,
        "comparison": comparison,
    }


def _run_full(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    output: Path,
    status_path: Path,
    data_dir: str | Path | None,
) -> dict[str, Any]:
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError("one-shot full metrics already exist; refusing a second full evaluation")
    smoke_path = output / "smoke.json"
    if not smoke_path.is_file():
        raise FileNotFoundError("run --mode smoke successfully before the one-shot full evaluation")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("config_sha256") != _sha256(config_path):
        raise ValueError("config changed after smoke; one-shot is not authorized")

    start_time = time.perf_counter()
    frozen_before = _frozen_hashes(root, config)
    _status(
        status_path,
        state="running",
        phase="build_full_trajectory",
        progress=25,
        detail="72-step 완전경로와 event weights를 구성 중",
        eta="약 5~15분",
    )
    _, dataset, public_audit = _load_and_audit(data_dir)
    compatibility = _compatibility_check(root, dataset, config)
    windows = tuple(tuple(item) for item in config["validation"]["windows"])
    folds = build_forecast_folds(
        dataset.anchors,
        windows=windows,
        embargo_hours=int(config["validation"]["embargo_hours"]),
    )
    variants = list(config["models"]["variants"])
    blind_frames: list[pd.DataFrame] = []
    fold_diagnostics: dict[str, Any] = {}
    model_paths: list[Path] = []

    for fold_number, fold in enumerate(folds):
        progress = 30 + 15 * fold_number
        _status(
            status_path,
            state="running",
            phase=f"fit_{fold.name}",
            progress=progress,
            detail=f"{fold.name}: NLinear·DLinear 고정 모델 학습/블라인드 예측",
            eta="약 3~12분",
        )
        train_ids = fold.train_ids[dataset.complete_path[fold.train_ids]]
        train_weights = event_balanced_weights(dataset.anchors, train_ids)
        train_current = dataset.anchors.set_index("anchor_id").loc[
            train_ids, "current_hs"
        ].to_numpy(dtype=np.float64)
        train_target = dataset.path_target[train_ids].astype(np.float64) - train_current[:, None]
        selection_ids = fold.validation_ids
        holdout_ids = select_lattice_phase(
            dataset.anchors,
            start=fold.validation_start,
            end=fold.validation_end,
            phase_hours=int(config["validation"]["final_holdout_lattice_phase_hours"]),
            gap_hours=int(config["validation"]["lattice_gap_hours"]),
        )
        fold_diagnostics[fold.name] = {
            "outer_train_eligible_anchors": int(len(fold.train_ids)),
            "dense_complete_train_anchors": int(len(train_ids)),
            "train_episodes": int(
                dataset.anchors.set_index("anchor_id").loc[train_ids, "episode_id"].nunique()
            ),
            "event_weight_min": float(train_weights.min()),
            "event_weight_max": float(train_weights.max()),
            "event_weight_mean": float(train_weights.mean()),
            "selection_cases": int(len(selection_ids)),
            "holdout_cases": int(len(holdout_ids)),
            "selection_holdout_anchor_overlap": int(
                np.intersect1d(selection_ids, holdout_ids).size
            ),
        }
        for variant in variants:
            model = _model_from_config(variant, config)
            station_train = dataset.anchors.set_index("anchor_id").loc[
                train_ids, "station"
            ].to_numpy(dtype=str)
            model.fit(
                dataset.history[train_ids], station_train, train_target, train_weights
            )
            model_path = output / "models" / f"{fold.name}_{variant}.npz"
            model.save(model_path)
            model_paths.append(model_path)
            for phase, ids in (("selection_phase_0h", selection_ids), ("final_holdout_phase_39h", holdout_ids)):
                station = dataset.anchors.set_index("anchor_id").loc[
                    ids, "station"
                ].to_numpy(dtype=str)
                delta = model.predict_delta(dataset.history[ids], station)
                blind_frames.append(
                    build_blind_prediction_frame(
                        dataset,
                        ids,
                        delta,
                        fold=fold.name,
                        phase=phase,
                        variant=variant,
                    )
                )

    blind = pd.concat(blind_frames, ignore_index=True)
    blind_keys = ["variant", "phase", *PAIR_KEYS]
    if blind.duplicated(blind_keys).any():
        raise ValueError("duplicate blind prediction keys")
    forbidden = {"target_hs", "official_target", "path_target"}.intersection(blind.columns)
    if forbidden:
        raise ValueError(f"holdout target leaked before prediction write: {sorted(forbidden)}")
    blind_path = output / "blind_predictions.parquet"
    blind.to_parquet(blind_path, index=False, compression="zstd")
    blind_sha = _sha256(blind_path)
    _status(
        status_path,
        state="running",
        phase="blind_predictions_frozen_opening_labels",
        progress=78,
        detail=f"블라인드 예측 SHA {blind_sha[:12]} 고정 후에만 validation label 개방",
        eta="약 1~3분",
    )

    evaluated = attach_official_targets(dataset, blind)
    phase_metrics: dict[str, Any] = {}
    for variant in variants:
        phase_metrics[variant] = {}
        for phase in ("selection_phase_0h", "final_holdout_phase_39h"):
            group = evaluated.loc[
                evaluated["variant"].eq(variant) & evaluated["phase"].eq(phase)
            ]
            phase_metrics[variant][phase] = {
                "candidate": metric_slices(group, "prediction"),
                "persistence": metric_slices(group, "persistence"),
            }

    selected_variant = min(
        variants,
        key=lambda variant: (
            phase_metrics[variant]["selection_phase_0h"]["candidate"]["rmse"],
            variants.index(variant),
        ),
    )
    selection = evaluated.loc[
        evaluated["variant"].eq(selected_variant)
        & evaluated["phase"].eq("selection_phase_0h")
    ].copy()
    incumbent_path = root / config["frozen_inputs"]["incumbent_oof"]
    incumbent = pd.read_parquet(incumbent_path).sort_values(PAIR_KEYS).reset_index(drop=True)
    incumbent = incumbent[PAIR_KEYS + ["target_hs", "prediction"]].rename(
        columns={"target_hs": "incumbent_target_hs", "prediction": "incumbent_prediction"}
    )
    paired = selection.merge(incumbent, on=PAIR_KEYS, how="inner", validate="one_to_one")
    if len(paired) != len(incumbent) or len(paired) != len(selection):
        raise ValueError("selection phase does not exactly match the frozen 182-case incumbent")
    if not np.allclose(paired["target_hs"], paired["incumbent_target_hs"], atol=1e-6):
        raise ValueError("candidate/incumbent validation targets differ")
    bootstrap = paired_event_bootstrap(
        paired,
        candidate_column="prediction",
        baseline_column="incumbent_prediction",
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    gate = _gate(paired, bootstrap, config)

    holdout = evaluated.loc[
        evaluated["variant"].eq(selected_variant)
        & evaluated["phase"].eq("final_holdout_phase_39h")
    ].copy()
    holdout_bootstrap = paired_event_bootstrap(
        holdout,
        candidate_column="prediction",
        baseline_column="persistence",
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["seed"]) + 1,
    )
    evaluated_path = output / "evaluated_predictions.parquet"
    paired_path = output / "paired_selection_vs_incumbent.parquet"
    evaluated.to_parquet(evaluated_path, index=False, compression="zstd")
    paired.to_parquet(paired_path, index=False, compression="zstd")

    frozen_after = _frozen_hashes(root, config)
    if frozen_after != frozen_before:
        raise RuntimeError("a frozen incumbent/model/submission artifact changed during the probe")
    model_sha = {str(path.relative_to(output)): _sha256(path) for path in model_paths}
    elapsed = time.perf_counter() - start_time
    result = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": "gate_passed" if gate["passed"] else "gate_failed_stop_before_nhits_vmd",
        "elapsed_seconds": float(elapsed),
        "config_sha256": _sha256(config_path),
        "git": _git_state(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "public_audit": public_audit,
        "trajectory_audit": _episode_summary(dataset),
        "compatibility": compatibility,
        "preregistered_model_selection": {
            "variants": variants,
            "rule": config["models"]["selection_rule"],
            "selected_variant": selected_variant,
        },
        "fold_diagnostics": fold_diagnostics,
        "metrics": phase_metrics,
        "paired_selection_vs_incumbent": {
            "bootstrap": bootstrap,
            "gate": gate,
        },
        "final_holdout_opened_after_blind_prediction_sha256": {
            "blind_prediction_sha256": blind_sha,
            "selected_variant_metrics": metric_slices(holdout, "prediction"),
            "persistence_metrics": metric_slices(holdout, "persistence"),
            "paired_event_bootstrap_vs_persistence": holdout_bootstrap,
        },
        "invariants": {
            "source_rows_mutated": 0,
            "external_observations_used": 0,
            "test_context_used_for_training": False,
            "hidden_test_labels_used": 0,
            "hyperparameter_search_run": False,
            "n_hits_or_vmd_run": False,
            "submission_written_or_uploaded": False,
            "frozen_artifacts_unchanged": True,
        },
        "sha256": {
            "smoke": _sha256(smoke_path),
            "blind_predictions": blind_sha,
            "evaluated_predictions": _sha256(evaluated_path),
            "paired_selection": _sha256(paired_path),
            "models": model_sha,
            "frozen_before_and_after": frozen_after,
        },
    }
    _atomic_json(metrics_path, result)
    _status(
        status_path,
        state="completed_gate_pass" if gate["passed"] else "completed_gate_fail",
        phase="complete",
        progress=100,
        detail=(
            f"{selected_variant} RMSE {gate['comparison']['candidate']['rmse']:.6f} · "
            f"gate {'PASS' if gate['passed'] else 'FAIL'} · N-HiTS/VMD 미실행"
        ),
        eta="완료",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--status-path", default=DEFAULT_STATUS)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (root / config["output_dir"]).resolve()
    status_path = (root / args.status_path).resolve()
    data_dir = args.data_dir or os.environ.get("P3_DATA_DIR")
    try:
        if args.mode == "smoke":
            result = _run_smoke(
                root=root,
                config_path=config_path,
                config=config,
                output=output,
                status_path=status_path,
                data_dir=data_dir,
            )
        else:
            result = _run_full(
                root=root,
                config_path=config_path,
                config=config,
                output=output,
                status_path=status_path,
                data_dir=data_dir,
            )
    except Exception as error:
        _status(
            status_path,
            state="failed",
            phase="stopped",
            progress=100,
            detail=f"{type(error).__name__}: {error}",
            eta="중단",
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
