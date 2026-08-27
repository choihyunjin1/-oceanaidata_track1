"""Run the preregistered local-only P2 physical-state analog one-shot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.deep_data import build_panel
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_teos_analog import (
    AnalogConfig,
    AnalogResidualModel,
    LinearEOSConfig,
    PublicState,
    blend_with_frozen,
    build_public_state,
    catalog_split,
    mask_target_interval,
)

EXPECTED_EXPERIMENT = "p2_teos_analog_v1"
EXPECTED_REFERENCE_RMSE = 0.7744179315612375


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _status(
    path: Path,
    *,
    progress: float,
    phase: str,
    detail: str,
    started: float,
    status: str = "running",
) -> None:
    elapsed = max(time.perf_counter() - started, 0.001)
    remaining = elapsed * max(100.0 - progress, 0.0) / max(progress, 1.0)
    eta = datetime.now().astimezone() + timedelta(seconds=remaining)
    _write_json(
        path,
        {
            "title": "P2 물리상태 analog one-shot",
            "experiment_id": EXPECTED_EXPERIMENT,
            "status": status,
            "progress": float(progress),
            "phase": phase,
            "detail": detail,
            "elapsed_seconds": elapsed,
            "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST"),
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def _load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPECTED_EXPERIMENT:
        raise ValueError("unexpected P2 analog experiment id")
    if value.get("status") != "preregistered_local_one_shot":
        raise ValueError("P2 analog contract is not preregistered")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("P2 analog must remain local-only")
    if value.get("external_values_used") is not False:
        raise ValueError("P2 analog contract unexpectedly permits external values")
    analog = value["analog"]
    if analog.get("selection_policy") is None:
        raise ValueError("fixed one-shot selection policy is absent")
    if value["outputs"].get("submission_created") is not False:
        raise ValueError("this runner must not create a submission")
    return value


def _linear_eos(contract: dict[str, Any]) -> LinearEOSConfig:
    proxy = contract["physical_proxy"]
    return LinearEOSConfig(
        rho0=float(proxy["rho0_kg_m3"]),
        reference_temp=float(proxy["reference_temp_c"]),
        reference_psal=float(proxy["reference_psal"]),
        thermal_expansion=float(proxy["thermal_expansion_per_c"]),
        haline_contraction=float(proxy["haline_contraction_per_psal"]),
    )


def _analog_config(contract: dict[str, Any]) -> AnalogConfig:
    value = contract["analog"]
    clip = value["residual_clip_quantiles"]
    return AnalogConfig(
        neighbors=int(value["neighbors"]),
        pca_components=int(value["pca_components"]),
        ridge=float(value["local_linear_ridge"]),
        blend=float(value["blend"]),
        max_normalized_neighbor_distance=float(value["max_normalized_neighbor_distance"]),
        min_effective_neighbors=float(value["min_effective_neighbors"]),
        max_query_missing_fraction=float(value["max_query_missing_fraction"]),
        minimum_feature_coverage=float(value["minimum_feature_coverage"]),
        residual_clip_low=float(clip[0]),
        residual_clip_high=float(clip[1]),
        batch_size=int(value["batch_size"]),
        n_jobs=int(value["n_jobs"]),
        seed=int(value["seed"]),
    )


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(truth) - np.asarray(prediction)))))


def _cut(
    truth: np.ndarray,
    frozen: np.ndarray,
    candidate: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int]:
    base = _rmse(truth[selected], frozen[selected])
    current = _rmse(truth[selected], candidate[selected])
    return {
        "rows": int(selected.sum()),
        "frozen_rmse": base,
        "candidate_rmse": current,
        "delta_rmse": current - base,
    }


def _metrics(
    frame: pd.DataFrame,
    frozen: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    truth = frame["truth"].to_numpy(float)
    all_rows = np.ones(len(frame), dtype=bool)
    return {
        **_cut(truth, frozen, candidate, all_rows),
        "by_block": {
            str(block): _cut(
                truth, frozen, candidate, frame["block"].eq(block).to_numpy()
            )
            for block in frame["block"].drop_duplicates()
        },
        "by_layer": {
            str(layer): _cut(
                truth, frozen, candidate, frame["layer"].eq(layer).to_numpy()
            )
            for layer in (2, 3, 4)
        },
    }


def _paired_day_bootstrap(
    frame: pd.DataFrame,
    frozen: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    truth = frame["truth"].to_numpy(float)
    day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    unique = np.unique(day)
    blocks = [np.flatnonzero(day == value) for value in unique]
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=np.float64)
    for number in range(replicates):
        chosen = rng.integers(0, len(blocks), len(blocks))
        rows = np.concatenate([blocks[index] for index in chosen])
        delta[number] = _rmse(truth[rows], candidate[rows]) - _rmse(truth[rows], frozen[rows])
    return {
        "replicates": int(replicates),
        "kst_days": int(len(unique)),
        "delta_rmse": _rmse(truth, candidate) - _rmse(truth, frozen),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def _weekly_metrics(
    frame: pd.DataFrame,
    frozen: np.ndarray,
    candidate: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    truth = frame["truth"].to_numpy(float)
    kst = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul")
    normalized = kst.dt.normalize()
    week = (normalized - pd.to_timedelta(normalized.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    keyed = frame["block"].astype(str) + "__" + week
    rows: list[dict[str, float | int | str]] = []
    for key in keyed.drop_duplicates():
        selected = keyed.eq(key).to_numpy()
        if selected.sum() < 100:
            continue
        metrics = _cut(truth, frozen, candidate, selected)
        block, week_start = key.split("__", maxsplit=1)
        rows.append({"block": block, "week_start_kst": week_start, **metrics})
    delta = np.array([float(row["delta_rmse"]) for row in rows], dtype=np.float64)
    summary = {
        "groups": float(len(rows)),
        "delta_rmse_median": float(np.median(delta)),
        "delta_rmse_p90": float(np.quantile(delta, 0.9)),
        "delta_rmse_worst": float(np.max(delta)),
        "degraded_share": float(np.mean(delta > 0.0)),
    }
    return rows, summary


def _validate_oof(frame: pd.DataFrame, contract: dict[str, Any]) -> None:
    required = {"time", "layer", "truth", "block", "prediction"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"frozen physical OOF is missing columns: {sorted(missing)}")
    if len(frame) != 69_850 or frame.duplicated(["time", "layer"]).any():
        raise ValueError("frozen physical OOF grain changed")
    if not np.isfinite(frame[["truth", "prediction"]].to_numpy(float)).all():
        raise ValueError("frozen physical OOF contains non-finite values")
    if set(frame["block"].unique()) != set(contract["validation"]["blocks"]):
        raise ValueError("frozen physical OOF block set changed")


def _frozen_hashes(contract: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name in ("physical_oof", "deep_oof"):
        expected = contract["frozen_inputs"][name]
        path = Path(expected["path"])
        digest = _sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"frozen {name} SHA256 changed")
        result[name] = {"path": path.as_posix(), "sha256": digest}
    return result


def _run(
    *,
    data_dir: Path,
    config_path: Path,
    output_dir: Path,
    status_path: Path,
    started: float,
) -> dict[str, Any]:
    contract = _load_contract(config_path)
    if (output_dir / "result.json").exists():
        raise FileExistsError("one-shot result already exists; refusing a second evaluation")
    frozen_before = _frozen_hashes(contract)
    _status(
        status_path,
        progress=5,
        phase="계약·입력 검증",
        detail="사전등록값과 동결 Deep/물리 OOF SHA 확인",
        started=started,
    )
    data = load_p2_data(resolve_data_dir(data_dir))
    physical_path = Path(contract["frozen_inputs"]["physical_oof"]["path"])
    oof = pd.read_parquet(physical_path)
    _validate_oof(oof, contract)
    frozen = oof["prediction"].to_numpy(float)
    reference_rmse = _rmse(oof["truth"].to_numpy(float), frozen)
    if not np.isclose(reference_rmse, EXPECTED_REFERENCE_RMSE, rtol=0.0, atol=1e-12):
        raise ValueError("frozen physical OOF reference RMSE changed")

    _status(
        status_path,
        progress=12,
        phase="동시 blackout",
        detail="각 outer validation에서 layer 2·3·4 temp+psal 동시 마스킹 검사",
        started=started,
    )
    mask_diagnostics: dict[str, dict[str, Any]] = {}
    for block, (start, stop) in contract["validation"]["blocks"].items():
        masked, diagnostics = mask_target_interval(data.observations, start, stop)
        selected = (
            pd.to_datetime(masked["time"], utc=True)
            .ge(pd.Timestamp(start, tz="Asia/Seoul").tz_convert("UTC"))
            & pd.to_datetime(masked["time"], utc=True)
            .lt(pd.Timestamp(stop, tz="Asia/Seoul").tz_convert("UTC"))
            & masked["layer"].isin((2, 3, 4))
        )
        if not masked.loc[selected, ["temp", "psal"]].isna().all().all():
            raise AssertionError("outer blackout contract failed")
        mask_diagnostics[block] = diagnostics
        del masked
        gc.collect()

    panel = build_panel(data.observations)
    public_only = data.observations.copy()
    public_only.loc[public_only["layer"].isin((2, 3, 4)), ["temp", "psal"]] = np.nan
    state = build_public_state(
        public_only,
        eos=_linear_eos(contract),
        trajectory_horizons=tuple(contract["physical_proxy"]["trajectory_horizons_minutes"]),
    )
    del public_only
    gc.collect()
    if not state.frame.index.equals(panel.times):
        raise ValueError("physical state and target panel timestamps differ")

    setting = _analog_config(contract)
    candidate_unprojected = frozen.copy()
    analog_absolute_rows = np.full(len(oof), np.nan, dtype=np.float64)
    support_rows = np.zeros(len(oof), dtype=bool)
    distance_rows = np.full(len(oof), np.nan, dtype=np.float64)
    effective_rows = np.full(len(oof), np.nan, dtype=np.float64)
    fold_diagnostics: dict[str, dict[str, Any]] = {}
    oof_times = pd.DatetimeIndex(pd.to_datetime(oof["time"], utc=True))
    panel_lookup = pd.Series(np.arange(len(panel.times)), index=panel.times)

    blocks = list(contract["validation"]["blocks"].items())
    for number, (block, (start, stop)) in enumerate(blocks):
        _status(
            status_path,
            progress=25 + number * 18,
            phase=f"analog {number + 1}/3",
            detail=f"{block}: outer-train catalog kNN + local-linear residual",
            started=started,
        )
        split = catalog_split(
            panel.times,
            panel.target_mask,
            start,
            stop,
            purge_days=int(contract["validation"]["purge_days"]),
        )
        train_state = PublicState(state.frame.loc[split.training], state.feature_columns)
        train_residual = (panel.target - panel.baseline)[split.training]
        model = AnalogResidualModel.fit(train_state, train_residual, config=setting)

        block_rows = oof["block"].eq(block).to_numpy()
        query_times = pd.DatetimeIndex(oof_times[block_rows].unique()).sort_values()
        query_state = PublicState(state.frame.reindex(query_times), state.feature_columns)
        analog = model.predict(query_state)
        panel_rows = panel_lookup.reindex(query_times).to_numpy()
        if not np.isfinite(panel_rows).all():
            raise ValueError("OOF query time is absent from target panel")
        query_baseline = panel.baseline[panel_rows.astype(int)]
        analog_absolute = query_baseline + analog.residual

        query_lookup = pd.Series(np.arange(len(query_times)), index=query_times)
        query_rows = query_lookup.reindex(oof_times[block_rows]).to_numpy()
        if not np.isfinite(query_rows).all():
            raise ValueError("OOF rows failed query-time alignment")
        query_rows = query_rows.astype(int)
        layer_offset = oof.loc[block_rows, "layer"].to_numpy(int) - 2
        row_analog = analog_absolute[query_rows, layer_offset]
        row_support = analog.supported[query_rows]
        row_distance = analog.normalized_neighbor_distance[query_rows]
        row_effective = analog.effective_neighbors[query_rows]
        candidate_unprojected[block_rows] = blend_with_frozen(
            frozen[block_rows], row_analog, row_support, blend=setting.blend
        )
        analog_absolute_rows[block_rows] = row_analog
        support_rows[block_rows] = row_support
        distance_rows[block_rows] = row_distance
        effective_rows[block_rows] = row_effective
        fold_diagnostics[block] = {
            "catalog_complete_profiles": int(split.training.sum()),
            "validation_grid_times": int(split.validation.sum()),
            "purged_grid_times": int(split.purged.sum()),
            "selected_state_features": int(model.transform.selected.sum()),
            "pca_components": int(model.transform.pca.n_components_),
            "pca_explained_variance_share": float(
                model.transform.pca.explained_variance_ratio_.sum()
            ),
            **analog.diagnostics(),
        }
        del model, train_state, train_residual, query_state, analog
        gc.collect()

    if not np.isfinite(candidate_unprojected).all():
        raise ValueError("analog blend produced non-finite OOF predictions")
    _status(
        status_path,
        progress=79,
        phase="물리 투영",
        detail="지원 query만 수정 후 공개층 envelope·순서 투영; fallback exact 복원",
        started=started,
    )
    endpoints = public_endpoint_frame(data.observations)
    projection = project_profiles_vectorized(oof, candidate_unprojected, endpoints)
    candidate = projection.prediction
    candidate[~support_rows] = frozen[~support_rows]
    if not np.array_equal(candidate[~support_rows], frozen[~support_rows]):
        raise AssertionError("unsupported rows differ from frozen physical projection")

    metrics = _metrics(oof, frozen, candidate)
    bootstrap = _paired_day_bootstrap(
        oof,
        frozen,
        candidate,
        replicates=int(contract["bootstrap"]["replicates"]),
        seed=int(contract["bootstrap"]["seed"]),
    )
    weekly, weekly_summary = _weekly_metrics(oof, frozen, candidate)
    gate = contract["gate"]
    checks = {
        "candidate_rmse": metrics["candidate_rmse"] <= float(gate["candidate_rmse_max"]),
        "delta_rmse": metrics["delta_rmse"] <= float(gate["required_delta_rmse_max"]),
        "bootstrap_ci90_high": bootstrap["ci90_high"] < float(
            gate["bootstrap_ci90_high_max"]
        ),
        "layer4": metrics["by_layer"]["4"]["delta_rmse"]
        <= float(gate["layer4_delta_rmse_max"]),
        "weekly_p90": weekly_summary["delta_rmse_p90"]
        <= float(gate["weekly_delta_rmse_p90_max"]),
        "weekly_worst": weekly_summary["delta_rmse_worst"]
        <= float(gate["weekly_delta_rmse_worst_max"]),
        "weekly_degraded_share": weekly_summary["degraded_share"]
        <= float(gate["weekly_degraded_share_max"]),
    }
    passed = all(checks.values())
    decision = "PASS_ALLOW_BAYOTIDE_RESEARCH" if passed else "REJECT_STOP_BEFORE_BAYOTIDE"

    _status(
        status_path,
        progress=93,
        phase="artifact 검증",
        detail="OOF·bootstrap·주별 gate 및 동결 SHA 재검증",
        started=started,
    )
    frozen_after = _frozen_hashes(contract)
    if frozen_after != frozen_before:
        raise AssertionError("frozen Deep/physical artifacts changed during experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / "oof.parquet"
    saved = oof.loc[:, ["time", "layer", "truth", "block"]].copy()
    saved["frozen_prediction"] = frozen
    saved["analog_absolute"] = analog_absolute_rows
    saved["supported"] = support_rows
    saved["normalized_neighbor_distance"] = distance_rows
    saved["effective_neighbors"] = effective_rows
    saved["candidate_unprojected"] = candidate_unprojected
    saved["prediction"] = candidate
    saved.to_parquet(oof_path, index=False, compression="zstd")

    result: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": EXPECTED_EXPERIMENT,
        "research_only": True,
        "adaptive_after_prior_outer_exposure": True,
        "fresh_holdout_claimed": False,
        "external_values_used": False,
        "submission_created": False,
        "uploaded": False,
        "elapsed_seconds": time.perf_counter() - started,
        "reference_rmse": reference_rmse,
        "method": {
            "density_warning": contract["physical_proxy"]["warning"],
            "public_layers": contract["physical_proxy"]["public_layers"],
            "trajectory_horizons_minutes": contract["physical_proxy"][
                "trajectory_horizons_minutes"
            ],
            "analog": {
                key: value
                for key, value in contract["analog"].items()
                if key != "selection_policy"
            },
        },
        "mask_diagnostics": mask_diagnostics,
        "fold_diagnostics": fold_diagnostics,
        "projection": {
            **projection.diagnostics(),
            "supported_rows": int(support_rows.sum()),
            "supported_share": float(support_rows.mean()),
            "fallback_exact": True,
        },
        "metrics": metrics,
        "paired_kst_day_bootstrap": bootstrap,
        "weekly": weekly,
        "weekly_summary": weekly_summary,
        "gate": {"thresholds": gate, "checks": checks, "passed": passed},
        "decision": decision,
        "frozen_artifacts": frozen_after,
        "artifacts": {"oof": {"path": oof_path.as_posix(), "sha256": _sha256(oof_path)}},
    }
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    manifest_path = output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "experiment_id": EXPECTED_EXPERIMENT,
            "config": {"path": config_path.as_posix(), "sha256": _sha256(config_path)},
            "source": {
                "path": "src/p2_teos_analog.py",
                "sha256": _sha256(Path("src/p2_teos_analog.py")),
            },
            "runner": {
                "path": "scripts/run_p2_teos_analog.py",
                "sha256": _sha256(Path("scripts/run_p2_teos_analog.py")),
            },
            "test": {
                "path": "tests/test_p2_teos_analog.py",
                "sha256": _sha256(Path("tests/test_p2_teos_analog.py")),
            },
            "frozen": frozen_after,
            "oof": {"path": oof_path.as_posix(), "sha256": _sha256(oof_path)},
            "result": {"path": result_path.as_posix(), "sha256": _sha256(result_path)},
            "submission_created": False,
            "uploaded": False,
        },
    )
    result["artifacts"]["result"] = {
        "path": result_path.as_posix(),
        "sha256": _sha256(result_path),
    }
    result["artifacts"]["manifest"] = {
        "path": manifest_path.as_posix(),
        "sha256": _sha256(manifest_path),
    }
    _status(
        status_path,
        progress=100,
        phase="완료",
        detail=(
            f"{decision} · RMSE {metrics['candidate_rmse']:.6f}℃ "
            f"· Δ {metrics['delta_rmse']:+.6f}℃ · 제출 생성 없음"
        ),
        started=started,
        status="complete",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/p2_teos_analog_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p2_teos_analog_v1")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/p2_teos_analog_v1.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        result = _run(
            data_dir=resolve_data_dir(args.data_dir),
            config_path=args.config,
            output_dir=args.output_dir,
            status_path=args.status_file,
            started=started,
        )
    except Exception as error:
        _status(
            args.status_file,
            progress=100,
            phase="실패",
            detail=f"{type(error).__name__}: {error}",
            started=started,
            status="failed",
        )
        raise
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "metrics": result["metrics"],
                "bootstrap": result["paired_kst_day_bootstrap"],
                "submission_created": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
