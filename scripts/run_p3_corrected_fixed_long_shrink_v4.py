"""Complete the append-only P3 corrected fixed-long-shrink improvement cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from p3_wave.corrected_fixed_long_shrink import FixedLongLeadShrinkCalibrator
from p3_wave.loss_router import (
    OBSERVED_FEATURES,
    build_inference_router_features,
    expand_case_router_features,
    route_row_predictions,
)
from p3_wave.one_shot_guard import acquire_persistent_attempt_lock
from p3_wave.submission import build_submission, validate_submission

LEADS = (3, 6, 9, 12, 18, 24)
KEYS = ["case_id", "station", "lead_h"]
OOF_KEYS = ["fold", "anchor_id", "station", "lead_h"]
FOLDS = ("2024_h2_storm", "winter_transition", "2025_h1")
CONFIG_RELATIVE = "configs/experiments/p3_corrected_fixed_long_shrink_v4.json"
CONFIG_SHA256 = "6d20e321086aef44f98f3fff1506ead7637b63136c4bff937faf91b0b8c7b1a8"
OUTPUT_RELATIVE = "artifacts/p3_corrected_fixed_long_shrink_v4"
LOCK_RELATIVE = "artifacts/p3_corrected_fixed_long_shrink_v4.ATTEMPT_LOCK.json"
BASE_RELATIVE = "artifacts/p3_corrected_repeated_forward_catboost_v2"
CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
PRIOR_RELATIVE = "artifacts/p3/long_persistence_shrink/metrics.json"
CURRENT_RELATIVE = "output/2026-08-20/ready/P3_submission.csv"
INCUMBENT_RMSE = 0.7791048399763751
INCUMBENT_WEIGHT = 0.20


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(truth: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    left = np.asarray(truth, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(left - right))))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _metric_slices(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    values = np.asarray(prediction, dtype=np.float64)
    result: dict[str, Any] = {"rmse_m": _rmse(frame["target_hs"], values), "rows": len(frame)}
    result["by_fold"] = {
        str(name): _rmse(group["target_hs"], values[group.index])
        for name, group in frame.groupby("fold", sort=True, observed=True)
    }
    result["by_station"] = {
        str(name): _rmse(group["target_hs"], values[group.index])
        for name, group in frame.groupby("station", sort=True, observed=True)
    }
    result["by_lead"] = {
        str(int(name)): _rmse(group["target_hs"], values[group.index])
        for name, group in frame.groupby("lead_h", sort=True, observed=True)
    }
    return result


def _paired_case_bootstrap(
    frame: pd.DataFrame, candidate: np.ndarray, *, replicates: int = 5_000
) -> dict[str, Any]:
    work = frame[["anchor_id", "target_hs", "final_prediction"]].copy()
    work["candidate"] = np.asarray(candidate, dtype=np.float64)
    grouped = list(work.groupby("anchor_id", sort=True, observed=True))
    if len(grouped) != 181 or any(len(group) != 6 for _, group in grouped):
        raise ValueError("bootstrap requires 181 complete six-lead cases")
    truth = np.stack([group["target_hs"].to_numpy(float) for _, group in grouped])
    incumbent = np.stack([group["final_prediction"].to_numpy(float) for _, group in grouped])
    proposed = np.stack([group["candidate"].to_numpy(float) for _, group in grouped])
    generator = np.random.default_rng(20260822)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = generator.integers(0, len(grouped), len(grouped))
        deltas[index] = _rmse(truth[sampled], proposed[sampled]) - _rmse(
            truth[sampled], incumbent[sampled]
        )
    return {
        "unit": "complete_six_lead_case",
        "replicates": replicates,
        "seed": 20260822,
        "ci90_delta_candidate_minus_incumbent_m": [
            float(np.quantile(deltas, 0.05)),
            float(np.quantile(deltas, 0.95)),
        ],
        "median_delta_m": float(np.median(deltas)),
        "probability_improves_descriptive": float(np.mean(deltas < 0.0)),
    }


def _expected_inputs(root: Path, data_dir: Path) -> dict[str, tuple[Path, str]]:
    base = root / BASE_RELATIVE
    cache = root / CACHE_RELATIVE
    return {
        "config": (root / CONFIG_RELATIVE, CONFIG_SHA256),
        "prior_sealed_metrics": (
            root / PRIOR_RELATIVE,
            "41ed5676fc993fc6debefd271f74f28be69340bb4b7a5621ab5d83322a81c46e",
        ),
        "incumbent_oof": (
            base / "oof.parquet",
            "eb0af75ec29210254da0d13d1bb8164c0d6b427f4ad5853622144a11fe795f7e",
        ),
        "incumbent_metrics": (
            base / "metrics.json",
            "2c797e6169b7af27d343edb31fae5acfd4ce704149c63b732480fe33692c22e6",
        ),
        "incumbent_candidate": (
            base / "candidate/submission.csv",
            "24a360dd85978155b883378459f6d4d46a6b847569f1c3b6636a728c96e5ba11",
        ),
        "feature_columns": (
            base / "feature_columns.json",
            "49f1e03b43cce691f046027a4c59aac1bcb6846970c32ebf94c4b02382d4fa7a",
        ),
        "full_single": (
            base / "models/full/single.cbm",
            "b59972b88932a860ae38318f626e13f70e9429dd71234c2b4d52fc7369eedeac",
        ),
        "full_multi": (
            base / "models/full/multi.cbm",
            "9dffba79e9ccf33482fd7212b3eda5ec3a98606d5b14a2e34e156988b87280cc",
        ),
        "full_router": (
            base / "models/full/router.joblib",
            "897caa8e357b58db59f3ce8746a01fa85750f9379fc56d8ea00986cf5fed4bca",
        ),
        "test_features": (
            cache / "test_features.parquet",
            "004018935c155b0ab4fea18bdcfa2c99bdef265734c1ddcdd5ea5c2fee68312d",
        ),
        "test_index": (
            data_dir / "test_index.csv",
            "004551346ca5be6e3445d8b8e9c8121c16283eea72a363cbd673c5e3edcd2acc",
        ),
        "baseline_persistence": (
            data_dir / "baseline_persistence.csv",
            "0533ef3ad4bdff406a7680c9d4b17033d7d81afddad0cd5579ccbca80ec43110",
        ),
        "frozen_current": (
            root / CURRENT_RELATIVE,
            "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
        ),
    }


def _verify_inputs(inputs: dict[str, tuple[Path, str]]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, (path, expected) in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"required pinned input is missing: {name}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"pinned input SHA differs: {name}")
        observed[name] = digest
    return observed


def evaluate_identical_oof(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    oof = pd.read_parquet(root / BASE_RELATIVE / "oof.parquet").reset_index(drop=True)
    required = {
        *OOF_KEYS,
        "target_hs",
        "persistence",
        "routed_prediction",
        "final_prediction",
    }
    if not required.issubset(oof.columns):
        raise ValueError("incumbent OOF schema differs")
    if len(oof) != 1_086 or oof["anchor_id"].nunique() != 181:
        raise ValueError("incumbent OOF row/case contract differs")
    if oof.duplicated(OOF_KEYS).any() or oof[OOF_KEYS].isna().any().any():
        raise ValueError("incumbent OOF keys are missing or duplicated")
    lead_contract = oof.groupby("anchor_id", observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(values.astype(int)))
    )
    if not lead_contract.map(lambda value: value == LEADS).all():
        raise ValueError("incumbent OOF cases are incomplete")
    if tuple(dict.fromkeys(oof["fold"].astype(str))) != FOLDS:
        raise ValueError("incumbent OOF fold order differs")
    incumbent = oof["final_prediction"].to_numpy(float)
    incumbent_rmse = _rmse(oof["target_hs"], incumbent)
    if abs(incumbent_rmse - INCUMBENT_RMSE) > 1e-15:
        raise ValueError("incumbent OOF RMSE does not reproduce")
    calibrator = FixedLongLeadShrinkCalibrator()
    candidate = calibrator.predict(
        oof["routed_prediction"].to_numpy(float),
        oof["persistence"].to_numpy(float),
        oof["lead_h"].to_numpy(int),
    )
    incumbent_metrics = _metric_slices(oof, incumbent)
    candidate_metrics = _metric_slices(oof, candidate)
    fold_deltas = {
        fold: candidate_metrics["by_fold"][fold] - incumbent_metrics["by_fold"][fold]
        for fold in FOLDS
    }
    station_deltas = {
        station: candidate_metrics["by_station"][station] - incumbent_metrics["by_station"][station]
        for station in sorted(incumbent_metrics["by_station"])
    }
    long_mask = oof["lead_h"].isin([18, 24]).to_numpy()
    long_delta = _rmse(oof.loc[long_mask, "target_hs"], candidate[long_mask]) - _rmse(
        oof.loc[long_mask, "target_hs"], incumbent[long_mask]
    )
    checks = {
        "strict_pooled_rmse_below_incumbent": candidate_metrics["rmse_m"]
        < incumbent_metrics["rmse_m"],
        "at_least_two_folds_strictly_improve": sum(value < 0.0 for value in fold_deltas.values())
        >= 2,
        "all_major_stations_within_degradation_guard": max(station_deltas.values()) <= 0.01,
        "lead_18_24_combined_within_degradation_guard": long_delta <= 0.01,
        "finite_and_range_valid": bool(
            np.isfinite(candidate).all() and np.all((candidate >= 0.0) & (candidate <= 30.0))
        ),
        "same_keys_truth_and_split_surface": True,
        "coefficient_search_on_corrected_oof_run_zero": True,
    }
    detail = {
        "surface": {"cases": 181, "rows": 1_086, "folds": list(FOLDS)},
        "incumbent": incumbent_metrics,
        "candidate": candidate_metrics,
        "delta_candidate_minus_incumbent_m": candidate_metrics["rmse_m"]
        - incumbent_metrics["rmse_m"],
        "fold_delta_m": fold_deltas,
        "strictly_improved_fold_count": int(sum(value < 0.0 for value in fold_deltas.values())),
        "station_delta_m": station_deltas,
        "lead_18_24_combined_delta_m": long_delta,
        "paired_case_bootstrap": _paired_case_bootstrap(oof, candidate),
        "gate": {"passed": bool(all(checks.values())), "checks": checks},
    }
    oof["candidate_prediction"] = candidate
    return oof, detail


def _cat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station"] = result["station"].astype(str)
    if "lead_h" in result:
        result["lead_h"] = result["lead_h"].astype(str)
    return result


def _saved_base_inference(
    *,
    test_features: pd.DataFrame,
    test_index: pd.DataFrame,
    feature_columns: list[str],
    single_path: Path,
    multi_path: Path,
    router_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if list(test_index.columns) != KEYS or len(test_index) != 1_200:
        raise ValueError("test_index schema/rows differ")
    if test_index.duplicated(KEYS).any() or test_index[KEYS].isna().any().any():
        raise ValueError("test_index keys are missing or duplicated")
    contracts = test_index.groupby("case_id", sort=False, observed=True)["lead_h"].agg(tuple)
    if len(contracts) != 200 or not contracts.map(lambda values: values == LEADS).all():
        raise ValueError("test_index case/lead contract differs")
    if len(test_features) != 200 or test_features.duplicated(["case_id", "station"]).any():
        raise ValueError("same-case test feature cache contract differs")
    forbidden = {
        column
        for column in test_features.columns
        if column.lower() in {"time", "timestamp", "date", "target_hs", "truth", "label"}
    }
    if forbidden:
        raise ValueError(f"anonymous test feature cache has forbidden columns: {sorted(forbidden)}")
    case_order = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    cases = case_order.merge(
        test_features, on=["case_id", "station"], how="left", validate="one_to_one"
    )
    if len(cases) != 200 or cases[feature_columns].isna().all(axis=1).any():
        raise ValueError("same-case test feature alignment failed")
    source = cases.set_index(["case_id", "station"])
    row_keys = pd.MultiIndex.from_frame(test_index[["case_id", "station"]])
    single_x = source.loc[row_keys, feature_columns].reset_index(drop=True)
    single_x.insert(0, "lead_h", test_index["lead_h"].to_numpy())
    single_x.insert(0, "station", test_index["station"].astype(str).to_numpy())
    current_rows = source.loc[row_keys, "hs_current"].to_numpy(float)
    single_x.insert(2, "current_hs_for_residual", current_rows)
    single = CatBoostRegressor()
    single.load_model(single_path)
    single_prediction = np.clip(current_rows + single.predict(_cat_frame(single_x)), 0.0, 30.0)
    multi_x = cases[["station", *feature_columns]].copy()
    multi_x["station"] = multi_x["station"].astype(str)
    multi = CatBoostRegressor()
    multi.load_model(multi_path)
    multi_delta = np.asarray(multi.predict(multi_x), dtype=np.float64)
    current_case = cases["hs_current"].to_numpy(float)
    multi_prediction = np.clip(current_case[:, None] + multi_delta, 0.0, 30.0)
    components = np.stack(
        [
            single_prediction.reshape(200, len(LEADS)),
            multi_prediction,
            np.repeat(current_case[:, None], len(LEADS), axis=1),
        ],
        axis=2,
    )
    case_x = build_inference_router_features(
        cases.loc[:, OBSERVED_FEATURES],
        cases["station"].to_numpy(str),
        current_case,
        components,
    )
    meta = pd.DataFrame(
        {
            "fold": "anonymous_test",
            "anchor_id": np.arange(200, dtype=np.int64),
            "station": cases["station"].astype(str),
            "anchor_time": pd.NaT,
        }
    )
    row_x, row_meta, row_components = expand_case_router_features(case_x, meta, components)
    router = joblib.load(router_path)
    weights = router.predict_weights(row_x)
    inactive = ~row_meta["lead_h"].isin([12, 18, 24]).to_numpy()
    weights[inactive] = np.array([0.5, 0.5, 0.0])
    routed = route_row_predictions(row_components, weights)
    if not np.isfinite(routed).all():
        raise ValueError("saved base-model inference is non-finite")
    return routed, current_rows


def _write_submission_exclusive(frame: pd.DataFrame, index: pd.DataFrame, path: Path) -> None:
    validate_submission(frame, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", mode="x")
    reread = pd.read_csv(path)
    validate_submission(reread, index)
    if not reread[KEYS].equals(index[KEYS]):
        raise AssertionError("candidate keys/order changed on CSV roundtrip")


def _copy_full_bundle(root: Path, stage: Path) -> dict[str, Path]:
    source = root / BASE_RELATIVE / "models/full"
    destination = stage / "models/full"
    destination.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    for name in ("single.cbm", "multi.cbm", "router.joblib"):
        target = destination / name
        shutil.copy2(source / name, target)
        paths[name] = target
    shutil.copy2(
        root / BASE_RELATIVE / "feature_columns.json", destination / "feature_columns.json"
    )
    paths["feature_columns.json"] = destination / "feature_columns.json"
    calibrator_path = destination / "calibrator.joblib"
    joblib.dump(FixedLongLeadShrinkCalibrator(), calibrator_path)
    paths["calibrator.joblib"] = calibrator_path
    return paths


def _infer_candidate(
    *, data_dir: Path, cache_dir: Path, model_paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    feature_payload = json.loads(model_paths["feature_columns.json"].read_text(encoding="utf-8"))
    feature_columns = feature_payload["columns"]
    test_features = pd.read_parquet(cache_dir / "test_features.parquet")
    test_index = pd.read_csv(data_dir / "test_index.csv")
    persistence = pd.read_csv(data_dir / "baseline_persistence.csv")
    if not persistence[KEYS].equals(test_index[KEYS]):
        raise ValueError("persistence keys/order differ from test_index")
    routed, current_rows = _saved_base_inference(
        test_features=test_features,
        test_index=test_index,
        feature_columns=feature_columns,
        single_path=model_paths["single.cbm"],
        multi_path=model_paths["multi.cbm"],
        router_path=model_paths["router.joblib"],
    )
    if not np.allclose(current_rows, persistence["hs_pred"].to_numpy(float), atol=0.0, rtol=0.0):
        raise ValueError("test persistence differs from same-case hs_current")
    calibrator: FixedLongLeadShrinkCalibrator = joblib.load(model_paths["calibrator.joblib"])
    prediction = calibrator.predict(
        routed,
        persistence["hs_pred"].to_numpy(float),
        test_index["lead_h"].to_numpy(int),
    )
    incumbent_prediction = routed.copy()
    active = test_index["lead_h"].isin([12, 18, 24]).to_numpy()
    incumbent_prediction[active] = 0.8 * routed[active] + 0.2 * persistence.loc[
        active, "hs_pred"
    ].to_numpy(float)
    incumbent_saved = pd.read_csv(
        Path(__file__).resolve().parents[1] / BASE_RELATIVE / "candidate/submission.csv"
    )
    if not incumbent_saved[KEYS].equals(test_index[KEYS]):
        raise ValueError("incumbent candidate keys/order differ")
    incumbent_reproduction_max_abs = float(
        np.max(np.abs(incumbent_prediction - incumbent_saved["hs_pred"].to_numpy(float)))
    )
    if incumbent_reproduction_max_abs > 1e-12:
        raise RuntimeError("saved full base models do not reproduce incumbent inference")
    candidate = build_submission(test_index, prediction)
    return (
        candidate,
        test_index,
        {
            "minimum_m": float(np.min(prediction)),
            "median_m": float(np.median(prediction)),
            "maximum_m": float(np.max(prediction)),
            "mean_m": float(np.mean(prediction)),
            "incumbent_saved_model_reproduction_max_abs_m": incumbent_reproduction_max_abs,
            "candidate_minus_incumbent_prediction_rmse_m": _rmse(prediction, incumbent_prediction),
            "candidate_minus_incumbent_prediction_max_abs_m": float(
                np.max(np.abs(prediction - incumbent_prediction))
            ),
        },
    )


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in stage.rglob("*") if value.is_file()):
        relative = path.relative_to(stage).as_posix()
        if relative in {"manifest.json", "manifest.sha256"}:
            continue
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _git_state(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "head": head.stdout.strip() if head.returncode == 0 else "unknown",
        "dirty": bool(dirty),
        "dirty_entry_count": len(dirty),
    }


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    config_path = (root / CONFIG_RELATIVE).resolve(strict=True)
    if config_path != root.resolve(strict=True) / CONFIG_RELATIVE:
        raise PermissionError("non-canonical config path")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "p3_corrected_fixed_long_shrink_v4":
        raise ValueError("unexpected experiment identity")
    if config["model"]["persistence_weight"] != 0.25:
        raise ValueError("fixed coefficient differs")
    if not all(config["prohibitions"].values()):
        raise ValueError("all prohibitions must remain enabled")
    inputs = _expected_inputs(root, data_dir)
    snapshot = _verify_inputs(inputs)
    output = root / OUTPUT_RELATIVE
    lock = root / LOCK_RELATIVE
    return {
        "status": "CHECK_ONLY_PASS",
        "config_sha256": snapshot["config"],
        "pinned_input_count": len(snapshot),
        "output_absent": not output.exists(),
        "attempt_lock_absent": not lock.exists(),
    }


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    preflight = check_only(root=root, data_dir=data_dir)
    if not preflight["output_absent"] or not preflight["attempt_lock_absent"]:
        raise FileExistsError("append-only output or attempt lock already exists")
    inputs = _expected_inputs(root, data_dir)
    before = _verify_inputs(inputs)
    lock_receipt = acquire_persistent_attempt_lock(
        root / LOCK_RELATIVE,
        experiment_id="p3_corrected_fixed_long_shrink_v4",
        config_sha256=CONFIG_SHA256,
        created_at=_now(),
    )
    started = time.perf_counter()
    oof, evaluation = evaluate_identical_oof(root)
    if not evaluation["gate"]["passed"]:
        raise RuntimeError("winner gate failed closed before anonymous test inference")
    temporary_root = root / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_corrected_fixed_shrink_v4_", dir=temporary_root))
    try:
        model_paths = _copy_full_bundle(root, stage)
        candidate, test_index, prediction_summary = _infer_candidate(
            data_dir=data_dir,
            cache_dir=root / CACHE_RELATIVE,
            model_paths=model_paths,
        )
        candidate_path = stage / "candidate/submission.csv"
        _write_submission_exclusive(candidate, test_index, candidate_path)
        # Freshly reload every saved model and the calibrator for independent materialization.
        reproduced, reproduced_index, reproduced_summary = _infer_candidate(
            data_dir=data_dir,
            cache_dir=root / CACHE_RELATIVE,
            model_paths=model_paths,
        )
        reproduced_path = stage / "candidate/reproduced_submission.csv"
        _write_submission_exclusive(reproduced, reproduced_index, reproduced_path)
        if candidate_path.read_bytes() != reproduced_path.read_bytes():
            raise RuntimeError("saved-model candidate reproduction is not byte-identical")
        oof.to_parquet(stage / "oof.parquet", index=False, compression="zstd")
        _atomic_json(
            stage / "metrics.json",
            {
                "created_at": _now(),
                "experiment_id": "p3_corrected_fixed_long_shrink_v4",
                "status": "WINNER_FULL_MODEL_CANDIDATE_CREATED_NOT_UPLOADED",
                "prior_provenance": {
                    "artifact": PRIOR_RELATIVE,
                    "artifact_sha256": before["prior_sealed_metrics"],
                    "persistence_weight": 0.25,
                    "corrected_oof_coefficient_search": False,
                },
                "evaluation": evaluation,
                "candidate_validation": {
                    "rows": len(candidate),
                    "cases": int(candidate["case_id"].nunique()),
                    "key_order_exact": True,
                    "finite": bool(np.isfinite(candidate["hs_pred"]).all()),
                    "range_0_to_30_m": bool(candidate["hs_pred"].between(0.0, 30.0).all()),
                    "saved_model_reproduction_byte_identical": True,
                    "prediction_summary": prediction_summary,
                    "reproduction_summary": reproduced_summary,
                    "candidate_sha256": sha256_file(candidate_path),
                    "reproduced_candidate_sha256": sha256_file(reproduced_path),
                },
                "access_counters": {
                    "test_feature_cache_reads": 2,
                    "test_index_reads": 2,
                    "test_context_reads": 0,
                    "test_target_or_hidden_label_reads": 0,
                    "absolute_test_timestamp_recovery_attempts": 0,
                    "current_or_frozen_writes": 0,
                    "upload_attempts": 0,
                },
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        after = _verify_inputs(inputs)
        if before != after:
            raise RuntimeError("pinned source/cache/base/current inputs changed during run")
        implementation_paths = {
            "config": root / CONFIG_RELATIVE,
            "runner": Path(__file__).resolve(),
            "calibrator_module": root / "src/p3_wave/corrected_fixed_long_shrink.py",
            "test": root / "tests/test_p3_corrected_fixed_long_shrink_v4.py",
        }
        manifest = {
            "created_at": _now(),
            "experiment_id": "p3_corrected_fixed_long_shrink_v4",
            "status": "WINNER_FULL_MODEL_CANDIDATE_CREATED_NOT_UPLOADED",
            "append_only_generation": True,
            "attempt_lock": lock_receipt,
            "git": _git_state(root),
            "input_sha256_before": before,
            "input_sha256_after": after,
            "input_unchanged": True,
            "implementation_sha256": {
                name: sha256_file(path) for name, path in implementation_paths.items()
            },
            "output_files": _artifact_hashes(stage),
            "candidate_created": True,
            "candidate_uploaded": False,
            "test_target_or_hidden_labels_used": 0,
            "absolute_test_timestamp_recovered": False,
            "current_or_frozen_mutated": False,
            "no_raw_values_in_manifest": True,
        }
        _atomic_json(stage / "manifest.json", manifest)
        manifest_sha = sha256_file(stage / "manifest.json")
        (stage / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n", encoding="ascii")
        output = root / OUTPUT_RELATIVE
        if output.exists():
            raise FileExistsError("append-only output appeared before atomic finalize")
        stage.replace(output)
    except Exception:
        # Preserve the attempt lock and failed stage for audit; never make a second attempt.
        raise
    result = {
        "status": "WINNER_FULL_MODEL_CANDIDATE_CREATED_NOT_UPLOADED",
        "artifact_dir": OUTPUT_RELATIVE,
        "candidate_sha256": sha256_file(root / OUTPUT_RELATIVE / "candidate/submission.csv"),
        "reproduced_candidate_sha256": sha256_file(
            root / OUTPUT_RELATIVE / "candidate/reproduced_submission.csv"
        ),
        "metrics_sha256": sha256_file(root / OUTPUT_RELATIVE / "metrics.json"),
        "oof_sha256": sha256_file(root / OUTPUT_RELATIVE / "oof.parquet"),
        "manifest_sha256": sha256_file(root / OUTPUT_RELATIVE / "manifest.json"),
        "candidate_rmse_m": evaluation["candidate"]["rmse_m"],
        "delta_m": evaluation["delta_candidate_minus_incumbent_m"],
        "improved_folds": evaluation["strictly_improved_fold_count"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", choices=("check-only", "run"), default="check-only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).expanduser().resolve(strict=True)
    if args.mode == "check-only":
        print(json.dumps(check_only(root=root, data_dir=data_dir), ensure_ascii=False, indent=2))
        return 0
    run_experiment(root=root, data_dir=data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
