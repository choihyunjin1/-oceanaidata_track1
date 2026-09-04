"""One-shot, append-only P1 performance-improvement cycle.

The outer comparison is walk-forward: a target fold is predicted with ensemble
parameters selected only on earlier outer folds.  The first fold is an exact
incumbent fallback.  Full-data refitting happens only after an outer winner is
fixed, and test labels are never read.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from p1_qc.config import load_config
from p1_qc.data import KEY_COLUMNS, load_dataset
from p1_qc.pipeline import (
    SavedTabularModel,
    apply_postprocess,
    load_or_build_features,
    predict_submission,
    train_full_model,
)
from p1_qc.rules import detect_plateaus, detect_singleton_spikes
from p1_qc.submission import build_submission, validate_submission, write_submission

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KST = ZoneInfo("Asia/Seoul")


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _resolve(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
        raise ValueError(f"path escapes project root: {relative}")
    return path


def load_cycle_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if PROJECT_ROOT not in config_path.parents:
        raise ValueError("cycle config must be inside project root")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "p1_full_improvement_cycle_v1":
        raise ValueError("unexpected experiment_id")
    if len(config["branch_order"]) != 3:
        raise ValueError("exactly three preregistered branches are required")
    return config_path, config


def _implementation_paths() -> dict[str, Path]:
    return {
        "config": _resolve("configs/experiments/p1_full_improvement_cycle_v1.json"),
        "module": _resolve("src/p1_qc/improvement_cycle.py"),
        "runner": _resolve("scripts/run_p1_full_improvement_cycle.py"),
        "tests": _resolve("tests/test_p1_full_improvement_cycle.py"),
    }


def seal_preexecution(config_path: str | Path) -> dict[str, Any]:
    resolved_config, config = load_cycle_config(config_path)
    if resolved_config != _implementation_paths()["config"]:
        raise ValueError("only the canonical cycle config may be sealed")
    paths = {name: _resolve(value) for name, value in config["paths"].items()}
    artifact_root = paths.pop("artifact_root")
    if artifact_root.exists():
        raise FileExistsError(f"append-only artifact root already exists: {artifact_root}")
    for name, expected in config["expected_sha256"].items():
        actual = sha256_file(paths[name])
        if actual != expected:
            raise RuntimeError(f"input SHA mismatch for {name}: {actual}")
    implementation = {name: sha256_file(path) for name, path in _implementation_paths().items()}
    receipt = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "sealed_at_kst": datetime.now(KST).isoformat(),
        "config_sha256": sha256_file(resolved_config),
        "implementation_sha256": implementation,
        "registered_input_sha256": config["expected_sha256"],
        "branch_order": config["branch_order"],
        "fold_order": config["fold_order"],
        "outer_row_label_reads": 0,
        "model_fits": 0,
        "test_prediction_generations": 0,
        "submission_uploads": 0,
        "decision": "SEALED_FOR_EXACTLY_ONE_EXECUTION",
    }
    artifact_root.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(artifact_root / "preexecution_seal.json", receipt)
    return receipt


def _verify_seal(config_path: Path, config: dict[str, Any], root: Path) -> dict[str, Any]:
    seal_path = root / "preexecution_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("config changed after preexecution seal")
    actual_implementation = {
        name: sha256_file(path) for name, path in _implementation_paths().items()
    }
    if seal["implementation_sha256"] != actual_implementation:
        raise RuntimeError("implementation changed after preexecution seal")
    for name, expected in config["expected_sha256"].items():
        actual = sha256_file(_resolve(config["paths"][name]))
        if actual != expected:
            raise RuntimeError(f"registered input changed: {name}")
    return seal


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("truth/prediction shape mismatch")
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": int(np.sum(y)),
        "predicted_positive": int(np.sum(p)),
    }


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _physical_order(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    work = frame.loc[:, ["station", "layer", "time"]].copy()
    work["_position"] = np.arange(len(work), dtype=np.int64)
    work["_time"] = pd.to_datetime(work["time"], utc=True, errors="raise")
    work.sort_values(["station", "layer", "_time", "_position"], inplace=True)
    positions = work["_position"].to_numpy(dtype=np.int64)
    groups = pd.factorize(pd.MultiIndex.from_frame(work[["station", "layer"]]), sort=False)[0]
    times = work["_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    breaks = np.ones(len(work), dtype=bool)
    if len(work) > 1:
        breaks[1:] = ~(
            (groups[1:] == groups[:-1]) & (times[1:] - times[:-1] == pd.Timedelta("10min").value)
        )
    return positions, breaks


def causal_event_rescue(
    frame: pd.DataFrame,
    incumbent_prediction: np.ndarray,
    causal_prediction: np.ndarray,
    incumbent_probability: np.ndarray,
    causal_probability: np.ndarray,
    *,
    causal_floor: float,
    incumbent_floor: float,
) -> np.ndarray:
    """Add whole causal-positive physical events with a causal-only seed."""

    x = np.asarray(incumbent_prediction, dtype=np.int8)
    c = np.asarray(causal_prediction, dtype=np.int8)
    px = np.asarray(incumbent_probability, dtype=float)
    pc = np.asarray(causal_probability, dtype=float)
    positions, breaks = _physical_order(frame)
    ox, oc, opx, opc = x[positions], c[positions], px[positions], pc[positions]
    result = ox.copy()
    start: int | None = None
    for i in range(len(oc) + 1):
        boundary = i == len(oc) or (i < len(oc) and breaks[i])
        active = i < len(oc) and oc[i] == 1
        if boundary and start is not None:
            stop = i
            seed = (
                (ox[start:stop] == 0)
                & (opc[start:stop] >= causal_floor)
                & (opx[start:stop] >= incumbent_floor)
            )
            if seed.any():
                result[start:stop] = 1
            start = None
        if active and start is None:
            start = i
        if start is not None and (i == len(oc) or not active):
            stop = i
            seed = (
                (ox[start:stop] == 0)
                & (opc[start:stop] >= causal_floor)
                & (opx[start:stop] >= incumbent_floor)
            )
            if seed.any():
                result[start:stop] = 1
            start = None
    restored = np.zeros(len(result), dtype=np.int8)
    restored[positions] = result
    return restored


def _select_rescue(
    frame: pd.DataFrame,
    truth: np.ndarray,
    x_pred: np.ndarray,
    c_pred: np.ndarray,
    px: np.ndarray,
    pc: np.ndarray,
    branch: dict[str, Any],
) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    best: tuple[tuple[float, int, float, float], dict[str, float], np.ndarray] | None = None
    for causal_floor in branch["causal_probability_floors"]:
        for incumbent_floor in branch["incumbent_probability_floors"]:
            prediction = causal_event_rescue(
                frame,
                x_pred,
                c_pred,
                px,
                pc,
                causal_floor=float(causal_floor),
                incumbent_floor=float(incumbent_floor),
            )
            metrics = binary_metrics(truth, prediction)
            added = int(np.sum((prediction == 1) & (x_pred == 0)))
            key = (metrics["f1"], -added, float(causal_floor), float(incumbent_floor))
            parameters = {
                "causal_floor": float(causal_floor),
                "incumbent_floor": float(incumbent_floor),
            }
            if best is None or key > best[0]:
                best = (key, parameters, prediction)
    if best is None:
        raise RuntimeError("empty causal rescue grid")
    return best[1], best[2], binary_metrics(truth, best[2])


def _postprocess_profiles(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(profile)
        for profile in config["branches"]["logit_convex_walk_forward"]["postprocess_profiles"]
    ]


def _blend_probability(px: np.ndarray, pc: np.ndarray, weight: float) -> np.ndarray:
    return _sigmoid(weight * _logit(px) + (1.0 - weight) * _logit(pc))


def _select_convex(
    frame: pd.DataFrame,
    truth: np.ndarray,
    px: np.ndarray,
    pc: np.ndarray,
    plateau: np.ndarray,
    spike: np.ndarray,
    branch: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    best: tuple[tuple[float, float, int], dict[str, Any], np.ndarray] | None = None
    profiles = [dict(value) for value in branch["postprocess_profiles"]]
    for weight in branch["incumbent_logit_weights"]:
        probability = _blend_probability(px, pc, float(weight))
        for index, profile in enumerate(profiles):
            prediction = apply_postprocess(frame, probability, plateau, spike, profile)
            score = binary_metrics(truth, prediction)["f1"]
            key = (score, float(weight), -index)
            parameters = {"incumbent_logit_weight": float(weight), "profile": profile}
            if best is None or key > best[0]:
                best = (key, parameters, prediction)
    if best is None:
        raise RuntimeError("empty convex grid")
    return best[1], best[2], binary_metrics(truth, best[2])


def _stack_matrix(frame: pd.DataFrame, px: np.ndarray, pc: np.ndarray) -> np.ndarray:
    station_levels = ("G-ORS", "I-ORS", "S-ORS")
    layer_levels = tuple(range(1, 9))
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].to_numpy(dtype=int)
    numeric = np.column_stack(
        (
            _logit(px),
            _logit(pc),
            np.asarray(px) - np.asarray(pc),
            np.abs(np.asarray(px) - np.asarray(pc)),
            frame["plateau"].to_numpy(dtype=float),
            frame["spike_candidate"].to_numpy(dtype=float),
        )
    )
    categorical = [station == value for value in station_levels]
    categorical.extend(layer == value for value in layer_levels)
    return np.column_stack((numeric, *categorical)).astype(np.float64, copy=False)


def _fit_logistic(
    matrix: np.ndarray, truth: np.ndarray, c_value: float, branch: dict[str, Any]
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        solver=str(branch["solver"]),
        max_iter=int(branch["max_iter"]),
        class_weight=branch["class_weight"],
        random_state=20260813,
        n_jobs=1,
    )
    model.fit(matrix, truth)
    return model


def _select_logistic(
    frame: pd.DataFrame,
    truth: np.ndarray,
    px: np.ndarray,
    pc: np.ndarray,
    profiles: list[dict[str, Any]],
    branch: dict[str, Any],
) -> tuple[dict[str, Any], LogisticRegression, np.ndarray, dict[str, Any]]:
    matrix = _stack_matrix(frame, px, pc)
    plateau = frame["plateau"].to_numpy(dtype=bool)
    spike = frame["spike_candidate"].to_numpy(dtype=bool)
    best: tuple[tuple[float, float, int], dict[str, Any], LogisticRegression, np.ndarray] | None = (
        None
    )
    for c_value in branch["regularization_C"]:
        model = _fit_logistic(matrix, truth, float(c_value), branch)
        probability = model.predict_proba(matrix)[:, 1]
        for index, profile in enumerate(profiles):
            prediction = apply_postprocess(frame, probability, plateau, spike, profile)
            score = binary_metrics(truth, prediction)["f1"]
            key = (score, -float(c_value), -index)
            parameters = {"C": float(c_value), "profile": profile}
            if best is None or key > best[0]:
                best = (key, parameters, model, prediction)
    if best is None:
        raise RuntimeError("empty logistic grid")
    return best[1], best[2], best[3], binary_metrics(truth, best[3])


def _metrics_by(
    frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray, column: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    values = frame[column].astype(str).to_numpy()
    for value in sorted(set(values.tolist())):
        mask = values == value
        result[value] = binary_metrics(truth[mask], prediction[mask])
    return result


def _guard_report(
    frame: pd.DataFrame,
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    guards: dict[str, Any],
) -> dict[str, Any]:
    base = binary_metrics(truth, incumbent)
    cand = binary_metrics(truth, candidate)
    fold_base = _metrics_by(frame, truth, incumbent, "fold")
    fold_cand = _metrics_by(frame, truth, candidate, "fold")
    station_base = _metrics_by(frame, truth, incumbent, "station")
    station_cand = _metrics_by(frame, truth, candidate, "station")
    fold_delta = {name: fold_cand[name]["f1"] - fold_base[name]["f1"] for name in fold_base}
    station_delta = {
        name: station_cand[name]["f1"] - station_base[name]["f1"]
        for name in station_base
        if station_base[name]["support"] >= guards["minimum_station_positive_support"]
    }
    checks = {
        "pooled_micro_f1_delta": cand["f1"] - base["f1"] >= guards["minimum_pooled_micro_f1_delta"],
        "fold_degradation": min(fold_delta.values()) >= -guards["maximum_fold_f1_degradation"],
        "station_degradation": min(station_delta.values())
        >= -guards["maximum_station_f1_degradation"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "incumbent": base,
        "candidate": cand,
        "micro_f1_delta": cand["f1"] - base["f1"],
        "by_fold_incumbent": fold_base,
        "by_fold_candidate": fold_cand,
        "by_fold_f1_delta": fold_delta,
        "by_station_incumbent": station_base,
        "by_station_candidate": station_cand,
        "by_station_f1_delta": station_delta,
    }


def _bootstrap_day_delta(
    frame: pd.DataFrame,
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    day = (
        pd.to_datetime(frame["time"], utc=True, errors="raise")
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
    )
    rows = []
    for _, indices in pd.Series(np.arange(len(frame))).groupby(day, sort=True):
        idx = indices.to_numpy(dtype=np.int64)
        base = binary_metrics(truth[idx], incumbent[idx])
        cand = binary_metrics(truth[idx], candidate[idx])
        rows.append((base["tp"], base["fp"], base["fn"], cand["tp"], cand["fp"], cand["fn"]))
    counts = np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        picked = counts[rng.integers(0, len(counts), size=len(counts))].sum(axis=0)
        btp, bfp, bfn, ctp, cfp, cfn = picked
        bf1 = 2 * btp / (2 * btp + bfp + bfn)
        cf1 = 2 * ctp / (2 * ctp + cfp + cfn)
        delta[replicate] = cf1 - bf1
    return {
        "unit": "KST calendar day",
        "days": int(len(counts)),
        "replicates": int(replicates),
        "seed": int(seed),
        "delta_mean": float(delta.mean()),
        "delta_ci90": [float(np.quantile(delta, 0.05)), float(np.quantile(delta, 0.95))],
        "probability_delta_positive": float(np.mean(delta > 0)),
    }


def _aligned_oof(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pd.read_parquet(_resolve(config["paths"]["incumbent_oof"]))
    c = pd.read_parquet(_resolve(config["paths"]["causal_oof"]))
    keys = config["key_columns"]
    required = {*keys, "label", "probability", "prediction", "plateau", "spike_candidate", "fold"}
    if not required.issubset(x.columns) or not required.issubset(c.columns):
        raise ValueError("OOF schema is incomplete")
    if len(x) != 421032 or len(c) != len(x):
        raise ValueError("unexpected OOF row count")
    if not x[keys].equals(c[keys]):
        raise ValueError("OOF keys/order differ")
    if not x["label"].equals(c["label"]):
        raise ValueError("OOF labels differ")
    if x.duplicated(keys).any():
        raise ValueError("duplicate OOF keys")
    if list(pd.unique(x["fold"])) != config["fold_order"]:
        raise ValueError("unexpected fold order")
    frame = x.loc[:, [*keys, "fold", "plateau", "spike_candidate"]].copy()
    truth = x["label"].to_numpy(dtype=np.int8)
    x_pred = x["prediction"].to_numpy(dtype=np.int8)
    c_pred = c["prediction"].to_numpy(dtype=np.int8)
    px = x["probability"].to_numpy(dtype=float)
    pc = c["probability"].to_numpy(dtype=float)
    if binary_metrics(truth, x_pred)["f1"] != 0.8603708380408055:
        raise RuntimeError("incumbent F1 did not reproduce")
    return frame, truth, x_pred, c_pred, px, pc


def _evaluate_branch(
    name: str,
    config: dict[str, Any],
    frame: pd.DataFrame,
    truth: np.ndarray,
    x_pred: np.ndarray,
    c_pred: np.ndarray,
    px: np.ndarray,
    pc: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], Any | None]:
    fold_order = config["fold_order"]
    prediction = np.zeros(len(frame), dtype=np.int8)
    selections: dict[str, Any] = {}
    branch = config["branches"][name]
    profiles = _postprocess_profiles(config)
    for index, fold in enumerate(fold_order):
        target = frame["fold"].eq(fold).to_numpy()
        if index == 0:
            prediction[target] = x_pred[target]
            selections[fold] = {"rule": "exact_incumbent_fallback"}
            continue
        calibration_folds = fold_order[:index]
        calibration = frame["fold"].isin(calibration_folds).to_numpy()
        if name == "causal_event_rescue_walk_forward":
            parameters, _, calibration_metrics = _select_rescue(
                frame.loc[calibration].reset_index(drop=True),
                truth[calibration],
                x_pred[calibration],
                c_pred[calibration],
                px[calibration],
                pc[calibration],
                branch,
            )
            prediction[target] = causal_event_rescue(
                frame.loc[target].reset_index(drop=True),
                x_pred[target],
                c_pred[target],
                px[target],
                pc[target],
                **parameters,
            )
            model = None
        elif name == "logit_convex_walk_forward":
            parameters, _, calibration_metrics = _select_convex(
                frame.loc[calibration].reset_index(drop=True),
                truth[calibration],
                px[calibration],
                pc[calibration],
                frame.loc[calibration, "plateau"].to_numpy(dtype=bool),
                frame.loc[calibration, "spike_candidate"].to_numpy(dtype=bool),
                branch,
            )
            probability = _blend_probability(
                px[target], pc[target], parameters["incumbent_logit_weight"]
            )
            prediction[target] = apply_postprocess(
                frame.loc[target].reset_index(drop=True),
                probability,
                frame.loc[target, "plateau"].to_numpy(dtype=bool),
                frame.loc[target, "spike_candidate"].to_numpy(dtype=bool),
                parameters["profile"],
            )
            model = None
        elif name == "logistic_stack_walk_forward":
            parameters, model, _, calibration_metrics = _select_logistic(
                frame.loc[calibration].reset_index(drop=True),
                truth[calibration],
                px[calibration],
                pc[calibration],
                profiles,
                branch,
            )
            target_frame = frame.loc[target].reset_index(drop=True)
            probability = model.predict_proba(_stack_matrix(target_frame, px[target], pc[target]))[
                :, 1
            ]
            prediction[target] = apply_postprocess(
                target_frame,
                probability,
                target_frame["plateau"].to_numpy(dtype=bool),
                target_frame["spike_candidate"].to_numpy(dtype=bool),
                parameters["profile"],
            )
        else:
            raise ValueError(f"unknown branch: {name}")
        selections[fold] = {
            "calibration_folds": calibration_folds,
            "calibration_rows": int(calibration.sum()),
            "parameters": parameters,
            "calibration_metrics": calibration_metrics,
            "target_label_reads_before_prediction": 0,
        }

    if name == "causal_event_rescue_walk_forward":
        deploy_parameters, _, deploy_metrics = _select_rescue(
            frame, truth, x_pred, c_pred, px, pc, branch
        )
        deploy_model = None
    elif name == "logit_convex_walk_forward":
        deploy_parameters, _, deploy_metrics = _select_convex(
            frame,
            truth,
            px,
            pc,
            frame["plateau"].to_numpy(dtype=bool),
            frame["spike_candidate"].to_numpy(dtype=bool),
            branch,
        )
        deploy_model = None
    else:
        deploy_parameters, deploy_model, _, deploy_metrics = _select_logistic(
            frame, truth, px, pc, profiles, branch
        )
    report = {
        "branch": name,
        "walk_forward_selections": selections,
        "deployment_parameters": deploy_parameters,
        "deployment_resubstitution_metrics_not_outer": deploy_metrics,
        "guard": _guard_report(frame, truth, x_pred, prediction, config["success_guards"]),
    }
    return prediction, report, deploy_model


@dataclass
class P1ImprovementEnsemble:
    incumbent_model: SavedTabularModel
    causal_model: SavedTabularModel
    winner_branch: str
    deployment_parameters: dict[str, Any]
    logistic_model: LogisticRegression | None
    config_sha256: str


def _predict_ensemble(
    bundle: P1ImprovementEnsemble,
    test: pd.DataFrame,
    offline_features: Any,
    causal_features: Any,
) -> pd.DataFrame:
    x_submission, px = predict_submission(bundle.incumbent_model, test, offline_features)
    c_submission, pc = predict_submission(bundle.causal_model, test, causal_features)
    x_pred = x_submission["label"].to_numpy(dtype=np.int8)
    c_pred = c_submission["label"].to_numpy(dtype=np.int8)
    plateau = detect_plateaus(test).to_numpy(dtype=bool)
    spike = detect_singleton_spikes(test).to_numpy(dtype=bool)
    params = bundle.deployment_parameters
    if bundle.winner_branch == "causal_event_rescue_walk_forward":
        label = causal_event_rescue(test, x_pred, c_pred, px, pc, **params)
    elif bundle.winner_branch == "logit_convex_walk_forward":
        probability = _blend_probability(px, pc, params["incumbent_logit_weight"])
        label = apply_postprocess(test, probability, plateau, spike, params["profile"])
    elif bundle.winner_branch == "logistic_stack_walk_forward":
        if bundle.logistic_model is None:
            raise RuntimeError("logistic winner has no fitted meta-model")
        meta_frame = test.loc[:, [*KEY_COLUMNS]].copy()
        meta_frame["plateau"] = plateau
        meta_frame["spike_candidate"] = spike
        probability = bundle.logistic_model.predict_proba(_stack_matrix(meta_frame, px, pc))[:, 1]
        label = apply_postprocess(test, probability, plateau, spike, params["profile"])
    else:
        raise RuntimeError("unknown saved winner branch")
    anomaly = np.full(len(test), "", dtype=object)
    anomaly[(label == 1) & plateau] = "flatline"
    anomaly[(label == 1) & spike & ~plateau] = "spike"
    return build_submission(test, label, anomaly)


def _full_fit_and_candidate(
    config_path: Path,
    config: dict[str, Any],
    root: Path,
    winner_name: str,
    winner_report: dict[str, Any],
    deployment_model: LogisticRegression | None,
) -> dict[str, Any]:
    data_dir = _resolve(config["paths"]["train_csv"]).parent
    p1_config_path = _resolve(config["paths"]["p1_config"])
    base_env = {"P1_DATA_DIR": str(data_dir), "P1QC_MODE": "offline"}
    offline_config = load_config(p1_config_path, env=base_env)
    causal_env = {"P1_DATA_DIR": str(data_dir), "P1QC_MODE": "causal"}
    causal_config = load_config(p1_config_path, env=causal_env)
    train = load_dataset(data_dir / "train.csv", kind="train", audit=True)
    if len(train) != config["full_fit"]["train_rows_expected"]:
        raise RuntimeError("unexpected full-train row count")
    offline_bundle = load_or_build_features(train, offline_config, kind="train", use_cache=True)
    causal_bundle = load_or_build_features(train, causal_config, kind="train", use_cache=True)
    x_selection = json.loads(
        _resolve(config["paths"]["incumbent_selection"]).read_text(encoding="utf-8")
    )
    c_selection = json.loads(
        _resolve(config["paths"]["causal_selection"]).read_text(encoding="utf-8")
    )
    x_model = train_full_model(train, offline_bundle, offline_config, x_selection)
    c_model = train_full_model(train, causal_bundle, causal_config, c_selection)
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=False)
    x_path = model_dir / "offline_xgboost_full.joblib"
    c_path = model_dir / "causal_lightgbm_full.joblib"
    joblib.dump(x_model, x_path, compress=3)
    joblib.dump(c_model, c_path, compress=3)
    ensemble = P1ImprovementEnsemble(
        incumbent_model=x_model,
        causal_model=c_model,
        winner_branch=winner_name,
        deployment_parameters=dict(winner_report["deployment_parameters"]),
        logistic_model=deployment_model,
        config_sha256=sha256_file(config_path),
    )
    ensemble_path = model_dir / "p1_improved_ensemble.joblib"
    joblib.dump(ensemble, ensemble_path, compress=3)

    test = load_dataset(data_dir / "test.csv", kind="test", audit=True)
    offline_test_bundle = load_or_build_features(test, offline_config, kind="test", use_cache=True)
    causal_test_bundle = load_or_build_features(test, causal_config, kind="test", use_cache=True)
    candidate = _predict_ensemble(ensemble, test, offline_test_bundle, causal_test_bundle)
    candidate_dir = root / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = candidate_dir / "P1_IMPROVED_ENSEMBLE_V1.csv"
    if candidate_path.exists():
        raise FileExistsError(candidate_path)
    write_submission(candidate, candidate_path)
    validation = validate_submission(candidate_path, test)
    if len(candidate) != config["full_fit"]["candidate_rows_expected"]:
        raise RuntimeError("unexpected candidate row count")

    loaded = joblib.load(ensemble_path)
    reproduced = _predict_ensemble(loaded, test, offline_test_bundle, causal_test_bundle)
    reproduced_path = candidate_dir / "reproduced.csv"
    write_submission(reproduced, reproduced_path)
    if candidate_path.read_bytes() != reproduced_path.read_bytes():
        raise RuntimeError("saved ensemble did not reproduce candidate bytes")
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "model_fit_count": 2 + int(deployment_model is not None),
        "models": {
            "offline_xgboost_full": {"path": str(x_path), "sha256": sha256_file(x_path)},
            "causal_lightgbm_full": {"path": str(c_path), "sha256": sha256_file(c_path)},
            "ensemble": {"path": str(ensemble_path), "sha256": sha256_file(ensemble_path)},
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "rows": len(candidate),
            "positive_rows": int(candidate["label"].sum()),
            "positive_rate": float(candidate["label"].mean()),
        },
        "reproduction": {
            "path": str(reproduced_path),
            "sha256": sha256_file(reproduced_path),
            "byte_identical": True,
        },
        "strict_validation": validation,
    }


def run_cycle(config_path: str | Path) -> dict[str, Any]:
    resolved_config, config = load_cycle_config(config_path)
    root = _resolve(config["paths"]["artifact_root"])
    seal = _verify_seal(resolved_config, config, root)
    lock = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "started_at_kst": datetime.now(KST).isoformat(),
        "pid": os.getpid(),
        "seal_sha256": sha256_file(root / "preexecution_seal.json"),
        "execution_ordinal": 1,
    }
    write_json_exclusive(root / "attempt_lock.json", lock)
    frozen_before = sha256_file(_resolve(config["paths"]["frozen_submission"]))
    source_before = {
        "train_csv": sha256_file(_resolve(config["paths"]["train_csv"])),
        "test_csv": sha256_file(_resolve(config["paths"]["test_csv"])),
    }
    try:
        frame, truth, x_pred, c_pred, px, pc = _aligned_oof(config)
        branch_reports: list[dict[str, Any]] = []
        winner_name: str | None = None
        winner_prediction: np.ndarray | None = None
        winner_report: dict[str, Any] | None = None
        deployment_model: LogisticRegression | None = None
        for name in config["branch_order"]:
            prediction, report, model = _evaluate_branch(
                name, config, frame, truth, x_pred, c_pred, px, pc
            )
            branch_reports.append(report)
            if report["guard"]["passed"]:
                winner_name = name
                winner_prediction = prediction
                winner_report = report
                deployment_model = model
                break
        if winner_name is None or winner_prediction is None or winner_report is None:
            failure = {
                "status": "NO_WINNER_WITHIN_PREREGISTERED_BRANCHES",
                "branch_reports": branch_reports,
            }
            write_json(root / "failure.json", failure)
            raise RuntimeError("all three preregistered branches failed the success guards")

        oof = frame.loc[:, [*config["key_columns"], "fold"]].copy()
        oof["label"] = truth
        oof["incumbent_prediction"] = x_pred
        oof["candidate_prediction"] = winner_prediction
        oof_path = root / "winner_oof.parquet"
        oof.to_parquet(oof_path, index=False, compression="zstd")
        bootstrap = _bootstrap_day_delta(
            frame,
            truth,
            x_pred,
            winner_prediction,
            replicates=int(config["bootstrap"]["replicates"]),
            seed=int(config["bootstrap"]["seed"]),
        )
        full_fit = _full_fit_and_candidate(
            resolved_config,
            config,
            root,
            winner_name,
            winner_report,
            deployment_model,
        )
        frozen_after = sha256_file(_resolve(config["paths"]["frozen_submission"]))
        source_after = {
            "train_csv": sha256_file(_resolve(config["paths"]["train_csv"])),
            "test_csv": sha256_file(_resolve(config["paths"]["test_csv"])),
        }
        if frozen_before != frozen_after or source_before != source_after:
            raise RuntimeError("protected source/frozen hash changed")
        metrics = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "winner": winner_name,
            "branch_reports": branch_reports,
            "winner_guard": winner_report["guard"],
            "bootstrap": bootstrap,
            "outer_rows": len(frame),
            "outer_key_duplicates": int(frame.duplicated(config["key_columns"]).sum()),
            "outer_key_alignment_exact": True,
            "target_fold_label_reads_before_prediction": 0,
            "deployment_resubstitution_is_not_outer_metric": True,
        }
        write_json(root / "metrics.json", metrics)
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": "COMPLETE_WINNER_TRAINED_CANDIDATE_REPRODUCED_NOT_UPLOADED",
            "completed_at_kst": datetime.now(KST).isoformat(),
            "winner": winner_name,
            "outer_incumbent_f1": winner_report["guard"]["incumbent"]["f1"],
            "outer_candidate_f1": winner_report["guard"]["candidate"]["f1"],
            "outer_f1_delta": winner_report["guard"]["micro_f1_delta"],
            "guard_passed": winner_report["guard"]["passed"],
            "bootstrap": bootstrap,
            "full_fit": full_fit,
            "protected_hashes": {
                "source_before": source_before,
                "source_after": source_after,
                "frozen_before": frozen_before,
                "frozen_after": frozen_after,
            },
            "operation_counters": {
                "outer_branch_evaluations": len(branch_reports),
                "full_base_model_fits": 2,
                "test_label_reads": 0,
                "submission_uploads": 0,
                "source_mutations": 0,
                "frozen_submission_mutations": 0,
            },
        }
        write_json(root / "result.json", result)
        manifest = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "created_at_kst": datetime.now(KST).isoformat(),
            "config_sha256": sha256_file(resolved_config),
            "seal_sha256": sha256_file(root / "preexecution_seal.json"),
            "attempt_lock_sha256": sha256_file(root / "attempt_lock.json"),
            "implementation_sha256": seal["implementation_sha256"],
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "artifacts": {},
        }
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                manifest["artifacts"][str(path.relative_to(PROJECT_ROOT))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        write_json(root / "manifest.json", manifest)
        return result
    except Exception as exc:
        failure_path = root / "execution_failure.json"
        if not failure_path.exists():
            write_json(
                failure_path,
                {
                    "status": "FAILED",
                    "failed_at_kst": datetime.now(KST).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "submission_uploads": 0,
                },
            )
        raise


__all__ = [
    "P1ImprovementEnsemble",
    "binary_metrics",
    "causal_event_rescue",
    "load_cycle_config",
    "run_cycle",
    "seal_preexecution",
]
