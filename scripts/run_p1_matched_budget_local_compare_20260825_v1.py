"""Run the sealed P1 matched-budget local comparison.

This program is intentionally OOF-only.  It does not resolve or read P1_DATA_DIR,
official test data, sample submissions, frozen submissions, ERA5, or leaderboards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_qc.improvement_cycle import causal_event_rescue
from p1_qc.pipeline import apply_postprocess


KST = ZoneInfo("Asia/Seoul")
KEY_COLUMNS = ("station", "year", "layer", "time")
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/experiments/p1_matched_budget_local_compare_20260825_v1.json"
)
FAMILY_ORDER = (
    "incumbent_offline_xgboost",
    "causal_event_rescue_ensemble",
    "event_day_balanced_lightgbm",
)


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object: {path}")
    return parsed


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip())
        handle.write("\n")


def _write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial output: {temporary}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _resolve(relative: str, *, must_exist: bool = True) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"path escapes repository: {relative}")
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _artifact_dir(config: Mapping[str, Any]) -> Path:
    path = (PROJECT_ROOT / str(config["artifact_dir"])).resolve()
    if not path.is_relative_to(PROJECT_ROOT / "artifacts"):
        raise RuntimeError("artifact_dir must remain under repository artifacts")
    return path


def _iter_input_specs(config: Mapping[str, Any]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = [dict(config["common_protocol"])]
    inputs = config["inputs"]
    for key in ("incumbent_oof", "causal_oof", "historical_round_a_oof"):
        values.append(dict(inputs[key]))
    values.extend(dict(value) for value in inputs["full_prefix_parts"])
    return values


def _validate_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p1_matched_budget_local_compare.v1":
        raise ValueError("unexpected schema version")
    if config.get("experiment_id") != "p1_matched_budget_local_compare_20260825_v1":
        raise ValueError("unexpected experiment id")
    surface = config["surface"]
    if tuple(surface["key_columns"]) != KEY_COLUMNS:
        raise ValueError("key contract changed")
    if list(surface["fold_order"]) != ["2025_q2", "2025_q3", "2025_q4"]:
        raise ValueError("fold order changed")
    if list(surface["seeds"]) != [20260813, 20260829, 20260847]:
        raise ValueError("seed budget changed")
    if int(config["matched_tuning"]["settings_per_family"]) != 3:
        raise ValueError("setting budget changed")
    if int(config["matched_tuning"]["seed_count_per_family"]) != 3:
        raise ValueError("seed count changed")
    families = config["families"]
    incumbent_settings = families["incumbent_offline_xgboost"]["settings"]
    rescue_settings = families["causal_event_rescue_ensemble"]["settings"]
    if len(incumbent_settings) != 3 or len(rescue_settings) != 3:
        raise ValueError("each family must have exactly three settings")
    if families["event_day_balanced_lightgbm"]["settings_from_family"] != (
        "incumbent_offline_xgboost"
    ):
        raise ValueError("shared postprocess opportunity changed")
    if int(config["metrics"]["bootstrap"]["replicates"]) != 5000:
        raise ValueError("bootstrap budget changed")
    forbidden = tuple(str(value).lower() for value in config["prohibitions"]["read_paths_containing"])
    for spec in _iter_input_specs(config):
        relative = str(spec["path"]).replace("\\", "/").lower()
        if any(token in relative for token in forbidden):
            raise RuntimeError(f"forbidden input path in sealed config: {relative}")


def _verify_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for spec in _iter_input_specs(config):
        path = _resolve(str(spec["path"]))
        digest = _sha256(path)
        if digest != str(spec["sha256"]):
            raise RuntimeError(f"immutable input drift: {spec['path']}")
        observed[str(spec["path"])] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return observed


def binary_metrics(truth: Sequence[int], prediction: Sequence[int]) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("truth and prediction must be equal one-dimensional vectors")
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("binary vectors required")
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denominator = 2 * tp + fp + fn
    return {
        "f1": float(2 * tp / denominator if denominator else 0.0),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": int(tp + fp),
        "positive_support": int(tp + fn),
        "rows": int(len(y)),
    }


def _postprocess_by_fold(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    plateau: np.ndarray,
    spike: np.ndarray,
    profile_schedule: Mapping[str, Mapping[str, Any]],
    fold_order: Sequence[str],
) -> np.ndarray:
    result = np.zeros(len(frame), dtype=np.int8)
    for fold in fold_order:
        mask = frame["fold"].eq(fold).to_numpy()
        fold_frame = frame.loc[mask, ["station", "layer", "time"]].reset_index(drop=True)
        result[mask] = apply_postprocess(
            fold_frame,
            np.asarray(probabilities)[mask],
            np.asarray(plateau)[mask],
            np.asarray(spike)[mask],
            profile_schedule[fold],
        )
    return result


def _rescue_by_fold(
    frame: pd.DataFrame,
    incumbent_prediction: np.ndarray,
    causal_prediction: np.ndarray,
    incumbent_probability: np.ndarray,
    causal_probability: np.ndarray,
    setting: Mapping[str, Any],
    fold_order: Sequence[str],
) -> np.ndarray:
    result = np.zeros(len(frame), dtype=np.int8)
    for fold in fold_order:
        mask = frame["fold"].eq(fold).to_numpy()
        result[mask] = causal_event_rescue(
            frame.loc[mask, ["station", "layer", "time"]].reset_index(drop=True),
            np.asarray(incumbent_prediction)[mask],
            np.asarray(causal_prediction)[mask],
            np.asarray(incumbent_probability)[mask],
            np.asarray(causal_probability)[mask],
            causal_floor=float(setting["causal_floor"]),
            incumbent_floor=float(setting["incumbent_floor"]),
        )
    return result


def _load_surface(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    incumbent = pd.read_parquet(_resolve(str(inputs["incumbent_oof"]["path"])))
    causal = pd.read_parquet(_resolve(str(inputs["causal_oof"]["path"])))
    historical_a = pd.read_parquet(_resolve(str(inputs["historical_round_a_oof"]["path"])))
    required = {*KEY_COLUMNS, "label", "fold", "probability", "prediction", "plateau", "spike_candidate"}
    if not required.issubset(incumbent.columns) or not required.issubset(causal.columns):
        raise RuntimeError("sealed OOF schema is incomplete")
    expected_rows = int(config["surface"]["expected_rows"])
    if len(incumbent) != expected_rows or len(causal) != expected_rows:
        raise RuntimeError("unexpected OOF row count")
    if incumbent.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("duplicate incumbent OOF keys")
    if not incumbent.loc[:, list(KEY_COLUMNS)].equals(causal.loc[:, list(KEY_COLUMNS)]):
        raise RuntimeError("causal OOF key/order mismatch")
    if not incumbent["label"].equals(causal["label"]):
        raise RuntimeError("causal OOF truth mismatch")
    if not incumbent.loc[:, list(KEY_COLUMNS)].equals(historical_a.loc[:, list(KEY_COLUMNS)]):
        raise RuntimeError("historical Round A key/order mismatch")
    if not incumbent["label"].equals(historical_a["label"]):
        raise RuntimeError("historical Round A truth mismatch")

    parts = []
    for spec in inputs["full_prefix_parts"]:
        part = pd.read_parquet(_resolve(str(spec["path"])))
        if len(part) == 0 or not part["fold"].eq(str(spec["fold"])).all():
            raise RuntimeError(f"invalid full-prefix part: {spec['path']}")
        if not np.isclose(part["fraction"].to_numpy(dtype=float), 1.0).all():
            raise RuntimeError("non-full prefix part supplied")
        parts.append(part)
    saved = pd.concat(parts, ignore_index=True)
    if len(saved) != expected_rows or saved.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("full-prefix prediction parts do not cover exact OOF keys")
    saved = incumbent.loc[:, list(KEY_COLUMNS)].merge(
        saved,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if saved.isna().all(axis=1).any() or len(saved) != expected_rows:
        raise RuntimeError("full-prefix prediction alignment failed")
    if not saved["fold"].astype(str).equals(incumbent["fold"].astype(str)):
        raise RuntimeError("full-prefix fold assignment differs from OOF")

    seeds = [int(value) for value in config["surface"]["seeds"]]
    required_saved = {"plateau", "spike_candidate"}
    for seed in seeds:
        required_saved.update(
            {
                f"baseline__seed_{seed}__probability",
                f"event_day_balanced_binary_lgbm__seed_{seed}__probability",
            }
        )
    missing = sorted(required_saved.difference(saved.columns))
    if missing:
        raise RuntimeError(f"saved prediction columns missing: {missing}")

    frame = incumbent.loc[:, [*KEY_COLUMNS, "fold"]].copy()
    return {
        "frame": frame,
        "truth": incumbent["label"].to_numpy(dtype=np.int8),
        "plateau": saved["plateau"].to_numpy(dtype=bool),
        "spike": saved["spike_candidate"].to_numpy(dtype=bool),
        "incumbent_probabilities": {
            seed: saved[f"baseline__seed_{seed}__probability"].to_numpy(dtype=float)
            for seed in seeds
        },
        "event_probabilities": {
            seed: saved[
                f"event_day_balanced_binary_lgbm__seed_{seed}__probability"
            ].to_numpy(dtype=float)
            for seed in seeds
        },
        "causal_probability": causal["probability"].to_numpy(dtype=float),
        "causal_prediction": causal["prediction"].to_numpy(dtype=np.int8),
        "historical_incumbent_prediction": incumbent["prediction"].to_numpy(dtype=np.int8),
        "historical_round_a_prediction": historical_a["candidate_prediction"].to_numpy(
            dtype=np.int8
        ),
    }


def _build_family_predictions(
    config: Mapping[str, Any], surface: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    frame = surface["frame"]
    plateau = surface["plateau"]
    spike = surface["spike"]
    folds = list(config["surface"]["fold_order"])
    seeds = [int(value) for value in config["surface"]["seeds"]]
    profile_settings = config["families"]["incumbent_offline_xgboost"]["settings"]
    output: dict[str, dict[str, dict[str, Any]]] = {
        family: {} for family in FAMILY_ORDER
    }

    for family, probability_key in (
        ("incumbent_offline_xgboost", "incumbent_probabilities"),
        ("event_day_balanced_lightgbm", "event_probabilities"),
    ):
        probabilities_by_seed = surface[probability_key]
        mean_probability = np.mean(
            np.vstack([probabilities_by_seed[seed] for seed in seeds]), axis=0
        )
        for setting in profile_settings:
            setting_id = str(setting["id"])
            ensemble = _postprocess_by_fold(
                frame,
                mean_probability,
                plateau,
                spike,
                setting["profiles"],
                folds,
            )
            seed_predictions = {
                seed: _postprocess_by_fold(
                    frame,
                    probabilities_by_seed[seed],
                    plateau,
                    spike,
                    setting["profiles"],
                    folds,
                )
                for seed in seeds
            }
            output[family][setting_id] = {
                "ensemble": ensemble,
                "seeds": seed_predictions,
            }

    incumbent_default = str(
        config["families"]["incumbent_offline_xgboost"]["default_setting"]
    )
    default_profile = next(
        value["profiles"] for value in profile_settings if value["id"] == incumbent_default
    )
    incumbent_probabilities = surface["incumbent_probabilities"]
    mean_incumbent_probability = np.mean(
        np.vstack([incumbent_probabilities[seed] for seed in seeds]), axis=0
    )
    incumbent_base_mean = _postprocess_by_fold(
        frame,
        mean_incumbent_probability,
        plateau,
        spike,
        default_profile,
        folds,
    )
    incumbent_base_seed = {
        seed: _postprocess_by_fold(
            frame,
            incumbent_probabilities[seed],
            plateau,
            spike,
            default_profile,
            folds,
        )
        for seed in seeds
    }
    for setting in config["families"]["causal_event_rescue_ensemble"]["settings"]:
        setting_id = str(setting["id"])
        ensemble = _rescue_by_fold(
            frame,
            incumbent_base_mean,
            surface["causal_prediction"],
            mean_incumbent_probability,
            surface["causal_probability"],
            setting,
            folds,
        )
        seed_predictions = {
            seed: _rescue_by_fold(
                frame,
                incumbent_base_seed[seed],
                surface["causal_prediction"],
                incumbent_probabilities[seed],
                surface["causal_probability"],
                setting,
                folds,
            )
            for seed in seeds
        }
        output["causal_event_rescue_ensemble"][setting_id] = {
            "ensemble": ensemble,
            "seeds": seed_predictions,
        }
    return output


def _select_best_setting(
    setting_predictions: Mapping[str, np.ndarray],
    setting_order: Sequence[str],
    truth: np.ndarray,
    mask: np.ndarray,
) -> tuple[str, list[dict[str, Any]]]:
    scores = []
    for order, setting in enumerate(setting_order):
        score = binary_metrics(truth[mask], setting_predictions[setting][mask])["f1"]
        scores.append({"setting": setting, "order": order, "f1": float(score)})
    selected = max(scores, key=lambda item: (item["f1"], -item["order"]))["setting"]
    return str(selected), scores


def _compose_modes(
    config: Mapping[str, Any],
    family: str,
    family_predictions: Mapping[str, Mapping[str, Any]],
    frame: pd.DataFrame,
    truth: np.ndarray,
) -> dict[str, Any]:
    folds = list(config["surface"]["fold_order"])
    seeds = [int(value) for value in config["surface"]["seeds"]]
    family_config = config["families"][family]
    if family == "event_day_balanced_lightgbm":
        setting_order = [
            str(value["id"])
            for value in config["families"]["incumbent_offline_xgboost"]["settings"]
        ]
    else:
        setting_order = [str(value["id"]) for value in family_config["settings"]]
    default_setting = str(family_config["default_setting"])
    ensemble_by_setting = {
        setting: np.asarray(family_predictions[setting]["ensemble"], dtype=np.int8)
        for setting in setting_order
    }
    default_prediction = ensemble_by_setting[default_setting].copy()
    tuned_prediction = np.zeros(len(frame), dtype=np.int8)
    tuned_seed_predictions = {
        seed: np.zeros(len(frame), dtype=np.int8) for seed in seeds
    }
    selections: dict[str, Any] = {}
    for fold_index, fold in enumerate(folds):
        target = frame["fold"].eq(fold).to_numpy()
        if fold_index == 0:
            selected = default_setting
            score_table: list[dict[str, Any]] = []
            calibration_folds: list[str] = []
        else:
            calibration_folds = folds[:fold_index]
            calibration = frame["fold"].isin(calibration_folds).to_numpy()
            selected, score_table = _select_best_setting(
                ensemble_by_setting,
                setting_order,
                truth,
                calibration,
            )
        tuned_prediction[target] = ensemble_by_setting[selected][target]
        for seed in seeds:
            tuned_seed_predictions[seed][target] = np.asarray(
                family_predictions[selected]["seeds"][seed], dtype=np.int8
            )[target]
        selections[fold] = {
            "selected_setting": selected,
            "calibration_folds": calibration_folds,
            "calibration_scores": score_table,
            "target_label_reads_before_prediction": 0,
        }
    oracle_setting, oracle_scores = _select_best_setting(
        ensemble_by_setting,
        setting_order,
        truth,
        np.ones(len(frame), dtype=bool),
    )
    return {
        "setting_order": setting_order,
        "default_setting": default_setting,
        "default": default_prediction,
        "walk_forward_tuned": tuned_prediction,
        "oracle_upper_bound": ensemble_by_setting[oracle_setting].copy(),
        "walk_forward_selections": selections,
        "oracle_setting": oracle_setting,
        "oracle_scores": oracle_scores,
        "seed_predictions": {
            "default": {
                seed: np.asarray(family_predictions[default_setting]["seeds"][seed], dtype=np.int8)
                for seed in seeds
            },
            "walk_forward_tuned": tuned_seed_predictions,
            "oracle_upper_bound": {
                seed: np.asarray(family_predictions[oracle_setting]["seeds"][seed], dtype=np.int8)
                for seed in seeds
            },
        },
    }


def _slice_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    frame: pd.DataFrame,
    column: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in pd.unique(frame[column]):
        mask = frame[column].eq(value).to_numpy()
        output[str(value)] = binary_metrics(truth[mask], prediction[mask])
    return output


def normal_fp_day_metrics(
    truth: Sequence[int], prediction: Sequence[int], frame: pd.DataFrame
) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    parsed = pd.to_datetime(frame["time"], utc=True, format="mixed", errors="raise")
    normal = y == 0
    work = frame.loc[normal, ["station", "layer"]].reset_index(drop=True).copy()
    work["kst_day"] = parsed[normal].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d").to_numpy()
    fp = p[normal] == 1
    blocks = int(work.drop_duplicates(["station", "layer", "kst_day"]).shape[0])
    rows = int(fp.sum())
    days = int(work.loc[fp, ["station", "layer", "kst_day"]].drop_duplicates().shape[0])
    return {
        "normal_rows": int(normal.sum()),
        "normal_station_layer_kst_days": blocks,
        "false_positive_rows": rows,
        "false_positive_rows_per_normal_station_layer_kst_day": rows / blocks if blocks else None,
        "normal_station_layer_kst_days_with_fp": days,
        "normal_station_layer_kst_day_fp_incidence": days / blocks if blocks else None,
    }


def paired_kst_day_bootstrap(
    truth: Sequence[int],
    candidate: Sequence[int],
    baseline: Sequence[int],
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.int8)
    c = np.asarray(candidate, dtype=np.int8)
    b = np.asarray(baseline, dtype=np.int8)
    if y.shape != c.shape or y.shape != b.shape:
        raise ValueError("paired bootstrap inputs differ")
    day = pd.to_datetime(frame["time"], utc=True, format="mixed", errors="raise").dt.tz_convert(
        "Asia/Seoul"
    ).dt.strftime("%Y-%m-%d")
    codes, uniques = pd.factorize(day, sort=True)
    counts = np.zeros((len(uniques), 6), dtype=np.int64)
    for block in range(len(uniques)):
        mask = codes == block
        bm = binary_metrics(y[mask], b[mask])
        cm = binary_metrics(y[mask], c[mask])
        counts[block] = (bm["tp"], bm["fp"], bm["fn"], cm["tp"], cm["fp"], cm["fn"])
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        selected = counts[rng.integers(0, len(counts), size=len(counts))].sum(axis=0)
        btp, bfp, bfn, ctp, cfp, cfn = selected
        baseline_denominator = 2 * btp + bfp + bfn
        candidate_denominator = 2 * ctp + cfp + cfn
        baseline_f1 = 2 * btp / baseline_denominator if baseline_denominator else 0.0
        candidate_f1 = 2 * ctp / candidate_denominator if candidate_denominator else 0.0
        differences[replicate] = candidate_f1 - baseline_f1
    quantiles = np.quantile(differences, [0.05, 0.5, 0.95])
    return {
        "unit": "KST calendar day",
        "days": int(len(uniques)),
        "replicates": int(replicates),
        "seed": int(seed),
        "delta_mean": float(differences.mean()),
        "delta_median": float(quantiles[1]),
        "delta_ci90": [float(quantiles[0]), float(quantiles[2])],
        "probability_delta_positive": float(np.mean(differences > 0)),
    }


def _seed_summary(
    truth: np.ndarray, seed_predictions: Mapping[int, np.ndarray]
) -> dict[str, Any]:
    values = {
        str(seed): binary_metrics(truth, prediction)["f1"]
        for seed, prediction in seed_predictions.items()
    }
    vector = np.asarray(list(values.values()), dtype=float)
    return {
        "by_seed_f1": values,
        "mean_f1": float(vector.mean()),
        "population_std_f1": float(vector.std(ddof=0)),
        "minimum_f1": float(vector.min()),
        "maximum_f1": float(vector.max()),
    }


def _relative_increase(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return (value - baseline) / baseline


def _evaluate(
    config: Mapping[str, Any], surface: Mapping[str, Any], modes: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], pd.DataFrame]:
    truth = surface["truth"]
    frame = surface["frame"]
    bootstrap_config = config["metrics"]["bootstrap"]
    family_metrics: dict[str, Any] = {}
    predictions = frame.loc[:, [*KEY_COLUMNS, "fold"]].copy()
    for family in FAMILY_ORDER:
        family_mode = modes[family]
        mode_metrics: dict[str, Any] = {}
        for mode in ("default", "walk_forward_tuned", "oracle_upper_bound"):
            prediction = np.asarray(family_mode[mode], dtype=np.int8)
            predictions[f"{family}__{mode}"] = prediction
            for seed, seed_prediction in family_mode["seed_predictions"][mode].items():
                predictions[f"{family}__{mode}__seed_{seed}"] = np.asarray(
                    seed_prediction, dtype=np.int8
                )
            mode_metrics[mode] = {
                "pooled": binary_metrics(truth, prediction),
                "by_fold": _slice_metrics(truth, prediction, frame, "fold"),
                "by_station": _slice_metrics(truth, prediction, frame, "station"),
                "normal_fp_day": normal_fp_day_metrics(truth, prediction, frame),
                "seed_robustness": _seed_summary(
                    truth, family_mode["seed_predictions"][mode]
                ),
            }
        default_f1 = mode_metrics["default"]["pooled"]["f1"]
        tuned_f1 = mode_metrics["walk_forward_tuned"]["pooled"]["f1"]
        oracle_f1 = mode_metrics["oracle_upper_bound"]["pooled"]["f1"]
        mode_metrics["tuning_effect"] = {
            "walk_forward_tuned_minus_default_f1": tuned_f1 - default_f1,
            "oracle_minus_default_f1": oracle_f1 - default_f1,
            "oracle_is_non_promotable": True,
        }
        mode_metrics["selection"] = {
            "walk_forward": family_mode["walk_forward_selections"],
            "oracle_setting": family_mode["oracle_setting"],
            "oracle_setting_scores": family_mode["oracle_scores"],
        }
        family_metrics[family] = mode_metrics

    incumbent_default_f1 = family_metrics["incumbent_offline_xgboost"]["default"]["pooled"]["f1"]
    incumbent_tuned_prediction = np.asarray(
        modes["incumbent_offline_xgboost"]["walk_forward_tuned"], dtype=np.int8
    )
    incumbent_tuned_f1 = family_metrics["incumbent_offline_xgboost"]["walk_forward_tuned"][
        "pooled"
    ]["f1"]
    comparisons: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        candidate_default_f1 = family_metrics[family]["default"]["pooled"]["f1"]
        candidate_tuned_f1 = family_metrics[family]["walk_forward_tuned"]["pooled"]["f1"]
        candidate_prediction = np.asarray(modes[family]["walk_forward_tuned"], dtype=np.int8)
        bootstrap_vs_incumbent = paired_kst_day_bootstrap(
            truth,
            candidate_prediction,
            incumbent_tuned_prediction,
            frame,
            replicates=int(bootstrap_config["replicates"]),
            seed=int(bootstrap_config["seed"]),
        )
        bootstrap_tuning = paired_kst_day_bootstrap(
            truth,
            candidate_prediction,
            np.asarray(modes[family]["default"], dtype=np.int8),
            frame,
            replicates=int(bootstrap_config["replicates"]),
            seed=int(bootstrap_config["seed"]) + 101,
        )
        fold_delta = {
            fold: (
                family_metrics[family]["walk_forward_tuned"]["by_fold"][fold]["f1"]
                - family_metrics["incumbent_offline_xgboost"]["walk_forward_tuned"]["by_fold"][
                    fold
                ]["f1"]
            )
            for fold in config["surface"]["fold_order"]
        }
        station_delta = {
            station: (
                family_metrics[family]["walk_forward_tuned"]["by_station"][station]["f1"]
                - family_metrics["incumbent_offline_xgboost"]["walk_forward_tuned"][
                    "by_station"
                ][station]["f1"]
            )
            for station in family_metrics["incumbent_offline_xgboost"]["walk_forward_tuned"][
                "by_station"
            ]
        }
        candidate_fp = family_metrics[family]["walk_forward_tuned"]["normal_fp_day"][
            "false_positive_rows_per_normal_station_layer_kst_day"
        ]
        incumbent_fp = family_metrics["incumbent_offline_xgboost"]["walk_forward_tuned"][
            "normal_fp_day"
        ]["false_positive_rows_per_normal_station_layer_kst_day"]
        relative_fp = _relative_increase(float(candidate_fp), float(incumbent_fp))
        gate = config["interpretation_gates"]["robust_local_win"]
        robust_checks = {
            "pooled_delta_positive": candidate_tuned_f1 - incumbent_tuned_f1 > 0.0,
            "paired_day_ci90_lower_positive": bootstrap_vs_incumbent["delta_ci90"][0] > 0.0,
            "minimum_nonnegative_fold_count": sum(value >= 0.0 for value in fold_delta.values())
            >= int(gate["minimum_nonnegative_fold_count"]),
            "worst_station_regression_within_limit": min(station_delta.values())
            >= -float(gate["maximum_worst_station_f1_regression"]),
            "normal_fp_relative_increase_within_limit": relative_fp
            <= float(gate["maximum_relative_normal_fp_per_day_increase"]),
        }
        default_gap = incumbent_default_f1 - candidate_default_f1
        tuning_gain = candidate_tuned_f1 - candidate_default_f1
        closure_fraction = 1.0 if default_gap <= 0 else tuning_gain / default_gap
        competitive = default_gap <= float(
            config["interpretation_gates"]["competitive_default_absolute_f1_gap_at_most"]
        )
        robust_win = all(robust_checks.values())
        maturity_supported = (
            competitive
            and closure_fraction
            >= float(config["interpretation_gates"]["material_gap_closure_fraction_at_least"])
        ) or robust_win
        comparisons[family] = {
            "default_structure_delta_vs_incumbent_default_f1": candidate_default_f1
            - incumbent_default_f1,
            "walk_forward_tuned_delta_vs_incumbent_tuned_f1": candidate_tuned_f1
            - incumbent_tuned_f1,
            "paired_kst_day_bootstrap_vs_incumbent_tuned": bootstrap_vs_incumbent,
            "paired_kst_day_bootstrap_tuned_vs_own_default": bootstrap_tuning,
            "fold_f1_deltas_vs_incumbent_tuned": fold_delta,
            "station_f1_deltas_vs_incumbent_tuned": station_delta,
            "normal_fp_per_day_relative_increase_vs_incumbent_tuned": relative_fp,
            "robust_local_win_checks": robust_checks,
            "robust_local_win": robust_win,
            "default_gap_to_incumbent_default_f1": default_gap,
            "own_walk_forward_tuning_gain_f1": tuning_gain,
            "default_gap_closure_fraction": closure_fraction,
            "competitive_default": competitive,
            "maturity_bias_supported": maturity_supported,
        }

    metrics = {
        "schema_version": "p1_matched_budget_local_compare.metrics.v1",
        "experiment_id": config["experiment_id"],
        "surface": {
            "rows": len(frame),
            "folds": list(config["surface"]["fold_order"]),
            "stations": sorted(frame["station"].astype(str).unique().tolist()),
            "positive_rows": int(truth.sum()),
            "same_keys_all_families": True,
            "same_three_seeds_all_families": True,
            "same_three_settings_all_families": True,
        },
        "families": family_metrics,
        "comparisons": comparisons,
        "historical_context_not_used_for_selection": {
            "historical_mature_incumbent_oof": binary_metrics(
                truth, surface["historical_incumbent_prediction"]
            ),
            "historical_round_a_walk_forward_oof": binary_metrics(
                truth, surface["historical_round_a_prediction"]
            ),
            "matched_surface_warning": "Historical references used older retained probability/selection surfaces and are context only, not matched-budget winners.",
        },
    }
    return metrics, predictions


def _conclusion(metrics: Mapping[str, Any]) -> dict[str, Any]:
    families = metrics["families"]
    comparisons = metrics["comparisons"]
    tuned = {
        family: float(families[family]["walk_forward_tuned"]["pooled"]["f1"])
        for family in FAMILY_ORDER
    }
    default = {
        family: float(families[family]["default"]["pooled"]["f1"])
        for family in FAMILY_ORDER
    }
    best = max(FAMILY_ORDER, key=lambda family: (tuned[family], -FAMILY_ORDER.index(family)))
    challenger_support = {
        family: bool(comparisons[family]["maturity_bias_supported"])
        for family in FAMILY_ORDER[1:]
    }
    return {
        "matched_default_f1": default,
        "matched_walk_forward_tuned_f1": tuned,
        "best_matched_walk_forward_family": best,
        "robust_local_winners_vs_incumbent": [
            family for family in FAMILY_ORDER[1:] if comparisons[family]["robust_local_win"]
        ],
        "maturity_bias_supported_for_challenger": challenger_support,
        "interpretation": (
            "Maturity bias is supported only for challengers marked true under the sealed gate. "
            "A higher oracle value is diagnostic headroom and is not an honest promotion result."
        ),
    }


def _render_report(metrics: Mapping[str, Any], conclusion: Mapping[str, Any]) -> str:
    labels = {
        "incumbent_offline_xgboost": "기존 offline XGBoost",
        "causal_event_rescue_ensemble": "Round A causal event rescue",
        "event_day_balanced_lightgbm": "Round B event-day LightGBM",
    }
    lines = [
        "# P1 matched-budget 로컬 비교",
        "",
        "## 결론",
        "",
        f"동일 예산 walk-forward 최고 계열은 **{labels[conclusion['best_matched_walk_forward_family']]}**입니다.",
        "공식 테스트·리더보드 수치는 후보 선택이나 튜닝에 사용하지 않았습니다.",
        "",
        "## 동일 예산 결과",
        "",
        "| 계열 | 기본 F1 | walk-forward 튜닝 F1 | 튜닝 이득 | 3설정 oracle F1 | incumbent 대비 튜닝 delta | day CI90 | 성숙도 편향 지지 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for family in FAMILY_ORDER:
        values = metrics["families"][family]
        comparison = metrics["comparisons"][family]
        ci = comparison["paired_kst_day_bootstrap_vs_incumbent_tuned"]["delta_ci90"]
        lines.append(
            "| {label} | {default:.6f} | {tuned:.6f} | {gain:+.6f} | {oracle:.6f} | "
            "{delta:+.6f} | [{low:+.6f}, {high:+.6f}] | {maturity} |".format(
                label=labels[family],
                default=values["default"]["pooled"]["f1"],
                tuned=values["walk_forward_tuned"]["pooled"]["f1"],
                gain=values["tuning_effect"]["walk_forward_tuned_minus_default_f1"],
                oracle=values["oracle_upper_bound"]["pooled"]["f1"],
                delta=comparison["walk_forward_tuned_delta_vs_incumbent_tuned_f1"],
                low=ci[0],
                high=ci[1],
                maturity="YES" if comparison["maturity_bias_supported"] else "NO",
            )
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- 세 계열 모두 동일 421,032 OOF 키, 동일 세 outer fold, 동일 3설정·3시드 예산입니다.",
            "- Round A의 causal 성분은 저장된 단일 시드 OOF만 존재하여 고정했고, incumbent 성분만 3시드입니다. 따라서 Round A seed 분산은 부분적입니다.",
            "- oracle은 전체 outer 라벨을 보고 고른 낙관적 상한이며 후보 승격 근거로 사용할 수 없습니다.",
            "- 과거 완성형 점수는 성숙도 문맥만 제공하며 동일 예산 승자 선택에는 사용하지 않았습니다.",
            "- 결과물은 로컬 OOF 분석뿐이며 제출 CSV를 생성하지 않았습니다.",
        ]
    )
    return "\n".join(lines)


def _implementation_paths() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "validator": PROJECT_ROOT
        / "scripts/validate_p1_matched_budget_local_compare_20260825_v1.py",
        "tests": PROJECT_ROOT / "tests/test_p1_matched_budget_local_compare_20260825_v1.py",
        "causal_event_rescue_module": PROJECT_ROOT / "src/p1_qc/improvement_cycle.py",
        "postprocess_module": PROJECT_ROOT / "src/p1_qc/pipeline.py",
    }


def seal(config_path: Path) -> Path:
    config = _read_json(config_path)
    _validate_contract(config)
    observed_inputs = _verify_inputs(config)
    artifact = _artifact_dir(config)
    if artifact.exists():
        raise FileExistsError(f"artifact namespace already exists: {artifact}")
    implementation = _implementation_paths()
    for path in implementation.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    value = {
        "schema_version": "p1_matched_budget_local_compare.preexecution_seal.v1",
        "experiment_id": config["experiment_id"],
        "sealed_at_kst": _now_kst(),
        "status": "SEALED_BEFORE_ANY_MATCHED_GRID_SCORE",
        "config_sha256": _sha256(config_path),
        "common_protocol_sha256": config["common_protocol"]["sha256"],
        "input_sha256": observed_inputs,
        "implementation_sha256": {
            name: _sha256(path) for name, path in implementation.items()
        },
        "budgets": {
            "families": 3,
            "settings_per_family": 3,
            "seeds_per_family": 3,
            "bootstrap_replicates": 5000,
            "new_model_fits": 0,
            "official_test_reads": 0,
            "submission_files_generated": 0,
            "uploads": 0,
        },
        "result_driven_grid_changes": 0,
        "result_driven_reruns": 0,
    }
    _write_json_new(artifact / "preexecution_seal.json", value)
    return artifact / "preexecution_seal.json"


def _verify_seal(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    seal_path = _artifact_dir(config) / "preexecution_seal.json"
    seal_value = _read_json(seal_path)
    if seal_value["status"] != "SEALED_BEFORE_ANY_MATCHED_GRID_SCORE":
        raise RuntimeError("invalid seal status")
    if seal_value["config_sha256"] != _sha256(config_path):
        raise RuntimeError("config changed after seal")
    if seal_value["common_protocol_sha256"] != config["common_protocol"]["sha256"]:
        raise RuntimeError("common protocol pin changed")
    for name, path in _implementation_paths().items():
        if seal_value["implementation_sha256"][name] != _sha256(path):
            raise RuntimeError(f"implementation changed after seal: {name}")
    observed = _verify_inputs(config)
    if observed != seal_value["input_sha256"]:
        raise RuntimeError("input inventory differs from seal")
    return seal_value


def execute(config_path: Path) -> Path:
    config = _read_json(config_path)
    _validate_contract(config)
    seal_value = _verify_seal(config, config_path)
    artifact = _artifact_dir(config)
    forbidden_outputs = [
        artifact / "metrics.json",
        artifact / "predictions.parquet",
        artifact / "result.json",
        artifact / "report_ko.md",
        artifact / "manifest.json",
    ]
    if any(path.exists() for path in forbidden_outputs):
        raise FileExistsError("matched comparison is append-only and already has outputs")

    surface = _load_surface(config)
    family_predictions = _build_family_predictions(config, surface)
    modes = {
        family: _compose_modes(
            config,
            family,
            family_predictions[family],
            surface["frame"],
            surface["truth"],
        )
        for family in FAMILY_ORDER
    }
    metrics, predictions = _evaluate(config, surface, modes)
    conclusion = _conclusion(metrics)
    seeds = [int(value) for value in config["surface"]["seeds"]]
    for family in FAMILY_ORDER:
        for setting, values in family_predictions[family].items():
            predictions[f"{family}__setting__{setting}"] = np.asarray(
                values["ensemble"], dtype=np.int8
            )
            for seed in seeds:
                predictions[f"{family}__setting__{setting}__seed_{seed}"] = np.asarray(
                    values["seeds"][seed], dtype=np.int8
                )
    predictions_path = artifact / "predictions.parquet"
    _write_parquet_new(predictions_path, predictions)
    metrics_path = artifact / "metrics.json"
    _write_json_new(metrics_path, metrics)
    report_path = artifact / "report_ko.md"
    _write_text_new(report_path, _render_report(metrics, conclusion))
    result = {
        "schema_version": "p1_matched_budget_local_compare.result.v1",
        "experiment_id": config["experiment_id"],
        "completed_at_kst": _now_kst(),
        "status": "COMPLETE_LOCAL_ONLY_AWAITING_INDEPENDENT_QA",
        "conclusion": conclusion,
        "common_protocol_sha256": config["common_protocol"]["sha256"],
        "preexecution_seal_sha256": _sha256(artifact / "preexecution_seal.json"),
        "outputs": {
            "metrics": {"path": str(metrics_path.relative_to(PROJECT_ROOT)), "sha256": _sha256(metrics_path)},
            "predictions": {"path": str(predictions_path.relative_to(PROJECT_ROOT)), "sha256": _sha256(predictions_path)},
            "report": {"path": str(report_path.relative_to(PROJECT_ROOT)), "sha256": _sha256(report_path)},
        },
        "operation_counters": {
            "new_model_fits": 0,
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
            "submission_files_generated": 0,
            "uploads": 0,
            "source_mutations": 0,
            "frozen_or_package_mutations": 0,
            "result_driven_grid_changes": 0,
            "result_driven_reruns": 0,
        },
        "seal_verified_before_scoring": True,
        "seal_status": seal_value["status"],
    }
    result_path = artifact / "result.json"
    _write_json_new(result_path, result)
    manifest = {
        "schema_version": "p1_matched_budget_local_compare.manifest.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now_kst(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "common_protocol": {
            "path": config["common_protocol"]["path"],
            "sha256": config["common_protocol"]["sha256"],
        },
        "config": {
            "path": str(config_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(config_path),
        },
        "implementation_sha256": {
            name: _sha256(path) for name, path in _implementation_paths().items()
        },
        "inputs": _verify_inputs(config),
        "artifacts": {
            str(path.relative_to(PROJECT_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (
                artifact / "preexecution_seal.json",
                metrics_path,
                predictions_path,
                report_path,
                result_path,
            )
        },
    }
    manifest_path = artifact / "manifest.json"
    _write_json_new(manifest_path, manifest)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError("config must remain inside repository")
    output = seal(config_path) if args.seal else execute(config_path)
    print(json.dumps({"status": "ok", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
