"""Run the preregistered exploratory P2 group-balanced residual experiment.

Only ``observations.csv`` and a truth-free historical prediction artifact are
opened.  There is deliberately no official query, sample, submission, hidden
answer, materialization, or upload code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)

EXPERIMENT_ID = "p2_group_balanced_raw_residual_20260901_v8"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
ENGINE_PATH = Path(__file__)
RUNNER_PATH = ENGINE_PATH
RESULT_SCHEMA = "p2.group_balanced_raw_residual.result.20260901.v8"
REPORT_TITLE = "# P2 group-balanced raw-residual exploratory cycle 20260901 v8"
FOLD_ORDER = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")
TARGET_LAYERS = (2, 3, 4)
KEYS = ("time", "layer")


class ContractError(RuntimeError):
    """Raised when the sealed experiment contract drifts."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    objective: str
    conditional: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def canonical_time_ns(values: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Normalize any pandas datetime resolution to UTC nanosecond integers."""

    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns")
    return index.asi8.copy()


def resolve_observations(config: dict[str, Any]) -> Path:
    raw = os.environ.get(config["source_contract"]["environment_variable"])
    if not raw:
        raise ContractError("P2_DATA_DIR is required")
    directory = Path(raw).resolve()
    path = directory / config["source_contract"]["only_source_filename"]
    if path.name != "observations.csv" or not path.is_file():
        raise ContractError("only P2_DATA_DIR/observations.csv may be opened")
    if sha256_file(path) != config["source_contract"]["observations_sha256"]:
        raise ContractError("observations.csv hash drift")
    return path


def registered_window_ids(
    times: pd.Series | pd.DatetimeIndex, windows: list[dict[str, str]]
) -> np.ndarray:
    local = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).tz_convert("Asia/Seoul")
    output = np.full(len(local), "", dtype=object)
    for window in windows:
        start = pd.Timestamp(window["start_inclusive"]).tz_convert("Asia/Seoul")
        end = pd.Timestamp(window["end_exclusive"]).tz_convert("Asia/Seoul")
        selected = (local >= start) & (local < end)
        if np.any(output[selected] != ""):
            raise ContractError("registered training windows overlap")
        output[selected] = window["id"]
    return output


def group_balanced_weights(
    layer: np.ndarray, window: np.ndarray, kst_date: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.DataFrame({"layer": layer, "window": window, "kst_date": kst_date})
    if (frame["window"] == "").any():
        raise ContractError("unregistered row entered group weighting")
    group_keys = sorted(frame[["layer", "window"]].drop_duplicates().itertuples(index=False, name=None))
    if len(group_keys) != 6:
        raise ContractError(f"expected six layer-window groups, got {len(group_keys)}")
    raw = np.zeros(len(frame), dtype=float)
    receipts: dict[str, Any] = {}
    for layer_value, window_value in group_keys:
        group_mask = frame["layer"].eq(layer_value) & frame["window"].eq(window_value)
        days = sorted(frame.loc[group_mask, "kst_date"].unique())
        if not days:
            raise ContractError("empty layer-window group")
        for day in days:
            day_mask = group_mask & frame["kst_date"].eq(day)
            count = int(day_mask.sum())
            raw[day_mask.to_numpy()] = 1.0 / (6.0 * len(days) * count)
        key = f"layer{int(layer_value)}:{window_value}"
        receipts[key] = {
            "rows": int(group_mask.sum()),
            "days": len(days),
            "raw_weight_sum": float(raw[group_mask.to_numpy()].sum()),
        }
    if not (np.isfinite(raw).all() and (raw > 0.0).all()):
        raise ContractError("group weights must be finite and positive")
    weights = raw / raw.mean()
    return weights, {
        "groups": receipts,
        "raw_weight_sum": float(raw.sum()),
        "normalized_mean": float(weights.mean()),
        "normalized_min": float(weights.min()),
        "normalized_max": float(weights.max()),
    }


def strict_gate(
    *,
    pooled_delta: float,
    by_fold: dict[str, dict[str, float]],
    by_layer: dict[str, dict[str, float]],
    bootstrap_ci90_high: float,
    calibrated_expected_points: float,
) -> dict[str, bool]:
    return {
        "pooled_improves": pooled_delta < 0.0,
        "minimum_two_folds_improve": sum(item["delta_rmse"] < 0.0 for item in by_fold.values())
        >= 2,
        "official_like_fold_improves": by_fold["2024_sep_oct"]["delta_rmse"] < 0.0,
        "all_layers_within_0_003C": max(item["delta_rmse"] for item in by_layer.values())
        <= 0.003,
        "pooled_ci90_high_below_zero": bootstrap_ci90_high < 0.0,
        "calibrated_expected_points_gte_0_01": calibrated_expected_points >= 0.01,
    }


def grouped_bootstrap(
    frame: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    work = frame[["fold", "kst_date"]].copy()
    work["truth"] = truth
    work["reference"] = reference
    work["candidate"] = candidate
    pieces = list(work.groupby(["fold", "kst_date"], sort=True, observed=True))
    reference_sse = np.asarray(
        [np.square(part["reference"] - part["truth"]).sum() for _, part in pieces]
    )
    candidate_sse = np.asarray(
        [np.square(part["candidate"] - part["truth"]).sum() for _, part in pieces]
    )
    counts = np.asarray([len(part) for _, part in pieces], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        draw = rng.integers(0, len(pieces), len(pieces))
        denominator = counts[draw].sum()
        deltas[replicate] = math.sqrt(candidate_sse[draw].sum() / denominator) - math.sqrt(
            reference_sse[draw].sum() / denominator
        )
    return {
        "unit": "fold_x_kst_date",
        "groups": len(pieces),
        "replicates": replicates,
        "mean_delta_rmse": float(deltas.mean()),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def official_like_bootstrap(
    frame: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    mask = frame["fold"].eq("2024_sep_oct").to_numpy()
    subset = frame.loc[mask, ["kst_date"]].copy()
    subset["truth"] = truth[mask]
    subset["reference"] = reference[mask]
    subset["candidate"] = candidate[mask]
    pieces = list(subset.groupby("kst_date", sort=True, observed=True))
    ref_sse = np.asarray([np.square(part.reference - part.truth).sum() for _, part in pieces])
    cand_sse = np.asarray([np.square(part.candidate - part.truth).sum() for _, part in pieces])
    counts = np.asarray([len(part) for _, part in pieces], dtype=float)
    rng = np.random.default_rng(seed + 1)
    delta = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        draw = rng.integers(0, len(pieces), len(pieces))
        count = counts[draw].sum()
        delta[replicate] = math.sqrt(cand_sse[draw].sum() / count) - math.sqrt(
            ref_sse[draw].sum() / count
        )
    return {
        "unit": "kst_date",
        "groups": len(pieces),
        "replicates": replicates,
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def make_reference(
    observations: pd.DataFrame, scored: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    stations = observations.loc[:, ["station", "time", "layer"]].drop_duplicates()
    blind = scored.merge(stations, on=["time", "layer"], how="left", validate="one_to_one")
    if blind["station"].isna().any() or blind.duplicated(["time", "layer"]).any():
        raise ContractError("historical scoring keys are not unique and complete")
    local = blind["time"].dt.tz_convert("Asia/Seoul")
    blind["kst_date"] = local.dt.date
    season_bin = ((local.dt.dayofyear - 1) // 14).astype(int)
    active = season_bin.eq(17).to_numpy()
    unprojected = scored["reference"].to_numpy(float) + np.where(
        active,
        scored["candidate"].to_numpy(float) - scored["reference"].to_numpy(float),
        0.0,
    )
    endpoints = public_endpoint_frame(observations)
    reference = project_profiles_vectorized(
        blind[["station", "layer", "time"]], unprojected, endpoints
    ).prediction
    return blind, reference


def fit_predict(
    spec: CandidateSpec,
    config: dict[str, Any],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    sample_weight: np.ndarray,
    query_x: pd.DataFrame,
    query_baseline: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = []
    receipts = []
    parameters = dict(config["training"]["common_lightgbm"])
    parameters["objective"] = spec.objective
    for seed in config["training"]["seeds"]:
        model = lgb.LGBMRegressor(random_state=int(seed), **parameters)
        model.fit(train_x, train_y, sample_weight=sample_weight)
        prediction = np.asarray(model.predict(query_x), dtype=float)
        predictions.append(prediction)
        receipts.append(
            {
                "seed": int(seed),
                "objective": spec.objective,
                "best_iteration": int(model.n_estimators_),
                "feature_count": int(model.n_features_in_),
            }
        )
    residual = np.mean(np.vstack(predictions), axis=0)
    candidate = query_baseline + residual
    if not np.isfinite(candidate).all():
        raise ContractError("candidate prediction is nonfinite")
    return candidate, receipts


def evaluate_candidate(
    spec: CandidateSpec,
    blind: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    reference_rmse = rmse(truth, reference)
    candidate_rmse = rmse(truth, candidate)
    by_fold: dict[str, dict[str, float]] = {}
    by_layer: dict[str, dict[str, float]] = {}
    by_fold_layer: dict[str, dict[str, dict[str, float]]] = {}
    for fold in FOLD_ORDER:
        mask = blind["fold"].eq(fold).to_numpy()
        ref = rmse(truth[mask], reference[mask])
        cand = rmse(truth[mask], candidate[mask])
        by_fold[fold] = {
            "rows": int(mask.sum()),
            "reference_rmse": ref,
            "candidate_rmse": cand,
            "delta_rmse": cand - ref,
        }
        by_fold_layer[fold] = {}
        for layer in TARGET_LAYERS:
            selected = mask & blind["layer"].eq(layer).to_numpy()
            layer_ref = rmse(truth[selected], reference[selected])
            layer_cand = rmse(truth[selected], candidate[selected])
            by_fold_layer[fold][str(layer)] = {
                "rows": int(selected.sum()),
                "reference_rmse": layer_ref,
                "candidate_rmse": layer_cand,
                "delta_rmse": layer_cand - layer_ref,
            }
    for layer in TARGET_LAYERS:
        mask = blind["layer"].eq(layer).to_numpy()
        ref = rmse(truth[mask], reference[mask])
        cand = rmse(truth[mask], candidate[mask])
        by_layer[str(layer)] = {
            "rows": int(mask.sum()),
            "reference_rmse": ref,
            "candidate_rmse": cand,
            "delta_rmse": cand - ref,
        }
    evaluation = config["evaluation"]
    bootstrap = grouped_bootstrap(
        blind,
        truth,
        reference,
        candidate,
        seed=int(evaluation["bootstrap_seed"]),
        replicates=int(evaluation["bootstrap_replicates"]),
    )
    official_like = official_like_bootstrap(
        blind,
        truth,
        reference,
        candidate,
        seed=int(evaluation["bootstrap_seed"]),
        replicates=int(evaluation["bootstrap_replicates"]),
    )
    raw_expected = max(
        0.0,
        -official_like["ci90_high"] * float(evaluation["points_per_rmse_C"]),
    )
    calibrated = raw_expected - float(evaluation["transport_penalty_points"])
    pooled_delta = candidate_rmse - reference_rmse
    checks = strict_gate(
        pooled_delta=pooled_delta,
        by_fold=by_fold,
        by_layer=by_layer,
        bootstrap_ci90_high=bootstrap["ci90_high"],
        calibrated_expected_points=calibrated,
    )
    return {
        "name": spec.name,
        "objective": spec.objective,
        "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "reference_rmse": reference_rmse,
        "candidate_rmse": candidate_rmse,
        "delta_rmse": pooled_delta,
        "by_fold": by_fold,
        "by_layer": by_layer,
        "by_fold_layer": by_fold_layer,
        "bootstrap": bootstrap,
        "official_like_bootstrap": official_like,
        "raw_expected_points_delta": raw_expected,
        "transport_calibrated_expected_points_delta": calibrated,
        "candidate_change_rms_C": float(np.sqrt(np.mean(np.square(candidate - reference)))),
        "candidate_change_abs_max_C": float(np.max(np.abs(candidate - reference))),
        "gate_checks": checks,
        "strict_exploratory_pass": bool(all(checks.values())),
    }


def write_report(result: dict[str, Any]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = [
        REPORT_TITLE,
        "",
        "## 결론",
        "",
        f"상태: `{result['status']}`. 이 결과는 exposed historical surface의 탐색 결과이며 fresh confirmation이 아니다.",
        "",
        "| 후보 | pooled ΔRMSE | Sep-Oct | Jul-Aug | Nov-Dec | raw 예상점수 | transport 보정점수 | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["candidates"]:
        folds = item["by_fold"]
        lines.append(
            "| {name} | {pooled:+.9f} | {sep:+.9f} | {jul:+.9f} | {nov:+.9f} | {raw:+.6f} | {cal:+.6f} | {gate} |".format(
                name=item["name"],
                pooled=item["delta_rmse"],
                sep=folds["2024_sep_oct"]["delta_rmse"],
                jul=folds["2025_jul_aug"]["delta_rmse"],
                nov=folds["2025_nov_dec"]["delta_rmse"],
                raw=item["raw_expected_points_delta"],
                cal=item["transport_calibrated_expected_points_delta"],
                gate="PASS" if item["strict_exploratory_pass"] else "NO_GO",
            )
        )
    lines.extend(
        [
            "",
            "## 구조와 중복 배제",
            "",
            "공개층만으로 만든 고정 55개 특징에서 raw-Celsius 선형보간 잔차를 학습했고, layer×등록창×KST-day가 동일 위험을 갖도록 sample weight를 고정했다. DINEOF/GP/CatBoost, soft benefit gate, PAVA/isotonic, rank·season·bin search는 사용하지 않았다.",
            "",
            "행 제거는 하지 않았다. primary L2가 strict gate에 실패한 경우에만 결과 전에 등록한 L1 robust-loss 후보를 한 번 실행했다.",
            "",
            "## 검증 경계",
            "",
            "2024 Sep-Oct 61일과 2025 Jul-Aug/Nov-Dec는 과거 연구에서 이미 노출됐다. 따라서 fold, layer, paired KST-day bootstrap, 점수 환산은 후보 폐쇄와 우선순위 판단용 탐색 증거일 뿐 독립 확인이 아니다.",
            "",
            f"fits={result['fit_count']}; official/test/sample/query/hidden rows=0; submission CSV=0; upload=0.",
        ]
    )
    (REPORT / "report-source.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, resume_technical: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    if ARTIFACT.exists():
        existing = {path.name for path in ARTIFACT.iterdir()}
        if not resume_technical or existing != {"attempt_lock.json"}:
            raise FileExistsError(ARTIFACT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED":
        raise ContractError("config is not in preregistered state")
    if config["postprocess"] != "NONE":
        raise ContractError("postprocess drift")
    if resume_technical:
        lock = json.loads((ARTIFACT / "attempt_lock.json").read_text(encoding="utf-8"))
        atomic_json(
            ARTIFACT / "technical_recovery.json",
            {
                "reason": "pandas Series date accessor required .dt.date",
                "repair": "replace Series.date with Series.dt.date",
                "original_runner_sha256": lock["runner_sha256"],
                "recovery_runner_sha256": sha256_file(RUNNER_PATH),
                "engine_sha256": sha256_file(ENGINE_PATH),
                "model_fits_before_failure": 0,
                "predictions_before_failure": 0,
                "metrics_before_failure": 0,
                "official_rows_before_failure": 0,
                "candidate_or_parameter_change": False,
            },
        )
    else:
        ARTIFACT.mkdir(parents=True)
        atomic_json(
            ARTIFACT / "attempt_lock.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "config_sha256": sha256_file(CONFIG),
                "runner_sha256": sha256_file(RUNNER_PATH),
                "engine_sha256": sha256_file(ENGINE_PATH),
                "candidate_order": [item["name"] for item in config["candidates"]],
                "fallback_condition_frozen": "primary_strict_gate_fails",
                "result_adaptive_parameter_change": False,
            },
        )
    observations_path = resolve_observations(config)
    scoring_path = ROOT / config["source_contract"]["scoring_frame"]
    if sha256_file(scoring_path) != config["source_contract"]["scoring_frame_sha256"]:
        raise ContractError("truth-free scoring frame hash drift")
    observations = pd.read_csv(
        observations_path,
        dtype={"station": "string", "time": "string"},
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    if observations.duplicated(["time", "layer"]).any():
        raise ContractError("observation time-layer keys are not unique")
    scored = pd.read_parquet(scoring_path)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    if tuple(sorted(scored["fold"].unique())) != tuple(sorted(FOLD_ORDER)):
        raise ContractError("fold set drift")
    blind, reference = make_reference(observations, scored)

    feature_table = build_training_features(observations)
    design = build_normalized_curvature_design(feature_table.frame)
    local = design.keys["time"].dt.tz_convert("Asia/Seoul")
    window_id = registered_window_ids(local, config["training"]["registered_windows_kst"])
    train_mask = window_id != ""
    if int(train_mask.sum()) != int(config["training"]["expected_rows"]):
        raise ContractError(
            f"training row count drift: {int(train_mask.sum())} != {config['training']['expected_rows']}"
        )
    train_dates = np.asarray(local[train_mask].dt.date, dtype=object)
    weights, weight_receipt = group_balanced_weights(
        design.keys.loc[train_mask, "layer"].to_numpy(int),
        window_id[train_mask],
        train_dates,
    )
    train_x = design.features.loc[train_mask].reset_index(drop=True)
    train_y = (design.truth - design.baseline)[train_mask]

    design_index = pd.MultiIndex.from_arrays(
        [canonical_time_ns(design.keys["time"]), design.keys["layer"]], names=KEYS
    )
    query_index = pd.MultiIndex.from_arrays(
        [canonical_time_ns(blind["time"]), blind["layer"]], names=KEYS
    )
    if design_index.has_duplicates:
        raise ContractError("feature design keys are duplicated")
    positions = design_index.get_indexer(query_index)
    if np.any(positions < 0):
        raise ContractError("historical query feature alignment failed")
    query_x = design.features.iloc[positions].reset_index(drop=True)
    query_baseline = design.baseline[positions]

    specs = [
        CandidateSpec(
            name=item["name"],
            objective=item["objective"],
            conditional=item["run"] != "always",
        )
        for item in config["candidates"]
    ]
    candidates: list[dict[str, Any]] = []
    fit_count = 0
    truth = design.truth[positions]
    for spec in specs:
        if spec.conditional and candidates[0]["strict_exploratory_pass"]:
            break
        candidate, receipts = fit_predict(
            spec,
            config,
            train_x,
            train_y,
            weights,
            query_x,
            query_baseline,
        )
        fit_count += len(receipts)
        output = ARTIFACT / f"{spec.name}.npz"
        np.savez_compressed(
            output,
            time_ns=canonical_time_ns(blind["time"]),
            layer=blind["layer"].to_numpy(np.int16),
            reference=reference,
            candidate=candidate,
            fold=blind["fold"].to_numpy(str),
        )
        commitment = {
            "name": spec.name,
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": len(candidate),
            "fit_receipts": receipts,
            "metric_computed_at_commitment": False,
        }
        atomic_json(ARTIFACT / f"{spec.name}.commitment.json", commitment)
        record = evaluate_candidate(spec, blind, truth, reference, candidate, config)
        record["prediction_commitment"] = commitment
        candidates.append(record)

    primary_pass = candidates[0]["strict_exploratory_pass"]
    fallback_executed = len(candidates) == 2
    status = (
        "EXPLORATORY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if any(item["strict_exploratory_pass"] for item in candidates)
        else "EXPLORATORY_NO_GO_BOTH_PREDECLARED_OBJECTIVES"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": fit_count,
        "primary_pass": primary_pass,
        "fallback_executed": fallback_executed,
        "training": {
            "rows": int(train_mask.sum()),
            "feature_count": int(train_x.shape[1]),
            "weight_receipt": weight_receipt,
        },
        "candidates": candidates,
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "historical_scoring_rows_read": int(len(scored)),
            "official_test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_file_rows_read": 0,
            "score_file_rows_read": 0,
            "query_support_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config": sha256_file(CONFIG),
            "runner": sha256_file(RUNNER_PATH),
            "engine": sha256_file(ENGINE_PATH),
            "observations": sha256_file(observations_path),
            "scoring_frame": sha256_file(scoring_path),
        },
    }
    atomic_json(ARTIFACT / "result.json", result)
    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT / "result.json", result)
    write_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-technical", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("Use --execute for the sealed exploratory run")
    print(
        json.dumps(
            run(resume_technical=arguments.resume_technical),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
