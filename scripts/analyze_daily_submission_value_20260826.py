from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "daily_submission_value_20260826_v1"
DELIVERY = Path.home() / "Downloads" / "해양 해커톤 제출용" / "20260826_value_of_information_v1"
SEED = 20260826
BOOTSTRAP_REPLICATES = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    denom = 2 * tp + fp + fn
    return {
        "f1": float(2 * tp / denom if denom else 0.0),
        "precision": float(tp / (tp + fp) if tp + fp else 0.0),
        "recall": float(tp / (tp + fn) if tp + fn else 0.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": int(np.sum(p)),
        "rows": int(len(y)),
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def kst_day(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    return parsed.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")


def bootstrap_f1_delta(
    day: pd.Series,
    truth: np.ndarray,
    candidate: np.ndarray,
    incumbent: np.ndarray,
) -> dict[str, float | int | list[float]]:
    work = pd.DataFrame(
        {
            "day": day.to_numpy(),
            "truth": np.asarray(truth, dtype=np.int8),
            "candidate": np.asarray(candidate, dtype=np.int8),
            "incumbent": np.asarray(incumbent, dtype=np.int8),
        }
    )
    rows: list[list[int]] = []
    for _, group in work.groupby("day", sort=True):
        y = group["truth"].to_numpy()
        c = group["candidate"].to_numpy()
        i = group["incumbent"].to_numpy()
        rows.append(
            [
                int(np.sum((y == 1) & (c == 1))),
                int(np.sum((y == 0) & (c == 1))),
                int(np.sum((y == 1) & (c == 0))),
                int(np.sum((y == 1) & (i == 1))),
                int(np.sum((y == 0) & (i == 1))),
                int(np.sum((y == 1) & (i == 0))),
            ]
        )
    blocks = np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = blocks[rng.integers(0, len(blocks), size=len(blocks))].sum(axis=0)
        cdenom = 2 * sample[0] + sample[1] + sample[2]
        idenom = 2 * sample[3] + sample[4] + sample[5]
        deltas[index] = (2 * sample[0] / cdenom) - (2 * sample[3] / idenom)
    return {
        "unit": "KST calendar day",
        "days": int(len(blocks)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "delta_mean": float(np.mean(deltas)),
        "delta_ci90": [float(v) for v in np.quantile(deltas, [0.05, 0.95])],
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }


def bootstrap_rmse_delta(
    day: pd.Series,
    truth: np.ndarray,
    candidate: np.ndarray,
    incumbent: np.ndarray,
) -> dict[str, float | int | list[float]]:
    work = pd.DataFrame(
        {
            "day": day.to_numpy(),
            "candidate_se": np.square(np.asarray(candidate) - np.asarray(truth)),
            "incumbent_se": np.square(np.asarray(incumbent) - np.asarray(truth)),
        }
    )
    blocks = (
        work.groupby("day", sort=True)
        .agg(candidate_sse=("candidate_se", "sum"), incumbent_sse=("incumbent_se", "sum"), rows=("day", "size"))
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = blocks[rng.integers(0, len(blocks), size=len(blocks))].sum(axis=0)
        deltas[index] = math.sqrt(sample[0] / sample[2]) - math.sqrt(sample[1] / sample[2])
    return {
        "unit": "KST calendar day",
        "days": int(len(blocks)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "delta_mean": float(np.mean(deltas)),
        "delta_ci90": [float(v) for v in np.quantile(deltas, [0.05, 0.95])],
        "probability_delta_negative": float(np.mean(deltas < 0)),
    }


def p1_analysis() -> tuple[dict, Path]:
    prediction_path = ROOT / "artifacts" / "p1_matched_budget_local_compare_20260825_v1" / "predictions.parquet"
    truth_path = ROOT / "artifacts" / "runs" / "20260813T153038+0900_cv_378a4e89" / "oof.parquet"
    predictions = pd.read_parquet(prediction_path)
    truth_frame = pd.read_parquet(truth_path, columns=["station", "year", "layer", "time", "label", "fold"])
    keys = ["station", "year", "layer", "time"]
    if not predictions[keys].equals(truth_frame[keys]):
        raise RuntimeError("P1 OOF key order mismatch")
    if not predictions["fold"].astype(str).equals(truth_frame["fold"].astype(str)):
        raise RuntimeError("P1 OOF fold mismatch")

    y = truth_frame["label"].to_numpy(dtype=np.int8)
    incumbent = predictions["incumbent_offline_xgboost__default"].to_numpy(dtype=np.int8)
    round_b = predictions["event_day_balanced_lightgbm__default"].to_numpy(dtype=np.int8)
    intersection = np.bitwise_and(incumbent, round_b)

    local = {
        "incumbent": binary_metrics(y, incumbent),
        "round_b_analogue": binary_metrics(y, round_b),
        "intersection": binary_metrics(y, intersection),
        "intersection_delta_vs_round_b_f1": float(binary_metrics(y, intersection)["f1"] - binary_metrics(y, round_b)["f1"]),
        "by_fold": {},
        "by_station": {},
        "bootstrap_vs_round_b": bootstrap_f1_delta(kst_day(predictions["time"]), y, intersection, round_b),
    }
    for fold in sorted(predictions["fold"].astype(str).unique()):
        mask = predictions["fold"].astype(str).eq(fold).to_numpy()
        local["by_fold"][fold] = {
            "round_b_analogue": binary_metrics(y[mask], round_b[mask]),
            "intersection": binary_metrics(y[mask], intersection[mask]),
        }
    for station in sorted(predictions["station"].astype(str).unique()):
        mask = predictions["station"].astype(str).eq(station).to_numpy()
        local["by_station"][station] = {
            "round_b_analogue": binary_metrics(y[mask], round_b[mask]),
            "intersection": binary_metrics(y[mask], intersection[mask]),
        }

    original_path = ROOT / "output" / "2026-08-20" / "ready" / "P1_submission.csv"
    round_b_path = Path.home() / "Downloads" / "해양 해커톤 제출용" / "20260825_round_B_target_adaptive" / "P1_submission.csv"
    original = pd.read_csv(original_path)
    current = pd.read_csv(round_b_path)
    if list(original.columns) != list(current.columns) or not original.iloc[:, :4].equals(current.iloc[:, :4]):
        raise RuntimeError("P1 submission schema/key mismatch")
    output = current.copy()
    output["label"] = (
        original["label"].to_numpy(dtype=np.int8) & current["label"].to_numpy(dtype=np.int8)
    ).astype(np.int8)
    output.loc[output["label"].eq(0), "anomaly_type"] = np.nan
    candidate_path = DELIVERY / "P1_PRECISION_INTERSECTION_V1.csv"
    output.to_csv(candidate_path, index=False, lineterminator="\n")
    official = {
        "rows": int(len(output)),
        "positive_counts": {
            "original": int(original["label"].sum()),
            "current_round_b": int(current["label"].sum()),
            "intersection": int(output["label"].sum()),
        },
        "removed_from_current_round_b": int(np.sum((current["label"] == 1) & (output["label"] == 0))),
        "candidate_sha256": sha256(candidate_path),
    }
    return {"local_oof": local, "official_candidate": official}, candidate_path


def fit_public_quadratic() -> dict[str, float]:
    alpha = np.asarray([0.0, 0.5, 1.0], dtype=float)
    rmse_values = np.asarray([0.541085, 0.599921, 0.713520], dtype=float)
    design = np.column_stack([alpha**2, alpha, np.ones_like(alpha)])
    a, b, c = np.linalg.solve(design, np.square(rmse_values))
    optimum = float(-b / (2 * a))
    predicted = float(math.sqrt(a * optimum**2 + b * optimum + c))
    return {
        "quadratic_a": float(a),
        "quadratic_b": float(b),
        "quadratic_c": float(c),
        "alpha_optimum": optimum,
        "alpha_safe_half": optimum / 2,
        "predicted_rmse_at_optimum": predicted,
        "predicted_gain_vs_incumbent": float(rmse_values[0] - predicted),
    }


def p2_analysis() -> tuple[dict, list[Path]]:
    curve = fit_public_quadratic()
    oof_path = ROOT / "artifacts" / "p2_authoritative_nested_surrogate_actual_20260825_v5" / "evaluated_oof_100.parquet"
    oof = pd.read_parquet(oof_path)
    y = oof["truth"].to_numpy(dtype=float)
    incumbent = oof["INCUMBENT_NOOP"].to_numpy(dtype=float)
    challenger = oof["STACK_W0625"].to_numpy(dtype=float)
    midpoint = oof["FALLBACK_BLEND50_A0625"].to_numpy(dtype=float)
    midpoint_error = np.abs(midpoint - (0.5 * incumbent + 0.5 * challenger))
    alphas = [curve["alpha_safe_half"], curve["alpha_optimum"]]
    local_candidates = {}
    for alpha in alphas:
        name = "safe_half" if alpha == alphas[0] else "public_optimum"
        prediction = incumbent + alpha * (challenger - incumbent)
        entry = {
            "alpha": float(alpha),
            "rmse": rmse(y, prediction),
            "delta_vs_incumbent_rmse": float(rmse(y, prediction) - rmse(y, incumbent)),
            "bootstrap_vs_incumbent": bootstrap_rmse_delta(kst_day(oof["time"]), y, prediction, incumbent),
            "by_fold": {},
            "by_station": {},
        }
        for fold in sorted(oof["fold"].astype(str).unique()):
            mask = oof["fold"].astype(str).eq(fold).to_numpy()
            entry["by_fold"][fold] = {
                "incumbent_rmse": rmse(y[mask], incumbent[mask]),
                "candidate_rmse": rmse(y[mask], prediction[mask]),
            }
        for station in sorted(oof["station"].astype(str).unique()):
            mask = oof["station"].astype(str).eq(station).to_numpy()
            entry["by_station"][station] = {
                "incumbent_rmse": rmse(y[mask], incumbent[mask]),
                "candidate_rmse": rmse(y[mask], prediction[mask]),
            }
        local_candidates[name] = entry

    incumbent_path = ROOT / "output" / "2026-08-20" / "ready" / "P2_submission.csv"
    challenger_path = ROOT / "artifacts" / "p2_conservative_stack_improvement_v1" / "candidate" / "P2_CONSERVATIVE_STACK_IMPROVEMENT_V1.csv"
    midpoint_path = ROOT / "artifacts" / "p2_public_layer_causal_residual_correction_v1" / "candidate" / "P2_PUBLIC_LAYER_CAUSAL_RESIDUAL_CORRECTION_V1_FALLBACK_BLEND50.csv"
    inc = pd.read_csv(incumbent_path)
    cha = pd.read_csv(challenger_path)
    mid = pd.read_csv(midpoint_path)
    key_columns = ["station", "layer", "time"]
    if not inc[key_columns].equals(cha[key_columns]) or not inc[key_columns].equals(mid[key_columns]):
        raise RuntimeError("P2 submission key mismatch")
    official_midpoint_error = np.abs(mid["temp"] - (0.5 * inc["temp"] + 0.5 * cha["temp"]))

    paths: list[Path] = []
    official_candidates = {}
    for name, alpha in (("SAFE", curve["alpha_safe_half"]), ("OPT", curve["alpha_optimum"])):
        output = inc.copy()
        output["temp"] = inc["temp"] + alpha * (cha["temp"] - inc["temp"])
        path = DELIVERY / f"P2_PUBLIC_QUADRATIC_{name}_V1.csv"
        output.to_csv(path, index=False, float_format="%.12f", lineterminator="\n")
        paths.append(path)
        official_candidates[name.lower()] = {
            "alpha": float(alpha),
            "rows": int(len(output)),
            "temp_min": float(output["temp"].min()),
            "temp_max": float(output["temp"].max()),
            "sha256": sha256(path),
        }
    return {
        "public_quadratic": curve,
        "local_oof": {
            "incumbent_rmse": rmse(y, incumbent),
            "challenger_rmse": rmse(y, challenger),
            "midpoint_max_abs_identity_error": float(midpoint_error.max()),
            "candidates": local_candidates,
        },
        "official_path_identity": {
            "midpoint_max_abs_error": float(official_midpoint_error.max()),
            "midpoint_rms_error": float(np.sqrt(np.mean(np.square(official_midpoint_error)))),
        },
        "official_candidates": official_candidates,
    }, paths


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    DELIVERY.mkdir(parents=True, exist_ok=False)
    p1, p1_path = p1_analysis()
    p2, p2_paths = p2_analysis()
    result = {
        "schema_version": "daily_submission_value_20260826.v1",
        "policy": {
            "candidates_preregistered_before_any_2026-08-26_official_submission": True,
            "official_submissions_performed": 0,
            "p3_excluded_due_to_active_frozen_era5_experiment": True,
        },
        "p1": p1,
        "p2": p2,
        "files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in [p1_path, *p2_paths]
        },
    }
    result_path = ARTIFACT / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DELIVERY / "ANALYSIS_RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
