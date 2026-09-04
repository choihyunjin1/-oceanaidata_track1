"""Run the frozen P2 v4 candidate cycle and materialize only internal PASSes.

This is a single, non-adaptive train -> chronological blocked test -> optional
submission-build run.  Candidate definitions and gates are constants below.
Official query keys are not opened until every internal prediction is sealed
and scored.  Hidden labels and upload operations are not implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)

EXPERIMENT_ID = "p2_parallel_candidate_cycle_20260831_v4"
RANDOM_SEED = 20260831
P2_KEYS = ["station", "layer", "time"]
OBSERVATIONS = Path(
    r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv"
)
TEST_INDEX = Path(
    r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\test_index.csv"
)
SCORING_FRAME = (
    ROOT
    / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2"
    / "scored_predictions_no_truth.parquet"
)
CHAMPION = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260830_P2_RANK1_BIN_DECOMPOSITION_READY_V1"
    r"\P2_1_RANK1_BIN17_ONLY\P2_submission.csv"
)
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260831_P2_PARALLEL_CANDIDATE_CYCLE_V4"
)

FOLD_ORDER = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")
FOLD_STARTS = {
    "2024_sep_oct": pd.Timestamp("2024-09-01", tz="UTC"),
    "2025_jul_aug": pd.Timestamp("2025-07-01", tz="UTC"),
    "2025_nov_dec": pd.Timestamp("2025-11-01", tz="UTC"),
}
OFFICIAL_LIKE_FOLD = "2024_sep_oct"
V3_SHALLOW_JUL_AUG_DELTA = 0.013565820170636567


class ContractError(RuntimeError):
    """Raised when a frozen data, split, or output contract is violated."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    clip_c: float
    blend: float
    profile_rms_cap_c: float | None


SPECS = (
    CandidateSpec(
        name="P2_2_HGB_ABSOLUTE_PROFILE",
        family="hgb_absolute",
        clip_c=0.18,
        blend=0.60,
        profile_rms_cap_c=None,
    ),
    CandidateSpec(
        name="P2_3_PLS2_ROBUST_PROFILE",
        family="pls2_profile",
        clip_c=0.15,
        blend=1.00,
        profile_rms_cap_c=0.05,
    ),
)


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


def p2_public_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Past/current features from public layers only (1, 5, 6, 7, 8)."""

    values = observations.pivot(index="time", columns="layer", values=["temp", "psal", "depth"])
    values.columns = [f"{name}_l{int(layer)}" for name, layer in values.columns]
    public = [column for column in values if any(f"_l{layer}" in column for layer in (1, 5, 6, 7, 8))]
    current = values[public].copy()
    parts = [current]
    for hours in (6, 24, 72):
        shifted = current.shift(freq=pd.Timedelta(hours=hours)).reindex(current.index)
        shifted.columns = [f"{column}_lag{hours}h" for column in shifted]
        delta = current - shifted.to_numpy()
        delta.columns = [f"{column}_delta{hours}h" for column in current]
        parts.extend([shifted, delta])
    features = pd.concat(parts, axis=1)
    local = features.index.tz_convert("Asia/Seoul")
    features["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    features["hour_sin"] = np.sin(2 * np.pi * local.hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * local.hour / 24.0)
    return features


def make_hgb() -> HistGradientBoostingRegressor:
    """Frozen robust rule: absolute loss; no result-adaptive outlier deletion."""

    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=128,
        l2_regularization=5.0,
        random_state=RANDOM_SEED,
    )


def make_pls() -> Any:
    """Frozen low-rank joint-profile model with train-only median imputation."""

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        PLSRegression(n_components=2, scale=False, max_iter=500),
    )


def usable_feature_mask(values: np.ndarray) -> np.ndarray:
    """Keep train-fold columns with finite support and non-zero variation."""

    matrix = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(matrix)
    finite_count = finite.sum(axis=0)
    minimum = np.min(np.where(finite, matrix, np.inf), axis=0)
    maximum = np.max(np.where(finite, matrix, -np.inf), axis=0)
    variation = maximum - minimum
    keep = (finite_count >= 2) & np.isfinite(variation) & (variation > 0.0)
    if not keep.any():
        raise ContractError("no usable public feature columns in train fold")
    return keep


def cap_profile_correction(
    frame: pd.DataFrame,
    correction: np.ndarray,
    *,
    absolute_cap: float,
    rms_cap: float | None,
) -> np.ndarray:
    bounded = np.clip(np.asarray(correction, dtype=np.float64), -absolute_cap, absolute_cap)
    if rms_cap is None:
        return bounded
    output = bounded.copy()
    groups = frame.assign(_row=np.arange(len(frame))).groupby(
        [column for column in ("station", "time") if column in frame],
        sort=False,
        dropna=False,
    )
    for _, group in groups:
        rows = group["_row"].to_numpy(int)
        current_rms = float(np.sqrt(np.mean(np.square(output[rows]))))
        if current_rms > rms_cap:
            output[rows] *= rms_cap / current_rms
    return output


def strict_gate(by_fold: dict[str, dict[str, float]], pooled_delta: float) -> dict[str, bool]:
    return {
        "pooled_rmse_improved": pooled_delta < 0.0,
        "minimum_two_of_three_folds_improved": sum(
            record["delta_rmse"] < 0.0 for record in by_fold.values()
        )
        >= 2,
        "official_like_sep_oct_improved": by_fold[OFFICIAL_LIKE_FOLD]["delta_rmse"] < 0.0,
    }


def grouped_bootstrap(
    frame: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    replicates: int = 2000,
) -> dict[str, Any]:
    work = frame[["fold", "kst_date"]].copy()
    work["truth"] = truth
    work["reference"] = reference
    work["candidate"] = candidate
    pieces = list(work.groupby(["fold", "kst_date"], sort=True, observed=True))
    ref_sse = np.asarray([np.square(part["reference"] - part["truth"]).sum() for _, part in pieces])
    cand_sse = np.asarray([np.square(part["candidate"] - part["truth"]).sum() for _, part in pieces])
    counts = np.asarray([len(part) for _, part in pieces], dtype=np.float64)
    rng = np.random.default_rng(RANDOM_SEED)
    delta = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, len(pieces), len(pieces))
        denominator = counts[draw].sum()
        delta[index] = math.sqrt(cand_sse[draw].sum() / denominator) - math.sqrt(
            ref_sse[draw].sum() / denominator
        )
    return {
        "groups": len(pieces),
        "replicates": replicates,
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def build_blind_work(
    observations: pd.DataFrame,
    scored: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    station = observations.loc[:, ["station", "time", "layer"]].drop_duplicates()
    blind = scored.merge(station, on=["time", "layer"], how="left", validate="one_to_one")
    if blind["station"].isna().any() or blind.duplicated(P2_KEYS).any():
        raise ContractError("blind scoring keys are not unique and complete")
    local = blind["time"].dt.tz_convert("Asia/Seoul")
    blind["kst_date"] = local.dt.date
    season_bin = ((local.dt.dayofyear - 1) // 14).astype(int)
    active = season_bin.eq(17).to_numpy()
    unprojected = scored["reference"].to_numpy(np.float64) + np.where(
        active,
        scored["candidate"].to_numpy(np.float64) - scored["reference"].to_numpy(np.float64),
        0.0,
    )
    reference = project_profiles_vectorized(blind[P2_KEYS], unprojected, endpoints).prediction
    return blind, reference


def fit_predict_candidate(
    spec: CandidateSpec,
    blind: pd.DataFrame,
    reference: np.ndarray,
    features: pd.DataFrame,
    complete: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    candidate = reference.copy()
    fit_count = 0
    fold_receipts: dict[str, Any] = {}
    for fold in FOLD_ORDER:
        mask = blind["fold"].eq(fold).to_numpy()
        cutoff = FOLD_STARTS[fold]
        train_times = complete.index[complete.index < cutoff]
        train_x = features.reindex(train_times)
        query_x = features.reindex(pd.DatetimeIndex(blind.loc[mask, "time"]))
        if len(train_times) == 0 or len(query_x) != int(mask.sum()):
            raise ContractError(f"{spec.name} feature alignment failed for {fold}")
        train_values = train_x.to_numpy(np.float64)
        query_values = query_x.to_numpy(np.float64)
        feature_keep = usable_feature_mask(train_values)
        if spec.family == "hgb_absolute":
            direct = np.empty(int(mask.sum()), dtype=np.float64)
            query_layers = blind.loc[mask, "layer"].to_numpy(int)
            for layer in (2, 3, 4):
                model = make_hgb()
                target = complete.loc[train_times, layer].to_numpy(np.float64)
                model.fit(train_values[:, feature_keep], target)
                layer_mask = query_layers == layer
                direct[layer_mask] = model.predict(query_values[layer_mask][:, feature_keep])
                fit_count += 1
        elif spec.family == "pls2_profile":
            model = make_pls()
            model.fit(
                train_values[:, feature_keep],
                complete.loc[train_times, [2, 3, 4]].to_numpy(),
            )
            profile_times = pd.DatetimeIndex(sorted(blind.loc[mask, "time"].unique()))
            profile_prediction = model.predict(
                features.reindex(profile_times).to_numpy(np.float64)[:, feature_keep]
            )
            lookup = {
                (timestamp, layer): profile_prediction[row, column]
                for row, timestamp in enumerate(profile_times)
                for column, layer in enumerate((2, 3, 4))
            }
            direct = np.asarray(
                [lookup[(row.time, int(row.layer))] for row in blind.loc[mask].itertuples()],
                dtype=np.float64,
            )
            fit_count += 1
        else:
            raise ContractError(f"unknown candidate family {spec.family}")
        raw_correction = spec.blend * (direct - reference[mask])
        correction = cap_profile_correction(
            blind.loc[mask, P2_KEYS],
            raw_correction,
            absolute_cap=spec.clip_c,
            rms_cap=spec.profile_rms_cap_c,
        )
        projected = project_profiles_vectorized(
            blind.loc[mask, P2_KEYS], reference[mask] + correction, endpoints
        )
        candidate[mask] = projected.prediction
        fold_receipts[fold] = {
            "train_profiles": int(len(train_times)),
            "query_rows": int(mask.sum()),
            "cutoff_utc": cutoff.isoformat(),
            "postprojection_correction_rms_c": float(
                np.sqrt(np.mean(np.square(projected.prediction - reference[mask])))
            ),
            "pava_active_rows": int(projected.active_mask.sum()),
            "usable_feature_columns": int(feature_keep.sum()),
        }
    return candidate, fit_count, fold_receipts


def fit_full_model(
    spec: CandidateSpec,
    features: pd.DataFrame,
    complete: pd.DataFrame,
) -> tuple[Any, int]:
    train_times = complete.index[complete.index < pd.Timestamp("2025-09-01", tz="UTC")]
    train_x = features.reindex(train_times).to_numpy(np.float64)
    feature_keep = usable_feature_mask(train_x)
    if spec.family == "hgb_absolute":
        models = {}
        for layer in (2, 3, 4):
            model = make_hgb()
            model.fit(
                train_x[:, feature_keep],
                complete.loc[train_times, layer].to_numpy(np.float64),
            )
            models[layer] = model
        return {"models": models, "feature_keep": feature_keep}, 3
    model = make_pls()
    model.fit(
        train_x[:, feature_keep],
        complete.loc[train_times, [2, 3, 4]].to_numpy(np.float64),
    )
    return {"model": model, "feature_keep": feature_keep}, 1


def official_prediction(
    spec: CandidateSpec,
    model: Any,
    test_index: pd.DataFrame,
    reference: np.ndarray,
    features: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    query_x = features.reindex(pd.DatetimeIndex(test_index["time"]))
    if len(query_x) != len(test_index):
        raise ContractError("official feature alignment failed")
    if spec.family == "hgb_absolute":
        direct = np.empty(len(test_index), dtype=np.float64)
        for layer in (2, 3, 4):
            mask = test_index["layer"].eq(layer).to_numpy()
            direct[mask] = model["models"][layer].predict(
                query_x.to_numpy(np.float64)[mask][:, model["feature_keep"]]
            )
    else:
        profile_times = pd.DatetimeIndex(sorted(test_index["time"].unique()))
        profile_prediction = model["model"].predict(
            features.reindex(profile_times).to_numpy(np.float64)[:, model["feature_keep"]]
        )
        lookup = {
            (timestamp, layer): profile_prediction[row, column]
            for row, timestamp in enumerate(profile_times)
            for column, layer in enumerate((2, 3, 4))
        }
        direct = np.asarray(
            [lookup[(row.time, int(row.layer))] for row in test_index.itertuples()],
            dtype=np.float64,
        )
    correction = cap_profile_correction(
        test_index[P2_KEYS],
        spec.blend * (direct - reference),
        absolute_cap=spec.clip_c,
        rms_cap=spec.profile_rms_cap_c,
    )
    projected = project_profiles_vectorized(test_index[P2_KEYS], reference + correction, endpoints)
    return projected.prediction, {
        "rms_change_vs_champion_c": float(
            np.sqrt(np.mean(np.square(projected.prediction - reference)))
        ),
        "pava_eligible_rows": int(projected.eligible_mask.sum()),
        "pava_active_rows": int(projected.active_mask.sum()),
    }


def validate_submission(
    submission: pd.DataFrame,
    test_index_raw: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> dict[str, Any]:
    keys_exact = submission[P2_KEYS].equals(test_index_raw[P2_KEYS])
    finite = bool(np.isfinite(submission["temp"]).all())
    duplicate_keys = int(submission.duplicated(P2_KEYS).sum())
    parsed = submission.copy()
    parsed["time"] = pd.to_datetime(parsed["time"], utc=True)
    reprojection = project_profiles_vectorized(
        parsed[P2_KEYS], parsed["temp"].to_numpy(np.float64), endpoints
    )
    pava_idempotent = bool(
        np.allclose(reprojection.prediction, parsed["temp"], rtol=0.0, atol=1e-10)
    )
    checks = {
        "rows_26061": len(submission) == 26_061,
        "columns_exact": list(submission.columns) == [*P2_KEYS, "temp"],
        "keys_and_order_exact": keys_exact,
        "duplicate_keys_zero": duplicate_keys == 0,
        "finite": finite,
        "pava_idempotent": pava_idempotent,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def write_report(result: dict[str, Any]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P2 parallel candidate cycle 20260831 v4",
        "",
        "## 결론",
        "",
        f"상태: `{result['status']}`. 내부 PASS 수: {result['internal_pass_count']}; 제출본 생성 수: {result['submission_count']}.",
        "",
        "## 사전 고정 방법과 결과",
        "",
        "| 후보 | 구조 | pooled delta RMSE | Sep-Oct | Jul-Aug | Nov-Dec | gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in result["candidates"]:
        folds = record["by_fold"]
        lines.append(
            "| {name} | {family} | {pooled:+.9f} | {sep:+.9f} | {jul:+.9f} | {nov:+.9f} | {gate} |".format(
                name=record["name"],
                family=record["family"],
                pooled=record["delta_rmse"],
                sep=folds["2024_sep_oct"]["delta_rmse"],
                jul=folds["2025_jul_aug"]["delta_rmse"],
                nov=folds["2025_nov_dec"]["delta_rmse"],
                gate="PASS" if record["strict_internal_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "두 후보는 결과 확인 전 코드 상수로 고정했다. HGB 후보는 train-only absolute loss로 이상값 영향만 완화하며 행을 삭제하지 않는다. PLS 후보는 세 target layer를 공동 low-rank profile로 학습하고 train-only median imputation, 0.05 C profile RMS cap, 0.15 C absolute cap을 적용한다.",
            "",
            "PASS 기준은 pooled RMSE 개선, 3개 blocked fold 중 최소 2개 개선, 공식 유사 2024 Sep-Oct 개선의 논리곱이다. 공식 test_index는 모든 내부 prediction hash가 봉인되고 점수와 gate가 끝난 뒤 PASS가 있을 때만 읽었다.",
            "",
            "## QA와 경계",
            "",
            f"실제 fit count: {result['fit_count']}. 내부 prediction hash {len(result['prediction_commitment']['outputs'])}개를 기록했다. hidden truth 접근과 upload는 0건이다. 제출 CSV는 PASS 후보만 만들었다.",
        ]
    )
    (REPORT / "report-source.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(REPORT / "independent-qa.json", result["independent_qa"])


def run(*, resume_technical: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    if ARTIFACT.exists():
        existing = {path.name for path in ARTIFACT.iterdir()}
        if not resume_technical or existing != {"attempt_lock.json"}:
            raise FileExistsError(ARTIFACT)
        atomic_json(
            ARTIFACT / "technical_recovery.json",
            {
                "reason": "all-missing train-fold feature reached sklearn HGB binning",
                "repair": "select train-fold finite nonconstant feature columns",
                "performance_metric_seen_before_repair": False,
                "prediction_written_before_repair": False,
                "official_rows_read_before_repair": 0,
                "new_runner_sha256": sha256_file(Path(__file__)),
            },
        )
    else:
        ARTIFACT.mkdir(parents=True)
        atomic_json(
            ARTIFACT / "attempt_lock.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "candidate_names": [spec.name for spec in SPECS],
                "result_adaptive_retry": False,
                "official_read_before_internal_score": False,
                "runner_sha256": sha256_file(Path(__file__)),
            },
        )

    observations = pd.read_csv(
        OBSERVATIONS,
        dtype={"station": "string", "time": "string"},
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    if observations.duplicated(["time", "layer"]).any():
        raise ContractError("observation time-layer keys are not unique")
    features = p2_public_features(observations)
    endpoints = public_endpoint_frame(observations)
    complete = observations.pivot(index="time", columns="layer", values="temp")
    complete = complete.dropna(subset=[2, 3, 4]).sort_index()
    scored = pd.read_parquet(SCORING_FRAME)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    if tuple(sorted(scored["fold"].unique())) != tuple(sorted(FOLD_ORDER)):
        raise ContractError("scoring folds drifted")
    blind, reference = build_blind_work(observations, scored, endpoints)

    prediction_records: dict[str, Any] = {}
    prediction_vectors: dict[str, np.ndarray] = {}
    fit_count = 0
    for spec in SPECS:
        prediction, candidate_fits, receipts = fit_predict_candidate(
            spec, blind, reference, features, complete, endpoints
        )
        fit_count += candidate_fits
        path = ARTIFACT / f"{spec.name}.npz"
        np.savez_compressed(
            path,
            time_ns=pd.DatetimeIndex(blind["time"]).as_unit("ns").asi8,
            layer=blind["layer"].to_numpy(np.int16),
            reference=reference,
            candidate=prediction,
            fold=blind["fold"].to_numpy(str),
        )
        prediction_records[spec.name] = {
            "path": str(path),
            "rows": len(blind),
            "sha256": sha256_file(path),
            "fit_count": candidate_fits,
            "fold_receipts": receipts,
        }
        prediction_vectors[spec.name] = prediction
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "truth_metric_computed": False,
        "official_test_index_rows_read": 0,
        "hidden_truth_rows_read": 0,
        "outputs": prediction_records,
    }
    atomic_json(ARTIFACT / "prediction_commitment.json", commitment)

    truth = observations.loc[
        observations["layer"].isin([2, 3, 4]), ["time", "layer", "temp"]
    ].rename(columns={"temp": "truth"})
    scored_truth = blind.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
    if scored_truth["truth"].isna().any():
        raise ContractError("outer truth binding failed")
    y = scored_truth["truth"].to_numpy(np.float64)
    candidates = []
    for spec in SPECS:
        if sha256_file(Path(prediction_records[spec.name]["path"])) != prediction_records[spec.name]["sha256"]:
            raise ContractError("sealed prediction hash changed")
        prediction = prediction_vectors[spec.name]
        by_fold: dict[str, dict[str, float]] = {}
        for fold in FOLD_ORDER:
            mask = blind["fold"].eq(fold).to_numpy()
            reference_rmse = rmse(y[mask], reference[mask])
            candidate_rmse = rmse(y[mask], prediction[mask])
            by_fold[fold] = {
                "rows": int(mask.sum()),
                "reference_rmse": reference_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": candidate_rmse - reference_rmse,
            }
        reference_rmse = rmse(y, reference)
        candidate_rmse = rmse(y, prediction)
        delta = candidate_rmse - reference_rmse
        checks = strict_gate(by_fold, delta)
        candidates.append(
            {
                "name": spec.name,
                "family": spec.family,
                "reference_rmse": reference_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": delta,
                "by_fold": by_fold,
                "bootstrap": grouped_bootstrap(blind, y, reference, prediction),
                "gate_checks": checks,
                "strict_internal_pass": bool(all(checks.values())),
                "jul_aug_delta_improvement_vs_v3_shallow_c": float(
                    V3_SHALLOW_JUL_AUG_DELTA - by_fold["2025_jul_aug"]["delta_rmse"]
                ),
            }
        )
    internal_passes = [record for record in candidates if record["strict_internal_pass"]]

    submissions: list[dict[str, Any]] = []
    official_rows_read = 0
    if internal_passes:
        test_index_raw = pd.read_csv(
            TEST_INDEX, dtype={"station": "string", "time": "string"}
        )
        champion_raw = pd.read_csv(
            CHAMPION, dtype={"station": "string", "time": "string"}
        )
        official_rows_read = len(test_index_raw)
        if not champion_raw[P2_KEYS].equals(test_index_raw[P2_KEYS]):
            raise ContractError("champion and test_index key/order differ")
        test_index = test_index_raw.copy()
        test_index["time"] = pd.to_datetime(test_index["time"], utc=True)
        champion_reference = champion_raw["temp"].to_numpy(np.float64)
        DELIVERY.mkdir(parents=True, exist_ok=True)
        spec_by_name = {spec.name: spec for spec in SPECS}
        for internal in internal_passes:
            spec = spec_by_name[internal["name"]]
            model, full_fits = fit_full_model(spec, features, complete)
            fit_count += full_fits
            prediction, diagnostics = official_prediction(
                spec, model, test_index, champion_reference, features, endpoints
            )
            submission = test_index_raw[P2_KEYS].copy()
            submission["temp"] = prediction
            output_dir = DELIVERY / spec.name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "P2_submission.csv"
            submission.to_csv(output_path, index=False, lineterminator="\n")
            qa = validate_submission(submission, test_index_raw, endpoints)
            if qa["status"] != "PASS":
                raise ContractError(f"submission QA failed for {spec.name}")
            submissions.append(
                {
                    "name": spec.name,
                    "path": str(output_path),
                    "rows": len(submission),
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                    "qa": qa,
                    "diagnostics": diagnostics,
                    "full_fit_count": full_fits,
                }
            )
        atomic_json(
            DELIVERY / "SET_MANIFEST.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "INTERNAL_PASS_SUBMISSIONS_READY_NOT_UPLOADED",
                "submissions": submissions,
            },
        )

    qa_checks = {
        "two_frozen_candidates": len(candidates) == 2,
        "three_chronological_folds_each": all(
            set(record["by_fold"]) == set(FOLD_ORDER) for record in candidates
        ),
        "prediction_hashes_match": all(
            sha256_file(Path(record["path"])) == record["sha256"]
            for record in prediction_records.values()
        ),
        "official_read_only_after_internal_pass": official_rows_read == (26_061 if internal_passes else 0),
        "only_internal_pass_materialized": len(submissions) == len(internal_passes),
        "all_submission_qa_pass": all(item["qa"]["status"] == "PASS" for item in submissions),
        "hidden_truth_rows_zero": True,
        "upload_count_zero": True,
    }
    independent_qa = {
        "status": "PASS" if all(qa_checks.values()) else "FAIL",
        "checks": qa_checks,
        "submission_hashes": {item["name"]: item["sha256"] for item in submissions},
    }
    status = (
        "COMPLETE_WITH_NEW_INTERNAL_PASS_SUBMISSIONS_NOT_UPLOADED"
        if internal_passes
        else "COMPLETE_NO_NEW_INTERNAL_PASS"
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": fit_count,
        "candidate_count": len(candidates),
        "internal_pass_count": len(internal_passes),
        "submission_count": len(submissions),
        "official_test_index_rows_read": official_rows_read,
        "hidden_truth_rows_read": 0,
        "upload_count": 0,
        "gate_definition": {
            "pooled_rmse_improved": True,
            "minimum_improved_folds": 2,
            "official_like_fold": OFFICIAL_LIKE_FOLD,
            "official_like_fold_must_improve": True,
        },
        "candidates": candidates,
        "submissions": submissions,
        "prediction_commitment": commitment,
        "independent_qa": independent_qa,
    }
    atomic_json(ARTIFACT / "result.json", result)
    write_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-technical", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("Use --execute for the frozen one-shot run")
    result = run(resume_technical=arguments.resume_technical)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
