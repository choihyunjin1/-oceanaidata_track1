"""Run the sealed P3 Public-transport repair cycle.

The historical stage uses the already authorized selection-matched cohort and
the saved paired incumbent.  Candidate residual calibrators are strictly
prequential: the first forward window is an exact no-op, and later windows may
learn only from completed prior-window OOF residuals.  Official inputs are
opened only when a candidate clears every historical and transport gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for entry in (SCRIPTS, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from run_p3_selection_matched_masked_ssl_20260830_v1 import (  # noqa: E402
    _build_folds,
    _load_stage0_runner,
)

from p3_wave.data import LEADS  # noqa: E402
from p3_wave.selection_matched_masked_ssl_20260830_v1 import (  # noqa: E402
    apply_paired_prequential_reference,
    extract_history_sequences,
    saved_catboost_component_predictions,
    summarize_validation_histories,
)

EXPERIMENT_ID = "p3_public_transport_repair_cycle_20260831_v8"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
DELIVERY_DIR = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_PUBLIC_TRANSPORT_REPAIR_V8"
)
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
P3_DATA = Path(r"C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast")
STAGE0_CONFIG = ROOT / "configs/experiments/p3_selection_matched_cohort_preflight_20260830_v1.json"
MASKED_CONFIG = ROOT / "configs/experiments/p3_selection_matched_masked_ssl_20260830_v1.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v1/calibration.json"
OFFICIAL_LEDGER = ROOT / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json"
FEATURE_COLUMNS = ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/feature_columns.json"
MODEL_ROOT = ROOT / "artifacts/p3_corrected_repeated_forward_catboost_v2/models/folds"
TEST_FEATURES = ROOT / "artifacts/p3/features_all20_v1/test_features.parquet"
CHAMPION = ROOT / "submissions/p3_20260830_uniform_kma_0425_v1/P3_submission.csv"
TEST_INDEX = P3_DATA / "test_index.csv"

KEYS = ["case_id", "station", "lead_h"]
FOLD_ORDER = ("2024_h2_storm", "winter_transition", "2025_h1")
POINTS_PER_RMSE_M = 15.870739046986959
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260831
MAX_STATION_LEAD_REGRESSION_M = 0.01
MIN_CALIBRATED_POINTS = 0.01

CASE_FEATURES = (
    "hs_current",
    "hs_delta_3h",
    "hs_delta_6h",
    "hs_delta_12h",
    "hs_slope_6h",
    "hs_slope_12h",
    "hs_std_12h",
    "hs_std_24h",
    "tp_current",
    "hmax_hs_ratio_current",
    "wspd_current",
    "wspd_delta_6h",
    "wspd_delta_12h",
    "gust_current",
    "wind_wave_alignment_current",
    "wind_input_proxy_current",
)


class ContractError(RuntimeError):
    """Raised when a sealed v8 contract changes."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    trust: float
    correction_cap_m: float
    summary: str


SPECS = (
    CandidateSpec(
        "P3_1_SELECTION_MATCHED_HUBER_CALIBRATOR",
        "huber",
        0.50,
        0.40,
        "Robust shared residual calibrator on selection-matched past-only wave and wind state.",
    ),
    CandidateSpec(
        "P3_2_SELECTION_MATCHED_RIDGE_CALIBRATOR",
        "ridge",
        0.35,
        0.30,
        "Strongly regularized linear residual calibrator with station and lead effects.",
    ),
    CandidateSpec(
        "P3_3_SELECTION_MATCHED_SHALLOW_ET_CALIBRATOR",
        "extra_trees",
        0.40,
        0.35,
        "Shallow robust residual forest constrained by support abstention and correction caps.",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ContractError("invalid RMSE inputs")
    return float(np.sqrt(np.mean(np.square(p - y))))


def build_estimator(spec: CandidateSpec, seed: int) -> Any:
    if spec.family == "huber":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            HuberRegressor(epsilon=1.35, alpha=0.10, max_iter=1000, tol=1e-5),
        )
    if spec.family == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=100.0),
        )
    if spec.family == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=256,
                max_depth=4,
                min_samples_leaf=12,
                max_features=0.5,
                random_state=seed,
                n_jobs=6,
            ),
        )
    raise ContractError(f"unknown family: {spec.family}")


def design(frame: pd.DataFrame) -> np.ndarray:
    missing = set(CASE_FEATURES) - set(frame.columns)
    if missing:
        raise ContractError(f"case features missing: {sorted(missing)}")
    columns = [frame[column].to_numpy(dtype=np.float64) for column in CASE_FEATURES]
    lead = frame["lead_h"].to_numpy(dtype=int)
    station = frame["station"].astype(str)
    current = frame["hs_current"].to_numpy(dtype=np.float64)
    reference = frame["incumbent_prediction"].to_numpy(dtype=np.float64)
    columns.extend(
        [
            (lead / 24.0).astype(np.float64),
            reference - current,
            station.eq("G-ORS").to_numpy(dtype=np.float64),
            station.eq("I-ORS").to_numpy(dtype=np.float64),
            station.eq("S-ORS").to_numpy(dtype=np.float64),
        ]
    )
    columns.extend((lead == item).astype(np.float64) for item in LEADS)
    return np.column_stack(columns)


def support_stats(frame: pd.DataFrame) -> dict[str, list[float]]:
    matrix = frame[list(CASE_FEATURES)].to_numpy(dtype=np.float64)
    median = np.nanmedian(matrix, axis=0)
    q25, q75 = np.nanquantile(matrix, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale <= 1e-9)] = 1.0
    return {"median": median.tolist(), "scale": scale.tolist()}


def supported(frame: pd.DataFrame, stats: dict[str, list[float]]) -> np.ndarray:
    matrix = frame[list(CASE_FEATURES)].to_numpy(dtype=np.float64)
    median = np.asarray(stats["median"], dtype=np.float64)
    scale = np.asarray(stats["scale"], dtype=np.float64)
    filled = np.where(np.isfinite(matrix), matrix, median)
    robust_z = np.abs((filled - median) / scale)
    selection = (
        frame["hs_current"].between(1.5, 2.2, inclusive="left").to_numpy()
        & frame["hs_delta_12h"].gt(0.2).to_numpy()
    )
    return selection & (np.nanmax(robust_z, axis=1) <= 6.0)


def predict_candidate(
    spec: CandidateSpec,
    model: Any,
    frame: pd.DataFrame,
    stats: dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    correction = np.asarray(model.predict(design(frame)), dtype=np.float64)
    correction = np.clip(correction * spec.trust, -spec.correction_cap_m, spec.correction_cap_m)
    active = supported(frame, stats)
    correction[~active] = 0.0
    prediction = np.clip(
        frame["incumbent_prediction"].to_numpy(dtype=np.float64) + correction,
        0.0,
        30.0,
    )
    return prediction, active


def build_historical() -> tuple[pd.DataFrame, dict[str, Any]]:
    stage0 = _load_stage0_runner()
    stage0_config = stage0.load_config(STAGE0_CONFIG)
    masked_config = json.loads(MASKED_CONFIG.read_text(encoding="utf-8"))
    paths = stage0.resolve_train_only_source_paths(P3_DATA)
    wave, atmos, source_receipt = stage0.load_train_only_sources(paths, stage0_config)
    grid, anchors = stage0.build_canonical_train_only_surface(wave, atmos)
    anchors, footprint = stage0.enrich_and_check_anchor_footprints(grid, anchors)
    matched = stage0.build_selection_matched_cohort(anchors, stage0_config)
    validation, _ = _build_folds(matched, masked_config, stage0)
    histories = extract_history_sequences(grid, matched)
    lookup = {
        int(anchor_id): index
        for index, anchor_id in enumerate(matched["anchor_id"].to_numpy(dtype=np.int64))
    }
    valid_index = np.asarray([lookup[int(value)] for value in validation["anchor_id"]], dtype=int)
    features = summarize_validation_histories(histories[valid_index], validation.reset_index(drop=True))
    feature_columns = json.loads(FEATURE_COLUMNS.read_text(encoding="utf-8"))["columns"]
    components = saved_catboost_component_predictions(
        features,
        validation,
        fold_order=FOLD_ORDER,
        feature_columns=feature_columns,
        model_root=MODEL_ROOT,
    )
    reference, reference_receipt = apply_paired_prequential_reference(
        components,
        features,
        validation,
        fold_order=FOLD_ORDER,
        reference_config=masked_config["paired_incumbent_reference"],
    )
    frame = reference.merge(features, on=["anchor_id", "station"], validate="many_to_one")
    if len(frame) != 942 or frame.duplicated(["anchor_id", "station", "lead_h"]).any():
        raise ContractError("historical comparison contract changed")
    frame["fold"] = pd.Categorical(frame["fold"], categories=FOLD_ORDER, ordered=True)
    frame = frame.sort_values(["fold", "anchor_id", "lead_h"]).reset_index(drop=True)
    return frame, {
        "source_receipt": source_receipt,
        "canonical_anchor_count": int(len(anchors)),
        "selection_matched_dense_count": int(len(matched)),
        "validation_cases": int(validation["anchor_id"].nunique()),
        "validation_rows": int(len(frame)),
        "footprint": footprint,
        "reference_receipt": reference_receipt,
    }


def case_bootstrap(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    ordered = (
        frame.assign(candidate=prediction)
        .sort_values(["anchor_id", "lead_h"])
        .reset_index(drop=True)
    )
    case_ids = ordered["anchor_id"].drop_duplicates().to_numpy()
    groups = {key: part.index.to_numpy() for key, part in ordered.groupby("anchor_id", sort=False)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    y = ordered["target_hs"].to_numpy(dtype=np.float64)
    incumbent = ordered["incumbent_prediction"].to_numpy(dtype=np.float64)
    candidate = ordered["candidate"].to_numpy(dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(case_ids, size=len(case_ids), replace=True)
        rows = np.concatenate([groups[item] for item in selected])
        values[index] = rmse(y[rows], candidate[rows]) - rmse(y[rows], incumbent[rows])
    return {
        "unit": "78h-independent complete six-lead anchor episode",
        "replicates": BOOTSTRAP_REPLICATES,
        "ci90_m": [float(value) for value in np.quantile(values, [0.05, 0.95])],
        "median_m": float(np.median(values)),
        "probability_improve": float(np.mean(values < 0.0)),
    }


def group_bootstrap(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    work = frame.assign(candidate=prediction).reset_index(drop=True)
    keys = list(work.groupby(["fold", "station"], observed=True, sort=True).groups)
    groups = {
        key: work.groupby(["fold", "station"], observed=True, sort=True).groups[key].to_numpy()
        for key in keys
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    y = work["target_hs"].to_numpy(dtype=np.float64)
    incumbent = work["incumbent_prediction"].to_numpy(dtype=np.float64)
    candidate = work["candidate"].to_numpy(dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([groups[keys[item]] for item in selected])
        values[index] = rmse(y[rows], candidate[rows]) - rmse(y[rows], incumbent[rows])
    return {
        "unit": "station_x_forward_window group",
        "groups": len(keys),
        "replicates": BOOTSTRAP_REPLICATES,
        "ci90_m": [float(value) for value in np.quantile(values, [0.05, 0.95])],
        "median_m": float(np.median(values)),
        "probability_improve": float(np.mean(values < 0.0)),
    }


def evaluate_candidate(
    spec: CandidateSpec,
    frame: pd.DataFrame,
    transport_penalty: float,
) -> tuple[dict[str, Any], Any, dict[str, list[float]]]:
    prediction = frame["incumbent_prediction"].to_numpy(dtype=np.float64).copy()
    fit_receipts: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(FOLD_ORDER):
        valid = frame["fold"].astype(str).eq(fold).to_numpy()
        train = frame["fold"].astype(str).isin(FOLD_ORDER[:fold_number]).to_numpy()
        if fold_number == 0:
            fit_receipts.append({"fold": fold, "action": "exact_incumbent_no_op", "fits": 0})
            continue
        model = build_estimator(spec, BOOTSTRAP_SEED + fold_number)
        model.fit(
            design(frame.loc[train]),
            (
                frame.loc[train, "target_hs"].to_numpy(dtype=np.float64)
                - frame.loc[train, "incumbent_prediction"].to_numpy(dtype=np.float64)
            ),
        )
        stats = support_stats(frame.loc[train])
        fold_prediction, active = predict_candidate(spec, model, frame.loc[valid], stats)
        prediction[valid] = fold_prediction
        fit_receipts.append(
            {
                "fold": fold,
                "action": "fit_prior_completed_oof_only",
                "fits": 1,
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "active_rows": int(active.sum()),
            }
        )
    y = frame["target_hs"].to_numpy(dtype=np.float64)
    incumbent = frame["incumbent_prediction"].to_numpy(dtype=np.float64)
    before = rmse(y, incumbent)
    after = rmse(y, prediction)
    delta = after - before
    by_window: dict[str, Any] = {}
    for key, part in frame.assign(candidate=prediction).groupby("fold", observed=True, sort=True):
        by_window[str(key)] = {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["incumbent_prediction"].to_numpy()),
        }
    station_lead: dict[str, Any] = {}
    for (station, lead), part in frame.assign(candidate=prediction).groupby(
        ["station", "lead_h"], observed=True, sort=True
    ):
        station_lead[f"{station}|{int(lead)}"] = {
            "rows": int(len(part)),
            "delta_rmse_m": rmse(part["target_hs"].to_numpy(), part["candidate"].to_numpy())
            - rmse(part["target_hs"].to_numpy(), part["incumbent_prediction"].to_numpy()),
        }
    episode = case_bootstrap(frame, prediction)
    group = group_bootstrap(frame, prediction)
    conservative_raw_points = max(0.0, -float(episode["ci90_m"][1]) * POINTS_PER_RMSE_M)
    central_raw_points = -delta * POINTS_PER_RMSE_M
    calibrated_conservative = conservative_raw_points - transport_penalty
    worst_station_lead = max(item["delta_rmse_m"] for item in station_lead.values())
    checks = {
        "pooled_rmse_improves": delta < 0.0,
        "episode_ci90_upper_below_zero": float(episode["ci90_m"][1]) < 0.0,
        "group_ci90_upper_below_zero": float(group["ci90_m"][1]) < 0.0,
        "station_lead_worst_within_0p01m": worst_station_lead <= MAX_STATION_LEAD_REGRESSION_M,
        "calibrated_conservative_points_at_least_0p01": calibrated_conservative
        >= MIN_CALIBRATED_POINTS,
        "finite_predictions": bool(np.isfinite(prediction).all()),
    }
    passed = all(checks.values())
    full_model = None
    if passed:
        full_model = build_estimator(spec, BOOTSTRAP_SEED + 99)
        full_model.fit(design(frame), y - incumbent)
    stats = support_stats(frame)
    return (
        {
            "spec": asdict(spec),
            "fit_receipts": fit_receipts,
            "historical_fit_count": int(sum(item["fits"] for item in fit_receipts)),
            "reference_rmse_m": before,
            "candidate_rmse_m": after,
            "delta_candidate_minus_reference_rmse_m": delta,
            "by_window": by_window,
            "station_lead": station_lead,
            "worst_station_lead_delta_rmse_m": worst_station_lead,
            "episode_bootstrap": episode,
            "group_bootstrap": group,
            "expected_points": {
                "raw_central": central_raw_points,
                "raw_conservative_from_episode_ci90": conservative_raw_points,
                "public_reversal_penalty": transport_penalty,
                "calibrated_conservative": calibrated_conservative,
            },
            "gate_checks": checks,
            "passed": passed,
        },
        full_model,
        stats,
    )


def load_transport_penalty() -> tuple[float, dict[str, Any]]:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    ledger = json.loads(OFFICIAL_LEDGER.read_text(encoding="utf-8"))
    p3 = calibration["gates"]["P3"]
    penalty = float(p3["transport_penalty_points"])
    if abs(penalty - 0.3219056897594759) > 1e-12:
        raise ContractError("P3 Public transport penalty changed")
    official = [item for item in ledger["submissions"] if item["problem"] == "P3"]
    if len(official) != 1 or official[0]["official_metric"] != 0.590956:
        raise ContractError("authoritative P3 official reversal evidence changed")
    return penalty, {"calibration": calibration, "official_p3": official[0]}


def load_official_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    test_index = pd.read_csv(TEST_INDEX, dtype={"case_id": "string", "station": "string"})
    champion = pd.read_csv(CHAMPION, dtype={"case_id": "string", "station": "string"})
    if len(test_index) != 1200 or not test_index[KEYS].equals(champion[KEYS]):
        raise ContractError("official key/champion contract changed")
    features = pd.read_parquet(TEST_FEATURES)
    if len(features) != 200 or features.duplicated(["case_id", "station"]).any():
        raise ContractError("official case feature contract changed")
    frame = test_index.merge(features, on=["case_id", "station"], validate="many_to_one")
    frame["incumbent_prediction"] = champion["hs_pred"].to_numpy(dtype=np.float64)
    return frame, champion


def materialize(
    passing: list[tuple[dict[str, Any], Any, dict[str, list[float]]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not passing:
        return [], {
            "official_test_index_rows_read": 0,
            "official_case_feature_rows_read": 0,
            "official_champion_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "uploads": 0,
        }
    official, champion = load_official_frame()
    DELIVERY_DIR.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    digests: set[str] = set()
    for record, model, stats in passing[:3]:
        prediction, active = predict_candidate(
            CandidateSpec(**record["spec"]), model, official, stats
        )
        submission = official[KEYS].copy()
        submission["hs_pred"] = prediction
        if len(submission) != 1200 or submission.duplicated(KEYS).any() or not np.isfinite(prediction).all():
            raise ContractError("submission structure failed")
        payload = submission.to_csv(index=False, lineterminator="\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in digests:
            raise ContractError("duplicate passing candidate submission")
        digests.add(digest)
        directory = DELIVERY_DIR / record["spec"]["name"]
        path = directory / "P3_submission.csv"
        write_new(path, payload)
        output = {
            "candidate": record["spec"]["name"],
            "path": str(path),
            "rows": 1200,
            "sha256": digest,
            "bytes": len(payload),
            "minimum_m": float(prediction.min()),
            "maximum_m": float(prediction.max()),
            "changed_rows_vs_champion": int(
                np.sum(np.abs(prediction - champion["hs_pred"].to_numpy(dtype=np.float64)) > 1e-12)
            ),
            "support_active_rows": int(active.sum()),
            "rms_change_vs_champion_m": float(
                np.sqrt(np.mean(np.square(prediction - champion["hs_pred"].to_numpy(dtype=float))))
            ),
        }
        outputs.append(output)
        write_new(directory / "submission-info.json", canonical_bytes(output))
    write_new(
        DELIVERY_DIR / "SET_MANIFEST.json",
        canonical_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "PUBLIC_TRANSPORT_PASS_MATERIALIZED_NOT_UPLOADED",
                "outputs": outputs,
                "hidden_truth_rows_read": 0,
                "uploads": 0,
            }
        ),
    )
    return outputs, {
        "official_test_index_rows_read": 1200,
        "official_case_feature_rows_read": 200,
        "official_champion_rows_read": 1200,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# P3 Public transport repair cycle v8",
        "",
        "## 결론",
        "",
        f"- Public-risk calibrated PASS: **{result['passing_candidate_count']}/{result['candidate_count']}**",
        f"- 제출 CSV: **{len(result['outputs'])}개**, upload 0",
        "- 직전 v5 ExtraTrees의 내부 개선이 Public에서 반전된 실측값을 0.321905690점 페널티로 직접 차감했다.",
        "",
        "| candidate | delta RMSE(m) | episode CI90 upper | group CI90 upper | worst station-lead | raw conservative pts | calibrated pts | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["candidates"]:
        lines.append(
            "| {name} | {delta:.6f} | {episode:.6f} | {group:.6f} | {worst:.6f} | {raw:.6f} | {cal:.6f} | {passed} |".format(
                name=item["spec"]["name"],
                delta=item["delta_candidate_minus_reference_rmse_m"],
                episode=item["episode_bootstrap"]["ci90_m"][1],
                group=item["group_bootstrap"]["ci90_m"][1],
                worst=item["worst_station_lead_delta_rmse_m"],
                raw=item["expected_points"]["raw_conservative_from_episode_ci90"],
                cal=item["expected_points"]["calibrated_conservative"],
                passed=item["passed"],
            )
        )
    lines.extend(
        [
            "",
            "## 판정 해석",
            "",
            "- raw conservative gain은 78시간 독립 episode bootstrap의 90% CI 상단으로 계산했다.",
            "- station×forward-window group bootstrap과 station×lead 최악 회귀는 별도 hard gate다.",
            "- Public 결과를 학습 label로 사용하지 않았고, 과거 후보의 예측-실측 점수 잔차는 승격 페널티로만 사용했다.",
            "",
            "## 1차 출처",
            "",
            "- Sugiyama, Krauledat, Müller (JMLR 2007), Importance Weighted Cross Validation: https://jmlr.org/papers/v8/sugiyama07a.html",
            "- Tibshirani et al. (NeurIPS 2019), Conformal Prediction Under Covariate Shift: https://papers.nips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html",
            "- Shah et al. (ICML 2022), Selective Regression under Fairness Criteria: https://proceedings.mlr.press/v162/shah22a.html",
            "- Sagawa et al. (ICLR 2020), Distributionally Robust Neural Networks for Group Shifts: https://arxiv.org/abs/1911.08731",
            "",
            "## 경계",
            "",
            "- hidden truth 0, upload 0. PASS가 없으면 official test/index/champion 값도 0행 읽는다.",
            "- v4-v7 artifact와 다른 문제 lane은 변경하지 않았다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "candidates": [asdict(x) for x in SPECS]}))
        return 0
    for path in (
        P3_DATA,
        STAGE0_CONFIG,
        MASKED_CONFIG,
        CALIBRATION,
        OFFICIAL_LEDGER,
        FEATURE_COLUMNS,
        MODEL_ROOT,
    ):
        if not path.exists():
            raise ContractError(f"required dependency missing: {path}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runner_sha256": runner_hash,
        "candidates": [asdict(item) for item in SPECS],
        "maximum_fits": 9,
    }
    write_new(ATTEMPT_LOCK, canonical_bytes(lock))
    penalty, transport = load_transport_penalty()
    frame, historical = build_historical()
    candidates: list[dict[str, Any]] = []
    fitted: list[tuple[dict[str, Any], Any, dict[str, list[float]]]] = []
    for spec in SPECS:
        record, model, stats = evaluate_candidate(spec, frame, penalty)
        candidates.append(record)
        if record["passed"]:
            fitted.append((record, model, stats))
        print(
            json.dumps(
                {
                    "candidate": spec.name,
                    "delta_rmse_m": record["delta_candidate_minus_reference_rmse_m"],
                    "calibrated_conservative_points": record["expected_points"]["calibrated_conservative"],
                    "passed": record["passed"],
                }
            ),
            flush=True,
        )
    outputs, official_access = materialize(fitted)
    result = {
        "schema_version": "p3.public_transport_repair.result.v8",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE",
        "decision": "PASS_MATERIALIZED_NOT_UPLOADED" if fitted else "NO_GO_PUBLIC_TRANSPORT_GATE",
        "candidate_count": len(SPECS),
        "passing_candidate_count": len(fitted),
        "candidates": candidates,
        "fit_budget": {
            "maximum": 9,
            "actual_historical": int(sum(item["historical_fit_count"] for item in candidates)),
            "actual_full": len(fitted),
            "actual_total": int(sum(item["historical_fit_count"] for item in candidates) + len(fitted)),
        },
        "historical_surface": historical,
        "transport_calibration": {
            "penalty_points": penalty,
            "minimum_calibrated_points": MIN_CALIBRATED_POINTS,
            "minimum_raw_points": penalty + MIN_CALIBRATED_POINTS,
            "authoritative_evidence": transport,
        },
        "outputs": outputs,
        "data_access": official_access,
        "source_provenance": {
            "runner_sha256": runner_hash,
            "stage0_config_sha256": sha256(STAGE0_CONFIG),
            "masked_config_sha256": sha256(MASKED_CONFIG),
            "calibration_sha256": sha256(CALIBRATION),
            "official_ledger_sha256": sha256(OFFICIAL_LEDGER),
            "feature_columns_sha256": sha256(FEATURE_COLUMNS),
        },
        "execution": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "result_based_tuning_or_retry": False,
            "hidden_truth_rows_read": 0,
            "upload_attempt_count": 0,
        },
    }
    write_new(ARTIFACT_DIR / "result.json", canonical_bytes(result))
    write_new(REPORT_DIR / "report-source.md", report_markdown(result).encode())
    write_new(
        REPORT_DIR / "run-manifest.json",
        canonical_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "runner_sha256": runner_hash,
                "result_sha256": sha256(ARTIFACT_DIR / "result.json"),
                "report_sha256": sha256(REPORT_DIR / "report-source.md"),
                "outputs": outputs,
            }
        ),
    )
    print(json.dumps({"decision": result["decision"], "passing": len(fitted), "outputs": len(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
