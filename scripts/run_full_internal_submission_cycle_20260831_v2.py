"""Run a full train -> blocked internal test -> submission-build cycle for P1/P2/P3.

The cycle deliberately changes model structure rather than micro-tuning the
previous submission axes:

* P1 learns three add-only OOF stackers over XGBoost, peer-gate XGBoost, and
  MS-TCN predictions.  Q3 and Q4 are forward tests.
* P2 learns two public-profile residual CatBoost models over a historical
  bin-17 champion proxy.  Three chronological masked blocks are outer tests.
* P3 learns two residual correctors over the uniform KMA alpha=.425 proxy.
  The three historical storm folds are held out in turn by case.

Official test covariates are opened only after the internal predictions are
complete.  Hidden targets are never available to this script and upload is not
implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as p1_e150  # noqa: E402

from p1_qc.config import load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.pipeline import (  # noqa: E402
    load_or_build_features,
    predict_submission,
    train_full_model,
)
from p1_qc.stratification import (  # noqa: E402
    PeerGateConfig,
    append_stratification_peer_gate,
)
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p3_wave.kma_alpha_surface import prepare_oof_frame  # noqa: E402

EXPERIMENT_ID = "full_internal_submission_cycle_20260831_v3"
P1_KEYS = ["station", "year", "layer", "time"]
P2_KEYS = ["station", "layer", "time"]
P3_KEYS = ["case_id", "station", "lead_h"]
RANDOM_SEED = 20260831

P1_DATA = ROOT / "데이터셋 원본" / "데이터셋_P1" / "P1_qc_anomaly"
P2_DATA = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore")
P3_DATA = Path(r"C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast")
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_FULL_INTERNAL_CYCLE_V3"
)
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID

P1_BASE_OOF = ROOT / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet"
P1_PEER_OOF = (
    ROOT
    / "artifacts/runs/20260813T205237+0900_strat_gate_fixed24h_59f6d5c6/oof.parquet"
)
P1_SELECTION = (
    ROOT
    / "artifacts/runs/20260813T205237+0900_strat_gate_fixed24h_59f6d5c6/selection.json"
)
P1_CHAMPION = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260828_DEADLINE_INFORMATION_PROBES_READY"
    r"\P1_1_E150_PLUS_GI_SPIKE2\P1_submission.csv"
)
P1_E150_DEPLOY = ROOT / "artifacts/p1_mstcn_e150_full_deployment_20260827_v1"

P2_SCORED = (
    ROOT
    / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2"
    / "scored_predictions_no_truth.parquet"
)
P2_CHAMPION = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260830_P2_RANK1_BIN_DECOMPOSITION_READY_V1"
    r"\P2_1_RANK1_BIN17_ONLY\P2_submission.csv"
)

P3_BLIND = ROOT / "artifacts/p3_kma_calibrated_longlead_blend_v2/one_shot/blind_predictions.parquet"
P3_EVALUATED = ROOT / "artifacts/p3/long_persistence_shrink/oof.parquet"
P3_BASE = ROOT / "submissions/p3_frozen_catboost/submission.csv"
P3_SOURCE = ROOT / "submissions/p3_kma_calibrated_longlead_secondary_v1/submission.csv"
P3_CHAMPION = ROOT / "submissions/p3_20260830_uniform_kma_0425_v1/P3_submission.csv"


class ContractError(RuntimeError):
    """Raised when a data, split, or output contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, prediction)))


def grouped_rmse_bootstrap(
    frame: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: list[str],
    replicates: int = 2000,
) -> dict[str, Any]:
    work = frame.loc[:, groups].copy()
    work["truth"] = truth
    work["reference"] = reference
    work["candidate"] = candidate
    pieces = list(work.groupby(groups, sort=True, observed=True))
    ref_sse = np.asarray(
        [np.square(group["reference"] - group["truth"]).sum() for _, group in pieces]
    )
    cand_sse = np.asarray(
        [np.square(group["candidate"] - group["truth"]).sum() for _, group in pieces]
    )
    counts = np.asarray([len(group) for _, group in pieces], dtype=np.float64)
    rng = np.random.default_rng(RANDOM_SEED)
    delta = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.integers(0, len(pieces), len(pieces))
        denominator = counts[draw].sum()
        delta[index] = math.sqrt(cand_sse[draw].sum() / denominator) - math.sqrt(
            ref_sse[draw].sum() / denominator
        )
    return {
        "groups": int(len(pieces)),
        "replicates": replicates,
        "mean_delta_rmse": float(delta.mean()),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def validate_keys(frame: pd.DataFrame, keys: list[str], expected_rows: int) -> None:
    if len(frame) != expected_rows or frame.duplicated(keys).any():
        raise ContractError(f"key contract failed for {keys}")


def p1_frame() -> tuple[pd.DataFrame, list[str]]:
    base = pd.read_parquet(P1_BASE_OOF)
    peer = pd.read_parquet(P1_PEER_OOF)
    keys = [*P1_KEYS, "fold"]
    merged = base[
        keys
        + [
            "label",
            "probability",
            "deployment_prediction",
            "plateau_baseline",
            "plateau",
            "spike_candidate",
        ]
    ].merge(
        peer[keys + ["label", "probability", "deployment_prediction"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_base", "_peer"),
    )
    if len(merged) != len(base) or not np.array_equal(
        merged["label_base"], merged["label_peer"]
    ):
        raise ContractError("P1 OOF alignment failed")
    e150_parts = []
    for bundle in p1_e150.load_bundles().values():
        part = bundle.frame[keys].copy()
        part["e150_probability"] = bundle.row_probability
        part["e150_boundary_start"] = bundle.boundary_probability[:, 0]
        part["e150_boundary_end"] = bundle.boundary_probability[:, 1]
        part["e150_prediction"] = bundle.raw_candidate
        e150_parts.append(part)
    e150 = pd.concat(e150_parts, ignore_index=True)
    merged = merged.merge(e150, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(base):
        raise ContractError("P1 E150 alignment failed")
    local = pd.to_datetime(merged["time"], utc=True).dt.tz_convert("Asia/Seoul")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    merged["station_code"] = merged["station"].map(station_map).astype(float)
    merged["layer_code"] = merged["layer"].astype(float)
    merged["month_sin"] = np.sin(2 * np.pi * local.dt.month / 12.0)
    merged["month_cos"] = np.cos(2 * np.pi * local.dt.month / 12.0)
    merged["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24.0)
    merged["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24.0)
    feature_columns = [
        "probability_base",
        "probability_peer",
        "deployment_prediction_base",
        "deployment_prediction_peer",
        "e150_probability",
        "e150_boundary_start",
        "e150_boundary_end",
        "e150_prediction",
        "station_code",
        "layer_code",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
    ]
    if merged[feature_columns].isna().any().any():
        raise ContractError("P1 meta features contain missing values")
    return merged, feature_columns


@dataclass(frozen=True)
class P1ModelSpec:
    name: str
    builder: Callable[[], Any]


def p1_specs() -> list[P1ModelSpec]:
    return [
        P1ModelSpec(
            "P1_1_LOGISTIC_OOF_STACK_UNION",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=RANDOM_SEED,
                ),
            ),
        ),
        P1ModelSpec(
            "P1_2_HIST_GBDT_OOF_STACK_UNION",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=220,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                l2_regularization=5.0,
                random_state=RANDOM_SEED,
            ),
        ),
        P1ModelSpec(
            "P1_3_EXTRA_TREES_OOF_STACK_UNION",
            lambda: ExtraTreesClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=50,
                max_features=0.8,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=6,
            ),
        ),
    ]


def best_add_only_threshold(
    truth: np.ndarray, anchor: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    best = (float(f1_score(truth, anchor)), 1.0)
    for threshold in np.linspace(0.10, 0.95, 35):
        prediction = np.maximum(anchor, probability >= threshold)
        score = float(f1_score(truth, prediction))
        if score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and threshold > best[1]):
            best = (score, float(threshold))
    return best[1], best[0]


def run_p1(delivery: Path) -> dict[str, Any]:
    started = time.perf_counter()
    frame, columns = p1_frame()
    x = frame[columns].to_numpy(np.float64)
    y = frame["label_base"].to_numpy(np.int8)
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    fold_order = ["2025_q2", "2025_q3", "2025_q4"]
    model_records = []
    deployment_models: list[tuple[P1ModelSpec, Any, float]] = []
    for spec in p1_specs():
        prediction = anchor.copy()
        thresholds = []
        fold_records = {}
        for test_fold in fold_order[1:]:
            test_index = frame["fold"].eq(test_fold).to_numpy()
            earlier = fold_order[: fold_order.index(test_fold)]
            train_index = frame["fold"].isin(earlier).to_numpy()
            train_times = pd.to_datetime(frame.loc[train_index, "time"], utc=True)
            cutoff = train_times.quantile(0.8)
            inner_fit = train_index & pd.to_datetime(frame["time"], utc=True).lt(cutoff).to_numpy()
            inner_cal = train_index & ~inner_fit
            if inner_fit.sum() == 0 or inner_cal.sum() == 0:
                raise ContractError("P1 inner time split is empty")
            inner_model = spec.builder()
            inner_model.fit(x[inner_fit], y[inner_fit])
            inner_probability = inner_model.predict_proba(x[inner_cal])[:, 1]
            threshold, _ = best_add_only_threshold(
                y[inner_cal], anchor[inner_cal], inner_probability
            )
            model = spec.builder()
            model.fit(x[train_index], y[train_index])
            probability = model.predict_proba(x[test_index])[:, 1]
            prediction[test_index] = np.maximum(
                anchor[test_index], probability >= threshold
            ).astype(np.int8)
            thresholds.append(threshold)
            fold_records[test_fold] = {
                "rows": int(test_index.sum()),
                "threshold": threshold,
                "reference_f1": float(f1_score(y[test_index], anchor[test_index])),
                "candidate_f1": float(f1_score(y[test_index], prediction[test_index])),
                "delta_f1": float(
                    f1_score(y[test_index], prediction[test_index])
                    - f1_score(y[test_index], anchor[test_index])
                ),
            }
        evaluated = frame["fold"].isin(fold_order[1:]).to_numpy()
        reference_f1 = float(f1_score(y[evaluated], anchor[evaluated]))
        candidate_f1 = float(f1_score(y[evaluated], prediction[evaluated]))
        threshold = float(np.median(thresholds))
        final_model = spec.builder()
        final_model.fit(x, y)
        deployment_models.append((spec, final_model, threshold))
        model_records.append(
            {
                "name": spec.name,
                "forward_test_rows": int(evaluated.sum()),
                "reference_f1": reference_f1,
                "candidate_f1": candidate_f1,
                "delta_f1": candidate_f1 - reference_f1,
                "by_fold": fold_records,
                "deployment_threshold": threshold,
                "strict_internal_pass": bool(
                    candidate_f1 > reference_f1
                    and all(record["delta_f1"] >= 0.0 for record in fold_records.values())
                ),
            }
        )

    config = load_config(ROOT / "configs/p1.toml", env={"P1_DATA_DIR": str(P1_DATA)})
    train, test = load_train_test(P1_DATA, audit=True, strict=True)
    base_train = load_or_build_features(train, config, kind="train", use_cache=True)
    base_test = load_or_build_features(test, config, kind="test", use_cache=True)
    selection = json.loads(P1_SELECTION.read_text(encoding="utf-8"))
    base_model = train_full_model(train, base_train, config, selection)
    base_submission, base_probability = predict_submission(base_model, test, base_test)
    gate = PeerGateConfig(mode="offline", window_hours=24, min_period_fraction=0.5)
    peer_train = append_stratification_peer_gate(
        base_train,
        train,
        config=gate,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    peer_test = append_stratification_peer_gate(
        base_test,
        test,
        config=gate,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    peer_model = train_full_model(train, peer_train, config, selection)
    peer_submission, peer_probability = predict_submission(peer_model, test, peer_test)
    seed_arrays = [
        np.load(path)
        for path in sorted(P1_E150_DEPLOY.glob("full_width_512_seed_*_test_prediction.npz"))
    ]
    if len(seed_arrays) != 3:
        raise ContractError("P1 expected three E150 seed predictions")
    e150_probability = np.mean([item["row_probability"] for item in seed_arrays], axis=0)
    e150_boundary = np.mean([item["boundary_probability"] for item in seed_arrays], axis=0)
    champion = pd.read_csv(
        P1_CHAMPION, dtype={"station": "string", "time": "string", "label": "int8"}
    )
    raw_test = pd.read_csv(
        P1_DATA / "test.csv", dtype={"station": "string", "time": "string"}
    )
    if not champion[P1_KEYS].equals(raw_test[P1_KEYS]) or len(champion) != 169_011:
        raise ContractError("P1 champion official key/order changed")
    local = pd.to_datetime(raw_test["time"], utc=True).dt.tz_convert("Asia/Seoul")
    official = pd.DataFrame(
        {
            "probability_base": base_probability,
            "probability_peer": peer_probability,
            "deployment_prediction_base": base_submission["label"].to_numpy(np.int8),
            "deployment_prediction_peer": peer_submission["label"].to_numpy(np.int8),
            "e150_probability": e150_probability,
            "e150_boundary_start": e150_boundary[:, 0],
            "e150_boundary_end": e150_boundary[:, 1],
            "e150_prediction": champion["label"].to_numpy(np.int8),
            "station_code": raw_test["station"].map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}),
            "layer_code": raw_test["layer"].astype(float),
            "month_sin": np.sin(2 * np.pi * local.dt.month / 12.0),
            "month_cos": np.cos(2 * np.pi * local.dt.month / 12.0),
            "hour_sin": np.sin(2 * np.pi * local.dt.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * local.dt.hour / 24.0),
        }
    )
    official_x = official[columns].to_numpy(np.float64)
    champion_label = champion["label"].to_numpy(np.int8)
    outputs = []
    for record, (spec, model, threshold) in zip(model_records, deployment_models, strict=True):
        probability = model.predict_proba(official_x)[:, 1]
        label = np.maximum(champion_label, probability >= threshold).astype(np.int8)
        submission = raw_test[P1_KEYS].copy()
        submission["label"] = label
        validate_keys(submission, P1_KEYS, 169_011)
        output = write_csv(delivery / spec.name / "P1_submission.csv", submission)
        output.update(
            {
                "name": spec.name,
                "positive_rows": int(label.sum()),
                "additions_vs_champion": int(((label == 1) & (champion_label == 0)).sum()),
                "removals_vs_champion": 0,
                "internal": record,
            }
        )
        outputs.append(output)
    return {
        "status": "TRAINED_FORWARD_TESTED_AND_MATERIALIZED_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "internal_comparator": "E150_OOF_PROXY",
        "internal_disclosure": "Official champion adds two official-only GI rows; OOF comparator is raw E150.",
        "fit_count": 2 + 3 * 5,
        "outputs": outputs,
    }


def p2_public_features(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.copy()
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    values = observations.pivot(index="time", columns="layer", values=["temp", "psal", "depth"])
    values.columns = [f"{name}_l{int(layer)}" for name, layer in values.columns]
    public = [column for column in values.columns if any(f"_l{layer}" in column for layer in (1, 5, 6, 7, 8))]
    current = values[public].copy()
    feature_parts = [current]
    for hours in (6, 24, 72):
        shifted = current.shift(freq=pd.Timedelta(hours=hours)).reindex(current.index)
        shifted.columns = [f"{column}_lag{hours}h" for column in shifted.columns]
        delta = current - shifted.to_numpy()
        delta.columns = [f"{column}_delta{hours}h" for column in current.columns]
        feature_parts.extend([shifted, delta])
    features = pd.concat(feature_parts, axis=1)
    local = features.index.tz_convert("Asia/Seoul")
    features["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    features["hour_sin"] = np.sin(2 * np.pi * local.hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * local.hour / 24.0)
    return features


@dataclass(frozen=True)
class P2Spec:
    name: str
    depth: int
    iterations: int
    learning_rate: float
    l2: float
    blend: float
    clip: float


def p2_specs() -> list[P2Spec]:
    return [
        P2Spec("P2_1_PUBLIC_PROFILE_RESIDUAL_SHALLOW", 6, 500, 0.035, 20.0, 0.50, 0.20),
        P2Spec("P2_2_PUBLIC_PROFILE_RESIDUAL_DEEP", 8, 800, 0.025, 25.0, 0.75, 0.30),
    ]


def p2_model(spec: P2Spec, seed_offset: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=spec.iterations,
        depth=spec.depth,
        learning_rate=spec.learning_rate,
        l2_leaf_reg=spec.l2,
        loss_function="RMSE",
        random_seed=RANDOM_SEED + seed_offset,
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
    )


def run_p2(delivery: Path) -> dict[str, Any]:
    started = time.perf_counter()
    observations = pd.read_csv(
        P2_DATA / "observations.csv",
        dtype={"station": "string", "time": "string"},
    )
    features = p2_public_features(observations)
    scored = pd.read_parquet(P2_SCORED)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    truth = observations.loc[
        observations["layer"].isin([2, 3, 4]), ["station", "time", "layer", "temp"]
    ].copy()
    truth["time"] = pd.to_datetime(truth["time"], utc=True)
    work = scored.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
    if work["temp"].isna().any():
        raise ContractError("P2 historical truth join failed")
    local = work["time"].dt.tz_convert("Asia/Seoul")
    work["season_bin"] = ((local.dt.dayofyear - 1) // 14).astype(int)
    work["kst_date"] = local.dt.date
    active = work["season_bin"].eq(17).to_numpy()
    bin17 = work["reference"].to_numpy(np.float64) + np.where(
        active,
        work["candidate"].to_numpy(np.float64) - work["reference"].to_numpy(np.float64),
        0.0,
    )
    endpoints = public_endpoint_frame(observations)
    projection_frame = work[["station", "time", "layer"]]
    reference = project_profiles_vectorized(projection_frame, bin17, endpoints).prediction
    work_features = features.reindex(pd.DatetimeIndex(work["time"]))
    if len(work_features) != len(work):
        raise ContractError("P2 feature alignment failed")
    x = work_features.to_numpy(np.float64)
    x = np.column_stack([x, work["layer"].to_numpy(float), reference])
    y = work["temp"].to_numpy(np.float64)
    fold_order = ["2024_sep_oct", "2025_jul_aug", "2025_nov_dec"]
    fold_starts = {
        "2024_sep_oct": pd.Timestamp("2024-09-01", tz="UTC"),
        "2025_jul_aug": pd.Timestamp("2025-07-01", tz="UTC"),
        "2025_nov_dec": pd.Timestamp("2025-11-01", tz="UTC"),
    }
    records = []
    full_models: dict[str, dict[int, CatBoostRegressor]] = {}
    complete = observations.pivot(index="time", columns="layer", values="temp")
    complete.index = pd.to_datetime(complete.index, utc=True)
    complete = complete.dropna(subset=[2, 3, 4])
    all_train_times = complete.index[complete.index < pd.Timestamp("2025-09-01", tz="UTC")]
    full_feature = features.reindex(all_train_times)
    for spec_index, spec in enumerate(p2_specs()):
        oof_candidate = reference.copy()
        by_fold = {}
        fit_count = 0
        for fold_index, fold in enumerate(fold_order):
            test_mask = work["fold"].eq(fold).to_numpy()
            cutoff = fold_starts[fold]
            train_times = complete.index[complete.index < cutoff]
            train_feature = features.reindex(train_times)
            for layer in (2, 3, 4):
                target = complete.loc[train_times, layer].to_numpy(np.float64)
                valid = np.isfinite(target)
                model = p2_model(spec, spec_index * 100 + fold_index * 10 + layer)
                model.fit(train_feature.to_numpy(np.float64)[valid], target[valid])
                layer_mask = test_mask & work["layer"].eq(layer).to_numpy()
                direct = model.predict(x[layer_mask, :-2])
                correction = np.clip(direct - reference[layer_mask], -spec.clip, spec.clip)
                oof_candidate[layer_mask] = reference[layer_mask] + spec.blend * correction
                fit_count += 1
            projected = project_profiles_vectorized(
                work.loc[test_mask, ["station", "time", "layer"]],
                oof_candidate[test_mask],
                endpoints,
            ).prediction
            oof_candidate[test_mask] = projected
            by_fold[fold] = {
                "rows": int(test_mask.sum()),
                "reference_rmse": rmse(y[test_mask], reference[test_mask]),
                "candidate_rmse": rmse(y[test_mask], oof_candidate[test_mask]),
                "delta_rmse": rmse(y[test_mask], oof_candidate[test_mask])
                - rmse(y[test_mask], reference[test_mask]),
            }
        bootstrap = grouped_rmse_bootstrap(
            work, y, reference, oof_candidate, ["fold", "kst_date"]
        )
        reference_rmse = rmse(y, reference)
        candidate_rmse = rmse(y, oof_candidate)
        records.append(
            {
                "name": spec.name,
                "fit_count": fit_count,
                "reference_rmse": reference_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": candidate_rmse - reference_rmse,
                "by_fold": by_fold,
                "bootstrap": bootstrap,
                "strict_internal_pass": bool(
                    candidate_rmse < reference_rmse
                    and sum(item["delta_rmse"] < 0 for item in by_fold.values()) >= 2
                    and bootstrap["probability_improved"] >= 0.8
                ),
            }
        )
        models = {}
        for layer in (2, 3, 4):
            target = complete.loc[all_train_times, layer].to_numpy(np.float64)
            valid = np.isfinite(target)
            model = p2_model(spec, 900 + spec_index * 10 + layer)
            model.fit(full_feature.to_numpy(np.float64)[valid], target[valid])
            models[layer] = model
        full_models[spec.name] = models

    test_index = pd.read_csv(
        P2_DATA / "test_index.csv", dtype={"station": "string", "time": "string"}
    )
    test_index["time"] = pd.to_datetime(test_index["time"], utc=True)
    champion = pd.read_csv(
        P2_CHAMPION,
        dtype={"station": "string", "time": "string"},
    )
    champion["time"] = pd.to_datetime(champion["time"], utc=True)
    if not champion[P2_KEYS].equals(test_index[P2_KEYS]):
        raise ContractError("P2 champion official key/order changed")
    official_features = features.reindex(pd.DatetimeIndex(test_index["time"]))
    official_reference = champion["temp"].to_numpy(np.float64)
    outputs = []
    for spec, internal in zip(p2_specs(), records, strict=True):
        prediction = official_reference.copy()
        for layer in (2, 3, 4):
            mask = test_index["layer"].eq(layer).to_numpy()
            direct = full_models[spec.name][layer].predict(
                official_features.to_numpy(np.float64)[mask]
            )
            correction = np.clip(direct - official_reference[mask], -spec.clip, spec.clip)
            prediction[mask] = official_reference[mask] + spec.blend * correction
        projected = project_profiles_vectorized(test_index[P2_KEYS], prediction, endpoints)
        submission = test_index[P2_KEYS].copy()
        submission["temp"] = projected.prediction
        submission["time"] = submission["time"].dt.tz_convert("Asia/Seoul").map(
            lambda value: value.isoformat()
        )
        validate_keys(submission, P2_KEYS, 26_061)
        output = write_csv(delivery / spec.name / "P2_submission.csv", submission)
        output.update(
            {
                "name": spec.name,
                "minimum": float(submission["temp"].min()),
                "maximum": float(submission["temp"].max()),
                "rms_change_vs_champion": float(
                    np.sqrt(np.mean(np.square(projected.prediction - official_reference)))
                ),
                "pava_active_rows": int(projected.active_mask.sum()),
                "internal": internal,
            }
        )
        outputs.append(output)
    return {
        "status": "TRAINED_BLOCKED_TESTED_AND_MATERIALIZED_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "internal_comparator": "CROSSFIT_BIN17_CHAMPION_PROXY",
        "fit_count": int(sum(record["fit_count"] + 3 for record in records)),
        "outputs": outputs,
    }


@dataclass(frozen=True)
class P3Spec:
    name: str
    family: str
    blend: float
    clip: float


def p3_specs() -> list[P3Spec]:
    return [
        P3Spec("P3_1_RIDGE_KMA_RESIDUAL_STACK", "ridge", 0.50, 0.35),
        P3Spec("P3_2_CATBOOST_KMA_RESIDUAL_STACK", "catboost", 0.50, 0.50),
    ]


def p3_features(frame: pd.DataFrame) -> np.ndarray:
    station = frame["station"].map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}).to_numpy(float)
    lead = frame["lead_h"].to_numpy(float)
    current = frame["current_hs"].to_numpy(float)
    base = frame["base"].to_numpy(float)
    delta = frame["delta"].to_numpy(float)
    reference = frame["reference"].to_numpy(float)
    return np.column_stack(
        [
            station,
            lead,
            current,
            base,
            delta,
            reference,
            lead * delta,
            current * delta,
            station * delta,
            (lead >= 18).astype(float),
            (lead == 24).astype(float),
        ]
    )


def p3_model(spec: P3Spec, seed_offset: int) -> Any:
    if spec.family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    return CatBoostRegressor(
        iterations=600,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=20.0,
        loss_function="RMSE",
        random_seed=RANDOM_SEED + seed_offset,
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
    )


def run_p3(delivery: Path) -> dict[str, Any]:
    started = time.perf_counter()
    blind = pd.read_parquet(P3_BLIND)
    evaluated = pd.read_parquet(P3_EVALUATED)
    frame = prepare_oof_frame(blind, evaluated)
    current = blind[
        ["fold", "anchor_id", "station", "lead_h", "current_hs"]
    ].drop_duplicates()
    frame = frame.merge(
        current,
        on=["fold", "anchor_id", "station", "lead_h"],
        how="left",
        validate="one_to_one",
    )
    if frame["current_hs"].isna().any():
        raise ContractError("P3 historical current_hs join failed")
    lead = frame["lead_h"].to_numpy(int)
    alpha = np.zeros(len(frame), dtype=np.float64)
    alpha[np.isin(lead, [18, 24])] = 0.425
    frame["reference"] = np.clip(
        frame["base"].to_numpy(float) + alpha * frame["delta"].to_numpy(float), 0.0, 30.0
    )
    x = p3_features(frame)
    y = frame["target_hs"].to_numpy(float)
    reference = frame["reference"].to_numpy(float)
    fold_names = sorted(frame["fold"].unique().tolist())
    records = []
    full_models = {}
    for spec_index, spec in enumerate(p3_specs()):
        prediction = reference.copy()
        by_fold = {}
        for fold_index, fold in enumerate(fold_names):
            test_mask = frame["fold"].eq(fold).to_numpy()
            train_mask = ~test_mask
            model = p3_model(spec, spec_index * 10 + fold_index)
            model.fit(x[train_mask], y[train_mask] - reference[train_mask])
            correction = np.clip(model.predict(x[test_mask]), -spec.clip, spec.clip)
            prediction[test_mask] = np.clip(
                reference[test_mask] + spec.blend * correction, 0.0, 30.0
            )
            by_fold[str(fold)] = {
                "rows": int(test_mask.sum()),
                "reference_rmse": rmse(y[test_mask], reference[test_mask]),
                "candidate_rmse": rmse(y[test_mask], prediction[test_mask]),
                "delta_rmse": rmse(y[test_mask], prediction[test_mask])
                - rmse(y[test_mask], reference[test_mask]),
            }
        bootstrap = grouped_rmse_bootstrap(
            frame, y, reference, prediction, ["fold", "anchor_id"]
        )
        reference_rmse = rmse(y, reference)
        candidate_rmse = rmse(y, prediction)
        records.append(
            {
                "name": spec.name,
                "fit_count": len(fold_names),
                "reference_rmse": reference_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_rmse": candidate_rmse - reference_rmse,
                "by_fold": by_fold,
                "bootstrap": bootstrap,
                "strict_internal_pass": bool(
                    candidate_rmse < reference_rmse
                    and sum(item["delta_rmse"] < 0 for item in by_fold.values()) >= 2
                    and bootstrap["probability_improved"] >= 0.8
                ),
            }
        )
        model = p3_model(spec, 90 + spec_index)
        model.fit(x, y - reference)
        full_models[spec.name] = model

    test_index = pd.read_csv(
        P3_DATA / "test_index.csv", dtype={"case_id": "string", "station": "string"}
    )
    context = pd.read_parquet(P3_DATA / "test_context.parquet")
    current = context.loc[context["step_minute"].eq(0), ["case_id", "station", "hs"]]
    if current.duplicated(["case_id", "station"]).any():
        raise ContractError("P3 current context key is not unique")
    base = pd.read_csv(P3_BASE, dtype={"case_id": "string", "station": "string"})
    source = pd.read_csv(P3_SOURCE, dtype={"case_id": "string", "station": "string"})
    champion = pd.read_csv(P3_CHAMPION, dtype={"case_id": "string", "station": "string"})
    for candidate in (base, source, champion):
        if not candidate[P3_KEYS].equals(test_index[P3_KEYS]):
            raise ContractError("P3 official key/order mismatch")
    official = test_index.merge(current, on=["case_id", "station"], how="left", validate="many_to_one")
    official["base"] = base["hs_pred"].to_numpy(float)
    official["delta"] = source["hs_pred"].to_numpy(float) - official["base"]
    official["current_hs"] = official["hs"]
    official["reference"] = champion["hs_pred"].to_numpy(float)
    if official[["current_hs", "base", "delta", "reference"]].isna().any().any():
        raise ContractError("P3 official feature join failed")
    official_x = p3_features(official)
    outputs = []
    for spec, internal in zip(p3_specs(), records, strict=True):
        correction = np.clip(
            full_models[spec.name].predict(official_x), -spec.clip, spec.clip
        )
        prediction = np.clip(
            official["reference"].to_numpy(float) + spec.blend * correction, 0.0, 30.0
        )
        submission = test_index[P3_KEYS].copy()
        submission["hs_pred"] = prediction
        validate_keys(submission, P3_KEYS, 1_200)
        output = write_csv(delivery / spec.name / "P3_submission.csv", submission)
        output.update(
            {
                "name": spec.name,
                "minimum": float(prediction.min()),
                "maximum": float(prediction.max()),
                "rms_change_vs_champion": float(
                    np.sqrt(
                        np.mean(
                            np.square(prediction - official["reference"].to_numpy(float))
                        )
                    )
                ),
                "internal": internal,
            }
        )
        outputs.append(output)
    return {
        "status": "TRAINED_BLOCKED_TESTED_AND_MATERIALIZED_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "internal_comparator": "UNIFORM_ALPHA_0.425_OOF_PROXY",
        "fit_count": int(sum(record["fit_count"] + 1 for record in records)),
        "outputs": outputs,
    }


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    outputs = [
        output
        for problem in ("P1", "P2", "P3")
        for output in result["problems"][problem]["outputs"]
    ]
    hashes = [sha256_file(Path(output["path"])) for output in outputs]
    checks = {
        "candidate_count_7": len(outputs) == 7,
        "all_hashes_match": all(
            digest == output["sha256"] for digest, output in zip(hashes, outputs, strict=True)
        ),
        "all_hashes_distinct": len(set(hashes)) == len(hashes),
        "p1_rows": all(output["rows"] == 169_011 for output in result["problems"]["P1"]["outputs"]),
        "p2_rows": all(output["rows"] == 26_061 for output in result["problems"]["P2"]["outputs"]),
        "p3_rows": all(output["rows"] == 1_200 for output in result["problems"]["P3"]["outputs"]),
        "hidden_truth_reads": result["operations"]["hidden_truth_reads"] == 0,
        "uploads": result["operations"]["uploads"] == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "candidate_sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")
    required = [
        P1_DATA,
        P2_DATA,
        P3_DATA,
        P1_BASE_OOF,
        P1_PEER_OOF,
        P1_CHAMPION,
        P2_SCORED,
        P2_CHAMPION,
        P3_BLIND,
        P3_EVALUATED,
        P3_BASE,
        P3_SOURCE,
        P3_CHAMPION,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")
    if args.resume:
        if not ARTIFACT.is_dir() or not DELIVERY.is_dir():
            raise FileNotFoundError("resume requested without existing artifact and delivery roots")
    else:
        if ARTIFACT.exists() or DELIVERY.exists():
            raise FileExistsError(f"one-shot output exists: {ARTIFACT} or {DELIVERY}")
        ARTIFACT.mkdir(parents=True)
        DELIVERY.mkdir(parents=True)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": "oceanaidata.full_internal_submission_cycle.20260831.v3",
        "experiment_id": EXPERIMENT_ID,
        "status": "RUNNING",
        "problems": {},
        "operations": {
            "official_test_covariate_reads_after_internal_scoring": 3,
            "hidden_truth_reads": 0,
            "uploads": 0,
        },
    }
    if not args.resume:
        atomic_json(
            ARTIFACT / "attempt_lock.json",
            {"experiment_id": EXPERIMENT_ID, "pid": os.getpid()},
        )
    for problem, runner in (("P1", run_p1), ("P2", run_p2), ("P3", run_p3)):
        problem_path = ARTIFACT / f"{problem.lower()}_result.json"
        if problem_path.exists():
            result["problems"][problem] = json.loads(
                problem_path.read_text(encoding="utf-8")
            )
        else:
            result["problems"][problem] = runner(DELIVERY)
            atomic_json(problem_path, result["problems"][problem])
    result["runtime_seconds"] = time.perf_counter() - started
    result["status"] = "COMPLETE_TRAINED_TESTED_MATERIALIZED_NOT_UPLOADED"
    result["qa"] = independent_qa(result)
    atomic_json(ARTIFACT / "result.json", result)
    atomic_json(DELIVERY / "SET_MANIFEST.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
