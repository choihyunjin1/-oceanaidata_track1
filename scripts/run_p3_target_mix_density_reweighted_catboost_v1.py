"""One-shot P3 target-mix density reweighting with a fixed safe fallback.

This experiment is intentionally isolated from the P3 ERA5 context-transfer work.  It
uses only the organizer-supplied P3 feature cache and, after the local gate, anonymous
same-case test features.  No absolute test time or hidden target is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_p3_corrected_repeated_forward_catboost_v1 as base
from p3_wave.corrected_repeated_forward import (
    ACTIVE_ROUTER_LEADS,
    build_corrected_repeated_forward_folds,
    paired_case_bootstrap,
)
from p3_wave.loss_router import (
    OBSERVED_FEATURES,
    build_inference_router_features,
    expand_case_router_features,
    route_row_predictions,
)
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.revin_patch import assign_storm_episodes_from_wave
from p3_wave.submission import build_submission, validate_submission, write_submission
from p3_wave.validation import expand_leads, rmse

CONFIG_REL = Path("configs/experiments/p3_target_mix_density_reweighted_catboost_v1.json")
BASE_CONFIG_REL = Path("configs/experiments/p3_corrected_repeated_forward_catboost_v2.json")
CACHE_REL = Path("artifacts/p3/features_all20_v1")
LEADS = (3, 6, 9, 12, 18, 24)
KEYS = ["case_id", "station", "lead_h"]


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _ess(values: np.ndarray) -> float:
    weight = np.asarray(values, dtype=float)
    return float(weight.sum() ** 2 / np.square(weight).sum())


def _domain_pipeline(config: dict[str, Any]) -> Pipeline:
    classifier = config["domain"]["classifier"]
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, list(OBSERVED_FEATURES)),
            (
                "station",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["station"],
            ),
        ],
        sparse_threshold=0.0,
    )
    model = LogisticRegression(
        C=float(classifier["C"]),
        solver=str(classifier["solver"]),
        max_iter=int(classifier["max_iter"]),
        tol=float(classifier["tol"]),
        class_weight=classifier["class_weight"],
        random_state=int(classifier["random_state"]),
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", model)])


def _density_weights(
    *,
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_groups: np.ndarray,
    target_groups: np.ndarray,
    base_weight: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return source-case weights from grouped OOF domain probabilities."""

    required = {"station", *OBSERVED_FEATURES}
    if required.difference(source.columns) or required.difference(target.columns):
        raise ValueError("domain feature surface is incomplete")
    if len(source) != len(source_groups) or len(target) != len(target_groups):
        raise ValueError("domain groups are not aligned")
    if len(source) != len(base_weight):
        raise ValueError("base weight is not aligned")

    columns = ["station", *OBSERVED_FEATURES]
    combined_frame = pd.concat(
        [source.loc[:, columns], target.loc[:, columns]], ignore_index=True
    )
    combined_frame["station"] = combined_frame["station"].astype(str)
    domain = np.concatenate(
        [np.zeros(len(source), dtype=np.int8), np.ones(len(target), dtype=np.int8)]
    )
    groups = np.concatenate(
        [
            np.asarray([f"source|{value}" for value in source_groups], dtype=object),
            np.asarray([f"target|{value}" for value in target_groups], dtype=object),
        ]
    )
    cross_fit = config["domain"]["cross_fit"]
    splitter = StratifiedGroupKFold(
        n_splits=int(cross_fit["n_splits"]),
        shuffle=bool(cross_fit["shuffle"]),
        random_state=int(cross_fit["random_state"]),
    )
    probability = np.full(len(combined_frame), np.nan, dtype=float)
    prior = np.full(len(combined_frame), np.nan, dtype=float)
    fold_receipts: list[dict[str, Any]] = []
    for number, (fit_index, held_index) in enumerate(
        splitter.split(combined_frame, domain, groups), start=1
    ):
        if set(groups[fit_index]).intersection(set(groups[held_index])):
            raise AssertionError("domain group overlap")
        fit_domain = domain[fit_index]
        source_count = int(np.sum(fit_domain == 0))
        target_count = int(np.sum(fit_domain == 1))
        if source_count == 0 or target_count == 0:
            raise ValueError("domain fold is missing a class")
        pipeline = _domain_pipeline(config)
        pipeline.fit(combined_frame.iloc[fit_index], fit_domain)
        probability[held_index] = pipeline.predict_proba(
            combined_frame.iloc[held_index]
        )[:, 1]
        prior[held_index] = source_count / target_count
        fold_receipts.append(
            {
                "fold": number,
                "fit_source": source_count,
                "fit_target": target_count,
                "held_source": int(np.sum(domain[held_index] == 0)),
                "held_target": int(np.sum(domain[held_index] == 1)),
                "group_overlap": 0,
            }
        )
    if not np.isfinite(probability).all() or not np.isfinite(prior).all():
        raise RuntimeError("domain OOF coverage is incomplete")

    p_low, p_high = map(float, config["domain"]["probability_clip"])
    r_low, r_high = map(float, config["domain"]["ratio_clip"])
    probability = np.clip(probability, p_low, p_high)
    ratio_all = np.clip(probability / (1.0 - probability) * prior, r_low, r_high)
    ratio = ratio_all[: len(source)]
    base_values = np.asarray(base_weight, dtype=float)
    combined = base_values * ratio
    combined /= float(np.mean(combined))
    if not np.isfinite(combined).all() or np.any(combined <= 0.0):
        raise ValueError("combined weights are invalid")

    hard = config["label_free_hard_gate"]
    ratio_ess_fraction = _ess(ratio) / len(ratio)
    combined_ess_fraction = _ess(combined) / len(combined)
    station_ess = {
        str(station): _ess(combined[source["station"].astype(str).eq(str(station))])
        / int(source["station"].astype(str).eq(str(station)).sum())
        for station in sorted(source["station"].astype(str).unique())
    }
    auc = float(roc_auc_score(domain, probability))
    checks = {
        "oof_coverage_100pct": bool(np.isfinite(probability).all()),
        "group_overlap_zero": all(row["group_overlap"] == 0 for row in fold_receipts),
        "finite_positive": bool(np.isfinite(combined).all() and np.all(combined > 0.0)),
        "ratio_ess_fraction": ratio_ess_fraction
        >= float(hard["ratio_ess_fraction_min"]),
        "combined_ess_fraction": combined_ess_fraction
        >= float(hard["combined_ess_fraction_min"]),
        "station_combined_ess_fraction": min(station_ess.values())
        >= float(hard["station_combined_ess_fraction_min"]),
        "domain_auc_below_limit": auc < float(hard["domain_oof_auc_strictly_below"]),
    }
    receipt = {
        "source_cases": int(len(source)),
        "target_cases": int(len(target)),
        "numeric_feature_count": int(len(OBSERVED_FEATURES)),
        "domain_oof_auc": auc,
        "ratio_clip": [r_low, r_high],
        "ratio_ess_fraction": ratio_ess_fraction,
        "combined_ess_fraction": combined_ess_fraction,
        "station_combined_ess_fraction": station_ess,
        "ratio_min": float(np.min(ratio)),
        "ratio_median": float(np.median(ratio)),
        "ratio_max": float(np.max(ratio)),
        "combined_weight_mean": float(np.mean(combined)),
        "ratio_sha256": _array_sha(ratio),
        "combined_weight_sha256": _array_sha(combined),
        "folds": fold_receipts,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "outcome_labels_used": 0,
    }
    return combined, receipt


def _load_preflight(root: Path, data_dir: Path) -> dict[str, Any]:
    config = json.loads((root / CONFIG_REL).read_text(encoding="utf-8"))
    base_config = json.loads((root / BASE_CONFIG_REL).read_text(encoding="utf-8"))
    if config["experiment_id"] != "p3_target_mix_density_reweighted_catboost_v1":
        raise ValueError("unexpected experiment config")
    if tuple(OBSERVED_FEATURES) != tuple(config["domain"]["numeric_features_source"]):
        # The JSON records the symbol, not a duplicated mutable list.
        if config["domain"]["numeric_features_source"] != "p3_wave.loss_router.OBSERVED_FEATURES":
            raise ValueError("domain feature source changed")
    if len(OBSERVED_FEATURES) != int(config["domain"]["numeric_feature_count"]):
        raise ValueError("domain feature count changed")
    if get_gpu_device_count() != 1:
        raise RuntimeError("the frozen multi-output CatBoost requires one visible GPU")

    cache = root / CACHE_REL
    pinned = base._resolved_input_paths(root=root, data_dir=data_dir, cache_dir=cache)
    input_hashes = base._verify_input_hashes(pinned, base_config["expected_sha256"])
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    if len(features) != 24_360 or len(anchors) != 24_360:
        raise ValueError("P3 training cache row count changed")
    if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("P3 feature/anchor keys changed")
    compact = compact_feature_columns(
        [column for column in features.columns if column not in {"anchor_id", "station"}]
    )
    if len(compact) != 591:
        raise ValueError("compact outcome feature surface changed")
    wave = pd.read_csv(data_dir / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors,
        windows=base_config["validation"]["windows"],
        gap_hours=78,
        footprint_hours=72,
    )
    if len(selected) != 181 or [len(f.validation_ids) for f in folds] != [49, 79, 53]:
        raise ValueError("corrected validation selection changed")
    return {
        "config": config,
        "base_config": base_config,
        "cache": cache,
        "input_hashes": input_hashes,
        "features": features,
        "anchors": anchors,
        "feature_columns": compact,
        "folds": folds,
        "selected": selected,
        "split_audit": split_audit,
    }


def _fold_domain_weights(preflight: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    features = preflight["features"].set_index("anchor_id", drop=False)
    anchors = preflight["anchors"].set_index("anchor_id", drop=False)
    weights: dict[str, np.ndarray] = {}
    receipts: list[dict[str, Any]] = []
    for fold in preflight["folds"]:
        source = features.loc[fold.train_ids].reset_index(drop=True)
        target = features.loc[fold.validation_ids].reset_index(drop=True)
        source_meta = anchors.loc[fold.train_ids]
        base_weight = threshold_case_weights(
            source_meta["current_hs"].to_numpy(dtype=float)
        )
        current, receipt = _density_weights(
            source=source,
            target=target,
            source_groups=np.asarray(
                [
                    f"{row.station}|{int(row.episode_id)}"
                    for row in source_meta.itertuples(index=False)
                ],
                dtype=object,
            ),
            target_groups=np.asarray(
                [f"{row.station}|{int(row.anchor_id)}" for row in target.itertuples(index=False)],
                dtype=object,
            ),
            base_weight=base_weight,
            config=preflight["config"],
        )
        receipt["outer_fold"] = fold.name
        receipt["source_anchor_order_sha256"] = hashlib.sha256(
            np.ascontiguousarray(fold.train_ids.astype("<i8")).tobytes()
        ).hexdigest()
        weights[fold.name] = current
        receipts.append(receipt)
    return weights, receipts


def _call_with_injected_weights(function, *, case_weights: np.ndarray, expected_current: np.ndarray, **kwargs):
    """Inject the frozen case weights without changing the shared base runner."""

    case_values = np.asarray(case_weights, dtype=float)
    current_case = np.asarray(expected_current, dtype=float)
    current_rows = np.repeat(current_case, len(LEADS))
    row_values = np.repeat(case_values, len(LEADS))
    original = base.threshold_case_weights

    def injected(current_hs: np.ndarray) -> np.ndarray:
        current = np.asarray(current_hs, dtype=float)
        if current.shape == current_case.shape and np.allclose(current, current_case, equal_nan=True):
            return case_values.copy()
        if current.shape == current_rows.shape and np.allclose(current, current_rows, equal_nan=True):
            return row_values.copy()
        raise AssertionError("unexpected threshold-weight call surface")

    base.threshold_case_weights = injected
    try:
        return function(**kwargs)
    finally:
        base.threshold_case_weights = original


def _local_gate(candidate: pd.DataFrame, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    gate_config = config["local_candidate_gate"]
    incumbent_path = root / gate_config["matched_incumbent_oof"]
    if _sha(incumbent_path) != "eb0af75ec29210254da0d13d1bb8164c0d6b427f4ad5853622144a11fe795f7e":
        raise ValueError("matched incumbent OOF hash changed")
    incumbent = pd.read_parquet(incumbent_path)
    keys = ["fold", "anchor_id", "station", "lead_h"]
    joined = candidate.merge(
        incumbent[keys + ["target_hs", "final_prediction"]].rename(
            columns={"target_hs": "incumbent_truth", "final_prediction": "incumbent_prediction"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != 1086 or not np.allclose(
        joined["target_hs"], joined["incumbent_truth"], atol=0.0, rtol=0.0
    ):
        raise ValueError("candidate/incumbent matched surface changed")
    candidate_rmse = rmse(joined["target_hs"], joined["final_prediction"])
    incumbent_rmse = rmse(joined["target_hs"], joined["incumbent_prediction"])
    bootstrap = paired_case_bootstrap(
        joined,
        candidate_column="final_prediction",
        baseline_column="incumbent_prediction",
        replicates=int(gate_config["bootstrap_replicates"]),
        seed=int(gate_config["bootstrap_seed"]),
    )
    fold_delta = {
        str(name): rmse(group["target_hs"], group["final_prediction"])
        - rmse(group["target_hs"], group["incumbent_prediction"])
        for name, group in joined.groupby("fold", sort=True, observed=True)
    }
    station_delta = {
        str(name): rmse(group["target_hs"], group["final_prediction"])
        - rmse(group["target_hs"], group["incumbent_prediction"])
        for name, group in joined.groupby("station", sort=True, observed=True)
    }
    long = joined["lead_h"].isin([18, 24])
    long_delta = rmse(joined.loc[long, "target_hs"], joined.loc[long, "final_prediction"]) - rmse(
        joined.loc[long, "target_hs"], joined.loc[long, "incumbent_prediction"]
    )
    delta = candidate_rmse - incumbent_rmse
    ci90 = bootstrap["delta_candidate_minus_persistence_ci90_m"]
    checks = {
        "rmse_improves_by_at_least_0p005m": delta
        <= -float(gate_config["minimum_rmse_improvement_m"]),
        "ci90_upper_below_zero": float(ci90[1]) < 0.0,
        "at_least_two_folds_improve": sum(value < 0.0 for value in fold_delta.values()) >= 2,
        "station_degradation_within_0p010m": max(station_delta.values())
        <= float(gate_config["maximum_station_degradation_m"]),
        "lead_18_24_nonworsening": long_delta <= 0.0,
    }
    return {
        "candidate_rmse_m": candidate_rmse,
        "incumbent_rmse_m": incumbent_rmse,
        "delta_candidate_minus_incumbent_m": delta,
        "fold_delta_m": fold_delta,
        "station_delta_m": station_delta,
        "lead_18_24_delta_m": long_delta,
        "paired_complete_case_bootstrap": bootstrap,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _fallback(stage: Path, root: Path, data_dir: Path, config: dict[str, Any], reason: str) -> dict[str, Any]:
    fallback = config["fallback"]
    current_path = root / fallback["current_csv"]
    round_a_path = root / fallback["round_a_csv"]
    if _sha(current_path) != fallback["current_sha256"]:
        raise ValueError("frozen current P3 CSV hash changed")
    if _sha(round_a_path) != fallback["round_a_sha256"]:
        raise ValueError("Round A P3 CSV hash changed")
    current = pd.read_csv(current_path)
    round_a = pd.read_csv(round_a_path)
    test_index = pd.read_csv(data_dir / "test_index.csv")
    validate_submission(current, test_index)
    validate_submission(round_a, test_index)
    if not current[KEYS].equals(round_a[KEYS]):
        raise ValueError("P3 fallback source keys differ")
    active = current["lead_h"].isin(fallback["active_leads"]).to_numpy()
    prediction = current["hs_pred"].to_numpy(dtype=float).copy()
    prediction[active] = 0.5 * (
        current.loc[active, "hs_pred"].to_numpy(dtype=float)
        + round_a.loc[active, "hs_pred"].to_numpy(dtype=float)
    )
    candidate = build_submission(test_index, prediction)
    candidate_path = write_submission(candidate, test_index, stage / "candidate/submission.csv")
    reproduced_path = write_submission(
        candidate, test_index, stage / "candidate/reproduced_submission.csv"
    )
    if _sha(candidate_path) != _sha(reproduced_path):
        raise AssertionError("fallback CSV is not byte-identically reproducible")
    short = ~active
    return {
        "selected_candidate": fallback["id"],
        "selection_reason": reason,
        "rows": int(len(candidate)),
        "short_lead_exact_no_op": bool(
            np.array_equal(
                prediction[short], current.loc[short, "hs_pred"].to_numpy(dtype=float)
            )
        ),
        "candidate_sha256": _sha(candidate_path),
        "reproduced_sha256": _sha(reproduced_path),
        "byte_identical_reproduction": True,
        "uploaded": False,
    }


def _reproduce_density(
    *, stage: Path, cache: Path, data_dir: Path, feature_columns: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    test_features = pd.read_parquet(cache / "test_features.parquet")
    test_index = pd.read_csv(data_dir / "test_index.csv")
    case_order = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    test_cases = case_order.merge(test_features, on=["case_id", "station"], validate="one_to_one")
    source = test_cases.set_index(["case_id", "station"])
    repeated = pd.MultiIndex.from_frame(test_index[["case_id", "station"]])
    single_x = source.loc[repeated, feature_columns].reset_index(drop=True)
    single_x.insert(0, "lead_h", test_index["lead_h"].to_numpy())
    single_x.insert(0, "station", test_index["station"].astype(str).to_numpy())
    current_rows = source.loc[repeated, "hs_current"].to_numpy(dtype=float)
    single_x.insert(2, "current_hs_for_residual", current_rows)
    single = CatBoostRegressor()
    multi = CatBoostRegressor()
    single.load_model(stage / "models/full/single.cbm")
    multi.load_model(stage / "models/full/multi.cbm")
    router = joblib.load(stage / "models/full/router.joblib")
    single_prediction = np.clip(
        current_rows + single.predict(base._cat_frame(single_x)), 0.0, 30.0
    )
    multi_x = test_cases[["station", *feature_columns]].copy()
    multi_x["station"] = multi_x["station"].astype(str)
    current_case = test_cases["hs_current"].to_numpy(dtype=float)
    multi_prediction = np.clip(
        current_case[:, None] + np.asarray(multi.predict(multi_x), dtype=float), 0.0, 30.0
    )
    components = np.stack(
        [
            single_prediction.reshape(200, 6),
            multi_prediction,
            np.repeat(current_case[:, None], 6, axis=1),
        ],
        axis=2,
    )
    case_x = build_inference_router_features(
        test_cases.loc[:, OBSERVED_FEATURES],
        test_cases["station"].to_numpy(str),
        current_case,
        components,
    )
    meta = pd.DataFrame(
        {
            "fold": "anonymous_test",
            "anchor_id": np.arange(200, dtype=np.int64),
            "station": test_cases["station"].astype(str),
            "anchor_time": pd.NaT,
        }
    )
    row_x, row_meta, row_components = expand_case_router_features(case_x, meta, components)
    weights = router.predict_weights(row_x)
    inactive = ~row_meta["lead_h"].isin(ACTIVE_ROUTER_LEADS).to_numpy()
    weights[inactive] = np.array([0.5, 0.5, 0.0])
    routed = route_row_predictions(row_components, weights)
    final = apply_long_lead_persistence_shrink(
        routed,
        np.repeat(current_case[:, None], 6, axis=1).reshape(-1),
        row_meta["lead_h"].to_numpy(dtype=int),
        config=LongLeadPersistenceShrink(weight=0.2, active_leads=(12, 18, 24)),
    )
    reproduced = build_submission(test_index, final)
    reproduced_path = write_submission(
        reproduced, test_index, stage / "candidate/reproduced_submission.csv"
    )
    candidate_path = stage / "candidate/submission.csv"
    return {
        "candidate_sha256": _sha(candidate_path),
        "reproduced_sha256": _sha(reproduced_path),
        "byte_identical_reproduction": _sha(candidate_path) == _sha(reproduced_path),
    }


def check_only(root: Path, data_dir: Path) -> dict[str, Any]:
    preflight = _load_preflight(root, data_dir)
    _, receipts = _fold_domain_weights(preflight)
    result = {
        "status": "CHECK_ONLY_PASS" if all(row["passed"] for row in receipts) else "CHECK_ONLY_DENSITY_NO_GO",
        "validation_cases": int(len(preflight["selected"])),
        "feature_count": int(len(preflight["feature_columns"])),
        "domain_receipts": receipts,
        "all_local_domain_gates_pass": bool(all(row["passed"] for row in receipts)),
        "test_feature_values_read": 0,
        "era5_paths_read": 0,
    }
    return result


def execute(root: Path, data_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = _load_preflight(root, data_dir)
    config = preflight["config"]
    output = root / config["output"]
    lock = root / config["attempt_lock"]
    if output.exists() or lock.exists():
        raise FileExistsError("one-shot output or attempt lock already exists")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment_id": config["experiment_id"],
                "created_at": _now(),
                "status": "ATTEMPT_CONSUMED",
                "upload_authorized": False,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_density_v1_", dir=tmp_root))
    fold_weights, domain_receipts = _fold_domain_weights(preflight)
    _json(stage / "domain_receipts.json", {"folds": domain_receipts})
    density_local_gate: dict[str, Any] | None = None
    density_candidate: dict[str, Any] | None = None
    training_receipts: list[dict[str, Any]] = []
    status: str
    if not all(row["passed"] for row in domain_receipts):
        density_candidate = _fallback(
            stage, root, data_dir, config, "NO_GO_LABEL_FREE_DOMAIN_GATE"
        )
        status = "DENSITY_NO_GO__FIXED_22P5_FALLBACK_CREATED"
    else:
        fold_oof: list[pd.DataFrame] = []
        anchor_lookup = preflight["anchors"].set_index("anchor_id")
        for number, fold in enumerate(preflight["folds"]):
            expected_current = anchor_lookup.loc[
                fold.train_ids, "current_hs"
            ].to_numpy(dtype=float)
            current_oof, receipt = _call_with_injected_weights(
                base._fit_fold_components,
                case_weights=fold_weights[fold.name],
                expected_current=expected_current,
                fold=fold,
                fold_number=number,
                features=preflight["features"],
                anchors=preflight["anchors"],
                feature_columns=preflight["feature_columns"],
                config=preflight["base_config"],
                model_dir=stage / "models/folds",
            )
            fold_oof.append(current_oof)
            training_receipts.append(receipt)
        component_oof = pd.concat(fold_oof, ignore_index=True)
        oof, evaluation, router_material = base._evaluate_fixed_structure(
            component_oof=component_oof,
            train_features=preflight["features"],
            anchors=preflight["anchors"],
            fold_order=tuple(fold.name for fold in preflight["folds"]),
            config=preflight["base_config"],
            split_audit=preflight["split_audit"],
            expected_validation_ids=preflight["selected"]["anchor_id"].to_numpy(dtype=np.int64),
        )
        oof.to_parquet(stage / "oof.parquet", index=False)
        density_local_gate = _local_gate(oof, root, config)
        _json(stage / "local_gate.json", density_local_gate)
        if not density_local_gate["passed"]:
            density_candidate = _fallback(
                stage, root, data_dir, config, "NO_GO_DENSITY_LOCAL_CANDIDATE_GATE"
            )
            status = "DENSITY_LOCAL_NO_GO__FIXED_22P5_FALLBACK_CREATED"
        else:
            test_features = pd.read_parquet(preflight["cache"] / "test_features.parquet")
            source = preflight["features"].reset_index(drop=True)
            source_meta = preflight["anchors"].reset_index(drop=True)
            full_base = threshold_case_weights(source_meta["current_hs"].to_numpy(dtype=float))
            full_weights, full_domain = _density_weights(
                source=source,
                target=test_features.reset_index(drop=True),
                source_groups=np.asarray(
                    [
                        f"{row.station}|{int(row.episode_id)}"
                        for row in source_meta.itertuples(index=False)
                    ],
                    dtype=object,
                ),
                target_groups=np.asarray(
                    [f"{row.station}|{row.case_id}" for row in test_features.itertuples(index=False)],
                    dtype=object,
                ),
                base_weight=full_base,
                config=config,
            )
            _json(stage / "full_domain_receipt.json", full_domain)
            if not full_domain["passed"]:
                density_candidate = _fallback(
                    stage, root, data_dir, config, "NO_GO_OFFICIAL_TARGET_DOMAIN_GATE"
                )
                status = "DENSITY_DEPLOYMENT_NO_GO__FIXED_22P5_FALLBACK_CREATED"
            else:
                expected_current = source_meta["current_hs"].to_numpy(dtype=float)
                candidate_receipt, access = _call_with_injected_weights(
                    base._fit_full_and_infer,
                    case_weights=full_weights,
                    expected_current=expected_current,
                    root=root,
                    data_dir=data_dir,
                    cache_dir=preflight["cache"],
                    stage=stage,
                    features=preflight["features"],
                    anchors=preflight["anchors"],
                    feature_columns=preflight["feature_columns"],
                    router_material=router_material,
                    config=preflight["base_config"],
                )
                reproduction = _reproduce_density(
                    stage=stage,
                    cache=preflight["cache"],
                    data_dir=data_dir,
                    feature_columns=preflight["feature_columns"],
                    config=preflight["base_config"],
                )
                if not reproduction["byte_identical_reproduction"]:
                    raise AssertionError("density candidate model reload reproduction failed")
                density_candidate = {
                    "selected_candidate": "P3_TARGET_MIX_DENSITY_REWEIGHTED_CATBOOST_V1",
                    "candidate_validation": candidate_receipt,
                    "reproduction": reproduction,
                    "access": access,
                    "uploaded": False,
                }
                status = "DENSITY_LOCAL_GO__CANDIDATE_CREATED_NOT_UPLOADED"

    metrics = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "domain_receipts": domain_receipts,
        "training_receipts": training_receipts,
        "density_local_gate": density_local_gate,
        "candidate": density_candidate,
        "split_audit": preflight["split_audit"],
        "invariants": {
            "hidden_target_or_test_label_reads": 0,
            "absolute_test_timestamp_recovery": False,
            "era5_paths_read_or_imported": 0,
            "external_data_quarantine_reads_or_writes": 0,
            "official_uploads": 0,
            "outcome_feature_count": 591,
            "domain_numeric_feature_count": len(OBSERVED_FEATURES),
            "router_search_count": 0,
            "long_lead_persistence_shrink": 0.2,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _json(stage / "metrics.json", metrics)
    output_files = {
        str(path.relative_to(stage)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "config_sha256": _sha(root / CONFIG_REL),
        "base_config_sha256": _sha(root / BASE_CONFIG_REL),
        "runner_sha256": _sha(Path(__file__)),
        "input_sha256": preflight["input_hashes"],
        "output_files": output_files,
        "candidate_uploaded": False,
        "no_raw_values_in_manifest": True,
    }
    _json(stage / "manifest.json", manifest)
    if output.exists():
        raise FileExistsError("canonical output appeared during execution")
    stage.replace(output)
    return {
        "status": status,
        "artifact_dir": str(output.relative_to(root)).replace("\\", "/"),
        "candidate_path": str((output / "candidate/submission.csv").relative_to(root)).replace("\\", "/"),
        "candidate_sha256": _sha(output / "candidate/submission.csv"),
        "metrics_sha256": _sha(output / "metrics.json"),
        "manifest_sha256": _sha(output / "manifest.json"),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    data_dir = args.data_dir
    if data_dir is None:
        value = os.environ.get("P3_DATA_DIR")
        if not value:
            raise RuntimeError("P3_DATA_DIR or --data-dir is required")
        data_dir = Path(value)
    data_dir = data_dir.resolve()
    result = check_only(root, data_dir) if args.check_only else execute(root, data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
