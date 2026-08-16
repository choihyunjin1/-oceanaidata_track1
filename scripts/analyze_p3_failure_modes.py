"""Aggregate-only failure and distribution-shift reconnaissance for P3."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from p3_wave.validation import rmse

SHIFT_FEATURES = (
    "hs_current",
    "hs_delta_1h",
    "hs_delta_3h",
    "hs_delta_6h",
    "hs_delta_12h",
    "hs_std_3h",
    "hs_std_6h",
    "hs_std_12h",
    "hs_mean_24h",
    "tp_current",
    "hmax_current",
    "wspd_current",
    "wspd_delta_3h",
    "wspd_mean_12h",
    "gust_current",
    "caph_current",
    "caph_delta_6h",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(frame: pd.DataFrame, prediction: str) -> dict[str, float | int]:
    truth = frame["target_hs"].to_numpy(float)
    values = frame[prediction].to_numpy(float)
    return {
        "rows": int(len(frame)),
        "rmse": rmse(truth, values),
        "bias": float(np.mean(values - truth)),
        "mae": float(np.mean(np.abs(values - truth))),
    }


def _group_metrics(frame: pd.DataFrame, group: str) -> list[dict[str, object]]:
    total_squared_error = float(
        np.sum((frame["prediction"].to_numpy(float) - frame["target_hs"].to_numpy(float)) ** 2)
    )
    rows: list[dict[str, object]] = []
    for value, part in frame.groupby(group, observed=True, dropna=False):
        squared_error = float(
            np.sum((part["prediction"].to_numpy(float) - part["target_hs"].to_numpy(float)) ** 2)
        )
        rows.append(
            {
                "segment": str(value),
                **_metric(part, "prediction"),
                "single_rmse": _metric(part, "single_prediction")["rmse"],
                "multi_rmse": _metric(part, "multi_prediction")["rmse"],
                "persistence_rmse": _metric(part, "persistence")["rmse"],
                "squared_error_share": squared_error / total_squared_error,
            }
        )
    return rows


def _trajectory_class(case: pd.Series) -> str:
    current = float(case["current_hs"])
    future = np.asarray([case[f"target_{lead}"] for lead in (3, 6, 9, 12, 18, 24)])
    peak_gain = float(np.max(future) - current)
    final_gain = float(future[-1] - current)
    drawdown = float(np.max(future) - future[-1])
    if peak_gain >= 0.30 and drawdown >= 0.30:
        return "peak_then_decay"
    if final_gain >= 0.30:
        return "continued_growth"
    if final_gain <= -0.30:
        return "decay"
    return "near_flat"


def _domain_auc(validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    columns = [column for column in SHIFT_FEATURES if column in validation and column in test]
    combined = pd.concat(
        [
            validation[["station", *columns]].assign(domain=0),
            test[["station", *columns]].assign(domain=1),
        ],
        ignore_index=True,
    )
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                columns,
            ),
            (
                "station",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["station"],
            ),
        ]
    )
    classifier = Pipeline(
        [
            ("transform", transform),
            ("model", LogisticRegression(C=0.1, max_iter=2_000)),
        ]
    )
    y = combined.pop("domain").to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260817)
    scores: list[float] = []
    for train_index, validation_index in splitter.split(combined, y):
        classifier.fit(combined.iloc[train_index], y[train_index])
        probability = classifier.predict_proba(combined.iloc[validation_index])[:, 1]
        scores.append(float(roc_auc_score(y[validation_index], probability)))
    return {
        "features": columns,
        "fold_auc": scores,
        "mean_auc": float(np.mean(scores)),
        "std_auc": float(np.std(scores)),
        "interpretation": "1.0 means easily separable validation/test covariates; 0.5 means indistinguishable.",
    }


def _shift_table(validation: pd.DataFrame, test: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in SHIFT_FEATURES:
        if feature not in validation or feature not in test:
            continue
        left = validation[feature].dropna().to_numpy(float)
        right = test[feature].dropna().to_numpy(float)
        pooled_std = float(np.nanstd(np.concatenate([left, right])))
        rows.append(
            {
                "feature": feature,
                "validation_finite_share": float(validation[feature].notna().mean()),
                "test_finite_share": float(test[feature].notna().mean()),
                "validation_median": float(np.median(left)) if len(left) else None,
                "test_median": float(np.median(right)) if len(right) else None,
                "standardized_mean_difference": (
                    float((np.mean(right) - np.mean(left)) / pooled_std)
                    if pooled_std > 0.0 and len(left) and len(right)
                    else None
                ),
                "ks_statistic": float(ks_2samp(left, right).statistic)
                if len(left) and len(right)
                else None,
            }
        )
    return sorted(rows, key=lambda row: row["ks_statistic"] or 0.0, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--oof", default="artifacts/p3/final_ensemble_validation/oof.parquet")
    parser.add_argument("--submission-dir", default="submissions/p3_frozen_catboost")
    parser.add_argument("--output-dir", default="artifacts/p3/failure_recon")
    args = parser.parse_args()
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(args.oof)
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    test_features = pd.read_parquet(cache / "test_features.parquet")
    validation_ids = oof["anchor_id"].drop_duplicates().to_numpy(dtype=np.int64)
    validation_features = features.set_index("anchor_id").loc[validation_ids].reset_index()

    anchor_lookup = anchors.set_index("anchor_id")
    case = anchor_lookup.loc[validation_ids].reset_index()
    case["trajectory_class"] = case.apply(_trajectory_class, axis=1)
    phase = validation_features[["anchor_id", "hs_delta_3h", "hs_std_6h"]].copy()
    phase["observed_phase"] = pd.cut(
        phase["hs_delta_3h"],
        bins=[-np.inf, -0.20, 0.20, np.inf],
        labels=["falling", "stable", "rising"],
    ).astype(str)
    phase["volatility_quartile"] = pd.qcut(
        phase["hs_std_6h"].rank(method="first"),
        4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    ).astype(str)
    oof = oof.merge(case[["anchor_id", "trajectory_class"]], on="anchor_id", validate="many_to_one")
    oof = oof.merge(
        phase[["anchor_id", "observed_phase", "volatility_quartile"]],
        on="anchor_id",
        validate="many_to_one",
    )
    oof["current_bin"] = pd.cut(
        oof["current_hs"],
        bins=[1.5 - 1e-9, 1.7, 2.0, 2.5, np.inf],
        labels=["1.5-1.7", "1.7-2.0", "2.0-2.5", "2.5+"],
        include_lowest=True,
    ).astype(str)
    oof["component_disagreement"] = np.abs(oof["single_prediction"] - oof["multi_prediction"])
    oof["disagreement_quartile"] = pd.qcut(
        oof["component_disagreement"].rank(method="first"),
        4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    ).astype(str)

    truth = oof["target_hs"].to_numpy(float)
    component = np.column_stack(
        [
            oof["single_prediction"].to_numpy(float),
            oof["multi_prediction"].to_numpy(float),
            oof["persistence"].to_numpy(float),
        ]
    )
    row_oracle = component[
        np.arange(len(oof)), np.argmin(np.abs(component - truth[:, None]), axis=1)
    ]
    case_oracle_parts: list[pd.DataFrame] = []
    for _, part in oof.groupby(["fold", "anchor_id"], sort=False):
        errors = [
            rmse(part["target_hs"], part[column])
            for column in ("single_prediction", "multi_prediction", "persistence")
        ]
        selected = ("single_prediction", "multi_prediction", "persistence")[int(np.argmin(errors))]
        chosen = part.copy()
        chosen["case_oracle"] = chosen[selected]
        case_oracle_parts.append(chosen)
    case_oracle = pd.concat(case_oracle_parts, ignore_index=True)

    submission_dir = Path(args.submission_dir)
    test_single = pd.read_csv(submission_dir / "submission_raw.csv")
    test_multi = pd.read_csv(submission_dir / "submission_multi.csv")
    keys = ["case_id", "station", "lead_h"]
    test_predictions = test_single.merge(
        test_multi,
        on=keys,
        validate="one_to_one",
        suffixes=("_single", "_multi"),
    )
    test_predictions["disagreement"] = np.abs(
        test_predictions["hs_pred_single"] - test_predictions["hs_pred_multi"]
    )

    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "aggregate_only_research_diagnostic",
        "metric_definition": "pooled row RMSE over six leads; 182 independent local cases",
        "overall": {
            "ensemble": _metric(oof, "prediction"),
            "single": _metric(oof, "single_prediction"),
            "multi": _metric(oof, "multi_prediction"),
            "persistence": _metric(oof, "persistence"),
            "row_oracle_rmse_unimplementable": rmse(truth, row_oracle),
            "case_oracle_rmse_unimplementable": rmse(
                case_oracle["target_hs"], case_oracle["case_oracle"]
            ),
        },
        "cuts": {
            "lead": _group_metrics(oof, "lead_h"),
            "station": _group_metrics(oof, "station"),
            "fold": _group_metrics(oof, "fold"),
            "current_hs": _group_metrics(oof, "current_bin"),
            "observed_phase": _group_metrics(oof, "observed_phase"),
            "future_trajectory_research_only": _group_metrics(oof, "trajectory_class"),
            "volatility": _group_metrics(oof, "volatility_quartile"),
            "component_disagreement": _group_metrics(oof, "disagreement_quartile"),
        },
        "data_shift": {
            "validation_cases": int(len(validation_features)),
            "test_cases": int(len(test_features)),
            "domain_classifier": _domain_auc(validation_features, test_features),
            "feature_shift": _shift_table(validation_features, test_features),
            "phase_share": {
                "validation": phase["observed_phase"].value_counts(normalize=True).to_dict(),
                "test": pd.cut(
                    test_features["hs_delta_3h"],
                    bins=[-np.inf, -0.20, 0.20, np.inf],
                    labels=["falling", "stable", "rising"],
                )
                .astype(str)
                .value_counts(normalize=True)
                .to_dict(),
            },
            "station_share": {
                "validation": validation_features["station"].value_counts(normalize=True).to_dict(),
                "test": test_features["station"].value_counts(normalize=True).to_dict(),
            },
        },
        "test_label_free": {
            "component_disagreement_mean": float(test_predictions["disagreement"].mean()),
            "component_disagreement_p90": float(test_predictions["disagreement"].quantile(0.9)),
            "component_disagreement_max": float(test_predictions["disagreement"].max()),
            "by_lead": test_predictions.groupby("lead_h")["disagreement"].mean().to_dict(),
            "by_station": test_predictions.groupby("station")["disagreement"].mean().to_dict(),
        },
        "provenance": {
            "oof_sha256": _sha256(Path(args.oof)),
            "train_features_sha256": _sha256(cache / "train_features.parquet"),
            "test_features_sha256": _sha256(cache / "test_features.parquet"),
            "submission_single_sha256": _sha256(submission_dir / "submission_raw.csv"),
            "submission_multi_sha256": _sha256(submission_dir / "submission_multi.csv"),
            "raw_rows_written": 0,
            "external_observations_used": 0,
        },
    }
    output_path = output / "diagnostics.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": output_path.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
