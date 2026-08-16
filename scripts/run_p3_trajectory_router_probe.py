"""Past-only CatBoost feasibility probe for routing future P3 trajectory regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss

from p3_wave.models import compact_feature_columns
from p3_wave.regimes import TRAJECTORY_CLASSES, classify_future_trajectory
from p3_wave.validation import build_forecast_folds

LEADS = (3, 6, 9, 12, 18, 24)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(anchors: pd.DataFrame) -> np.ndarray:
    future = anchors[[f"target_{lead}" for lead in LEADS]].to_numpy(float)
    return classify_future_trajectory(anchors["current_hs"].to_numpy(float), future)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--output-dir", default="artifacts/p3/trajectory_router_probe")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    cache = Path(args.cache_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    feature_lookup = features.set_index("anchor_id")
    all_targets = pd.Series(_target(anchors), index=anchors["anchor_id"], name="trajectory")
    class_to_index = {name: index for index, name in enumerate(TRAJECTORY_CLASSES)}
    fold_results: dict[str, object] = {}
    truth_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    for fold_number, fold in enumerate(build_forecast_folds(anchors)):
        x_train = feature_lookup.loc[fold.train_ids, ["station", *feature_columns]].copy()
        x_validation = feature_lookup.loc[fold.validation_ids, ["station", *feature_columns]].copy()
        x_train["station"] = x_train["station"].astype(str)
        x_validation["station"] = x_validation["station"].astype(str)
        y_train = all_targets.loc[fold.train_ids].to_numpy(str)
        y_validation = all_targets.loc[fold.validation_ids].to_numpy(str)
        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=500,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=8.0,
            random_strength=0.2,
            auto_class_weights="Balanced",
            random_seed=20260817 + fold_number,
            thread_count=args.threads,
            task_type=args.task_type,
            devices="0" if args.task_type == "GPU" else None,
            boosting_type="Plain" if args.task_type == "GPU" else None,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train, cat_features=[0], verbose=False)
        raw_probability = np.asarray(model.predict_proba(x_validation), dtype=float)
        probability = np.zeros((len(x_validation), len(TRAJECTORY_CLASSES)), dtype=float)
        for source_index, class_name in enumerate(model.classes_.astype(str)):
            probability[:, class_to_index[class_name]] = raw_probability[:, source_index]
        prediction = np.asarray(TRAJECTORY_CLASSES)[np.argmax(probability, axis=1)]
        truth_parts.append(y_validation)
        probability_parts.append(probability)
        train_counts = pd.Series(y_train).value_counts().to_dict()
        majority = max(train_counts, key=train_counts.get)
        fold_results[fold.name] = {
            "train_cases": int(len(y_train)),
            "validation_cases": int(len(y_validation)),
            "train_class_counts": {str(key): int(value) for key, value in train_counts.items()},
            "validation_class_counts": {
                str(key): int(value)
                for key, value in pd.Series(y_validation).value_counts().to_dict().items()
            },
            "balanced_accuracy": float(balanced_accuracy_score(y_validation, prediction)),
            "macro_f1": float(f1_score(y_validation, prediction, average="macro")),
            "log_loss": float(log_loss(y_validation, probability, labels=TRAJECTORY_CLASSES)),
            "majority_accuracy": float(np.mean(y_validation == majority)),
            "confusion": confusion_matrix(
                y_validation, prediction, labels=TRAJECTORY_CLASSES
            ).tolist(),
        }

    truth = np.concatenate(truth_parts)
    probability = np.concatenate(probability_parts)
    prediction = np.asarray(TRAJECTORY_CLASSES)[np.argmax(probability, axis=1)]
    result = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "research_only_past_48h_router_feasibility",
        "trajectory_threshold_m": 0.30,
        "classes": list(TRAJECTORY_CLASSES),
        "features": {
            "count": int(len(feature_columns) + 1),
            "station_categorical": True,
            "future_targets_in_features": False,
        },
        "training_device": args.task_type,
        "gpu_result_is_not_bitwise_deterministic": args.task_type == "GPU",
        "overall": {
            "cases": int(len(truth)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro")),
            "log_loss": float(log_loss(truth, probability, labels=TRAJECTORY_CLASSES)),
            "class_counts": {
                str(key): int(value)
                for key, value in pd.Series(truth).value_counts().to_dict().items()
            },
            "confusion": confusion_matrix(truth, prediction, labels=TRAJECTORY_CLASSES).tolist(),
        },
        "folds": fold_results,
        "interpretation_gate": {
            "minimum_balanced_accuracy_for_moe_followup": 0.40,
            "minimum_macro_f1_for_moe_followup": 0.35,
        },
        "provenance": {
            "train_features_sha256": _sha256(cache / "train_features.parquet"),
            "train_anchors_sha256": _sha256(cache / "train_anchors.parquet"),
            "raw_case_predictions_written": 0,
            "external_observations_used": 0,
        },
    }
    path = output / "metrics.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
