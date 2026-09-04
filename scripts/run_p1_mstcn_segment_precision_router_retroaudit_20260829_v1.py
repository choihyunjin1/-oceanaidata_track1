"""Retrospective, prequential audit of an event-level MSTCN addition router.

The script reads only historical OOF predictions and labels. It never reads the
official P1 test frame, creates a submission, or uploads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_segment_precision_router_retroaudit_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
OUTPUT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ANCHOR_PATH = ROOT / "artifacts" / "p1_current_router_oof_anchor_v1" / "anchor.parquet"
TRUTH_PATH = (
    ROOT / "artifacts" / "runs" / "20260813T153038+0900_cv_378a4e89" / "oof.parquet"
)
Q2_PATH = (
    ROOT
    / "artifacts"
    / "p1_incumbent_preserving_mstcn_asrf_v2"
    / "q2_qualification_grid_blind.npz"
)
CHECKPOINT_DIR = ROOT / "artifacts" / "p1_mstcn_checkpoint_diagnostic_20260827_v2"
FOLD_PATHS = {
    "2025_q2": Q2_PATH,
    "2025_q3": CHECKPOINT_DIR / "q3_blind_checkpoint_curve.npz",
    "2025_q4": CHECKPOINT_DIR / "q4_blind_checkpoint_curve.npz",
}
KEYS = ["station", "year", "layer", "time", "fold"]
TYPE_NAMES = ["spike", "noise", "flatline", "offset", "drift"]
CORE_NUMERIC = [
    "log_length",
    "row_mean",
    "row_min",
    "row_max",
    "row_q10",
    "row_median",
    "row_q90",
    "row_std",
    "boundary_start_mean",
    "boundary_start_max",
    "boundary_end_mean",
    "boundary_end_max",
    "incumbent_probability_mean",
    "incumbent_probability_max",
]
TYPE_NUMERIC = [*(f"type_{name}_mean" for name in TYPE_NAMES), "type_entropy_mean"]
CATEGORICAL = ["station", "layer"]


@dataclass
class FoldBundle:
    fold: str
    frame: pd.DataFrame
    labels: np.ndarray
    incumbent: np.ndarray
    raw_candidate: np.ndarray
    row_probability: np.ndarray
    boundary_probability: np.ndarray
    type_probability: np.ndarray | None
    segments: pd.DataFrame
    segment_indices: list[np.ndarray]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def metric(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((labels == 1) & (prediction == 1)))
    fp = int(np.sum((labels == 0) & (prediction == 1)))
    fn = int(np.sum((labels == 1) & (prediction == 0)))
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "f1": float(2 * tp / denominator) if denominator else 1.0,
    }


def segment_is_beneficial(
    *, true_positive_rows: int, false_positive_rows: int, incumbent_f1: float
) -> bool:
    """Exact add-only F1 improvement condition for one disjoint segment."""

    total = true_positive_rows + false_positive_rows
    if total == 0:
        return False
    return true_positive_rows / total > incumbent_f1 / 2.0


def _select_archive_arrays(fold: str, path: Path) -> dict[str, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        if fold == "2025_q2":
            capacity_index = int(
                np.flatnonzero((archive["widths"] == 512) & (archive["epochs"] == 150))[0]
            )
            threshold_index = int(np.flatnonzero(np.isclose(archive["thresholds"], 0.8))[0])
            return {
                "candidate": archive["candidate"][capacity_index, threshold_index].astype(
                    np.int8
                ),
                "row_probability": archive["row_probability"][capacity_index].astype(
                    np.float64
                ),
                "boundary_probability": archive["boundary_probability"][capacity_index].astype(
                    np.float64
                ),
                "type_probability": None,
            }
        checkpoint_index = int(np.flatnonzero(archive["epochs"] == 150)[0])
        return {
            "candidate": archive["candidate"][checkpoint_index].astype(np.int8),
            "row_probability": archive["row_probability"][checkpoint_index].astype(np.float64),
            "boundary_probability": archive["boundary_probability"][checkpoint_index].astype(
                np.float64
            ),
            "type_probability": archive["type_probability"][checkpoint_index].astype(
                np.float64
            ),
        }


def _entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-8, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _build_segments(
    frame: pd.DataFrame,
    labels: np.ndarray,
    incumbent: np.ndarray,
    raw_candidate: np.ndarray,
    row_probability: np.ndarray,
    boundary_probability: np.ndarray,
    type_probability: np.ndarray | None,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    addition = (raw_candidate == 1) & (incumbent == 0)
    parsed_time = pd.to_datetime(frame["time"], utc=True)
    continuation = (
        pd.Series(addition).shift(fill_value=False).to_numpy()
        & frame["station"].astype(str).eq(frame["station"].astype(str).shift()).to_numpy()
        & frame["layer"].eq(frame["layer"].shift()).to_numpy()
        & parsed_time.diff().eq(pd.Timedelta(minutes=10)).to_numpy()
    )
    segment_ids = np.cumsum(addition & ~continuation)
    incumbent_f1 = float(metric(labels, incumbent)["f1"])
    rows: list[dict[str, Any]] = []
    indices: list[np.ndarray] = []
    for segment_id in np.unique(segment_ids[addition]):
        positions = np.flatnonzero(addition & (segment_ids == segment_id))
        y = labels[positions]
        row = row_probability[positions]
        boundary = boundary_probability[positions]
        true_positive_rows = int(y.sum())
        false_positive_rows = int(len(y) - true_positive_rows)
        record: dict[str, Any] = {
            "station": str(frame.loc[int(positions[0]), "station"]),
            "layer": str(frame.loc[int(positions[0]), "layer"]),
            "length": int(len(positions)),
            "log_length": float(math.log1p(len(positions))),
            "row_mean": float(np.mean(row)),
            "row_min": float(np.min(row)),
            "row_max": float(np.max(row)),
            "row_q10": float(np.quantile(row, 0.10)),
            "row_median": float(np.median(row)),
            "row_q90": float(np.quantile(row, 0.90)),
            "row_std": float(np.std(row)),
            "boundary_start_mean": float(np.mean(boundary[:, 0])),
            "boundary_start_max": float(np.max(boundary[:, 0])),
            "boundary_end_mean": float(np.mean(boundary[:, 1])),
            "boundary_end_max": float(np.max(boundary[:, 1])),
            "incumbent_probability_mean": float(
                np.mean(frame.loc[positions, "incumbent_probability"].to_numpy(float))
            ),
            "incumbent_probability_max": float(
                np.max(frame.loc[positions, "incumbent_probability"].to_numpy(float))
            ),
            "true_positive_rows": true_positive_rows,
            "false_positive_rows": false_positive_rows,
            "segment_precision": float(true_positive_rows / len(positions)),
            "beneficial": int(
                segment_is_beneficial(
                    true_positive_rows=true_positive_rows,
                    false_positive_rows=false_positive_rows,
                    incumbent_f1=incumbent_f1,
                )
            ),
        }
        if type_probability is not None:
            types = type_probability[positions]
            for index, name in enumerate(TYPE_NAMES):
                record[f"type_{name}_mean"] = float(np.mean(types[:, index]))
            record["type_entropy_mean"] = float(np.mean(_entropy(types)))
        rows.append(record)
        indices.append(positions)
    return pd.DataFrame(rows), indices


def load_bundles() -> dict[str, FoldBundle]:
    anchor = pd.read_parquet(ANCHOR_PATH)
    truth = pd.read_parquet(
        TRUTH_PATH,
        columns=[*KEYS, "label", "probability"],
    )
    require(anchor[KEYS].astype(str).equals(truth[KEYS].astype(str)), "anchor/truth keys differ")
    bundles: dict[str, FoldBundle] = {}
    for fold, path in FOLD_PATHS.items():
        archive = _select_archive_arrays(fold, path)
        anchor_fold = anchor.loc[anchor["fold"].eq(fold)].reset_index(drop=True)
        truth_fold = truth.loc[truth["fold"].eq(fold)].reset_index(drop=True)
        frame = anchor_fold.copy()
        frame["incumbent_probability"] = truth_fold["probability"].to_numpy(float)
        labels = truth_fold["label"].to_numpy(np.int8)
        incumbent = frame["current_router_prediction"].to_numpy(np.int8)
        raw_candidate = np.asarray(archive["candidate"], dtype=np.int8)
        row_probability = np.asarray(archive["row_probability"], dtype=np.float64)
        boundary_probability = np.asarray(archive["boundary_probability"], dtype=np.float64)
        type_probability = archive["type_probability"]
        require(
            len(frame) == len(raw_candidate) == len(row_probability) == len(boundary_probability),
            f"fold array length mismatch: {fold}",
        )
        require(np.isin(raw_candidate, [0, 1]).all(), f"invalid candidate: {fold}")
        require(np.isfinite(row_probability).all(), f"non-finite row probability: {fold}")
        require(np.isfinite(boundary_probability).all(), f"non-finite boundary: {fold}")
        if type_probability is not None:
            type_probability = np.asarray(type_probability, dtype=np.float64)
            require(type_probability.shape == (len(frame), 5), f"type shape: {fold}")
            require(np.isfinite(type_probability).all(), f"non-finite type: {fold}")
        segments, segment_indices = _build_segments(
            frame,
            labels,
            incumbent,
            raw_candidate,
            row_probability,
            boundary_probability,
            type_probability,
        )
        bundles[fold] = FoldBundle(
            fold=fold,
            frame=frame,
            labels=labels,
            incumbent=incumbent,
            raw_candidate=raw_candidate,
            row_probability=row_probability,
            boundary_probability=boundary_probability,
            type_probability=type_probability,
            segments=segments,
            segment_indices=segment_indices,
        )
    return bundles


def _model(numeric_features: list[str], config: dict[str, Any]) -> Pipeline:
    specification = config["model"]
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=float(specification["C"]),
        class_weight=specification["class_weight"],
        solver=str(specification["solver"]),
        max_iter=int(specification["maximum_iterations"]),
        random_state=int(specification["random_state"]),
    )
    return Pipeline([("transform", transform), ("classifier", classifier)])


def evaluate_arm(
    *,
    arm: str,
    fit_folds: list[str],
    evaluation_fold: str,
    bundles: dict[str, FoldBundle],
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    type_augmented = arm == "type_augmented"
    numeric_features = [*CORE_NUMERIC, *(TYPE_NUMERIC if type_augmented else [])]
    training = pd.concat([bundles[name].segments for name in fit_folds], ignore_index=True)
    evaluation = bundles[evaluation_fold]
    require(len(training) > 0 and len(evaluation.segments) > 0, "empty segment surface")
    require(training["beneficial"].nunique() == 2, "training surface has one target class")
    require(
        not type_augmented or all(name in training.columns for name in TYPE_NUMERIC),
        "type-augmented features unavailable",
    )
    pipeline = _model(numeric_features, config)
    columns = [*numeric_features, *CATEGORICAL]
    pipeline.fit(training[columns], training["beneficial"].to_numpy(np.int8))
    probability = pipeline.predict_proba(evaluation.segments[columns])[:, 1]
    accepted = probability >= float(config["model"]["acceptance_probability"])
    added = np.zeros(len(evaluation.labels), dtype=np.int8)
    for keep, positions in zip(accepted, evaluation.segment_indices, strict=True):
        if keep:
            added[positions] = 1
    candidate = np.maximum(evaluation.incumbent, added)
    incumbent_metric = metric(evaluation.labels, evaluation.incumbent)
    raw_metric = metric(evaluation.labels, evaluation.raw_candidate)
    candidate_metric = metric(evaluation.labels, candidate)
    added_mask = (candidate == 1) & (evaluation.incumbent == 0)
    report = {
        "arm": arm,
        "fit_folds": fit_folds,
        "evaluation_fold": evaluation_fold,
        "training_segments": int(len(training)),
        "training_beneficial_segments": int(training["beneficial"].sum()),
        "evaluation_segments": int(len(evaluation.segments)),
        "accepted_segments": int(accepted.sum()),
        "accepted_rows": int(added_mask.sum()),
        "accepted_true_positive_rows": int(evaluation.labels[added_mask].sum()),
        "accepted_false_positive_rows": int((1 - evaluation.labels[added_mask]).sum()),
        "accepted_row_precision": (
            float(evaluation.labels[added_mask].mean()) if added_mask.any() else 1.0
        ),
        "incumbent": incumbent_metric,
        "raw_e150": raw_metric,
        "candidate": candidate_metric,
        "delta_f1_vs_incumbent": float(candidate_metric["f1"] - incumbent_metric["f1"]),
        "delta_f1_vs_raw_e150": float(candidate_metric["f1"] - raw_metric["f1"]),
        "probability_summary": {
            "minimum": float(np.min(probability)),
            "median": float(np.median(probability)),
            "maximum": float(np.max(probability)),
        },
    }
    return report, candidate


def _pooled_metric(
    bundles: dict[str, FoldBundle], predictions: dict[str, np.ndarray], folds: list[str]
) -> dict[str, float | int]:
    labels = np.concatenate([bundles[fold].labels for fold in folds])
    prediction = np.concatenate([predictions[fold] for fold in folds])
    return metric(labels, prediction)


def execute() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(config["experiment_id"] == EXPERIMENT_ID, "experiment id mismatch")
    require(config["official_test_reads_allowed"] == 0, "official read contract changed")
    require(config["submission_creation_allowed"] is False, "submission contract changed")
    bundles = load_bundles()
    reports: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for row in config["fold_schedule"]:
        report, candidate = evaluate_arm(
            arm=str(row["arm"]),
            fit_folds=[str(value) for value in row["fit_folds"]],
            evaluation_fold=str(row["evaluation_fold"]),
            bundles=bundles,
            config=config,
        )
        reports.append(report)
        predictions[(str(row["arm"]), str(row["evaluation_fold"]))] = candidate

    by_key = {(row["arm"], row["evaluation_fold"]): row for row in reports}
    q3_core = by_key[("core", "2025_q3")]
    q4_core = by_key[("core", "2025_q4")]
    q4_type = by_key[("type_augmented", "2025_q4")]
    gate_checks = {
        "core_improves_raw_e150_on_q3": q3_core["delta_f1_vs_raw_e150"] > 0.0,
        "core_improves_raw_e150_on_q4": q4_core["delta_f1_vs_raw_e150"] > 0.0,
        "core_nonnegative_vs_incumbent_on_q3": q3_core["delta_f1_vs_incumbent"] >= 0.0,
        "core_nonnegative_vs_incumbent_on_q4": q4_core["delta_f1_vs_incumbent"] >= 0.0,
        "type_augmented_improves_core_on_q4": q4_type["candidate"]["f1"]
        > q4_core["candidate"]["f1"],
    }
    core_predictions = {
        "2025_q3": predictions[("core", "2025_q3")],
        "2025_q4": predictions[("core", "2025_q4")],
    }
    pooled_core = _pooled_metric(bundles, core_predictions, ["2025_q3", "2025_q4"])
    pooled_incumbent = _pooled_metric(
        bundles,
        {fold: bundles[fold].incumbent for fold in ["2025_q3", "2025_q4"]},
        ["2025_q3", "2025_q4"],
    )
    pooled_raw = _pooled_metric(
        bundles,
        {fold: bundles[fold].raw_candidate for fold in ["2025_q3", "2025_q4"]},
        ["2025_q3", "2025_q4"],
    )
    passed = all(gate_checks.values())
    return {
        "schema_version": "p1.mstcn_segment_precision_router.retroaudit.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "PASS_RETROSPECTIVE_SIGNAL" if passed else "NO_GO_RETROSPECTIVE_GATE",
        "passed_all_diagnostic_gates": passed,
        "role": config["role"],
        "fold_results": reports,
        "pooled_q3_q4": {
            "incumbent": pooled_incumbent,
            "raw_e150": pooled_raw,
            "core_router": pooled_core,
            "core_delta_f1_vs_incumbent": float(pooled_core["f1"] - pooled_incumbent["f1"]),
            "core_delta_f1_vs_raw_e150": float(pooled_core["f1"] - pooled_raw["f1"]),
        },
        "fold_segment_inventory": {
            fold: {
                "segments": int(len(bundle.segments)),
                "beneficial_segments": int(bundle.segments["beneficial"].sum()),
                "added_rows": int(
                    ((bundle.raw_candidate == 1) & (bundle.incumbent == 0)).sum()
                ),
            }
            for fold, bundle in bundles.items()
        },
        "gate_checks": gate_checks,
        "limitations": config["limitations"],
        "input_hashes": {
            "config": sha256_file(CONFIG_PATH),
            "anchor": sha256_file(ANCHOR_PATH),
            "truth": sha256_file(TRUTH_PATH),
            **{fold: sha256_file(path) for fold, path in FOLD_PATHS.items()},
        },
        "operation_counters": {
            "official_test_rows_read": 0,
            "submission_files_created": 0,
            "uploads": 0,
            "model_fits": 3,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    required = [CONFIG_PATH, ANCHOR_PATH, TRUTH_PATH, *FOLD_PATHS.values()]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, f"missing required inputs: {missing}")
    if not args.execute:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "READY_RETROSPECTIVE_ONLY",
                    "official_test_rows_read": 0,
                    "submission_files_created": 0,
                },
                indent=2,
            )
        )
        return
    result = execute()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "result.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed_all_diagnostic_gates": result["passed_all_diagnostic_gates"],
                "pooled_q3_q4": result["pooled_q3_q4"],
                "gate_checks": result["gate_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
