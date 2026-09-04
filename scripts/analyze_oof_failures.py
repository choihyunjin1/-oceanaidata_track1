"""Aggregate research-only failure diagnostics across P1 outer-fold OOF predictions.

This script intentionally reads organizer-provided train labels.  Its output is
descriptive/oracle diagnostic evidence only: it must never be used to select a
candidate, tune a threshold, fit a gate, or claim hidden-test performance.
Only aggregate counts and rates are written; no observation rows are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEY_COLUMNS = ("station", "year", "layer", "time")
SOURCE_COLUMNS = (*KEY_COLUMNS, "temp", "depth", "label", "anomaly_type")
REQUIRED_DATA_FILES = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "baseline_rule.csv",
    "README.md",
)
ANOMALY_TYPES = ("spike", "noise", "flatline", "offset", "drift")
TREE_COLUMNS = (
    *KEY_COLUMNS,
    "label",
    "probability",
    "prediction",
    "anomaly_type",
    "fold",
)
SCORE_EDGES = (-np.inf, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95, np.inf)
SCORE_LABELS = (
    "<0.05",
    "0.05-<0.10",
    "0.10-<0.20",
    "0.20-<0.40",
    "0.40-<0.60",
    "0.60-<0.80",
    "0.80-<0.95",
    ">=0.95",
)
RESEARCH_ONLY_WARNING = (
    "OUTER-LABEL RESEARCH DIAGNOSTIC ONLY. Do not use these aggregates or oracle "
    "results for candidate selection, threshold tuning, promotion, training, or hidden-test claims."
)


@dataclass(frozen=True)
class ModelOOF:
    """One aligned OOF prediction surface and its provenance files."""

    name: str
    display_name: str
    probability: np.ndarray
    prediction: np.ndarray
    source_paths: tuple[Path, ...]
    metadata: Mapping[str, Any]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_data_dir(project_root: Path, override: str | Path | None = None) -> Path:
    """Resolve P1_DATA_DIR or one unambiguous complete organizer file set."""

    configured = override or os.environ.get("P1_DATA_DIR")
    if configured:
        candidates = [Path(configured).expanduser().resolve()]
    else:
        candidates = sorted(
            {
                path.parent.resolve()
                for path in project_root.rglob("train.csv")
                if all((path.parent / name).is_file() for name in REQUIRED_DATA_FILES)
            }
        )
    if len(candidates) != 1:
        raise FileNotFoundError(
            "set P1_DATA_DIR or --data-dir; exactly one complete P1 file set is required, "
            f"found {len(candidates)}"
        )
    candidate = candidates[0]
    missing = [name for name in REQUIRED_DATA_FILES if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P1 data directory is missing {missing}: {candidate}")
    return candidate


def duration_bucket(rows: pd.Series | np.ndarray) -> pd.Categorical:
    """Bucket 10-minute row counts using official anomaly-duration landmarks."""

    values = np.asarray(rows, dtype=float)
    labels = np.select(
        [
            values == 1,
            (values > 1) & (values < 18),
            (values >= 18) & (values < 48),
            (values >= 48) & (values < 144),
            (values >= 144) & (values < 288),
            values >= 288,
        ],
        ["10m", "20m-<3h", "3h-<8h", "8h-<24h", "24h-<48h", ">=48h"],
        default="not_applicable",
    )
    categories = ["10m", "20m-<3h", "3h-<8h", "8h-<24h", "24h-<48h", ">=48h"]
    return pd.Categorical(labels, categories=categories, ordered=True)


def confusion_metrics(label: Sequence[int], prediction: Sequence[int]) -> dict[str, Any]:
    truth = np.asarray(label, dtype=np.int8)
    pred = np.asarray(prediction, dtype=np.int8)
    if truth.shape != pred.shape:
        raise ValueError("label and prediction shapes differ")
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "rows": int(len(truth)),
        "positive_rows": int(np.sum(truth)),
        "predicted_positive_rows": int(np.sum(pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def confusion_by_dimension(
    frame: pd.DataFrame,
    prediction: Sequence[int],
    dimension: str,
) -> list[dict[str, Any]]:
    """Return additive confusion counts and contextual error shares per category."""

    pred = np.asarray(prediction, dtype=np.int8)
    truth = frame["label"].to_numpy(dtype=np.int8)
    if len(pred) != len(frame):
        raise ValueError("prediction length does not match frame")
    work = pd.DataFrame(
        {
            "category": frame[dimension].astype("string").fillna("missing"),
            "rows": np.ones(len(frame), dtype=np.int64),
            "positive_rows": truth,
            "predicted_positive_rows": pred,
            "tp": (truth == 1) & (pred == 1),
            "fp": (truth == 0) & (pred == 1),
            "fn": (truth == 1) & (pred == 0),
            "tn": (truth == 0) & (pred == 0),
        }
    )
    grouped = work.groupby("category", sort=True, observed=True).sum().reset_index()
    total_fp = int(grouped["fp"].sum())
    total_fn = int(grouped["fn"].sum())
    records: list[dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        tp, fp, fn = int(row.tp), int(row.fp), int(row.fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        records.append(
            {
                "category": str(row.category),
                "rows": int(row.rows),
                "positive_rows": int(row.positive_rows),
                "predicted_positive_rows": int(row.predicted_positive_rows),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": int(row.tn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp_share_of_model": fp / total_fp if total_fp else 0.0,
                "fn_share_of_model": fn / total_fn if total_fn else 0.0,
            }
        )
    return records


def anomaly_membership_summary(
    frame: pd.DataFrame, prediction: Sequence[int]
) -> list[dict[str, Any]]:
    """Summarize normal rows and multi-label type memberships (non-additive)."""

    signature = frame["anomaly_signature"].astype(str)
    records: list[dict[str, Any]] = []
    categories: Iterable[tuple[str, np.ndarray]] = [
        ("normal", frame["label"].to_numpy(dtype=np.int8) == 0),
        *[
            (
                anomaly_type,
                signature.str.split("+")
                .map(lambda values, target=anomaly_type: target in values)
                .to_numpy(),
            )
            for anomaly_type in ANOMALY_TYPES
        ],
    ]
    pred = np.asarray(prediction, dtype=np.int8)
    for category, mask in categories:
        metrics = confusion_metrics(frame.loc[mask, "label"], pred[mask])
        records.append({"category": category, **metrics})
    return records


def build_event_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach true-event duration/composite state without crossing gaps or groups."""

    work = frame.copy()
    work["_original_order"] = np.arange(len(work), dtype=np.int64)
    work["_time_kst"] = pd.to_datetime(work["time"], utc=True).dt.tz_convert("Asia/Seoul")
    work = work.sort_values(["station", "layer", "_time_kst"], kind="stable").reset_index(drop=True)
    same_group = work["station"].eq(work["station"].shift()) & work["layer"].eq(
        work["layer"].shift()
    )
    cadence = work["_time_kst"].diff().eq(pd.Timedelta(minutes=10))
    work["_stream_break"] = ~(same_group & cadence)
    positive = work["label"].eq(1)
    new_event = positive & (work["_stream_break"] | ~positive.shift(fill_value=False))
    event_id = new_event.cumsum().astype("int64")
    work["true_event_id"] = event_id.where(positive, pd.NA).astype("Int64")
    work["_sequence_order"] = np.arange(len(work), dtype=np.int64)

    positive_rows = work.loc[positive, ["true_event_id", "anomaly_signature"]].copy()
    event_sizes = positive_rows.groupby("true_event_id", observed=True).size()

    def union_signature(values: pd.Series) -> str:
        memberships: set[str] = set()
        for value in values.astype(str):
            memberships.update(part for part in value.split("+") if part in ANOMALY_TYPES)
        return "+".join(value for value in ANOMALY_TYPES if value in memberships)

    event_signatures = positive_rows.groupby("true_event_id", observed=True)[
        "anomaly_signature"
    ].agg(union_signature)
    work["true_event_rows"] = work["true_event_id"].map(event_sizes).astype("Float64")
    work["true_event_signature"] = work["true_event_id"].map(event_signatures).fillna("normal")
    duration = pd.Series("normal", index=work.index, dtype="string")
    duration.loc[positive] = duration_bucket(work.loc[positive, "true_event_rows"]).astype(str)
    work["true_event_duration_bucket"] = duration
    event_type_count = work["true_event_signature"].str.count(r"\+") + 1
    work["event_composite_state"] = np.select(
        [~positive, event_type_count.gt(1)], ["normal", "composite"], default="single"
    )
    work = work.sort_values("_original_order", kind="stable").reset_index(drop=True)
    return work.drop(columns=["_original_order", "_time_kst"])


def true_event_diagnostics(
    frame: pd.DataFrame, prediction: Sequence[int], group_column: str
) -> list[dict[str, Any]]:
    """Aggregate event detection and FN rows by duration or composite category."""

    pred = np.asarray(prediction, dtype=np.int8)
    work = frame.loc[frame["label"].eq(1), ["true_event_id", group_column]].copy()
    work["prediction"] = pred[frame["label"].to_numpy(dtype=np.int8) == 1]
    events = (
        work.groupby("true_event_id", observed=True)
        .agg(
            category=(group_column, "first"),
            positive_rows=("prediction", "size"),
            detected_rows=("prediction", "sum"),
            detected=("prediction", "max"),
        )
        .reset_index(drop=True)
    )
    grouped = (
        events.groupby("category", observed=True, sort=True)
        .agg(
            events=("detected", "size"),
            detected_events=("detected", "sum"),
            positive_rows=("positive_rows", "sum"),
            detected_rows=("detected_rows", "sum"),
        )
        .reset_index()
    )
    records: list[dict[str, Any]] = []
    for row in grouped.itertuples(index=False):
        records.append(
            {
                "category": str(row.category),
                "events": int(row.events),
                "detected_events": int(row.detected_events),
                "missed_events": int(row.events - row.detected_events),
                "event_recall": float(row.detected_events / row.events),
                "positive_rows": int(row.positive_rows),
                "fn_rows": int(row.positive_rows - row.detected_rows),
                "row_recall": float(row.detected_rows / row.positive_rows),
            }
        )
    return records


def false_positive_run_diagnostics(
    frame: pd.DataFrame, prediction: Sequence[int]
) -> list[dict[str, Any]]:
    """Describe contiguous 10-minute false-positive runs without exporting rows."""

    ordered = frame.assign(_prediction=np.asarray(prediction, dtype=np.int8)).sort_values(
        "_sequence_order", kind="stable"
    )
    fp = ordered["label"].eq(0) & ordered["_prediction"].eq(1)
    new_run = fp & (ordered["_stream_break"] | ~fp.shift(fill_value=False))
    ordered["_fp_run_id"] = new_run.cumsum().where(fp, pd.NA).astype("Int64")
    runs = ordered.loc[fp].groupby("_fp_run_id", observed=True).size().rename("rows")
    if runs.empty:
        return []
    run_frame = runs.reset_index(drop=True).to_frame()
    run_frame["category"] = duration_bucket(run_frame["rows"]).astype(str)
    grouped = run_frame.groupby("category", observed=True, sort=True)["rows"].agg(
        runs="size", fp_rows="sum"
    )
    return [
        {"category": str(category), "runs": int(row.runs), "fp_rows": int(row.fp_rows)}
        for category, row in grouped.iterrows()
    ]


def oracle_diagnostics(
    label: Sequence[int], predictions: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    """Compute explicitly forbidden label-aware ceilings and label-free set operations."""

    truth = np.asarray(label, dtype=np.int8)
    names = list(predictions)
    matrix = np.column_stack([np.asarray(predictions[name], dtype=np.int8) for name in names])
    if matrix.shape[0] != len(truth):
        raise ValueError("prediction matrix and label lengths differ")
    union = matrix.max(axis=1)
    intersection = matrix.min(axis=1)
    label_oracle = np.where(truth == 1, union, intersection).astype(np.int8)
    per_model: dict[str, Any] = {}
    for column, name in enumerate(names):
        pred = matrix[:, column]
        other = np.delete(matrix, column, axis=1)
        fn = (truth == 1) & (pred == 0)
        fp = (truth == 0) & (pred == 1)
        recoverable = fn & (other.max(axis=1) == 1)
        rejectable = fp & (other.min(axis=1) == 0)
        per_model[name] = {
            "fn_rows": int(fn.sum()),
            "fn_recoverable_by_at_least_one_other_model": int(recoverable.sum()),
            "fn_recoverable_share": float(recoverable.sum() / fn.sum()) if fn.any() else 0.0,
            "fp_rows": int(fp.sum()),
            "fp_rejectable_by_at_least_one_other_model": int(rejectable.sum()),
            "fp_rejectable_share": float(rejectable.sum() / fp.sum()) if fp.any() else 0.0,
        }
    return {
        "warning": RESEARCH_ONLY_WARNING,
        "union_prediction_label_free": confusion_metrics(truth, union),
        "intersection_prediction_label_free": confusion_metrics(truth, intersection),
        "per_row_label_oracle_unimplementable": confusion_metrics(truth, label_oracle),
        "per_model_label_aware_opportunity": per_model,
    }


def disagreement_diagnostics(frame: pd.DataFrame, models: Sequence[ModelOOF]) -> dict[str, Any]:
    predictions = {model.name: model.prediction for model in models}
    matrix = np.column_stack(list(predictions.values())).astype(np.int8)
    truth = frame["label"].to_numpy(dtype=np.int8)
    votes = matrix.sum(axis=1)
    vote_frame = frame.assign(model_positive_votes=votes.astype(str))
    by_vote = confusion_by_dimension(
        vote_frame, (votes >= 1).astype(np.int8), "model_positive_votes"
    )
    for record in by_vote:
        vote_count = int(record["category"])
        mask = votes == vote_count
        record["actual_positive_rate"] = float(truth[mask].mean()) if mask.any() else 0.0
        record["unanimous"] = vote_count in {0, matrix.shape[1]}

    unanimous_error = ((votes == 0) & (truth == 1)) | ((votes == matrix.shape[1]) & (truth == 0))
    consensus_prediction = (votes == matrix.shape[1]).astype(np.int8)
    unanimous_residual = {
        dimension: confusion_by_dimension(
            frame.loc[unanimous_error].reset_index(drop=True),
            consensus_prediction[unanimous_error],
            dimension,
        )
        for dimension in (
            "station_layer",
            "month",
            "anomaly_signature",
            "true_event_duration_bucket",
            "event_composite_state",
            "peer_state",
            "depth_state",
        )
    }

    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            a, b = left.prediction, right.prediction
            disagree = a != b
            a_correct = a == truth
            b_correct = b == truth
            union_positive = (a == 1) | (b == 1)
            both_positive = (a == 1) & (b == 1)
            pairs.append(
                {
                    "model_a": left.name,
                    "model_b": right.name,
                    "disagreement_rows": int(disagree.sum()),
                    "disagreement_rate": float(disagree.mean()),
                    "a_correct_b_wrong": int(np.sum(disagree & a_correct & ~b_correct)),
                    "b_correct_a_wrong": int(np.sum(disagree & b_correct & ~a_correct)),
                    "positive_jaccard": float(both_positive.sum() / union_positive.sum())
                    if union_positive.any()
                    else 1.0,
                    "probability_correlation": float(
                        np.corrcoef(left.probability, right.probability)[0, 1]
                    ),
                }
            )
    return {
        "model_count": len(models),
        "by_positive_vote_count": by_vote,
        "unanimous_error_rows": int(unanimous_error.sum()),
        "unanimous_residual_by_dimension": unanimous_residual,
        "pairwise": pairs,
        "oracle": oracle_diagnostics(truth, predictions),
    }


def _relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _discover_tree_candidates(project_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    selected: dict[str, tuple[Path, dict[str, Any]]] = {}
    for metrics_path in sorted((project_root / "artifacts" / "runs").glob("*_cv_*/metrics.json")):
        run_dir = metrics_path.parent
        oof_path, manifest_path = run_dir / "oof.parquet", run_dir / "manifest.json"
        if not oof_path.is_file() or not manifest_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("mode") != "offline":
            continue
        backend = str(metrics.get("backend", ""))
        augmented = bool(metrics.get("augmentation", False))
        if backend == "lightgbm" and augmented:
            name = "lightgbm_augmented"
        elif backend in {"lightgbm", "xgboost", "catboost"} and not augmented:
            name = backend
        else:
            continue
        selected[name] = (run_dir, metrics)
    required = {"lightgbm", "lightgbm_augmented", "xgboost", "catboost"}
    missing = sorted(required - set(selected))
    if missing:
        raise FileNotFoundError(f"missing complete offline tree OOF artifacts: {missing}")
    return selected


def _load_tree_model(
    project_root: Path,
    name: str,
    run_dir: Path,
    metrics: Mapping[str, Any],
    master: pd.DataFrame,
) -> ModelOOF:
    oof_path = run_dir / "oof.parquet"
    oof = pd.read_parquet(oof_path, columns=list(TREE_COLUMNS))
    if oof.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{name} OOF contains duplicate keys")
    aligned = master.loc[:, [*KEY_COLUMNS]].merge(
        oof,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned["probability"].isna().any() or len(aligned) != len(master):
        raise ValueError(f"{name} OOF does not cover the common outer population")
    prediction = aligned["prediction"].to_numpy(dtype=np.int8)
    probability = aligned["probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"{name} OOF probability is outside [0, 1]")
    if not np.isin(prediction, [0, 1]).all():
        raise ValueError(f"{name} OOF prediction is not binary")
    if not np.array_equal(aligned["label"].to_numpy(dtype=np.int8), master["label"]):
        raise ValueError(f"{name} OOF labels disagree with source train labels")
    display = {
        "lightgbm": "LightGBM",
        "lightgbm_augmented": "LightGBM + synthetic augmentation",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
    }[name]
    return ModelOOF(
        name=name,
        display_name=display,
        probability=probability,
        prediction=prediction,
        source_paths=(oof_path, run_dir / "metrics.json", run_dir / "manifest.json"),
        metadata={
            "family": "tree",
            "backend": metrics.get("backend"),
            "augmentation": bool(metrics.get("augmentation", False)),
            "mode": metrics.get("mode"),
            "run_id": run_dir.name,
        },
    )


def _discover_sequence_experiment(project_root: Path) -> tuple[Path, dict[str, Any]]:
    eligible: list[tuple[Path, dict[str, Any]]] = []
    for path in (project_root / "artifacts").glob("sequence*/sequence_experiment.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        experiment = payload.get("experiment", {})
        if (
            experiment.get("smoke") is False
            and experiment.get("fold_count") == 3
            and experiment.get("selection_uses_outer_labels") is False
        ):
            eligible.append((path, payload))
    if len(eligible) != 1:
        raise FileNotFoundError(
            "expected exactly one complete non-smoke, three-fold sequence experiment; "
            f"found {len(eligible)}"
        )
    return eligible[0]


def _load_deep_models(
    experiment_path: Path,
    payload: Mapping[str, Any],
    master: pd.DataFrame,
) -> list[ModelOOF]:
    models: list[ModelOOF] = []
    expected_rows = master["source_row_index"].to_numpy(dtype=np.int64)
    for architecture in ("tcn", "patch_transformer"):
        path = experiment_path.parent / f"oof_{architecture}.npz"
        with np.load(path, allow_pickle=False) as archive:
            required = {"row_index", "label", "probability", "prediction"}
            if set(archive.files) != required:
                raise ValueError(f"unexpected {architecture} OOF arrays: {archive.files}")
            row_index = archive["row_index"].astype(np.int64)
            if len(np.unique(row_index)) != len(row_index):
                raise ValueError(f"{architecture} OOF row_index is not unique")
            positions = pd.Series(np.arange(len(row_index)), index=row_index).reindex(expected_rows)
            if positions.isna().any() or len(row_index) != len(expected_rows):
                raise ValueError(
                    f"{architecture} OOF population differs from tree outer population"
                )
            order = positions.to_numpy(dtype=np.int64)
            label = archive["label"][order].astype(np.int8)
            probability = archive["probability"][order].astype(np.float64)
            prediction = archive["prediction"][order].astype(np.int8)
        if not np.array_equal(label, master["label"].to_numpy(dtype=np.int8)):
            raise ValueError(f"{architecture} OOF labels disagree with source train labels")
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{architecture} OOF probability is outside [0, 1]")
        if not np.isin(prediction, [0, 1]).all():
            raise ValueError(f"{architecture} OOF prediction is not binary")
        models.append(
            ModelOOF(
                name=architecture,
                display_name="TCN" if architecture == "tcn" else "Patch Transformer",
                probability=probability,
                prediction=prediction,
                source_paths=(path, experiment_path),
                metadata={
                    "family": "deep",
                    "architecture": architecture,
                    "mode": payload["experiment"].get("feature_mode"),
                    "selection_uses_outer_labels": payload["experiment"].get(
                        "selection_uses_outer_labels"
                    ),
                    "outer_labels_used_once_for_diagnostics": payload["experiment"].get(
                        "outer_labels_used_once_for_diagnostics"
                    ),
                    "experiment_dir": experiment_path.parent.name,
                },
            )
        )
    return models


def _load_master_population(
    train_path: Path,
    reference_oof_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        train_path,
        usecols=list(SOURCE_COLUMNS),
        dtype={"station": "string", "time": "string", "anomaly_type": "string"},
    )
    source.insert(0, "source_row_index", np.arange(len(source), dtype=np.int64))
    if source.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("source train keys are not unique")
    reference = pd.read_parquet(reference_oof_path, columns=list(TREE_COLUMNS))
    if reference.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("reference OOF keys are not unique")
    master = reference.loc[:, list(KEY_COLUMNS)].merge(
        source,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if master["source_row_index"].isna().any() or len(master) != len(reference):
        raise ValueError("reference OOF keys do not map one-to-one to source train")
    master["source_row_index"] = master["source_row_index"].astype(np.int64)
    if not np.array_equal(reference["label"].to_numpy(dtype=np.int8), master["label"]):
        raise ValueError("reference OOF labels disagree with source train labels")
    source_signature = master["anomaly_type"].fillna("").astype(str)
    reference_signature = reference["anomaly_type"].fillna("").astype(str)
    if not source_signature.reset_index(drop=True).equals(
        reference_signature.reset_index(drop=True)
    ):
        raise ValueError("reference OOF anomaly_type disagrees with source train")
    master["fold"] = reference["fold"].astype(str).to_numpy()
    return source, master


def _attach_context(
    project_root: Path,
    train_path: Path,
    source: pd.DataFrame,
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, Path | None, dict[str, Any]]:
    source_hash = sha256_file(train_path)
    cache_path: Path | None = None
    cache_metadata: dict[str, Any] = {}
    for metadata_path in sorted(
        (project_root / "artifacts" / "cache").glob("train_offline_*.json")
    ):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        parquet_path = metadata_path.with_suffix(".parquet")
        if payload.get("source_sha256") == source_hash and parquet_path.is_file():
            if cache_path is not None:
                raise ValueError("multiple offline feature caches match the source train hash")
            cache_path = parquet_path
            cache_metadata = payload
    if cache_path is not None:
        context = pd.read_parquet(
            cache_path, columns=["peer_available", "peer_count", "depth_missing"]
        )
        if len(context) != len(source):
            raise ValueError("offline feature cache row count differs from source train")
        expected_hash = cache_metadata.get("parquet_sha256")
        if expected_hash and sha256_file(cache_path) != expected_hash:
            raise ValueError("offline feature cache SHA-256 differs from its metadata")
        selected = context.iloc[master["source_row_index"].to_numpy(dtype=np.int64)].reset_index(
            drop=True
        )
        peer_available = selected["peer_available"].fillna(0).astype(bool)
        depth_missing = selected["depth_missing"].fillna(1).astype(bool)
        context_source = "verified offline feature cache"
    else:
        simultaneous = source.groupby(["station", "time"], observed=True)["temp"].transform("count")
        peer_count = simultaneous - source["temp"].notna().astype(int)
        positions = master["source_row_index"].to_numpy(dtype=np.int64)
        peer_available = peer_count.iloc[positions].reset_index(drop=True).gt(0)
        depth_missing = source["depth"].iloc[positions].reset_index(drop=True).isna()
        context_source = "source-derived fallback"

    result = master.copy().reset_index(drop=True)
    result["month"] = (
        pd.to_datetime(result["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m")
    )
    result["station_layer"] = result["station"].astype(str) + "/L" + result["layer"].astype(str)
    result["anomaly_signature"] = result["anomaly_type"].fillna("").astype(str)
    result.loc[result["label"].eq(0), "anomaly_signature"] = "normal"
    plus_count = result["anomaly_signature"].str.count(r"\+")
    result["composite_state"] = np.select(
        [result["label"].eq(0), plus_count.gt(0)],
        ["normal", "composite"],
        default="single",
    )
    result["peer_state"] = np.where(peer_available, "peer", "no_peer")
    result["depth_state"] = np.where(depth_missing, "depth_missing", "depth_present")
    result = build_event_context(result)
    result["station_layer_anomaly"] = result["station_layer"] + "|" + result["anomaly_signature"]
    result["month_station"] = result["month"] + "|" + result["station"].astype(str)
    result["peer_station"] = result["peer_state"] + "|" + result["station"].astype(str)
    result["depth_station"] = result["depth_state"] + "|" + result["station"].astype(str)
    result["event_duration_composite"] = (
        result["true_event_duration_bucket"] + "|" + result["event_composite_state"]
    )
    quality = {
        "context_source": context_source,
        "feature_cache_rows": len(source) if cache_path is not None else None,
        "feature_cache_source_sha256_match": cache_path is not None,
        "outer_rows": len(result),
        "outer_unique_keys": int(result.drop_duplicates(list(KEY_COLUMNS)).shape[0]),
        "outer_positive_rows": int(result["label"].sum()),
        "outer_time_min_kst": str(result["time"].min()),
        "outer_time_max_kst": str(result["time"].max()),
        "peer_available_rate": float(peer_available.mean()),
        "depth_missing_rate": float(depth_missing.mean()),
    }
    return result, cache_path, quality


def _model_summary(frame: pd.DataFrame, model: ModelOOF) -> dict[str, Any]:
    score_frame = frame.assign(
        score_band=pd.cut(
            model.probability,
            bins=SCORE_EDGES,
            labels=SCORE_LABELS,
            right=False,
            include_lowest=True,
        ).astype(str)
    )
    dimensions = {
        name: confusion_by_dimension(score_frame, model.prediction, name)
        for name in (
            "station",
            "layer",
            "station_layer",
            "station_layer_anomaly",
            "fold",
            "month",
            "month_station",
            "anomaly_signature",
            "composite_state",
            "true_event_duration_bucket",
            "event_composite_state",
            "event_duration_composite",
            "peer_state",
            "peer_station",
            "depth_state",
            "depth_station",
            "score_band",
        )
    }
    return {
        "display_name": model.display_name,
        "metadata": dict(model.metadata),
        "overall": confusion_metrics(frame["label"], model.prediction),
        "dimensions": dimensions,
        "anomaly_type_membership_non_additive": anomaly_membership_summary(frame, model.prediction),
        "true_events_by_duration": true_event_diagnostics(
            frame, model.prediction, "true_event_duration_bucket"
        ),
        "true_events_by_composite_state": true_event_diagnostics(
            frame, model.prediction, "event_composite_state"
        ),
        "false_positive_runs_by_duration": false_positive_run_diagnostics(frame, model.prediction),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def validate_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile all additive cuts to their controlling model-level totals."""

    count_fields = (
        "rows",
        "positive_rows",
        "predicted_positive_rows",
        "tp",
        "fp",
        "fn",
        "tn",
    )
    additive_dimensions = (
        "station",
        "layer",
        "station_layer",
        "station_layer_anomaly",
        "fold",
        "month",
        "month_station",
        "anomaly_signature",
        "composite_state",
        "true_event_duration_bucket",
        "event_composite_state",
        "event_duration_composite",
        "peer_state",
        "peer_station",
        "depth_state",
        "depth_station",
        "score_band",
    )
    checks: list[str] = []
    event_count: int | None = None
    for model_name, model in summary["models"].items():
        overall = model["overall"]
        for dimension in additive_dimensions:
            records = model["dimensions"][dimension]
            for field in count_fields:
                subtotal = sum(int(record[field]) for record in records)
                if subtotal != int(overall[field]):
                    raise ValueError(
                        f"{model_name}/{dimension}/{field} subtotal {subtotal} "
                        f"does not match overall {overall[field]}"
                    )
        checks.append(f"{model_name}: all additive dimension subtotals reconcile")

        for event_cut in ("true_events_by_duration", "true_events_by_composite_state"):
            records = model[event_cut]
            positive_rows = sum(int(record["positive_rows"]) for record in records)
            fn_rows = sum(int(record["fn_rows"]) for record in records)
            events = sum(int(record["events"]) for record in records)
            if positive_rows != int(overall["positive_rows"]) or fn_rows != int(overall["fn"]):
                raise ValueError(f"{model_name}/{event_cut} row subtotals do not reconcile")
            if event_count is None:
                event_count = events
            elif events != event_count:
                raise ValueError(f"{model_name}/{event_cut} true-event count changed")
        fp_run_rows = sum(
            int(record["fp_rows"]) for record in model["false_positive_runs_by_duration"]
        )
        if fp_run_rows != int(overall["fp"]):
            raise ValueError(f"{model_name} false-positive run rows do not reconcile")
        checks.append(f"{model_name}: true events and FP-run rows reconcile")

    disagreement = summary["disagreement"]
    vote_rows = sum(int(record["rows"]) for record in disagreement["by_positive_vote_count"])
    expected_rows = int(summary["data_quality"]["outer_rows"])
    if vote_rows != expected_rows:
        raise ValueError("model-vote row subtotal does not match outer population")
    oracle = disagreement["oracle"]["per_row_label_oracle_unimplementable"]
    oracle_errors = int(oracle["fp"]) + int(oracle["fn"])
    if oracle_errors != int(disagreement["unanimous_error_rows"]):
        raise ValueError("unanimous error count does not match per-row oracle residual")
    checks.append("model-vote rows and unanimous oracle residual reconcile")
    return {
        "status": "passed",
        "checks": checks,
        "model_count": len(summary["models"]),
        "outer_rows": expected_rows,
        "true_events": event_count,
    }


def run_analysis(
    project_root: Path,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the full aggregate-only research diagnostic and write JSON + manifest."""

    train_path = data_dir / "train.csv"
    tree_candidates = _discover_tree_candidates(project_root)
    reference_run, _ = tree_candidates["xgboost"]
    source, master = _load_master_population(train_path, reference_run / "oof.parquet")
    frame, cache_path, quality = _attach_context(project_root, train_path, source, master)

    tree_models = [
        _load_tree_model(project_root, name, run_dir, metrics, frame)
        for name, (run_dir, metrics) in tree_candidates.items()
    ]
    sequence_path, sequence_payload = _discover_sequence_experiment(project_root)
    deep_models = _load_deep_models(sequence_path, sequence_payload, frame)
    model_order = (
        "lightgbm",
        "xgboost",
        "catboost",
        "lightgbm_augmented",
        "tcn",
        "patch_transformer",
    )
    by_name = {model.name: model for model in [*tree_models, *deep_models]}
    models = [by_name[name] for name in model_order]

    artifact_paths = {train_path, sequence_path}
    if cache_path is not None:
        artifact_paths.add(cache_path)
        artifact_paths.add(cache_path.with_suffix(".json"))
    for model in models:
        artifact_paths.update(model.source_paths)
    provenance = [
        {
            "logical_path": (
                f"P1_DATA_DIR/{path.name}"
                if path.resolve() == train_path.resolve()
                else _relative(project_root, path)
            ),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(artifact_paths, key=lambda item: str(item))
    ]

    summary: dict[str, Any] = {
        "contract_version": 1,
        "warning": RESEARCH_ONLY_WARNING,
        "use_policy": {
            "research_only": True,
            "outer_validation_labels_used": True,
            "allowed_uses": [
                "aggregate failure diagnosis",
                "hypothesis generation",
                "unimplementable oracle upper-bound estimation",
            ],
            "forbidden_uses": [
                "candidate selection",
                "threshold or post-processing tuning",
                "model fitting or stacking",
                "promotion or submission decision",
                "hidden-test or official-score claim",
            ],
        },
        "scope": {
            "grain": "one outer-validation station-layer-time row",
            "timezone": "Asia/Seoul (+09:00)",
            "population": "three purged rolling-origin outer holdouts shared by all models",
            "model_order": list(model_order),
            "model_count": len(models),
        },
        "data_quality": quality,
        "models": {model.name: _model_summary(frame, model) for model in models},
        "disagreement": disagreement_diagnostics(frame, models),
        "provenance": provenance,
    }
    summary["validation"] = validate_summary(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "research_only_failure_summary.json"
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n"
    ).encode("utf-8")
    summary_path.write_bytes(summary_bytes)
    manifest = {
        "contract_version": 1,
        "warning": RESEARCH_ONLY_WARNING,
        "generated_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": _relative(project_root, Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "summary": _relative(project_root, summary_path),
        "summary_bytes": len(summary_bytes),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "source_train_sha256": sha256_file(train_path),
        "raw_rows_exported": 0,
        "external_observations_used": 0,
        "aggregate_validation": summary["validation"],
        "models": [model.name for model in models],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "warning": RESEARCH_ONLY_WARNING,
        "summary": str(summary_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "summary_sha256": manifest["summary_sha256"],
        "models": manifest["models"],
        "outer_rows": len(frame),
        "raw_rows_exported": 0,
        "external_observations_used": 0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/failure_recon_20260813"),
    )
    parser.add_argument(
        "--acknowledge-research-only",
        action="store_true",
        help="required acknowledgement that outer labels cannot tune or promote a candidate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.acknowledge_research_only:
        print(f"FAIL: {RESEARCH_ONLY_WARNING}")
        print("Re-run with --acknowledge-research-only to produce aggregate diagnostics.")
        return 2
    project_root = args.project_root.expanduser().resolve()
    try:
        data_dir = resolve_data_dir(project_root, args.data_dir)
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        receipt = run_analysis(project_root, data_dir, output_dir.resolve())
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
