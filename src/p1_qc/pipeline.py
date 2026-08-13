"""End-to-end tabular CV, training, inference, and model persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from p1_qc.augment import AugmentConfig, augment_training_fold
from p1_qc.config import P1QCConfig
from p1_qc.data import ANOMALY_TYPES
from p1_qc.experiment import sha256_file, stable_hash, write_json
from p1_qc.features import FeatureBundle, build_features
from p1_qc.metrics import evaluate_predictions, group_row_shares, micro_f1
from p1_qc.models_tabular import DeterministicTabularClassifier, make_tabular_classifier
from p1_qc.postprocess import (
    PostprocessConfig,
    _chronological_order,
    close_short_gaps,
    hysteresis_threshold,
    remove_short_runs,
)
from p1_qc.rules import detect_plateaus, detect_singleton_spikes, evaluate_binary_rule
from p1_qc.splits import Fold, outer_folds
from p1_qc.submission import build_submission, validate_submission, write_submission
from p1_qc.validation import paired_block_bootstrap

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_data_dir(config: P1QCConfig, override: str | Path | None = None) -> Path:
    if override is not None:
        candidate = Path(override)
    elif config.paths.data_dir is not None:
        candidate = config.paths.data_dir
    else:
        matches = [
            path.parent
            for path in PROJECT_ROOT.rglob("train.csv")
            if path.parent.name == "P1_qc_anomaly"
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                "set P1_DATA_DIR or --data-dir; safe fallback requires exactly one P1_qc_anomaly"
            )
        candidate = matches[0]
    candidate = candidate.expanduser().resolve()
    required = ("train.csv", "test.csv", "sample_submission.csv", "baseline_rule.csv", "README.md")
    missing = [name for name in required if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P1 data directory is missing {missing}: {candidate}")
    return candidate


def _cache_directory(config: P1QCConfig) -> Path:
    raw_project = config.raw.get("project", {})
    value = (
        raw_project.get("cache_dir", "artifacts/cache")
        if isinstance(raw_project, Mapping)
        else "artifacts/cache"
    )
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_or_build_features(
    frame: pd.DataFrame,
    config: P1QCConfig,
    *,
    kind: str,
    use_cache: bool = True,
) -> FeatureBundle:
    source_hash = frame.attrs.get("source_sha256", "unknown")
    key = stable_hash(
        {
            "source_sha256": source_hash,
            "kind": kind,
            "features": asdict(config.features),
            "cadence": config.data.cadence_minutes,
        }
    )[:16]
    cache = _cache_directory(config)
    parquet = cache / f"{kind}_{config.features.mode}_{key}.parquet"
    metadata_path = cache / f"{kind}_{config.features.mode}_{key}.json"
    if use_cache and parquet.is_file() and metadata_path.is_file():
        feature_frame = pd.read_parquet(parquet)
        metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
        if len(feature_frame) == len(frame) and metadata.get("source_sha256") == source_hash:
            feature_frame.index = frame.index.copy()
            return FeatureBundle(
                feature_frame,
                tuple(metadata["feature_columns"]),
                tuple(metadata["categorical_columns"]),
            )
    bundle = build_features(
        frame,
        config=config,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    if use_cache:
        cache.mkdir(parents=True, exist_ok=True)
        bundle.frame.to_parquet(parquet, index=False, compression="zstd")
        write_json(
            metadata_path,
            {
                "source_sha256": source_hash,
                "feature_columns": list(bundle.feature_columns),
                "categorical_columns": list(bundle.categorical_columns),
                "rows": len(frame),
                "parquet_sha256": sha256_file(parquet),
            },
        )
    return bundle


@dataclass
class TabularEncoder:
    feature_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    category_maps: dict[str, dict[str, int]] | None = None

    def fit(self, bundle: FeatureBundle, indices: Sequence[int] | np.ndarray) -> TabularEncoder:
        self.feature_columns = bundle.feature_columns
        self.categorical_columns = bundle.categorical_columns
        part = bundle.frame.iloc[np.asarray(indices, dtype=np.int64)]
        self.category_maps = {}
        for column in self.categorical_columns:
            values = sorted(part[column].astype("string").fillna("<NA>").unique().tolist())
            self.category_maps[column] = {str(value): index for index, value in enumerate(values)}
        return self

    def transform(
        self,
        bundle: FeatureBundle,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        if self.category_maps is None:
            raise RuntimeError("fit must be called before transform")
        frame = (
            bundle.frame
            if indices is None
            else bundle.frame.iloc[np.asarray(indices, dtype=np.int64)]
        )
        columns: list[np.ndarray] = []
        for column in self.feature_columns:
            if column in self.categorical_columns:
                mapping = self.category_maps[column]
                encoded = (
                    frame[column]
                    .astype("string")
                    .fillna("<NA>")
                    .map(mapping)
                    .fillna(-1)
                    .to_numpy(dtype=np.float32)
                )
            else:
                encoded = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                    dtype=np.float32, copy=True
                )
                encoded[~np.isfinite(encoded)] = np.nan
            columns.append(encoded)
        return np.column_stack(columns).astype(np.float32, copy=False)


@dataclass
class SavedTabularModel:
    backend: str
    encoder: TabularEncoder
    model: DeterministicTabularClassifier
    postprocess: dict[str, Any]
    feature_mode: str
    feature_hash: str
    iteration_count: int
    seed: int
    type_order: tuple[str, ...] = ANOMALY_TYPES


def _model_parameters(config: P1QCConfig, backend: str) -> dict[str, Any]:
    models = config.raw.get("models", {})
    if isinstance(models, Mapping):
        values = models.get(backend, {})
        if isinstance(values, Mapping):
            return dict(values)
    return {}


def _threads(config: P1QCConfig) -> int:
    project = config.raw.get("project", {})
    if isinstance(project, Mapping):
        return int(project.get("threads", 8))
    return 8


def _sample_weights(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.int8)
    positive = max(1, int(target.sum()))
    negative = max(1, len(target) - positive)
    positive_weight = float(np.sqrt(negative / positive))
    return np.where(target == 1, positive_weight, 1.0).astype(np.float32)


def _augmented_fit_data(
    full_frame: pd.DataFrame,
    original_bundle: FeatureBundle,
    indices: np.ndarray,
    config: P1QCConfig,
    *,
    seed: int,
    enabled: bool,
) -> tuple[TabularEncoder, np.ndarray, np.ndarray, dict[str, Any]]:
    if not enabled:
        encoder = TabularEncoder().fit(original_bundle, indices)
        return (
            encoder,
            encoder.transform(original_bundle, indices),
            full_frame.iloc[indices]["label"].to_numpy(dtype=np.int8),
            {"enabled": False, "injected_rows": 0, "events": 0},
        )
    fold_frame = full_frame.iloc[indices].copy()
    augmented = augment_training_fold(
        fold_frame,
        AugmentConfig(target_fraction=0.04, overlap_fraction=0.15, seed=seed),
    )
    modified = full_frame.copy()
    for column in ("temp", "label", "anomaly_type"):
        column_position = modified.columns.get_loc(column)
        modified.iloc[indices, column_position] = augmented.frame[column].to_numpy()
    augmented_bundle = build_features(
        modified,
        config=config,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    encoder = TabularEncoder().fit(augmented_bundle, indices)
    audit = {
        "enabled": True,
        "injected_rows": int(augmented.injected_mask.sum()),
        "injected_fraction_of_fold": float(augmented.injected_mask.mean()),
        "events": len(augmented.events),
        "event_types": augmented.events["anomaly_type"].value_counts().to_dict(),
        "overlap_events": int(augmented.events["is_overlap"].sum()),
        "original_positive_rows": int(fold_frame["label"].sum()),
        "result_positive_rows": int(augmented.frame["label"].sum()),
    }
    return (
        encoder,
        encoder.transform(augmented_bundle, indices),
        augmented.frame["label"].to_numpy(dtype=np.int8),
        audit,
    )


def _fit_model(
    backend: str,
    parameters: Mapping[str, Any],
    seed: int,
    threads: int,
    features: np.ndarray,
    target: np.ndarray,
    *,
    evaluation: tuple[np.ndarray, np.ndarray] | None = None,
) -> DeterministicTabularClassifier:
    model = make_tabular_classifier(backend, seed=seed, n_jobs=threads, parameters=dict(parameters))
    fit_parameters: dict[str, Any] = {}
    evaluation_set = None
    if evaluation is not None:
        evaluation_set = [evaluation]
        if backend == "lightgbm":
            import lightgbm as lgb

            fit_parameters["callbacks"] = [
                lgb.early_stopping(80, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        elif backend == "catboost":
            fit_parameters.update({"early_stopping_rounds": 80, "use_best_model": True})
    model.fit(
        features,
        target,
        sample_weight=_sample_weights(target),
        eval_set=evaluation_set,
        **fit_parameters,
    )
    return model


def _best_iteration(model: DeterministicTabularClassifier, fallback: int) -> int:
    backend = model.model
    for attribute in ("best_iteration_", "best_iteration"):
        value = getattr(backend, attribute, None)
        if value is not None and int(value) >= 0:
            return max(1, int(value) + (1 if attribute == "best_iteration" else 0))
    getter = getattr(backend, "get_best_iteration", None)
    if callable(getter):
        value = int(getter())
        if value >= 0:
            return value + 1
    return fallback


def _iteration_parameter(backend: str) -> str:
    return "iterations" if backend == "catboost" else "n_estimators"


def _postprocess_grid(config: P1QCConfig) -> tuple[list[float], list[float], list[int], list[int]]:
    section = config.raw.get("postprocess", {})
    if not isinstance(section, Mapping):
        section = {}
    return (
        [float(value) for value in section.get("threshold_grid", np.linspace(0.15, 0.7, 12))],
        [float(value) for value in section.get("low_ratio_grid", (0.5, 0.7, 0.85))],
        [int(value) for value in section.get("gap_grid", (0, 1, 2, 3, 6))],
        [int(value) for value in section.get("continuous_min_grid", (1, 3, 6, 12))],
    )


def apply_postprocess(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    plateau: np.ndarray,
    spike_candidates: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    config = PostprocessConfig(
        high_threshold=float(parameters["high_threshold"]),
        low_threshold=float(parameters["low_threshold"]),
        close_gap_rows=int(parameters["close_gap_rows"]),
        minimum_positive_run=int(parameters["minimum_positive_run"]),
    )
    positions, breaks = _chronological_order(
        frame,
        group_columns=config.group_columns,
        time_column=config.time_column,
        expected_interval=config.expected_interval,
    )
    ordered_probability = np.asarray(probabilities, dtype=float)[positions]
    ordered_plateau = np.asarray(plateau, dtype=bool)[positions]
    ordered_spike = np.asarray(spike_candidates, dtype=bool)[positions]
    preserve = ordered_spike & (ordered_probability >= config.high_threshold)
    label = hysteresis_threshold(
        ordered_probability,
        high_threshold=config.high_threshold,
        low_threshold=config.low_threshold,
        breaks=breaks,
    )
    label |= ordered_plateau | preserve
    label = close_short_gaps(label, max_gap_rows=config.close_gap_rows, breaks=breaks)
    label = remove_short_runs(
        label,
        minimum_run=config.minimum_positive_run,
        preserve=preserve,
        breaks=breaks,
    )
    result = np.zeros(len(frame), dtype=np.int8)
    result[positions] = label.astype(np.int8)
    return result


def tune_postprocess(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    truth: np.ndarray,
    plateau: np.ndarray,
    spike_candidates: np.ndarray,
    config: P1QCConfig,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    thresholds, low_ratios, gaps, minimum_runs = _postprocess_grid(config)
    simple_scores = []
    for threshold in thresholds:
        prediction = (np.asarray(probabilities) >= threshold) | np.asarray(plateau, dtype=bool)
        simple_scores.append((micro_f1(truth, prediction), threshold))
    top_thresholds = [value for _, value in sorted(simple_scores, reverse=True)[:3]]
    best_score = -1.0
    best_parameters: dict[str, Any] | None = None
    best_prediction: np.ndarray | None = None
    for high in top_thresholds:
        for ratio in low_ratios:
            low = min(high, high * ratio)
            for gap in gaps:
                for minimum in minimum_runs:
                    parameters = {
                        "high_threshold": high,
                        "low_threshold": low,
                        "close_gap_rows": gap,
                        "minimum_positive_run": minimum,
                    }
                    prediction = apply_postprocess(
                        frame, probabilities, plateau, spike_candidates, parameters
                    )
                    score = micro_f1(truth, prediction)
                    if score > best_score + 1.0e-12:
                        best_score = score
                        best_parameters = parameters
                        best_prediction = prediction
    if best_parameters is None or best_prediction is None:
        raise RuntimeError("postprocess search produced no configuration")
    diagnostics = {
        "selected_f1": best_score,
        "plateau": asdict(evaluate_binary_rule(plateau, truth)),
        "spike_standalone": asdict(evaluate_binary_rule(spike_candidates, truth)),
        "searched_high_thresholds": top_thresholds,
        "grid_candidates": len(top_thresholds) * len(low_ratios) * len(gaps) * len(minimum_runs),
    }
    return best_parameters, best_prediction, diagnostics


def _inner_calibration_indices(
    frame: pd.DataFrame,
    fold: Fold,
    *,
    calibration_days: int,
    purge_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    time = pd.to_datetime(frame["time"], errors="raise", utc=True)
    calibration_start = fold.train_end - pd.Timedelta(days=calibration_days)
    fit_end = calibration_start - pd.Timedelta(days=purge_days)
    outer_train = np.zeros(len(frame), dtype=bool)
    outer_train[fold.train_idx] = True
    fit = np.flatnonzero(outer_train & time.le(fit_end).to_numpy())
    calibration = np.flatnonzero(
        outer_train & time.ge(calibration_start).to_numpy() & time.le(fold.train_end).to_numpy()
    )
    if len(fit) == 0 or len(calibration) == 0:
        raise ValueError(f"fold {fold.name} inner calibration is empty")
    if frame.iloc[fit]["label"].sum() == 0 or frame.iloc[calibration]["label"].sum() == 0:
        raise ValueError(f"fold {fold.name} inner calibration lacks positive labels")
    return fit, calibration


def run_cross_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bundle: FeatureBundle,
    config: P1QCConfig,
    *,
    backend: str = "lightgbm",
    bootstrap_replicates: int = 2000,
    augmentation: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    folds = outer_folds(
        train,
        config=config.splits,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    test_shares = group_row_shares(test)
    parameters = _model_parameters(config, backend)
    configured_iterations = int(parameters.get(_iteration_parameter(backend), 400))
    project_validation = config.raw.get("validation", {})
    calibration_days = (
        int(project_validation.get("calibration_days", 60))
        if isinstance(project_validation, Mapping)
        else 60
    )
    fold_reports: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    iteration_counts: list[int] = []

    for fold_number, fold in enumerate(folds):
        inner_fit, calibration = _inner_calibration_indices(
            train,
            fold,
            calibration_days=calibration_days,
            purge_days=config.splits.purge_days,
        )
        inner_encoder, inner_train_features, inner_target, inner_augmentation = _augmented_fit_data(
            train,
            bundle,
            inner_fit,
            config,
            seed=config.seed + fold_number * 10,
            enabled=augmentation,
        )
        calibration_features = inner_encoder.transform(bundle, calibration)
        calibration_target = train.iloc[calibration]["label"].to_numpy(dtype=np.int8)
        selection_model = _fit_model(
            backend,
            parameters,
            config.seed + fold_number,
            _threads(config),
            inner_train_features,
            inner_target,
            evaluation=(calibration_features, calibration_target),
        )
        best_iterations = _best_iteration(selection_model, configured_iterations)
        iteration_counts.append(best_iterations)
        calibration_probability = selection_model.predict_proba(calibration_features)[:, 1]
        calibration_frame = train.iloc[calibration].copy()
        calibration_plateau = detect_plateaus(calibration_frame).to_numpy()
        calibration_spike = detect_singleton_spikes(calibration_frame).to_numpy()
        selected_postprocess, _, inner_diagnostics = tune_postprocess(
            calibration_frame,
            calibration_probability,
            calibration_target,
            calibration_plateau,
            calibration_spike,
            config,
        )

        outer_parameters = dict(parameters)
        outer_parameters[_iteration_parameter(backend)] = best_iterations
        encoder, outer_train_features, outer_target, outer_augmentation = _augmented_fit_data(
            train,
            bundle,
            fold.train_idx,
            config,
            seed=config.seed + fold_number * 10 + 1,
            enabled=augmentation,
        )
        validation_features = encoder.transform(bundle, fold.val_idx)
        model = _fit_model(
            backend,
            outer_parameters,
            config.seed + fold_number,
            _threads(config),
            outer_train_features,
            outer_target,
        )
        probability = model.predict_proba(validation_features)[:, 1]
        validation_frame = train.iloc[fold.val_idx].copy()
        validation_target = validation_frame["label"].to_numpy(dtype=np.int8)
        plateau = detect_plateaus(validation_frame).to_numpy()
        spike = detect_singleton_spikes(validation_frame).to_numpy()
        prediction = apply_postprocess(
            validation_frame, probability, plateau, spike, selected_postprocess
        )
        baseline = plateau.astype(np.int8)
        report = evaluate_predictions(
            validation_target,
            prediction,
            validation_frame,
            group_weights=test_shares,
            anomaly_type=validation_frame["anomaly_type"],
        )
        baseline_report = evaluate_predictions(
            validation_target,
            baseline,
            validation_frame,
            group_weights=test_shares,
            anomaly_type=validation_frame["anomaly_type"],
        )
        fold_reports.append(
            {
                "fold": fold.name,
                "train_rows": len(fold.train_idx),
                "validation_rows": len(fold.val_idx),
                "best_iterations": best_iterations,
                "postprocess": selected_postprocess,
                "inner_diagnostics": inner_diagnostics,
                "inner_augmentation": inner_augmentation,
                "outer_augmentation": outer_augmentation,
                "candidate": report.to_dict(),
                "plateau_baseline": baseline_report.to_dict(),
                "weighted_f1_improvement": report.weighted.f1 - baseline_report.weighted.f1,
            }
        )
        keys = validation_frame.loc[:, ["station", "year", "layer", "time"]].copy()
        keys["label"] = validation_target
        keys["probability"] = probability.astype(np.float32)
        keys["prediction"] = prediction
        keys["plateau_baseline"] = baseline
        keys["plateau"] = plateau
        keys["spike_candidate"] = spike
        keys["anomaly_type"] = validation_frame["anomaly_type"].fillna("").to_numpy()
        keys["fold"] = fold.name
        oof_parts.append(keys)

    oof = pd.concat(oof_parts, ignore_index=True)
    deployment_postprocess, deployment_prediction, deployment_diagnostics = tune_postprocess(
        oof,
        oof["probability"].to_numpy(),
        oof["label"].to_numpy(),
        oof["plateau"].to_numpy(),
        oof["spike_candidate"].to_numpy(),
        config,
    )
    oof["deployment_prediction"] = deployment_prediction
    # Honest nested-CV aggregate: every row keeps the post-processing chosen
    # only from that fold's inner calibration block.  The deployment threshold
    # below is tuned on all OOF rows for final test inference, so its score is
    # explicitly reported as resubstitution and never used as an outer estimate.
    aggregate = evaluate_predictions(
        oof["label"].to_numpy(),
        oof["prediction"].to_numpy(),
        oof,
        group_weights=test_shares,
        anomaly_type=oof["anomaly_type"],
    )
    baseline_aggregate = evaluate_predictions(
        oof["label"].to_numpy(),
        oof["plateau_baseline"].to_numpy(),
        oof,
        group_weights=test_shares,
        anomaly_type=oof["anomaly_type"],
    )
    bootstrap = paired_block_bootstrap(
        oof["label"].to_numpy(),
        oof["prediction"].to_numpy(),
        oof["plateau_baseline"].to_numpy(),
        oof,
        replicates=bootstrap_replicates,
        seed=config.seed,
    )
    metrics = {
        "backend": backend,
        "mode": config.features.mode,
        "augmentation": augmentation,
        "folds": fold_reports,
        "aggregate": aggregate.to_dict(),
        "deployment_resubstitution": evaluate_predictions(
            oof["label"].to_numpy(),
            deployment_prediction,
            oof,
            group_weights=test_shares,
            anomaly_type=oof["anomaly_type"],
        ).to_dict(),
        "deployment_resubstitution_is_not_outer_estimate": True,
        "plateau_baseline": baseline_aggregate.to_dict(),
        "bootstrap_vs_plateau": bootstrap,
        "official_hidden_baseline_reference": 0.548255,
        "official_reference_is_not_directly_comparable": True,
    }
    selection = {
        "backend": backend,
        "feature_mode": config.features.mode,
        "augmentation": augmentation,
        "iteration_count": int(round(float(np.median(iteration_counts)))),
        "fold_iteration_counts": iteration_counts,
        "postprocess": deployment_postprocess,
        "deployment_diagnostics": deployment_diagnostics,
        "feature_hash": stable_hash(asdict(config.features)),
        "promotion_gate": {
            "all_folds_improve_plateau_by_0_01": all(
                report["weighted_f1_improvement"] >= 0.01 for report in fold_reports
            ),
            "bootstrap_ci90_lower_above_zero": bootstrap["difference_ci90"][0] > 0,
        },
    }
    return oof, metrics, selection


def train_full_model(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    config: P1QCConfig,
    selection: Mapping[str, Any],
) -> SavedTabularModel:
    backend = str(selection["backend"])
    encoder = TabularEncoder().fit(bundle, np.arange(len(train)))
    features = encoder.transform(bundle)
    target = train["label"].to_numpy(dtype=np.int8)
    parameters = _model_parameters(config, backend)
    iterations = int(selection["iteration_count"])
    parameters[_iteration_parameter(backend)] = iterations
    model = _fit_model(backend, parameters, config.seed, _threads(config), features, target)
    return SavedTabularModel(
        backend=backend,
        encoder=encoder,
        model=model,
        postprocess=dict(selection["postprocess"]),
        feature_mode=str(selection["feature_mode"]),
        feature_hash=str(selection["feature_hash"]),
        iteration_count=iterations,
        seed=config.seed,
    )


def save_model(model: SavedTabularModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load_model(path: str | Path) -> SavedTabularModel:
    value = joblib.load(path)
    if not isinstance(value, SavedTabularModel):
        raise TypeError("model artifact is not a SavedTabularModel")
    return value


def predict_submission(
    model: SavedTabularModel,
    test: pd.DataFrame,
    bundle: FeatureBundle,
) -> tuple[pd.DataFrame, np.ndarray]:
    if bundle.frame.attrs.get("feature_mode", model.feature_mode) != model.feature_mode:
        raise ValueError("model and inference feature modes differ")
    features = model.encoder.transform(bundle)
    probability = model.model.predict_proba(features)[:, 1]
    plateau = detect_plateaus(test).to_numpy()
    spike = detect_singleton_spikes(test).to_numpy()
    label = apply_postprocess(test, probability, plateau, spike, model.postprocess)
    anomaly_type = np.full(len(test), "", dtype=object)
    anomaly_type[(label == 1) & plateau] = "flatline"
    confirmed_spike = (label == 1) & spike & (probability >= model.postprocess["high_threshold"])
    anomaly_type[confirmed_spike & ~plateau] = "spike"
    submission = build_submission(test, label, anomaly_type)
    return submission, probability


def reproduce_submission(
    model_path: str | Path,
    test: pd.DataFrame,
    bundle: FeatureBundle,
    output_path: str | Path,
    *,
    expected_path: str | Path | None = None,
) -> dict[str, Any]:
    model = load_model(model_path)
    submission, _ = predict_submission(model, test, bundle)
    output = write_submission(submission, output_path)
    report = validate_submission(output, test)
    if expected_path is not None:
        expected = pd.read_csv(expected_path, keep_default_na=False)
        actual = pd.read_csv(output, keep_default_na=False)
        report["row_identical_to_expected"] = actual.equals(expected)
        report["expected_sha256"] = sha256_file(expected_path)
        if not report["row_identical_to_expected"]:
            raise RuntimeError("reproduced submission differs from expected rows")
    return report
