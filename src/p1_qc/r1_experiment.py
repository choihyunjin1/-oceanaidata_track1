"""Honest nested-CV orchestration for the R1 boundary-completion experiment."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .change_points import (
    ChangePointConfig,
    proposals_to_mask,
    propose_change_intervals,
)
from .config import P1QCConfig
from .data import segment_timeseries
from .features import FeatureBundle
from .metrics import evaluate_predictions, group_row_shares
from .pipeline import (
    _augmented_fit_data,
    _best_iteration,
    _fit_model,
    _inner_calibration_indices,
    _iteration_parameter,
    _model_parameters,
    _threads,
    apply_postprocess,
    tune_postprocess,
)
from .r1_validation import (
    CandidateOutput,
    GridSelectionResult,
    SelectionProvenance,
    apply_selected_interval_grid,
    candidate_output_to_mask,
    select_interval_grid,
)
from .rules import detect_plateaus, detect_singleton_spikes
from .splits import _positive_run_ids, outer_folds


class ProposalBuilder(Protocol):
    """Label-blind proposal callback used for both inner and outer rows."""

    def __call__(
        self,
        *,
        frame: pd.DataFrame,
        features: FeatureBundle,
        probabilities: np.ndarray,
        base_prediction: np.ndarray,
        parameters: Mapping[str, Any],
    ) -> CandidateOutput: ...


def change_point_proposal_builder(
    *,
    frame: pd.DataFrame,
    features: FeatureBundle,
    probabilities: np.ndarray,
    base_prediction: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    """Apply the preregistered label-blind CAPA/CPOP-lite proposal generator."""

    del base_prediction  # The frozen XGB score, not a label, defines seed runs.
    if parameters.get("enabled") is False:
        return np.zeros(len(frame), dtype=bool)
    required = {
        "signal",
        "max_flank_rows",
        "proposal_score_quantile",
        "top_proposals_per_seed",
        "high_seed_threshold",
        "low_seed_threshold",
    }
    missing = sorted(required.difference(parameters))
    if missing:
        raise KeyError(f"R1 proposal parameters are missing {missing}")
    signal_name = str(parameters["signal"])
    if signal_name not in features.frame:
        raise KeyError(f"R1 signal is absent from the feature bundle: {signal_name}")

    probability = np.asarray(probabilities, dtype=float).copy()
    if probability.shape != (len(frame),):
        raise ValueError("R1 probabilities must align one-for-one with frame rows")
    # Flatline and spike are already handled by protected rules.  They must not
    # become slow-anomaly expansion seeds.
    protected_seed = detect_plateaus(frame).to_numpy() | detect_singleton_spikes(frame).to_numpy()
    probability[protected_seed] = 0.0
    segmented = segment_timeseries(frame, cadence_minutes=10)
    result = propose_change_intervals(
        pd.to_numeric(features.frame[signal_name], errors="coerce").to_numpy(dtype=float),
        probability,
        segmented["segment_id"].to_numpy(),
        station=frame["station"].to_numpy(),
        layer=frame["layer"].to_numpy(),
        row_ids=np.arange(len(frame), dtype=np.int64),
        times=frame["time"].to_numpy(),
        config=ChangePointConfig(
            mode="offline",
            high_seed_threshold=float(parameters["high_seed_threshold"]),
            low_seed_threshold=float(parameters["low_seed_threshold"]),
            max_flank_rows=int(parameters["max_flank_rows"]),
            min_interval_rows=6,
            max_interval_rows=600,
            min_baseline_rows=6,
            min_return_rows=3,
            mean_gain_threshold=0.5,
            variance_gain_threshold=0.25,
            slope_gain_threshold=0.25,
            baseline_z_threshold=3.0,
            return_z_threshold=3.0,
            max_candidates_per_seed_run=8,
        ),
    )
    scores = np.asarray([proposal.total_score for proposal in result.proposals], dtype=float)
    cutoff = (
        float(np.quantile(scores, float(parameters["proposal_score_quantile"])))
        if len(scores)
        else np.inf
    )
    return proposals_to_mask(
        result,
        len(frame),
        top_k_per_seed=int(parameters["top_proposals_per_seed"]),
        min_total_score=cutoff,
        require_return=True,
    )


@dataclass(frozen=True)
class InnerBoundaryAudit:
    fit_indices: np.ndarray
    calibration_indices: np.ndarray
    dropped_runs: int
    dropped_fit_rows: int
    dropped_calibration_rows: int
    crossed_boundaries: Mapping[str, int]


@dataclass(frozen=True)
class R1CVResult:
    oof: pd.DataFrame
    metrics: Mapping[str, Any]
    selection: Mapping[str, Any]


def preregistered_r1_grid() -> list[dict[str, Any]]:
    """Return the frozen 37-candidate R1 grid, including a no-op."""

    grid: list[dict[str, Any]] = [{"enabled": False}]
    for signal in ("reference_resid_7d", "reference_resid_14d"):
        for flank in (72, 144, 288):
            for quantile in (0.75, 0.9, 0.95):
                for top in (1, 2):
                    grid.append(
                        {
                            "enabled": True,
                            "signal": signal,
                            "max_flank_rows": flank,
                            "proposal_score_quantile": quantile,
                            "top_proposals_per_seed": top,
                        }
                    )
    if len(grid) != 37:
        raise RuntimeError("R1 preregistered grid must contain exactly 37 candidates")
    return grid


def _subset_bundle(bundle: FeatureBundle, indices: np.ndarray) -> FeatureBundle:
    frame = bundle.frame.iloc[indices].reset_index(drop=True).copy()
    frame.attrs = dict(bundle.frame.attrs)
    return FeatureBundle(frame, bundle.feature_columns, bundle.categorical_columns)


def _safe_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["label", "anomaly_type"], errors="ignore").reset_index(drop=True)


def audit_inner_event_boundaries(
    frame: pd.DataFrame,
    fit_indices: Sequence[int] | np.ndarray,
    calibration_indices: Sequence[int] | np.ndarray,
    *,
    scope_indices: Sequence[int] | np.ndarray | None = None,
    cadence_minutes: int = 10,
    group_columns: Sequence[str] = ("station", "layer"),
) -> InnerBoundaryAudit:
    """Drop positive runs touching an inner boundary inside an explicit scope.

    ``scope_indices`` is normally the outer-train population.  Labels outside
    that population are never inspected.  For backward API compatibility,
    omitting it scopes the audit to the whole frame.  A positive run that
    reaches the calibration end is conservatively dropped without consulting
    any later row.
    """

    def positions(
        values: Sequence[int] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if array.dtype.kind == "b":
            raise TypeError(f"{name} must contain positional integers, not booleans")
        try:
            numeric = array.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain positional integers") from exc
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{name} must contain finite positional integers")
        result = numeric.astype(np.int64)
        if ((result < 0) | (result >= len(frame))).any():
            raise IndexError(f"{name} contains a row outside the frame")
        if len(np.unique(result)) != len(result):
            raise ValueError(f"{name} contains duplicate rows")
        return result

    fit = positions(fit_indices, name="fit_indices")
    calibration = positions(calibration_indices, name="calibration_indices")
    scope = (
        np.arange(len(frame), dtype=np.int64)
        if scope_indices is None
        else positions(scope_indices, name="scope_indices")
    )
    if not len(fit) or not len(calibration) or not len(scope):
        raise ValueError("fit, calibration, and scope indices must be non-empty")
    if np.intersect1d(fit, calibration).size:
        raise ValueError("fit_indices and calibration_indices must be disjoint")
    if not np.isin(fit, scope).all() or not np.isin(calibration, scope).all():
        raise ValueError("fit and calibration indices must be contained in scope_indices")
    time = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    scoped_frame = frame.iloc[scope].reset_index(drop=True)
    scoped_time = time.iloc[scope].reset_index(drop=True)
    scoped_run_ids = _positive_run_ids(
        scoped_frame,
        scoped_time,
        cadence_minutes=cadence_minutes,
        group_columns=group_columns,
        label_column="label",
    )
    run_ids = np.full(len(frame), -1, dtype=np.int64)
    run_ids[scope] = scoped_run_ids
    fit_end = time.iloc[fit].max()
    calibration_start = time.iloc[calibration].min()
    calibration_end = time.iloc[calibration].max()
    boundaries = {
        "fit_end": (lambda start, stop: start <= fit_end < stop),
        "calibration_start": (lambda start, stop: start < calibration_start <= stop),
        # Inclusive on the right: a run reaching the last calibration instant
        # is removed without looking outside the outer-train scope to see
        # whether it continues.
        "calibration_end": (lambda start, stop: start <= calibration_end <= stop),
    }
    crossing_runs: set[int] = set()
    counts = {name: 0 for name in boundaries}
    positive_positions = np.flatnonzero(run_ids >= 0)
    if len(positive_positions):
        events = (
            pd.DataFrame(
                {
                    "run": run_ids[positive_positions],
                    "time": time.iloc[positive_positions].to_numpy(),
                }
            )
            .groupby("run", sort=False)["time"]
            .agg(["min", "max"])
        )
        for run_id, event in events.iterrows():
            for name, crosses in boundaries.items():
                if crosses(event["min"], event["max"]):
                    crossing_runs.add(int(run_id))
                    counts[name] += 1
    drop = np.isin(run_ids, np.asarray(sorted(crossing_runs), dtype=np.int64))
    clean_fit = fit[~drop[fit]]
    clean_calibration = calibration[~drop[calibration]]
    if not len(clean_fit) or not len(clean_calibration):
        raise ValueError("event-boundary filtering emptied an inner split")
    return InnerBoundaryAudit(
        fit_indices=clean_fit,
        calibration_indices=clean_calibration,
        dropped_runs=len(crossing_runs),
        dropped_fit_rows=int(drop[fit].sum()),
        dropped_calibration_rows=int(drop[calibration].sum()),
        crossed_boundaries=counts,
    )


def _candidate_factory(
    builder: ProposalBuilder,
    frame: pd.DataFrame,
    features: FeatureBundle,
    probabilities: np.ndarray,
    base_prediction: np.ndarray,
) -> Callable[[Mapping[str, Any]], CandidateOutput]:
    forbidden = {"label", "anomaly_type"}
    exposed = forbidden.intersection(
        set(features.frame.columns)
        | set(features.feature_columns)
        | set(features.categorical_columns)
    )
    if exposed:
        raise ValueError(f"R1 FeatureBundle exposes forbidden target columns: {sorted(exposed)}")
    if len(features.frame) != len(frame):
        raise ValueError("R1 FeatureBundle must align one-for-one with frame rows")
    safe_frame = _safe_metadata(frame)

    def factory(parameters: Mapping[str, Any]) -> CandidateOutput:
        if parameters.get("enabled") is False:
            return np.zeros(len(safe_frame), dtype=bool)
        return builder(
            frame=safe_frame.copy(deep=False),
            features=features,
            probabilities=np.asarray(probabilities, dtype=float).copy(),
            base_prediction=np.asarray(base_prediction, dtype=np.int8).copy(),
            parameters=dict(parameters),
        )

    return factory


def consensus_boundary_parameters(
    selections: Sequence[GridSelectionResult | Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose a deterministic modal whole configuration without labels."""

    canonical: list[str] = []
    decoded: dict[str, dict[str, Any]] = {}
    for selection in selections:
        parameters = (
            selection.parameters if isinstance(selection, GridSelectionResult) else selection
        )
        value = json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"))
        canonical.append(value)
        decoded[value] = dict(parameters)
    if not canonical:
        raise ValueError("at least one fold selection is required")
    counts = Counter(canonical)
    winner = min(value for value, count in counts.items() if count == max(counts.values()))
    return decoded[winner]


def run_r1_nested_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bundle: FeatureBundle,
    config: P1QCConfig,
    proposal_builder: ProposalBuilder,
    *,
    parameter_grid: Sequence[Mapping[str, Any]] | None = None,
    backend: str = "xgboost",
    augmentation: bool = False,
    primary_metric: str = "micro_f1",
    fit_model_fn: Callable[..., Any] | None = None,
) -> R1CVResult:
    """Run nested R1 selection without using outer labels for model or boundary tuning.

    The shared outer-fold contract is label-aware only to keep each positive
    event wholly inside one evaluation fold.  Once that evaluation membership
    is fixed, outer labels do not enter model fitting, postprocess tuning, or
    boundary selection.
    """

    grid = list(preregistered_r1_grid() if parameter_grid is None else parameter_grid)
    if not any(candidate.get("enabled") is False for candidate in grid):
        raise ValueError("R1 parameter grid must include the no-op candidate")
    fit_model = _fit_model if fit_model_fn is None else fit_model_fn
    folds = outer_folds(
        train,
        config=config.splits,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    test_shares = group_row_shares(test)
    model_parameters = _model_parameters(config, backend)
    configured_iterations = int(model_parameters.get(_iteration_parameter(backend), 400))
    validation = config.raw.get("validation", {})
    calibration_days = (
        int(validation.get("calibration_days", 60)) if isinstance(validation, Mapping) else 60
    )
    fold_reports: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    fold_selections: list[GridSelectionResult] = []
    iteration_counts: list[int] = []

    for fold_number, fold in enumerate(folds):
        raw_fit, raw_calibration = _inner_calibration_indices(
            train,
            fold,
            calibration_days=calibration_days,
            purge_days=config.splits.purge_days,
        )
        boundary_audit = audit_inner_event_boundaries(
            train,
            raw_fit,
            raw_calibration,
            scope_indices=fold.train_idx,
            cadence_minutes=config.data.cadence_minutes,
            group_columns=config.data.group_columns,
        )
        # Preserve the frozen XGB baseline exactly.  The model and its
        # postprocess use the original inner split; boundary grid selection
        # alone excludes positive events crossing an inner boundary.
        inner_fit = raw_fit
        calibration = raw_calibration
        inner_encoder, inner_x, inner_y, inner_augmentation = _augmented_fit_data(
            train,
            bundle,
            inner_fit,
            config,
            seed=config.seed + fold_number * 10,
            enabled=augmentation,
        )
        calibration_x = inner_encoder.transform(bundle, calibration)
        calibration_y = train.iloc[calibration]["label"].to_numpy(dtype=np.int8)
        selection_model = fit_model(
            backend,
            model_parameters,
            config.seed + fold_number,
            _threads(config),
            inner_x,
            inner_y,
            evaluation=(calibration_x, calibration_y),
        )
        best_iterations = _best_iteration(selection_model, configured_iterations)
        iteration_counts.append(best_iterations)
        calibration_probability = selection_model.predict_proba(calibration_x)[:, 1]
        calibration_frame = train.iloc[calibration].reset_index(drop=True).copy()
        calibration_plateau = detect_plateaus(calibration_frame).to_numpy()
        calibration_spike = detect_singleton_spikes(calibration_frame).to_numpy()
        postprocess, calibration_base, postprocess_diagnostics = tune_postprocess(
            calibration_frame,
            calibration_probability,
            calibration_y,
            calibration_plateau,
            calibration_spike,
            config,
        )
        selection_keep = np.isin(calibration, boundary_audit.calibration_indices)
        selection_indices = calibration[selection_keep]
        selection_frame = train.iloc[selection_indices].reset_index(drop=True).copy()
        selection_bundle = _subset_bundle(bundle, selection_indices)
        selection_probability = calibration_probability[selection_keep]
        selection_base = calibration_base[selection_keep]
        selection_target = calibration_y[selection_keep]
        selection_plateau = calibration_plateau[selection_keep]
        selection_spike = calibration_spike[selection_keep]
        fold_grid = [
            (
                dict(candidate)
                if candidate.get("enabled") is False
                else {
                    **dict(candidate),
                    "high_seed_threshold": float(postprocess["high_threshold"]),
                    "low_seed_threshold": float(postprocess["low_threshold"]),
                }
            )
            for candidate in grid
        ]
        calibration_factory = _candidate_factory(
            proposal_builder,
            selection_frame,
            selection_bundle,
            selection_probability,
            selection_base,
        )
        boundary_selection = select_interval_grid(
            _safe_metadata(selection_frame),
            selection_target,
            selection_base,
            fold_grid,
            calibration_factory,
            provenance=SelectionProvenance(
                fit_rows=raw_fit,
                inner_validation_rows=selection_indices,
                outer_validation_rows=fold.val_idx,
                label_scope="inner_validation",
                generator_id="r1_boundary_completion_v1",
            ),
            group_weights=test_shares,
            # The union operation cannot remove any base-positive row.  Only
            # already-positive rule rows are "protected"; raw low-precision
            # spike candidates must never be promoted by a no-op boundary.
            spike_protected=selection_spike & selection_base.astype(bool),
            plateau_protected=selection_plateau & selection_base.astype(bool),
            primary_metric=primary_metric,
            group_columns=config.data.group_columns,
            cadence_minutes=config.data.cadence_minutes,
        )
        fold_selections.append(boundary_selection)

        outer_parameters = dict(model_parameters)
        outer_parameters[_iteration_parameter(backend)] = best_iterations
        encoder, outer_x, outer_y, outer_augmentation = _augmented_fit_data(
            train,
            bundle,
            fold.train_idx,
            config,
            seed=config.seed + fold_number * 10 + 1,
            enabled=augmentation,
        )
        validation_x = encoder.transform(bundle, fold.val_idx)
        model = fit_model(
            backend,
            outer_parameters,
            config.seed + fold_number,
            _threads(config),
            outer_x,
            outer_y,
        )
        probability = model.predict_proba(validation_x)[:, 1]
        validation_frame = train.iloc[fold.val_idx].reset_index(drop=True).copy()
        validation_bundle = _subset_bundle(bundle, fold.val_idx)
        plateau = detect_plateaus(validation_frame).to_numpy()
        spike = detect_singleton_spikes(validation_frame).to_numpy()
        base_prediction = apply_postprocess(
            validation_frame, probability, plateau, spike, postprocess
        )
        outer_factory = _candidate_factory(
            proposal_builder,
            validation_frame,
            validation_bundle,
            probability,
            base_prediction,
        )
        prediction = apply_selected_interval_grid(
            _safe_metadata(validation_frame),
            base_prediction,
            boundary_selection,
            outer_factory,
            spike_protected=spike & base_prediction.astype(bool),
            plateau_protected=plateau & base_prediction.astype(bool),
            group_columns=config.data.group_columns,
            cadence_minutes=config.data.cadence_minutes,
        )
        proposal = candidate_output_to_mask(
            _safe_metadata(validation_frame),
            outer_factory(boundary_selection.parameters),
            group_columns=config.data.group_columns,
            cadence_minutes=config.data.cadence_minutes,
        )
        expected = np.asarray(base_prediction, dtype=bool) | proposal
        if not np.array_equal(prediction.astype(bool), expected):
            raise RuntimeError("R1 outer prediction violates proposal-union invariant")
        if boundary_selection.parameters.get("enabled") is False and not np.array_equal(
            prediction, base_prediction
        ):
            raise RuntimeError("R1 no-op boundary must exactly reproduce the frozen baseline")

        # After the label-aware event-protected fold membership was fixed,
        # this is the first point where its outer truth values are consumed.
        # They are used for evaluation only, never for model/boundary selection.
        outer_truth = validation_frame["label"].to_numpy(dtype=np.int8)
        candidate_report = evaluate_predictions(
            outer_truth,
            prediction,
            validation_frame,
            group_weights=test_shares,
            anomaly_type=validation_frame["anomaly_type"],
        )
        base_report = evaluate_predictions(
            outer_truth,
            base_prediction,
            validation_frame,
            group_weights=test_shares,
            anomaly_type=validation_frame["anomaly_type"],
        )
        fold_reports.append(
            {
                "fold": fold.name,
                "best_iterations": best_iterations,
                "postprocess": postprocess,
                "postprocess_diagnostics": postprocess_diagnostics,
                "boundary_parameters": dict(boundary_selection.parameters),
                "boundary_diagnostics": dict(boundary_selection.diagnostics),
                "inner_boundary_audit": {
                    "scope_rows": len(fold.train_idx),
                    "dropped_runs": boundary_audit.dropped_runs,
                    "dropped_fit_rows": boundary_audit.dropped_fit_rows,
                    "dropped_calibration_rows": boundary_audit.dropped_calibration_rows,
                    "crossed_boundaries": dict(boundary_audit.crossed_boundaries),
                },
                "inner_augmentation": inner_augmentation,
                "outer_augmentation": outer_augmentation,
                "candidate": candidate_report.to_dict(),
                "base": base_report.to_dict(),
            }
        )
        keys = validation_frame.loc[:, ["station", "year", "layer", "time"]].copy()
        keys["label"] = outer_truth
        keys["probability"] = probability.astype(np.float32)
        keys["base_prediction"] = np.asarray(base_prediction, dtype=np.int8)
        keys["proposal"] = proposal
        keys["prediction"] = prediction
        keys["plateau"] = plateau
        keys["spike_candidate"] = spike
        keys["anomaly_type"] = validation_frame["anomaly_type"].fillna("")
        keys["fold"] = fold.name
        oof_parts.append(keys)

    oof = pd.concat(oof_parts, ignore_index=True)
    aggregate = evaluate_predictions(
        oof["label"].to_numpy(),
        oof["prediction"].to_numpy(),
        oof,
        group_weights=test_shares,
        anomaly_type=oof["anomaly_type"],
    )
    base_aggregate = evaluate_predictions(
        oof["label"].to_numpy(),
        oof["base_prediction"].to_numpy(),
        oof,
        group_weights=test_shares,
        anomaly_type=oof["anomaly_type"],
    )
    consensus = consensus_boundary_parameters(fold_selections)
    metrics = {
        "backend": backend,
        "mode": config.features.mode,
        "folds": fold_reports,
        "aggregate": aggregate.to_dict(),
        "base_aggregate": base_aggregate.to_dict(),
        "outer_labels_used_for_selection": False,
    }
    selection = {
        "backend": backend,
        "feature_mode": config.features.mode,
        "iteration_count": int(round(float(np.median(iteration_counts)))),
        "fold_iteration_counts": iteration_counts,
        "boundary": consensus,
        "fold_boundary_parameters": [dict(item.parameters) for item in fold_selections],
        "boundary_consensus_uses_outer_labels": False,
    }
    return R1CVResult(oof=oof, metrics=metrics, selection=selection)


__all__ = [
    "InnerBoundaryAudit",
    "ProposalBuilder",
    "R1CVResult",
    "audit_inner_event_boundaries",
    "change_point_proposal_builder",
    "consensus_boundary_parameters",
    "preregistered_r1_grid",
    "run_r1_nested_cv",
]
