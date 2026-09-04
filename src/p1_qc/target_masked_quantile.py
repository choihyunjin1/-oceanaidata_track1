"""Target-masked conditional quantile features for P1.

The conditional model is deliberately denied the target row's temperature and
every feature derived from that value.  It may use salinity, contemporaneous
*other-layer* observations, calendar/clock phase, and explicit missingness
masks.  Temperature is consulted only after prediction to form auditable
residual scores for the frozen tabular QC model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from .data import segment_timeseries

TARGET_MASKED_CATEGORICAL_COLUMNS = ("station", "layer_category")
TARGET_MASKED_NUMERIC_COLUMNS = (
    "psal_raw",
    "psal_missing",
    "peer_temp_mean",
    "peer_temp_std",
    "peer_temp_count",
    "peer_temp_missing",
    "peer_psal_mean",
    "peer_psal_std",
    "peer_psal_count",
    "peer_psal_missing",
    "day_sin",
    "day_cos",
    "hour_sin",
    "hour_cos",
    "m2_sin",
    "m2_cos",
)
TARGET_MASKED_FEATURE_COLUMNS = (
    *TARGET_MASKED_CATEGORICAL_COLUMNS,
    *TARGET_MASKED_NUMERIC_COLUMNS,
)

# These are the only target-temperature-derived values exposed to the outer
# XGBoost.  The quantile regressors never receive them as inputs.
QUANTILE_SCORE_COLUMNS = (
    "tmq_signed_residual",
    "tmq_interval_width",
    "tmq_lower_tail_distance",
    "tmq_upper_tail_distance",
    "tmq_outside_tail_distance",
    "tmq_signed_residual_mean_24h",
    "tmq_signed_residual_mean_72h",
    "tmq_outside_tail_mean_24h",
    "tmq_quantile_available",
)


@dataclass(frozen=True)
class QuantileModelConfig:
    """One fixed, non-searchable LightGBM quantile configuration."""

    alphas: tuple[float, float, float] = (0.05, 0.50, 0.95)
    n_estimators: int = 180
    learning_rate: float = 0.04
    num_leaves: int = 31
    min_child_samples: int = 200
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    crossfit_folds: int = 2
    threads: int = 8
    seed: int = 20260813

    def __post_init__(self) -> None:
        if tuple(sorted(self.alphas)) != self.alphas:
            raise ValueError("quantile alphas must be strictly ordered")
        if len(self.alphas) != 3 or not all(0.0 < value < 1.0 for value in self.alphas):
            raise ValueError("exactly three alphas in (0, 1) are required")
        if self.alphas[1] != 0.5:
            raise ValueError("the middle quantile must be 0.5")
        if self.n_estimators < 1 or self.learning_rate <= 0:
            raise ValueError("invalid LightGBM iteration configuration")
        if self.num_leaves < 2 or self.min_child_samples < 1:
            raise ValueError("invalid LightGBM tree configuration")
        if self.crossfit_folds < 2:
            raise ValueError("crossfit_folds must be at least two")
        if self.threads == 0 or self.threads < -1:
            raise ValueError("threads must be -1 or positive")


@dataclass(frozen=True)
class TargetMaskedDesign:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...] = TARGET_MASKED_FEATURE_COLUMNS
    categorical_columns: tuple[str, ...] = TARGET_MASKED_CATEGORICAL_COLUMNS

    def __post_init__(self) -> None:
        missing = sorted(set(self.feature_columns).difference(self.frame.columns))
        if missing:
            raise ValueError(f"target-masked design is missing columns: {missing}")
        assert_target_masked_contract(self)


@dataclass
class TargetMaskedEncoder:
    """Fold-local deterministic encoding for the two identity categories."""

    category_maps: dict[str, dict[str, int]] | None = None

    def fit(
        self,
        design: TargetMaskedDesign,
        indices: Sequence[int] | np.ndarray,
    ) -> TargetMaskedEncoder:
        positions = _positions(indices, len(design.frame), name="encoder fit indices")
        part = design.frame.iloc[positions]
        self.category_maps = {}
        for column in design.categorical_columns:
            values = sorted(part[column].astype("string").fillna("<NA>").unique().tolist())
            self.category_maps[column] = {str(value): code for code, value in enumerate(values)}
        return self

    def transform(
        self,
        design: TargetMaskedDesign,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        if self.category_maps is None:
            raise RuntimeError("fit must be called before transform")
        positions = _positions(indices, len(design.frame), name="encoder transform indices")
        part = design.frame.iloc[positions]
        columns: list[np.ndarray] = []
        for column in design.feature_columns:
            if column in design.categorical_columns:
                values = (
                    part[column]
                    .astype("string")
                    .fillna("<NA>")
                    .map(self.category_maps[column])
                    .fillna(-1)
                    .to_numpy(dtype=np.float32)
                )
            else:
                values = pd.to_numeric(part[column], errors="coerce").to_numpy(
                    dtype=np.float32, copy=True
                )
                values[~np.isfinite(values)] = np.nan
            columns.append(values)
        return np.column_stack(columns).astype(np.float32, copy=False)


def _positions(values: Sequence[int] | np.ndarray, length: int, *, name: str) -> np.ndarray:
    positions = np.asarray(values)
    if positions.ndim != 1 or positions.dtype.kind == "b":
        raise ValueError(f"{name} must be one-dimensional integer positions")
    try:
        numeric = positions.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer positions") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer positions")
    result = numeric.astype(np.int64)
    if (result < 0).any() or (result >= length).any():
        raise IndexError(f"{name} are outside a frame of length {length}")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contain duplicate positions")
    return result


def _leave_one_out_moments(
    values: pd.Series,
    groupers: Sequence[pd.Series],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return other-row mean/std/count for each contemporaneous group."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric.notna()
    filled = numeric.fillna(0.0)
    grouped_sum = filled.groupby(list(groupers), sort=False, observed=True).transform("sum")
    grouped_square = (
        (filled * filled).groupby(list(groupers), sort=False, observed=True).transform("sum")
    )
    grouped_count = (
        finite.astype(np.int64).groupby(list(groupers), sort=False, observed=True).transform("sum")
    )
    own = filled.where(finite, 0.0)
    peer_sum = grouped_sum - own
    peer_square = grouped_square - own * own
    peer_count = grouped_count - finite.astype(np.int64)
    mean = (peer_sum / peer_count.where(peer_count > 0)).astype(float)
    numerator = peer_square - peer_sum * peer_sum / peer_count.where(peer_count > 0)
    variance = numerator / (peer_count - 1).where(peer_count > 1)
    variance = variance.clip(lower=0.0)
    std = np.sqrt(variance)
    return mean, std, peer_count.astype(float)


def build_target_masked_design(frame: pd.DataFrame) -> TargetMaskedDesign:
    """Build conditional-model inputs without using the row's own temperature.

    ``peer_temp_*`` is computed from other layers at exactly the same station
    and timestamp.  A one-layer station therefore receives count zero, NaN
    moments, and an explicit missing flag.  Depth is intentionally absent, so
    the G-ORS 2026 all-missing-depth deployment has the same input contract.
    """

    required = {"station", "layer", "time", "temp", "psal"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing target-masked source columns: {missing}")
    parsed = pd.to_datetime(frame["time"], errors="coerce", utc=True, format="mixed")
    if parsed.isna().any():
        raise ValueError("target-masked timestamps must all be parseable")
    station = frame["station"].astype("string")
    time_key = frame["time"].astype("string")
    groupers = (station, time_key)
    peer_temp_mean, peer_temp_std, peer_temp_count = _leave_one_out_moments(frame["temp"], groupers)
    peer_psal_mean, peer_psal_std, peer_psal_count = _leave_one_out_moments(frame["psal"], groupers)

    local = parsed.dt.tz_convert("Asia/Seoul")
    day_phase = 2.0 * np.pi * (local.dt.dayofyear.to_numpy(dtype=float) - 1.0) / 365.2425
    hour = local.dt.hour.to_numpy(dtype=float) + local.dt.minute.to_numpy(dtype=float) / 60.0
    hour_phase = 2.0 * np.pi * hour / 24.0
    # pandas 3 may preserve a microsecond datetime resolution, so explicitly
    # convert units instead of assuming integer timestamps are nanoseconds.
    epoch_seconds = parsed.dt.as_unit("s").astype("int64").to_numpy(dtype=np.float64)
    m2_period_seconds = 12.42 * 60.0 * 60.0
    m2_phase = 2.0 * np.pi * np.remainder(epoch_seconds, m2_period_seconds) / m2_period_seconds

    psal = pd.to_numeric(frame["psal"], errors="coerce")
    out = pd.DataFrame(index=frame.index)
    out["station"] = station
    out["layer_category"] = frame["layer"].astype("string")
    out["psal_raw"] = psal
    out["psal_missing"] = psal.isna().astype(np.int8)
    out["peer_temp_mean"] = peer_temp_mean
    out["peer_temp_std"] = peer_temp_std
    out["peer_temp_count"] = peer_temp_count
    out["peer_temp_missing"] = peer_temp_count.eq(0).astype(np.int8)
    out["peer_psal_mean"] = peer_psal_mean
    out["peer_psal_std"] = peer_psal_std
    out["peer_psal_count"] = peer_psal_count
    out["peer_psal_missing"] = peer_psal_count.eq(0).astype(np.int8)
    out["day_sin"] = np.sin(day_phase)
    out["day_cos"] = np.cos(day_phase)
    out["hour_sin"] = np.sin(hour_phase)
    out["hour_cos"] = np.cos(hour_phase)
    out["m2_sin"] = np.sin(m2_phase)
    out["m2_cos"] = np.cos(m2_phase)
    return TargetMaskedDesign(out.loc[:, TARGET_MASKED_FEATURE_COLUMNS].copy())


def assert_target_masked_contract(design: TargetMaskedDesign) -> None:
    """Fail closed if a forbidden own-temperature or depth input appears."""

    columns = tuple(str(column) for column in design.feature_columns)
    if columns != TARGET_MASKED_FEATURE_COLUMNS:
        raise ValueError("target-masked feature columns differ from the frozen contract")
    forbidden_exact = {
        "temp",
        "temp_raw",
        "depth",
        "depth_raw",
        "nominal_depth_m",
        "depth_regime",
        "year",
    }
    forbidden = sorted(forbidden_exact.intersection(columns))
    if forbidden:
        raise ValueError(f"forbidden conditional-model inputs: {forbidden}")
    for column in columns:
        lowered = column.lower()
        if "temp" in lowered and not lowered.startswith("peer_temp_"):
            raise ValueError(f"own-temperature-derived input is forbidden: {column}")
        if "depth" in lowered:
            raise ValueError(f"depth input is forbidden by the G fallback contract: {column}")


def _quantile_parameters(config: QuantileModelConfig, alpha: float, seed: int) -> dict[str, Any]:
    return {
        "objective": "quantile",
        "alpha": float(alpha),
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "random_state": seed,
        "n_jobs": config.threads,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }


def fit_predict_quantiles(
    design: TargetMaskedDesign,
    target_temp: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    fit_indices: Sequence[int] | np.ndarray,
    predict_indices: Sequence[int] | np.ndarray,
    *,
    config: QuantileModelConfig | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit on fold-local normal rows and predict three conditional quantiles."""

    config = config or QuantileModelConfig()
    fit_positions = _positions(fit_indices, len(design.frame), name="quantile fit indices")
    predict_positions = _positions(
        predict_indices, len(design.frame), name="quantile predict indices"
    )
    overlap = np.intersect1d(fit_positions, predict_positions, assume_unique=False)
    if len(overlap):
        raise ValueError("quantile fit and prediction positions must be disjoint")
    target = np.asarray(target_temp, dtype=float)
    label = np.asarray(labels)
    if target.shape != (len(design.frame),) or label.shape != target.shape:
        raise ValueError("target_temp and labels must align to the design")
    if not np.isin(label, [0, 1]).all():
        raise ValueError("labels must be binary")
    normal_fit = fit_positions[(label[fit_positions] == 0) & np.isfinite(target[fit_positions])]
    if len(normal_fit) < max(100, config.min_child_samples * 2):
        raise ValueError("too few finite fold-train normal rows for quantile fitting")

    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment contract
        raise ImportError("LightGBM is required for target-masked quantiles") from exc

    encoder = TargetMaskedEncoder().fit(design, normal_fit)
    train_matrix = encoder.transform(design, normal_fit)
    predict_matrix = encoder.transform(design, predict_positions)
    predictions = np.empty((len(predict_positions), len(config.alphas)), dtype=np.float64)
    for quantile_number, alpha in enumerate(config.alphas):
        seed = config.seed + seed_offset + quantile_number
        model = lgb.LGBMRegressor(**_quantile_parameters(config, alpha, seed))
        model.fit(train_matrix, target[normal_fit])
        predictions[:, quantile_number] = np.asarray(model.predict(predict_matrix), dtype=float)
    if not np.isfinite(predictions).all():
        raise RuntimeError("quantile model produced non-finite predictions")
    # Quantile crossing is a numerical/model artifact.  Sorting is fixed and
    # label-free; it is not a validation-tuned correction.
    predictions.sort(axis=1)
    audit = {
        "fit_scope_rows": int(len(fit_positions)),
        "normal_fit_rows": int(len(normal_fit)),
        "positive_fit_rows_used": int(label[normal_fit].sum()),
        "predict_rows": int(len(predict_positions)),
        "fit_predict_overlap_rows": int(len(overlap)),
        "feature_columns": list(design.feature_columns),
        "conditional_model_reads_own_temp": False,
        "conditional_model_reads_depth": False,
        "parameters": asdict(config),
    }
    return predictions, audit


def _crossfit_blocks(frame: pd.DataFrame, indices: np.ndarray, folds: int) -> np.ndarray:
    parsed = pd.to_datetime(frame.iloc[indices]["time"], errors="coerce", utc=True, format="mixed")
    if parsed.isna().any():
        raise ValueError("cross-fit timestamps must all be parseable")
    local = parsed.dt.tz_convert("Asia/Seoul")
    # A calendar ordinal need only be deterministic and advance once per KST
    # day.  Avoid integer datetime units, whose resolution differs by pandas
    # version.
    day_number = local.dt.year.to_numpy(dtype=np.int64) * 400 + local.dt.dayofyear.to_numpy(
        dtype=np.int64
    )
    blocks = np.mod(day_number, folds).astype(np.int8)
    if len(np.unique(blocks)) != folds:
        raise ValueError("cross-fit scope does not contain every frozen day block")
    return blocks


def cross_fitted_quantiles(
    source_frame: pd.DataFrame,
    design: TargetMaskedDesign,
    target_temp: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scope_indices: Sequence[int] | np.ndarray,
    *,
    config: QuantileModelConfig | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return out-of-block quantiles for every row in one training scope.

    All rows from a KST calendar day share a block, so contemporaneous peers
    cannot straddle quantile fit and holdout sides.  Only fold-scope normal rows
    enter each fit; outer/calibration rows are not accepted by this function.
    """

    config = config or QuantileModelConfig()
    scope = _positions(scope_indices, len(design.frame), name="cross-fit scope")
    blocks = _crossfit_blocks(source_frame, scope, config.crossfit_folds)
    output = np.full((len(design.frame), len(config.alphas)), np.nan, dtype=np.float64)
    block_audits: list[dict[str, Any]] = []
    for block in range(config.crossfit_folds):
        predict_positions = scope[blocks == block]
        fit_positions = scope[blocks != block]
        prediction, audit = fit_predict_quantiles(
            design,
            target_temp,
            labels,
            fit_positions,
            predict_positions,
            config=config,
            seed_offset=seed_offset + block * 10,
        )
        output[predict_positions] = prediction
        block_audits.append({"block": block, **audit})
    if not np.isfinite(output[scope]).all():
        raise RuntimeError("cross-fit did not predict every scope row")
    return output, {
        "scope_rows": int(len(scope)),
        "crossfit_folds": config.crossfit_folds,
        "all_scope_rows_predicted_once": True,
        "block_assignment": "KST calendar-day ordinal modulo crossfit_folds",
        "blocks": block_audits,
    }


def place_quantiles(
    length: int,
    indices: Sequence[int] | np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Place subset predictions into a full aligned NaN matrix."""

    positions = _positions(indices, length, name="quantile placement indices")
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(positions), 3):
        raise ValueError("quantile predictions must have shape (len(indices), 3)")
    result = np.full((length, 3), np.nan, dtype=np.float64)
    result[positions] = values
    return result


def _gap_aware_centered_mean(
    values: pd.Series,
    segments: pd.Series,
    *,
    rows: int,
) -> pd.Series:
    minimum = max(3, int(np.ceil(rows * 0.25)))
    result = (
        values.groupby(segments, sort=False, observed=True)
        .rolling(
            window=rows,
            min_periods=minimum,
            center=True,
        )
        .mean()
    )
    return result.reset_index(level=0, drop=True).sort_index()


def build_quantile_scores(
    source_frame: pd.DataFrame,
    quantiles: np.ndarray,
    *,
    cadence_minutes: int = 10,
) -> pd.DataFrame:
    """Convert conditional quantiles to the frozen residual-score contract."""

    values = np.asarray(quantiles, dtype=float)
    if values.shape != (len(source_frame), 3):
        raise ValueError("quantiles must have shape (len(source_frame), 3)")
    segmented = segment_timeseries(source_frame, cadence_minutes=cadence_minutes)
    segmented["__position"] = np.arange(len(segmented), dtype=np.int64)
    ordered = segmented.sort_values(
        ["station", "layer", "parsed_time", "__position"], kind="mergesort"
    ).reset_index(drop=True)
    ordered_quantiles = values[ordered["__position"].to_numpy(dtype=np.int64)]
    temp = pd.to_numeric(ordered["temp"], errors="coerce").to_numpy(dtype=float)
    available = np.isfinite(temp) & np.isfinite(ordered_quantiles).all(axis=1)
    q05, q50, q95 = ordered_quantiles.T
    signed = np.where(available, temp - q50, np.nan)
    width = np.where(available, np.maximum(q95 - q05, 1.0e-6), np.nan)
    lower = np.where(available, np.maximum(q05 - temp, 0.0), np.nan)
    upper = np.where(available, np.maximum(temp - q95, 0.0), np.nan)
    outside = lower + upper
    segment = ordered["segment_id"]

    ordered_scores = pd.DataFrame(index=ordered.index)
    ordered_scores["tmq_signed_residual"] = signed
    ordered_scores["tmq_interval_width"] = width
    ordered_scores["tmq_lower_tail_distance"] = lower
    ordered_scores["tmq_upper_tail_distance"] = upper
    ordered_scores["tmq_outside_tail_distance"] = outside
    ordered_scores["tmq_signed_residual_mean_24h"] = _gap_aware_centered_mean(
        pd.Series(signed), segment, rows=int(round(24 * 60 / cadence_minutes))
    )
    ordered_scores["tmq_signed_residual_mean_72h"] = _gap_aware_centered_mean(
        pd.Series(signed), segment, rows=int(round(72 * 60 / cadence_minutes))
    )
    ordered_scores["tmq_outside_tail_mean_24h"] = _gap_aware_centered_mean(
        pd.Series(outside), segment, rows=int(round(24 * 60 / cadence_minutes))
    )
    ordered_scores["tmq_quantile_available"] = available.astype(np.int8)

    restored = ordered_scores.copy()
    restored["__position"] = ordered["__position"].to_numpy(dtype=np.int64)
    restored.sort_values("__position", kind="mergesort", inplace=True)
    restored.drop(columns="__position", inplace=True)
    restored.index = source_frame.index.copy()
    return restored.loc[:, QUANTILE_SCORE_COLUMNS]


def append_score_matrix(base_matrix: np.ndarray, score_frame: pd.DataFrame) -> np.ndarray:
    """Append only the frozen score columns to an existing encoded matrix."""

    base = np.asarray(base_matrix, dtype=np.float32)
    if base.ndim != 2 or len(base) != len(score_frame):
        raise ValueError("base matrix and score frame must have matching rows")
    if tuple(score_frame.columns) != QUANTILE_SCORE_COLUMNS:
        raise ValueError("score columns differ from the frozen contract")
    scores = score_frame.to_numpy(dtype=np.float32, copy=True)
    scores[~np.isfinite(scores)] = np.nan
    return np.column_stack((base, scores)).astype(np.float32, copy=False)


def design_contract_hash() -> str:
    payload = "\n".join((*TARGET_MASKED_FEATURE_COLUMNS, "--", *QUANTILE_SCORE_COLUMNS))
    return sha256(payload.encode("utf-8")).hexdigest()


def synthetic_offset_smoke(config: QuantileModelConfig | None = None) -> dict[str, Any]:
    """Small deterministic end-to-end check with an unseen temperature offset."""

    cfg = config or QuantileModelConfig(
        n_estimators=30,
        min_child_samples=8,
        crossfit_folds=2,
        threads=1,
    )
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp("2024-01-01T00:00:00+09:00")
    steps = 12 * 24 * 6
    for step in range(steps):
        timestamp = start + pd.Timedelta(minutes=10 * step)
        phase = 2.0 * np.pi * step / (6 * 24)
        for layer in (1, 2):
            psal = 32.0 + 0.15 * np.sin(phase + layer * 0.05)
            temp = 18.0 - 0.7 * layer + 0.8 * np.sin(phase) - 0.45 * (psal - 32.0)
            label = 0
            if layer == 1 and step >= 10 * 24 * 6:
                temp += 2.5
                label = 1
            rows.append(
                {
                    "station": "SYN",
                    "year": 2024,
                    "layer": layer,
                    "time": timestamp.isoformat(),
                    "temp": temp,
                    "psal": psal,
                    "depth": float(layer * 10),
                    "label": label,
                }
            )
    frame = pd.DataFrame(rows)
    design = build_target_masked_design(frame)
    time = pd.to_datetime(frame["time"], utc=True)
    fit = np.flatnonzero(time.lt(pd.Timestamp("2024-01-10T00:00:00+09:00").tz_convert("UTC")))
    predict = np.flatnonzero(~np.isin(np.arange(len(frame)), fit))
    prediction, audit = fit_predict_quantiles(
        design,
        frame["temp"].to_numpy(),
        frame["label"].to_numpy(),
        fit,
        predict,
        config=cfg,
    )
    aligned = place_quantiles(len(frame), predict, prediction)
    scores = build_quantile_scores(frame, aligned)
    predicted_label = frame.iloc[predict]["label"].to_numpy(dtype=np.int8)
    tail = scores.iloc[predict]["tmq_outside_tail_distance"].to_numpy(dtype=float)
    normal_median = float(np.nanmedian(tail[predicted_label == 0]))
    offset_median = float(np.nanmedian(tail[predicted_label == 1]))
    passed = bool(offset_median > normal_median + 0.5)
    if not passed:
        raise RuntimeError("synthetic offset did not increase target-masked tail distance")
    return {
        "passed": passed,
        "rows": len(frame),
        "normal_tail_median": normal_median,
        "offset_tail_median": offset_median,
        "positive_fit_rows_used": audit["positive_fit_rows_used"],
        "gors_style_fallback_supported": True,
        "design_contract_hash": design_contract_hash(),
    }


__all__ = [
    "QUANTILE_SCORE_COLUMNS",
    "TARGET_MASKED_CATEGORICAL_COLUMNS",
    "TARGET_MASKED_FEATURE_COLUMNS",
    "TARGET_MASKED_NUMERIC_COLUMNS",
    "QuantileModelConfig",
    "TargetMaskedDesign",
    "TargetMaskedEncoder",
    "append_score_matrix",
    "assert_target_masked_contract",
    "build_quantile_scores",
    "build_target_masked_design",
    "cross_fitted_quantiles",
    "design_contract_hash",
    "fit_predict_quantiles",
    "place_quantiles",
    "synthetic_offset_smoke",
]
