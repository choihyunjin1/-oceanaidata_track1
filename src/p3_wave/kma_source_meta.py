"""Isolated KMA point-prediction meta features for the P3 paired ablation.

The source model is trained only on pre-2024 KMA buoy cases.  It never sees a
KMA station identifier, proxy mapping, calendar field, P3 target, or anonymous
test case.  Its six point predictions are the only values allowed to cross the
external-to-P3 boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEADS = (3, 6, 9, 12, 18, 24)
PAIR_KEYS = ("fold", "anchor_id", "station", "lead_h")
META_COLUMNS = tuple(f"kma_source_hs_pred_{lead}h" for lead in LEADS)
ROUTER_COLUMNS = (
    "multi_prediction",
    "persistence",
    "weight_single",
    "weight_multi",
    "weight_persistence",
    "second_stage_persistence_weight",
    "prediction",
)
SCALAR_VARIABLES = ("hs", "hmax", "wspd", "gust", "airt", "relh", "caph")
DIRECTION_VARIABLES = ("wvdir", "wdir")
ARRAY_ORDER = (
    *SCALAR_VARIABLES,
    "wvdir_sin",
    "wvdir_cos",
    "wdir_sin",
    "wdir_cos",
    "wave_energy",
    "hmax_hs_ratio",
    "gust_excess",
    "wind_wave_alignment",
    "wind_input_proxy",
)
LAG_HOURS = (1, 3, 6, 12, 24, 48)
WINDOW_HOURS = (3, 6, 12, 24, 48)
SUMMARY_STATISTICS = ("mean", "std", "delta", "slope")
HISTORY_ROWS = 97
SOURCE_CUTOFF = pd.Timestamp("2023-12-31T23:30:00+09:00")
FORBIDDEN_MODEL_TOKENS = (
    "station",
    "proxy",
    "time",
    "date",
    "year",
    "month",
    "day",
    "hour",
    "case_id",
)


class KMASourceMetaError(RuntimeError):
    """Fail-closed error for the isolated source-prediction experiment."""


@dataclass(frozen=True)
class DomainRoute:
    auc: float | None
    route: str
    direct_concat_allowed: bool
    requires_inner_incremental_signal: bool


@dataclass(frozen=True)
class SourceCases:
    features: pd.DataFrame
    residual_targets: np.ndarray
    current_hs: np.ndarray
    source_case_keys: pd.DataFrame


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preregistration(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KMASourceMetaError("preregistration must be a JSON object")
    validate_preregistration(value)
    return value


def validate_preregistration(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "1.0":
        raise KMASourceMetaError("unsupported preregistration schema")
    if config.get("experiment_id") != "p3_kma_source_prediction_meta_v1":
        raise KMASourceMetaError("unexpected experiment id")
    representation = config["representation"]
    if representation.get("history_hours") != 48:
        raise KMASourceMetaError("history must be frozen at 48 hours")
    if representation.get("grid_minutes") != 30:
        raise KMASourceMetaError("common grid must be frozen at 30 minutes")
    if representation.get("history_rows_including_anchor") != HISTORY_ROWS:
        raise KMASourceMetaError("common history row count changed")
    if tuple(representation.get("lead_hours", ())) != LEADS:
        raise KMASourceMetaError("official lead order changed")
    if tuple(representation.get("lag_hours", ())) != LAG_HOURS:
        raise KMASourceMetaError("lag surface changed")
    if tuple(representation.get("summary_windows_hours", ())) != WINDOW_HOURS:
        raise KMASourceMetaError("summary window surface changed")
    if tuple(representation.get("summary_statistics", ())) != SUMMARY_STATISTICS:
        raise KMASourceMetaError("summary statistics changed")
    if representation.get("expected_source_feature_count") != 447:
        raise KMASourceMetaError("source feature count is not frozen at 447")
    mismatch = representation.get("excluded_semantic_mismatch_variables", {})
    if mismatch.get("source") != ["WP"] or mismatch.get("target") != ["tp"]:
        raise KMASourceMetaError("period semantic-mismatch exclusion changed")
    if representation.get("absolute_calendar_features") is not False:
        raise KMASourceMetaError("absolute calendar features are prohibited")
    if representation.get("source_station_or_proxy_features") is not False:
        raise KMASourceMetaError("external identity features are prohibited")
    if representation.get("test_case_or_timestamp_mapping") is not False:
        raise KMASourceMetaError("anonymous test mapping is prohibited")
    missingness = representation.get("missingness_harmonization", {})
    if missingness.get("causal_forward_fill_limit_minutes") != 60:
        raise KMASourceMetaError("common causal fill limit changed")
    if missingness.get("applied_identically_to_source_and_target") is not True:
        raise KMASourceMetaError("missingness harmonization must apply to both domains")
    if missingness.get("missingness_or_valid_fraction_features") is not False:
        raise KMASourceMetaError("missingness fingerprint features are prohibited")
    if missingness.get("target_fit_statistics_used") is not False:
        raise KMASourceMetaError("target-fit imputation statistics are prohibited")

    source_model = config["source_model"]
    target_model = config["target_model"]
    if source_model.get("hyperparameter_grid_size") != 0:
        raise KMASourceMetaError("source hyperparameter search is prohibited")
    if target_model.get("hyperparameter_grid_size") != 0:
        raise KMASourceMetaError("target hyperparameter search is prohibited")
    if source_model.get("external_station_feature") is not False:
        raise KMASourceMetaError("source model cannot use station identity")
    if tuple(target_model.get("challenger_added_columns", ())) != META_COLUMNS:
        raise KMASourceMetaError("challenger must add exactly six frozen meta columns")
    if target_model.get("base_feature_count") != 591:
        raise KMASourceMetaError("target base surface changed")
    integration = config["frozen_final_integration"]
    if tuple(integration.get("allowed_pretruth_columns", ())) != (
        *PAIR_KEYS,
        *ROUTER_COLUMNS,
    ):
        raise KMASourceMetaError("frozen router pre-truth allowlist changed")
    if integration.get("forbidden_pretruth_column") != "target_hs":
        raise KMASourceMetaError("outer target prohibition is missing")
    if integration.get("long_lead_persistence_weight") != 0.2:
        raise KMASourceMetaError("frozen long-lead persistence weight changed")
    if integration.get("weight_router_or_shrink_reselection") is not False:
        raise KMASourceMetaError("router or shrink reselection is prohibited")

    domain = config["domain_shift"]
    if domain.get("direct_source_target_row_concatenation_allowed") is not False:
        raise KMASourceMetaError("direct source-target row concatenation must remain disabled")
    routes = domain.get("auc_routes")
    if not isinstance(routes, list) or [row.get("maximum_inclusive") for row in routes] != [
        0.65,
        0.8,
        1.0,
    ]:
        raise KMASourceMetaError("domain AUC routes changed")
    high = domain["high_auc_inner_gate"]
    if high.get("required_above_auc") != 0.8:
        raise KMASourceMetaError("high-domain-shift threshold changed")
    if high.get("minimum_improved_inner_blocks") != 2:
        raise KMASourceMetaError("high-AUC inner block gate changed")
    if high.get("station_lead_or_ci_veto") is not False:
        raise KMASourceMetaError("station/lead/CI veto belongs only to outer promotion")
    if high.get("failure_action") != "stop_before_outer_prediction_and_truth":
        raise KMASourceMetaError("high-AUC failure must stop before outer truth")

    validation = config["validation"]
    if validation.get("outer_membership") != "frozen_incumbent_oof_keys_only":
        raise KMASourceMetaError("outer membership must come from frozen OOF keys")
    if validation.get("embargo_hours") != 78:
        raise KMASourceMetaError("outer embargo changed")
    if validation.get("inner_validation_days") != 45:
        raise KMASourceMetaError("inner validation length changed")
    if validation.get("prediction_contract") != (
        "each_fold_prediction_excludes_that_folds_validation_targets_and_global_blind_"
        "predictions_are_fsynced_hashed_and_reloaded_before_designated_scoring_read"
    ):
        raise KMASourceMetaError("fold-local prediction sealing contract changed")
    rolling_scope = validation.get("rolling_origin_label_scope", {})
    expected_rolling_scope = {
        "current_fold_validation_targets_excluded_from_that_fold_training_and_inner_selection": (
            True
        ),
        "earlier_fold_validation_targets_allowed_only_as_later_fold_training_history": True,
        "future_fold_validation_targets_forbidden_from_earlier_fold_training": True,
        "global_process_level_zero_outer_target_exposure_before_blind_seal_claimed": False,
        "designated_scoring_read_occurs_after_global_blind_seal": True,
    }
    if rolling_scope != expected_rolling_scope:
        raise KMASourceMetaError("rolling-origin fold-local label scope changed")
    if validation.get("one_shot_no_rerun") is not True:
        raise KMASourceMetaError("one-shot contract must remain enabled")
    if config["prohibitions"].get("direct_source_target_row_concatenation") is not True:
        raise KMASourceMetaError("direct concatenation prohibition is missing")
    if (
        config["prohibitions"].get(
            "current_fold_validation_target_in_that_fold_training_or_inner_selection"
        )
        is not True
    ):
        raise KMASourceMetaError("current-fold validation target prohibition is missing")
    promotion = config["promotion_gate"]
    if promotion.get("applies_to") != "challenger_final_vs_exact_incumbent_final_only":
        raise KMASourceMetaError("promotion must be measured against exact final incumbent")
    execution = config["execution"]
    if execution.get("actual_authorized") is not False:
        raise KMASourceMetaError("canonical preregistration must remain dry-only")
    if execution.get("actual_authorization_mechanism") != (
        "separate_O_EXCL_amendment_bound_to_exact_dry_receipt_and_implementation_SHA"
    ):
        raise KMASourceMetaError("actual authorization mechanism changed")


def resolve_domain_route(auc: float | None) -> DomainRoute:
    """Route domain AUC without using it as a prediction-meta veto."""

    if auc is None:
        return DomainRoute(None, "pending_domain_result", False, False)
    value = float(auc)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise KMASourceMetaError("domain AUC must be finite and within [0, 1]")
    if value <= 0.65:
        route = "full_six_prediction_meta"
        high_gate = False
    elif value <= 0.8:
        route = "prediction_meta_only"
        high_gate = False
    else:
        route = "prediction_meta_only_with_inner_incremental_signal_gate"
        high_gate = True
    return DomainRoute(value, route, False, high_gate)


def compact_source_feature_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for name in ARRAY_ORDER:
        columns.append(f"{name}_current")
        columns.extend(f"{name}_lag_{hour}h" for hour in LAG_HOURS)
        for hour in WINDOW_HOURS:
            columns.extend(f"{name}_{statistic}_{hour}h" for statistic in SUMMARY_STATISTICS)
    for hour in (1, 3, 6, 12, 24):
        columns.extend((f"hs_change_{hour}h", f"wspd_change_{hour}h", f"caph_change_{hour}h"))
    if len(columns) != 447 or len(set(columns)) != len(columns):
        raise AssertionError("compact common feature surface is not exactly 447 unique columns")
    forbidden = {
        column
        for column in columns
        if any(
            column.lower() == token or column.lower().startswith(f"{token}_")
            for token in FORBIDDEN_MODEL_TOKENS
        )
    }
    if forbidden:
        raise AssertionError(f"forbidden identity/calendar feature names: {sorted(forbidden)}")
    return tuple(columns)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def canonicalize_kma_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Map KMA names to the station-agnostic P3 common variable vocabulary."""

    required = {
        "TM",
        "STN",
        "WD1",
        "WS1",
        "WS1_GST",
        "WD2",
        "WS2",
        "WS2_GST",
        "PA",
        "HM",
        "TA",
        "WH_MAX",
        "WH_SIG",
        "WP",
        "WO",
    }
    if not required <= set(frame.columns):
        raise KMASourceMetaError(f"KMA source is missing {sorted(required - set(frame.columns))}")
    result = pd.DataFrame()
    timestamp = pd.to_datetime(frame["TM"], errors="raise")
    if timestamp.dt.tz is None:
        raise KMASourceMetaError("KMA source timestamps must be timezone aware")
    timestamp = timestamp.dt.tz_convert("Asia/Seoul")
    if timestamp.max() > SOURCE_CUTOFF:
        raise KMASourceMetaError("KMA source crosses the pre-2024 cutoff")
    if not timestamp.dt.minute.isin([0, 30]).all():
        raise KMASourceMetaError("KMA source is not on its native 30-minute grid")
    station = pd.to_numeric(frame["STN"], errors="raise").astype("int64")
    if pd.DataFrame({"station": station, "time": timestamp}).duplicated().any():
        raise KMASourceMetaError("KMA source contains duplicate station timestamps")

    ws1 = _numeric(frame, "WS1")
    ws2 = _numeric(frame, "WS2")
    wd1 = _numeric(frame, "WD1")
    wd2 = _numeric(frame, "WD2")
    gust1 = _numeric(frame, "WS1_GST")
    gust2 = _numeric(frame, "WS2_GST")
    use_sensor1 = np.isfinite(ws1)
    result["_source_group"] = station
    result["_time"] = timestamp
    result["hs"] = _numeric(frame, "WH_SIG")
    result["hmax"] = _numeric(frame, "WH_MAX")
    result["wvdir"] = _numeric(frame, "WO")
    result["wspd"] = np.where(use_sensor1, ws1, ws2)
    result["gust"] = np.where(np.isfinite(gust1), gust1, gust2)
    result["wdir"] = np.where(use_sensor1 & np.isfinite(wd1), wd1, wd2)
    result["caph"] = _numeric(frame, "PA")
    result["airt"] = _numeric(frame, "TA")
    result["relh"] = _numeric(frame, "HM")
    return result.sort_values(["_source_group", "_time"]).reset_index(drop=True)


def _compact_summary(values: np.ndarray, x_hours: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(values)
    if not mask.any():
        return {name: np.nan for name in SUMMARY_STATISTICS}
    y = values[mask]
    x = x_hours[mask]
    slope = np.nan
    if len(y) >= 3 and float(np.ptp(x)) > 0:
        centered = x - x.mean()
        denominator = float(np.dot(centered, centered))
        if denominator > 0:
            slope = float(np.dot(centered, y - y.mean()) / denominator)
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "delta": (
            float(values[-1] - values[0])
            if np.isfinite(values[-1]) and np.isfinite(values[0])
            else np.nan
        ),
        "slope": slope,
    }


def summarize_common_history(history: pd.DataFrame) -> dict[str, float]:
    """Create the frozen 591-column surface from one causal 30-minute history."""

    if len(history) != HISTORY_ROWS:
        raise KMASourceMetaError(f"common history must contain {HISTORY_ROWS} rows")
    required = set(SCALAR_VARIABLES + DIRECTION_VARIABLES)
    if not required <= set(history.columns):
        raise KMASourceMetaError(
            f"common history is missing {sorted(required - set(history.columns))}"
        )
    # Apply the same strictly causal, short-gap fill to both domains.  No
    # availability fraction or missingness mask is exposed to the source model;
    # remaining gaps are imputed from source-training medians after summarizing.
    harmonized = history.loc[:, [*SCALAR_VARIABLES, *DIRECTION_VARIABLES]].ffill(limit=2)
    arrays: dict[str, np.ndarray] = {
        column: _numeric(harmonized, column) for column in SCALAR_VARIABLES + DIRECTION_VARIABLES
    }
    for direction in DIRECTION_VARIABLES:
        radians = np.deg2rad(arrays.pop(direction))
        arrays[f"{direction}_sin"] = np.sin(radians)
        arrays[f"{direction}_cos"] = np.cos(radians)

    hs = arrays["hs"]
    hmax = arrays["hmax"]
    wspd = arrays["wspd"]
    gust = arrays["gust"]
    wave_sin = arrays["wvdir_sin"]
    wave_cos = arrays["wvdir_cos"]
    wind_sin = arrays["wdir_sin"]
    wind_cos = arrays["wdir_cos"]
    with np.errstate(divide="ignore", invalid="ignore"):
        arrays["wave_energy"] = hs**2
        arrays["hmax_hs_ratio"] = np.where(hs > 0.05, hmax / hs, np.nan)
        arrays["gust_excess"] = gust - wspd
        alignment = wind_cos * wave_cos + wind_sin * wave_sin
        alignment[~(np.isfinite(wind_cos) & np.isfinite(wave_cos))] = np.nan
        arrays["wind_wave_alignment"] = alignment
        arrays["wind_input_proxy"] = wspd**2 * np.maximum(alignment, 0.0)

    if tuple(arrays) != ARRAY_ORDER:
        raise AssertionError("common array ordering drifted")
    row: dict[str, float] = {}
    for name, values in arrays.items():
        row[f"{name}_current"] = float(values[-1]) if np.isfinite(values[-1]) else np.nan
        for hour in LAG_HOURS:
            position = -1 - hour * 2
            row[f"{name}_lag_{hour}h"] = (
                float(values[position]) if np.isfinite(values[position]) else np.nan
            )
        for hour in WINDOW_HOURS:
            length = hour * 2 + 1
            section = values[-length:]
            x = np.arange(-len(section) + 1, 1, dtype=np.float64) / 2.0
            for statistic, value in _compact_summary(section, x).items():
                row[f"{name}_{statistic}_{hour}h"] = value
    for hour in (1, 3, 6, 12, 24):
        row[f"hs_change_{hour}h"] = row["hs_current"] - row[f"hs_lag_{hour}h"]
        row[f"wspd_change_{hour}h"] = row["wspd_current"] - row[f"wspd_lag_{hour}h"]
        row[f"caph_change_{hour}h"] = row["caph_current"] - row[f"caph_lag_{hour}h"]
    if tuple(row) != compact_source_feature_columns():
        raise AssertionError("common feature ordering does not match the frozen surface")
    return row


def _validate_source_anchor_spacing(anchors: pd.DataFrame) -> None:
    required = {"station_id", "anchor_time_kst"}
    if not required <= set(anchors.columns):
        raise KMASourceMetaError(f"source anchors are missing {sorted(required - set(anchors))}")
    value = anchors.copy()
    value["anchor_time_kst"] = pd.to_datetime(value["anchor_time_kst"], errors="raise")
    if value["anchor_time_kst"].dt.tz is None:
        raise KMASourceMetaError("source anchor timestamps must be timezone aware")
    for _, group in value.groupby("station_id", observed=True):
        delta = group["anchor_time_kst"].sort_values().diff().dropna()
        if not delta.ge(pd.Timedelta(hours=78)).all():
            raise KMASourceMetaError("source anchors violate 78-hour independence")


def build_source_cases(observations: pd.DataFrame, anchors: pd.DataFrame) -> SourceCases:
    """Build external-only source features and six residual targets."""

    _validate_source_anchor_spacing(anchors)
    mapped = canonicalize_kma_observations(observations)
    by_station = {
        int(key): group.set_index("_time").drop(columns="_source_group").sort_index()
        for key, group in mapped.groupby("_source_group", sort=True, observed=True)
    }
    rows: list[dict[str, float]] = []
    residuals: list[np.ndarray] = []
    currents: list[float] = []
    keys: list[dict[str, Any]] = []
    for number, anchor in enumerate(anchors.itertuples(index=False), start=1):
        station_id = int(anchor.station_id)
        if station_id not in by_station:
            raise KMASourceMetaError("source anchor references an unknown station epoch")
        timestamp = pd.Timestamp(anchor.anchor_time_kst).tz_convert("Asia/Seoul")
        history_index = pd.date_range(timestamp - pd.Timedelta(hours=48), timestamp, freq="30min")
        history = by_station[station_id].reindex(history_index)
        row = summarize_common_history(history)
        current = float(history["hs"].iloc[-1])
        target_index = pd.DatetimeIndex([timestamp + pd.Timedelta(hours=lead) for lead in LEADS])
        target = by_station[station_id]["hs"].reindex(target_index).to_numpy(dtype=np.float64)
        if not np.isfinite(current) or current < 1.5:
            raise KMASourceMetaError("source anchor lacks its required storm-state hs")
        if not np.isfinite(target).all():
            raise KMASourceMetaError("source anchor lacks one or more six-lead targets")
        rows.append(row)
        residuals.append(target - current)
        currents.append(current)
        keys.append({"source_case_number": number, "anchor_time_kst": timestamp})
    features = pd.DataFrame(rows, columns=compact_source_feature_columns())
    if any(
        any(
            column.lower() == token or column.lower().startswith(f"{token}_")
            for token in FORBIDDEN_MODEL_TOKENS
        )
        for column in features
    ):
        raise AssertionError("source model matrix contains a forbidden identity/calendar column")
    return SourceCases(
        features=features,
        residual_targets=np.vstack(residuals),
        current_hs=np.asarray(currents, dtype=np.float64),
        source_case_keys=pd.DataFrame(keys),
    )


def _causal_asof(
    query_times: pd.DatetimeIndex,
    source: pd.DataFrame,
    columns: Sequence[str],
    *,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    query = pd.DataFrame({"_query_time": query_times})
    selected = source.loc[:, ["time", *columns]].sort_values("time").copy()
    selected["time"] = pd.to_datetime(selected["time"], utc=True, errors="raise")
    merged = pd.merge_asof(
        query.sort_values("_query_time"),
        selected,
        left_on="_query_time",
        right_on="time",
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )
    if merged["time"].notna().any():
        delay = (
            merged.loc[merged["time"].notna(), "_query_time"]
            - merged.loc[merged["time"].notna(), "time"]
        )
        if delay.lt(pd.Timedelta(0)).any() or delay.gt(tolerance).any():
            raise AssertionError("causal as-of joined a future or stale observation")
    return merged.loc[:, columns].reset_index(drop=True)


def extract_target_common_history(
    wave: pd.DataFrame,
    atmos: pd.DataFrame,
    *,
    station: str,
    anchor_time: pd.Timestamp,
) -> pd.DataFrame:
    """Build one P3 history without reading any future target or test case."""

    timestamp = pd.Timestamp(anchor_time)
    if timestamp.tzinfo is None:
        raise KMASourceMetaError("P3 anchor time must be timezone aware")
    timestamp = timestamp.tz_convert("UTC")
    query_times = pd.date_range(timestamp - pd.Timedelta(hours=48), timestamp, freq="30min")
    current_wave = wave.loc[wave["station"].astype(str).eq(str(station))]
    current_atmos = atmos.loc[atmos["station"].astype(str).eq(str(station))]
    if current_wave.empty or current_atmos.empty:
        raise KMASourceMetaError("target station is absent from raw training inputs")
    wave_values = _causal_asof(
        query_times,
        current_wave,
        ("hs", "hmax", "wvdir"),
        tolerance=pd.Timedelta(minutes=20) - pd.Timedelta(nanoseconds=1),
    )
    atmos_values = _causal_asof(
        query_times,
        current_atmos,
        ("wspd", "gust", "wdir", "airt", "relh", "caph"),
        tolerance=pd.Timedelta(minutes=10) - pd.Timedelta(nanoseconds=1),
    )
    history = pd.concat([wave_values, atmos_values], axis=1)
    return history.loc[:, [*SCALAR_VARIABLES, *DIRECTION_VARIABLES]]


def build_target_source_features(
    wave: pd.DataFrame,
    atmos: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Create label-free external-model inputs for P3 training anchors."""

    required = {"anchor_id", "station", "anchor_time"}
    if not required <= set(anchors.columns):
        raise KMASourceMetaError(
            f"target anchor metadata is missing {sorted(required - set(anchors))}"
        )
    if anchors["anchor_id"].duplicated().any():
        raise KMASourceMetaError("target anchor metadata contains duplicate ids")
    rows: list[dict[str, float | int]] = []
    total = len(anchors)
    for number, anchor in enumerate(anchors.itertuples(index=False), start=1):
        history = extract_target_common_history(
            wave,
            atmos,
            station=str(anchor.station),
            anchor_time=pd.Timestamp(anchor.anchor_time),
        )
        row: dict[str, float | int] = {"anchor_id": int(anchor.anchor_id)}
        row.update(summarize_common_history(history))
        rows.append(row)
        if progress is not None and (number == total or number % 250 == 0):
            progress(number, total)
    return pd.DataFrame(rows, columns=["anchor_id", *compact_source_feature_columns()])


def fit_source_median_imputer(features: pd.DataFrame) -> pd.Series:
    """Fit one source-only imputer without exposing a missingness fingerprint."""

    columns = list(compact_source_feature_columns())
    if list(features.columns) != columns:
        raise KMASourceMetaError("source imputer received a non-canonical feature surface")
    medians = features.median(axis=0, skipna=True).astype("float64")
    if medians.isna().any() or not np.isfinite(medians.to_numpy()).all():
        missing = medians.index[medians.isna()].tolist()
        raise KMASourceMetaError(f"source features are entirely missing: {missing}")
    return medians


def apply_source_median_imputer(features: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Apply source-fit medians identically to source and target feature rows."""

    columns = list(compact_source_feature_columns())
    if list(features.columns) != columns or list(medians.index) != columns:
        raise KMASourceMetaError("source imputer surface or ordering changed")
    result = features.astype("float64").fillna(medians)
    if not np.isfinite(result.to_numpy()).all():
        raise KMASourceMetaError("source median imputation left a non-finite feature")
    return result


def source_predictions_to_meta(
    residual_prediction: np.ndarray,
    *,
    anchor_ids: Sequence[int],
    current_hs: Sequence[float],
) -> pd.DataFrame:
    prediction = np.asarray(residual_prediction, dtype=np.float64)
    ids = np.asarray(anchor_ids, dtype=np.int64)
    current = np.asarray(current_hs, dtype=np.float64)
    if prediction.shape != (len(ids), len(LEADS)):
        raise KMASourceMetaError("source prediction matrix must have six columns")
    if current.shape != (len(ids),):
        raise KMASourceMetaError("current hs vector does not match source predictions")
    absolute = np.clip(current[:, None] + prediction, 0.0, 30.0)
    if not np.isfinite(absolute).all():
        raise KMASourceMetaError("source meta prediction contains a non-finite value")
    result = pd.DataFrame(absolute, columns=META_COLUMNS)
    result.insert(0, "anchor_id", ids)
    if result["anchor_id"].duplicated().any():
        raise KMASourceMetaError("source meta prediction contains duplicate anchor ids")
    return result


def append_meta_features(
    base_features: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    expected_base_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Append exactly six sealed source predictions, never source rows or identities."""

    expected = tuple(expected_base_columns or ())
    required_base = {"anchor_id", "station", *expected}
    if not required_base <= set(base_features.columns):
        raise KMASourceMetaError("base target feature surface is incomplete")
    if expected and len(expected) != 591:
        raise KMASourceMetaError("target base feature surface must contain exactly 591 columns")
    if list(meta.columns) != ["anchor_id", *META_COLUMNS]:
        raise KMASourceMetaError("meta table must contain one key and exactly six predictions")
    if base_features["anchor_id"].duplicated().any() or meta["anchor_id"].duplicated().any():
        raise KMASourceMetaError("feature or meta key is duplicated")
    merged = base_features.merge(
        meta, on="anchor_id", how="left", validate="one_to_one", sort=False
    )
    if merged[list(META_COLUMNS)].isna().any().any():
        raise KMASourceMetaError("source meta coverage is incomplete")
    if not np.isfinite(merged[list(META_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise KMASourceMetaError("source meta contains non-finite values")
    return merged


def read_frozen_outer_key_membership(
    path: str | Path,
    *,
    expected_folds: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read only key columns from the incumbent OOF; prediction/truth stay unopened."""

    keys = pd.read_parquet(path, columns=list(PAIR_KEYS))
    if list(keys.columns) != list(PAIR_KEYS):
        raise KMASourceMetaError("frozen OOF key schema changed")
    if keys.duplicated(list(PAIR_KEYS)).any():
        raise KMASourceMetaError("frozen OOF contains duplicate keys")
    if set(keys["fold"].astype(str)) != set(expected_folds):
        raise KMASourceMetaError("frozen OOF fold set changed")
    leads = keys.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
        lambda value: tuple(sorted(int(item) for item in value))
    )
    if not leads.map(lambda value: value == LEADS).all():
        raise KMASourceMetaError("each frozen OOF case must contain all six leads")
    station_count = keys.groupby(["fold", "anchor_id"], observed=True)["station"].nunique()
    if not station_count.eq(1).all():
        raise KMASourceMetaError("a frozen OOF case maps to multiple stations")
    membership = {
        name: np.sort(
            keys.loc[keys["fold"].astype(str).eq(name), "anchor_id"].unique().astype(np.int64)
        )
        for name in expected_folds
    }
    return keys, membership


def read_frozen_router_components(path: str | Path) -> pd.DataFrame:
    """Read the frozen label-free router/shrink columns and exact incumbent prediction."""

    columns = [*PAIR_KEYS, *ROUTER_COLUMNS]
    frame = pd.read_parquet(path, columns=columns)
    if list(frame.columns) != columns or frame.duplicated(list(PAIR_KEYS)).any():
        raise KMASourceMetaError("frozen router component schema or keys changed")
    numeric = frame[list(ROUTER_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise KMASourceMetaError("frozen router component contains a non-finite value")
    weights = frame[["weight_single", "weight_multi", "weight_persistence"]].to_numpy(
        dtype=np.float64
    )
    if (weights < 0.0).any() or not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise KMASourceMetaError("frozen router weights are invalid")
    expected_shrink = np.where(frame["lead_h"].isin([12, 18, 24]), 0.2, 0.0)
    if not np.array_equal(
        frame["second_stage_persistence_weight"].to_numpy(dtype=np.float64),
        expected_shrink,
    ):
        raise KMASourceMetaError("frozen long-lead persistence shrink changed")
    if frame["prediction"].lt(0.0).any() or frame["prediction"].gt(30.0).any():
        raise KMASourceMetaError("exact frozen incumbent prediction is outside [0, 30]")
    return frame


def integrate_frozen_router(single_blind: pd.DataFrame, router: pd.DataFrame) -> pd.DataFrame:
    """Replace only the single component; router weights and 0.2 shrink stay frozen."""

    expected_single = [
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "current_hs",
        "control_single_prediction",
        "challenger_single_prediction",
    ]
    if list(single_blind.columns) != expected_single:
        raise KMASourceMetaError("single-model blind prediction schema changed")
    if list(router.columns) != [*PAIR_KEYS, *ROUTER_COLUMNS]:
        raise KMASourceMetaError("router integration received a non-canonical component table")
    merged = single_blind.merge(
        router,
        on=list(PAIR_KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(single_blind) or len(merged) != len(router):
        raise KMASourceMetaError("single prediction and frozen router keys differ")
    weight_single = merged["weight_single"].to_numpy(dtype=np.float64)
    weight_multi = merged["weight_multi"].to_numpy(dtype=np.float64)
    weight_persistence = merged["weight_persistence"].to_numpy(dtype=np.float64)
    multi = merged["multi_prediction"].to_numpy(dtype=np.float64)
    persistence = merged["persistence"].to_numpy(dtype=np.float64)
    shrink = merged["second_stage_persistence_weight"].to_numpy(dtype=np.float64)
    control_routed = (
        weight_single * merged["control_single_prediction"].to_numpy(dtype=np.float64)
        + weight_multi * multi
        + weight_persistence * persistence
    )
    challenger_routed = (
        weight_single * merged["challenger_single_prediction"].to_numpy(dtype=np.float64)
        + weight_multi * multi
        + weight_persistence * persistence
    )
    merged["control_final"] = (1.0 - shrink) * control_routed + shrink * persistence
    merged["challenger_final"] = (1.0 - shrink) * challenger_routed + shrink * persistence
    merged = merged.rename(columns={"prediction": "incumbent_final"})
    result = merged[
        [
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "current_hs",
            "control_single_prediction",
            "challenger_single_prediction",
            "multi_prediction",
            "persistence",
            "weight_single",
            "weight_multi",
            "weight_persistence",
            "second_stage_persistence_weight",
            "control_final",
            "challenger_final",
            "incumbent_final",
        ]
    ]
    validate_blind_prediction_frame(result)
    return result


def validate_outer_membership_against_anchors(keys: pd.DataFrame, anchors: pd.DataFrame) -> None:
    lookup = anchors.set_index("anchor_id")["station"].astype(str)
    station = keys["anchor_id"].map(lookup)
    if station.isna().any() or not station.eq(keys["station"].astype(str)).all():
        raise KMASourceMetaError("frozen OOF keys do not match anchor metadata")


def expand_target_rows(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    targets: pd.DataFrame,
    anchor_ids: Sequence[int],
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Expand only explicitly released target rows to the six-lead regression grain."""

    ids = np.asarray(anchor_ids, dtype=np.int64)
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    target_lookup = targets.set_index("anchor_id")
    if not np.isin(ids, target_lookup.index.to_numpy(dtype=np.int64)).all():
        raise KMASourceMetaError("target vault did not release every requested anchor")
    blocks: list[pd.DataFrame] = []
    residuals: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for lead in LEADS:
        block = feature_lookup.loc[ids, list(feature_columns)].reset_index(drop=True)
        station = anchor_lookup.loc[ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[ids, "current_hs"].to_numpy(dtype=np.float64)
        target = target_lookup.loc[ids, f"target_{lead}"].to_numpy(dtype=np.float64)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", lead)
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        residuals.append(target - current)
        metadata.append(
            pd.DataFrame(
                {
                    "anchor_id": ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                    "target_hs": target,
                }
            )
        )
    return (
        pd.concat(blocks, ignore_index=True),
        np.concatenate(residuals),
        pd.concat(metadata, ignore_index=True),
    )


def expand_prediction_rows(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_ids: Sequence[int],
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand label-free validation rows while keeping outer truth unopened."""

    ids = np.asarray(anchor_ids, dtype=np.int64)
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    metadata: list[pd.DataFrame] = []
    for lead in LEADS:
        block = feature_lookup.loc[ids, list(feature_columns)].reset_index(drop=True)
        station = anchor_lookup.loc[ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[ids, "current_hs"].to_numpy(dtype=np.float64)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", lead)
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        metadata.append(
            pd.DataFrame(
                {
                    "anchor_id": ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True), pd.concat(metadata, ignore_index=True)


def catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station"] = result["station"].astype(str)
    result["lead_h"] = result["lead_h"].astype(str)
    return result


def source_catboost(parameters: Mapping[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    expected = {
        "loss_function": "MultiRMSE",
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.1,
        "random_seed": 20260821,
        "task_type": "CPU",
        "thread_count": 8,
        "allow_writing_files": False,
    }
    current = {key: parameters.get(key) for key in expected}
    if current != expected:
        raise KMASourceMetaError("source CatBoost parameters differ from preregistration")
    return CatBoostRegressor(verbose=False, **expected)


def target_catboost(parameters: Mapping[str, Any], *, iterations: int) -> Any:
    from catboost import CatBoostRegressor

    if not 1 <= int(iterations) <= int(parameters["maximum_iterations"]):
        raise KMASourceMetaError("target CatBoost iteration is outside the frozen limit")
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=int(iterations),
        learning_rate=float(parameters["learning_rate"]),
        depth=int(parameters["depth"]),
        l2_leaf_reg=float(parameters["l2_leaf_reg"]),
        random_strength=float(parameters["random_strength"]),
        random_seed=int(parameters["random_seed"]),
        task_type=str(parameters["task_type"]),
        thread_count=int(parameters["thread_count"]),
        allow_writing_files=bool(parameters["allow_writing_files"]),
        verbose=False,
    )


def validate_blind_prediction_frame(frame: pd.DataFrame) -> dict[str, Any]:
    expected = [
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "current_hs",
        "control_single_prediction",
        "challenger_single_prediction",
        "multi_prediction",
        "persistence",
        "weight_single",
        "weight_multi",
        "weight_persistence",
        "second_stage_persistence_weight",
        "control_final",
        "challenger_final",
        "incumbent_final",
    ]
    if list(frame.columns) != expected:
        raise KMASourceMetaError("blind prediction schema changed")
    forbidden = [
        column for column in frame if "target" in column.lower() or "truth" in column.lower()
    ]
    if forbidden:
        raise KMASourceMetaError("outer truth leaked into blind predictions")
    if frame.duplicated(list(PAIR_KEYS)).any():
        raise KMASourceMetaError("blind prediction contains duplicate keys")
    prediction_columns = [
        "control_single_prediction",
        "challenger_single_prediction",
        "multi_prediction",
        "persistence",
        "control_final",
        "challenger_final",
        "incumbent_final",
    ]
    numeric = frame[["current_hs", *prediction_columns]].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or (numeric[:, 1:] < 0.0).any()
        or (numeric[:, 1:] > 30.0).any()
    ):
        raise KMASourceMetaError("blind prediction is non-finite or outside [0, 30]")
    weights = frame[["weight_single", "weight_multi", "weight_persistence"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise KMASourceMetaError("blind router weights are invalid")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise KMASourceMetaError("blind router weights do not sum to one")
    shrink = frame["second_stage_persistence_weight"].to_numpy(dtype=np.float64)
    expected_shrink = np.where(frame["lead_h"].isin([12, 18, 24]), 0.2, 0.0)
    if not np.array_equal(shrink, expected_shrink):
        raise KMASourceMetaError("blind long-lead shrink differs from frozen 0.2 policy")
    persistence = frame["persistence"].to_numpy(dtype=np.float64)
    multi = frame["multi_prediction"].to_numpy(dtype=np.float64)
    for single_column, final_column in (
        ("control_single_prediction", "control_final"),
        ("challenger_single_prediction", "challenger_final"),
    ):
        routed = (
            weights[:, 0] * frame[single_column].to_numpy(dtype=np.float64)
            + weights[:, 1] * multi
            + weights[:, 2] * persistence
        )
        reconstructed = (1.0 - shrink) * routed + shrink * persistence
        if not np.allclose(
            reconstructed,
            frame[final_column].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise KMASourceMetaError(f"{final_column} does not reconstruct from frozen router")
    return {"rows": int(len(frame)), "cases": int(frame["anchor_id"].nunique())}


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def metric_slices(evaluated: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    truth = evaluated["target_hs"].to_numpy(dtype=np.float64)
    prediction = evaluated[prediction_column].to_numpy(dtype=np.float64)
    result: dict[str, Any] = {"rmse": _rmse(truth, prediction), "rows": int(len(evaluated))}
    result["by_fold"] = {
        str(key): _rmse(group["target_hs"].to_numpy(), group[prediction_column].to_numpy())
        for key, group in evaluated.groupby("fold", observed=True)
    }
    result["by_station"] = {
        str(key): _rmse(group["target_hs"].to_numpy(), group[prediction_column].to_numpy())
        for key, group in evaluated.groupby("station", observed=True)
    }
    result["by_lead"] = {
        str(int(key)): _rmse(group["target_hs"].to_numpy(), group[prediction_column].to_numpy())
        for key, group in evaluated.groupby("lead_h", observed=True)
    }
    return result


def evaluate_inner_incremental_signal(inner: pd.DataFrame) -> dict[str, Any]:
    """Minimal high-AUC routing gate; station/lead/CI vetoes remain outer-only."""

    required = {"fold", "target_hs", "control_prediction", "challenger_prediction"}
    if not required <= set(inner.columns):
        raise KMASourceMetaError("inner utility frame is incomplete")
    if inner.empty or inner["fold"].nunique() != 3:
        raise KMASourceMetaError("inner utility gate requires exactly three blocks")
    control = _rmse(inner["target_hs"].to_numpy(), inner["control_prediction"].to_numpy())
    challenger = _rmse(inner["target_hs"].to_numpy(), inner["challenger_prediction"].to_numpy())
    delta_by_block = {
        str(name): _rmse(group["target_hs"].to_numpy(), group["challenger_prediction"].to_numpy())
        - _rmse(group["target_hs"].to_numpy(), group["control_prediction"].to_numpy())
        for name, group in inner.groupby("fold", observed=True)
    }
    improved = int(sum(value < 0.0 for value in delta_by_block.values()))
    delta = float(challenger - control)
    return {
        "pooled_control_rmse": control,
        "pooled_challenger_rmse": challenger,
        "pooled_delta_rmse": delta,
        "improved_blocks": improved,
        "delta_by_block": delta_by_block,
        "pass": delta < 0.0 and improved >= 2,
        "station_lead_or_ci_veto_applied": False,
    }


def paired_case_bootstrap(
    evaluated: pd.DataFrame,
    *,
    replicates: int = 5000,
    seed: int = 20260821,
    control_column: str = "control_prediction",
    challenger_column: str = "challenger_prediction",
) -> dict[str, float | int]:
    """Bootstrap challenger-minus-control RMSE at independent case grain."""

    case_columns = ["fold", "anchor_id"]
    grouped = list(evaluated.groupby(case_columns, sort=True, observed=True))
    if not grouped:
        raise KMASourceMetaError("paired bootstrap received no cases")
    control_sse = np.asarray(
        [np.square(g[control_column] - g["target_hs"]).sum() for _, g in grouped],
        dtype=np.float64,
    )
    challenger_sse = np.asarray(
        [np.square(g[challenger_column] - g["target_hs"]).sum() for _, g in grouped],
        dtype=np.float64,
    )
    counts = np.asarray([len(g) for _, g in grouped], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        draw = rng.integers(0, len(grouped), size=len(grouped))
        denominator = counts[draw].sum()
        control = np.sqrt(control_sse[draw].sum() / denominator)
        challenger = np.sqrt(challenger_sse[draw].sum() / denominator)
        deltas[index] = challenger - control
    return {
        "replicates": int(replicates),
        "case_count": int(len(grouped)),
        "delta_rmse_mean": float(deltas.mean()),
        "ci90_lower": float(np.quantile(deltas, 0.05)),
        "ci90_upper": float(np.quantile(deltas, 0.95)),
    }


def paired_comparison(
    evaluated: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    control_column: str,
    challenger_column: str,
) -> dict[str, Any]:
    control = metric_slices(evaluated, control_column)
    challenger = metric_slices(evaluated, challenger_column)
    delta = float(challenger["rmse"] - control["rmse"])
    by_fold = {
        key: float(challenger["by_fold"][key] - control["by_fold"][key])
        for key in control["by_fold"]
    }
    by_station = {
        key: float(challenger["by_station"][key] - control["by_station"][key])
        for key in control["by_station"]
    }
    by_lead = {
        key: float(challenger["by_lead"][key] - control["by_lead"][key])
        for key in control["by_lead"]
    }
    validation = config["validation"]
    bootstrap = paired_case_bootstrap(
        evaluated,
        replicates=int(validation["bootstrap_replicates"]),
        seed=int(validation["bootstrap_seed"]),
        control_column=control_column,
        challenger_column=challenger_column,
    )
    return {
        "control_column": control_column,
        "challenger_column": challenger_column,
        "control": control,
        "challenger": challenger,
        "challenger_minus_control_rmse": delta,
        "delta_by_fold": by_fold,
        "delta_by_station": by_station,
        "delta_by_lead": by_lead,
        "paired_case_bootstrap": bootstrap,
    }


def evaluate_promotion(
    evaluated: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    control_column: str = "incumbent_final",
    challenger_column: str = "challenger_final",
) -> dict[str, Any]:
    if (control_column, challenger_column) != ("incumbent_final", "challenger_final"):
        raise KMASourceMetaError(
            "promotion gate may only compare challenger_final to incumbent_final"
        )
    comparison = paired_comparison(
        evaluated,
        config,
        control_column=control_column,
        challenger_column=challenger_column,
    )
    delta = comparison["challenger_minus_control_rmse"]
    by_fold = comparison["delta_by_fold"]
    by_station = comparison["delta_by_station"]
    by_lead = comparison["delta_by_lead"]
    bootstrap = comparison["paired_case_bootstrap"]
    gate = config["promotion_gate"]
    checks = {
        "pooled_rmse_improvement": delta <= -float(gate["minimum_pooled_rmse_improvement_meters"]),
        "bootstrap_ci90_upper_below_zero": bootstrap["ci90_upper"] < 0.0,
        "minimum_improved_outer_folds": sum(value < 0.0 for value in by_fold.values())
        >= int(gate["minimum_improved_outer_folds"]),
        "station_degradation_limit": max(by_station.values())
        <= float(gate["maximum_any_station_rmse_degradation_meters"]),
        "lead_degradation_limit": max(by_lead.values())
        <= float(gate["maximum_any_lead_rmse_degradation_meters"]),
        "lead_18_non_degrading": by_lead["18"] <= 0.0,
        "lead_24_non_degrading": by_lead["24"] <= 0.0,
    }
    return {
        **comparison,
        "decision": "GO_TO_INTEGRATION" if all(checks.values()) else "NO_GO_FINAL_INCUMBENT",
        "checks": checks,
    }


__all__ = [
    "ARRAY_ORDER",
    "DomainRoute",
    "HISTORY_ROWS",
    "KMASourceMetaError",
    "LEADS",
    "META_COLUMNS",
    "PAIR_KEYS",
    "ROUTER_COLUMNS",
    "SourceCases",
    "append_meta_features",
    "build_source_cases",
    "build_target_source_features",
    "canonicalize_kma_observations",
    "catboost_frame",
    "compact_source_feature_columns",
    "evaluate_promotion",
    "evaluate_inner_incremental_signal",
    "paired_comparison",
    "expand_target_rows",
    "expand_prediction_rows",
    "extract_target_common_history",
    "fit_source_median_imputer",
    "apply_source_median_imputer",
    "load_preregistration",
    "metric_slices",
    "paired_case_bootstrap",
    "read_frozen_outer_key_membership",
    "read_frozen_router_components",
    "integrate_frozen_router",
    "resolve_domain_route",
    "sha256_file",
    "source_catboost",
    "source_predictions_to_meta",
    "summarize_common_history",
    "target_catboost",
    "validate_blind_prediction_frame",
    "validate_outer_membership_against_anchors",
    "validate_preregistration",
]
