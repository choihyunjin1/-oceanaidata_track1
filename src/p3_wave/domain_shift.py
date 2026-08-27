"""Fail-closed domain-shift gate for quarantined pre-2024 P3 data.

Only the KMA quarantine Parquet and the two distributed P3 training tables are
accepted.  The module builds a common causal 30-minute representation and
evaluates whether a classifier can distinguish the two domains under
station-year grouped cross-validation.  It never reads test cases, trains a
wave forecast, writes row-level features, or creates a submission.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

KST = "Asia/Seoul"
ALLOWED_P3_TRAIN_FILES = ("train_wave.csv", "train_atmos.csv")
FORBIDDEN_P3_FILES = frozenset(
    {
        "test_context.parquet",
        "test_index.csv",
        "sample_submission.csv",
        "baseline_persistence.csv",
        "answer.csv",
    }
)
COMMON_COLUMNS = (
    "hs",
    "tp",
    "hmax",
    "wvdir",
    "wspd",
    "gust",
    "wdir",
    "caph",
    "airt",
    "relh",
)
SOURCE_COLUMNS = (
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
)
WAVE_COLUMNS = ("station", "time", "hs", "tp", "hmax", "wvdir")
ATMOS_COLUMNS = ("station", "time", "wspd", "gust", "wdir", "airt", "relh", "caph")


class DomainShiftError(RuntimeError):
    """Raised when a fixed safety, schema, or evaluation contract fails."""


@dataclass(frozen=True)
class SampledRepresentation:
    """In-memory row-level representation; this object must never be persisted."""

    features: pd.DataFrame
    domain: np.ndarray
    station_keys: np.ndarray
    group_keys: np.ndarray
    summary: dict[str, Any]


@dataclass(frozen=True)
class ClassifierEvaluation:
    auc: float
    fold_auc: tuple[float, ...]
    fold_summaries: tuple[dict[str, Any], ...]
    probabilities: np.ndarray
    normalized_oof: np.ndarray


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assert_allowed_input(path: Path, *, role: str) -> None:
    """Reject any P3 evaluation or output-like input by basename."""

    name = path.name.casefold()
    forbidden = {item.casefold() for item in FORBIDDEN_P3_FILES}
    if name in forbidden:
        raise DomainShiftError(f"forbidden P3 evaluation input requested for {role}: {path.name}")
    if role == "p3_train" and name not in {item.casefold() for item in ALLOWED_P3_TRAIN_FILES}:
        raise DomainShiftError(f"P3 input is outside the two-file training allowlist: {path.name}")
    lower_parts = {part.casefold() for part in path.parts}
    if lower_parts.intersection({"submissions", "models", "oof", "frozen", "saved_weight"}):
        raise DomainShiftError(
            f"model/OOF/submission path is forbidden for domain gate: {path.name}"
        )


def resolve_p3_train_paths(data_dir: Path) -> dict[str, Path]:
    root = data_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DomainShiftError("P3_DATA_DIR is not a directory")
    paths: dict[str, Path] = {}
    for name in ALLOWED_P3_TRAIN_FILES:
        candidate = (root / name).resolve(strict=True)
        if candidate.parent != root:
            raise DomainShiftError(f"P3 training input escapes P3_DATA_DIR: {name}")
        assert_allowed_input(candidate, role="p3_train")
        paths[name] = candidate
    return paths


def _parse_time_kst(values: pd.Series, *, field_name: str) -> pd.Series:
    raw = values.astype("string")
    has_offset = raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)
    if bool(has_offset.any()) and not bool(has_offset.all()):
        raise DomainShiftError(f"mixed timezone-aware and naive values in {field_name}")
    try:
        if bool(has_offset.all()):
            parsed = pd.to_datetime(raw, errors="raise", utc=True).dt.tz_convert(KST)
        else:
            parsed = pd.to_datetime(raw, errors="raise").dt.tz_localize(
                KST, ambiguous="raise", nonexistent="raise"
            )
    except (TypeError, ValueError) as exc:
        raise DomainShiftError(f"unparseable timestamps in {field_name}") from exc
    return parsed


def _validate_columns(frame: pd.DataFrame, required: Sequence[str], *, table: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise DomainShiftError(f"{table} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: Iterable[str], *, table: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
        finite_or_missing = (
            np.isfinite(result[column].to_numpy(dtype=float, na_value=np.nan))
            | result[column].isna().to_numpy()
        )
        if not bool(finite_or_missing.all()):
            raise DomainShiftError(f"{table}.{column} contains infinity")
    return result


def load_external_canonical(parquet_path: Path, *, maximum_time_kst: str) -> pd.DataFrame:
    """Load only fixed source columns and map them to the common schema."""

    path = parquet_path.expanduser().resolve(strict=True)
    assert_allowed_input(path, role="external")
    frame = pd.read_parquet(path, columns=list(SOURCE_COLUMNS))
    _validate_columns(frame, SOURCE_COLUMNS, table="KMA external")
    if frame.empty:
        raise DomainShiftError("KMA external table is empty")
    time_values = frame["TM"]
    if isinstance(time_values.dtype, pd.DatetimeTZDtype):
        time_kst = time_values.dt.tz_convert(KST)
    else:
        time_kst = _parse_time_kst(time_values, field_name="KMA.TM")
    cutoff = pd.Timestamp(maximum_time_kst).tz_convert(KST)
    if bool((time_kst > cutoff).any()):
        raise DomainShiftError("KMA source contains an observation after the pre-2024 cutoff")
    if bool(frame.duplicated(["STN", "TM"]).any()):
        raise DomainShiftError("KMA source has duplicate station-time keys")

    numeric_columns = [column for column in SOURCE_COLUMNS if column not in {"TM", "STN"}]
    frame = _numeric(frame, numeric_columns, table="KMA external")
    canonical = pd.DataFrame(
        {
            "station": "source:" + frame["STN"].astype("string"),
            "time": time_kst,
            "hs": frame["WH_SIG"],
            "tp": frame["WP"],
            "hmax": frame["WH_MAX"],
            "wvdir": frame["WO"],
            "wspd": frame["WS1"].combine_first(frame["WS2"]),
            "gust": frame["WS1_GST"].combine_first(frame["WS2_GST"]),
            "wdir": frame["WD1"].combine_first(frame["WD2"]),
            "caph": frame["PA"],
            "airt": frame["TA"],
            "relh": frame["HM"],
        }
    )
    return canonical.sort_values(["station", "time"], kind="stable").reset_index(drop=True)


def load_target_train_canonical(train_paths: Mapping[str, Path]) -> pd.DataFrame:
    """Load exactly the two distributed P3 train tables; no test path is accepted."""

    if set(train_paths) != set(ALLOWED_P3_TRAIN_FILES):
        raise DomainShiftError(
            "P3 training mapping must contain exactly train_wave.csv and train_atmos.csv"
        )
    wave_path = train_paths["train_wave.csv"]
    atmos_path = train_paths["train_atmos.csv"]
    assert_allowed_input(wave_path, role="p3_train")
    assert_allowed_input(atmos_path, role="p3_train")
    wave = pd.read_csv(wave_path, usecols=list(WAVE_COLUMNS))
    atmos = pd.read_csv(atmos_path, usecols=list(ATMOS_COLUMNS))
    _validate_columns(wave, WAVE_COLUMNS, table="train_wave")
    _validate_columns(atmos, ATMOS_COLUMNS, table="train_atmos")
    wave["time"] = _parse_time_kst(wave["time"], field_name="train_wave.time")
    atmos["time"] = _parse_time_kst(atmos["time"], field_name="train_atmos.time")
    if bool(wave.duplicated(["station", "time"]).any()):
        raise DomainShiftError("train_wave has duplicate station-time keys")
    if bool(atmos.duplicated(["station", "time"]).any()):
        raise DomainShiftError("train_atmos has duplicate station-time keys")
    wave = _numeric(wave, WAVE_COLUMNS[2:], table="train_wave")
    atmos = _numeric(atmos, ATMOS_COLUMNS[2:], table="train_atmos")
    merged = wave.merge(atmos, on=["station", "time"], how="outer", validate="one_to_one")
    merged["station"] = "target:" + merged["station"].astype("string")
    if merged.empty:
        raise DomainShiftError("P3 training tables are empty")
    return (
        merged.loc[:, ["station", "time", *COMMON_COLUMNS]]
        .sort_values(["station", "time"], kind="stable")
        .reset_index(drop=True)
    )


def _causal_resample(group: pd.DataFrame, *, grid_minutes: int) -> pd.DataFrame:
    if grid_minutes <= 0:
        raise DomainShiftError("grid_minutes must be positive")
    ordered = group.sort_values("time", kind="stable")
    indexed = ordered.set_index("time").loc[:, list(COMMON_COLUMNS)]
    if indexed.index.has_duplicates:
        raise DomainShiftError("duplicate timestamps within station-year group")
    grid = indexed.resample(
        f"{grid_minutes}min", label="right", closed="right", origin="start_day"
    ).last()
    return grid.astype(float)


def _peak_recency(values: np.ndarray, *, window_size: int, step_hours: float) -> np.ndarray:
    """Hours since the most recent rolling maximum, in O(n) causal time."""

    array = np.asarray(values, dtype=float)
    result = np.full(len(array), np.nan, dtype=np.float32)
    candidates: deque[int] = deque()
    for index, value in enumerate(array):
        left = index - window_size + 1
        while candidates and candidates[0] < left:
            candidates.popleft()
        if np.isfinite(value):
            while candidates and array[candidates[-1]] <= value:
                candidates.pop()
            candidates.append(index)
        if candidates:
            result[index] = np.float32((index - candidates[0]) * step_hours)
    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, floor: float) -> pd.Series:
    valid = denominator.abs() >= floor
    return numerator.where(valid) / denominator.where(valid)


def build_causal_features(grid: pd.DataFrame, representation: Mapping[str, Any]) -> pd.DataFrame:
    """Build the pre-registered common features using current and past rows only."""

    _validate_columns(grid, COMMON_COLUMNS, table="common 30-minute grid")
    grid_minutes = int(representation["grid_minutes"])
    windows = tuple(int(value) for value in representation["history_windows_hours"])
    if windows != (3, 6, 12, 24, 48):
        raise DomainShiftError("history windows must remain fixed at 3/6/12/24/48 hours")
    step_hours = grid_minutes / 60.0
    minimum_fraction = float(representation["rolling_minimum_fraction"])
    scalar_variables = tuple(representation["scalar_variables"])
    direction_variables = tuple(representation["direction_variables"])
    peak_variables = frozenset(representation["peak_recency_variables"])
    floors = representation["ratio_denominator_floors"]
    features: dict[str, np.ndarray] = {}

    for variable in scalar_variables:
        series = grid[variable].astype(float)
        features[f"{variable}__current"] = series.to_numpy(dtype=np.float32)
        features[f"{variable}__missing_current"] = series.isna().to_numpy(dtype=np.float32)
        for hours in windows:
            steps = int(round(hours / step_hours))
            window_size = steps + 1
            minimum_periods = max(2, int(np.ceil(window_size * minimum_fraction)))
            rolling = series.rolling(window_size, min_periods=minimum_periods)
            features[f"{variable}__delta_{hours}h"] = (series - series.shift(steps)).to_numpy(
                dtype=np.float32
            )
            features[f"{variable}__var_{hours}h"] = rolling.var(ddof=0).to_numpy(dtype=np.float32)
            features[f"{variable}__max_{hours}h"] = rolling.max().to_numpy(dtype=np.float32)
            observed = series.notna().rolling(window_size, min_periods=1).mean()
            features[f"{variable}__missing_frac_{hours}h"] = (1.0 - observed).to_numpy(
                dtype=np.float32
            )
            if variable in peak_variables:
                features[f"{variable}__peak_recency_{hours}h"] = _peak_recency(
                    series.to_numpy(dtype=float), window_size=window_size, step_hours=step_hours
                )

    direction_radians: dict[str, pd.Series] = {}
    for variable in direction_variables:
        radians = np.deg2rad(grid[variable].astype(float))
        direction_radians[variable] = radians
        sin_value = np.sin(radians)
        cos_value = np.cos(radians)
        features[f"{variable}__sin_current"] = sin_value.to_numpy(dtype=np.float32)
        features[f"{variable}__cos_current"] = cos_value.to_numpy(dtype=np.float32)
        features[f"{variable}__missing_current"] = radians.isna().to_numpy(dtype=np.float32)
        for hours in windows:
            steps = int(round(hours / step_hours))
            window_size = steps + 1
            minimum_periods = max(2, int(np.ceil(window_size * minimum_fraction)))
            lagged = radians.shift(steps)
            delta = radians - lagged
            features[f"{variable}__delta_sin_{hours}h"] = np.sin(delta).to_numpy(dtype=np.float32)
            features[f"{variable}__delta_cos_{hours}h"] = np.cos(delta).to_numpy(dtype=np.float32)
            mean_sin = sin_value.rolling(window_size, min_periods=minimum_periods).mean()
            mean_cos = cos_value.rolling(window_size, min_periods=minimum_periods).mean()
            concentration = np.sqrt(mean_sin.pow(2) + mean_cos.pow(2))
            features[f"{variable}__circular_var_{hours}h"] = (1.0 - concentration).to_numpy(
                dtype=np.float32
            )
            observed = radians.notna().rolling(window_size, min_periods=1).mean()
            features[f"{variable}__missing_frac_{hours}h"] = (1.0 - observed).to_numpy(
                dtype=np.float32
            )

    wind_radians = direction_radians["wdir"]
    wave_radians = direction_radians["wvdir"]
    derived = {
        "wind_u": grid["wspd"] * np.cos(wind_radians),
        "wind_v": grid["wspd"] * np.sin(wind_radians),
        "wind_wave_alignment": np.cos(wind_radians - wave_radians),
        "gust_wspd_ratio": _safe_ratio(grid["gust"], grid["wspd"], float(floors["wspd"])),
        "hmax_hs_ratio": _safe_ratio(grid["hmax"], grid["hs"], float(floors["hs"])),
        "wave_energy_proxy": grid["hs"].pow(2),
        "hs_tp2_proxy": _safe_ratio(grid["hs"], grid["tp"].pow(2), float(floors["tp_squared"])),
    }
    for name, series in derived.items():
        series = pd.Series(series, index=grid.index, dtype=float)
        features[f"{name}__current"] = series.to_numpy(dtype=np.float32)
        features[f"{name}__missing_current"] = series.isna().to_numpy(dtype=np.float32)
        for hours in windows:
            steps = int(round(hours / step_hours))
            window_size = steps + 1
            minimum_periods = max(2, int(np.ceil(window_size * minimum_fraction)))
            features[f"{name}__delta_{hours}h"] = (series - series.shift(steps)).to_numpy(
                dtype=np.float32
            )
            features[f"{name}__var_{hours}h"] = (
                series.rolling(window_size, min_periods=minimum_periods)
                .var(ddof=0)
                .to_numpy(dtype=np.float32)
            )

    result = pd.DataFrame(features, index=grid.index, dtype=np.float32)
    values = result.to_numpy(dtype=np.float32, copy=False)
    if bool(np.isinf(values).any()):
        raise DomainShiftError("engineered representation contains infinity")
    if result.columns.duplicated().any():
        raise DomainShiftError("engineered representation has duplicate feature names")
    return result


def _stable_sample_indices(length: int, maximum: int, *, seed: int, group_key: str) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    group_seed = int.from_bytes(sha256(f"{seed}|{group_key}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(group_seed)
    return np.sort(rng.choice(length, size=maximum, replace=False)).astype(np.int64)


def build_sampled_representation(
    source: pd.DataFrame,
    target: pd.DataFrame,
    representation: Mapping[str, Any],
    *,
    seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SampledRepresentation:
    """Build and deterministically sample station-year groups in memory."""

    grid_minutes = int(representation["grid_minutes"])
    minimum_history_hours = int(representation["minimum_history_hours"])
    warmup_rows = int(round(minimum_history_hours * 60 / grid_minutes))
    maximum_rows = int(representation["maximum_rows_per_station_year"])
    if warmup_rows != 96 or maximum_rows <= 0:
        raise DomainShiftError("fixed 48-hour warmup or sampling contract changed")

    feature_parts: list[pd.DataFrame] = []
    domain_parts: list[np.ndarray] = []
    station_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    group_counts = {"source": 0, "target": 0}
    station_sets: dict[str, set[str]] = {"source": set(), "target": set()}
    year_sets: dict[str, set[int]] = {"source": set(), "target": set()}

    domain_frames = (("source", 0, source), ("target", 1, target))
    total_groups = sum(
        int(frame.assign(year=frame["time"].dt.year).groupby(["station", "year"]).ngroups)
        for _, _, frame in domain_frames
    )
    completed_groups = 0
    for domain_name, domain_label, frame in domain_frames:
        working = frame.copy()
        working["year"] = working["time"].dt.year.astype(int)
        for (station, year), group in working.groupby(["station", "year"], sort=True):
            station_text = str(station)
            group_key = f"{domain_name}|{station_text}|{int(year)}"
            grid = _causal_resample(group, grid_minutes=grid_minutes)
            if len(grid) <= warmup_rows:
                completed_groups += 1
                if progress_callback is not None:
                    progress_callback(completed_groups, total_groups)
                continue
            features = build_causal_features(grid, representation).iloc[warmup_rows:]
            chosen = _stable_sample_indices(
                len(features), maximum_rows, seed=seed, group_key=group_key
            )
            selected = features.iloc[chosen].reset_index(drop=True)
            if selected.empty:
                continue
            feature_parts.append(selected)
            domain_parts.append(np.full(len(selected), domain_label, dtype=np.uint8))
            station_key = f"{domain_name}|{station_text}"
            station_parts.append(np.full(len(selected), station_key, dtype=object))
            group_parts.append(np.full(len(selected), group_key, dtype=object))
            group_counts[domain_name] += 1
            station_sets[domain_name].add(station_text)
            year_sets[domain_name].add(int(year))
            completed_groups += 1
            if progress_callback is not None:
                progress_callback(completed_groups, total_groups)

    if not feature_parts:
        raise DomainShiftError("no sampled feature groups were produced")
    feature_columns = feature_parts[0].columns
    if any(not part.columns.equals(feature_columns) for part in feature_parts[1:]):
        raise DomainShiftError("feature schema changed across station-year groups")
    combined = pd.concat(feature_parts, ignore_index=True)
    labels = np.concatenate(domain_parts)
    stations = np.concatenate(station_parts)
    groups = np.concatenate(group_parts)
    if set(np.unique(labels)) != {0, 1}:
        raise DomainShiftError("both source and target domains are required")
    if len(np.unique(groups[labels == 1])) < 5:
        raise DomainShiftError("fewer than five target station-year groups remain")
    summary = {
        "sample_rows": {
            "source": int((labels == 0).sum()),
            "target": int((labels == 1).sum()),
            "total": int(len(labels)),
        },
        "station_year_group_counts": group_counts,
        "station_counts": {key: len(value) for key, value in station_sets.items()},
        "year_counts": {key: len(value) for key, value in year_sets.items()},
        "feature_count": int(combined.shape[1]),
        "sampling": "deterministic_sha256_seeded_without_replacement_per_station_year",
        "warmup_rows_excluded_per_group": warmup_rows,
    }
    return SampledRepresentation(combined, labels, stations, groups, summary)


class StationRobustNormalizer:
    """Fold-train-only station median/IQR normalization and median imputation."""

    def __init__(self, *, iqr_floor: float, clip: float) -> None:
        self.iqr_floor = float(iqr_floor)
        self.clip = float(clip)
        self.global_center_: np.ndarray | None = None
        self.global_scale_: np.ndarray | None = None
        self.station_stats_: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _stats(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            center = np.nanmedian(values, axis=0)
            q25 = np.nanpercentile(values, 25.0, axis=0)
            q75 = np.nanpercentile(values, 75.0, axis=0)
        scale = q75 - q25
        return center, scale

    def fit(self, frame: pd.DataFrame, station_keys: np.ndarray) -> StationRobustNormalizer:
        values = frame.to_numpy(dtype=np.float64, copy=True)
        global_center, global_scale = self._stats(values)
        global_center = np.where(np.isfinite(global_center), global_center, 0.0)
        global_scale = np.where(
            np.isfinite(global_scale) & (global_scale > self.iqr_floor), global_scale, 1.0
        )
        self.global_center_ = global_center
        self.global_scale_ = global_scale
        self.station_stats_ = {}
        for station in np.unique(station_keys):
            center, scale = self._stats(values[station_keys == station])
            center = np.where(np.isfinite(center), center, global_center)
            scale = np.where(np.isfinite(scale) & (scale > self.iqr_floor), scale, global_scale)
            self.station_stats_[str(station)] = (center, scale)
        return self

    def transform(self, frame: pd.DataFrame, station_keys: np.ndarray) -> np.ndarray:
        if self.global_center_ is None or self.global_scale_ is None:
            raise DomainShiftError("StationRobustNormalizer must be fit before transform")
        values = frame.to_numpy(dtype=np.float64, copy=True)
        transformed = np.empty_like(values, dtype=np.float32)
        for station in np.unique(station_keys):
            mask = station_keys == station
            center, scale = self.station_stats_.get(
                str(station), (self.global_center_, self.global_scale_)
            )
            block = (values[mask] - center) / scale
            block = np.where(np.isfinite(block), block, 0.0)
            transformed[mask] = np.clip(block, -self.clip, self.clip).astype(np.float32)
        return transformed


def evaluate_domain_classifier(
    sampled: SampledRepresentation,
    classifier_config: Mapping[str, Any],
    representation_config: Mapping[str, Any],
    *,
    seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ClassifierEvaluation:
    labels = sampled.domain.astype(int)
    groups = sampled.group_keys
    stations = sampled.station_keys
    n_splits = int(classifier_config["n_splits"])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probability = np.full(len(labels), np.nan, dtype=np.float64)
    normalized_oof = np.full(sampled.features.shape, np.nan, dtype=np.float32)
    fold_auc: list[float] = []
    fold_summaries: list[dict[str, Any]] = []
    seen_validation_groups: set[str] = set()

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(sampled.features, labels, groups), start=1
    ):
        train_groups = set(groups[train_index])
        validation_groups = set(groups[validation_index])
        if train_groups.intersection(validation_groups):
            raise DomainShiftError("station-year group leakage detected")
        if seen_validation_groups.intersection(validation_groups):
            raise DomainShiftError("station-year group evaluated more than once")
        seen_validation_groups.update(validation_groups)
        if set(labels[train_index]) != {0, 1} or set(labels[validation_index]) != {0, 1}:
            raise DomainShiftError("each grouped fold must contain both domains")

        normalizer = StationRobustNormalizer(
            iqr_floor=float(representation_config["iqr_floor"]),
            clip=float(representation_config["normalized_clip"]),
        ).fit(sampled.features.iloc[train_index], stations[train_index])
        train_values = normalizer.transform(
            sampled.features.iloc[train_index], stations[train_index]
        )
        validation_values = normalizer.transform(
            sampled.features.iloc[validation_index], stations[validation_index]
        )
        train_labels = labels[train_index]
        class_counts = np.bincount(train_labels, minlength=2).astype(float)
        sample_weight = len(train_labels) / (2.0 * class_counts[train_labels])
        model = HistGradientBoostingClassifier(
            learning_rate=float(classifier_config["learning_rate"]),
            max_iter=int(classifier_config["max_iter"]),
            max_leaf_nodes=int(classifier_config["max_leaf_nodes"]),
            min_samples_leaf=int(classifier_config["min_samples_leaf"]),
            l2_regularization=float(classifier_config["l2_regularization"]),
            max_bins=int(classifier_config["max_bins"]),
            max_features=float(classifier_config["max_features"]),
            early_stopping=False,
            random_state=seed + fold,
        )
        model.fit(train_values, train_labels, sample_weight=sample_weight)
        probability = model.predict_proba(validation_values)[:, 1]
        oof_probability[validation_index] = probability
        normalized_oof[validation_index] = validation_values
        score = float(roc_auc_score(labels[validation_index], probability))
        fold_auc.append(score)
        fold_summaries.append(
            {
                "fold": fold,
                "auc": score,
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "train_group_count": len(train_groups),
                "validation_group_count": len(validation_groups),
                "validation_source_rows": int((labels[validation_index] == 0).sum()),
                "validation_target_rows": int((labels[validation_index] == 1).sum()),
                "group_overlap_count": 0,
            }
        )
        if progress_callback is not None:
            progress_callback(fold, n_splits)

    if not bool(np.isfinite(oof_probability).all()):
        raise DomainShiftError("OOF probability coverage is incomplete")
    if not bool(np.isfinite(normalized_oof).all()):
        raise DomainShiftError("OOF normalized feature coverage is incomplete")
    if seen_validation_groups != set(groups):
        raise DomainShiftError("not every station-year group was evaluated exactly once")
    overall_auc = float(roc_auc_score(labels, oof_probability))
    return ClassifierEvaluation(
        auc=overall_auc,
        fold_auc=tuple(fold_auc),
        fold_summaries=tuple(fold_summaries),
        probabilities=oof_probability,
        normalized_oof=normalized_oof,
    )


def gate_decision(auc: float, gate_config: Mapping[str, Any]) -> dict[str, Any]:
    full_max = float(gate_config["full_ablation_max_auc"])
    pretrain_max = float(gate_config["pretrain_only_max_auc"])
    if not 0.5 <= full_max < pretrain_max < 1.0:
        raise DomainShiftError("invalid pre-registered AUC thresholds")
    if not np.isfinite(auc):
        raise DomainShiftError("domain classifier AUC is not finite")
    if auc <= full_max:
        return {
            "tier": "full_ablation_allowed",
            "source_concat_or_full_finetune_allowed": True,
            "pretrain_only_challenger_allowed": True,
            "reason": "auc_at_or_below_full_ablation_threshold",
        }
    if auc <= pretrain_max:
        return {
            "tier": "pretrain_only_challenger",
            "source_concat_or_full_finetune_allowed": False,
            "pretrain_only_challenger_allowed": True,
            "reason": "auc_above_full_threshold_at_or_below_pretrain_threshold",
        }
    return {
        "tier": "no_go_source_concat_or_full_finetune",
        "source_concat_or_full_finetune_allowed": False,
        "pretrain_only_challenger_allowed": False,
        "reason": "auc_above_pretrain_only_threshold",
    }


def density_ratio_summary(
    labels: np.ndarray, probabilities: np.ndarray, gate_config: Mapping[str, Any]
) -> dict[str, Any]:
    probability_clip = float(gate_config["density_probability_clip"])
    ratio_clip = float(gate_config["density_ratio_clip"])
    source_probability = np.clip(probabilities[labels == 0], probability_clip, 1 - probability_clip)
    weights = source_probability / (1.0 - source_probability)
    weights = np.clip(weights, 0.0, ratio_clip)
    ess = float(weights.sum() ** 2 / np.square(weights).sum()) if np.square(weights).sum() else 0.0
    quantiles = np.quantile(weights, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "source_row_count": int(len(weights)),
        "effective_sample_size": ess,
        "effective_sample_fraction": ess / len(weights) if len(weights) else 0.0,
        "probability_clip": probability_clip,
        "ratio_clip": ratio_clip,
        "weight_quantiles": {
            key: float(value)
            for key, value in zip(
                ("min", "p50", "p90", "p95", "p99", "max"), quantiles, strict=True
            )
        },
    }


def standardized_drift_summary(
    sampled: SampledRepresentation,
    normalized_oof: np.ndarray,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    labels = sampled.domain
    source = normalized_oof[labels == 0].astype(np.float64)
    target = normalized_oof[labels == 1].astype(np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    pooled_scale = np.sqrt((source.var(axis=0) + target.var(axis=0)) / 2.0)
    smd = np.divide(
        target_mean - source_mean,
        pooled_scale,
        out=np.zeros_like(source_mean),
        where=pooled_scale > 1e-12,
    )
    raw_missing = sampled.features.isna().to_numpy()
    missing_delta = raw_missing[labels == 1].mean(axis=0) - raw_missing[labels == 0].mean(axis=0)
    names = np.asarray(sampled.features.columns, dtype=object)
    top_smd_index = np.argsort(-np.abs(smd))[:top_n]
    top_missing_index = np.argsort(-np.abs(missing_delta))[:top_n]
    absolute_smd = np.abs(smd)
    return {
        "normalization": "out_of_fold_stationwise_train_only_median_iqr",
        "absolute_smd_quantiles": {
            key: float(value)
            for key, value in zip(
                ("p50", "p75", "p90", "p95", "p99", "max"),
                np.quantile(absolute_smd, [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
                strict=True,
            )
        },
        "top_features_by_absolute_smd": [
            {
                "feature": str(names[index]),
                "standardized_mean_difference": float(smd[index]),
                "absolute_standardized_mean_difference": float(abs(smd[index])),
            }
            for index in top_smd_index
        ],
        "top_features_by_missing_rate_difference": [
            {
                "feature": str(names[index]),
                "target_minus_source_missing_rate": float(missing_delta[index]),
            }
            for index in top_missing_index
        ],
    }


def validate_external_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("source_id") != "kma_ocean_buoy_pre2024":
        raise DomainShiftError("unexpected KMA external source_id")
    precheck = manifest.get("precheck")
    if not isinstance(precheck, Mapping) or precheck.get("accepted") is not True:
        raise DomainShiftError("KMA source precheck has not passed")
    if precheck.get("domain_shift_local_comparison") not in {"pending", "passed", "completed"}:
        raise DomainShiftError("KMA manifest has an invalid domain-shift state")


def aggregate_input_receipts(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    receipts = []
    for role, path in sorted(paths.items()):
        receipts.append(
            {
                "role": role,
                "basename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return receipts
