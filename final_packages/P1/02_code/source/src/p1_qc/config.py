"""Configuration loading for the P1 quality-control pipeline.

The loader intentionally has no third-party dependency: Python's ``tomllib``
parses the optional project file and environment variables provide small,
auditable overrides.  Unknown TOML keys are retained in ``P1QCConfig.raw`` so
an experiment record never silently loses configuration supplied by a caller.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path | None = None
    artifacts_dir: Path = Path("artifacts")


@dataclass(frozen=True)
class DataConfig:
    cadence_minutes: int = 10
    timezone: str = "Asia/Seoul"
    key_columns: tuple[str, ...] = ("station", "year", "layer", "time")
    group_columns: tuple[str, ...] = ("station", "layer")
    anomaly_types: tuple[str, ...] = (
        "spike",
        "noise",
        "flatline",
        "offset",
        "drift",
    )


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = "offline"
    rolling_hours: tuple[int, ...] = (3, 6, 12, 24, 72)
    long_windows_days: tuple[int, ...] = (7, 14)
    min_period_fraction: float = 0.25
    depth_regime_width_m: float = 2.5
    plateau_round_decimals: int | None = None
    robust_epsilon: float = 1.0e-4


@dataclass(frozen=True)
class FoldWindowConfig:
    name: str
    train_end: str
    val_start: str
    val_end: str


def _default_fold_windows() -> tuple[FoldWindowConfig, ...]:
    return (
        FoldWindowConfig(
            "2025_q2",
            "2025-03-24T23:50:00+09:00",
            "2025-04-01T00:00:00+09:00",
            "2025-07-01T00:00:00+09:00",
        ),
        FoldWindowConfig(
            "2025_q3",
            "2025-06-23T23:50:00+09:00",
            "2025-07-01T00:00:00+09:00",
            "2025-10-01T00:00:00+09:00",
        ),
        FoldWindowConfig(
            "2025_q4",
            "2025-09-23T23:50:00+09:00",
            "2025-10-01T00:00:00+09:00",
            "2025-12-11T00:00:00+09:00",
        ),
    )


@dataclass(frozen=True)
class SplitConfig:
    purge_days: int = 7
    protect_positive_runs: bool = True
    folds: tuple[FoldWindowConfig, ...] = field(default_factory=_default_fold_windows)


@dataclass(frozen=True)
class MetricsConfig:
    group_columns: tuple[str, ...] = ("station", "layer")
    event_cadence_minutes: int = 10
    event_min_iou: float = 0.0


@dataclass(frozen=True)
class P1QCConfig:
    seed: int = 20260813
    mode: str = "offline"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable effective mapping, retaining unknown keys."""

        result = deepcopy(dict(self.raw))
        known = asdict(self)
        known.pop("raw", None)
        known["paths"] = {
            "data_dir": None if self.paths.data_dir is None else str(self.paths.data_dir),
            "artifacts_dir": str(self.paths.artifacts_dir),
        }
        known["splits"]["folds"] = [asdict(item) for item in self.splits.folds]
        _deep_merge(result, known)
        return result


def _defaults_mapping() -> dict[str, Any]:
    config = P1QCConfig()
    data = asdict(config)
    data.pop("raw", None)
    data["paths"] = {
        "data_dir": None,
        "artifacts_dir": str(config.paths.artifacts_dir),
    }
    data["splits"]["folds"] = [asdict(item) for item in config.splits.folds]
    return data


def _deep_merge(
    target: MutableMapping[str, Any], incoming: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_merge(target[key], value)  # type: ignore[index]
        else:
            target[key] = deepcopy(value)
    return target


def _parse_env_value(value: str) -> Any:
    try:
        return tomllib.loads(f"value = {value}")["value"]
    except tomllib.TOMLDecodeError:
        return value


def _environment_mapping(env: Mapping[str, str], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    aliases = {
        "DATA_DIR": ("paths", "data_dir"),
        "ARTIFACTS_DIR": ("paths", "artifacts_dir"),
        "SEED": ("seed",),
        "MODE": ("mode",),
    }
    for key, raw_value in env.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix:
            continue
        path = aliases.get(suffix)
        if path is None:
            # Double underscores are the unambiguous nested-key delimiter.
            path = tuple(part.lower() for part in suffix.split("__") if part)
        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _parse_env_value(raw_value)
    return result


def _normalise_legacy_sections(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Map the implementation-guide TOML vocabulary to canonical sections.

    Original sections remain in the effective raw mapping for provenance.
    """

    result = deepcopy(dict(parsed))
    project = parsed.get("project", {})
    if isinstance(project, Mapping):
        for key in ("seed", "mode"):
            if key in project and key not in parsed:
                result[key] = project[key]
        paths = result.setdefault("paths", {})
        if isinstance(paths, MutableMapping) and "artifacts_dir" in project:
            paths.setdefault("artifacts_dir", project["artifacts_dir"])
    validation = parsed.get("validation")
    if isinstance(validation, Mapping) and "splits" not in parsed:
        result["splits"] = deepcopy(dict(validation))
    features = result.get("features")
    if isinstance(features, MutableMapping):
        if "windows_hours" in features and "rolling_hours" not in features:
            features["rolling_hours"] = deepcopy(features["windows_hours"])
        if "depth_regime_tolerance_m" in features and "depth_regime_width_m" not in features:
            features["depth_regime_width_m"] = features["depth_regime_tolerance_m"]
    return result


def _tuple_of(values: Any, cast: type = str) -> tuple[Any, ...]:
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    if not isinstance(values, Sequence):
        raise TypeError(f"expected a sequence, got {type(values).__name__}")
    return tuple(cast(value) for value in values)


def _folds_from_mapping(values: Any) -> tuple[FoldWindowConfig, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("splits.folds must be an array of tables")
    folds: list[FoldWindowConfig] = []
    for item in values:
        if not isinstance(item, Mapping):
            raise TypeError("each splits.folds item must be a mapping")
        folds.append(
            FoldWindowConfig(
                name=str(item["name"]),
                train_end=str(item["train_end"]),
                val_start=str(item["val_start"]),
                val_end=str(item["val_end"]),
            )
        )
    return tuple(folds)


def _build_config(raw: Mapping[str, Any]) -> P1QCConfig:
    paths = raw.get("paths", {})
    data = raw.get("data", {})
    features = raw.get("features", {})
    splits = raw.get("splits", {})
    metrics = raw.get("metrics", {})

    mode = str(raw.get("mode", features.get("mode", "offline"))).lower()
    if mode not in {"offline", "causal"}:
        raise ValueError("mode must be 'offline' or 'causal'")
    feature_mode = str(features.get("mode", mode)).lower()
    if feature_mode not in {"offline", "causal"}:
        raise ValueError("features.mode must be 'offline' or 'causal'")

    data_dir = paths.get("data_dir")
    path_config = PathsConfig(
        data_dir=None if data_dir in {None, ""} else Path(str(data_dir)).expanduser(),
        artifacts_dir=Path(str(paths.get("artifacts_dir", "artifacts"))).expanduser(),
    )
    data_config = DataConfig(
        cadence_minutes=int(data.get("cadence_minutes", 10)),
        timezone=str(data.get("timezone", "Asia/Seoul")),
        key_columns=_tuple_of(data.get("key_columns", DataConfig().key_columns)),
        group_columns=_tuple_of(data.get("group_columns", DataConfig().group_columns)),
        anomaly_types=_tuple_of(data.get("anomaly_types", DataConfig().anomaly_types)),
    )
    feature_config = FeatureConfig(
        mode=feature_mode,
        rolling_hours=_tuple_of(features.get("rolling_hours", (3, 6, 12, 24, 72)), int),
        long_windows_days=_tuple_of(features.get("long_windows_days", (7, 14)), int),
        min_period_fraction=float(features.get("min_period_fraction", 0.25)),
        depth_regime_width_m=float(features.get("depth_regime_width_m", 2.5)),
        plateau_round_decimals=(
            None
            if features.get("plateau_round_decimals") is None
            else int(features["plateau_round_decimals"])
        ),
        robust_epsilon=float(features.get("robust_epsilon", 1.0e-4)),
    )
    split_config = SplitConfig(
        purge_days=int(splits.get("purge_days", 7)),
        protect_positive_runs=bool(splits.get("protect_positive_runs", True)),
        folds=_folds_from_mapping(
            splits.get("folds", [asdict(x) for x in _default_fold_windows()])
        ),
    )
    metrics_config = MetricsConfig(
        group_columns=_tuple_of(metrics.get("group_columns", ("station", "layer"))),
        event_cadence_minutes=int(metrics.get("event_cadence_minutes", 10)),
        event_min_iou=float(metrics.get("event_min_iou", 0.0)),
    )
    if data_config.cadence_minutes <= 0:
        raise ValueError("data.cadence_minutes must be positive")
    if not 0 < feature_config.min_period_fraction <= 1:
        raise ValueError("features.min_period_fraction must be in (0, 1]")
    if not 0 <= metrics_config.event_min_iou <= 1:
        raise ValueError("metrics.event_min_iou must be in [0, 1]")

    return P1QCConfig(
        seed=int(raw.get("seed", 20260813)),
        mode=mode,
        paths=path_config,
        data=data_config,
        features=feature_config,
        splits=split_config,
        metrics=metrics_config,
        raw=deepcopy(dict(raw)),
    )


def load_config(
    path: str | Path | None = None,
    *,
    env_prefix: str = "P1QC_",
    env: Mapping[str, str] | None = None,
) -> P1QCConfig:
    """Load defaults, an optional TOML file, then environment overrides.

    Nested environment keys use double underscores, for example
    ``P1QC_FEATURES__MODE=\"causal\"``.  ``P1QC_DATA_DIR``,
    ``P1QC_ARTIFACTS_DIR``, ``P1QC_SEED`` and ``P1QC_MODE`` are convenience
    aliases.
    """

    effective = _defaults_mapping()
    config_base: Path | None = None
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        config_base = (
            config_path.parent.parent
            if config_path.parent.name == "configs"
            else config_path.parent
        )
        with config_path.open("rb") as handle:
            parsed = tomllib.load(handle)
        _deep_merge(effective, _normalise_legacy_sections(parsed))
    environment = os.environ if env is None else env
    _deep_merge(effective, _environment_mapping(environment, env_prefix))
    data_section = effective.get("data", {})
    paths_section = effective.setdefault("paths", {})
    if isinstance(data_section, Mapping) and isinstance(paths_section, MutableMapping):
        if paths_section.get("data_dir") in {None, ""}:
            configured_env = data_section.get("env_var")
            if configured_env and environment.get(str(configured_env)):
                paths_section["data_dir"] = environment[str(configured_env)]
            elif config_base is not None and data_section.get("relative_dir"):
                paths_section["data_dir"] = str(
                    (config_base / str(data_section["relative_dir"])).resolve()
                )
    return _build_config(effective)


__all__ = [
    "DataConfig",
    "FeatureConfig",
    "FoldWindowConfig",
    "MetricsConfig",
    "P1QCConfig",
    "PathsConfig",
    "SplitConfig",
    "load_config",
]
