"""Run one sealed P3 ERA5 wave-directional-memory experiment.

Only pre-2024 ERA5 and the official local training files are reachable.  The
runner cannot read test context/index/sample/submission files or create a CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.corrected_repeated_forward import (  # noqa: E402
    build_corrected_repeated_forward_folds,
)
from p3_wave.era5_context_transfer import (  # noqa: E402
    LEADS,
    LOCAL_CATBOOST_PARAMETERS,
    SOURCE_CATBOOST_PARAMETERS,
    canonicalize_era5_hourly,
    common_feature_columns,
    select_source_year_validation,
    summarize_past_48h,
)
from p3_wave.revin_patch import assign_storm_episodes_from_wave  # noqa: E402
from p3_wave.wave_directional_energy_memory import (  # noqa: E402
    DIRECTIONAL_FEATURES,
    VALUE_FEATURES,
    DirectionalContextTransferRegressor,
    apply_directional_increment,
    apply_frozen_persistence_shrink,
    build_local_directional_features,
    summarize_directional_energy_memory,
)

EXPERIMENT_ID = "p3_era5_wave_directional_energy_memory_20260828_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = Path(__file__).resolve()
MODULE = ROOT / "src" / "p3_wave" / "wave_directional_energy_memory.py"
TEST = ROOT / "tests" / "test_p3_wave_directional_energy_memory_20260828_v1.py"
SOURCE_TRAIN_YEARS = tuple(range(2014, 2021))
SOURCE_HELD_YEARS = (2021, 2022, 2023)
TARGET_COLUMNS = tuple(f"target_{lead}" for lead in LEADS)
KEY_COLUMNS = ("fold", "anchor_id", "station", "lead_h")


class ContractError(RuntimeError):
    """Raised when the preregistered one-shot contract changes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    return sha256_file(path)


def _create_attempt_lock(path: Path, config_sha: str) -> str:
    payload = {
        "schema_version": "p3.directional_memory.attempt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_sha,
        "pid": os.getpid(),
        "maximum_executions": 1,
        "official_access": False,
        "upload_allowed": False,
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _load_frozen_runner() -> Any:
    path = ROOT / "scripts" / "run_p3_era5_context_transfer_v1.py"
    spec = importlib.util.spec_from_file_location("_p3_directional_frozen", path)
    if spec is None or spec.loader is None:
        raise ContractError("could not load frozen ERA5 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_data_dir(argument: Path | None) -> Path:
    value = argument or (Path(os.environ["P3_DATA_DIR"]) if "P3_DATA_DIR" in os.environ else None)
    if value is None:
        raise ContractError("P3_DATA_DIR or --data-dir is required")
    return value.resolve()


def _verify_file(path: Path, record: dict[str, Any], label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ContractError(f"{label} byte size changed")
    observed = sha256_file(path)
    if observed != str(record["sha256"]).lower():
        raise ContractError(f"{label} SHA256 changed")
    return observed


def _load_contract(data_dir: Path) -> tuple[dict[str, Any], Any, Any, dict[str, str]]:
    config = _read_json(CONFIG)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID drifted")
    if config.get("status") != "PREREGISTERED_ONE_SHOT_LOCAL_ONLY_EXECUTION_APPROVED_2026-08-28":
        raise ContractError("approval status changed")
    policy = config["execution_policy"]
    if (
        int(policy["maximum_executions"]) != 1
        or bool(policy["result_based_retry_or_tuning"])
        or bool(policy["official_test_context_index_sample_submission_access_allowed"])
        or bool(policy["submission_csv_generation_allowed"])
        or bool(policy["upload_allowed"])
    ):
        raise ContractError("execution boundary changed")
    if tuple(config["base_contract"]["source_train_years"]) != SOURCE_TRAIN_YEARS:
        raise ContractError("source train years changed")
    if tuple(config["base_contract"]["source_held_years"]) != SOURCE_HELD_YEARS:
        raise ContractError("source held years changed")
    if int(config["base_contract"]["frozen_feature_count"]) != len(common_feature_columns()):
        raise ContractError("frozen 286-feature count changed")
    if int(config["directional_features"]["total_enriched_feature_count"]) != 306:
        raise ContractError("enriched feature count changed")
    if int(config["model"]["maximum_fits"]) != 4:
        raise ContractError("fit budget changed")
    if dict(config["model"]["source_pretrain"]) != dict(SOURCE_CATBOOST_PARAMETERS):
        raise ContractError("source CatBoost schedule changed")
    configured_local = dict(config["model"]["local_continuation"])
    configured_local.pop("sample_weight")
    if configured_local != dict(LOCAL_CATBOOST_PARAMETERS):
        raise ContractError("local CatBoost schedule changed")
    increment = config["model"]["candidate_increment"]
    if (
        float(increment["weight"]) != 0.20
        or tuple(increment["active_leads_h"]) != (18, 24)
        or tuple(increment["protected_leads_h"]) != (3, 6, 9, 12)
    ):
        raise ContractError("candidate increment changed")

    hashes: dict[str, str] = {}
    base_path = ROOT / config["base_contract"]["path"]
    hashes["base_contract"] = _verify_file(base_path, config["base_contract"], "base contract")
    for label, record in config["immutable_inputs"].items():
        hashes[label] = _verify_file(ROOT / record["path"], record, label)
    wave_path = data_dir / config["source_train_wave"]["filename"]
    hashes["train_wave"] = _verify_file(wave_path, config["source_train_wave"], "train_wave")
    hashes.update(
        {
            "config": sha256_file(CONFIG),
            "module": sha256_file(MODULE),
            "runner": sha256_file(RUNNER),
            "test": sha256_file(TEST),
        }
    )
    frozen = _load_frozen_runner()
    _, _, frozen_paths = frozen._load_contract(ROOT)
    return config, frozen, frozen_paths, hashes


def _read_local_metadata(anchor_path: Path) -> pd.DataFrame:
    anchors = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", "station", "anchor_time", "current_hs"],
    )
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    if anchors["anchor_id"].duplicated().any() or anchors["current_hs"].lt(1.5).any():
        raise ContractError("local anchor metadata changed")
    return anchors


def _shadow_split(
    config: dict[str, Any], data_dir: Path, anchor_path: Path
) -> tuple[tuple[Any, ...], pd.DataFrame, dict[str, Any]]:
    anchors = _read_local_metadata(anchor_path)
    wave = pd.read_csv(
        data_dir / config["source_train_wave"]["filename"],
        usecols=["station", "time", "hs"],
    )
    with_episodes = assign_storm_episodes_from_wave(anchors, wave)
    shadow = config["validation"]["fresh_shadow"]
    return build_corrected_repeated_forward_folds(
        with_episodes,
        windows=shadow["windows"],
        gap_hours=int(shadow["gap_hours"]),
        footprint_hours=int(shadow["footprint_hours"]),
    )


def _support_receipt(selected: pd.DataFrame) -> dict[str, Any]:
    return {
        "cases": int(len(selected)),
        "by_station": {
            str(key): int(value)
            for key, value in selected.groupby("station", sort=True, observed=True).size().items()
        },
        "outcome_values_read": 0,
    }


def _support_passed(config: dict[str, Any], receipt: dict[str, Any]) -> bool:
    gate = config["validation"]["fresh_shadow"]
    return bool(
        int(receipt["cases"]) >= int(gate["minimum_cases"])
        and min(receipt["by_station"].values(), default=0)
        >= int(gate["minimum_cases_per_station"])
    )


def check_only(data_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve():
        raise ContractError("runtime root changed")
    config, frozen, frozen_paths, hashes = _load_contract(data_dir)
    source_receipt, _, _ = frozen._external_preflight(frozen_paths)
    folds, selected, audit = _shadow_split(config, data_dir, frozen_paths.train_anchors)
    support = _support_receipt(selected)
    raw = np.load(ROOT / config["immutable_inputs"]["raw_contexts"]["path"], mmap_mode="r")
    station = np.load(
        ROOT / config["immutable_inputs"]["station_codes"]["path"], mmap_mode="r"
    )
    if raw.shape != (24360, 289, 10) or station.shape != (24360,):
        raise ContractError("local sequence cache shape changed")
    return {
        "schema_version": "p3.directional_memory.check.v1",
        "experiment_id": EXPERIMENT_ID,
        "passed": True,
        "writes": 0,
        "model_fits": 0,
        "official_rows_read": 0,
        "hashes": hashes,
        "environment": {
            "python": platform.python_version(),
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "executable": sys.executable,
        },
        "source_external_preflight": source_receipt,
        "source_feature_count": 286,
        "enriched_feature_count": 306,
        "directional_feature_names": list(DIRECTIONAL_FEATURES),
        "fresh_shadow_support": support,
        "fresh_shadow_support_passed": _support_passed(config, support),
        "fresh_shadow_audit": audit,
        "fresh_shadow_fold_count": len(folds),
        "output_exists": (ROOT / config["artifact_directory"]).exists(),
        "attempt_lock_exists": (
            ROOT / "artifacts" / f"{EXPERIMENT_ID}.attempt.lock"
        ).exists(),
    }


def _build_source_past_surface(
    hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build eligibility metadata and past features without retaining held targets."""

    metadata_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, float]] = []
    direction_rows: list[dict[str, float]] = []
    next_id = 1
    for station, group in hourly.groupby("station", sort=False, observed=True):
        canonical = canonicalize_era5_hourly(group.drop(columns="station"), time_column="time")
        first = canonical["time"].iloc[0] + pd.Timedelta(hours=48)
        last = canonical["time"].iloc[-1] - pd.Timedelta(hours=24)
        candidates = pd.date_range(first, last, freq="6h")
        hs_by_time = canonical.set_index("time")["hs"]
        current = hs_by_time.reindex(candidates).to_numpy(dtype=np.float64)
        # Future values are used only for the preregistered complete-case predicate.
        # Their numeric values are neither returned nor passed to feature/model code.
        complete = np.ones(len(candidates), dtype=bool)
        for lead in LEADS:
            future = hs_by_time.reindex(candidates + pd.Timedelta(hours=lead)).to_numpy(
                dtype=np.float64
            )
            complete &= np.isfinite(future) & (future >= 0.0)
        valid = np.isfinite(current) & (current >= 1.5) & complete
        station_times = (
            canonical["time"]
            .dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        for anchor_time, current_hs in zip(candidates[valid], current[valid], strict=True):
            start_ns = (anchor_time - pd.Timedelta(hours=48)).value
            stop_ns = anchor_time.value
            left = int(np.searchsorted(station_times, start_ns, side="left"))
            right = int(np.searchsorted(station_times, stop_ns, side="right"))
            history = canonical.iloc[left:right]
            if len(history) != 49:
                raise ContractError("source context is not exactly 49 hourly rows")
            anchor_id = next_id
            next_id += 1
            metadata_rows.append(
                {
                    "anchor_id": anchor_id,
                    "station": str(station),
                    "anchor_time": anchor_time,
                    "current_hs": float(current_hs),
                }
            )
            base_rows.append(summarize_past_48h(history, anchor_time=anchor_time))
            direction_rows.append(
                summarize_directional_energy_memory(
                    history["hs"].to_numpy(dtype=np.float64),
                    history["wvdir"].to_numpy(dtype=np.float64),
                    np.arange(-48, 1, dtype=np.float64),
                )
            )
    metadata = pd.DataFrame(metadata_rows)
    base = pd.DataFrame(base_rows, columns=common_feature_columns())
    direction = pd.DataFrame(direction_rows, columns=DIRECTIONAL_FEATURES)
    if len(metadata) != len(base) or len(base) != len(direction):
        raise ContractError("source past-only surfaces do not align")
    enriched = pd.concat([base, direction], axis=1)
    coverage = {
        name: float(np.isfinite(direction[name].to_numpy(dtype=np.float64)).mean())
        for name in VALUE_FEATURES
    }
    return metadata, enriched, {
        "cases": int(len(metadata)),
        "base_feature_count": len(common_feature_columns()),
        "enriched_feature_count": int(enriched.shape[1]),
        "directional_value_coverage": coverage,
        "held_numeric_target_values_retained_before_prediction_seal": 0,
        "complete_target_availability_predicate_only": True,
    }


def _source_split(metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    times = pd.to_datetime(metadata["anchor_time"], utc=True, errors="raise")
    years = times.dt.year
    complete_year = (
        (times - pd.Timedelta(hours=48)).dt.year.eq(years)
        & (times + pd.Timedelta(hours=24)).dt.year.eq(years)
    )
    train_ids = metadata.loc[
        complete_year & years.isin(SOURCE_TRAIN_YEARS), "anchor_id"
    ].to_numpy(dtype=np.int64)
    held = select_source_year_validation(
        metadata,
        held_years=SOURCE_HELD_YEARS,
        station_column="station",
    )
    if set(train_ids).intersection(held["anchor_id"].astype(int)):
        raise ContractError("source train and held IDs overlap")
    if len(train_ids) != 7311 or len(held) != 492:
        raise ContractError("frozen source split counts changed")
    return train_ids, held


def _source_targets(
    hourly: pd.DataFrame, metadata: pd.DataFrame, anchor_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    lookup = metadata.set_index("anchor_id").loc[ids]
    indexed = hourly.set_index(["station", "time"])["hs"]
    future = np.empty((len(ids), len(LEADS)), dtype=np.float64)
    for row_position, row in enumerate(lookup.itertuples()):
        for lead_position, lead in enumerate(LEADS):
            future[row_position, lead_position] = float(
                indexed.loc[(str(row.station), pd.Timestamp(row.anchor_time) + pd.Timedelta(hours=lead))]
            )
    current = lookup["current_hs"].to_numpy(dtype=np.float64)
    if not np.isfinite(future).all() or (future < 0).any():
        raise ContractError("source targets are invalid")
    log_delta = np.log1p(future) - np.log1p(current)[:, None]
    return log_delta, future


def _matrix_rows(
    metadata: pd.DataFrame,
    case_ids: np.ndarray,
    group_metadata: pd.DataFrame,
    base: np.ndarray,
    enriched: np.ndarray,
    candidate: np.ndarray,
    *,
    fold_column: str,
) -> pd.DataFrame:
    source = metadata.set_index("anchor_id").loc[case_ids]
    group = group_metadata.set_index("anchor_id")
    rows: list[dict[str, Any]] = []
    for position, anchor_id in enumerate(case_ids):
        current = source.loc[int(anchor_id)]
        extra = group.loc[int(anchor_id)]
        for lead_position, lead in enumerate(LEADS):
            rows.append(
                {
                    "fold": str(extra[fold_column]),
                    "anchor_id": int(anchor_id),
                    "station": str(current["station"]),
                    "lead_h": int(lead),
                    "episode_id": int(extra["episode_id"]),
                    "base_prediction": float(base[position, lead_position]),
                    "enriched_prediction": float(enriched[position, lead_position]),
                    "candidate_prediction": float(candidate[position, lead_position]),
                }
            )
    frame = pd.DataFrame(rows).sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ContractError("prediction rows are duplicated")
    return frame


def _attach_matrix_truth(
    frame: pd.DataFrame,
    case_ids: np.ndarray,
    truth: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for position, anchor_id in enumerate(case_ids):
        for lead_position, lead in enumerate(LEADS):
            rows.append(
                {
                    "anchor_id": int(anchor_id),
                    "lead_h": int(lead),
                    "target_hs": float(truth[position, lead_position]),
                }
            )
    target = pd.DataFrame(rows)
    result = frame.merge(target, on=["anchor_id", "lead_h"], how="left", validate="one_to_one")
    if result["target_hs"].isna().any():
        raise ContractError("truth attachment failed")
    return result


def _rmse(truth: pd.Series, prediction: pd.Series) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction.to_numpy(dtype=np.float64) - truth.to_numpy(dtype=np.float64)
                )
            )
        )
    )


def _metric(frame: pd.DataFrame) -> dict[str, float | int]:
    baseline = _rmse(frame["target_hs"], frame["base_prediction"])
    candidate = _rmse(frame["target_hs"], frame["candidate_prediction"])
    return {
        "rows": int(len(frame)),
        "base_rmse_m": baseline,
        "candidate_rmse_m": candidate,
        "delta_m": candidate - baseline,
    }


def _breakdown(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float | int]]:
    return {
        str(key): _metric(group)
        for key, group in frame.groupby(column, sort=True, observed=True)
    }


def _station_lead_breakdown(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    return {
        f"{station}|{int(lead)}": _metric(group)
        for (station, lead), group in frame.groupby(["station", "lead_h"], sort=True)
    }


def _paired_bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, Any]:
    grouped = []
    for _, block in frame.groupby("episode_id", sort=True, observed=True):
        truth = block["target_hs"].to_numpy(dtype=np.float64)
        base = block["base_prediction"].to_numpy(dtype=np.float64)
        candidate = block["candidate_prediction"].to_numpy(dtype=np.float64)
        grouped.append(
            (
                float(np.square(base - truth).sum()),
                float(np.square(candidate - truth).sum()),
                int(len(block)),
            )
        )
    if not grouped:
        raise ContractError("bootstrap has no episodes")
    base_sse = np.asarray([item[0] for item in grouped], dtype=np.float64)
    candidate_sse = np.asarray([item[1] for item in grouped], dtype=np.float64)
    counts = np.asarray([item[2] for item in grouped], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        denominator = counts[selected].sum()
        deltas[draw] = np.sqrt(candidate_sse[selected].sum() / denominator) - np.sqrt(
            base_sse[selected].sum() / denominator
        )
    return {
        "unit": "episode_id",
        "episodes": len(grouped),
        "replicates": replicates,
        "seed": seed,
        "observed_delta_m": _metric(frame)["delta_m"],
        "ci90_lower_m": float(np.quantile(deltas, 0.05)),
        "ci90_upper_m": float(np.quantile(deltas, 0.95)),
    }


def _source_gate(
    frame: pd.DataFrame, config: dict[str, Any], coverage: dict[str, float]
) -> dict[str, Any]:
    gate = config["validation"]["source_gate_first"]
    overall = _metric(frame)
    by_year = _breakdown(frame, "fold")
    by_station = _breakdown(frame, "station")
    by_lead = _breakdown(frame, "lead_h")
    bootstrap = _paired_bootstrap(
        frame, int(gate["bootstrap_replicates"]), int(gate["bootstrap_seed"])
    )
    slices = [*by_year.values(), *by_station.values(), *by_lead.values()]
    maximum = max(float(item["delta_m"]) for item in slices)
    checks = {
        "pooled_delta_below_zero": float(overall["delta_m"]) < float(gate["pooled_delta_below_m"]),
        "bootstrap_ci90_upper_below_zero": float(bootstrap["ci90_upper_m"])
        < float(gate["bootstrap_ci90_upper_below_m"]),
        "all_three_years_non_degrading": len(by_year) == 3
        and all(float(value["delta_m"]) <= 0.0 for value in by_year.values()),
        "maximum_slice_regression": maximum
        <= float(gate["maximum_year_station_or_lead_regression_m"]),
        "directional_value_coverage": min(coverage.values(), default=0.0)
        >= float(gate["minimum_directional_value_coverage"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "overall": overall,
        "by_year": by_year,
        "by_station": by_station,
        "by_lead": by_lead,
        "maximum_year_station_or_lead_regression_m": maximum,
        "bootstrap": bootstrap,
        "directional_value_coverage": coverage,
    }


def _read_local_surface(
    config: dict[str, Any], frozen: Any, frozen_paths: Any
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    anchors = _read_local_metadata(frozen_paths.train_anchors)
    base = frozen._read_local_features(frozen_paths.train_features)
    raw = np.load(ROOT / config["immutable_inputs"]["raw_contexts"]["path"], mmap_mode="r")
    if not np.array_equal(anchors["anchor_id"].to_numpy(dtype=np.int64), np.arange(len(anchors))):
        raise ContractError("local anchor id is not the raw-cache row")
    direction = build_local_directional_features(raw, anchors["anchor_id"].to_numpy(dtype=np.int64))
    merged = base.merge(direction, on="anchor_id", how="left", validate="one_to_one")
    coverage = {
        name: float(np.isfinite(merged[name].to_numpy(dtype=np.float64)).mean())
        for name in VALUE_FEATURES
    }
    return anchors, merged, {
        "rows": int(len(merged)),
        "base_feature_count": len(common_feature_columns()),
        "enriched_feature_count": len(common_feature_columns()) + len(DIRECTIONAL_FEATURES),
        "directional_value_coverage": coverage,
    }


def _shadow_target_matrix(anchor_path: Path, ids: np.ndarray) -> np.ndarray:
    selected = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", *TARGET_COLUMNS],
        filters=[("anchor_id", "in", np.asarray(ids, dtype=np.int64).tolist())],
    ).set_index("anchor_id")
    result = selected.loc[ids, list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if result.shape != (len(ids), len(LEADS)) or not np.isfinite(result).all():
        raise ContractError("shadow truth matrix changed")
    return result


def _shadow_gate(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    gate = config["validation"]["fresh_shadow"]
    active = frame.loc[frame["lead_h"].isin(gate["active_leads_h"])].copy()
    overall = _metric(active)
    by_station = _breakdown(active, "station")
    by_lead = _breakdown(active, "lead_h")
    by_station_lead = _station_lead_breakdown(active)
    bootstrap = _paired_bootstrap(
        active, int(gate["bootstrap_replicates"]), int(gate["bootstrap_seed"])
    )
    worst = max(float(value["delta_m"]) for value in by_station_lead.values())
    checks = {
        "pooled_delta": float(overall["delta_m"])
        <= float(gate["pooled_delta_at_most_m"]),
        "bootstrap_ci90_upper": float(bootstrap["ci90_upper_m"])
        < float(gate["bootstrap_ci90_upper_below_m"]),
        "all_stations_non_degrading": len(by_station) == 3
        and all(float(value["delta_m"]) <= 0.0 for value in by_station.values()),
        "both_active_leads_improve": set(by_lead) == {"18", "24"}
        and all(float(value["delta_m"]) < 0.0 for value in by_lead.values()),
        "maximum_station_lead_regression": worst
        <= float(gate["maximum_station_lead_regression_m"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "active_leads_only": [18, 24],
        "overall": overall,
        "by_station": by_station,
        "by_lead": by_lead,
        "by_station_lead": by_station_lead,
        "maximum_station_lead_regression_m": worst,
        "bootstrap": bootstrap,
    }


def _write_terminal_result(
    output: Path, result: dict[str, Any], artifact_hashes: dict[str, str]
) -> dict[str, Any]:
    result["artifact_hashes"] = dict(artifact_hashes)
    result_path = output / "result.json"
    result_hash = _atomic_json(result_path, result)
    manifest = {
        "schema_version": "p3.directional_memory.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "result_sha256": result_hash,
        "artifacts": artifact_hashes,
        "official_rows_read": 0,
        "submission_generated": False,
        "uploaded": False,
    }
    manifest_hash = _atomic_json(output / "manifest.json", manifest)
    return {**result, "result_sha256": result_hash, "manifest_sha256": manifest_hash}


def execute_once(data_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve():
        raise ContractError("runtime root changed")
    started = time.perf_counter()
    config, frozen, frozen_paths, hashes = _load_contract(data_dir)
    output = ROOT / config["artifact_directory"]
    lock_path = ROOT / "artifacts" / f"{EXPERIMENT_ID}.attempt.lock"
    if output.exists() or lock_path.exists():
        raise FileExistsError("one-shot attempt is already consumed")
    preflight = check_only(data_dir, root)
    lock_sha = _create_attempt_lock(lock_path, hashes["config"])
    output.mkdir(parents=True, exist_ok=False)
    artifact_hashes: dict[str, str] = {}
    fit_count = 0

    source_hourly, source_provenance = frozen._load_source_hourly(frozen_paths)
    source_metadata, source_features, source_feature_receipt = _build_source_past_surface(
        source_hourly
    )
    source_train_ids, source_held = _source_split(source_metadata)
    feature_lookup = source_features.copy()
    feature_lookup.insert(0, "anchor_id", source_metadata["anchor_id"].to_numpy(dtype=np.int64))
    feature_lookup = feature_lookup.set_index("anchor_id")
    source_train_target, _ = _source_targets(source_hourly, source_metadata, source_train_ids)
    base_columns = tuple(common_feature_columns())
    enriched_columns = (*base_columns, *DIRECTIONAL_FEATURES)
    base_model = DirectionalContextTransferRegressor(base_columns).fit_pretrain(
        feature_lookup.loc[source_train_ids, list(base_columns)].reset_index(drop=True),
        source_train_target,
    )
    enriched_model = DirectionalContextTransferRegressor(enriched_columns).fit_pretrain(
        feature_lookup.loc[source_train_ids, list(enriched_columns)].reset_index(drop=True),
        source_train_target,
    )
    fit_count += 2
    source_model_dir = output / "models"
    base_model.save_model(source_model_dir / "source_base.cbm")
    enriched_model.save_model(source_model_dir / "source_enriched.cbm")
    artifact_hashes["source_base_model"] = sha256_file(source_model_dir / "source_base.cbm")
    artifact_hashes["source_enriched_model"] = sha256_file(
        source_model_dir / "source_enriched.cbm"
    )

    held_ids = source_held["anchor_id"].to_numpy(dtype=np.int64)
    held_current = source_metadata.set_index("anchor_id").loc[held_ids, "current_hs"].to_numpy(
        dtype=np.float64
    )
    base_source_prediction = apply_frozen_persistence_shrink(
        base_model.predict_hs(
            feature_lookup.loc[held_ids, list(base_columns)].reset_index(drop=True),
            current_hs=held_current,
        ),
        held_current,
    )
    enriched_source_prediction = apply_frozen_persistence_shrink(
        enriched_model.predict_hs(
            feature_lookup.loc[held_ids, list(enriched_columns)].reset_index(drop=True),
            current_hs=held_current,
        ),
        held_current,
    )
    source_candidate = apply_directional_increment(
        base_source_prediction, enriched_source_prediction
    )
    source_blind = _matrix_rows(
        source_metadata,
        held_ids,
        source_held.assign(year=source_held["year"].astype(str)),
        base_source_prediction,
        enriched_source_prediction,
        source_candidate,
        fold_column="year",
    )
    if "target_hs" in source_blind:
        raise ContractError("source target entered the blind prediction surface")
    source_seal = output / "source_predictions_sealed.parquet"
    artifact_hashes["source_prediction_seal"] = _atomic_parquet(source_seal, source_blind)
    artifact_hashes["source_prediction_receipt"] = _atomic_json(
        output / "source_prediction_seal.json",
        {
            "rows": int(len(source_blind)),
            "prediction_sha256": array_sha256(source_blind["candidate_prediction"]),
            "numeric_held_target_values_attached_before_seal": 0,
            "complete_target_availability_predicate_only": True,
            "fit_count": fit_count,
        },
    )
    _, source_truth = _source_targets(source_hourly, source_metadata, held_ids)
    source_evaluated = _attach_matrix_truth(source_blind, held_ids, source_truth)
    source_gate = _source_gate(
        source_evaluated,
        config,
        source_feature_receipt["directional_value_coverage"],
    )
    base_result: dict[str, Any] = {
        "schema_version": "p3.directional_memory.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_only": True,
        "attempt_lock_sha256": lock_sha,
        "preflight": preflight,
        "input_and_code_hashes": hashes,
        "source_provenance": source_provenance,
        "source_feature_receipt": source_feature_receipt,
        "source_split": {
            "train_cases": int(len(source_train_ids)),
            "held_cases": int(len(held_ids)),
            "train_years": list(SOURCE_TRAIN_YEARS),
            "held_years": list(SOURCE_HELD_YEARS),
            "station_gap_hours": 78,
        },
        "source_gate": source_gate,
        "fit_count": fit_count,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "result_based_retry": False,
    }
    if not source_gate["passed"]:
        base_result.update(
            {
                "status": "NO_GO_SOURCE_GATE",
                "shadow_support": None,
                "shadow_gate": None,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        return _write_terminal_result(output, base_result, artifact_hashes)

    folds, selected, shadow_audit = _shadow_split(
        config, data_dir, frozen_paths.train_anchors
    )
    support = _support_receipt(selected)
    base_result["shadow_support"] = support
    base_result["shadow_audit"] = shadow_audit
    if not _support_passed(config, support):
        base_result.update(
            {
                "status": "NO_GO_SUPPORT",
                "shadow_gate": None,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        return _write_terminal_result(output, base_result, artifact_hashes)

    anchors, local_features, local_feature_receipt = _read_local_surface(
        config, frozen, frozen_paths
    )
    local_lookup = local_features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    prediction_blocks: list[pd.DataFrame] = []
    local_model_hashes: dict[str, str] = {}
    for fold in folds:
        train_ids = np.asarray(fold.train_ids, dtype=np.int64)
        validation_ids = np.asarray(fold.validation_ids, dtype=np.int64)
        local_target = frozen._read_training_targets(frozen_paths.train_anchors, train_ids)
        log_target = frozen._log_delta_targets(local_target)
        train_current = local_target["current_hs"].to_numpy(dtype=np.float64)
        base_local = base_model.clone_pretrained().continue_local(
            local_lookup.loc[train_ids, list(base_columns)].reset_index(drop=True),
            log_target,
            current_hs=train_current,
        )
        enriched_local = enriched_model.clone_pretrained().continue_local(
            local_lookup.loc[train_ids, list(enriched_columns)].reset_index(drop=True),
            log_target,
            current_hs=train_current,
        )
        fit_count += 2
        base_model_path = output / "models" / f"{fold.name}_base.cbm"
        enriched_model_path = output / "models" / f"{fold.name}_enriched.cbm"
        base_local.save_model(base_model_path)
        enriched_local.save_model(enriched_model_path)
        local_model_hashes[f"{fold.name}_base_model"] = sha256_file(base_model_path)
        local_model_hashes[f"{fold.name}_enriched_model"] = sha256_file(enriched_model_path)
        current = anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(dtype=np.float64)
        base_prediction = apply_frozen_persistence_shrink(
            base_local.predict_hs(
                local_lookup.loc[validation_ids, list(base_columns)].reset_index(drop=True),
                current_hs=current,
            ),
            current,
        )
        enriched_prediction = apply_frozen_persistence_shrink(
            enriched_local.predict_hs(
                local_lookup.loc[validation_ids, list(enriched_columns)].reset_index(drop=True),
                current_hs=current,
            ),
            current,
        )
        candidate = apply_directional_increment(base_prediction, enriched_prediction)
        fold_metadata = selected.loc[selected["fold"].eq(fold.name)].copy()
        prediction_blocks.append(
            _matrix_rows(
                anchors,
                validation_ids,
                fold_metadata,
                base_prediction,
                enriched_prediction,
                candidate,
                fold_column="fold",
            )
        )
    if fit_count != int(config["model"]["maximum_fits"]):
        raise ContractError("fit count differs from preregistered maximum")
    artifact_hashes.update(local_model_hashes)
    shadow_blind = (
        pd.concat(prediction_blocks, ignore_index=True)
        .sort_values(list(KEY_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )
    protected = shadow_blind["lead_h"].isin((3, 6, 9, 12))
    if not np.array_equal(
        shadow_blind.loc[protected, "candidate_prediction"].to_numpy(),
        shadow_blind.loc[protected, "base_prediction"].to_numpy(),
    ):
        raise ContractError("protected early leads changed")
    shadow_seal = output / "shadow_predictions_sealed.parquet"
    artifact_hashes["shadow_prediction_seal"] = _atomic_parquet(shadow_seal, shadow_blind)
    artifact_hashes["shadow_prediction_receipt"] = _atomic_json(
        output / "shadow_prediction_seal.json",
        {
            "rows": int(len(shadow_blind)),
            "cases": int(shadow_blind["anchor_id"].nunique()),
            "candidate_sha256": array_sha256(shadow_blind["candidate_prediction"]),
            "truth_rows_read_before_seal": 0,
            "fit_count": fit_count,
        },
    )
    validation_ids = selected.sort_values("anchor_id")["anchor_id"].to_numpy(dtype=np.int64)
    truth = _shadow_target_matrix(frozen_paths.train_anchors, validation_ids)
    # Rebind truth by ID; row order of the sealed surface is irrelevant.
    shadow_evaluated = _attach_matrix_truth(shadow_blind, validation_ids, truth)
    shadow_gate = _shadow_gate(shadow_evaluated, config)
    base_result.update(
        {
            "status": "RESEARCH_GATE_PASS" if shadow_gate["passed"] else "NO_GO_SHADOW_GATE",
            "shadow_support": support,
            "shadow_audit": shadow_audit,
            "local_feature_receipt": local_feature_receipt,
            "shadow_gate": shadow_gate,
            "fit_count": fit_count,
            "runtime_seconds": time.perf_counter() - started,
            "promotion": "research candidate only" if shadow_gate["passed"] else "no-go",
        }
    )
    return _write_terminal_result(output, base_result, artifact_hashes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_dir = _resolve_data_dir(args.data_dir)
    result = execute_once(data_dir, args.root) if args.execute else check_only(data_dir, args.root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
