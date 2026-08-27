"""Fail-closed one-shot mechanics for the P2 ERA5 mixing-gate experiment."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from p2_restore.era5_mixing_gate import ERA5_MIXING_FEATURES
from p2_restore.regime_gate import (
    STATE_FEATURES,
    build_public_state_features,
    fit_soft_gate,
    predict_soft_gate,
    soft_gate_weights,
)

KEY_COLUMNS = ("time", "layer", "block")
EXPERT_COLUMNS = ("deep_prediction", "physical_prediction")
BLIND_COLUMNS = (
    "time",
    "layer",
    "block",
    "deep_prediction",
    "physical_prediction",
    "control_physical_weight",
    "selected_physical_weight",
    "control_prediction",
    "selected_prediction",
    "selected_arm",
)
OUTER_BLOCKS = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")
TARGET_LAYERS = (2, 3, 4)
TARGET_COLUMNS = ("temp", "psal")
NATIVE_FLUX_SIGN_SEMANTICS = {
    "surface_net_solar_radiation": "positive downward accumulated J m-2",
    "surface_net_thermal_radiation": "positive downward accumulated J m-2",
    "surface_latent_heat_flux": "positive downward accumulated J m-2",
    "surface_sensible_heat_flux": "positive downward accumulated J m-2",
}
NATIVE_QNET_DEFINITION = (
    "surface_net_solar_radiation + surface_net_thermal_radiation + "
    "surface_latent_heat_flux + surface_sensible_heat_flux"
)
BLOCK_INTERVALS_KST = {
    "2024_sep_oct": ("2024-09-01T00:00:00+09:00", "2024-11-01T00:00:00+09:00"),
    "2025_jul_aug": ("2025-07-01T00:00:00+09:00", "2025-09-01T00:00:00+09:00"),
    "2025_nov_dec": ("2025-11-01T00:00:00+09:00", "2026-01-01T00:00:00+09:00"),
}


def validate_native_flux_sign_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Pin corrected ECMWF positive-downward semantics without changing raw values."""

    expected = {
        "source_convention": "ECMWF vertical surface fluxes use positive downward values.",
        "accumulated_units": "J m-2 over the one-hour ERA5 accumulation interval",
        "sign_semantics": NATIVE_FLUX_SIGN_SEMANTICS,
        "numeric_transform": (
            "No component sign flip; sum the four stored accumulated values exactly, "
            "then divide by 3600 seconds for W m-2."
        ),
        "qnet_definition": NATIVE_QNET_DEFINITION,
        "qnet_sign_semantics": "positive downward net surface heat flux into the ocean",
        "legacy_dry_metadata_status": (
            "Known metadata-only error: the immutable dry config and retrieval manifest "
            "described STR, SLHF, and SSHF as upward-positive. Their numeric values and the "
            "preregistered native sum are unchanged."
        ),
        "legacy_artifacts_mutated": False,
        "official_evidence": [
            "https://codes.ecmwf.int/grib/param-db/177",
            "https://confluence.ecmwf.int/spaces/TIGGE/pages/51721809/Surface+latent+heat+flux",
            "https://confluence.ecmwf.int/spaces/TIGGE/pages/51721815/Surface+sensible+heat+flux",
        ],
    }
    observed = contract.get("native_flux_semantics")
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise ValueError("corrected ECMWF native-flux sign contract changed")
    return {
        "component_sign_semantics": dict(NATIVE_FLUX_SIGN_SEMANTICS),
        "qnet_definition": NATIVE_QNET_DEFINITION,
        "qnet_sign_semantics": expected["qnet_sign_semantics"],
        "component_sign_flips_applied": 0,
        "legacy_artifacts_mutated": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def write_json_exclusive_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one JSON file exactly once, flushing bytes before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # The partial file intentionally remains as a fail-closed attempt marker.
        raise


class AppendOnlyLedger:
    """Durable append-only JSONL ledger with a monotonic hash chain."""

    def __init__(self, path: Path, *, experiment_id: str) -> None:
        self.path = path
        self.experiment_id = experiment_id
        self.sequence = 0
        self.previous_sha256 = "0" * 64
        self._event_signatures: set[str] = set()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        self.append("ledger_created", {"experiment_id": experiment_id})

    def append(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        signature = hashlib.sha256(
            _json_bytes({"event": event, "payload": dict(payload)})
        ).hexdigest()
        if signature in self._event_signatures:
            raise ValueError("duplicate append-only ledger event/payload")
        body: dict[str, Any] = {
            "sequence": self.sequence,
            "created_at": datetime.now().astimezone().isoformat(),
            "experiment_id": self.experiment_id,
            "event": event,
            "previous_sha256": self.previous_sha256,
            "payload": dict(payload),
        }
        record_sha256 = hashlib.sha256(_json_bytes(body)).hexdigest()
        record = {**body, "record_sha256": record_sha256}
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        with os.fdopen(descriptor, "ab", closefd=True) as stream:
            stream.write(_json_bytes(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1
        self.previous_sha256 = record_sha256
        self._event_signatures.add(signature)
        return record


def verify_append_only_ledger(path: Path, *, experiment_id: str) -> dict[str, Any]:
    previous = "0" * 64
    count = 0
    signatures: set[str] = set()
    with path.open("rb") as stream:
        for raw in stream:
            record = json.loads(raw.decode("utf-8"))
            if record.get("sequence") != count or record.get("experiment_id") != experiment_id:
                raise ValueError("ledger sequence or experiment identity changed")
            if record.get("previous_sha256") != previous:
                raise ValueError("ledger hash chain is broken")
            claimed = record.pop("record_sha256", None)
            observed = hashlib.sha256(_json_bytes(record)).hexdigest()
            if claimed != observed:
                raise ValueError("ledger record SHA-256 is invalid")
            signature = hashlib.sha256(
                _json_bytes({"event": record["event"], "payload": record["payload"]})
            ).hexdigest()
            if signature in signatures:
                raise ValueError("ledger contains a duplicate event/payload")
            signatures.add(signature)
            previous = observed
            count += 1
    if count == 0:
        raise ValueError("ledger is empty")
    return {"record_count": count, "tail_sha256": previous}


class GrantBoundEra5Reader:
    """Instrumentation that makes reading ERA5 before a scope grant impossible."""

    def __init__(self) -> None:
        self._granted_path: Path | None = None
        self.events: list[str] = []
        self.value_read_count = 0

    def bind_grant(self, path: Path) -> None:
        if self._granted_path is not None:
            raise RuntimeError("ERA5 scope grant was already bound")
        self._granted_path = path.resolve()
        self.events.append("scope_grant_bound")

    def read_values(self, *, columns: Sequence[str]) -> pd.DataFrame:
        if self._granted_path is None:
            raise PermissionError("ERA5 values cannot be read before the canonical scope grant")
        self.events.append("era5_value_read")
        self.value_read_count += 1
        return pq.read_table(self._granted_path, columns=list(columns)).to_pandas()


def load_observations_only(
    data_directory: Path,
    *,
    expected_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only observations.csv; test/submission files are neither resolved nor opened."""

    root = data_directory.expanduser().resolve()
    path = (root / "observations.csv").resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("observations path escaped the supplied P2 directory") from error
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("observations.csv is missing or its SHA-256 changed")
    observations = pd.read_csv(path, dtype={"station": "string", "time": "string"})
    expected_columns = [
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
        "nominal_depth",
    ]
    if observations.columns.tolist() != expected_columns:
        raise ValueError("observations-only schema changed")
    if observations.duplicated(["station", "year", "layer", "time"]).any():
        raise ValueError("observations-only keys are duplicated")
    times = pd.to_datetime(observations["time"], utc=True, errors="raise").dt.tz_convert(
        "Asia/Seoul"
    )
    hidden = (
        times.ge(pd.Timestamp("2025-09-01", tz="Asia/Seoul"))
        & times.lt(pd.Timestamp("2025-11-01", tz="Asia/Seoul"))
        & observations["layer"].isin(TARGET_LAYERS)
    )
    if not observations.loc[hidden, TARGET_COLUMNS].isna().all().all():
        raise ValueError("hidden 2025 Sep-Oct target temperature/salinity is populated")
    return observations, {
        "rows": int(len(observations)),
        "duplicate_key_count": 0,
        "hidden_target_rows": int(hidden.sum()),
        "hidden_target_nonmissing_values": 0,
        "files_opened": ["observations.csv"],
        "test_or_submission_files_opened": 0,
    }


def mask_target_layers_for_validation_block(
    observations: pd.DataFrame,
    validation_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Jointly mask target-layer temp and psal over one complete validation interval."""

    if not {"time", "layer", *TARGET_COLUMNS}.issubset(observations.columns):
        raise ValueError("observations mask schema is incomplete")
    if not set(KEY_COLUMNS).issubset(validation_keys.columns) or validation_keys.empty:
        raise ValueError("validation keys are incomplete")
    blocks = validation_keys["block"].astype(str).unique()
    if len(blocks) != 1:
        raise ValueError("joint target mask requires exactly one validation block")
    block = str(blocks[0])
    if block not in BLOCK_INTERVALS_KST:
        raise ValueError("validation block has no canonical KST mask interval")
    obs_time = pd.to_datetime(observations["time"], utc=True, errors="raise")
    start = pd.Timestamp(BLOCK_INTERVALS_KST[block][0]).tz_convert("UTC")
    end = pd.Timestamp(BLOCK_INTERVALS_KST[block][1]).tz_convert("UTC")
    selected = observations["layer"].isin(TARGET_LAYERS) & obs_time.ge(start) & obs_time.lt(end)
    result = observations.copy(deep=True)
    before_temp = int(result.loc[selected, "temp"].notna().sum())
    before_psal = int(result.loc[selected, "psal"].notna().sum())
    result.loc[selected, ["temp", "psal"]] = np.nan
    if not result.loc[selected, ["temp", "psal"]].isna().all().all():
        raise AssertionError("target temperature and salinity were not jointly masked")
    if not result.loc[~selected].equals(observations.loc[~selected]):
        raise AssertionError("joint target mask modified rows outside the validation interval")
    return result, {
        "block": block,
        "interval_start_utc": start.isoformat(),
        "interval_end_exclusive_utc": end.isoformat(),
        "target_rows_jointly_masked": int(selected.sum()),
        "temperature_values_removed": before_temp,
        "salinity_values_removed": before_psal,
        "post_mask_nonmissing_target_values": 0,
    }


def build_block_masked_public_state_panel(
    observations: pd.DataFrame,
    keys: pd.DataFrame,
    *,
    outer_blocks: Sequence[str] = OUTER_BLOCKS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build each block's public state only after its target temp+psal joint mask."""

    keyed = keys.loc[:, KEY_COLUMNS].copy().reset_index(drop=True)
    keyed["_row_order"] = np.arange(len(keyed), dtype=np.int64)
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for block in outer_blocks:
        current = keyed.loc[keyed["block"].eq(block)].copy()
        if current.empty:
            raise ValueError(f"OOF keys are missing outer block {block}")
        masked, audit = mask_target_layers_for_validation_block(observations, current)
        features = build_public_state_features(masked, current.loc[:, ["time", "layer"]])
        if tuple(features.columns[2:]) != STATE_FEATURES:
            raise ValueError("public-state feature order changed")
        features["block"] = block
        features["_row_order"] = current["_row_order"].to_numpy()
        outputs.append(features.loc[:, ["time", "layer", "block", "_row_order", *STATE_FEATURES]])
        audits.append(audit)
    result = (
        pd.concat(outputs, ignore_index=True)
        .sort_values("_row_order", kind="stable")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )
    if not result.loc[:, KEY_COLUMNS].equals(keyed.loc[:, KEY_COLUMNS]):
        raise ValueError("block-masked public-state panel lost key order")
    return result, audits


def canonical_inner_partition_ids(
    frame: pd.DataFrame,
    *,
    partition_count: int = 3,
) -> np.ndarray:
    """Assign contiguous deterministic within-block timestamp partitions."""

    if partition_count != 3:
        raise ValueError("this generation requires exactly three inner partitions")
    result = np.full(len(frame), -1, dtype=np.int8)
    times = pd.to_datetime(frame["time"], utc=True, errors="raise")
    blocks = frame["block"].astype(str).to_numpy()
    for block in sorted(set(blocks)):
        block_rows = blocks == block
        unique_times = pd.DatetimeIndex(sorted(times.loc[block_rows].unique()))
        if len(unique_times) < partition_count:
            raise ValueError("inner block has too few unique timestamps")
        for partition, values in enumerate(np.array_split(unique_times, partition_count)):
            selected = block_rows & times.isin(values).to_numpy()
            result[selected] = partition
    if np.any(result < 0) or set(result) != {0, 1, 2}:
        raise ValueError("inner partition assignment is incomplete")
    return result


def purge_training_rows(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    purge_hours: int = 168,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove same-block training rows around each validation interval."""

    if purge_hours != 168:
        raise ValueError("this generation requires an exact seven-day purge")
    train_time = pd.to_datetime(train["time"], utc=True, errors="raise")
    validation_time = pd.to_datetime(validation["time"], utc=True, errors="raise")
    keep = np.ones(len(train), dtype=bool)
    pad = pd.Timedelta(hours=purge_hours)
    interval_count = 0
    for block in sorted(validation["block"].astype(str).unique()):
        valid_block = validation["block"].astype(str).eq(block)
        start = validation_time.loc[valid_block].min() - pad
        end = validation_time.loc[valid_block].max() + pad
        same_train_block = train["block"].astype(str).eq(block).to_numpy()
        inside = (train_time.ge(start) & train_time.lt(end)).to_numpy()
        keep &= ~(same_train_block & inside)
        interval_count += 1
    return keep, {
        "purge_hours": purge_hours,
        "validation_interval_count": interval_count,
        "input_train_rows": int(len(train)),
        "removed_rows": int((~keep).sum()),
        "retained_rows": int(keep.sum()),
    }


def _resolve_truth_shard_contract(
    repo_root: Path,
    shards: Mapping[str, Mapping[str, Any]],
    *,
    outer_blocks: Sequence[str],
) -> dict[str, dict[str, Any]]:
    root = repo_root.resolve()
    if set(shards) != set(outer_blocks):
        raise ValueError("truth-vault shard block set changed")
    resolved: dict[str, dict[str, Any]] = {}
    for block in outer_blocks:
        spec = dict(shards[block])
        path = (root / str(spec["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("truth-vault shard escaped the repository") from error
        if spec.get("row_group_block_min") != block or spec.get("row_group_block_max") != block:
            raise ValueError("truth-vault shard block-stat contract changed")
        resolved[block] = {**spec, "resolved_path": path}
    return resolved


def validate_truth_shard_key_contract(
    repo_root: Path,
    shards: Mapping[str, Mapping[str, Any]],
    *,
    expected_union_rows: int,
    expected_union_key_sha256: str,
    outer_blocks: Sequence[str] = OUTER_BLOCKS,
) -> dict[str, Any]:
    """Read shard metadata and keys only; never request the truth column."""

    resolved = _resolve_truth_shard_contract(repo_root, shards, outer_blocks=outer_blocks)
    key_frames: list[pd.DataFrame] = []
    audits: dict[str, dict[str, Any]] = {}
    for block in outer_blocks:
        spec = resolved[block]
        path = spec["resolved_path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError("truth-vault shard SHA-256 changed")
        parquet = pq.ParquetFile(path)
        block_index = parquet.schema_arrow.get_field_index("block")
        if block_index < 0 or parquet.num_row_groups != spec["row_groups"]:
            raise ValueError("truth-vault shard row-group structure changed")
        if parquet.metadata.num_rows != spec["rows"]:
            raise ValueError("truth-vault shard row count changed")
        row_group_stats = []
        for number in range(parquet.num_row_groups):
            statistics = parquet.metadata.row_group(number).column(block_index).statistics
            if statistics is None:
                raise ValueError("truth-vault shard block statistics are absent")
            minimum = str(statistics.min)
            maximum = str(statistics.max)
            if minimum != block or maximum != block:
                raise ValueError("truth-vault shard is not physically block-isolated")
            row_group_stats.append(
                {"row_group": number, "block_min": minimum, "block_max": maximum}
            )
        frame = pq.read_table(path, columns=list(KEY_COLUMNS)).to_pandas()
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
        frame["layer"] = frame["layer"].astype("int64")
        frame["block"] = frame["block"].astype("string")
        if len(frame) != spec["rows"] or set(frame["block"].astype(str)) != {block}:
            raise ValueError("truth-vault shard key rows changed")
        if frame.loc[:, KEY_COLUMNS].duplicated().any():
            raise ValueError("truth-vault shard keys are duplicated")
        key_frames.append(frame)
        audits[block] = {
            "path": str(spec["path"]),
            "sha256": str(spec["sha256"]),
            "rows": int(len(frame)),
            "row_groups": int(parquet.num_row_groups),
            "row_group_stats": row_group_stats,
            "columns_requested": list(KEY_COLUMNS),
            "truth_value_column_requested": False,
        }
    union = pd.concat(key_frames, ignore_index=True)
    payload = union.loc[:, KEY_COLUMNS].copy()
    payload["time"] = payload["time"].map(lambda value: value.isoformat())
    digest = hashlib.sha256(
        payload.to_csv(index=False, header=False, lineterminator="\n").encode()
    ).hexdigest()
    if len(union) != expected_union_rows or digest != expected_union_key_sha256:
        raise ValueError("truth-vault shard union key contract changed")
    if union.loc[:, KEY_COLUMNS].duplicated().any():
        raise ValueError("truth-vault shard union keys are duplicated")
    return {
        "shards": audits,
        "union_rows": int(len(union)),
        "union_key_sha256": digest,
        "union_keys_unique": True,
        "truth_value_columns_requested": 0,
    }


class FoldLocalTruthVault:
    """Read physically isolated truth shards; never open the current outer shard."""

    def __init__(
        self,
        repo_root: Path,
        shards: Mapping[str, Mapping[str, Any]],
        *,
        truth_column: str = "truth",
        outer_blocks: Sequence[str] = OUTER_BLOCKS,
    ) -> None:
        self.truth_column = truth_column
        self.outer_blocks = tuple(outer_blocks)
        self.shards = _resolve_truth_shard_contract(
            repo_root,
            shards,
            outer_blocks=self.outer_blocks,
        )
        self.fold_outer: str | None = None
        self.designated_open_count = 0
        self.value_open_log: list[dict[str, Any]] = []

    def _read_truth_shard(self, block: str, *, purpose: str) -> pd.DataFrame:
        spec = self.shards[block]
        path = spec["resolved_path"]
        self.value_open_log.append(
            {
                "block": block,
                "path": str(spec["path"]),
                "purpose": purpose,
                "truth_value_column_requested": True,
            }
        )
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError("truth-vault shard SHA-256 changed at semantic open")
        frame = pq.read_table(path, columns=[*KEY_COLUMNS, self.truth_column]).to_pandas()
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
        frame["layer"] = frame["layer"].astype("int64")
        frame["block"] = frame["block"].astype("string")
        if len(frame) != spec["rows"] or set(frame["block"].astype(str)) != {block}:
            raise ValueError("truth-vault shard semantic rows changed")
        if frame.loc[:, KEY_COLUMNS].duplicated().any() or frame[self.truth_column].isna().any():
            raise ValueError("truth-vault shard keys/values are invalid")
        return frame

    def open_fold_train(self, outer_block: str) -> pd.DataFrame:
        if outer_block not in self.outer_blocks or self.fold_outer is not None:
            raise PermissionError("one isolated vault instance may serve exactly one outer fold")
        allowed = [block for block in self.outer_blocks if block != outer_block]
        frames = [
            self._read_truth_shard(block, purpose=f"train_outer_{outer_block}") for block in allowed
        ]
        if any(entry["block"] == outer_block for entry in self.value_open_log):
            raise AssertionError("current outer truth shard was opened before prediction")
        frame = pd.concat(frames, ignore_index=True)
        if set(frame["block"].astype(str)) != set(allowed):
            raise ValueError("fold-local truth vault returned the wrong training blocks")
        if frame.loc[:, KEY_COLUMNS].duplicated().any() or frame[self.truth_column].isna().any():
            raise ValueError("fold-local truth vault keys/values are invalid")
        self.fold_outer = outer_block
        return frame

    def open_designated_outer_once(
        self,
        *,
        blind_seal_path: Path,
        expected_blind_seal_sha256: str,
        completed_outer_blocks: Sequence[str],
    ) -> pd.DataFrame:
        if self.designated_open_count != 0:
            raise PermissionError("designated outer truth may be opened exactly once")
        if self.fold_outer is not None:
            raise PermissionError("designated scoring requires a fresh post-seal vault instance")
        if (
            not blind_seal_path.is_file()
            or sha256_file(blind_seal_path) != expected_blind_seal_sha256
        ):
            raise PermissionError("blind outer-prediction seal is absent or changed")
        if set(completed_outer_blocks) != set(self.outer_blocks):
            raise PermissionError("all three fold-local predictions must precede outer truth")
        frames = [
            self._read_truth_shard(block, purpose="designated_outer_scoring")
            for block in self.outer_blocks
        ]
        frame = pd.concat(frames, ignore_index=True)
        if frame.loc[:, KEY_COLUMNS].duplicated().any() or frame[self.truth_column].isna().any():
            raise ValueError("designated outer truth keys/values are invalid")
        self.designated_open_count += 1
        return frame

    def audit(self) -> dict[str, Any]:
        return {
            "fold_outer": self.fold_outer,
            "designated_outer_open_count": self.designated_open_count,
            "truth_value_open_log": list(self.value_open_log),
        }


@dataclass(frozen=True)
class FoldOutcome:
    outer_block: str
    selected_arm: str
    inner_gate_passed: bool
    predictions: pd.DataFrame
    summary: dict[str, Any]


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


def arm_specification(feature_names: Sequence[str]) -> dict[str, Any]:
    return {
        "model": "p2_restore.regime_gate.fit_soft_gate",
        "prediction_columns": list(EXPERT_COLUMNS),
        "feature_names": list(feature_names),
        "regularization": 10.0,
        "max_iterations": 1000,
        "seed": 20260821,
        "weighting": "unweighted pooled row-level MSE",
        "parameter_grid": [],
    }


def assert_arm_symmetry() -> None:
    control = arm_specification(STATE_FEATURES)
    challenger = arm_specification((*STATE_FEATURES, *ERA5_MIXING_FEATURES))
    for key in control:
        if key == "feature_names":
            continue
        if control[key] != challenger[key]:
            raise AssertionError(f"control/challenger model contract differs at {key}")
    if challenger["feature_names"] != [*control["feature_names"], *ERA5_MIXING_FEATURES]:
        raise AssertionError("challenger differs by more than the ERA5 feature suffix")


def _fit_predict_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gate = fit_soft_gate(
        train,
        feature_names=feature_names,
        prediction_columns=EXPERT_COLUMNS,
        regularization=10.0,
    )
    iterations = {str(layer): fitted.optimizer_iterations for layer, fitted in gate.layers.items()}
    if any(value > 1000 for value in iterations.values()):
        raise AssertionError("gate optimizer exceeded the fixed iteration cap")
    prediction = predict_soft_gate(gate, test)
    weights = soft_gate_weights(gate, test)
    physical_weight = weights[:, 1]
    if not np.isfinite(physical_weight).all() or np.any(
        (physical_weight < 0.0) | (physical_weight > 1.0)
    ):
        raise AssertionError("soft gate produced a non-convex physical expert weight")
    return prediction, physical_weight, {"optimizer_iterations_by_layer": iterations}


def run_fold_local_gate(
    design: pd.DataFrame,
    fold_truth: pd.DataFrame,
    *,
    outer_block: str,
    purge_hours: int = 168,
) -> FoldOutcome:
    """Select ERA5 on/off without ever receiving the current outer block truth."""

    assert_arm_symmetry()
    if outer_block in set(fold_truth["block"].astype(str)):
        raise PermissionError("current outer validation truth reached fold-local selection")
    required = {
        *KEY_COLUMNS,
        *EXPERT_COLUMNS,
        *STATE_FEATURES,
        *ERA5_MIXING_FEATURES,
    }
    if not required.issubset(design.columns):
        raise ValueError("fold-local design schema is incomplete")
    outer_rows = design["block"].astype(str).eq(outer_block)
    train_design = design.loc[~outer_rows].copy()
    outer_design = design.loc[outer_rows].copy()
    train = train_design.merge(fold_truth, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    if train["truth"].isna().any() or len(train) != len(train_design):
        raise ValueError("fold-local training truth did not attach exactly")
    partition = canonical_inner_partition_ids(train)
    control_inner = np.full(len(train), np.nan, dtype=np.float64)
    challenger_inner = np.full(len(train), np.nan, dtype=np.float64)
    partition_metrics: list[dict[str, Any]] = []
    fit_audits: list[dict[str, Any]] = []
    for inner in range(3):
        validation_rows = partition == inner
        candidate_train = train.loc[~validation_rows].copy()
        validation = train.loc[validation_rows].copy()
        keep, purge_audit = purge_training_rows(
            candidate_train,
            validation,
            purge_hours=purge_hours,
        )
        fit_frame = candidate_train.loc[keep]
        if set(fit_frame["layer"].astype(int)) != set(TARGET_LAYERS):
            raise ValueError("purged inner training frame lost a target layer")
        control_prediction, _, control_audit = _fit_predict_arm(
            fit_frame,
            validation,
            feature_names=STATE_FEATURES,
        )
        challenger_prediction, _, challenger_audit = _fit_predict_arm(
            fit_frame,
            validation,
            feature_names=(*STATE_FEATURES, *ERA5_MIXING_FEATURES),
        )
        control_inner[validation_rows] = control_prediction
        challenger_inner[validation_rows] = challenger_prediction
        truth = validation["truth"].to_numpy(float)
        control_rmse = _rmse(truth, control_prediction)
        challenger_rmse = _rmse(truth, challenger_prediction)
        partition_metrics.append(
            {
                "inner_block": inner,
                "rows": int(len(validation)),
                "control_rmse": control_rmse,
                "challenger_rmse": challenger_rmse,
                "delta_rmse": challenger_rmse - control_rmse,
            }
        )
        fit_audits.append(
            {
                "inner_block": inner,
                "purge": purge_audit,
                "control": control_audit,
                "challenger": challenger_audit,
            }
        )
    if not np.isfinite(control_inner).all() or not np.isfinite(challenger_inner).all():
        raise ValueError("inner predictions are incomplete")
    truth = train["truth"].to_numpy(float)
    pooled_control = _rmse(truth, control_inner)
    pooled_challenger = _rmse(truth, challenger_inner)
    pooled_delta = pooled_challenger - pooled_control
    improved = sum(metric["delta_rmse"] < 0.0 for metric in partition_metrics)
    inner_gate_passed = pooled_delta < 0.0 and improved >= 2
    selected_arm = "challenger" if inner_gate_passed else "control"

    outer_train_keep, outer_purge = purge_training_rows(
        train,
        outer_design,
        purge_hours=purge_hours,
    )
    final_train = train.loc[outer_train_keep]
    control_outer, control_weight, control_final_audit = _fit_predict_arm(
        final_train,
        outer_design,
        feature_names=STATE_FEATURES,
    )
    if inner_gate_passed:
        selected_outer, selected_weight, selected_final_audit = _fit_predict_arm(
            final_train,
            outer_design,
            feature_names=(*STATE_FEATURES, *ERA5_MIXING_FEATURES),
        )
    else:
        selected_outer = control_outer.copy()
        selected_weight = control_weight.copy()
        selected_final_audit = {"exact_control_no_op": True}
    predictions = outer_design.loc[:, KEY_COLUMNS].copy()
    predictions["deep_prediction"] = outer_design["deep_prediction"].to_numpy(float)
    predictions["physical_prediction"] = outer_design["physical_prediction"].to_numpy(float)
    predictions["control_physical_weight"] = control_weight
    predictions["selected_physical_weight"] = selected_weight
    predictions["control_prediction"] = control_outer
    predictions["selected_prediction"] = selected_outer
    predictions["selected_arm"] = selected_arm
    if selected_arm == "control" and not np.array_equal(control_outer, selected_outer):
        raise AssertionError("failed fold-local gate is not an exact no-op")
    summary = {
        "outer_block": outer_block,
        "outer_truth_rows_seen_during_fit": 0,
        "train_blocks": sorted(train["block"].astype(str).unique()),
        "inner_control_rmse": pooled_control,
        "inner_challenger_rmse": pooled_challenger,
        "inner_delta_rmse": pooled_delta,
        "improved_inner_blocks": int(improved),
        "inner_gate_passed": bool(inner_gate_passed),
        "selected_arm": selected_arm,
        "inner_blocks": partition_metrics,
        "inner_fit_audits": fit_audits,
        "outer_train_purge": outer_purge,
        "control_outer_fit": control_final_audit,
        "selected_outer_fit": selected_final_audit,
        "outer_prediction_rows": int(len(predictions)),
    }
    return FoldOutcome(outer_block, selected_arm, inner_gate_passed, predictions, summary)


def _normalize_blind_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != BLIND_COLUMNS:
        raise ValueError("blind outer-prediction column order changed")
    result = frame.copy().reset_index(drop=True)
    result["time"] = pd.to_datetime(result["time"], utc=True, errors="raise")
    result["layer"] = result["layer"].astype("int64")
    result["block"] = result["block"].astype("string")
    result["deep_prediction"] = result["deep_prediction"].astype("float64")
    result["physical_prediction"] = result["physical_prediction"].astype("float64")
    result["control_physical_weight"] = result["control_physical_weight"].astype("float64")
    result["selected_physical_weight"] = result["selected_physical_weight"].astype("float64")
    result["control_prediction"] = result["control_prediction"].astype("float64")
    result["selected_prediction"] = result["selected_prediction"].astype("float64")
    result["selected_arm"] = result["selected_arm"].astype("string")
    if result.loc[:, KEY_COLUMNS].duplicated().any():
        raise ValueError("blind outer predictions contain duplicate keys")
    numeric_columns = [
        "deep_prediction",
        "physical_prediction",
        "control_physical_weight",
        "selected_physical_weight",
        "control_prediction",
        "selected_prediction",
    ]
    if not np.isfinite(result[numeric_columns].to_numpy(float)).all():
        raise ValueError("blind outer predictions contain non-finite values")
    for prefix in ("control", "selected"):
        weight = result[f"{prefix}_physical_weight"].to_numpy(float)
        if np.any((weight < 0.0) | (weight > 1.0)):
            raise ValueError("blind physical expert weight is outside [0,1]")
        expected = (1.0 - weight) * result["deep_prediction"].to_numpy(float) + weight * result[
            "physical_prediction"
        ].to_numpy(float)
        if not np.allclose(
            expected,
            result[f"{prefix}_prediction"].to_numpy(float),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("blind prediction is not the registered convex two-expert blend")
    if not set(result["selected_arm"].astype(str)).issubset({"control", "challenger"}):
        raise ValueError("blind selected arm is invalid")
    return result


def write_and_seal_blind_predictions(
    frame: pd.DataFrame,
    *,
    parquet_path: Path,
    seal_path: Path,
    seal_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """O_EXCL-write, fsync, reload, compare, hash, then seal blind predictions."""

    normalized = _normalize_blind_frame(frame)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(parquet_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        normalized.to_parquet(stream, index=False, compression="zstd")
        stream.flush()
        os.fsync(stream.fileno())
    parquet_sha256_before_reload = sha256_file(parquet_path)
    reloaded = pd.read_parquet(parquet_path, columns=list(BLIND_COLUMNS))
    reloaded = _normalize_blind_frame(reloaded)
    pd.testing.assert_frame_equal(
        normalized,
        reloaded,
        check_dtype=True,
        check_exact=True,
        check_like=False,
    )
    schema = pq.ParquetFile(parquet_path).schema_arrow
    if tuple(schema.names) != BLIND_COLUMNS:
        raise ValueError("reloaded blind Parquet schema changed")
    parquet_sha256 = sha256_file(parquet_path)
    if parquet_sha256 != parquet_sha256_before_reload:
        raise ValueError("blind Parquet SHA-256 changed during reload validation")
    seal = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage": "all_outer_predictions_complete_before_designated_outer_truth",
        "truth_columns_present": False,
        "rows": int(len(normalized)),
        "columns": list(BLIND_COLUMNS),
        "outer_blocks": sorted(normalized["block"].astype(str).unique()),
        "parquet_sha256": parquet_sha256,
        "parquet_sha256_unchanged_after_reload": True,
        "parquet_bytes": int(parquet_path.stat().st_size),
        "reload_schema_equal": True,
        "reload_dtype_equal": True,
        "reload_key_order_equal": True,
        "reload_values_exact_equal": True,
        **dict(seal_metadata),
    }
    write_json_exclusive_fsync(seal_path, seal)
    return {
        "parquet_sha256": parquet_sha256,
        "parquet_bytes": int(parquet_path.stat().st_size),
        "seal_sha256": sha256_file(seal_path),
        "rows": int(len(normalized)),
    }


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["truth"].to_numpy(float)
    control = frame["control_prediction"].to_numpy(float)
    selected = frame["selected_prediction"].to_numpy(float)

    def current(rows: np.ndarray) -> dict[str, Any]:
        control_rmse = _rmse(truth[rows], control[rows])
        selected_rmse = _rmse(truth[rows], selected[rows])
        return {
            "rows": int(rows.sum()),
            "control_rmse": control_rmse,
            "selected_rmse": selected_rmse,
            "delta_rmse": selected_rmse - control_rmse,
            "improvement_rmse": control_rmse - selected_rmse,
        }

    all_rows = np.ones(len(frame), dtype=bool)
    return {
        "pooled": current(all_rows),
        "by_block": {
            block: current(frame["block"].astype(str).eq(block).to_numpy())
            for block in OUTER_BLOCKS
        },
        "by_layer": {
            str(layer): current(frame["layer"].astype(int).eq(layer).to_numpy())
            for layer in TARGET_LAYERS
        },
    }


def paired_kst_day_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int = 5000,
    seed: int = 20260821,
) -> dict[str, Any]:
    work = frame.loc[:, ["time", "truth", "control_prediction", "selected_prediction"]].copy()
    work["day"] = pd.to_datetime(work["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.date
    work["control_se"] = (work["control_prediction"] - work["truth"]) ** 2
    work["selected_se"] = (work["selected_prediction"] - work["truth"]) ** 2
    daily = work.groupby("day", sort=True).agg(
        control_se=("control_se", "sum"),
        selected_se=("selected_se", "sum"),
        rows=("truth", "size"),
    )
    rng = np.random.default_rng(seed)
    count = len(daily)
    deltas = np.empty(replicates, dtype=np.float64)
    control_se = daily["control_se"].to_numpy(float)
    selected_se = daily["selected_se"].to_numpy(float)
    rows = daily["rows"].to_numpy(float)
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        sample = rng.integers(0, count, size=(size, count))
        sampled_rows = rows[sample].sum(axis=1)
        control_rmse = np.sqrt(control_se[sample].sum(axis=1) / sampled_rows)
        selected_rmse = np.sqrt(selected_se[sample].sum(axis=1) / sampled_rows)
        deltas[start : start + size] = selected_rmse - control_rmse
    lower, upper = np.quantile(deltas, [0.05, 0.95])
    return {
        "unit": "paired KST day",
        "day_count": int(count),
        "replicates": replicates,
        "seed": seed,
        "ci_level": 0.9,
        "delta_rmse_ci_lower": float(lower),
        "delta_rmse_ci_upper": float(upper),
    }


def outer_promotion_decision(
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    pooled = metrics["pooled"]
    improved_blocks = sum(row["delta_rmse"] < 0.0 for row in metrics["by_block"].values())
    worst_layer_delta = max(row["delta_rmse"] for row in metrics["by_layer"].values())
    checks = {
        "pooled_improvement_at_least_0_005c": pooled["improvement_rmse"] >= 0.005,
        "ci90_upper_below_zero": bootstrap["delta_rmse_ci_upper"] < 0.0,
        "at_least_two_of_three_outer_blocks_improve": improved_blocks >= 2,
        "worst_layer_delta_at_most_plus_0_01c": worst_layer_delta <= 0.01,
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "improved_outer_blocks": int(improved_blocks),
        "worst_layer_delta_rmse": float(worst_layer_delta),
    }
