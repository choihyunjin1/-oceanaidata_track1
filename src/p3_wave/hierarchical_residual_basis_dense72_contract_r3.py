"""Fail-closed Gen5r3 contract correcting the three Gen5r2 QA findings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

import p3_wave.hierarchical_residual_basis_dense72_contract_r1 as predecessor_guard
from ocean_goal.meaningful_score_ledger_v5 import validate_ledger
from p3_wave.corrected_repeated_forward import CorrectedFold
from p3_wave.dense72_targets_r1 import Dense72TargetAccessor, sha256_file
from p3_wave.hierarchical_residual_basis import (
    HierarchicalResidualBasisConfig,
    HierarchicalResidualBasisForecaster,
    prepare_hierarchical_context,
)
from p3_wave.meaningful_learning_curve import PREFIX_FRACTIONS, chronological_prefix_ids
from p3_wave.models import compact_feature_columns
from p3_wave.revin_patch import RAW_COLUMNS, validate_raw_context

CONFIG_RELATIVE = "configs/experiments/p3_hierarchical_residual_basis_dense72_r3.json"
CONFIG_SHA256 = "b874ee578a34741d31ae0779229617fc65d0a187cd6ef2b6699acf45d95549c4"
STAGE = "P3_HIERARCHICAL_RESIDUAL_BASIS_GEN5R3_DENSE72_R1"
COMPARISON_MODE = "STRUCTURE_MATCHED_FRESH_REFIT_PENDING_OFFICIAL_PAIRED_AB"
FOLD_ORDER = ("2024_h2_storm", "winter_transition", "2025_h1")
EXPECTED_ROLES = frozenset(
    {
        "CONFIG",
        "GUARD",
        "ENGINE",
        "RUNNER",
        "TESTS",
        "RUNNER_TESTS",
        "R2_CONFIG",
        "R2_TARGET_ACCESSOR",
        "R2_MODEL",
        "R2_GUARD",
        "R2_ENGINE",
        "R2_RUNNER",
        "R2_MODEL_TESTS",
        "R2_RUNNER_TESTS",
    }
)


class Dense72R3ContractError(RuntimeError):
    """Raised when the append-only r3 contract drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def strict_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Dense72R3ContractError("JSON document must be an object")
    return value


def _origin_url(root: Path) -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def workspace_path(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise PermissionError("r3 path must be non-traversing and workspace-relative")
    workspace = root.resolve(strict=True)
    path = (workspace / candidate).resolve(strict=must_exist)
    if path != workspace and workspace not in path.parents:
        raise PermissionError("r3 path escaped the canonical workspace")
    return path


def _canonical_workspace(root: Path, config: dict[str, Any] | None = None) -> Path:
    workspace = root.resolve(strict=True)
    if Path.cwd().resolve(strict=True) != workspace:
        raise PermissionError("r3 must run from the supplied canonical workspace root")
    if not (workspace / ".git").is_dir():
        raise PermissionError("canonical workspace lacks its Git boundary")
    expected = config or _read_config_unchecked(workspace)
    identity = expected["canonical_workspace_identity"]
    stat = workspace.stat()
    if (
        int(stat.st_dev) != int(identity["root_st_dev"])
        or int(stat.st_ino) != int(identity["root_st_ino"])
        or sha256_file(workspace / ".git/config") != identity["git_config_sha256"]
        or _origin_url(workspace) != identity["origin_url"]
    ):
        raise PermissionError("canonical workspace identity differs")
    return workspace


def _pin(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _read_config_unchecked(root: Path) -> dict[str, Any]:
    path = workspace_path(root, CONFIG_RELATIVE)
    if sha256_file(path) != CONFIG_SHA256:
        raise Dense72R3ContractError("canonical r3 config byte SHA differs")
    return strict_json_object(path)


def _validate_overlay(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "created_at_kst",
        "status",
        "problem",
        "comparison_mode",
        "exact_official_incumbent_comparison",
        "local_numeric_curve_qualification_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
        "canonical_paths",
        "canonical_workspace_identity",
        "predecessor_r2",
        "scientific_structure",
        "implementation_roles",
        "target_free_split",
        "correction_contract",
        "execution_policy",
        "qa_receipt_contract",
        "authorization_contract",
        "static_counters",
    }
    if set(config) != required:
        raise Dense72R3ContractError("r3 config top-level surface changed")
    if (
        config["schema_version"] != "p3_hierarchical_residual_basis.gen5r3_dense72.r1"
        or config["experiment_id"] != "p3_hierarchical_residual_basis_dense72_r3"
        or config["problem"] != "P3"
        or config["comparison_mode"] != COMPARISON_MODE
        or config["exact_official_incumbent_comparison"] is not False
        or config["local_numeric_curve_qualification_allowed"] is not True
        or config["official_promotion_allowed"] is not False
        or config["candidate_or_test_prediction_allowed"] is not False
        or config["upload_allowed"] is not False
        or set(config["implementation_roles"]) != EXPECTED_ROLES
    ):
        raise Dense72R3ContractError("r3 identity or role surface changed")
    correction = config["correction_contract"]
    required_true = (
        "attempt_lock_created_and_deep_verified_before_capability_mint",
        "capability_binds_attempt_lock_sha256",
        "capability_binds_canonical_output_stage",
        "capability_binds_operational_snapshot_canonical_json_and_sha256",
        "full_predecessor_transitive_dependency_closure_pinned",
        "direct_engine_requires_live_locked_phase",
        "direct_private_curve_requires_single_use_curve_phase",
        "preflight_full_train_wave_hs_float_decode_allowed",
        "full_preflight_poison_test_required",
        "duplicate_and_partial_write_tests_required",
    )
    if not all(
        correction[name] is True
        for name in required_true
        if name != "preflight_full_train_wave_hs_float_decode_allowed"
    ) or correction["preflight_full_train_wave_hs_float_decode_allowed"] is not False:
        raise Dense72R3ContractError("r3 correction semantics changed")
    if correction["capability_replay_allowed"] is not False:
        raise Dense72R3ContractError("r3 capability replay must remain forbidden")
    if (
        correction["cell_training_current_hs_decode_mode"]
        != "selective_train_ids_only"
        or correction[
            "unreleased_validation_current_hs_float_decodes_before_raw_fold_commitment"
        ]
        != 0
        or correction[
            "raw_delta_fold_commitment_precedes_validation_current_hs_decode"
        ]
        is not True
        or correction["disclosed_input_cache_current_hs_allowed_for_model_features"]
        is not True
    ):
        raise Dense72R3ContractError("r3 selective current-hs semantics changed")


def _scientific_surface(predecessor: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    keys = config["scientific_structure"]["deep_equal_keys"]
    result = {str(key): predecessor[str(key)] for key in keys}
    digest = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if digest != config["scientific_structure"]["deep_sha256"]:
        raise Dense72R3ContractError("r3 scientific structure differs from r2")
    return result


def _verify_predecessor_files(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, expected in config["predecessor_r2"]["files"].items():
        observed = _pin(workspace_path(root, expected["path"]), root)
        if observed != expected:
            raise PermissionError(f"preserved r2 bytes changed: {role}")
        result[role] = observed
    predecessor_paths = predecessor_guard.stage_paths(
        root,
        predecessor_guard.load_canonical_config(root),
    )
    if any(predecessor_paths[name].exists() for name in predecessor_paths):
        raise PermissionError("NO-GO r2 unexpectedly acquired control or output state")
    return result


def load_canonical_config(
    root: Path,
    requested_config: Path | None = None,
    *,
    supplied_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = root.resolve(strict=True)
    config = _read_config_unchecked(workspace)
    workspace = _canonical_workspace(workspace, config)
    _validate_overlay(config)
    if requested_config is not None and requested_config.resolve(strict=True) != workspace_path(
        workspace, CONFIG_RELATIVE
    ):
        raise PermissionError("noncanonical r3 config was requested")
    if supplied_config is not None and canonical_json_bytes(supplied_config) != canonical_json_bytes(
        config
    ):
        raise PermissionError("supplied r3 config differs from canonical bytes")
    predecessor = predecessor_guard.load_canonical_config(workspace)
    _scientific_surface(predecessor, config)
    _verify_predecessor_files(workspace, config)
    return config, predecessor


def implementation_pins(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    workspace = _canonical_workspace(root, config)
    pins = {
        role: _pin(workspace_path(workspace, relative), workspace)
        for role, relative in config["implementation_roles"].items()
    }
    predecessor = predecessor_guard.load_canonical_config(workspace)
    transitive = {
        f"R2_TRANSITIVE_{role}": _pin(
            workspace_path(workspace, relative), workspace
        )
        for role, relative in predecessor["implementation_roles"].items()
    }
    overlap = set(pins).intersection(transitive)
    if overlap or len(transitive) != len(predecessor["implementation_roles"]):
        raise PermissionError("r3 transitive implementation role expansion changed")
    return {**pins, **transitive}


def stage_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    workspace = _canonical_workspace(root, config)
    return {
        key: workspace_path(workspace, config["canonical_paths"][key], must_exist=False)
        for key in (
            "output",
            "control",
            "pre_execution_qa",
            "authorization",
            "attempt_lock",
            "run_failure_receipt",
        )
    }


def _ids_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _array_sha(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _metadata_sha(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, ["fold", "anchor_id", "station", "episode_id"]].sort_values(
        ["fold", "anchor_id"]
    )
    return hashlib.sha256(
        ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _station_episode_sha(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, ["anchor_id", "station", "episode_id"]].sort_values("anchor_id")
    return hashlib.sha256(
        ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _minimum_gap(train: pd.DataFrame, validation: pd.DataFrame) -> float:
    minima: list[float] = []
    for station, current in validation.groupby("station", sort=True, observed=True):
        times = train.loc[train["station"].eq(station), "anchor_time"]
        for timestamp in current["anchor_time"]:
            minima.append(float((times - timestamp).abs().min().total_seconds() / 3600.0))
    return min(minima)


class SelectiveCurrentHsAccessor:
    """Decode anchor-current train-wave hs only for authorized rolling train IDs."""

    def __init__(
        self,
        dense_accessor: Dense72TargetAccessor,
        anchors: pd.DataFrame,
        *,
        validation_groups: dict[str, np.ndarray],
    ) -> None:
        if set(anchors.columns) != {"anchor_id", "station", "anchor_time"}:
            raise ValueError("current-hs accessor received a non-target-free anchor surface")
        ordered = anchors.sort_values("anchor_id").reset_index(drop=True)
        if not np.array_equal(ordered["anchor_id"].to_numpy(np.int64), np.arange(len(ordered))):
            raise ValueError("current-hs anchor row identity changed")
        station_codes = ordered["station"].map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2})
        if station_codes.isna().any():
            raise ValueError("current-hs anchor station changed")
        times = pd.DatetimeIndex(
            pd.to_datetime(ordered["anchor_time"], utc=True, errors="raise")
        ).as_unit("ns")
        source_rows = np.full(len(ordered), -1, dtype=np.int64)
        for code in (0, 1, 2):
            local = np.flatnonzero(station_codes.to_numpy(np.int8) == code)
            source_times = dense_accessor._station_times[code]
            source_ids = dense_accessor._station_rows[code]
            query = times.asi8[local]
            positions = np.searchsorted(source_times, query)
            safe = np.minimum(positions, len(source_times) - 1)
            found = (positions < len(source_times)) & (source_times[safe] == query)
            if not found.all():
                raise ValueError("anchor current time is absent from train_wave identity")
            source_rows[local] = source_ids[positions]
        if (source_rows < 0).any() or not dense_accessor._source_available[source_rows].all():
            raise ValueError("anchor current train-wave hs is unavailable")
        self._dense = dense_accessor
        self._source_rows = source_rows
        self._validation_ids = {
            str(name): np.asarray(values, dtype=np.int64).copy()
            for name, values in validation_groups.items()
        }
        if set(self._validation_ids) != set(validation_groups):
            raise ValueError("current-hs validation labels changed")
        merged = np.concatenate(list(self._validation_ids.values()))
        if np.unique(merged).size != len(merged):
            raise PermissionError("current-hs validation IDs overlap")
        self._validation_source_rows = {
            name: self._source_rows[case_ids].copy()
            for name, case_ids in self._validation_ids.items()
        }
        self._released: set[str] = set()
        self._commitments: dict[str, str] = {}
        self._cache: dict[int, float] = {}
        self._total_decodes = 0
        self._forbidden_decodes = 0

    def _ids(self, values: np.ndarray, *, role: str) -> np.ndarray:
        result = np.asarray(values)
        if (
            result.ndim != 1
            or len(result) == 0
            or not np.issubdtype(result.dtype, np.integer)
        ):
            raise ValueError(f"{role} IDs must be a nonempty integer vector")
        result = result.astype(np.int64, copy=False)
        if (
            np.unique(result).size != len(result)
            or result.min() < 0
            or result.max() >= len(self._source_rows)
        ):
            raise ValueError(f"{role} IDs are duplicated or outside the anchor table")
        return result

    def _unreleased_source_rows(self) -> set[int]:
        return {
            int(self._source_rows[case_id])
            for name, case_ids in self._validation_ids.items()
            if name not in self._released
            for case_id in case_ids
        }

    def assert_training_target_current_isolation(
        self, train_case_ids: np.ndarray
    ) -> None:
        train = self._ids(train_case_ids, role="dense-target current-isolation train")
        rows, mask = self._dense._locate(train)
        overlap = self._unreleased_source_rows().intersection(
            int(value) for value in rows[mask]
        )
        if overlap:
            self._forbidden_decodes += len(overlap)
            raise PermissionError(
                "training future targets intersect unreleased validation current hs"
            )

    def _decode_rows(self, source_rows: np.ndarray) -> np.ndarray:
        result = np.empty(len(source_rows), dtype=np.float64)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        descriptor = os.open(self._dense.wave_path, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                descriptor = -1
                for index, source_row in enumerate(source_rows):
                    row = int(source_row)
                    if row not in self._cache:
                        stream.seek(int(self._dense._source_offsets[row]))
                        fields = stream.readline().rstrip(b"\r\n").split(b",")
                        if len(fields) != 6 or not fields[2]:
                            raise ValueError("selected current-hs source row is malformed")
                        value = float(fields[2].decode("ascii"))
                        if not np.isfinite(value):
                            raise ValueError("selected current-hs scalar is non-finite")
                        self._cache[row] = value
                        self._total_decodes += 1
                    result[index] = self._cache[row]
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return result

    def load_training_current_hs(
        self,
        train_case_ids: np.ndarray,
        *,
        active_validation_case_ids: np.ndarray,
    ) -> np.ndarray:
        train = self._ids(train_case_ids, role="current-hs train")
        validation = self._ids(active_validation_case_ids, role="current-hs validation")
        matching = [
            name
            for name, values in self._validation_ids.items()
            if np.array_equal(np.sort(values), np.sort(validation))
        ]
        if len(matching) != 1 or matching[0] in self._released:
            raise PermissionError("active current-hs validation group is invalid")
        if np.intersect1d(train, validation).size:
            raise PermissionError("current-hs train and validation IDs overlap")
        rows = self._source_rows[train]
        overlap = self._unreleased_source_rows().intersection(int(value) for value in rows)
        if overlap:
            self._forbidden_decodes += len(overlap)
            raise PermissionError("train current-hs request intersects unreleased validation")
        return self._decode_rows(rows)

    def release_validation_group(self, name: str, *, fold_commitment_sha256: str) -> None:
        label = str(name)
        digest = str(fold_commitment_sha256)
        if label not in self._validation_ids or label in self._released:
            raise PermissionError("current-hs validation group cannot be released")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("current-hs fold commitment is not a lowercase SHA-256")
        self._released.add(label)
        self._commitments[label] = digest

    def load_released_validation_current_hs(self, name: str) -> np.ndarray:
        label = str(name)
        if label not in self._released:
            raise PermissionError("validation current hs requires a prior blind fold commitment")
        return self._decode_rows(self._source_rows[self._validation_ids[label]])

    def validation_group_scalar_decodes(self, name: str) -> int:
        label = str(name)
        if label not in self._validation_source_rows:
            raise KeyError("unknown current-hs validation group")
        return len(set(int(row) for row in self._validation_source_rows[label]) & self._cache.keys())

    def validation_group_process_scalar_decodes(self, name: str) -> int:
        label = str(name)
        if label not in self._validation_source_rows:
            raise KeyError("unknown current-hs validation group")
        decoded = set(self._cache) | set(self._dense._decoded_rows)
        return len(set(int(row) for row in self._validation_source_rows[label]) & decoded)

    @property
    def total_scalar_decodes(self) -> int:
        return self._total_decodes

    @property
    def forbidden_scalar_decodes(self) -> int:
        return self._forbidden_decodes

    @property
    def released_groups(self) -> tuple[str, ...]:
        return tuple(sorted(self._released))

    def access_audit(self) -> dict[str, Any]:
        return {
            "source": "train_wave.csv_anchor_current_hs_selective_only",
            "registered_validation_groups": sorted(self._validation_ids),
            "released_validation_groups": sorted(self._released),
            "release_commitment_sha256": dict(sorted(self._commitments.items())),
            "unique_current_hs_scalar_decodes": self._total_decodes,
            "forbidden_validation_current_hs_scalar_decodes": self._forbidden_decodes,
            "validation_group_current_hs_scalar_decodes": {
                name: self.validation_group_scalar_decodes(name)
                for name in sorted(self._validation_ids)
            },
            "validation_group_process_wide_current_hs_scalar_decodes": {
                name: self.validation_group_process_scalar_decodes(name)
                for name in sorted(self._validation_ids)
            },
        }


def build_target_free_folds(
    anchors: pd.DataFrame,
    validation_keys: pd.DataFrame,
    predecessor: dict[str, Any],
    config: dict[str, Any],
) -> tuple[tuple[CorrectedFold, ...], pd.DataFrame, dict[str, Any]]:
    if set(anchors.columns) != {"anchor_id", "station", "anchor_time"}:
        raise ValueError("target-free anchor surface changed")
    if set(validation_keys.columns) != {"fold", "anchor_id", "station", "episode_id"}:
        raise ValueError("pinned target-free validation metadata surface changed")
    source = anchors.sort_values("anchor_id").reset_index(drop=True).copy()
    if not np.array_equal(source["anchor_id"].to_numpy(np.int64), np.arange(24_360)):
        raise ValueError("target-free anchors differ from canonical row identity")
    source["anchor_time"] = pd.to_datetime(source["anchor_time"], utc=True, errors="raise")
    keys = validation_keys.copy()
    if len(keys) != 181 or keys["anchor_id"].duplicated().any():
        raise ValueError("pinned target-free validation rows differ")
    selected = keys.merge(source, on=["anchor_id", "station"], validate="one_to_one")
    selected = selected.sort_values(["anchor_time", "station", "anchor_id"]).reset_index(
        drop=True
    )
    split = config["target_free_split"]
    if _metadata_sha(selected) != split["canonical_rows_csv_sha256"]:
        raise PermissionError("pinned target-free validation metadata changed")
    folds: list[CorrectedFold] = []
    summaries: dict[str, Any] = {}
    windows = predecessor["validation"]["windows"]
    for name, start, end in windows:
        fold_pin = split["folds"][name]
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        validation = selected.loc[selected["fold"].eq(name)].copy()
        if not validation["anchor_time"].ge(start_ts).all() or not validation[
            "anchor_time"
        ].lt(end_ts).all():
            raise PermissionError("pinned validation IDs escaped their historical window")
        validation_ids = np.sort(validation["anchor_id"].to_numpy(np.int64))
        train_end = start_ts - pd.Timedelta(hours=78)
        train = source.loc[source["anchor_time"].lt(train_end)].copy()
        train_ids = np.sort(train["anchor_id"].to_numpy(np.int64))
        if (
            len(validation_ids) != fold_pin["validation_count"]
            or _ids_sha(validation_ids) != fold_pin["validation_ids_sha256"]
            or _station_episode_sha(validation) != fold_pin["station_episode_rows_sha256"]
            or len(train_ids) != fold_pin["train_count"]
            or _ids_sha(train_ids) != fold_pin["train_ids_sha256"]
            or np.intersect1d(train_ids, validation_ids).size
        ):
            raise PermissionError("target-free pinned fold identity changed")
        minimum_gap = _minimum_gap(train, validation)
        if minimum_gap < 78.0:
            raise PermissionError("target-free fold embargo changed")
        folds.append(
            CorrectedFold(
                name=name,
                train_ids=train_ids,
                validation_ids=validation_ids,
                validation_start=start_ts,
                validation_end=end_ts,
            )
        )
        summaries[name] = {
            "train_anchor_count": len(train_ids),
            "validation_case_count": len(validation_ids),
            "validation_by_station": {
                str(key): int(value)
                for key, value in validation.groupby("station", observed=True).size().items()
            },
            "removed_same_episode_train_anchors": int(
                split["historical_same_episode_train_removals"][name]
            ),
            "shared_train_validation_station_episode_count": 0,
            "minimum_train_validation_anchor_gap_hours": minimum_gap,
            "minimum_train_validation_footprint_separation_hours": minimum_gap - 72.0,
        }
    station_gap: dict[str, float] = {}
    for station, group in selected.groupby("station", sort=True, observed=True):
        gaps = (
            group.sort_values("anchor_time")["anchor_time"]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(3600.0)
        )
        station_gap[str(station)] = float(gaps.min())
    audit = {
        "selection_rule": "pinned_historical_station_global_ids_no_r3_hs_recompute",
        "historical_label_derived": True,
        "r3_train_wave_hs_float_decodes": 0,
        "validation_case_count": len(selected),
        "validation_row_count": len(selected) * 6,
        "unique_station_episode_count": int(
            selected[["station", "episode_id"]].drop_duplicates().shape[0]
        ),
        "repeated_station_episode_count": int(
            selected.duplicated(["station", "episode_id"]).sum()
        ),
        "station_global_minimum_gap_hours": station_gap,
        "cross_window_pairs_below_78h": 0,
        "context48_plus_target24_footprint_overlap_pairs": 0,
        "footprint_definition": "[anchor-48h, anchor+24h] within station",
        "folds": summaries,
    }
    if audit["repeated_station_episode_count"] != 0 or min(station_gap.values()) < 78.0:
        raise PermissionError("pinned historical validation isolation changed")
    return tuple(folds), selected, audit


def _anchor_identity_sha(anchors: pd.DataFrame) -> str:
    ordered = anchors.sort_values("anchor_id")
    digest = hashlib.sha256()
    digest.update(ordered["anchor_id"].to_numpy(dtype="<i8").tobytes())
    times = pd.DatetimeIndex(ordered["anchor_time"]).as_unit("ns").asi8.astype("<i8")
    digest.update(times.tobytes())
    digest.update("\n".join(ordered["station"].astype(str)).encode("utf-8"))
    return digest.hexdigest()


def _dense_accessor_deep_snapshot(accessor: Dense72TargetAccessor) -> dict[str, Any]:
    return {
        "anchor_station_sha256": _array_sha(accessor._anchor_station, dtype="|i1"),
        "anchor_time_sha256": _array_sha(accessor._anchor_time_ns, dtype="<i8"),
        "placeholder_current_hs_sha256": _array_sha(
            accessor._current_hs, dtype="<f8"
        ),
        "source_offsets_sha256": _array_sha(accessor._source_offsets, dtype="<i8"),
        "source_available_sha256": _array_sha(
            accessor._source_available, dtype="|b1"
        ),
        "station_indices": {
            str(code): {
                "times_sha256": _array_sha(accessor._station_times[code], dtype="<i8"),
                "rows_sha256": _array_sha(accessor._station_rows[code], dtype="<i8"),
            }
            for code in sorted(accessor._station_times)
        },
        "validation_ids_sha256": {
            name: _ids_sha(values)
            for name, values in sorted(accessor._validation_case_ids.items())
        },
        "validation_target_rows_sha256": {
            name: _ids_sha(np.asarray(sorted(values), dtype=np.int64))
            for name, values in sorted(accessor._validation_target_rows.items())
        },
    }


def _current_accessor_deep_snapshot(accessor: SelectiveCurrentHsAccessor) -> dict[str, Any]:
    return {
        "source_rows_sha256": _ids_sha(accessor._source_rows),
        "validation_ids_sha256": {
            name: _ids_sha(values)
            for name, values in sorted(accessor._validation_ids.items())
        },
        "validation_source_rows_sha256": {
            name: _ids_sha(values)
            for name, values in sorted(accessor._validation_source_rows.items())
        },
    }


def operational_snapshot(preflight: dict[str, Any]) -> dict[str, Any]:
    compact = preflight["compact"]
    station = preflight["station"]
    anchors = preflight["anchors"]
    folds = preflight["folds"]
    prefixes = preflight["prefix_ids"]
    accessor = preflight["target_accessor"]
    current_accessor = preflight["current_hs_accessor"]
    scientific = _scientific_surface(preflight["predecessor"], preflight["config"])
    scientific_sha = hashlib.sha256(canonical_json_bytes(scientific)).hexdigest()
    return {
        "schema_version": "p3_gen5r3.operational_snapshot.r1",
        "config_sha256": CONFIG_SHA256,
        "config_object_sha256": hashlib.sha256(
            canonical_json_bytes(preflight["config"])
        ).hexdigest(),
        "predecessor_object_sha256": hashlib.sha256(
            canonical_json_bytes(preflight["predecessor"])
        ).hexdigest(),
        "scientific_structure_sha256": scientific_sha,
        "input_snapshot": preflight["input_snapshot"],
        "implementation_pins": implementation_pins(preflight["root"], preflight["config"]),
        "raw": {
            "shape": list(preflight["raw"].shape),
            "dtype": str(preflight["raw"].dtype),
            "source_sha256": preflight["input_snapshot"]["sequence_cache/train_values.npy"][
                "sha256"
            ],
        },
        "station": {
            "shape": list(station.shape),
            "dtype": str(station.dtype),
            "sha256": _array_sha(station, dtype="<i8"),
        },
        "compact": {
            "shape": list(compact.shape),
            "dtype": str(compact.dtype),
            "sha256": _array_sha(compact, dtype="<f4"),
            "feature_names_sha256": hashlib.sha256(
                canonical_json_bytes(list(preflight["feature_columns"]))
            ).hexdigest(),
        },
        "anchors": {
            "rows": len(anchors),
            "identity_sha256": _anchor_identity_sha(anchors),
        },
        "folds": [
            {
                "name": fold.name,
                "train_count": len(fold.train_ids),
                "train_ids_sha256": _ids_sha(fold.train_ids),
                "validation_count": len(fold.validation_ids),
                "validation_ids_sha256": _ids_sha(fold.validation_ids),
                "validation_start": pd.Timestamp(fold.validation_start).isoformat(),
                "validation_end": pd.Timestamp(fold.validation_end).isoformat(),
            }
            for fold in folds
        ],
        "prefix_ids": {
            f"{fraction:.2f}": {
                fold.name: {
                    "count": len(prefixes[fraction][fold.name]),
                    "sha256": _ids_sha(prefixes[fraction][fold.name]),
                }
                for fold in folds
            }
            for fraction in PREFIX_FRACTIONS
        },
        "target_accessor": accessor.access_audit(),
        "target_accessor_deep": _dense_accessor_deep_snapshot(accessor),
        "current_hs_accessor": current_accessor.access_audit(),
        "current_hs_accessor_deep": _current_accessor_deep_snapshot(current_accessor),
        "gen1_metrics_object_sha256": hashlib.sha256(
            canonical_json_bytes(preflight["gen1_metrics"])
        ).hexdigest(),
        "current_hs_feature_index": int(preflight["current_hs_feature_index"]),
        "preflight_train_wave_hs_float_decodes": 0,
    }


def _build_preflight(
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    predecessor: dict[str, Any],
) -> dict[str, Any]:
    workspace = _canonical_workspace(root, config)
    paths = stage_paths(workspace, config)
    if any(paths[name].exists() for name in ("output", "attempt_lock", "run_failure_receipt")):
        raise FileExistsError("r3 append-only state is already consumed")
    predecessor_guard.verify_runtime_environment(workspace, predecessor)
    input_paths, input_snapshot = predecessor_guard.verify_input_pins(
        workspace, data_dir, predecessor
    )
    records = validate_ledger(workspace, input_paths["v5/registry.jsonl"])
    ledger = predecessor["central_ledger_anchor"]
    if len(records) != ledger["event_count"] or records[-1]["event_sha256"] != ledger[
        "head_event_sha256"
    ]:
        raise PermissionError("central ledger anchor changed")

    anchors = pd.read_parquet(
        input_paths["compact_cache/train_anchors.parquet"],
        columns=["anchor_id", "station", "anchor_time"],
    )
    features = pd.read_parquet(input_paths["compact_cache/train_features.parquet"])
    if anchors.shape != (24_360, 3) or features.shape != (24_360, 1_277):
        raise ValueError("target-free anchor or compact source shape changed")
    ids = np.arange(24_360, dtype=np.int64)
    if not np.array_equal(anchors["anchor_id"].to_numpy(np.int64), ids) or not np.array_equal(
        features["anchor_id"].to_numpy(np.int64), ids
    ):
        raise ValueError("anchor and compact row identities changed")
    if not anchors["station"].astype(str).equals(features["station"].astype(str)):
        raise ValueError("anchor and compact station identities changed")
    feature_columns = tuple(compact_feature_columns(list(features.columns)))
    if len(feature_columns) != 591 or "hs_current" not in feature_columns or any(
        name.startswith("target_") for name in feature_columns
    ):
        raise ValueError("compact input-only feature surface changed")
    compact = features.loc[:, feature_columns].to_numpy(np.float32)
    del features
    if compact.shape != (24_360, 591):
        raise ValueError("compact 24360x591 matrix changed")
    current_hs_index = feature_columns.index("hs_current")
    if not np.isfinite(compact[:, current_hs_index]).all():
        raise ValueError("disclosed current-hs input feature is non-finite")

    raw = np.load(input_paths["sequence_cache/train_values.npy"], mmap_mode="r")
    station = np.load(input_paths["sequence_cache/train_station.npy"], mmap_mode="r")
    if raw.shape != (24_360, 289, 10) or raw.dtype != np.float32:
        raise ValueError("raw sequence cache changed")
    if station.shape != (24_360,) or station.dtype != np.int64:
        raise ValueError("station sequence cache changed")
    validate_raw_context(torch.from_numpy(np.array(raw[:8], copy=True)))
    raw_hs_index = RAW_COLUMNS.index("hs")
    raw_current_hs = np.asarray(raw[:, -1, raw_hs_index], dtype=np.float32)
    if not np.array_equal(compact[:, current_hs_index], raw_current_hs):
        raise ValueError("disclosed compact and sequence current-hs inputs differ")

    validation_keys = pd.read_parquet(input_paths["gen4/validation_keys.parquet"])
    folds, selected, split_audit = build_target_free_folds(
        anchors, validation_keys, predecessor, config
    )
    if tuple(fold.name for fold in folds) != FOLD_ORDER:
        raise ValueError("target-free fold order changed")
    prefix_ids: dict[float, dict[str, np.ndarray]] = {}
    prefix_audit: dict[str, Any] = {}
    lookup = anchors.set_index("anchor_id")
    for fraction in PREFIX_FRACTIONS:
        prefix_ids[fraction] = {}
        tag = f"{int(round(fraction * 100)):03d}"
        prefix_audit[tag] = {}
        for fold in folds:
            current = chronological_prefix_ids(anchors, fold.train_ids, fraction)
            prefix_ids[fraction][fold.name] = current
            gap = float(
                (
                    pd.Timestamp(fold.validation_start)
                    - pd.to_datetime(lookup.loc[current, "anchor_time"], utc=True).max()
                ).total_seconds()
                / 3600.0
            )
            prefix_audit[tag][fold.name] = {
                "count": len(current),
                "full_count": len(fold.train_ids),
                "id_sha256_little_endian_int64": _ids_sha(current),
                "nested_subset_of_safe_outer_train": bool(
                    np.isin(current, fold.train_ids).all()
                ),
                "maximum_anchor_before_validation_start_hours": gap,
            }
    leakage_checks = {
        "station_global_validation_gap_at_least_78h": all(
            value >= 78.0 for value in split_audit["station_global_minimum_gap_hours"].values()
        ),
        "validation_station_episode_reuse_zero": split_audit[
            "repeated_station_episode_count"
        ]
        == 0,
        "validation_72h_footprint_overlap_zero": split_audit[
            "context48_plus_target24_footprint_overlap_pairs"
        ]
        == 0,
        "outer_train_validation_episode_overlap_zero": all(
            row["shared_train_validation_station_episode_count"] == 0
            for row in split_audit["folds"].values()
        ),
        "outer_train_validation_gap_at_least_78h": all(
            row["minimum_train_validation_anchor_gap_hours"] >= 78.0
            for row in split_audit["folds"].values()
        ),
        "all_prefixes_nested_in_safe_outer_train": all(
            row["nested_subset_of_safe_outer_train"]
            for by_fold in prefix_audit.values()
            for row in by_fold.values()
        ),
        "r3_split_recomputation_from_hs_zero": True,
    }
    if not all(leakage_checks.values()):
        raise PermissionError("target-free split leakage checks failed")

    accessor_anchors = anchors.copy()
    accessor_anchors["current_hs"] = np.zeros(len(accessor_anchors), dtype=np.float64)
    accessor = Dense72TargetAccessor(
        input_paths["source/train_wave.csv"],
        accessor_anchors.loc[:, ["anchor_id", "station", "anchor_time", "current_hs"]],
        validation_groups={fold.name: fold.validation_ids for fold in folds},
        expected_source_sha256=predecessor["input_pins"]["source/train_wave.csv"][
            "sha256"
        ],
        expected_source_bytes=predecessor["input_pins"]["source/train_wave.csv"]["bytes"],
    )
    availability = accessor.availability_audit().as_dict()
    if availability["scalar_decodes"] != 0 or accessor.total_scalar_decodes != 0:
        raise PermissionError("r3 preflight decoded a train-wave hs float")
    current_accessor = SelectiveCurrentHsAccessor(
        accessor,
        anchors,
        validation_groups={fold.name: fold.validation_ids for fold in folds},
    )
    if current_accessor.total_scalar_decodes != 0:
        raise PermissionError("r3 preflight decoded an anchor-current hs float")

    gen1_oof = pd.read_parquet(
        input_paths["gen1/learning_curve_oof.parquet"],
        columns=[
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "current_hs",
            "persistence",
            "incumbent_prediction",
            "prefix_fraction",
        ],
    )
    if len(gen1_oof) != 5 * 1_086:
        raise ValueError("sealed comparator key surface changed")
    gen1_metrics = strict_json_object(input_paths["gen1/metrics.json"])
    probe = prepare_hierarchical_context(torch.from_numpy(np.array(raw[:2], copy=True)))
    model_config = HierarchicalResidualBasisConfig(
        static_feature_count=591,
        hidden_width=192,
        conditioning_width=128,
        dropout=0.1,
        context_steps=144,
        input_channels=24,
        forecast_steps=72,
        pooling_factors=(12, 4, 1),
        forecast_knots=(6, 18, 72),
        blocks_per_stack=2,
    )
    parameter_count = HierarchicalResidualBasisForecaster(model_config).trainable_parameter_count
    expected_steps = sum(
        math.ceil(len(prefix_ids[fraction][fold.name]) / 512) * 12
        for fold in folds
        for fraction in PREFIX_FRACTIONS
        for _seed in predecessor["validation"]["seed_replicates"]
    )
    if parameter_count != 4_125_120 or expected_steps != 10_260:
        raise ValueError("scientific model accounting changed")

    scientific = _scientific_surface(predecessor, config)
    scientific_sha = hashlib.sha256(canonical_json_bytes(scientific)).hexdigest()
    preflight: dict[str, Any] = {
        "root": workspace,
        "config": config,
        "predecessor": predecessor,
        "scientific_structure": scientific,
        "scientific_structure_sha256": scientific_sha,
        "input_paths": input_paths,
        "input_snapshot": input_snapshot,
        "anchors": anchors,
        "raw": raw,
        "station": station,
        "compact": compact,
        "feature_columns": feature_columns,
        "current_hs_feature_index": current_hs_index,
        "folds": folds,
        "selected": selected,
        "split_audit": split_audit,
        "prefix_ids": prefix_ids,
        "prefix_audit": prefix_audit,
        "leakage_checks": leakage_checks,
        "target_accessor": accessor,
        "current_hs_accessor": current_accessor,
        "gen1_metrics": gen1_metrics,
    }
    snapshot = operational_snapshot(preflight)
    snapshot_sha = hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()
    summary = {
        "schema_version": "p3_hierarchical_residual_basis.gen5r3_dense72.preflight.r1",
        "status": "PASS_STATIC_IMPLEMENTATION_ONLY_NO_FIT_NO_LOCK",
        "problem": "P3",
        "comparison_mode": COMPARISON_MODE,
        "scientific_structure_deep_equal_to_r2": True,
        "scientific_structure_sha256": scientific_sha,
        "historical_split_label_derived": True,
        "r3_split_or_episode_recomputed_from_hs": False,
        "process_train_wave_hs_float_decodes": 0,
        "validation_current_train_wave_hs_float_decodes_before_blind_commitment": 0,
        "compact": {
            "shape": [24_360, 591],
            "dtype": "float32",
            "matrix_sha256": _array_sha(compact, dtype="<f4"),
            "feature_names_sha256": hashlib.sha256(
                canonical_json_bytes(list(feature_columns))
            ).hexdigest(),
            "target_columns_selected": 0,
        },
        "dense72_availability": availability,
        "current_hs_access": current_accessor.access_audit(),
        "validation": {
            "cases": len(selected),
            "rows": len(selected) * 6,
            "fold_order": list(FOLD_ORDER),
            "split_audit": split_audit,
            "prefix_audit": prefix_audit,
            "leakage_checks": leakage_checks,
        },
        "model": {
            "context_probe_shape": list(probe.values.shape),
            "trainable_parameter_count": parameter_count,
            "actual_fit_cells": 45,
            "optimizer_steps": expected_steps,
            "optimizer_target_surface": "all_available_dense72_train_only_steps",
        },
        "comparator": {
            "reference_seed_full_prediction_exact_to_historical_frozen_oof": False,
            "local_numeric_qualification_allowed": True,
            "official_promotion_requires_future_paired_ab": True,
        },
        "fits": 0,
        "predictions": 0,
        "scores": 0,
        "test_predictions": 0,
        "uploads": 0,
    }
    preflight["summary"] = summary
    preflight["summary_sha256"] = hashlib.sha256(canonical_json_bytes(summary)).hexdigest()
    preflight["implementation_pins"] = implementation_pins(workspace, config)
    preflight["operational_snapshot"] = snapshot
    preflight["operational_snapshot_sha256"] = snapshot_sha
    return preflight


def prepare_execution_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, predecessor = load_canonical_config(
        root, requested_config, supplied_config=supplied_config
    )
    return config, _build_preflight(
        root.resolve(strict=True), data_dir.resolve(strict=True), config, predecessor
    )


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
) -> dict[str, Any]:
    config, preflight = prepare_execution_preflight(
        root, data_dir, requested_config=requested_config
    )
    paths = stage_paths(root, config)
    return {
        **preflight["summary"],
        "config": _pin(workspace_path(root, CONFIG_RELATIVE), root.resolve(strict=True)),
        "predecessor_r2_preserved": config["predecessor_r2"],
        "implementation_pins": preflight["implementation_pins"],
        "operational_snapshot_sha256": preflight["operational_snapshot_sha256"],
        "static_preflight_sha256": preflight["summary_sha256"],
        "control_state": {name: path.exists() for name, path in paths.items()},
        "files_written": 0,
        "attempt_locks_created": 0,
    }


def _write_all_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            try:
                written = os.write(descriptor, view[offset:])
            except InterruptedError:
                continue
            if written <= 0 or written > len(view) - offset:
                raise OSError("exclusive write made invalid progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    _write_all_exclusive(path, canonical_json_bytes(value) + b"\n")


def verify_pre_execution_qa(
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = stage_paths(root, config)["pre_execution_qa"]
    if not path.is_file():
        raise PermissionError("independent r3 pre-execution QA receipt is missing")
    receipt = strict_json_object(path)
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "reviewer",
        "decision",
        "p0_count",
        "p1_count",
        "config",
        "implementation_pins",
        "static_preflight_sha256",
        "operational_snapshot_sha256",
        "predecessor_r2_preserved",
        "lock_before_capability_verified",
        "target_free_preflight_verified",
        "exclusive_parquet_verified",
        "notes",
    }
    contract = config["qa_receipt_contract"]
    checks = {
        "keys": set(receipt) == expected_keys,
        "schema": receipt.get("schema_version") == contract["schema_version"],
        "reviewer": bool(receipt.get("reviewer")),
        "decision": receipt.get("decision") == contract["decision"],
        "p0": receipt.get("p0_count") == 0,
        "p1": receipt.get("p1_count") == 0,
        "config": receipt.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation": receipt.get("implementation_pins")
        == implementation_pins(root, config),
        "preflight": receipt.get("static_preflight_sha256")
        == preflight["summary_sha256"],
        "operational": receipt.get("operational_snapshot_sha256")
        == preflight["operational_snapshot_sha256"],
        "predecessor": receipt.get("predecessor_r2_preserved") is True,
        "lock": receipt.get("lock_before_capability_verified") is True,
        "target_free": receipt.get("target_free_preflight_verified") is True,
        "parquet": receipt.get("exclusive_parquet_verified") is True,
        "notes": isinstance(receipt.get("notes"), list) and bool(receipt["notes"]),
    }
    if not all(checks.values()):
        raise PermissionError(
            f"r3 QA receipt failed: {sorted(k for k, value in checks.items() if not value)}"
        )
    return receipt, sha256_file(path)


def verify_execution_authorization(
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    *,
    qa_sha256: str,
) -> tuple[dict[str, Any], str]:
    paths = stage_paths(root, config)
    if any(paths[name].exists() for name in ("output", "attempt_lock", "run_failure_receipt")):
        raise FileExistsError("r3 one-shot state is already consumed")
    if not paths["authorization"].is_file():
        raise PermissionError("separate r3 execution authorization is missing")
    authorization = strict_json_object(paths["authorization"])
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "stage",
        "config",
        "authorization",
        "user_message_reference",
        "qa_receipt",
        "implementation_pins",
        "static_preflight_sha256",
        "operational_snapshot_sha256",
        "curve_execution_authorized",
        "full_fit_or_candidate_authorized",
        "test_prediction_authorized",
        "upload_authorized",
    }
    contract = config["authorization_contract"]
    checks = {
        "keys": set(authorization) == expected_keys,
        "schema": authorization.get("schema_version") == contract["schema_version"],
        "stage": authorization.get("stage") == STAGE,
        "config": authorization.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "phrase": authorization.get("authorization")
        == contract["authorization_phrase_prefix"] + CONFIG_SHA256,
        "user": bool(authorization.get("user_message_reference")),
        "qa": authorization.get("qa_receipt")
        == {"path": config["canonical_paths"]["pre_execution_qa"], "sha256": qa_sha256},
        "implementation": authorization.get("implementation_pins")
        == implementation_pins(root, config),
        "preflight": authorization.get("static_preflight_sha256")
        == preflight["summary_sha256"],
        "operational": authorization.get("operational_snapshot_sha256")
        == preflight["operational_snapshot_sha256"],
        "curve": authorization.get("curve_execution_authorized") is True,
        "full_fit": authorization.get("full_fit_or_candidate_authorized") is False,
        "test": authorization.get("test_prediction_authorized") is False,
        "upload": authorization.get("upload_authorized") is False,
    }
    if not all(checks.values()):
        raise PermissionError(
            f"r3 authorization failed: {sorted(k for k, value in checks.items() if not value)}"
        )
    return authorization, sha256_file(paths["authorization"])


def _validate_live_preflight(root: Path, config: dict[str, Any], preflight: dict[str, Any]) -> None:
    canonical_config, canonical_predecessor = load_canonical_config(root)
    if (
        canonical_json_bytes(config) != canonical_json_bytes(canonical_config)
        or canonical_json_bytes(preflight["config"])
        != canonical_json_bytes(canonical_config)
        or canonical_json_bytes(preflight["predecessor"])
        != canonical_json_bytes(canonical_predecessor)
    ):
        raise PermissionError("r3 live config or predecessor object changed")
    predecessor_guard.verify_runtime_environment(root, preflight["predecessor"])
    input_paths = preflight["input_paths"]
    expected_inputs = preflight["input_snapshot"]
    if set(input_paths) != set(expected_inputs):
        raise PermissionError("r3 live input key surface changed")
    for label, path in input_paths.items():
        source = Path(path).resolve(strict=True)
        expected = expected_inputs[label]
        if source.stat().st_size != expected["bytes"] or sha256_file(source) != expected[
            "sha256"
        ]:
            raise PermissionError(f"r3 live input pin changed: {label}")
    live_gen1_metrics = strict_json_object(input_paths["gen1/metrics.json"])
    if canonical_json_bytes(live_gen1_metrics) != canonical_json_bytes(
        preflight["gen1_metrics"]
    ):
        raise PermissionError("r3 live Gen1 comparator metrics object changed")
    if hashlib.sha256(canonical_json_bytes(preflight["summary"])).hexdigest() != preflight[
        "summary_sha256"
    ]:
        raise PermissionError("r3 preflight summary changed")
    if implementation_pins(root, config) != preflight["implementation_pins"]:
        raise PermissionError("r3 implementation changed after preflight")
    current = operational_snapshot(preflight)
    if current != preflight["operational_snapshot"] or hashlib.sha256(
        canonical_json_bytes(current)
    ).hexdigest() != preflight["operational_snapshot_sha256"]:
        raise PermissionError("r3 operational snapshot changed")


def _lock_payload(
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    workspace = _canonical_workspace(root, config)
    return {
        "schema_version": "p3_hierarchical_residual_basis.gen5r3_dense72.attempt_lock.r1",
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "stage": STAGE,
        "workspace_identity": {
            "root_st_dev": int(workspace.stat().st_dev),
            "root_st_ino": int(workspace.stat().st_ino),
        },
        "canonical_stage_relative": config["canonical_paths"]["output"],
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "static_preflight_sha256": preflight["summary_sha256"],
        "operational_snapshot": preflight["operational_snapshot"],
        "operational_snapshot_sha256": preflight["operational_snapshot_sha256"],
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "implementation_pins": implementation_pins(root, config),
        "status": "ATTEMPT_CONSUMED_BEFORE_CAPABILITY_MINT",
        "capability_minted": False,
        "rerun_allowed": False,
        "resume_allowed": False,
        "candidate_or_test_prediction_allowed": False,
        "upload_allowed": False,
    }


def create_and_verify_attempt_lock(
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> tuple[Path, dict[str, Any], str, str, str]:
    _validate_live_preflight(root, config, preflight)
    _qa, qa_sha = verify_pre_execution_qa(root, config, preflight)
    _authorization, authorization_sha = verify_execution_authorization(
        root, config, preflight, qa_sha256=qa_sha
    )
    paths = stage_paths(root, config)
    payload = _lock_payload(
        root,
        config,
        preflight,
        qa_sha256=qa_sha,
        authorization_sha256=authorization_sha,
    )
    exclusive_json(paths["attempt_lock"], payload)
    observed = strict_json_object(paths["attempt_lock"])
    if observed != payload:
        raise PermissionError("persisted r3 attempt lock differs immediately after O_EXCL write")
    lock_sha = sha256_file(paths["attempt_lock"])
    return paths["attempt_lock"], observed, lock_sha, qa_sha, authorization_sha


def verify_consumed_attempt_lock(
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    *,
    expected_lock_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    path = stage_paths(root, config)["attempt_lock"]
    if not path.is_file():
        raise PermissionError("canonical r3 attempt lock is missing")
    observed = strict_json_object(path)
    expected_keys = set(
        _lock_payload(
            root,
            config,
            preflight,
            qa_sha256=str(observed.get("qa_receipt_sha256")),
            authorization_sha256=str(observed.get("authorization_sha256")),
        )
    )
    lock_sha = sha256_file(path)
    checks = {
        "keys": set(observed) == expected_keys,
        "schema": observed.get("schema_version")
        == "p3_hierarchical_residual_basis.gen5r3_dense72.attempt_lock.r1",
        "stage": observed.get("stage") == STAGE,
        "workspace": observed.get("workspace_identity")
        == {
            "root_st_dev": int(root.resolve(strict=True).stat().st_dev),
            "root_st_ino": int(root.resolve(strict=True).stat().st_ino),
        },
        "canonical_stage": observed.get("canonical_stage_relative")
        == config["canonical_paths"]["output"],
        "config": observed.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "preflight": observed.get("static_preflight_sha256")
        == preflight["summary_sha256"],
        "operational": observed.get("operational_snapshot")
        == preflight["operational_snapshot"],
        "operational_sha": observed.get("operational_snapshot_sha256")
        == preflight["operational_snapshot_sha256"],
        "implementation": observed.get("implementation_pins")
        == implementation_pins(root, config),
        "qa_file": sha256_file(stage_paths(root, config)["pre_execution_qa"])
        == observed.get("qa_receipt_sha256"),
        "authorization_file": sha256_file(stage_paths(root, config)["authorization"])
        == observed.get("authorization_sha256"),
        "status": observed.get("status") == "ATTEMPT_CONSUMED_BEFORE_CAPABILITY_MINT",
        "mint_false_at_lock": observed.get("capability_minted") is False,
        "rerun": observed.get("rerun_allowed") is False,
        "resume": observed.get("resume_allowed") is False,
    }
    if expected_lock_sha256 is not None:
        checks["lock_sha"] = lock_sha == expected_lock_sha256
    if not all(checks.values()):
        raise PermissionError(
            f"consumed r3 attempt lock failed: {sorted(k for k, value in checks.items() if not value)}"
        )
    return observed, lock_sha


@dataclass(frozen=True)
class ExecutionCapability:
    root_st_dev: int
    root_st_ino: int
    config_sha256: str
    canonical_stage_relative: str
    attempt_lock_sha256: str
    static_preflight_sha256: str
    operational_snapshot_sha256: str
    operational_snapshot_canonical_json: str
    qa_sha256: str
    authorization_sha256: str
    nonce: str


_LIVE_CAPABILITY: ExecutionCapability | None = None
_LIVE_PHASE: str | None = None
_LIVE_TEMP_STAGE: Path | None = None


def issue_execution_capability(
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    *,
    lock_sha256: str,
) -> ExecutionCapability:
    global _LIVE_CAPABILITY, _LIVE_PHASE, _LIVE_TEMP_STAGE
    workspace = _canonical_workspace(root, config)
    if _LIVE_CAPABILITY is not None or _LIVE_PHASE is not None:
        raise PermissionError("a canonical r3 capability is already live")
    _validate_live_preflight(workspace, config, preflight)
    lock, observed_sha = verify_consumed_attempt_lock(
        workspace,
        config,
        preflight,
        expected_lock_sha256=lock_sha256,
    )
    if observed_sha != lock_sha256:
        raise PermissionError("r3 lock SHA changed before capability mint")
    operational_json = canonical_json_bytes(preflight["operational_snapshot"]).decode("utf-8")
    nonce = hashlib.sha256(
        canonical_json_bytes(
            {
                "pid": os.getpid(),
                "lock": observed_sha,
                "operational": preflight["operational_snapshot_sha256"],
                "stage": config["canonical_paths"]["output"],
                "qa": lock["qa_receipt_sha256"],
                "authorization": lock["authorization_sha256"],
            }
        )
    ).hexdigest()
    capability = ExecutionCapability(
        root_st_dev=int(workspace.stat().st_dev),
        root_st_ino=int(workspace.stat().st_ino),
        config_sha256=CONFIG_SHA256,
        canonical_stage_relative=config["canonical_paths"]["output"],
        attempt_lock_sha256=observed_sha,
        static_preflight_sha256=preflight["summary_sha256"],
        operational_snapshot_sha256=preflight["operational_snapshot_sha256"],
        operational_snapshot_canonical_json=operational_json,
        qa_sha256=lock["qa_receipt_sha256"],
        authorization_sha256=lock["authorization_sha256"],
        nonce=nonce,
    )
    _LIVE_CAPABILITY = capability
    _LIVE_PHASE = "LOCK_VERIFIED_CAPABILITY_MINTED"
    _LIVE_TEMP_STAGE = None
    return capability


def _require_core(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> ExecutionCapability:
    workspace = _canonical_workspace(root, config)
    if capability is not _LIVE_CAPABILITY or not isinstance(capability, ExecutionCapability):
        raise PermissionError("canonical live r3 capability is required")
    _validate_live_preflight(workspace, config, preflight)
    if (
        capability.root_st_dev != int(workspace.stat().st_dev)
        or capability.root_st_ino != int(workspace.stat().st_ino)
        or capability.config_sha256 != CONFIG_SHA256
        or capability.canonical_stage_relative != config["canonical_paths"]["output"]
        or capability.static_preflight_sha256 != preflight["summary_sha256"]
        or capability.operational_snapshot_sha256
        != preflight["operational_snapshot_sha256"]
        or capability.operational_snapshot_canonical_json
        != canonical_json_bytes(preflight["operational_snapshot"]).decode("utf-8")
    ):
        raise PermissionError("forged or stale r3 capability")
    lock, lock_sha = verify_consumed_attempt_lock(
        workspace,
        config,
        preflight,
        expected_lock_sha256=capability.attempt_lock_sha256,
    )
    if (
        lock_sha != capability.attempt_lock_sha256
        or lock["qa_receipt_sha256"] != capability.qa_sha256
        or lock["authorization_sha256"] != capability.authorization_sha256
    ):
        raise PermissionError("r3 capability no longer binds the consumed lock")
    return capability


def begin_execution_stage(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    global _LIVE_PHASE
    _require_core(capability, root=root, config=config, preflight=preflight)
    if _LIVE_PHASE != "LOCK_VERIFIED_CAPABILITY_MINTED":
        raise PermissionError("r3 execution-stage capability phase was consumed")
    if stage_paths(root, config)["output"].exists():
        raise FileExistsError("canonical r3 stage already exists")
    _LIVE_PHASE = "EXECUTION_STAGE_ENTERED"


def authorize_curve_phase(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    temporary_stage: Path,
) -> None:
    global _LIVE_PHASE, _LIVE_TEMP_STAGE
    _require_core(capability, root=root, config=config, preflight=preflight)
    workspace = root.resolve(strict=True)
    stage = temporary_stage.resolve(strict=True)
    tmp = (workspace / "tmp").resolve(strict=True)
    if _LIVE_PHASE != "EXECUTION_STAGE_ENTERED":
        raise PermissionError("r3 curve authorization phase is not available")
    if tmp not in stage.parents or not stage.is_dir() or any(stage.iterdir()):
        raise PermissionError("r3 temporary stage is noncanonical or nonempty")
    _LIVE_TEMP_STAGE = stage
    _LIVE_PHASE = "CURVE_CALL_AUTHORIZED"


def require_curve_capability(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: dict[str, Any],
    preflight: dict[str, Any],
    temporary_stage: Path,
) -> None:
    global _LIVE_PHASE
    _require_core(capability, root=root, config=config, preflight=preflight)
    if (
        _LIVE_PHASE != "CURVE_CALL_AUTHORIZED"
        or _LIVE_TEMP_STAGE is None
        or temporary_stage.resolve(strict=True) != _LIVE_TEMP_STAGE
    ):
        raise PermissionError("single-use live r3 curve phase is required")
    _LIVE_PHASE = "CURVE_CALL_CONSUMED"


def revoke_execution_capability(capability: ExecutionCapability) -> None:
    global _LIVE_CAPABILITY, _LIVE_PHASE, _LIVE_TEMP_STAGE
    if capability is not _LIVE_CAPABILITY:
        raise PermissionError("cannot revoke a noncanonical r3 capability")
    _LIVE_CAPABILITY = None
    _LIVE_PHASE = None
    _LIVE_TEMP_STAGE = None


def write_run_failure_receipt(
    root: Path,
    config: dict[str, Any],
    *,
    exception: BaseException,
) -> Path:
    paths = stage_paths(root, config)
    lock = paths["attempt_lock"].resolve(strict=True)
    entries: list[str] = []
    if paths["output"].is_dir():
        entries = sorted(
            item.relative_to(paths["output"]).as_posix()
            for item in paths["output"].rglob("*")
        )
    receipt = {
        "schema_version": "p3_hierarchical_residual_basis.gen5r3_dense72.failure.r1",
        "stage": STAGE,
        "classification": "POST_LOCK_FAILURE_NO_RETRY_NO_RESUME",
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "attempt_lock": _pin(lock, root.resolve(strict=True)),
        "implementation_pins": implementation_pins(root, config),
        "exception_type": type(exception).__name__,
        "exception_message_sha256": hashlib.sha256(str(exception).encode()).hexdigest(),
        "raw_exception_message_persisted": False,
        "output_recursive_entry_count": len(entries),
        "output_registered_relative_entries": entries,
        "rerun_allowed": False,
        "resume_allowed": False,
        "candidate_or_test_prediction_allowed": False,
        "upload_allowed": False,
        "uploads": 0,
    }
    exclusive_json(paths["run_failure_receipt"], receipt)
    return paths["run_failure_receipt"]


__all__ = [
    "COMPARISON_MODE",
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "Dense72R3ContractError",
    "ExecutionCapability",
    "FOLD_ORDER",
    "STAGE",
    "SelectiveCurrentHsAccessor",
    "authorize_curve_phase",
    "begin_execution_stage",
    "build_target_free_folds",
    "canonical_json_bytes",
    "create_and_verify_attempt_lock",
    "exclusive_json",
    "implementation_pins",
    "issue_execution_capability",
    "load_canonical_config",
    "operational_snapshot",
    "prepare_execution_preflight",
    "require_curve_capability",
    "revoke_execution_capability",
    "sha256_file",
    "stage_paths",
    "static_preflight",
    "strict_json_object",
    "verify_consumed_attempt_lock",
    "workspace_path",
    "write_run_failure_receipt",
]
