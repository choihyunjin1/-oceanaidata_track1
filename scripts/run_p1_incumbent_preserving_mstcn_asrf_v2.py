"""Sealed Q2 qualification and Q3+Q4 confirmation for MS-TCN++/ASRF.

This runner has three deliberately separate modes:

* ``--check-only`` verifies the preregistration and every immutable input pin;
* ``--smoke`` runs a two-epoch synthetic end-to-end exercise without locks or data;
* ``--execute-protocol`` performs Q2 finite-grid selection followed by the
  independently sealed Q3+Q4 confirmatory protocol.

Every holdout target is inaccessible to its training/prediction path.  The only
truth reader requires verified receipts for atomically committed blind files;
both confirmatory files are sealed before either confirmatory score is opened.
The runner never creates a deployable prediction file.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_incumbent_preserving_mstcn_asrf_v2"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
KEY_COLUMNS = ("station", "year", "layer", "time")
TYPE_NAMES = ("spike", "noise", "flatline", "offset", "drift")
Q2_FOLD = "2025_q2"
CURRENT_ROUTER_ORIGINAL_COLUMN = "incumbent_offline_xgboost__default"
CURRENT_ROUTER_BEST_COLUMN = "event_day_balanced_lightgbm__default"

# These columns are computed from an unbounded amount of the cached full-series
# covariate surface.  They are never projected into this experiment.  Depth is
# represented by train-only thresholds fitted to the current-row ``depth_raw``
# value instead of the cached station/year/layer median.
UNBOUNDED_CACHED_FEATURES = frozenset(
    {
        "nominal_depth_m",
        "plateau_full_length",
        "plateau_count",
        "depth_regime",
    }
)
SAFE_CURRENT_OR_PAST_CACHED_FEATURES = frozenset(
    {
        "station",
        "layer_category",
        "temp_raw",
        "psal_raw",
        "depth_raw",
        "psal_missing",
        "depth_missing",
        "has_gap_before",
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
        "temp_diff_1",
        "temp_abs_diff_1",
        "temp_backward_acceleration",
        "psal_diff_1",
        "psal_abs_diff_1",
        "depth_diff_1",
        "depth_abs_diff_1",
        "plateau_elapsed",
        "peer_count",
        "peer_available",
        "peer_temp_mean",
        "temp_peer_residual",
        "temp_abs_peer_residual",
        "station_layer_temp_std",
    }
)

# Updated only after the preregistration is final.  Keeping the expected digest
# in executable code makes a config edit fail before any scientific lock is
# acquired; a digest stored inside the mutable JSON would be self-authorizing.
EXPECTED_CONFIG_SHA256 = "1f8940d29ea6b047273e4f53445f62230e7d72bf1f0b14abe9fb18476f0345f0"
EXTERNAL_LAUNCHER_NAME = "launch_p1_incumbent_preserving_mstcn_asrf_v2.py"
_SEALED_LAUNCHER_CAPABILITY = object()


class ContractError(RuntimeError):
    """Raised when a fixed preregistration or leakage boundary is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    _atomic_bytes(path, _json_bytes(value), replace=replace)


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, *, replace: bool = False, **arrays: Any) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _atomic_torch_save(path: Path, value: Any, *, replace: bool = False) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        if path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _canonical_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    observed_digest = _sha256(path)
    if observed_digest != EXPECTED_CONFIG_SHA256:
        raise ContractError("full preregistration SHA-256 seal changed")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    if config.get("status") != "PREREGISTERED_LOCAL_ONLY_NOT_SUBMISSION_AUTHORIZED":
        raise ContractError("preregistration status changed")
    if config["windowing"] != {
        **config["windowing"],
        "rows": 2048,
        "stride_rows": 512,
    }:
        raise ContractError("registered window geometry changed")
    architecture = config["architecture"]
    if not (
        architecture["width"] == 256
        and architecture["q2_capacity_width_grid"] == [256, 512]
        and architecture["batch_size_by_width"] == {"256": 128, "512": 64}
        and architecture["prediction_generator_layers"] == 10
        and architecture["refinement_stages"] == 3
        and architecture["refinement_layers_per_stage"] == 10
        and architecture["dual_dilations"] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ):
        raise ContractError("registered model topology changed")
    training = config["training"]
    if not (
        training["maximum_epochs"] == 300
        and int(training["batch_size"]) >= 1
        and int(training["gradient_accumulation_steps"]) >= 1
        and training["q2_capacity_runs_complete_all_300_epochs"] is True
        and training["early_stopping_used_for_q2_selection"] is False
        and training["ensemble_seeds"] == [20260827, 20260839, 20260863]
        and training["result_based_retry"] is False
    ):
        raise ContractError("registered training contract changed")
    if config["decoder"]["q2_high_threshold_grid"] != [
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]:
        raise ContractError("registered Q2 decoder grid changed")
    if not (
        config["decoder"]["maximum_added_segment_rows"] is None
        and config["decoder"]["type_posterior_integration"][
            "long_event_type_heads"
        ]
        == ["noise", "offset", "drift"]
        and float(
            config["decoder"]["type_posterior_integration"][
                "multiplicative_weight"
            ]
        )
        == 0.25
    ):
        raise ContractError("registered long-event decoder changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ContractError("a preregistered prohibition was disabled")
    finite_count = (
        len(architecture["q2_capacity_width_grid"])
        * len(_checkpoint_epochs(config))
        * len(config["decoder"]["q2_high_threshold_grid"])
    )
    if finite_count != int(
        training["checkpoint_and_threshold_selection"]["finite_candidate_count"]
    ):
        raise ContractError("registered Q2 finite-grid candidate count changed")
    return config


def verify_immutable_inputs(config: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Verify the exact registered data, lineage, calibration, and code inventory."""

    immutable = config.get("immutable_inputs", {})
    expected_names = {
        "feature_cache",
        "feature_metadata",
        "feature_key_sidecar",
        "training_labels",
        "frozen_truth_and_folds",
        "frozen_round_b_anchor",
        "frozen_current_router_components",
        "frozen_current_router_anchor",
        "frozen_current_router_manifest",
        "capacity_calibration_receipt",
        "capacity_calibration_builder",
        "current_router_anchor_builder",
        "model_implementation",
        "data_implementation",
        "package_init_implementation",
    }
    if set(immutable) != expected_names:
        raise ContractError("immutable input inventory differs from the registered fifteen")
    records: dict[str, Any] = {}
    for name in sorted(expected_names):
        records[name] = _verify_registered_input(config, name, root=root)
    return records


def _verify_registered_input(
    config: dict[str, Any], name: str, *, root: Path = ROOT
) -> dict[str, Any]:
    immutable = config.get("immutable_inputs", {})
    if name not in immutable:
        raise ContractError(f"unregistered immutable input requested: {name}")
    record = immutable[name]
    root_resolved = root.resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root_resolved):
        raise ContractError(f"immutable input escapes repository: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = {
        "path": record["path"],
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    if (
        observed["bytes"] != int(record["bytes"])
        or observed["sha256"] != record["sha256"]
    ):
        raise ContractError(f"immutable input identity changed: {name}")
    return observed


@contextlib.contextmanager
def _verified_immutable_read(
    config: dict[str, Any], name: str, *, root: Path = ROOT
) -> Iterable[Path]:
    """Verify one delayed input immediately before and after its complete read."""

    before = _verify_registered_input(config, name, root=root)
    path = (root / before["path"]).resolve()
    try:
        yield path
    finally:
        after = _verify_registered_input(config, name, root=root)
        if after != before:
            raise ContractError(f"immutable input changed during read: {name}")


def _verify_capacity_calibration_contract(
    config: dict[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    immutable = config["immutable_inputs"]
    with _verified_immutable_read(
        config, "capacity_calibration_receipt", root=root
    ) as receipt_path:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not (
        receipt.get("status") == "PASS"
        and int(receipt.get("scientific_feature_rows_read", -1)) == 0
        and int(receipt.get("scientific_labels_read", -1)) == 0
        and int(receipt.get("official_test_sample_submission_reads", -1)) == 0
        and receipt.get("model_source_sha256")
        == immutable["model_implementation"]["sha256"]
        and receipt.get("torch_version") == config["runtime_identity"]["torch"]
        and receipt.get("torch_cuda_version")
        == config["runtime_identity"]["cuda_runtime"]
        and receipt.get("device") == config["runtime_identity"]["cuda_device"]
    ):
        raise ContractError("capacity calibration provenance changed")
    observed: dict[str, Any] = {}
    for cell in receipt.get("results", []):
        width = str(int(cell["width"]))
        if width in observed:
            raise ContractError("capacity calibration width is duplicated")
        if not (
            list(cell["input_shape"])[1:] == [2048, 165]
            and int(cell["batch_size"])
            == int(config["architecture"]["batch_size_by_width"][width])
            and int(cell["parameter_count"])
            == int(
                config["architecture"]["exact_parameter_count_by_width_at_input_165"][
                    width
                ]
            )
            and bool(cell["finite_final_loss"])
            and int(cell["peak_allocated_bytes"])
            < int(receipt["device_total_memory_bytes"])
        ):
            raise ContractError(f"capacity calibration cell changed: width {width}")
        observed[width] = {
            "batch_size": int(cell["batch_size"]),
            "parameter_count": int(cell["parameter_count"]),
            "peak_allocated_bytes": int(cell["peak_allocated_bytes"]),
            "optimizer_step_seconds_mean": float(cell["optimizer_step_seconds_mean"]),
        }
    if set(observed) != {"256", "512"}:
        raise ContractError("capacity calibration grid changed")
    return {
        "status": "PASS",
        "scientific_rows_or_labels_read": 0,
        "model_source_sha256": receipt["model_source_sha256"],
        "runtime_identity": {
            "torch": receipt["torch_version"],
            "cuda_runtime": receipt["torch_cuda_version"],
            "cuda_device": receipt["device"],
        },
        "cells": observed,
    }


def _verify_anchor_manifest_contract(
    config: dict[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    immutable = config["immutable_inputs"]
    with _verified_immutable_read(
        config, "frozen_current_router_manifest", root=root
    ) as manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("status") == "PASS_LABEL_FREE_RECONSTRUCTION"
        and int(manifest.get("truth_columns_read", -1)) == 0
        and int(manifest.get("rows", -1))
        == int(immutable["frozen_current_router_anchor"]["rows"])
        and manifest.get("source_sha256")
        == immutable["frozen_current_router_components"]["sha256"]
        and manifest.get("output_sha256")
        == immutable["frozen_current_router_anchor"]["sha256"]
        and int(manifest.get("output_bytes", -1))
        == int(immutable["frozen_current_router_anchor"]["bytes"])
    ):
        raise ContractError("current-Router anchor manifest provenance changed")
    return {
        "status": "PASS_LABEL_FREE_RECONSTRUCTION",
        "rows": int(manifest["rows"]),
        "positive_rows": int(manifest["positive_rows"]),
        "truth_columns_read": 0,
        "source_sha256": manifest["source_sha256"],
        "output_sha256": manifest["output_sha256"],
    }


def _observe_runtime_identity() -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import pyarrow
    import torch

    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "pyarrow": str(pyarrow.__version__),
        "torch": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": (
            None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version())
        ),
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _validate_runtime_identity(
    observed: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    exact_fields = (
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "cuda_runtime",
        "cudnn",
        "cuda_device",
    )
    mismatches = {
        name: {"expected": expected.get(name), "observed": observed.get(name)}
        for name in exact_fields
        if observed.get(name) != expected.get(name)
    }
    if bool(expected.get("require_cuda")) and observed.get("cuda_available") is not True:
        mismatches["cuda_available"] = {"expected": True, "observed": False}
    if mismatches:
        raise ContractError(f"registered runtime identity changed: {mismatches}")
    return {**observed, "result": "PASS_EXACT_RUNTIME_IDENTITY"}


def verify_runtime_identity(config: dict[str, Any]) -> dict[str, Any]:
    return _validate_runtime_identity(
        _observe_runtime_identity(), config["runtime_identity"]
    )


def check_only(*, root: Path = ROOT) -> dict[str, Any]:
    config_path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config = _canonical_config(config_path)
    return {
        "schema_version": "p1.mstcn_asrf.preflight.v2",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(config_path),
        "immutable_inputs": verify_immutable_inputs(config, root=root),
        "capacity_calibration_contract": _verify_capacity_calibration_contract(
            config, root=root
        ),
        "current_router_anchor_manifest_contract": _verify_anchor_manifest_contract(
            config, root=root
        ),
        "runtime_identity": verify_runtime_identity(config),
        "scientific_execution_entrypoint": f"scripts/{EXTERNAL_LAUNCHER_NAME}",
        "direct_runner_scientific_execution_authorized": False,
        "protected_interface_reads": 0,
        "result": "PASS",
    }


def _implementation_identity(*, root: Path = ROOT) -> dict[str, Any]:
    """Hash the transitive local code/config surface verified by the launcher."""

    paths = {
        "runner": Path("scripts") / f"run_{EXPERIMENT_ID}.py",
        "config": Path("configs") / "experiments" / f"{EXPERIMENT_ID}.json",
        "package_init": Path("src") / "p1_qc" / "__init__.py",
        "model": Path("src") / "p1_qc" / "ms_tcn_asrf.py",
        "data": Path("src") / "p1_qc" / "ms_tcn_asrf_data.py",
        "current_router_anchor_builder": Path("scripts")
        / "build_p1_current_router_oof_anchor_v1.py",
        "capacity_calibration_builder": Path("scripts")
        / "benchmark_p1_mstcn_capacity_grid_v2.py",
    }
    result: dict[str, Any] = {}
    root_resolved = root.resolve()
    for name, relative in paths.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root_resolved) or not path.is_file():
            raise ContractError(f"implementation path invalid: {name}")
        result[name] = {
            "path": relative.as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    return result


def _verify_external_implementation_attestation(
    attestation: dict[str, Any] | None,
    *,
    launcher_capability: object | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Require the reviewed launcher path before a scientific lock.

    This is an integrity and accidental-bypass guard.  It is deliberately not
    described as a hostile same-process Python security boundary.
    """

    if launcher_capability is not _SEALED_LAUNCHER_CAPABILITY:
        raise ContractError("scientific execution lacks the launcher capability")
    if not isinstance(attestation, dict):
        raise ContractError(
            f"scientific execution requires scripts/{EXTERNAL_LAUNCHER_NAME}"
        )
    if attestation.get("verified_by") != EXTERNAL_LAUNCHER_NAME:
        raise ContractError("external implementation verifier identity changed")
    launcher_path = root / "scripts" / EXTERNAL_LAUNCHER_NAME
    if (
        not launcher_path.is_file()
        or attestation.get("launcher_sha256") != _sha256(launcher_path)
        or attestation.get("externally_expected_launcher_sha256")
        != _sha256(launcher_path)
        or attestation.get("external_launcher_hash_acknowledged") is not True
    ):
        raise ContractError("external implementation verifier bytes changed")
    observed = _implementation_identity(root=root)
    if attestation.get("identities") != observed:
        raise ContractError("external implementation attestation differs from observed code")
    if not bool(attestation.get("all_pins_matched")):
        raise ContractError("external implementation verifier did not match every pin")
    return {
        "verified_by": EXTERNAL_LAUNCHER_NAME,
        "launcher_sha256": str(attestation.get("launcher_sha256", "")),
        "externally_expected_launcher_sha256": str(
            attestation.get("externally_expected_launcher_sha256", "")
        ),
        "external_launcher_hash_acknowledged": True,
        "identities": observed,
        "all_pins_matched": True,
    }


def _load_scientific() -> tuple[Any, Any, Any, Any, Any]:
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import numpy as np
    import pandas as pd
    import torch

    from p1_qc import ms_tcn_asrf as model_api
    from p1_qc import ms_tcn_asrf_data as data_api

    return np, pd, torch, model_api, data_api


def _ordered_key_sha(frame: Any) -> str:
    digest = hashlib.sha256()
    for column in KEY_COLUMNS:
        digest.update(column.encode("ascii") + b"\0")
        for value in frame[column].tolist():
            raw = str(value).encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
    return digest.hexdigest()


def _keys_equal(left: Any, right: Any) -> bool:
    if len(left) != len(right):
        return False
    return all(
        left[column].astype("string").fillna("").reset_index(drop=True).equals(
            right[column].astype("string").fillna("").reset_index(drop=True)
        )
        for column in KEY_COLUMNS
    )


def _kst_cutoff_text(utc_text: str) -> str:
    import pandas as pd

    timestamp = pd.Timestamp(utc_text)
    if timestamp.tzinfo is None:
        raise ContractError("registered cutoff must be timezone-aware")
    return timestamp.tz_convert("Asia/Seoul").isoformat()


def _read_training_targets_prefix(path: Path, cutoff_utc: str) -> Any:
    """Projection/filter pushdown is the only pre-receipt label-cache reader."""

    import pyarrow.dataset as dataset

    cutoff = _kst_cutoff_text(cutoff_utc)
    scanner = dataset.dataset(path, format="parquet").scanner(
        columns=[*KEY_COLUMNS, "label", "anomaly_type"],
        filter=dataset.field("time") <= cutoff,
        use_threads=True,
    )
    return scanner.to_table().to_pandas()


def _read_fold_membership_without_truth(path: Path, fold: str) -> Any:
    """Read only public keys/fold; target columns are absent from the projection."""

    import pyarrow.dataset as dataset

    scanner = dataset.dataset(path, format="parquet").scanner(
        columns=[*KEY_COLUMNS, "fold"],
        filter=dataset.field("fold") == fold,
        use_threads=True,
    )
    return scanner.to_table().to_pandas().reset_index(drop=True)


def _phase_protocol_for_fold(config: dict[str, Any], fold: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (phase, config["phase_protocols"][phase])
        for phase in ("q2", "q3", "q4")
        if config["phase_protocols"][phase].get("fold") == fold
    ]
    if len(matches) != 1:
        raise ContractError(f"fold must map to exactly one registered phase: {fold}")
    return matches[0]


def _validate_registered_holdout_membership(
    frame: Any,
    config: dict[str, Any],
    *,
    fold: str,
) -> tuple[Any, dict[str, Any]]:
    """Validate one exact frozen fold against its registered time envelope."""

    import pandas as pd

    phase, protocol = _phase_protocol_for_fold(config, fold)
    if "fold" not in frame.columns or len(frame) == 0:
        raise ContractError(f"{fold} membership is empty or lacks its fold column")
    observed_folds = set(frame["fold"].astype(str).unique().tolist())
    if observed_folds != {fold}:
        raise ContractError(f"{fold} membership is empty or mixed")
    source_rows = int(len(frame))
    timestamps = pd.to_datetime(frame["time"], utc=True, format="mixed", errors="raise")
    start = pd.Timestamp(protocol["holdout_start_utc"])
    stop = pd.Timestamp(protocol["holdout_end_utc_exclusive"])
    if start.tzinfo is None or stop.tzinfo is None or not start < stop:
        raise ContractError(f"{phase} registered holdout bounds are invalid")
    keep = (timestamps >= start) & (timestamps < stop)
    if not bool(keep.all()):
        raise ContractError(f"{fold} key escaped its registered holdout interval")
    membership = frame.reset_index(drop=True)
    membership_times = timestamps.reset_index(drop=True)
    key_index = pd.MultiIndex.from_frame(membership.loc[:, KEY_COLUMNS].astype(str))
    if not key_index.is_unique:
        raise ContractError(f"{fold} registered holdout keys are not unique")
    receipt = {
        "phase": phase,
        "fold": fold,
        "registered_start_utc_inclusive": start.isoformat(),
        "registered_end_utc_exclusive": stop.isoformat(),
        "source_fold_rows": source_rows,
        "membership_rows": int(len(membership)),
        "excluded_out_of_bounds_rows": 0,
        "membership_min_time_utc": membership_times.min().isoformat(),
        "membership_max_time_utc": membership_times.max().isoformat(),
        "ordered_key_sha256": _ordered_key_sha(membership),
        "all_membership_keys_inside_registered_bounds": True,
    }
    expected = protocol.get("membership_identity")
    if not isinstance(expected, dict):
        raise ContractError(f"{phase} membership identity pin is absent")
    pinned = {
        "source_fold_rows": int(expected.get("source_fold_rows", -1)),
        "membership_rows": int(expected.get("membership_rows", -1)),
        "excluded_out_of_bounds_rows": int(
            expected.get("excluded_out_of_bounds_rows", -1)
        ),
        "ordered_key_sha256": str(expected.get("ordered_key_sha256", "")),
    }
    if any(receipt[name] != value for name, value in pinned.items()):
        raise ContractError(f"{phase} membership identity changed")
    return membership, receipt


def _assert_series_local_fold_chronology(fold_keys: dict[str, Any]) -> dict[str, Any]:
    """Prove exact-key disjointness and chronology within every physical series."""

    import pandas as pd

    ordered_folds = ("2025_q2", "2025_q3", "2025_q4")
    key_sets: dict[str, set[tuple[str, ...]]] = {}
    day_sets: dict[str, set[Any]] = {}
    normalized: dict[str, Any] = {}
    for fold in ordered_folds:
        keys = fold_keys[fold].loc[:, KEY_COLUMNS].copy()
        keys["_time_utc"] = pd.to_datetime(keys["time"], utc=True, format="mixed")
        normalized[fold] = keys
        key_sets[fold] = set(
            keys.loc[:, KEY_COLUMNS].astype(str).itertuples(index=False, name=None)
        )
        day_sets[fold] = set(keys["_time_utc"].dt.tz_convert("Asia/Seoul").dt.date)
    pairwise_key_overlap = 0
    pairwise_day_overlap: dict[str, int] = {}
    minimum_series_gap_minutes: float | None = None
    checked_series_pairs = 0
    for left_index, left in enumerate(ordered_folds):
        for right in ordered_folds[left_index + 1 :]:
            key_overlap = key_sets[left].intersection(key_sets[right])
            pairwise_key_overlap += len(key_overlap)
            if key_overlap:
                raise ContractError(f"registered folds share exact keys: {left}|{right}")
            pairwise_day_overlap[f"{left}|{right}"] = len(
                day_sets[left].intersection(day_sets[right])
            )
            left_groups = normalized[left].groupby(
                ["station", "year", "layer"], sort=False, observed=True
            )["_time_utc"].max()
            right_groups = normalized[right].groupby(
                ["station", "year", "layer"], sort=False, observed=True
            )["_time_utc"].min()
            shared_series = left_groups.index.intersection(right_groups.index)
            for series in shared_series:
                gap_minutes = float(
                    (right_groups.loc[series] - left_groups.loc[series]).total_seconds()
                    / 60.0
                )
                if gap_minutes <= 0.0:
                    raise ContractError(
                        f"registered fold chronology overlaps within series {series}"
                    )
                checked_series_pairs += 1
                minimum_series_gap_minutes = (
                    gap_minutes
                    if minimum_series_gap_minutes is None
                    else min(minimum_series_gap_minutes, gap_minutes)
                )
    return {
        "pairwise_exact_key_overlap_rows": pairwise_key_overlap,
        "series_local_chronology_violations": 0,
        "series_pairs_checked": checked_series_pairs,
        "minimum_series_local_gap_minutes": minimum_series_gap_minutes,
        "pairwise_shared_kst_calendar_days": pairwise_day_overlap,
        "global_kst_calendar_overlap_allowed": True,
    }


def _feature_dependency_audit(
    metadata: dict[str, Any], config: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Classify every cached feature and return the exact safe projection.

    The immutable cache was built in offline mode.  Bounded, two-sided
    features are admissible only behind the registered time purge; full-run
    features are not admissible at all.  This function fails closed if the
    metadata or the registered exclusion inventory changes.
    """

    feature_columns = tuple(str(value) for value in metadata["feature_columns"])
    categorical = tuple(str(value) for value in metadata["categorical_columns"])
    registered = config["data_contract"]["cached_feature_dependency_audit"]
    registered_excluded = frozenset(str(value) for value in registered["excluded_unbounded"])
    if registered_excluded != UNBOUNDED_CACHED_FEATURES:
        raise ContractError("registered unbounded feature inventory changed")
    if not UNBOUNDED_CACHED_FEATURES.issubset(feature_columns):
        raise ContractError("immutable feature metadata lacks a registered exclusion")

    bounded_exact = {"temp_diff_next", "spike_min_abs_diff", "temp_center_curvature"}
    bounded_prefixes = (
        "temp_median_resid_",
        "temp_abs_median_resid_",
        "temp_roll_std_",
        "diff_roll_std_",
        "temp_robust_z_",
        "temp_long_resid_",
        "peer_detrended_resid_",
        "peer_abs_detrended_resid_",
        "reference_resid_",
        "reference_abs_resid_",
        "reference_slope_1h_",
    )
    bounded_future = tuple(
        value
        for value in feature_columns
        if value in bounded_exact or value.startswith(bounded_prefixes)
    )
    bounded_set = set(bounded_future)
    classified = (
        set(UNBOUNDED_CACHED_FEATURES)
        | set(SAFE_CURRENT_OR_PAST_CACHED_FEATURES)
        | bounded_set
    )
    observed = set(feature_columns)
    if classified != observed:
        raise ContractError(
            "immutable cached feature classification is not exact: "
            f"unknown={sorted(observed - classified)}, absent={sorted(classified - observed)}"
        )
    numeric = tuple(
        value
        for value in feature_columns
        if value not in categorical
        and value in (SAFE_CURRENT_OR_PAST_CACHED_FEATURES | bounded_set)
    )
    projected = ("station", "layer_category", *numeric)
    if len(numeric) != int(registered["model_numeric_feature_count"]):
        raise ContractError("safe model numeric width differs from preregistration")
    if any(value in projected for value in UNBOUNDED_CACHED_FEATURES):
        raise ContractError("unbounded cached feature entered the model projection")
    receipt = {
        "schema_version": "p1.mstcn_asrf.feature_dependency.v1",
        "cache_feature_count": len(feature_columns),
        "model_numeric_feature_count": len(numeric),
        "projected_cache_columns": list(projected),
        "excluded_unbounded": sorted(UNBOUNDED_CACHED_FEATURES),
        "bounded_future_columns": list(bounded_future),
        "bounded_future_support_hours_max": int(registered["bounded_future_support_hours_max"]),
        "depth_regime_source": "train-only thresholds over current-row depth_raw",
        "all_cache_features_classified": True,
        "unbounded_features_projected": 0,
    }
    return numeric, projected, receipt


def _current_router_bits(frame: Any) -> Any:
    """Reconstruct the exact pinned current-Router rule from frozen O/B bits."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    original = frame[CURRENT_ROUTER_ORIGINAL_COLUMN].to_numpy(dtype=np.int8)
    best = frame[CURRENT_ROUTER_BEST_COLUMN].to_numpy(dtype=np.int8)
    if not (np.isin(original, [0, 1]).all() and np.isin(best, [0, 1]).all()):
        raise ContractError("frozen current-Router components are not binary")
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].to_numpy(dtype=int)
    original_only = (original == 1) & (best == 0)
    best_only = (original == 0) & (best == 1)
    add = original_only & (
        ((station == "G-ORS") & (layer == 1))
        | ((station == "I-ORS") & (layer == 2))
    )
    remove = best_only & (
        ((station == "S-ORS") & np.isin(layer, [1, 5, 6]))
        | ((station == "I-ORS") & (layer == 4))
    )
    result = best.copy()
    result[add] = 1
    result[remove] = 0
    return result.astype(np.int8, copy=False)


def _read_current_router_fold(path: Path, fold: str) -> Any:
    import pyarrow.dataset as dataset

    columns = [
        *KEY_COLUMNS,
        "fold",
        CURRENT_ROUTER_ORIGINAL_COLUMN,
        CURRENT_ROUTER_BEST_COLUMN,
    ]
    scanner = dataset.dataset(path, format="parquet").scanner(
        columns=columns,
        filter=dataset.field("fold") == fold,
        use_threads=True,
    )
    return scanner.to_table().to_pandas().reset_index(drop=True)


def _read_current_router_anchor_fold(path: Path, fold: str, *, column: str) -> Any:
    import pyarrow.dataset as dataset

    scanner = dataset.dataset(path, format="parquet").scanner(
        columns=[*KEY_COLUMNS, "fold", column],
        filter=dataset.field("fold") == fold,
        use_threads=True,
    )
    return scanner.to_table().to_pandas().reset_index(drop=True)


@dataclass
class RowSurface:
    keys: Any
    numeric: Any
    station: Any
    layer_category: Any
    depth_regime: Any
    labels: Any | None = None
    anomaly_type: Any | None = None
    anchor: Any | None = None
    depth: Any | None = None

    @property
    def rows(self) -> int:
        return int(len(self.keys))


@dataclass
class FrozenSurfaces:
    training: RowSurface
    q2: RowSurface
    q3: RowSurface
    q4: RowSurface
    numeric_names: tuple[str, ...]
    q2_membership_sha256: str
    membership_sha256: dict[str, str]
    membership_receipts: dict[str, dict[str, Any]]
    dependency_receipt: dict[str, Any]
    all_keys: Any
    feature_table: Any


def load_blind_surfaces(config: dict[str, Any], *, root: Path = ROOT) -> FrozenSurfaces:
    """Load features, prefix targets, Q2 keys, and anchor without Q2 target columns."""

    np, pd, _torch, _model_api, _data_api = _load_scientific()
    immutable = config["immutable_inputs"]
    router_anchor_record = immutable["frozen_current_router_anchor"]

    with _verified_immutable_read(config, "feature_metadata", root=root) as metadata_path:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_numeric_count = len(metadata["feature_columns"]) - len(metadata["categorical_columns"])
    if cache_numeric_count != int(immutable["feature_metadata"]["numeric_feature_count"]):
        raise ContractError("registered cache numeric count differs from metadata")
    numeric_names, projected_columns, dependency_receipt = _feature_dependency_audit(
        metadata, config
    )
    # Column projection is the enforcement boundary: excluded full-run values
    # never enter process memory, even transiently.
    with _verified_immutable_read(config, "feature_cache", root=root) as feature_path:
        features = pd.read_parquet(feature_path, columns=list(projected_columns))
    with _verified_immutable_read(config, "feature_key_sidecar", root=root) as key_path:
        keys = pd.read_parquet(key_path, columns=["ordinal", *KEY_COLUMNS])
    if len(features) != int(immutable["feature_cache"]["rows"]) or len(keys) != len(features):
        raise ContractError("feature/key row count changed")
    ordinal = keys["ordinal"].to_numpy(dtype=np.int64)
    if not np.array_equal(ordinal, np.arange(len(keys), dtype=np.int64)):
        raise ContractError("feature key ordinal is not the canonical row order")
    keys = keys.loc[:, KEY_COLUMNS].reset_index(drop=True)

    q2_protocol = config["chronological_q2_qualification"]
    cutoff = q2_protocol["training_max_time_utc"]
    with _verified_immutable_read(config, "training_labels", root=root) as target_path:
        prefix_targets = _read_training_targets_prefix(target_path, cutoff)
    cutoff_kst = _kst_cutoff_text(cutoff)
    prefix_ids = np.flatnonzero(keys["time"].astype(str).to_numpy() <= cutoff_kst)
    prefix_keys = keys.iloc[prefix_ids].reset_index(drop=True)
    if not _keys_equal(prefix_keys, prefix_targets.loc[:, KEY_COLUMNS]):
        raise ContractError("prefix label/key alignment changed")
    labels = prefix_targets["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ContractError("prefix target is not binary")

    key_index = pd.MultiIndex.from_frame(keys.loc[:, KEY_COLUMNS].astype(str))
    if not key_index.is_unique:
        raise ContractError("feature keys are not unique")

    def make_surface(row_ids: Any, surface_keys: Any, **targets: Any) -> RowSurface:
        selected = features.iloc[row_ids].reset_index(drop=True)
        return RowSurface(
            keys=surface_keys.loc[:, KEY_COLUMNS].reset_index(drop=True),
            numeric=selected.loc[:, numeric_names].to_numpy(dtype=np.float32),
            station=selected["station"].astype(str).to_numpy(),
            layer_category=selected["layer_category"].astype(str).to_numpy(),
            depth_regime=None,
            depth=selected["depth_raw"].to_numpy(dtype=np.float32),
            **targets,
        )

    training = make_surface(
        prefix_ids,
        prefix_keys,
        labels=labels,
        anomaly_type=prefix_targets["anomaly_type"].fillna("").astype(str).to_numpy(),
    )
    holdouts: dict[str, RowSurface] = {}
    membership_sha: dict[str, str] = {}
    membership_receipts: dict[str, dict[str, Any]] = {}
    for fold in ("2025_q2", "2025_q3", "2025_q4"):
        with _verified_immutable_read(
            config, "frozen_truth_and_folds", root=root
        ) as oof_path:
            membership = _read_fold_membership_without_truth(oof_path, fold)
        membership, membership_receipt = _validate_registered_holdout_membership(
            membership, config, fold=fold
        )
        holdout_index = pd.MultiIndex.from_frame(
            membership.loc[:, KEY_COLUMNS].astype(str)
        )
        holdout_ids = key_index.get_indexer(holdout_index)
        if np.any(holdout_ids < 0) or len(np.unique(holdout_ids)) != len(holdout_ids):
            raise ContractError(f"{fold} membership does not map one-to-one to feature keys")
        with _verified_immutable_read(
            config, "frozen_current_router_components", root=root
        ) as router_path:
            router = _read_current_router_fold(router_path, fold)
        router, router_receipt = _validate_registered_holdout_membership(
            router, config, fold=fold
        )
        if router_receipt != membership_receipt:
            raise ContractError(f"{fold} Router filter receipt differs from membership")
        if not _keys_equal(membership, router):
            raise ContractError(f"{fold} membership and frozen current-Router keys differ")
        reconstructed_anchor = _current_router_bits(router)
        with _verified_immutable_read(
            config, "frozen_current_router_anchor", root=root
        ) as router_anchor_path:
            router_anchor = _read_current_router_anchor_fold(
                router_anchor_path,
                fold,
                column=router_anchor_record["column"],
            )
        router_anchor, anchor_receipt = _validate_registered_holdout_membership(
            router_anchor, config, fold=fold
        )
        if anchor_receipt != membership_receipt:
            raise ContractError(f"{fold} anchor filter receipt differs from membership")
        if not _keys_equal(membership, router_anchor):
            raise ContractError(
                f"{fold} membership and current-Router anchor sidecar keys differ"
            )
        anchor_bits = router_anchor[router_anchor_record["column"]].to_numpy(
            dtype=np.int8
        )
        if not np.array_equal(anchor_bits, reconstructed_anchor):
            raise ContractError(
                f"{fold} current-Router sidecar differs from exact O/B reconstruction"
            )
        holdouts[fold] = make_surface(holdout_ids, membership, anchor=anchor_bits)
        membership_sha[fold] = _ordered_key_sha(membership)
        membership_receipts[fold] = membership_receipt
    fold_separation = _assert_series_local_fold_chronology(
        {fold: surface.keys for fold, surface in holdouts.items()}
    )
    q2 = holdouts["2025_q2"]
    q2_start = pd.Timestamp(q2_protocol["validation_start_utc"])
    q2_train_max = pd.Timestamp(q2_protocol["training_max_time_utc"])
    support_hours = int(dependency_receipt["bounded_future_support_hours_max"])
    past_support_hours = int(
        config["data_contract"]["cached_feature_dependency_audit"][
            "validation_bounded_past_support_hours_max"
        ]
    )
    separation_hours = float((q2_start - q2_train_max).total_seconds() / 3600.0)
    required_separation_hours = support_hours + past_support_hours
    dependency_receipt.update(
        {
            "holdout_membership_projected_columns": [*KEY_COLUMNS, "fold"],
            "holdout_truth_columns_read_before_blind_receipts": 0,
            "holdout_covariate_rows_used_to_fit_preprocessing": 0,
            "holdout_covariate_rows_used_to_train": 0,
            "q2_training_to_holdout_gap_hours": separation_hours,
            "required_feature_non_overlap_hours": required_separation_hours,
            "feature_non_overlap_slack_hours": separation_hours - required_separation_hours,
            "q2_training_bounded_future_reaches_holdout": bool(
                separation_hours <= required_separation_hours
            ),
            "current_router_projection": [
                *KEY_COLUMNS,
                "fold",
                CURRENT_ROUTER_ORIGINAL_COLUMN,
                CURRENT_ROUTER_BEST_COLUMN,
            ],
            "round_b_anchor_role": "pinned research lineage only; not the primary gate anchor",
            "registered_holdout_membership_identities": membership_receipts,
            "registered_holdout_fold_separation": fold_separation,
        }
    )
    if dependency_receipt["q2_training_bounded_future_reaches_holdout"]:
        raise ContractError("bounded cached future support reaches the Q2 covariate surface")
    return FrozenSurfaces(
        training,
        q2,
        holdouts["2025_q3"],
        holdouts["2025_q4"],
        numeric_names,
        membership_sha["2025_q2"],
        membership_sha,
        membership_receipts,
        dependency_receipt,
        keys,
        features,
    )


def _subset_surface(surface: RowSurface, row_ids: Any) -> RowSurface:
    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    ids = np.asarray(row_ids, dtype=np.int64)
    return RowSurface(
        keys=surface.keys.iloc[ids].reset_index(drop=True),
        numeric=surface.numeric[ids],
        station=surface.station[ids],
        layer_category=surface.layer_category[ids],
        depth_regime=(
            None if surface.depth_regime is None else surface.depth_regime[ids]
        ),
        labels=None if surface.labels is None else surface.labels[ids],
        anomaly_type=None if surface.anomaly_type is None else surface.anomaly_type[ids],
        anchor=None if surface.anchor is None else surface.anchor[ids],
        depth=None if surface.depth is None else surface.depth[ids],
    )


def _time_mask(keys: Any, *, maximum: str | None = None, start: str | None = None, stop: str | None = None) -> Any:
    import pandas as pd

    values = pd.to_datetime(keys["time"], utc=True, format="mixed")
    mask = values.notna().to_numpy(copy=True)
    if maximum is not None:
        mask &= (values <= pd.Timestamp(maximum)).to_numpy()
    if start is not None:
        mask &= (values >= pd.Timestamp(start)).to_numpy()
    if stop is not None:
        mask &= (values < pd.Timestamp(stop)).to_numpy()
    return mask


@dataclass
class EncodedSurface:
    surface: RowSurface
    features: Any
    layout: Any
    targets: Any | None


def _fit_encoder_and_transform(
    train: RowSurface,
    others: Sequence[RowSurface],
    *,
    fit_ids: Any,
    forbidden_ids: Any,
    numeric_names: tuple[str, ...],
) -> tuple[Any, list[EncodedSurface]]:
    np, pd, _torch, _model_api, data_api = _load_scientific()
    if train.depth is None or any(surface.depth is None for surface in others):
        raise ContractError("current-row depth is required for split-local depth regimes")
    layout_train = data_api.SegmentLayout.from_aligned(
        train.keys["station"], train.keys["year"], train.keys["layer"], train.keys["time"]
    )
    encoder = data_api.RobustRowEncoder.fit(
        train.numeric,
        train.station,
        train.layer_category,
        fit_ids,
        depth=train.depth,
        forbidden_row_ids=forbidden_ids,
        numeric_names=numeric_names,
    )

    encoded: list[EncodedSurface] = []
    for index, surface in enumerate((train, *others)):
        layout = layout_train if index == 0 else data_api.SegmentLayout.from_aligned(
            surface.keys["station"],
            surface.keys["year"],
            surface.keys["layer"],
            surface.keys["time"],
        )
        rows = encoder.transform(
            surface.numeric,
            surface.station,
            surface.layer_category,
            depth=surface.depth,
            row_valid=np.ones(surface.rows, dtype=bool),
            gap=layout.gap_by_row.astype(bool),
            one_hot_categories=True,
        )
        targets = None
        if surface.labels is not None:
            targets = data_api.build_asrf_targets(
                surface.labels,
                surface.anomaly_type,
                layout,
                sigma_rows=3.0,
            )
        encoded.append(EncodedSurface(surface, rows.dense, layout, targets))
    return encoder, encoded


def _window_rank(window: Any, seed: int) -> bytes:
    payload = f"{seed}|{window.segment_id}|{window.start}|{window.valid_length}".encode("ascii")
    return hashlib.sha256(payload).digest()


def _selected_windows(encoded: EncodedSurface, config: dict[str, Any]) -> tuple[Any, ...]:
    _np, _pd, _torch, _model_api, data_api = _load_scientific()
    if encoded.targets is None:
        raise ContractError("training windows require targets")
    all_windows = data_api.build_window_index(
        encoded.layout,
        window_size=int(config["windowing"]["rows"]),
        stride=int(config["windowing"]["stride_rows"]),
    )
    return data_api.select_training_windows(
        all_windows,
        encoded.targets.row_label,
        negative_ratio=float(config["windowing"]["negative_window_ratio"]),
        seed=int(config["training"]["seed"]),
    )


def _all_windows(encoded: EncodedSurface, config: dict[str, Any]) -> tuple[Any, ...]:
    _np, _pd, _torch, _model_api, data_api = _load_scientific()
    return data_api.build_window_index(
        encoded.layout,
        window_size=int(config["windowing"]["rows"]),
        stride=int(config["windowing"]["stride_rows"]),
    )


def _materialize_target_batch(targets: Any, windows: Sequence[Any]) -> tuple[Any, Any, Any, Any]:
    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    size = windows[0].window_size
    batch = len(windows)
    event = np.zeros((batch, size), dtype=np.float32)
    boundary = np.zeros((batch, size, 2), dtype=np.float32)
    kinds = np.zeros((batch, size, 5), dtype=np.float32)
    valid = np.zeros((batch, size), dtype=bool)
    for index, window in enumerate(windows):
        length = window.valid_length
        rows = window.row_ids
        event[index, :length] = targets.row_label[rows]
        boundary[index, :length, 0] = targets.start_boundary[rows]
        boundary[index, :length, 1] = targets.end_boundary[rows]
        kinds[index, :length] = targets.anomaly_type[rows]
        valid[index, :length] = True
    return event, boundary, kinds, valid


def _batches(windows: Sequence[Any], batch_size: int, *, seed: int, shuffle: bool) -> Iterable[tuple[Any, ...]]:
    order = list(range(len(windows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield tuple(windows[index] for index in order[start : start + batch_size])


def _runs(binary: Any) -> list[tuple[int, int]]:
    values = [int(value) for value in binary]
    result: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(values):
        if values[cursor] == 0:
            cursor += 1
            continue
        stop = cursor + 1
        while stop < len(values) and values[stop] == 1:
            stop += 1
        result.append((cursor, stop))
        cursor = stop
    return result


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return float(overlap / union) if union else 0.0


def event_metrics(truth: Any, prediction: Any, *, iou_threshold: float = 0.70) -> dict[str, float]:
    """Greedy one-to-one event recall and IoU summary used by the sanity gate."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    true_runs = _runs(truth)
    predicted_runs = _runs(prediction)
    remaining = set(range(len(predicted_runs)))
    matched_ious: list[float] = []
    for true_run in true_runs:
        candidates = sorted(
            ((_iou(true_run, predicted_runs[index]), index) for index in remaining),
            reverse=True,
        )
        if candidates and candidates[0][0] >= iou_threshold:
            score, index = candidates[0]
            matched_ious.append(score)
            remaining.remove(index)
    recall = len(matched_ious) / len(true_runs) if true_runs else 1.0
    median = float(np.median(np.asarray(matched_ious))) if matched_ious else 0.0
    return {
        "true_events": float(len(true_runs)),
        "predicted_events": float(len(predicted_runs)),
        "matched_events": float(len(matched_ious)),
        "event_recall_iou_0_70": float(recall),
        "median_event_iou": float(median),
    }


def binary_metrics(truth: Any, prediction: Any) -> dict[str, float | int]:
    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    y = np.asarray(truth, dtype=np.int8)
    pred = np.asarray(prediction, dtype=np.int8)
    if y.shape != pred.shape or not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("binary metric inputs must be aligned binary vectors")
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def decode_long_event_segments(
    row_probability: Any,
    boundary_probability: Any,
    layout: Any,
    *,
    high_threshold: float,
    low_threshold: float | None = None,
    snap_radius: int = 12,
    minimum_rows: int = 19,
    maximum_rows: int | None = None,
) -> Any:
    """Segment-local hysteresis followed by ASRF start/end boundary snapping."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    probability = np.asarray(row_probability, dtype=np.float64)
    boundary = np.asarray(boundary_probability, dtype=np.float64)
    if probability.shape != (layout.n_rows,) or boundary.shape != (layout.n_rows, 2):
        raise ValueError("decoder inputs do not align with the segment layout")
    if not np.isfinite(probability).all() or not np.isfinite(boundary).all():
        raise ValueError("decoder probabilities must be finite")
    low = high_threshold / 2.0 if low_threshold is None else float(low_threshold)
    if not 0.0 <= low <= high_threshold <= 1.0:
        raise ValueError("decoder thresholds are invalid")
    result = np.zeros(layout.n_rows, dtype=np.int8)
    for segment in layout.segments:
        rows = segment.row_ids
        values = probability[rows]
        eligible = values >= low
        seeds = values >= high_threshold
        cursor = 0
        candidates: list[tuple[int, int]] = []
        while cursor < len(rows):
            if not eligible[cursor]:
                cursor += 1
                continue
            stop = cursor + 1
            while stop < len(rows) and eligible[stop]:
                stop += 1
            if bool(seeds[cursor:stop].any()):
                candidates.append((cursor, stop))
            cursor = stop
        for start, stop in candidates:
            left_lo, left_hi = max(0, start - snap_radius), min(len(rows), start + snap_radius + 1)
            right_center = stop - 1
            right_lo = max(0, right_center - snap_radius)
            right_hi = min(len(rows), right_center + snap_radius + 1)
            snapped_start = left_lo + int(np.argmax(boundary[rows[left_lo:left_hi], 0]))
            snapped_end = right_lo + int(np.argmax(boundary[rows[right_lo:right_hi], 1])) + 1
            if snapped_end <= snapped_start:
                snapped_start, snapped_end = start, stop
            length = snapped_end - snapped_start
            if minimum_rows <= length and (maximum_rows is None or length <= maximum_rows):
                result[rows[snapped_start:snapped_end]] = 1
    return result


def _long_type_conditioned_row_probability(
    row_probability: Any,
    type_probability: Any,
    *,
    weight: float,
) -> Any:
    """Use the conditional noise/offset/drift head in the fixed decoder score."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    row = np.asarray(row_probability, dtype=np.float32)
    kinds = np.asarray(type_probability, dtype=np.float32)
    if kinds.shape != (len(row), len(TYPE_NAMES)):
        raise ValueError("type probabilities do not align with row probabilities")
    if not np.isfinite(row).all() or not np.isfinite(kinds).all():
        raise ValueError("type-conditioned decoder probabilities must be finite")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("type posterior decoder weight is outside [0, 1]")
    long_support = np.max(kinds[:, [1, 3, 4]], axis=1)
    return row * ((1.0 - weight) + weight * long_support)


def _decoder_row_probability(bundle: PredictionBundle, config: dict[str, Any]) -> Any:
    integration = config["decoder"]["type_posterior_integration"]
    if integration["long_event_type_heads"] != ["noise", "offset", "drift"]:
        raise ContractError("registered long-event type-head inventory changed")
    return _long_type_conditioned_row_probability(
        bundle.row_probability,
        bundle.type_probability,
        weight=float(integration["multiplicative_weight"]),
    )


def _maximum_segment_rows(config: dict[str, Any]) -> int | None:
    value = config["decoder"]["maximum_added_segment_rows"]
    return None if value is None else int(value)


def anchor_preserving_union(anchor: Any, proposal: Any) -> Any:
    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    frozen = np.asarray(anchor, dtype=np.int8)
    addition = np.asarray(proposal, dtype=np.int8)
    if frozen.shape != addition.shape or not np.isin(frozen, [0, 1]).all():
        raise ValueError("anchor/proposal must be aligned binary vectors")
    result = np.maximum(frozen, addition).astype(np.int8)
    if np.any((frozen == 1) & (result == 0)):
        raise AssertionError("anchor-positive row was removed")
    return result


def _positive_weight(labels: Any) -> float:
    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    values = np.asarray(labels, dtype=np.int8)
    positive = int(values.sum())
    negative = int(len(values) - positive)
    if positive == 0:
        raise ContractError("training surface has no positive row")
    return float(np.clip(negative / positive, 1.0, 20.0))


def _autocast(device: Any) -> Any:
    _np, _pd, torch, _model_api, _data_api = _load_scientific()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _new_model(input_features: int, config: dict[str, Any], device: Any) -> Any:
    _np, _pd, torch, model_api, _data_api = _load_scientific()
    architecture = config["architecture"]
    model_config = model_api.MSTCNASRFConfig(
        input_feature_count=int(input_features),
        width=int(architecture["width"]),
        generator_dilations=tuple(int(value) for value in architecture["dual_dilations"]),
        refinement_stages=int(architecture["refinement_stages"]),
        refinement_dilations=tuple(int(value) for value in architecture["dual_dilations"]),
        dropout=float(architecture["dropout"]),
    )
    model = model_api.MSTCNASRF(model_config).to(device)
    return model


def _loss_config(config: dict[str, Any], positive_weight: float) -> Any:
    _np, _pd, _torch, model_api, _data_api = _load_scientific()
    weights = config["training"]["loss_weights"]
    return model_api.MSTCNASRFLossConfig(
        stage_weights=tuple(float(value) for value in config["training"]["stage_weights"]),
        event_bce_weight=float(weights["row_bce"]),
        event_dice_weight=float(weights["row_soft_dice"]),
        smoothing_weight=float(weights["truncated_temporal_smoothing"]),
        boundary_weight=float(weights["boundary_bce"]),
        type_weight=float(weights["type_bce"]),
        event_positive_weight=float(positive_weight),
    )


def _lr_at_step(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    maximum_lr: float,
    minimum_lr: float,
) -> float:
    if step < warmup_steps:
        return maximum_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
    progress = min(1.0, max(0.0, progress))
    return minimum_lr + 0.5 * (maximum_lr - minimum_lr) * (1.0 + math.cos(math.pi * progress))


def _schedule_geometry(config: dict[str, Any], *, window_count: int) -> tuple[int, int, int]:
    """Return fixed full-budget optimizer, total, and warmup step counts.

    Q2 capacity runs and every fresh confirmatory refit use the registered
    300-epoch horizon as the cosine denominator.  A confirmatory fit merely
    stops at the Q2-selected epoch; it must not compress the learning-rate
    curve into that shorter interval.
    """

    if window_count < 1:
        raise ValueError("window_count must be positive")
    training = config["training"]
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    if batch_size < 1 or accumulation < 1:
        raise ContractError("batch size and accumulation must be positive")
    batches_per_epoch = math.ceil(window_count / batch_size)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = int(training["maximum_epochs"]) * optimizer_steps_per_epoch
    warmup_steps = int(training["warmup_epochs"]) * optimizer_steps_per_epoch
    return optimizer_steps_per_epoch, total_steps, warmup_steps


def _checkpoint_epochs(config: dict[str, Any]) -> tuple[int, ...]:
    maximum = int(config["training"]["maximum_epochs"])
    interval = int(config["training"]["validation_interval_epochs"])
    required = {
        int(value) for value in config["training"]["required_learning_curve_epochs"]
    }
    epochs = tuple(sorted(required | set(range(interval, maximum + 1, interval))))
    if epochs != tuple(sorted(set(epochs))) or epochs[-1] != maximum or len(epochs) != 63:
        raise ContractError("effective checkpoint epoch union changed")
    return epochs


def _config_for_capacity(
    config: dict[str, Any], *, width: int, seed: int
) -> dict[str, Any]:
    cloned = json.loads(json.dumps(config))
    allowed = tuple(int(value) for value in config["architecture"]["q2_capacity_width_grid"])
    if width not in allowed:
        raise ContractError("capacity width is outside the preregistered grid")
    cloned["architecture"]["width"] = int(width)
    cloned["training"]["seed"] = int(seed)
    cloned["training"]["batch_size"] = int(
        config["architecture"]["batch_size_by_width"][str(width)]
    )
    return cloned


def _move_targets(batch: tuple[Any, Any, Any, Any], device: Any) -> tuple[Any, Any, Any, Any]:
    _np, _pd, torch, _model_api, _data_api = _load_scientific()
    event, boundary, kinds, valid = batch
    return (
        torch.from_numpy(event).to(device, non_blocking=True),
        torch.from_numpy(boundary).to(device, non_blocking=True),
        torch.from_numpy(kinds).to(device, non_blocking=True),
        torch.from_numpy(valid).to(device, non_blocking=True),
    )


@dataclass
class PredictionBundle:
    row_probability: Any
    boundary_probability: Any
    type_probability: Any


def predict_encoded(
    model: Any,
    encoded: EncodedSurface,
    windows: Sequence[Any],
    *,
    batch_size: int,
    device: Any,
) -> PredictionBundle:
    np, _pd, torch, _model_api, data_api = _load_scientific()
    row_parts: list[Any] = []
    boundary_parts: list[Any] = []
    type_parts: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch_windows in _batches(windows, batch_size, seed=0, shuffle=False):
            values, valid = data_api.materialize_windows(encoded.features, batch_windows)
            tensor = torch.from_numpy(values).to(device, non_blocking=True)
            valid_tensor = torch.from_numpy(valid).to(device, dtype=torch.bool, non_blocking=True)
            with _autocast(device):
                output = model(tensor, valid_mask=valid_tensor)
            row_parts.append(torch.sigmoid(output.final_logits).float().cpu().numpy())
            boundary_parts.append(torch.sigmoid(output.boundary_logits).float().cpu().numpy())
            type_parts.append(torch.sigmoid(output.type_logits).float().cpu().numpy())
    row_windows = np.concatenate(row_parts, axis=0)
    boundary_windows = np.concatenate(boundary_parts, axis=0)
    type_windows = np.concatenate(type_parts, axis=0)
    required = np.arange(encoded.surface.rows, dtype=np.int64)
    row = data_api.stitch_center_weighted(
        row_windows, windows, n_rows=encoded.surface.rows, require_row_ids=required
    )
    boundary = data_api.stitch_center_weighted(
        boundary_windows, windows, n_rows=encoded.surface.rows, require_row_ids=required
    )
    kinds = data_api.stitch_center_weighted(
        type_windows, windows, n_rows=encoded.surface.rows, require_row_ids=required
    )
    if not (np.isfinite(row).all() and np.isfinite(boundary).all() and np.isfinite(kinds).all()):
        raise ContractError("stitched prediction contains non-finite values")
    return PredictionBundle(row, boundary, kinds)


def _train_epoch(
    model: Any,
    optimizer: Any,
    encoded: EncodedSurface,
    windows: Sequence[Any],
    *,
    config: dict[str, Any],
    positive_weight: float,
    device: Any,
    epoch: int,
    global_step: int,
    total_steps: int,
) -> tuple[dict[str, Any], int, float]:
    np, _pd, torch, model_api, data_api = _load_scientific()
    if encoded.targets is None:
        raise ContractError("training targets are absent")
    training = config["training"]
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    if batch_size < 1 or accumulation < 1:
        raise ContractError("batch size and accumulation must be positive")
    batches = list(_batches(windows, batch_size, seed=int(training["seed"]) + epoch, shuffle=True))
    optimizer.zero_grad(set_to_none=True)
    component_sums = {
        "total_loss": 0.0,
        "event_loss": 0.0,
        "temporal_smoothing_loss": 0.0,
        "boundary_loss": 0.0,
        "anomaly_type_loss": 0.0,
    }
    observed = 0
    grad_norms: list[float] = []
    gradient_clip_count = 0
    optimizer_steps_at_start = global_step
    nonfinite_count = 0
    loss_config = _loss_config(config, positive_weight)
    maximum_lr = float(training["learning_rate"])
    minimum_lr = 3e-6
    _steps_per_epoch, registered_total_steps, warmup_steps = _schedule_geometry(
        config, window_count=len(windows)
    )
    if total_steps != registered_total_steps:
        raise ContractError("training caller supplied a compressed LR horizon")
    model.train()
    for batch_index, batch_windows in enumerate(batches):
        values, feature_valid = data_api.materialize_windows(encoded.features, batch_windows)
        targets = _materialize_target_batch(encoded.targets, batch_windows)
        tensor = torch.from_numpy(values).to(device, non_blocking=True)
        event, boundary, kinds, valid = _move_targets(targets, device)
        feature_valid_tensor = torch.from_numpy(feature_valid).to(
            device, dtype=torch.bool, non_blocking=True
        )
        if not bool(torch.equal(feature_valid_tensor, valid)):
            raise ContractError("feature and target valid masks differ")
        with _autocast(device):
            output = model(tensor, valid_mask=feature_valid_tensor)
            loss_output = model_api.compute_ms_tcn_asrf_loss(
                output,
                event,
                boundary,
                kinds,
                valid_mask=valid,
                config=loss_config,
            )
            loss = loss_output.total / accumulation
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        batch_weight = len(batch_windows)
        loss_components = {
            "total_loss": loss_output.total,
            "event_loss": loss_output.event,
            "temporal_smoothing_loss": loss_output.temporal_smoothing,
            "boundary_loss": loss_output.boundary,
            "anomaly_type_loss": loss_output.anomaly_type,
        }
        for name, value in loss_components.items():
            scalar = float(value.detach().float().cpu())
            if not math.isfinite(scalar):
                nonfinite_count += 1
                raise FloatingPointError(f"non-finite {name}")
            component_sums[name] += scalar * batch_weight
        observed += len(batch_windows)
        at_boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches)
        if at_boundary:
            lr = _lr_at_step(
                global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                maximum_lr=maximum_lr,
                minimum_lr=minimum_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite gradient norm")
            grad_norm = float(norm.detach().float().cpu())
            grad_norms.append(grad_norm)
            if grad_norm > float(training["gradient_clip_norm"]):
                gradient_clip_count += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
    if not grad_norms:
        raise ContractError("training epoch performed no optimizer step")
    denominator = max(1, observed)
    telemetry = {
        name: value / denominator for name, value in component_sums.items()
    }
    telemetry.update(
        {
            "grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "grad_norm_max": max(grad_norms),
            "grad_norm_last": grad_norms[-1],
            "gradient_clip_count": int(gradient_clip_count),
            "optimizer_steps_epoch": int(global_step - optimizer_steps_at_start),
            "observed_windows": int(observed),
            "nonfinite_count": int(nonfinite_count),
        }
    )
    return telemetry, global_step, float(optimizer.param_groups[0]["lr"])


def _history_record(
    *,
    epoch: int,
    telemetry: dict[str, Any],
    global_step: int,
    learning_rate: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Create one loss/optimization receipt row without hiding components."""

    return {
        "epoch": int(epoch),
        **telemetry,
        "train_loss": float(telemetry["total_loss"]),
        "learning_rate": float(learning_rate),
        "optimizer_steps_cumulative": int(global_step),
        "epoch_wall_seconds": float(elapsed_seconds),
    }


def _reset_cuda_peak_memory(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory_receipt(torch: Any, device: Any) -> dict[str, Any]:
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return {
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
        }
    torch.cuda.synchronize(device)
    return {
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _optional_artifact_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "bytes": None, "sha256": None}
    if not path.is_file():
        raise ContractError(f"expected run artifact is absent: {path.name}")
    return {
        "path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _smoke_refit_and_predict(
    training: EncodedSurface,
    q2: EncodedSurface,
    *,
    config: dict[str, Any],
    selected_epoch: int,
    device: Any,
    artifact_dir: Path | None,
) -> tuple[PredictionBundle, dict[str, Any]]:
    """Exercise training/inference on synthetic data only; never used scientifically."""
    _np, _pd, torch, _model_api, _data_api = _load_scientific()
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _new_model(training.features.shape[1], config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    windows = _selected_windows(training, config)
    positive_weight = _positive_weight(training.surface.labels)
    batch_size = int(config["training"]["batch_size"])
    _steps_per_epoch, total_steps, _warmup_steps = _schedule_geometry(
        config, window_count=len(windows)
    )
    global_step = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, selected_epoch + 1):
        started = time.perf_counter()
        telemetry, global_step, lr = _train_epoch(
            model,
            optimizer,
            training,
            windows,
            config=config,
            positive_weight=positive_weight,
            device=device,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
        )
        history.append(
            _history_record(
                epoch=epoch,
                telemetry=telemetry,
                global_step=global_step,
                learning_rate=lr,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        if artifact_dir is not None:
            _atomic_json(artifact_dir / "synthetic_smoke_history.json", history, replace=True)
    q2_windows = _all_windows(q2, config)
    blind = predict_encoded(model, q2, q2_windows, batch_size=batch_size, device=device)
    checkpoint = {
        "schema_version": "p1.mstcn_asrf.synthetic_smoke_checkpoint.v1",
        "experiment_id": EXPERIMENT_ID,
        "selected_epoch": selected_epoch,
        "input_features": int(training.features.shape[1]),
        "parameter_count": int(model.trainable_parameter_count),
        "positive_weight": positive_weight,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    checkpoint_sha = None
    if artifact_dir is not None:
        checkpoint_sha = _atomic_torch_save(
            artifact_dir / "synthetic_smoke_checkpoint.pt", checkpoint
        )
    return blind, {
        "epochs": selected_epoch,
        "optimizer_steps": global_step,
        "positive_weight": positive_weight,
        "parameter_count": int(model.trainable_parameter_count),
        "checkpoint_sha256": checkpoint_sha,
        "history_rows": len(history),
    }


def _fit_capacity_seed_checkpoints(
    training: EncodedSurface,
    holdout: EncodedSurface,
    *,
    config: dict[str, Any],
    width: int,
    seed: int,
    device: Any,
    artifact_dir: Path | None,
    phase: str,
) -> tuple[dict[int, PredictionBundle], dict[str, Any]]:
    """Run one full-budget seed and capture raw predictions at all checkpoints."""

    _np, _pd, torch, _model_api, _data_api = _load_scientific()
    capacity = _config_for_capacity(config, width=width, seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _reset_cuda_peak_memory(torch, device)
    model = _new_model(training.features.shape[1], capacity, device)
    expected_parameters = int(
        config["architecture"]["exact_parameter_count_by_width_at_input_165"][str(width)]
    )
    if int(model.trainable_parameter_count) != expected_parameters:
        raise ContractError("capacity-grid parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(capacity["training"]["learning_rate"]),
        weight_decay=float(capacity["training"]["weight_decay"]),
    )
    windows = _selected_windows(training, capacity)
    holdout_windows = _all_windows(holdout, capacity)
    positive_weight = _positive_weight(training.surface.labels)
    _steps, total_steps, _warmup = _schedule_geometry(
        capacity, window_count=len(windows)
    )
    checkpoint_epochs = set(_checkpoint_epochs(capacity))
    global_step = 0
    history: list[dict[str, Any]] = []
    predictions: dict[int, PredictionBundle] = {}
    history_path = (
        None
        if artifact_dir is None
        else artifact_dir / f"{phase}_width_{width}_seed_{seed}_training_history.json"
    )
    for epoch in range(1, int(capacity["training"]["maximum_epochs"]) + 1):
        started = time.perf_counter()
        telemetry, global_step, lr = _train_epoch(
            model,
            optimizer,
            training,
            windows,
            config=capacity,
            positive_weight=positive_weight,
            device=device,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
        )
        record = _history_record(
            epoch=epoch,
            telemetry=telemetry,
            global_step=global_step,
            learning_rate=lr,
            elapsed_seconds=time.perf_counter() - started,
        )
        if epoch in checkpoint_epochs:
            predictions[epoch] = predict_encoded(
                model,
                holdout,
                holdout_windows,
                batch_size=int(capacity["training"]["batch_size"]),
                device=device,
            )
            record["blind_checkpoint_captured"] = True
        history.append(record)
        if history_path is not None:
            _atomic_json(
                history_path,
                history,
                replace=True,
            )
    if tuple(sorted(predictions)) != _checkpoint_epochs(capacity):
        raise ContractError("a preregistered qualification checkpoint is absent")
    return predictions, {
        "phase": phase,
        "width": width,
        "seed": seed,
        "epochs": int(capacity["training"]["maximum_epochs"]),
        "checkpoint_epochs": list(sorted(predictions)),
        "optimizer_steps": global_step,
        "positive_weight": positive_weight,
        "parameter_count": int(model.trainable_parameter_count),
        "batch_size": int(capacity["training"]["batch_size"]),
        "prediction_representation": "raw",
        "type_head_used_by_decoder": True,
        "history_artifact": _optional_artifact_identity(history_path),
        "loss_history_fields": list(history[-1]) if history else [],
        "nonfinite_count_total": int(sum(row["nonfinite_count"] for row in history)),
        "gradient_clip_count_total": int(
            sum(row["gradient_clip_count"] for row in history)
        ),
        **_cuda_peak_memory_receipt(torch, device),
    }


@dataclass
class QualificationGrid:
    widths: Any
    epochs: Any
    thresholds: Any
    row_probability: Any
    boundary_probability: Any
    proposal: Any
    candidate: Any
    fit_receipts: list[dict[str, Any]]


def fit_q2_qualification_grid(
    training: EncodedSurface,
    q2: EncodedSurface,
    *,
    config: dict[str, Any],
    device: Any,
    artifact_dir: Path | None,
) -> QualificationGrid:
    """Fit the sealed width×seed grid and average seeds before decoding."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    if q2.surface.anchor is None:
        raise ContractError("Q2 current-Router anchor is absent")
    widths: list[int] = []
    epochs: list[int] = []
    row_parts: list[Any] = []
    boundary_parts: list[Any] = []
    proposal_parts: list[Any] = []
    candidate_parts: list[Any] = []
    receipts: list[dict[str, Any]] = []
    checkpoint_epochs = _checkpoint_epochs(config)
    threshold_grid = np.asarray(
        config["decoder"]["q2_high_threshold_grid"], dtype=np.float64
    )
    seeds = tuple(int(value) for value in config["training"]["ensemble_seeds"])
    for width in (int(value) for value in config["architecture"]["q2_capacity_width_grid"]):
        row_sum = {
            epoch: np.zeros(q2.surface.rows, dtype=np.float32)
            for epoch in checkpoint_epochs
        }
        boundary_sum = {
            epoch: np.zeros((q2.surface.rows, 2), dtype=np.float32)
            for epoch in checkpoint_epochs
        }
        type_sum = {
            epoch: np.zeros((q2.surface.rows, len(TYPE_NAMES)), dtype=np.float32)
            for epoch in checkpoint_epochs
        }
        for seed in seeds:
            seed_predictions, receipt = _fit_capacity_seed_checkpoints(
                training,
                q2,
                config=config,
                width=width,
                seed=seed,
                device=device,
                artifact_dir=artifact_dir,
                phase="q2",
            )
            receipts.append(receipt)
            for epoch in checkpoint_epochs:
                row_sum[epoch] += seed_predictions[epoch].row_probability.astype(
                    np.float32, copy=False
                )
                boundary_sum[epoch] += seed_predictions[epoch].boundary_probability.astype(
                    np.float32, copy=False
                )
                type_sum[epoch] += seed_predictions[epoch].type_probability.astype(
                    np.float32, copy=False
                )
        for epoch in checkpoint_epochs:
            row_mean = row_sum[epoch] / float(len(seeds))
            boundary_mean = boundary_sum[epoch] / float(len(seeds))
            type_mean = type_sum[epoch] / float(len(seeds))
            decoder_row_mean = _decoder_row_probability(
                PredictionBundle(row_mean, boundary_mean, type_mean), config
            )
            proposals: list[Any] = []
            candidates: list[Any] = []
            for threshold in threshold_grid:
                proposal = decode_long_event_segments(
                    decoder_row_mean,
                    boundary_mean,
                    q2.layout,
                    high_threshold=float(threshold),
                    snap_radius=int(config["decoder"]["boundary_peak_snap_radius_rows"]),
                    minimum_rows=int(config["decoder"]["minimum_added_segment_rows"]),
                    maximum_rows=_maximum_segment_rows(config),
                )
                proposals.append(proposal)
                candidates.append(anchor_preserving_union(q2.surface.anchor, proposal))
            widths.append(width)
            epochs.append(epoch)
            row_parts.append(decoder_row_mean)
            boundary_parts.append(boundary_mean)
            proposal_parts.append(np.stack(proposals, axis=0))
            candidate_parts.append(np.stack(candidates, axis=0))
    return QualificationGrid(
        np.asarray(widths, dtype=np.int16),
        np.asarray(epochs, dtype=np.int16),
        threshold_grid,
        np.stack(row_parts, axis=0),
        np.stack(boundary_parts, axis=0),
        np.stack(proposal_parts, axis=0),
        np.stack(candidate_parts, axis=0),
        receipts,
    )


def _fit_raw_seed_to_epoch(
    training: EncodedSurface,
    holdout: EncodedSurface,
    *,
    config: dict[str, Any],
    width: int,
    seed: int,
    selected_epoch: int,
    phase: str,
    device: Any,
    artifact_dir: Path | None,
) -> tuple[PredictionBundle, dict[str, Any]]:
    """Fresh raw refit using the fixed 300-epoch LR horizon, stopped at selection."""

    _np, _pd, torch, _model_api, _data_api = _load_scientific()
    capacity = _config_for_capacity(config, width=width, seed=seed)
    if not 1 <= selected_epoch <= int(capacity["training"]["maximum_epochs"]):
        raise ContractError("selected epoch is outside the preregistered horizon")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _reset_cuda_peak_memory(torch, device)
    model = _new_model(training.features.shape[1], capacity, device)
    expected_parameters = int(
        config["architecture"]["exact_parameter_count_by_width_at_input_165"][str(width)]
    )
    if int(model.trainable_parameter_count) != expected_parameters:
        raise ContractError("confirmatory parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(capacity["training"]["learning_rate"]),
        weight_decay=float(capacity["training"]["weight_decay"]),
    )
    windows = _selected_windows(training, capacity)
    positive_weight = _positive_weight(training.surface.labels)
    _steps, total_steps, _warmup = _schedule_geometry(
        capacity, window_count=len(windows)
    )
    global_step = 0
    history: list[dict[str, Any]] = []
    history_path = (
        None
        if artifact_dir is None
        else artifact_dir / f"{phase}_width_{width}_seed_{seed}_refit_history.json"
    )
    for epoch in range(1, selected_epoch + 1):
        started = time.perf_counter()
        telemetry, global_step, lr = _train_epoch(
            model,
            optimizer,
            training,
            windows,
            config=capacity,
            positive_weight=positive_weight,
            device=device,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
        )
        history.append(
            _history_record(
                epoch=epoch,
                telemetry=telemetry,
                global_step=global_step,
                learning_rate=lr,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        if history_path is not None:
            _atomic_json(
                history_path,
                history,
                replace=True,
            )
    blind = predict_encoded(
        model,
        holdout,
        _all_windows(holdout, capacity),
        batch_size=int(capacity["training"]["batch_size"]),
        device=device,
    )
    checkpoint_path = (
        None
        if artifact_dir is None
        else artifact_dir / f"{phase}_width_{width}_seed_{seed}_raw_checkpoint.pt"
    )
    if artifact_dir is not None:
        _atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": "p1.mstcn_asrf.confirmatory_raw_checkpoint.v1",
                "experiment_id": EXPERIMENT_ID,
                "phase": phase,
                "width": width,
                "seed": seed,
                "selected_epoch": selected_epoch,
                "input_features": int(training.features.shape[1]),
                "parameter_count": int(model.trainable_parameter_count),
                "state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
            },
        )
    return blind, {
        "phase": phase,
        "width": width,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "optimizer_steps": global_step,
        "batch_size": int(capacity["training"]["batch_size"]),
        "parameter_count": int(model.trainable_parameter_count),
        "history_artifact": _optional_artifact_identity(history_path),
        "checkpoint_artifact": _optional_artifact_identity(checkpoint_path),
        "prediction_representation": "raw",
        "type_head_used_by_decoder": True,
        "loss_history_fields": list(history[-1]) if history else [],
        "nonfinite_count_total": int(sum(row["nonfinite_count"] for row in history)),
        "gradient_clip_count_total": int(
            sum(row["gradient_clip_count"] for row in history)
        ),
        **_cuda_peak_memory_receipt(torch, device),
    }


def refit_confirmatory_ensemble(
    training: EncodedSurface,
    holdout: EncodedSurface,
    *,
    config: dict[str, Any],
    width: int,
    selected_epoch: int,
    selected_threshold: float,
    phase: str,
    device: Any,
    artifact_dir: Path | None,
) -> tuple[PredictionBundle, Any, Any, list[dict[str, Any]]]:
    """Fresh-refit all three seeds and decode their probability mean once."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    row_sum = np.zeros(holdout.surface.rows, dtype=np.float32)
    boundary_sum = np.zeros((holdout.surface.rows, 2), dtype=np.float32)
    type_sum = np.zeros((holdout.surface.rows, len(TYPE_NAMES)), dtype=np.float32)
    receipts: list[dict[str, Any]] = []
    seeds = tuple(int(value) for value in config["training"]["ensemble_seeds"])
    for seed in seeds:
        blind, receipt = _fit_raw_seed_to_epoch(
            training,
            holdout,
            config=config,
            width=width,
            seed=seed,
            selected_epoch=selected_epoch,
            phase=phase,
            device=device,
            artifact_dir=artifact_dir,
        )
        row_sum += blind.row_probability.astype(np.float32, copy=False)
        boundary_sum += blind.boundary_probability.astype(np.float32, copy=False)
        type_sum += blind.type_probability.astype(np.float32, copy=False)
        receipts.append(receipt)
    bundle = PredictionBundle(
        row_sum / float(len(seeds)),
        boundary_sum / float(len(seeds)),
        type_sum / float(len(seeds)),
    )
    proposal = decode_long_event_segments(
        _decoder_row_probability(bundle, config),
        bundle.boundary_probability,
        holdout.layout,
        high_threshold=selected_threshold,
        snap_radius=int(config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=_maximum_segment_rows(config),
    )
    candidate = anchor_preserving_union(holdout.surface.anchor, proposal)
    return bundle, proposal, candidate, receipts


def _predict_window_outputs(
    model: Any,
    encoded: EncodedSurface,
    windows: Sequence[Any],
    *,
    batch_size: int,
    device: Any,
) -> PredictionBundle:
    np, _pd, torch, _model_api, data_api = _load_scientific()
    row_parts: list[Any] = []
    boundary_parts: list[Any] = []
    type_parts: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch_windows in _batches(windows, batch_size, seed=0, shuffle=False):
            values, valid = data_api.materialize_windows(encoded.features, batch_windows)
            valid_tensor = torch.from_numpy(valid).to(device, dtype=torch.bool, non_blocking=True)
            with _autocast(device):
                output = model(
                    torch.from_numpy(values).to(device, non_blocking=True),
                    valid_mask=valid_tensor,
                )
            row_parts.append(torch.sigmoid(output.final_logits).float().cpu().numpy())
            boundary_parts.append(torch.sigmoid(output.boundary_logits).float().cpu().numpy())
            type_parts.append(torch.sigmoid(output.type_logits).float().cpu().numpy())
    return PredictionBundle(
        np.concatenate(row_parts, axis=0),
        np.concatenate(boundary_parts, axis=0),
        np.concatenate(type_parts, axis=0),
    )


def run_sanity_gate(
    encoded_train: EncodedSurface,
    *,
    config: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    """Overfit the fixed 32-positive/32-normal deterministic implementation sample."""

    np, pd, torch, _model_api, data_api = _load_scientific()
    if encoded_train.targets is None:
        raise ContractError("sanity surface has no targets")
    all_windows = _all_windows(encoded_train, config)
    long_rows = _long_event_mask(
        pd.DataFrame(
            {
                "label": encoded_train.surface.labels,
                "anomaly_type": encoded_train.surface.anomaly_type,
            }
        ),
        encoded_train.layout,
        minimum_rows=int(config["decoder"]["minimum_added_segment_rows"]),
    )
    positive = [window for window in all_windows if long_rows[window.row_ids].any()]
    normal = [window for window in all_windows if not encoded_train.surface.labels[window.row_ids].any()]
    seed = int(config["training"]["seed"])
    positive = sorted(positive, key=lambda window: _window_rank(window, seed))[:32]
    normal = sorted(normal, key=lambda window: _window_rank(window, seed))[:32]
    if not positive or not normal:
        raise ContractError("sanity gate needs positive and normal windows")
    windows = tuple(positive + normal)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _new_model(encoded_train.features.shape[1], config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    maximum_epochs = int(config["implementation_sanity_gate"]["maximum_epochs"])
    batch_size = int(config["training"]["batch_size"])
    _steps_per_epoch, total_steps, _warmup_steps = _schedule_geometry(
        config, window_count=len(windows)
    )
    positive_weight = _positive_weight(encoded_train.surface.labels)
    global_step = 0
    history: list[dict[str, Any]] = []
    final_metrics: dict[str, Any] = {}
    passed = False
    for epoch in range(1, maximum_epochs + 1):
        epoch_started = time.perf_counter()
        telemetry, global_step, lr = _train_epoch(
            model,
            optimizer,
            encoded_train,
            windows,
            config=config,
            positive_weight=positive_weight,
            device=device,
            epoch=epoch,
            global_step=global_step,
            total_steps=total_steps,
        )
        if epoch <= 3 or epoch % 5 == 0 or epoch == maximum_epochs:
            # Sanity evaluates raw weights.  A 0.999 EMA initialized from a
            # random model can lag a tiny overfit set for hundreds of updates
            # and create a false implementation failure.
            window_output = _predict_window_outputs(
                model, encoded_train, windows, batch_size=batch_size, device=device
            )
            covered = np.unique(np.concatenate([window.row_ids for window in windows]))
            row_probability = data_api.stitch_center_weighted(
                window_output.row_probability,
                windows,
                n_rows=encoded_train.surface.rows,
                require_row_ids=covered,
            )
            boundary_probability = data_api.stitch_center_weighted(
                window_output.boundary_probability,
                windows,
                n_rows=encoded_train.surface.rows,
                require_row_ids=covered,
            )
            type_probability = data_api.stitch_center_weighted(
                window_output.type_probability,
                windows,
                n_rows=encoded_train.surface.rows,
                require_row_ids=covered,
            )
            row_probability = np.nan_to_num(row_probability, nan=0.0)
            boundary_probability = np.nan_to_num(boundary_probability, nan=0.0)
            type_probability = np.nan_to_num(type_probability, nan=0.0)
            proposal = decode_long_event_segments(
                _decoder_row_probability(
                    PredictionBundle(
                        row_probability, boundary_probability, type_probability
                    ),
                    config,
                ),
                boundary_probability,
                encoded_train.layout,
                high_threshold=min(
                    float(value)
                    for value in config["decoder"]["q2_high_threshold_grid"]
                ),
                snap_radius=int(config["decoder"]["boundary_peak_snap_radius_rows"]),
                minimum_rows=int(config["decoder"]["minimum_added_segment_rows"]),
                maximum_rows=_maximum_segment_rows(config),
            )
            candidate = anchor_preserving_union(
                np.zeros(encoded_train.surface.rows, dtype=np.int8), proposal
            )
            truth_windows, _valid = data_api.materialize_windows(
                long_rows.astype(np.float32)[:, None], windows
            )
            truth_windows = truth_windows[:, :, 0]
            event_truth: list[int] = []
            event_prediction: list[int] = []
            normal_with_any = 0
            for index, window in enumerate(windows):
                length = window.valid_length
                if index < len(positive):
                    event_truth.extend(truth_windows[index, :length].astype(np.int8).tolist())
                    event_prediction.extend(candidate[window.row_ids].tolist())
                    event_truth.append(0)
                    event_prediction.append(0)
                elif bool(candidate[window.row_ids].any()):
                    normal_with_any += 1
            events = event_metrics(event_truth, event_prediction)
            final_metrics = {
                **events,
                "normal_windows_with_any_prediction": int(normal_with_any),
                "finite_loss_and_gradients": bool(
                    telemetry["nonfinite_count"] == 0
                    and math.isfinite(float(telemetry["total_loss"]))
                    and math.isfinite(float(telemetry["grad_norm_last"]))
                ),
                "evaluation_weights": "raw",
                "path_exercised": [
                    "valid_mask",
                    "stitch",
                    "type_conditioning",
                    "decoder",
                    "anchor_union",
                ],
            }
            gate = config["implementation_sanity_gate"]["pass_if_all"]
            passed = bool(
                final_metrics["finite_loss_and_gradients"]
                and final_metrics["event_recall_iou_0_70"]
                >= float(gate["event_recall_at_iou_0_70_min"])
                and final_metrics["median_event_iou"] >= float(gate["median_event_iou_min"])
                and final_metrics["normal_windows_with_any_prediction"]
                <= int(gate["normal_windows_with_any_prediction_max"])
            )
            history.append(
                {
                    **_history_record(
                        epoch=epoch,
                        telemetry=telemetry,
                        global_step=global_step,
                        learning_rate=lr,
                        elapsed_seconds=time.perf_counter() - epoch_started,
                    ),
                    **final_metrics,
                }
            )
            if passed:
                break
    return {
        "schema_version": "p1.mstcn_asrf.sanity.v1",
        "sample": {"positive_windows": len(positive), "normal_windows": len(normal)},
        "epochs": history[-1]["epoch"] if history else maximum_epochs,
        "positive_weight": positive_weight,
        "metrics": final_metrics,
        "history": history,
        "result": "PASS" if passed else "FAIL",
    }


def _encoder_receipt(encoder: Any) -> dict[str, Any]:
    return {
        "center": [float(value) for value in encoder.center],
        "scale": [float(value) for value in encoder.scale],
        "station_vocab": list(encoder.station_vocab),
        "layer_vocab": list(encoder.layer_vocab),
        "depth_regime_vocab": list(encoder.depth_regime_vocab),
        "numeric_names": list(encoder.numeric_names),
        "fit_ids_sha256": encoder.fit_ids_sha256,
        "uses_supplied_depth_regime": bool(encoder.uses_supplied_depth_regime),
        "depth_thresholds": (
            None
            if encoder.depth_thresholds is None
            else [float(value) for value in encoder.depth_thresholds]
        ),
        "preprocessing_fit_uses_holdout_rows": False,
    }


def commit_q2_qualification_grid(
    grid: QualificationGrid,
    *,
    key_sha256: str,
    config_sha256: str,
    artifact_dir: Path,
) -> Path:
    """Seal every Q2 selectable candidate before the sole Q2 truth read."""

    score_path = artifact_dir / "q2_qualification_grid_blind.npz"
    score_arrays = {
        "widths": grid.widths,
        "epochs": grid.epochs,
        "thresholds": grid.thresholds,
        "row_probability": grid.row_probability,
        "boundary_probability": grid.boundary_probability,
        "proposal": grid.proposal,
        "candidate": grid.candidate,
    }
    score_sha = _atomic_npz(
        score_path,
        **score_arrays,
    )
    receipt_path = artifact_dir / "q2_qualification_grid_receipt.json"
    _atomic_json(
        receipt_path,
        {
            "schema_version": "p1.mstcn_asrf.q2_qualification_blind.v1",
            "experiment_id": EXPERIMENT_ID,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "fold": Q2_FOLD,
            "score_path": score_path.name,
            "score_bytes": score_path.stat().st_size,
            "score_sha256": score_sha,
            "config_sha256": config_sha256,
            "ordered_holdout_key_sha256": key_sha256,
            "capacity_rows": int(len(grid.widths)),
            "threshold_count": int(len(grid.thresholds)),
            "holdout_rows": int(grid.row_probability.shape[1]),
            "array_inventory": _array_inventory(score_arrays),
            "seed_fit_count": int(len(grid.fit_receipts)),
            "fit_receipts": list(grid.fit_receipts),
            "prediction_representation": "raw_three_seed_ensemble_mean",
            "row_probability_array_role": "fixed long-type-conditioned decoder score",
            "type_head_used_by_decoder": True,
            "same_fold_holdout_truth_columns_opened_before_receipt": 0,
            "historical_prior_fold_labels_allowed": False,
            "prior_fold_metrics_computed_before_both_confirmatory_seals": False,
        },
    )
    return receipt_path


def commit_confirmatory_blind(
    blind: PredictionBundle,
    proposal: Any,
    candidate: Any,
    *,
    phase: str,
    fold: str,
    key_sha256: str,
    config_sha256: str,
    selected_recipe: dict[str, Any],
    fit_receipts: Sequence[dict[str, Any]],
    artifact_dir: Path,
) -> Path:
    score_path = artifact_dir / f"{phase}_confirmatory_blind.npz"
    score_arrays = {
        "row_probability": blind.row_probability,
        "boundary_probability": blind.boundary_probability,
        "type_probability": blind.type_probability,
        "proposal": proposal,
        "candidate": candidate,
    }
    score_sha = _atomic_npz(
        score_path,
        **score_arrays,
    )
    receipt_path = artifact_dir / f"{phase}_confirmatory_blind_receipt.json"
    _atomic_json(
        receipt_path,
        {
            "schema_version": "p1.mstcn_asrf.confirmatory_blind.v1",
            "experiment_id": EXPERIMENT_ID,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "phase": phase,
            "fold": fold,
            "score_path": score_path.name,
            "score_bytes": score_path.stat().st_size,
            "score_sha256": score_sha,
            "config_sha256": config_sha256,
            "ordered_holdout_key_sha256": key_sha256,
            "holdout_rows": int(blind.row_probability.shape[0]),
            "array_inventory": _array_inventory(score_arrays),
            "selected_recipe": selected_recipe,
            "fit_receipts": list(fit_receipts),
            "row_probability_array_role": "raw event probability",
            "decoder_uses_type_probability": True,
            "same_fold_holdout_truth_columns_opened_before_receipt": 0,
            "historical_prior_fold_labels_allowed": phase == "q4",
            "historical_prior_fold_labels_role": (
                "Q3 rows are historical training prefix rows for Q4 only"
                if phase == "q4"
                else "no earlier confirmatory fold is in the Q3 training prefix"
            ),
            "prior_fold_metrics_computed_before_both_confirmatory_seals": False,
        },
    )
    return receipt_path


def _array_inventory(arrays: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


def _resolve_blind_score_path(receipt_path: Path, receipt: dict[str, Any]) -> Path:
    parent = receipt_path.resolve().parent
    raw = receipt.get("score_path")
    if not isinstance(raw, str) or not raw or raw != Path(raw).name:
        raise ContractError("blind score_path must be one basename")
    if "/" in raw or "\\" in raw or Path(raw).suffix != ".npz":
        raise ContractError("blind score_path is not a local NPZ basename")
    score_path = (parent / raw).resolve()
    if score_path.parent != parent:
        raise ContractError("blind score_path escapes the run namespace")
    return score_path


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"sealed file is absent: {path.name}")
    return {"bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _held_file_identity(handle: Any) -> dict[str, Any]:
    position = handle.tell()
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
    size = int(os.fstat(handle.fileno()).st_size)
    handle.seek(position)
    return {"bytes": size, "sha256": digest.hexdigest()}


def verify_blind_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    if not receipt_path.is_file():
        raise ContractError("blind receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("blind receipt identity changed")
    if receipt.get("config_sha256") != _sha256(CONFIG_PATH):
        raise ContractError("blind receipt config identity changed")
    score_path = _resolve_blind_score_path(receipt_path, receipt)
    expected_score_identity = {
        "bytes": int(receipt["score_bytes"]),
        "sha256": str(receipt["score_sha256"]),
    }
    if _file_identity(score_path) != expected_score_identity:
        raise ContractError("blind probability bytes changed after receipt")
    if int(
        receipt.get("same_fold_holdout_truth_columns_opened_before_receipt", -1)
    ) != 0 or receipt.get(
        "prior_fold_metrics_computed_before_both_confirmatory_seals"
    ) is not False:
        raise ContractError("receipt does not attest a same-fold blind prediction")
    schema = receipt.get("schema_version")
    expected_prior_labels = (
        receipt.get("phase") == "q4"
        if schema == "p1.mstcn_asrf.confirmatory_blind.v1"
        else False
    )
    if receipt.get("historical_prior_fold_labels_allowed") is not expected_prior_labels:
        raise ContractError("blind receipt historical-label semantics changed")
    for fit_receipt in receipt.get("fit_receipts", []):
        for field in ("history_artifact", "checkpoint_artifact"):
            identity = fit_receipt.get(field)
            if not identity or identity.get("path") is None:
                continue
            artifact_path = (receipt_path.parent / identity["path"]).resolve()
            if artifact_path.parent != receipt_path.parent.resolve():
                raise ContractError("fit artifact path escapes the run namespace")
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != int(identity["bytes"])
                or _sha256(artifact_path) != identity["sha256"]
            ):
                raise ContractError(f"sealed {field} bytes changed")
    return receipt


def _read_verified_blind_npz(
    receipt_path: Path,
    *,
    expected_schema: str,
    expected_names: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one sealed NPZ from a held file with pre/post identity checks."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    receipt = verify_blind_receipt(receipt_path)
    if receipt.get("schema_version") != expected_schema:
        raise ContractError("blind NPZ receipt schema changed")
    score_path = _resolve_blind_score_path(receipt_path, receipt)
    expected_identity = {
        "bytes": int(receipt["score_bytes"]),
        "sha256": str(receipt["score_sha256"]),
    }
    before_path = _file_identity(score_path)
    if before_path != expected_identity:
        raise ContractError("blind NPZ identity changed before load")
    with score_path.open("rb") as handle:
        before_handle = _held_file_identity(handle)
        if before_handle != expected_identity:
            raise ContractError("held blind NPZ identity differs before load")
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            if set(archive.files) != expected_names:
                raise ContractError("sealed NPZ array inventory changed")
            arrays = {name: archive[name].copy() for name in archive.files}
        after_handle = _held_file_identity(handle)
        if after_handle != before_handle:
            raise ContractError("held blind NPZ changed during load")
    after_path = _file_identity(score_path)
    if after_path != before_path:
        raise ContractError("blind NPZ path identity changed during load")
    observed_inventory = _array_inventory(arrays)
    if receipt.get("array_inventory") != observed_inventory:
        raise ContractError("sealed NPZ shape/dtype inventory changed")
    return receipt, arrays


def load_sealed_q2_grid(receipt_path: Path) -> QualificationGrid:
    """Reload the immutable Q2 grid used for post-truth selection."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    expected_names = {
        "widths",
        "epochs",
        "thresholds",
        "row_probability",
        "boundary_probability",
        "proposal",
        "candidate",
    }
    receipt, arrays = _read_verified_blind_npz(
        receipt_path,
        expected_schema="p1.mstcn_asrf.q2_qualification_blind.v1",
        expected_names=expected_names,
    )
    config = _canonical_config()
    expected_epochs = np.asarray(_checkpoint_epochs(config), dtype=np.int16)
    expected_widths = np.repeat(
        np.asarray(config["architecture"]["q2_capacity_width_grid"], dtype=np.int16),
        len(expected_epochs),
    )
    expected_epoch_rows = np.tile(
        expected_epochs, len(config["architecture"]["q2_capacity_width_grid"])
    )
    expected_thresholds = np.asarray(
        config["decoder"]["q2_high_threshold_grid"], dtype=np.float64
    )
    rows = int(config["phase_protocols"]["q2"]["membership_identity"]["membership_rows"])
    capacities = len(expected_widths)
    thresholds = len(expected_thresholds)
    expected_dtypes = {
        "widths": "int16",
        "epochs": "int16",
        "thresholds": "float64",
        "row_probability": "float32",
        "boundary_probability": "float32",
        "proposal": "int8",
        "candidate": "int8",
    }
    expected_shapes = {
        "widths": (capacities,),
        "epochs": (capacities,),
        "thresholds": (thresholds,),
        "row_probability": (capacities, rows),
        "boundary_probability": (capacities, rows, 2),
        "proposal": (capacities, thresholds, rows),
        "candidate": (capacities, thresholds, rows),
    }
    if any(
        arrays[name].shape != expected_shapes[name]
        or str(arrays[name].dtype) != expected_dtypes[name]
        for name in expected_names
    ):
        raise ContractError("sealed Q2 grid shape or dtype differs from preregistration")
    if not (
        int(receipt.get("capacity_rows", -1)) == capacities == 126
        and int(receipt.get("threshold_count", -1)) == thresholds == 7
        and capacities * thresholds == 882
        and int(receipt.get("holdout_rows", -1)) == rows
        and int(receipt.get("seed_fit_count", -1)) == 6
        and len(receipt.get("fit_receipts", [])) == 6
        and np.array_equal(arrays["widths"], expected_widths)
        and np.array_equal(arrays["epochs"], expected_epoch_rows)
        and np.array_equal(arrays["thresholds"], expected_thresholds)
    ):
        raise ContractError("sealed Q2 finite grid identity changed")
    if not (
        np.isfinite(arrays["row_probability"]).all()
        and np.isfinite(arrays["boundary_probability"]).all()
        and ((arrays["row_probability"] >= 0.0) & (arrays["row_probability"] <= 1.0)).all()
        and (
            (arrays["boundary_probability"] >= 0.0)
            & (arrays["boundary_probability"] <= 1.0)
        ).all()
        and np.isin(arrays["proposal"], [0, 1]).all()
        and np.isin(arrays["candidate"], [0, 1]).all()
    ):
        raise ContractError("sealed Q2 arrays are nonfinite, nonprobabilistic, or nonbinary")
    return QualificationGrid(
        widths=arrays["widths"],
        epochs=arrays["epochs"],
        thresholds=arrays["thresholds"],
        row_probability=arrays["row_probability"],
        boundary_probability=arrays["boundary_probability"],
        proposal=arrays["proposal"],
        candidate=arrays["candidate"],
        fit_receipts=[],
    )


def validate_sealed_q2_decoder_semantics(
    grid: QualificationGrid,
    q2: EncodedSurface,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Replay every selectable Q2 decoder/union cell before opening Q2 truth."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    if q2.surface.anchor is None:
        raise ContractError("Q2 anchor is absent from semantic replay")
    checks = 0
    for capacity_index in range(len(grid.widths)):
        for threshold_index, threshold in enumerate(grid.thresholds.tolist()):
            replay_proposal = decode_long_event_segments(
                grid.row_probability[capacity_index],
                grid.boundary_probability[capacity_index],
                q2.layout,
                high_threshold=float(threshold),
                snap_radius=int(config["decoder"]["boundary_peak_snap_radius_rows"]),
                minimum_rows=int(config["decoder"]["minimum_added_segment_rows"]),
                maximum_rows=_maximum_segment_rows(config),
            )
            if not np.array_equal(
                replay_proposal, grid.proposal[capacity_index, threshold_index]
            ):
                raise ContractError("sealed Q2 proposal differs from decoder replay")
            replay_candidate = anchor_preserving_union(
                q2.surface.anchor, replay_proposal
            )
            if not np.array_equal(
                replay_candidate, grid.candidate[capacity_index, threshold_index]
            ):
                raise ContractError("sealed Q2 candidate differs from anchor union replay")
            checks += 1
    if checks != 882:
        raise ContractError("sealed Q2 semantic replay cell count changed")
    return {
        "schema_version": "p1.mstcn_asrf.q2_semantic_replay.v1",
        "decoder_cells_replayed": checks,
        "expected_decoder_cells": 882,
        "same_fold_truth_columns_read": 0,
        "result": "PASS",
    }


def load_sealed_confirmatory_candidate(
    receipt_path: Path,
    *,
    holdout: EncodedSurface,
    config: dict[str, Any],
    selected_recipe: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Reload and semantically replay one confirmatory blind candidate."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    expected_names = {
        "row_probability",
        "boundary_probability",
        "type_probability",
        "proposal",
        "candidate",
    }
    receipt, arrays = _read_verified_blind_npz(
        receipt_path,
        expected_schema="p1.mstcn_asrf.confirmatory_blind.v1",
        expected_names=expected_names,
    )
    phase = str(receipt.get("phase", ""))
    if phase not in {"q3", "q4"}:
        raise ContractError("confirmatory phase is not registered")
    protocol = config["phase_protocols"][phase]
    rows = int(protocol["membership_identity"]["membership_rows"])
    if receipt.get("fold") != protocol["fold"] or int(
        receipt.get("holdout_rows", -1)
    ) != rows or len(receipt.get("fit_receipts", [])) != 3:
        raise ContractError("confirmatory receipt population changed")
    if receipt.get("selected_recipe") != selected_recipe:
        raise ContractError("confirmatory selected recipe changed")
    expected_dtypes = {
        "row_probability": "float32",
        "boundary_probability": "float32",
        "type_probability": "float32",
        "proposal": "int8",
        "candidate": "int8",
    }
    expected_shapes = {
        "row_probability": (rows,),
        "boundary_probability": (rows, 2),
        "type_probability": (rows, len(TYPE_NAMES)),
        "proposal": (rows,),
        "candidate": (rows,),
    }
    if any(
        arrays[name].shape != expected_shapes[name]
        or str(arrays[name].dtype) != expected_dtypes[name]
        for name in expected_names
    ):
        raise ContractError("sealed confirmatory shape or dtype changed")
    if not (
        all(
            np.isfinite(arrays[name]).all()
            and ((arrays[name] >= 0.0) & (arrays[name] <= 1.0)).all()
            for name in ("row_probability", "boundary_probability", "type_probability")
        )
        and np.isin(arrays["proposal"], [0, 1]).all()
        and np.isin(arrays["candidate"], [0, 1]).all()
    ):
        raise ContractError(
            "sealed confirmatory arrays are nonfinite, nonprobabilistic, or nonbinary"
        )
    bundle = PredictionBundle(
        arrays["row_probability"],
        arrays["boundary_probability"],
        arrays["type_probability"],
    )
    replay_proposal = decode_long_event_segments(
        _decoder_row_probability(bundle, config),
        bundle.boundary_probability,
        holdout.layout,
        high_threshold=float(selected_recipe["threshold"]),
        snap_radius=int(config["decoder"]["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(config["decoder"]["minimum_added_segment_rows"]),
        maximum_rows=_maximum_segment_rows(config),
    )
    if not np.array_equal(replay_proposal, arrays["proposal"]):
        raise ContractError("sealed confirmatory proposal differs from decoder replay")
    if holdout.surface.anchor is None:
        raise ContractError("confirmatory anchor is absent from semantic replay")
    replay_candidate = anchor_preserving_union(
        holdout.surface.anchor, replay_proposal
    )
    if not np.array_equal(replay_candidate, arrays["candidate"]):
        raise ContractError(
            "sealed confirmatory candidate differs from anchor union replay"
        )
    replay_receipt = {
        "schema_version": "p1.mstcn_asrf.confirmatory_semantic_replay.v1",
        "phase": phase,
        "fold": protocol["fold"],
        "rows": rows,
        "same_fold_truth_columns_read": 0,
        "decoder_replayed": True,
        "anchor_union_replayed": True,
        "result": "PASS",
    }
    return arrays["candidate"], replay_receipt


def select_q2_recipe(
    truth: Any,
    q2: EncodedSurface,
    grid: QualificationGrid,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Select one sealed capacity recipe using current-Router union F1."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    y = truth["label"].to_numpy(dtype=np.int8)
    anchor = np.asarray(q2.surface.anchor, dtype=np.int8)
    anchor_score = binary_metrics(y, anchor)
    long_mask = _long_event_mask(
        truth, q2.layout, minimum_rows=int(config["decoder"]["minimum_added_segment_rows"])
    )
    records: list[dict[str, Any]] = []
    best_key: tuple[float, float, int, int, float, int] | None = None
    best_record: dict[str, Any] | None = None
    for capacity_index, (width, epoch) in enumerate(
        zip(grid.widths.tolist(), grid.epochs.tolist(), strict=True)
    ):
        for threshold_index, threshold in enumerate(grid.thresholds.tolist()):
            candidate = grid.candidate[capacity_index, threshold_index]
            proposal = grid.proposal[capacity_index, threshold_index]
            score = binary_metrics(y, candidate)
            proposal_score = binary_metrics(y, proposal)
            added = (candidate == 1) & (anchor == 0)
            added_count = int(added.sum())
            added_tp = int(np.sum(added & (y == 1)))
            added_fp = int(np.sum(added & (y == 0)))
            added_precision = float(added_tp / added_count) if added_count else 0.0
            anchor_long = float(anchor[long_mask].mean()) if long_mask.any() else 0.0
            candidate_long = (
                float(candidate[long_mask].mean()) if long_mask.any() else 0.0
            )
            record = {
                "capacity_index": capacity_index,
                "threshold_index": threshold_index,
                "width": int(width),
                "epoch": int(epoch),
                "threshold": float(threshold),
                "representation": "raw_three_seed_ensemble_mean",
                "candidate": score,
                "standalone_proposal": proposal_score,
                "delta_f1": float(score["f1"] - anchor_score["f1"]),
                "added_rows": added_count,
                "added_true_positive_rows": added_tp,
                "added_false_positive_rows": added_fp,
                "added_row_precision": added_precision,
                "long_event_recall_gain": candidate_long - anchor_long,
            }
            key = (
                float(score["f1"]),
                added_precision,
                -added_fp,
                -int(width),
                float(threshold),
                -int(epoch),
            )
            records.append(record)
            if best_key is None or key > best_key:
                best_key = key
                best_record = record
    if best_record is None:
        raise ContractError("Q2 qualification grid is empty")
    capacity_index = int(best_record["capacity_index"])
    threshold_index = int(best_record["threshold_index"])
    selected_candidate = grid.candidate[capacity_index, threshold_index]
    selected_proposal = grid.proposal[capacity_index, threshold_index]
    metrics = evaluate_q2(
        truth,
        q2,
        selected_proposal,
        selected_candidate,
        config=config,
    )
    selected_epoch = int(best_record["epoch"])
    maximum_epoch = int(config["training"]["maximum_epochs"])
    return {
        "schema_version": "p1.mstcn_asrf.q2_selection.v1",
        "role": "qualification_and_finite_grid_selection_only_not_promotion_evidence",
        "anchor": anchor_score,
        "selected": best_record,
        "selected_metrics": metrics,
        "grid_candidates": len(records),
        "grid_records": records,
        "convergence_evidence": {
            "curve_rows": len(records),
            "widths": sorted({int(record["width"]) for record in records}),
            "epochs": sorted({int(record["epoch"]) for record in records}),
            "thresholds": sorted({float(record["threshold"]) for record in records}),
            "selected_epoch": selected_epoch,
            "maximum_epoch": maximum_epoch,
            "right_censored_at_max_epoch": selected_epoch == maximum_epoch,
            "interpretation": (
                "RIGHT_CENSORED_AT_MAX_EPOCH_NOT_PROVEN_CONVERGED"
                if selected_epoch == maximum_epoch
                else "Q2_SELECTED_BEFORE_MAX_EPOCH_NO_CONVERGENCE_CLAIM"
            ),
        },
        "optimistic_max_selection_acknowledged": True,
        "result": "PASS" if metrics["result"] == "PASS" else "FAIL",
    }


def load_fold_truth_after_receipts(
    config: dict[str, Any],
    holdout: RowSurface,
    receipt_paths: Sequence[Path],
    *,
    fold: str,
    root: Path = ROOT,
) -> Any:
    """Open one fold target only after every supplied blind receipt verifies."""

    import pyarrow.dataset as dataset

    expected_key = _ordered_key_sha(holdout.keys)
    if not receipt_paths:
        raise ContractError("at least one blind receipt is required before truth access")
    for path in receipt_paths:
        receipt = verify_blind_receipt(path)
        sealed_key = receipt.get("ordered_holdout_key_sha256")
        if sealed_key != expected_key:
            raise ContractError("blind receipt was committed for different holdout keys")
    with _verified_immutable_read(
        config, "frozen_truth_and_folds", root=root
    ) as oof_path:
        scanner = dataset.dataset(oof_path, format="parquet").scanner(
            columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
            filter=dataset.field("fold") == fold,
            use_threads=True,
        )
        truth = scanner.to_table().to_pandas().reset_index(drop=True)
    truth, _membership_receipt = _validate_registered_holdout_membership(
        truth, config, fold=fold
    )
    if not _keys_equal(holdout.keys, truth):
        raise ContractError(f"opened {fold} truth keys differ from the blind surface")
    return truth


def _long_event_mask(truth: Any, layout: Any, *, minimum_rows: int = 19) -> Any:
    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    labels = truth["label"].to_numpy(dtype=np.int8)
    types = truth["anomaly_type"].fillna("").astype(str).str.casefold().to_numpy()
    eligible_type = np.asarray(
        [
            ("offset" in value) or ("drift" in value) or ("noise" in value)
            for value in types
        ],
        dtype=bool,
    )
    result = np.zeros(len(labels), dtype=bool)
    for segment in layout.segments:
        rows = segment.row_ids
        values = labels[rows]
        cursor = 0
        while cursor < len(rows):
            if values[cursor] == 0:
                cursor += 1
                continue
            stop = cursor + 1
            while stop < len(rows) and values[stop] == 1:
                stop += 1
            event_rows = rows[cursor:stop]
            if len(event_rows) >= minimum_rows and bool(eligible_type[event_rows].any()):
                result[event_rows] = True
            cursor = stop
    return result


def evaluate_q2(
    truth: Any,
    q2: EncodedSurface,
    proposal: Any,
    candidate: Any,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    y = truth["label"].to_numpy(dtype=np.int8)
    anchor = np.asarray(q2.surface.anchor, dtype=np.int8)
    proposal = np.asarray(proposal, dtype=np.int8)
    candidate = np.asarray(candidate, dtype=np.int8)
    anchor_score = binary_metrics(y, anchor)
    candidate_score = binary_metrics(y, candidate)
    added = (candidate == 1) & (anchor == 0)
    added_count = int(added.sum())
    added_precision = float(y[added].mean()) if added_count else 0.0
    long_mask = _long_event_mask(
        truth, q2.layout, minimum_rows=int(config["decoder"]["minimum_added_segment_rows"])
    )
    denominator = int(long_mask.sum())
    anchor_long_recall = float(anchor[long_mask].mean()) if denominator else 0.0
    candidate_long_recall = float(candidate[long_mask].mean()) if denominator else 0.0
    normal = y == 0
    anchor_fp_rows = int(np.sum(normal & (anchor == 1)))
    candidate_fp_rows = int(np.sum(normal & (candidate == 1)))
    if anchor_fp_rows == 0:
        fp_ratio = 1.0 if candidate_fp_rows == 0 else 1.0e12
    else:
        fp_ratio = float(candidate_fp_rows / anchor_fp_rows)
    removed = int(np.sum((anchor == 1) & (candidate == 0)))
    delta_f1 = float(candidate_score["f1"] - anchor_score["f1"])
    long_gain = candidate_long_recall - anchor_long_recall
    gate = config["q2_continuation_gate"]
    checks = {
        "delta_f1": delta_f1 >= float(gate["delta_f1_min"]),
        "added_row_precision": added_precision >= float(gate["added_row_precision_min"]),
        "long_event_recall_gain": long_gain >= float(gate["long_event_recall_gain_min"]),
        "normal_false_positive_row_ratio": fp_ratio
        <= float(gate["normal_false_positive_row_ratio_max"]),
        "anchor_positive_removed_rows": removed == int(gate["anchor_positive_removed_rows"]),
    }
    return {
        "schema_version": "p1.mstcn_asrf.q2_metrics.v1",
        "q2_rows": len(y),
        "anchor": anchor_score,
        "candidate": candidate_score,
        "delta_f1": delta_f1,
        "added_rows": added_count,
        "added_row_precision": added_precision,
        "long_event_rows": denominator,
        "anchor_long_event_recall": anchor_long_recall,
        "candidate_long_event_recall": candidate_long_recall,
        "long_event_recall_gain": long_gain,
        "anchor_normal_false_positive_rows": anchor_fp_rows,
        "candidate_normal_false_positive_rows": candidate_fp_rows,
        "normal_false_positive_row_ratio": fp_ratio,
        "anchor_positive_removed_rows": removed,
        "gate_checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }


def _paired_day_block_bootstrap(
    folds: Sequence[tuple[Any, Any, Any, Any]],
    *,
    replicates: int,
    block_days: int,
    seed: int,
) -> dict[str, Any]:
    """Paired circular whole-day bootstrap over the pooled Q3+Q4 cross-section."""

    np, pd, _torch, _model_api, _data_api = _load_scientific()
    if not folds:
        raise ContractError("confirmatory bootstrap population is empty")
    rng = np.random.default_rng(seed)
    date_parts: list[Any] = []
    truth_parts: list[Any] = []
    anchor_parts: list[Any] = []
    candidate_parts: list[Any] = []
    fold_day_sets: list[set[Any]] = []
    for keys, truth, anchor, candidate in folds:
        dates = pd.to_datetime(keys["time"], utc=True, format="mixed").dt.tz_convert(
            "Asia/Seoul"
        ).dt.date
        y = np.asarray(truth, dtype=np.int8)
        a = np.asarray(anchor, dtype=np.int8)
        c = np.asarray(candidate, dtype=np.int8)
        if not (len(dates) == len(y) == len(a) == len(c)):
            raise ContractError("confirmatory bootstrap arrays are misaligned")
        if len(dates) == 0:
            raise ContractError("confirmatory bootstrap fold has no KST calendar day")
        date_parts.append(dates.reset_index(drop=True))
        truth_parts.append(y)
        anchor_parts.append(a)
        candidate_parts.append(c)
        fold_day_sets.append(set(dates.tolist()))
    pooled_dates = pd.concat(date_parts, ignore_index=True)
    y = np.concatenate(truth_parts)
    a = np.concatenate(anchor_parts)
    c = np.concatenate(candidate_parts)
    unique_dates = tuple(sorted(set(pooled_dates.tolist())))
    anchor_counts = np.zeros((len(unique_dates), 3), dtype=np.int64)
    candidate_counts = np.zeros((len(unique_dates), 3), dtype=np.int64)
    date_values = pooled_dates.to_numpy()
    for index, day in enumerate(unique_dates):
        mask = date_values == day
        for output, pred in ((anchor_counts, a), (candidate_counts, c)):
            output[index, 0] = np.sum(mask & (y == 1) & (pred == 1))
            output[index, 1] = np.sum(mask & (y == 0) & (pred == 1))
            output[index, 2] = np.sum(mask & (y == 1) & (pred == 0))
    deltas = np.empty(replicates, dtype=np.float64)
    days = len(unique_dates)
    chunks = math.ceil(days / block_days)
    for replicate in range(replicates):
        sampled: list[int] = []
        for _chunk in range(chunks):
            start = int(rng.integers(0, days))
            sampled.extend((start + offset) % days for offset in range(block_days))
        ids = np.asarray(sampled[:days], dtype=np.int64)
        anchor_total = anchor_counts[ids].sum(axis=0)
        candidate_total = candidate_counts[ids].sum(axis=0)
        anchor_denominator = 2 * anchor_total[0] + anchor_total[1] + anchor_total[2]
        candidate_denominator = (
            2 * candidate_total[0] + candidate_total[1] + candidate_total[2]
        )
        anchor_f1 = (
            2.0 * anchor_total[0] / anchor_denominator if anchor_denominator else 0.0
        )
        candidate_f1 = (
            2.0 * candidate_total[0] / candidate_denominator
            if candidate_denominator
            else 0.0
        )
        deltas[replicate] = candidate_f1 - anchor_f1
    return {
        "method": "paired circular moving-block bootstrap over pooled Q3+Q4 whole KST-day cross-sections",
        "replicates": replicates,
        "block_days": block_days,
        "seed": seed,
        "pooled_unique_kst_calendar_days": days,
        "source_fold_shared_kst_calendar_days": sum(
            len(fold_day_sets[left].intersection(fold_day_sets[right]))
            for left in range(len(fold_day_sets))
            for right in range(left + 1, len(fold_day_sets))
        ),
        "shared_kst_calendar_day_cross_sections_sampled_once": True,
        "delta_f1_mean": float(deltas.mean()),
        "ci90_lower": float(np.quantile(deltas, 0.05)),
        "ci90_upper": float(np.quantile(deltas, 0.95)),
    }


def evaluate_confirmatory_folds(
    truths: dict[str, Any],
    holdouts: dict[str, EncodedSurface],
    candidates: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Primary Q3+Q4 pooled evaluation after both blind seals exist."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    fold_metrics: dict[str, Any] = {}
    truth_parts: list[Any] = []
    anchor_parts: list[Any] = []
    candidate_parts: list[Any] = []
    station_parts: list[Any] = []
    bootstrap_folds: list[tuple[Any, Any, Any, Any]] = []
    for phase in ("q3", "q4"):
        truth = truths[phase]["label"].to_numpy(dtype=np.int8)
        anchor = np.asarray(holdouts[phase].surface.anchor, dtype=np.int8)
        candidate = np.asarray(candidates[phase], dtype=np.int8)
        anchor_score = binary_metrics(truth, anchor)
        candidate_score = binary_metrics(truth, candidate)
        fold_metrics[phase] = {
            "anchor": anchor_score,
            "candidate": candidate_score,
            "delta_f1": float(candidate_score["f1"] - anchor_score["f1"]),
        }
        truth_parts.append(truth)
        anchor_parts.append(anchor)
        candidate_parts.append(candidate)
        station_parts.append(
            holdouts[phase].surface.keys["station"].astype(str).to_numpy()
        )
        bootstrap_folds.append(
            (holdouts[phase].surface.keys, truth, anchor, candidate)
        )
    y = np.concatenate(truth_parts)
    anchor = np.concatenate(anchor_parts)
    candidate = np.concatenate(candidate_parts)
    stations = np.concatenate(station_parts)
    anchor_score = binary_metrics(y, anchor)
    candidate_score = binary_metrics(y, candidate)
    expected_counts = config["confirmatory_gate"][
        "expected_q3_q4_current_router_counts"
    ]
    if any(
        int(anchor_score[name]) != int(expected_counts[name])
        for name in ("tp", "fp", "fn")
    ):
        raise ContractError("confirmatory current-Router baseline identity changed")
    expected_formula = config["confirmatory_gate"][
        "expected_q3_q4_current_router_f1_exact_formula"
    ]
    expected_f1 = 17922.0 / 19849.0
    if expected_formula != "17922/19849" or not math.isclose(
        float(anchor_score["f1"]), expected_f1, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ContractError("confirmatory current-Router exact F1 identity changed")
    added = (candidate == 1) & (anchor == 0)
    added_count = int(added.sum())
    added_precision = float(y[added].mean()) if added_count else 0.0
    removed = int(np.sum((anchor == 1) & (candidate == 0)))
    delta = float(candidate_score["f1"] - anchor_score["f1"])
    by_station: dict[str, Any] = {}
    for station in sorted(set(stations.tolist())):
        mask = stations == station
        station_anchor = binary_metrics(y[mask], anchor[mask])
        station_candidate = binary_metrics(y[mask], candidate[mask])
        by_station[station] = {
            "anchor": station_anchor,
            "candidate": station_candidate,
            "delta_f1": float(station_candidate["f1"] - station_anchor["f1"]),
        }
    stations_improved = sum(
        int(value["delta_f1"] > 0.0) for value in by_station.values()
    )
    gate = config["confirmatory_gate"]
    expected_station_total = int(
        gate["high_impact_official_probe"]["stations_total"]
    )
    if len(by_station) != expected_station_total:
        raise ContractError("confirmatory station inventory changed")
    bootstrap = _paired_day_block_bootstrap(
        bootstrap_folds,
        replicates=int(config["confirmatory_gate"]["bootstrap_replicates"]),
        block_days=int(config["confirmatory_gate"]["bootstrap_block_kst_days"]),
        seed=int(config["confirmatory_gate"]["bootstrap_seed"]),
    )
    research_checks = {
        "pooled_delta_f1": delta >= float(gate["research_success"]["pooled_delta_f1_min"]),
        "q3_positive": fold_metrics["q3"]["delta_f1"] > 0.0,
        "q4_positive": fold_metrics["q4"]["delta_f1"] > 0.0,
        "ci90_lower_positive": bootstrap["ci90_lower"] > 0.0,
        "anchor_positive_removed_rows": removed == 0,
    }
    high_checks = {
        "pooled_f1_endpoint": candidate_score["f1"]
        >= float(gate["high_impact_official_probe"]["pooled_f1_min"]),
        "pooled_delta_f1": delta
        >= float(gate["high_impact_official_probe"]["pooled_delta_f1_min"]),
        "q3_delta_f1": fold_metrics["q3"]["delta_f1"]
        >= float(gate["high_impact_official_probe"]["each_fold_delta_f1_min"]),
        "q4_delta_f1": fold_metrics["q4"]["delta_f1"]
        >= float(gate["high_impact_official_probe"]["each_fold_delta_f1_min"]),
        "added_row_precision": added_precision
        >= float(gate["high_impact_official_probe"]["added_row_precision_min"]),
        "ci90_lower": bootstrap["ci90_lower"]
        >= float(gate["high_impact_official_probe"]["ci90_lower_min"]),
        "anchor_positive_removed_rows": removed == 0,
        "stations_improved": stations_improved
        >= int(gate["high_impact_official_probe"]["stations_improved_min"]),
    }
    return {
        "schema_version": "p1.mstcn_asrf.confirmatory_metrics.v1",
        "role": "Q3+Q4 primary confirmatory; Q2 excluded from promotion evidence",
        "folds": fold_metrics,
        "pooled": {
            "rows": len(y),
            "anchor": anchor_score,
            "candidate": candidate_score,
            "delta_f1": delta,
            "added_rows": added_count,
            "added_row_precision": added_precision,
            "anchor_positive_removed_rows": removed,
        },
        "bootstrap": bootstrap,
        "by_station": by_station,
        "stations_improved": stations_improved,
        "research_success_checks": research_checks,
        "high_impact_official_probe_checks": high_checks,
        "research_result": "PASS" if all(research_checks.values()) else "FAIL",
        "high_impact_official_probe_result": (
            "PASS" if all(high_checks.values()) else "FAIL"
        ),
        "three_official_points_claimed": False,
    }


def _training_surface_for_cutoff(
    surfaces: FrozenSurfaces,
    config: dict[str, Any],
    cutoff_utc: str,
    *,
    root: Path = ROOT,
) -> RowSurface:
    """Build a phase prefix using target projection/filter pushdown only."""

    np, _pd, _torch, _model_api, _data_api = _load_scientific()
    with _verified_immutable_read(config, "training_labels", root=root) as target_path:
        targets = _read_training_targets_prefix(target_path, cutoff_utc)
    cutoff_kst = _kst_cutoff_text(cutoff_utc)
    ids = np.flatnonzero(
        surfaces.all_keys["time"].astype(str).to_numpy() <= cutoff_kst
    )
    keys = surfaces.all_keys.iloc[ids].reset_index(drop=True)
    if not _keys_equal(keys, targets.loc[:, KEY_COLUMNS]):
        raise ContractError("phase training target/key alignment changed")
    labels = targets["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ContractError("phase training target is not binary")
    selected = surfaces.feature_table.iloc[ids].reset_index(drop=True)
    return RowSurface(
        keys=keys,
        numeric=selected.loc[:, surfaces.numeric_names].to_numpy(dtype=np.float32),
        station=selected["station"].astype(str).to_numpy(),
        layer_category=selected["layer_category"].astype(str).to_numpy(),
        depth_regime=None,
        labels=labels,
        anomaly_type=targets["anomaly_type"].fillna("").astype(str).to_numpy(),
        depth=selected["depth_raw"].to_numpy(dtype=np.float32),
    )


def _prepare_phase_surfaces(
    surfaces: FrozenSurfaces,
    config: dict[str, Any],
    phase: str,
    *,
    root: Path = ROOT,
) -> tuple[Any, EncodedSurface, EncodedSurface, dict[str, Any]]:
    """Fit preprocessing on one phase prefix and transform one blind fold."""

    np, pd, _torch, _model_api, _data_api = _load_scientific()
    protocol = config["phase_protocols"][phase]
    holdout = {"q2": surfaces.q2, "q3": surfaces.q3, "q4": surfaces.q4}[phase]
    training = _training_surface_for_cutoff(
        surfaces, config, protocol["training_max_time_utc"], root=root
    )
    encoder, encoded = _fit_encoder_and_transform(
        training,
        [holdout],
        fit_ids=np.arange(training.rows, dtype=np.int64),
        forbidden_ids=np.asarray([], dtype=np.int64),
        numeric_names=surfaces.numeric_names,
    )
    train_max = pd.to_datetime(training.keys["time"], utc=True, format="mixed").max()
    holdout_min = pd.to_datetime(holdout.keys["time"], utc=True, format="mixed").min()
    dependency = config["data_contract"]["cached_feature_dependency_audit"]
    required_hours = int(dependency["bounded_future_support_hours_max"]) + int(
        dependency["validation_bounded_past_support_hours_max"]
    )
    actual_hours = float((holdout_min - train_max).total_seconds() / 3600.0)
    if actual_hours <= required_hours:
        raise ContractError(f"{phase} cached feature supports overlap")
    expected_width = int(config["data_contract"]["expected_runtime_input_features"])
    if encoded[0].features.shape[1] != expected_width:
        raise ContractError(f"{phase} runtime input width changed")
    if encoded[1].features.shape[1] != expected_width:
        raise ContractError(f"{phase} train/holdout encoded widths differ")
    receipt = {
        "schema_version": "p1.mstcn_asrf.phase_split.v1",
        "phase": phase,
        "fold": protocol["fold"],
        "training_rows": training.rows,
        "holdout_rows": holdout.rows,
        "training_max_time_utc": protocol["training_max_time_utc"],
        "holdout_start_utc": protocol["holdout_start_utc"],
        "holdout_end_utc_exclusive": protocol["holdout_end_utc_exclusive"],
        "actual_separation_hours": actual_hours,
        "required_feature_non_overlap_hours": required_hours,
        "feature_non_overlap_slack_hours": actual_hours - required_hours,
        "split_before_windowing": True,
        "cross_split_window_count": 0,
        "training_key_sha256": _ordered_key_sha(training.keys),
        "holdout_key_sha256": _ordered_key_sha(holdout.keys),
        "holdout_membership_sha256": surfaces.membership_sha256[protocol["fold"]],
        "holdout_membership_identity": surfaces.membership_receipts[
            protocol["fold"]
        ],
        "holdout_rows_used_to_fit_preprocessing": 0,
        "holdout_rows_used_to_train": 0,
        "holdout_truth_columns_read": 0,
        "runtime_input_features": int(encoded[0].features.shape[1]),
    }
    return encoder, encoded[0], encoded[1], receipt


def _attempt_receipt(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "p1.mstcn_asrf.attempt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "config_sha256": preflight["config_sha256"],
        "immutable_input_sha256": {
            name: value["sha256"] for name, value in preflight["immutable_inputs"].items()
        },
        "execution_implementation_sha256": {
            name: value["sha256"]
            for name, value in preflight["external_implementation_attestation"][
                "identities"
            ].items()
        },
        "execution_launcher_sha256": preflight["external_implementation_attestation"][
            "launcher_sha256"
        ],
        "runtime_identity": preflight["runtime_identity_immediately_before_lock"],
        "one_shot": True,
    }


def acquire_attempt_namespace(
    preflight: dict[str, Any],
    *,
    path: Path = ATTEMPT_LOCK,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, Any]:
    """Acquire the lock and artifact namespace with exact-lock rollback on failure."""

    if path.exists() or artifact_dir.exists():
        raise FileExistsError("scientific attempt namespace already exists")
    if not path.parent.is_dir() or not artifact_dir.parent.is_dir():
        raise FileNotFoundError("scientific attempt namespace parent is absent")
    receipt = _attempt_receipt(preflight)
    payload = _json_bytes(receipt)
    _exclusive_json(path, receipt)
    try:
        artifact_dir.mkdir(parents=False, exist_ok=False)
    except BaseException:
        # Never remove a lock that another actor changed after our exclusive
        # create.  Roll back only the exact bytes created by this invocation.
        if path.is_file() and path.read_bytes() == payload:
            path.unlink()
        raise
    return receipt


def execute_protocol(
    *,
    root: Path = ROOT,
    artifact_dir: Path = ARTIFACT_DIR,
    implementation_attestation: dict[str, Any] | None = None,
    launcher_capability: object | None = None,
) -> dict[str, Any]:
    """Execute Q2 qualification, then sealed Q3+Q4 confirmation exactly once."""

    if launcher_capability is not _SEALED_LAUNCHER_CAPABILITY:
        raise ContractError("direct runner API scientific execution is disabled")
    config_path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    lock_path = root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    if lock_path.exists() or artifact_dir.exists():
        raise FileExistsError("scientific attempt namespace already exists")

    # Every read-only infrastructure and data-dependency check precedes the
    # one-shot scientific lock.  No stochastic fit or holdout truth read occurs
    # above acquire_attempt_namespace.
    preflight = check_only(root=root)
    preflight["external_implementation_attestation"] = (
        _verify_external_implementation_attestation(
            implementation_attestation,
            launcher_capability=launcher_capability,
            root=root,
        )
    )
    config = _canonical_config(config_path)
    _np, _pd, torch, _model_api, _data_api = _load_scientific()
    if not torch.cuda.is_available():
        raise ContractError("the registered bf16 protocol requires a CUDA device")
    device = torch.device("cuda")
    surfaces = load_blind_surfaces(config, root=root)
    q2_encoder, q2_train, q2, q2_split = _prepare_phase_surfaces(
        surfaces, config, "q2", root=root
    )
    if q2.surface.anchor is None:
        raise ContractError("Q2 current-Router anchor is absent before scientific lock")
    if int(q2_train.features.shape[1]) != int(
        config["data_contract"]["expected_runtime_input_features"]
    ):
        raise ContractError("runtime input width differs from preregistration")

    implementation_before_lock = _verify_external_implementation_attestation(
        implementation_attestation,
        launcher_capability=launcher_capability,
        root=root,
    )
    if implementation_before_lock != preflight["external_implementation_attestation"]:
        raise ContractError("implementation identity changed during read-only preflight")
    preflight["external_implementation_attestation_immediately_before_lock"] = (
        implementation_before_lock
    )
    runtime_before_lock = verify_runtime_identity(config)
    if runtime_before_lock != preflight["runtime_identity"]:
        raise ContractError("runtime identity changed during read-only preflight")
    preflight["runtime_identity_immediately_before_lock"] = runtime_before_lock
    acquire_attempt_namespace(
        preflight,
        path=lock_path,
        artifact_dir=artifact_dir,
    )
    started = datetime.now(UTC)
    terminal_path = artifact_dir / "terminal_result.json"
    try:
        _atomic_json(artifact_dir / "preflight.json", preflight)
        _atomic_json(
            artifact_dir / "feature_dependency_receipt.json",
            surfaces.dependency_receipt,
        )
        _atomic_json(artifact_dir / "q2_split.json", q2_split)
        _atomic_json(artifact_dir / "q2_encoder.json", _encoder_receipt(q2_encoder))

        sanity_config = _config_for_capacity(
            config,
            width=int(config["architecture"]["q2_capacity_width_grid"][0]),
            seed=int(config["training"]["ensemble_seeds"][0]),
        )
        sanity = run_sanity_gate(q2_train, config=sanity_config, device=device)
        _atomic_json(artifact_dir / "sanity_gate.json", sanity)
        if sanity["result"] != "PASS":
            result = {
                "schema_version": "p1.mstcn_asrf.terminal.v2",
                "experiment_id": EXPERIMENT_ID,
                "status": "NO_GO_IMPLEMENTATION_GATE",
                "started_at_utc": started.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "all_holdout_truth_reads": 0,
                "sanity_gate": sanity,
            }
            _atomic_json(terminal_path, result)
            return result

        grid = fit_q2_qualification_grid(
            q2_train,
            q2,
            config=config,
            device=device,
            artifact_dir=artifact_dir,
        )
        _atomic_json(artifact_dir / "q2_capacity_fit_receipts.json", grid.fit_receipts)
        q2_receipt = commit_q2_qualification_grid(
            grid,
            key_sha256=surfaces.membership_sha256["2025_q2"],
            config_sha256=preflight["config_sha256"],
            artifact_dir=artifact_dir,
        )
        sealed_q2_grid = load_sealed_q2_grid(q2_receipt)
        q2_semantic_replay = validate_sealed_q2_decoder_semantics(
            sealed_q2_grid, q2, config=config
        )
        _atomic_json(
            artifact_dir / "q2_blind_semantic_replay.json", q2_semantic_replay
        )
        q2_truth = load_fold_truth_after_receipts(
            config,
            q2.surface,
            [q2_receipt],
            fold="2025_q2",
            root=root,
        )
        selection = select_q2_recipe(
            q2_truth, q2, sealed_q2_grid, config=config
        )
        _atomic_json(artifact_dir / "q2_selection.json", selection)
        if selection["result"] != "PASS":
            result = {
                "schema_version": "p1.mstcn_asrf.terminal.v2",
                "experiment_id": EXPERIMENT_ID,
                "status": "NO_GO_Q2_QUALIFICATION",
                "started_at_utc": started.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "q2_role": "selection_only_not_promotion_evidence",
                "q2_selection": selection,
                "q3_q4_training_started": False,
            }
            _atomic_json(terminal_path, result)
            return result

        selected = selection["selected"]
        selected_recipe = {
            "schema_version": "p1.mstcn_asrf.selected_recipe.v1",
            "width": int(selected["width"]),
            "batch_size": int(
                config["architecture"]["batch_size_by_width"][str(selected["width"])]
            ),
            "epoch": int(selected["epoch"]),
            "threshold": float(selected["threshold"]),
            "representation": "raw_three_seed_ensemble_mean",
            "seeds": list(config["training"]["ensemble_seeds"]),
            "decoder": config["decoder"],
            "source": "sealed Q2 qualification grid",
            "q3_or_q4_result_driven_changes_authorized": False,
            "convergence_assessment": (
                "RIGHT_CENSORED_AT_MAX_EPOCH_NOT_PROVEN_CONVERGED"
                if int(selected["epoch"]) == int(config["training"]["maximum_epochs"])
                else "Q2_SELECTED_BEFORE_MAX_EPOCH_NO_CONVERGENCE_CLAIM"
            ),
        }
        _atomic_json(artifact_dir / "selected_recipe.json", selected_recipe)

        confirm_receipts: dict[str, Path] = {}
        confirm_encoded: dict[str, EncodedSurface] = {}
        for phase in ("q3", "q4"):
            encoder, training, holdout, split = _prepare_phase_surfaces(
                surfaces, config, phase, root=root
            )
            _atomic_json(artifact_dir / f"{phase}_split.json", split)
            _atomic_json(artifact_dir / f"{phase}_encoder.json", _encoder_receipt(encoder))
            blind, proposal, candidate, fit_receipts = refit_confirmatory_ensemble(
                training,
                holdout,
                config=config,
                width=selected_recipe["width"],
                selected_epoch=selected_recipe["epoch"],
                selected_threshold=selected_recipe["threshold"],
                phase=phase,
                device=device,
                artifact_dir=artifact_dir,
            )
            fold = config["phase_protocols"][phase]["fold"]
            receipt = commit_confirmatory_blind(
                blind,
                proposal,
                candidate,
                phase=phase,
                fold=fold,
                key_sha256=surfaces.membership_sha256[fold],
                config_sha256=preflight["config_sha256"],
                selected_recipe=selected_recipe,
                fit_receipts=fit_receipts,
                artifact_dir=artifact_dir,
            )
            confirm_receipts[phase] = receipt
            confirm_encoded[phase] = holdout

        # Both confirmatory files must exist and verify before either metric
        # reader is invoked.  Q3 labels may have become historical training
        # input for the Q4 refit, but no Q3 score/gate was computed or observed.
        verify_blind_receipt(confirm_receipts["q3"])
        verify_blind_receipt(confirm_receipts["q4"])
        sealed_confirmatory_candidates: dict[str, Any] = {}
        confirmatory_semantic_replays: dict[str, Any] = {}
        for phase in ("q3", "q4"):
            candidate, replay = load_sealed_confirmatory_candidate(
                confirm_receipts[phase],
                holdout=confirm_encoded[phase],
                config=config,
                selected_recipe=selected_recipe,
            )
            sealed_confirmatory_candidates[phase] = candidate
            confirmatory_semantic_replays[phase] = replay
        _atomic_json(
            artifact_dir / "confirmatory_blind_semantic_replays.json",
            confirmatory_semantic_replays,
        )
        truths = {
            phase: load_fold_truth_after_receipts(
                config,
                confirm_encoded[phase].surface,
                [confirm_receipts[phase]],
                fold=config["phase_protocols"][phase]["fold"],
                root=root,
            )
            for phase in ("q3", "q4")
        }
        confirm_metrics = evaluate_confirmatory_folds(
            truths,
            confirm_encoded,
            sealed_confirmatory_candidates,
            config=config,
        )
        _atomic_json(artifact_dir / "confirmatory_metrics.json", confirm_metrics)
        if confirm_metrics["high_impact_official_probe_result"] == "PASS":
            status = "GO_HIGH_IMPACT_OFFICIAL_PROBE_ELIGIBLE_NOT_AUTHORIZED"
        elif confirm_metrics["research_result"] == "PASS":
            status = "GO_RESEARCH_ONLY_NOT_OFFICIAL_PROBE_ELIGIBLE"
        else:
            status = "NO_GO_CONFIRMATORY"
        result = {
            "schema_version": "p1.mstcn_asrf.terminal.v2",
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "device": torch.cuda.get_device_name(device),
            "q2_role": "selection_only_not_promotion_evidence",
            "selected_recipe": selected_recipe,
            "confirmatory_metrics": confirm_metrics,
            "submission_created": False,
            "upload_performed": False,
            "official_three_point_gain_claimed": False,
        }
        _atomic_json(terminal_path, result)
        return result
    except BaseException as error:
        if not terminal_path.exists():
            _atomic_json(
                terminal_path,
                {
                    "schema_version": "p1.mstcn_asrf.terminal.v2",
                    "experiment_id": EXPERIMENT_ID,
                    "status": "FAILED_EXECUTION_NO_RETRY_AUTHORIZED",
                    "started_at_utc": started.isoformat(),
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "result_based_retry_authorized": False,
                },
            )
        raise


def run_smoke() -> dict[str, Any]:
    """Two-epoch synthetic CPU/GPU pipeline; no immutable input or lock is touched."""

    np, pd, torch, _model_api, _data_api = _load_scientific()
    config = json.loads(json.dumps(_canonical_config()))
    config["windowing"]["rows"] = 64
    config["windowing"]["stride_rows"] = 32
    config["architecture"]["width"] = 8
    config["architecture"]["dropout"] = 0.0
    config["training"]["maximum_epochs"] = 2
    config["training"]["batch_size"] = 8
    config["training"]["gradient_accumulation_steps"] = 1
    config["training"]["warmup_epochs"] = 1
    config["training"]["learning_rate"] = 0.003
    config["implementation_sanity_gate"]["maximum_epochs"] = 2
    seed = int(config["training"]["seed"])
    rng = np.random.default_rng(seed)

    def synthetic(rows: int, *, start: str, events: Sequence[tuple[int, int]]) -> RowSurface:
        time_values = pd.date_range(start, periods=rows, freq="10min", tz="Asia/Seoul")
        keys = pd.DataFrame(
            {
                "station": ["SYN"] * rows,
                "year": time_values.year.astype(np.int16),
                "layer": np.ones(rows, dtype=np.int8),
                "time": time_values.astype(str),
            }
        )
        numeric = rng.normal(0.0, 0.15, size=(rows, 6)).astype(np.float32)
        labels = np.zeros(rows, dtype=np.int8)
        anomaly = np.full(rows, "", dtype=object)
        for event_index, (left, right) in enumerate(events):
            labels[left:right] = 1
            anomaly[left:right] = "offset" if event_index % 2 == 0 else "drift"
            numeric[left:right, 0] += 3.0
        return RowSurface(
            keys,
            numeric,
            np.asarray(["SYN"] * rows),
            np.asarray(["layer_1"] * rows),
            np.asarray(["mid"] * rows),
            labels,
            anomaly,
            np.zeros(rows, dtype=np.int8),
            np.full(rows, 10.0, dtype=np.float32),
        )

    training = synthetic(384, start="2025-01-01", events=((48, 80), (176, 220), (300, 330)))
    q2_surface = synthetic(192, start="2025-04-01", events=((50, 82), (130, 166)))
    encoder, encoded = _fit_encoder_and_transform(
        training,
        [q2_surface],
        fit_ids=np.arange(training.rows, dtype=np.int64),
        forbidden_ids=np.asarray([], dtype=np.int64),
        numeric_names=tuple(f"feature_{index}" for index in range(training.numeric.shape[1])),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blind, fit_receipt = _smoke_refit_and_predict(
        encoded[0],
        encoded[1],
        config=config,
        selected_epoch=2,
        device=device,
        artifact_dir=None,
    )
    proposal = decode_long_event_segments(
        _decoder_row_probability(blind, config),
        blind.boundary_probability,
        encoded[1].layout,
        high_threshold=0.5,
        minimum_rows=5,
        maximum_rows=100,
    )
    candidate = anchor_preserving_union(q2_surface.anchor, proposal)
    metrics = binary_metrics(q2_surface.labels, candidate)
    if not (
        np.isfinite(blind.row_probability).all()
        and np.isfinite(blind.boundary_probability).all()
        and fit_receipt["epochs"] == 2
    ):
        raise ContractError("synthetic smoke output is incomplete")
    return {
        "schema_version": "p1.mstcn_asrf.smoke.v1",
        "experiment_id": EXPERIMENT_ID,
        "result": "PASS",
        "device": str(device),
        "epochs": 2,
        "rows": {"training": training.rows, "blind": q2_surface.rows},
        "input_features": int(encoded[0].features.shape[1]),
        "encoder_fit_ids_sha256": encoder.fit_ids_sha256,
        "metrics": metrics,
        "real_input_reads": 0,
        "attempt_lock_created": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute-protocol", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve(strict=True)
    if args.check_only:
        result = check_only(root=root)
    elif args.smoke:
        if root != ROOT.resolve():
            raise ContractError("synthetic smoke uses the canonical code/config root")
        result = run_smoke()
    else:
        raise ContractError(
            f"direct scientific execution is disabled; use scripts/{EXTERNAL_LAUNCHER_NAME}"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
