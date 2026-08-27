"""Sealed zero-fit runner for the preregistered P1 Cycle-1 screen.

The checked-in state is intentionally NOT_AUTHORIZED.  ``--seal`` and
``--preflight`` are read-only with respect to scientific data and may only
publish aggregate integrity receipts.  ``--execute`` rejects before a claim,
feature/proposal materialization, numerical import, or model fit until an
independent authorization file is created and its exact digest is embedded in
this runner.  The design JSON is never rewritten by this runner.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v1_design.json"
)
TRIGGER_RESOLUTION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v1_trigger_resolution.json"
)
OPERATIONAL_AMENDMENT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v2_operational_amendment.json"
)
MODULE_PATH = PROJECT_ROOT / "src" / "p1_qc" / "long_event_segment_proposal_rescore.py"
CHANGE_POINTS_PATH = PROJECT_ROOT / "src" / "p1_qc" / "change_points.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_run_p1_long_event_segment_proposal_rescore_v1.py"
AUTHORIZATION_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v1_execution_authorization_template.json"
)
AUTHORIZATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v1_execution_authorization.json"
)
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "p1_long_event_segment_proposal_rescore_20260826_v1"
SEAL_PATH = ARTIFACT_DIR / "preexecution_seal.json"
PREFLIGHT_PATH = ARTIFACT_DIR / "strict_preflight_receipt.json"
SUPERSEDED_INFRA_RUNNER_PATH = (
    PROJECT_ROOT / "scripts" / "run_p1_round_b_nonspike_long_event_residual_v1r6.py"
)
V1R6_STRICT_PREFLIGHT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "p1_round_b_nonspike_long_event_residual_v1r6"
    / "strict_preflight_receipt.json"
)

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
DESIGN_SHA256 = "31b0bde27d8ef7e2b42135709563cca0bcca61c6ec6fdabefbb3530906869563"
TRIGGER_RESOLUTION_SHA256 = "f70a17ff6a30d59522a007b9605da12b23f8b5fb0c3247e45c9b30772c1cc2f1"
OPERATIONAL_AMENDMENT_SHA256 = "b33f7d386e05cd7ab79976f58e9f4ab752f37cfe6a8856849867ef5f541cb276"
AUTHORIZATION_ENV_VAR = "P1_LONG_EVENT_SEGMENT_EXECUTION_AUTHORIZATION_SHA256"
AUTHORIZATION_SHA256 = "<NOT_AUTHORIZED_PENDING_INDEPENDENT_QA>"
MAXIMUM_LIFETIME_PHYSICAL_FITS = 72
MAXIMUM_FEATURE_OR_PROPOSAL_MATERIALIZATIONS = 21
HARD_WALL_SECONDS = 21600
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
SEEDS = (20260826, 20260843, 20260871)
ROUND_B_SEEDS = (20260813, 20260829, 20260847)
INNER_WINDOW_IDS = (
    "inner_2024_jul_aug",
    "inner_2024_oct_nov",
    "inner_2025_jan_feb",
)
CONTEXT_BANK_IDS = ("24_72", "48_168", "24_72_168")
DECODER_IDS = ("connected_only", "dual_boundary_disconnected_allowed")
STRUCTURE_CELL_IDS = tuple(
    f"bank_{bank}__{decoder}" for bank in CONTEXT_BANK_IDS for decoder in DECODER_IDS
)
NUMERICAL_LINEAGE_SECTIONS = (
    "base_config",
    "surface",
    "residual_target",
    "residual_model",
    "rescue_decoder",
    "outer_protocol",
    "fail_fast_gates",
    "resource_budget",
)
REQUIRED_PREDECESSOR_GATE_NAMES = (
    "adequately_supported_worst_cell_f1_delta",
    "all_seed_f1_deltas",
    "equal_weight_station_fold_f1_delta",
    "new_disconnected_events",
    "new_singletons",
    "nonnegative_fold_count",
    "nonnegative_station_count",
    "normal_fp_per_day_ratio",
    "paired_bootstrap_ci90_lower",
    "pooled_micro_f1_delta",
    "precision_delta",
    "recall_delta",
    "spike_recall_delta",
)

_AUTH_PATTERN = re.compile(rb'(?m)^AUTHORIZATION_SHA256 = "[^"]+"$')
_SENSITIVE_BASENAMES = {
    "test.csv",
    "sample_submission.csv",
    "submission.csv",
}


class AuthorizationError(RuntimeError):
    """Raised before any numerical work when execution is not authorized."""


class ProcessTreeTerminationError(RuntimeError):
    """Raised when a timed-out worker tree cannot be proved terminated."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


@dataclass(frozen=True)
class HeldValidationSurface:
    """Verified in-memory outer truth and anchor keys; no scoring path reopen."""

    key_tuples: tuple[tuple[str, int, int, str], ...]
    truth: tuple[int, ...]
    fold: tuple[str, ...]
    anchor_probability: tuple[float, ...]
    anchor_prediction: tuple[int, ...]
    plateau: tuple[bool, ...]
    spike_candidate: tuple[bool, ...]
    input_receipts: Mapping[str, Any]


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bound_bytes(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Read one held handle once and bind the returned bytes to its digest."""

    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        value = handle.read()
        after = os.fstat(handle.fileno())
    before_signature = (before.st_size, before.st_mtime_ns, before.st_ino)
    after_signature = (after.st_size, after.st_mtime_ns, after.st_ino)
    if before_signature != after_signature or len(value) != before.st_size:
        raise RuntimeError(f"held file changed while being read: {path.name}")
    observed = _sha256_bytes(value)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"held file digest mismatch: {path.name}")
    return value, {"bytes": len(value), "actual_read_sha256": observed}


def _json_from_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {label}")
    return value


def _read_bound_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, receipt = _read_bound_bytes(path, expected_sha256=expected_sha256)
    return _json_from_bytes(raw, label=path.name), receipt


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return f"{text}\n".encode()


def _normalised_runner_bytes(path: Path | None = None) -> bytes:
    raw = (path or Path(__file__).resolve()).read_bytes()
    matches = list(_AUTH_PATTERN.finditer(raw))
    if len(matches) != 1:
        raise RuntimeError("runner authorization anchor must occur exactly once")
    replacement = b'AUTHORIZATION_SHA256 = "<EXTERNAL_AUTHORIZATION_SHA256>"'
    return raw[: matches[0].start()] + replacement + raw[matches[0].end() :]


def _normalised_runner_sha256(path: Path | None = None) -> str:
    return _sha256_bytes(_normalised_runner_bytes(path))


def _fsync_directory(path: Path) -> None:
    """Durably flush a directory where the platform exposes that operation."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        flush = kernel32.FlushFileBuffers
        flush.argtypes = [wintypes.HANDLE]
        flush.restype = wintypes.BOOL
        close = kernel32.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        handle = create_file(
            str(path.resolve(strict=True)),
            0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise OSError(ctypes.get_last_error(), "directory open for flush failed")
        try:
            if not flush(handle):
                raise OSError(ctypes.get_last_error(), "directory flush failed")
        finally:
            close(handle)
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_bytes(path: Path, payload: bytes) -> Path:
    """Create-only publication: a pre-existing target is never overwritten."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    linked = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        linked = True
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            if linked:
                _fsync_directory(path.parent)
    return path


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _atomic_create_bytes(path, _json_bytes(value))


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _sanitize_error(error: BaseException, sensitive_roots: Sequence[Path] = ()) -> dict[str, str]:
    """Return full, path-sanitized error provenance without silent truncation."""

    raw = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    roots = {
        PROJECT_ROOT,
        Path.home(),
        *sensitive_roots,
    }
    sanitized = raw
    for root in sorted(roots, key=lambda value: len(str(value)), reverse=True):
        text = str(root)
        if text:
            sanitized = sanitized.replace(text, f"<REDACTED:{root.name or 'ROOT'}>")
    sanitized = "".join(
        character if character in "\n\r\t" or ord(character) >= 32 else "?"
        for character in sanitized
    )
    return {
        "type": type(error).__name__,
        "message": str(error).replace(str(PROJECT_ROOT), "<PROJECT_ROOT>"),
        "sanitized_traceback": sanitized,
        "sanitized_traceback_sha256": _sha256_bytes(sanitized.encode("utf-8")),
    }


def _reject_sensitive_path(path: Path) -> None:
    lowered = path.name.lower()
    if lowered in _SENSITIVE_BASENAMES or (
        "submission" in lowered and path.suffix.lower() in {".csv", ".parquet"}
    ):
        raise RuntimeError("official/sample/submission path is prohibited")


def _verify_hash_map(hash_map: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(hash_map.items()):
        path = PROJECT_ROOT / relative
        _reject_sensitive_path(path)
        raw, receipt = _read_bound_bytes(path, expected_sha256=str(expected))
        del raw
        receipts[relative] = receipt
    return receipts


def _runtime_versions() -> dict[str, str]:
    package_names = {
        "joblib": "joblib",
        "lightgbm": "lightgbm",
        "narwhals": "narwhals",
        "numpy": "numpy",
        "p1-qc": "p1-qc",
        "pandas": "pandas",
        "psutil": "psutil",
        "pyarrow": "pyarrow",
        "python-dateutil": "python-dateutil",
        "scikit-learn": "scikit-learn",
        "scipy": "scipy",
        "six": "six",
        "threadpoolctl": "threadpoolctl",
    }
    versions = {"python": platform.python_version()}
    for output_name, distribution_name in package_names.items():
        versions[output_name] = importlib.metadata.version(distribution_name)
    return versions


def _verify_runtime(template: Mapping[str, Any]) -> dict[str, Any]:
    expected = template["runtime_versions"]
    observed = _runtime_versions()
    if observed != expected:
        raise RuntimeError("runtime version map differs from authorization template")
    native = template["lightgbm_native"]
    distribution = importlib.metadata.distribution("lightgbm")
    native_path = Path(distribution.locate_file(native["distribution_relative_path"]))
    raw, receipt = _read_bound_bytes(native_path, expected_sha256=native["sha256"])
    if len(raw) != int(native["bytes"]):
        raise RuntimeError("native LightGBM byte count differs")
    return {"versions": observed, "lightgbm_native": receipt}


def _reconstruct_frozen_v1r4(
    current_bytes: bytes,
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    proof = resolution["design_predecessor_reconstruction"]
    substitution = proof["single_permitted_reconstruction"]
    current = substitution["current_value"].encode("ascii")
    placeholder = substitution["reconstructed_value"].encode("ascii")
    if current_bytes.count(current) != substitution["required_current_value_occurrences"]:
        raise RuntimeError("v1r4 seal-anchor occurrence count changed")
    reconstructed = current_bytes.replace(current, placeholder)
    if len(reconstructed) != int(substitution["reconstructed_bytes"]):
        raise RuntimeError("v1r4 reconstructed byte count differs")
    digest = _sha256_bytes(reconstructed)
    if digest != substitution["reconstructed_sha256"]:
        raise RuntimeError("v1r4 placeholder reconstruction differs")
    if digest != proof["design_expected_sha256"]:
        raise RuntimeError("v1r4 reconstruction does not match frozen design")
    return {
        "current_value_occurrences": current_bytes.count(current),
        "reconstructed_bytes": len(reconstructed),
        "reconstructed_sha256": digest,
        "only_changed_json_pointer": substitution["json_pointer"],
    }


def _verify_trigger_resolution(
    design: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    if resolution.get("status") != (
        "TRIGGER_RESOLVED_INFRASTRUCTURE_ONLY_SUCCESSOR_SCIENTIFIC_NO_GO"
    ):
        raise RuntimeError("trigger resolution is not terminal scientific NO_GO")
    if resolution["frozen_design"]["sha256"] != DESIGN_SHA256:
        raise RuntimeError("trigger resolution references another design")
    versions = resolution["infrastructure_only_lineage"]["versions"]
    parsed_configs: list[dict[str, Any]] = []
    config_raws: list[bytes] = []
    config_receipts: dict[str, Any] = {}
    for version in versions:
        path = PROJECT_ROOT / version["config_path"]
        raw, receipt = _read_bound_bytes(path, expected_sha256=version["config_sha256"])
        config_raws.append(raw)
        parsed_configs.append(_json_from_bytes(raw, label=path.name))
        config_receipts[version["experiment_id"]] = receipt
        receipt_path = PROJECT_ROOT / version["supersession_receipt_path"]
        _read_bound_bytes(
            receipt_path,
            expected_sha256=version["supersession_receipt_sha256"],
        )
    reconstruction = _reconstruct_frozen_v1r4(
        config_raws[0],
        resolution,
    )
    for section in NUMERICAL_LINEAGE_SECTIONS:
        reference = parsed_configs[0][section]
        if any(config[section] != reference for config in parsed_configs[1:]):
            raise RuntimeError(f"infrastructure successor changed {section}")
    terminal = resolution["terminal_scientific_successor"]
    result, result_receipt = _read_bound_json(
        PROJECT_ROOT / terminal["result"]["path"],
        expected_sha256=terminal["result"]["sha256"],
    )
    manifest, manifest_receipt = _read_bound_json(
        PROJECT_ROOT / terminal["manifest"]["path"],
        expected_sha256=terminal["manifest"]["sha256"],
    )
    metrics, metrics_receipt = _read_bound_json(
        PROJECT_ROOT / terminal["metrics"]["path"],
        expected_sha256=terminal["metrics"]["sha256"],
    )
    if result.get("status") != terminal["result"]["status"]:
        raise RuntimeError("successor result is not terminal")
    if result.get("decision") != "NO_GO_LOCAL_GATE":
        raise RuntimeError("successor scientific decision changed")
    checks = metrics.get("gate_checks")
    if not isinstance(checks, Mapping) or set(checks) != set(REQUIRED_PREDECESSOR_GATE_NAMES):
        raise RuntimeError("successor original gate set is incomplete")
    if result.get("passed_all_gates") is not False or metrics.get("passed_all_gates") is not False:
        raise RuntimeError("successor aggregate NO_GO evidence changed")
    if all(bool(value) for value in checks.values()):
        raise RuntimeError("successor gate booleans contradict NO_GO")
    if manifest.get("experiment_id") != terminal["experiment_id"]:
        raise RuntimeError("successor manifest identity changed")
    if resolution.get("resolved_anchor_branch") != "FROZEN_ROUND_B":
        raise RuntimeError("trigger resolution selected a non-preregistered branch")
    frozen = design["fixed_structure_search"]
    if (
        int(frozen["inner_physical_fit_calls"]) != 54
        or int(frozen["outer_locked_physical_fit_calls"]) != 9
        or int(frozen["maximum_lifetime_physical_fit_calls"]) != 63
    ):
        raise RuntimeError("frozen 54+9=63 fit accounting changed")
    return {
        "anchor_branch": "FROZEN_ROUND_B",
        "v1r4_reconstruction": reconstruction,
        "numerical_sections_equal": list(NUMERICAL_LINEAGE_SECTIONS),
        "successor_result": result_receipt,
        "successor_manifest": manifest_receipt,
        "successor_metrics": metrics_receipt,
        "gate_count": len(checks),
        "successor_pooled_f1_delta": metrics["pooled"]["f1_delta"],
        "successor_rescued_rows": metrics["structural"]["rescued_rows"],
        "config_receipts": config_receipts,
    }


def _load_module_from_verified_path(
    path: Path,
    *,
    module_name: str,
) -> ModuleType:
    source_root = str(PROJECT_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    if module_name.startswith("p1_qc."):
        importlib.import_module("p1_qc")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verified module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _verify_top_level_stdlib_only(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    allowed = {
        "__future__",
        "argparse",
        "ast",
        "collections",
        "dataclasses",
        "datetime",
        "hashlib",
        "importlib",
        "io",
        "json",
        "os",
        "pathlib",
        "platform",
        "re",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "traceback",
        "types",
        "typing",
        "uuid",
        "zoneinfo",
    }
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "").split(".")[0])
    unexpected = sorted(set(imports).difference(allowed))
    if unexpected:
        raise RuntimeError("pre-import runner imported non-stdlib modules")
    return imports


def _read_template() -> tuple[dict[str, Any], dict[str, Any]]:
    template, receipt = _read_bound_json(AUTHORIZATION_TEMPLATE_PATH)
    if template.get("status") != "NOT_AUTHORIZED_PENDING_INDEPENDENT_QA":
        raise RuntimeError("authorization template status changed")
    if template.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("authorization template identity changed")
    return template, receipt


def _static_package_checks() -> dict[str, Any]:
    design, design_receipt = _read_bound_json(
        DESIGN_PATH,
        expected_sha256=DESIGN_SHA256,
    )
    resolution, resolution_receipt = _read_bound_json(
        TRIGGER_RESOLUTION_PATH,
        expected_sha256=TRIGGER_RESOLUTION_SHA256,
    )
    amendment, amendment_receipt = _read_bound_json(
        OPERATIONAL_AMENDMENT_PATH,
        expected_sha256=OPERATIONAL_AMENDMENT_SHA256,
    )
    template, template_receipt = _read_template()
    if template["runner_normalized_sha256"] != _normalised_runner_sha256():
        raise RuntimeError("normalized runner differs from authorization template")
    closure = _verify_hash_map(template["dependency_sha256"])
    runtime = _verify_runtime(template)
    trigger = _verify_trigger_resolution(design, resolution)
    imports = _verify_top_level_stdlib_only(Path(__file__).resolve())
    module = _load_module_from_verified_path(
        MODULE_PATH,
        module_name=f"p1_qc.long_event_segment_proposal_rescore_preflight_{uuid.uuid4().hex}",
    )
    module.assert_design_contract(design)
    module.assert_operational_amendment(amendment)
    contract = module.implementation_contract()
    if contract["fit_budget"] != {
        "inner_seed_cells_per_window": 18,
        "inner_anchor_physical_fits": 9,
        "inner_segment_physical_fits": 54,
        "outer_segment_physical_fits": 9,
        "segment_physical_fits": 63,
        "maximum_lifetime_physical_fits": 72,
        "maximum_feature_or_proposal_materializations": 21,
    }:
        raise RuntimeError("module fit accounting differs from operational amendment")
    return {
        "design": design_receipt,
        "trigger_resolution": resolution_receipt,
        "operational_amendment": amendment_receipt,
        "authorization_template": template_receipt,
        "dependency_closure": closure,
        "runtime": runtime,
        "trigger": trigger,
        "runner_top_level_imports": imports,
        "implementation_contract_sha256": contract["contract_sha256"],
        "runner_normalized_sha256": _normalised_runner_sha256(),
    }


def _p1_source_train_path() -> Path:
    value = os.environ.get("P1_DATA_DIR")
    if not value:
        raise RuntimeError("P1_DATA_DIR is required for strict data readiness")
    directory = Path(value).resolve(strict=True)
    path = (directory / "train.csv").resolve(strict=True)
    if path.parent != directory or path.name.lower() != "train.csv":
        raise RuntimeError("P1 source path is not the exact historical train file")
    return path


def _verify_existing_v1r6_readiness(template: Mapping[str, Any]) -> dict[str, Any]:
    expected = template["immutable_input_sha256"]
    source_path = _p1_source_train_path()
    source_raw, source_receipt = _read_bound_bytes(
        source_path,
        expected_sha256=expected["P1_DATA_DIR/train.csv"],
    )
    del source_raw
    current = _verify_hash_map(
        {key: value for key, value in expected.items() if key != "P1_DATA_DIR/train.csv"}
    )
    receipt, receipt_read = _read_bound_json(
        V1R6_STRICT_PREFLIGHT_PATH,
        expected_sha256=expected[_relative(V1R6_STRICT_PREFLIGHT_PATH)],
    )
    readiness = receipt.get("strict_readiness", {})
    if receipt.get("status") != "PASS_STRICT_COMPLETE_READINESS_NO_CLAIM_NO_FIT":
        raise RuntimeError("v1r6 readiness receipt status changed")
    if readiness.get("status") != "PASS_COMPLETE_READINESS_BEFORE_CLAIM":
        raise RuntimeError("v1r6 inner readiness status changed")
    full = readiness.get("full_feature_cache_binding", {})
    if (
        full.get("status") != "PASS_REGENERATED_ALL_MODEL_INPUT_FEATURES_EXACT_ORDERED"
        or full.get("actual_cache_order_and_values_exact") is not True
        or full.get("synthetic_swap_probe", {}).get("full_feature_mismatch_detected") is not True
    ):
        raise RuntimeError("v1r6 full feature binding proof changed")
    if readiness.get("left_censored_positive_connected_event_count_by_fold") != {
        "2025_q2": 0,
        "2025_q3": 0,
        "2025_q4": 0,
    }:
        raise RuntimeError("left-censor proof changed")
    exact = readiness.get("exact_round_b_equivalence", {})
    if exact.get("truth_key_alignment") != "one_to_one":
        raise RuntimeError("Round-B truth/key proof changed")
    return {
        "source_train": source_receipt,
        "immutable_inputs": current,
        "v1r6_strict_preflight": receipt_read,
        "full_feature_cache_binding": full,
        "left_censor_counts": readiness["left_censored_positive_connected_event_count_by_fold"],
        "exact_round_b_equivalence": exact,
    }


def _table_from_held_parquet(raw: bytes, columns: Sequence[str]) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pq.read_table(pa.BufferReader(raw), columns=list(columns))


def _held_outer_truth_and_keys(template: Mapping[str, Any]) -> HeldValidationSurface:
    """Verify and parse outer truth/anchor bytes once, before any scorer call."""

    import pandas as pd

    paths = template["outer_surface_inputs"]
    oof_path = PROJECT_ROOT / paths["truth_oof"]["path"]
    oof_raw, oof_receipt = _read_bound_bytes(
        oof_path,
        expected_sha256=paths["truth_oof"]["sha256"],
    )
    oof = _table_from_held_parquet(
        oof_raw,
        ["station", "year", "layer", "time", "label", "fold"],
    ).to_pandas()
    del oof_raw
    part_frames = []
    receipts: dict[str, Any] = {"truth_oof": oof_receipt}
    for item in paths["round_b_parts"]:
        path = PROJECT_ROOT / item["path"]
        raw, receipt = _read_bound_bytes(path, expected_sha256=item["sha256"])
        frame = _table_from_held_parquet(
            raw,
            [
                "station",
                "year",
                "layer",
                "time",
                "fold",
                "event_day_balanced_binary_lgbm__probability",
                "event_day_balanced_binary_lgbm__prediction",
                "plateau",
                "spike_candidate",
            ],
        ).to_pandas()
        del raw
        if set(frame["fold"].astype(str)) != {item["fold"]}:
            raise RuntimeError("Round-B part fold identity changed")
        part_frames.append(frame)
        receipts[item["fold"]] = receipt
    anchor = pd.concat(part_frames, ignore_index=True)
    key_columns = ["station", "year", "layer", "time", "fold"]
    if len(oof) != 421032 or len(anchor) != 421032:
        raise RuntimeError("outer surface row count changed")
    left = oof[key_columns].astype({"station": "string", "time": "string", "fold": "string"})
    right = anchor[key_columns].astype({"station": "string", "time": "string", "fold": "string"})
    if not left.equals(right):
        raise RuntimeError("held Round-B and truth ordered keys differ")
    key_tuples = tuple(
        (str(row.station), int(row.year), int(row.layer), str(row.time))
        for row in left.itertuples(index=False)
    )
    truth = tuple(int(value) for value in oof["label"])
    prediction = tuple(int(value) for value in anchor["event_day_balanced_binary_lgbm__prediction"])
    if set(truth).difference({0, 1}) or set(prediction).difference({0, 1}):
        raise RuntimeError("held truth or anchor prediction is non-binary")
    return HeldValidationSurface(
        key_tuples=key_tuples,
        truth=truth,
        fold=tuple(str(value) for value in anchor["fold"]),
        anchor_probability=tuple(
            float(value) for value in anchor["event_day_balanced_binary_lgbm__probability"]
        ),
        anchor_prediction=prediction,
        plateau=tuple(bool(value) for value in anchor["plateau"]),
        spike_candidate=tuple(bool(value) for value in anchor["spike_candidate"]),
        input_receipts=receipts,
    )


def score_from_held_truth(
    truth: Sequence[int],
    prediction: Sequence[int],
) -> dict[str, float | int]:
    """Pure scorer receiving verified truth directly; it never opens a path."""

    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    tp = fp = fn = tn = 0
    for actual, predicted in zip(truth, prediction, strict=True):
        if actual not in (0, 1) or predicted not in (0, 1):
            raise ValueError("truth and prediction must be binary")
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": len(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _strict_data_preflight(template: Mapping[str, Any]) -> dict[str, Any]:
    amendment, amendment_receipt = _read_bound_json(
        OPERATIONAL_AMENDMENT_PATH,
        expected_sha256=OPERATIONAL_AMENDMENT_SHA256,
    )
    readiness = _verify_existing_v1r6_readiness(template)
    held = _held_outer_truth_and_keys(template)
    baseline = score_from_held_truth(held.truth, held.anchor_prediction)
    expected = readiness["exact_round_b_equivalence"]["baseline_metrics"]
    for name in ("f1", "precision", "recall"):
        if float(baseline[name]) != float(expected[name]):
            raise RuntimeError("held Round-B baseline metric differs")
    for name in ("tp", "fp", "fn", "tn"):
        if int(baseline[name]) != int(expected[name]):
            raise RuntimeError("held Round-B baseline count differs")
    plan = template["scientific_input_binding"]["inner_anchor_construction"]
    if plan.get("status") != "PROSPECTIVELY_PINNED_ZERO_FIT":
        raise RuntimeError("inner anchor construction status changed")
    if plan.get("operational_amendment_sha256") != OPERATIONAL_AMENDMENT_SHA256:
        raise RuntimeError("inner anchor construction pins another amendment")
    windows = amendment["inner_anchor_fits"]["windows"]
    if len(windows) != 3 or any(tuple(item["fit_seeds"]) != ROUND_B_SEEDS for item in windows):
        raise RuntimeError("inner anchor seed plan changed")
    if amendment["round_b_anchor_lineage"]["postprocess"] != {
        "source": amendment["round_b_anchor_lineage"]["postprocess"]["source"],
        "high_threshold": 0.2,
        "low_threshold": 0.1,
        "close_gap_rows": 0,
        "minimum_positive_run": 12,
        "plateau_detector": "p1_qc.rules.detect_plateaus",
        "singleton_spike_detector": "p1_qc.rules.detect_singleton_spikes",
        "implementation": "p1_qc.pipeline.apply_postprocess",
    }:
        raise RuntimeError("inner anchor deployment postprocess changed")
    return {
        "v1r6_reused_readiness": readiness,
        "held_outer_surface": {
            "rows": len(held.truth),
            "ordered_key_sha256": _sha256_bytes(
                json.dumps(held.key_tuples, separators=(",", ":")).encode("utf-8")
            ),
            "input_receipts": dict(held.input_receipts),
            "truth_passed_directly_to_scorer": True,
            "truth_path_reopens_by_scorer": 0,
            "round_b_baseline": baseline,
        },
        "inner_anchor_construction": {
            **plan,
            "operational_amendment": amendment_receipt,
            "windows": [item["id"] for item in windows],
            "registered_seeds_unchanged_per_window": list(ROUND_B_SEEDS),
            "base_physical_fit_calls": 9,
            "postprocess_uniform_all_inner_windows": True,
            "fits_observed": 0,
            "materializations_observed": 0,
        },
        "scientific_input_ready_before_claim": True,
        "execution_ready": False,
        "execution_blocker": "NO_INDEPENDENT_EXECUTION_AUTHORIZATION",
    }


def _require_execution_authorization_preimport() -> dict[str, Any]:
    """Authenticate an external authorization before config/numerical import."""

    if not re.fullmatch(r"[0-9a-f]{64}", AUTHORIZATION_SHA256):
        raise AuthorizationError("runner is sealed NOT_AUTHORIZED pending independent QA")
    supplied = os.environ.get(AUTHORIZATION_ENV_VAR)
    if supplied != AUTHORIZATION_SHA256:
        raise AuthorizationError("external execution authorization digest is missing or differs")
    authorization, _receipt = _read_bound_json(
        AUTHORIZATION_PATH,
        expected_sha256=AUTHORIZATION_SHA256,
    )
    if authorization.get("status") != "AUTHORIZED_INDEPENDENT_QA_PASS":
        raise AuthorizationError("external authorization status is not executable")
    if authorization.get("experiment_id") != EXPERIMENT_ID:
        raise AuthorizationError("external authorization identity differs")
    if authorization.get("runner_normalized_sha256") != _normalised_runner_sha256():
        raise AuthorizationError("external authorization pins another runner")
    _verify_hash_map(authorization["dependency_sha256"])
    _verify_runtime(authorization)
    plan = authorization["scientific_input_binding"]["inner_anchor_construction"]
    if (
        plan.get("status") != "AUTHORIZED_PROSPECTIVE_CONSTRUCTION"
        or plan.get("operational_amendment_sha256") != OPERATIONAL_AMENDMENT_SHA256
    ):
        raise AuthorizationError("prospective inner anchor construction is not authorized")
    return authorization


def _load_supervisor_infrastructure(template: Mapping[str, Any]) -> ModuleType:
    expected = template["dependency_sha256"][_relative(SUPERSEDED_INFRA_RUNNER_PATH)]
    _read_bound_bytes(SUPERSEDED_INFRA_RUNNER_PATH, expected_sha256=expected)
    return _load_module_from_verified_path(
        SUPERSEDED_INFRA_RUNNER_PATH,
        module_name=f"p1_v1r6_supervisor_{uuid.uuid4().hex}",
    )


class AttemptJournal:
    """Create-only 9-anchor + 63-segment fit / 21-materialization ledger."""

    def __init__(self, artifact: Path, deadline_epoch: float) -> None:
        self.artifact = artifact
        self.deadline_epoch = deadline_epoch
        self.lock_path = artifact / "execution.lock"
        self.journal_dir = artifact / "attempt_journal"
        self.attempt_id = uuid.uuid4().hex
        self.fit_reservations = 0
        self.fits_completed = 0
        self.materializations = 0
        self._last_sha256: str | None = None
        self._entries: dict[str, str] = {}
        self._lock_descriptor: int | None = None
        self._selected_outer_cell: str | None = None

    @classmethod
    def begin(cls, artifact: Path, deadline_epoch: float) -> AttemptJournal:
        if time.time() >= deadline_epoch:
            raise TimeoutError("deadline expired before execution claim")
        artifact.mkdir(parents=True, exist_ok=True)
        journal = cls(artifact, deadline_epoch)
        if journal.journal_dir.exists():
            raise FileExistsError("lifetime attempt journal already exists")
        journal._lock_descriptor = os.open(
            journal.lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
        phase = "LOCK_WRITE"
        try:
            payload = _json_bytes(
                {
                    "attempt_id": journal.attempt_id,
                    "deadline_epoch": deadline_epoch,
                    "created_at_kst": _now_kst(),
                }
            )
            written = os.write(journal._lock_descriptor, payload)
            if written != len(payload):
                raise OSError("short execution lock write")
            phase = "LOCK_FSYNC"
            os.fsync(journal._lock_descriptor)
            phase = "LOCK_DIR_FSYNC"
            _fsync_directory(artifact)
            phase = "JOURNAL_CREATE"
            os.mkdir(journal.journal_dir)
            _fsync_directory(artifact)
            journal._entry(
                "000_started.json",
                {
                    "attempt_id": journal.attempt_id,
                    "inner_anchor_physical_fits": 9,
                    "inner_segment_physical_fits": 54,
                    "outer_segment_physical_fits": 9,
                    "maximum_lifetime_physical_fits": 72,
                    "maximum_materializations": 21,
                    "created_at_kst": _now_kst(),
                },
            )
            return journal
        except BaseException as error:
            terminal = artifact / "initialization_failed.json"
            try:
                _atomic_create_json(
                    terminal,
                    {
                        "attempt_id": journal.attempt_id,
                        "phase": phase,
                        "error": _sanitize_error(error),
                        "execution_lock_retained": True,
                        "fit_reservations": 0,
                        "materializations": 0,
                        "created_at_kst": _now_kst(),
                    },
                )
            finally:
                if journal._lock_descriptor is not None:
                    os.close(journal._lock_descriptor)
                    journal._lock_descriptor = None
            raise

    def _entry(self, name: str, value: Mapping[str, Any]) -> Path:
        record = dict(value)
        record["previous_entry_sha256"] = self._last_sha256
        path = _atomic_create_json(self.journal_dir / name, record)
        digest = _sha256_bytes(path.read_bytes())
        self._entries[name] = digest
        self._last_sha256 = digest
        return path

    def reserve_materialization(self, label: str) -> int:
        if time.time() >= self.deadline_epoch:
            raise TimeoutError("deadline expired before materialization reservation")
        if self.materializations >= MAXIMUM_FEATURE_OR_PROPOSAL_MATERIALIZATIONS:
            raise RuntimeError("materialization ceiling would be exceeded")
        ordinal = self.materializations + 1
        self._entry(
            f"m{ordinal:03d}_reserved.json",
            {"ordinal": ordinal, "label": label, "created_at_kst": _now_kst()},
        )
        self.materializations = ordinal
        return ordinal

    def reserve_fit(self, phase: str, window: str, cell: str, seed: int) -> int:
        if time.time() >= self.deadline_epoch:
            raise TimeoutError("deadline expired before physical fit reservation")
        if self.fit_reservations >= MAXIMUM_LIFETIME_PHYSICAL_FITS:
            raise RuntimeError("72-fit lifetime ceiling would be exceeded")
        ordinal = self.fit_reservations + 1
        if ordinal <= 9:
            zero_based = ordinal - 1
            expected_window = INNER_WINDOW_IDS[zero_based // len(ROUND_B_SEEDS)]
            expected_seed = ROUND_B_SEEDS[zero_based % len(ROUND_B_SEEDS)]
            expected = ("INNER_ANCHOR", expected_window, "ROUND_B_SHARED", expected_seed)
            if (phase, window, cell, seed) != expected:
                raise RuntimeError("inner anchor fit reservation differs from 9-fit plan")
        elif ordinal <= 63:
            zero_based = ordinal - 10
            per_window = len(STRUCTURE_CELL_IDS) * len(SEEDS)
            expected_window = INNER_WINDOW_IDS[zero_based // per_window]
            within_window = zero_based % per_window
            expected_cell = STRUCTURE_CELL_IDS[within_window // len(SEEDS)]
            expected_seed = SEEDS[within_window % len(SEEDS)]
            expected = ("INNER_SEGMENT", expected_window, expected_cell, expected_seed)
            if (phase, window, cell, seed) != expected:
                raise RuntimeError("inner segment fit reservation differs from 54-fit plan")
        else:
            zero_based = ordinal - 64
            expected_window = FOLD_ORDER[zero_based // len(SEEDS)]
            expected_seed = SEEDS[zero_based % len(SEEDS)]
            if cell not in STRUCTURE_CELL_IDS:
                raise RuntimeError("outer selected cell is not preregistered")
            if self._selected_outer_cell is None:
                self._selected_outer_cell = cell
            if (
                phase != "OUTER_SEGMENT"
                or window != expected_window
                or seed != expected_seed
                or cell != self._selected_outer_cell
            ):
                raise RuntimeError("outer fit reservation differs from locked 9-fit plan")
        self._entry(
            f"f{ordinal:03d}_reserved.json",
            {
                "ordinal": ordinal,
                "phase": phase,
                "window": window,
                "cell": cell,
                "seed": seed,
                "created_at_kst": _now_kst(),
            },
        )
        self.fit_reservations = ordinal
        return ordinal

    def complete_fit(self, ordinal: int) -> None:
        if ordinal != self.fits_completed + 1 or ordinal > self.fit_reservations:
            raise RuntimeError("fit completion order differs from reservations")
        self._entry(
            f"f{ordinal:03d}_completed.json",
            {"ordinal": ordinal, "created_at_kst": _now_kst()},
        )
        self.fits_completed = ordinal

    def fail(self, phase: str, error: BaseException) -> Path:
        name = "999_failed.json"
        path = self.journal_dir / name
        if path.exists():
            return path
        return self._entry(
            name,
            {
                "phase": phase,
                "error": _sanitize_error(error),
                "fit_reservations": self.fit_reservations,
                "fits_completed": self.fits_completed,
                "materializations": self.materializations,
                "execution_lock_retained": True,
                "journal_prefix_sha256": self._last_sha256,
                "created_at_kst": _now_kst(),
            },
        )

    def close_descriptor(self) -> None:
        if self._lock_descriptor is not None:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None


def _run_supervised(
    command: Sequence[str], deadline_epoch: float, template: Mapping[str, Any]
) -> tuple[str, str]:
    supervisor = _load_supervisor_infrastructure(template)
    return supervisor._run_supervised(command, deadline_epoch)


def seal() -> Path:
    if SEAL_PATH.exists():
        raise FileExistsError("preexecution seal already exists")
    checks = _static_package_checks()
    template, template_receipt = _read_template()
    seal_value = {
        "schema_version": "p1_long_event_segment_proposal_rescore.preexecution_seal.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "SEALED_STATIC_NOT_AUTHORIZED_PENDING_INDEPENDENT_QA",
        "created_at_kst": _now_kst(),
        "design_sha256": DESIGN_SHA256,
        "trigger_resolution_sha256": TRIGGER_RESOLUTION_SHA256,
        "operational_amendment_sha256": OPERATIONAL_AMENDMENT_SHA256,
        "runner_sha256": _sha256_bytes(Path(__file__).resolve().read_bytes()),
        "runner_normalized_sha256": _normalised_runner_sha256(),
        "module_sha256": _sha256_bytes(MODULE_PATH.read_bytes()),
        "test_sha256": _sha256_bytes(TEST_PATH.read_bytes()),
        "authorization_template_sha256": template_receipt["actual_read_sha256"],
        "actual_execution_authorization_exists": AUTHORIZATION_PATH.exists(),
        "actual_execution_authorization_sha256_literal": AUTHORIZATION_SHA256,
        "fit_budget": {
            "inner_round_b_anchor": 9,
            "inner_segment": 54,
            "outer_segment": 9,
            "segment_total": 63,
            "lifetime": 72,
        },
        "materialization_ceiling": 21,
        "static_checks": checks,
        "operation_counters": {
            "claims": 0,
            "physical_fits": 0,
            "feature_or_proposal_materializations": 0,
            "outer_scores": 0,
            "candidate_files": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
            "uploads": 0,
        },
        "authorization_template_status": template["status"],
    }
    if AUTHORIZATION_PATH.exists():
        raise RuntimeError("unexpected actual execution authorization exists")
    return _atomic_create_json(SEAL_PATH, seal_value)


def strict_preflight() -> Path:
    if PREFLIGHT_PATH.exists():
        raise FileExistsError("strict preflight receipt already exists")
    seal_value, seal_receipt = _read_bound_json(SEAL_PATH)
    if seal_value.get("status") != "SEALED_STATIC_NOT_AUTHORIZED_PENDING_INDEPENDENT_QA":
        raise RuntimeError("preexecution seal status changed")
    if seal_value.get("runner_sha256") != _sha256_bytes(Path(__file__).resolve().read_bytes()):
        raise RuntimeError("runner changed after seal")
    if seal_value.get("module_sha256") != _sha256_bytes(MODULE_PATH.read_bytes()):
        raise RuntimeError("module changed after seal")
    if seal_value.get("test_sha256") != _sha256_bytes(TEST_PATH.read_bytes()):
        raise RuntimeError("tests changed after seal")
    package = _static_package_checks()
    template, _template_receipt = _read_template()
    data = _strict_data_preflight(template)
    if AUTHORIZATION_PATH.exists():
        raise RuntimeError("unexpected actual execution authorization exists")
    receipt = {
        "schema_version": "p1_long_event_segment_proposal_rescore.strict_preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS_STRICT_COMPLETE_READINESS_NOT_AUTHORIZED_NO_CLAIM_NO_FIT",
        "created_at_kst": _now_kst(),
        "preexecution_seal": seal_receipt,
        "package": package,
        "data_readiness": data,
        "authorization": {
            "actual_execution_authorization_exists": False,
            "fit_authorized": False,
            "materialization_authorized": False,
            "official_action_authorized": False,
            "execution_ready": False,
        },
        "operation_counters": {
            "claims": 0,
            "physical_fits": 0,
            "feature_or_proposal_materializations": 0,
            "outer_scores": 0,
            "candidate_files": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
            "uploads": 0,
        },
    }
    return _atomic_create_json(PREFLIGHT_PATH, receipt)


def execute_parent() -> None:
    authorization = _require_execution_authorization_preimport()
    deadline = time.time() + HARD_WALL_SECONDS
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--deadline-epoch",
        repr(deadline),
    ]
    _run_supervised(command, deadline, authorization)


def execute_worker(deadline_epoch: float) -> None:
    authorization = _require_execution_authorization_preimport()
    if time.time() >= deadline_epoch:
        raise TimeoutError("deadline expired before worker readiness")
    _static_package_checks()
    _strict_data_preflight(authorization)
    raise AuthorizationError(
        "numeric worker is unreachable in the sealed zero-fit package until an "
        "independent QA successor embeds an actual authorization digest"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", action="store_true")
    parser.add_argument("--deadline-epoch", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seal:
        print(seal())
        return
    if args.preflight:
        print(strict_preflight())
        return
    if args.execute:
        execute_parent()
        return
    if args.deadline_epoch is None:
        raise SystemExit("--worker requires --deadline-epoch")
    execute_worker(float(args.deadline_epoch))


if __name__ == "__main__":
    main()
