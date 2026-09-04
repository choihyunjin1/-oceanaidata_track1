"""Externally authorized one-shot runner for the P1 Cycle-1 segment screen.

The checked-in authorization anchor is deliberately non-executable.  Static
trust, runtime, immutable training inputs, and the full 80-feature binding are
verified before any scientific claim.  When a later independent QA issues a
digest-bound authorization, the same sealed file supervises a private worker
that can execute the fixed 9 + 54 + 9 fit graph exactly once.

No official test, sample, submission, or candidate path is accepted here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

# Private snapshot code is always imported from source bytes whose digests are
# pinned below.  Never create or accept bytecode beside those source files.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v2_execution_contract.json"
)
DESIGN_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v1_design.json"
)
TRIGGER_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v1_trigger_resolution.json"
)
AMENDMENT_V2_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v2_operational_amendment.json"
)
CLOSURE_V3_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v3_execution_closure_amendment.json"
)
TRUST_FIREWALL_V5_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v5_trust_firewall_amendment.json"
)
EXECUTION_MODULE_PATH = (
    PROJECT_ROOT / "src" / "p1_qc" / "long_event_segment_proposal_rescore_execution_v2.py"
)
LEGACY_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "p1_round_b_nonspike_long_event_residual_v1r6.json"
)
LEGACY_RUNNER_PATH = (
    PROJECT_ROOT / "scripts" / "run_p1_round_b_nonspike_long_event_residual_v1r6.py"
)
LEGACY_PREFLIGHT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "p1_round_b_nonspike_long_event_residual_v1r6"
    / "strict_preflight_receipt.json"
)
TEST_PATH = PROJECT_ROOT / "tests" / "test_run_p1_long_event_segment_proposal_rescore_v2.py"
AUTHORIZATION_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v2_execution_authorization_template.json"
)
AUTHORIZATION_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_long_event_segment_proposal_rescore_v2_execution_authorization.json"
)
R5_QA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "p1_long_event_segment_proposal_rescore_v2_independent_preexecution_qa_r5_20260826.json"
)
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "p1_long_event_segment_proposal_rescore_20260826_v2"
SUPERSEDED_SEAL_R1_PATH = ARTIFACT_DIR / "preexecution_seal.json"
SUPERSEDED_SEAL_R2_PATH = ARTIFACT_DIR / "preexecution_seal_r2.json"
SUPERSEDED_SEAL_R3_PATH = ARTIFACT_DIR / "preexecution_seal_r3.json"
SUPERSEDED_SEAL_R4_PATH = ARTIFACT_DIR / "preexecution_seal_r4.json"
SEAL_PATH = ARTIFACT_DIR / "preexecution_seal_r5.json"

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v2"
SCIENTIFIC_EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
AUTHORIZATION_ENV_VAR = "P1_LONG_EVENT_SEGMENT_V2_AUTHORIZATION_SHA256"
CONFIG_SHA256 = "f64704a47751756cd88d9802eb6cf6ec160290af1f3112051ced91fa72b0febc"
DESIGN_SHA256 = "31b0bde27d8ef7e2b42135709563cca0bcca61c6ec6fdabefbb3530906869563"
TRIGGER_SHA256 = "f70a17ff6a30d59522a007b9605da12b23f8b5fb0c3247e45c9b30772c1cc2f1"
AMENDMENT_V2_SHA256 = "b33f7d386e05cd7ab79976f58e9f4ab752f37cfe6a8856849867ef5f541cb276"
CLOSURE_V3_SHA256 = "b7afec5a11e908f6e5fd8a1ef28404cbb062da39764b2649ddabc2659a56ad99"
TRUST_FIREWALL_V5_SHA256 = "bd0370c7100ae7602eb2b045b4ef69bf7808d345fcb3255234c0726225e57563"
LEGACY_CONFIG_SHA256 = "e447106eef3123f52b06a1577944f006ec9fd469c325069bda8b87cd37fbacc1"
LEGACY_RUNNER_SHA256 = "47528af803da0d582a17db41b03b189240a79f5735aac041cae7e331c0d9d94a"
LEGACY_PREFLIGHT_SHA256 = "075fa781ccd1201ea3476e4219868eac583992f1860af28f0ab62b5c9c60f131"
EXECUTION_MODULE_SHA256 = "68b644b1523fef498c00e03ee09058a7f9280761de70ec771e09e8c32a768e81"

MAXIMUM_LIFETIME_PHYSICAL_FITS = 72
MAXIMUM_SCIENTIFIC_MATERIALIZATIONS = 21
HARD_WALL_SECONDS = 21600
ROUND_B_SEEDS = (20260813, 20260829, 20260847)
SEGMENT_SEEDS = (20260826, 20260843, 20260871)
INNER_WINDOW_IDS = (
    "inner_2024_jul_aug",
    "inner_2024_oct_nov",
    "inner_2025_jan_feb",
)
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
CONTEXT_BANK_IDS = ("24_72", "48_168", "24_72_168")
DECODER_IDS = ("connected_only", "dual_boundary_disconnected_allowed")
STRUCTURE_CELL_IDS = tuple(
    f"bank_{bank}__{decoder}" for bank in CONTEXT_BANK_IDS for decoder in DECODER_IDS
)
AUTHORIZATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "status",
        "authorized",
        "contract_sha256",
        "trust_firewall_v5_sha256",
        "runner_sha256",
        "runner_normalized_sha256",
        "preexecution_seal",
        "independent_qa",
        "readonly_preflight_verification_sha256",
        "zero_prior_state",
        "operation_authorization",
    }
)

# Filled and sealed only after the numerical module and focused tests are
# final.  These are runner-owned trust roots; an authorization cannot replace
# or extend the map.
PROJECT_FILE_SHA256: dict[str, str] = {
    "configs/p1_meaningful_learning_curve_generation_v1.json": "45ed39e3cf2d84c254828b217d46166c2a511d4c7e7f9111d02896352c4d45f7",
    "scripts/run_p1_meaningful_learning_curve_generation_v1.py": "c22896fb17a5ece10831759631b9f7414cb9564ca80af2d1a4f3272fdeed6b9a",
    "scripts/run_p1_round_b_nonspike_long_event_residual_v1.py": "867e1f9c4d42564be8419f4612bdf595fc11005f460989b45b0c088f00ca6b45",
    "src/p1_qc/__init__.py": "5a7743ab77b2f9bd6851f12ebe694313c4fb361611821ed68dfd3b7574a82088",
    "src/p1_qc/audit.py": "fe339592bdacf6d8dbbb3f87dbfdbb11976767cb52f30a116046da4d1cbf1e62",
    "src/p1_qc/augment.py": "290a9b17d5c32940c745f70ea5f5ffe803180e401005a1febfc17e575d3a884a",
    "src/p1_qc/change_points.py": "99aed99428070104033053be885d42d2da7f61de35434fbd9867ad58bb569ae7",
    "src/p1_qc/config.py": "adf2e04123b95e51bcc7989e2c80b9faee776cb32fe63179b984a48cba2651f1",
    "src/p1_qc/data.py": "5dc1dac588faa2f50323d15ab2d83231eb06c3a709845d539ba7efbb99addd69",
    "src/p1_qc/experiment.py": "a331d15447abe1601dc98f351e5a47c23c8d2f50cc3145c48d07dd6814b75ab1",
    "src/p1_qc/features.py": "e769dbe187eeaa5061047a634bb2bcac6ea9a2ad9a359d48d98f49ba4b4b7162",
    "src/p1_qc/long_event_segment_proposal_rescore.py": "68aaf1fae9902b386b024500b5873328af5b0dddaf9d99243d799d68aa5c5feb",
    "src/p1_qc/metrics.py": "8ef58064c09b4d9710b1485151321d90ee0433b45f125dd52da9b3808464abbc",
    "src/p1_qc/models_tabular.py": "07b068a21c9101f9a4de53ee5f458d89ee9101468a148d8cde6f0afc50599311",
    "src/p1_qc/nonspike_long_event_residual.py": "32a2c993001813c42fee73235862b185e62cdbaa01bcca7bd6293e4be8a089b2",
    "src/p1_qc/pipeline.py": "389a905abbaf4b62e7d862c44fa25bba2e58dae7b7a7f5bcb4e1e8438d914669",
    "src/p1_qc/postprocess.py": "2066d8a45c71cdd1b77365a2334c16efb075c7e6d8feb7808d76aaef22cb5bf5",
    "src/p1_qc/rules.py": "ec921139f210f3b264c519346547fa0e17b094f54ce83780f21cf48f5287069c",
    "src/p1_qc/splits.py": "86e0f3990c73a0dbfa0810dce403001dbae726b8806df0e4f9483d3a05d7eb9d",
    "src/p1_qc/submission.py": "3d36d58c839536823c080d9ac7d377af73dae0ea00f3bf0d098f6d2fd125bb10",
    "src/p1_qc/validation.py": "7794d8c2c69fd35f93272088fecf9b80be553581351c0dbf6c5211b11df8ada6",
}

_SENSITIVE_NAMES = {
    "test.csv",
    "sample_submission.csv",
    "submission.csv",
}


class AuthorizationError(RuntimeError):
    """Raised before any repository/data read when execute is unauthorized."""


class ProcessTreeTerminationError(RuntimeError):
    """A supervised process tree could not be proved terminated."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


class WorkerTimeoutError(TimeoutError):
    """The fixed wall deadline expired."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bound_bytes(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        value = handle.read()
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ) or len(value) != before.st_size:
        raise RuntimeError(f"held file changed while read: {path.name}")
    observed = _sha256_bytes(value)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"held file digest mismatch: {path.name}")
    return value, {
        "bytes": len(value),
        "actual_read_sha256": observed,
        "parsed_from_same_held_bytes": True,
        "held_file_identity": {
            "size": int(before.st_size),
            "mtime_ns": int(before.st_mtime_ns),
            "inode": int(before.st_ino),
        },
    }


def _verify_held_path_identity(path: Path, held: Mapping[str, Any]) -> None:
    identity = held["receipt"]["held_file_identity"]
    observed = path.stat()
    if (
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ino),
    ) != (
        int(identity["size"]),
        int(identity["mtime_ns"]),
        int(identity["inode"]),
    ):
        raise RuntimeError(f"held authority path identity changed: {path.name}")


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


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )


def _relative(path: Path) -> str:
    return str(path.resolve(strict=True).relative_to(PROJECT_ROOT)).replace("\\", "/")


def _relative_literal(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _resolve_repo_path(relative: str, *, must_exist: bool = True) -> Path:
    path = (PROJECT_ROOT / relative).resolve(strict=must_exist)
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError("path escapes repository")
    _reject_sensitive_path(path)
    return path


def _reject_sensitive_path(path: Path) -> None:
    name = path.name.lower()
    if name in _SENSITIVE_NAMES or (
        "submission" in name and path.suffix.lower() in {".csv", ".parquet"}
    ):
        raise RuntimeError("official/sample/submission path is prohibited")


def _normalised_runner_bytes(path: Path | None = None) -> bytes:
    # r5 is byte-immutable across authorization. No authorization digest is
    # embedded in source, so the normalized and raw runner trust roots agree.
    return (path or Path(__file__).resolve()).read_bytes()


def _normalised_runner_sha256(path: Path | None = None) -> str:
    return _sha256_bytes(_normalised_runner_bytes(path))


def _verify_top_level_stdlib_only(path: Path | None = None) -> list[str]:
    runner = path or Path(__file__).resolve()
    tree = ast.parse(runner.read_text(encoding="utf-8"), filename=str(runner))
    imported: list[str] = []
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name.split(".", maxsplit=1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [str(node.module or "").split(".", maxsplit=1)[0]]
        else:
            continue
        imported.extend(names)
        disallowed = [name for name in names if name not in allowed]
        if disallowed:
            raise RuntimeError(f"non-stdlib top-level import: {disallowed}")
    return sorted(set(imported))


def _fsync_directory(path: Path) -> None:
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


def _publish_temp_create_only(temporary: Path, target: Path) -> None:
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise
    except OSError as error:
        raise RuntimeError("atomic create-only hardlink publication unavailable") from error
    _fsync_directory(target.parent)
    temporary.unlink()


def _atomic_create_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_create_only(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _atomic_create_bytes(path, _json_bytes(value))


def _sanitize_error(error: BaseException, sensitive_roots: Sequence[Path] = ()) -> dict[str, str]:
    raw = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    roots = {PROJECT_ROOT, Path.home(), *sensitive_roots}
    sanitized = raw
    for root in sorted(roots, key=lambda value: len(str(value)), reverse=True):
        sanitized = sanitized.replace(str(root), f"<REDACTED:{root.name or 'ROOT'}>")
    sanitized = "".join(
        character if character in "\n\r\t" or ord(character) >= 32 else "?"
        for character in sanitized
    )
    return {
        "type": type(error).__name__,
        "message": str(error).replace(str(PROJECT_ROOT), "<PROJECT_ROOT>"),
        "sanitized_traceback": sanitized,
        "sanitized_traceback_sha256": _sha256_bytes(sanitized.encode()),
    }


def _verify_hash_map() -> dict[str, dict[str, Any]]:
    if not PROJECT_FILE_SHA256:
        raise RuntimeError("runner project-file trust map is not sealed")
    receipts: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(PROJECT_FILE_SHA256.items()):
        path = _resolve_repo_path(relative)
        _raw, receipt = _read_bound_bytes(path, expected_sha256=expected)
        receipts[relative] = receipt
    return receipts


def _validate_closure(
    contract: Mapping[str, Any],
    design: Mapping[str, Any],
    amendment: Mapping[str, Any],
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("execution contract identity changed")
    if contract.get("scientific_experiment_id") != SCIENTIFIC_EXPERIMENT_ID:
        raise RuntimeError("scientific experiment identity changed")
    if contract.get("artifact_dir") != (
        "artifacts/p1_long_event_segment_proposal_rescore_20260826_v2"
    ):
        raise RuntimeError("artifact namespace changed")
    if contract.get("status") != "IMPLEMENTED_ZERO_FIT_NOT_AUTHORIZED_PENDING_INDEPENDENT_QA":
        raise RuntimeError("prospective execution status changed")
    authorities = contract["authorities"]
    expected_authorities = {
        "scientific_design": DESIGN_SHA256,
        "trigger_resolution": TRIGGER_SHA256,
        "operational_amendment_v2": AMENDMENT_V2_SHA256,
        "execution_closure_v3": CLOSURE_V3_SHA256,
    }
    for name, expected in expected_authorities.items():
        if authorities[name]["sha256"] != expected:
            raise RuntimeError(f"authority hash changed: {name}")
    graph = contract["fixed_operation_graph"]
    if {
        "inner_anchor_physical_fits": int(graph["inner_anchor_physical_fits"]),
        "inner_segment_physical_fits": int(graph["inner_segment_physical_fits"]),
        "outer_segment_physical_fits": int(graph["outer_segment_physical_fits"]),
        "maximum_lifetime_physical_fits": int(graph["maximum_lifetime_physical_fits"]),
        "maximum_lifetime_scientific_materializations": int(
            graph["maximum_lifetime_scientific_materializations"]
        ),
        "hard_wall_seconds": int(graph["hard_wall_seconds"]),
    } != {
        "inner_anchor_physical_fits": 9,
        "inner_segment_physical_fits": 54,
        "outer_segment_physical_fits": 9,
        "maximum_lifetime_physical_fits": 72,
        "maximum_lifetime_scientific_materializations": 21,
        "hard_wall_seconds": 21600,
    }:
        raise RuntimeError("fixed operation graph changed")
    if any(int(value) != 0 for value in contract["prohibitions"].values()):
        raise RuntimeError("execution prohibition counter changed")
    if closure.get("status") != "PROSPECTIVE_EXECUTION_CLOSURE_NOT_AUTHORIZED":
        raise RuntimeError("execution closure status changed")
    if closure.get("created_before_any_cycle1_fit_materialization_score_or_result") is not True:
        raise RuntimeError("execution closure is not prospective")
    if closure["authorization"] != {
        "implementation_allowed": True,
        "numerical_execute_authorized": False,
        "fit_authorized": False,
        "scientific_materialization_authorized": False,
        "official_action_authorized": False,
        "requires_independent_qa_and_new_digest_bound_authorization": True,
    }:
        raise RuntimeError("closure authorization state changed")
    expected_windows = [
        {
            "id": "inner_2024_jul_aug",
            "anchor_fit_end_inclusive": "2024-05-24T23:50:00+09:00",
            "support_surface_start": "2024-05-25T00:00:00+09:00",
            "segment_calibration_start": "2024-06-01T00:00:00+09:00",
            "segment_calibration_end_exclusive": "2024-07-01T00:00:00+09:00",
            "support_surface_end_exclusive": "2024-07-08T00:00:00+09:00",
            "inner_validation_start": "2024-07-08T00:00:00+09:00",
            "inner_validation_end_inclusive": "2024-08-31T23:50:00+09:00",
            "central_grid_slots_per_continuous_station_layer": 4320,
        },
        {
            "id": "inner_2024_oct_nov",
            "anchor_fit_end_inclusive": "2024-08-24T23:50:00+09:00",
            "support_surface_start": "2024-08-25T00:00:00+09:00",
            "segment_calibration_start": "2024-09-01T00:00:00+09:00",
            "segment_calibration_end_exclusive": "2024-10-01T00:00:00+09:00",
            "support_surface_end_exclusive": "2024-10-08T00:00:00+09:00",
            "inner_validation_start": "2024-10-08T00:00:00+09:00",
            "inner_validation_end_inclusive": "2024-11-30T23:50:00+09:00",
            "central_grid_slots_per_continuous_station_layer": 4320,
        },
        {
            "id": "inner_2025_jan_feb",
            "anchor_fit_end_inclusive": "2024-11-23T23:50:00+09:00",
            "support_surface_start": "2024-11-24T00:00:00+09:00",
            "segment_calibration_start": "2024-12-01T00:00:00+09:00",
            "segment_calibration_end_exclusive": "2025-01-01T00:00:00+09:00",
            "support_surface_end_exclusive": "2025-01-08T00:00:00+09:00",
            "inner_validation_start": "2025-01-08T00:00:00+09:00",
            "inner_validation_end_inclusive": "2025-03-01T23:50:00+09:00",
            "central_grid_slots_per_continuous_station_layer": 4464,
        },
    ]
    observed_windows = closure["out_of_sample_anchor_surfaces"]["windows"]
    if observed_windows != expected_windows:
        raise RuntimeError("frozen anchor cutoffs/shelves changed")
    if closure["out_of_sample_anchor_surfaces"]["round_b_seeds_unchanged"] != list(ROUND_B_SEEDS):
        raise RuntimeError("Round-B anchor seeds changed")
    plan = closure["exact_resource_and_materialization_plan"]
    if (
        int(plan["inner_anchor_fits"]),
        int(plan["inner_segment_fits"]),
        int(plan["outer_segment_fits"]),
        int(plan["maximum_lifetime_physical_fits"]),
        int(plan["maximum_lifetime_scientific_materializations"]),
    ) != (9, 54, 9, 72, 21):
        raise RuntimeError("closure fit/materialization accounting changed")
    if closure["segment_training_and_outer_rule"]["outer_training_corpus"] != (
        "After inner selection is frozen, each outer-fold segment model is trained "
        "only on the three OOS central calibration-shelf proposal pools for the "
        "selected bank. Inner validation labels remain selection-only and outer "
        "labels remain evaluation-only."
    ):
        raise RuntimeError("outer-label firewall changed")
    fixed = closure["fixed_segment_model_authority"]
    if fixed["registered_segment_seeds"] != list(SEGMENT_SEEDS):
        raise RuntimeError("segment seeds changed")
    if tuple(
        tuple(bank)
        for bank in closure["bounded_target_free_proposal_contract"]["context_banks_hours"]
    ) != (
        (24, 72),
        (48, 168),
        (24, 72, 168),
    ):
        raise RuntimeError("context banks changed")
    amendment_windows = amendment["inner_anchor_fits"]["windows"]
    if [window["fit_end_inclusive"] for window in amendment_windows] != [
        window["anchor_fit_end_inclusive"] for window in expected_windows
    ]:
        raise RuntimeError("closure cutoffs differ from frozen v2 amendment")
    frozen_search = design["fixed_structure_search"]
    if (
        int(frozen_search["inner_physical_fit_calls"]),
        int(frozen_search["outer_locked_physical_fit_calls"]),
    ) != (54, 9):
        raise RuntimeError("frozen design segment fit graph changed")
    return {
        "cutoffs_equal_frozen_v2": True,
        "oos_shelves": [window["id"] for window in expected_windows],
        "outer_labels_in_fit_or_selection": 0,
        "fit_plan": {"anchor": 9, "inner_segment": 54, "outer_segment": 9},
        "materialization_ceiling": 21,
    }


def _static_package_checks(
    held_authorities: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _verify_top_level_stdlib_only()
    expected_paths = {
        CONFIG_PATH: CONFIG_SHA256,
        DESIGN_PATH: DESIGN_SHA256,
        TRIGGER_PATH: TRIGGER_SHA256,
        AMENDMENT_V2_PATH: AMENDMENT_V2_SHA256,
        CLOSURE_V3_PATH: CLOSURE_V3_SHA256,
        TRUST_FIREWALL_V5_PATH: TRUST_FIREWALL_V5_SHA256,
        LEGACY_CONFIG_PATH: LEGACY_CONFIG_SHA256,
        LEGACY_RUNNER_PATH: LEGACY_RUNNER_SHA256,
        LEGACY_PREFLIGHT_PATH: LEGACY_PREFLIGHT_SHA256,
        EXECUTION_MODULE_PATH: EXECUTION_MODULE_SHA256,
    }
    raws: dict[Path, bytes] = {}
    receipts: dict[str, Any] = {}
    for path, expected in expected_paths.items():
        relative = _relative(path)
        held = (held_authorities or {}).get(relative)
        if held is None:
            raw, receipt = _read_bound_bytes(path, expected_sha256=expected)
        else:
            raw = bytes(held["raw"])
            observed = _sha256_bytes(raw)
            if observed != expected or held["receipt"]["actual_read_sha256"] != expected:
                raise RuntimeError(f"held authority digest changed: {path.name}")
            receipt = dict(held["receipt"])
        raws[path] = raw
        receipts[relative] = receipt
    dependency_receipts = _verify_hash_map()
    contract = _json_from_bytes(raws[CONFIG_PATH], label=CONFIG_PATH.name)
    design = _json_from_bytes(raws[DESIGN_PATH], label=DESIGN_PATH.name)
    amendment = _json_from_bytes(raws[AMENDMENT_V2_PATH], label=AMENDMENT_V2_PATH.name)
    closure = _json_from_bytes(raws[CLOSURE_V3_PATH], label=CLOSURE_V3_PATH.name)
    trust = _json_from_bytes(
        raws[TRUST_FIREWALL_V5_PATH],
        label=TRUST_FIREWALL_V5_PATH.name,
    )
    closure_receipt = _validate_closure(contract, design, amendment, closure)
    if (
        trust.get("status") != "PROSPECTIVE_R5_TRUST_FIREWALL_NOT_AUTHORIZED"
        or trust.get("created_before_any_cycle1_fit_materialization_score_or_result") is not True
        or trust["immutable_scientific_authorities"]["scientific_values_modified"] is not False
        or trust["immutable_scientific_authorities"]["maximum_lifetime_physical_fits"] != 72
        or trust["immutable_scientific_authorities"]["maximum_lifetime_scientific_materializations"]
        != 21
        or any(int(value) != 0 for value in trust["prohibitions"].values())
    ):
        raise RuntimeError("r5 trust-firewall authority changed")
    for evidence_name in ("independent_qa_json", "independent_qa_korean_report"):
        evidence = trust["supersedes"][evidence_name]
        evidence_path = _resolve_repo_path(str(evidence["path"]))
        raw, receipt = _read_bound_bytes(
            evidence_path,
            expected_sha256=str(evidence["sha256"]),
        )
        if len(raw) != int(evidence["bytes"]):
            raise RuntimeError(f"r4 QA evidence byte count changed: {evidence_name}")
        receipts[_relative(evidence_path)] = receipt
    legacy = contract["legacy_verified_readiness"]
    if (
        legacy["config_sha256"] != LEGACY_CONFIG_SHA256
        or legacy["runner_sha256"] != LEGACY_RUNNER_SHA256
        or legacy["strict_preflight_sha256"] != LEGACY_PREFLIGHT_SHA256
    ):
        raise RuntimeError("legacy full-readiness lineage changed")
    old = closure["superseded_zero_fit_package"]
    for key in ("runner_path", "module_path", "seal_path", "preflight_path"):
        path = _resolve_repo_path(str(old[key]))
        hash_key = key.replace("_path", "_sha256")
        if _sha256(path) != old[hash_key]:
            raise RuntimeError(f"superseded v1 proof changed: {key}")
    return {
        "status": "PASS_RUNNER_OWNED_TRANSITIVE_TRUST_ROOTS",
        "runner_normalized_sha256": _normalised_runner_sha256(),
        "authority_receipts": receipts,
        "project_dependency_receipts": dependency_receipts,
        "project_dependency_count": len(dependency_receipts),
        "closure": closure_receipt,
        "trust_firewall_v5_sha256": TRUST_FIREWALL_V5_SHA256,
        "superseded_v1_preserved": True,
    }


def _load_module_from_path(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"unable to load verified module: {path.name}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_snapshot_legacy_runner(snapshot_root: Path, name_prefix: str) -> ModuleType:
    path = snapshot_root / _relative_literal(LEGACY_RUNNER_PATH)
    _read_bound_bytes(path, expected_sha256=LEGACY_RUNNER_SHA256)
    module = _load_module_from_path(path, f"{name_prefix}_{uuid.uuid4().hex}")
    _read_bound_bytes(path, expected_sha256=LEGACY_RUNNER_SHA256)
    return module


def _runtime_receipt(legacy_config: Mapping[str, Any]) -> dict[str, Any]:
    expected = legacy_config["trust_contract"]["runtime_versions"]
    observed = {"python": platform.python_version()}
    for distribution in expected:
        if distribution == "python":
            continue
        observed[distribution] = importlib.metadata.version(distribution)
    if observed != expected:
        raise RuntimeError("runtime version closure changed")
    native = legacy_config["trust_contract"]["lightgbm_native"]
    distribution = importlib.metadata.distribution("lightgbm")
    native_path = Path(distribution.locate_file(native["distribution_relative_path"])).resolve(
        strict=True
    )
    raw, receipt = _read_bound_bytes(native_path, expected_sha256=native["sha256"])
    if len(raw) != int(native["bytes"]):
        raise RuntimeError("LightGBM native byte count changed")
    return {"runtime_versions": observed, "lightgbm_native": receipt | dict(native)}


def _remove_snapshot_modules(snapshot_root: Path) -> None:
    for name, module in list(sys.modules.items()):
        value = getattr(module, "__file__", None)
        if not value:
            continue
        try:
            path = Path(value).resolve()
        except (OSError, RuntimeError):
            continue
        if path.is_relative_to(snapshot_root):
            sys.modules.pop(name, None)
    sys.path[:] = [
        value
        for value in sys.path
        if not Path(value or ".").resolve().is_relative_to(snapshot_root)
    ]


def _input_specs(legacy_config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    immutable = legacy_config["immutable_inputs"]
    evidence = legacy_config["preexecution_evidence"]
    values: list[tuple[str, Mapping[str, Any]]] = [
        ("base_config", legacy_config["base_config"]),
        ("feature_cache", immutable["feature_cache"]),
        ("feature_metadata", immutable["feature_metadata"]),
        ("feature_cache_key_binding", evidence["feature_cache_key_binding"]),
        (
            "feature_cache_key_binding_receipt",
            evidence["feature_cache_key_binding_receipt"],
        ),
        ("frozen_truth_oof", immutable["frozen_truth_oof"]),
        ("matched_budget_predictions", immutable["matched_budget_predictions"]),
    ]
    values.extend(
        (f"round_b_full_prefix:{part['fold']}", part)
        for part in immutable["round_b_full_prefix_parts"]
    )
    return values


def _training_source_path(legacy_config: Mapping[str, Any]) -> Path:
    raw = os.environ.get("P1_DATA_DIR")
    if not raw:
        raise RuntimeError("P1_DATA_DIR is required before any execution claim")
    directory = Path(raw).expanduser().resolve(strict=True)
    path = (directory / "train.csv").resolve(strict=True)
    if path.parent != directory or not path.is_file() or path.name != "train.csv":
        raise RuntimeError("only P1 train.csv is accepted as the external source")
    expected = str(legacy_config["immutable_inputs"]["feature_cache"]["source_sha256"])
    if _sha256(path) != expected:
        raise RuntimeError("P1 training source digest changed")
    return path


def _stdlib_data_readiness(
    legacy_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    source = _training_source_path(legacy_config)
    files: dict[str, dict[str, Any]] = {
        "training_source": {
            "filename": "train.csv",
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
    }
    for name, specification in _input_specs(legacy_config):
        path = _resolve_repo_path(str(specification["path"]))
        digest = _sha256(path)
        if digest != specification["sha256"]:
            raise RuntimeError(f"immutable input mismatch before claim: {name}")
        files[name] = {
            "project_path": _relative(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return {
        "status": "PASS_STDLIB_INPUT_READINESS_NO_CLAIM",
        "runtime": dict(runtime),
        "files": files,
        "source_path": source,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
    }


def _copy_verified_file(source: Path, destination: Path, expected: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.snapshot"
    digest = hashlib.sha256()
    try:
        with source.open("rb") as source_handle:
            before = os.fstat(source_handle.fileno())
            with temporary.open("xb") as target_handle:
                for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                    digest.update(block)
                    target_handle.write(block)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            after = os.fstat(source_handle.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise RuntimeError(f"held snapshot source changed: {source.name}")
        observed = digest.hexdigest()
        if observed != expected:
            raise RuntimeError(f"held snapshot source digest mismatch: {source.name}")
        _publish_temp_create_only(temporary, destination)
        snapshot_digest = _sha256(destination)
        if snapshot_digest != expected:
            raise RuntimeError(f"private snapshot digest mismatch: {destination.name}")
        return {
            "bytes": destination.stat().st_size,
            "sha256": snapshot_digest,
            "actual_read_sha256": observed,
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def _snapshot_held_bytes(
    destination: Path,
    held: Mapping[str, Any],
    expected: str,
) -> dict[str, Any]:
    raw = bytes(held["raw"])
    if (
        _sha256_bytes(raw) != expected
        or held["receipt"]["actual_read_sha256"] != expected
        or int(held["receipt"]["bytes"]) != len(raw)
    ):
        raise RuntimeError(f"held authority changed before snapshot: {destination.name}")
    _atomic_create_bytes(destination, raw)
    return {
        "bytes": len(raw),
        "sha256": expected,
        "actual_read_sha256": expected,
        "copied_from_single_held_authority_snapshot": True,
    }


def _prepare_snapshot(
    legacy_config: Mapping[str, Any],
    readiness: Mapping[str, Any],
    held_authorities: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    root = Path(tempfile.mkdtemp(prefix="p1_segment_v2_verified_snapshot_"))
    records: dict[str, dict[str, Any]] = {}
    try:
        source = Path(readiness["source_path"])
        source_expected = str(legacy_config["immutable_inputs"]["feature_cache"]["source_sha256"])
        records["inputs/train.csv"] = _copy_verified_file(
            source,
            root / "inputs" / "train.csv",
            source_expected,
        )
        for _name, specification in _input_specs(legacy_config):
            relative = str(specification["path"])
            records[relative] = _copy_verified_file(
                _resolve_repo_path(relative),
                root / relative,
                str(specification["sha256"]),
            )
        for relative, expected in PROJECT_FILE_SHA256.items():
            if relative in records:
                continue
            records[relative] = _copy_verified_file(
                _resolve_repo_path(relative),
                root / relative,
                expected,
            )
        fixed = {
            _relative(CONFIG_PATH): CONFIG_SHA256,
            _relative(DESIGN_PATH): DESIGN_SHA256,
            _relative(TRIGGER_PATH): TRIGGER_SHA256,
            _relative(AMENDMENT_V2_PATH): AMENDMENT_V2_SHA256,
            _relative(CLOSURE_V3_PATH): CLOSURE_V3_SHA256,
            _relative(TRUST_FIREWALL_V5_PATH): TRUST_FIREWALL_V5_SHA256,
            _relative(LEGACY_CONFIG_PATH): LEGACY_CONFIG_SHA256,
            _relative(LEGACY_RUNNER_PATH): LEGACY_RUNNER_SHA256,
            _relative(LEGACY_PREFLIGHT_PATH): LEGACY_PREFLIGHT_SHA256,
            _relative(EXECUTION_MODULE_PATH): EXECUTION_MODULE_SHA256,
        }
        for relative, expected in fixed.items():
            if relative in records:
                continue
            held = (held_authorities or {}).get(relative)
            records[relative] = (
                _snapshot_held_bytes(root / relative, held, expected)
                if held is not None
                else _copy_verified_file(
                    _resolve_repo_path(relative),
                    root / relative,
                    expected,
                )
            )
        for relative, held in sorted((held_authorities or {}).items()):
            if relative in records:
                continue
            expected = str(held["receipt"]["actual_read_sha256"])
            records[relative] = _snapshot_held_bytes(
                root / relative,
                held,
                expected,
            )
        return root, records
    except BaseException:
        _cleanup_snapshot(root)
        raise


def _cleanup_snapshot(root: Path) -> None:
    resolved = root.resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if resolved.parent != temporary_root or not resolved.name.startswith(
        "p1_segment_v2_verified_snapshot_"
    ):
        raise RuntimeError("refusing unsafe private snapshot cleanup")
    if resolved.exists():
        shutil.rmtree(resolved)


def _load_snapshot_numerical(
    snapshot_root: Path,
    legacy_config: Mapping[str, Any],
    legacy: ModuleType,
) -> tuple[ModuleType, ModuleType, dict[str, Any]]:
    numerical, runtime = legacy._load_snapshot_numerical(snapshot_root, legacy_config)
    source_root = snapshot_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    expected_path = snapshot_root / _relative_literal(EXECUTION_MODULE_PATH)
    _read_bound_bytes(expected_path, expected_sha256=EXECUTION_MODULE_SHA256)
    module = importlib.import_module("p1_qc.long_event_segment_proposal_rescore_execution_v2")
    module_path = Path(str(module.__file__)).resolve(strict=True)
    expected_path = expected_path.resolve(strict=True)
    if module_path != expected_path or _sha256(module_path) != EXECUTION_MODULE_SHA256:
        raise RuntimeError("numerical execution module was not loaded from snapshot")
    _read_bound_bytes(expected_path, expected_sha256=EXECUTION_MODULE_SHA256)
    if module.EXECUTION_CLOSURE_V3_SHA256 != CLOSURE_V3_SHA256:
        raise RuntimeError("numerical execution module closure anchor changed")
    if module.TRUST_FIREWALL_V5_SHA256 != TRUST_FIREWALL_V5_SHA256:
        raise RuntimeError("numerical execution module r5 trust anchor changed")
    return numerical, module, runtime


def _strict_target_free_snapshot_readiness(
    snapshot_root: Path,
    config: Mapping[str, Any],
    numerical: ModuleType,
    runtime_receipt: Mapping[str, Any],
    legacy: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce v1r6 readiness without parsing outer target columns.

    The complete frozen-truth file is held-read and hashed once, while only
    its target-free ordered keys/folds are parsed before the prediction-freeze
    boundary.  Label/anomaly columns are parsed later from these same held
    bytes by the numerical module after ``record_outer_freeze``.
    """

    reader = legacy.HeldSnapshotInputs(snapshot_root)
    immutable = config["immutable_inputs"]
    cache_spec = immutable["feature_cache"]
    train = reader.read_csv(
        "inputs/train.csv",
        str(cache_spec["source_sha256"]),
        numerical,
    )
    train.attrs.update(
        {
            "source_path": "PRIVATE_VERIFIED_SNAPSHOT/inputs/train.csv",
            "source_size": reader.receipts["inputs/train.csv"]["bytes"],
            "source_sha256": str(cache_spec["source_sha256"]),
            "dataset_kind": "train",
        }
    )
    feature_relative = str(cache_spec["path"])
    features = reader.read_parquet(
        feature_relative,
        str(cache_spec["sha256"]),
        numerical,
    )
    metadata_spec = immutable["feature_metadata"]
    metadata = reader.read_json(
        str(metadata_spec["path"]),
        str(metadata_spec["sha256"]),
    )
    if int(metadata["rows"]) != len(train) or len(features) != int(cache_spec["rows"]):
        raise RuntimeError("feature cache row contract changed")
    if metadata.get("source_sha256") != cache_spec["source_sha256"]:
        raise RuntimeError("feature metadata source binding changed")
    if metadata.get("parquet_sha256") != cache_spec["sha256"]:
        raise RuntimeError("feature metadata cache binding changed")
    if tuple(features.columns) != tuple(metadata["feature_columns"]):
        raise RuntimeError("feature cache schema differs from metadata")
    if len(features.columns) != int(cache_spec["columns"]):
        raise RuntimeError("feature cache column count changed")
    if {"label", "anomaly_type"}.intersection(features.columns):
        raise RuntimeError("feature cache contains protected target columns")
    feature_binding = legacy._validate_feature_row_binding(train, features, numerical)

    evidence = config["preexecution_evidence"]
    sidecar_spec = evidence["feature_cache_key_binding"]
    sidecar = reader.read_parquet(
        str(sidecar_spec["path"]),
        str(sidecar_spec["sha256"]),
        numerical,
    )
    binding_receipt_spec = evidence["feature_cache_key_binding_receipt"]
    binding_receipt = reader.read_json(
        str(binding_receipt_spec["path"]),
        str(binding_receipt_spec["sha256"]),
    )
    feature_binding = legacy._validate_pinned_feature_key_binding(
        train,
        sidecar,
        feature_binding,
        cache_spec,
        sidecar_spec,
        binding_receipt_spec,
        binding_receipt,
        numerical,
    )
    p1_config = legacy._load_base_config_from_held(
        snapshot_root,
        config,
        numerical,
        reader,
    )
    full_feature_binding = legacy._full_feature_regeneration_readiness(
        train,
        features,
        p1_config,
        numerical,
        config,
        snapshot_root,
    )
    bundle = numerical.FeatureBundle(
        features,
        tuple(str(value) for value in metadata["feature_columns"]),
        tuple(str(value) for value in metadata["categorical_columns"]),
    )
    surface = legacy._load_base_surface_from_held(config, numerical, reader)

    truth_spec = immutable["frozen_truth_oof"]
    truth_relative = str(truth_spec["path"])
    truth_raw = reader.read_bytes(truth_relative, str(truth_spec["sha256"]))
    target_free_columns = [*legacy.KEY_COLUMNS, "fold"]
    truth_keys = numerical.pd.read_parquet(
        io.BytesIO(truth_raw),
        columns=target_free_columns,
    )
    if len(truth_keys) != int(truth_spec["rows"]):
        raise RuntimeError("frozen truth key row count changed")
    if truth_keys.duplicated(target_free_columns).any():
        raise RuntimeError("frozen truth target-free keys are duplicated")
    surface_keys = surface.loc[:, target_free_columns].reset_index(drop=True)
    if not surface_keys.equals(truth_keys.loc[:, target_free_columns].reset_index(drop=True)):
        raise RuntimeError("frozen truth ordered target-free keys differ from Round-B surface")

    folds = numerical._fold_runtime(train, p1_config, surface)
    left_counts = legacy._left_censored_positive_event_counts(train, folds, numerical)
    exact_equivalence = {
        "status": "PASS_TARGET_FREE_KEYS_EXACT_TRUTH_METRICS_DEFERRED_POST_FREEZE",
        "rows": len(truth_keys),
        "truth_key_alignment": "exact_ordered",
        "frozen_truth_bytes_sha256_verified": True,
        "pre_freeze_parsed_columns": target_free_columns,
        "pre_freeze_target_columns_parsed": 0,
        "expected_base_metrics_pinned": dict(config["surface"]["expected_base_metrics"]),
    }
    receipt = {
        "status": "PASS_COMPLETE_TARGET_FREE_READINESS_BEFORE_CLAIM",
        "rows": len(train),
        "oof_rows": len(surface),
        "fold_rows": {fold: int(surface["fold"].eq(fold).sum()) for fold in FOLD_ORDER},
        "feature_row_binding": feature_binding,
        "full_feature_cache_binding": full_feature_binding,
        "left_censored_positive_connected_event_count_by_fold": left_counts,
        "exact_round_b_equivalence": exact_equivalence,
        "held_snapshot_input_reads": reader.receipts,
        "runtime": runtime_receipt,
        "residual_model_fits": 0,
        "outer_scores": 0,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
    }
    state = {
        "train": train,
        "bundle": bundle,
        "feature_metadata": metadata,
        "surface": surface,
        "frozen_truth_oof_bytes": truth_raw,
        "frozen_truth_oof_sha256": str(truth_spec["sha256"]),
        "frozen_truth_oof_rows": int(truth_spec["rows"]),
        "expected_base_metrics": dict(config["surface"]["expected_base_metrics"]),
        "truth_read_receipt": dict(reader.receipts[truth_relative]),
        "folds": folds,
        "left_counts": left_counts,
    }
    return receipt, state


def _selected_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    runtime = readiness["runtime"]
    return {
        "status": readiness["status"],
        "rows": int(readiness["rows"]),
        "oof_rows": int(readiness["oof_rows"]),
        "fold_rows": readiness["fold_rows"],
        "feature_row_binding": readiness["feature_row_binding"],
        "full_feature_cache_binding": readiness["full_feature_cache_binding"],
        "left_censored_positive_connected_event_count_by_fold": readiness[
            "left_censored_positive_connected_event_count_by_fold"
        ],
        "exact_round_b_equivalence": readiness["exact_round_b_equivalence"],
        "held_snapshot_input_reads": readiness["held_snapshot_input_reads"],
        "runtime": {
            "runtime_versions": runtime["runtime_versions"],
            "loaded_distribution_names": runtime["loaded_distribution_names"],
            "loaded_runtime_file_count": runtime["loaded_runtime_file_count"],
            "loaded_runtime_file_aggregate_sha256": runtime["loaded_runtime_file_aggregate_sha256"],
            "lightgbm_native": runtime["lightgbm_native"],
        },
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
        "physical_fits": 0,
        "scientific_materializations": 0,
        "outer_scores": 0,
    }


def _attach_stable_verification_and_live_authorization(
    immutable_result: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(immutable_result)
    result["verification_sha256"] = _canonical_sha(immutable_result)
    supplied = os.environ.get(AUTHORIZATION_ENV_VAR, "")
    result["live_authorization_inspection"] = {
        "excluded_from_verification_sha256": True,
        "actual_authorization_exists": AUTHORIZATION_PATH.exists(),
        "external_digest_is_64_lower_hex": bool(re.fullmatch(r"[0-9a-f]{64}", supplied)),
    }
    return result


def _complete_readiness(
    *,
    retain_snapshot: bool,
    held_authorities: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path | None, dict[str, dict[str, Any]] | None]:
    static = _static_package_checks(held_authorities)
    legacy_raw, _receipt = _read_bound_bytes(
        LEGACY_CONFIG_PATH,
        expected_sha256=LEGACY_CONFIG_SHA256,
    )
    legacy_config = _json_from_bytes(legacy_raw, label=LEGACY_CONFIG_PATH.name)
    runtime = _runtime_receipt(legacy_config)
    stdlib = _stdlib_data_readiness(legacy_config, runtime)
    snapshot_root, records = _prepare_snapshot(
        legacy_config,
        stdlib,
        held_authorities,
    )
    try:
        _verify_exact_snapshot_inventory(snapshot_root, records, records)
        legacy = _load_snapshot_legacy_runner(snapshot_root, "_p1_v1r6_parent_readiness")
        numerical, _execution, loaded_runtime = _load_snapshot_numerical(
            snapshot_root,
            legacy_config,
            legacy,
        )
        readiness, _state = _strict_target_free_snapshot_readiness(
            snapshot_root,
            legacy_config,
            numerical,
            loaded_runtime,
            legacy,
        )
        selected = _selected_readiness(readiness)
        result = {
            "schema_version": "p1_long_event_segment_proposal_rescore.readonly_preflight.v2",
            "experiment_id": EXPERIMENT_ID,
            "status": "PASS_STRICT_COMPLETE_READINESS_NO_CLAIM_NO_FIT",
            "static": {
                "status": static["status"],
                "runner_normalized_sha256": static["runner_normalized_sha256"],
                "project_dependency_count": static["project_dependency_count"],
                "project_dependency_sha256": {
                    key: value["actual_read_sha256"]
                    for key, value in static["project_dependency_receipts"].items()
                },
                "closure": static["closure"],
                "trust_firewall_v5_sha256": static["trust_firewall_v5_sha256"],
                "superseded_v1_preserved": True,
            },
            "readiness": selected,
            "snapshot_static_inventory": {
                relative: {
                    "bytes": int(record["bytes"]),
                    "sha256": str(record["sha256"]),
                }
                for relative, record in sorted(records.items())
                if relative
                not in {
                    _relative_literal(AUTHORIZATION_PATH),
                    _relative_literal(SEAL_PATH),
                    _relative_literal(R5_QA_PATH),
                }
            },
            "namespace": {
                "clean_except_preexecution_seal": _namespace_clean_for_claim(),
                "claim_created": False,
            },
            "operation_counters": {
                "claims": 0,
                "physical_fits": 0,
                "scientific_materializations": 0,
                "outer_scores": 0,
                "candidate_files": 0,
                "official_test_reads": 0,
                "sample_format_reads": 0,
                "submission_candidate_reads": 0,
                "uploads": 0,
            },
        }
        result = _attach_stable_verification_and_live_authorization(result)
        if retain_snapshot:
            return result, snapshot_root, records
        _remove_snapshot_modules(snapshot_root)
        _cleanup_snapshot(snapshot_root)
        return result, None, None
    except BaseException:
        _remove_snapshot_modules(snapshot_root)
        _cleanup_snapshot(snapshot_root)
        raise


def _namespace_clean_for_claim() -> bool:
    if not ARTIFACT_DIR.exists():
        return True
    allowed = {
        SUPERSEDED_SEAL_R1_PATH.name,
        SUPERSEDED_SEAL_R2_PATH.name,
        SUPERSEDED_SEAL_R3_PATH.name,
        SUPERSEDED_SEAL_R4_PATH.name,
        SEAL_PATH.name,
    }
    return {path.name for path in ARTIFACT_DIR.iterdir()} <= allowed


class AttemptJournal:
    """Create-only hash-chained ledger for one exact 72-fit attempt."""

    def __init__(
        self,
        artifact: Path,
        lock_path: Path,
        lock_descriptor: int,
        journal_dir: Path,
        attempt_id: str,
        deadline_epoch: float,
    ) -> None:
        self.artifact = artifact
        self.lock_path = lock_path
        self.lock_descriptor = lock_descriptor
        self.journal_dir = journal_dir
        self.attempt_id = attempt_id
        self.deadline_epoch = deadline_epoch
        self.fit_reservations = 0
        self.fits_completed = 0
        self.materializations = 0
        self._selected_outer_cell: str | None = None
        self._entry_hashes: dict[str, str] = {}
        self._last_sha256: str | None = None
        self._sequence = 0

    @classmethod
    def begin(
        cls,
        artifact: Path,
        deadline_epoch: float,
        *,
        snapshot_manifest_sha256: str = "TEST_ONLY",
    ) -> AttemptJournal:
        if time.time() >= deadline_epoch:
            raise TimeoutError("deadline expired before execution claim")
        artifact.mkdir(parents=True, exist_ok=True)
        lock_path = artifact / "execution.lock"
        journal_dir = artifact / "attempt_journal"
        initialization_failure = artifact / "initialization_failed.json"
        if journal_dir.exists() or initialization_failure.exists():
            raise FileExistsError("lifetime attempt state already exists before claim")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        attempt_id = uuid.uuid4().hex
        created_journal = False
        phase = "LOCK_PAYLOAD_WRITE"
        try:
            payload = _json_bytes(
                {
                    "schema_version": "p1_segment_rescore.execution_lock.v2",
                    "experiment_id": EXPERIMENT_ID,
                    "attempt_id": attempt_id,
                    "pid": os.getpid(),
                    "deadline_epoch": deadline_epoch,
                    "created_at_kst": _now_kst(),
                }
            )
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("short execution-lock write")
            phase = "LOCK_PAYLOAD_FSYNC"
            os.fsync(descriptor)
            phase = "LOCK_DIRECTORY_FSYNC"
            _fsync_directory(artifact)
            phase = "JOURNAL_CREATE"
            if journal_dir.exists():
                raise FileExistsError("attempt journal appeared after claim")
            os.mkdir(journal_dir)
            created_journal = True
            phase = "JOURNAL_DIRECTORY_FSYNC"
            _fsync_directory(artifact)
            journal = cls(
                artifact,
                lock_path,
                descriptor,
                journal_dir,
                attempt_id,
                deadline_epoch,
            )
            journal._entry(
                "started",
                {
                    "schema_version": "p1_segment_rescore.attempt_started.v2",
                    "snapshot_manifest_sha256": snapshot_manifest_sha256,
                    "maximum_lifetime_physical_fits": 72,
                    "maximum_scientific_materializations": 21,
                    "created_at_kst": _now_kst(),
                },
            )
            return journal
        except BaseException as error:
            try:
                target = (
                    journal_dir / "0997_failed.json"
                    if created_journal and journal_dir.is_dir()
                    else initialization_failure
                )
                previous_entry_sha256 = None
                if target.parent == journal_dir:
                    prefix = sorted(journal_dir.glob("*.json"))
                    if prefix:
                        previous_entry_sha256 = _sha256(prefix[-1])
                _atomic_create_json(
                    target,
                    {
                        "schema_version": "p1_segment_rescore.initialization_failed.v2",
                        "experiment_id": EXPERIMENT_ID,
                        "attempt_id": attempt_id,
                        "status": "FAILED_INITIALIZATION_LOCK_RETAINED",
                        "phase": phase,
                        "error": _sanitize_error(error),
                        "fit_reservations": 0,
                        "fits_completed": 0,
                        "scientific_materializations": 0,
                        "execution_lock_retained": True,
                        "previous_entry_sha256": previous_entry_sha256,
                        "created_at_kst": _now_kst(),
                    },
                )
                _fsync_directory(target.parent)
                _fsync_directory(artifact)
            finally:
                os.close(descriptor)
            raise

    def _verify_entries(self) -> None:
        names = {path.name for path in self.journal_dir.glob("*.json")}
        if names != set(self._entry_hashes):
            raise RuntimeError("attempt journal membership changed")
        for name, expected in self._entry_hashes.items():
            if _sha256(self.journal_dir / name) != expected:
                raise RuntimeError(f"attempt journal entry changed: {name}")

    def _entry(self, kind: str, value: Mapping[str, Any]) -> Path:
        self._verify_entries()
        self._sequence += 1
        name = f"{self._sequence:04d}_{kind}.json"
        record = dict(value)
        record.update(
            {
                "attempt_id": self.attempt_id,
                "sequence": self._sequence,
                "previous_entry_sha256": self._last_sha256,
            }
        )
        path = _atomic_create_json(self.journal_dir / name, record)
        digest = _sha256(path)
        self._entry_hashes[name] = digest
        self._last_sha256 = digest
        return path

    def record_readiness(self, readiness: Mapping[str, Any]) -> None:
        if self.fit_reservations or self.materializations:
            raise RuntimeError("readiness must precede every scientific operation")
        self._entry(
            "readiness",
            {
                "schema_version": "p1_segment_rescore.readiness.v2",
                "status": readiness["status"],
                "feature_binding_sha256": _canonical_sha(readiness["full_feature_cache_binding"]),
                "exact_round_b_equivalence": readiness["exact_round_b_equivalence"],
                "left_censor_counts": readiness[
                    "left_censored_positive_connected_event_count_by_fold"
                ],
                "physical_fits_before_receipt": 0,
                "scientific_materializations_before_receipt": 0,
                "created_at_kst": _now_kst(),
            },
        )

    def _expected_fit(self, ordinal: int) -> tuple[str, str, str | None, int]:
        if ordinal <= 9:
            zero = ordinal - 1
            return (
                "INNER_ANCHOR",
                INNER_WINDOW_IDS[zero // 3],
                "ROUND_B_SHARED",
                ROUND_B_SEEDS[zero % 3],
            )
        if ordinal <= 63:
            zero = ordinal - 10
            per_window = len(STRUCTURE_CELL_IDS) * len(SEGMENT_SEEDS)
            window = INNER_WINDOW_IDS[zero // per_window]
            within = zero % per_window
            cell = STRUCTURE_CELL_IDS[within // len(SEGMENT_SEEDS)]
            seed = SEGMENT_SEEDS[within % len(SEGMENT_SEEDS)]
            return "INNER_SEGMENT", window, cell, seed
        zero = ordinal - 64
        return (
            "OUTER_SEGMENT",
            FOLD_ORDER[zero // len(SEGMENT_SEEDS)],
            None,
            SEGMENT_SEEDS[zero % len(SEGMENT_SEEDS)],
        )

    def reserve_fit(self, phase: str, window: str, cell: str, seed: int) -> int:
        if time.time() >= self.deadline_epoch:
            raise TimeoutError("deadline expired before physical fit reservation")
        if self.fit_reservations >= MAXIMUM_LIFETIME_PHYSICAL_FITS:
            raise RuntimeError("72-fit lifetime ceiling would be exceeded")
        ordinal = self.fit_reservations + 1
        expected_phase, expected_window, expected_cell, expected_seed = self._expected_fit(ordinal)
        if ordinal >= 64:
            if cell not in STRUCTURE_CELL_IDS:
                raise RuntimeError("outer selected cell is not preregistered")
            if self._selected_outer_cell is None:
                self._selected_outer_cell = cell
            expected_cell = self._selected_outer_cell
        observed = (phase, window, cell, int(seed))
        if observed != (expected_phase, expected_window, expected_cell, expected_seed):
            raise RuntimeError(f"fit reservation differs from frozen plan at slot {ordinal}")
        self._entry(
            "fit_reserved",
            {
                "schema_version": "p1_segment_rescore.fit_reserved.v2",
                "ordinal": ordinal,
                "phase": phase,
                "window_or_fold": window,
                "cell": cell,
                "seed": int(seed),
                "reserved_at_epoch": time.time(),
                "created_at_kst": _now_kst(),
            },
        )
        self.fit_reservations = ordinal
        return ordinal

    def complete_fit(self, ordinal: int) -> None:
        if ordinal != self.fits_completed + 1 or ordinal > self.fit_reservations:
            raise RuntimeError("fit completion order differs from reservation order")
        self._entry(
            "fit_completed",
            {
                "schema_version": "p1_segment_rescore.fit_completed.v2",
                "ordinal": ordinal,
                "created_at_kst": _now_kst(),
            },
        )
        self.fits_completed = ordinal

    def _expected_materialization_label(self, ordinal: int) -> re.Pattern[str]:
        if ordinal <= 3:
            window = re.escape(INNER_WINDOW_IDS[ordinal - 1])
            return re.compile(rf"^inner_anchor_surface:{window}$")
        if ordinal <= 12:
            zero = ordinal - 4
            window = re.escape(INNER_WINDOW_IDS[zero // 3])
            bank = re.escape(CONTEXT_BANK_IDS[zero % 3])
            return re.compile(rf"^inner_context_surface:{window}:{bank}$")
        zero = ordinal - 13
        fold = re.escape(FOLD_ORDER[zero // 3])
        bank = re.escape(CONTEXT_BANK_IDS[zero % 3])
        return re.compile(rf"^outer_context_surface:{fold}:{bank}$")

    def reserve_materialization(self, label: str) -> int:
        if time.time() >= self.deadline_epoch:
            raise TimeoutError("deadline expired before scientific materialization")
        if self.materializations >= MAXIMUM_SCIENTIFIC_MATERIALIZATIONS:
            raise RuntimeError("21-materialization lifetime ceiling would be exceeded")
        ordinal = self.materializations + 1
        if self._expected_materialization_label(ordinal).fullmatch(label) is None:
            raise RuntimeError(f"scientific materialization differs from plan at slot {ordinal}")
        self._entry(
            "materialization_reserved",
            {
                "schema_version": "p1_segment_rescore.materialization_reserved.v2",
                "ordinal": ordinal,
                "label": label,
                "created_at_kst": _now_kst(),
            },
        )
        self.materializations = ordinal
        return ordinal

    def record_outer_freeze(self, freeze: Mapping[str, Any]) -> None:
        if self.fit_reservations != 72 or self.fits_completed != 72 or self.materializations != 21:
            raise RuntimeError("outer freeze requires completed 72-fit/21-materialization plan")
        self._entry(
            "outer_predictions_frozen",
            {
                "schema_version": "p1_segment_rescore.outer_freeze.v2",
                "status": "ALL_OUTER_PREDICTIONS_FROZEN_BEFORE_SCORE",
                "freeze": dict(freeze),
                "outer_scores_before_receipt": 0,
                "created_at_kst": _now_kst(),
            },
        )

    def record_aggregate(self, result: Mapping[str, Any]) -> None:
        self._entry(
            "aggregate_scored",
            {
                "schema_version": "p1_segment_rescore.aggregate_scored.v2",
                "result_payload_sha256": _canonical_sha(result),
                "outer_scores": 1,
                "physical_fit_reservations": self.fit_reservations,
                "physical_fits_completed": self.fits_completed,
                "scientific_materializations": self.materializations,
                "created_at_kst": _now_kst(),
            },
        )

    def terminal_entry(
        self,
        name: str,
        value: Mapping[str, Any],
    ) -> Path:
        self._verify_entries()
        record = dict(value)
        record.update(
            {
                "attempt_id": self.attempt_id,
                "previous_entry_sha256": self._last_sha256,
            }
        )
        path = _atomic_create_json(self.journal_dir / name, record)
        digest = _sha256(path)
        self._entry_hashes[name] = digest
        self._last_sha256 = digest
        return path

    def fail_terminal(
        self,
        phase: str,
        error: BaseException,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        failed = [
            path
            for path in self.journal_dir.glob("*.json")
            if path.name in {"0997_failed.json", "0999_failed.json"}
        ]
        if len(failed) == 1:
            self.close_handle_keep_lock()
            return failed[0]
        if failed:
            raise RuntimeError("multiple failure terminals exist")
        name = (
            "0999_failed.json"
            if (self.journal_dir / "0998_worker_terminal.json").exists()
            else "0997_failed.json"
        )
        path = self.terminal_entry(
            name,
            {
                "schema_version": "p1_segment_rescore.failed_terminal.v2",
                "status": "FAILED_FAIL_CLOSED_LOCK_RETAINED",
                "failure_actor": "worker_or_parent",
                "phase": phase,
                "error": _sanitize_error(error),
                "failure_provenance": dict(provenance or {}),
                "fit_reservations": self.fit_reservations,
                "fits_completed": self.fits_completed,
                "scientific_materializations": self.materializations,
                "journal_prefix": {
                    "entry_count": len(self._entry_hashes),
                    "last_entry_sha256": self._last_sha256,
                },
                "execution_lock_retained": True,
                "created_at_kst": _now_kst(),
            },
        )
        self.close_handle_keep_lock()
        return path

    def close_handle_keep_lock(self) -> None:
        if self.lock_descriptor >= 0:
            os.close(self.lock_descriptor)
            self.lock_descriptor = -1

    def manifest_records(self) -> dict[str, dict[str, Any]]:
        self._verify_entries()
        return {
            _relative(self.journal_dir / name): {
                "bytes": (self.journal_dir / name).stat().st_size,
                "sha256": digest,
            }
            for name, digest in self._entry_hashes.items()
        }


def _require_execution_authorization_preimport() -> dict[str, Any]:
    """Authenticate the acyclic external capability from single held bytes."""

    supplied = os.environ.get(AUTHORIZATION_ENV_VAR, "")
    if not re.fullmatch(r"[0-9a-f]{64}", supplied):
        raise AuthorizationError("external authorization digest is absent or malformed")
    raw, receipt = _read_bound_bytes(
        AUTHORIZATION_PATH,
        expected_sha256=supplied,
    )
    authorization = _json_from_bytes(raw, label=AUTHORIZATION_PATH.name)
    if set(authorization) != AUTHORIZATION_REQUIRED_KEYS:
        raise AuthorizationError("authorization schema membership changed")
    if (
        authorization["experiment_id"] != EXPERIMENT_ID
        or authorization["status"] != "AUTHORIZED_INDEPENDENT_QA_PASS_ONE_SHOT"
        or authorization["authorized"] is not True
        or authorization["contract_sha256"] != CONFIG_SHA256
        or authorization["trust_firewall_v5_sha256"] != TRUST_FIREWALL_V5_SHA256
        or authorization["runner_sha256"] != _sha256(Path(__file__).resolve())
        or authorization["runner_normalized_sha256"] != _normalised_runner_sha256()
    ):
        raise AuthorizationError("authorization identity/lineage changed")
    seal_spec = authorization["preexecution_seal"]
    if seal_spec.get("path") != _relative(SEAL_PATH):
        raise AuthorizationError("authorization seal path changed")
    seal_raw, seal_read = _read_bound_bytes(
        SEAL_PATH,
        expected_sha256=str(seal_spec["sha256"]),
    )
    seal = _json_from_bytes(seal_raw, label=SEAL_PATH.name)
    if int(seal_spec["bytes"]) != len(seal_raw):
        raise AuthorizationError("authorization seal byte count changed")
    _verify_seal_value(seal, seal_read)
    qa_spec = authorization["independent_qa"]
    if qa_spec.get("path") != _relative(R5_QA_PATH):
        raise AuthorizationError("authorization QA path changed")
    qa_path = R5_QA_PATH
    qa_raw, qa_read = _read_bound_bytes(
        qa_path,
        expected_sha256=str(qa_spec["sha256"]),
    )
    if len(qa_raw) != int(qa_spec["bytes"]):
        raise AuthorizationError("independent QA byte count changed")
    qa = _json_from_bytes(qa_raw, label=qa_path.name)
    if (
        qa_spec.get("verdict") != "PASS"
        or qa.get("verdict") != "PASS"
        or qa.get("experiment_id") != EXPERIMENT_ID
        or qa.get("preexecution_seal_sha256") != seal_read["actual_read_sha256"]
        or qa.get("contract_sha256") != CONFIG_SHA256
        or qa.get("trust_firewall_v5_sha256") != TRUST_FIREWALL_V5_SHA256
        or qa.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or qa.get("execution_module_sha256") != EXECUTION_MODULE_SHA256
    ):
        raise AuthorizationError("independent QA proof is not a matching PASS")
    if authorization["readonly_preflight_verification_sha256"] != seal.get(
        "readonly_preflight_verification_sha256"
    ):
        raise AuthorizationError("authorization preflight proof changed")
    if authorization["zero_prior_state"] != {
        "claims": 0,
        "physical_fits": 0,
        "scientific_materializations": 0,
        "outer_scores": 0,
        "candidate_files": 0,
    }:
        raise AuthorizationError("authorization zero-prior-state proof changed")
    if authorization["operation_authorization"] != {
        "single_attempt": True,
        "maximum_lifetime_physical_fits": 72,
        "maximum_scientific_materializations": 21,
        "outer_scores": 1,
        "candidate_files": 0,
        "uploads": 0,
    }:
        raise AuthorizationError("authorization operation ceiling changed")

    held_authorities: dict[str, dict[str, Any]] = {
        _relative(AUTHORIZATION_PATH): {"raw": raw, "receipt": receipt},
        _relative(SEAL_PATH): {"raw": seal_raw, "receipt": seal_read},
        _relative(qa_path): {"raw": qa_raw, "receipt": qa_read},
    }
    for path, expected in (
        (DESIGN_PATH, DESIGN_SHA256),
        (EXECUTION_MODULE_PATH, EXECUTION_MODULE_SHA256),
        (TRUST_FIREWALL_V5_PATH, TRUST_FIREWALL_V5_SHA256),
    ):
        held_raw, held_receipt = _read_bound_bytes(path, expected_sha256=expected)
        held_authorities[_relative(path)] = {
            "raw": held_raw,
            "receipt": held_receipt,
        }
    return {
        "value": authorization,
        "read_receipt": receipt,
        "external_authorization_sha256": supplied,
        "seal": seal,
        "seal_read_receipt": seal_read,
        "qa": qa,
        "qa_read_receipt": qa_read,
        "held_authorities": held_authorities,
    }


def _verify_seal_value(
    seal: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if seal.get("status") != ("SEALED_STRICT_ZERO_FIT_NOT_AUTHORIZED_PENDING_INDEPENDENT_QA"):
        raise RuntimeError("preexecution seal status changed")
    checks = {
        "contract_sha256": CONFIG_SHA256,
        "trust_firewall_v5_sha256": TRUST_FIREWALL_V5_SHA256,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "runner_normalized_sha256": _normalised_runner_sha256(),
        "execution_module_sha256": EXECUTION_MODULE_SHA256,
        "test_sha256": _sha256(TEST_PATH),
        "authorization_template_sha256": _sha256(AUTHORIZATION_TEMPLATE_PATH),
    }
    for name, expected in checks.items():
        if seal.get(name) != expected:
            raise RuntimeError(f"preexecution seal binding changed: {name}")
    if seal.get("project_file_sha256") != PROJECT_FILE_SHA256:
        raise RuntimeError("preexecution transitive project map changed")
    return {"value": dict(seal), "read_receipt": dict(receipt)}


def _verify_seal() -> dict[str, Any]:
    seal, receipt = _read_bound_json(SEAL_PATH)
    return _verify_seal_value(seal, receipt)


def seal() -> Path:
    if os.environ.get(AUTHORIZATION_ENV_VAR):
        raise AuthorizationError("external authorization capability must be absent while sealing")
    if SEAL_PATH.exists():
        raise FileExistsError("preexecution seal already exists")
    if AUTHORIZATION_PATH.exists():
        raise RuntimeError("unexpected actual authorization exists before seal")
    if not AUTHORIZATION_TEMPLATE_PATH.is_file():
        raise RuntimeError("authorization false template is missing")
    static = _static_package_checks()
    preflight, _snapshot, _records = _complete_readiness(retain_snapshot=False)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _fsync_directory(ARTIFACT_DIR.parent)
    receipt = {
        "schema_version": "p1_long_event_segment_proposal_rescore.preexecution_seal.v2",
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "status": "SEALED_STRICT_ZERO_FIT_NOT_AUTHORIZED_PENDING_INDEPENDENT_QA",
        "sealed_at_kst": _now_kst(),
        "contract_sha256": CONFIG_SHA256,
        "design_sha256": DESIGN_SHA256,
        "trigger_resolution_sha256": TRIGGER_SHA256,
        "operational_amendment_v2_sha256": AMENDMENT_V2_SHA256,
        "execution_closure_v3_sha256": CLOSURE_V3_SHA256,
        "trust_firewall_v5_sha256": TRUST_FIREWALL_V5_SHA256,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "runner_normalized_sha256": _normalised_runner_sha256(),
        "execution_module_sha256": EXECUTION_MODULE_SHA256,
        "test_sha256": _sha256(TEST_PATH),
        "authorization_template_sha256": _sha256(AUTHORIZATION_TEMPLATE_PATH),
        "project_file_sha256": dict(PROJECT_FILE_SHA256),
        "project_dependency_count": len(PROJECT_FILE_SHA256),
        "runtime": preflight["readiness"]["runtime"],
        "immutable_input_receipts": preflight["readiness"]["held_snapshot_input_reads"],
        "snapshot_static_inventory": preflight["snapshot_static_inventory"],
        "readonly_preflight_verification_sha256": preflight["verification_sha256"],
        "closure_receipt": static["closure"],
        "supersedes": {
            "preexecution_seal_r4_sha256": "c00fc68ff68e5d3883bf892d05312c6962567b4b340f288c57e9dc2aab6dde68",
            "preexecution_seal_r4_status": "IMMUTABLE_NO_GO_P0_ACYCLIC_AUTH_AND_PREFLIGHT_TRANSITION_P1_HELD_SEAL_INVENTORY_AND_OUTER_FIREWALL_P2_POSTCOMMIT_RECOVERY",
            "preexecution_seal_r3_sha256": "c28d28d1dacfa0821ab634a8522bf368aff9d41c835ba770be5868342b458350",
            "preexecution_seal_r3_status": "SUPERSEDED_NO_GO_BEFORE_INDEPENDENT_QA_FULL_SUPPORT_ANCHOR_SURFACE_WAS_SENTINEL_FILLED_OUTSIDE_CENTRAL_SHELF",
            "preexecution_seal_r2_sha256": "e3c08f01b4725c51785146c1c48de11cae331410184bfb1d51860679b5876316",
            "preexecution_seal_r2_status": "SUPERSEDED_BEFORE_INDEPENDENT_QA_TO_ADD_PARENT_CRASH_TERMINAL_AND_INITIALIZATION_HASH_CHAIN_CLOSURE",
            "preexecution_seal_r1_sha256": "e3321d32bf3ffc0290b722e6251fb0d86366bd2f5bca05151df27ad457e6a911",
            "preexecution_seal_r1_status": "SUPERSEDED_BEFORE_INDEPENDENT_QA_AFTER_TEST_HARNESS_DISCOVERED_IN_PROCESS_RUNTIME_CONTAMINATION",
            "v1_runner_sha256": "2c1bacac8039fec9b5370cddf452fac1516bb4d014e6ebd9e08e683688b5faba",
            "v1_module_sha256": "68aaf1fae9902b386b024500b5873328af5b0dddaf9d99243d799d68aa5c5feb",
            "v1_seal_sha256": "ebd54c60e2953ec66956d94a632f7b949dba321cdddbab3640dd0ce058d1c03e",
            "v1_preflight_sha256": "ffafcb55531964d463e836ecb71a339c8d7d0c0c14b8f3533fc8dda97b6f2103",
            "reason": "P0_NONEXECUTABLE_WORKER_AND_UNTRUSTED_SELF_SUPPLIED_AUTH_MAPS",
        },
        "fixed_operation_graph": {
            "inner_anchor_fits": 9,
            "inner_segment_fits": 54,
            "outer_segment_fits": 9,
            "maximum_lifetime_physical_fits": 72,
            "maximum_scientific_materializations": 21,
        },
        "operation_counters_at_seal": {
            "claims": 0,
            "physical_fits": 0,
            "scientific_materializations": 0,
            "outer_scores": 0,
            "candidate_files": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
            "uploads": 0,
        },
    }
    return _atomic_create_json(SEAL_PATH, receipt)


def read_only_preflight() -> dict[str, Any]:
    preflight, _snapshot, _records = _complete_readiness(retain_snapshot=False)
    if SEAL_PATH.exists():
        seal_receipt = _verify_seal()
        expected = seal_receipt["value"]["readonly_preflight_verification_sha256"]
        if preflight["verification_sha256"] != expected:
            raise RuntimeError("read-only preflight differs from sealed verification")
        preflight["seal_sha256"] = seal_receipt["read_receipt"]["actual_read_sha256"]
    else:
        preflight["seal_sha256"] = None
    return preflight


def _snapshot_manifest(
    snapshot_root: Path,
    records: Mapping[str, Mapping[str, Any]],
    readiness: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> Path:
    manifest = {
        "schema_version": "p1_segment_rescore.private_snapshot.v2",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_PRIVATE_SNAPSHOT_READY_FOR_WORKER_REVERIFY",
        "authorization_sha256": authorization["external_authorization_sha256"],
        "contract_sha256": CONFIG_SHA256,
        "trust_firewall_v5_sha256": TRUST_FIREWALL_V5_SHA256,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "runner_normalized_sha256": _normalised_runner_sha256(),
        "seal_sha256": authorization["seal_read_receipt"]["actual_read_sha256"],
        "files": dict(records),
        "strict_readiness": readiness,
    }
    path = snapshot_root / "snapshot_manifest.json"
    _atomic_create_json(path, manifest)
    return path


def _verify_exact_snapshot_inventory(
    root: Path,
    entries: Mapping[str, Mapping[str, Any]],
    expected_inventory: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(entries) != set(expected_inventory):
        raise RuntimeError("snapshot inventory membership differs from sealed exact set")
    expected_directories: set[str] = set()
    for relative in expected_inventory:
        parts = Path(relative).parts
        expected_directories.update(
            Path(*parts[:index]).as_posix() for index in range(1, len(parts))
        )
    actual_files: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        is_junction = bool(getattr(candidate, "is_junction", lambda: False)())
        if candidate.is_symlink() or is_junction:
            raise RuntimeError(f"snapshot tree contains a link/reparse point: {relative}")
        if candidate.is_dir():
            if candidate.name == "__pycache__":
                raise RuntimeError(f"snapshot tree contains bytecode directory: {relative}")
            if relative not in expected_directories:
                raise RuntimeError(f"snapshot tree contains undeclared directory: {relative}")
            continue
        if not candidate.is_file():
            raise RuntimeError(f"snapshot tree contains a non-regular entry: {relative}")
        if relative == "snapshot_manifest.json":
            continue
        if candidate.suffix.lower() in {".pyc", ".pyo"}:
            raise RuntimeError(f"snapshot tree contains bytecode: {relative}")
        actual_files.add(relative)
    if actual_files != set(expected_inventory):
        missing = sorted(set(expected_inventory) - actual_files)
        extra = sorted(actual_files - set(expected_inventory))
        raise RuntimeError(
            f"actual snapshot tree differs from sealed exact set: missing={missing}, extra={extra}"
        )
    executable_pins = {
        _relative_literal(LEGACY_RUNNER_PATH): LEGACY_RUNNER_SHA256,
        _relative_literal(EXECUTION_MODULE_PATH): EXECUTION_MODULE_SHA256,
        _relative_literal(DESIGN_PATH): DESIGN_SHA256,
        _relative_literal(TRUST_FIREWALL_V5_PATH): TRUST_FIREWALL_V5_SHA256,
    }
    for relative, expected in executable_pins.items():
        if expected_inventory.get(relative, {}).get("sha256") != expected:
            raise RuntimeError(f"snapshot executable pin changed: {relative}")
    for relative, record in expected_inventory.items():
        declared = entries[relative]
        if int(declared["bytes"]) != int(record["bytes"]) or declared["sha256"] != record["sha256"]:
            raise RuntimeError(f"snapshot declared inventory changed: {relative}")
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root):
            raise RuntimeError("snapshot path escapes private root")
        if candidate.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"snapshot byte count changed: {relative}")
        if _sha256(candidate) != record["sha256"]:
            raise RuntimeError(f"snapshot digest changed: {relative}")


def _verify_snapshot_manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest, _receipt = _read_bound_json(path, expected_sha256=expected_sha256)
    root = path.parent.resolve(strict=True)
    supplied = os.environ.get(AUTHORIZATION_ENV_VAR, "")
    if not re.fullmatch(r"[0-9a-f]{64}", supplied):
        raise AuthorizationError("worker external authorization digest is absent or malformed")
    checks = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_PRIVATE_SNAPSHOT_READY_FOR_WORKER_REVERIFY",
        "authorization_sha256": supplied,
        "contract_sha256": CONFIG_SHA256,
        "trust_firewall_v5_sha256": TRUST_FIREWALL_V5_SHA256,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "runner_normalized_sha256": _normalised_runner_sha256(),
    }
    for name, expected in checks.items():
        if manifest.get(name) != expected:
            raise RuntimeError(f"snapshot manifest binding changed: {name}")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("snapshot manifest inventory is empty")
    auth_relative = _relative_literal(AUTHORIZATION_PATH)
    seal_relative = _relative_literal(SEAL_PATH)
    qa_relative = _relative_literal(R5_QA_PATH)
    authority_required = {auth_relative, seal_relative, qa_relative}
    if not authority_required.issubset(entries):
        raise RuntimeError("snapshot authority inventory is incomplete")

    auth_raw, auth_receipt = _read_bound_bytes(
        root / auth_relative,
        expected_sha256=supplied,
    )
    authorization = _json_from_bytes(auth_raw, label=AUTHORIZATION_PATH.name)
    if set(authorization) != AUTHORIZATION_REQUIRED_KEYS:
        raise AuthorizationError("snapshot authorization schema membership changed")
    seal_spec = authorization.get("preexecution_seal", {})
    if seal_spec.get("path") != seal_relative:
        raise AuthorizationError("snapshot authorization seal path changed")
    seal_raw, seal_receipt = _read_bound_bytes(
        root / seal_relative,
        expected_sha256=str(seal_spec.get("sha256")),
    )
    seal = _json_from_bytes(seal_raw, label=SEAL_PATH.name)
    _verify_seal_value(seal, seal_receipt)
    qa_spec = authorization.get("independent_qa", {})
    if qa_spec.get("path") != qa_relative:
        raise AuthorizationError("snapshot authorization QA path changed")
    qa_raw, qa_receipt = _read_bound_bytes(
        root / qa_relative,
        expected_sha256=str(qa_spec.get("sha256")),
    )
    qa = _json_from_bytes(qa_raw, label=R5_QA_PATH.name)
    if (
        authorization.get("experiment_id") != EXPERIMENT_ID
        or int(seal_spec.get("bytes", -1)) != len(seal_raw)
        or authorization.get("status") != "AUTHORIZED_INDEPENDENT_QA_PASS_ONE_SHOT"
        or authorization.get("authorized") is not True
        or authorization.get("contract_sha256") != CONFIG_SHA256
        or authorization.get("trust_firewall_v5_sha256") != TRUST_FIREWALL_V5_SHA256
        or authorization.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or authorization.get("runner_normalized_sha256") != _normalised_runner_sha256()
        or authorization.get("readonly_preflight_verification_sha256")
        != seal.get("readonly_preflight_verification_sha256")
        or qa_spec.get("verdict") != "PASS"
        or int(qa_spec.get("bytes", -1)) != len(qa_raw)
        or qa.get("verdict") != "PASS"
        or qa.get("preexecution_seal_sha256") != seal_receipt["actual_read_sha256"]
        or qa.get("contract_sha256") != CONFIG_SHA256
        or qa.get("trust_firewall_v5_sha256") != TRUST_FIREWALL_V5_SHA256
        or qa.get("runner_sha256") != _sha256(Path(__file__).resolve())
        or qa.get("execution_module_sha256") != EXECUTION_MODULE_SHA256
    ):
        raise AuthorizationError("snapshot authorization lineage changed")
    if authorization.get("zero_prior_state") != {
        "claims": 0,
        "physical_fits": 0,
        "scientific_materializations": 0,
        "outer_scores": 0,
        "candidate_files": 0,
    } or authorization.get("operation_authorization") != {
        "single_attempt": True,
        "maximum_lifetime_physical_fits": 72,
        "maximum_scientific_materializations": 21,
        "outer_scores": 1,
        "candidate_files": 0,
        "uploads": 0,
    }:
        raise AuthorizationError("snapshot authorization operation boundary changed")

    expected_inventory = dict(seal.get("snapshot_static_inventory", {}))
    expected_inventory.update(
        {
            auth_relative: {
                "bytes": auth_receipt["bytes"],
                "sha256": supplied,
            },
            seal_relative: {
                "bytes": seal_receipt["bytes"],
                "sha256": seal_receipt["actual_read_sha256"],
            },
            qa_relative: {
                "bytes": qa_receipt["bytes"],
                "sha256": qa_receipt["actual_read_sha256"],
            },
        }
    )
    _verify_exact_snapshot_inventory(root, entries, expected_inventory)
    if manifest.get("seal_sha256") != seal_receipt["actual_read_sha256"]:
        raise RuntimeError("snapshot seal digest differs from held authority")
    return (
        root,
        manifest,
        {
            "value": authorization,
            "read_receipt": auth_receipt,
            "external_authorization_sha256": supplied,
            "seal": seal,
            "seal_read_receipt": seal_receipt,
            "qa": qa,
            "qa_read_receipt": qa_receipt,
            "held_authority_records": {
                relative: {
                    "bytes": int(expected_inventory[relative]["bytes"]),
                    "sha256": str(expected_inventory[relative]["sha256"]),
                    "source": "AUTHENTICATED_PRIVATE_SNAPSHOT",
                }
                for relative in (
                    auth_relative,
                    seal_relative,
                    qa_relative,
                    _relative_literal(DESIGN_PATH),
                    _relative_literal(EXECUTION_MODULE_PATH),
                    _relative_literal(TRUST_FIREWALL_V5_PATH),
                )
            },
        },
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_journal_chain(
    journal_dir: Path,
    *,
    required_last: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    paths = sorted(journal_dir.glob("*.json"))
    if not paths:
        raise RuntimeError("attempt journal is empty")
    if required_last is not None and paths[-1].name != required_last:
        raise RuntimeError(f"required journal terminal is missing: {required_last}")
    previous: str | None = None
    values: list[dict[str, Any]] = []
    for path in paths:
        value, receipt = _read_bound_json(path)
        if value.get("previous_entry_sha256") != previous:
            raise RuntimeError(f"journal hash chain mismatch: {path.name}")
        previous = receipt["actual_read_sha256"]
        values.append(value)
    assert previous is not None
    return previous, values


def _record_parent_failure_if_claimed(
    phase: str,
    error: BaseException,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path | None:
    """Publish one hash-chained parent failure only while a claim is held.

    The supervisor proves the worker tree is gone before returning an error, so
    the parent is the sole possible writer here. A missing lock means either no
    claim was made or success crossed the final lock-unlink boundary; both
    cases remain strictly read-only. Every recorded failure retains the lock.
    """

    lock_path = ARTIFACT_DIR / "execution.lock"
    if not lock_path.is_file():
        return None

    journal_dir = ARTIFACT_DIR / "attempt_journal"
    initialization_failure = ARTIFACT_DIR / "initialization_failed.json"
    if not journal_dir.is_dir():
        if initialization_failure.is_file():
            existing, _receipt = _read_bound_json(initialization_failure)
            if existing.get("execution_lock_retained") is not True:
                raise RuntimeError("initialization failure lock policy changed")
            return initialization_failure
        path = _atomic_create_json(
            initialization_failure,
            {
                "schema_version": "p1_segment_rescore.parent_initialization_failed.v2",
                "experiment_id": EXPERIMENT_ID,
                "status": "FAILED_PARENT_AFTER_CLAIM_BEFORE_JOURNAL_LOCK_RETAINED",
                "failure_actor": "parent_supervisor",
                "phase": phase,
                "error": _sanitize_error(error),
                "failure_provenance": dict(provenance or {}),
                "fit_reservations": 0,
                "fits_completed": 0,
                "scientific_materializations": 0,
                "execution_lock_retained": True,
                "created_at_kst": _now_kst(),
            },
        )
        _fsync_directory(ARTIFACT_DIR)
        return path

    paths = sorted(journal_dir.glob("*.json"))
    if not paths:
        lock_value, _lock_receipt = _read_bound_json(lock_path)
        attempt_id = lock_value.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise RuntimeError("empty claimed journal has no bound attempt identity")
        path = _atomic_create_json(
            journal_dir / "0997_failed.json",
            {
                "schema_version": "p1_segment_rescore.parent_failed_terminal.v2",
                "experiment_id": EXPERIMENT_ID,
                "attempt_id": attempt_id,
                "status": "FAILED_PARENT_AFTER_JOURNAL_CREATE_BEFORE_START",
                "failure_actor": "parent_supervisor",
                "phase": phase,
                "error": _sanitize_error(error),
                "failure_provenance": dict(provenance or {}),
                "fit_reservations": 0,
                "fits_completed": 0,
                "scientific_materializations": 0,
                "journal_prefix": {
                    "entry_count": 0,
                    "last_entry_sha256": None,
                },
                "execution_lock_retained": True,
                "previous_entry_sha256": None,
                "created_at_kst": _now_kst(),
            },
        )
        _fsync_directory(journal_dir)
        _fsync_directory(ARTIFACT_DIR)
        return path

    previous, entries = _verify_journal_chain(journal_dir, required_last=None)
    failure_names = {
        "0997_failed.json",
        "0999_failed.json",
        "1000_postcompletion_failed.json",
    }
    existing_failures = [path for path in paths if path.name in failure_names]
    if len(existing_failures) == 1:
        return existing_failures[0]
    if existing_failures:
        raise RuntimeError("multiple claimed failure terminals exist")

    names = {path.name for path in paths}
    if "0999_completed.json" in names:
        terminal_name = "1000_postcompletion_failed.json"
        status = "FAILED_AFTER_COMPLETION_BEFORE_LOCK_RELEASE"
    elif "0998_worker_terminal.json" in names:
        terminal_name = "0999_failed.json"
        status = "FAILED_AFTER_WORKER_OUTPUT_BEFORE_COMPLETION"
    else:
        terminal_name = "0997_failed.json"
        status = "FAILED_PARENT_SUPERVISION_LOCK_RETAINED"

    def count(schema: str) -> int:
        return sum(value.get("schema_version") == schema for value in entries)

    attempt_ids = {
        str(value["attempt_id"]) for value in entries if isinstance(value.get("attempt_id"), str)
    }
    if len(attempt_ids) != 1:
        raise RuntimeError("claimed journal attempt identity is ambiguous")
    path = _atomic_create_json(
        journal_dir / terminal_name,
        {
            "schema_version": "p1_segment_rescore.parent_failed_terminal.v2",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": next(iter(attempt_ids)),
            "status": status,
            "failure_actor": "parent_supervisor",
            "phase": phase,
            "error": _sanitize_error(error),
            "failure_provenance": dict(provenance or {}),
            "fit_reservations": count("p1_segment_rescore.fit_reserved.v2"),
            "fits_completed": count("p1_segment_rescore.fit_completed.v2"),
            "scientific_materializations": count("p1_segment_rescore.materialization_reserved.v2"),
            "journal_prefix": {
                "entry_count": len(entries),
                "last_entry_sha256": previous,
            },
            "execution_lock_retained": True,
            "previous_entry_sha256": previous,
            "created_at_kst": _now_kst(),
        },
    )
    _fsync_directory(journal_dir)
    _fsync_directory(ARTIFACT_DIR)
    return path


def _render_report(result: Mapping[str, Any]) -> bytes:
    metrics = result["metrics"]
    pooled = metrics["pooled"]
    bootstrap = metrics["paired_bootstrap"]
    selected = result["selected_inner_cell"]
    lines = [
        "# P1 장기 이벤트 구간 제안·재채점 고정 실험",
        "",
        f"결론: **{result['decision']}**",
        "",
        "이 보고서는 공식 평가·제출이 아닌 사전등록된 로컬 historical screen이다.",
        "",
        "## 핵심 수치",
        "",
        f"- 후보 F1: {pooled['candidate']['f1']:.9f}",
        f"- Round-B anchor F1: {pooled['anchor']['f1']:.9f}",
        f"- 후보−anchor F1 Δ: {pooled['f1_delta']:+.9f}",
        f"- paired bootstrap 90% CI: {bootstrap['difference_ci90']}",
        f"- 선택 구조: {selected['cell_id']}, threshold={selected['threshold']}",
        f"- RESEARCH_GO: {result['RESEARCH_GO']}",
        f"- SUBMISSION_GO_RESEARCH_ONLY: {result['SUBMISSION_GO_RESEARCH_ONLY']}",
        "",
        "## 고정 실행 계수",
        "",
        "- inner Round-B anchors: 9 fits",
        "- inner segment models: 54 fits",
        "- outer segment models: 9 fits",
        "- total: 72 fits; scientific materializations: 21; outer score: 1",
        "",
        "## 한계",
        "",
        "- 로컬 historical windows와 공식 평가 간 분포 차이는 남는다.",
        "- 통과하더라도 별도 독립 QA 및 명시적 제출 승인이 필요하다.",
        "- 행별 prediction 또는 제출 후보는 생성하지 않았다.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _publish_success(
    screen: Mapping[str, Any],
    journal: AttemptJournal,
    authorization: Mapping[str, Any],
    snapshot_manifest_sha256: str,
) -> Path:
    metrics_path = ARTIFACT_DIR / "metrics.json"
    report_path = ARTIFACT_DIR / "report_ko.md"
    result_path = ARTIFACT_DIR / "result.json"
    manifest_path = ARTIFACT_DIR / "manifest.json"
    result = dict(screen)
    result.update(
        {
            "attempt_id": journal.attempt_id,
            "authorization_sha256": authorization["external_authorization_sha256"],
            "contract_sha256": CONFIG_SHA256,
            "preexecution_seal_sha256": authorization["seal_read_receipt"]["actual_read_sha256"],
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "completed_at_kst": _now_kst(),
        }
    )
    journal.record_aggregate(result)
    _atomic_create_json(metrics_path, result["metrics"])
    _atomic_create_bytes(report_path, _render_report(result))
    _atomic_create_json(result_path, result)
    inventory = {
        _relative(path): _file_record(path)
        for path in (
            CONFIG_PATH,
            TRIGGER_PATH,
            AMENDMENT_V2_PATH,
            CLOSURE_V3_PATH,
            Path(__file__).resolve(),
            metrics_path,
            report_path,
            result_path,
        )
    }
    inventory.update(journal.manifest_records())
    manifest = {
        "schema_version": "p1_long_event_segment_proposal_rescore.manifest.v2",
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "attempt_id": journal.attempt_id,
        "status": "WORKER_OUTPUTS_COMPLETE_BEFORE_FINAL_COMMIT",
        "authorization_sha256": authorization["external_authorization_sha256"],
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "artifacts": inventory,
        "held_authority_records": authorization["held_authority_records"],
        "operation_counters": result["operation_counters"],
        "forbidden_outputs": {
            "row_level_prediction_files": 0,
            "submission_candidate_files": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
            "uploads": 0,
        },
        "created_at_kst": _now_kst(),
    }
    _atomic_create_json(manifest_path, manifest)
    # Re-read every declared byte before the commit boundary while the claim
    # can still be failed closed.
    for relative, record in manifest["artifacts"].items():
        path = _resolve_repo_path(relative)
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"worker manifest verification failed: {relative}")
    journal.terminal_entry(
        "0998_worker_terminal.json",
        {
            "schema_version": "p1_segment_rescore.worker_terminal.v2",
            "status": "WORKER_SUCCESS_COMMIT_PREPARED",
            "result_sha256": _sha256(result_path),
            "manifest_sha256": _sha256(manifest_path),
            "physical_fit_reservations": journal.fit_reservations,
            "physical_fits_completed": journal.fits_completed,
            "scientific_materializations": journal.materializations,
            "outer_scores": 1,
            "created_at_kst": _now_kst(),
        },
    )
    previous, _entries = _verify_journal_chain(
        journal.journal_dir,
        required_last="0998_worker_terminal.json",
    )
    journal.terminal_entry(
        "0999_completed.json",
        {
            "schema_version": "p1_segment_rescore.completed.v2",
            "status": "SUCCESS_ALL_OUTPUTS_VERIFIED_READY_FOR_LOCK_RELEASE",
            "result_sha256": _sha256(result_path),
            "manifest_sha256": _sha256(manifest_path),
            "worker_terminal_sha256": previous,
            "physical_fit_reservations": 72,
            "physical_fits_completed": 72,
            "scientific_materializations": 21,
            "outer_scores": 1,
            "created_at_kst": _now_kst(),
        },
    )
    _final_success_commit(journal)
    return result_path


def _final_success_commit(journal: AttemptJournal) -> None:
    """Flush under lock; successful unlink is the final filesystem operation.

    Windows has no portable directory-fsync guarantee for the deletion.  The
    completed artifacts and both parent directories are flushed first.  If
    close or unlink fails, the lock remains and the outer failure path records
    a terminal; after a successful unlink there is deliberately no filesystem
    call in this function.
    """

    if journal.fit_reservations != 72 or journal.fits_completed != 72:
        raise RuntimeError("final commit requires exactly 72 completed fits")
    if journal.materializations != 21:
        raise RuntimeError("final commit requires exactly 21 materializations")
    _verify_journal_chain(journal.journal_dir, required_last="0999_completed.json")
    _fsync_directory(journal.journal_dir)
    _fsync_directory(journal.artifact)
    if not journal.lock_path.is_file():
        raise RuntimeError("execution lock disappeared before final commit")
    if journal.lock_descriptor >= 0:
        os.close(journal.lock_descriptor)
        journal.lock_descriptor = -1
    journal.lock_path.unlink()


def _assert_output_namespace_ready() -> None:
    if not ARTIFACT_DIR.is_dir() or not SEAL_PATH.is_file():
        raise RuntimeError("sealed artifact namespace is missing")
    allowed = {
        SUPERSEDED_SEAL_R1_PATH.name,
        SUPERSEDED_SEAL_R2_PATH.name,
        SUPERSEDED_SEAL_R3_PATH.name,
        SUPERSEDED_SEAL_R4_PATH.name,
        SEAL_PATH.name,
    }
    unexpected = {path.name for path in ARTIFACT_DIR.iterdir() if path.name not in allowed}
    if unexpected:
        raise FileExistsError(f"one-shot artifact namespace is not clean: {sorted(unexpected)}")


def _load_worker_state(
    snapshot_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[ModuleType, ModuleType, dict[str, Any], dict[str, Any]]:
    legacy_raw, _receipt = _read_bound_bytes(
        snapshot_root / _relative_literal(LEGACY_CONFIG_PATH),
        expected_sha256=LEGACY_CONFIG_SHA256,
    )
    legacy_config = _json_from_bytes(legacy_raw, label=LEGACY_CONFIG_PATH.name)
    legacy = _load_snapshot_legacy_runner(snapshot_root, "_p1_v1r6_worker")
    numerical, execution, runtime = _load_snapshot_numerical(
        snapshot_root,
        legacy_config,
        legacy,
    )
    readiness, state = _strict_target_free_snapshot_readiness(
        snapshot_root,
        legacy_config,
        numerical,
        runtime,
        legacy,
    )
    selected = _selected_readiness(readiness)
    if selected != manifest["strict_readiness"]:
        raise RuntimeError("worker strict readiness differs from parent receipt")
    return numerical, execution, selected, state


def _worker_execute(
    snapshot_manifest_path: Path,
    snapshot_manifest_sha256: str,
    deadline_epoch: float,
) -> Path:
    snapshot_root, manifest, authorization = _verify_snapshot_manifest(
        snapshot_manifest_path,
        snapshot_manifest_sha256,
    )
    numerical, execution, readiness, state = _load_worker_state(snapshot_root, manifest)
    if time.time() >= deadline_epoch:
        raise TimeoutError("deadline expired before execution claim")
    _assert_output_namespace_ready()
    journal = AttemptJournal.begin(
        ARTIFACT_DIR,
        deadline_epoch,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
    )
    phase = "CLAIM_CREATED"
    try:
        phase = "READINESS_RECONFIRMED"
        journal.record_readiness(readiness)
        phase = "FIXED_72_FIT_NUMERICAL_SCREEN"
        closure_raw, _closure_receipt = _read_bound_bytes(
            snapshot_root / _relative(CLOSURE_V3_PATH),
            expected_sha256=CLOSURE_V3_SHA256,
        )
        closure = _json_from_bytes(closure_raw, label=CLOSURE_V3_PATH.name)
        screen = execution.run_authorized_screen(
            state,
            numerical,
            closure,
            journal,
            deadline_epoch,
        )
        phase = "AGGREGATE_ONLY_SUCCESS_PUBLICATION"
        return _publish_success(
            screen,
            journal,
            authorization,
            snapshot_manifest_sha256,
        )
    except BaseException as error:
        if journal.lock_path.exists():
            try:
                journal.fail_terminal(phase, error, provenance={"snapshot_root": "PRIVATE"})
            except BaseException as terminal_error:
                journal.close_handle_keep_lock()
                raise RuntimeError(
                    "claimed failure terminal could not be published"
                ) from terminal_error
        raise


def _worker_command(
    snapshot_manifest_path: Path,
    snapshot_manifest_sha256: str,
    deadline_epoch: float,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--snapshot-manifest",
        str(snapshot_manifest_path),
        "--snapshot-manifest-sha256",
        snapshot_manifest_sha256,
        "--deadline-epoch",
        repr(deadline_epoch),
    ]


def _run_supervised(
    command: Sequence[str],
    deadline_epoch: float,
    snapshot_root: Path,
) -> tuple[str, str]:
    """Use the pinned v1r6 Windows tree-kill supervisor with a 6-hour deadline."""

    legacy = _load_snapshot_legacy_runner(snapshot_root, "_p1_v1r6_supervisor")
    try:
        return legacy._run_supervised(command, deadline_epoch)
    except BaseException as error:
        receipt = getattr(error, "termination_receipt", None)
        if isinstance(error, TimeoutError) and isinstance(receipt, Mapping):
            raise WorkerTimeoutError(
                "absolute 21600-second timeout terminated the worker tree",
                receipt,
            ) from error
        raise


def _parent_read_only_verify(result_path: Path) -> dict[str, Any]:
    if ARTIFACT_DIR.joinpath("execution.lock").exists():
        raise RuntimeError("success returned while execution lock remains")
    expected_result = ARTIFACT_DIR / "result.json"
    if result_path.resolve(strict=True) != expected_result.resolve(strict=True):
        raise RuntimeError("worker returned an unexpected result path")
    manifest_path = ARTIFACT_DIR / "manifest.json"
    result, result_receipt = _read_bound_json(result_path)
    manifest, manifest_receipt = _read_bound_json(manifest_path)
    previous, entries = _verify_journal_chain(
        ARTIFACT_DIR / "attempt_journal",
        required_last="0999_completed.json",
    )
    completed = entries[-1]
    if completed.get("status") != "SUCCESS_ALL_OUTPUTS_VERIFIED_READY_FOR_LOCK_RELEASE":
        raise RuntimeError("completed terminal status changed")
    if completed.get("result_sha256") != result_receipt["actual_read_sha256"]:
        raise RuntimeError("completed terminal result hash changed")
    if completed.get("manifest_sha256") != manifest_receipt["actual_read_sha256"]:
        raise RuntimeError("completed terminal manifest hash changed")
    if result.get("attempt_id") != manifest.get("attempt_id"):
        raise RuntimeError("result/manifest attempt identity differs")
    held = manifest.get("held_authority_records")
    expected_held_paths = {
        _relative_literal(AUTHORIZATION_PATH),
        _relative_literal(SEAL_PATH),
        _relative_literal(R5_QA_PATH),
        _relative_literal(DESIGN_PATH),
        _relative_literal(EXECUTION_MODULE_PATH),
        _relative_literal(TRUST_FIREWALL_V5_PATH),
    }
    if not isinstance(held, dict) or set(held) != expected_held_paths:
        raise RuntimeError("held authority manifest membership changed")
    held_expected_hashes = {
        _relative_literal(AUTHORIZATION_PATH): result["authorization_sha256"],
        _relative_literal(SEAL_PATH): result["preexecution_seal_sha256"],
        _relative_literal(DESIGN_PATH): DESIGN_SHA256,
        _relative_literal(EXECUTION_MODULE_PATH): EXECUTION_MODULE_SHA256,
        _relative_literal(TRUST_FIREWALL_V5_PATH): TRUST_FIREWALL_V5_SHA256,
    }
    for relative, expected in held_expected_hashes.items():
        if held[relative].get("sha256") != expected:
            raise RuntimeError(f"held authority manifest digest changed: {relative}")
    qa_record = held[_relative_literal(R5_QA_PATH)]
    if not re.fullmatch(r"[0-9a-f]{64}", str(qa_record.get("sha256", ""))) or any(
        record.get("source") != "AUTHENTICATED_PRIVATE_SNAPSHOT" for record in held.values()
    ):
        raise RuntimeError("held QA/source authority receipt changed")
    for relative, record in manifest["artifacts"].items():
        path = _resolve_repo_path(relative)
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"parent manifest verification failed: {relative}")
    reservations = [
        value
        for value in entries
        if value.get("schema_version") == "p1_segment_rescore.fit_reserved.v2"
    ]
    completions = [
        value
        for value in entries
        if value.get("schema_version") == "p1_segment_rescore.fit_completed.v2"
    ]
    materializations = [
        value
        for value in entries
        if value.get("schema_version") == "p1_segment_rescore.materialization_reserved.v2"
    ]
    if len(reservations) != 72 or len(completions) != 72 or len(materializations) != 21:
        raise RuntimeError("parent journal operation accounting changed")
    if result["operation_counters"] != {
        "claims": 1,
        "inner_anchor_physical_fits": 9,
        "inner_segment_physical_fits": 54,
        "outer_segment_physical_fits": 9,
        "physical_fits": 72,
        "scientific_materializations": 21,
        "outer_scores": 1,
        "candidate_files": 0,
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
        "uploads": 0,
    }:
        raise RuntimeError("result operation counters changed")
    return {
        "status": "PASS_PARENT_READ_ONLY_INDEPENDENT_MANIFEST_QA",
        "result_sha256": result_receipt["actual_read_sha256"],
        "manifest_sha256": manifest_receipt["actual_read_sha256"],
        "completed_terminal_sha256": previous,
        "physical_fit_reservations": 72,
        "physical_fits_completed": 72,
        "scientific_materializations": 21,
        "outer_scores": 1,
    }


def _recover_committed_success_after_supervision_failure(
    error: BaseException,
) -> tuple[Path, dict[str, Any]] | None:
    """Recover only a fully committed lock-free success, without any write."""

    if ARTIFACT_DIR.joinpath("execution.lock").exists():
        return None
    result_path = ARTIFACT_DIR / "result.json"
    manifest_path = ARTIFACT_DIR / "manifest.json"
    completed_path = ARTIFACT_DIR / "attempt_journal" / "0999_completed.json"
    exists = [result_path.is_file(), manifest_path.is_file(), completed_path.is_file()]
    if not any(exists):
        return None
    if not all(exists):
        raise RuntimeError("lock-free namespace has partial success evidence") from error
    verification = _parent_read_only_verify(result_path)
    verification = dict(verification)
    verification["status"] = "PASS_RECOVERED_DURABLE_SUCCESS_AFTER_STDOUT_LOSS"
    verification["recovery_writes"] = 0
    verification["supervision_error"] = _sanitize_error(error)
    return result_path, verification


def execute_parent() -> tuple[Path, dict[str, Any]]:
    started = time.time()
    authorization = _require_execution_authorization_preimport()
    for path in (
        AUTHORIZATION_PATH,
        SEAL_PATH,
        R5_QA_PATH,
        DESIGN_PATH,
        EXECUTION_MODULE_PATH,
        TRUST_FIREWALL_V5_PATH,
    ):
        _verify_held_path_identity(
            path,
            authorization["held_authorities"][_relative(path)],
        )
    _static_package_checks(authorization["held_authorities"])
    seal_receipt = {
        "value": authorization["seal"],
        "read_receipt": authorization["seal_read_receipt"],
    }
    preflight, snapshot_root, records = _complete_readiness(
        retain_snapshot=True,
        held_authorities=authorization["held_authorities"],
    )
    assert snapshot_root is not None and records is not None
    if (
        preflight["verification_sha256"]
        != seal_receipt["value"]["readonly_preflight_verification_sha256"]
    ):
        _remove_snapshot_modules(snapshot_root)
        _cleanup_snapshot(snapshot_root)
        raise RuntimeError("execution readiness differs from sealed read-only preflight")
    deadline = started + HARD_WALL_SECONDS
    manifest_path = _snapshot_manifest(
        snapshot_root,
        records,
        preflight["readiness"],
        authorization,
    )
    manifest_sha = _sha256(manifest_path)
    _remove_snapshot_modules(snapshot_root)
    phase = "PARENT_WORKER_SUPERVISION"
    try:
        stdout, _stderr = _run_supervised(
            _worker_command(manifest_path, manifest_sha, deadline),
            deadline,
            snapshot_root,
        )
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("worker returned no receipt")
        receipt = json.loads(lines[-1])
        if receipt.get("status") != "worker_ok":
            raise RuntimeError("worker success receipt changed")
        result_path = Path(str(receipt["result_path"]))
        phase = "PARENT_READ_ONLY_POSTCOMMIT_VERIFICATION"
        verification = _parent_read_only_verify(result_path)
        return result_path, verification
    except BaseException as error:
        # A proved-dead worker cannot publish its own terminal after a hard
        # timeout/crash. The parent records that claimed failure exactly once;
        # success after lock unlink and every pre-claim failure remain read-only.
        if ARTIFACT_DIR.joinpath("execution.lock").exists():
            provenance = getattr(error, "receipt", None) or getattr(
                error, "termination_receipt", None
            )
            try:
                _record_parent_failure_if_claimed(
                    phase,
                    error,
                    provenance=provenance if isinstance(provenance, Mapping) else {},
                )
            except BaseException as terminal_error:
                raise RuntimeError(
                    f"{phase} failed and parent failure terminal publication failed; "
                    "execution lock remains fail-closed"
                ) from terminal_error
            raise RuntimeError(
                f"{phase} failed; worker claim remains fail-closed; "
                f"termination_provenance={provenance}"
            ) from error
        recovered = _recover_committed_success_after_supervision_failure(error)
        if recovered is not None:
            return recovered
        raise
    finally:
        try:
            _cleanup_snapshot(snapshot_root)
        except BaseException:
            # Snapshot cleanup is outside the repository and never reopens a
            # committed repository artifact.  Preserve the primary result.
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--seal", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-epoch", type=float, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seal:
        output: Any = str(seal())
    elif args.preflight:
        output = read_only_preflight()
    elif args.execute:
        path, verification = execute_parent()
        output = {"result_path": str(path), "parent_verification": verification}
    else:
        if (
            args.snapshot_manifest is None
            or args.snapshot_manifest_sha256 is None
            or args.deadline_epoch is None
        ):
            raise RuntimeError("hidden worker requires a complete parent capability")
        result = _worker_execute(
            args.snapshot_manifest.resolve(strict=True),
            str(args.snapshot_manifest_sha256),
            float(args.deadline_epoch),
        )
        print(
            json.dumps(
                {"status": "worker_ok", "result_path": str(result)},
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return
    print(json.dumps({"status": "ok", "output": output}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
