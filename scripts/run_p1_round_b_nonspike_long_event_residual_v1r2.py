"""Fail-closed wrapper for the sealed P1 Round-B residual local screen.

Only Python's standard library is imported until the complete pre-import trust
receipt has been verified.  The numerical implementation is the immutable v1
runner, loaded dynamically after trust succeeds; this wrapper adds no new
model, feature, decoder, scoring, or gate logic.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1r2.json"
)
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
NUMERICAL_SECTIONS = (
    "base_config",
    "immutable_inputs",
    "surface",
    "residual_target",
    "residual_model",
    "rescue_decoder",
    "outer_protocol",
    "fail_fast_gates",
    "resource_budget",
    "interpretation",
    "prohibitions",
)
REQUIRED_DEPENDENCY_PATHS = (
    "src/p1_qc/__init__.py",
    "src/p1_qc/audit.py",
    "src/p1_qc/augment.py",
    "src/p1_qc/config.py",
    "src/p1_qc/data.py",
    "src/p1_qc/experiment.py",
    "src/p1_qc/features.py",
    "src/p1_qc/metrics.py",
    "src/p1_qc/models_tabular.py",
    "src/p1_qc/nonspike_long_event_residual.py",
    "src/p1_qc/pipeline.py",
    "src/p1_qc/postprocess.py",
    "src/p1_qc/rules.py",
    "src/p1_qc/splits.py",
    "src/p1_qc/submission.py",
    "src/p1_qc/validation.py",
    "scripts/run_p1_round_b_nonspike_long_event_residual_v1.py",
)
RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "lightgbm": "lightgbm",
    "scikit-learn": "scikit-learn",
}
REQUIRED_VERIFICATION_PATHS = (
    "tests/test_run_p1_round_b_nonspike_long_event_residual_v1r2.py",
)
FORBIDDEN_PRETRUST_MODULE_ROOTS = {
    "lightgbm",
    "numpy",
    "p1_qc",
    "pandas",
    "pyarrow",
    "sklearn",
}

_TRUST_VERIFIED = False
_NUMERICAL_MODULE: ModuleType | None = None


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return f"{text}\n".encode()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability (Windows may reject directory handles)."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes_new(path, _json_bytes(value))


def _atomic_parquet_new(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.parquet"
    try:
        with temporary.open("xb") as handle:
            frame.to_parquet(handle, index=False, compression="zstd")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_repo_path(value: str, *, must_exist: bool = True) -> Path:
    candidate = PROJECT_ROOT / value
    path = candidate.resolve(strict=must_exist)
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"path escapes repository: {value}")
    return path


def _artifact_dir(config: Mapping[str, Any]) -> Path:
    path = _resolve_repo_path(str(config["artifact_dir"]), must_exist=False)
    root = (PROJECT_ROOT / "artifacts").resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("artifact_dir must remain under artifacts")
    return path


def _assert_pretrust_clean() -> None:
    loaded = {
        name.split(".", maxsplit=1)[0]
        for name in sys.modules
        if name.split(".", maxsplit=1)[0] in FORBIDDEN_PRETRUST_MODULE_ROOTS
    }
    if loaded:
        raise RuntimeError(
            "numerical/project modules were loaded before trust verification: "
            + ", ".join(sorted(loaded))
        )


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    versions.update(
        {
            name: importlib.metadata.version(distribution)
            for name, distribution in RUNTIME_DISTRIBUTIONS.items()
        }
    )
    return versions


def _project_module_path(module_name: str) -> Path | None:
    if module_name != "p1_qc" and not module_name.startswith("p1_qc."):
        return None
    parts = module_name.split(".")
    candidate = PROJECT_ROOT / "src" / Path(*parts)
    module_file = candidate.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = candidate / "__init__.py"
    if package_file.is_file():
        return package_file
    raise RuntimeError(f"unresolved project import: {module_name}")


def _discover_project_dependency_closure(entrypoint: Path) -> set[str]:
    pending = [entrypoint, PROJECT_ROOT / "src" / "p1_qc" / "__init__.py"]
    visited: set[Path] = set()
    while pending:
        path = pending.pop().resolve(strict=True)
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(PROJECT_ROOT / "src") if path.is_relative_to(
            PROJECT_ROOT / "src"
        ) else None
        if relative is None:
            package_parts: list[str] = []
        elif relative.name == "__init__.py":
            package_parts = list(relative.parent.parts)
        else:
            package_parts = list(relative.parent.parts)
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if not package_parts:
                        continue
                    retain = len(package_parts) - (node.level - 1)
                    if retain < 0:
                        raise RuntimeError(f"invalid relative import in {path}")
                    base = package_parts[:retain]
                    if node.module:
                        base.extend(node.module.split("."))
                    module_names.append(".".join(base))
                elif node.module:
                    module_names.append(node.module)
            for module_name in module_names:
                dependency = _project_module_path(module_name)
                if dependency is not None and dependency not in visited:
                    pending.append(dependency)
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in visited
    }


def _validate_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != (
        "p1_round_b_nonspike_long_event_residual.preregistration.v1r2"
    ):
        raise RuntimeError("unexpected preregistration schema")
    if config.get("experiment_id") != (
        "p1_round_b_nonspike_long_event_residual_v1r2"
    ):
        raise RuntimeError("unexpected experiment id")
    if str(config.get("artifact_dir")) != (
        "artifacts/p1_round_b_nonspike_long_event_residual_v1r2"
    ):
        raise RuntimeError("unexpected artifact namespace")
    trust = config["trust_contract"]
    if trust.get("stdlib_only_before_trust_verification") is not True:
        raise RuntimeError("stdlib-only trust stage is disabled")
    if tuple(trust["project_dependency_paths"]) != REQUIRED_DEPENDENCY_PATHS:
        raise RuntimeError("transitive project dependency closure changed")
    if str(trust["runner_path"]) != str(Path(__file__).relative_to(PROJECT_ROOT)).replace(
        "\\", "/"
    ):
        raise RuntimeError("runner path binding changed")
    if str(trust["core_module_path"]) not in REQUIRED_DEPENDENCY_PATHS:
        raise RuntimeError("core module missing from dependency closure")
    if str(trust["numerical_entrypoint_path"]) not in REQUIRED_DEPENDENCY_PATHS:
        raise RuntimeError("numerical entrypoint missing from dependency closure")
    if tuple(trust["verification_paths"]) != REQUIRED_VERIFICATION_PATHS:
        raise RuntimeError("focused verification-file binding changed")
    discovered = _discover_project_dependency_closure(
        _resolve_repo_path(str(trust["numerical_entrypoint_path"]))
    )
    if discovered != set(REQUIRED_DEPENDENCY_PATHS):
        missing = sorted(discovered - set(REQUIRED_DEPENDENCY_PATHS))
        surplus = sorted(set(REQUIRED_DEPENDENCY_PATHS) - discovered)
        raise RuntimeError(
            f"actual project import closure differs: missing={missing}, surplus={surplus}"
        )

    surface = config["surface"]
    target = config["residual_target"]
    decoder = config["rescue_decoder"]
    budget = config["resource_budget"]
    safety = config["execution_safety"]
    if tuple(surface["fold_order"]) != FOLD_ORDER:
        raise RuntimeError("outer fold order changed")
    if list(surface["seeds"]) != [20260813, 20260829, 20260847]:
        raise RuntimeError("registered seeds changed")
    if int(surface["expected_rows"]) != 421032:
        raise RuntimeError("OOF row contract changed")
    if int(target["minimum_event_rows"]) != 19 or int(target["cadence_minutes"]) != 10:
        raise RuntimeError("residual target contract changed")
    if float(decoder["probability_threshold"]) != 0.8:
        raise RuntimeError("residual threshold changed")
    if int(decoder["maximum_anchor_distance_rows"]) != 18:
        raise RuntimeError("anchor distance changed")
    if decoder["threshold_tuning_grid"] != []:
        raise RuntimeError("threshold tuning is prohibited")
    if int(budget["round_b_base_model_fits"]) != 0:
        raise RuntimeError("Round-B refits are prohibited")
    if int(budget["residual_model_fits"]) != 9:
        raise RuntimeError("residual fit budget changed")
    if int(budget["result_driven_reruns"]) != 0:
        raise RuntimeError("result-driven reruns are prohibited")
    if int(safety["maximum_lifetime_physical_model_fits"]) != 9:
        raise RuntimeError("lifetime fit ceiling changed")
    if safety["incomplete_attempt_policy"] != "FAIL_CLOSED_NO_RETRY":
        raise RuntimeError("incomplete-attempt policy changed")
    if safety["exclusive_lock"] != "execution.lock":
        raise RuntimeError("exclusive lock path changed")
    if safety["attempt_journal_directory"] != "attempt_journal":
        raise RuntimeError("attempt journal path changed")
    left_gate = safety["pre_fit_left_censor_gate"]
    if int(left_gate["maximum_count_per_fold"]) != 0:
        raise RuntimeError("left-censored positive-event allowance changed")
    if left_gate["failure_policy"] != (
        "NO_GO_BEFORE_ANY_MODEL_FIT_NO_SELECTION_RULE_CHANGE"
    ):
        raise RuntimeError("left-censor fail-closed policy changed")
    if safety["terminal_order"] != [
        "result.json",
        "manifest.json",
        "999_completed.json",
        "release execution.lock",
    ]:
        raise RuntimeError("terminal crash-order contract changed")
    if not all(int(value) == 0 for value in config["prohibitions"].values()):
        raise RuntimeError("a prohibition counter is nonzero")

    supersedes = config["supersedes"]
    old_config_path = _resolve_repo_path(str(supersedes["config_path"]))
    old_runner_path = _resolve_repo_path(str(supersedes["runner_path"]))
    old_seal_path = _resolve_repo_path(str(supersedes["seal_path"]))
    bindings = (
        ("superseded config", old_config_path, str(supersedes["config_sha256"])),
        ("superseded runner", old_runner_path, str(supersedes["runner_sha256"])),
        ("superseded seal", old_seal_path, str(supersedes["seal_sha256"])),
    )
    for name, path, expected in bindings:
        if _sha256(path) != expected:
            raise RuntimeError(f"{name} SHA mismatch")
    old_config = _json_load(old_config_path)
    changed = [name for name in NUMERICAL_SECTIONS if config[name] != old_config[name]]
    if changed:
        raise RuntimeError("numerical preregistration changed: " + ", ".join(changed))
    return old_config


def _immutable_specs(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    immutable = config["immutable_inputs"]
    specs: list[tuple[str, Mapping[str, Any]]] = [
        ("base_config", config["base_config"]),
        ("feature_cache", immutable["feature_cache"]),
        ("feature_metadata", immutable["feature_metadata"]),
        ("frozen_truth_oof", immutable["frozen_truth_oof"]),
        ("matched_budget_predictions", immutable["matched_budget_predictions"]),
    ]
    specs.extend(
        (f"round_b_full_prefix:{part['fold']}", part)
        for part in immutable["round_b_full_prefix_parts"]
    )
    return specs


def _verify_file_bindings(
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    inputs: dict[str, dict[str, Any]] = {}
    for name, spec in _immutable_specs(config):
        path = _resolve_repo_path(str(spec["path"]))
        observed = _sha256(path)
        if observed != str(spec["sha256"]):
            raise RuntimeError(f"immutable input SHA mismatch: {name}")
        inputs[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    dependencies = {
        name: _sha256(_resolve_repo_path(name)) for name in REQUIRED_DEPENDENCY_PATHS
    }
    return inputs, dependencies


def _training_source_stdlib(config: Mapping[str, Any]) -> Path:
    raw = os.environ.get("P1_DATA_DIR")
    if not raw:
        raise RuntimeError("P1_DATA_DIR must identify the immutable P1 source directory")
    directory = Path(raw).expanduser().resolve(strict=True)
    path = (directory / "train.csv").resolve(strict=True)
    if path.parent != directory or path.name != "train.csv" or not path.is_file():
        raise RuntimeError("only the P1 training source train.csv is allowed")
    expected = str(config["immutable_inputs"]["feature_cache"]["source_sha256"])
    if _sha256(path) != expected:
        raise RuntimeError("training source SHA differs from feature-cache source binding")
    return path


def _parse_aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("training time must include an explicit timezone")
    return parsed


def _stdlib_left_censored_positive_event_counts(
    config: Mapping[str, Any], source_path: Path
) -> dict[str, int]:
    base_path = _resolve_repo_path(str(config["base_config"]["path"]))
    with base_path.open("rb") as handle:
        base = tomllib.load(handle)
    folds = list(base["validation"]["folds"])
    if tuple(str(fold["name"]) for fold in folds) != FOLD_ORDER:
        raise RuntimeError("stdlib left-censor audit fold order changed")
    train_ends = {
        str(fold["name"]): _parse_aware_datetime(str(fold["train_end"]))
        for fold in folds
    }
    earliest: dict[str, dict[tuple[str, str], tuple[datetime, int, int]]] = {
        fold: {} for fold in FOLD_ORDER
    }
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"station", "layer", "time", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("training source lacks left-censor audit columns")
        for ordinal, row in enumerate(reader):
            timestamp = _parse_aware_datetime(str(row["time"]))
            label = int(str(row["label"]))
            if label not in (0, 1):
                raise RuntimeError("training source contains a non-binary label")
            key = (str(row["station"]), str(row["layer"]))
            for fold, train_end in train_ends.items():
                if timestamp > train_end:
                    continue
                current = earliest[fold].get(key)
                candidate = (timestamp, ordinal, label)
                if current is None or candidate[:2] < current[:2]:
                    earliest[fold][key] = candidate
    return {
        fold: sum(record[2] == 1 for record in earliest[fold].values())
        for fold in FOLD_ORDER
    }


def seal(config_path: Path) -> Path:
    _assert_pretrust_clean()
    config = _json_load(config_path)
    _validate_contract(config)
    inputs, dependencies = _verify_file_bindings(config)
    training_source = _training_source_stdlib(config)
    left_censor_counts = _stdlib_left_censored_positive_event_counts(
        config, training_source
    )
    if any(count != 0 for count in left_censor_counts.values()):
        raise RuntimeError(
            "NO_GO_LEFT_CENSORED_POSITIVE_EVENT_AT_SEAL: "
            + json.dumps(left_censor_counts, sort_keys=True)
        )
    expected_versions = dict(config["trust_contract"]["runtime_versions"])
    observed_versions = _runtime_versions()
    if observed_versions != expected_versions:
        raise RuntimeError(
            f"runtime versions differ: expected={expected_versions}, observed={observed_versions}"
        )
    runner_path = Path(__file__).resolve()
    receipt = {
        "schema_version": "p1_round_b_nonspike_long_event_residual.seal.v1r2",
        "experiment_id": config["experiment_id"],
        "status": "SEALED_STDLIB_PREIMPORT_NO_NUMERICAL_EXECUTION",
        "sealed_at_kst": _now_kst(),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "runner_path": str(runner_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "runner_sha256": _sha256(runner_path),
        "dependency_sha256": dependencies,
        "verification_sha256": {
            name: _sha256(_resolve_repo_path(name)) for name in REQUIRED_VERIFICATION_PATHS
        },
        "immutable_inputs": inputs,
        "runtime_versions": observed_versions,
        "supersedes": dict(config["supersedes"]),
        "numerical_sections_equal_superseded_v1": list(NUMERICAL_SECTIONS),
        "registered_model_fits": 9,
        "pre_fit_left_censor_gate": {
            "registered_required_count_by_fold": {fold: 0 for fold in FOLD_ORDER},
            "observed_count_by_fold": left_censor_counts,
            "maximum_count_per_fold": 0,
            "observation_stage": (
                "STDLIB_SEAL_SCAN_THEN_INDEPENDENT_READ_ONLY_PREFLIGHT_AND_"
                "REPEAT_BEFORE_FIRST_FIT"
            ),
            "training_source": {
                "filename": training_source.name,
                "bytes": training_source.stat().st_size,
                "sha256": _sha256(training_source),
            },
        },
        "operation_counters_at_seal": {
            "round_b_base_model_fits": 0,
            "residual_model_fits": 0,
            "outer_scores": 0,
            "full_fits": 0,
            "candidate_files": 0,
            "uploads": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
            "stdlib_training_source_scans": 1,
        },
    }
    artifact = _artifact_dir(config)
    path = artifact / "preexecution_seal.json"
    _atomic_json_new(path, receipt)
    return path


def _verify_preimport_trust(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    global _TRUST_VERIFIED
    _assert_pretrust_clean()
    config = _json_load(config_path)
    _validate_contract(config)
    seal_path = _artifact_dir(config) / "preexecution_seal.json"
    receipt = _json_load(seal_path)
    if receipt.get("status") != "SEALED_STDLIB_PREIMPORT_NO_NUMERICAL_EXECUTION":
        raise RuntimeError("invalid v1r2 preexecution seal status")
    if receipt.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("config changed after seal")
    if receipt.get("runner_sha256") != _sha256(Path(__file__).resolve()):
        raise RuntimeError("runner changed after seal")
    expected_dependencies = dict(receipt.get("dependency_sha256", {}))
    if len(expected_dependencies) != len(REQUIRED_DEPENDENCY_PATHS) or set(
        expected_dependencies
    ) != set(REQUIRED_DEPENDENCY_PATHS):
        raise RuntimeError("sealed dependency closure is incomplete or reordered")
    for name, expected in expected_dependencies.items():
        if _sha256(_resolve_repo_path(name)) != expected:
            raise RuntimeError(f"dependency changed after seal: {name}")
    expected_verification = dict(receipt.get("verification_sha256", {}))
    if set(expected_verification) != set(REQUIRED_VERIFICATION_PATHS):
        raise RuntimeError("sealed focused verification set changed")
    for name, expected in expected_verification.items():
        if _sha256(_resolve_repo_path(name)) != expected:
            raise RuntimeError(f"focused verification file changed after seal: {name}")
    for name, record in dict(receipt.get("immutable_inputs", {})).items():
        path = _resolve_repo_path(str(record["path"]))
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"immutable input changed after seal: {name}")
    observed_versions = _runtime_versions()
    if observed_versions != receipt.get("runtime_versions"):
        raise RuntimeError("runtime versions changed after seal")
    if observed_versions != config["trust_contract"]["runtime_versions"]:
        raise RuntimeError("runtime versions differ from preregistration")
    if receipt.get("supersedes") != config["supersedes"]:
        raise RuntimeError("superseded-v1 binding differs from seal")
    training_source = _training_source_stdlib(config)
    sealed_source = receipt.get("pre_fit_left_censor_gate", {}).get(
        "training_source", {}
    )
    if sealed_source.get("filename") != "train.csv" or sealed_source.get(
        "sha256"
    ) != _sha256(training_source):
        raise RuntimeError("training source differs from stdlib seal scan")
    _TRUST_VERIFIED = True
    return config, receipt


def _load_v1_numerical(config: Mapping[str, Any] | None = None) -> ModuleType:
    global _NUMERICAL_MODULE
    if not _TRUST_VERIFIED:
        raise RuntimeError("numerical imports are forbidden before trust verification")
    if _NUMERICAL_MODULE is not None:
        return _NUMERICAL_MODULE
    if config is None:
        config = _json_load(DEFAULT_CONFIG)
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    entrypoint = _resolve_repo_path(
        str(config["trust_contract"]["numerical_entrypoint_path"])
    )
    specification = importlib.util.spec_from_file_location(
        "_sealed_p1_round_b_residual_v1", entrypoint
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load sealed v1 numerical entrypoint")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    sklearn = importlib.import_module("sklearn")
    observed = {
        "python": platform.python_version(),
        "numpy": str(module.np.__version__),
        "pandas": str(module.pd.__version__),
        "lightgbm": str(module.lgb.__version__),
        "scikit-learn": str(sklearn.__version__),
    }
    if observed != config["trust_contract"]["runtime_versions"]:
        raise RuntimeError("loaded numerical runtime differs from sealed versions")
    _NUMERICAL_MODULE = module
    return module


class AttemptJournal:
    """Exclusive, immutable, fail-closed lifetime execution record."""

    def __init__(
        self,
        artifact: Path,
        lock_path: Path,
        lock_descriptor: int,
        journal_dir: Path,
        attempt_id: str,
        maximum_fits: int,
    ) -> None:
        self.artifact = artifact
        self.lock_path = lock_path
        self.lock_descriptor = lock_descriptor
        self.journal_dir = journal_dir
        self.attempt_id = attempt_id
        self.maximum_fits = maximum_fits
        self.completed_fits = 0
        self._entry_hashes: dict[str, str] = {}
        self._last_entry_sha256: str | None = None

    @classmethod
    def begin(
        cls,
        artifact: Path,
        *,
        config_sha256: str,
        seal_sha256: str,
        maximum_fits: int,
    ) -> AttemptJournal:
        artifact.mkdir(parents=True, exist_ok=True)
        lock_path = artifact / "execution.lock"
        journal_dir = artifact / "attempt_journal"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        attempt_id = uuid.uuid4().hex
        lock_payload = _json_bytes(
            {
                "attempt_id": attempt_id,
                "created_at_kst": _now_kst(),
                "pid": os.getpid(),
            }
        )
        journal_created = False
        try:
            os.write(descriptor, lock_payload)
            os.fsync(descriptor)
            _fsync_directory(artifact)
            if journal_dir.exists():
                raise FileExistsError(
                    "attempt journal already exists; lifetime rerun is prohibited"
                )
            os.mkdir(journal_dir)
            journal_created = True
            _fsync_directory(artifact)
            journal = cls(
                artifact,
                lock_path,
                descriptor,
                journal_dir,
                attempt_id,
                maximum_fits,
            )
            journal._entry(
                "000_started.json",
                {
                    "schema_version": "p1_round_b_residual.attempt.started.v1r2",
                    "attempt_id": attempt_id,
                    "created_at_kst": _now_kst(),
                    "pid": os.getpid(),
                    "config_sha256": config_sha256,
                    "preexecution_seal_sha256": seal_sha256,
                    "maximum_lifetime_physical_model_fits": maximum_fits,
                    "planned_fold_order": list(FOLD_ORDER),
                    "fits_completed_before_attempt": 0,
                },
            )
            return journal
        except BaseException:
            os.close(descriptor)
            if not journal_created and lock_path.exists():
                lock_path.unlink()
                _fsync_directory(artifact)
            raise

    def _entry(self, name: str, payload: Mapping[str, Any]) -> None:
        self.verify_integrity()
        record = dict(payload)
        record["previous_entry_sha256"] = self._last_entry_sha256
        path = self.journal_dir / name
        _atomic_json_new(path, record)
        observed = _sha256(path)
        self._entry_hashes[name] = observed
        self._last_entry_sha256 = observed

    def verify_integrity(self) -> None:
        observed_names = {path.name for path in self.journal_dir.glob("*.json")}
        if observed_names != set(self._entry_hashes):
            raise RuntimeError("attempt journal membership changed")
        for name, expected in self._entry_hashes.items():
            if _sha256(self.journal_dir / name) != expected:
                raise RuntimeError(f"attempt journal entry changed: {name}")

    def manifest_records(self) -> dict[str, dict[str, Any]]:
        self.verify_integrity()
        return {
            str((self.journal_dir / name).relative_to(PROJECT_ROOT)).replace("\\", "/"): {
                "bytes": (self.journal_dir / name).stat().st_size,
                "sha256": expected,
            }
            for name, expected in self._entry_hashes.items()
        }

    def before_fold(self, ordinal: int, fold: str, planned_fits: int) -> None:
        if planned_fits != 3:
            raise RuntimeError("each sealed fold must fit exactly three seed models")
        if self.completed_fits + planned_fits > self.maximum_fits:
            raise RuntimeError("lifetime physical model-fit ceiling would be exceeded")
        self._entry(
            f"{10 + ordinal * 20:03d}_{fold}_intent.json",
            {
                "schema_version": "p1_round_b_residual.attempt.fold_intent.v1r2",
                "attempt_id": self.attempt_id,
                "fold": fold,
                "ordinal": ordinal,
                "planned_physical_model_fits": planned_fits,
                "fits_completed_before_fold": self.completed_fits,
                "created_at_kst": _now_kst(),
            },
        )

    def record_left_censor_gate(self, counts: Mapping[str, int]) -> None:
        normalized = {fold: int(counts[fold]) for fold in FOLD_ORDER}
        if any(value != 0 for value in normalized.values()):
            raise RuntimeError("cannot record a nonzero left-censor gate as PASS")
        self._entry(
            "005_left_censor_gate_passed.json",
            {
                "schema_version": "p1_round_b_residual.left_censor_gate.v1r2",
                "attempt_id": self.attempt_id,
                "definition": (
                    "station-layer positive connected events touching the first "
                    "chronological row of each outer training prefix"
                ),
                "observed_count_by_fold": normalized,
                "maximum_count_per_fold": 0,
                "physical_model_fits_before_gate": self.completed_fits,
                "status": "PASS_ZERO_BEFORE_ANY_MODEL_FIT",
                "completed_at_kst": _now_kst(),
            },
        )

    def after_fold(
        self,
        ordinal: int,
        fold: str,
        actual_fits: int,
        part_path: Path,
        audit_path: Path,
    ) -> None:
        if actual_fits != 3:
            raise RuntimeError("sealed fold returned a non-three model-fit count")
        self.completed_fits += actual_fits
        if self.completed_fits > self.maximum_fits:
            raise RuntimeError("lifetime physical model-fit ceiling exceeded")
        self._entry(
            f"{20 + ordinal * 20:03d}_{fold}_completed.json",
            {
                "schema_version": "p1_round_b_residual.attempt.fold_completed.v1r2",
                "attempt_id": self.attempt_id,
                "fold": fold,
                "ordinal": ordinal,
                "actual_physical_model_fits": actual_fits,
                "cumulative_physical_model_fits": self.completed_fits,
                "part_path": str(part_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "part_sha256": _sha256(part_path),
                "audit_path": str(audit_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "audit_sha256": _sha256(audit_path),
                "completed_at_kst": _now_kst(),
            },
        )

    def complete(self, result_path: Path, manifest_path: Path) -> None:
        if self.completed_fits != self.maximum_fits:
            raise RuntimeError("terminal completion requires exactly nine physical fits")
        self._entry(
            "999_completed.json",
            {
                "schema_version": "p1_round_b_residual.attempt.completed.v1r2",
                "attempt_id": self.attempt_id,
                "physical_model_fits": self.completed_fits,
                "result_sha256": _sha256(result_path),
                "manifest_sha256": _sha256(manifest_path),
                "completed_at_kst": _now_kst(),
            },
        )
        os.close(self.lock_descriptor)
        self.lock_descriptor = -1
        self.lock_path.unlink()
        _fsync_directory(self.artifact)

    def abandon(self) -> None:
        """Close the handle but deliberately retain lock and journal on failure."""
        if self.lock_descriptor >= 0:
            os.close(self.lock_descriptor)
            self.lock_descriptor = -1


def _file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _commit_terminal_artifacts(
    artifact: Path,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    attempt: AttemptJournal,
    *,
    crash_after: str | None = None,
) -> Path:
    result_path = artifact / "result.json"
    manifest_path = artifact / "manifest.json"
    _atomic_json_new(result_path, result)
    if crash_after == "result":
        raise RuntimeError("injected crash after result")
    _atomic_json_new(manifest_path, manifest)
    if crash_after == "manifest":
        raise RuntimeError("injected crash after manifest")
    attempt.complete(result_path, manifest_path)
    return result_path


def _left_censored_positive_event_counts(
    train: Any,
    folds: Sequence[Mapping[str, Any]],
    numerical: ModuleType,
) -> dict[str, int]:
    """Count positive events touching each station-layer prefix's left edge.

    One such event exists exactly when the first chronological row in a
    station-layer training prefix is positive.  This is an audit-only gate;
    it does not alter eligibility, weights, predictions, or decoder behavior.
    """
    counts: dict[str, int] = {}
    for fold in folds:
        fold_name = str(fold["name"])
        indices = numerical.np.asarray(fold["train_idx"], dtype=numerical.np.int64)
        frame = train.iloc[indices][["station", "layer", "time", "label"]].copy()
        frame["_source_row"] = indices
        frame["_parsed"] = numerical.pd.to_datetime(
            frame["time"], errors="raise", utc=True, format="mixed"
        )
        frame.sort_values(
            ["station", "layer", "_parsed", "_source_row"],
            kind="stable",
            inplace=True,
        )
        first = frame.groupby(
            ["station", "layer"], sort=False, observed=True
        ).head(1)
        labels = numerical.pd.to_numeric(first["label"], errors="raise").to_numpy(
            dtype=numerical.np.int8
        )
        if not numerical.np.isin(labels, [0, 1]).all():
            raise RuntimeError(f"non-binary label in left-censor audit: {fold_name}")
        counts[fold_name] = int((labels == 1).sum())
    if tuple(counts) != FOLD_ORDER:
        raise RuntimeError("left-censor audit fold order changed")
    return counts


def _enforce_left_censor_gate(
    train: Any,
    folds: Sequence[Mapping[str, Any]],
    numerical: ModuleType,
    config: Mapping[str, Any],
) -> dict[str, int]:
    counts = _left_censored_positive_event_counts(train, folds, numerical)
    maximum = int(
        config["execution_safety"]["pre_fit_left_censor_gate"][
            "maximum_count_per_fold"
        ]
    )
    failures = {fold: count for fold, count in counts.items() if count > maximum}
    if failures:
        raise RuntimeError(
            "NO_GO_LEFT_CENSORED_POSITIVE_EVENT_BEFORE_FIT: "
            + json.dumps(failures, sort_keys=True)
        )
    return counts


def preflight(config_path: Path) -> dict[str, Any]:
    config, receipt = _verify_preimport_trust(config_path)
    numerical = _load_v1_numerical(config)
    pins = numerical._verify_immutable_inputs(config)
    surface = numerical._load_base_surface(config)
    train = numerical.load_dataset(_training_source(numerical), kind="train", audit=False)
    if train.attrs.get("source_sha256") != config["immutable_inputs"]["feature_cache"][
        "source_sha256"
    ]:
        raise RuntimeError("training source differs from frozen feature-cache binding")
    p1_config = numerical.load_config(
        numerical._resolve_repo_path(str(config["base_config"]["path"]))
    )
    folds = numerical._fold_runtime(train, p1_config, surface)
    left_censor_counts = _enforce_left_censor_gate(train, folds, numerical, config)
    if left_censor_counts != receipt["pre_fit_left_censor_gate"][
        "observed_count_by_fold"
    ]:
        raise RuntimeError("left-censor counts differ from stdlib seal scan")
    return {
        "schema_version": "p1_round_b_nonspike_long_event_residual.preflight.v1r2",
        "experiment_id": config["experiment_id"],
        "status": "PASS_READ_ONLY_READY_FOR_SINGLE_EXECUTE",
        "rows": len(surface),
        "fold_rows": {
            fold: int(surface["fold"].eq(fold).sum()) for fold in FOLD_ORDER
        },
        "row_positions_unique": bool(surface["row_position"].is_unique),
        "exact_round_b_default_equivalence": True,
        "round_b_base_model_fits_required": 0,
        "residual_model_fits_registered": 9,
        "immutable_inputs": pins,
        "trust_seal_sha256": _sha256(_artifact_dir(config) / "preexecution_seal.json"),
        "trust_status": receipt["status"],
        "left_censored_positive_connected_event_count_by_fold": left_censor_counts,
        "left_censored_positive_connected_event_count_total_fold_incidence": sum(
            left_censor_counts.values()
        ),
        "left_censor_gate_status": "PASS_ZERO_BEFORE_ANY_MODEL_FIT",
        "training_source_kind": "train_only",
        "training_source_sha256": train.attrs["source_sha256"],
        "official_test_reads": 0,
        "sample_format_reads": 0,
        "submission_candidate_reads": 0,
        "outer_scores_computed": 0,
    }


def _training_source(numerical: ModuleType) -> Path:
    return numerical._training_source()


def execute(config_path: Path) -> Path:
    started = time.perf_counter()
    config, seal_receipt = _verify_preimport_trust(config_path)
    artifact = _artifact_dir(config)
    attempt = AttemptJournal.begin(
        artifact,
        config_sha256=_sha256(config_path),
        seal_sha256=_sha256(artifact / "preexecution_seal.json"),
        maximum_fits=int(
            config["execution_safety"]["maximum_lifetime_physical_model_fits"]
        ),
    )
    try:
        numerical = _load_v1_numerical(config)
        numerical._verify_immutable_inputs(config)
        forbidden_existing = [
            artifact / "predictions_complete.json",
            artifact / "predictions.parquet",
            artifact / "metrics.json",
            artifact / "result.json",
            artifact / "manifest.json",
        ]
        forbidden_existing.extend(
            artifact / "prediction_parts" / f"{fold}.parquet" for fold in FOLD_ORDER
        )
        forbidden_existing.extend(
            artifact / "prediction_parts" / f"{fold}.json" for fold in FOLD_ORDER
        )
        if any(path.exists() for path in forbidden_existing):
            raise FileExistsError("one-shot output already exists; rerun is prohibited")

        train = numerical.load_dataset(_training_source(numerical), kind="train", audit=False)
        if train.attrs.get("source_sha256") != config["immutable_inputs"]["feature_cache"][
            "source_sha256"
        ]:
            raise RuntimeError("training source differs from frozen feature-cache binding")
        surface = numerical._load_base_surface(config)
        p1_config = numerical.load_config(
            numerical._resolve_repo_path(str(config["base_config"]["path"]))
        )
        folds = numerical._fold_runtime(train, p1_config, surface)
        left_censor_counts = _enforce_left_censor_gate(train, folds, numerical, config)
        if left_censor_counts != seal_receipt["pre_fit_left_censor_gate"][
            "observed_count_by_fold"
        ]:
            raise RuntimeError("left-censor counts differ from stdlib seal scan")
        attempt.record_left_censor_gate(left_censor_counts)
        bundle, feature_metadata = numerical._load_feature_bundle(train, config)
        part_frames: list[Any] = []
        fit_audits: dict[str, Any] = {}
        wall_cap = float(config["resource_budget"]["wall_clock_cap_seconds"])
        for ordinal, fold in enumerate(folds):
            if time.perf_counter() - started > wall_cap:
                raise TimeoutError("wall-clock cap exceeded before next outer fold")
            fold_name = str(fold["name"])
            attempt.before_fold(ordinal, fold_name, 3)
            output, audit = numerical._fit_fold(train, bundle, config, fold)
            part_path = artifact / "prediction_parts" / f"{fold_name}.parquet"
            audit_path = artifact / "prediction_parts" / f"{fold_name}.json"
            _atomic_parquet_new(part_path, output)
            audit.update(
                {
                    "parquet_path": str(part_path.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"
                    ),
                    "parquet_sha256": _sha256(part_path),
                    "completed_at_kst": _now_kst(),
                }
            )
            _atomic_json_new(audit_path, audit)
            actual_fits = int(audit["model_fits"])
            attempt.after_fold(
                ordinal, fold_name, actual_fits, part_path, audit_path
            )
            part_frames.append(output)
            fit_audits[fold_name] = audit

        if attempt.completed_fits != int(config["resource_budget"]["residual_model_fits"]):
            raise RuntimeError("residual fit count differs from preregistration")
        predictions = numerical.pd.concat(part_frames, ignore_index=True)
        if len(predictions) != int(config["surface"]["expected_rows"]):
            raise RuntimeError("completed prediction surface has wrong row count")
        predictions_path = artifact / "predictions.parquet"
        _atomic_parquet_new(predictions_path, predictions)
        complete_path = artifact / "predictions_complete.json"
        _atomic_json_new(
            complete_path,
            {
                "schema_version": (
                    "p1_round_b_nonspike_long_event_residual."
                    "predictions_complete.v1r2"
                ),
                "experiment_id": config["experiment_id"],
                "status": "ALL_OUTER_PREDICTIONS_FROZEN_BEFORE_SCORING",
                "rows": len(predictions),
                "residual_model_fits": attempt.completed_fits,
                "round_b_base_model_fits": 0,
                "prediction_sha256": _sha256(predictions_path),
                "fold_audits": fit_audits,
                "left_censored_positive_connected_event_count_by_fold": (
                    left_censor_counts
                ),
                "target_fold_scores_computed_before_receipt": 0,
                "completed_at_kst": _now_kst(),
            },
        )
        if time.perf_counter() - started > wall_cap:
            raise TimeoutError("wall-clock cap exceeded after prediction freeze")

        metrics, checks = numerical._score(config, predictions)
        metrics_path = artifact / "metrics.json"
        _atomic_json_new(metrics_path, metrics)
        passed = all(checks.values())
        result = {
            "schema_version": "p1_round_b_nonspike_long_event_residual.result.v1r2",
            "experiment_id": config["experiment_id"],
            "status": "COMPLETE_LOCAL_SCREEN_ONLY",
            "decision": config["interpretation"]["pass_label"]
            if passed
            else config["interpretation"]["fail_label"],
            "passed_all_gates": passed,
            "completed_at_kst": _now_kst(),
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": _sha256(config_path),
            "preexecution_seal_sha256": _sha256(
                artifact / "preexecution_seal.json"
            ),
            "prediction_sha256": _sha256(predictions_path),
            "metrics_sha256": _sha256(metrics_path),
            "feature_cache_sha256": feature_metadata["parquet_sha256"],
            "operation_counters": {
                "round_b_base_model_fits": 0,
                "residual_model_fits": attempt.completed_fits,
                "outer_scores": 1,
                "full_fits": 0,
                "candidate_files": 0,
                "uploads": 0,
                "source_mutations": 0,
                "frozen_artifact_mutations": 0,
                "official_test_reads": 0,
                "sample_format_reads": 0,
                "submission_candidate_reads": 0,
            },
            "independent_confirmation": config["interpretation"][
                "independent_confirmation"
            ],
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                **config["trust_contract"]["runtime_versions"],
            },
            "seal_status": seal_receipt["status"],
            "attempt_id": attempt.attempt_id,
            "left_censored_positive_connected_event_count_by_fold": left_censor_counts,
        }
        result_path = artifact / "result.json"
        result_payload = _json_bytes(result)
        manifest_files: list[Path] = [
            config_path,
            artifact / "preexecution_seal.json",
            predictions_path,
            complete_path,
            metrics_path,
        ]
        manifest_files.extend(
            artifact / "prediction_parts" / f"{fold}.parquet" for fold in FOLD_ORDER
        )
        manifest_files.extend(
            artifact / "prediction_parts" / f"{fold}.json" for fold in FOLD_ORDER
        )
        artifacts = {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _file_record(path)
            for path in manifest_files
        }
        artifacts.update(attempt.manifest_records())
        artifacts[str(result_path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = {
            "bytes": len(result_payload),
            "sha256": _sha256_bytes(result_payload),
        }
        manifest = {
            "schema_version": "p1_round_b_nonspike_long_event_residual.manifest.v1r2",
            "experiment_id": config["experiment_id"],
            "attempt_id": attempt.attempt_id,
            "created_at_kst": _now_kst(),
            "artifacts": artifacts,
            "terminal_order": [
                "result.json",
                "manifest.json",
                "attempt_journal/999_completed.json",
                "release execution.lock",
            ],
        }
        return _commit_terminal_artifacts(artifact, result, manifest, attempt)
    except BaseException:
        attempt.abandon()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--seal", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    if not config_path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError("config must remain inside repository")
    if args.seal:
        output: Any = str(seal(config_path))
    elif args.preflight:
        output = preflight(config_path)
    else:
        output = str(execute(config_path))
    print(json.dumps({"status": "ok", "output": output}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
