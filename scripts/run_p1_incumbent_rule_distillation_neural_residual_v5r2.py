"""Run the append-only P1 Gen5r2 blind-boundary correction exactly once."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import stat
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocean_goal.meaningful_score_v3 import evaluate_learning_curve, load_contract
from p1_qc.causal_raw_features_v4r2 import build_exact_prefix_causal_matrix
from p1_qc.selective_targets_v5r2 import (
    SelectiveTargetAccessor,
    load_frozen_oof_keys_only,
    load_input_only_train,
)

_GEN5_RUNNER = PROJECT_ROOT / "scripts/run_p1_incumbent_rule_distillation_neural_residual_v5.py"
_GEN5_SPEC = importlib.util.spec_from_file_location("p1_gen5r2_preserved_engine", _GEN5_RUNNER)
if _GEN5_SPEC is None or _GEN5_SPEC.loader is None:
    raise ImportError("failed to load pinned tombstoned P1 Gen5 engine")
gen5 = importlib.util.module_from_spec(_GEN5_SPEC)
sys.modules[_GEN5_SPEC.name] = gen5
_GEN5_SPEC.loader.exec_module(gen5)
shared = gen5.shared

EXPECTED_CONFIG_SHA256 = "2dbc5a98305c2b567ccdfdf6d31549b215ddfd7cf54d6869222f74f277fe0566"
EXPECTED_CONFIG_DEEP_SHA256 = "21e657f2b441707d1823a262425422e82a29350efff9ec1929cc627dcb725dd5"
CANONICAL_CONFIG = (
    "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5r2.json"
)
CANONICAL_ARTIFACT = "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r2"
CANONICAL_CONTROL = "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r2_control"
CANONICAL_QA_RECEIPT = f"{CANONICAL_CONTROL}/independent_static_qa.json"
CANONICAL_AUTHORIZATION = f"{CANONICAL_CONTROL}/execution_authorization.json"
CANONICAL_LOCK = f"{CANONICAL_CONTROL}/attempt.lock"
PREDECESSOR_CONFIG_SHA256 = "da7427dcfa58daff7d9825653c34296aeb6c4d0648d0d2295715c5e8c0179396"
PREDECESSOR_PREREG_SHA256 = "612543d63eeac8f8ed444d1bfadcac6663b67d31c265082a011cb7a7f3221014"
PREDECESSOR_PRESEAL_SHA256 = "8e530ab36c61b50b8f5bca14245578230dad37b386c1d71f60726cc300723449"
PREDECESSOR_SUPERSESSION_SHA256 = (
    "1c3da3bd4bc7c14c44319ecefc8c21e5a698671bfa4f49060afc8b97d58245f4"
)
PREDECESSOR_TOMBSTONE_SHA256 = (
    "00f990794a1ef1d9ebab9c1fc27a1d6d41c495944da34da9f4e2cda4e8bd7fc9"
)
EXECUTION_TOMBSTONE = (
    "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r2_control/"
    "EXECUTION_TOMBSTONE.json"
)
EXECUTION_TOMBSTONE_SHA256 = (
    "e761088364873ee5064ba71103818a0770a9b20a89e276e47b26af2a0e49422a"
)
HYPOTHESIS = "incumbent_rule_distillation_with_out_of_fold_neural_residual"
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260813, 20260829, 20260847)
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
CANONICAL_ROOT_PATH = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
CANONICAL_DATA_DIR_PATH = (
    CANONICAL_ROOT_PATH / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly"
)


def _enforce_execution_tombstone() -> None:
    path = (Path(__file__).resolve().parents[1] / EXECUTION_TOMBSTONE).resolve(strict=True)
    if _sha(path) != EXECUTION_TOMBSTONE_SHA256:
        raise PermissionError("invalidated Gen5r2 execution tombstone SHA differs")
    value = _strict_json(path)
    if not (
        value.get("generation")
        == "p1_incumbent_rule_distillation_neural_residual_v5r2"
        and value.get("successor_generation")
        == "p1_incumbent_rule_distillation_neural_residual_v5r3"
        and value.get("execution_prohibited") is True
        and value.get("authorization_must_fail_before_attempt_lock") is True
        and value.get("attempt_lock_created") is False
        and value.get("model_fits") == 0
        and value.get("uploads") == 0
    ):
        raise PermissionError("invalidated Gen5r2 execution tombstone semantics differ")
    raise PermissionError(
        "p1_incumbent_rule_distillation_neural_residual_v5r2 is superseded and non-executable"
    )


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / CANONICAL_CONFIG,
        "artifact": root / CANONICAL_ARTIFACT,
        "control": root / CANONICAL_CONTROL,
        "qa_receipt": root / CANONICAL_QA_RECEIPT,
        "authorization": root / CANONICAL_AUTHORIZATION,
        "lock": root / CANONICAL_LOCK,
        "predecessor_config": root
        / "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5.json",
        "predecessor_artifact": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5",
        "predecessor_lock": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5.ATTEMPT_LOCK.json",
        "predecessor_prereg": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5/preregistration.json",
        "predecessor_preseal": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5/preseal_static_qa.json",
        "predecessor_supersession": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5/STATIC_QA_NO_GO_SUPERSEDED_20260823.json",
        "predecessor_tombstone": root
        / "artifacts/p1_incumbent_rule_distillation_neural_residual_v5/EXECUTION_TOMBSTONE.json",
        "base_config": root / "configs/p1.toml",
        "goal": root / "configs/goals/meaningful_score_maximization_v3.json",
        "feature_cache": root / "artifacts/cache/train_causal_raw_prefix_safe_v4r2.parquet",
        "feature_metadata": root / "artifacts/cache/train_causal_raw_prefix_safe_v4r2.json",
        "gen1": root / "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen4r4": root / "artifacts/p1_masked_pretrain_binary_event_v4r4",
        "frozen_oof": root / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
        "ledger": root / "artifacts/meaningful_score_goal_v5/registry.jsonl",
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token in {path}: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_single_link(path: Path, *, role: str) -> None:
    if os.name == "nt" and os.stat(path).st_nlink != 1:
        raise PermissionError(f"hardlinked canonical identity anchor is forbidden: {role}")


def _assert_canonical_identity(root: Path, data_dir: Path) -> tuple[Path, Path]:
    lexical_root = Path(os.path.abspath(root))
    lexical_data = Path(os.path.abspath(data_dir))
    if lexical_root != CANONICAL_ROOT_PATH:
        raise PermissionError("workspace clone or non-canonical root is forbidden")
    if lexical_data != CANONICAL_DATA_DIR_PATH:
        raise PermissionError("non-canonical P1 data directory is forbidden")
    if root.resolve(strict=True) != CANONICAL_ROOT_PATH:
        raise PermissionError("canonical workspace resolves through an alias or reparse point")
    if data_dir.resolve(strict=True) != CANONICAL_DATA_DIR_PATH:
        raise PermissionError("canonical data directory resolves through an alias or reparse point")
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    anchors = {
        "workspace": CANONICAL_ROOT_PATH,
        "data_dir": CANONICAL_DATA_DIR_PATH,
        "config": CANONICAL_ROOT_PATH / CANONICAL_CONFIG,
        "runner": Path(__file__).resolve(strict=True),
        "train": CANONICAL_DATA_DIR_PATH / "train.csv",
        "test": CANONICAL_DATA_DIR_PATH / "test.csv",
        "sample_submission": CANONICAL_DATA_DIR_PATH / "sample_submission.csv",
    }
    for role, path in anchors.items():
        if getattr(os.lstat(path), "st_file_attributes", 0) & reparse:
            raise PermissionError(f"reparse-point canonical identity anchor is forbidden: {role}")
        _assert_single_link(path, role=role)
    return CANONICAL_ROOT_PATH, CANONICAL_DATA_DIR_PATH


def _verify_predecessor_tombstone(paths: dict[str, Path]) -> dict[str, Any]:
    if paths["predecessor_lock"].exists():
        raise PermissionError("invalidated Gen5 predecessor attempt lock unexpectedly exists")
    expected = {
        "predecessor_config": PREDECESSOR_CONFIG_SHA256,
        "predecessor_prereg": PREDECESSOR_PREREG_SHA256,
        "predecessor_preseal": PREDECESSOR_PRESEAL_SHA256,
        "predecessor_supersession": PREDECESSOR_SUPERSESSION_SHA256,
        "predecessor_tombstone": PREDECESSOR_TOMBSTONE_SHA256,
    }
    observed = {name: _sha(paths[name]) for name in expected}
    if observed != expected:
        raise PermissionError("Gen5 predecessor seal, NO-GO receipt, or tombstone differs")
    tombstone = _strict_json(paths["predecessor_tombstone"])
    if not (
        tombstone.get("generation") == "p1_incumbent_rule_distillation_neural_residual_v5"
        and tombstone.get("successor_generation")
        == "p1_incumbent_rule_distillation_neural_residual_v5r2"
        and tombstone.get("execution_prohibited") is True
        and tombstone.get("authorization_must_fail_before_attempt_lock") is True
        and tombstone.get("attempt_lock_created") is False
        and tombstone.get("curve_model_fits") == 0
        and tombstone.get("uploads") == 0
    ):
        raise PermissionError("Gen5 predecessor tombstone semantics differ")
    return {**observed, "execution_prohibited": True}


def _verify_execution_closure(root: Path, config: dict[str, Any]) -> dict[str, str]:
    expected = config["execution_closure_sha256"]
    if not isinstance(expected, dict) or not expected:
        raise PermissionError("Gen5r2 executable closure is empty")
    observed: dict[str, str] = {}
    for relative, expected_sha in expected.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise PermissionError("Gen5r2 executable closure path is non-canonical")
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise PermissionError("Gen5r2 executable closure escaped the workspace")
        observed[relative] = _sha(path)
        if observed[relative] != expected_sha:
            raise PermissionError(f"Gen5r2 executable closure SHA differs: {relative}")
    required_suffixes = {
        "src/p1_qc/binary_event_tcn.py",
        "src/p1_qc/data.py",
        "src/p1_qc/config.py",
        "src/p1_qc/models_tabular.py",
        "src/p1_qc/rules.py",
        "src/p1_qc/validation.py",
        "requirements-lock.txt",
    }
    if not required_suffixes.issubset(expected):
        raise PermissionError("Gen5r2 required transitive closure entries are absent")
    return observed


def _runtime_config(root: Path, overlay: dict[str, Any]) -> dict[str, Any]:
    predecessor = _strict_json(root / overlay["predecessor_config_path"])
    if _sha(root / overlay["predecessor_config_path"]) != PREDECESSOR_CONFIG_SHA256:
        raise PermissionError("preserved Gen5 config SHA differs")
    if not (
        predecessor["hypotheses"] == overlay["preserved_structure"]["hypotheses"]
        and predecessor["model"] == overlay["preserved_structure"]["model"]
        and predecessor["prefix_fractions"] == list(FRACTIONS)
        and predecessor["seeds"] == list(SEEDS)
        and predecessor["train_only_no_op_gate"]
        == overlay["preserved_structure"]["train_only_no_op_gate"]
        and predecessor["fixed_fold_postprocess"]
        == overlay["preserved_structure"]["fixed_fold_postprocess"]
        and predecessor["inner_cross_fit"]
        == overlay["preserved_structure"]["inner_cross_fit"]
        and predecessor["training"] == overlay["preserved_structure"]["training"]
    ):
        raise PermissionError("Gen5r2 changed the registered scientific structure")
    runtime = dict(predecessor)
    runtime["experiment_id"] = overlay["experiment_id"]
    runtime["status"] = overlay["status"]
    runtime["canonical_paths"] = {
        **predecessor["canonical_paths"],
        "config": CANONICAL_CONFIG,
        "artifact": CANONICAL_ARTIFACT,
        "attempt_lock": CANONICAL_LOCK,
    }
    return runtime


def authorize_entry(
    *,
    root: Path,
    data_dir: Path,
    requested_config: Path,
    requested_artifact: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], dict[str, str]]:
    _enforce_execution_tombstone()
    root, data_dir = _assert_canonical_identity(root, data_dir)
    paths = _paths(root)
    if requested_config.resolve(strict=True) != paths["config"].resolve(strict=True):
        raise PermissionError("non-canonical Gen5r2 config path is forbidden")
    if requested_artifact.resolve(strict=False) != paths["artifact"].resolve(strict=False):
        raise PermissionError("non-canonical Gen5r2 artifact path is forbidden")
    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical Gen5r2 config byte SHA differs")
    overlay = json.loads(content)
    if _deep_sha(overlay) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical Gen5r2 config deep JSON differs")
    if not (
        overlay["experiment_id"]
        == "p1_incumbent_rule_distillation_neural_residual_v5r2"
        and overlay["problem"] == "P1"
        and overlay["comparison_mode"] == "EXACT_OFFICIAL_PREFIX_REFIT"
        and overlay["v5_ledger_binding"]["head_seq"] == 9
        and overlay["v5_ledger_binding"]["semantic_upload_count"] == 0
        and overlay["correction_contract"]["fold_major_order"] == list(FOLD_ORDER)
        and overlay["correction_contract"]["prefix_order"] == list(FRACTIONS)
    ):
        raise PermissionError("Gen5r2 identity, exact comparator, or rolling order differs")
    _verify_predecessor_tombstone(paths)
    closure = _verify_execution_closure(root, overlay)
    runtime = _runtime_config(root, overlay)
    shared._verify_input_pins(root, data_dir, runtime)
    gen5._verify_failed_generation(runtime, paths)
    gen5._verify_v5_ledger_binding(root, runtime, paths["ledger"])
    return overlay, runtime, paths, closure


def _verify_independent_qa(
    root: Path,
    overlay: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], str]:
    if not paths["qa_receipt"].is_file():
        raise PermissionError("canonical Gen5r2 independent-QA receipt is missing")
    qa = _strict_json(paths["qa_receipt"])
    required = {
        "schema_version",
        "problem",
        "generation",
        "verdict",
        "p0_count",
        "p1_count",
        "config",
        "runner",
        "execution_closure_sha256",
        "checks",
        "reviewer",
        "actual_run_performed",
        "files_modified",
        "test_value_reads",
        "candidate_files",
        "uploads",
    }
    if set(qa) != required:
        raise PermissionError("Gen5r2 independent-QA receipt keys differ")
    expected_config = {"path": CANONICAL_CONFIG, "sha256": EXPECTED_CONFIG_SHA256}
    expected_runner = {
        "path": Path(__file__).resolve().relative_to(root).as_posix(),
        "sha256": _sha(Path(__file__).resolve()),
    }
    if not (
        qa["schema_version"] == "p1_gen5r2_independent_static_qa.v1"
        and qa["problem"] == "P1"
        and qa["generation"] == overlay["experiment_id"]
        and qa["verdict"] == "GO"
        and qa["p0_count"] == qa["p1_count"] == 0
        and qa["config"] == expected_config
        and qa["runner"] == expected_runner
        and qa["execution_closure_sha256"] == overlay["execution_closure_sha256"]
        and isinstance(qa["checks"], dict)
        and qa["checks"]
        and all(value is True for value in qa["checks"].values())
        and bool(qa["reviewer"])
        and qa["actual_run_performed"] is False
        and qa["files_modified"] == 0
        and qa["test_value_reads"] == qa["candidate_files"] == qa["uploads"] == 0
    ):
        raise PermissionError("Gen5r2 independent-QA receipt did not authorize GO")
    return qa, _sha(paths["qa_receipt"])


def _verify_execution_authorization(
    root: Path,
    overlay: dict[str, Any],
    paths: dict[str, Path],
    *,
    qa_sha256: str,
) -> tuple[dict[str, Any], str]:
    if paths["lock"].exists():
        raise FileExistsError("Gen5r2 one-shot attempt lock already exists")
    if not paths["authorization"].is_file():
        raise PermissionError("separate Gen5r2 execution authorization is missing")
    authorization = _strict_json(paths["authorization"])
    required = {
        "schema_version",
        "problem",
        "generation",
        "authorization",
        "user_message_reference",
        "config",
        "runner",
        "qa_receipt",
        "execution_closure_sha256",
        "execution_authorized",
        "test_prediction_allowed",
        "candidate_creation_allowed",
        "upload_allowed",
    }
    if set(authorization) != required:
        raise PermissionError("Gen5r2 execution authorization keys differ")
    runner_pin = {
        "path": Path(__file__).resolve().relative_to(root).as_posix(),
        "sha256": _sha(Path(__file__).resolve()),
    }
    phrase = f"AUTHORIZE_P1_GEN5R2_EXECUTION:{EXPECTED_CONFIG_SHA256}:{runner_pin['sha256']}"
    if not (
        authorization["schema_version"] == "p1_gen5r2_execution_authorization.v1"
        and authorization["problem"] == "P1"
        and authorization["generation"] == overlay["experiment_id"]
        and authorization["authorization"] == phrase
        and bool(authorization["user_message_reference"])
        and authorization["config"]
        == {"path": CANONICAL_CONFIG, "sha256": EXPECTED_CONFIG_SHA256}
        and authorization["runner"] == runner_pin
        and authorization["qa_receipt"]
        == {"path": CANONICAL_QA_RECEIPT, "sha256": qa_sha256}
        and authorization["execution_closure_sha256"]
        == overlay["execution_closure_sha256"]
        and authorization["execution_authorized"] is True
        and authorization["test_prediction_allowed"] is False
        and authorization["candidate_creation_allowed"] is False
        and authorization["upload_allowed"] is False
    ):
        raise PermissionError("Gen5r2 execution authorization failed")
    return authorization, _sha(paths["authorization"])


def _acquire_lock(
    path: Path,
    *,
    qa_sha256: str,
    authorization_sha256: str,
    closure: dict[str, str],
) -> dict[str, Any]:
    receipt = {
        "schema_version": "p1_gen5r2_attempt_lock.v1",
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "experiment_id": "p1_incumbent_rule_distillation_neural_residual_v5r2",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "runner_sha256": _sha(Path(__file__).resolve()),
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "execution_closure_sha256": closure,
        "o_excl": True,
        "rerun_forbidden": True,
    }
    _json_new(path, receipt)
    return {**receipt, "sha256": _sha(path)}


def _verify_lock(path: Path, receipt: dict[str, Any]) -> None:
    if not path.is_file() or _sha(path) != receipt["sha256"]:
        raise PermissionError("Gen5r2 attempt lock differs")
    persisted = _strict_json(path)
    if persisted != {key: value for key, value in receipt.items() if key != "sha256"}:
        raise PermissionError("Gen5r2 persisted attempt lock payload differs")


_CAPABILITY_MINT = object()
_LIVE_CAPABILITIES: dict[str, _ExecutionCapability] = {}


class _ExecutionCapability:
    __slots__ = (
        "root",
        "config_sha256",
        "runner_sha256",
        "qa_sha256",
        "authorization_sha256",
        "lock_sha256",
        "closure",
        "phase",
    )

    def __init__(
        self,
        mint: object,
        *,
        root: Path,
        qa_sha256: str,
        authorization_sha256: str,
        lock_sha256: str,
        closure: dict[str, str],
    ) -> None:
        if mint is not _CAPABILITY_MINT:
            raise PermissionError("canonical Gen5r2 capability mint is required")
        self.root = root
        self.config_sha256 = EXPECTED_CONFIG_SHA256
        self.runner_sha256 = _sha(Path(__file__).resolve())
        self.qa_sha256 = qa_sha256
        self.authorization_sha256 = authorization_sha256
        self.lock_sha256 = lock_sha256
        self.closure = dict(closure)
        self.phase = "BLIND_CURVE"


def _mint_capability(
    *,
    root: Path,
    qa_sha256: str,
    authorization_sha256: str,
    lock: dict[str, Any],
    closure: dict[str, str],
) -> _ExecutionCapability:
    if _LIVE_CAPABILITIES:
        raise PermissionError("a canonical Gen5r2 capability is already live")
    _verify_lock(_paths(root)["lock"], lock)
    capability = _ExecutionCapability(
        _CAPABILITY_MINT,
        root=root,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
        lock_sha256=lock["sha256"],
        closure=closure,
    )
    _LIVE_CAPABILITIES[lock["sha256"]] = capability
    return capability


def _require_capability(capability: object, phase: str) -> _ExecutionCapability:
    if not isinstance(capability, _ExecutionCapability):
        raise PermissionError("canonical live Gen5r2 execution capability is required")
    live = _LIVE_CAPABILITIES.get(capability.lock_sha256)
    if live is not capability:
        raise PermissionError("forged or stale Gen5r2 execution capability")
    if capability.root != CANONICAL_ROOT_PATH:
        raise PermissionError("Gen5r2 capability root differs")
    if capability.config_sha256 != EXPECTED_CONFIG_SHA256:
        raise PermissionError("Gen5r2 capability config pin differs")
    if capability.runner_sha256 != _sha(Path(__file__).resolve()):
        raise PermissionError("Gen5r2 capability runner pin differs")
    if capability.phase != phase:
        raise PermissionError(f"Gen5r2 capability phase must be {phase}")
    return capability


class _BlindCommitmentLedger:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact
        self._expected = [(fold, fraction) for fold in FOLD_ORDER for fraction in FRACTIONS]
        self._cells: list[dict[str, Any]] = []
        self._folds: dict[str, dict[str, Any]] = {}
        self._head = "0" * 64
        self._global: dict[str, Any] | None = None

    def is_fold_committed(self, fold: str) -> bool:
        return fold in self._folds

    def is_global_committed(self) -> bool:
        return self._global is not None

    def commit_cell(
        self,
        *,
        fold: str,
        fraction: float,
        validation_ids_sha256: str,
        seed_blind_predictions: list[dict[str, Any]],
        prediction_part: dict[str, Any],
        target_accessor: SelectiveTargetAccessor,
    ) -> dict[str, Any]:
        sequence = len(self._cells) + 1
        if self._expected[sequence - 1] != (fold, fraction):
            raise PermissionError("Gen5r2 cell commitment order is not fold-major monotone")
        if [row["seed"] for row in seed_blind_predictions] != list(SEEDS):
            raise PermissionError("Gen5r2 cell commitment seed order differs")
        for row in [*seed_blind_predictions, prediction_part]:
            target = self.artifact / row["path"]
            if not target.is_file() or _sha(target) != row["sha256"]:
                raise PermissionError("Gen5r2 blind prediction pin differs before commitment")
        validation_ids = target_accessor.validation_rows(fold)
        if shared.ids_sha256(validation_ids) != validation_ids_sha256:
            raise PermissionError("Gen5r2 active validation ID pin differs")
        target_counts = target_accessor.validation_target_decode_counts(fold)
        if target_counts != {"label": 0, "anomaly_type": 0}:
            raise PermissionError(
                "Gen5r2 active outer-validation target decoded before commitment"
            )
        fraction_tag = gen5.shared._tag(fraction)
        relative = f"blind_commitments/cell_{sequence:02d}_{fold}_{fraction_tag}.json"
        path = self.artifact / relative
        payload = {
            "schema_version": "p1_gen5r2_blind_cell_commitment.v1",
            "sequence": sequence,
            "previous_event_sha256": self._head,
            "fold": fold,
            "fraction": fraction,
            "validation_ids_sha256": validation_ids_sha256,
            "seed_blind_predictions": seed_blind_predictions,
            "prediction_part": prediction_part,
            "active_outer_validation_labels_decoded_before_commitment": target_counts[
                "label"
            ],
            "active_outer_validation_anomaly_types_decoded_before_commitment": target_counts[
                "anomaly_type"
            ],
            "o_excl": True,
        }
        _json_new(path, payload)
        event = {"path": relative, "sha256": _sha(path), "payload": payload}
        self._head = event["sha256"]
        self._cells.append(event)
        fold_cells = [row for row in self._cells if row["payload"]["fold"] == fold]
        if len(fold_cells) == len(FRACTIONS):
            fold_relative = f"blind_commitments/fold_{fold}.json"
            fold_path = self.artifact / fold_relative
            fold_payload = {
                "schema_version": "p1_gen5r2_blind_fold_commitment.v1",
                "fold": fold,
                "fractions": list(FRACTIONS),
                "cell_commitments": [
                    {"path": row["path"], "sha256": row["sha256"]}
                    for row in fold_cells
                ],
                "chain_head_sha256": self._head,
                "all_five_cells_committed": True,
                "rolling_origin_training_reuse_now_allowed": True,
                "o_excl": True,
            }
            _json_new(fold_path, fold_payload)
            self._folds[fold] = {
                "path": fold_relative,
                "sha256": _sha(fold_path),
                "payload": fold_payload,
            }
        return event

    def finalize_ledger(self) -> dict[str, Any]:
        if len(self._cells) != 15 or tuple(self._folds) != FOLD_ORDER:
            raise PermissionError("Gen5r2 blind ledger is incomplete")
        relative = "blind_commitments/ledger_complete.json"
        path = self.artifact / relative
        payload = {
            "schema_version": "p1_gen5r2_blind_commitment_ledger.v1",
            "order": "fold_major_then_prefix_fraction",
            "fold_order": list(FOLD_ORDER),
            "prefix_order": list(FRACTIONS),
            "cell_count": 15,
            "seed_blind_prediction_count": 45,
            "cell_commitments": [
                {"path": row["path"], "sha256": row["sha256"]} for row in self._cells
            ],
            "fold_commitments": [
                {"path": self._folds[fold]["path"], "sha256": self._folds[fold]["sha256"]}
                for fold in FOLD_ORDER
            ],
            "chain_head_sha256": self._head,
            "target_fold_scores": 0,
            "o_excl": True,
        }
        _json_new(path, payload)
        return {"path": relative, "sha256": _sha(path), "payload": payload}

    def mark_predictions_complete(self, path: Path) -> None:
        value = _strict_json(path)
        if not (
            value.get("fit_cells") == 225
            and value.get("blind_commitment_ledger", {}).get("sha256")
            == _sha(self.artifact / "blind_commitments/ledger_complete.json")
            and len(value.get("prediction_parts", [])) == 15
            and len(value.get("model_receipts", [])) == 45
        ):
            raise PermissionError("Gen5r2 predictions_complete did not bind every blind prediction")
        self._global = {"path": path, "sha256": _sha(path)}

    @property
    def cell_receipts(self) -> list[dict[str, Any]]:
        return [
            {"path": row["path"], "sha256": row["sha256"]} for row in self._cells
        ]

    @property
    def fold_receipts(self) -> list[dict[str, Any]]:
        return [
            {"path": self._folds[fold]["path"], "sha256": self._folds[fold]["sha256"]}
            for fold in FOLD_ORDER
        ]


def _run_curve(
    *,
    capability: _ExecutionCapability,
    root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: pd.DataFrame,
    targets: SelectiveTargetAccessor,
    commitment: _BlindCommitmentLedger,
    features: np.ndarray,
    feature_columns: list[str],
    layout: Any,
    folds: list[dict[str, Any]],
    prefix_ids: dict[tuple[str, float], np.ndarray],
    comparator_parts: dict[tuple[str, float], Path],
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    _require_capability(capability, "BLIND_CURVE")
    if tuple(str(fold["name"]) for fold in folds) != FOLD_ORDER:
        raise PermissionError("Gen5r2 outer fold order differs")
    p1_config = shared.load_config(paths["base_config"])
    model_config = gen5._model_config(config, len(feature_columns), layout.group_count)
    training_config = gen5._training_config(config)
    working_train = train.copy()
    working_train["label"] = pd.Series(pd.NA, index=working_train.index, dtype="Int8")
    all_receipts: list[dict[str, Any]] = []
    primary_receipts: list[dict[str, Any]] = []
    teacher_receipts: list[dict[str, Any]] = []
    gate_receipts: list[dict[str, Any]] = []
    part_receipts: list[dict[str, Any]] = []
    cell_commitments: list[dict[str, Any]] = []
    completed_primary = 0
    for fold in folds:
        fold_name = str(fold["name"])
        validation_ids = np.asarray(fold["val_idx"], dtype=np.int64)
        if not np.array_equal(validation_ids, targets.validation_rows(fold_name)):
            raise PermissionError("Gen5r2 selective accessor validation IDs differ")
        for fraction in FRACTIONS:
            working_train["label"] = pd.Series(
                pd.NA, index=working_train.index, dtype="Int8"
            )
            fraction_tag = shared._tag(fraction)
            train_ids = prefix_ids[(fold_name, fraction)]
            if np.intersect1d(train_ids, validation_ids).size:
                raise PermissionError("Gen5r2 active train/validation IDs overlap")
            planned_splits = gen5.build_three_block_inner_splits(
                train.loc[:, ["time"]],
                train_ids,
                purge_days=int(config["inner_cross_fit"]["purge_days"]),
            )
            pre_gate_label_ids = np.unique(
                np.concatenate(
                    [
                        *(split.teacher_train_ids for split in planned_splits),
                        planned_splits[0].teacher_prediction_ids,
                        planned_splits[1].teacher_prediction_ids,
                    ]
                )
            )
            selected_labels = targets.labels_for(
                pre_gate_label_ids,
                commitment=commitment,
                purpose=f"curve_pre_gate_training:{fold_name}:{fraction_tag}",
                active_fold=fold_name,
            )
            label_column = working_train.columns.get_loc("label")
            working_train.iloc[pre_gate_label_ids, label_column] = selected_labels
            comparator = shared._comparator_frame(
                comparator_parts[(fold_name, fraction)], fold, fraction
            )
            prefix_features, prefix_feature_sha = build_exact_prefix_causal_matrix(
                train,
                train_ids,
                full_reference=features,
            )
            teacher, cell_teacher_receipts = gen5._teacher_oof(
                config=config,
                paths=paths,
                train=working_train,
                p1_config=p1_config,
                outer_prefix_ids=train_ids,
                outer_forbidden_ids=validation_ids,
                fold_name=fold_name,
                fold_ordinal=int(fold["ordinal"]),
                fraction_tag=fraction_tag,
                scope="curve",
            )
            teacher_receipts.extend(cell_teacher_receipts)
            all_receipts.extend(cell_teacher_receipts)
            splits = teacher["splits"]
            if any(
                left.block != right.block
                or not np.array_equal(left.teacher_train_ids, right.teacher_train_ids)
                or not np.array_equal(left.teacher_prediction_ids, right.teacher_prediction_ids)
                for left, right in zip(planned_splits, splits, strict=True)
            ):
                raise PermissionError("Gen5r2 planned and executed inner splits differ")
            residual_fit_ids = np.concatenate(
                [splits[0].teacher_prediction_ids, splits[1].teacher_prediction_ids]
            )
            gate_ids = splits[2].teacher_prediction_ids
            if working_train.iloc[gate_ids]["label"].notna().any():
                raise PermissionError("Gen5r2 gate labels were decoded before blind prediction")
            refit_ids = teacher["oof_ids"]
            outer_context = np.unique(np.concatenate([train_ids, validation_ids]))
            prediction_features = prefix_features.copy()
            prediction_features[validation_ids] = features[validation_ids]
            seed_final_probabilities: list[np.ndarray] = []
            seed_final_predictions: list[np.ndarray] = []
            seed_blind_pins: list[dict[str, Any]] = []
            cell_gate_passes: list[bool] = []
            gate_stages: dict[int, dict[str, Any]] = {}
            gate_forbidden = np.unique(np.concatenate([validation_ids, gate_ids]))
            residual_fit_labels = pd.to_numeric(
                working_train.iloc[residual_fit_ids]["label"], errors="raise"
            ).to_numpy(np.int8)
            for gate_seed in SEEDS:
                gate_started = time.perf_counter()
                gate_model = gen5.fit_incumbent_residual_model(
                    prefix_features,
                    layout,
                    residual_fit_ids,
                    residual_fit_labels,
                    teacher["seed_probability"][gate_seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][gate_seed],
                    context_ids=train_ids,
                    forbidden_ids=gate_forbidden,
                    seed=int(gate_seed) + int(fold["ordinal"]),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                )
                gate_probability = gen5.predict_incumbent_residual_probability(
                    gate_model,
                    prefix_features,
                    layout,
                    gate_ids,
                    teacher["seed_probability"][gate_seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][gate_seed],
                    context_ids=train_ids,
                    device="cuda",
                )
                gate_model_relative = (
                    f"gate_models/{fraction_tag}/{fold_name}/seed_{gate_seed}.pt"
                )
                gate_model_path = shared._safe_path(paths["artifact"], gate_model_relative)
                gen5.save_fitted_incumbent_residual_model(gate_model, gate_model_path)
                loaded_gate = gen5.load_fitted_incumbent_residual_model(gate_model_path)
                reproduced_gate = gen5.predict_incumbent_residual_probability(
                    loaded_gate,
                    prefix_features,
                    layout,
                    gate_ids,
                    teacher["seed_probability"][gate_seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][gate_seed],
                    context_ids=train_ids,
                    device="cuda",
                )
                if not np.array_equal(gate_probability, reproduced_gate):
                    raise RuntimeError("saved Gen5r2 gate residual did not reproduce")
                gate_blind_relative = (
                    f"gate_blind_predictions/{fraction_tag}/{fold_name}/seed_{gate_seed}.npy"
                )
                gate_blind_path = shared._safe_path(paths["artifact"], gate_blind_relative)
                gate_blind_sha = shared._npy_new(gate_blind_path, gate_probability)
                gate_residual_prediction = gen5._postprocess_ids(
                    working_train,
                    gate_ids,
                    gate_probability,
                    config["fixed_fold_postprocess"][fold_name],
                )
                gate_stages[gate_seed] = {
                    "model": gate_model,
                    "prediction": gate_residual_prediction,
                    "blind_relative": gate_blind_relative,
                    "blind_sha256": gate_blind_sha,
                    "model_relative": gate_model_relative,
                    "model_path": gate_model_path,
                    "elapsed_seconds": float(time.perf_counter() - gate_started),
                }
            if tuple(gate_stages) != SEEDS or any(
                not (paths["artifact"] / stage["blind_relative"]).is_file()
                or _sha(paths["artifact"] / stage["blind_relative"])
                != stage["blind_sha256"]
                for stage in gate_stages.values()
            ):
                raise PermissionError("Gen5r2 did not seal all three gate predictions")
            if working_train.iloc[gate_ids]["label"].notna().any():
                raise PermissionError("Gen5r2 gate labels changed before all blind seals")
            gate_truth = targets.labels_for(
                gate_ids,
                commitment=commitment,
                purpose=f"gate_after_three_blind_seals:{fold_name}:{fraction_tag}",
                active_fold=fold_name,
            )
            working_train.iloc[gate_ids, label_column] = gate_truth
            for seed in SEEDS:
                stage = gate_stages[seed]
                gate_model = stage["model"]
                gate_residual_prediction = stage["prediction"]
                gate_blind_relative = stage["blind_relative"]
                gate_blind_sha = stage["blind_sha256"]
                gate_model_relative = stage["model_relative"]
                gate_model_path = stage["model_path"]
                gate_result = gen5._gate_decision(
                    train=working_train,
                    gate_ids=gate_ids,
                    truth=gate_truth,
                    base_prediction=teacher["seed_decision"][seed][gate_ids],
                    residual_prediction=gate_residual_prediction,
                    config=config,
                )
                cell_gate_passes.append(bool(gate_result["passed"]))
                gate_receipt = {
                    "role": "residual_gate",
                    "fraction": fraction,
                    "fold": fold_name,
                    "seed": seed,
                    "train_ids_sha256": gate_model.train_ids_sha256,
                    "gate_ids_sha256": shared.ids_sha256(gate_ids),
                    "optimizer_steps": training_config.optimizer_steps,
                    "gate_prediction_sealed_before_gate_label_read": True,
                    "all_three_gate_predictions_sealed_before_gate_label_read": True,
                    "gate_blind_prediction_relative_path": gate_blind_relative,
                    "gate_blind_prediction_sha256": gate_blind_sha,
                    "model_relative_path": gate_model_relative,
                    "model_sha256": _sha(gate_model_path),
                    "model_state_sha256": gate_model.model_state_sha256,
                    "saved_model_reload_prediction_exact": True,
                    "gate": gate_result,
                    "outer_validation_target_reads": 0,
                    "elapsed_seconds": stage["elapsed_seconds"],
                    "test_value_reads": 0,
                }
                gate_receipts.append(gate_receipt)
                all_receipts.append(gate_receipt)

                refit_started = time.perf_counter()
                refit_labels = pd.to_numeric(
                    working_train.iloc[refit_ids]["label"], errors="raise"
                ).to_numpy(np.int8)
                refit_model = gen5.fit_incumbent_residual_model(
                    prefix_features,
                    layout,
                    refit_ids,
                    refit_labels,
                    teacher["seed_probability"][seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][seed],
                    context_ids=train_ids,
                    forbidden_ids=validation_ids,
                    seed=int(seed) + int(fold["ordinal"]),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                )
                n_rows = len(working_train)
                outer_seed = np.full(n_rows, 0.5, dtype=np.float32)
                outer_mean = np.full(n_rows, 0.5, dtype=np.float32)
                outer_std = np.zeros(n_rows, dtype=np.float32)
                outer_decision = np.zeros(n_rows, dtype=np.int8)
                seed_column = f"baseline__seed_{seed}__probability"
                decision_column = f"baseline__seed_{seed}__prediction"
                outer_seed[validation_ids] = comparator[seed_column].to_numpy(np.float32)
                outer_mean[validation_ids] = comparator["baseline_probability"].to_numpy(
                    np.float32
                )
                comparator_seed_matrix = np.column_stack(
                    [
                        comparator[
                            f"baseline__seed_{registered}__probability"
                        ].to_numpy(np.float32)
                        for registered in SEEDS
                    ]
                )
                outer_std[validation_ids] = comparator_seed_matrix.std(axis=1).astype(
                    np.float32
                )
                outer_decision[validation_ids] = comparator[decision_column].to_numpy(np.int8)
                residual_probability = gen5.predict_incumbent_residual_probability(
                    refit_model,
                    prediction_features,
                    layout,
                    validation_ids,
                    outer_seed,
                    outer_mean,
                    outer_std,
                    outer_decision,
                    context_ids=outer_context,
                    device="cuda",
                )
                model_relative = f"models/{fraction_tag}/{fold_name}/seed_{seed}.pt"
                model_path = shared._safe_path(paths["artifact"], model_relative)
                gen5.save_fitted_incumbent_residual_model(refit_model, model_path)
                loaded = gen5.load_fitted_incumbent_residual_model(model_path)
                reproduced = gen5.predict_incumbent_residual_probability(
                    loaded,
                    prediction_features,
                    layout,
                    validation_ids,
                    outer_seed,
                    outer_mean,
                    outer_std,
                    outer_decision,
                    context_ids=outer_context,
                    device="cuda",
                )
                reload_exact = bool(np.array_equal(residual_probability, reproduced))
                if not reload_exact:
                    raise RuntimeError("saved Gen5r2 refit did not reproduce blind probability")
                final_probability = gen5.exact_identity_or_residual(
                    comparator[seed_column].to_numpy(np.float32),
                    residual_probability,
                    gate_passed=bool(gate_result["passed"]),
                )
                if gate_result["passed"]:
                    final_prediction = shared.apply_postprocess(
                        working_train.iloc[validation_ids],
                        final_probability,
                        comparator["plateau"].to_numpy(bool),
                        comparator["spike_candidate"].to_numpy(bool),
                        config["fixed_fold_postprocess"][fold_name],
                    ).astype(np.int8)
                else:
                    final_prediction = comparator[decision_column].to_numpy(np.int8).copy()
                    if not (
                        np.array_equal(
                            final_probability,
                            comparator[seed_column].to_numpy(np.float32),
                        )
                        and np.array_equal(
                            final_prediction,
                            comparator[decision_column].to_numpy(np.int8),
                        )
                    ):
                        raise AssertionError("failed Gen5r2 gate changed incumbent seed")
                blind_relative = (
                    f"blind_predictions/{fraction_tag}/{fold_name}/seed_{seed}.npy"
                )
                blind_path = shared._safe_path(paths["artifact"], blind_relative)
                blind_sha = shared._npy_new(blind_path, final_probability)
                seed_blind_pins.append(
                    {"seed": seed, "path": blind_relative, "sha256": blind_sha}
                )
                seed_final_probabilities.append(final_probability)
                seed_final_predictions.append(final_prediction)
                completed_primary += 1
                primary_receipt = {
                    "role": "residual_refit",
                    "fraction": fraction,
                    "fold": fold_name,
                    "seed": seed,
                    "train_rows": int(len(refit_ids)),
                    "validation_rows": int(len(validation_ids)),
                    "optimizer_steps": training_config.optimizer_steps,
                    "raw_prefix_causal_feature_sha256": prefix_feature_sha,
                    "gate_passed": bool(gate_result["passed"]),
                    "failed_gate_exact_incumbent_identity": not gate_result["passed"],
                    "train_ids_sha256": refit_model.train_ids_sha256,
                    "validation_ids_sha256": shared.ids_sha256(validation_ids),
                    "model_relative_path": model_relative,
                    "model_sha256": _sha(model_path),
                    "model_state_sha256": refit_model.model_state_sha256,
                    "scaler_sha256": refit_model.scaler.state_sha256,
                    "blind_prediction_relative_path": blind_relative,
                    "blind_prediction_sha256": blind_sha,
                    "blind_prediction_sealed_before_validation_target_read": True,
                    "saved_model_reload_prediction_exact": reload_exact,
                    "elapsed_seconds": float(time.perf_counter() - refit_started),
                    "validation_target_reads": 0,
                    "test_value_reads": 0,
                }
                primary_receipts.append(primary_receipt)
                all_receipts.append(primary_receipt)
                shared._emit(
                    "gen5r2_primary_fit_complete",
                    completed=completed_primary,
                    total=45,
                    fraction=fraction,
                    fold=fold_name,
                    seed=seed,
                    gate_passed=bool(gate_result["passed"]),
                )
            part = comparator.copy()
            for seed, probability, prediction in zip(
                SEEDS,
                seed_final_probabilities,
                seed_final_predictions,
                strict=True,
            ):
                part[f"challenger__seed_{seed}__probability"] = probability
                part[f"challenger__seed_{seed}__prediction"] = prediction
            if not any(cell_gate_passes):
                mean_probability = comparator["baseline_probability"].to_numpy(
                    np.float32
                ).copy()
                mean_prediction = comparator["baseline_prediction"].to_numpy(np.int8).copy()
                if not (
                    np.array_equal(
                        mean_probability,
                        comparator["baseline_probability"].to_numpy(np.float32),
                    )
                    and np.array_equal(
                        mean_prediction,
                        comparator["baseline_prediction"].to_numpy(np.int8),
                    )
                ):
                    raise AssertionError("all-failed Gen5r2 gates changed incumbent ensemble")
            else:
                mean_probability = np.mean(
                    np.column_stack(seed_final_probabilities), axis=1
                ).astype(np.float32)
                mean_prediction = shared.apply_postprocess(
                    working_train.iloc[validation_ids],
                    mean_probability,
                    comparator["plateau"].to_numpy(bool),
                    comparator["spike_candidate"].to_numpy(bool),
                    config["fixed_fold_postprocess"][fold_name],
                ).astype(np.int8)
            part["challenger_probability"] = mean_probability
            part["challenger_prediction"] = mean_prediction
            part_relative = f"prediction_parts/{fold_name}_{fraction_tag}.parquet"
            part_path = shared._safe_path(paths["artifact"], part_relative)
            part_sha = shared._parquet_new(part_path, part)
            part_receipt = {
                "fraction": fraction,
                "fold": fold_name,
                "rows": int(len(part)),
                "path": part_relative,
                "sha256": part_sha,
                "gate_pass_count": int(sum(cell_gate_passes)),
                "key_order_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(
                        part.loc[:, [*shared.KEY_COLUMNS, "fold"]], index=False
                    ).to_numpy("<u8").tobytes()
                ).hexdigest(),
            }
            part_receipts.append(part_receipt)
            event = commitment.commit_cell(
                fold=fold_name,
                fraction=fraction,
                validation_ids_sha256=shared.ids_sha256(validation_ids),
                seed_blind_predictions=seed_blind_pins,
                prediction_part={
                    "path": part_relative,
                    "sha256": part_sha,
                    "key_order_sha256": part_receipt["key_order_sha256"],
                },
                target_accessor=targets,
            )
            cell_commitments.append(
                {"path": event["path"], "sha256": event["sha256"]}
            )
        if not commitment.is_fold_committed(fold_name):
            raise PermissionError("Gen5r2 fold did not reach its five-cell commitment")
    if not (
        len(teacher_receipts) == 135
        and len(gate_receipts) == 45
        and len(primary_receipts) == 45
        and len(all_receipts) == 225
        and sum(row["optimizer_steps"] for row in gate_receipts) == 5400
        and sum(row["optimizer_steps"] for row in primary_receipts) == 5400
        and len(part_receipts) == len(cell_commitments) == 15
    ):
        raise AssertionError("Gen5r2 curve fit, part, or optimizer-step count differs")
    ledger_receipt = commitment.finalize_ledger()
    completion = {
        "schema_version": "p1_incumbent_residual_predictions_complete.v5r2",
        "created_at": shared._now(),
        "fit_cells": 225,
        "teacher_fit_count": 135,
        "residual_gate_fit_count": 45,
        "residual_refit_count": 45,
        "optimizer_steps": 5400,
        "gate_optimizer_steps": 5400,
        "total_residual_optimizer_steps": 10800,
        "teacher_model_receipts": teacher_receipts,
        "gate_model_receipts": gate_receipts,
        "model_receipts": primary_receipts,
        "prediction_parts": part_receipts,
        "cell_commitments": cell_commitments,
        "fold_commitments": commitment.fold_receipts,
        "blind_commitment_ledger": {
            "path": ledger_receipt["path"],
            "sha256": ledger_receipt["sha256"],
        },
        "execution_order": "fold_major_then_prefix_fraction",
        "rolling_origin_reuse": (
            "earlier-fold validation labels become later-fold training history only after "
            "all five blind cells for the earlier fold are O_EXCL committed"
        ),
        "all_inner_and_outer_blind_predictions_sealed_before_their_target_reads": True,
        "aggregate_scores_computed_before_completion": 0,
        "frozen_oof_target_columns_decoded": 0,
        "train_anomaly_type_rows_decoded": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    completion_path = paths["artifact"] / "predictions_complete.json"
    _json_new(completion_path, completion)
    commitment.mark_predictions_complete(completion_path)
    capability.phase = "BLIND_COMMITTED"
    return all_receipts, completion, working_train


def _full_fit_models(
    *,
    capability: _ExecutionCapability,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: pd.DataFrame,
    features: np.ndarray,
    feature_columns: list[str],
    layout: Any,
) -> dict[str, Any]:
    _require_capability(capability, "FULL_FIT_AUTHORIZED")
    return gen5._full_fit_models(
        config=config,
        paths=paths,
        train=train,
        features=features,
        feature_columns=feature_columns,
        layout=layout,
    )


def _score_json_new(path: Path, value: Any) -> None:
    if isinstance(value, dict) and path.name == "learning_curve_evidence.json":
        value = {
            **value,
            "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT",
            "leakage_checks": {
                **value["leakage_checks"],
                "input_frame_loaded_without_label_or_anomaly_type": True,
                "frozen_oof_projected_to_keys_fold_prediction_only": True,
                "target_csv_indexed_as_opaque_byte_spans": True,
                "active_cell_validation_targets_undecoded_until_its_o_excl_commitment": True,
                "earlier_fold_validation_reused_only_after_all_five_fold_commitments": True,
                "aggregate_scoring_targets_undecoded_until_predictions_complete": True,
                "anomaly_type_scalars_never_decoded": True,
                "inner_teacher_fit_and_prediction_rows_disjoint": True,
                "inner_teacher_seven_day_purge_exact": True,
                "teacher_raw_rows_restricted_to_exact_outer_prefix": True,
                "residual_raw_context_future_half_zero_masked": True,
                "gate_predictions_sealed_before_gate_block_labels_read": True,
                "failed_gate_returns_exact_incumbent_seed_probability_and_prediction": True,
            },
            "reproducibility_checks": {
                **value["reproducibility_checks"],
                "fold_major_then_fraction_commitment_order_fixed": True,
                "fifteen_cell_and_three_fold_commitments_hash_chained": True,
                "predictions_complete_binds_all_forty_five_seed_blind_files": True,
                "fixed_135_teacher_45_gate_45_refit_curve_fits": True,
                "fixed_10800_total_residual_optimizer_steps": True,
                "all_teacher_and_residual_saved_models_reload_exact": True,
                "full_transitive_executable_closure_same_at_start_and_end": True,
            },
        }
    elif isinstance(value, dict) and path.name == "result.json":
        value = {
            **value,
            "operation_counters": {
                **value["operation_counters"],
                "curve_teacher_model_fits": 135,
                "curve_residual_gate_model_fits": 45,
                "curve_residual_refit_model_fits": 45,
                "curve_total_model_fits": 225,
                "curve_gate_optimizer_steps": 5400,
                "curve_refit_optimizer_steps": 5400,
                "curve_total_residual_optimizer_steps": 10800,
                "frozen_oof_target_columns_decoded": 0,
                "train_anomaly_type_rows_decoded": 0,
                "test_value_reads": 0,
                "uploads": 0,
            },
        }
    gen5._SHARED_JSON_NEW(path, value)


def _patch_shared_engine() -> None:
    shared.__file__ = str(Path(__file__).resolve())
    shared.EXPECTED_CONFIG_SHA256 = EXPECTED_CONFIG_SHA256
    shared.EXPECTED_CONFIG_DEEP_SHA256 = EXPECTED_CONFIG_DEEP_SHA256
    shared.CANONICAL_CONFIG = CANONICAL_CONFIG
    shared.CANONICAL_ARTIFACT = CANONICAL_ARTIFACT
    shared.CANONICAL_LOCK = CANONICAL_LOCK
    shared.HYPOTHESIS = HYPOTHESIS
    shared.FRACTIONS = FRACTIONS
    shared.SEEDS = SEEDS
    shared._paths = _paths
    shared._json_new = _score_json_new
    shared.evaluate_learning_curve = evaluate_learning_curve
    shared.load_contract = load_contract


def _artifact_hashes(artifact: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(artifact).as_posix(): {
            "sha256": _sha(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }


def _run_after_lock(
    *,
    root: Path,
    data_dir: Path,
    overlay: dict[str, Any],
    config: dict[str, Any],
    paths: dict[str, Path],
    closure_before: dict[str, str],
    qa_sha256: str,
    authorization_sha256: str,
    lock: dict[str, Any],
) -> dict[str, Any]:
    _verify_lock(paths["lock"], lock)
    if paths["artifact"].exists():
        raise FileExistsError("Gen5r2 artifact already exists")
    capability = _mint_capability(
        root=root,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
        lock=lock,
        closure=closure_before,
    )
    _patch_shared_engine()
    pins_before = shared._verify_input_pins(root, data_dir, config)
    paths["artifact"].mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    train = load_input_only_train(data_dir / "train.csv")
    if len(train) == 0 or {"label", "anomaly_type"}.intersection(train.columns):
        raise PermissionError("Gen5r2 input-only train load differs")
    feature_metadata = _strict_json(paths["feature_metadata"])
    feature_columns = [str(value) for value in config["features"]["selected_numeric_columns"]]
    if any(column not in feature_metadata["feature_columns"] for column in feature_columns):
        raise ValueError("registered Gen5r2 causal feature column is absent")
    feature_frame = pd.read_parquet(paths["feature_cache"], columns=feature_columns)
    if len(feature_frame) != len(train) or {"label", "anomaly_type"}.intersection(
        feature_frame.columns
    ):
        raise PermissionError("Gen5r2 feature cache row or target exclusion differs")
    features = feature_frame.to_numpy(np.float32)
    layout = shared.SequenceLayout.build(train.loc[:, ["station", "layer", "time"]])
    p1_config = shared.load_config(paths["base_config"])
    frozen_oof = load_frozen_oof_keys_only(paths["frozen_oof"])
    folds, scope_audit = shared.gen1._fold_runtime(
        train,
        p1_config,
        frozen_oof,
    )
    if tuple(str(fold["name"]) for fold in folds) != FOLD_ORDER:
        raise PermissionError("Gen5r2 frozen outer fold order differs")
    prefix_ids, prefix_audit = gen5.r4._pinned_label_free_prefixes(
        train,
        folds,
        int(config["features"]["cadence_minutes"]),
    )
    comparator_parts = shared._verify_gen1_parts(root, paths)
    validation_rows_by_fold = {
        str(fold["name"]): np.asarray(fold["val_idx"], dtype=np.int64) for fold in folds
    }
    targets = SelectiveTargetAccessor(
        data_dir / "train.csv",
        expected_sha256=config["immutable_inputs"]["train.csv"],
        expected_rows=len(train),
        validation_rows_by_fold=validation_rows_by_fold,
        fold_order=FOLD_ORDER,
    )
    if targets.decoded_target_scalars != 0:
        raise PermissionError("Gen5r2 target accessor decoded targets during indexing")
    _json_new(
        paths["artifact"] / "split_audit.json",
        {
            "folds": scope_audit,
            "prefixes": prefix_audit,
            "execution_order": "fold_major_then_prefix_fraction",
            "rolling_origin_reuse": True,
            "input_columns": list(train.columns),
            "input_target_columns_decoded": 0,
            "frozen_oof_columns": list(frozen_oof.columns),
            "frozen_oof_target_columns_decoded": 0,
            "target_accessor_opaque_index_rows": targets.row_count,
            "target_accessor_target_scalars_decoded_at_split_seal": 0,
        },
    )
    commitment = _BlindCommitmentLedger(paths["artifact"])
    receipts, completion, working_train = _run_curve(
        capability=capability,
        root=root,
        config=config,
        paths=paths,
        train=train,
        targets=targets,
        commitment=commitment,
        features=features,
        feature_columns=feature_columns,
        layout=layout,
        folds=folds,
        prefix_ids=prefix_ids,
        comparator_parts=comparator_parts,
    )
    _require_capability(capability, "BLIND_COMMITTED")
    full_ids = np.arange(len(train), dtype=np.int64)
    all_labels = targets.labels_for(
        full_ids,
        commitment=commitment,
        purpose="post_predictions_complete_scoring_and_optional_full_fit",
        active_fold=None,
        require_global=True,
    )
    working_train.iloc[:, working_train.columns.get_loc("label")] = all_labels
    if targets.decoded_anomaly_rows != 0:
        raise PermissionError("Gen5r2 decoded anomaly_type despite binary-only structure")
    target_audit = targets.audit()
    _json_new(paths["artifact"] / "selective_target_audit.json", target_audit)
    report, evidence, central = shared._score(
        root=root,
        config=config,
        paths=paths,
        train=working_train,
        frozen_oof=frozen_oof,
    )
    if central["passed"]:
        capability.phase = "FULL_FIT_AUTHORIZED"
        full_fit = _full_fit_models(
            capability=capability,
            config=config,
            paths=paths,
            train=working_train,
            features=features,
            feature_columns=feature_columns,
            layout=layout,
        )
        next_generation = None
    else:
        capability.phase = "SCORED_NO_PASS"
        full_fit = {
            "performed": False,
            "reason": "curve did not satisfy every preregistered meaningful-improvement gate",
            "model_count": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
        }
        next_generation = config["on_no_pass"]["exactly_one_next_structural_diagnosis"]
    closure_after = _verify_execution_closure(root, overlay)
    if closure_before != closure_after:
        raise RuntimeError("Gen5r2 executable closure changed during the run")
    pins_after = shared._verify_input_pins(root, data_dir, config)
    if pins_before != pins_after:
        raise RuntimeError("Gen5r2 immutable inputs changed during the run")
    _verify_lock(paths["lock"], lock)
    result = {
        "schema_version": "p1_incumbent_residual_result.v5r2",
        "experiment_id": overlay["experiment_id"],
        "completed_at": shared._now(),
        "status": (
            "CURVE_QUALIFIED_FULL_FIT_MODELS_SAVED"
            if central["passed"]
            else "RESEARCH_ONLY_NO_PASS"
        ),
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "hypothesis": HYPOTHESIS,
        "points": report["points"],
        "gate_checks": report["gate_checks"],
        "full_fit": full_fit,
        "exactly_one_next_structural_diagnosis": next_generation,
        "attempt": lock,
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "prediction_completion_sha256": _sha(
            paths["artifact"] / "predictions_complete.json"
        ),
        "blind_commitment_ledger_sha256": completion["blind_commitment_ledger"][
            "sha256"
        ],
        "selective_target_audit_sha256": _sha(
            paths["artifact"] / "selective_target_audit.json"
        ),
        "operation_counters": {
            "curve_model_fits": len(receipts),
            "curve_optimizer_steps": completion["optimizer_steps"],
            "full_fit_model_fits": int(full_fit["model_count"]),
            "target_fold_scores": len(FRACTIONS),
            "frozen_oof_target_columns_decoded": 0,
            "train_anomaly_type_rows_decoded": 0,
            "test_value_reads": 0,
            "test_prediction_generations": 0,
            "candidate_files": 0,
            "uploads": 0,
            "source_mutations": 0,
            "frozen_mutations": 0,
        },
        "protected_input_sha256_unchanged": True,
        "executable_closure_sha256_unchanged": True,
        "elapsed_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    _score_json_new(paths["artifact"] / "result.json", result)
    registry = {
        "schema_version": "p1_incumbent_residual_registry.v5r2",
        "experiment_id": overlay["experiment_id"],
        "registered_at": shared._now(),
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "hypothesis": HYPOTHESIS,
        "learning_curve_evidence_sha256": _sha(
            paths["artifact"] / "learning_curve_evidence.json"
        ),
        "canonical_decision_sha256": _sha(
            paths["artifact"] / "canonical_curve_decision.json"
        ),
        "result_sha256": _sha(paths["artifact"] / "result.json"),
        "full_fit_models": full_fit.get("models"),
        "candidate": None,
        "test_value_reads": 0,
        "uploads": 0,
    }
    _json_new(paths["artifact"] / "registry.json", registry)
    manifest = {
        "schema_version": "p1_incumbent_residual_manifest.v5r2",
        "experiment_id": overlay["experiment_id"],
        "created_at": shared._now(),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "runner_sha256": _sha(Path(__file__).resolve()),
        "execution_closure_before": closure_before,
        "execution_closure_after": closure_after,
        "attempt_lock_path": CANONICAL_LOCK,
        "attempt_lock_sha256": lock["sha256"],
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "immutable_inputs_before": pins_before,
        "immutable_inputs_after": pins_after,
        "source_cache_current_frozen_unchanged": True,
        "artifacts": _artifact_hashes(paths["artifact"]),
        "candidate_created": False,
        "uploaded": False,
    }
    _json_new(paths["artifact"] / "manifest.json", manifest)
    manifest_sha = _sha(paths["artifact"] / "manifest.json")
    with (paths["artifact"] / "manifest.sha256").open(
        "x", encoding="ascii", newline="\n"
    ) as handle:
        handle.write(f"{manifest_sha}  manifest.json\n")
    final = {
        "status": result["status"],
        "decision": central["decision"],
        "passed": bool(central["passed"]),
        "artifact": CANONICAL_ARTIFACT,
        "metrics_sha256": _sha(paths["artifact"] / "metrics.json"),
        "evidence_sha256": _sha(paths["artifact"] / "learning_curve_evidence.json"),
        "result_sha256": _sha(paths["artifact"] / "result.json"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": None,
        "test_value_reads": 0,
        "uploads": 0,
        "elapsed_seconds": result["elapsed_seconds"],
    }
    shared._emit("gen5r2_generation_complete", **final)
    return final


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    overlay, config, paths, closure = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=root / CANONICAL_CONFIG,
        requested_artifact=root / CANONICAL_ARTIFACT,
    )
    if paths["artifact"].exists() or paths["lock"].exists():
        raise FileExistsError("Gen5r2 artifact or attempt lock already exists")
    comparator = shared._verify_gen1_parts(root, paths)
    train = load_input_only_train(data_dir / "train.csv")
    frozen_oof = load_frozen_oof_keys_only(paths["frozen_oof"])
    p1_config = shared.load_config(paths["base_config"])
    folds, scope_audit = shared.gen1._fold_runtime(train, p1_config, frozen_oof)
    prefix_ids, prefix_audit = gen5.r4._pinned_label_free_prefixes(
        train,
        folds,
        int(config["features"]["cadence_minutes"]),
    )
    validation_rows_by_fold = {
        str(fold["name"]): np.asarray(fold["val_idx"], dtype=np.int64) for fold in folds
    }
    targets = SelectiveTargetAccessor(
        data_dir / "train.csv",
        expected_sha256=config["immutable_inputs"]["train.csv"],
        expected_rows=len(train),
        validation_rows_by_fold=validation_rows_by_fold,
        fold_order=FOLD_ORDER,
    )
    split_cells: list[dict[str, Any]] = []
    for fold in folds:
        fold_name = str(fold["name"])
        validation_ids = np.asarray(fold["val_idx"], dtype=np.int64)
        for fraction in FRACTIONS:
            train_ids = prefix_ids[(fold_name, fraction)]
            splits = gen5.build_three_block_inner_splits(
                train.loc[:, ["time"]],
                train_ids,
                purge_days=int(config["inner_cross_fit"]["purge_days"]),
            )
            if (
                len(splits) != 3
                or np.intersect1d(train_ids, validation_ids).size
                or any(
                    np.intersect1d(split.teacher_train_ids, split.teacher_prediction_ids).size
                    for split in splits
                )
            ):
                raise PermissionError("Gen5r2 static split firewall differs")
            split_cells.append(
                {
                    "fold": fold_name,
                    "fraction": fraction,
                    "prefix_rows": int(len(train_ids)),
                    "validation_rows": int(len(validation_ids)),
                    "inner_split_count": len(splits),
                    "active_validation_target_decodes": 0,
                }
            )
    if targets.decoded_target_scalars != 0 or targets.decoded_anomaly_rows != 0:
        raise PermissionError("Gen5r2 check-only decoded a target scalar")
    causal_audit = gen5._causal_feature_audit(data_dir=data_dir, paths=paths)
    ledger_binding = gen5._verify_v5_ledger_binding(root, config, paths["ledger"])
    direct_call_rejected = False
    try:
        _require_capability(None, "BLIND_CURVE")
    except PermissionError:
        direct_call_rejected = True
    if not direct_call_rejected:
        raise PermissionError("Gen5r2 forged capability was not rejected")
    required_control_absence = {
        "independent_qa_receipt": paths["qa_receipt"].exists(),
        "execution_authorization": paths["authorization"].exists(),
        "attempt_lock": paths["lock"].exists(),
    }
    return {
        "status": "CANONICAL_GEN5R2_CHECK_ONLY_PASS",
        "experiment_id": overlay["experiment_id"],
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "runner_sha256": _sha(Path(__file__).resolve()),
        "execution_closure_sha256": closure,
        "execution_closure_count": len(closure),
        "predecessor_tombstone": _verify_predecessor_tombstone(paths),
        "input_only_columns": list(train.columns),
        "input_target_columns_decoded": 0,
        "frozen_oof_columns": list(frozen_oof.columns),
        "frozen_oof_target_columns_decoded": 0,
        "opaque_target_index_rows": targets.row_count,
        "opaque_target_index_decoded_scalars": targets.decoded_target_scalars,
        "outer_fold_scope": scope_audit,
        "prefix_audit": prefix_audit,
        "split_cells": split_cells,
        "split_cell_count": len(split_cells),
        "execution_order": "fold_major_then_prefix_fraction",
        "planned_cell_commitments": 15,
        "planned_fold_commitments": 3,
        "planned_seed_blind_predictions": 45,
        "curve_fit_cells": 225,
        "curve_total_residual_optimizer_steps": 10800,
        "maximum_full_fit_fits_on_pass": 18,
        "gen1_comparator_parts": len(comparator),
        "causal_feature_audit": causal_audit,
        "v5_ledger_binding": ledger_binding,
        "canonical_capability_direct_call_rejected": direct_call_rejected,
        "control_files_present": required_control_absence,
        "artifact_absent": not paths["artifact"].exists(),
        "attempt_lock_absent": not paths["lock"].exists(),
        "model_fits": 0,
        "target_fold_scores": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    overlay, config, paths, closure = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=root / CANONICAL_CONFIG,
        requested_artifact=root / CANONICAL_ARTIFACT,
    )
    if paths["artifact"].exists():
        raise FileExistsError("Gen5r2 append-only artifact already exists")
    qa, qa_sha256 = _verify_independent_qa(root, overlay, paths)
    authorization, authorization_sha256 = _verify_execution_authorization(
        root,
        overlay,
        paths,
        qa_sha256=qa_sha256,
    )
    replayed_closure = _verify_execution_closure(root, overlay)
    if replayed_closure != closure:
        raise PermissionError("Gen5r2 closure changed before lock")
    lock = _acquire_lock(
        paths["lock"],
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
        closure=closure,
    )
    del qa, authorization
    return _run_after_lock(
        root=root,
        data_dir=data_dir,
        overlay=overlay,
        config=config,
        paths=paths,
        closure_before=closure,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
        lock=lock,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    if args.check_only:
        result = check_only(root=root, data_dir=data_dir)
    elif args.run:
        result = run_experiment(root=root, data_dir=data_dir)
    else:
        raise AssertionError("unreachable Gen5r2 mode")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
