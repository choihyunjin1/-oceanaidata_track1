"""Fail-closed contract for the P2 joint-hydrographic Layer-4 curve.

This module intentionally imports only the Python standard library.  Static
preflight may launch a disposable runtime probe, but neither importing this
module nor calling :func:`static_preflight` loads the numerical stack in the
calling process.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_RELATIVE = (
    "configs/experiments/p2_joint_hydrographic_multitask_layer4_execution_v1.json"
)
CONFIG_SHA256 = "d583320c5926f4d5e0d2857d5c56ae6685b83a66d06d61db4d1ba052b761b194"
STAGE = "P2_JOINT_HYDROGRAPHIC_MULTITASK_LAYER4_EXECUTION_V1"
MODE = "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
ENGINE_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_execution_v1"
ENGINE_RELATIVE = "src/p2_restore/joint_hydrographic_multitask_layer4_execution_v1.py"
IMPLEMENTATION_ROLES = {
    "CONFIG": CONFIG_RELATIVE,
    "GUARD": "src/p2_restore/joint_hydrographic_multitask_layer4_contract_v1.py",
    "ENGINE": ENGINE_RELATIVE,
    "RUNNER": "scripts/run_p2_joint_hydrographic_multitask_layer4_v1.py",
    "TESTS": "tests/test_p2_joint_hydrographic_multitask_layer4_execution_v1.py",
}
STAGE_A_ROLES = {
    "CONFIG",
    "MANIFEST",
    "SEAL",
    "ARCHITECTURE_MANIFEST",
    "TRAINING_RECIPE",
    "TRAINING_RECEIPT",
    "CURVE_METRICS",
    "OOF_040",
    "OOF_055",
    "OOF_070",
    "OOF_085",
    "OOF_100",
    "QA_RECEIPT",
    "AUTHORIZATION",
    "ATTEMPT_LOCK",
}
FRACTION_ROLES = {
    "0.4": "OOF_040",
    "0.55": "OOF_055",
    "0.7": "OOF_070",
    "0.85": "OOF_085",
    "1.0": "OOF_100",
}
NUMERICAL_PREFIXES = ("numpy", "pandas", "scipy", "sklearn", "torch")


class Layer4ContractError(ValueError):
    """Raised when the immutable curve contract or one of its pins drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def strict_json_object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Layer4ContractError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    def finite_only(value: str) -> Any:
        raise Layer4ContractError(f"non-finite JSON number in {path.name}: {value}")

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=unique,
            parse_constant=finite_only,
        )
    if not isinstance(value, dict):
        raise Layer4ContractError(f"JSON root must be an object: {path}")
    return value


def workspace_path(root: Path, relative: str | Path, *, must_exist: bool = True) -> Path:
    workspace = root.resolve(strict=True)
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts:
        raise Layer4ContractError("workspace path must be portable and relative")
    candidate = (workspace / child).resolve(strict=must_exist)
    if candidate != workspace and workspace not in candidate.parents:
        raise Layer4ContractError("workspace path escaped the canonical root")
    return candidate


def contained_path(output: Path, relative: str | Path, *, must_exist: bool = False) -> Path:
    stage = output.resolve(strict=True)
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts:
        raise Layer4ContractError("artifact path must be portable and relative")
    candidate = (stage / child).resolve(strict=must_exist)
    if candidate == stage or stage not in candidate.parents:
        raise Layer4ContractError("artifact path escaped the canonical stage")
    return candidate


def exclusive_bytes(path: Path, payload: bytes) -> None:
    """Write all bytes once using O_EXCL and O_BINARY where available."""

    if not isinstance(payload, bytes):
        raise TypeError("exclusive payload must already be immutable bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("exclusive full-write loop made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    exclusive_bytes(path, canonical_json_bytes(value) + b"\n")


def _pin(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise Layer4ContractError(f"{label} exact keys changed")


def _validate_pin(pin: Any, *, label: str) -> None:
    if not isinstance(pin, Mapping) or set(pin) != {"path", "sha256", "bytes"}:
        raise Layer4ContractError(f"{label} pin shape changed")
    path = pin["path"]
    if (
        not isinstance(path, str)
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or not _is_sha256(pin["sha256"])
        or not isinstance(pin["bytes"], int)
        or pin["bytes"] <= 0
    ):
        raise Layer4ContractError(f"{label} pin is invalid")


def _validate_pin_map(pins: Any, roles: set[str], *, label: str) -> None:
    if not isinstance(pins, Mapping) or set(pins) != roles:
        raise Layer4ContractError(f"{label} roles changed")
    for role, pin in pins.items():
        _validate_pin(pin, label=f"{label}:{role}")


def _frozen_prefix_rows() -> list[tuple[str, float, int, str, int]]:
    return [
        ("outer_2024_sep_oct", 0.4, 6129, "977ccdc9946969ac09e4ddebf4d23aa21db7afae0347b0bca4d57ae0d368edca", 18),
        ("outer_2024_sep_oct", 0.55, 8428, "f11c535aa26bb86bdbc69a42d2fd7861a14750cd39f3b18bdf4985b3481c4f59", 24),
        ("outer_2024_sep_oct", 0.7, 10726, "82e2683c158c606c5952051df7d172d8dff5af6ff03d78b6fb00254c18257ac2", 30),
        ("outer_2024_sep_oct", 0.85, 13024, "94e172a46caef0571383476c998ccd74c060be1bb47cb39b0ccf3696c8b6802e", 36),
        ("outer_2024_sep_oct", 1.0, 15322, "95454eb3e242dda58c4901bcef3acd5f0062a8ff908d1df70c4114773bbc7b35", 42),
        ("outer_2025_may_jun", 0.4, 12584, "daa35d3301fe57eccf1f24343ac3f2e8f1e12d39b3aa2315caa5421581ecadfe", 34),
        ("outer_2025_may_jun", 0.55, 17303, "e55693a7409a60ac90f6fb2f8b22c741707c4614de8aead50e84631d1f5c32d7", 47),
        ("outer_2025_may_jun", 0.7, 22022, "dffd622ff779eb7e643b2d89b68868e46c12f076097a8447572d062c692f09c6", 60),
        ("outer_2025_may_jun", 0.85, 26741, "c68b84ffa8a3eac89a11228eae4ef96eac1b0b546afec1d0e14a27332860ef77", 72),
        ("outer_2025_may_jun", 1.0, 31459, "75ad1de46507002cc9f56c58d431fa9f21cfea70157b531e0a82f3eb6a923e64", 85),
        ("outer_2025_jul_aug", 0.4, 15972, "d4445b19de8bed903cfb7f1b58d3cdaf74848ae07bd1ca6233064c371da7df28", 44),
        ("outer_2025_jul_aug", 0.55, 21961, "6b35645fae508afd1c79f9a72dc397b8eee359c2ff735523528895905d112686", 60),
        ("outer_2025_jul_aug", 0.7, 27951, "75ef670bb039b10526ed1fa6a4d3b79191f7c4aaed5b7833c900ec9c2bddf8b8", 75),
        ("outer_2025_jul_aug", 0.85, 33940, "f086fdafe66919d95fe07ca34ea6054ee09dfcf79bf76393ef116b4ce7541817", 91),
        ("outer_2025_jul_aug", 1.0, 39929, "b6bd5e3b6b9d7f94d79e7c5bbef2988993bd147300f4dc06993ca16484b03169", 108),
    ]


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p2_joint_hydrographic_multitask_layer4.execution.v1":
        raise Layer4ContractError("Layer-4 execution schema changed")
    if config.get("problem") != "P2" or config.get("comparison_mode") != MODE:
        raise Layer4ContractError("Layer-4 execution identity changed")
    false_flags = (
        "exact_official_incumbent_comparison",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
    )
    if any(config.get(name) is not False for name in false_flags):
        raise Layer4ContractError("research-only firewall changed")
    if (
        config.get("local_qualification_only") is not True
        or config.get("research_only") is not True
        or config.get("official_submission_count") != 0
    ):
        raise Layer4ContractError("local-only status changed")

    surface = config.get("scientific_surface")
    if not isinstance(surface, Mapping) or surface.get("scientific_changes_from_frozen_design") != 0:
        raise Layer4ContractError("frozen scientific surface changed")
    _exact_keys(
        surface,
        {"design", "pure_model", "pure_model_tests", "static_preflight", "execution_design", "scientific_changes_from_frozen_design"},
        label="scientific surface",
    )
    for role in ("design", "pure_model", "pure_model_tests", "static_preflight", "execution_design"):
        _validate_pin(surface[role], label=f"scientific surface:{role}")
    _validate_pin_map(config.get("source_pins"), {"PACKAGE_INIT", "DEEP_DATA", "FEATURES", "DATA", "CURVE_GATE"}, label="transitive source")
    if config.get("implementation_roles") != IMPLEMENTATION_ROLES:
        raise Layer4ContractError("implementation roles changed")

    reference = config.get("stage_a_reference")
    if not isinstance(reference, Mapping) or reference.get("identity") != "P2_ARCHITECTURE_MATCHED_STAGE_A_EXECUTION_V3":
        raise Layer4ContractError("Stage-A reference identity changed")
    _validate_pin_map(reference.get("artifacts"), STAGE_A_ROLES, label="Stage-A")
    expected_reference = {
        "seal_complete": True,
        "all_five_prefixes_sealed": True,
        "curve_cells": 45,
        "full_reference_rmse_c": 1.0109798870010898,
        "challenger_fit_or_score_count_before_seal": 0,
        "submission_predictions": 0,
        "uploads": 0,
    }
    if reference.get("expected") != expected_reference:
        raise Layer4ContractError("Stage-A expected state changed")

    expected_paths = {
        "config": CONFIG_RELATIVE,
        "output": "artifacts/p2_joint_hydrographic_multitask_layer4_execution_v1",
        "control": "artifacts/p2_joint_hydrographic_multitask_layer4_execution_v1_control",
        "pre_execution_qa": "artifacts/p2_joint_hydrographic_multitask_layer4_execution_v1_control/pre_execution_qa.json",
        "authorization": "artifacts/p2_joint_hydrographic_multitask_layer4_execution_v1_control/authorization.json",
        "attempt_lock": "artifacts/p2_joint_hydrographic_multitask_layer4_execution_v1_control/attempt.lock",
    }
    if config.get("canonical_paths") != expected_paths:
        raise Layer4ContractError("canonical paths changed")

    protocol = config.get("curve_protocol")
    expected_folds = ["outer_2024_sep_oct", "outer_2025_may_jun", "outer_2025_jul_aug"]
    if not isinstance(protocol, Mapping) or any(
        protocol.get(key) != expected
        for key, expected in {
            "fold_major_order": expected_folds,
            "prefix_fractions": [0.4, 0.55, 0.7, 0.85, 1.0],
            "seed_ids": [20260823, 20260824, 20260825],
            "cell_order": "FOLD_THEN_FRACTION_THEN_SEED",
            "embargo_days": 7,
            "fit_cells": 45,
            "predictions_per_fold_before_commitment": 15,
            "epochs_per_fit": 28,
            "batch_size": 12,
            "total_optimizer_steps": 6132,
            "bootstrap_replicates": 5000,
            "bootstrap_cluster": "KST_day",
            "bootstrap_seed": 20260823,
            "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
            "all_folds_committed_before_metric_truth": True,
        }.items()
    ):
        raise Layer4ContractError("curve protocol changed")
    folds = protocol.get("folds")
    if not isinstance(folds, list) or [fold.get("name") for fold in folds] != expected_folds:
        raise Layer4ContractError("outer fold order changed")

    observed_prefix = [
        (
            item.get("fold"),
            float(item.get("fraction")),
            item.get("timestamps"),
            item.get("timestamp_order_sha256"),
            item.get("training_chunks"),
        )
        for item in config.get("prefix_pins", [])
        if isinstance(item, Mapping)
    ]
    if observed_prefix != _frozen_prefix_rows():
        raise Layer4ContractError("frozen prefix pins changed")
    steps = sum(28 * ((row[4] + 11) // 12) * 3 for row in _frozen_prefix_rows())
    if steps != 6132:
        raise Layer4ContractError("optimizer-step derivation changed")

    model = config.get("model_and_training")
    if not isinstance(model, Mapping) or any(
        model.get(key) != expected
        for key, expected in {
            "hypothesis_id": "H1_JOINT_HYDROGRAPHIC_MULTITASK_TCN_LAYER4",
            "input_channels": 54,
            "hidden_width": 160,
            "dilations": [1, 2, 4, 8, 16, 32],
            "dropout": 0.05,
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "chunk_length": 512,
            "chunk_stride": 384,
            "gradient_clip_norm": 1.0,
            "vertical_difference_weight": 0.25,
            "cuda_bfloat16": True,
            "parameter_count": 1021602,
            "layer4_clip_c": [-5.0, 45.0],
            "layer2_and_layer3_from_exact_stage_a_seed_columns": True,
            "post_hoc_blend_router_or_projection": False,
        }.items()
    ):
        raise Layer4ContractError("model or training contract changed")

    policy = config.get("execution_policy")
    required_true = {
        "check_only_is_default",
        "fresh_static_preflight_before_qa_and_before_lock",
        "independent_p0_p1_zero_qa_required",
        "separate_curve_only_authorization_required",
        "persistent_o_excl_attempt_lock_before_engine_import",
        "post_lock_opaque_single_use_capability_required",
        "caller_supplied_qa_authorization_or_lock_hashes_ignored",
        "direct_fit_curve_or_truth_call_without_live_capability_rejected",
        "capability_replay_or_arbitrary_stage_rejected",
        "fold_major_15_predictions_then_o_excl_commitment",
        "previous_fold_truth_for_later_training_requires_verified_commitment",
        "all_45_predictions_and_three_fold_commitments_before_metric_truth",
        "all_binary_artifacts_bytesio_then_o_excl_o_binary_full_write",
        "exists_then_file_write_forbidden",
    }
    required_false = {
        "rerun_allowed",
        "resume_allowed",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
    }
    if (
        not isinstance(policy, Mapping)
        or any(policy.get(key) is not True for key in required_true)
        or any(policy.get(key) is not False for key in required_false)
        or policy.get("active_fold_target_temp_psal_scalar_decode_before_commitment") != 0
    ):
        raise Layer4ContractError("execution firewall changed")

    output = config.get("output_contract")
    if (
        not isinstance(output, Mapping)
        or output.get("per_cell")
        != {"model_bundle": "model.pt", "blind_prediction_array": "prediction.npy", "cell_receipt": "receipt.json"}
        or output.get("per_fold_commitment") != "fold_commitment.json"
        or set(output.get("aggregate", {}))
        != {"prediction_commitment", "learning_curve_oof", "metrics", "bootstrap_receipt", "learning_curve_evidence", "gate_decision", "training_receipt", "manifest", "manifest_sidecar", "seal"}
        or any(output.get(key) is not False for key in ("full_fit_performed", "candidate_generated", "test_prediction_generated", "upload_performed"))
    ):
        raise Layer4ContractError("output allowlist or firewall changed")

    data = config.get("data_contract")
    if (
        not isinstance(data, Mapping)
        or data.get("personal_absolute_path_in_code_or_config") is not False
        or data.get("observations_rows") != 789408
        or data.get("test_index_or_sample_submission_semantic_reads") != 0
        or set(data.get("source_pins", {})) != {"observations.csv", "README.md", "score.py"}
    ):
        raise Layer4ContractError("runtime-injected data contract changed")
    for name, pin in data["source_pins"].items():
        if set(pin) != {"sha256", "bytes"} or not _is_sha256(pin["sha256"]):
            raise Layer4ContractError(f"data pin changed: {name}")


def load_canonical_config(
    root: Path,
    requested_path: Path | None = None,
    *,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    canonical = workspace_path(workspace, CONFIG_RELATIVE)
    requested = (requested_path or canonical).resolve(strict=True)
    if requested != canonical:
        raise Layer4ContractError("only the canonical Layer-4 config is accepted")
    if sha256_file(canonical) != CONFIG_SHA256:
        raise Layer4ContractError("canonical Layer-4 config SHA changed")
    config = strict_json_object(canonical)
    validate_config(config)
    if supplied_config is not None and dict(supplied_config) != config:
        raise Layer4ContractError("supplied config fails full deep equality")
    return config


def implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    return {
        role: _pin(workspace_path(workspace, relative), workspace)
        for role, relative in IMPLEMENTATION_ROLES.items()
    }


def verify_pin_map(
    root: Path,
    pins: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    result: dict[str, dict[str, Any]] = {}
    for role, expected in pins.items():
        path = workspace_path(workspace, str(expected["path"]))
        observed = _pin(path, workspace)
        if observed != dict(expected):
            raise Layer4ContractError(f"{label} pin drift: {role}")
        result[str(role)] = observed
    return result


def stage_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        key: workspace_path(root, config["canonical_paths"][key], must_exist=False)
        for key in ("output", "control", "pre_execution_qa", "authorization", "attempt_lock")
    }


def verify_scientific_surface(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    registered = {
        key: value
        for key, value in config["scientific_surface"].items()
        if key != "scientific_changes_from_frozen_design"
    }
    observed = verify_pin_map(workspace, registered, label="scientific surface")
    design = strict_json_object(workspace_path(workspace, observed["design"]["path"]))
    execution_design = strict_json_object(
        workspace_path(workspace, observed["execution_design"]["path"])
    )
    static = strict_json_object(workspace_path(workspace, observed["static_preflight"]["path"]))
    cells = static.get("real_source_probe", {}).get("cells", [])
    normalized_cells = [
        (str(row[0]), float(row[1]), int(row[2]), str(row[3]))
        for row in cells
        if isinstance(row, list) and len(row) == 4
    ]
    expected_cells = [(fold, fraction, count, digest) for fold, fraction, count, digest, _ in _frozen_prefix_rows()]
    checks = {
        "design_schema": design.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.design.v2",
        "design_not_executable": design.get("status")
        == "DESIGN_ONLY_NOT_EXECUTABLE_NOT_PREREGISTERED",
        "design_hypothesis": design.get("hypothesis", {}).get("id")
        == "H1_JOINT_HYDROGRAPHIC_MULTITASK_TCN_LAYER4",
        "design_no_candidate": design.get("candidate_or_test_prediction_allowed") is False,
        "design_no_upload": design.get("upload_allowed") is False,
        "execution_design_schema": execution_design.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.execution_design.v1",
        "execution_design_surface": execution_design.get("scientific_surface", {}).get(
            "scientific_changes_from_design"
        )
        == 0,
        "execution_design_cells": execution_design.get("curve_contract", {}).get(
            "fold_major_cells"
        )
        == 45,
        "static_schema": static.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.static_preflight.v2",
        "static_prefixes": normalized_cells == expected_cells,
        "static_steps": static.get("real_source_probe", {}).get("exact_curve_optimizer_steps")
        == 6132,
        "static_blind": static.get("real_source_probe", {}).get(
            "all_15_active_fold_withheld_target_scalar_decodes"
        )
        == 0,
        "static_no_fit": static.get("quality", {}).get("actual_curve_fits") == 0,
        "static_no_score": static.get("quality", {}).get("actual_scores") == 0,
        "static_no_test": static.get("quality", {}).get("test_predictions") == 0,
        "static_no_upload": static.get("quality", {}).get("uploads") == 0,
    }
    if not all(checks.values()):
        raise Layer4ContractError(
            f"frozen scientific surface failed: {sorted(k for k, v in checks.items() if not v)}"
        )
    return {"pins": observed, "checks": checks}


def verify_stage_a_reference(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    pins = verify_pin_map(
        workspace,
        config["stage_a_reference"]["artifacts"],
        label="Stage-A reference",
    )
    seal = strict_json_object(workspace_path(workspace, pins["SEAL"]["path"]))
    manifest = strict_json_object(workspace_path(workspace, pins["MANIFEST"]["path"]))
    metrics = strict_json_object(workspace_path(workspace, pins["CURVE_METRICS"]["path"]))
    receipt = strict_json_object(workspace_path(workspace, pins["TRAINING_RECEIPT"]["path"]))
    expected = config["stage_a_reference"]["expected"]
    sealed_oof = seal.get("reference_oof_by_fraction")
    checks = {
        "seal_schema": seal.get("schema_version") == "p2_architecture_matched_reference.seal.v3",
        "seal_complete": seal.get("complete") is expected["seal_complete"],
        "seal_prefixes": seal.get("all_five_prefixes_sealed")
        is expected["all_five_prefixes_sealed"],
        "seal_challenger_zero": seal.get("challenger_import_fit_or_score_count_before_seal")
        == 0,
        "seal_no_promotion": seal.get("official_promotion_allowed") is False,
        "seal_no_upload": seal.get("upload_count") == 0,
        "manifest_schema": manifest.get("schema_version")
        == "p2_architecture_matched_reference.manifest.v3",
        "manifest_no_upload": manifest.get("uploads") == 0,
        "receipt_cells": len(receipt.get("cells", [])) == expected["curve_cells"],
        "receipt_no_submission": receipt.get("plan", {}).get("submission_predictions") == 0,
        "metrics_full": abs(
            float(metrics.get("points", [{}])[-1].get("prediction_mean_metric", float("nan")))
            - expected["full_reference_rmse_c"]
        )
        <= 1e-15,
        "seal_oof_roles": isinstance(sealed_oof, Mapping)
        and set(sealed_oof) == set(FRACTION_ROLES),
    }
    if checks["seal_oof_roles"]:
        checks["seal_oof_pins"] = all(
            sealed_oof[fraction]
            == {
                "path": Path(pins[role]["path"]).name,
                "sha256": pins[role]["sha256"],
                "bytes": pins[role]["bytes"],
            }
            for fraction, role in FRACTION_ROLES.items()
        )
    if not all(checks.values()):
        raise Layer4ContractError(
            f"Stage-A v3 binding failed: {sorted(k for k, v in checks.items() if not v)}"
        )
    return {"status": "PASS_EXACT_STAGE_A_V3_REFERENCE", "pins": pins, "checks": checks}


def _runtime_probe(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    script = r"""
import json, platform, sys
import numpy, pandas, scipy, sklearn, torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
print(json.dumps({
  "python": platform.python_version(),
  "numpy": numpy.__version__,
  "pandas": pandas.__version__,
  "scipy": scipy.__version__,
  "scikit_learn": sklearn.__version__,
  "torch": torch.__version__,
  "torch_cuda": torch.version.cuda,
  "cuda_available": torch.cuda.is_available(),
  "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
  "cudnn_benchmark": torch.backends.cudnn.benchmark,
  "cudnn_deterministic": torch.backends.cudnn.deterministic,
  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
}, sort_keys=True))
"""
    environment = os.environ.copy()
    source = str(workspace_path(root, "src"))
    prior = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not prior else os.pathsep.join((source, prior))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise Layer4ContractError(f"isolated runtime probe failed: {detail}")
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise Layer4ContractError("isolated runtime probe returned invalid JSON") from exc
    expected = dict(contract)
    gpu_contains = str(expected.pop("gpu_name_contains"))
    checks = {key: observed.get(key) == value for key, value in expected.items()}
    checks["gpu_name"] = gpu_contains in str(observed.get("gpu_name"))
    if not all(checks.values()):
        raise Layer4ContractError(
            f"runtime contract failed: {sorted(k for k, v in checks.items() if not v)}"
        )
    return observed


def _data_preflight(data_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    directory = data_dir.resolve(strict=True)
    if not directory.is_dir():
        raise Layer4ContractError("runtime P2 data directory is not a directory")
    observed: dict[str, dict[str, Any]] = {}
    for name, expected in config["data_contract"]["source_pins"].items():
        path = (directory / name).resolve(strict=True)
        if path.parent != directory or not path.is_file():
            raise Layer4ContractError(f"runtime data path escaped or is not a file: {name}")
        pin = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if pin != dict(expected):
            raise Layer4ContractError(f"runtime data pin changed: {name}")
        observed[name] = pin
    observations = (directory / "observations.csv").resolve(strict=True)
    with observations.open("rb") as stream:
        raw_header = stream.readline()
        rows = sum(1 for _ in stream)
    try:
        header = next(csv.reader([raw_header.decode("utf-8-sig").rstrip("\r\n")]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise Layer4ContractError("observations header cannot be decoded") from exc
    if header != config["data_contract"]["observations_header"]:
        raise Layer4ContractError("observations header changed")
    if rows != config["data_contract"]["observations_rows"]:
        raise Layer4ContractError("observations row count changed")
    return {
        "pins": observed,
        "observations_header": header,
        "observations_rows": rows,
        "raw_full_file_read_for_sha256_integrity_only": True,
        "target_temp_psal_scalar_fields_decoded_or_converted": 0,
        "test_index_or_sample_submission_semantic_reads": 0,
    }


def _loaded_numerical_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in NUMERICAL_PREFIXES)
    )


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify all immutable bytes without writing or importing the engine."""

    workspace = root.resolve(strict=True)
    before = set(_loaded_numerical_modules())
    config = load_canonical_config(
        workspace,
        requested_config,
        supplied_config=supplied_config,
    )
    scientific = verify_scientific_surface(workspace, config)
    sources = verify_pin_map(workspace, config["source_pins"], label="transitive source")
    stage_a = verify_stage_a_reference(workspace, config)
    data = _data_preflight(data_dir, config)
    runtime = _runtime_probe(workspace, config["runtime_contract"])
    implementation = implementation_pins(workspace)
    operational = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.operational_snapshot.v1",
        "config": {
            "path": CONFIG_RELATIVE,
            "sha256": CONFIG_SHA256,
            "bytes": workspace_path(workspace, CONFIG_RELATIVE).stat().st_size,
        },
        "scientific_surface_pins": scientific["pins"],
        "source_pins": sources,
        "implementation_pins": implementation,
        "stage_a_reference_pins": stage_a["pins"],
        "data_pins": data["pins"],
        "observations_schema": {
            "header": data["observations_header"],
            "rows": data["observations_rows"],
            "raw_integrity_read_only": True,
            "target_scalar_decodes": 0,
        },
        "prefix_pins": config["prefix_pins"],
        "runtime": runtime,
        "fit_cells": 45,
        "optimizer_steps": 6132,
        "candidate_or_test_prediction_allowed": False,
        "upload_allowed": False,
    }
    summary_sha = hashlib.sha256(canonical_json_bytes(operational)).hexdigest()
    after = set(_loaded_numerical_modules())
    paths = stage_paths(workspace, config)
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.static_preflight.v1",
        "status": "PASS_STATIC_IMPLEMENTATION_ONLY_AWAITING_QA_AUTHORIZATION",
        "stage": STAGE,
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "operational_snapshot": operational,
        "summary_sha256": summary_sha,
        "scientific_surface": scientific,
        "stage_a_reference": stage_a,
        "data_audit": data,
        "runtime_probe_isolated": True,
        "preflight_process_loaded_numerical_modules": sorted(after),
        "preflight_process_new_numerical_modules": sorted(after - before),
        "execution_engine_imported": ENGINE_MODULE in sys.modules,
        "canonical_path_state": {key: path.exists() for key, path in paths.items()},
        "files_written": 0,
        "qa_receipts_created": 0,
        "authorizations_created": 0,
        "attempt_locks_created": 0,
        "model_fits": 0,
        "predictions": 0,
        "scores": 0,
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }


def verify_pre_execution_qa(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    path = stage_paths(root, config)["pre_execution_qa"]
    if not path.is_file():
        raise PermissionError("independent Layer-4 P0/P1-zero QA receipt is missing")
    receipt = strict_json_object(path)
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "decision",
        "p0_count",
        "p1_count",
        "config",
        "scientific_surface_pins",
        "stage_a_reference_pins",
        "source_pins",
        "implementation_pins",
        "reviewer",
        "notes",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
    }
    _exact_keys(receipt, expected_keys, label="QA receipt")
    contract = config["qa_receipt_contract"]
    checks = {
        "schema": receipt.get("schema_version") == contract["schema_version"],
        "decision": receipt.get("decision") == contract["decision"],
        "p0": receipt.get("p0_count") == 0,
        "p1": receipt.get("p1_count") == 0,
        "config": receipt.get("config") == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "scientific": receipt.get("scientific_surface_pins")
        == {key: value for key, value in config["scientific_surface"].items() if key != "scientific_changes_from_frozen_design"},
        "stage_a": receipt.get("stage_a_reference_pins")
        == config["stage_a_reference"]["artifacts"],
        "sources": receipt.get("source_pins") == config["source_pins"],
        "implementation": receipt.get("implementation_pins") == implementation_pins(root),
        "roles": set(receipt.get("implementation_pins", {})) == set(IMPLEMENTATION_ROLES),
        "reviewer": bool(receipt.get("reviewer")),
        "notes": isinstance(receipt.get("notes"), list),
        "no_promotion": receipt.get("official_promotion_allowed") is False,
        "no_candidate": receipt.get("candidate_or_test_prediction_allowed") is False,
        "no_upload": receipt.get("upload_allowed") is False,
    }
    if not all(checks.values()):
        raise PermissionError(f"Layer-4 QA failed: {sorted(k for k, v in checks.items() if not v)}")
    return receipt, sha256_file(path)


def verify_execution_authorization(
    root: Path,
    config: Mapping[str, Any],
    *,
    require_unconsumed: bool = True,
    require_output_absent: bool = True,
) -> tuple[dict[str, Any], str]:
    paths = stage_paths(root, config)
    if require_output_absent and paths["output"].exists():
        raise FileExistsError("append-only Layer-4 output already exists")
    if require_unconsumed and paths["attempt_lock"].exists():
        raise FileExistsError("one-shot Layer-4 attempt is already consumed")
    qa, qa_sha = verify_pre_execution_qa(root, config)
    del qa
    if not paths["authorization"].is_file():
        raise PermissionError("separate curve-only Layer-4 authorization is missing")
    authorization = strict_json_object(paths["authorization"])
    expected_keys = {
        "schema_version",
        "created_at_kst",
        "stage",
        "config",
        "authorization",
        "user_message_reference",
        "qa_receipt",
        "scientific_design",
        "implementation_pins",
        "execution_authorized",
        "curve_only",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "upload_allowed",
    }
    _exact_keys(authorization, expected_keys, label="authorization")
    contract = config["authorization_contract"]
    checks = {
        "schema": authorization.get("schema_version") == contract["schema_version"],
        "stage": authorization.get("stage") == STAGE,
        "config": authorization.get("config") == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "phrase": authorization.get("authorization")
        == contract["authorization_phrase_prefix"] + CONFIG_SHA256,
        "user_reference": bool(authorization.get("user_message_reference")),
        "qa": authorization.get("qa_receipt")
        == {"path": config["canonical_paths"]["pre_execution_qa"], "sha256": qa_sha},
        "design": authorization.get("scientific_design") == config["scientific_surface"]["design"],
        "implementation": authorization.get("implementation_pins") == implementation_pins(root),
        "execution": authorization.get("execution_authorized") is True,
        "curve_only": authorization.get("curve_only") is True,
        "no_promotion": authorization.get("official_promotion_allowed") is False,
        "no_candidate": authorization.get("candidate_or_test_prediction_allowed") is False,
        "no_upload": authorization.get("upload_allowed") is False,
    }
    if not all(checks.values()):
        raise PermissionError(
            f"Layer-4 authorization failed: {sorted(k for k, v in checks.items() if not v)}"
        )
    return authorization, sha256_file(paths["authorization"])


def _lock_payload(
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    qa_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.attempt_lock.v1",
        "stage": STAGE,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "preflight_summary_sha256": preflight["summary_sha256"],
        "qa_receipt_sha256": qa_sha256,
        "authorization_sha256": authorization_sha256,
        "scientific_design": config["scientific_surface"]["design"],
        "implementation_pins": preflight["operational_snapshot"]["implementation_pins"],
        "source_pins": config["source_pins"],
        "stage_a_reference_pins": config["stage_a_reference"]["artifacts"],
        "canonical_output": config["canonical_paths"]["output"],
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "curve_only": True,
        "official_promotion_allowed": False,
        "candidate_or_test_prediction_allowed": False,
        "rerun_allowed": False,
        "resume_allowed": False,
        "upload_allowed": False,
    }


def consume_attempt_lock(root: Path, data_dir: Path, config: Mapping[str, Any]) -> Path:
    """Consume the canonical attempt without accepting caller-provided hashes."""

    workspace = root.resolve(strict=True)
    canonical = load_canonical_config(workspace, supplied_config=config)
    preflight = static_preflight(workspace, data_dir, supplied_config=canonical)
    _qa, qa_sha = verify_pre_execution_qa(workspace, canonical)
    _authorization, authorization_sha = verify_execution_authorization(workspace, canonical)
    paths = stage_paths(workspace, canonical)
    if paths["output"].exists():
        raise FileExistsError("append-only Layer-4 output already exists")
    paths["control"].mkdir(parents=True, exist_ok=True)
    exclusive_json(
        paths["attempt_lock"],
        _lock_payload(
            canonical,
            preflight,
            qa_sha256=qa_sha,
            authorization_sha256=authorization_sha,
        ),
    )
    observed = strict_json_object(paths["attempt_lock"])
    expected = _lock_payload(
        canonical,
        preflight,
        qa_sha256=qa_sha,
        authorization_sha256=authorization_sha,
    )
    if observed != expected:
        raise PermissionError("new Layer-4 attempt lock failed full deep equality")
    return paths["attempt_lock"]


def verify_consumed_attempt_lock(
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    workspace = root.resolve(strict=True)
    canonical = load_canonical_config(workspace, supplied_config=config)
    preflight = static_preflight(workspace, data_dir, supplied_config=canonical)
    _qa, qa_sha = verify_pre_execution_qa(workspace, canonical)
    _authorization, authorization_sha = verify_execution_authorization(
        workspace,
        canonical,
        require_unconsumed=False,
        require_output_absent=False,
    )
    path = stage_paths(workspace, canonical)["attempt_lock"]
    if not path.is_file():
        raise PermissionError("consumed Layer-4 attempt lock is missing")
    observed = strict_json_object(path)
    expected = _lock_payload(
        canonical,
        preflight,
        qa_sha256=qa_sha,
        authorization_sha256=authorization_sha,
    )
    if observed != expected:
        raise PermissionError("consumed Layer-4 attempt lock fails full deep equality")
    return observed, sha256_file(path), preflight


@dataclass(frozen=True)
class ExecutionCapability:
    """Opaque by identity: constructed lookalikes are rejected by every gate."""

    root: str
    config_sha256: str
    preflight_summary_sha256: str
    attempt_lock_sha256: str
    qa_sha256: str
    authorization_sha256: str
    implementation_pins_sha256: str
    nonce: str


_LIVE_CAPABILITY: ExecutionCapability | None = None
_LIVE_PHASE: str | None = None
_COMPLETED_CELLS: list[tuple[str, float, int]] = []
_ACTIVE_CELL: tuple[str, float, int] | None = None
_FOLD_COMMITMENTS: list[dict[str, Any]] = []
_AGGREGATE_COMMITMENT: dict[str, Any] | None = None


def _cell_order(config: Mapping[str, Any]) -> list[tuple[str, float, int]]:
    return [
        (str(fold), float(fraction), int(seed))
        for fold in config["curve_protocol"]["fold_major_order"]
        for fraction in config["curve_protocol"]["prefix_fractions"]
        for seed in config["curve_protocol"]["seed_ids"]
    ]


def issue_execution_capability(
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
) -> tuple[ExecutionCapability, dict[str, Any]]:
    """Mint only after a fresh deep verification of the consumed lock."""

    global _LIVE_CAPABILITY, _LIVE_PHASE, _COMPLETED_CELLS, _ACTIVE_CELL
    global _FOLD_COMMITMENTS, _AGGREGATE_COMMITMENT
    if _LIVE_CAPABILITY is not None or _LIVE_PHASE is not None:
        raise PermissionError("a canonical Layer-4 capability is already live")
    workspace = root.resolve(strict=True)
    canonical = load_canonical_config(workspace, supplied_config=config)
    lock, lock_sha, preflight = verify_consumed_attempt_lock(workspace, data_dir, canonical)
    pins_sha = hashlib.sha256(
        canonical_json_bytes(preflight["operational_snapshot"]["implementation_pins"])
    ).hexdigest()
    nonce = hashlib.sha256(
        canonical_json_bytes(
            {
                "pid": os.getpid(),
                "root": str(workspace),
                "lock": lock_sha,
                "preflight": preflight["summary_sha256"],
                "qa": lock["qa_receipt_sha256"],
                "authorization": lock["authorization_sha256"],
            }
        )
    ).hexdigest()
    capability = ExecutionCapability(
        root=str(workspace),
        config_sha256=CONFIG_SHA256,
        preflight_summary_sha256=preflight["summary_sha256"],
        attempt_lock_sha256=lock_sha,
        qa_sha256=lock["qa_receipt_sha256"],
        authorization_sha256=lock["authorization_sha256"],
        implementation_pins_sha256=pins_sha,
        nonce=nonce,
    )
    _LIVE_CAPABILITY = capability
    _LIVE_PHASE = "LOCK_VERIFIED_CAPABILITY_MINTED"
    _COMPLETED_CELLS = []
    _ACTIVE_CELL = None
    _FOLD_COMMITMENTS = []
    _AGGREGATE_COMMITMENT = None
    return capability, preflight


def _require_core(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
) -> ExecutionCapability:
    if capability is not _LIVE_CAPABILITY or not isinstance(capability, ExecutionCapability):
        raise PermissionError("canonical live post-lock Layer-4 capability is required")
    workspace = root.resolve(strict=True)
    canonical = load_canonical_config(workspace, supplied_config=config)
    paths = stage_paths(workspace, canonical)
    pins_sha = hashlib.sha256(canonical_json_bytes(implementation_pins(workspace))).hexdigest()
    if (
        capability.root != str(workspace)
        or capability.config_sha256 != CONFIG_SHA256
        or capability.implementation_pins_sha256 != pins_sha
        or not paths["attempt_lock"].is_file()
        or sha256_file(paths["attempt_lock"]) != capability.attempt_lock_sha256
        or not paths["pre_execution_qa"].is_file()
        or sha256_file(paths["pre_execution_qa"]) != capability.qa_sha256
        or not paths["authorization"].is_file()
        or sha256_file(paths["authorization"]) != capability.authorization_sha256
    ):
        raise PermissionError("forged or stale Layer-4 capability")
    lock = strict_json_object(paths["attempt_lock"])
    if (
        lock.get("qa_receipt_sha256") != capability.qa_sha256
        or lock.get("authorization_sha256") != capability.authorization_sha256
        or lock.get("preflight_summary_sha256") != capability.preflight_summary_sha256
        or lock.get("candidate_or_test_prediction_allowed") is not False
        or lock.get("upload_allowed") is not False
    ):
        raise PermissionError("Layer-4 capability no longer binds the consumed lock")
    return capability


def begin_execution(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    global _LIVE_PHASE
    _require_core(capability, root=root, config=config)
    if (
        _LIVE_PHASE != "LOCK_VERIFIED_CAPABILITY_MINTED"
        or preflight.get("summary_sha256") != capability.preflight_summary_sha256
        or preflight.get("operational_snapshot", {}).get("implementation_pins")
        != implementation_pins(root)
    ):
        raise PermissionError("fresh preflight does not authorize execution entry")
    if stage_paths(root, config)["output"].exists():
        raise FileExistsError("append-only Layer-4 output already exists")
    _LIVE_PHASE = "EXECUTION_ACTIVE"


def claim_cell(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
    fold: str,
    fraction: float,
    seed: int,
) -> None:
    global _LIVE_PHASE, _ACTIVE_CELL
    _require_core(capability, root=root, config=config)
    if _LIVE_PHASE != "EXECUTION_ACTIVE" or _ACTIVE_CELL is not None:
        raise PermissionError("Layer-4 cell capability phase is unavailable")
    order = _cell_order(config)
    if len(_COMPLETED_CELLS) >= len(order):
        raise PermissionError("all registered Layer-4 cells were already consumed")
    expected = order[len(_COMPLETED_CELLS)]
    requested = (str(fold), float(fraction), int(seed))
    if requested != expected:
        raise PermissionError("arbitrary or replayed Layer-4 cell order is rejected")
    fold_index = config["curve_protocol"]["fold_major_order"].index(str(fold))
    if len(_FOLD_COMMITMENTS) != fold_index:
        raise PermissionError("prior fold commitment is not verified")
    _ACTIVE_CELL = requested
    _LIVE_PHASE = "CELL_CLAIMED"


def complete_cell(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
    fold: str,
    fraction: float,
    seed: int,
    artifact_pins: Mapping[str, Mapping[str, Any]],
) -> None:
    global _LIVE_PHASE, _ACTIVE_CELL
    _require_core(capability, root=root, config=config)
    requested = (str(fold), float(fraction), int(seed))
    if _LIVE_PHASE != "CELL_CLAIMED" or _ACTIVE_CELL != requested:
        raise PermissionError("Layer-4 cell completion is stale or out of order")
    if set(artifact_pins) != {"model_bundle", "blind_prediction_array", "cell_receipt"}:
        raise PermissionError("Layer-4 cell artifacts changed")
    output = stage_paths(root, config)["output"].resolve(strict=True)
    token = {0.4: "040", 0.55: "055", 0.7: "070", 0.85: "085", 1.0: "100"}[
        float(fraction)
    ]
    registered_names = config["output_contract"]["per_cell"]
    for role, pin in artifact_pins.items():
        _validate_pin(pin, label=f"cell artifact:{role}")
        path = workspace_path(root, str(pin["path"]))
        expected = output / "cells" / str(fold) / f"fraction_{token}" / f"seed_{seed}" / registered_names[role]
        if path != expected or _pin(path, root.resolve(strict=True)) != dict(pin):
            raise PermissionError(f"Layer-4 cell artifact pin failed: {role}")
    _COMPLETED_CELLS.append(requested)
    _ACTIVE_CELL = None
    _LIVE_PHASE = "EXECUTION_ACTIVE"


def claim_fold_commitment(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
    fold: str,
) -> None:
    global _LIVE_PHASE
    _require_core(capability, root=root, config=config)
    folds = config["curve_protocol"]["fold_major_order"]
    index = len(_FOLD_COMMITMENTS)
    if (
        _LIVE_PHASE != "EXECUTION_ACTIVE"
        or index >= len(folds)
        or str(fold) != folds[index]
        or len(_COMPLETED_CELLS) != (index + 1) * 15
    ):
        raise PermissionError("fold commitment cannot precede all 15 registered cells")
    _LIVE_PHASE = "FOLD_COMMITMENT_CLAIMED"


def complete_fold_commitment(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
    fold: str,
    commitment_pin: Mapping[str, Any],
) -> None:
    global _LIVE_PHASE
    _require_core(capability, root=root, config=config)
    folds = config["curve_protocol"]["fold_major_order"]
    if _LIVE_PHASE != "FOLD_COMMITMENT_CLAIMED" or str(fold) != folds[len(_FOLD_COMMITMENTS)]:
        raise PermissionError("fold commitment completion is stale")
    _validate_pin(commitment_pin, label="fold commitment")
    path = workspace_path(root, str(commitment_pin["path"]))
    expected = (
        stage_paths(root, config)["output"]
        / "folds"
        / str(fold)
        / config["output_contract"]["per_fold_commitment"]
    )
    if path != expected or _pin(path, root.resolve(strict=True)) != dict(commitment_pin):
        raise PermissionError("fold commitment pin failed")
    _FOLD_COMMITMENTS.append(dict(commitment_pin))
    _LIVE_PHASE = "EXECUTION_ACTIVE"


def claim_aggregate_commitment(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
) -> None:
    global _LIVE_PHASE
    _require_core(capability, root=root, config=config)
    if (
        _LIVE_PHASE != "EXECUTION_ACTIVE"
        or len(_COMPLETED_CELLS) != 45
        or len(_FOLD_COMMITMENTS) != 3
    ):
        raise PermissionError("aggregate commitment requires 45 cells and three folds")
    _LIVE_PHASE = "AGGREGATE_COMMITMENT_CLAIMED"


def complete_aggregate_commitment(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
    commitment_pin: Mapping[str, Any],
) -> None:
    global _LIVE_PHASE, _AGGREGATE_COMMITMENT
    _require_core(capability, root=root, config=config)
    if _LIVE_PHASE != "AGGREGATE_COMMITMENT_CLAIMED" or _AGGREGATE_COMMITMENT is not None:
        raise PermissionError("aggregate commitment capability was consumed")
    _validate_pin(commitment_pin, label="aggregate commitment")
    path = workspace_path(root, str(commitment_pin["path"]))
    expected = (
        stage_paths(root, config)["output"]
        / config["output_contract"]["aggregate"]["prediction_commitment"]
    )
    if path != expected or _pin(path, root.resolve(strict=True)) != dict(commitment_pin):
        raise PermissionError("aggregate commitment pin failed")
    _AGGREGATE_COMMITMENT = dict(commitment_pin)
    _LIVE_PHASE = "BLIND_CURVE_COMMITTED"


def claim_truth_and_score_phase(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
) -> None:
    global _LIVE_PHASE
    _require_core(capability, root=root, config=config)
    if _LIVE_PHASE != "BLIND_CURVE_COMMITTED" or _AGGREGATE_COMMITMENT is None:
        raise PermissionError("metric truth is unavailable before aggregate commitment")
    path = workspace_path(root, str(_AGGREGATE_COMMITMENT["path"]))
    if _pin(path, root.resolve(strict=True)) != _AGGREGATE_COMMITMENT:
        raise PermissionError("aggregate commitment changed before truth access")
    _LIVE_PHASE = "TRUTH_AND_SCORE_PHASE_CONSUMED"


def complete_execution_phase(
    capability: ExecutionCapability | object,
    *,
    root: Path,
    config: Mapping[str, Any],
) -> None:
    global _LIVE_PHASE
    _require_core(capability, root=root, config=config)
    if _LIVE_PHASE != "TRUTH_AND_SCORE_PHASE_CONSUMED":
        raise PermissionError("Layer-4 execution completion is out of order")
    _LIVE_PHASE = "COMPLETE"


def revoke_execution_capability(capability: ExecutionCapability | object) -> None:
    global _LIVE_CAPABILITY, _LIVE_PHASE, _COMPLETED_CELLS, _ACTIVE_CELL
    global _FOLD_COMMITMENTS, _AGGREGATE_COMMITMENT
    if capability is not _LIVE_CAPABILITY:
        raise PermissionError("cannot revoke a noncanonical Layer-4 capability")
    _LIVE_CAPABILITY = None
    _LIVE_PHASE = None
    _COMPLETED_CELLS = []
    _ACTIVE_CELL = None
    _FOLD_COMMITMENTS = []
    _AGGREGATE_COMMITMENT = None


def expected_output_files(config: Mapping[str, Any]) -> set[str]:
    files: set[str] = set()
    cell = config["output_contract"]["per_cell"]
    for fold in config["curve_protocol"]["fold_major_order"]:
        for fraction in config["curve_protocol"]["prefix_fractions"]:
            token = {0.4: "040", 0.55: "055", 0.7: "070", 0.85: "085", 1.0: "100"}[float(fraction)]
            for seed in config["curve_protocol"]["seed_ids"]:
                base = f"cells/{fold}/fraction_{token}/seed_{seed}"
                files.update(f"{base}/{name}" for name in cell.values())
        files.add(f"folds/{fold}/{config['output_contract']['per_fold_commitment']}")
    files.update(config["output_contract"]["aggregate"].values())
    return files


def expected_output_directories(config: Mapping[str, Any]) -> set[str]:
    directories: set[str] = set()
    for relative in expected_output_files(config):
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _relative_pin(path: Path, output: Path) -> dict[str, Any]:
    return {"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _npy_little_endian_float64_payload(path: Path) -> tuple[bytes, int]:
    """Read the exact raw vector payload emitted by np.save without importing NumPy."""

    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise Layer4ContractError("blind prediction is not an NPY artifact")
        version = stream.read(2)
        if version == b"\x01\x00":
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise Layer4ContractError("blind prediction NPY header is truncated")
            header_length = struct.unpack("<H", length_bytes)[0]
            encoding = "latin1"
        elif version in {b"\x02\x00", b"\x03\x00"}:
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise Layer4ContractError("blind prediction NPY header is truncated")
            header_length = struct.unpack("<I", length_bytes)[0]
            encoding = "utf-8" if version == b"\x03\x00" else "latin1"
        else:
            raise Layer4ContractError("blind prediction NPY version changed")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise Layer4ContractError("blind prediction NPY header is truncated")
        try:
            header = ast.literal_eval(header_bytes.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            raise Layer4ContractError("blind prediction NPY header is invalid") from exc
        if (
            not isinstance(header, dict)
            or set(header) != {"descr", "fortran_order", "shape"}
            or header["descr"] != "<f8"
            or header["fortran_order"] is not False
            or not isinstance(header["shape"], tuple)
            or len(header["shape"]) != 1
            or not isinstance(header["shape"][0], int)
            or isinstance(header["shape"][0], bool)
            or header["shape"][0] <= 0
        ):
            raise Layer4ContractError("blind prediction NPY contract changed")
        payload = stream.read()
    rows = int(header["shape"][0])
    if len(payload) != rows * 8:
        raise Layer4ContractError("blind prediction NPY payload size changed")
    if not all(math.isfinite(value[0]) for value in struct.iter_unpack("<d", payload)):
        raise Layer4ContractError("blind prediction NPY payload is non-finite")
    return payload, rows


def _csv_header_and_rows(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise Layer4ContractError("sealed CSV is empty") from exc
        rows = sum(1 for _ in reader)
    return header, rows


def _verify_prediction_commitment_graph(
    workspace: Path,
    output: Path,
    config: Mapping[str, Any],
    commitment: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the complete cell/fold/aggregate commitment graph from disk."""

    expected_order = [
        (str(fold), float(fraction), int(seed))
        for fold in config["curve_protocol"]["fold_major_order"]
        for fraction in config["curve_protocol"]["prefix_fractions"]
        for seed in config["curve_protocol"]["seed_ids"]
    ]
    prefix_pins = {
        (str(item["fold"]), float(item["fraction"])): item
        for item in config["prefix_pins"]
    }
    expected_fold_pins: list[dict[str, Any]] = []
    aggregate_cells: list[dict[str, Any]] = []
    rows_by_cell: list[int] = []
    values_hasher = hashlib.sha256()
    per_cell_names = config["output_contract"]["per_cell"]
    fraction_tokens = {0.4: "040", 0.55: "055", 0.7: "070", 0.85: "085", 1.0: "100"}

    for fold_index, fold in enumerate(config["curve_protocol"]["fold_major_order"]):
        fold = str(fold)
        fold_path = contained_path(
            output,
            f"folds/{fold}/{config['output_contract']['per_fold_commitment']}",
            must_exist=True,
        )
        fold_pin = _pin(fold_path, workspace)
        fold_payload = strict_json_object(fold_path)
        fold_cells: list[dict[str, Any]] = []
        key_digests: set[str] = set()
        fold_row_counts: set[int] = set()
        expected_fold_order = [
            (float(fraction), int(seed))
            for fraction in config["curve_protocol"]["prefix_fractions"]
            for seed in config["curve_protocol"]["seed_ids"]
        ]
        observed_cells = fold_payload.get("cells")
        if not isinstance(observed_cells, list) or len(observed_cells) != 15:
            raise Layer4ContractError("fold commitment cell projection changed")
        for (fraction, seed), projected in zip(
            expected_fold_order,
            observed_cells,
            strict=True,
        ):
            if not isinstance(projected, Mapping):
                raise Layer4ContractError("fold commitment cell projection is invalid")
            token = fraction_tokens[fraction]
            base = f"cells/{fold}/fraction_{token}/seed_{seed}"
            model_path = contained_path(
                output,
                f"{base}/{per_cell_names['model_bundle']}",
                must_exist=True,
            )
            prediction_path = contained_path(
                output,
                f"{base}/{per_cell_names['blind_prediction_array']}",
                must_exist=True,
            )
            receipt_path = contained_path(
                output,
                f"{base}/{per_cell_names['cell_receipt']}",
                must_exist=True,
            )
            model_pin = _pin(model_path, workspace)
            prediction_pin = _pin(prediction_path, workspace)
            receipt_pin = _pin(receipt_path, workspace)
            receipt = strict_json_object(receipt_path)
            _exact_keys(
                receipt,
                {
                    "schema_version",
                    "fold",
                    "fraction",
                    "seed",
                    "prefix_timestamps",
                    "prefix_timestamp_order_sha256",
                    "training_chunks",
                    "epochs",
                    "optimizer_steps",
                    "final_training_loss",
                    "trainable_parameters",
                    "model_state_sha256",
                    "validation_rows",
                    "validation_key_order_sha256",
                    "prediction_values_sha256",
                    "model_bundle",
                    "blind_prediction_array",
                    "blindness",
                    "official_promotion_allowed",
                    "candidate_or_test_prediction",
                    "upload_performed",
                },
                label="cell receipt",
            )
            prefix = prefix_pins[(fold, fraction)]
            payload, rows = _npy_little_endian_float64_payload(prediction_path)
            prediction_values_sha = hashlib.sha256(payload).hexdigest()
            blindness = receipt.get("blindness")
            if isinstance(blindness, Mapping):
                _exact_keys(
                    blindness,
                    {
                        "active_fold_target_temp_psal_scalar_fields_decoded_or_converted",
                        "outer_truth_used_for_fit_or_epoch_selection",
                        "future_target_truth_used_for_fit",
                        "layer2_and_layer3_exact_stage_a_seed_values",
                        "only_layer4_temperature_replaced",
                        "prediction_persisted_before_active_fold_truth_decode",
                    },
                    label="cell blindness",
                )
            receipt_checks = (
                receipt.get("schema_version")
                == "p2_joint_hydrographic_multitask_layer4.cell_receipt.v1",
                receipt.get("fold") == fold,
                receipt.get("fraction") == fraction,
                receipt.get("seed") == seed,
                receipt.get("prefix_timestamps") == prefix["timestamps"],
                receipt.get("prefix_timestamp_order_sha256")
                == prefix["timestamp_order_sha256"],
                receipt.get("training_chunks") == prefix["training_chunks"],
                receipt.get("epochs") == config["curve_protocol"]["epochs_per_fit"],
                receipt.get("optimizer_steps")
                == config["curve_protocol"]["epochs_per_fit"]
                * ((int(prefix["training_chunks"]) + config["curve_protocol"]["batch_size"] - 1)
                   // config["curve_protocol"]["batch_size"]),
                isinstance(receipt.get("final_training_loss"), (int, float)),
                not isinstance(receipt.get("final_training_loss"), bool),
                isinstance(receipt.get("final_training_loss"), (int, float))
                and math.isfinite(float(receipt["final_training_loss"])),
                receipt.get("trainable_parameters")
                == config["model_and_training"]["parameter_count"],
                _is_sha256(receipt.get("model_state_sha256")),
                receipt.get("validation_rows") == rows,
                _is_sha256(receipt.get("validation_key_order_sha256")),
                receipt.get("prediction_values_sha256") == prediction_values_sha,
                receipt.get("model_bundle") == model_pin,
                receipt.get("blind_prediction_array") == prediction_pin,
                isinstance(blindness, Mapping),
                isinstance(blindness, Mapping)
                and blindness.get(
                    "active_fold_target_temp_psal_scalar_fields_decoded_or_converted"
                )
                == 0,
                isinstance(blindness, Mapping)
                and blindness.get("outer_truth_used_for_fit_or_epoch_selection") is False,
                isinstance(blindness, Mapping)
                and blindness.get("future_target_truth_used_for_fit") is False,
                isinstance(blindness, Mapping)
                and blindness.get("layer2_and_layer3_exact_stage_a_seed_values") is True,
                isinstance(blindness, Mapping)
                and blindness.get("only_layer4_temperature_replaced") is True,
                isinstance(blindness, Mapping)
                and blindness.get(
                    "prediction_persisted_before_active_fold_truth_decode"
                )
                is True,
                receipt.get("official_promotion_allowed") is False,
                receipt.get("candidate_or_test_prediction") is False,
                receipt.get("upload_performed") is False,
            )
            if not all(receipt_checks):
                raise Layer4ContractError("cell receipt fails the sealed execution contract")
            expected_projection = {
                "fraction": fraction,
                "seed": seed,
                "model_bundle": model_pin,
                "blind_prediction_array": prediction_pin,
                "cell_receipt": receipt_pin,
                "prediction_values_sha256": prediction_values_sha,
                "model_state_sha256": receipt["model_state_sha256"],
                "optimizer_steps": receipt["optimizer_steps"],
            }
            if dict(projected) != expected_projection:
                raise Layer4ContractError("fold commitment does not match its cell artifacts")
            fold_cells.append(expected_projection)
            key_digests.add(str(receipt["validation_key_order_sha256"]))
            fold_row_counts.add(rows)
            aggregate_cells.append(
                {
                    "fold": fold,
                    "fraction": fraction,
                    "seed": seed,
                    "blind_prediction_array": prediction_pin,
                    "cell_receipt": receipt_pin,
                    "model_bundle": model_pin,
                    "prediction_values_sha256": prediction_values_sha,
                }
            )
            rows_by_cell.append(rows)
            values_hasher.update(f"{fold}|{fraction}|{seed}\0".encode("ascii"))
            values_hasher.update(payload)
        if len(key_digests) != 1 or len(fold_row_counts) != 1:
            raise Layer4ContractError("fold validation key order or row count differs across cells")
        _exact_keys(
            fold_payload,
            {
                "schema_version",
                "stage",
                "config",
                "fold",
                "fold_order_index",
                "cell_order",
                "cell_prediction_count",
                "validation_rows",
                "validation_key_order_sha256",
                "cells_sha256",
                "combined_fold_commitment_sha256",
                "cells",
                "prior_fold_commitments",
                "blind_input_audit",
                "active_fold_target_temp_psal_scalar_decodes_before_commitment",
                "truth_columns_present_in_predictions",
                "official_promotion_allowed",
                "candidate_or_test_prediction",
                "upload_performed",
            },
            label="fold commitment",
        )
        cells_sha = hashlib.sha256(canonical_json_bytes({"cells": fold_cells})).hexdigest()
        key_digest = next(iter(key_digests))
        combined = hashlib.sha256(
            bytes.fromhex(key_digest) + bytes.fromhex(cells_sha)
        ).hexdigest()
        audit = fold_payload.get("blind_input_audit")
        fold_checks = (
            fold_payload.get("schema_version")
            == "p2_joint_hydrographic_multitask_layer4.fold_commitment.v1",
            fold_payload.get("stage") == "ACTIVE_FOLD_15_BLIND_PREDICTIONS_COMMITTED",
            fold_payload.get("config") == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
            fold_payload.get("fold") == fold,
            fold_payload.get("fold_order_index") == fold_index,
            fold_payload.get("cell_order") == "FRACTION_THEN_SEED",
            fold_payload.get("cell_prediction_count") == 15,
            fold_payload.get("validation_rows") == next(iter(fold_row_counts)),
            fold_payload.get("validation_key_order_sha256") == key_digest,
            fold_payload.get("cells_sha256") == cells_sha,
            fold_payload.get("combined_fold_commitment_sha256") == combined,
            fold_payload.get("cells") == fold_cells,
            fold_payload.get("prior_fold_commitments") == expected_fold_pins,
            isinstance(audit, Mapping),
            isinstance(audit, Mapping) and audit.get("fold") == fold,
            isinstance(audit, Mapping)
            and audit.get("verified_prior_fold_commitments") == fold_index,
            isinstance(audit, Mapping)
            and audit.get(
                "active_fold_target_temp_psal_scalar_fields_decoded_or_converted"
            )
            == 0,
            isinstance(audit, Mapping)
            and audit.get(
                "withheld_target_temp_psal_scalar_fields_decoded_or_converted"
            )
            == 0,
            isinstance(audit, Mapping)
            and audit.get("anomaly_or_hidden_target_proxy_reads") == 0,
            fold_payload.get(
                "active_fold_target_temp_psal_scalar_decodes_before_commitment"
            )
            == 0,
            fold_payload.get("truth_columns_present_in_predictions") is False,
            fold_payload.get("official_promotion_allowed") is False,
            fold_payload.get("candidate_or_test_prediction") is False,
            fold_payload.get("upload_performed") is False,
        )
        if not all(fold_checks):
            raise Layer4ContractError("fold commitment graph verification failed")
        expected_fold_pins.append(fold_pin)

    if [(cell["fold"], cell["fraction"], cell["seed"]) for cell in aggregate_cells] != expected_order:
        raise Layer4ContractError("aggregate cell order changed")
    cells_sha = hashlib.sha256(
        canonical_json_bytes({"cells": aggregate_cells})
    ).hexdigest()
    folds_sha = hashlib.sha256(
        canonical_json_bytes({"fold_commitments": expected_fold_pins})
    ).hexdigest()
    values_sha = values_hasher.hexdigest()
    combined = hashlib.sha256(
        bytes.fromhex(values_sha) + bytes.fromhex(cells_sha) + bytes.fromhex(folds_sha)
    ).hexdigest()
    _exact_keys(
        commitment,
        {
            "schema_version",
            "stage",
            "config",
            "implementation_pins",
            "stage_a_seal",
            "cell_order",
            "cell_prediction_count",
            "fold_commitment_count",
            "rows_by_cell_in_order",
            "prediction_values_sha256",
            "cell_artifacts_sha256",
            "fold_commitments_sha256",
            "combined_prediction_commitment_sha256",
            "cells",
            "fold_commitments",
            "validation_truth_scalar_decodes_before_commitment",
            "truth_columns_present",
            "all_prediction_and_model_artifacts_o_excl_o_binary",
            "official_promotion_allowed",
            "candidate_or_test_prediction",
            "upload_performed",
        },
        label="aggregate prediction commitment",
    )
    aggregate_checks = (
        commitment.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.prediction_commitment.v1",
        commitment.get("stage")
        == "ALL_45_BLIND_PREDICTIONS_AND_THREE_FOLDS_COMMITTED_BEFORE_METRIC_TRUTH",
        commitment.get("config") == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        commitment.get("implementation_pins") == implementation_pins(workspace),
        commitment.get("stage_a_seal") == config["stage_a_reference"]["artifacts"]["SEAL"],
        commitment.get("cell_order") == "FOLD_THEN_FRACTION_THEN_SEED",
        commitment.get("cell_prediction_count") == 45,
        commitment.get("fold_commitment_count") == 3,
        commitment.get("rows_by_cell_in_order") == rows_by_cell,
        commitment.get("prediction_values_sha256") == values_sha,
        commitment.get("cell_artifacts_sha256") == cells_sha,
        commitment.get("fold_commitments_sha256") == folds_sha,
        commitment.get("combined_prediction_commitment_sha256") == combined,
        commitment.get("cells") == aggregate_cells,
        commitment.get("fold_commitments") == expected_fold_pins,
        commitment.get("validation_truth_scalar_decodes_before_commitment") == 0,
        commitment.get("truth_columns_present") is False,
        commitment.get("all_prediction_and_model_artifacts_o_excl_o_binary") is True,
        commitment.get("official_promotion_allowed") is False,
        commitment.get("candidate_or_test_prediction") is False,
        commitment.get("upload_performed") is False,
    )
    if not all(aggregate_checks):
        raise Layer4ContractError("aggregate prediction commitment graph verification failed")
    return {
        "cells": 45,
        "folds": 3,
        "fold_pins": expected_fold_pins,
        "rows_by_cell": rows_by_cell,
        "validation_rows_per_fraction": {
            str(float(fraction)): sum(
                rows_by_cell[expected_order.index((str(fold), float(fraction), int(config["curve_protocol"]["seed_ids"][0])))]
                for fold in config["curve_protocol"]["fold_major_order"]
            )
            for fraction in config["curve_protocol"]["prefix_fractions"]
        },
        "optimizer_steps": sum(
            int(item["optimizer_steps"])
            for fold in config["curve_protocol"]["fold_major_order"]
            for item in strict_json_object(
                contained_path(
                    output,
                    f"folds/{fold}/{config['output_contract']['per_fold_commitment']}",
                    must_exist=True,
                )
            )["cells"]
        ),
        "combined_prediction_commitment_sha256": combined,
    }


def verify_seal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    canonical = load_canonical_config(workspace, supplied_config=config)
    output = stage_paths(workspace, canonical)["output"].resolve(strict=True)
    expected = expected_output_files(canonical)
    observed = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_file()
    }
    observed_directories = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_dir()
    }
    if (
        observed != expected
        or observed_directories != expected_output_directories(canonical)
        or any(item.is_symlink() for item in output.rglob("*"))
    ):
        raise Layer4ContractError("sealed output allowlist changed")
    aggregate = canonical["output_contract"]["aggregate"]
    manifest_path = contained_path(output, aggregate["manifest"], must_exist=True)
    sidecar_path = contained_path(output, aggregate["manifest_sidecar"], must_exist=True)
    seal_path = contained_path(output, aggregate["seal"], must_exist=True)
    manifest = strict_json_object(manifest_path)
    seal = strict_json_object(seal_path)
    commitment = strict_json_object(
        contained_path(output, aggregate["prediction_commitment"], must_exist=True)
    )
    evidence = strict_json_object(
        contained_path(output, aggregate["learning_curve_evidence"], must_exist=True)
    )
    decision = strict_json_object(contained_path(output, aggregate["gate_decision"], must_exist=True))
    receipt = strict_json_object(contained_path(output, aggregate["training_receipt"], must_exist=True))
    metrics = strict_json_object(contained_path(output, aggregate["metrics"], must_exist=True))
    bootstrap = strict_json_object(
        contained_path(output, aggregate["bootstrap_receipt"], must_exist=True)
    )
    artifact_pins = manifest.get("artifacts")
    if not isinstance(artifact_pins, Mapping) or set(artifact_pins) != expected - {
        aggregate["manifest"],
        aggregate["manifest_sidecar"],
        aggregate["seal"],
    }:
        raise Layer4ContractError("manifest artifact roles changed")
    for relative, pin in artifact_pins.items():
        if (
            not isinstance(pin, Mapping)
            or _relative_pin(contained_path(output, relative, must_exist=True), output)
            != dict(pin)
        ):
            raise Layer4ContractError(f"sealed artifact pin changed: {relative}")
    graph = _verify_prediction_commitment_graph(workspace, output, canonical, commitment)
    commitment_workspace_pin = _pin(
        contained_path(output, aggregate["prediction_commitment"], must_exist=True),
        workspace,
    )
    commitment_output_pin = _relative_pin(
        contained_path(output, aggregate["prediction_commitment"], must_exist=True),
        output,
    )
    _exact_keys(
        metrics,
        {
            "schema_version",
            "comparison_mode",
            "exact_official_incumbent_comparison",
            "points",
            "fold_delta_order",
            "fold_deltas_challenger_minus_reference",
            "slice_deltas_challenger_minus_reference",
            "official_promotion_allowed",
        },
        label="metrics",
    )
    _exact_keys(
        bootstrap,
        {
            "schema_version",
            "prediction_commitment",
            "points",
            "all_five_fractions_exactly_5000_paired_kst_day_replicates",
            "candidate_or_test_prediction",
            "upload_performed",
        },
        label="bootstrap receipt",
    )
    _exact_keys(
        evidence,
        {
            "schema_version",
            "problem",
            "comparison_mode",
            "exact_official_incumbent_comparison",
            "local_qualification_only",
            "prediction_commitment",
            "points",
            "fold_deltas_candidate_minus_incumbent",
            "slice_deltas_candidate_minus_incumbent",
            "local_numeric_gates",
            "local_qualification",
            "leakage_checks",
            "reproducibility_checks",
            "output_firewall",
        },
        label="learning curve evidence",
    )
    _exact_keys(
        decision,
        {
            "schema_version",
            "status",
            "local_numeric_gates",
            "local_qualification",
            "passed",
            "official_promotion",
            "official_promotion_allowed",
            "candidate_generated",
            "test_prediction_generated",
            "upload_performed",
        },
        label="gate decision",
    )
    _exact_keys(
        receipt,
        {
            "schema_version",
            "started_at_kst",
            "completed_at_kst",
            "config",
            "plan",
            "runtime",
            "prediction_commitment",
            "fold_commitments",
            "fold_blind_input_audits",
            "truth_access_audit",
            "model_fits",
            "blind_prediction_arrays",
            "optimizer_steps",
            "candidate_predictions",
            "test_predictions",
            "uploads",
        },
        label="training receipt",
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "created_at_kst",
            "append_only",
            "problem",
            "comparison_mode",
            "exact_official_incumbent_comparison",
            "config",
            "scientific_surface_pins",
            "implementation_pins",
            "source_pins",
            "stage_a_reference_pins",
            "data_pins",
            "preflight_summary_sha256",
            "prediction_commitment",
            "artifacts",
            "local_qualification",
            "official_promotion_allowed",
            "full_fit_performed",
            "candidate_generated",
            "test_prediction_generated",
            "uploads",
        },
        label="manifest",
    )
    _exact_keys(
        seal,
        {
            "schema_version",
            "complete",
            "status",
            "comparison_mode",
            "exact_official_incumbent_comparison",
            "local_qualification",
            "official_promotion_allowed",
            "config",
            "prediction_commitment",
            "manifest",
            "manifest_sidecar",
            "candidate_generated",
            "test_prediction_generated",
            "upload_count",
        },
        label="seal",
    )
    metric_points = metrics.get("points")
    bootstrap_points = bootstrap.get("points")
    fractions = [float(value) for value in canonical["curve_protocol"]["prefix_fractions"]]
    point_fractions = (
        [float(point.get("fraction")) for point in metric_points]
        if isinstance(metric_points, list)
        and all(isinstance(point, Mapping) for point in metric_points)
        else []
    )
    bootstrap_fractions = (
        [float(point.get("fraction")) for point in bootstrap_points]
        if isinstance(bootstrap_points, list)
        and all(isinstance(point, Mapping) for point in bootstrap_points)
        else []
    )
    oof_header, oof_rows = _csv_header_and_rows(
        contained_path(output, aggregate["learning_curve_oof"], must_exist=True)
    )
    expected_oof_header = [
        "fraction",
        "fold",
        "station",
        "layer",
        "time",
        *(f"seed_{seed}" for seed in canonical["curve_protocol"]["seed_ids"]),
        "prediction_mean",
        *(f"challenger_seed_{seed}" for seed in canonical["curve_protocol"]["seed_ids"]),
        "challenger_mean",
        "truth",
    ]
    expected_oof_rows = sum(graph["validation_rows_per_fraction"].values())
    fold_audits = receipt.get("fold_blind_input_audits")
    truth_audit = receipt.get("truth_access_audit")
    local_gates = decision.get("local_numeric_gates")
    plan = receipt.get("plan")
    runtime = receipt.get("runtime")
    paths = stage_paths(workspace, canonical)
    _qa, qa_sha = verify_pre_execution_qa(workspace, canonical)
    _authorization, authorization_sha = verify_execution_authorization(
        workspace,
        canonical,
        require_unconsumed=False,
        require_output_absent=False,
    )
    lock = strict_json_object(paths["attempt_lock"])
    _exact_keys(
        lock,
        {
            "schema_version",
            "stage",
            "config",
            "preflight_summary_sha256",
            "qa_receipt_sha256",
            "authorization_sha256",
            "scientific_design",
            "implementation_pins",
            "source_pins",
            "stage_a_reference_pins",
            "canonical_output",
            "comparison_mode",
            "exact_official_incumbent_comparison",
            "curve_only",
            "official_promotion_allowed",
            "candidate_or_test_prediction_allowed",
            "rerun_allowed",
            "resume_allowed",
            "upload_allowed",
        },
        label="attempt lock",
    )
    checks = {
        "manifest_schema": manifest.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.manifest.v1",
        "manifest_append_only": manifest.get("append_only") is True,
        "manifest_identity": manifest.get("problem") == "P2"
        and manifest.get("comparison_mode") == MODE
        and manifest.get("exact_official_incumbent_comparison") is False,
        "manifest_config": manifest.get("config") == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "manifest_scientific": manifest.get("scientific_surface_pins")
        == {
            key: value
            for key, value in canonical["scientific_surface"].items()
            if key != "scientific_changes_from_frozen_design"
        },
        "manifest_implementation": manifest.get("implementation_pins") == implementation_pins(workspace),
        "manifest_sources": manifest.get("source_pins") == canonical["source_pins"],
        "manifest_stage_a": manifest.get("stage_a_reference_pins")
        == canonical["stage_a_reference"]["artifacts"],
        "manifest_data": manifest.get("data_pins")
        == canonical["data_contract"]["source_pins"],
        "manifest_preflight": _is_sha256(manifest.get("preflight_summary_sha256")),
        "manifest_commitment": manifest.get("prediction_commitment")
        == commitment_output_pin,
        "manifest_artifacts": True,
        "manifest_no_promotion": manifest.get("official_promotion_allowed") is False,
        "manifest_no_candidate": manifest.get("candidate_generated") is False,
        "manifest_no_test": manifest.get("test_prediction_generated") is False,
        "manifest_no_upload": manifest.get("uploads") == 0,
        "manifest_no_full_fit": manifest.get("full_fit_performed") is False,
        "commitment_schema": commitment.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.prediction_commitment.v1",
        "commitment_cells": commitment.get("cell_prediction_count") == 45,
        "commitment_folds": commitment.get("fold_commitment_count") == 3,
        "commitment_truth_zero": commitment.get("validation_truth_scalar_decodes_before_commitment") == 0,
        "metrics_schema": metrics.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.metrics.v1",
        "metrics_identity": metrics.get("comparison_mode") == MODE
        and metrics.get("exact_official_incumbent_comparison") is False
        and metrics.get("official_promotion_allowed") is False,
        "metrics_fractions": point_fractions == fractions,
        "metrics_folds": metrics.get("fold_delta_order")
        == canonical["curve_protocol"]["fold_major_order"],
        "bootstrap_schema": bootstrap.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.bootstrap_receipt.v1",
        "bootstrap_commitment": bootstrap.get("prediction_commitment")
        == commitment_workspace_pin,
        "bootstrap_fractions": bootstrap_fractions == fractions,
        "bootstrap_protocol": isinstance(bootstrap_points, list)
        and all(
            point.get("replicates") == 5000
            and point.get("cluster") == "KST_day"
            and point.get("seed") == canonical["curve_protocol"]["bootstrap_seed"]
            and point.get("paired_reference_and_challenger") is True
            for point in bootstrap_points
        )
        and bootstrap.get(
            "all_five_fractions_exactly_5000_paired_kst_day_replicates"
        )
        is True,
        "bootstrap_no_candidate": bootstrap.get("candidate_or_test_prediction") is False,
        "bootstrap_no_upload": bootstrap.get("upload_performed") is False,
        "oof_header": oof_header == expected_oof_header,
        "oof_rows": oof_rows == expected_oof_rows,
        "evidence_schema": evidence.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.learning_curve_evidence.v1",
        "evidence_identity": evidence.get("problem") == "P2"
        and evidence.get("comparison_mode") == MODE
        and evidence.get("exact_official_incumbent_comparison") is False
        and evidence.get("local_qualification_only") is True,
        "evidence_commitment": evidence.get("prediction_commitment")
        == commitment_workspace_pin,
        "evidence_points": evidence.get("points") == metric_points,
        "evidence_deltas": evidence.get("fold_deltas_candidate_minus_incumbent")
        == metrics.get("fold_deltas_challenger_minus_reference")
        and evidence.get("slice_deltas_candidate_minus_incumbent")
        == metrics.get("slice_deltas_challenger_minus_reference"),
        "evidence_leakage": isinstance(evidence.get("leakage_checks"), Mapping)
        and bool(evidence["leakage_checks"])
        and all(value is True for value in evidence["leakage_checks"].values()),
        "evidence_repro": isinstance(evidence.get("reproducibility_checks"), Mapping)
        and bool(evidence["reproducibility_checks"])
        and all(value is True for value in evidence["reproducibility_checks"].values()),
        "evidence_firewall": evidence.get("output_firewall")
        == {
            "research_only": True,
            "official_promotion_allowed": False,
            "full_fit_performed": False,
            "candidate_generated": False,
            "test_prediction_generated": False,
            "upload_performed": False,
        },
        "decision_schema": decision.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.gate_decision.v1",
        "decision_gates": isinstance(local_gates, Mapping)
        and bool(local_gates)
        and all(isinstance(value, bool) for value in local_gates.values())
        and evidence.get("local_numeric_gates") == local_gates,
        "decision_qualification": decision.get("local_qualification")
        is all(local_gates.values())
        if isinstance(local_gates, Mapping) and bool(local_gates)
        else False,
        "decision_evidence": evidence.get("local_qualification")
        is decision.get("local_qualification"),
        "decision_research": decision.get("official_promotion") is False
        and decision.get("passed") is False,
        "decision_no_candidate": decision.get("candidate_generated") is False,
        "decision_no_test": decision.get("test_prediction_generated") is False,
        "decision_no_upload": decision.get("upload_performed") is False,
        "receipt_fits": receipt.get("model_fits") == 45,
        "receipt_predictions": receipt.get("blind_prediction_arrays") == 45,
        "receipt_steps": receipt.get("optimizer_steps") == 6132,
        "receipt_schema": receipt.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.training_receipt.v1",
        "receipt_plan": isinstance(plan, Mapping)
        and plan.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.execution_plan.v1"
        and plan.get("stage") == STAGE
        and plan.get("problem") == "P2"
        and plan.get("comparison_mode") == MODE
        and plan.get("exact_official_incumbent_comparison") is False
        and plan.get("fold_major_order")
        == canonical["curve_protocol"]["fold_major_order"]
        and plan.get("prefix_fractions")
        == canonical["curve_protocol"]["prefix_fractions"]
        and plan.get("pipeline_seeds") == canonical["curve_protocol"]["seed_ids"]
        and plan.get("cell_order") == "FOLD_THEN_FRACTION_THEN_SEED"
        and plan.get("fit_cells") == 45
        and plan.get("blind_prediction_arrays") == 45
        and plan.get("fold_commitments") == 3
        and plan.get("epochs_per_fit") == 28
        and plan.get("optimizer_steps") == 6132
        and plan.get("bootstrap_replicates_per_fraction") == 5000
        and plan.get("full_fit_jobs") == 0
        and plan.get("candidate_predictions") == 0
        and plan.get("test_predictions") == 0
        and plan.get("uploads") == 0,
        "receipt_runtime": isinstance(runtime, Mapping)
        and set(runtime)
        == {
            "python",
            "numpy",
            "pandas",
            "torch",
            "torch_cuda",
            "cuda_available",
            "gpu_name",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
        }
        and all(
            runtime.get(key) == canonical["runtime_contract"][key]
            for key in (
                "python",
                "numpy",
                "pandas",
                "torch",
                "torch_cuda",
                "cuda_available",
                "cudnn_benchmark",
                "cudnn_deterministic",
                "deterministic_algorithms",
            )
        )
        and canonical["runtime_contract"]["gpu_name_contains"]
        in str(runtime.get("gpu_name")),
        "receipt_config": receipt.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "receipt_commitment": receipt.get("prediction_commitment")
        == commitment_workspace_pin,
        "receipt_folds": receipt.get("fold_commitments") == graph["fold_pins"],
        "receipt_fold_audits": isinstance(fold_audits, Mapping)
        and list(fold_audits) == canonical["curve_protocol"]["fold_major_order"]
        and all(
            audit.get(
                "active_fold_target_temp_psal_scalar_fields_decoded_or_converted"
            )
            == 0
            and audit.get(
                "withheld_target_temp_psal_scalar_fields_decoded_or_converted"
            )
            == 0
            and audit.get("anomaly_or_hidden_target_proxy_reads") == 0
            for audit in fold_audits.values()
        ),
        "receipt_truth_audit": isinstance(truth_audit, Mapping)
        and truth_audit.get("nonvalidation_target_scalars_converted") == 0
        and truth_audit.get("hidden_test_target_scalars_converted") == 0
        and truth_audit.get("test_index_or_sample_submission_semantic_reads") == 0,
        "commitment_graph_cells": graph["cells"] == 45,
        "commitment_graph_folds": graph["folds"] == 3,
        "commitment_graph_steps": graph["optimizer_steps"] == 6132,
        "receipt_no_candidate": receipt.get("candidate_predictions") == 0,
        "receipt_no_test": receipt.get("test_predictions") == 0,
        "receipt_no_upload": receipt.get("uploads") == 0,
        "lock_schema": lock.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.attempt_lock.v1",
        "lock_identity": lock.get("stage") == STAGE
        and lock.get("config") == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256}
        and lock.get("comparison_mode") == MODE
        and lock.get("exact_official_incumbent_comparison") is False,
        "lock_preflight": lock.get("preflight_summary_sha256")
        == manifest.get("preflight_summary_sha256"),
        "lock_controls": lock.get("qa_receipt_sha256") == qa_sha
        and lock.get("authorization_sha256") == authorization_sha,
        "lock_dependencies": lock.get("scientific_design")
        == canonical["scientific_surface"]["design"]
        and lock.get("implementation_pins") == implementation_pins(workspace)
        and lock.get("source_pins") == canonical["source_pins"]
        and lock.get("stage_a_reference_pins")
        == canonical["stage_a_reference"]["artifacts"],
        "lock_output": lock.get("canonical_output")
        == canonical["canonical_paths"]["output"],
        "lock_firewall": lock.get("curve_only") is True
        and lock.get("official_promotion_allowed") is False
        and lock.get("candidate_or_test_prediction_allowed") is False
        and lock.get("rerun_allowed") is False
        and lock.get("resume_allowed") is False
        and lock.get("upload_allowed") is False,
        "sidecar": sidecar_path.read_bytes()
        == f"{sha256_file(manifest_path)}  manifest.json\n".encode("ascii"),
        "seal_schema": seal.get("schema_version")
        == "p2_joint_hydrographic_multitask_layer4.seal.v1",
        "seal_complete": seal.get("complete") is True,
        "seal_identity": seal.get("comparison_mode") == MODE
        and seal.get("exact_official_incumbent_comparison") is False,
        "seal_config": seal.get("config")
        == {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "seal_status": seal.get("status") == decision.get("status"),
        "qualification_relation": manifest.get("local_qualification")
        is decision.get("local_qualification")
        and seal.get("local_qualification") is decision.get("local_qualification"),
        "seal_commitment": seal.get("prediction_commitment")
        == commitment_output_pin,
        "seal_manifest": seal.get("manifest") == _relative_pin(manifest_path, output),
        "seal_sidecar": seal.get("manifest_sidecar") == _relative_pin(sidecar_path, output),
        "seal_no_promotion": seal.get("official_promotion_allowed") is False,
        "seal_no_candidate": seal.get("candidate_generated") is False,
        "seal_no_test": seal.get("test_prediction_generated") is False,
        "seal_no_upload": seal.get("upload_count") == 0,
    }
    if not all(checks.values()):
        raise Layer4ContractError(
            f"Layer-4 seal verification failed: {sorted(k for k, v in checks.items() if not v)}"
        )
    return {
        "status": "PASS_SEALED_P2_JOINT_HYDROGRAPHIC_LAYER4_V1",
        "manifest_sha256": sha256_file(manifest_path),
        "seal_sha256": sha256_file(seal_path),
        "local_qualification": bool(seal.get("local_qualification")),
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }


__all__ = [
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "ENGINE_MODULE",
    "ENGINE_RELATIVE",
    "ExecutionCapability",
    "FRACTION_ROLES",
    "IMPLEMENTATION_ROLES",
    "Layer4ContractError",
    "MODE",
    "STAGE",
    "begin_execution",
    "canonical_json_bytes",
    "claim_aggregate_commitment",
    "claim_cell",
    "claim_fold_commitment",
    "claim_truth_and_score_phase",
    "complete_aggregate_commitment",
    "complete_cell",
    "complete_execution_phase",
    "complete_fold_commitment",
    "consume_attempt_lock",
    "contained_path",
    "exclusive_bytes",
    "exclusive_json",
    "expected_output_files",
    "expected_output_directories",
    "implementation_pins",
    "issue_execution_capability",
    "load_canonical_config",
    "revoke_execution_capability",
    "sha256_file",
    "stage_paths",
    "static_preflight",
    "strict_json_object",
    "validate_config",
    "verify_consumed_attempt_lock",
    "verify_execution_authorization",
    "verify_pin_map",
    "verify_pre_execution_qa",
    "verify_scientific_surface",
    "verify_seal",
    "verify_stage_a_reference",
    "workspace_path",
]
