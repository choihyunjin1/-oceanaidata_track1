"""Authenticated execution contract for P1 multiscale Gen6r2.

This module is compiled only from the source buffer authenticated by the
noncyclic bootstrap.  Execution documents are revalidated before capability
minting and again before every state transition.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

try:
    _CONTEXT = _P1_V6R2_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r2 contract requires the authenticated bootstrap") from exc

if not isinstance(_CONTEXT, dict) or _CONTEXT.get("all_owner_roles_authenticated") is not True:
    raise RuntimeError("P1 Gen6r2 contract loaded before source authentication")

GENERATION = "p1_multiscale_cross_layer_offset_drift_unary_v6r2"
QA_SCHEMA = "p1_multiscale_cross_layer_offset_drift_unary.v6r2.independent_qa.v1"
AUTH_SCHEMA = "p1_multiscale_cross_layer_offset_drift_unary.v6r2.execution_authorization.v1"
LOCK_SCHEMA = "p1_multiscale_cross_layer_offset_drift_unary.v6r2.attempt_lock.v1"
AUTHORIZATION_PHRASE = (
    "AUTHORIZE_P1_MULTISCALE_CROSS_LAYER_OFFSET_DRIFT_V6R2_ONE_SHOT_RESEARCH_CURVE_ONLY"
)
LOWER_SHA = set("0123456789abcdef")
REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
O_BINARY = getattr(os, "O_BINARY", 0)


class ContractError(RuntimeError):
    """The authenticated execution contract failed closed."""


class CapabilityError(PermissionError):
    """A capability was absent, forged, stale, or replayed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def deep_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= LOWER_SHA


def _exact_int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _relative_parts(relative: str) -> tuple[str, ...]:
    if type(relative) is not str or not relative or "\\" in relative:
        raise ContractError("path must be canonical POSIX relative text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError("path is not a contained relative path")
    return pure.parts


def _has_reparse(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(
        int(getattr(info, "st_file_attributes", 0)) & REPARSE_ATTRIBUTE
    )


def _plain_chain(path: Path, *, require_target: bool) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    for entry in (*reversed(lexical.parents), lexical):
        if os.path.lexists(entry) and _has_reparse(entry):
            raise ContractError(f"link/reparse path forbidden: {entry}")
    if require_target and not os.path.lexists(lexical):
        raise FileNotFoundError(lexical)
    return lexical


def contained_path(root: Path, relative: str, *, must_exist: bool, kind: str | None = None) -> Path:
    base = _plain_chain(root, require_target=True).resolve(strict=True)
    candidate = _plain_chain(base.joinpath(*_relative_parts(relative)), require_target=must_exist)
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ContractError("path escaped its canonical root") from exc
    if must_exist:
        info = resolved.lstat()
        if kind == "file" and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise ContractError("regular single-link file required")
        if kind == "directory" and not stat.S_ISDIR(info.st_mode):
            raise ContractError("plain directory required")
    return resolved


def file_pin(path: Path, *, relative: str | None = None) -> dict[str, Any]:
    checked = _plain_chain(path, require_target=True).resolve(strict=True)
    info_before = checked.lstat()
    if not stat.S_ISREG(info_before.st_mode) or info_before.st_nlink != 1:
        raise ContractError("single-link regular file required for pin")
    digest = hashlib.sha256()
    with checked.open("rb") as stream:
        descriptor_before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        descriptor_after = os.fstat(stream.fileno())
    info_after = checked.lstat()
    identity_before = (
        info_before.st_dev,
        info_before.st_ino,
        info_before.st_size,
        info_before.st_mtime_ns,
        info_before.st_ctime_ns,
    )
    identity_after = (
        info_after.st_dev,
        info_after.st_ino,
        info_after.st_size,
        info_after.st_mtime_ns,
        info_after.st_ctime_ns,
    )
    if identity_before != identity_after or (
        descriptor_before.st_dev,
        descriptor_before.st_ino,
        descriptor_before.st_size,
    ) != (
        descriptor_after.st_dev,
        descriptor_after.st_ino,
        descriptor_after.st_size,
    ):
        raise ContractError("file identity changed while hashing")
    if (
        descriptor_before.st_dev,
        descriptor_before.st_ino,
        descriptor_before.st_size,
    ) != (info_before.st_dev, info_before.st_ino, info_before.st_size) or (
        descriptor_after.st_dev,
        descriptor_after.st_ino,
        descriptor_after.st_size,
    ) != (info_after.st_dev, info_after.st_ino, info_after.st_size):
        raise ContractError("opened descriptor differs from the pinned pathname")
    return {
        "path": relative if relative is not None else checked.as_posix(),
        "bytes": int(info_after.st_size),
        "sha256": digest.hexdigest(),
        "device": int(info_after.st_dev),
        "inode": int(info_after.st_ino),
        "nlink": int(info_after.st_nlink),
        "non_reparse": True,
    }


def _assert_expected_pin(root: Path, expected: dict[str, Any], *, label: str) -> dict[str, Any]:
    if type(expected) is not dict or set(expected) != {"path", "bytes", "sha256"}:
        raise ContractError(f"{label} pin schema differs")
    if (
        type(expected["path"]) is not str
        or not _exact_int(expected["bytes"])
        or not _is_sha(expected["sha256"])
    ):
        raise ContractError(f"{label} pin value differs")
    path = contained_path(root, expected["path"], must_exist=True, kind="file")
    observed = file_pin(path, relative=expected["path"])
    if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
        raise ContractError(f"{label} pin drift")
    return observed


def _config() -> dict[str, Any]:
    value = _CONTEXT.get("config")
    if type(value) is not dict:
        raise ContractError("authenticated config is unavailable")
    return value


def _projection() -> dict[str, Any]:
    value = _CONTEXT.get("science_projection")
    if type(value) is not dict:
        raise ContractError("authenticated science projection is unavailable")
    return value


def _workspace() -> Path:
    value = _CONTEXT.get("workspace")
    if not isinstance(value, Path):
        raise ContractError("authenticated workspace is unavailable")
    return value


def _data_dir() -> Path:
    value = _CONTEXT.get("data_dir")
    if not isinstance(value, Path):
        raise ContractError("authenticated data directory is unavailable")
    return value


def execution_closure() -> dict[str, Any]:
    config = _config()
    return {
        "generation": GENERATION,
        "bootstrap_observed_pin": _CONTEXT["bootstrap_observed_pin"],
        "owner_role_pins": _CONTEXT["owner_role_pins"],
        "science_projection_sha256": config["science_projection"]["sha256"],
        "v6_disposition": config["append_only_lineage"],
        "source_pins": config["source_pins"],
        "runtime_trust_contract": config["runtime_trust_contract"],
        "incumbent_binding": config["incumbent_binding"],
        "inner_incumbent_binding": config["inner_incumbent_binding"],
        "split_and_dependency_contract": config["split_and_dependency_contract"],
        "selective_target_contract": config["selective_target_contract"],
        "commitment_contract": config["commitment_contract"],
        "metric_contract": config["metric_contract"],
        "resource_ceiling": config["resource_ceiling"],
        "output_contract": config["output_contract"],
        "v9_binding": config["v9_binding"],
    }


def execution_closure_sha256() -> str:
    return deep_sha256(execution_closure())


def _verify_config_semantics() -> dict[str, Any]:
    config = _config()
    projection = _projection()
    if (
        config.get("schema_version")
        != "p1_multiscale_cross_layer_offset_drift_unary.v6r2.execution_contract.v1"
        or config.get("generation") != GENERATION
        or config.get("stage") != "STATIC_OWNER_IMPLEMENTATION_AWAITING_FRESH_INDEPENDENT_QA"
        or config.get("actual_execution_authorized") is not False
        or projection.get("generation") != GENERATION
        or projection.get("hypothesis_count") != 1
        or projection.get("hypothesis_id") != config.get("scientific_identity")
    ):
        raise ContractError("config/science identity differs")
    runtime = config.get("runtime_trust_contract", {})
    runtime_audit = _CONTEXT.get("runtime_static_audit")
    if (
        type(runtime) is not dict
        or runtime.get("base_prefix_source")
        != "observed_sys_base_prefix_never_personal_path_literal"
        or runtime.get("bytecode_loader_present") is not False
        or runtime.get("source_loader_uses_authenticated_buffers") is not True
        or type(runtime.get("distribution_records")) is not dict
        or len(runtime["distribution_records"]) != 12
        or type(runtime_audit) is not dict
        or runtime_audit.get("canonical_flags") != ["-I", "-S", "-B"]
        or runtime_audit.get("safe_path") is not True
        or runtime_audit.get("ignore_environment") is not True
        or runtime_audit.get("pycache_prefix_absent") is not True
        or runtime_audit.get("startup_sys_path_exact") is not True
        or runtime_audit.get("startup_meta_path_exact") is not True
        or runtime_audit.get("earliest_audit_hook_installed") is not True
        or runtime_audit.get("stdlib_source_buffer_loader_installed") is not True
    ):
        raise ContractError("runtime trust contract differs")
    train_gate = projection.get("train_only_gate", {})
    final_gate = projection.get("final_curve_gate", {})
    if (
        train_gate.get("minimum_worst_station_layer_f1_delta") != 0.0
        or final_gate.get("worst_station_layer_f1_delta_at_least") != 0.0
        or train_gate.get("minimum_spike_f1_delta") != 0.0
        or final_gate.get("full_spike_f1_delta_at_least") != 0.0
        or train_gate.get("required_inner_blocks") != 3
        or train_gate.get("minimum_nondegrading_inner_blocks") != 3
        or train_gate.get("count_domain") != "non_boolean_integral_json_number_exactly_3"
    ):
        raise ContractError("corrected gate semantics differ")
    if config.get("inner_incumbent_binding", {}).get(
        "frozen_reference_source_pins"
    ) != projection.get("inner_incumbent", {}).get("frozen_reference_source_pins") or config.get(
        "inner_incumbent_binding", {}
    ).get("fixed_postprocess_golden") != {
        "fixture_sha256": projection.get("inner_incumbent", {})
        .get("golden_fixture", {})
        .get("sha256"),
        **projection.get("inner_incumbent", {})
        .get("golden_fixture", {})
        .get("output_int8_sha256_by_fold", {}),
    }:
        raise ContractError("fixed incumbent source/golden projection differs")
    resource = config.get("resource_ceiling", {})
    if not (
        resource.get("curve_cells") == 15
        and resource.get("maximum_label_free_baseline_fit_calls") == 60
        and resource.get("maximum_supervised_unary_fit_calls") == 60
        and resource.get("maximum_top_level_fit_calls") == 120
        and resource.get("maximum_seasonal_irls_steps") == 7680
        and resource.get("maximum_unary_lbfgs_iterations") == 3840
        and resource.get("maximum_total_iterative_steps") == 11520
        and resource.get("maximum_total_iterative_steps")
        == resource.get("maximum_seasonal_irls_steps")
        + resource.get("maximum_unary_lbfgs_iterations")
        and resource.get("maximum_vram_bytes") == 0
        and resource.get("maximum_files_written") == 192
        and resource.get("gpu_allowed") is False
    ):
        raise ContractError("resource arithmetic differs")
    if any(value is not False for value in config.get("static_prohibitions", {}).values()):
        raise ContractError("static prohibition differs")
    return {
        "single_hypothesis": True,
        "corrected_worst_station_layer_floor": 0.0,
        "strict_count_domain": True,
        "resource_arithmetic": True,
        "runtime_trust": True,
    }


def _verify_v6_disposition() -> dict[str, Any]:
    root = _workspace()
    lineage = _config()["append_only_lineage"]
    observed = {
        role: _assert_expected_pin(root, pin, label=f"v6 {role}")
        for role, pin in lineage["superseded_v6"].items()
    }
    owner = _assert_expected_pin(root, lineage["owner_no_go"], label="v6 owner NO-GO")
    tombstone = _assert_expected_pin(
        root, lineage["execution_tombstone"], label="v6 execution tombstone"
    )
    parse = _CONTEXT["authenticated_json_for_pin"]
    owner_value = parse(lineage["owner_no_go"], "v6 owner NO-GO")
    tombstone_value = parse(lineage["execution_tombstone"], "v6 execution tombstone")
    if (
        owner_value.get("verdict") != "P0=0_P1=2_NO_GO"
        or owner_value.get("p0_count") != 0
        or owner_value.get("p1_count") != 2
        or tombstone_value.get("status") != "PERMANENTLY_TOMBSTONED_NEVER_EXECUTE"
        or tombstone_value.get("owner_no_go_receipt") != lineage["owner_no_go"]
    ):
        raise ContractError("v6 disposition semantics differ")
    return {"v6": observed, "owner_no_go": owner, "tombstone": tombstone}


def _verify_v9() -> dict[str, Any]:
    expected = _config()["v9_binding"]
    pin = {name: expected[name] for name in ("path", "bytes", "sha256")}
    observed = _assert_expected_pin(_workspace(), pin, label="v9")
    raw = _CONTEXT["authenticated_bytes_for_pin"](pin, "v9")
    lines = raw.decode("utf-8", errors="strict").splitlines()
    parser = _CONTEXT["parse_json_text"]
    events = [parser(line, f"v9 line {index + 1}") for index, line in enumerate(lines) if line]
    if (
        len(events) != 3
        or events[-1].get("seq") != expected["head_seq"]
        or events[-1].get("event_sha256") != expected["head_event_sha256"]
        or sum(int(event.get("official_uploads", 0)) for event in events)
        != expected["semantic_upload_count"]
    ):
        raise ContractError("v9 semantic anchor differs")
    lock = contained_path(_workspace(), f"{pin['path']}.append.lock", must_exist=False)
    if os.path.lexists(lock):
        raise ContractError("v9 append lock exists")
    return {**observed, "head_seq": events[-1]["seq"], "uploads": 0, "append_lock": False}


def _verify_incumbent_binding() -> dict[str, Any]:
    root = _workspace()
    binding = _config()["incumbent_binding"]
    roles = ("predictions_complete", "manifest", "fold_scope_audit", "full_fraction_oof_keys")
    observed = {
        role: _assert_expected_pin(root, binding[role], label=f"incumbent {role}") for role in roles
    }
    complete = _CONTEXT["authenticated_json_for_pin"](
        binding["predictions_complete"], "incumbent predictions_complete"
    )
    parts = complete.get("parts")
    expected_pairs = [
        (fold, fraction)
        for fold in ("2025_q2", "2025_q3", "2025_q4")
        for fraction in (0.4, 0.55, 0.7, 0.85, 1.0)
    ]
    if (
        complete.get("schema_version") != "p1_learning_curve_predictions_complete.v1"
        or complete.get("part_count") != 15
        or type(parts) is not list
        or [(part.get("fold"), part.get("fraction")) for part in parts] != expected_pairs
        or complete.get("target_fold_validation_label_reads_before_its_prediction") != 0
        or complete.get("test_value_reads") != 0
        or complete.get("uploads") != 0
    ):
        raise ContractError("incumbent prediction binding differs")
    for part in parts:
        parquet_relative = str(part["parquet"]).replace("\\", "/")
        audit_relative = parquet_relative.removesuffix(".parquet") + ".json"
        parquet_pin = {
            "path": parquet_relative,
            "bytes": contained_path(root, parquet_relative, must_exist=True, kind="file")
            .stat()
            .st_size,
            "sha256": part["parquet_sha256"],
        }
        _assert_expected_pin(
            root, parquet_pin, label=f"incumbent part {part['fold']} {part['fraction']}"
        )
        audit_path = contained_path(root, audit_relative, must_exist=True, kind="file")
        audit_pin = {
            "path": audit_relative,
            "bytes": audit_path.stat().st_size,
            "sha256": part["audit_sha256"],
        }
        audit = _CONTEXT["authenticated_json_for_pin"](audit_pin, "incumbent part audit")
        if (
            audit.get("prefix_positions_sha256") is None
            or audit.get("parquet_sha256") != part["parquet_sha256"]
            or audit.get("target_fold_validation_labels_read_before_prediction") != 0
            or audit.get("validation_key_order_matches_frozen_oof") is not True
        ):
            raise ContractError("incumbent part row/prefix binding differs")
    return {"pins": observed, "part_count": 15, "exact_part_bindings": True}


def _verify_inner_incumbent_binding() -> dict[str, Any]:
    root = _workspace()
    binding = _config()["inner_incumbent_binding"]
    roles = ("gen5r6_config", "gen5r6_manifest", "gen5r6_split_audit")
    observed = {
        role: _assert_expected_pin(root, binding[role], label=f"inner incumbent {role}")
        for role in roles
    }
    reference_pins = binding.get("frozen_reference_source_pins")
    if type(reference_pins) is not dict or tuple(reference_pins) != (
        "science_projection",
        "pipeline",
        "postprocess",
        "rules",
    ):
        raise ContractError("fixed incumbent reference pin set differs")
    observed["frozen_reference_sources"] = {
        role: _assert_expected_pin(root, pin, label=f"fixed incumbent reference {role}")
        for role, pin in reference_pins.items()
    }
    manifest = _CONTEXT["authenticated_json_for_pin"](
        binding["gen5r6_manifest"], "inner incumbent manifest"
    )
    split_audit = _CONTEXT["authenticated_json_for_pin"](
        binding["gen5r6_split_audit"], "inner incumbent split audit"
    )
    artifacts = manifest.get("artifacts")
    teacher_prefix = "teacher_blind_predictions/curve/"
    teacher = (
        {
            relative: pin
            for relative, pin in artifacts.items()
            if type(relative) is str
            and relative.startswith(teacher_prefix)
            and relative.endswith(".npy")
        }
        if type(artifacts) is dict
        else {}
    )
    if (
        manifest.get("schema_version") != "p1_incumbent_residual_manifest.v5r6"
        or manifest.get("experiment_id") != "p1_incumbent_rule_distillation_neural_residual_v5r6"
        or manifest.get("candidate_created") is not False
        or manifest.get("uploaded") is not False
        or len(teacher) != binding["teacher_probability_files"]
        or binding["teacher_probability_files"] != 135
        or binding["teacher_probability_seed_order"] != [20260813, 20260829, 20260847]
    ):
        raise ContractError("inner incumbent manifest semantics differ")
    expected_names = {
        f"teacher_blind_predictions/curve/p{int(round(fraction * 100)):03d}/{fold}/"
        f"block_{block}/seed_{seed}.npy"
        for fold in ("2025_q2", "2025_q3", "2025_q4")
        for fraction in (0.4, 0.55, 0.7, 0.85, 1.0)
        for block in (1, 2, 3)
        for seed in (20260813, 20260829, 20260847)
    }
    if set(teacher) != expected_names:
        raise ContractError("inner incumbent teacher artifact set differs")
    artifact_root = contained_path(
        root,
        "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r6",
        must_exist=True,
        kind="directory",
    )
    for relative, expected in teacher.items():
        path = contained_path(artifact_root, relative, must_exist=True, kind="file")
        observed_pin = file_pin(path, relative=relative)
        if (
            type(expected) is not dict
            or set(expected) != {"bytes", "sha256"}
            or observed_pin["bytes"] != expected["bytes"]
            or observed_pin["sha256"] != expected["sha256"]
        ):
            raise ContractError(f"inner incumbent teacher pin differs: {relative}")
    prefixes = split_audit.get("prefixes")
    if (
        split_audit.get("input_target_columns_decoded") != 0
        or split_audit.get("frozen_oof_target_columns_decoded") != 0
        or split_audit.get("target_accessor_target_scalars_decoded_at_split_seal") != 0
        or type(prefixes) is not dict
        or tuple(prefixes) != ("p040", "p055", "p070", "p085", "p100")
        or any(tuple(prefixes[tag]) != ("2025_q2", "2025_q3", "2025_q4") for tag in prefixes)
        or any(
            item.get("current_run_prefix_selector_target_reads") != 0
            or item.get("exact_to_immutable_incumbent_fold_train_ids") is not True
            or not _is_sha(item.get("id_sha256_little_endian_int64"))
            or not _is_sha(item.get("validation_id_sha256_little_endian_int64"))
            for by_fold in prefixes.values()
            for item in by_fold.values()
        )
    ):
        raise ContractError("inner incumbent split audit semantics differ")
    return {
        "pins": observed,
        "teacher_probability_files": len(teacher),
        "prefix_cells": 15,
        "target_decodes_at_seal": 0,
    }


def _source_stat_boundary(*, hash_values: bool) -> dict[str, Any]:
    data = _data_dir()
    data_info = data.lstat()
    if not stat.S_ISDIR(data_info.st_mode) or _has_reparse(data):
        raise ContractError("data directory identity differs")
    result: dict[str, Any] = {
        "directory": {
            "device": int(data_info.st_dev),
            "inode": int(data_info.st_ino),
            "non_reparse": True,
        },
        "files": {},
    }
    for name, expected in _config()["source_pins"].items():
        path = contained_path(data, expected["path"], must_exist=True, kind="file")
        info = path.lstat()
        observed: dict[str, Any] = {
            "path": expected["path"],
            "bytes": int(info.st_size),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "nlink": int(info.st_nlink),
            "non_reparse": True,
            "opened": hash_values,
        }
        if observed["bytes"] != expected["bytes"] or observed["nlink"] != 1:
            raise ContractError(f"source stat differs: {name}")
        if hash_values:
            pin = file_pin(path, relative=expected["path"])
            if pin["sha256"] != expected["sha256"]:
                raise ContractError(f"source content differs: {name}")
            observed["sha256"] = pin["sha256"]
        result["files"][name] = observed
    return result


def _future_state() -> dict[str, bool]:
    root = _workspace()
    return {
        relative: os.path.lexists(contained_path(root, relative, must_exist=False))
        for relative in _config()["static_expected_absence"]
    }


def static_preflight(science_audit: dict[str, Any]) -> dict[str, Any]:
    if _CONTEXT.get("mode") != "check-only":
        raise ContractError("static preflight requires check-only mode")
    before = _CONTEXT["reverify_owner_roles"]()
    config_audit = _verify_config_semantics()
    lineage = _verify_v6_disposition()
    v9 = _verify_v9()
    incumbent = _verify_incumbent_binding()
    inner_incumbent = _verify_inner_incumbent_binding()
    source = _source_stat_boundary(hash_values=False)
    future = _future_state()
    if any(future.values()):
        raise ContractError("v6r2 future execution path exists during static stage")
    if any(science_audit.get(name) for name in ("fits", "predictions", "scores", "target_decodes")):
        raise ContractError("science static audit performed an operation")
    if science_audit.get("minimum_worst_station_layer_f1_delta") != 0.0:
        raise ContractError("science helper retains the old worst-group allowance")
    if (
        science_audit.get("fixed_postprocess_golden")
        != _config()["inner_incumbent_binding"]["fixed_postprocess_golden"]
    ):
        raise ContractError("science fixed postprocess golden differs")
    after = _CONTEXT["reverify_owner_roles"]()
    if before != after:
        raise ContractError("owner role pin map changed during static preflight")
    return {
        "status": "P1_MULTISCALE_CROSS_LAYER_OFFSET_DRIFT_V6R2_STATIC_CHECK_PASS",
        "verdict": "OWNER_STATIC_IMPLEMENTATION_AWAITING_FRESH_INDEPENDENT_QA",
        "generation": GENERATION,
        "config_audit": config_audit,
        "science_audit": science_audit,
        "lineage": lineage,
        "incumbent_binding": incumbent,
        "inner_incumbent_binding": inner_incumbent,
        "v9": v9,
        "source_boundary": source,
        "runtime_trust": _CONTEXT["runtime_static_audit"],
        "future_paths": future,
        "execution_closure_sha256": execution_closure_sha256(),
        "owner_role_pins": after,
        "bootstrap_observed_pin": _CONTEXT["bootstrap_observed_pin"],
        "engine_loaded": False,
        "operation_counts": {
            "independent_qa_receipts_created": 0,
            "execution_authorizations_created": 0,
            "attempt_locks_created": 0,
            "fits": 0,
            "predictions": 0,
            "target_decodes": 0,
            "scores": 0,
            "outputs": 0,
            "test_value_reads": 0,
            "candidates": 0,
            "ledger_appends": 0,
            "uploads": 0,
        },
        "actual_execution_authorized": False,
    }


def _dynamic_pin(relative: str) -> dict[str, Any]:
    path = contained_path(_workspace(), relative, must_exist=True, kind="file")
    observed = file_pin(path, relative=relative)
    return {name: observed[name] for name in ("path", "bytes", "sha256")}


def _execution_documents(
    *, hash_source: bool = True
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        _CONTEXT.get("mode") != "execute"
        or _CONTEXT.get("bootstrap_documents_prevalidated") is not True
    ):
        raise ContractError("bootstrap did not prevalidate execution documents")
    paths = _config()["canonical_paths"]
    qa_pin = _dynamic_pin(paths["independent_qa"])
    auth_pin = _dynamic_pin(paths["execution_authorization"])
    prevalidated = _CONTEXT.get("prevalidated_execution_documents")
    if (
        type(prevalidated) is not dict
        or prevalidated.get("qa_pin") != qa_pin
        or prevalidated.get("auth_pin") != auth_pin
        or type(prevalidated.get("qa_raw")) is not bytes
        or type(prevalidated.get("auth_raw")) is not bytes
        or type(prevalidated.get("qa")) is not dict
        or type(prevalidated.get("auth")) is not dict
    ):
        raise ContractError("prevalidated execution document buffers differ")
    qa_raw = _CONTEXT["authenticated_bytes_for_pin"](qa_pin, "independent QA")
    auth_raw = _CONTEXT["authenticated_bytes_for_pin"](auth_pin, "execution authorization")
    if qa_raw != prevalidated["qa_raw"] or auth_raw != prevalidated["auth_raw"]:
        raise ContractError("execution document bytes changed after prevalidation")
    # Semantics are consumed from the exact buffers parsed before any owner
    # execution module was loaded; path bytes are only re-authenticated for
    # freshness and are never reparsed into a second semantic object.
    qa = prevalidated["qa"]
    auth = prevalidated["auth"]
    expected_qa_fields = {
        "schema_version",
        "created_at_kst",
        "problem",
        "generation",
        "reviewer",
        "reviewer_independent",
        "verdict",
        "p0_count",
        "p1_count",
        "bootstrap",
        "owner_role_pins",
        "execution_closure_sha256",
        "v6_disposition_verified",
        "v9_binding",
        "source_identity",
        "incumbent_binding_verified",
        "inner_incumbent_binding_verified",
        "resource_ceiling_verified",
        "static_report_sha256",
        "actual_run_performed",
        "counters",
    }
    if type(qa) is not dict or set(qa) != expected_qa_fields:
        raise ContractError("independent QA field set differs")
    zero_counters = {
        "authorizations": 0,
        "attempt_locks": 0,
        "fits": 0,
        "predictions": 0,
        "target_decodes": 0,
        "scores": 0,
        "candidates": 0,
        "test_predictions": 0,
        "ledger_appends": 0,
        "uploads": 0,
    }
    if not (
        qa["schema_version"] == QA_SCHEMA
        and qa["problem"] == "P1"
        and qa["generation"] == GENERATION
        and type(qa["reviewer"]) is str
        and bool(qa["reviewer"])
        and qa["reviewer_independent"] is True
        and qa["verdict"] == "GO"
        and type(qa["p0_count"]) is int
        and qa["p0_count"] == 0
        and type(qa["p1_count"]) is int
        and qa["p1_count"] == 0
        and qa["bootstrap"] == _CONTEXT["bootstrap_observed_pin"]
        and qa["owner_role_pins"] == _CONTEXT["owner_role_pins"]
        and qa["execution_closure_sha256"] == execution_closure_sha256()
        and qa["v6_disposition_verified"] is True
        and qa["v9_binding"] == _config()["v9_binding"]
        and qa["incumbent_binding_verified"] is True
        and qa["inner_incumbent_binding_verified"] is True
        and qa["resource_ceiling_verified"] is True
        and _is_sha(qa["static_report_sha256"])
        and qa["actual_run_performed"] is False
        and type(qa["counters"]) is dict
        and set(qa["counters"]) == set(zero_counters)
        and all(
            type(qa["counters"][name]) is int and qa["counters"][name] == expected
            for name, expected in zero_counters.items()
        )
    ):
        raise ContractError("independent QA semantics differ")
    if hash_source:
        live_source = _source_stat_boundary(hash_values=True)
        if qa["source_identity"] != live_source:
            raise ContractError("independent QA source identity differs")
    expected_auth_fields = {
        "schema_version",
        "created_at_kst",
        "problem",
        "generation",
        "authorization",
        "user_message_reference",
        "qa_receipt",
        "bootstrap",
        "owner_role_pins_sha256",
        "execution_closure_sha256",
        "v9_binding",
        "execution_authorized",
        "research_curve_only",
        "one_shot_no_resume",
        "test_prediction_allowed",
        "candidate_creation_allowed",
        "ledger_append_allowed",
        "upload_allowed",
    }
    if type(auth) is not dict or set(auth) != expected_auth_fields:
        raise ContractError("execution authorization field set differs")
    if not (
        auth["schema_version"] == AUTH_SCHEMA
        and auth["problem"] == "P1"
        and auth["generation"] == GENERATION
        and auth["authorization"] == AUTHORIZATION_PHRASE
        and type(auth["user_message_reference"]) is str
        and bool(auth["user_message_reference"])
        and auth["qa_receipt"] == qa_pin
        and auth["bootstrap"] == _CONTEXT["bootstrap_observed_pin"]
        and auth["owner_role_pins_sha256"] == deep_sha256(_CONTEXT["owner_role_pins"])
        and auth["execution_closure_sha256"] == execution_closure_sha256()
        and auth["v9_binding"] == _config()["v9_binding"]
        and auth["execution_authorized"] is True
        and auth["research_curve_only"] is True
        and auth["one_shot_no_resume"] is True
        and auth["test_prediction_allowed"] is False
        and auth["candidate_creation_allowed"] is False
        and auth["ledger_append_allowed"] is False
        and auth["upload_allowed"] is False
    ):
        raise ContractError("execution authorization semantics differ")
    return qa, qa_pin, auth, auth_pin


_PRELOCK_MINT = object()
_POSTLOCK_MINT = object()
_LIVE_PRELOCK: dict[str, _LiveRecord] = {}
_LIVE_EXECUTION: dict[str, _LiveRecord] = {}


class PrelockCapability:
    __slots__ = ("_token",)

    def __init__(self, mint: object, token: str) -> None:
        if mint is not _PRELOCK_MINT:
            raise CapabilityError("canonical prelock mint required")
        self._token = token


class ExecutionCapability:
    __slots__ = ("_token",)

    def __init__(self, mint: object, token: str) -> None:
        if mint is not _POSTLOCK_MINT:
            raise CapabilityError("canonical postlock mint required")
        self._token = token


@dataclass
class _LiveRecord:
    capability: object
    qa_pin: dict[str, Any]
    auth_pin: dict[str, Any]
    closure_sha256: str
    source_identity_sha256: str
    v9_sha256: str
    lock_pin: dict[str, Any] | None
    phase: str
    started_monotonic: float
    counters: dict[str, int] = field(default_factory=dict)
    outputs: set[str] = field(default_factory=set)
    output_order: list[str] = field(default_factory=list)
    consumed: bool = False


def authorize_entry() -> PrelockCapability:
    if _LIVE_PRELOCK or _LIVE_EXECUTION:
        raise CapabilityError("a live P1 Gen6r2 capability already exists")
    _CONTEXT["reverify_owner_roles"]()
    qa, qa_pin, _auth, auth_pin = _execution_documents()
    v9 = _verify_v9()
    source = _source_stat_boundary(hash_values=True)
    if qa["source_identity"] != source:
        raise ContractError("source changed before prelock mint")
    token = secrets.token_hex(32)
    capability = PrelockCapability(_PRELOCK_MINT, token)
    _LIVE_PRELOCK[token] = _LiveRecord(
        capability=capability,
        qa_pin=qa_pin,
        auth_pin=auth_pin,
        closure_sha256=execution_closure_sha256(),
        source_identity_sha256=deep_sha256(source),
        v9_sha256=v9["sha256"],
        lock_pin=None,
        phase="authorized_prelock",
        started_monotonic=time.monotonic(),
        counters={},
    )
    return capability


def _prelock_record(capability: object) -> _LiveRecord:
    if not isinstance(capability, PrelockCapability):
        raise CapabilityError("opaque prelock capability required")
    record = _LIVE_PRELOCK.get(capability._token)
    if record is None or record.capability is not capability or record.consumed:
        raise CapabilityError("forged stale or replayed prelock capability")
    return record


def robust_write_exclusive(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes:
        raise ContractError("exclusive payload must be bytes")
    parent = _plain_chain(path.parent, require_target=True).resolve(strict=True)
    target = _plain_chain(path, require_target=False)
    if target.parent.resolve(strict=True) != parent:
        raise ContractError("exclusive write parent identity differs")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_BINARY
    descriptor = os.open(target, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("exclusive write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or _has_reparse(target):
        raise ContractError("exclusive output identity differs")
    if target.read_bytes() != payload:
        raise ContractError("exclusive output bytes differ after write")


def acquire_attempt_lock(capability: object) -> ExecutionCapability:
    record = _prelock_record(capability)
    _qa, qa_pin, _auth, auth_pin = _execution_documents(hash_source=False)
    if (
        qa_pin != record.qa_pin
        or auth_pin != record.auth_pin
        or execution_closure_sha256() != record.closure_sha256
        or _verify_v9()["sha256"] != record.v9_sha256
        or deep_sha256(_source_stat_boundary(hash_values=True)) != record.source_identity_sha256
    ):
        raise ContractError("authorization snapshot changed before attempt lock")
    paths = _config()["canonical_paths"]
    control = contained_path(_workspace(), paths["control"], must_exist=True, kind="directory")
    lock_path = contained_path(_workspace(), paths["attempt_lock"], must_exist=False)
    if lock_path.parent != control:
        raise ContractError("attempt lock escaped control directory")
    payload = canonical_json_bytes(
        {
            "schema_version": LOCK_SCHEMA,
            "generation": GENERATION,
            "qa_receipt": qa_pin,
            "execution_authorization": auth_pin,
            "bootstrap": _CONTEXT["bootstrap_observed_pin"],
            "owner_role_pins_sha256": deep_sha256(_CONTEXT["owner_role_pins"]),
            "execution_closure_sha256": record.closure_sha256,
            "source_identity_sha256": record.source_identity_sha256,
            "v9_sha256": record.v9_sha256,
            "prelock_nonce_sha256": hashlib.sha256(capability._token.encode()).hexdigest(),
            "one_shot": True,
        }
    )
    robust_write_exclusive(lock_path, payload)
    lock_observed = file_pin(lock_path, relative=paths["attempt_lock"])
    lock_pin = {name: lock_observed[name] for name in ("path", "bytes", "sha256")}
    token = secrets.token_hex(32)
    execution = ExecutionCapability(_POSTLOCK_MINT, token)
    record.consumed = True
    del _LIVE_PRELOCK[capability._token]
    _LIVE_EXECUTION[token] = _LiveRecord(
        capability=execution,
        qa_pin=qa_pin,
        auth_pin=auth_pin,
        closure_sha256=record.closure_sha256,
        source_identity_sha256=record.source_identity_sha256,
        v9_sha256=record.v9_sha256,
        lock_pin=lock_pin,
        phase="locked",
        started_monotonic=record.started_monotonic,
        counters={},
    )
    return execution


def _execution_record(capability: object) -> _LiveRecord:
    if not isinstance(capability, ExecutionCapability):
        raise CapabilityError("opaque postlock capability required")
    record = _LIVE_EXECUTION.get(capability._token)
    if record is None or record.capability is not capability or record.consumed:
        raise CapabilityError("forged stale or replayed postlock capability")
    _qa, qa_pin, _auth, auth_pin = _execution_documents(hash_source=False)
    if qa_pin != record.qa_pin or auth_pin != record.auth_pin:
        raise CapabilityError("execution documents changed after lock")
    if execution_closure_sha256() != record.closure_sha256:
        raise CapabilityError("execution closure changed after lock")
    _CONTEXT["reverify_owner_roles"]()
    if _verify_v9()["sha256"] != record.v9_sha256:
        raise CapabilityError("v9 anchor changed after lock")
    if (
        time.monotonic() - record.started_monotonic
        > _config()["resource_ceiling"]["maximum_wall_clock_seconds"]
    ):
        raise CapabilityError("wall-clock ceiling exceeded")
    lock = _config()["canonical_paths"]["attempt_lock"]
    observed = _dynamic_pin(lock)
    if observed != record.lock_pin:
        raise CapabilityError("attempt lock changed after mint")
    return record


def require_engine_capability(capability: object, entry_name: str) -> dict[str, Any]:
    record = _execution_record(capability)
    allowed_curve_entries = {
        "load_input_only_train",
        "target_accessor_init",
        "target_release",
        "ledger_init",
        "load_incumbent_catalog",
        "load_outer_cell",
        "load_inner_incumbent",
        "fit_predict_unit",
        "run_curve",
        "verify_dependency_closed_split",
        "build_three_block_inner_splits",
        "mean_seed_incumbent_probability",
        "fixed_incumbent_postprocess",
        "fit_robust_seasonal_graph_state",
        "apply_robust_seasonal_graph_state",
        "build_multiscale_geometry",
        "fit_fixed_slow_unary_head",
        "predict_fixed_slow_unary_probability",
        "protected_incumbent_union",
        "score_candidate_delta",
        "paired_bootstrap_f1_delta_ci90",
    }
    if type(entry_name) is not str or entry_name not in allowed_curve_entries:
        raise CapabilityError("engine entry name required")
    if record.phase != "curve":
        raise CapabilityError("science entry requires the live curve phase")
    return {
        "phase": record.phase,
        "lock": record.lock_pin,
        "closure_sha256": record.closure_sha256,
        "entry_name": entry_name,
    }


def enter_phase(capability: object, *, expected: str, new: str) -> None:
    record = _execution_record(capability)
    allowed = {
        ("locked", "loaded"),
        ("loaded", "curve"),
        ("curve", "publishing"),
    }
    if record.phase != expected or (expected, new) not in allowed:
        raise CapabilityError("invalid or replayed engine phase transition")
    if deep_sha256(_source_stat_boundary(hash_values=True)) != record.source_identity_sha256:
        raise ContractError("source identity changed across phase transition")
    record.phase = new


def bump_counter(capability: object, name: str, amount: int = 1) -> int:
    record = _execution_record(capability)
    if type(name) is not str or not name or not _exact_int(amount, minimum=1):
        raise ContractError("resource counter update differs")
    ceilings = {
        "baseline_fits": 60,
        "unary_fits": 60,
        "top_level_fits": 120,
        "seasonal_subfits": 960,
        "graph_edges": 780,
        "seasonal_irls_steps": 7680,
        "unary_lbfgs_iterations": 3840,
        "iterative_steps": 11520,
        "predictions": 60,
        "inner_commitments": 45,
        "cell_commitments": 15,
        "fold_commitments": 3,
        "predictions_complete": 1,
        "bootstrap_replicates": 25_000,
        "target_decodes": 10_000_000,
        "scores": 53,
        "files_written": 192,
    }
    if name not in ceilings:
        raise ContractError("unknown resource counter")
    value = record.counters.get(name, 0) + amount
    if value > ceilings[name]:
        raise ContractError(f"resource counter exceeded: {name}")
    record.counters[name] = value
    return value


def resource_snapshot(capability: object) -> dict[str, Any]:
    record = _execution_record(capability)
    elapsed = time.monotonic() - record.started_monotonic
    if elapsed > _config()["resource_ceiling"]["maximum_wall_clock_seconds"]:
        raise ContractError("wall-clock ceiling exceeded")
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    threads: dict[str, int] = {}
    for name in thread_variables:
        raw = os.environ.get(name, "1")
        if not raw.isdecimal() or not 1 <= int(raw) <= 8:
            raise ContractError(f"thread ceiling differs: {name}")
        threads[name] = int(raw)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise ContractError("GPU visibility is forbidden")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            raise ContractError("peak RSS measurement failed")
        peak_rss = int(counters.PeakWorkingSetSize)
    else:
        import resource

        raw_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_rss = raw_peak if sys.platform == "darwin" else raw_peak * 1024
    if peak_rss > _config()["resource_ceiling"]["maximum_peak_rss_bytes"]:
        raise ContractError("peak RSS ceiling exceeded")
    return {
        "elapsed_seconds": elapsed,
        "counters": dict(sorted(record.counters.items())),
        "thread_limits": threads,
        "gpu_visible": False,
        "vram_bytes": 0,
        "peak_rss_bytes": peak_rss,
    }


def _expected_output_sequence() -> list[str]:
    sequence: list[str] = []
    inner = 0
    cell = 0
    for fold in ("2025_q2", "2025_q3", "2025_q4"):
        for _fraction in (0.4, 0.55, 0.7, 0.85, 1.0):
            cell += 1
            for _block in (1, 2, 3):
                inner += 1
                sequence.extend(
                    (
                        f"models/inner_{inner:02d}.json",
                        f"inner_predictions/inner_{inner:02d}.bin",
                        f"blind_commitments/inner_{inner:02d}.json",
                    )
                )
            sequence.extend(
                (
                    f"models/cell_{cell:02d}.json",
                    f"prediction_parts/cell_{cell:02d}.bin",
                    f"blind_commitments/cell_{cell:02d}.json",
                )
            )
        sequence.append(f"blind_commitments/fold_{fold}.json")
    sequence.extend(
        (
            "blind_commitments/predictions_complete.json",
            "split_audit.json",
            "selective_target_audit.json",
            "metrics.json",
            "learning_curve_evidence.json",
            "result.json",
            "resource_audit.json",
            "manifest.json",
            "manifest.sha256",
        )
    )
    return sequence


def _output_allowlist() -> set[str]:
    return set(_expected_output_sequence())


def create_output_tree(capability: object) -> Path:
    record = _execution_record(capability)
    if record.phase != "locked":
        raise CapabilityError("output tree requires locked phase")
    root = _workspace()
    output_relative = _config()["canonical_paths"]["output"]
    output = contained_path(root, output_relative, must_exist=False)
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.mkdir(mode=0o700)
    for name in ("blind_commitments", "inner_predictions", "models", "prediction_parts"):
        (output / name).mkdir(mode=0o700)
    for path in (
        output,
        output / "blind_commitments",
        output / "inner_predictions",
        output / "models",
        output / "prediction_parts",
    ):
        if _has_reparse(path) or not path.is_dir():
            raise ContractError("output directory identity differs")
    return output


def write_output_exclusive(capability: object, relative: str, value: Any) -> dict[str, Any]:
    record = _execution_record(capability)
    if relative not in _output_allowlist():
        raise ContractError("output path is outside the exact allowlist")
    publishing = {
        "resource_audit.json",
        "metrics.json",
        "learning_curve_evidence.json",
        "result.json",
        "manifest.json",
        "manifest.sha256",
    }
    if relative in publishing:
        if record.phase != "publishing":
            raise CapabilityError("final artifact requires publishing phase")
    elif record.phase != "curve":
        raise CapabilityError("curve artifact requires curve phase")
    expected_sequence = _expected_output_sequence()
    ordinal = len(record.output_order)
    if ordinal >= len(expected_sequence) or relative != expected_sequence[ordinal]:
        raise ContractError("output creation order differs from the exact fold-major chain")
    if relative == "manifest.json":
        required = _output_allowlist() - {"manifest.json", "manifest.sha256"}
        if record.outputs != required:
            raise ContractError("manifest preceded the exact pre-manifest inventory")
    if relative == "manifest.sha256" and "manifest.json" not in record.outputs:
        raise ContractError("manifest sidecar preceded manifest")
    if relative == "manifest.sha256":
        output_root = contained_path(
            _workspace(), _config()["canonical_paths"]["output"], must_exist=True, kind="directory"
        )
        manifest = contained_path(output_root, "manifest.json", must_exist=True, kind="file")
        expected_sidecar = (file_pin(manifest, relative="manifest.json")["sha256"] + "\n").encode(
            "ascii"
        )
        if type(value) is not bytes or value != expected_sidecar:
            raise ContractError("manifest sidecar does not bind the exact manifest bytes")
    if relative in record.outputs:
        raise FileExistsError(relative)
    output = contained_path(
        _workspace(), _config()["canonical_paths"]["output"], must_exist=True, kind="directory"
    )
    target = contained_path(output, relative, must_exist=False)
    payload = value if type(value) is bytes else canonical_json_bytes(value)
    robust_write_exclusive(target, payload)
    bump_counter(capability, "files_written")
    observed = file_pin(target, relative=relative)
    record.outputs.add(relative)
    record.output_order.append(relative)
    return {name: observed[name] for name in ("path", "bytes", "sha256")}


def verify_output_inventory(capability: object, *, final: bool) -> dict[str, Any]:
    _execution_record(capability)
    output = contained_path(
        _workspace(), _config()["canonical_paths"]["output"], must_exist=True, kind="directory"
    )
    files: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()
    for path in output.rglob("*"):
        relative = path.relative_to(output).as_posix()
        if _has_reparse(path):
            raise ContractError("output link/reparse entry forbidden")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            if path.lstat().st_nlink != 1 or relative not in _output_allowlist():
                raise ContractError("output file identity/allowlist differs")
            files[relative] = file_pin(path, relative=relative)
        else:
            raise ContractError("special output entry forbidden")
    if directories != {"blind_commitments", "inner_predictions", "models", "prediction_parts"}:
        raise ContractError("output directory inventory differs")
    if final and set(files) != _output_allowlist():
        raise ContractError("final output file inventory differs")
    total_bytes = sum(pin["bytes"] for pin in files.values())
    if total_bytes > _config()["resource_ceiling"]["maximum_artifact_disk_bytes"]:
        raise ContractError("artifact disk ceiling exceeded")
    return {
        "directories": sorted(directories),
        "files": {name: files[name] for name in sorted(files)},
        "file_count": len(files),
        "total_bytes": total_bytes,
        "inventory_sha256": deep_sha256(
            {
                name: {key: pin[key] for key in ("bytes", "sha256")}
                for name, pin in sorted(files.items())
            }
        ),
    }


def _verify_commitment_chain(inventory: dict[str, Any]) -> str:
    output_relative = _config()["canonical_paths"]["output"]
    names = [name for name in _expected_output_sequence() if name.startswith("blind_commitments/")]
    prior = hashlib.sha256(b"p1_v6r2_blind_commitment_genesis").hexdigest()
    inner = 0
    cell = 0
    folds = 0
    completed = 0
    for name in names:
        observed = inventory["files"][name]
        pin = {
            "path": f"{output_relative}/{name}",
            "bytes": observed["bytes"],
            "sha256": observed["sha256"],
        }
        event = _CONTEXT["strict_dynamic_json_for_pin"](pin, f"commitment {name}")
        if (
            type(event) is not dict
            or event.get("generation") != GENERATION
            or event.get("prior_event_sha256") != prior
        ):
            raise ContractError("blind commitment prior hash differs")
        claimed = event.get("event_sha256")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if not _is_sha(claimed) or claimed != deep_sha256(body):
            raise ContractError("blind commitment event hash differs")
        if name.startswith("blind_commitments/inner_"):
            inner += 1
            expected_cell = (inner - 1) // 3 + 1
            expected_block = (inner - 1) % 3 + 1
            expected_fold = ("2025_q2", "2025_q3", "2025_q4")[(expected_cell - 1) // 5]
            expected_fraction = (0.4, 0.55, 0.7, 0.85, 1.0)[(expected_cell - 1) % 5]
            prediction_name = f"inner_predictions/inner_{inner:02d}.bin"
            model_name = f"models/inner_{inner:02d}.json"
            expected_prediction_pin = {
                key: inventory["files"][prediction_name][key] for key in ("path", "bytes", "sha256")
            }
            expected_model_pin = {
                key: inventory["files"][model_name][key] for key in ("path", "bytes", "sha256")
            }
            if not (
                event.get("schema_version") == "p1_v6r2_inner_commitment.v1"
                and type(event.get("ordinal")) is int
                and event.get("ordinal") == inner
                and type(event.get("cell")) is int
                and event.get("cell") == expected_cell
                and type(event.get("block")) is int
                and event.get("block") == expected_block
                and event.get("fold") == expected_fold
                and event.get("fraction") == expected_fraction
                and type(event.get("target_scalars_decoded_before_commitment")) is int
                and event.get("target_scalars_decoded_before_commitment") == 0
                and event.get("prediction_bundle") == expected_prediction_pin
                and event.get("model") == expected_model_pin
            ):
                raise ContractError("inner commitment semantics/order differs")
        elif name.startswith("blind_commitments/cell_"):
            cell += 1
            expected_fold = ("2025_q2", "2025_q3", "2025_q4")[(cell - 1) // 5]
            expected_fraction = (0.4, 0.55, 0.7, 0.85, 1.0)[(cell - 1) % 5]
            prediction_name = f"prediction_parts/cell_{cell:02d}.bin"
            model_name = f"models/cell_{cell:02d}.json"
            expected_prediction_pin = {
                key: inventory["files"][prediction_name][key] for key in ("path", "bytes", "sha256")
            }
            expected_model_pin = {
                key: inventory["files"][model_name][key] for key in ("path", "bytes", "sha256")
            }
            if not (
                event.get("schema_version") == "p1_v6r2_cell_commitment.v1"
                and type(event.get("cell")) is int
                and event.get("cell") == cell
                and event.get("fold") == expected_fold
                and event.get("fraction") == expected_fraction
                and type(event.get("active_outer_target_scalars_decoded_before_commitment")) is int
                and event.get("active_outer_target_scalars_decoded_before_commitment") == 0
                and event.get("prediction_bundle") == expected_prediction_pin
                and event.get("model") == expected_model_pin
            ):
                raise ContractError("cell commitment semantics/order differs")
        elif name.startswith("blind_commitments/fold_"):
            expected_fold = ("2025_q2", "2025_q3", "2025_q4")[folds]
            folds += 1
            if not (
                event.get("schema_version") == "p1_v6r2_fold_commitment.v1"
                and event.get("fold") == expected_fold
                and type(event.get("cell_count")) is int
                and event.get("cell_count") == 5
                and type(event.get("active_fold_target_scalars_decoded_before_commitment")) is int
                and event.get("active_fold_target_scalars_decoded_before_commitment") == 0
            ):
                raise ContractError("fold commitment semantics/order differs")
        else:
            completed += 1
            if not (
                name == "blind_commitments/predictions_complete.json"
                and event.get("schema_version") == "p1_v6r2_predictions_complete.v1"
                and type(event.get("inner_commitments")) is int
                and event.get("inner_commitments") == 45
                and type(event.get("cell_commitments")) is int
                and event.get("cell_commitments") == 15
                and type(event.get("fold_commitments")) is int
                and event.get("fold_commitments") == 3
                and type(event.get("aggregate_target_scalars_decoded_before_completion")) is int
                and event.get("aggregate_target_scalars_decoded_before_completion") == 0
                and event.get("candidate_created") is False
                and event.get("test_prediction_created") is False
                and event.get("ledger_appended") is False
                and event.get("uploaded") is False
            ):
                raise ContractError("predictions_complete semantics differs")
        prior = claimed
    if (inner, cell, folds, completed) != (45, 15, 3, 1):
        raise ContractError("blind commitment chain cardinality differs")
    return prior


def _exact_completion_counters() -> dict[str, int]:
    return {
        "baseline_fits": 60,
        "unary_fits": 60,
        "top_level_fits": 120,
        "predictions": 60,
        "inner_commitments": 45,
        "cell_commitments": 15,
        "fold_commitments": 3,
        "predictions_complete": 1,
        "bootstrap_replicates": 25_000,
        # 45 sealed inner blocks + 5 fraction aggregates + 3 full-fold
        # aggregates. Inner gates reuse those block scores.
        "scores": 53,
        "files_written": 192,
    }


def _validate_completion_counters(
    observed: dict[str, Any], *, expected_file_count: int
) -> dict[str, int]:
    exact = _exact_completion_counters()
    if exact["files_written"] != expected_file_count:
        raise ContractError("completion file count arithmetic differs")
    if type(observed) is not dict or any(
        type(observed.get(name)) is not int or observed.get(name) != value
        for name, value in exact.items()
    ):
        raise ContractError("completion resource/commitment minima differ")
    return exact


def complete_capability(capability: object) -> dict[str, Any]:
    record = _execution_record(capability)
    if record.phase != "publishing":
        raise CapabilityError("completion requires publishing phase")
    expected_sequence = _expected_output_sequence()
    if record.output_order != expected_sequence or record.outputs != set(expected_sequence):
        raise ContractError("completion preceded the exact output sequence")
    _validate_completion_counters(record.counters, expected_file_count=len(expected_sequence))
    inventory = verify_output_inventory(capability, final=True)
    commitment_head = _verify_commitment_chain(inventory)
    output_relative = _config()["canonical_paths"]["output"]
    manifest_pin = {
        "path": f"{output_relative}/manifest.json",
        "bytes": inventory["files"]["manifest.json"]["bytes"],
        "sha256": inventory["files"]["manifest.json"]["sha256"],
    }
    manifest = _CONTEXT["strict_dynamic_json_for_pin"](manifest_pin, "final manifest")
    pre_manifest_files = {
        name: pin
        for name, pin in inventory["files"].items()
        if name not in {"manifest.json", "manifest.sha256"}
    }
    pre_manifest_sha = deep_sha256(
        {
            name: {key: pin[key] for key in ("bytes", "sha256")}
            for name, pin in sorted(pre_manifest_files.items())
        }
    )
    if (
        type(manifest) is not dict
        or manifest.get("schema_version") != "p1_v6r2_manifest.v1"
        or manifest.get("generation") != GENERATION
        or manifest.get("execution_closure_sha256") != record.closure_sha256
        or manifest.get("pre_manifest_inventory_sha256") != pre_manifest_sha
        or manifest.get("commitment_chain_head_sha256") != commitment_head
        or manifest.get("candidate_created") is not False
        or manifest.get("test_prediction_created") is not False
        or manifest.get("ledger_appended") is not False
        or manifest.get("uploaded") is not False
    ):
        raise ContractError("final manifest semantics differ")
    sidecar_pin = {
        "path": f"{output_relative}/manifest.sha256",
        "bytes": inventory["files"]["manifest.sha256"]["bytes"],
        "sha256": inventory["files"]["manifest.sha256"]["sha256"],
    }
    sidecar = _CONTEXT["authenticated_bytes_for_pin"](sidecar_pin, "manifest sidecar")
    if sidecar != (manifest_pin["sha256"] + "\n").encode("ascii"):
        raise ContractError("final manifest sidecar binding differs")
    if deep_sha256(_source_stat_boundary(hash_values=True)) != record.source_identity_sha256:
        raise ContractError("source identity changed before completion")
    snapshot = resource_snapshot(capability)
    snapshot["final_inventory"] = inventory
    record.phase = "complete"
    record.consumed = True
    del _LIVE_EXECUTION[capability._token]
    return snapshot


def capability_registry_counts() -> dict[str, int]:
    return {"prelock": len(_LIVE_PRELOCK), "postlock": len(_LIVE_EXECUTION)}


_CONTEXT["require_engine_capability"] = require_engine_capability

__all__ = [
    "AUTHORIZATION_PHRASE",
    "CapabilityError",
    "ContractError",
    "ExecutionCapability",
    "PrelockCapability",
    "acquire_attempt_lock",
    "authorize_entry",
    "bump_counter",
    "capability_registry_counts",
    "canonical_json_bytes",
    "complete_capability",
    "create_output_tree",
    "deep_sha256",
    "enter_phase",
    "execution_closure_sha256",
    "file_pin",
    "require_engine_capability",
    "resource_snapshot",
    "robust_write_exclusive",
    "static_preflight",
    "verify_output_inventory",
    "write_output_exclusive",
]
