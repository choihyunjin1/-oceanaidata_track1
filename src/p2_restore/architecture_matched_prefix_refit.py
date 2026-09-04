"""Fail-closed static contract for the P2 architecture-matched reference.

This module contains only contract, path, SHA, key-schema, and append-only I/O
guards.  It deliberately imports no numerical training framework and exposes no
fit or prediction function.  Stage runners may import a separately authorized
execution engine only after these guards succeed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MODE = "ARCHITECTURE_MATCHED_TIME_SAFE_BASELINE"
CONFIG_RELATIVE = "configs/experiments/p2_architecture_matched_time_safe_baseline_v1.json"
CONFIG_SHA256 = "0f03001e8890683ad2721b15de3609cefed868c44cb835a9a47c8ca62b01369b"
STAGE_A = "STAGE_A_REFERENCE"
STAGE_B = "STAGE_B_CHALLENGER"
PREFIX_FRACTIONS = [0.4, 0.55, 0.7, 0.85, 1.0]
PIPELINE_SEEDS = [20260823, 20260824, 20260825]
OUTER_FOLDS = ["outer_2024_sep_oct", "outer_2025_may_jun", "outer_2025_jul_aug"]
DEEP_COMPONENTS = [
    "depth_query_bitcn",
    "moment_units_scratch",
    "lsti_style",
    "timemixerpp_style",
]


class ArchitectureContractError(ValueError):
    """Raised before any execution when the static contract is not exact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArchitectureContractError(f"expected JSON object: {path}")
    return value


def workspace_path(
    root: Path, relative: str | Path, *, must_exist: bool = True
) -> Path:
    workspace = root.resolve(strict=True)
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ArchitectureContractError("path must be workspace-relative and non-traversing")
    resolved = (workspace / candidate).resolve(strict=must_exist)
    if not resolved.is_relative_to(workspace):
        raise ArchitectureContractError("path escapes workspace")
    return resolved


def contained_path(output_root: Path, child: str | Path) -> Path:
    base = output_root.resolve(strict=False)
    candidate = Path(child)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ArchitectureContractError("output child path is unsafe")
    target = (base / candidate).resolve(strict=False)
    if target == base or not target.is_relative_to(base):
        raise ArchitectureContractError("output path escapes its canonical directory")
    return target


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ArchitectureContractError(f"{name} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p2_architecture_matched_time_safe_baseline.v1":
        raise ArchitectureContractError("config schema changed")
    if config.get("problem") != "P2" or config.get("comparison_mode") != MODE:
        raise ArchitectureContractError("P2-only comparison identity changed")
    if config.get("exact_official_incumbent_comparison") is not False:
        raise ArchitectureContractError("architecture-matched baseline must never be exact")
    if config.get("explicitly_not_exact_official_incumbent") is not True:
        raise ArchitectureContractError("explicit non-exact label is required")
    if config.get("training_recipe_origin") != "NEW_PREREGISTERED_TIME_SAFE_RECIPE":
        raise ArchitectureContractError("training recipe origin changed")
    if config.get("upload_allowed") is not False or config.get("official_submission_count") != 0:
        raise ArchitectureContractError("upload state changed")

    canonical = config.get("canonical_paths")
    if not isinstance(canonical, Mapping):
        raise ArchitectureContractError("canonical paths are missing")
    expected_paths = {
        "config": CONFIG_RELATIVE,
        "stage_a_output": "artifacts/p2_architecture_matched_reference_v1",
        "stage_a_control": "artifacts/p2_architecture_matched_reference_v1_control",
        "stage_a_authorization": (
            "artifacts/p2_architecture_matched_reference_v1_control/authorization.json"
        ),
        "stage_a_attempt_lock": (
            "artifacts/p2_architecture_matched_reference_v1_control/attempt.lock"
        ),
        "stage_b_output": "artifacts/p2_meaningful_learning_curve_generation_v2",
        "stage_b_control": "artifacts/p2_meaningful_learning_curve_generation_v2_control",
        "stage_b_authorization": (
            "artifacts/p2_meaningful_learning_curve_generation_v2_control/authorization.json"
        ),
        "stage_b_attempt_lock": (
            "artifacts/p2_meaningful_learning_curve_generation_v2_control/attempt.lock"
        ),
    }
    if dict(canonical) != expected_paths:
        raise ArchitectureContractError("canonical path mapping changed")

    policy = config.get("execution_policy")
    required_true = {
        "static_check_only_now",
        "actual_stage_a_requires_separate_append_only_authorization",
        "actual_stage_b_requires_separate_append_only_authorization",
        "check_only_must_not_import_training_or_prediction_modules",
        "stage_b_must_verify_stage_a_seal_before_challenger_import",
        "stage_b_must_verify_stage_a_seal_before_attempt_lock",
    }
    if not isinstance(policy, Mapping) or any(policy.get(key) is not True for key in required_true):
        raise ArchitectureContractError("execution fail-close policy changed")
    if policy.get("output_materialization") != "O_EXCL_ONLY":
        raise ArchitectureContractError("output materialization must remain O_EXCL")
    for key in (
        "rerun_allowed",
        "frozen_mutation_allowed",
        "submission_mutation_allowed",
        "automatic_upload_allowed",
    ):
        if policy.get(key) is not False:
            raise ArchitectureContractError(f"{key} must remain false")

    graph = config.get("deployed_inference_graph")
    if not isinstance(graph, Mapping):
        raise ArchitectureContractError("deployed inference graph is missing")
    if graph.get("gate_route_layers") != [2, 4]:
        raise ArchitectureContractError("gate route layers changed")
    if graph.get("layer_extrapolation_factors") != {"2": 10.0, "3": 0.0, "4": 2.0}:
        raise ArchitectureContractError("fixed extrapolation factors changed")
    for role in ("source_pins", "deployed_asset_pins"):
        pins = graph.get(role)
        if not isinstance(pins, Mapping) or not pins:
            raise ArchitectureContractError(f"{role} is empty")
        if any(not _is_sha256(digest) for digest in pins.values()):
            raise ArchitectureContractError(f"{role} contains an invalid SHA")
    topology = graph.get("deployed_checkpoint_topology")
    if not isinstance(topology, Mapping) or list(topology) != DEEP_COMPONENTS:
        raise ArchitectureContractError("deployed checkpoint topology changed")
    if [len(topology[name]) for name in DEEP_COMPONENTS] != [1, 1, 3, 3]:
        raise ArchitectureContractError("deployed checkpoint counts changed")
    for entries in topology.values():
        for entry in entries:
            if (
                not isinstance(entry, Mapping)
                or not _is_sha256(entry.get("sha256"))
                or not isinstance(entry.get("seed"), int)
                or not isinstance(entry.get("epochs"), int)
            ):
                raise ArchitectureContractError("invalid deployed checkpoint pin")

    recipe = config.get("training_recipe")
    if not isinstance(recipe, Mapping):
        raise ArchitectureContractError("training recipe is missing")
    if recipe.get("complete_pipeline_seed_ids") != PIPELINE_SEEDS:
        raise ArchitectureContractError("three complete pipeline seeds changed")
    if recipe.get("seed_aggregation") != "PREDICTION_MEAN_THEN_METRIC":
        raise ArchitectureContractError("seed aggregation changed")
    if recipe.get("prefix_fractions") != PREFIX_FRACTIONS or recipe.get("embargo_days") != 7:
        raise ArchitectureContractError("prefix or embargo contract changed")
    folds = recipe.get("outer_folds")
    if not isinstance(folds, Sequence) or [fold.get("name") for fold in folds] != OUTER_FOLDS:
        raise ArchitectureContractError("outer fold keys changed")
    inner = recipe.get("inner_oof")
    if not isinstance(inner, Mapping) or inner.get("validation_fraction_edges") != [
        0.55,
        0.7,
        0.85,
        1.0,
    ]:
        raise ArchitectureContractError("nested chronological inner OOF changed")
    if inner.get("validation_block_count") != 3 or inner.get("future_or_outer_labels_allowed") is not False:
        raise ArchitectureContractError("inner OOF safety rule changed")
    epochs = recipe.get("epoch_selection")
    if not isinstance(epochs, Mapping) or epochs.get("epoch_grid") != [12, 20, 28, 36, 44, 52]:
        raise ArchitectureContractError("epoch selection grid changed")
    meta = recipe.get("meta_training")
    if not isinstance(meta, Mapping) or meta.get("frozen_stack_or_gate_reuse_allowed") is not False:
        raise ArchitectureContractError("prefix-local meta-training rule changed")
    bootstrap = recipe.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or bootstrap != {
        "cluster": "KST_day",
        "replicates": 5000,
        "confidence_level": 0.9,
        "seed": 20260823,
    }:
        raise ArchitectureContractError("bootstrap contract changed")

    stage_a = config.get("stage_a_reference_contract")
    if not isinstance(stage_a, Mapping):
        raise ArchitectureContractError("Stage-A contract is missing")
    if (
        stage_a.get("generate_all_five_prefix_oof_before_challenger") is not True
        or stage_a.get("reference_100_percent_oof_must_be_sealed_before_challenger_scoring")
        is not True
        or stage_a.get("row_predictions_may_contain_targets") is not False
    ):
        raise ArchitectureContractError("Stage-A seal-before-challenger rule changed")
    stage_b = config.get("stage_b_challenger_contract")
    if not isinstance(stage_b, Mapping) or stage_b.get("hypothesis_count") != 1:
        raise ArchitectureContractError("Stage-B preregistration changed")
    if stage_b.get("stage_a_seal_must_verify_before_any_challenger_import_fit_or_score") is not True:
        raise ArchitectureContractError("Stage-B import barrier changed")
    hypothesis = stage_b.get("hypotheses", [{}])[0]
    if (
        hypothesis.get("id") != "H2_JOINT_PROFILE_LATENT_RESIDUAL"
        or hypothesis.get("rank") != 2
        or hypothesis.get("residual_scale") != 1.0
        or hypothesis.get("hyperparameter_searches") != 0
        or hypothesis.get("score_derived_tuning") is not False
        or hypothesis.get("fixed_complete_pipeline_seed_ids") != PIPELINE_SEEDS
    ):
        raise ArchitectureContractError("Stage-B structural hypothesis changed")


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
        raise ArchitectureContractError("only the canonical config path is accepted")
    observed = sha256_file(canonical)
    if observed != CONFIG_SHA256:
        raise ArchitectureContractError("canonical config SHA mismatch")
    config = _json_object(canonical)
    validate_config(config)
    if supplied_config is not None and dict(supplied_config) != config:
        raise ArchitectureContractError("supplied config fails full deep equality")
    return config


def _verify_pin_group(root: Path, pins: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for relative, expected in pins.items():
        path = workspace_path(root, relative)
        observed = sha256_file(path)
        if observed != expected:
            raise ArchitectureContractError(f"{role} SHA mismatch: {relative}")
        verified[str(relative)] = {"sha256": observed, "bytes": path.stat().st_size}
    return verified


def verify_deployed_graph(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    graph = config["deployed_inference_graph"]
    source = _verify_pin_group(root, graph["source_pins"], role="graph source")
    assets = _verify_pin_group(root, graph["deployed_asset_pins"], role="deployed asset")
    checkpoints: dict[str, Any] = {}
    for name, entries in graph["deployed_checkpoint_topology"].items():
        checkpoints[name] = []
        for entry in entries:
            verified = _verify_pin_group(
                root, {entry["path"]: entry["sha256"]}, role=f"{name} checkpoint"
            )
            checkpoints[name].append({**dict(entry), "bytes": verified[entry["path"]]["bytes"]})

    deep_result_path = workspace_path(root, "artifacts/p2_deep_finalists_v1/result.json")
    deep_result = _json_object(deep_result_path)
    observed_topology: dict[str, list[dict[str, Any]]] = {}
    for name in DEEP_COMPONENTS:
        observed_topology[name] = [
            {
                "seed": entry["seed"],
                "epochs": entry["epochs"],
                "path": Path(entry["checkpoint"]).as_posix(),
                "sha256": entry["checkpoint_sha256"],
            }
            for entry in deep_result.get("full_models", {}).get(name, [])
        ]
    if observed_topology != graph["deployed_checkpoint_topology"]:
        raise ArchitectureContractError("deep-result checkpoint topology fails deep equality")
    return {
        "source_pins": source,
        "deployed_asset_pins": assets,
        "checkpoint_pins": checkpoints,
        "deep_result_topology_deep_equal": True,
        "deployed_graph_manifest_sha256": canonical_mapping_sha256(graph),
    }


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.reader(handle), None)
    if row is None:
        raise ArchitectureContractError(f"empty CSV: {path.name}")
    return row


def _key_digest(path: Path, expected_header: list[str]) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ArchitectureContractError(f"CSV header changed: {path.name}")
        for row in reader:
            key = [row["station"], row["layer"], row["time"]]
            digest.update(
                json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
            rows += 1
    return rows, digest.hexdigest()


def inspect_schema_and_keys(data_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Read headers and public station/layer/time keys only; never read target values."""

    directory = data_dir.resolve(strict=True)
    data = config["data_contract"]
    observations = directory / "observations.csv"
    test_index = directory / "test_index.csv"
    sample = directory / "sample_submission.csv"
    for path in (observations, test_index, sample):
        if not path.is_file():
            raise ArchitectureContractError(f"required data file is missing: {path.name}")
    if _header(observations) != data["observations_header"]:
        raise ArchitectureContractError("observations header changed")
    test_rows, test_digest = _key_digest(test_index, data["test_index_header"])
    sample_rows, sample_digest = _key_digest(sample, data["sample_submission_header"])
    if test_rows != data["canonical_test_rows"] or sample_rows != test_rows:
        raise ArchitectureContractError("public key row count changed")
    if test_digest != data["canonical_station_layer_time_key_sha256"]:
        raise ArchitectureContractError("test-index key digest changed")
    if sample_digest != test_digest:
        raise ArchitectureContractError("sample-submission keys differ from test-index keys")
    return {
        "access_scope": "HEADERS_AND_STATION_LAYER_TIME_KEYS_ONLY",
        "observations_header_only": True,
        "target_columns_read": [],
        "rows": test_rows,
        "station_layer_time_key_sha256": test_digest,
        "test_sample_key_deep_equal": True,
    }


def stage_paths(root: Path, config: Mapping[str, Any], stage: str) -> dict[str, Path]:
    if stage not in {STAGE_A, STAGE_B}:
        raise ArchitectureContractError("unknown stage")
    prefix = "stage_a" if stage == STAGE_A else "stage_b"
    canonical = config["canonical_paths"]
    return {
        "output": workspace_path(root, canonical[f"{prefix}_output"], must_exist=False),
        "control": workspace_path(root, canonical[f"{prefix}_control"], must_exist=False),
        "authorization": workspace_path(
            root, canonical[f"{prefix}_authorization"], must_exist=False
        ),
        "attempt_lock": workspace_path(
            root, canonical[f"{prefix}_attempt_lock"], must_exist=False
        ),
    }


def static_preflight(
    root: Path,
    data_dir: Path,
    *,
    requested_config: Path | None = None,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_canonical_config(
        root, requested_config, supplied_config=supplied_config
    )
    graph = verify_deployed_graph(root, config)
    keys = inspect_schema_and_keys(data_dir, config)
    stage_a = stage_paths(root, config, STAGE_A)
    stage_b = stage_paths(root, config, STAGE_B)
    central = config["central_contract"]
    central_verified = _verify_pin_group(root, {central["path"]: central["sha256"]}, role="central")
    return {
        "schema_version": "p2_architecture_matched_static_preflight.v1",
        "status": "PASS_STATIC_CHECK_ONLY",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "canonical_config_path": CONFIG_RELATIVE,
        "canonical_config_sha256": CONFIG_SHA256,
        "deployed_graph": graph,
        "training_recipe_sha256": canonical_mapping_sha256(config["training_recipe"]),
        "central_contract": central_verified,
        "schema_and_keys": keys,
        "stage_a_state": {name: path.exists() for name, path in stage_a.items()},
        "stage_b_state": {name: path.exists() for name, path in stage_b.items()},
        "training_modules_imported": False,
        "prediction_modules_imported": False,
        "files_written": 0,
        "attempt_locks_created": 0,
        "registries_created": 0,
        "fits": 0,
        "predictions": 0,
        "uploads": 0,
        "resource_estimate": config["resource_estimate"],
    }


def exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short exclusive write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    exclusive_bytes(path, canonical_json_bytes(value) + b"\n")


def verify_execution_authorization(
    root: Path, config: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    paths = stage_paths(root, config, stage)
    if paths["output"].exists():
        raise FileExistsError("canonical append-only output already exists")
    if paths["attempt_lock"].exists():
        raise FileExistsError("canonical one-shot attempt was already consumed")
    if not paths["authorization"].is_file():
        raise PermissionError("separate append-only execution authorization is missing")
    authorization = _json_object(paths["authorization"])
    expected_phrase = f"AUTHORIZE_P2_{stage}:{CONFIG_SHA256}"
    checks = {
        "schema": authorization.get("schema_version") == "p2_architecture_execution_authorization.v1",
        "stage": authorization.get("stage") == stage,
        "config_path": authorization.get("config_path") == CONFIG_RELATIVE,
        "config_sha": authorization.get("config_sha256") == CONFIG_SHA256,
        "explicit": authorization.get("authorization") == expected_phrase,
        "user_reference": bool(authorization.get("user_message_reference")),
        "execution_authorized": authorization.get("execution_authorized") is True,
        "upload_allowed": authorization.get("upload_allowed") is False,
    }
    if not all(checks.values()):
        raise PermissionError(
            f"execution authorization failed: {sorted(key for key, value in checks.items() if not value)}"
        )
    return authorization


def consume_attempt_lock(
    root: Path,
    config: Mapping[str, Any],
    stage: str,
    *,
    authorization_sha256: str,
) -> Path:
    paths = stage_paths(root, config, stage)
    payload = {
        "schema_version": "p2_architecture_attempt_lock.v1",
        "stage": stage,
        "config_path": CONFIG_RELATIVE,
        "config_sha256": CONFIG_SHA256,
        "authorization_sha256": authorization_sha256,
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "rerun_allowed": False,
        "upload_allowed": False,
    }
    exclusive_json(paths["attempt_lock"], payload)
    return paths["attempt_lock"]


def verify_stage_a_seal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = stage_paths(root, config, STAGE_A)
    output = paths["output"].resolve(strict=True)
    artifacts = config["stage_a_reference_contract"]["artifacts"]
    resolved = {name: contained_path(output, relative) for name, relative in artifacts.items()}
    for name, path in resolved.items():
        if not path.is_file():
            raise ArchitectureContractError(f"Stage-A artifact is missing: {name}")
    seal = _json_object(resolved["seal"])
    if seal.get("schema_version") != "p2_architecture_matched_reference.seal.v1":
        raise ArchitectureContractError("Stage-A seal schema changed")
    if seal.get("comparison_mode") != MODE or seal.get("exact_official_incumbent_comparison") is not False:
        raise ArchitectureContractError("Stage-A seal identity changed")
    if seal.get("complete") is not True or seal.get("all_five_prefixes_sealed") is not True:
        raise ArchitectureContractError("Stage-A curve is incomplete")
    if seal.get("challenger_fit_or_score_count_before_seal") != 0:
        raise ArchitectureContractError("challenger was evaluated before Stage-A sealing")
    manifest = _json_object(resolved["manifest"])
    if manifest.get("schema_version") != "p2_architecture_matched_reference.manifest.v1":
        raise ArchitectureContractError("Stage-A manifest schema changed")
    oof_by_fraction = manifest.get("reference_oof_by_fraction")
    expected_fraction_keys = {str(value) for value in PREFIX_FRACTIONS}
    if not isinstance(oof_by_fraction, Mapping) or set(oof_by_fraction) != expected_fraction_keys:
        raise ArchitectureContractError("Stage-A manifest must bind all five prefix OOF files")
    verified_oof: dict[str, dict[str, Any]] = {}
    for fraction, pin in oof_by_fraction.items():
        if not isinstance(pin, Mapping) or not _is_sha256(pin.get("sha256")):
            raise ArchitectureContractError("invalid prefix OOF pin")
        oof_path = contained_path(output, str(pin.get("path", "")))
        if not oof_path.is_file() or sha256_file(oof_path) != pin["sha256"]:
            raise ArchitectureContractError(f"prefix OOF SHA mismatch: {fraction}")
        verified_oof[fraction] = {
            "path": oof_path.relative_to(output).as_posix(),
            "sha256": pin["sha256"],
        }
    if resolved["reference_oof_100"] != contained_path(
        output, verified_oof["1.0"]["path"]
    ):
        raise ArchitectureContractError("100% reference OOF canonical path changed")
    if seal.get("reference_oof_by_fraction") != verified_oof:
        raise ArchitectureContractError("Stage-A seal does not deeply bind all prefix OOF files")
    binding = seal.get("binding")
    expected = {
        "stage_a_config_sha256": CONFIG_SHA256,
        "deployed_graph_manifest_sha256": sha256_file(resolved["deployed_graph_manifest"]),
        "training_recipe_sha256": sha256_file(resolved["training_recipe"]),
        "reference_oof_100_sha256": sha256_file(resolved["reference_oof_100"]),
    }
    if binding != expected:
        raise ArchitectureContractError("Stage-A seal binding fails deep equality")
    graph_manifest = _json_object(resolved["deployed_graph_manifest"])
    recipe = _json_object(resolved["training_recipe"])
    if graph_manifest != config["deployed_inference_graph"]:
        raise ArchitectureContractError("sealed graph manifest differs from preregistration")
    if recipe != config["training_recipe"]:
        raise ArchitectureContractError("sealed training recipe differs from preregistration")
    return {
        "seal_path": resolved["seal"].relative_to(root.resolve(strict=True)).as_posix(),
        "seal_sha256": sha256_file(resolved["seal"]),
        "binding": expected,
        "reference_oof_by_fraction": verified_oof,
        "verified_before_challenger_import_fit_score": True,
    }


__all__ = [
    "ArchitectureContractError",
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "MODE",
    "STAGE_A",
    "STAGE_B",
    "canonical_mapping_sha256",
    "consume_attempt_lock",
    "contained_path",
    "exclusive_bytes",
    "exclusive_json",
    "inspect_schema_and_keys",
    "load_canonical_config",
    "sha256_file",
    "stage_paths",
    "static_preflight",
    "validate_config",
    "verify_deployed_graph",
    "verify_execution_authorization",
    "verify_stage_a_seal",
    "workspace_path",
]
