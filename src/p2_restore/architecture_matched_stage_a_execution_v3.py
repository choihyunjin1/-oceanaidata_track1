"""Complete executable P2 architecture-matched Stage-A pipeline v3.

The numerical graph is deliberately delegated to the byte-pinned immutable v2
engine helpers.  V3 owns a new append-only output namespace and, on every
direct entry, performs a fresh canonical preflight plus an in-process runtime
check before it creates output or begins a fit.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from p2_restore.architecture_matched_stage_a_contract_v3 import (
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    MODE,
    canonical_mapping_sha256,
    contained_path,
    exclusive_bytes,
    exclusive_json,
    implementation_pins,
    load_canonical_config,
    sha256_file,
    stage_paths,
    static_preflight,
    verify_consumed_attempt_lock,
    verify_execution_authorization,
    verify_pre_execution_qa,
)

Progress = Callable[[dict[str, Any]], None]


def build_execution_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    recipe = config["training_recipe"]
    folds = [fold["name"] for fold in recipe["outer_folds"]]
    fractions = list(recipe["prefix_fractions"])
    seeds = list(recipe["complete_pipeline_seed_ids"])
    components = list(recipe["deep_training"]["components"])
    inner_splits = len(recipe["inner_oof"]["validation_blocks"])
    cells = len(folds) * len(fractions)
    return {
        "schema_version": "p2_architecture_matched_stage_a_execution.plan.v3",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "folds": folds,
        "prefix_fractions": fractions,
        "complete_pipeline_seeds": seeds,
        "deep_components": components,
        "inner_splits_per_cell": inner_splits,
        "outer_prefix_cells": cells,
        "deep_training_jobs": cells * len(seeds) * len(components) * (inner_splits + 1),
        "router_training_jobs": cells * len(seeds) * (inner_splits + 1),
        "challenger_jobs": 0,
        "full_fit_jobs": 0,
        "submission_predictions": 0,
        "uploads": 0,
        "implementation_lineage": "BYTE_PINNED_V2_NUMERICAL_GRAPH_WITH_V3_GUARDS",
        "additional_runtime_source_pins": ["MODEL_MODULE", "PACKAGE_INIT"],
    }


def _pin(path: Path, output: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _load_pinned_v2_engine(
    workspace: Path,
    config: Mapping[str, Any],
) -> Any:
    """Late-import v2 numerical code only after v3's fresh preflight passed."""

    module = importlib.import_module("p2_restore.architecture_matched_stage_a_execution_v2")
    expected_engine = config["immutable_v2_implementation_pins"]["ENGINE"]
    engine_path = Path(module.__file__).resolve(strict=True)
    canonical_engine = (workspace / expected_engine["path"]).resolve(strict=True)
    if engine_path != canonical_engine or _pin(engine_path, workspace) != expected_engine:
        raise PermissionError("late-imported v2 numerical engine fails its exact pin")
    for role, module_name in (
        ("MODEL_MODULE", "p2_restore.model"),
        ("PACKAGE_INIT", "p2_restore"),
    ):
        loaded = sys.modules.get(module_name)
        if loaded is None or not getattr(loaded, "__file__", None):
            raise PermissionError(f"late-imported dependency missing: {role}")
        expected = config["additional_runtime_source_pins"][role]
        loaded_path = Path(loaded.__file__).resolve(strict=True)
        canonical_path = (workspace / expected["path"]).resolve(strict=True)
        if loaded_path != canonical_path or _pin(loaded_path, workspace) != expected:
            raise PermissionError(f"late-imported dependency fails its exact pin: {role}")
    return module


def execute_stage_a(
    *,
    root: Path,
    data_dir: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
    attempt_lock: Path,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the full reference curve after independently redoing every guard.

    ``preflight`` is accepted only for runner API compatibility.  It is never
    trusted for authorization or pin checks; a new canonical preflight is
    always performed first.
    """

    del preflight
    workspace = root.resolve(strict=True)
    resolved_data_dir = data_dir.resolve(strict=True)

    # This must remain the first action capable of reaching execution state.
    # It rechecks v3/v2 implementations, architecture sources, deployed graph,
    # data schema/keys, the two additional source pins, and the exact runtime in
    # an isolated process.  No output path or training helper is touched first.
    fresh_preflight = static_preflight(
        workspace,
        resolved_data_dir,
        supplied_config=config,
    )
    if fresh_preflight.get("status") != "PASS_STATIC_IMPLEMENTATION_ONLY":
        raise PermissionError("fresh canonical Stage-A v3 preflight did not pass")
    canonical = load_canonical_config(workspace, supplied_config=config)
    expected_implementation_pins = fresh_preflight["implementation_pins"]
    if expected_implementation_pins != implementation_pins(workspace):
        raise PermissionError("v3 implementation bytes changed after fresh preflight")

    engine_v2 = _load_pinned_v2_engine(workspace, canonical)
    pd = engine_v2.pd
    # Recheck the runtime in the process that will actually fit.  This is
    # intentionally before QA/lock verification and before output creation.
    engine_v2.set_deterministic_seed(engine_v2.PIPELINE_SEEDS[0])
    runtime = engine_v2._verify_runtime(canonical)
    if runtime != fresh_preflight["runtime_probe"]["runtime"]:
        raise RuntimeError("isolated and execution-process runtime reports differ")
    data_pins = engine_v2._verify_data_pins(resolved_data_dir, canonical)

    paths = stage_paths(workspace, canonical)
    received_lock = attempt_lock.resolve(strict=True)
    canonical_lock = paths["attempt_lock"].resolve(strict=True)
    if received_lock != canonical_lock:
        raise PermissionError("engine did not receive the canonical consumed v3 attempt lock")
    _qa, qa_sha256 = verify_pre_execution_qa(workspace, canonical)
    _authorization, authorization_sha256 = verify_execution_authorization(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        require_unconsumed=False,
    )
    verify_consumed_attempt_lock(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )
    if paths["output"].exists():
        raise FileExistsError("append-only Stage-A v3 output already exists")

    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output)
    started = engine_v2._now_kst()
    artifacts = canonical["stage_a_reference_contract"]["artifacts"]
    exclusive_json(
        contained_path(output, artifacts["architecture_manifest"]),
        canonical["architecture_reference"],
    )
    exclusive_json(
        contained_path(output, artifacts["training_recipe"]),
        canonical["training_recipe"],
    )

    data = engine_v2.load_p2_data(resolved_data_dir)
    panel = engine_v2._joint_masked_panel(data.observations)
    router = engine_v2._build_router_context(data.observations)
    endpoints = engine_v2.public_endpoint_frame(data.observations)
    recipe = canonical["training_recipe"]
    plan = build_execution_plan(canonical)
    by_fraction: dict[float, dict[int, list[pd.DataFrame]]] = {
        fraction: {seed: [] for seed in engine_v2.PIPELINE_SEEDS}
        for fraction in engine_v2.PREFIX_FRACTIONS
    }
    cell_receipts: list[dict[str, Any]] = []
    for fraction in engine_v2.PREFIX_FRACTIONS:
        for fold in recipe["outer_folds"]:
            for pipeline_seed in engine_v2.PIPELINE_SEEDS:
                frame, receipt = engine_v2._run_cell_seed(
                    panel=panel,
                    router=router,
                    endpoints=endpoints,
                    recipe=recipe,
                    layer_factors=canonical["architecture_reference"][
                        "layer_extrapolation_factors"
                    ],
                    fold=fold,
                    fraction=fraction,
                    pipeline_seed=pipeline_seed,
                    progress=progress,
                )
                by_fraction[fraction][pipeline_seed].append(frame)
                cell_receipts.append(receipt)

    oof_roles = {
        0.4: "reference_oof_040",
        0.55: "reference_oof_055",
        0.7: "reference_oof_070",
        0.85: "reference_oof_085",
        1.0: "reference_oof_100",
    }
    curve_points: list[dict[str, Any]] = []
    output_columns = canonical["stage_a_reference_contract"]["reference_oof_columns"]
    for fraction in engine_v2.PREFIX_FRACTIONS:
        seed_frames = {
            seed: pd.concat(parts, ignore_index=True)
            for seed, parts in by_fraction[fraction].items()
        }
        merged, aggregate = engine_v2._merge_seed_predictions(
            seed_frames,
            recipe["metric"]["official_layer_counts"],
        )
        row_artifact = merged.drop(columns="truth")
        if set(row_artifact.columns) != set(output_columns):
            raise RuntimeError("target-free OOF schema is incomplete")
        path = contained_path(output, artifacts[oof_roles[fraction]])
        exclusive_bytes(path, engine_v2._csv_bytes(row_artifact, output_columns))
        curve_points.append({"fraction": fraction, **aggregate})

    curve_metrics = {
        "schema_version": "p2_architecture_matched_reference.curve_metrics.v3",
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
        "metric": recipe["metric"],
        "points": curve_points,
        "local_qualification_only": True,
        "official_promotion_allowed": False,
        "uploads": 0,
    }
    exclusive_json(
        contained_path(output, artifacts["reference_curve_metrics"]),
        curve_metrics,
    )
    training_receipt = {
        "schema_version": "p2_architecture_matched_reference.training_receipt.v3",
        "started_at_kst": started,
        "completed_at_kst": engine_v2._now_kst(),
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "plan": plan,
        "runtime": runtime,
        "cells": cell_receipts,
        "guard_summary": {
            "fresh_canonical_preflight_rerun_by_engine": True,
            "caller_preflight_trusted": False,
            "all_architecture_deployed_source_runtime_pins_before_output_or_fit": True,
            "joint_temp_psal_mask_applied_before_all_label_use": True,
            "outer_and_future_target_labels_used_for_fit": False,
            "frozen_stack_or_gate_reused": False,
            "all_five_prefixes_completed_before_seal": True,
            "challenger_import_fit_or_score_count": 0,
            "full_fit_count": 0,
            "submission_prediction_count": 0,
            "upload_count": 0,
        },
    }
    exclusive_json(
        contained_path(output, artifacts["training_receipt"]),
        training_receipt,
    )

    # Fail closed at seal time if any byte, graph, data file, runtime, QA,
    # authorization, or lock changed during the long-running fit.
    end_preflight = static_preflight(workspace, resolved_data_dir)
    if end_preflight["implementation_pins"] != expected_implementation_pins:
        raise PermissionError("v3 implementation bytes changed during Stage-A execution")
    if end_preflight["runtime_probe"]["runtime"] != runtime:
        raise PermissionError("runtime changed during Stage-A execution")
    if engine_v2._verify_runtime(canonical) != runtime:
        raise PermissionError("execution-process runtime changed during Stage-A execution")
    if engine_v2._verify_data_pins(resolved_data_dir, canonical) != data_pins:
        raise PermissionError("data source bytes changed during Stage-A execution")
    _qa_end, qa_sha256_end = verify_pre_execution_qa(workspace, canonical)
    if qa_sha256_end != qa_sha256:
        raise PermissionError("QA receipt changed during Stage-A execution")
    _authorization_end, authorization_sha256_end = verify_execution_authorization(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        require_unconsumed=False,
        require_output_absent=False,
    )
    if authorization_sha256_end != authorization_sha256:
        raise PermissionError("authorization changed during Stage-A execution")
    verify_consumed_attempt_lock(
        workspace,
        canonical,
        qa_sha256=qa_sha256,
        authorization_sha256=authorization_sha256,
    )

    artifact_pins: dict[str, dict[str, Any]] = {}
    for role, relative in artifacts.items():
        if role in {"manifest", "seal"}:
            continue
        path = contained_path(output, relative)
        artifact_pins[role] = _pin(path, output)
    manifest = {
        "schema_version": "p2_architecture_matched_reference.manifest.v3",
        "created_at_kst": engine_v2._now_kst(),
        "append_only": True,
        "problem": "P2",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "implementation_pins": expected_implementation_pins,
        "immutable_v2_implementation_pins": end_preflight[
            "immutable_v2_implementation_pins"
        ],
        "additional_runtime_source_pins": end_preflight[
            "additional_runtime_source_pins"
        ],
        "runtime": runtime,
        "data_source_pins": data_pins,
        "architecture_manifest_sha256": canonical_mapping_sha256(
            canonical["architecture_reference"]
        ),
        "training_recipe_sha256": canonical_mapping_sha256(recipe),
        "artifacts": artifact_pins,
        "challenger_import_fit_or_score_count": 0,
        "official_promotion_allowed": False,
        "uploads": 0,
    }
    manifest_path = contained_path(output, artifacts["manifest"])
    exclusive_json(manifest_path, manifest)
    reference_by_fraction = {
        str(fraction): artifact_pins[oof_roles[fraction]]
        for fraction in engine_v2.PREFIX_FRACTIONS
    }
    seal = {
        "schema_version": "p2_architecture_matched_reference.seal.v3",
        "complete": True,
        "all_five_prefixes_sealed": True,
        "challenger_import_fit_or_score_count_before_seal": 0,
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "upload_count": 0,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "manifest": _pin(manifest_path, output),
        "reference_oof_by_fraction": reference_by_fraction,
    }
    seal_path = contained_path(output, artifacts["seal"])
    exclusive_json(seal_path, seal)
    return {
        "schema_version": "p2_architecture_matched_stage_a_execution.result.v3",
        "status": "COMPLETE_SEALED_ARCHITECTURE_MATCHED_REFERENCE",
        "comparison_mode": MODE,
        "exact_official_incumbent_comparison": False,
        "official_promotion_allowed": False,
        "output": output.relative_to(workspace).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "seal_sha256": sha256_file(seal_path),
        "curve_metrics_sha256": sha256_file(
            contained_path(output, artifacts["reference_curve_metrics"])
        ),
        "challenger_fits": 0,
        "submission_predictions": 0,
        "uploads": 0,
    }


__all__ = ["build_execution_plan", "execute_stage_a"]
