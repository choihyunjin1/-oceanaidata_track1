"""Append-only evaluator recovery for the completed P2 checkpoint_v1 run.

The original run committed all 45 blind prediction arrays before target
decoding, then stopped because its evaluator incorrectly required the complete
truth surface to equal the (slightly smaller) sealed Stage-A/r3 key surface.
This module never trains.  It verifies every committed artifact, requires the
reference/r3 key surface to be an exact subset of truth, audits truth-only keys,
and appends the missing evaluation/seal files with O_EXCL.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from p2_restore import joint_hydrographic_multitask_layer4_checkpoint_v1 as original
from p2_restore import joint_hydrographic_multitask_layer4_execution_r3 as r3

RECOVERY_SCHEMA = "p2_joint_hydrographic_multitask_layer4.checkpoint_recovery_v1"
RECOVERY_FILES = (
    "recovery.lock",
    "recovery_receipt.json",
    "metrics.json",
    "checkpoint_oof.csv",
    "training_receipt.json",
    "manifest.json",
    "manifest.sha256",
    "seal.json",
)
RECOVERY_IMPLEMENTATION = {
    "engine": "src/p2_restore/joint_hydrographic_multitask_layer4_checkpoint_v1_recovery.py",
    "runner": "scripts/recover_p2_joint_hydrographic_multitask_layer4_checkpoint_v1.py",
    "tests": "tests/test_p2_joint_hydrographic_multitask_layer4_checkpoint_v1_recovery.py",
}


def _numerical() -> SimpleNamespace:
    import numpy as np
    import pandas as pd

    return SimpleNamespace(np=np, pd=pd, r3=r3)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _verify_exact_pin(root: Path, pin: Mapping[str, Any]) -> Path:
    path = original._workspace_path(root, str(pin["path"]))
    if original._pin(path, root) != dict(pin):
        raise PermissionError(f"committed artifact changed: {path.name}")
    return path


def validate_committed_state(
    *,
    root: Path,
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the immutable run without semantically reading target values."""

    workspace = root.resolve(strict=True)
    resolved_data = data_dir.resolve(strict=True)
    config_path, config = original.load_config(workspace)
    output = original._workspace_path(workspace, str(config["output"]["path"]))
    attempt_path = output / str(config["output"]["attempt_lock"])
    commitment_path = output / str(config["output"]["prediction_commitment"])
    if not attempt_path.is_file() or not commitment_path.is_file():
        raise FileNotFoundError("completed training commitment is absent")
    attempt = _read_json(attempt_path)
    config_pin = original._pin(config_path, workspace)
    if (
        attempt.get("schema_version")
        != "p2_joint_hydrographic_multitask_layer4.attempt_lock.v1"
        or attempt.get("execute") is not True
        or attempt.get("config") != config_pin
    ):
        raise PermissionError("attempt lock no longer binds the canonical config")

    for pin in config["pinned_implementation"].values():
        original._verify_pin(original._workspace_path(workspace, pin["path"]), pin)
    for pin in config["checkpoint_implementation"].values():
        original._verify_pin(original._workspace_path(workspace, pin["path"]), pin)
    for name, pin in config["source_boundary"]["allowed_files"].items():
        source = (resolved_data / name).resolve(strict=True)
        if source.parent != resolved_data:
            raise PermissionError("source path escaped P2_DATA_DIR")
        original._verify_pin(source, pin)

    commitment = _read_json(commitment_path)
    if (
        commitment.get("schema_version")
        != "p2_joint_hydrographic_multitask_layer4.prediction_commitment.v1"
        or commitment.get("outer_prediction_arrays") != 45
        or commitment.get("expected_outer_prediction_arrays") != 45
        or commitment.get("active_outer_target_scalars_decoded_before_commitment") != 0
        or len(commitment.get("fold_commitments", [])) != 3
        or len(commitment.get("selected_epochs", {})) != 15
    ):
        raise PermissionError("aggregate prediction commitment is incomplete")

    counts: dict[str, int] = {}
    artifact_pins: dict[str, dict[str, Any]] = {}
    for fold_pin in commitment["fold_commitments"]:
        fold_path = _verify_exact_pin(workspace, fold_pin)
        fold = _read_json(fold_path)
        if (
            fold.get("prediction_arrays") != 15
            or fold.get("active_outer_target_scalars_decoded_before_commitment") != 0
        ):
            raise PermissionError("fold commitment is incomplete")
        for relative, pin in fold.get("artifacts", {}).items():
            if relative in artifact_pins:
                raise PermissionError("artifact appears in more than one fold commitment")
            path = _verify_exact_pin(workspace, pin)
            if path.relative_to(output).as_posix() != relative:
                raise PermissionError("fold artifact path projection changed")
            artifact_pins[relative] = dict(pin)
            counts[path.name] = counts.get(path.name, 0) + 1
    required_counts = {
        "inner_best.pt": 45,
        "inner_history.json": 45,
        "checkpoint_selection.json": 15,
        "full_refit.pt": 45,
        "outer_prediction.npy": 45,
        "receipt.json": 45,
    }
    if counts != required_counts:
        raise PermissionError("committed cell artifact inventory changed")
    for name in RECOVERY_FILES:
        if (output / name).exists():
            raise FileExistsError(f"append-only recovery output already exists: {name}")
    recovery_implementation_pins = {
        role: original._pin(original._workspace_path(workspace, relative), workspace)
        for role, relative in RECOVERY_IMPLEMENTATION.items()
    }
    return config, {
        "workspace": workspace,
        "data_dir": resolved_data,
        "output": output,
        "attempt": attempt,
        "config_pin": config_pin,
        "commitment": commitment,
        "commitment_pin": original._pin(commitment_path, workspace),
        "fold_commitments": commitment["fold_commitments"],
        "artifact_pins": artifact_pins,
        "artifact_counts": counts,
        "recovery_implementation_pins": recovery_implementation_pins,
    }


def key_alignment_audit(
    reference: Any,
    truth: Any,
    comparator: Any,
    *,
    pd_module: Any,
) -> dict[str, Any]:
    """Require ref==r3 and ref subset truth; report only aggregate differences."""

    pd = pd_module
    keys = list(original.KEYS)
    for label, frame in (
        ("reference", reference),
        ("truth", truth),
        ("comparator", comparator),
    ):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} key surface is duplicated")
    ref_index = pd.MultiIndex.from_frame(reference.loc[:, keys])
    truth_index = pd.MultiIndex.from_frame(truth.loc[:, keys])
    comparator_index = pd.MultiIndex.from_frame(comparator.loc[:, keys])
    if not ref_index.equals(comparator_index):
        ref_only = ref_index.difference(comparator_index)
        comparator_only = comparator_index.difference(ref_index)
        raise ValueError(
            "reference/r3 key surfaces differ: "
            f"reference_only={len(ref_only)}, comparator_only={len(comparator_only)}"
        )
    missing_truth = ref_index.difference(truth_index)
    if len(missing_truth):
        raise ValueError(f"committed prediction keys missing truth: {len(missing_truth)}")
    truth_only = truth_index.difference(ref_index)
    truth_key_frame = truth.loc[:, keys].copy()
    truth_key_frame["_key"] = list(truth_index)
    extra = truth_key_frame.loc[truth_key_frame["_key"].isin(set(truth_only))]
    extra_counts = {
        f"{fold}|layer_{int(layer)}": int(count)
        for (fold, layer), count in extra.groupby(["fold", "layer"], sort=True).size().items()
    }
    return {
        "reference_rows": int(len(reference)),
        "r3_comparator_rows": int(len(comparator)),
        "truth_rows": int(len(truth)),
        "common_metric_rows": int(len(reference)),
        "reference_minus_truth_rows": 0,
        "reference_minus_r3_rows": 0,
        "r3_minus_reference_rows": 0,
        "truth_only_rows_excluded": int(len(truth_only)),
        "truth_only_by_fold_layer": extra_counts,
        "metric_domain": "EXACT_SEALED_STAGE_A_AND_R3_COMMON_KEY_SURFACE",
    }


def _load_predictions(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    numerical: SimpleNamespace,
) -> tuple[dict[tuple[str, float, int], Any], dict[str, Any]]:
    np = numerical.np
    output = state["output"]
    predictions: dict[tuple[str, float, int], Any] = {}
    inner_steps = 0
    full_steps = 0
    inner_fits = 0
    full_fits = 0
    for fold in config["folds"]:
        fold_name = str(fold["name"])
        for raw_fraction in config["prefix_fractions"]:
            fraction = float(raw_fraction)
            token = original.FRACTION_TOKENS[fraction]
            selection_path = output / "cells" / fold_name / f"fraction_{token}" / "checkpoint_selection.json"
            selection = _read_json(selection_path)
            selected_epoch = int(selection["selected_common_epoch"])
            if selected_epoch != int(state["commitment"]["selected_epochs"][f"{fold_name}|{fraction}"]):
                raise PermissionError("selected epoch differs from aggregate commitment")
            for raw_seed in config["seed_ids"]:
                seed = int(raw_seed)
                cell = output / "cells" / fold_name / f"fraction_{token}" / f"seed_{seed}"
                receipt = _read_json(cell / "receipt.json")
                history = _read_json(cell / "inner_history.json")
                if (
                    int(receipt["selected_common_epoch"]) != selected_epoch
                    or receipt.get("outer_truth_used_for_fit_or_checkpoint_selection") is not False
                    or history.get("outer_truth_used") is not False
                ):
                    raise PermissionError("cell blindness or selected checkpoint changed")
                prediction = np.load(cell / "outer_prediction.npy", allow_pickle=False)
                digest = hashlib.sha256(
                    np.asarray(prediction, dtype="<f8").tobytes(order="C")
                ).hexdigest()
                if digest != receipt["prediction_values_sha256"]:
                    raise PermissionError("prediction value digest changed")
                predictions[(fold_name, fraction, seed)] = np.asarray(
                    prediction, dtype=np.float64
                )
                inner_steps += int(history["optimizer_steps"])
                full_steps += int(receipt["optimizer_steps"])
                inner_fits += 1
                full_fits += 1
    if len(predictions) != 45 or inner_fits != 45 or full_fits != 45:
        raise RuntimeError("committed model/prediction inventory is incomplete")
    return predictions, {
        "inner_fits_reused": inner_fits,
        "full_prefix_refits_reused": full_fits,
        "total_model_fits_reused": inner_fits + full_fits,
        "new_model_fits": 0,
        "inner_optimizer_steps_reused": inner_steps,
        "full_optimizer_steps_reused": full_steps,
    }


def _prepare(
    *,
    root: Path,
    data_dir: Path,
) -> dict[str, Any]:
    config, state = validate_committed_state(root=root, data_dir=data_dir)
    numerical = _numerical()
    workspace = state["workspace"]
    predictions, reuse = _load_predictions(config, state, numerical)
    references = {
        float(fraction): original._load_reference(
            workspace, config, float(fraction), numerical
        )
        for fraction in config["prefix_fractions"]
    }

    # Target/comparator semantics occur only after all commitments were reverified.
    truth, truth_audit = original._load_truth_after_commitment(
        state["data_dir"] / "observations.csv",
        config=config,
        numerical=numerical,
    )
    comparator_pin = config["r3_final_comparator"]
    comparator_path = original._workspace_path(workspace, comparator_pin["path"])
    original._verify_pin(comparator_path, comparator_pin)
    r3_oof = numerical.pd.read_csv(
        comparator_path,
        dtype={"fold": "string", "station": "string", "time": "string"},
    )
    r3_oof["time"] = numerical.pd.to_datetime(
        r3_oof["time"], utc=True, format="mixed"
    )
    if r3_oof.duplicated(["fraction", *original.KEYS]).any():
        raise ValueError("r3 comparator keys are duplicated")

    points: list[dict[str, Any]] = []
    oof_parts: list[Any] = []
    alignments: dict[str, Any] = {}
    counts = config["metric_contract"]["official_layer_counts"]
    for raw_fraction in config["prefix_fractions"]:
        fraction = float(raw_fraction)
        reference = references[fraction]
        old = r3_oof.loc[
            numerical.np.isclose(r3_oof["fraction"].to_numpy(float), fraction)
        ].copy()
        old = old.loc[:, [*original.KEYS, "challenger_mean", "truth"]]
        alignment = key_alignment_audit(
            reference,
            truth,
            old,
            pd_module=numerical.pd,
        )
        alignments[str(fraction)] = alignment

        fold_parts = []
        for fold in config["folds"]:
            fold_name = str(fold["name"])
            current = reference.loc[reference["fold"].eq(fold_name)].reset_index(drop=True).copy()
            for seed in config["seed_ids"]:
                values = predictions[(fold_name, fraction, int(seed))]
                if len(values) != len(current):
                    raise ValueError("committed prediction length differs from reference")
                current[f"checkpoint_seed_{seed}"] = values
            fold_parts.append(current)
        frame = numerical.pd.concat(fold_parts, ignore_index=True)
        frame["reference_mean"] = frame["prediction_mean"].to_numpy(float)
        seed_columns = [f"checkpoint_seed_{seed}" for seed in config["seed_ids"]]
        frame["checkpoint_mean"] = frame[seed_columns].to_numpy(float).mean(axis=1)
        frame = frame.merge(truth, on=list(original.KEYS), how="inner", validate="one_to_one")
        old = old.rename(columns={"challenger_mean": "r3_final_mean", "truth": "r3_truth"})
        frame = frame.merge(old, on=list(original.KEYS), how="inner", validate="one_to_one")
        if len(frame) != int(alignment["common_metric_rows"]):
            raise ValueError("common metric key intersection is incomplete")
        if not numerical.np.allclose(
            frame["truth"].to_numpy(float),
            frame["r3_truth"].to_numpy(float),
            rtol=0,
            atol=1e-12,
        ):
            raise PermissionError("fresh truth differs from the sealed r3 truth")
        reference_metric = original._curve_metric(
            frame, "reference_mean", counts, numerical.np
        )
        r3_metric = original._curve_metric(frame, "r3_final_mean", counts, numerical.np)
        checkpoint_metric = original._curve_metric(
            frame, "checkpoint_mean", counts, numerical.np
        )
        points.append(
            {
                "fraction": fraction,
                "architecture_matched_stage_a": reference_metric,
                "r3_final_epoch": r3_metric,
                "checkpoint_v1": checkpoint_metric,
                "delta_checkpoint_minus_r3_final_c": checkpoint_metric["aggregate"]
                - r3_metric["aggregate"],
                "delta_checkpoint_minus_stage_a_c": checkpoint_metric["aggregate"]
                - reference_metric["aggregate"],
                "alignment": alignment,
            }
        )
        frame.insert(0, "fraction", fraction)
        oof_parts.append(
            frame.loc[
                :,
                [
                    "fraction",
                    *original.KEYS,
                    *seed_columns,
                    "checkpoint_mean",
                    "r3_final_mean",
                    "reference_mean",
                    "truth",
                ],
            ]
        )
    metrics = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.metrics.recovery_v1",
        "comparison_mode": "CHECKPOINT_POLICY_RETROSPECTIVE_DIAGNOSTIC",
        "key_domain_recovery": "TRUTH_SUPERSET_AUDITED_REFERENCE_R3_EXACT_DOMAIN",
        "exact_official_incumbent_comparison": False,
        "outer_windows_previously_exposed": True,
        "official_promotion_allowed": False,
        "points": points,
        "full_fraction_checkpoint_minus_r3_final_c": points[-1][
            "delta_checkpoint_minus_r3_final_c"
        ],
        "full_fraction_checkpoint_minus_stage_a_c": points[-1][
            "delta_checkpoint_minus_stage_a_c"
        ],
    }
    return {
        "config": config,
        "state": state,
        "reuse": reuse,
        "truth_audit": truth_audit,
        "alignments": alignments,
        "metrics": metrics,
        "oof": numerical.pd.concat(oof_parts, ignore_index=True),
    }


def preflight(*, root: Path, data_dir: Path) -> dict[str, Any]:
    prepared = _prepare(root=root, data_dir=data_dir)
    metrics = prepared["metrics"]
    result = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.recovery_preflight.v1",
        "status": "READY_TO_APPEND_EVALUATION_WITHOUT_REFIT",
        "checked_at_kst": original._now_kst(),
        "commitment": prepared["state"]["commitment_pin"],
        "artifact_counts": prepared["state"]["artifact_counts"],
        "recovery_implementation_pins": prepared["state"]["recovery_implementation_pins"],
        "reuse": prepared["reuse"],
        "key_alignment": prepared["alignments"],
        "full_fraction_checkpoint_minus_r3_final_c": metrics[
            "full_fraction_checkpoint_minus_r3_final_c"
        ],
        "full_fraction_checkpoint_minus_stage_a_c": metrics[
            "full_fraction_checkpoint_minus_stage_a_c"
        ],
        "writes": 0,
        "new_model_fits": 0,
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }
    result["summary_sha256"] = hashlib.sha256(
        original.canonical_json_bytes(result)
    ).hexdigest()
    return result


def execute_recovery(*, root: Path, data_dir: Path) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    prepared = _prepare(root=workspace, data_dir=data_dir)
    config = prepared["config"]
    state = prepared["state"]
    output = state["output"]
    recovery_lock = output / "recovery.lock"
    lock_payload = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.recovery_lock.v1",
        "created_at_kst": original._now_kst(),
        "original_config": state["config_pin"],
        "prediction_commitment": state["commitment_pin"],
        "recovery_implementation_pins": state["recovery_implementation_pins"],
        "new_model_fits_authorized": 0,
        "append_only_evaluation_only": True,
    }
    original.exclusive_json(recovery_lock, lock_payload)
    _verify_exact_pin(workspace, state["commitment_pin"])
    for pin in state["fold_commitments"]:
        _verify_exact_pin(workspace, pin)

    metrics_path = output / str(config["output"]["metrics"])
    oof_path = output / str(config["output"]["oof"])
    receipt_path = output / str(config["output"]["training_receipt"])
    recovery_receipt_path = output / "recovery_receipt.json"
    original.exclusive_json(metrics_path, prepared["metrics"])
    buffer = io.StringIO(newline="")
    prepared["oof"].to_csv(buffer, index=False, lineterminator="\n")
    original.exclusive_bytes(oof_path, buffer.getvalue().encode("utf-8"))
    training_receipt = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.training_receipt.recovery_v1",
        "original_attempt_created_at_kst": state["attempt"]["created_at_kst"],
        "evaluation_recovered_at_kst": original._now_kst(),
        "config": state["config_pin"],
        "prediction_commitment": state["commitment_pin"],
        "recovery_implementation_pins": state["recovery_implementation_pins"],
        "fold_commitments": state["fold_commitments"],
        "reuse": prepared["reuse"],
        "selected_common_epochs": state["commitment"]["selected_epochs"],
        "truth_access_audit": prepared["truth_audit"],
        "key_alignment": prepared["alignments"],
        "candidate_predictions": 0,
        "test_predictions": 0,
        "uploads": 0,
    }
    original.exclusive_json(receipt_path, training_receipt)
    recovery_receipt = {
        "schema_version": RECOVERY_SCHEMA,
        "recovered_at_kst": original._now_kst(),
        "root_cause": {
            "exception": "ValueError: checkpoint/reference/r3/truth key intersection is incomplete",
            "original_engine_line": 1214,
            "bug": "truth was a strict superset but evaluator required equal row counts",
            "committed_prediction_keys_missing_truth": 0,
            "truth_only_rows": 31,
        },
        "recovery": {
            "model_refits": 0,
            "prediction_regeneration": 0,
            "committed_predictions_reused": 45,
            "metric_domain": "EXACT_SEALED_STAGE_A_AND_R3_COMMON_KEY_SURFACE",
            "truth_only_rows_excluded_with_audit": 31,
            "semantic_truth_access_after_original_commitment": True,
        },
        "prediction_commitment": state["commitment_pin"],
        "recovery_implementation_pins": state["recovery_implementation_pins"],
        "key_alignment": prepared["alignments"],
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }
    original.exclusive_json(recovery_receipt_path, recovery_receipt)

    manifest_path = output / str(config["output"]["manifest"])
    sidecar_path = output / str(config["output"]["manifest_sidecar"])
    seal_path = output / str(config["output"]["seal"])
    artifacts = {
        path.relative_to(output).as_posix(): original._pin(path, workspace)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path not in {manifest_path, sidecar_path, seal_path}
    }
    manifest = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.manifest.recovery_v1",
        "created_at_kst": original._now_kst(),
        "append_only": True,
        "research_only": True,
        "original_config": state["config_pin"],
        "recovery_implementation_pins": state["recovery_implementation_pins"],
        "prediction_commitment": state["commitment_pin"],
        "artifacts": artifacts,
        "recovery_receipt": original._pin(recovery_receipt_path, workspace),
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }
    original.exclusive_json(manifest_path, manifest)
    original.exclusive_bytes(
        sidecar_path,
        f"{original.sha256_file(manifest_path)}  manifest.json\n".encode("ascii"),
    )
    seal = {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.seal.recovery_v1",
        "complete": True,
        "status": "RETROSPECTIVE_DIAGNOSTIC_COMPLETE_AFTER_APPEND_ONLY_KEY_RECOVERY",
        "manifest": original._pin(manifest_path, workspace),
        "manifest_sidecar": original._pin(sidecar_path, workspace),
        "prediction_commitment": state["commitment_pin"],
        "new_model_fits": 0,
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }
    original.exclusive_json(seal_path, seal)
    return {
        "schema_version": "p2_joint_hydrographic_multitask_layer4.recovery_result.v1",
        "status": seal["status"],
        "output": output.relative_to(workspace).as_posix(),
        "metrics": prepared["metrics"],
        "new_model_fits": 0,
        "manifest_sha256": original.sha256_file(manifest_path),
        "seal_sha256": original.sha256_file(seal_path),
        "official_promotion_allowed": False,
        "candidate_generated": False,
        "test_prediction_generated": False,
        "uploads": 0,
    }


__all__ = [
    "RECOVERY_FILES",
    "execute_recovery",
    "key_alignment_audit",
    "preflight",
    "validate_committed_state",
]
