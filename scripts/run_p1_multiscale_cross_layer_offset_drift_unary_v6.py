"""Static-only owner check for the P1 multiscale slow-anomaly unary design.

There is intentionally no execution path in this runner.  It verifies the
single preregistered structure, sealed predecessor evidence, the current v9
ledger anchor, source-path safety, pure helper invariants, and absence of every
future run/control/output path.  It never opens train.csv, test.csv, or the
submission template and never fits, predicts, scores, locks, appends, or writes.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT / "src"))

from ocean_goal.meaningful_score_ledger_v9 import validate_ledger
from p1_qc.multiscale_cross_layer_offset_drift import (
    DETERMINISTIC_SEED,
    EXPERIMENT_ID,
    GATE_THRESHOLDS,
    GEOMETRY_FEATURES,
    HYPOTHESIS_ID,
    MAX_SLOW_RUN_ROWS,
    MIN_SLOW_RUN_ROWS,
    MULTISCALE_ROWS,
    SEASONAL_IRLS_ITERATIONS,
    UNARY_C,
    UNARY_MAX_ITER,
    UNARY_THRESHOLD,
    UNARY_TOL,
    static_contract_audit,
)

WORKSPACE_ENV = "P1_WORKSPACE_ROOT"
DATA_ENV = "P1_DATA_DIR"
CANONICAL_CONFIG = "configs/experiments/p1_multiscale_cross_layer_offset_drift_unary_v6.json"
CANONICAL_HELPER = "src/p1_qc/multiscale_cross_layer_offset_drift.py"
CANONICAL_RUNNER = "scripts/run_p1_multiscale_cross_layer_offset_drift_unary_v6.py"
CANONICAL_TESTS = "tests/test_run_p1_multiscale_cross_layer_offset_drift_unary_v6.py"

EXPECTED_CONFIG_SHA256 = "132dd95a91687d47212ff4af653a9b2d2b3264705dce0821aaa2be73c4933838"
EXPECTED_HELPER_SHA256 = "434bce024437d5dea77a4827bfdde95f0726659f901993fd4c033df82ab605d8"
EXPECTED_TEST_SHA256 = "323e06c2c588db1bea20f2d1d58849d12384e2d58b0f7d4192587508f63998b3"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {token}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _reject_reparse_chain(path: Path, *, role: str) -> None:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    absolute = Path(os.path.abspath(path))
    for item in (absolute, *absolute.parents):
        if not item.exists():
            continue
        info = os.lstat(item)
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            raise PermissionError(f"reparse path forbidden for {role}")


def _regular_file(path: Path, *, role: str, allow_open: bool) -> dict[str, Any]:
    _reject_reparse_chain(path, role=role)
    resolved = path.resolve(strict=True)
    info = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PermissionError(f"regular single-link file required for {role}")
    value: dict[str, Any] = {
        "bytes": int(info.st_size),
        "nlink": int(info.st_nlink),
        "non_reparse": True,
        "opened": allow_open,
    }
    if allow_open:
        value["sha256"] = _sha(resolved)
    return value


def _environment_paths(
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    source = os.environ if environ is None else environ
    workspace_raw = source.get(WORKSPACE_ENV)
    data_raw = source.get(DATA_ENV)
    if not workspace_raw or not data_raw:
        raise PermissionError(f"{WORKSPACE_ENV} and {DATA_ENV} are required")
    workspace_path = Path(workspace_raw)
    data_path = Path(data_raw)
    if not workspace_path.is_absolute() or not data_path.is_absolute():
        raise PermissionError("workspace and data paths must be absolute")
    _reject_reparse_chain(workspace_path, role="workspace")
    _reject_reparse_chain(data_path, role="data")
    root = workspace_path.resolve(strict=True)
    data_dir = data_path.resolve(strict=True)
    if not os.path.samefile(root / CANONICAL_RUNNER, Path(__file__).resolve(strict=True)):
        raise PermissionError("workspace does not own this canonical runner")
    return root, data_dir


def _verify_pin(root: Path, pin: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    if set(pin) != {"path", "bytes", "sha256"}:
        raise PermissionError(f"sealed pin shape differs: {role}")
    relative = Path(str(pin["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError(f"sealed pin path is non-portable: {role}")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError(f"sealed pin escaped workspace: {role}")
    observed = _regular_file(path, role=role, allow_open=True)
    if observed["bytes"] != pin["bytes"] or observed["sha256"] != pin["sha256"]:
        raise PermissionError(f"sealed pin differs: {role}")
    return {"path": relative.as_posix(), **observed}


def _verify_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / CANONICAL_CONFIG
    config_identity = _regular_file(config_path, role="config", allow_open=True)
    if config_identity["sha256"] != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical config SHA differs")
    config = _strict_json(config_path)
    required_top = {
        "schema_version",
        "experiment_id",
        "problem",
        "stage",
        "created_at_kst",
        "metric",
        "direction",
        "comparison_mode",
        "single_hypothesis_contract",
        "diagnostic_identifiability",
        "sealed_history",
        "data_contract",
        "incumbent",
        "prefix_protocol",
        "label_free_baseline",
        "trajectory_geometry",
        "unary_head",
        "protected_union",
        "selective_target_and_blind_commitment_protocol",
        "train_only_gate",
        "final_curve_gate",
        "resource_ceiling",
        "v9_binding",
        "implementation_contract",
        "static_prohibitions",
        "future_output_paths_must_be_absent",
        "on_static_pass",
    }
    if set(config) != required_top:
        raise PermissionError("canonical config top-level keys differ")
    hypothesis = config["single_hypothesis_contract"]
    if not (
        config["schema_version"]
        == "p1_multiscale_cross_layer_offset_drift_unary.v6.static_preregistration.v1"
        and config["experiment_id"] == EXPERIMENT_ID
        and config["problem"] == "P1"
        and config["stage"] == "STATIC_OWNER_PREREGISTRATION_ONLY"
        and hypothesis["hypothesis_count"] == 1
        and hypothesis["hypothesis_id"] == HYPOTHESIS_ID
        and hypothesis["alternatives_registered"] == 0
        and all(
            hypothesis[key] is False
            for key in (
                "threshold_sweep",
                "alpha_sweep",
                "seed_sweep",
                "architecture_search",
                "hyperparameter_search",
                "posthoc_subgroup_selection",
            )
        )
        and config["diagnostic_identifiability"]["identifiable"] is True
    ):
        raise PermissionError("single-hypothesis or identifiability contract differs")

    trajectory = config["trajectory_geometry"]
    unary = config["unary_head"]
    protection = config["protected_union"]
    train_gate = config["train_only_gate"]
    gate = train_gate["apply_only_if_all"]
    required_groups = [
        "G-ORS|1",
        *[f"I-ORS|{layer}" for layer in range(1, 8)],
        *[f"S-ORS|{layer}" for layer in range(1, 9)],
    ]
    if not (
        tuple(trajectory["scales_rows"]) == MULTISCALE_ROWS
        and trajectory["feature_count"] == len(GEOMETRY_FEATURES) == 29
        and trajectory["maximum_dependency_rows"] == max(MULTISCALE_ROWS)
        and trajectory["purge_covers_dependency"] is True
        and unary["threshold"] == UNARY_THRESHOLD
        and unary["C"] == UNARY_C
        and unary["solver"] == "lbfgs"
        and unary["max_iter"] == UNARY_MAX_ITER
        and unary["tol"] == UNARY_TOL
        and unary["class_weight"] == "balanced"
        and unary["fit_intercept"] is True
        and unary["random_state"] == DETERMINISTIC_SEED
        and unary["scaler_with_centering"] is True
        and unary["scaler_with_scaling"] is True
        and unary["scaler_unit_variance"] is False
        and unary["fixed_before_any_score"] is True
        and protection["minimum_run_rows"] == MIN_SLOW_RUN_ROWS
        and protection["maximum_run_rows"] == MAX_SLOW_RUN_ROWS
        and protection["failed_gate_probability_bytes"] == "exact incumbent"
        and protection["failed_gate_prediction_bytes"] == "exact incumbent"
        and gate
        == {
            **GATE_THRESHOLDS,
            "both_slow_types_observed": True,
            "spike_observed": True,
            "all_required_station_layers_observed": True,
            "blind_predictions_sealed_before_gate_labels": True,
        }
        and train_gate["required_station_layers"] == required_groups
        and set(train_gate["metric_definitions"])
        == {
            "micro_f1_delta",
            "offset_recall_delta",
            "drift_recall_delta",
            "spike_f1_delta",
            "worst_station_layer_f1_delta",
            "normal_fp_relative_increase",
            "nondegrading_inner_blocks",
        }
    ):
        raise PermissionError("fixed geometry, unary, protection, or gate contract differs")

    prefix = config["prefix_protocol"]
    blind = config["selective_target_and_blind_commitment_protocol"]
    if not (
        prefix["outer_folds"] == ["2025_q2", "2025_q3", "2025_q4"]
        and prefix["fractions"] == [0.4, 0.55, 0.7, 0.85, 1.0]
        and prefix["execution_order"] == "fold_major_then_prefix_fraction"
        and prefix["deterministic_seeds"] == [20260823]
        and prefix["inner_blocks"] == 3
        and prefix["active_outer_target_reads_before_cell_commitment"] == 0
        and prefix["aggregate_target_reads_before_predictions_complete"] == 0
        and blind["inner_blind_commitments"] == 45
        and blind["outer_cell_commitments"] == 15
        and blind["fold_commitments"] == 3
        and blind["mixed_or_uncommitted_scoring_forbidden"] is True
    ):
        raise PermissionError("fold-major cross-fit or blind-commitment contract differs")

    resource = config["resource_ceiling"]
    expected_resource = {
        "curve_cells": 15,
        "deterministic_seed_count": 1,
        "inner_blocks_per_cell": 3,
        "maximum_label_free_baseline_fit_calls": 60,
        "maximum_supervised_unary_fit_calls": 60,
        "maximum_top_level_fit_calls": 120,
        "maximum_station_layer_seasonal_subfits": 960,
        "maximum_adjacent_graph_edge_estimations": 780,
        "maximum_seasonal_irls_steps": 7680,
        "maximum_unary_lbfgs_iterations": 3840,
        "maximum_total_iterative_steps": 11520,
        "maximum_wall_clock_seconds": 21600,
        "maximum_peak_rss_bytes": 8589934592,
        "maximum_vram_bytes": 0,
        "gpu_allowed": False,
        "maximum_threads": 8,
        "maximum_artifact_disk_bytes": 1073741824,
        "static_check_maximum_wall_clock_seconds": 60,
        "static_check_fit_calls": 0,
        "static_check_predictions": 0,
        "static_check_scores": 0,
        "static_check_test_value_reads": 0,
    }
    if resource != expected_resource:
        raise PermissionError("resource arithmetic or ceiling differs")
    if not (
        resource["maximum_label_free_baseline_fit_calls"]
        == resource["curve_cells"] * (resource["inner_blocks_per_cell"] + 1)
        and resource["maximum_supervised_unary_fit_calls"]
        == resource["curve_cells"] * (resource["inner_blocks_per_cell"] + 1)
        and resource["maximum_seasonal_irls_steps"]
        == resource["maximum_station_layer_seasonal_subfits"] * SEASONAL_IRLS_ITERATIONS
        and resource["maximum_unary_lbfgs_iterations"]
        == resource["maximum_supervised_unary_fit_calls"] * unary["max_iter"]
        and resource["maximum_total_iterative_steps"]
        == resource["maximum_seasonal_irls_steps"] + resource["maximum_unary_lbfgs_iterations"]
    ):
        raise PermissionError("resource ceiling arithmetic is inconsistent")

    prohibitions = config["static_prohibitions"]
    if not prohibitions or any(value is not False for value in prohibitions.values()):
        raise PermissionError("static prohibition changed")
    if not (
        config["implementation_contract"]["helper"]
        == {"path": CANONICAL_HELPER, "sha256": EXPECTED_HELPER_SHA256}
        and config["implementation_contract"]["runner"] == CANONICAL_RUNNER
        and config["implementation_contract"]["tests"] == CANONICAL_TESTS
        and config["implementation_contract"]["runner_mode"] == "CHECK_ONLY_NO_OUTPUT"
        and config["implementation_contract"]["actual_execution_entrypoint_present"] is False
        and config["on_static_pass"]["actual_run_allowed"] is False
    ):
        raise PermissionError("static implementation or authorization contract differs")
    return config, config_identity


def _verify_implementation(root: Path) -> dict[str, Any]:
    identities = {
        "config": _regular_file(root / CANONICAL_CONFIG, role="config", allow_open=True),
        "helper": _regular_file(root / CANONICAL_HELPER, role="helper", allow_open=True),
        "runner": _regular_file(root / CANONICAL_RUNNER, role="runner", allow_open=True),
        "tests": _regular_file(root / CANONICAL_TESTS, role="tests", allow_open=True),
    }
    if identities["config"]["sha256"] != EXPECTED_CONFIG_SHA256:
        raise PermissionError("config implementation pin differs")
    if identities["helper"]["sha256"] != EXPECTED_HELPER_SHA256:
        raise PermissionError("helper implementation pin differs")
    if identities["tests"]["sha256"] != EXPECTED_TEST_SHA256:
        raise PermissionError("test implementation pin differs")
    return identities


def _verify_source_boundary(data_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    pins = config["data_contract"]["source_pins"]
    readme = _regular_file(data_dir / "README.md", role="source README", allow_open=True)
    if (
        readme["bytes"] != pins["README.md"]["bytes"]
        or readme["sha256"] != pins["README.md"]["sha256"]
    ):
        raise PermissionError("source README pin differs")
    unopened: dict[str, Any] = {}
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        identity = _regular_file(data_dir / name, role=name, allow_open=False)
        if identity["bytes"] != pins[name]["bytes"] or pins[name]["static_check_open_allowed"]:
            raise PermissionError(f"static unopened source contract differs: {name}")
        unopened[name] = identity
    return {
        "README.md": readme,
        **unopened,
        "train_value_reads": 0,
        "test_value_reads": 0,
        "sample_submission_value_reads": 0,
    }


def _verify_sealed_history(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    history = config["sealed_history"]
    identities: dict[str, dict[str, Any]] = {}
    for family, payload in history.items():
        if family == "failure_recon":
            identities[family] = _verify_pin(root, payload, role=family)
            continue
        identities[family] = {
            role: _verify_pin(root, pin, role=f"{family}:{role}")
            for role, pin in payload["pins"].items()
        }

    typed = _strict_json(root / history["typed_duration_semimarkov_v2"]["pins"]["result"]["path"])
    typed_posthoc = _strict_json(
        root / history["typed_duration_semimarkov_v2"]["pins"]["posthoc_identifiability"]["path"]
    )
    if not (
        typed["decision"] == "NO_GO_EXACT_CONFIGURATION"
        and typed["outer_score_count"] == 0
        and typed["test_prediction_count"] == typed["submission_count"] == 0
        and typed["aggregate"]["micro_f1_delta"] == 0.0024336282407236842
        and typed["aggregate"]["recall_delta_by_type"]["drift"] == -0.007227671657201867
        and typed["aggregate"]["recall_delta_by_type"]["spike"] == -0.47058823529411764
        and typed["aggregate"]["worst_station_layer_f1_delta"] == -0.6666666666666666
        and typed_posthoc["decision"] == "BLOCKED_NOT_IDENTIFIABLE_FROM_SAVED_AGGREGATES"
        and typed_posthoc["operation_counts"]
        == {
            "inner_rescore": 0,
            "label_reopen": 0,
            "model_fit": 0,
            "outer_score": 0,
            "prediction_generation": 0,
            "submission": 0,
            "test_read": 0,
            "upload": 0,
        }
    ):
        raise PermissionError("typed semi-Markov closure semantics differ")

    tcn_metrics = _strict_json(
        root / history["station_layer_temporal_convolution_event_v2"]["pins"]["metrics"]["path"]
    )
    tcn_result = _strict_json(
        root / history["station_layer_temporal_convolution_event_v2"]["pins"]["result"]["path"]
    )
    if not (
        tcn_result["status"] == "RESEARCH_ONLY_NO_PASS"
        and tcn_result["passed"] is False
        and tcn_result["exactly_one_next_structural_diagnosis"]
        == "self_supervised_masked_sequence_pretraining_then_phase_head_finetune"
        and tcn_metrics["points"][-1]["delta_candidate_minus_incumbent"] == -0.3100741149973728
    ):
        raise PermissionError("TCN closure semantics differ")

    ssl_metrics = _strict_json(
        root / history["masked_pretrain_binary_event_v4r4"]["pins"]["metrics"]["path"]
    )
    ssl_result = _strict_json(
        root / history["masked_pretrain_binary_event_v4r4"]["pins"]["result"]["path"]
    )
    if not (
        ssl_result["status"] == "RESEARCH_ONLY_NO_PASS"
        and ssl_result["passed"] is False
        and ssl_result["exactly_one_next_structural_diagnosis"]
        == "incumbent_rule_distillation_with_out_of_fold_neural_residual"
        and ssl_metrics["points"][-1]["delta_candidate_minus_incumbent"] == -0.5080652138235318
    ):
        raise PermissionError("SSL closure semantics differ")

    gen5 = history["incumbent_rule_distillation_neural_residual_v5r6"]["pins"]
    gen5_metrics = _strict_json(root / gen5["metrics"]["path"])
    gen5_result = _strict_json(root / gen5["result"]["path"])
    completion = _strict_json(root / gen5["predictions_complete"]["path"])
    if not (
        gen5_result["status"] == "RESEARCH_ONLY_NO_PASS"
        and gen5_result["passed"] is False
        and all(
            point["delta_candidate_minus_incumbent"] == 0.0 and point["delta_ci90"] == [0.0, 0.0]
            for point in gen5_metrics["points"]
        )
        and completion["fit_cells"] == 225
        and completion["total_residual_optimizer_steps"] == 10800
        and len(completion["gate_model_receipts"]) == 45
        and sum(receipt["gate"]["passed"] for receipt in completion["gate_model_receipts"]) == 0
        and len(completion["model_receipts"]) == 45
        and all(
            receipt["failed_gate_exact_incumbent_identity"] is True
            for receipt in completion["model_receipts"]
        )
        and completion["test_value_reads"]
        == completion["candidate_files"]
        == completion["uploads"]
        == 0
    ):
        raise PermissionError("Gen5r6 exact-no-op closure semantics differ")

    matched = _strict_json(
        root / history["offset_drift_matched_filter_inner_v1"]["pins"]["metrics"]["path"]
    )
    matched_manifest = _strict_json(
        root / history["offset_drift_matched_filter_inner_v1"]["pins"]["manifest"]["path"]
    )
    if not (
        matched["passed"] is False
        and matched["outer_accessed"] is False
        and matched["aggregate"]["offset_recall_delta"] == 0.21981292517006806
        and matched["aggregate"]["drift_recall_delta"] == 0.12564766839378239
        and matched["aggregate"]["normal_fp_relative_increase"] == 1.0294117647058825
        and matched["aggregate"]["worst_group_f1_delta"] == -0.6481481481481481
        and matched_manifest["submission_created"] is False
        and matched_manifest["test_values_used"] is False
    ):
        raise PermissionError("matched-filter closure semantics differ")
    return {
        "pin_count": sum(
            len(payload.get("pins", {"self": payload})) for payload in history.values()
        ),
        "identities": identities,
        "typed_unary_family_closed": True,
        "tcn_replacement_closed": True,
        "ssl_replacement_closed": True,
        "gen5r6_all_45_gates_failed_exact_no_op": True,
        "matched_filter_fp_and_worst_group_failure_bound": True,
    }


def _verify_v9(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["v9_binding"]
    path = root / binding["path"]
    identity = _regular_file(path, role="v9 ledger", allow_open=True)
    if identity["bytes"] != binding["bytes"] or identity["sha256"] != binding["sha256"]:
        raise PermissionError("v9 ledger byte anchor differs")
    append_lock = path.with_name(f"{path.name}.append.lock")
    if append_lock.exists():
        raise PermissionError("v9 append lock must remain absent")
    records = validate_ledger(root, path)
    if not (
        len(records) == binding["local_event_count"] == 3
        and records[-1]["seq"] == binding["head_seq"] == 5
        and records[-1]["event_sha256"] == binding["head_event_sha256"]
        and [record["seq"] for record in records] == [3, 4, 5]
        and records[-1]["payload"]["decision"]["problem"] == "P1"
        and records[-1]["payload"]["upload_performed"] is False
        and records[-1]["payload"]["decision"]["decision"] == "RESEARCH_ONLY"
        and records[-1]["payload"]["decision"]["full_fraction_improvement"] == 0.0
        and all(record["payload"].get("upload_performed", False) is False for record in records)
        and binding["semantic_upload_count"] == 0
        and binding["append_allowed_during_static_stage"] is False
    ):
        raise PermissionError("v9 semantic anchor differs")
    return {
        **identity,
        "path": binding["path"],
        "local_event_count": len(records),
        "head_seq": records[-1]["seq"],
        "head_event_sha256": records[-1]["event_sha256"],
        "semantic_upload_count": 0,
        "append_lock_exists": False,
    }


def _future_path_state(root: Path, config: Mapping[str, Any]) -> dict[str, bool]:
    state: dict[str, bool] = {}
    for value in config["future_output_paths_must_be_absent"]:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("future output path is non-portable")
        resolved = (root / relative).resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise PermissionError("future output path escaped workspace")
        state[relative.as_posix()] = resolved.exists()
    if any(state.values()):
        raise PermissionError("future run/control/output path already exists")
    return state


def check_only(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root, data_dir = _environment_paths(environ)
    config, config_identity = _verify_config(root)
    future_before = _future_path_state(root, config)
    v9_before = _verify_v9(root, config)
    source_boundary = _verify_source_boundary(data_dir, config)
    history = _verify_sealed_history(root, config)
    implementation = _verify_implementation(root)
    helper_audit = static_contract_audit()
    if helper_audit["model_fits"] or helper_audit["predictions_generated"]:
        raise PermissionError("static helper audit executed a model path")
    if helper_audit["scores_computed"] or helper_audit["test_value_reads"]:
        raise PermissionError("static helper audit executed a score or test read")
    v9_after = _verify_v9(root, config)
    future_after = _future_path_state(root, config)
    if v9_after != v9_before or future_after != future_before:
        raise PermissionError("check-only mutated v9 or future output state")
    return {
        "status": "P1_MULTISCALE_CROSS_LAYER_OFFSET_DRIFT_V6_STATIC_CHECK_PASS",
        "verdict": "STATIC_OWNER_GO_AWAIT_INDEPENDENT_QA",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "config": {"path": CANONICAL_CONFIG, **config_identity},
        "implementation": implementation,
        "sealed_history": history,
        "diagnostic_identifiable": True,
        "feature_count": len(GEOMETRY_FEATURES),
        "resource_ceiling": config["resource_ceiling"],
        "v9_binding": v9_after,
        "source_boundary": source_boundary,
        "helper_static_audit": helper_audit,
        "future_paths_absent": not any(future_after.values()),
        "static_operation_counts": {
            "qa_receipts_created": 0,
            "authorizations_created": 0,
            "locks_created": 0,
            "model_fits": 0,
            "predictions_generated": 0,
            "target_scores": 0,
            "artifact_outputs": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "ledger_appends": 0,
            "uploads": 0,
        },
        "actual_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the static preregistration without creating any state",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check_only:
        raise PermissionError("this static owner runner exposes only --check-only")
    print(json.dumps(check_only(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
