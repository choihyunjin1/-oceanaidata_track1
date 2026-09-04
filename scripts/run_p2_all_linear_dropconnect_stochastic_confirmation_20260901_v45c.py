"""Confirm the frozen P2 v45 DropConnect candidate with one new seed trio."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_all_linear_dropconnect_deepset_20260901_v45 as v45  # noqa: E402

v37 = v45.v37
v13 = v45.v13
v12 = v45.v12

EXPERIMENT_ID = "p2_all_linear_dropconnect_stochastic_confirmation_20260901_v45c"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V45C_ALL_LINEAR_DROPCONNECT_CONFIRM_BLEND020"
RESULT_SCHEMA = (
    "p2.all_linear_dropconnect_deepset.stochastic_confirmation.result.20260901.v45c"
)

_BASE_RUN = v45._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v45._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v45._V13_RUNNER


def _bind_base() -> None:
    v13.EXPERIMENT_ID = EXPERIMENT_ID
    v13.CONFIG = CONFIG
    v13.ARTIFACT = ARTIFACT
    v13.REPORT = REPORT
    v13.RUNNER = RUNNER
    v13.PREDICTION_NAME = PREDICTION_NAME
    v13.load_config = load_config
    v13.domain_balanced_weights = _BASE_DOMAIN_BALANCED_WEIGHTS
    v13.train_predict_seed = v45.train_predict_seed
    v13.write_report = write_report


def _without_seeds(training: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(training)
    value.pop("seeds")
    return value


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = config["authorization_evidence"]
    paths = {
        name: ROOT / evidence[name]
        for name in (
            "v45_result",
            "v45_prediction",
            "v45_config",
            "v45_runner",
            "prospective_gate_amendment",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise v12.ContractError(f"v45c evidence missing: {path}")
    for name, path in paths.items():
        expected = evidence[f"{name}_sha256"]
        if v12.sha256_file(path) != expected:
            raise v12.ContractError(f"v45c evidence hash drift: {name}")

    v45_config = json.loads(paths["v45_config"].read_text(encoding="utf-8"))
    contract = config["confirmation_contract"]
    current_training = config["training"]
    source_training = v45_config["training"]
    safety = config["evaluation"]["safety_gate"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_STOCHASTIC_CONFIRMATION_NOT_EXECUTED"
        or config["claim_level"]
        != "STOCHASTIC_REPLICATION_ON_EXPOSED_BLOCKS_ONLY"
        or contract["original_v45_seeds"] != [20260901, 20260902, 20260903]
        or contract["confirmation_seeds"] != [20260904, 20260905, 20260906]
        or current_training["seeds"] != contract["confirmation_seeds"]
        or set(contract["original_v45_seeds"]) & set(contract["confirmation_seeds"])
        or not contract["same_exposed_folds_not_fresh_temporal_surface"]
        or contract["candidate_selection_between_seed_trios"]
        or not contract["v45_original_commitment_remains_representative"]
        or contract["automatic_retry_count"] != 0
        or _without_seeds(current_training) != _without_seeds(source_training)
        or config["source_contract"] != v45_config["source_contract"]
        or config["evaluation"] != v45_config["evaluation"]
        or safety["minimum_fold_layer_non_harm_cells"] != 8
        or safety["total_fold_layer_cells"] != 9
        or safety["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or config["operation_limits"]["maximum_fit_count"] != 9
        or config["operation_limits"]["automatic_retry_count"] != 0
    ):
        raise v12.ContractError("v45c frozen stochastic-confirmation contract drift")
    source_result = json.loads(paths["v45_result"].read_text(encoding="utf-8"))
    if (
        source_result["status"]
        != "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        or not source_result["candidate"]["safety_pass_with_v26a_amendment"]
        or source_result["fit_count"] != 9
    ):
        raise v12.ContractError("v45 source is not the frozen safety-pass candidate")
    return config


def _scientific_contract_sha256(config: dict[str, Any]) -> str:
    return v12.sha256_json(
        {
            "source_contract": config["source_contract"],
            "training_without_seeds": _without_seeds(config["training"]),
            "evaluation": config["evaluation"],
        }
    )


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = config["authorization_evidence"]
    v45_config = json.loads((ROOT / evidence["v45_config"]).read_text(encoding="utf-8"))
    return {
        "classification": "EXACT_V45_STOCHASTIC_CONFIRMATION_NEW_DISJOINT_SEEDS",
        "same_architecture_objective_optimizer_blend_caps": (
            _without_seeds(config["training"])
            == _without_seeds(v45_config["training"])
            and config["evaluation"] == v45_config["evaluation"]
        ),
        "original_seeds": config["confirmation_contract"]["original_v45_seeds"],
        "confirmation_seeds": config["confirmation_contract"]["confirmation_seeds"],
        "seed_sets_disjoint": not bool(
            set(config["confirmation_contract"]["original_v45_seeds"])
            & set(config["confirmation_contract"]["confirmation_seeds"])
        ),
        "same_exposed_folds_not_fresh_temporal_surface": True,
        "candidate_selection_between_seed_trios": False,
        "official_v23_feedback_used_for_settings_or_gate": False,
        "v45_original_commitment_remains_representative": True,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    contract = v45._dropconnect_contract_receipt()
    isolation = v45._isolation_receipt()
    audit = semantic_audit(config)
    if (
        not audit["same_architecture_objective_optimizer_blend_caps"]
        or not audit["seed_sets_disjoint"]
        or contract["module_count"] != 5
        or contract["module_shapes"]
        != [[32, 8], [32, 32], [32, 75], [32, 32], [1, 32]]
        or contract["parameters"] != 4865
        or contract["parameter_tensors"] != 10
        or contract["buffers"] != 0
        or contract["evaluation_initial_function_maximum_abs_error"] != 0.0
        or contract["deterministic_same_seed_training_maximum_abs_error"] != 0.0
        or not contract["deterministic_same_seed_mask_hashes"]
        or not contract["consecutive_step_masks_distinct"]
        or not 0.85 <= contract["first_step_keep_share"] <= 0.95
        or contract["zero_probability_training_maximum_abs_error"] != 0.0
        or not contract["evaluation_rng_unchanged"]
        or contract["dropout_module_count"] != 0
        or max(isolation.values()) > 1e-6
    ):
        raise v12.ContractError("v45c target-free confirmation preflight failed")
    evidence = config["authorization_evidence"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_CONFIRMATION_PREFLIGHT_READY",
        "semantic_audit": audit,
        "dropconnect_contract": contract,
        "isolation": isolation,
        "prospective_fold_layer_gate": config["evaluation"]["safety_gate"],
        "prefix_cutoffs": {
            fold: (pd.Timestamp(start) - pd.Timedelta(days=7)).isoformat()
            for fold, start in config["training"]["fold_starts_kst"].items()
        },
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "scientific_contract_sha256": _scientific_contract_sha256(config),
        "v45_result_sha256": evidence["v45_result_sha256"],
        "v45_prediction_sha256": evidence["v45_prediction_sha256"],
        "v45_config_sha256": evidence["v45_config_sha256"],
        "v45_runner_sha256": evidence["v45_runner_sha256"],
        "gate_amendment_sha256": evidence["prospective_gate_amendment_sha256"],
        "data_rows_read": 0,
        "model_fits": 0,
        "artifacts_written": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = v12.sha256_json(payload)
    return payload


def _load_committed_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name].copy() for name in bundle.files}


def _comparison_to_v45(
    result: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    evidence = config["authorization_evidence"]
    source_result = json.loads(
        (ROOT / evidence["v45_result"]).read_text(encoding="utf-8")
    )
    source_prediction = _load_committed_prediction(ROOT / evidence["v45_prediction"])
    current_path = ARTIFACT / f"{PREDICTION_NAME}.npz"
    current_prediction = _load_committed_prediction(current_path)
    for name in ("time_ns", "layer", "fold", "reference"):
        if not np.array_equal(current_prediction[name], source_prediction[name]):
            raise v12.ContractError(f"v45c commitment alignment drift: {name}")
    current = current_prediction["candidate"].astype(float)
    source = source_prediction["candidate"].astype(float)
    difference = current - source
    current_record = result["candidate"]
    source_record = source_result["candidate"]
    fold_layer_difference: dict[str, dict[str, float]] = {}
    for fold in v12.metric_engine.FOLD_ORDER:
        fold_layer_difference[fold] = {
            layer: float(
                current_record["by_fold_layer"][fold][layer]["delta_rmse"]
                - source_record["by_fold_layer"][fold][layer]["delta_rmse"]
            )
            for layer in ("2", "3", "4")
        }
    return {
        "use": "diagnostic_only_no_seed_trio_selection_or_ensemble",
        "v45_result_sha256": evidence["v45_result_sha256"],
        "v45_prediction_sha256": evidence["v45_prediction_sha256"],
        "v45_status": source_result["status"],
        "v45_delta_rmse_C": float(source_record["delta_rmse"]),
        "confirmation_delta_rmse_C": float(current_record["delta_rmse"]),
        "confirmation_minus_v45_delta_rmse_C": float(
            current_record["delta_rmse"] - source_record["delta_rmse"]
        ),
        "v45_candidate_rmse_C": float(source_record["candidate_rmse"]),
        "confirmation_candidate_rmse_C": float(current_record["candidate_rmse"]),
        "confirmation_minus_v45_candidate_rmse_C": float(
            current_record["candidate_rmse"] - source_record["candidate_rmse"]
        ),
        "candidate_prediction_difference": {
            "rows": int(len(difference)),
            "mean_C": float(np.mean(difference)),
            "rms_C": float(np.sqrt(np.mean(np.square(difference)))),
            "mean_abs_C": float(np.mean(np.abs(difference))),
            "p99_abs_C": float(np.quantile(np.abs(difference), 0.99)),
            "max_abs_C": float(np.max(np.abs(difference))),
            "pearson_correlation": float(np.corrcoef(current, source)[0, 1]),
        },
        "fold_layer_delta_rmse_difference_C": fold_layer_difference,
        "v45_original_commitment_remains_representative": True,
    }


def write_report(result: dict[str, Any]) -> None:
    if "confirmation_vs_v45" not in result:
        return
    record = result["candidate"]
    local = record["prospective_fold_layer_gate"]
    comparison = result["confirmation_vs_v45"]
    priority = result["priority_vs_v23"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v45c frozen DropConnect stochastic confirmation\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{record['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{record['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{record['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"v45 대비 delta-RMSE 차이: "
        f"`{comparison['confirmation_minus_v45_delta_rmse_C']:+.9f} C`; "
        f"prospective gate `{local['pass']}`, non-harm `{local['non_harm_cells']}/9`, "
        f"max cell `{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        f"우선순위: `{priority['decision']}`. 동일 exposed block의 새 seed 확인일 뿐 "
        "fresh temporal confirmation이 아니다. 원 v45 commitment를 대표 후보로 유지하며 "
        "seed trio 간 cherry-pick/ensemble/retune은 하지 않는다. "
        "official/test/sample/hidden/query/CSV/upload=0.\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    _bind_base()
    started = time.perf_counter()
    result = _BASE_RUN()
    config = load_config()
    record = result["candidate"]
    legacy_safety = bool(record["safety_pass"])
    amended = v37.prospective_fold_layer_gate(record, config)
    record["legacy_safety_pass_without_v26a_amendment"] = legacy_safety
    record["prospective_fold_layer_gate"] = amended
    record["safety_pass"] = bool(legacy_safety and amended["pass"])
    record["safety_pass_with_v26a_amendment"] = record["safety_pass"]
    passed = bool(record["strict_exploratory_pass"] and record["safety_pass"])
    result["schema_version"] = RESULT_SCHEMA
    result["status"] = (
        "STOCHASTIC_CONFIRMATION_PASS_EXPOSED_BLOCKS_ONLY"
        if passed
        else "STOCHASTIC_CONFIRMATION_NO_GO_ALL_LINEAR_DROPCONNECT"
    )
    result["claim_level"] = "STOCHASTIC_REPLICATION_ON_EXPOSED_BLOCKS_ONLY"
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["dropconnect_contract"] = v45._dropconnect_contract_receipt()
    result["isolation"] = v45._isolation_receipt()
    result["confirmation_vs_v45"] = _comparison_to_v45(result, config)
    result["priority_vs_v23"] = {
        "decision": (
            "V45_FAMILY_FIRST_INTERNAL_DEPLOYMENT_PREFLIGHT_PRIORITY"
            if passed
            else "V23_REMAINS_ONLY_DEPLOYMENT_PRIORITY"
        ),
        "v23_position": "OFFICIAL_ANCHOR_REMAINS_UNCHANGED",
        "v45_exact_representative": (
            "original_v45_commitment_no_seed_trio_cherry_pick" if passed else "HOLD"
        ),
        "official_v23_feedback_used_for_settings_gate_or_slice": False,
        "fresh_deployment_preflight_required_before_any_materialization": True,
    }
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer": config["training"]["optimizer"],
            "dropconnect": config["training"]["dropconnect"],
            "original_v45_seeds": config["confirmation_contract"][
                "original_v45_seeds"
            ],
            "confirmation_seeds": config["training"]["seeds"],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
            "extra_loss": 0,
            "extra_parameters": 0,
        }
    )
    evidence = config["authorization_evidence"]
    result["hashes"].update(
        {
            "v13_runner": v12.sha256_file(_V13_RUNNER),
            "v45_result": evidence["v45_result_sha256"],
            "v45_prediction": evidence["v45_prediction_sha256"],
            "v45_config": evidence["v45_config_sha256"],
            "v45_runner": evidence["v45_runner_sha256"],
            "prospective_gate_amendment": evidence[
                "prospective_gate_amendment_sha256"
            ],
            "scientific_contract": _scientific_contract_sha256(config),
        }
    )
    v12.atomic_json(ARTIFACT / "result.json", result)
    v12.atomic_json(REPORT / "result.json", result)
    write_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    value = preflight() if args.preflight else run()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
