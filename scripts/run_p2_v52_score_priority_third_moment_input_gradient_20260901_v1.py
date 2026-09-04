"""Run sealed P2 v52 score-priority third-moment plus input-gradient once."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_masked_third_central_moment_profile_pooling_deepset_20260901_v50 as v50  # noqa: E402
import run_p2_public_temperature_input_gradient_regularized_deepset_20260901_v23 as v23  # noqa: E402

v13 = v50.v13
v12 = v50.v12
v37 = v50.v37

EXPERIMENT_ID = "p2_v52_score_priority_third_moment_input_gradient_20260901_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V52_SCORE_PRIORITY_THIRD_MOMENT_INPUT_GRADIENT_BLEND020"
RESULT_SCHEMA = "p2.v52_score_priority_third_moment_input_gradient.result.20260901.v1"

_BASE_RUN = v50._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v50._BASE_DOMAIN_BALANCED_WEIGHTS
_EVIDENCE_NAMES = (
    "organizer_policy",
    "organizer_policy_registry",
    "negative_fingerprint",
    "v50_result",
    "v23_result",
    "v23_runner",
    "v23_official_receipt",
)


def _verify_file(relative: str, expected: str, label: str) -> Path:
    path = ROOT / relative
    if not path.is_file() or v12.sha256_file(path) != expected:
        raise v12.ContractError(f"v52 evidence drift: {label}")
    return path


def load_config() -> dict[str, Any]:
    """Resolve the sealed small override onto the hash-pinned v50 contract."""
    own = json.loads(CONFIG.read_text(encoding="utf-8"))
    base_contract = own["base_contract"]
    base_path = _verify_file(
        base_contract["config"], base_contract["config_sha256"], "base_config"
    )
    _verify_file(
        base_contract["runner"], base_contract["runner_sha256"], "base_runner"
    )
    for name in _EVIDENCE_NAMES:
        _verify_file(
            own["evidence"][name], own["evidence"][f"{name}_sha256"], name
        )
    policy = json.loads(
        (ROOT / own["evidence"]["organizer_policy_registry"]).read_text(
            encoding="utf-8"
        )
    )
    fingerprint = json.loads(
        (ROOT / own["evidence"]["negative_fingerprint"]).read_text(
            encoding="utf-8"
        )
    )
    base = json.loads(base_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(base)
    config["schema_version"] = own["schema_version"]
    config["experiment_id"] = own["experiment_id"]
    config["status"] = own["status"]
    config["claim_level"] = own["claim_level"]
    config["ready_preflight_contract"] = own["ready_preflight_contract"]
    config["operation_limits"] = own["operation_limits"]
    config["evaluation"]["score_priority_gate"] = own["evaluation"][
        "score_priority_gate"
    ]
    config["evaluation"]["stability_diagnostic_not_relaxed"] = own[
        "evaluation"
    ]["stability_diagnostic_not_relaxed"]
    config["evaluation"]["official_champion_reference"] = own["evaluation"][
        "official_champion_reference"
    ]
    config["training"]["architecture"] = own["method"]["architecture"]
    config["training"]["objective"] = own["method"]["objective"]
    config["training"]["input_gradient"] = {
        "coefficient": own["method"]["input_gradient_coefficient"],
        "token_channel": own["method"]["gradient_token_channel"],
        "observed_token_mask_only": own["method"]["observed_token_mask_only"],
        "penalize_psal_depth_nominal_presence_context": False,
        "coefficient_sweep": False,
    }
    source = config["source_contract"]
    own_source = own["source_contract"]
    method = own["method"]
    score = config["evaluation"]["score_priority_gate"]
    stability = config["evaluation"]["stability_diagnostic_not_relaxed"]
    ready = config["ready_preflight_contract"]
    if (
        own["experiment_id"] != EXPERIMENT_ID
        or own["status"] != "PREREGISTERED_SCORE_PRIORITY_NOT_EXECUTED"
        or policy["status"] != "ACTIVE_HIGHEST_PRECEDENCE"
        or not policy["distributed_data_only"]
        or fingerprint["experiment_id"] != EXPERIMENT_ID
        or not fingerprint["created_before_preregistration"]
        or fingerprint["repository_exact_execution_hits"] != 0
        or fingerprint["repository_combined_semantic_execution_hits"] != 0
        or method["architecture"] != "v50_masked_third_central_moment_pooling"
        or method["input_gradient_coefficient"] != 0.01
        or method["gradient_token_channel"] != 0
        or not method["observed_token_mask_only"]
        or method["training_initialization"]
        != "fresh_random_scratch_per_fold_seed"
        or method["seeds"] != [20260901, 20260902, 20260903]
        or method["epochs"] != 60
        or method["maximum_fit_count"] != 9
        or method["champion_preserving_weight"] != 0.8
        or method["model_weight"] != 0.2
        or method["model_minus_champion_clip_C"] != 2.5
        or method["maximum_final_action_C"] != 0.5
        or method["hyperparameter_sweep"]
        or method["result_adaptive_router"]
        or method["result_adaptive_ensemble"]
        or source["only_source_filename"] != "observations.csv"
        or own_source["only_direct_source_filename"] != "observations.csv"
        or not own_source["organizer_distributed_data_only"]
        or any(
            own_source[name]
            for name in (
                "external_observation_allowed",
                "external_reanalysis_allowed",
                "external_forecast_allowed",
                "pretrained_weights_allowed",
                "official_test_index_allowed",
                "official_sample_allowed",
                "official_baseline_allowed",
                "query_support_allowed",
                "hidden_truth_allowed",
                "submission_csv_allowed",
                "upload_allowed",
            )
        )
        or score["frozen_v23_internal_delta_rmse_C"]
        != -0.05189246657169555
        or score["frozen_v23_transport_adjusted_points"]
        != 0.5294402465540541
        or stability["minimum_fold_layer_non_harm_cells"] != 8
        or stability["total_fold_layer_cells"] != 9
        or stability["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or ready["required_receipts"] != 2
        or not ready["byte_identical"]
        or ready["status"] != "ZERO_OPERATION_PREFLIGHT_READY"
        or len(ready["paths"]) != 2
        or own["operation_limits"]["maximum_candidate_count"] != 1
        or own["operation_limits"]["maximum_fit_count"] != 9
        or own["operation_limits"]["automatic_retry_count"] != 0
    ):
        raise v12.ContractError("v52 score-priority contract drift")
    v50._prelock_source_contract_guard(config)
    return config


def train_predict_seed(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Use v23's fixed observed-temperature gradient penalty on v50's model."""
    return v23.train_predict_seed(*args, **kwargs)


def _bind_base() -> None:
    v13.EXPERIMENT_ID = EXPERIMENT_ID
    v13.CONFIG = CONFIG
    v13.ARTIFACT = ARTIFACT
    v13.REPORT = REPORT
    v13.RUNNER = RUNNER
    v13.PREDICTION_NAME = PREDICTION_NAME
    v13.load_config = load_config
    v13.domain_balanced_weights = _BASE_DOMAIN_BALANCED_WEIGHTS
    v13.train_predict_seed = train_predict_seed
    v13.write_report = write_report
    v12.VerticalDeepSet = v50.MaskedThirdCentralMomentProfileVerticalDeepSet


def _semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    fingerprint = json.loads(
        (ROOT / config["evidence"]["negative_fingerprint"]).read_text(
            encoding="utf-8"
        )
    ) if "evidence" in config else json.loads(
        (ROOT / json.loads(CONFIG.read_text(encoding="utf-8"))["evidence"]["negative_fingerprint"]).read_text(encoding="utf-8")
    )
    return {
        "classification": "NEW_FIXED_V50_POOLING_PLUS_V23_INPUT_GRADIENT",
        "exact_execution_hits_before_preregistration": fingerprint[
            "repository_exact_execution_hits"
        ],
        "combined_semantic_execution_hits_before_preregistration": fingerprint[
            "repository_combined_semantic_execution_hits"
        ],
        "component_hyperparameters_changed": False,
        "result_adaptive_router_or_ensemble": False,
        "official_feedback_used_to_tune_combination": False,
    }


def _source_guard(config: dict[str, Any]) -> dict[str, Any]:
    return v50._synthetic_source_contract_guard_receipt(config)


def _lineage() -> dict[str, Any]:
    return {
        "organizer_distributed_data_only": True,
        "only_direct_source_filename": "observations.csv",
        "truth_free_historical_scoring_frame": True,
        "fresh_random_scratch_initialization": True,
        "external_observation_rows": 0,
        "external_reanalysis_rows": 0,
        "external_forecast_rows": 0,
        "pretrained_weight_files_loaded": 0,
        "official_test_index_rows": 0,
        "sample_rows": 0,
        "baseline_file_rows": 0,
        "query_support_rows": 0,
        "hidden_truth_rows": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }


def preflight() -> dict[str, Any]:
    config = load_config()
    source_guard = _source_guard(config)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_READY",
        "method": {
            "pooling": v50._pooling_contract_receipt(),
            "input_gradient": v23._gradient_scope_receipt(),
            "isolation": v50._isolation_receipt(),
        },
        "source_contract_guard": source_guard,
        "lineage": _lineage(),
        "fold_cutoffs": {
            fold: (pd.Timestamp(start) - pd.Timedelta(days=7)).isoformat()
            for fold, start in config["training"]["fold_starts_kst"].items()
        },
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "data_rows_read": 0,
        "model_fits": 0,
        "official_test_index_rows_read": 0,
        "sample_rows_read": 0,
        "baseline_file_rows_read": 0,
        "query_support_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = v12.sha256_json(payload)
    return payload


def _ready_pair(config: dict[str, Any]) -> dict[str, Any]:
    paths = [ROOT / item for item in config["ready_preflight_contract"]["paths"]]
    if not all(path.is_file() for path in paths):
        raise v12.ContractError("v52 READY receipt missing")
    first, second = (path.read_bytes() for path in paths)
    if first != second:
        raise v12.ContractError("v52 READY receipts differ")
    stored = json.loads(first.decode("utf-8"))
    if stored != preflight():
        raise v12.ContractError("v52 READY receipt drift")
    return {
        "paths": [str(path.relative_to(ROOT)) for path in paths],
        "byte_identical": True,
        "bytes_each": len(first),
        "sha256_each": v12.sha256_file(paths[0]),
    }


def _score_gate(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = config["evaluation"]["score_priority_gate"]
    folds = record["by_fold"]
    checks = {
        "beats_frozen_v23_internal_delta_rmse": record["delta_rmse"]
        < gate["frozen_v23_internal_delta_rmse_C"],
        "all_three_folds_improve": all(
            item["delta_rmse"] < 0.0 for item in folds.values()
        ),
        "official_like_fold_improves": folds["2024_sep_oct"]["delta_rmse"] < 0.0,
        "pooled_bootstrap_ci90_high_below_zero": record["bootstrap"][
            "ci90_high"
        ]
        < 0.0,
        "beats_frozen_v23_transport_points": record[
            "canonical_transport_adjusted_pooled_points_delta"
        ]
        > gate["frozen_v23_transport_adjusted_points"],
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "frozen_v23_internal_delta_rmse_C": gate[
            "frozen_v23_internal_delta_rmse_C"
        ],
        "candidate_minus_v23_delta_rmse_C": record["delta_rmse"]
        - gate["frozen_v23_internal_delta_rmse_C"],
        "frozen_v23_transport_adjusted_points": gate[
            "frozen_v23_transport_adjusted_points"
        ],
        "candidate_minus_v23_transport_points": record[
            "canonical_transport_adjusted_pooled_points_delta"
        ]
        - gate["frozen_v23_transport_adjusted_points"],
    }


def _worst_fold_layer(record: dict[str, Any]) -> dict[str, Any]:
    cells = [
        (values["delta_rmse"], fold, layer, values)
        for fold, layers in record["by_fold_layer"].items()
        for layer, values in layers.items()
    ]
    delta, fold, layer, values = max(cells)
    return {"fold": fold, "layer": layer, **values, "delta_rmse": delta}


def write_report(result: dict[str, Any]) -> None:
    record = result["candidate"]
    score = result.get("score_priority_gate", {"pass": False})
    stability = result.get("stability_diagnostic", {"pass": False})
    worst = result.get("worst_fold_layer", {})
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v52 score-priority third moment + input gradient\n\n"
        "## 결론\n\n"
        f"상태: `{result.get('status', 'RUNNING')}`. pooled RMSE "
        f"`{record['candidate_rmse']:.9f} C`, ΔRMSE "
        f"`{record['delta_rmse']:+.9f} C`, nominal "
        f"`{record['canonical_nominal_pooled_points_delta']:+.6f}`점, transport "
        f"`{record['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점.\n\n"
        f"Score gate `{score['pass']}`, stability diagnostic "
        f"`{stability['pass']}`. Worst fold-layer: "
        f"`{worst.get('fold', 'pending')}/L{worst.get('layer', 'pending')}` "
        f"`{worst.get('delta_rmse', float('nan')):+.9f} C`.\n\n"
        "v50의 masked signed third-central-moment pooling과 v23의 observed-public-"
        "temperature input-gradient L2(lambda=0.01)를 사전 고정 결합했다. "
        "배포 observations.csv와 truth-free 파생 scoring frame만 사용하며 scratch "
        "9 fits다. 외부/사전학습/official/test/sample/baseline/query/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    config = load_config()
    source_guard = _source_guard(config)
    ready = _ready_pair(config)
    _bind_base()
    started = time.perf_counter()
    result = _BASE_RUN()
    config = load_config()
    record = result["candidate"]
    stability = v37.prospective_fold_layer_gate(record, config)
    score = _score_gate(record, config)
    result["schema_version"] = RESULT_SCHEMA
    result["score_priority_gate"] = score
    result["stability_diagnostic"] = stability
    result["worst_fold_layer"] = _worst_fold_layer(record)
    if score["pass"]:
        result["status"] = (
            "SCORE_PRIORITY_PASS_STABILITY_PASS"
            if stability["pass"]
            else "SCORE_PRIORITY_PASS_EXPLICIT_STABILITY_RISK"
        )
    else:
        result["status"] = "NO_GO_SCORE_PRIORITY_DID_NOT_BEAT_V23"
    result["worth_later_official_submission"] = bool(score["pass"])
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = _semantic_audit(config)
    result["prelock_source_contract_guard"] = source_guard
    result["ready_preflight_pair"] = ready
    result["lineage"] = _lineage()
    result["method_contract"] = {
        "masked_third_central_moment_pooling": True,
        "observed_public_temperature_input_gradient_L2": True,
        "input_gradient_coefficient": 0.01,
        "fresh_random_scratch": True,
        "result_adaptive_tuning": False,
    }
    result["operation_counters"].update(
        {
            "external_observation_rows_read": 0,
            "external_reanalysis_rows_read": 0,
            "external_forecast_rows_read": 0,
            "pretrained_weight_files_loaded": 0,
            "official_test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_file_rows_read": 0,
            "query_support_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        }
    )
    own = json.loads(CONFIG.read_text(encoding="utf-8"))
    result["hashes"]["base_config"] = own["base_contract"]["config_sha256"]
    result["hashes"]["base_runner"] = own["base_contract"]["runner_sha256"]
    for name in _EVIDENCE_NAMES:
        result["hashes"][name] = own["evidence"][f"{name}_sha256"]
    v12.atomic_json(ARTIFACT / "result.json", result)
    v12.atomic_json(REPORT / "result.json", result)
    write_report(result)
    return result


def _write_receipt(path_text: str, value: dict[str, Any]) -> None:
    config = load_config()
    allowed = {
        (ROOT / item).resolve()
        for item in config["ready_preflight_contract"]["paths"]
    }
    path = (ROOT / path_text).resolve()
    if path not in allowed:
        raise v12.ContractError("v52 preflight receipt path is not sealed")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v12.atomic_json(path, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    if args.execute and args.receipt:
        raise SystemExit("--receipt is valid only with --preflight")
    value = preflight() if args.preflight else run()
    if args.receipt:
        _write_receipt(args.receipt, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
