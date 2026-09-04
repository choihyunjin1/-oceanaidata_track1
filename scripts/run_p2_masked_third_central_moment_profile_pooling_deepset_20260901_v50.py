"""Run sealed P2 v50 masked third-central-moment DeepSets once."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_masked_logmeanexp_profile_pooling_deepset_20260901_v49 as v49  # noqa: E402

v48 = v49.v48
v46 = v49.v46
v37 = v49.v37
v13 = v49.v13
v12 = v49.v12

EXPERIMENT_ID = "p2_masked_third_central_moment_profile_pooling_deepset_20260901_v50"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V50_MASKED_THIRD_CENTRAL_MOMENT_PROFILE_POOLING_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.masked_third_central_moment_profile_pooling_deepset.result.20260901.v50"

_BASE_RUN = v49._BASE_RUN
_BASE_MODEL = v49._BASE_MODEL
_BASE_TRAIN_PREDICT_SEED = v49._BASE_TRAIN_PREDICT_SEED
_BASE_DOMAIN_BALANCED_WEIGHTS = v49._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = Path(v13.__file__)
_INHERITED_SOURCE_KEYS = (
    "environment_variable",
    "only_source_filename",
    "observations_sha256",
    "scoring_frame",
    "scoring_frame_sha256",
)
EVIDENCE_NAMES = (
    "organizer_policy",
    "organizer_policy_registry",
    "negative_fingerprint",
    "v13_result",
    "v15_result",
    "v16_result",
    "v20_result",
    "v43_result",
    "v45_result",
    "v45c_result",
    "v46_result",
    "v47_result",
    "v48_result",
    "v49_attempt_lock",
    "v49_stderr",
    "prospective_gate_amendment",
)


class MaskedThirdCentralMomentProfileVerticalDeepSet(_BASE_MODEL):
    """Exact v13 plus one identity-initialized signed profile-asymmetry summary."""

    def __init__(
        self, token_features: int, context_features: int, hidden: int = 32
    ) -> None:
        super().__init__(token_features, context_features, hidden)
        original = self.head[0]
        expanded = nn.Linear(hidden * 3 + context_features, hidden)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.bias.copy_(original.bias)
            expanded.weight[:, : hidden * 2].copy_(
                original.weight[:, : hidden * 2]
            )
            expanded.weight[:, hidden * 3 :].copy_(
                original.weight[:, hidden * 2 :]
            )
        self.head[0] = expanded

    @staticmethod
    def masked_third_central_moment(
        encoded: Tensor, token_mask: Tensor
    ) -> Tensor:
        mask = token_mask.unsqueeze(-1).to(encoded.dtype)
        raw_count = mask.sum(dim=1)
        count = raw_count.clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        centered = (encoded - mean.unsqueeze(1)) * mask
        moment = centered.pow(3).sum(dim=1) / count
        return torch.where(raw_count > 0, moment, torch.zeros_like(moment))

    def pooled_profile_context(
        self, tokens: Tensor, token_mask: Tensor, context: Tensor
    ) -> Tensor:
        encoded = self.element(tokens)
        mask = token_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool(), negative).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        third = self.masked_third_central_moment(encoded, token_mask)
        return torch.cat((mean, maximum, third, context), dim=1)

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        return self.head(
            self.pooled_profile_context(tokens, token_mask, context)
        ).squeeze(1)


def _bind_base() -> None:
    v13.EXPERIMENT_ID = EXPERIMENT_ID
    v13.CONFIG = CONFIG
    v13.ARTIFACT = ARTIFACT
    v13.REPORT = REPORT
    v13.RUNNER = RUNNER
    v13.PREDICTION_NAME = PREDICTION_NAME
    v13.load_config = load_config
    v13.domain_balanced_weights = _BASE_DOMAIN_BALANCED_WEIGHTS
    v13.train_predict_seed = _BASE_TRAIN_PREDICT_SEED
    v13.write_report = write_report
    v12.VerticalDeepSet = MaskedThirdCentralMomentProfileVerticalDeepSet


def _prelock_source_contract_guard(config: dict[str, Any]) -> dict[str, Any]:
    """Validate every key dereferenced by inherited v12/v13 before their lock."""

    source = config.get("source_contract")
    if not isinstance(source, dict):
        raise v12.ContractError("v50 source_contract must be an object")
    missing = [key for key in _INHERITED_SOURCE_KEYS if key not in source]
    if missing:
        raise v12.ContractError(
            "v50 inherited source-contract keys missing: " + ",".join(missing)
        )
    if tuple(source.get("inherited_required_keys", ())) != _INHERITED_SOURCE_KEYS:
        raise v12.ContractError("v50 inherited source-contract key registry drift")
    if not source.get("prelock_synthetic_key_validation_required"):
        raise v12.ContractError("v50 pre-lock source validation disabled")
    if (
        source["environment_variable"] != "P2_DATA_DIR"
        or source["only_source_filename"] != "observations.csv"
        or source.get("only_direct_source_filename")
        != source["only_source_filename"]
        or not isinstance(source["observations_sha256"], str)
        or len(source["observations_sha256"]) != 64
        or source["scoring_frame"] != source["allowed_derived_training_artifact"]
        or source["scoring_frame_sha256"]
        != source["allowed_derived_training_artifact_sha256"]
        or len(source["scoring_frame_sha256"]) != 64
    ):
        raise v12.ContractError("v50 inherited source-contract value drift")
    resolver_source = inspect.getsource(v12.resolve_observations)
    base_run_source = inspect.getsource(v13.run)
    referenced = {
        "environment_variable": 'source_contract"]["environment_variable' in resolver_source,
        "only_source_filename": 'source_contract"]["only_source_filename' in resolver_source,
        "observations_sha256": 'source_contract"]["observations_sha256' in resolver_source,
        "scoring_frame": 'source_contract"]["scoring_frame' in base_run_source,
        "scoring_frame_sha256": 'source_contract"]["scoring_frame_sha256' in base_run_source,
    }
    if not all(referenced.values()):
        raise v12.ContractError("v50 inherited source-contract implementation drift")
    return {
        "required_keys": list(_INHERITED_SOURCE_KEYS),
        "all_required_keys_present": True,
        "only_source_filename": source["only_source_filename"],
        "direct_alias_matches": True,
        "inherited_implementation_references": referenced,
        "guard_runs_before_base_lock": True,
    }


def _synthetic_source_contract_guard_receipt(
    config: dict[str, Any],
) -> dict[str, Any]:
    intact = _prelock_source_contract_guard(config)
    rejected: list[str] = []
    for key in _INHERITED_SOURCE_KEYS:
        broken = copy.deepcopy(config)
        del broken["source_contract"][key]
        try:
            _prelock_source_contract_guard(broken)
        except v12.ContractError:
            rejected.append(key)
        else:
            raise v12.ContractError(
                f"v50 synthetic missing-key case was accepted: {key}"
            )
    return {
        **intact,
        "synthetic_missing_key_cases": len(_INHERITED_SOURCE_KEYS),
        "synthetic_missing_key_rejections": rejected,
        "synthetic_files_opened": 0,
        "synthetic_data_rows_read": 0,
        "synthetic_model_fits": 0,
    }


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = config["authorization_evidence"]
    for name in EVIDENCE_NAMES:
        path = ROOT / evidence[name]
        expected = evidence[f"{name}_sha256"]
        if not path.is_file() or v12.sha256_file(path) != expected:
            raise v12.ContractError(f"v50 evidence drift: {name}")
    fingerprint = json.loads(
        (ROOT / evidence["negative_fingerprint"]).read_text(encoding="utf-8")
    )
    policy = json.loads(
        (ROOT / evidence["organizer_policy_registry"]).read_text(encoding="utf-8")
    )
    training = config["training"]
    pooling = training["masked_third_central_moment_pooling"]
    support = training["support_qualification"]
    source = config["source_contract"]
    gate = config["evaluation"]["safety_gate"]
    ready = config["ready_preflight_contract"]
    forbidden_source_flags = (
        source["official_test_index_allowed"],
        source["official_sample_allowed"],
        source["official_baseline_allowed"],
        source["query_support_allowed"],
        source["hidden_truth_allowed"],
        source["submission_csv_allowed"],
        source["upload_allowed"],
        source["external_observation_allowed"],
        source["external_reanalysis_allowed"],
        source["external_forecast_allowed"],
        source["pretrained_weights_allowed"],
    )
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or policy["status"] != "ACTIVE_HIGHEST_PRECEDENCE"
        or not policy["distributed_data_only"]
        or fingerprint["candidate_experiment_id"] != EXPERIMENT_ID
        or not fingerprint["created_before_preregistration"]
        or fingerprint["exact_execution_hit_count"] != 0
        or fingerprint["semantic_execution_hit_count"] != 0
        or not fingerprint["selected_axis"]["scratch_weights_only"]
        or fingerprint["selected_axis"]["moment_or_pooling_sweep"]
        or fingerprint["v49_failure_separation"]["v49_result_used_for_axis_selection"]
        or training["architecture"]
        != "v13_exact_five_Linear_DeepSets_plus_masked_third_central_moment_profile_pooling"
        or training["initialization"] != "fresh_random_scratch_per_fold_seed"
        or training["pretrained_weight_files_loaded"] != 0
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"] != "exact_v13_weighted_SmoothL1_beta_1.0"
        or training["optimizer"] != "exact_v13_AdamW"
        or training["learning_rate"] != 0.001
        or training["weight_decay"] != 0.0001
        or training["epochs"] != 60
        or training["batch_size"] != 4096
        or training["seeds"] != [20260901, 20260902, 20260903]
        or training["maximum_fit_count"] != 9
        or training["champion_preserving_weight"] != 0.8
        or training["model_weight"] != 0.2
        or training["model_minus_champion_clip_C"] != 2.5
        or training["maximum_final_action_C"] != 0.5
        or any(
            training[name]
            for name in (
                "row_deletion",
                "input_perturbation",
                "data_augmentation",
                "extra_loss",
                "normalization",
                "dropout",
                "early_stopping",
                "outer_fold_tuning",
            )
        )
        or pooling["input_embedding_width"] != 32
        or pooling["descriptor"]
        != "masked_mean((element_embedding-masked_mean(element_embedding))^3)"
        or pooling["moment_order"] != 3
        or pooling["all_missing_fallback"] != 0.0
        or pooling["full_head_input_width"] != 107
        or pooling["new_head_columns_initialization"] != 0.0
        or pooling["added_parameters"] != 1024
        or pooling["expected_parameters"] != 5889
        or pooling["expected_parameter_tensors"] != 10
        or pooling["expected_buffers"] != 0
        or pooling["normalization"]
        or pooling["moment_or_pooling_sweep"]
        or support["supported_folds"] != 3
        or support["training_rows_by_fold"] != [45935, 119667, 149384]
        or not support["finite_public_token_masks"]
        or source["only_source_filename"] != "observations.csv"
        or source["only_direct_source_filename"] != source["only_source_filename"]
        or source["scoring_frame"] != source["allowed_derived_training_artifact"]
        or source["scoring_frame_sha256"]
        != source["allowed_derived_training_artifact_sha256"]
        or source["derived_artifact_contains_hidden_truth"]
        or any(forbidden_source_flags)
        or gate["minimum_fold_layer_non_harm_cells"] != 8
        or gate["total_fold_layer_cells"] != 9
        or gate["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or ready["required_receipts"] != 2
        or not ready["byte_identical"]
        or ready["status"] != "ZERO_OPERATION_PREFLIGHT_READY"
        or len(ready["paths"]) != 2
        or config["result_adaptive_tuning"]
        or config["operation_limits"]["maximum_candidate_count"] != 1
        or config["operation_limits"]["maximum_fit_count"] != 9
        or config["operation_limits"]["automatic_retry_count"] != 0
    ):
        raise v12.ContractError("v50 scientific or policy contract drift")
    _prelock_source_contract_guard(config)
    return config


def _unchanged_state_error(base: nn.Module, candidate: nn.Module) -> float:
    return v49._unchanged_state_error(base, candidate)


def _pooling_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(5001)
    base = _BASE_MODEL(8, 11, hidden=32).eval()
    torch.manual_seed(5001)
    candidate = MaskedThirdCentralMomentProfileVerticalDeepSet(
        8, 11, hidden=32
    ).eval()
    unchanged_error = _unchanged_state_error(base, candidate)
    new_column_initial_max = float(
        torch.max(torch.abs(candidate.head[0].weight[:, 64:96].detach()))
    )
    tokens = torch.randn(7, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 1, 0, 0, 1]]
        + [[1, 1, 1, 0, 0]] * 3
        + [[0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    context = torch.randn(7, 11)
    with torch.no_grad():
        base_output = base(tokens, mask, context)
        initial_output = candidate(tokens, mask, context)
        descriptor = candidate.pooled_profile_context(tokens, mask, context)
        encoded = candidate.element(tokens)
        expanded_mask = mask.unsqueeze(-1).to(encoded.dtype)
        raw_count = expanded_mask.sum(dim=1)
        count = raw_count.clamp_min(1.0)
        mean = (encoded * expanded_mask).sum(dim=1) / count
        centered = (encoded - mean.unsqueeze(1)) * expanded_mask
        manual_third = centered.pow(3).sum(dim=1) / count
        manual_third = torch.where(
            raw_count > 0, manual_third, torch.zeros_like(manual_third)
        )
        manual = candidate.head(descriptor).squeeze(1)
        batch_first = candidate(tokens[:1], mask[:1], context[:1])
        candidate.head[0].weight[:, 64:96].copy_(
            torch.linspace(-0.01, 0.01, 32 * 32).reshape(32, 32)
        )
        learned_output = candidate(tokens, mask, context)
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    loss = candidate(tokens[:-1], mask[:-1], context[:-1]).square().mean()
    loss.backward()
    gradients = [value.grad for value in candidate.parameters()]
    return {
        "descriptor": "masked_mean((element_embedding-masked_mean(element_embedding))^3)",
        "moment_order": 3,
        "embedding_width": 32,
        "descriptor_width": 107,
        "new_head_columns": 32,
        "new_head_columns_initial_maximum_abs": new_column_initial_max,
        "parameters": int(sum(value.numel() for value in candidate.parameters())),
        "parameter_tensors": len(list(candidate.parameters())),
        "buffers": len(list(candidate.buffers())),
        "linear_count": sum(
            isinstance(module, nn.Linear) for module in candidate.modules()
        ),
        "unchanged_parameter_maximum_abs_error_vs_v13": unchanged_error,
        "initial_function_maximum_abs_error_vs_v13": float(
            torch.max(torch.abs(base_output - initial_output))
        ),
        "descriptor_forward_maximum_abs_error": float(
            torch.max(torch.abs(initial_output - manual))
        ),
        "third_moment_manual_maximum_abs_error": float(
            torch.max(torch.abs(descriptor[:, 64:96] - manual_third))
        ),
        "all_missing_third_moment_maximum_abs": float(
            torch.max(torch.abs(descriptor[-1, 64:96]))
        ),
        "third_moment_nonconstant_range": float(
            torch.max(descriptor[:-1, 64:96])
            - torch.min(descriptor[:-1, 64:96])
        ),
        "learned_pooling_columns_change_function_maximum_abs": float(
            torch.max(torch.abs(learned_output - initial_output))
        ),
        "batch_composition_maximum_abs_error": float(
            torch.max(torch.abs(initial_output[:1] - batch_first))
        ),
        "gradients_finite": all(
            value is not None and bool(torch.isfinite(value).all())
            for value in gradients
        ),
        "new_head_column_gradient_finite_nonzero": bool(
            candidate.head[0].weight.grad is not None
            and torch.isfinite(candidate.head[0].weight.grad[:, 64:96]).all()
            and torch.linalg.vector_norm(candidate.head[0].weight.grad[:, 64:96])
            > 0
        ),
        "normalization_count": sum(
            isinstance(module, (nn.LayerNorm, nn.BatchNorm1d))
            for module in candidate.modules()
        ),
        "dropout_count": sum(
            isinstance(module, nn.Dropout) for module in candidate.modules()
        ),
        "attention_count": sum(
            isinstance(module, nn.MultiheadAttention)
            for module in candidate.modules()
        ),
    }


def _isolation_receipt() -> dict[str, float]:
    torch.manual_seed(5002)
    model = MaskedThirdCentralMomentProfileVerticalDeepSet(
        8, 11, hidden=32
    ).eval()
    with torch.no_grad():
        model.head[0].weight[:, 64:96].copy_(
            torch.linspace(-0.01, 0.01, 32 * 32).reshape(32, 32)
        )
    tokens = torch.randn(6, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 1, 0, 0, 1]] * 2,
        dtype=torch.float32,
    )
    context = torch.randn(6, 11)
    order = torch.tensor([4, 2, 0, 3, 1])
    changed = tokens.clone()
    changed[mask == 0] = 1e6
    with torch.no_grad():
        base = model(tokens, mask, context)
        permuted = model(tokens[:, order], mask[:, order], context)
        masked = model(changed, mask, context)
        repeated = model(tokens, mask, context)
    return {
        "permutation_maximum_abs_error": float(
            torch.max(torch.abs(base - permuted))
        ),
        "masked_or_future_token_maximum_abs_error": float(
            torch.max(torch.abs(base - masked))
        ),
        "repeat_maximum_abs_error": float(torch.max(torch.abs(base - repeated))),
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    fingerprint = json.loads(
        (
            ROOT / config["authorization_evidence"]["negative_fingerprint"]
        ).read_text(encoding="utf-8")
    )
    return {
        "classification": config["semantic_audit"]["classification"],
        "exact_execution_hits_before_preregistration": fingerprint[
            "exact_execution_hit_count"
        ],
        "semantic_execution_hits_before_preregistration": fingerprint[
            "semantic_execution_hit_count"
        ],
        "v15_attention_distinguished": True,
        "v16_depth_graph_distinguished": True,
        "v20_coral_alignment_distinguished": True,
        "v43_film_distinguished": True,
        "v45_v45c_dropconnect_distinguished": True,
        "v46_layernorm_distinguished": True,
        "v47_cross_network_distinguished": True,
        "v48_diagonal_second_moment_distinguished": True,
        "v49_logmeanexp_distinguished": True,
        "primary_source_url": config["primary_source"]["url"],
        "primary_source_p2_performance_transfer": False,
        "result_adaptive_search": False,
        "seed_trio_selection_or_ensemble": False,
        "router_or_retune": False,
        "official_feedback_used_for_selection": False,
    }


def _lineage_receipt(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source_contract"]
    return {
        "organizer_distributed_data_only": True,
        "only_source_filename": source["only_source_filename"],
        "only_direct_source_filename": source["only_direct_source_filename"],
        "derived_training_artifact_has_no_truth": not source[
            "derived_artifact_contains_hidden_truth"
        ],
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
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    source_guard = _synthetic_source_contract_guard_receipt(config)
    contract = _pooling_contract_receipt()
    isolation = _isolation_receipt()
    if (
        contract["descriptor"]
        != "masked_mean((element_embedding-masked_mean(element_embedding))^3)"
        or contract["moment_order"] != 3
        or contract["embedding_width"] != 32
        or contract["descriptor_width"] != 107
        or contract["new_head_columns"] != 32
        or contract["new_head_columns_initial_maximum_abs"] != 0.0
        or contract["parameters"] != 5889
        or contract["parameter_tensors"] != 10
        or contract["buffers"] != 0
        or contract["linear_count"] != 5
        or contract["unchanged_parameter_maximum_abs_error_vs_v13"] != 0.0
        or contract["initial_function_maximum_abs_error_vs_v13"] != 0.0
        or contract["descriptor_forward_maximum_abs_error"] != 0.0
        or contract["third_moment_manual_maximum_abs_error"] != 0.0
        or contract["all_missing_third_moment_maximum_abs"] != 0.0
        or contract["third_moment_nonconstant_range"] <= 0.0
        or contract["learned_pooling_columns_change_function_maximum_abs"] <= 0.0
        or contract["batch_composition_maximum_abs_error"] > 1e-6
        or not contract["gradients_finite"]
        or not contract["new_head_column_gradient_finite_nonzero"]
        or contract["normalization_count"] != 0
        or contract["dropout_count"] != 0
        or contract["attention_count"] != 0
        or max(isolation.values()) > 1e-6
    ):
        raise v12.ContractError("v50 target-free third-moment preflight failed")
    evidence = config["authorization_evidence"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_READY",
        "semantic_audit": semantic_audit(config),
        "lineage": _lineage_receipt(config),
        "prelock_source_contract_guard": source_guard,
        "masked_third_central_moment_pooling_contract": contract,
        "isolation": isolation,
        "historical_block_support": {
            "source": evidence["v13_result"],
            "training_rows_by_fold": [45935, 119667, 149384],
            "finite_pooled_profile_context_width": 107,
            "supported_folds": 3,
        },
        "prospective_fold_layer_gate": config["evaluation"]["safety_gate"],
        "prefix_cutoffs": {
            fold: (pd.Timestamp(start) - pd.Timedelta(days=7)).isoformat()
            for fold, start in config["training"]["fold_starts_kst"].items()
        },
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        **{
            f"{name}_sha256": evidence[f"{name}_sha256"]
            for name in EVIDENCE_NAMES
        },
        "data_rows_read": 0,
        "model_fits": 0,
        "scientific_artifacts_written": 0,
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


def _ready_preflight_pair(config: dict[str, Any]) -> dict[str, Any]:
    paths = [
        ROOT / value for value in config["ready_preflight_contract"]["paths"]
    ]
    if any(not path.is_file() for path in paths):
        raise v12.ContractError("v50 READY preflight pair is incomplete")
    first, second = (path.read_bytes() for path in paths)
    if first != second:
        raise v12.ContractError("v50 READY preflights are not byte-identical")
    stored = json.loads(first.decode("utf-8"))
    current = preflight()
    if stored != current or stored["status"] != "ZERO_OPERATION_PREFLIGHT_READY":
        raise v12.ContractError("v50 READY preflight does not match sealed code")
    return {
        "paths": [
            str(path.relative_to(ROOT)).replace("\\", "/") for path in paths
        ],
        "bytes_each": len(first),
        "byte_identical": True,
        "sha256_each": hashlib.sha256(first).hexdigest(),
        "payload_sha256": stored["preflight_sha256"],
    }


def _comparison_to_frozen_candidates(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    evidence = config["authorization_evidence"]
    comparisons: dict[str, Any] = {}
    for name in ("v45", "v45c", "v46", "v47", "v48"):
        source = json.loads(
            (ROOT / evidence[f"{name}_result"]).read_text(encoding="utf-8")
        )["candidate"]
        comparisons[name] = {
            "source_result_sha256": evidence[f"{name}_result_sha256"],
            "source_delta_rmse_C": float(source["delta_rmse"]),
            "v50_delta_rmse_C": float(record["delta_rmse"]),
            "v50_minus_source_delta_rmse_C": float(
                record["delta_rmse"] - source["delta_rmse"]
            ),
            "source_canonical_transport_adjusted_points": float(
                source["canonical_transport_adjusted_pooled_points_delta"]
            ),
            "v50_canonical_transport_adjusted_points": float(
                record["canonical_transport_adjusted_pooled_points_delta"]
            ),
        }
    return {
        "use": "post_terminal_ledger_only_no_selection_router_retune_or_ensemble",
        "comparisons": comparisons,
        "source_results_not_used_to_choose_moment_or_gate": True,
    }


def write_report(result: dict[str, Any]) -> None:
    if "comparison_to_frozen_candidates" not in result:
        return
    record = result["candidate"]
    local = record["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v50 masked third-central-moment profile-pooling DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{record['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{record['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{record['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"prospective fold x layer gate `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13 masked mean/max summary에 masked signed third central "
        "moment 32개를 추가했다. 새 head columns는 zero-init되어 initial "
        "function이 v13과 같다. 두 READY preflight는 byte-identical이고 "
        "상속 source-contract 5개 key의 synthetic missing-key rejection을 "
        "lock 전에 검증했다. 배포 observations.csv 및 그 truth-free 파생 "
        "training frame 외 자료, pretrained weights, official/test/sample/"
        "baseline/query/hidden/CSV/upload는 사용하지 않았다.\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    config = load_config()
    source_guard = _synthetic_source_contract_guard_receipt(config)
    ready_pair = _ready_preflight_pair(config)
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
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if passed
        else "EXPLORATORY_NO_GO_MASKED_THIRD_CENTRAL_MOMENT_PROFILE_POOLING"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["lineage"] = _lineage_receipt(config)
    result["prelock_source_contract_guard"] = source_guard
    result["ready_preflight_pair"] = ready_pair
    result["masked_third_central_moment_pooling_contract"] = (
        _pooling_contract_receipt()
    )
    result["isolation"] = _isolation_receipt()
    result["comparison_to_frozen_candidates"] = _comparison_to_frozen_candidates(
        record, config
    )
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer": config["training"]["optimizer"],
            "initialization": config["training"]["initialization"],
            "pretrained_weight_files_loaded": 0,
            "masked_third_central_moment_pooling": config["training"][
                "masked_third_central_moment_pooling"
            ],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
            "extra_loss": 0,
            "normalization": 0,
            "dropout": 0,
        }
    )
    result["operation_counters"].update(
        {
            "external_observation_rows_read": 0,
            "external_reanalysis_rows_read": 0,
            "external_forecast_rows_read": 0,
            "pretrained_weight_files_loaded": 0,
        }
    )
    evidence = config["authorization_evidence"]
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    result["hashes"]["ready_preflight_receipt"] = ready_pair["sha256_each"]
    for name in EVIDENCE_NAMES:
        result["hashes"][name] = evidence[f"{name}_sha256"]
    v12.atomic_json(ARTIFACT / "result.json", result)
    v12.atomic_json(REPORT / "result.json", result)
    write_report(result)
    return result


def _write_preflight_receipt(path_text: str, value: dict[str, Any]) -> None:
    config = load_config()
    allowed = {
        (ROOT / item).resolve()
        for item in config["ready_preflight_contract"]["paths"]
    }
    path = (ROOT / path_text).resolve()
    if path not in allowed:
        raise v12.ContractError("v50 preflight receipt path is not sealed")
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
        _write_preflight_receipt(args.receipt, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
