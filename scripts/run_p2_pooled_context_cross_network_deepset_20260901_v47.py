"""Run sealed P2 v47 pooled-profile/context CrossNet DeepSets exactly once."""

from __future__ import annotations

import argparse
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

import run_p2_hidden_activation_layer_normalized_deepset_20260901_v46 as v46  # noqa: E402

v37 = v46.v37
v13 = v46.v13
v12 = v46.v12

EXPERIMENT_ID = "p2_pooled_context_cross_network_deepset_20260901_v47"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V47_POOLED_CONTEXT_CROSS_NETWORK_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.pooled_context_cross_network_deepset.result.20260901.v47"

_BASE_RUN = v46._BASE_RUN
_BASE_MODEL = v12.VerticalDeepSet
_BASE_TRAIN_PREDICT_SEED = v13.train_predict_seed
_BASE_DOMAIN_BALANCED_WEIGHTS = v46._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = Path(v13.__file__)
EVIDENCE_NAMES = (
    "negative_fingerprint",
    "v13_result",
    "v15_result",
    "v16_result",
    "v43_result",
    "v45_result",
    "v45c_result",
    "v46_result",
    "prospective_gate_amendment",
)


class PooledContextCrossVerticalDeepSet(_BASE_MODEL):
    """Exact v13 encoder/head with one identity-initialized DCN cross layer."""

    def __init__(
        self, token_features: int, context_features: int, hidden: int = 32
    ) -> None:
        super().__init__(token_features, context_features, hidden)
        width = hidden * 2 + context_features
        self.cross_weight = nn.Parameter(torch.zeros(width))
        self.cross_bias = nn.Parameter(torch.zeros(width))

    def pooled_context(
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
        return torch.cat((mean, maximum, context), dim=1)

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        x0 = self.pooled_context(tokens, token_mask, context)
        scalar = (x0 * self.cross_weight).sum(dim=1, keepdim=True)
        crossed = x0 * scalar + self.cross_bias + x0
        return self.head(crossed).squeeze(1)


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
    v12.VerticalDeepSet = PooledContextCrossVerticalDeepSet


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = config["authorization_evidence"]
    for name in EVIDENCE_NAMES:
        path = ROOT / evidence[name]
        if not path.is_file() or v12.sha256_file(path) != evidence[f"{name}_sha256"]:
            raise v12.ContractError(f"v47 evidence drift: {name}")
    fingerprint = json.loads(
        (ROOT / evidence["negative_fingerprint"]).read_text(encoding="utf-8")
    )
    training = config["training"]
    cross = training["cross_network"]
    gate = config["evaluation"]["safety_gate"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or fingerprint["candidate_experiment_id"] != EXPERIMENT_ID
        or fingerprint["exact_execution_hit_count"] != 0
        or fingerprint["semantic_execution_hit_count"] != 0
        or training["architecture"]
        != "v13_exact_five_Linear_DeepSets_plus_one_identity_initialized_pooled_profile_context_cross_layer"
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
        or cross["input_width"] != 75
        or cross["layer_count"] != 1
        or cross["weight_initialization"] != 0.0
        or cross["bias_initialization"] != 0.0
        or cross["added_parameters"] != 150
        or cross["expected_parameters"] != 5015
        or cross["expected_parameter_tensors"] != 12
        or cross["expected_buffers"] != 0
        or cross["coefficient_or_depth_sweep"]
        or gate["minimum_fold_layer_non_harm_cells"] != 8
        or gate["total_fold_layer_cells"] != 9
        or gate["maximum_any_fold_layer_delta_rmse_C"] != 0.003
    ):
        raise v12.ContractError("v47 scientific contract drift")
    return config


def _linear_state_error(
    base: nn.Module, candidate: nn.Module
) -> float:
    base_state = base.state_dict()
    candidate_state = candidate.state_dict()
    names = [name for name in base_state if name.startswith(("element.", "head."))]
    return max(
        float(torch.max(torch.abs(base_state[name] - candidate_state[name])))
        for name in names
    )


def _cross_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(4701)
    base = _BASE_MODEL(8, 11, hidden=32).eval()
    torch.manual_seed(4701)
    candidate = PooledContextCrossVerticalDeepSet(8, 11, hidden=32).eval()
    linear_error = _linear_state_error(base, candidate)
    initial_cross_weight_max = float(
        torch.max(torch.abs(candidate.cross_weight.detach()))
    )
    initial_cross_bias_max = float(torch.max(torch.abs(candidate.cross_bias.detach())))
    tokens = torch.randn(7, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 1, 0, 0, 1]]
        + [[1, 1, 1, 0, 0]] * 4,
        dtype=torch.float32,
    )
    context = torch.randn(7, 11)
    with torch.no_grad():
        base_output = base(tokens, mask, context)
        initial_output = candidate(tokens, mask, context)
        pooled = candidate.pooled_context(tokens, mask, context)
        manual = candidate.head(pooled).squeeze(1)
        batch_first = candidate(tokens[:1], mask[:1], context[:1])
        candidate.cross_weight.copy_(torch.linspace(-0.02, 0.02, 75))
        candidate.cross_bias.copy_(torch.linspace(0.01, -0.01, 75))
        crossed_output = candidate(tokens, mask, context)
        x0 = candidate.pooled_context(tokens, mask, context)
        scalar = (x0 * candidate.cross_weight).sum(dim=1, keepdim=True)
        equation_output = candidate.head(
            x0 * scalar + candidate.cross_bias + x0
        ).squeeze(1)
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    loss = candidate(tokens, mask, context).square().mean()
    loss.backward()
    gradients = [value.grad for value in candidate.parameters()]
    return {
        "cross_layer_count": 1,
        "cross_input_width": int(candidate.cross_weight.numel()),
        "cross_weight_initial_maximum_abs": initial_cross_weight_max,
        "cross_bias_initial_maximum_abs": initial_cross_bias_max,
        "parameters": int(sum(value.numel() for value in candidate.parameters())),
        "parameter_tensors": len(list(candidate.parameters())),
        "buffers": len(list(candidate.buffers())),
        "linear_count": sum(isinstance(module, nn.Linear) for module in candidate.modules()),
        "linear_parameter_maximum_abs_error_vs_v13": linear_error,
        "initial_function_maximum_abs_error_vs_v13": float(
            torch.max(torch.abs(base_output - initial_output))
        ),
        "identity_equation_maximum_abs_error": float(
            torch.max(torch.abs(initial_output - manual))
        ),
        "learned_cross_changes_function_maximum_abs": float(
            torch.max(torch.abs(crossed_output - initial_output))
        ),
        "nonzero_equation_maximum_abs_error": float(
            torch.max(torch.abs(crossed_output - equation_output))
        ),
        "batch_composition_maximum_abs_error": float(
            torch.max(torch.abs(initial_output[:1] - batch_first))
        ),
        "gradients_finite": all(
            value is not None and bool(torch.isfinite(value).all())
            for value in gradients
        ),
        "cross_weight_gradient_finite_nonzero": bool(
            candidate.cross_weight.grad is not None
            and torch.isfinite(candidate.cross_weight.grad).all()
            and torch.linalg.vector_norm(candidate.cross_weight.grad) > 0
        ),
        "cross_bias_gradient_finite_nonzero": bool(
            candidate.cross_bias.grad is not None
            and torch.isfinite(candidate.cross_bias.grad).all()
            and torch.linalg.vector_norm(candidate.cross_bias.grad) > 0
        ),
        "normalization_count": sum(
            isinstance(module, (nn.LayerNorm, nn.BatchNorm1d))
            for module in candidate.modules()
        ),
        "dropout_count": sum(isinstance(module, nn.Dropout) for module in candidate.modules()),
        "attention_count": sum(
            isinstance(module, nn.MultiheadAttention) for module in candidate.modules()
        ),
    }


def _isolation_receipt() -> dict[str, float]:
    torch.manual_seed(4702)
    model = PooledContextCrossVerticalDeepSet(8, 11, hidden=32).eval()
    with torch.no_grad():
        model.cross_weight.copy_(torch.linspace(-0.015, 0.015, 75))
        model.cross_bias.copy_(torch.linspace(0.005, -0.005, 75))
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
        "permutation_maximum_abs_error": float(torch.max(torch.abs(base - permuted))),
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
        "v43_film_distinguished": True,
        "v45_v45c_dropconnect_distinguished": True,
        "v46_layernorm_distinguished": True,
        "primary_source_url": config["primary_source"]["url"],
        "primary_source_p2_performance_transfer": False,
        "result_adaptive_search": False,
        "seed_trio_selection_or_ensemble": False,
        "router_or_retune": False,
        "official_v23_feedback_used_for_selection": False,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    contract = _cross_contract_receipt()
    isolation = _isolation_receipt()
    if (
        contract["cross_layer_count"] != 1
        or contract["cross_input_width"] != 75
        or contract["cross_weight_initial_maximum_abs"] != 0.0
        or contract["cross_bias_initial_maximum_abs"] != 0.0
        or contract["parameters"] != 5015
        or contract["parameter_tensors"] != 12
        or contract["buffers"] != 0
        or contract["linear_count"] != 5
        or contract["linear_parameter_maximum_abs_error_vs_v13"] != 0.0
        or contract["initial_function_maximum_abs_error_vs_v13"] != 0.0
        or contract["identity_equation_maximum_abs_error"] != 0.0
        or contract["learned_cross_changes_function_maximum_abs"] <= 0.0
        or contract["nonzero_equation_maximum_abs_error"] > 1e-6
        or contract["batch_composition_maximum_abs_error"] > 1e-6
        or not contract["gradients_finite"]
        or not contract["cross_weight_gradient_finite_nonzero"]
        or not contract["cross_bias_gradient_finite_nonzero"]
        or contract["normalization_count"] != 0
        or contract["dropout_count"] != 0
        or contract["attention_count"] != 0
        or max(isolation.values()) > 1e-6
    ):
        raise v12.ContractError("v47 target-free CrossNet preflight failed")
    evidence = config["authorization_evidence"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_READY",
        "semantic_audit": semantic_audit(config),
        "cross_network_contract": contract,
        "isolation": isolation,
        "historical_block_support": {
            "source": evidence["v13_result"],
            "training_rows_by_fold": [45935, 119667, 149384],
            "finite_pooled_context_width": 75,
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
        "artifacts_written": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = v12.sha256_json(payload)
    return payload


def _comparison_to_frozen_candidates(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    evidence = config["authorization_evidence"]
    comparisons: dict[str, Any] = {}
    for name in ("v45", "v45c", "v46"):
        source = json.loads(
            (ROOT / evidence[f"{name}_result"]).read_text(encoding="utf-8")
        )["candidate"]
        comparisons[name] = {
            "source_result_sha256": evidence[f"{name}_result_sha256"],
            "source_delta_rmse_C": float(source["delta_rmse"]),
            "v47_delta_rmse_C": float(record["delta_rmse"]),
            "v47_minus_source_delta_rmse_C": float(
                record["delta_rmse"] - source["delta_rmse"]
            ),
            "source_canonical_transport_adjusted_points": float(
                source["canonical_transport_adjusted_pooled_points_delta"]
            ),
            "v47_canonical_transport_adjusted_points": float(
                record["canonical_transport_adjusted_pooled_points_delta"]
            ),
        }
    return {
        "use": "post_terminal_ledger_only_no_selection_router_retune_or_ensemble",
        "comparisons": comparisons,
        "v45_original_commitment_preserved": True,
        "v45c_confirmation_not_used_as_seed_selection": True,
        "v46_no_go_preserved": True,
    }


def write_report(result: dict[str, Any]) -> None:
    if "comparison_to_frozen_candidates" not in result:
        return
    record = result["candidate"]
    local = record["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v47 pooled-profile/context CrossNet DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{record['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{record['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{record['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"prospective fold x layer gate `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13의 masked mean/max public-profile summary와 11 context를 합친 "
        "75차원 벡터에 identity-initialized one-layer DCN feature cross만 추가했다. "
        "v45/v45c/v46 비교는 terminal 후 ledger 진단만 수행했다. selection, "
        "retune, router, ensemble, official/test/sample/hidden/query/CSV/upload는 0이다.\n",
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
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if passed
        else "EXPLORATORY_NO_GO_POOLED_CONTEXT_CROSS_NETWORK"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["cross_network_contract"] = _cross_contract_receipt()
    result["isolation"] = _isolation_receipt()
    result["comparison_to_frozen_candidates"] = _comparison_to_frozen_candidates(
        record, config
    )
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer": config["training"]["optimizer"],
            "cross_network": config["training"]["cross_network"],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
            "extra_loss": 0,
            "normalization": 0,
            "dropout": 0,
        }
    )
    evidence = config["authorization_evidence"]
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    for name in EVIDENCE_NAMES:
        result["hashes"][name] = evidence[f"{name}_sha256"]
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
