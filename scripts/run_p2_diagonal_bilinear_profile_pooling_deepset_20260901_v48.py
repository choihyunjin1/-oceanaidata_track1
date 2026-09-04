"""Run sealed P2 v48 diagonal-bilinear profile-pooling DeepSets once."""

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

EXPERIMENT_ID = "p2_diagonal_bilinear_profile_pooling_deepset_20260901_v48"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V48_DIAGONAL_BILINEAR_PROFILE_POOLING_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.diagonal_bilinear_profile_pooling_deepset.result.20260901.v48"

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
    "v20_result",
    "v43_result",
    "v45_result",
    "v45c_result",
    "v46_result",
    "v47_result",
    "prospective_gate_amendment",
)


class DiagonalBilinearProfileVerticalDeepSet(_BASE_MODEL):
    """Exact v13 with an identity-initialized diagonal bilinear descriptor."""

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
        diagonal_second_moment = (encoded.square() * mask).sum(dim=1) / count
        return torch.cat(
            (mean, maximum, diagonal_second_moment, context), dim=1
        )

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        descriptor = self.pooled_profile_context(tokens, token_mask, context)
        return self.head(descriptor).squeeze(1)


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
    v12.VerticalDeepSet = DiagonalBilinearProfileVerticalDeepSet


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = config["authorization_evidence"]
    for name in EVIDENCE_NAMES:
        path = ROOT / evidence[name]
        if not path.is_file() or v12.sha256_file(path) != evidence[f"{name}_sha256"]:
            raise v12.ContractError(f"v48 evidence drift: {name}")
    fingerprint = json.loads(
        (ROOT / evidence["negative_fingerprint"]).read_text(encoding="utf-8")
    )
    training = config["training"]
    bilinear = training["diagonal_bilinear_pooling"]
    gate = config["evaluation"]["safety_gate"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or fingerprint["candidate_experiment_id"] != EXPERIMENT_ID
        or fingerprint["exact_execution_hit_count"] != 0
        or fingerprint["semantic_execution_hit_count"] != 0
        or training["architecture"]
        != "v13_exact_five_Linear_DeepSets_plus_masked_diagonal_bilinear_profile_pooling"
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
        or bilinear["input_embedding_width"] != 32
        or bilinear["descriptor"]
        != "masked_mean(element_embedding_squared)"
        or bilinear["full_head_input_width"] != 107
        or bilinear["new_head_columns_initialization"] != 0.0
        or bilinear["added_parameters"] != 1024
        or bilinear["expected_parameters"] != 5889
        or bilinear["expected_parameter_tensors"] != 10
        or bilinear["expected_buffers"] != 0
        or bilinear["normalization"]
        or bilinear["coefficient_or_rank_sweep"]
        or gate["minimum_fold_layer_non_harm_cells"] != 8
        or gate["total_fold_layer_cells"] != 9
        or gate["maximum_any_fold_layer_delta_rmse_C"] != 0.003
    ):
        raise v12.ContractError("v48 scientific contract drift")
    return config


def _unchanged_state_error(base: nn.Module, candidate: nn.Module) -> float:
    errors = [
        torch.max(torch.abs(base.element.state_dict()[name] - value))
        for name, value in candidate.element.state_dict().items()
    ]
    for index in (2, 4):
        for name, value in candidate.head[index].state_dict().items():
            errors.append(
                torch.max(torch.abs(base.head[index].state_dict()[name] - value))
            )
    errors.extend(
        (
            torch.max(
                torch.abs(
                    candidate.head[0].weight[:, :64] - base.head[0].weight[:, :64]
                )
            ),
            torch.max(
                torch.abs(
                    candidate.head[0].weight[:, 96:] - base.head[0].weight[:, 64:]
                )
            ),
            torch.max(torch.abs(candidate.head[0].bias - base.head[0].bias)),
        )
    )
    return float(torch.max(torch.stack(errors)).detach())


def _bilinear_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(4801)
    base = _BASE_MODEL(8, 11, hidden=32).eval()
    torch.manual_seed(4801)
    candidate = DiagonalBilinearProfileVerticalDeepSet(8, 11, hidden=32).eval()
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
        expanded_mask = mask.unsqueeze(-1)
        count = expanded_mask.sum(dim=1).clamp_min(1.0)
        manual_second = (encoded.square() * expanded_mask).sum(dim=1) / count
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
        "descriptor": "masked_mean(element_embedding_squared)",
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
        "diagonal_second_moment_maximum_abs_error": float(
            torch.max(torch.abs(descriptor[:, 64:96] - manual_second))
        ),
        "all_missing_second_moment_maximum_abs": float(
            torch.max(torch.abs(descriptor[-1, 64:96]))
        ),
        "learned_bilinear_columns_change_function_maximum_abs": float(
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
            isinstance(module, nn.MultiheadAttention) for module in candidate.modules()
        ),
    }


def _isolation_receipt() -> dict[str, float]:
    torch.manual_seed(4802)
    model = DiagonalBilinearProfileVerticalDeepSet(8, 11, hidden=32).eval()
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
        "v20_coral_alignment_distinguished": True,
        "v43_film_distinguished": True,
        "v45_v45c_dropconnect_distinguished": True,
        "v46_layernorm_distinguished": True,
        "v47_cross_network_distinguished": True,
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
    contract = _bilinear_contract_receipt()
    isolation = _isolation_receipt()
    if (
        contract["descriptor"] != "masked_mean(element_embedding_squared)"
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
        or contract["diagonal_second_moment_maximum_abs_error"] != 0.0
        or contract["all_missing_second_moment_maximum_abs"] != 0.0
        or contract["learned_bilinear_columns_change_function_maximum_abs"]
        <= 0.0
        or contract["batch_composition_maximum_abs_error"] > 1e-6
        or not contract["gradients_finite"]
        or not contract["new_head_column_gradient_finite_nonzero"]
        or contract["normalization_count"] != 0
        or contract["dropout_count"] != 0
        or contract["attention_count"] != 0
        or max(isolation.values()) > 1e-6
    ):
        raise v12.ContractError("v48 target-free bilinear preflight failed")
    evidence = config["authorization_evidence"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_READY",
        "semantic_audit": semantic_audit(config),
        "diagonal_bilinear_pooling_contract": contract,
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
    for name in ("v45", "v45c", "v46", "v47"):
        source = json.loads(
            (ROOT / evidence[f"{name}_result"]).read_text(encoding="utf-8")
        )["candidate"]
        comparisons[name] = {
            "source_result_sha256": evidence[f"{name}_result_sha256"],
            "source_delta_rmse_C": float(source["delta_rmse"]),
            "v48_delta_rmse_C": float(record["delta_rmse"]),
            "v48_minus_source_delta_rmse_C": float(
                record["delta_rmse"] - source["delta_rmse"]
            ),
            "source_canonical_transport_adjusted_points": float(
                source["canonical_transport_adjusted_pooled_points_delta"]
            ),
            "v48_canonical_transport_adjusted_points": float(
                record["canonical_transport_adjusted_pooled_points_delta"]
            ),
        }
    return {
        "use": "post_terminal_ledger_only_no_selection_router_retune_or_ensemble",
        "comparisons": comparisons,
        "v45_original_commitment_preserved": True,
        "v45c_confirmation_not_used_as_seed_selection": True,
        "v46_no_go_preserved": True,
        "v47_no_go_preserved": True,
    }


def write_report(result: dict[str, Any]) -> None:
    if "comparison_to_frozen_candidates" not in result:
        return
    record = result["candidate"]
    local = record["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v48 diagonal-bilinear profile-pooling DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{record['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{record['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{record['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"prospective fold x layer gate `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13의 masked mean/max summary에 token embedding의 masked diagonal "
        "second moment 32개를 추가했다. 새 head columns는 zero-init되어 initial "
        "function이 v13과 같다. v45/v45c/v46/v47 비교는 terminal 후 ledger "
        "진단만 수행했다. selection, "
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
        else "EXPLORATORY_NO_GO_DIAGONAL_BILINEAR_PROFILE_POOLING"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["diagonal_bilinear_pooling_contract"] = _bilinear_contract_receipt()
    result["isolation"] = _isolation_receipt()
    result["comparison_to_frozen_candidates"] = _comparison_to_frozen_candidates(
        record, config
    )
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer": config["training"]["optimizer"],
            "diagonal_bilinear_pooling": config["training"][
                "diagonal_bilinear_pooling"
            ],
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
