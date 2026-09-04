"""Run sealed P2 v46 hidden-activation LayerNorm DeepSets exactly once."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_all_linear_dropconnect_deepset_20260901_v45 as v45  # noqa: E402

v37 = v45.v37
v13 = v45.v13
v12 = v45.v12

EXPERIMENT_ID = "p2_hidden_activation_layer_normalized_deepset_20260901_v46"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V46_HIDDEN_ACTIVATION_LAYERNORM_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.hidden_activation_layer_normalized_deepset.result.20260901.v46"

_BASE_RUN = v45._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v45._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v45._V13_RUNNER


class LayerNormalizedVerticalDeepSet(v12.VerticalDeepSet):
    """Exact v13 Linear geometry with four pre-ReLU hidden LayerNorm modules."""

    def __init__(
        self,
        token_features: int,
        context_features: int,
        hidden: int = 32,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(token_features, context_features, hidden)
        element_linear_1 = self.element[0]
        element_relu_1 = self.element[1]
        element_linear_2 = self.element[2]
        element_relu_2 = self.element[3]
        head_linear_1 = self.head[0]
        head_relu_1 = self.head[1]
        head_linear_2 = self.head[2]
        head_relu_2 = self.head[3]
        head_output = self.head[4]
        self.element = nn.Sequential(
            element_linear_1,
            nn.LayerNorm(hidden, eps=eps, elementwise_affine=True),
            element_relu_1,
            element_linear_2,
            nn.LayerNorm(hidden, eps=eps, elementwise_affine=True),
            element_relu_2,
        )
        self.head = nn.Sequential(
            head_linear_1,
            nn.LayerNorm(hidden, eps=eps, elementwise_affine=True),
            head_relu_1,
            head_linear_2,
            nn.LayerNorm(hidden, eps=eps, elementwise_affine=True),
            head_relu_2,
            head_output,
        )


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


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = config["authorization_evidence"]
    evidence_names = (
        "negative_fingerprint",
        "design_fingerprint",
        "execution_decision",
        "v13_result",
        "v45_result",
        "v45c_result",
        "v45c_independent_qa",
        "prospective_gate_amendment",
    )
    for name in evidence_names:
        path = ROOT / evidence[name]
        if not path.is_file() or v12.sha256_file(path) != evidence[f"{name}_sha256"]:
            raise v12.ContractError(f"v46 evidence drift: {name}")
    fingerprint = json.loads(
        (ROOT / evidence["negative_fingerprint"]).read_text(encoding="utf-8")
    )
    v45c_result = json.loads(
        (ROOT / evidence["v45c_result"]).read_text(encoding="utf-8")
    )
    v45c_qa = json.loads(
        (ROOT / evidence["v45c_independent_qa"]).read_text(encoding="utf-8")
    )
    training = config["training"]
    layernorm = training["layer_normalization"]
    safety = config["evaluation"]["safety_gate"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or config["claim_level"] != "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION"
        or fingerprint["status"] != "DISTINCT_READY_FOR_PREREGISTRATION"
        or fingerprint["exact_execution_hit_count"] != 0
        or v45c_result["status"] != "STOCHASTIC_CONFIRMATION_PASS_EXPOSED_BLOCKS_ONLY"
        or v45c_qa["status"] != "PASS"
        or training["architecture"]
        != "v13_exact_five_Linear_DeepSets_plus_four_hidden_pre_ReLU_LayerNorm32"
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
        or training["row_deletion"]
        or training["input_perturbation"]
        or training["data_augmentation"]
        or training["extra_loss"]
        or layernorm["module"] != "torch.nn.LayerNorm"
        or layernorm["count"] != 4
        or layernorm["normalized_shape"] != 32
        or layernorm["eps"] != 1e-5
        or not layernorm["elementwise_affine"]
        or layernorm["placement"]
        != "after_each_of_four_hidden_Linear_layers_before_ReLU"
        or layernorm["initial_weight"] != 1.0
        or layernorm["initial_bias"] != 0.0
        or layernorm["batch_statistics"]
        or layernorm["running_statistics"]
        or layernorm["expected_parameters"] != 5121
        or layernorm["expected_parameter_tensors"] != 18
        or layernorm["expected_buffers"] != 0
        or layernorm["placement_or_epsilon_sweep"]
        or safety["minimum_fold_layer_non_harm_cells"] != 8
        or safety["total_fold_layer_cells"] != 9
        or safety["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or config["operation_limits"]["automatic_retry_count"] != 0
    ):
        raise v12.ContractError("v46 fixed scientific contract drift")
    return config


def train_predict_seed(
    tokens: np.ndarray,
    mask: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    query_tokens: np.ndarray,
    query_mask: np.ndarray,
    query_context: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = float(config["training"]["layer_normalization"]["eps"])
    model = LayerNormalizedVerticalDeepSet(8, 11, hidden=32, eps=eps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    train = tuple(
        torch.from_numpy(np.asarray(value).copy())
        for value in (
            tokens,
            mask,
            context,
            target.astype(np.float32),
            weights.astype(np.float32),
        )
    )
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        numerator = 0.0
        denominator = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            loss = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v46 training loss is non-finite")
            loss.backward()
            optimizer.step()
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
        losses.append(numerator / denominator)
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_tokens), batch_size):
            stop = start + batch_size
            output.append(
                model(
                    torch.from_numpy(np.asarray(query_tokens[start:stop]).copy()).to(
                        device
                    ),
                    torch.from_numpy(np.asarray(query_mask[start:stop]).copy()).to(
                        device
                    ),
                    torch.from_numpy(np.asarray(query_context[start:stop]).copy()).to(
                        device
                    ),
                )
                .cpu()
                .numpy()
            )
    prediction = np.concatenate(output).astype(float)
    finite = bool(np.isfinite(losses).all() and np.isfinite(prediction).all())
    layernorms = [
        module for module in model.modules() if isinstance(module, nn.LayerNorm)
    ]
    if not finite or len(layernorms) != 4:
        raise v12.ContractError("v46 LayerNorm training contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "parameter_tensors": len(list(model.parameters())),
        "buffers": len(list(model.buffers())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "layernorm_count": len(layernorms),
        "layernorm_names": [
            name for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)
        ],
        "layernorm_eps": [float(module.eps) for module in layernorms],
        "layernorm_affine": [bool(module.elementwise_affine) for module in layernorms],
        "batchnorm_count": sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in model.modules()
        ),
        "dropout_count": sum(isinstance(module, nn.Dropout) for module in model.modules()),
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _linear_weight_error() -> float:
    torch.manual_seed(46)
    base = v12.VerticalDeepSet(8, 11, hidden=32)
    torch.manual_seed(46)
    candidate = LayerNormalizedVerticalDeepSet(8, 11, hidden=32, eps=1e-5)
    pairs = (
        (base.element[0], candidate.element[0]),
        (base.element[2], candidate.element[3]),
        (base.head[0], candidate.head[0]),
        (base.head[2], candidate.head[3]),
        (base.head[4], candidate.head[6]),
    )
    errors = []
    for left, right in pairs:
        errors.append(float(torch.max(torch.abs(left.weight - right.weight)).detach()))
        errors.append(float(torch.max(torch.abs(left.bias - right.bias)).detach()))
    return max(errors)


def _layernorm_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(46)
    base = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    torch.manual_seed(46)
    candidate = LayerNormalizedVerticalDeepSet(8, 11, hidden=32, eps=1e-5).eval()
    layernorms = [
        (name, module)
        for name, module in candidate.named_modules()
        if isinstance(module, nn.LayerNorm)
    ]
    tokens = torch.randn(16, 5, 8)
    mask = torch.ones(16, 5)
    context = torch.randn(16, 11)
    with torch.inference_mode():
        base_output = base(tokens, mask, context)
        candidate_output = candidate(tokens, mask, context)
        repeated = candidate(tokens, mask, context)
        single = candidate(tokens[:1], mask[:1], context[:1])
        composed = candidate(tokens, mask, context)[:1]
    sample = torch.randn(64, 32)
    normalized = layernorms[0][1](sample)
    normalization_mean_error = float(torch.max(torch.abs(normalized.mean(dim=-1))).detach())
    normalization_variance_error = float(
        torch.max(torch.abs(normalized.var(dim=-1, unbiased=False) - 1.0)).detach()
    )
    loss = candidate(tokens, mask, context).square().mean()
    loss.backward()
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in candidate.parameters()
    )
    return {
        "layernorm_count": len(layernorms),
        "layernorm_names": [name for name, _ in layernorms],
        "normalized_shapes": [list(module.normalized_shape) for _, module in layernorms],
        "eps": [float(module.eps) for _, module in layernorms],
        "elementwise_affine": [
            bool(module.elementwise_affine) for _, module in layernorms
        ],
        "initial_weight_errors": [
            float(torch.max(torch.abs(module.weight - 1.0)).detach())
            for _, module in layernorms
        ],
        "initial_bias_errors": [
            float(torch.max(torch.abs(module.bias)).detach()) for _, module in layernorms
        ],
        "parameters": int(sum(value.numel() for value in candidate.parameters())),
        "parameter_tensors": len(list(candidate.parameters())),
        "buffers": len(list(candidate.buffers())),
        "linear_parameter_maximum_abs_error_vs_v13": _linear_weight_error(),
        "initial_function_maximum_abs_difference_vs_v13": float(
            torch.max(torch.abs(candidate_output - base_output)).detach()
        ),
        "repeat_maximum_abs_error": float(
            torch.max(torch.abs(candidate_output - repeated)).detach()
        ),
        "batch_composition_maximum_abs_error": float(
            torch.max(torch.abs(single - composed)).detach()
        ),
        "normalization_mean_maximum_abs_error": normalization_mean_error,
        "normalization_variance_maximum_abs_error": normalization_variance_error,
        "gradients_finite": gradients_finite,
        "batchnorm_count": sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in candidate.modules()
        ),
        "dropout_count": sum(
            isinstance(module, nn.Dropout) for module in candidate.modules()
        ),
    }


def _isolation_receipt() -> dict[str, float]:
    torch.manual_seed(64)
    model = LayerNormalizedVerticalDeepSet(8, 11, hidden=32, eps=1e-5).eval()
    tokens = torch.randn(4, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    context = torch.randn(4, 11)
    changed = tokens.clone()
    changed[mask == 0] = 1e6
    order = torch.tensor([4, 2, 0, 3, 1])
    with torch.inference_mode():
        base = model(tokens, mask, context)
        masked = model(changed, mask, context)
        permuted = model(tokens[:, order], mask[:, order], context)
        repeated = model(tokens, mask, context)
    return {
        "masked_or_future_token_maximum_abs_error": float(
            torch.max(torch.abs(base - masked))
        ),
        "permutation_maximum_abs_error": float(torch.max(torch.abs(base - permuted))),
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
        "v20_coral_distinguished": True,
        "v27_spectral_norm_distinguished": True,
        "v37_cmd_distinguished": True,
        "v40_dropout_consistency_distinguished": True,
        "v41_weight_norm_distinguished": True,
        "v45_v45c_dropconnect_distinguished": True,
        "primary_source_url": config["primary_source"]["url"],
        "primary_source_p2_performance_transfer": False,
        "result_adaptive_search": False,
        "seed_trio_selection_or_ensemble": False,
        "official_v23_feedback_used_for_selection": False,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    contract = _layernorm_contract_receipt()
    isolation = _isolation_receipt()
    expected_names = ["element.1", "element.4", "head.1", "head.4"]
    if (
        contract["layernorm_count"] != 4
        or contract["layernorm_names"] != expected_names
        or contract["normalized_shapes"] != [[32], [32], [32], [32]]
        or contract["eps"] != [1e-5, 1e-5, 1e-5, 1e-5]
        or not all(contract["elementwise_affine"])
        or max(contract["initial_weight_errors"]) != 0.0
        or max(contract["initial_bias_errors"]) != 0.0
        or contract["parameters"] != 5121
        or contract["parameter_tensors"] != 18
        or contract["buffers"] != 0
        or contract["linear_parameter_maximum_abs_error_vs_v13"] != 0.0
        or contract["initial_function_maximum_abs_difference_vs_v13"] <= 0.0
        or contract["repeat_maximum_abs_error"] != 0.0
        or contract["batch_composition_maximum_abs_error"] > 1e-6
        or contract["normalization_mean_maximum_abs_error"] > 1e-5
        or contract["normalization_variance_maximum_abs_error"] > 1e-3
        or not contract["gradients_finite"]
        or contract["batchnorm_count"] != 0
        or contract["dropout_count"] != 0
        or max(isolation.values()) > 1e-6
    ):
        raise v12.ContractError("v46 target-free LayerNorm preflight failed")
    evidence = config["authorization_evidence"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_READY",
        "semantic_audit": semantic_audit(config),
        "layernorm_contract": contract,
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
        **{
            f"{name}_sha256": evidence[f"{name}_sha256"]
            for name in (
                "negative_fingerprint",
                "design_fingerprint",
                "execution_decision",
                "v13_result",
                "v45_result",
                "v45c_result",
                "v45c_independent_qa",
                "prospective_gate_amendment",
            )
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


def _comparison_to_dropconnect(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    evidence = config["authorization_evidence"]
    comparisons: dict[str, Any] = {}
    for name in ("v45", "v45c"):
        source = json.loads(
            (ROOT / evidence[f"{name}_result"]).read_text(encoding="utf-8")
        )["candidate"]
        comparisons[name] = {
            "source_result_sha256": evidence[f"{name}_result_sha256"],
            "source_delta_rmse_C": float(source["delta_rmse"]),
            "v46_delta_rmse_C": float(record["delta_rmse"]),
            "v46_minus_source_delta_rmse_C": float(
                record["delta_rmse"] - source["delta_rmse"]
            ),
            "source_canonical_transport_adjusted_points": float(
                source["canonical_transport_adjusted_pooled_points_delta"]
            ),
            "v46_canonical_transport_adjusted_points": float(
                record["canonical_transport_adjusted_pooled_points_delta"]
            ),
        }
    return {
        "use": "post_terminal_ledger_only_no_selection_router_or_ensemble",
        "comparisons": comparisons,
        "v45_original_commitment_preserved": True,
        "v45c_confirmation_not_used_as_seed_selection": True,
    }


def write_report(result: dict[str, Any]) -> None:
    if "comparison_to_frozen_dropconnect" not in result:
        return
    record = result["candidate"]
    local = record["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v46 hidden-activation LayerNorm DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{record['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{record['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{record['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"prospective fold x layer gate `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13 five-Linear DeepSets의 네 hidden Linear 뒤, ReLU 전에 "
        "affine LayerNorm(32, eps=1e-5)만 추가했다. v45/v45c 비교는 terminal "
        "후 ledger 진단만 수행했고 selection/retune/router/ensemble은 없다. "
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
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if passed
        else "EXPLORATORY_NO_GO_HIDDEN_ACTIVATION_LAYERNORM"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["layernorm_contract"] = _layernorm_contract_receipt()
    result["isolation"] = _isolation_receipt()
    result["comparison_to_frozen_dropconnect"] = _comparison_to_dropconnect(
        record, config
    )
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer": config["training"]["optimizer"],
            "layer_normalization": config["training"]["layer_normalization"],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
            "extra_loss": 0,
        }
    )
    evidence = config["authorization_evidence"]
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    for name in (
        "negative_fingerprint",
        "design_fingerprint",
        "execution_decision",
        "v13_result",
        "v45_result",
        "v45c_result",
        "v45c_independent_qa",
        "prospective_gate_amendment",
    ):
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
