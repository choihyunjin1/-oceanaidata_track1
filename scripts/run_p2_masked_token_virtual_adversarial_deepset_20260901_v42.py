"""Run sealed P2 v42 masked-token virtual adversarial DeepSets exactly once."""

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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_all_linear_weight_normalized_deepset_20260901_v41 as v41  # noqa: E402

v40 = v41.v40
v37 = v41.v37
v13 = v41.v13
v12 = v41.v12

EXPERIMENT_ID = "p2_masked_token_virtual_adversarial_deepset_20260901_v42"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V42_MASKED_TOKEN_VIRTUAL_ADVERSARIAL_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.masked_token_virtual_adversarial_deepset.result.20260901.v42"

_BASE_RUN = v41._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v41._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v41._V13_RUNNER


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
    training = config["training"]
    vat = training["virtual_adversarial_training"]
    safety = config["evaluation"]["safety_gate"]
    evidence = config["authorization_evidence"]
    audit = ROOT / evidence["audit_result"]
    amendment = ROOT / evidence["prospective_gate_amendment"]
    fingerprint = ROOT / evidence["fingerprint"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"]
        != "weighted_SmoothL1_beta_1.0_plus_masked_token_virtual_adversarial_fixed_variance_Gaussian_consistency"
        or training["optimizer"] != "exact_v13_AdamW"
        or training["learning_rate"] != 0.001
        or training["weight_decay"] != 0.0001
        or training["epochs"] != 60
        or training["seeds"] != [20260901, 20260902, 20260903]
        or training["maximum_fit_count"] != 9
        or training["champion_preserving_weight"] != 0.8
        or training["model_weight"] != 0.2
        or training["model_minus_champion_clip_C"] != 2.5
        or training["maximum_final_action_C"] != 0.5
        or training["row_deletion"]
        or training["data_augmentation"]
        or vat["eligible_token_channels"] != [0, 1]
        or vat["presence_indicator_channels"] != [4, 5]
        or not vat["token_mask_required"]
        or vat["epsilon_normalized_L2_per_row"] != 0.05
        or vat["finite_difference_xi"] != 0.000001
        or vat["power_iterations"] != 1
        or vat["coefficient"] != 1.0
        or vat["direction_initialization"]
        != "deterministic_sine_of_observed_value_and_channel_no_layer_index"
        or vat["direction_objective"]
        != "0.5_weighted_prediction_difference_squared_divided_by_xi_squared"
        or vat["final_consistency"]
        != "0.5_times_coefficient_times_weighted_mean_squared_clean_stopgrad_minus_adversarial_prediction"
        or not vat["fixed_variance_Gaussian_interpretation"]
        or vat["perturb_masks"]
        or vat["perturb_depth_or_nominal"]
        or vat["perturb_presence_indicators"]
        or vat["perturb_context_or_target_layer_one_hot"]
        or vat["perturb_labels"]
        or vat["inference_perturbation"]
        or vat["sweep"]
        or safety["minimum_fold_layer_non_harm_cells"] != 8
        or safety["total_fold_layer_cells"] != 9
        or safety["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or not audit.is_file()
        or json.loads(audit.read_text(encoding="utf-8"))["status"]
        != evidence["required_status"]
        or not amendment.is_file()
        or v12.sha256_file(amendment)
        != evidence["prospective_gate_amendment_sha256"]
        or not fingerprint.is_file()
        or v12.sha256_file(fingerprint) != evidence["fingerprint_sha256"]
    ):
        raise v12.ContractError("v42 fixed scientific contract drift")
    return config


def eligible_value_mask(tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[-1] != 8 or token_mask.shape != tokens.shape[:2]:
        raise v12.ContractError("v42 eligible-value tensor contract failed")
    eligible = torch.zeros_like(tokens, dtype=torch.bool)
    active_token = token_mask > 0.5
    eligible[:, :, 0] = active_token & (tokens[:, :, 4] > 0.5)
    eligible[:, :, 1] = active_token & (tokens[:, :, 5] > 0.5)
    return eligible


def _normalize_per_row(
    values: torch.Tensor, eligible: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = torch.where(eligible, values, torch.zeros_like(values))
    norms = torch.linalg.vector_norm(masked.flatten(1), dim=1)
    scale = norms.clamp_min(1e-20).view(-1, 1, 1)
    normalized = masked / scale
    normalized = torch.where(
        (norms > 1e-20).view(-1, 1, 1), normalized, torch.zeros_like(normalized)
    )
    return normalized, norms


def deterministic_initial_direction(
    tokens: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    channel_offset = torch.zeros(8, dtype=tokens.dtype, device=tokens.device)
    channel_offset[0] = 0.5
    channel_offset[1] = 1.5
    raw = torch.sin(tokens.detach() * 12.9898 + channel_offset.view(1, 1, -1))
    normalized, _ = _normalize_per_row(raw, eligible)
    return normalized


def virtual_adversarial_direction(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    context: torch.Tensor,
    clean_prediction: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float = 0.05,
    xi: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if (
        clean_prediction.ndim != 1
        or weights.shape != clean_prediction.shape
        or epsilon not in (0.0, 0.05)
        or xi != 1e-6
        or bool(torch.any(weights <= 0.0))
    ):
        raise v12.ContractError("v42 VAT direction contract failed")
    # Canonicalize the set by the unchanged nominal-depth offset before the
    # finite-difference pass. This removes reduction-order ambiguity from the
    # direction while leaving the model, features and inference untouched.
    order = torch.argsort(tokens[:, :, 3], dim=1, stable=True)
    token_index = order.unsqueeze(-1).expand_as(tokens)
    canonical_tokens = torch.gather(tokens, 1, token_index)
    canonical_mask = torch.gather(token_mask, 1, order)
    canonical_clean = model(canonical_tokens, canonical_mask, context).detach()
    eligible = eligible_value_mask(canonical_tokens, canonical_mask)
    initial = deterministic_initial_direction(canonical_tokens, eligible)
    probe = (xi * initial).detach().requires_grad_(True)
    probe_prediction = model(canonical_tokens + probe, canonical_mask, context)
    denominator = weights.sum().clamp_min(1e-12)
    direction_loss = 0.5 * (((probe_prediction - canonical_clean).square() * weights).sum())
    direction_loss = direction_loss / denominator / (xi * xi)
    gradient = torch.autograd.grad(direction_loss, probe, only_inputs=True)[0]
    gradient_unit, gradient_norm = _normalize_per_row(gradient, eligible)
    initial_norm = torch.linalg.vector_norm(initial.flatten(1), dim=1)
    fallback = (gradient_norm <= 1e-20) & (initial_norm > 1e-20)
    unit = torch.where(fallback.view(-1, 1, 1), initial, gradient_unit)
    canonical_direction = (epsilon * unit).detach()
    direction = torch.zeros_like(canonical_direction).scatter(
        1, token_index, canonical_direction
    )
    eligible_rows = eligible.flatten(1).any(dim=1)
    final_norm = torch.linalg.vector_norm(canonical_direction.flatten(1), dim=1)
    norm_error = (
        torch.max(torch.abs(final_norm[eligible_rows] - epsilon))
        if bool(eligible_rows.any())
        else torch.tensor(0.0, device=tokens.device)
    )
    if (
        not bool(torch.isfinite(direction).all())
        or not bool(torch.isfinite(direction_loss))
        or float(norm_error.detach().cpu()) > 1e-6
        or bool(torch.any(canonical_direction[~eligible] != 0.0))
    ):
        raise v12.ContractError("v42 VAT direction finite/mask/norm guard failed")
    return direction, {
        "eligible_rows": int(eligible_rows.sum().detach().cpu()),
        "eligible_coordinates": int(eligible.sum().detach().cpu()),
        "fallback_rows": int(fallback.sum().detach().cpu()),
        "maximum_epsilon_norm_error": float(norm_error.detach().cpu()),
        "direction_objective": float(direction_loss.detach().cpu()),
        "canonical_clean_maximum_abs_error": float(
            torch.max(torch.abs(canonical_clean - clean_prediction.detach())).detach().cpu()
        ),
        "power_iterations": 1,
        "label_reads": 0,
    }


def vat_consistency_penalty(
    clean_prediction: torch.Tensor,
    adversarial_prediction: torch.Tensor,
    weights: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    if (
        clean_prediction.ndim != 1
        or adversarial_prediction.shape != clean_prediction.shape
        or weights.shape != clean_prediction.shape
        or coefficient not in (0.0, 1.0)
    ):
        raise v12.ContractError("v42 VAT consistency tensor contract failed")
    value = 0.5 * coefficient * (
        (adversarial_prediction - clean_prediction.detach()).square() * weights
    ).sum() / weights.sum().clamp_min(1e-12)
    if not bool(torch.isfinite(value)):
        raise v12.ContractError("v42 VAT consistency is non-finite")
    return value


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
    vat = config["training"]["virtual_adversarial_training"]
    model = v12.VerticalDeepSet(8, 11, hidden=32).to(device)
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
    base_losses: list[float] = []
    penalties: list[float] = []
    total_losses: list[float] = []
    maximum_norm_errors: list[float] = []
    optimizer_steps = 0
    direction_steps = 0
    eligible_rows_seen = 0
    fallback_rows_seen = 0
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        base_total = 0.0
        penalty_total = 0.0
        total_loss_total = 0.0
        norm_error = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            clean = model(batch[0], batch[1], batch[2])
            raw = F.smooth_l1_loss(clean, batch[3], beta=1.0, reduction="none")
            denominator = batch[4].sum().clamp_min(1e-12)
            base_loss = (raw * batch[4]).sum() / denominator
            direction, receipt = virtual_adversarial_direction(
                model,
                batch[0],
                batch[1],
                batch[2],
                clean,
                batch[4],
                epsilon=float(vat["epsilon_normalized_L2_per_row"]),
                xi=float(vat["finite_difference_xi"]),
            )
            adversarial = model(batch[0] + direction, batch[1], batch[2])
            penalty = vat_consistency_penalty(
                clean, adversarial, batch[4], coefficient=float(vat["coefficient"])
            )
            loss = base_loss + penalty
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v42 total training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            direction_steps += 1
            eligible_rows_seen += receipt["eligible_rows"]
            fallback_rows_seen += receipt["fallback_rows"]
            base_total += float(base_loss.detach().cpu())
            penalty_total += float(penalty.detach().cpu())
            total_loss_total += float(loss.detach().cpu())
            norm_error = max(norm_error, receipt["maximum_epsilon_norm_error"])
            batches += 1
        base_losses.append(base_total / batches)
        penalties.append(penalty_total / batches)
        total_losses.append(total_loss_total / batches)
        maximum_norm_errors.append(norm_error)

    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_tokens), batch_size):
            stop = start + batch_size
            output.append(
                model(
                    torch.from_numpy(np.asarray(query_tokens[start:stop]).copy()).to(device),
                    torch.from_numpy(np.asarray(query_mask[start:stop]).copy()).to(device),
                    torch.from_numpy(np.asarray(query_context[start:stop]).copy()).to(device),
                )
                .cpu()
                .numpy()
            )
    prediction = np.concatenate(output).astype(float)
    finite = bool(
        np.isfinite(base_losses).all()
        and np.isfinite(penalties).all()
        and np.isfinite(total_losses).all()
        and np.isfinite(maximum_norm_errors).all()
        and np.isfinite(prediction).all()
    )
    if not finite or max(maximum_norm_errors) > 1e-6:
        raise v12.ContractError("v42 VAT training receipt failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(base_losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "base_loss_first": base_losses[0],
        "base_loss_last": base_losses[-1],
        "vat_penalty_first": penalties[0],
        "vat_penalty_last": penalties[-1],
        "total_loss_first": total_losses[0],
        "total_loss_last": total_losses[-1],
        "optimizer_steps": optimizer_steps,
        "vat_direction_steps": direction_steps,
        "power_iterations_per_step": 1,
        "eligible_rows_seen": eligible_rows_seen,
        "fallback_rows_seen": fallback_rows_seen,
        "maximum_epsilon_norm_error": max(maximum_norm_errors),
        "epsilon": float(vat["epsilon_normalized_L2_per_row"]),
        "xi": float(vat["finite_difference_xi"]),
        "coefficient": float(vat["coefficient"]),
        "inference_perturbation": False,
        "label_reads_for_direction": 0,
        "parameter_perturbation_steps": 0,
        "row_mixing_steps": 0,
        "loss": config["training"]["objective"],
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _vat_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(42)
    model = v12.VerticalDeepSet(8, 11, hidden=32)
    tokens = torch.randn(4, 5, 8)
    tokens[:, :, 4:] = 1.0
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    context = torch.randn(4, 11)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    clean = model(tokens, mask, context)
    direction, receipt = virtual_adversarial_direction(
        model, tokens, mask, context, clean, weights
    )
    zero_direction, _ = virtual_adversarial_direction(
        model, tokens, mask, context, clean, weights, epsilon=0.0
    )
    adversarial = model(tokens + direction, mask, context)
    zero_penalty = vat_consistency_penalty(clean, adversarial, weights, coefficient=0.0)
    eligible = eligible_value_mask(tokens, mask)
    changed = tokens + direction
    untouched_error = torch.max(torch.abs((changed - tokens)[~eligible]))
    order = torch.tensor([4, 2, 0, 3, 1])
    permuted_tokens = tokens[:, order]
    permuted_mask = mask[:, order]
    permuted_clean = model(permuted_tokens, permuted_mask, context)
    permuted_direction, _ = virtual_adversarial_direction(
        model, permuted_tokens, permuted_mask, context, permuted_clean, weights
    )
    permutation_error = torch.max(torch.abs(direction[:, order] - permuted_direction))
    repeated, _ = virtual_adversarial_direction(
        model, tokens, mask, context, clean, weights
    )
    return {
        **receipt,
        "eligible_channels": [0, 1],
        "untouched_coordinate_maximum_abs_error": float(untouched_error.detach()),
        "epsilon_zero_exact_noop": bool(torch.equal(zero_direction, torch.zeros_like(zero_direction))),
        "coefficient_zero_exact_noop": float(zero_penalty.detach()) == 0.0,
        "permutation_equivariance_maximum_abs_error": float(permutation_error.detach()),
        "repeat_maximum_abs_error": float(torch.max(torch.abs(direction - repeated)).detach()),
        "direction_requires_grad": direction.requires_grad,
        "clean_stop_gradient_in_consistency": True,
        "target_reads": 0,
    }


def _isolation_receipt() -> dict[str, float]:
    torch.manual_seed(11)
    model = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(3, 5, 8)
    tokens[:, :, 4:] = 1.0
    mask = torch.tensor(
        [[1, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=torch.float32
    )
    context = torch.randn(3, 11)
    changed = tokens.clone()
    changed[mask == 0] = 1e6
    with torch.inference_mode():
        base = model(tokens, mask, context)
        masked = model(changed, mask, context)
    return {"masked_or_future_token_maximum_abs_error": float(torch.max(torch.abs(base - masked)))}


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    for key, relative in config["authorization_evidence"].items():
        if not (key.endswith("_result") or key in ("prospective_gate_amendment", "fingerprint")):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v23_direct_jacobian_distinguished": True,
        "v24_parameter_sam_distinguished": True,
        "v26_row_label_mixup_distinguished": True,
        "v31_domain_adversary_distinguished": True,
        "v40_dropout_consistency_distinguished": True,
        "v41_weight_norm_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    contract = _vat_contract_receipt()
    isolation = _isolation_receipt()
    if (
        contract["maximum_epsilon_norm_error"] > 1e-6
        or contract["untouched_coordinate_maximum_abs_error"] != 0.0
        or not contract["epsilon_zero_exact_noop"]
        or not contract["coefficient_zero_exact_noop"]
        or contract["permutation_equivariance_maximum_abs_error"] > 1e-6
        or contract["repeat_maximum_abs_error"] != 0.0
        or max(isolation.values()) > 1e-6
    ):
        raise v12.ContractError("v42 target-free VAT preflight failed")
    evidence = config["authorization_evidence"]
    audit = ROOT / evidence["audit_result"]
    amendment = ROOT / evidence["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "vat_contract": contract,
        "isolation": isolation,
        "validation_rows_used_for_direction": 0,
        "inference_perturbation": 0,
        "prospective_fold_layer_gate": config["evaluation"]["safety_gate"],
        "prefix_cutoffs": {
            fold: (pd.Timestamp(start) - pd.Timedelta(days=7)).isoformat()
            for fold, start in config["training"]["fold_starts_kst"].items()
        },
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "audit_result_sha256": v12.sha256_file(audit),
        "gate_amendment_sha256": v12.sha256_file(amendment),
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


def write_report(result: dict[str, Any]) -> None:
    item = result["candidate"]
    if "prospective_fold_layer_gate" not in item:
        return
    folds = item["by_fold"]
    local = item["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v42 masked-token virtual adversarial DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{item['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{item['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"fold delta RMSE: Sep-Oct `{folds['2024_sep_oct']['delta_rmse']:+.9f}`, "
        f"Jul-Aug `{folds['2025_jul_aug']['delta_rmse']:+.9f}`, "
        f"Nov-Dec `{folds['2025_nov_dec']['delta_rmse']:+.9f}`.\n\n"
        f"prospective fold x layer gate: `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13에 observed temperature/salinity token-value의 label-free one-step "
        "VAT consistency만 추가했다. Miyato et al.은 local smoothness 동기만 제공하며 "
        "P2 성능 근거가 아니다. sweep/router/ensemble/row deletion/Public selection/"
        "official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_MASKED_TOKEN_VIRTUAL_ADVERSARIAL"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["vat_contract"] = _vat_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "virtual_adversarial_training": config["training"][
                "virtual_adversarial_training"
            ],
            "row_deletion": 0,
            "data_augmentation": 0,
        }
    )
    result["comparison_to_preserved_candidates"] = {
        "use": "ledger_only_no_posthoc_selection_router_or_ensemble",
        "official_v23_feedback_used_for_selection": False,
        **{
            f"{name}_delta_rmse": json.loads(
                (ROOT / config["authorization_evidence"][f"{name}_result"]).read_text(
                    encoding="utf-8"
                )
            )["candidate"]["delta_rmse"]
            for name in ("v13", "v23", "v24", "v26", "v31", "v40", "v41")
        },
    }
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    result["hashes"]["prospective_gate_amendment"] = config[
        "authorization_evidence"
    ]["prospective_gate_amendment_sha256"]
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
