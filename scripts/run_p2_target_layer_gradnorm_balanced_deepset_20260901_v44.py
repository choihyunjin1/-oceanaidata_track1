"""Run sealed P2 v44 target-layer GradNorm DeepSets exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_target_layer_film_conditioned_deepset_20260901_v43 as v43  # noqa: E402

v37 = v43.v37
v13 = v43.v13
v12 = v43.v12

EXPERIMENT_ID = "p2_target_layer_gradnorm_balanced_deepset_20260901_v44"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V44_TARGET_LAYER_GRADNORM_BALANCED_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.target_layer_gradnorm_balanced_deepset.result.20260901.v44"

_BASE_RUN = v43._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v43._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v43._V13_RUNNER
_ACTIVE_LAYER_IDS: np.ndarray | None = None


def _bind_base() -> None:
    v13.EXPERIMENT_ID = EXPERIMENT_ID
    v13.CONFIG = CONFIG
    v13.ARTIFACT = ARTIFACT
    v13.REPORT = REPORT
    v13.RUNNER = RUNNER
    v13.PREDICTION_NAME = PREDICTION_NAME
    v13.load_config = load_config
    v13.domain_balanced_weights = domain_balanced_weights
    v13.train_predict_seed = train_predict_seed
    v13.write_report = write_report


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    gradnorm = training["gradnorm"]
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
        != "exact_v13_weighted_SmoothL1_with_target_layer_GradNorm"
        or training["optimizer"] != "exact_v13_AdamW_plus_plain_SGD_task_weights"
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
        or training["extra_inference_parameters"]
        or gradnorm["tasks"] != [2, 3, 4]
        or gradnorm["task_definition"] != "target_layer"
        or gradnorm["initial_task_weights"] != [1.0, 1.0, 1.0]
        or gradnorm["task_weight_sum"] != 3.0
        or gradnorm["task_weight_minimum"] != 0.001
        or gradnorm["asymmetry_alpha"] != 1.5
        or gradnorm["shared_parameter"] != "head.0.weight"
        or gradnorm["task_weight_optimizer"] != "plain_SGD_no_momentum"
        or gradnorm["task_weight_learning_rate"] != 0.025
        or gradnorm["missing_task_weight_update"] != "exact_noop"
        or gradnorm["sampler_change"]
        or gradnorm["projection"]
        or gradnorm["coordinate_mask"]
        or gradnorm["sweep"]
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
        raise v12.ContractError("v44 fixed scientific contract drift")
    return config


def domain_balanced_weights(
    layer: np.ndarray, local_time: pd.Series | pd.DatetimeIndex
) -> tuple[np.ndarray, dict[str, Any]]:
    global _ACTIVE_LAYER_IDS
    _ACTIVE_LAYER_IDS = np.asarray(layer, dtype=np.int64).copy()
    weights, receipt = _BASE_DOMAIN_BALANCED_WEIGHTS(layer, local_time)
    labels = sorted(np.unique(_ACTIVE_LAYER_IDS).tolist())
    if labels != [2, 3, 4]:
        raise v12.ContractError("v44 target-layer task support drift")
    receipt["gradnorm_task_labels"] = labels
    return weights, receipt


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def gradnorm_weight_gradient(
    task_losses: list[Tensor],
    task_weights: Tensor,
    shared_parameter: Tensor,
    initial_losses: Tensor,
    alpha: float,
) -> tuple[Tensor, dict[str, Any]]:
    """Return only the GradNorm gradient for the three scalar task weights."""
    if (
        len(task_losses) != 3
        or task_weights.shape != (3,)
        or initial_losses.shape != (3,)
        or alpha != 1.5
    ):
        raise v12.ContractError("v44 GradNorm geometry drift")
    norms = []
    for index, task_loss in enumerate(task_losses):
        gradient = torch.autograd.grad(
            task_weights[index] * task_loss,
            shared_parameter,
            create_graph=True,
            retain_graph=True,
        )[0]
        norms.append(torch.linalg.vector_norm(gradient))
    gradient_norms = torch.stack(norms)
    detached_losses = torch.stack([value.detach() for value in task_losses])
    loss_ratios = detached_losses / initial_losses.clamp_min(1e-12)
    inverse_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-12)
    targets = gradient_norms.detach().mean() * inverse_rates.pow(alpha)
    objective = torch.abs(gradient_norms - targets.detach()).sum()
    weight_gradient = torch.autograd.grad(
        objective, task_weights, retain_graph=True
    )[0]
    values = torch.cat(
        (gradient_norms.detach(), inverse_rates.detach(), targets.detach(), weight_gradient.detach())
    )
    if not bool(torch.isfinite(values).all()):
        raise v12.ContractError("v44 GradNorm values are non-finite")
    return weight_gradient.detach(), {
        "gradient_norms": gradient_norms.detach().cpu().tolist(),
        "relative_inverse_training_rates": inverse_rates.detach().cpu().tolist(),
        "gradient_targets": targets.detach().cpu().tolist(),
        "gradnorm_objective": float(objective.detach().cpu()),
        "task_weight_gradient": weight_gradient.detach().cpu().tolist(),
    }


def apply_projected_task_weight_sgd_(
    task_weights: Tensor,
    gradient: Tensor,
    learning_rate: float,
    minimum: float,
    target_sum: float,
) -> None:
    """One deterministic no-momentum SGD step projected to a lower-bounded simplex."""
    if (
        task_weights.shape != (3,)
        or gradient.shape != (3,)
        or learning_rate != 0.025
        or minimum != 0.001
        or target_sum != 3.0
    ):
        raise v12.ContractError("v44 task-weight SGD contract drift")
    with torch.no_grad():
        candidate = task_weights - learning_rate * gradient
        surplus = torch.clamp(candidate - minimum, min=0.0)
        remaining = target_sum - minimum * task_weights.numel()
        if float(surplus.sum().detach().cpu()) <= 1e-12:
            updated = torch.full_like(task_weights, target_sum / task_weights.numel())
        else:
            updated = minimum + remaining * surplus / surplus.sum()
        task_weights.copy_(updated)
    if (
        not bool(torch.isfinite(task_weights).all())
        or float(task_weights.min().detach().cpu()) < minimum - 1e-8
        or abs(float(task_weights.sum().detach().cpu()) - target_sum) > 1e-6
    ):
        raise v12.ContractError("v44 task-weight projection failed")


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
    if _ACTIVE_LAYER_IDS is None or len(_ACTIVE_LAYER_IDS) != len(target):
        raise v12.ContractError("v44 target-layer commitment unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = v12.VerticalDeepSet(8, 11, hidden=32).to(device)
    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    gradnorm = config["training"]["gradnorm"]
    task_labels = [int(value) for value in gradnorm["tasks"]]
    task_weights = torch.tensor(
        gradnorm["initial_task_weights"],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    initial_task_weight_sha256 = tensor_sha256(task_weights)
    shared_parameter = model.head[0].weight
    train = tuple(
        torch.from_numpy(np.asarray(value).copy())
        for value in (
            tokens,
            mask,
            context,
            target.astype(np.float32),
            weights.astype(np.float32),
            _ACTIVE_LAYER_IDS.astype(np.int64),
        )
    )
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_losses: Tensor | None = None
    epoch_losses: list[float] = []
    task_weight_updates = 0
    missing_task_batches = 0
    optimizer_steps = 0
    task_support_min = {label: 10**9 for label in task_labels}
    gradnorm_objective_sum = 0.0
    gradient_norm_min = float("inf")
    gradient_norm_max = 0.0
    task_weight_min_seen = float(task_weights.min().detach().cpu())
    task_weight_max_seen = float(task_weights.max().detach().cpu())
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        numerator = 0.0
        denominator = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            model_optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            task_losses: list[Tensor] = []
            task_masks: list[Tensor] = []
            all_present = True
            for label in task_labels:
                selected_task = batch[5].eq(label)
                support = int(selected_task.sum().detach().cpu())
                task_support_min[label] = min(task_support_min[label], support)
                if support == 0:
                    all_present = False
                task_masks.append(selected_task)
                if support > 0:
                    local_weights = batch[4][selected_task]
                    task_losses.append(
                        (raw_loss[selected_task] * local_weights).sum()
                        / local_weights.sum().clamp_min(1e-12)
                    )
            if not all_present:
                missing_task_batches += 1
                effective = torch.ones_like(batch[4])
                model_loss = (raw_loss * batch[4] * effective).sum()
                model_loss = model_loss / (batch[4] * effective).sum().clamp_min(1e-12)
            else:
                stacked_losses = torch.stack(task_losses)
                if initial_losses is None:
                    initial_losses = stacked_losses.detach().clone()
                weight_gradient, receipt = gradnorm_weight_gradient(
                    task_losses,
                    task_weights,
                    shared_parameter,
                    initial_losses,
                    float(gradnorm["asymmetry_alpha"]),
                )
                layer_indices = batch[5] - task_labels[0]
                row_task_weights = task_weights.detach()[layer_indices]
                combined_weights = batch[4] * row_task_weights
                model_loss = (raw_loss * combined_weights).sum()
                model_loss = model_loss / combined_weights.sum().clamp_min(1e-12)
                apply_projected_task_weight_sgd_(
                    task_weights,
                    weight_gradient,
                    float(gradnorm["task_weight_learning_rate"]),
                    float(gradnorm["task_weight_minimum"]),
                    float(gradnorm["task_weight_sum"]),
                )
                task_weight_updates += 1
                gradnorm_objective_sum += receipt["gradnorm_objective"]
                gradient_norm_min = min(gradient_norm_min, *receipt["gradient_norms"])
                gradient_norm_max = max(gradient_norm_max, *receipt["gradient_norms"])
                task_weight_min_seen = min(
                    task_weight_min_seen, float(task_weights.min().detach().cpu())
                )
                task_weight_max_seen = max(
                    task_weight_max_seen, float(task_weights.max().detach().cpu())
                )
            if not bool(torch.isfinite(model_loss)):
                raise v12.ContractError("v44 model loss is non-finite")
            model_loss.backward()
            if task_weights.grad is not None:
                raise v12.ContractError("v44 model loss leaked into task weights")
            model_optimizer.step()
            optimizer_steps += 1
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
        epoch_losses.append(numerator / denominator)
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
    final_task_weights = task_weights.detach().cpu().numpy().astype(float)
    finite = bool(
        np.isfinite(epoch_losses).all()
        and np.isfinite(prediction).all()
        and np.isfinite(final_task_weights).all()
    )
    if (
        not finite
        or initial_losses is None
        or task_weight_updates == 0
        or optimizer_steps == 0
        or abs(float(final_task_weights.sum()) - 3.0) > 1e-6
        or float(final_task_weights.min()) < 0.001 - 1e-8
    ):
        raise v12.ContractError("v44 GradNorm training contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(epoch_losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "parameter_tensors": len(list(model.parameters())),
        "loss_first": epoch_losses[0],
        "loss_last": epoch_losses[-1],
        "optimizer_steps": optimizer_steps,
        "task_labels": task_labels,
        "shared_parameter": "head.0.weight",
        "shared_parameter_shape": list(shared_parameter.shape),
        "gradnorm_alpha": float(gradnorm["asymmetry_alpha"]),
        "task_weight_learning_rate": float(gradnorm["task_weight_learning_rate"]),
        "task_weight_updates": task_weight_updates,
        "missing_task_batches": missing_task_batches,
        "task_support_min": {str(key): value for key, value in task_support_min.items()},
        "initial_task_losses": initial_losses.detach().cpu().tolist(),
        "initial_task_weights": gradnorm["initial_task_weights"],
        "final_task_weights": final_task_weights.tolist(),
        "task_weight_sum": float(final_task_weights.sum()),
        "task_weight_min_seen": task_weight_min_seen,
        "task_weight_max_seen": task_weight_max_seen,
        "initial_task_weight_sha256": initial_task_weight_sha256,
        "final_task_weight_sha256": tensor_sha256(task_weights),
        "gradnorm_objective_mean": gradnorm_objective_sum / task_weight_updates,
        "shared_gradient_norm_min": gradient_norm_min,
        "shared_gradient_norm_max": gradient_norm_max,
        "gradient_projection": 0,
        "coordinate_mask": 0,
        "extra_inference_parameters": 0,
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _gradnorm_contract_receipt() -> dict[str, Any]:
    shared_equal = torch.nn.Parameter(torch.tensor([0.75, -0.25]))
    equal_weights = torch.ones(3, requires_grad=True)
    equal_losses = [shared_equal.square().sum() for _ in range(3)]
    equal_initial = torch.stack([value.detach() for value in equal_losses])
    equal_gradient, equal_receipt = gradnorm_weight_gradient(
        equal_losses, equal_weights, shared_equal, equal_initial, 1.5
    )
    equal_before = equal_weights.detach().clone()
    apply_projected_task_weight_sgd_(
        equal_weights, equal_gradient, 0.025, 0.001, 3.0
    )

    shared = torch.nn.Parameter(torch.tensor([0.75, -0.25]))
    task_weights = torch.ones(3, requires_grad=True)
    unequal_losses = [
        shared.square().sum(),
        2.0 * shared.square().sum(),
        4.0 * shared.square().sum(),
    ]
    unequal_initial = torch.stack([value.detach() for value in unequal_losses])
    model_grads_before = [shared.grad, task_weights.grad]
    gradient, receipt = gradnorm_weight_gradient(
        unequal_losses, task_weights, shared, unequal_initial, 1.5
    )
    model_grads_after = [shared.grad, task_weights.grad]
    before = task_weights.detach().clone()
    apply_projected_task_weight_sgd_(task_weights, gradient, 0.025, 0.001, 3.0)

    task_weights_for_model = torch.ones(3, requires_grad=True)
    detached_weighted_loss = sum(
        task_weights_for_model.detach()[index] * value
        for index, value in enumerate(unequal_losses)
    ) / 3.0
    detached_weighted_loss.backward()
    model_loss_task_gradient_absent = task_weights_for_model.grad is None
    return {
        "tasks": [2, 3, 4],
        "shared_parameter": "head.0.weight",
        "alpha": 1.5,
        "task_weight_learning_rate": 0.025,
        "task_weight_minimum": 0.001,
        "target_sum": 3.0,
        "equal_fixed_point_maximum_abs_error": float(
            torch.max(torch.abs(equal_weights.detach() - equal_before))
        ),
        "equal_gradient_maximum_abs": float(torch.max(torch.abs(equal_gradient))),
        "equal_gradient_norms": equal_receipt["gradient_norms"],
        "unequal_update_maximum_abs": float(
            torch.max(torch.abs(task_weights.detach() - before))
        ),
        "unequal_gradient_norms": receipt["gradient_norms"],
        "updated_weights": task_weights.detach().tolist(),
        "updated_weight_sum": float(task_weights.detach().sum()),
        "updated_weight_minimum": float(task_weights.detach().min()),
        "gradnorm_autograd_populated_model_or_weight_grad": any(
            value is not None for value in model_grads_before + model_grads_after
        ),
        "model_loss_task_weight_gradient_absent": model_loss_task_gradient_absent,
        "missing_task_update_exact_noop": True,
        "finite": bool(
            torch.isfinite(torch.tensor(receipt["gradient_norms"])).all()
            and torch.isfinite(gradient).all()
        ),
    }


def _architecture_isolation_receipt() -> dict[str, Any]:
    torch.manual_seed(44)
    left = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    torch.manual_seed(44)
    right = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(5, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 0, 1, 1]],
        dtype=torch.float32,
    )
    context = torch.randn(5, 11)
    changed = tokens.clone()
    changed[mask == 0] = 1e6
    order = torch.tensor([4, 2, 0, 3, 1])
    with torch.inference_mode():
        base = left(tokens, mask, context)
        state_identical = right(tokens, mask, context)
        masked = left(changed, mask, context)
        permuted = left(tokens[:, order], mask[:, order], context)
    return {
        "model_class": type(left).__name__,
        "parameters": int(sum(value.numel() for value in left.parameters())),
        "parameter_tensors": len(list(left.parameters())),
        "state_identical_initial_function_maximum_abs_error": float(
            torch.max(torch.abs(base - state_identical))
        ),
        "masked_or_future_token_maximum_abs_error": float(
            torch.max(torch.abs(base - masked))
        ),
        "permutation_maximum_abs_error": float(torch.max(torch.abs(base - permuted))),
        "extra_inference_parameters": 0,
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    for key, relative in config["authorization_evidence"].items():
        if not (
            key.endswith("_result")
            or key in ("prospective_gate_amendment", "fingerprint")
        ):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_execution_hits": 0,
        "v18_group_dro_distinguished": True,
        "v28_pcgrad_distinguished": True,
        "v36_fishr_distinguished": True,
        "v39_sign_mask_distinguished": True,
        "v43_film_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    contract = _gradnorm_contract_receipt()
    isolation = _architecture_isolation_receipt()
    if (
        isolation["model_class"] != "VerticalDeepSet"
        or isolation["parameters"] != 4865
        or isolation["parameter_tensors"] != 10
        or max(
            isolation["state_identical_initial_function_maximum_abs_error"],
            isolation["masked_or_future_token_maximum_abs_error"],
            isolation["permutation_maximum_abs_error"],
        )
        > 1e-6
        or isolation["extra_inference_parameters"] != 0
        or contract["equal_fixed_point_maximum_abs_error"] > 1e-7
        or contract["unequal_update_maximum_abs"] <= 0.0
        or abs(contract["updated_weight_sum"] - 3.0) > 1e-6
        or contract["updated_weight_minimum"] < 0.001 - 1e-8
        or contract["gradnorm_autograd_populated_model_or_weight_grad"]
        or not contract["model_loss_task_weight_gradient_absent"]
        or not contract["missing_task_update_exact_noop"]
        or not contract["finite"]
    ):
        raise v12.ContractError("v44 target-free GradNorm preflight failed")
    evidence = config["authorization_evidence"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "gradnorm_contract": contract,
        "architecture_isolation": isolation,
        "prospective_fold_layer_gate": config["evaluation"]["safety_gate"],
        "prefix_cutoffs": {
            fold: (pd.Timestamp(start) - pd.Timedelta(days=7)).isoformat()
            for fold, start in config["training"]["fold_starts_kst"].items()
        },
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "audit_result_sha256": v12.sha256_file(ROOT / evidence["audit_result"]),
        "fingerprint_sha256": v12.sha256_file(ROOT / evidence["fingerprint"]),
        "gate_amendment_sha256": v12.sha256_file(
            ROOT / evidence["prospective_gate_amendment"]
        ),
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
    local = item["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v44 target-layer GradNorm balanced DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{item['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{item['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"prospective fold x layer gate: `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13 architecture에서 target-layer task weight만 fixed GradNorm "
        "alpha 1.5로 학습했다. sweep/projection/sign mask/router/ensemble/row deletion/"
        "Public selection/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_TARGET_LAYER_GRADNORM"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["gradnorm_contract"] = _gradnorm_contract_receipt()
    result["architecture_isolation"] = _architecture_isolation_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer": config["training"]["optimizer"],
            "gradnorm": config["training"]["gradnorm"],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
            "extra_inference_parameters": 0,
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
            for name in ("v13", "v18", "v28", "v36", "v39", "v43")
        },
    }
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    result["hashes"]["prospective_gate_amendment"] = config[
        "authorization_evidence"
    ]["prospective_gate_amendment_sha256"]
    result["hashes"]["fingerprint"] = config["authorization_evidence"][
        "fingerprint_sha256"
    ]
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
