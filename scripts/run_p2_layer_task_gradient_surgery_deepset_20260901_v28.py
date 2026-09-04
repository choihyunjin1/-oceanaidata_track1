"""Run sealed P2 v28 target-layer PCGrad DeepSets exactly once."""

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
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_layer_task_gradient_surgery_deepset_20260901_v28"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V28_LAYER_TASK_PCGRAD_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.layer_task_gradient_surgery_deepset.result.20260901.v28"

_BASE_LOAD_CONFIG = v13.load_config
_BASE_RUN = v13.run
_BASE_DOMAIN_BALANCED_WEIGHTS = v13.domain_balanced_weights
_V13_RUNNER = v13.RUNNER
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
    config = _BASE_LOAD_CONFIG()
    training = config["training"]
    surgery = training["gradient_surgery"]
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    if (
        training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["epochs"] != 60
        or training["seeds"] != [20260901, 20260902, 20260903]
        or training["maximum_fit_count"] != 9
        or training["champion_preserving_weight"] != 0.8
        or training["model_weight"] != 0.2
        or training["model_minus_champion_clip_C"] != 2.5
        or training["maximum_final_action_C"] != 0.5
        or training["row_deletion"]
        or training["input_perturbation"]
        or training["data_augmentation"]
        or surgery["algorithm"] != "PCGrad_negative_dot_normal_plane_projection"
        or surgery["task_axis"] != "target_layer"
        or surgery["task_labels"] != [2, 3, 4]
        or surgery["task_order"] != [2, 3, 4]
        or surgery["projection_reference"] != "unmodified_other_task_gradient"
        or surgery["combine"] != "arithmetic_mean_of_projected_task_gradients"
        or not surgery["negative_dot_only"]
        or surgery["denominator_epsilon"] != 1e-12
        or surgery["task_order_sweep"]
        or surgery["task_reweight_sweep"]
        or safety["minimum_fold_layer_non_harm_cells"] != 8
        or safety["total_fold_layer_cells"] != 9
        or safety["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or not amendment.is_file()
        or v12.sha256_file(amendment)
        != config["authorization_evidence"]["prospective_gate_amendment_sha256"]
    ):
        raise v12.ContractError("v28 fixed scientific contract drift")
    return config


def domain_balanced_weights(
    layer: np.ndarray, local_time: pd.Series | pd.DatetimeIndex
) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture the fixed target-layer task identity beside exact v13 weights."""
    global _ACTIVE_LAYER_IDS
    _ACTIVE_LAYER_IDS = np.asarray(layer, dtype=np.int64).copy()
    weights, receipt = _BASE_DOMAIN_BALANCED_WEIGHTS(layer, local_time)
    labels = sorted(np.unique(_ACTIVE_LAYER_IDS).tolist())
    if labels != [2, 3, 4]:
        raise v12.ContractError("v28 target-layer task support drift")
    receipt["pcgrad_task_labels"] = labels
    return weights, receipt


def project_conflicting_gradients(
    task_gradients: list[Tensor], epsilon: float = 1e-12
) -> tuple[list[Tensor], dict[str, Any]]:
    """Apply fixed-order PCGrad against unmodified other-task gradients."""
    if not task_gradients or any(value.ndim != 1 for value in task_gradients):
        raise v12.ContractError("v28 PCGrad vector contract drift")
    originals = [value.detach().clone() for value in task_gradients]
    projected: list[Tensor] = []
    negative_pairs = 0
    projection_norms = []
    for index, original in enumerate(originals):
        value = original.clone()
        before = value.clone()
        for other_index, other in enumerate(originals):
            if index == other_index:
                continue
            dot = torch.dot(value, other)
            if float(dot.detach().cpu()) < 0.0:
                value = value - dot * other / (torch.dot(other, other) + epsilon)
                negative_pairs += 1
        projected.append(value)
        projection_norms.append(float(torch.linalg.vector_norm(value - before).cpu()))
    return projected, {
        "directional_pair_count": len(originals) * (len(originals) - 1),
        "negative_dot_projection_count": negative_pairs,
        "projection_delta_l2_sum": float(sum(projection_norms)),
        "projection_delta_l2_max": float(max(projection_norms, default=0.0)),
    }


def _flatten_gradients(
    gradients: tuple[Tensor | None, ...], parameters: list[nn.Parameter]
) -> Tensor:
    return torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
            for gradient, parameter in zip(gradients, parameters, strict=True)
        ]
    )


def _assign_flat_gradient(parameters: list[nn.Parameter], gradient: Tensor) -> None:
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad = gradient[offset : offset + size].reshape_as(parameter).clone()
        offset += size
    if offset != gradient.numel():
        raise v12.ContractError("v28 flattened gradient geometry drift")


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
        raise v12.ContractError("v28 active target-layer task IDs unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = v12.VerticalDeepSet(8, 11, hidden=32).to(device)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    train = tuple(
        torch.from_numpy(value)
        for value in (
            tokens,
            mask,
            context,
            target.astype(np.float32),
            weights.astype(np.float32),
        )
    )
    task_labels = config["training"]["gradient_surgery"]["task_labels"]
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    total_negative_pairs = 0
    total_directional_pairs = 0
    total_projection_norm = 0.0
    maximum_projection_norm = 0.0
    missing_task_batches = 0
    batches = 0
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        numerator = 0.0
        denominator = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            layers = torch.from_numpy(_ACTIVE_LAYER_IDS[selected.numpy()]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            task_gradients = []
            for label in task_labels:
                task_mask = layers.eq(int(label))
                if not bool(task_mask.any()):
                    missing_task_batches += 1
                    continue
                task_weights = batch[4][task_mask]
                task_loss = (raw_loss[task_mask] * task_weights).sum()
                task_loss = task_loss / task_weights.sum().clamp_min(1e-12)
                gradients = torch.autograd.grad(
                    task_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                task_gradients.append(_flatten_gradients(gradients, parameters))
            if len(task_gradients) != 3:
                raise v12.ContractError("v28 minibatch missing a target-layer task")
            projected, receipt = project_conflicting_gradients(task_gradients)
            combined = torch.stack(projected).mean(dim=0)
            if not bool(torch.isfinite(combined).all()):
                raise v12.ContractError("v28 combined PCGrad is non-finite")
            _assign_flat_gradient(parameters, combined)
            optimizer.step()
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
            total_negative_pairs += receipt["negative_dot_projection_count"]
            total_directional_pairs += receipt["directional_pair_count"]
            total_projection_norm += receipt["projection_delta_l2_sum"]
            maximum_projection_norm = max(
                maximum_projection_norm, receipt["projection_delta_l2_max"]
            )
            batches += 1
        losses.append(numerator / denominator)
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_tokens), batch_size):
            stop = start + batch_size
            output.append(
                model(
                    torch.from_numpy(query_tokens[start:stop]).to(device),
                    torch.from_numpy(query_mask[start:stop]).to(device),
                    torch.from_numpy(query_context[start:stop]).to(device),
                )
                .cpu()
                .numpy()
            )
    prediction = np.concatenate(output).astype(float)
    finite = bool(np.isfinite(losses).all() and np.isfinite(prediction).all())
    if not finite or missing_task_batches != 0 or total_directional_pairs == 0:
        raise v12.ContractError("v28 training/PCGrad contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "task_labels": task_labels,
        "task_order": task_labels,
        "batches": batches,
        "directional_task_pairs": total_directional_pairs,
        "negative_dot_projections": total_negative_pairs,
        "negative_dot_projection_share": total_negative_pairs / total_directional_pairs,
        "projection_delta_l2_mean_per_batch": total_projection_norm / batches,
        "projection_delta_l2_max": maximum_projection_norm,
        "missing_task_batches": missing_task_batches,
        "input_gradient_penalty": 0,
        "parameter_neighborhood_perturbation": 0,
        "data_augmentation": 0,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _projection_contract_receipt() -> dict[str, Any]:
    conflict = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    first, first_receipt = project_conflicting_gradients(conflict)
    second, second_receipt = project_conflicting_gradients(conflict)
    no_conflict = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    untouched, untouched_receipt = project_conflicting_gradients(no_conflict)
    deterministic = all(
        torch.equal(left, right) for left, right in zip(first, second, strict=True)
    ) and first_receipt == second_receipt
    zero_conflict_noop = all(
        torch.equal(left, right)
        for left, right in zip(untouched, no_conflict, strict=True)
    )
    conflict_removed = all(float(torch.linalg.vector_norm(value)) <= 1e-12 for value in first)
    if (
        not deterministic
        or not zero_conflict_noop
        or not conflict_removed
        or first_receipt["negative_dot_projection_count"] != 2
        or untouched_receipt["negative_dot_projection_count"] != 0
    ):
        raise v12.ContractError("v28 PCGrad projection formula/order contract failed")
    return {
        "fixed_task_order": [2, 3, 4],
        "opposite_gradient_projection_count": first_receipt[
            "negative_dot_projection_count"
        ],
        "opposite_gradient_residual_l2_max": max(
            float(torch.linalg.vector_norm(value)) for value in first
        ),
        "zero_conflict_exact_noop": zero_conflict_noop,
        "byte_identical_replay": deterministic,
        "projection_reference": "unmodified_other_task_gradient",
    }


def _isolation_receipt() -> dict[str, Any]:
    torch.manual_seed(28)
    model = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(4, 5, 8)
    mask = torch.ones(4, 5)
    mask[:, -1] = 0.0
    context = torch.randn(4, 11)
    changed = tokens.clone()
    changed[:, -1] += 1000.0
    permutation = torch.tensor([2, 4, 0, 3, 1])
    with torch.inference_mode():
        base = model(tokens, mask, context)
        masked = model(changed, mask, context)
        permuted = model(tokens[:, permutation], mask[:, permutation], context)
    return {
        "masked_token_maximum_abs_error": float(torch.max(torch.abs(base - masked))),
        "permutation_maximum_abs_error": float(torch.max(torch.abs(base - permuted))),
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative in (
        config["authorization_evidence"]["v13_result"],
        config["authorization_evidence"]["v18_result"],
        config["authorization_evidence"]["v19_result"],
        config["authorization_evidence"]["v23_result"],
        config["authorization_evidence"]["v24_result"],
        config["authorization_evidence"]["v26_result"],
        config["authorization_evidence"]["audit_result"],
        config["authorization_evidence"]["prospective_gate_amendment"],
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v18_group_dro_distinguished": True,
        "v19_risk_variance_distinguished": True,
        "v23_input_gradient_distinguished": True,
        "v24_sam_distinguished": True,
        "v26_mixup_distinguished": True,
        "evidence_sha256": evidence,
    }


def prospective_fold_layer_gate(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    values = [
        float(layer["delta_rmse"])
        for fold in record["by_fold_layer"].values()
        for layer in fold.values()
    ]
    gate = config["evaluation"]["safety_gate"]
    non_harm = int(sum(value <= 0.0 for value in values))
    maximum = float(max(values))
    checks = {
        "minimum_eight_of_nine_non_harm": non_harm
        >= int(gate["minimum_fold_layer_non_harm_cells"]),
        "all_cells_within_plus_0_003C": maximum
        <= float(gate["maximum_any_fold_layer_delta_rmse_C"]),
    }
    return {
        "source": config["authorization_evidence"]["prospective_gate_amendment"],
        "source_sha256": config["authorization_evidence"][
            "prospective_gate_amendment_sha256"
        ],
        "non_harm_cells": non_harm,
        "total_cells": len(values),
        "non_harm_coverage": non_harm / len(values),
        "maximum_cell_delta_rmse_C": maximum,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    isolation = _isolation_receipt()
    if max(isolation.values()) > 1e-6:
        raise v12.ContractError("v28 masked/future or permutation isolation failed")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "projection_contract": _projection_contract_receipt(),
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
        "audit_result_sha256": v12.sha256_file(audit_path),
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
    folds = item["by_fold"]
    local = item["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v28 fixed target-layer PCGrad DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled ΔRMSE `{item['delta_rmse']:+.9f} C`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점.\n\n"
        f"fold ΔRMSE: Sep-Oct `{folds['2024_sep_oct']['delta_rmse']:+.9f}`, "
        f"Jul-Aug `{folds['2025_jul_aug']['delta_rmse']:+.9f}`, "
        f"Nov-Dec `{folds['2025_nov_dec']['delta_rmse']:+.9f}`.\n\n"
        f"prospective fold×layer gate: `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "v13 science를 고정하고 target layers [2,3,4]의 conflicting parameter gradients에만 "
        "fixed-order PCGrad를 적용했다. Yu et al. (NeurIPS 2020)은 최적화 동기만 제공하며 "
        "P2 성능 근거가 아니다. order/reweight sweep/router/ensemble/official/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    _bind_base()
    started = time.perf_counter()
    result = _BASE_RUN()
    config = load_config()
    record = result["candidate"]
    legacy_safety = bool(record["safety_pass"])
    amended = prospective_fold_layer_gate(record, config)
    record["legacy_safety_pass_without_v26a_amendment"] = legacy_safety
    record["prospective_fold_layer_gate"] = amended
    record["safety_pass"] = bool(legacy_safety and amended["pass"])
    record["safety_pass_with_v26a_amendment"] = record["safety_pass"]
    passed = bool(record["strict_exploratory_pass"] and record["safety_pass"])
    result["schema_version"] = RESULT_SCHEMA
    result["status"] = (
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if passed
        else "EXPLORATORY_NO_GO_LAYER_TASK_PCGRAD"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["projection_contract"] = _projection_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "gradient_surgery": config["training"]["gradient_surgery"],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
        }
    )
    result["comparison_to_preserved_candidates"] = {
        "use": "ledger_only_no_posthoc_selection_router_or_ensemble",
        "v13_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v13_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v23_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v23_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v24_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v24_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v26_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v26_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
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
