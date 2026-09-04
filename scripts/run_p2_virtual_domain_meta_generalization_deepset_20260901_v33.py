"""Run sealed P2 v33 virtual-domain MLDG DeepSets exactly once."""

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

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_layer_task_gradient_surgery_deepset_20260901_v28 as v28  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_virtual_domain_meta_generalization_deepset_20260901_v33"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V33_VIRTUAL_DOMAIN_MLDG_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.virtual_domain_meta_generalization_deepset.result.20260901.v33"

_BASE_LOAD_CONFIG = v13.load_config
_BASE_RUN = v13.run
_BASE_DOMAIN_BALANCED_WEIGHTS = v13.domain_balanced_weights
_V13_RUNNER = v13.RUNNER
_CURRENT_ENVIRONMENTS: np.ndarray | None = None
_CURRENT_ENVIRONMENT_LABELS: dict[int, str] = {}


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
    meta = training["meta_objective"]
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    audit = ROOT / config["authorization_evidence"]["audit_result"]
    if (
        training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"]
        != "weighted_SmoothL1_MLDG_virtual_layer_month_transfer"
        or training["optimizer"] != "exact_v13_AdamW_outer"
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
        or training["input_perturbation"]
        or training["data_augmentation"]
        or meta["environment"] != "target_layer_x_calendar_month"
        or meta["inner_learning_rate"] != 0.001
        or meta["inner_steps"] != 1
        or meta["meta_test_coefficient"] != 1.0
        or not meta["second_order"]
        or meta["domain_classifier"]
        or meta["gradient_projection"]
        or meta["latent_alignment"]
        or meta["loss_sweep"]
        or safety["minimum_fold_layer_non_harm_cells"] != 8
        or safety["total_fold_layer_cells"] != 9
        or safety["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or not amendment.is_file()
        or v12.sha256_file(amendment)
        != config["authorization_evidence"]["prospective_gate_amendment_sha256"]
        or not audit.is_file()
        or json.loads(audit.read_text(encoding="utf-8"))["status"]
        != config["authorization_evidence"]["required_status"]
    ):
        raise v12.ContractError("v33 fixed scientific contract drift")
    return config


def encode_layer_month_environments(
    layer: np.ndarray, local_time: pd.Series | pd.DatetimeIndex
) -> tuple[np.ndarray, dict[int, str]]:
    layers = np.asarray(layer, dtype=int)
    months = pd.DatetimeIndex(local_time).month.to_numpy(dtype=int)
    pairs = sorted(set(zip(layers.tolist(), months.tolist(), strict=True)))
    mapping = {pair: index for index, pair in enumerate(pairs)}
    encoded = np.fromiter(
        (mapping[(int(target_layer), int(month))] for target_layer, month in zip(layers, months, strict=True)),
        dtype=np.int16,
        count=len(layers),
    )
    labels = {
        index: f"layer{target_layer}:month{month:02d}"
        for (target_layer, month), index in mapping.items()
    }
    if len(encoded) != len(layers) or len(labels) < 2:
        raise v12.ContractError("v33 environment encoding failed")
    return encoded, labels


def domain_balanced_weights(
    layer: np.ndarray, local_time: pd.Series | pd.DatetimeIndex
) -> tuple[np.ndarray, dict[str, Any]]:
    global _CURRENT_ENVIRONMENTS, _CURRENT_ENVIRONMENT_LABELS
    weights, receipt = _BASE_DOMAIN_BALANCED_WEIGHTS(layer, local_time)
    environments, labels = encode_layer_month_environments(layer, local_time)
    _CURRENT_ENVIRONMENTS = environments
    _CURRENT_ENVIRONMENT_LABELS = labels
    counts = {labels[int(value)]: int(np.sum(environments == value)) for value in np.unique(environments)}
    receipt["mldg_environment"] = {
        "definition": "target_layer_x_calendar_month",
        "count": len(labels),
        "labels": {str(key): value for key, value in labels.items()},
        "rows": counts,
    }
    return weights, receipt


def weighted_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = F.smooth_l1_loss(prediction, target, beta=1.0, reduction="none")
    loss = (raw * weights).sum() / weights.sum().clamp_min(1e-12)
    return loss, raw


def select_virtual_domain(environments: torch.Tensor, cycle_index: int) -> int:
    present = torch.unique(environments.detach().cpu(), sorted=True).tolist()
    if len(present) < 2:
        raise v12.ContractError("v33 batch has fewer than two virtual domains")
    return int(present[int(cycle_index) % len(present)])


def mldg_batch_objective(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    environments: torch.Tensor,
    cycle_index: int,
    inner_learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    held_domain = select_virtual_domain(environments, cycle_index)
    meta_test = environments.eq(held_domain)
    meta_train = ~meta_test
    if not bool(meta_train.any()) or not bool(meta_test.any()):
        raise v12.ContractError("v33 meta-train/meta-test split is empty")

    base_prediction = model(tokens, mask, context)
    base_loss, raw = weighted_smooth_l1(base_prediction, target, weights)
    meta_train_loss, _ = weighted_smooth_l1(
        base_prediction[meta_train], target[meta_train], weights[meta_train]
    )
    parameters = dict(model.named_parameters())
    gradients = torch.autograd.grad(
        meta_train_loss, tuple(parameters.values()), create_graph=True
    )
    if not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise v12.ContractError("v33 non-finite differentiable inner gradient")
    adapted = {
        name: value - float(inner_learning_rate) * gradient
        for (name, value), gradient in zip(parameters.items(), gradients, strict=True)
    }
    if not all(bool(torch.isfinite(value).all()) for value in adapted.values()):
        raise v12.ContractError("v33 non-finite adapted parameter")
    adapted_prediction = torch.func.functional_call(
        model,
        adapted,
        (tokens[meta_test], mask[meta_test], context[meta_test]),
    )
    meta_test_loss, _ = weighted_smooth_l1(
        adapted_prediction, target[meta_test], weights[meta_test]
    )
    outer = meta_train_loss + meta_test_loss
    if not bool(torch.isfinite(outer)):
        raise v12.ContractError("v33 non-finite MLDG outer objective")
    inner_norm = float(
        torch.sqrt(sum(value.detach().square().sum() for value in gradients)).cpu()
    )
    return outer, raw, {
        "held_domain": held_domain,
        "present_domain_count": int(torch.unique(environments).numel()),
        "meta_train_rows": int(meta_train.sum()),
        "meta_test_rows": int(meta_test.sum()),
        "base_loss": float(base_loss.detach().cpu()),
        "meta_train_loss": float(meta_train_loss.detach().cpu()),
        "meta_test_loss": float(meta_test_loss.detach().cpu()),
        "inner_gradient_norm": inner_norm,
        "adapted_parameters_finite": True,
        "second_order_graph": bool(meta_test_loss.grad_fn is not None),
    }


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
    if _CURRENT_ENVIRONMENTS is None or len(_CURRENT_ENVIRONMENTS) != len(target):
        raise v12.ContractError("v33 environment contract was not initialized")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = v12.VerticalDeepSet(8, 11, hidden=32).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            _CURRENT_ENVIRONMENTS.astype(np.int64),
        )
    )
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    inner_learning_rate = float(
        config["training"]["meta_objective"]["inner_learning_rate"]
    )
    steps_per_epoch = int(np.ceil(len(target) / batch_size))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    meta_train_losses: list[float] = []
    meta_test_losses: list[float] = []
    environment_steps = {int(value): 0 for value in np.unique(_CURRENT_ENVIRONMENTS)}
    optimizer_steps = 0
    minimum_present_domains = len(environment_steps)
    maximum_inner_gradient_norm = 0.0
    maximum_outer_gradient_norm = 0.0
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(target), generator=generator)
        numerator = 0.0
        denominator = 0.0
        epoch_meta_train = 0.0
        epoch_meta_test = 0.0
        epoch_steps = 0
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            cycle_index = epoch * steps_per_epoch + batch_index
            optimizer.zero_grad(set_to_none=True)
            objective, raw_loss, receipt = mldg_batch_objective(
                model,
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[5],
                cycle_index,
                inner_learning_rate,
            )
            objective.backward()
            gradients = [value.grad for value in model.parameters()]
            if any(value is None for value in gradients) or not all(
                bool(torch.isfinite(value).all()) for value in gradients if value is not None
            ):
                raise v12.ContractError("v33 non-finite second-order outer gradient")
            outer_norm = float(
                torch.sqrt(
                    sum(value.detach().square().sum() for value in gradients if value is not None)
                ).cpu()
            )
            optimizer.step()
            optimizer_steps += 1
            environment_steps[receipt["held_domain"]] += 1
            minimum_present_domains = min(
                minimum_present_domains, receipt["present_domain_count"]
            )
            maximum_inner_gradient_norm = max(
                maximum_inner_gradient_norm, receipt["inner_gradient_norm"]
            )
            maximum_outer_gradient_norm = max(maximum_outer_gradient_norm, outer_norm)
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
            epoch_meta_train += receipt["meta_train_loss"]
            epoch_meta_test += receipt["meta_test_loss"]
            epoch_steps += 1
        losses.append(numerator / denominator)
        meta_train_losses.append(epoch_meta_train / epoch_steps)
        meta_test_losses.append(epoch_meta_test / epoch_steps)

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
    finite = bool(
        np.isfinite(losses).all()
        and np.isfinite(meta_train_losses).all()
        and np.isfinite(meta_test_losses).all()
        and np.isfinite(prediction).all()
        and np.isfinite(maximum_inner_gradient_norm)
        and np.isfinite(maximum_outer_gradient_norm)
    )
    if not finite or min(environment_steps.values()) < 1:
        raise v12.ContractError("v33 MLDG training/coverage contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "meta_train_loss_first": meta_train_losses[0],
        "meta_train_loss_last": meta_train_losses[-1],
        "meta_test_loss_first": meta_test_losses[0],
        "meta_test_loss_last": meta_test_losses[-1],
        "optimizer_steps": optimizer_steps,
        "second_order_steps": optimizer_steps,
        "inner_steps_per_outer_step": 1,
        "inner_learning_rate": inner_learning_rate,
        "meta_test_coefficient": 1.0,
        "minimum_present_domains_per_batch": minimum_present_domains,
        "held_domain_steps": {
            _CURRENT_ENVIRONMENT_LABELS[key]: value
            for key, value in environment_steps.items()
        },
        "maximum_inner_gradient_norm": maximum_inner_gradient_norm,
        "maximum_outer_gradient_norm": maximum_outer_gradient_norm,
        "loss": "weighted_SmoothL1_beta_1.0_MLDG",
        "loss_finite": finite,
        "second_order": True,
        "row_deletion": 0,
    }


def _mldg_scalar_contract_receipt() -> dict[str, Any]:
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    train_prediction = theta * torch.tensor([1.0, 2.0], dtype=torch.float64)
    train_target = torch.tensor([0.0, 0.5], dtype=torch.float64)
    train_loss = F.smooth_l1_loss(
        train_prediction, train_target, beta=1.0, reduction="mean"
    )
    inner_gradient = torch.autograd.grad(train_loss, theta, create_graph=True)[0]
    adapted = theta - 0.001 * inner_gradient
    test_prediction = adapted * torch.tensor([3.0], dtype=torch.float64)
    test_loss = F.smooth_l1_loss(
        test_prediction, torch.tensor([0.25], dtype=torch.float64), beta=1.0
    )
    outer = train_loss + test_loss
    outer_gradient = torch.autograd.grad(outer, theta)[0]
    values = [train_loss, inner_gradient, adapted, test_loss, outer, outer_gradient]
    finite = all(bool(torch.isfinite(value)) for value in values)
    if not finite or test_loss.grad_fn is None:
        raise v12.ContractError("v33 scalar second-order contract failed")
    return {
        "inner_learning_rate": 0.001,
        "inner_steps": 1,
        "meta_test_coefficient": 1.0,
        "train_loss": float(train_loss.detach()),
        "inner_gradient": float(inner_gradient.detach()),
        "adapted_parameter": float(adapted.detach()),
        "meta_test_loss": float(test_loss.detach()),
        "outer_loss": float(outer.detach()),
        "outer_gradient": float(outer_gradient.detach()),
        "second_order_graph": True,
        "all_finite": finite,
    }


def _virtual_domain_cycle_receipt() -> dict[str, Any]:
    environments = torch.tensor([5, 2, 5, 9, 2, 9], dtype=torch.int64)
    sequence = [select_virtual_domain(environments, index) for index in range(6)]
    expected = [2, 5, 9, 2, 5, 9]
    if sequence != expected:
        raise v12.ContractError("v33 deterministic virtual-domain cycle failed")
    return {
        "present_sorted": [2, 5, 9],
        "first_two_cycles": sequence,
        "expected": expected,
        "deterministic": True,
    }


def _isolation_receipt() -> dict[str, Any]:
    return v28._isolation_receipt()


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for key, relative in config["authorization_evidence"].items():
        if not key.endswith("_result") and key != "prospective_gate_amendment":
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v18_group_dro_distinguished": True,
        "v19_vrex_distinguished": True,
        "v20_coral_distinguished": True,
        "v28_pcgrad_distinguished": True,
        "v30_irmv1_distinguished": True,
        "v31_dann_distinguished": True,
        "v32_weighted_mse_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def prospective_fold_layer_gate(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    return v28.prospective_fold_layer_gate(record, config)


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    isolation = _isolation_receipt()
    if max(isolation.values()) > 1e-6:
        raise v12.ContractError("v33 masked/future or permutation isolation failed")
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "mldg_scalar_contract": _mldg_scalar_contract_receipt(),
        "virtual_domain_cycle": _virtual_domain_cycle_receipt(),
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
    if "prospective_fold_layer_gate" not in item:
        return
    folds = item["by_fold"]
    local = item["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v33 virtual-domain MLDG DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE `{item['delta_rmse']:+.9f} C`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"fold delta RMSE: Sep-Oct `{folds['2024_sep_oct']['delta_rmse']:+.9f}`, "
        f"Jul-Aug `{folds['2025_jul_aug']['delta_rmse']:+.9f}`, "
        f"Nov-Dec `{folds['2025_nov_dec']['delta_rmse']:+.9f}`.\n\n"
        f"prospective fold x layer gate: `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13 SmoothL1 pipeline에 deterministic layer x month virtual meta-test와 "
        "one differentiable base-LR inner step만 추가했다. Li et al. (AAAI 2018)은 "
        "domain-shift simulation 동기만 제공하며 P2 성능 근거가 아니다. sweep/router/"
        "ensemble/row deletion/official-feedback selection/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_VIRTUAL_DOMAIN_MLDG"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["mldg_contract"] = {
        "scalar": _mldg_scalar_contract_receipt(),
        "virtual_domain_cycle": _virtual_domain_cycle_receipt(),
        "environment": "target_layer_x_calendar_month",
        "inner_learning_rate": 0.001,
        "inner_steps": 1,
        "meta_test_coefficient": 1.0,
        "second_order": True,
    }
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "meta_objective": config["training"]["meta_objective"],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
        }
    )
    result["comparison_to_preserved_candidates"] = {
        "use": "ledger_only_no_posthoc_selection_router_or_ensemble",
        "official_v23_feedback_used_for_selection": False,
        "v13_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v13_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v20_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v20_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v23_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v23_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v32_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v32_result"]).read_text(
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
