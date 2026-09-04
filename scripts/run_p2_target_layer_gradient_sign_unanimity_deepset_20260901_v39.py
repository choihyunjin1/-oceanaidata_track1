"""Run sealed P2 v39 target-layer gradient-sign unanimity exactly once."""

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

import run_p2_spectral_decoupled_output_regularized_deepset_20260901_v38 as v38  # noqa: E402

v12 = v38.v12
v13 = v38.v13
v37 = v38.v37

EXPERIMENT_ID = "p2_target_layer_gradient_sign_unanimity_deepset_20260901_v39"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V39_TARGET_LAYER_GRADIENT_SIGN_UNANIMITY_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.target_layer_gradient_sign_unanimity_deepset.result.20260901.v39"

_BASE_RUN = v38._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v37._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v38._V13_RUNNER
_CURRENT_TARGET_LAYERS: np.ndarray | None = None


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
    mask = training["gradient_sign_unanimity"]
    safety = config["evaluation"]["safety_gate"]
    audit = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"]
        != "per_target_layer_weighted_SmoothL1_gradients_with_fixed_parameter_coordinate_sign_unanimity"
        or training["optimizer"] != "exact_v13_AdamW_with_gradient_replacement_only"
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
        or mask["tasks"] != [2, 3, 4]
        or mask["task_definition"] != "target_layer"
        or mask["task_order_used_for_result"]
        or mask["agreement_threshold"] != 1.0
        or mask["formula"] != "mean_layer_gradient_times_abs_mean_sign_ge_1"
        or not mask["zero_gradient_counts_as_non_unanimous"]
        or mask["projection"]
        or mask["partial_agreement"]
        or mask["group_reweighting"]
        or mask["sweep"]
        or safety["minimum_fold_layer_non_harm_cells"] != 8
        or safety["total_fold_layer_cells"] != 9
        or safety["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or not audit.is_file()
        or json.loads(audit.read_text(encoding="utf-8"))["status"]
        != config["authorization_evidence"]["required_status"]
        or not amendment.is_file()
        or v12.sha256_file(amendment)
        != config["authorization_evidence"]["prospective_gate_amendment_sha256"]
    ):
        raise v12.ContractError("v39 fixed scientific contract drift")
    return config


def domain_balanced_weights(
    layer: np.ndarray, local_time: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    global _CURRENT_TARGET_LAYERS
    layer_array = np.asarray(layer, dtype=np.int64)
    if not np.isin(layer_array, [2, 3, 4]).all():
        raise v12.ContractError("v39 target-layer commitment drift")
    weights, receipt = _BASE_DOMAIN_BALANCED_WEIGHTS(layer_array, local_time)
    _CURRENT_TARGET_LAYERS = layer_array
    unique, counts = np.unique(layer_array, return_counts=True)
    receipt["gradient_task_contract"] = {
        "tasks": unique.tolist(),
        "rows": counts.tolist(),
        "expected_tasks": [2, 3, 4],
        "exact": unique.tolist() == [2, 3, 4],
    }
    if not receipt["gradient_task_contract"]["exact"]:
        raise v12.ContractError("v39 missing target-layer gradient task")
    return weights, receipt


def and_mask_gradients(
    task_gradients: list[list[torch.Tensor]], threshold: float = 1.0
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if len(task_gradients) != 3 or threshold != 1.0:
        raise v12.ContractError("v39 task-count/threshold contract failed")
    parameter_count = len(task_gradients[0])
    if parameter_count == 0 or any(
        len(gradients) != parameter_count for gradients in task_gradients
    ):
        raise v12.ContractError("v39 parameter gradient alignment failed")
    masked: list[torch.Tensor] = []
    kept = 0
    total = 0
    finite = True
    for parameter_index in range(parameter_count):
        stacked = torch.stack(
            [gradients[parameter_index] for gradients in task_gradients]
        )
        agreement = torch.abs(torch.sign(stacked).mean(dim=0))
        use = agreement >= threshold - 1e-7
        value = stacked.mean(dim=0) * use
        finite &= bool(torch.isfinite(value).all())
        masked.append(value)
        kept += int(use.sum().detach().cpu())
        total += int(use.numel())
    if not finite:
        raise v12.ContractError("v39 masked gradient is non-finite")
    return masked, {
        "kept_coordinates": kept,
        "total_coordinates": total,
        "kept_share": kept / total,
        "threshold": threshold,
        "task_count": len(task_gradients),
        "finite": finite,
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
    if _CURRENT_TARGET_LAYERS is None or len(_CURRENT_TARGET_LAYERS) != len(target):
        raise v12.ContractError("v39 target-layer commitment missing or misaligned")
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
    parameters = list(model.parameters())
    train = tuple(
        torch.from_numpy(value)
        for value in (
            tokens,
            mask,
            context,
            target.astype(np.float32),
            weights.astype(np.float32),
            _CURRENT_TARGET_LAYERS,
        )
    )
    batch_size = int(config["training"]["batch_size"])
    tasks = [int(value) for value in config["training"]["gradient_sign_unanimity"]["tasks"]]
    threshold = float(
        config["training"]["gradient_sign_unanimity"]["agreement_threshold"]
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base_losses: list[float] = []
    kept_shares: list[float] = []
    optimizer_steps = 0
    task_support_min = {value: 10**9 for value in tasks}
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        numerator = 0.0
        denominator = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            prediction = model(batch[0], batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            task_gradients: list[list[torch.Tensor]] = []
            for index, target_layer in enumerate(tasks):
                task_selected = batch[5] == target_layer
                support = int(task_selected.sum().detach().cpu())
                task_support_min[target_layer] = min(
                    task_support_min[target_layer], support
                )
                if support < 2:
                    raise v12.ContractError("v39 insufficient batch task support")
                task_loss = (
                    raw_loss[task_selected] * batch[4][task_selected]
                ).sum() / batch[4][task_selected].sum().clamp_min(1e-12)
                gradients = torch.autograd.grad(
                    task_loss,
                    parameters,
                    retain_graph=index < len(tasks) - 1,
                )
                task_gradients.append([value.detach() for value in gradients])
            masked, receipt = and_mask_gradients(task_gradients, threshold)
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, masked, strict=True):
                parameter.grad = gradient
            optimizer.step()
            optimizer_steps += 1
            kept_shares.append(receipt["kept_share"])
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
        base_losses.append(numerator / denominator)

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
                ).cpu().numpy()
            )
    prediction_array = np.concatenate(output).astype(float)
    finite = bool(
        np.isfinite(base_losses).all()
        and np.isfinite(kept_shares).all()
        and np.isfinite(prediction_array).all()
    )
    if not finite:
        raise v12.ContractError("v39 AND-mask training contract failed")
    return prediction_array, {
        "seed": seed,
        "device": str(device),
        "epochs": len(base_losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": base_losses[0],
        "loss_last": base_losses[-1],
        "optimizer_steps": optimizer_steps,
        "gradient_mask_steps": optimizer_steps,
        "task_order": tasks,
        "task_order_used_for_result": False,
        "agreement_threshold": threshold,
        "mean_kept_coordinate_share": float(np.mean(kept_shares)),
        "minimum_kept_coordinate_share": float(np.min(kept_shares)),
        "maximum_kept_coordinate_share": float(np.max(kept_shares)),
        "minimum_task_rows_per_batch": task_support_min,
        "projection_steps": 0,
        "partial_agreement_steps": 0,
        "group_reweighting_steps": 0,
        "loss": config["training"]["objective"],
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _and_mask_contract_receipt() -> dict[str, Any]:
    first = [torch.tensor([1.0, 1.0, 0.0, -2.0])]
    second = [torch.tensor([2.0, -1.0, 1.0, -4.0])]
    third = [torch.tensor([3.0, 2.0, 1.0, -6.0])]
    masked, receipt = and_mask_gradients([first, second, third])
    expected = torch.tensor([2.0, 0.0, 0.0, -4.0])
    permuted, permuted_receipt = and_mask_gradients([third, first, second])
    exact = bool(torch.equal(masked[0], expected))
    task_permutation_invariant = bool(torch.equal(masked[0], permuted[0]))
    if not exact or not task_permutation_invariant or receipt != permuted_receipt:
        raise v12.ContractError("v39 synthetic AND-mask contract failed")
    return {
        "masked_gradient": masked[0].tolist(),
        "expected_gradient": expected.tolist(),
        "formula_exact": exact,
        "conflicting_coordinate_zeroed": float(masked[0][1]) == 0.0,
        "zero_coordinate_non_unanimous": float(masked[0][2]) == 0.0,
        "all_agree_mean_identity": float(masked[0][0]) == 2.0,
        "task_permutation_invariant": task_permutation_invariant,
        **receipt,
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for key, relative in config["authorization_evidence"].items():
        if not (
            key.endswith("_result")
            or key in ("prospective_gate_amendment", "p1_v45_report")
        ):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "p1_v45_cross_problem_adjacency_disclosed": True,
        "v18_group_dro_distinguished": True,
        "v19_vrex_distinguished": True,
        "v28_pcgrad_distinguished": True,
        "v30_irm_distinguished": True,
        "v36_fishr_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    isolation = v37._isolation_receipt()
    if max(isolation.values()) > 1e-6:
        raise v12.ContractError("v39 masked/future or permutation isolation failed")
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "and_mask_contract": _and_mask_contract_receipt(),
        "permutation_invariance": v12.permutation_invariance_receipt(),
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
        "# P2 v39 target-layer gradient-sign unanimity\n\n"
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
        "Exact v13의 layer 2/3/4 weighted SmoothL1 gradients에 parameter-coordinate "
        "sign unanimity mask만 적용했다. Parascandolo et al. (ICLR 2021)은 동기만 "
        "제공하며 P2 성능 근거가 아니다. P1-v45 adjacency는 공개했고 P1 결과 이전은 "
        "0이다. projection/partial agreement/sweep/router/ensemble/official/hidden/CSV/"
        "upload=0.\n",
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
        else "EXPLORATORY_NO_GO_TARGET_LAYER_GRADIENT_SIGN_UNANIMITY"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["and_mask_contract"] = _and_mask_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "gradient_sign_unanimity": config["training"][
                "gradient_sign_unanimity"
            ],
            "row_deletion": 0,
            "input_perturbation": 0,
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
            for name in ("v13", "v18", "v19", "v28", "v30", "v36", "v38")
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
