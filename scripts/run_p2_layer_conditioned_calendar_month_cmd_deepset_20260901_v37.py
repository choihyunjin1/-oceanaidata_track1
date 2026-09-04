"""Run sealed P2 v37 layer-conditioned calendar-month CMD exactly once."""

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

import run_p2_layer_month_fishr_gradient_variance_deepset_20260901_v36 as v36  # noqa: E402

v12 = v36.v12
v13 = v36.v13
v28 = v36.v28

EXPERIMENT_ID = "p2_layer_conditioned_calendar_month_cmd_deepset_20260901_v37"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V37_LAYER_CONDITIONED_MONTH_CMD_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.layer_conditioned_calendar_month_cmd_deepset.result.20260901.v37"

_BASE_LOAD_CONFIG = v36._BASE_LOAD_CONFIG
_BASE_RUN = v36._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v36._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v36._V13_RUNNER
_CURRENT_ENVIRONMENT_IDS: np.ndarray | None = None


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
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
    ):
        raise v12.ContractError("v37 preregistration identity drift")
    training = config["training"]
    cmd = training["central_moment_discrepancy"]
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    audit = ROOT / config["authorization_evidence"]["audit_result"]
    if (
        training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"]
        != "weighted_SmoothL1_beta_1.0_plus_fixed_layer_conditioned_calendar_month_CMD_orders1to5"
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
        or training["input_perturbation"]
        or training["data_augmentation"]
        or cmd["coefficient"] != 1.0
        or cmd["environment"] != "target_layer_x_KST_calendar_month"
        or cmd["representation"] != "tanh_of_v13_first_head_hidden_width32"
        or cmd["orders"] != [1, 2, 3, 4, 5]
        or cmd["reference"] != "all_rows_of_same_target_layer_in_batch"
        or cmd["minimum_rows_per_month_in_batch"] != 2
        or cmd["minimum_supported_months_per_layer_in_batch"] != 2
        or cmd["kernel"]
        or cmd["adversary"]
        or cmd["ema"]
        or cmd["warmup"]
        or cmd["sweep"]
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
        raise v12.ContractError("v37 fixed scientific contract drift")
    return config


def environment_ids(layer: np.ndarray, local_time: Any) -> np.ndarray:
    return v36.environment_ids(layer, local_time)


def domain_balanced_weights(
    layer: np.ndarray, local_time: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    global _CURRENT_ENVIRONMENT_IDS
    weights, receipt = _BASE_DOMAIN_BALANCED_WEIGHTS(layer, local_time)
    _CURRENT_ENVIRONMENT_IDS = environment_ids(layer, local_time)
    unique, counts = np.unique(_CURRENT_ENVIRONMENT_IDS, return_counts=True)
    receipt["cmd_environment_contract"] = {
        "encoding": "100*target_layer+KST_calendar_month",
        "environment_ids": unique.tolist(),
        "environment_rows": counts.tolist(),
        "environment_count": len(unique),
    }
    return weights, receipt


def forward_with_latent(
    model: v12.VerticalDeepSet,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = model.element(tokens)
    expanded_mask = token_mask.unsqueeze(-1)
    count = expanded_mask.sum(dim=1).clamp_min(1.0)
    mean = (encoded * expanded_mask).sum(dim=1) / count
    negative = torch.finfo(encoded.dtype).min
    maximum = encoded.masked_fill(~expanded_mask.bool(), negative).amax(dim=1)
    maximum = torch.where(
        torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
    )
    pooled = torch.cat((mean, maximum, context), dim=1)
    latent = model.head[1](model.head[0](pooled))
    hidden = model.head[3](model.head[2](latent))
    prediction = model.head[4](hidden).squeeze(1)
    return prediction, latent


def _moment(sample: torch.Tensor, order: int) -> torch.Tensor:
    if order == 1:
        return sample.mean(dim=0)
    centered = sample - sample.mean(dim=0, keepdim=True)
    return centered.pow(order).mean(dim=0)


def layer_conditioned_month_cmd(
    latent: torch.Tensor,
    environment: torch.Tensor,
    orders: tuple[int, ...] = (1, 2, 3, 4, 5),
    minimum_rows: int = 2,
    minimum_months: int = 2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if latent.ndim != 2 or environment.ndim != 1 or len(latent) != len(environment):
        raise v12.ContractError("v37 CMD tensor contract failed")
    if orders != (1, 2, 3, 4, 5):
        raise v12.ContractError("v37 CMD order contract failed")
    bounded = torch.tanh(latent)
    terms: list[torch.Tensor] = []
    supported: list[int] = []
    skipped: list[int] = []
    supported_layers: list[int] = []
    dimension_scale = float(bounded.shape[1]) ** 0.5
    for target_layer in sorted({int(value) // 100 for value in environment.tolist()}):
        layer_selected = environment.div(100, rounding_mode="floor") == target_layer
        layer_ids = torch.unique(environment[layer_selected], sorted=True)
        month_ids = [
            value for value in layer_ids if int((environment == value).sum()) >= minimum_rows
        ]
        skipped.extend(
            int(value.detach().cpu()) for value in layer_ids if value not in month_ids
        )
        if len(month_ids) < minimum_months:
            skipped.extend(int(value.detach().cpu()) for value in month_ids)
            continue
        supported_layers.append(target_layer)
        reference = bounded[layer_selected]
        reference_moments = {order: _moment(reference, order) for order in orders}
        for value in month_ids:
            month = bounded[environment == value]
            supported.append(int(value.detach().cpu()))
            for order in orders:
                distance = torch.linalg.vector_norm(
                    _moment(month, order) - reference_moments[order], ord=2
                ) / dimension_scale
                terms.append(distance)
    if not terms:
        raise v12.ContractError("v37 no supported layer-month CMD terms")
    penalty = torch.stack(terms).mean()
    finite = bool(torch.isfinite(penalty))
    if not finite:
        raise v12.ContractError("v37 CMD penalty is non-finite")
    return penalty, {
        "bounded_latent_dimension": int(bounded.shape[1]),
        "orders": list(orders),
        "supported_environment_ids": sorted(set(supported)),
        "supported_environment_count": len(set(supported)),
        "supported_target_layers": supported_layers,
        "supported_target_layer_count": len(supported_layers),
        "skipped_environment_ids": sorted(set(skipped)),
        "skipped_environment_count": len(set(skipped)),
        "moment_term_count": len(terms),
        "penalty_finite": finite,
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
    if _CURRENT_ENVIRONMENT_IDS is None or len(_CURRENT_ENVIRONMENT_IDS) != len(target):
        raise v12.ContractError("v37 environment commitment missing or misaligned")
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
            _CURRENT_ENVIRONMENT_IDS,
        )
    )
    batch_size = int(config["training"]["batch_size"])
    contract = config["training"]["central_moment_discrepancy"]
    coefficient = float(contract["coefficient"])
    minimum_rows = int(contract["minimum_rows_per_month_in_batch"])
    minimum_months = int(contract["minimum_supported_months_per_layer_in_batch"])
    orders = tuple(int(value) for value in contract["orders"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base_losses: list[float] = []
    penalties: list[float] = []
    total_losses: list[float] = []
    supported_environment_min = 10**9
    supported_layer_min = 10**9
    term_min = 10**9
    optimizer_steps = 0
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        base_numerator = 0.0
        weight_denominator = 0.0
        penalty_total = 0.0
        total_loss_total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            prediction, latent = forward_with_latent(
                model, batch[0], batch[1], batch[2]
            )
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            base_loss = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            penalty, receipt = layer_conditioned_month_cmd(
                latent,
                batch[5],
                orders=orders,
                minimum_rows=minimum_rows,
                minimum_months=minimum_months,
            )
            loss = base_loss + coefficient * penalty
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v37 total training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            supported_environment_min = min(
                supported_environment_min, receipt["supported_environment_count"]
            )
            supported_layer_min = min(
                supported_layer_min, receipt["supported_target_layer_count"]
            )
            term_min = min(term_min, receipt["moment_term_count"])
            base_numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            weight_denominator += float(batch[4].sum().cpu())
            penalty_total += float(penalty.detach().cpu())
            total_loss_total += float(loss.detach().cpu())
            batches += 1
        base_losses.append(base_numerator / weight_denominator)
        penalties.append(penalty_total / batches)
        total_losses.append(total_loss_total / batches)

    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_tokens), batch_size):
            stop = start + batch_size
            prediction, _latent = forward_with_latent(
                model,
                torch.from_numpy(query_tokens[start:stop]).to(device),
                torch.from_numpy(query_mask[start:stop]).to(device),
                torch.from_numpy(query_context[start:stop]).to(device),
            )
            output.append(prediction.cpu().numpy())
    prediction_array = np.concatenate(output).astype(float)
    finite = bool(
        np.isfinite(base_losses).all()
        and np.isfinite(penalties).all()
        and np.isfinite(total_losses).all()
        and np.isfinite(prediction_array).all()
    )
    if not finite or supported_environment_min < 2 or supported_layer_min < 1:
        raise v12.ContractError("v37 CMD training contract failed")
    return prediction_array, {
        "seed": seed,
        "device": str(device),
        "epochs": len(base_losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "base_loss_first": base_losses[0],
        "base_loss_last": base_losses[-1],
        "cmd_penalty_first": penalties[0],
        "cmd_penalty_last": penalties[-1],
        "total_loss_first": total_losses[0],
        "total_loss_last": total_losses[-1],
        "optimizer_steps": optimizer_steps,
        "cmd_penalty_steps": optimizer_steps,
        "coefficient": coefficient,
        "orders": list(orders),
        "bounded_latent_dimension": 32,
        "supported_environment_count_min": supported_environment_min,
        "supported_target_layer_count_min": supported_layer_min,
        "moment_term_count_min": term_min,
        "kernel_steps": 0,
        "adversary_steps": 0,
        "ema_steps": 0,
        "warmup_steps": 0,
        "loss": config["training"]["objective"],
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _cmd_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(37)
    latent = torch.randn(18, 7, requires_grad=True)
    environment = torch.tensor([201] * 3 + [202] * 3 + [203] * 3 + [301] * 3 + [302] * 3 + [303] * 3)
    penalty, receipt = layer_conditioned_month_cmd(latent, environment)
    order = torch.tensor([17, 2, 10, 4, 14, 0, 16, 8, 11, 7, 1, 15, 5, 9, 13, 3, 12, 6])
    permuted, permuted_receipt = layer_conditioned_month_cmd(
        latent[order], environment[order]
    )
    penalty.backward()
    invariant = bool(
        torch.allclose(penalty.detach(), permuted.detach(), atol=1e-7, rtol=0.0)
    )
    gradients_finite = bool(latent.grad is not None and torch.isfinite(latent.grad).all())
    if not invariant or not gradients_finite or receipt != permuted_receipt:
        raise v12.ContractError("v37 synthetic CMD contract failed")
    return {
        "penalty": float(penalty.detach()),
        "permuted_penalty": float(permuted.detach()),
        "permutation_invariant": invariant,
        "gradients_finite": gradients_finite,
        "coefficient": 1.0,
        **receipt,
    }


def _forward_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(37)
    model = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(5, 5, 8)
    mask = torch.ones(5, 5)
    context = torch.randn(5, 11)
    with torch.inference_mode():
        reference = model(tokens, mask, context)
        candidate, latent = forward_with_latent(model, tokens, mask, context)
    maximum_error = float(torch.max(torch.abs(reference - candidate)))
    if maximum_error > 1e-7 or latent.shape != (5, 32):
        raise v12.ContractError("v37 latent forward contract failed")
    return {
        "prediction_maximum_abs_error": maximum_error,
        "latent_shape": list(latent.shape),
        "bounded_latent_maximum_abs_value": float(torch.tanh(latent).abs().max()),
    }


def _isolation_receipt() -> dict[str, Any]:
    return v28._isolation_receipt()


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for key, relative in config["authorization_evidence"].items():
        if not (key.endswith("_result") or key == "prospective_gate_amendment"):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v20_covariance_only_distinguished": True,
        "v31_adversarial_distinguished": True,
        "v36_gradient_variance_distinguished": True,
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
        raise v12.ContractError("v37 masked/future or permutation isolation failed")
    synthetic_ids = environment_ids(
        np.array([2, 2, 3, 4]),
        pd.DatetimeIndex(
            [
                "2025-01-01T00:00:00+09:00",
                "2025-02-01T00:00:00+09:00",
                "2025-01-01T00:00:00+09:00",
                "2025-12-01T00:00:00+09:00",
            ]
        ),
    )
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "cmd_contract": _cmd_contract_receipt(),
        "forward_contract": _forward_contract_receipt(),
        "environment_contract": {
            "synthetic_ids": synthetic_ids.tolist(),
            "expected_ids": [201, 202, 301, 412],
            "exact": synthetic_ids.tolist() == [201, 202, 301, 412],
        },
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
    if not payload["environment_contract"]["exact"]:
        raise v12.ContractError("v37 synthetic environment encoding failed")
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
        "# P2 v37 layer-conditioned calendar-month CMD DeepSets\n\n"
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
        "Exact v13에 coefficient 1.0의 bounded latent calendar-month CMD "
        "orders 1-5만 추가했다. Zellinger et al. (ICLR 2017)은 representation "
        "동기만 제공하며 P2 성능 근거가 아니다. kernel/adversary/EMA/sweep/"
        "router/ensemble/official-feedback selection/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_LAYER_CONDITIONED_MONTH_CMD"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["cmd_contract"] = _cmd_contract_receipt()
    result["forward_contract"] = _forward_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "central_moment_discrepancy": config["training"][
                "central_moment_discrepancy"
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
            for name in ("v13", "v20", "v31", "v36")
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
