"""Run sealed P2 v38 output spectral-decoupling DeepSets exactly once."""

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

import run_p2_layer_conditioned_calendar_month_cmd_deepset_20260901_v37 as v37  # noqa: E402

v12 = v37.v12
v13 = v37.v13

EXPERIMENT_ID = "p2_spectral_decoupled_output_regularized_deepset_20260901_v38"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V38_SPECTRAL_DECOUPLED_OUTPUT_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.spectral_decoupled_output_regularized_deepset.result.20260901.v38"

_BASE_RUN = v37._BASE_RUN
_V13_RUNNER = v37._V13_RUNNER


def _bind_base() -> None:
    v13.EXPERIMENT_ID = EXPERIMENT_ID
    v13.CONFIG = CONFIG
    v13.ARTIFACT = ARTIFACT
    v13.REPORT = REPORT
    v13.RUNNER = RUNNER
    v13.PREDICTION_NAME = PREDICTION_NAME
    v13.load_config = load_config
    v13.domain_balanced_weights = v37._BASE_DOMAIN_BALANCED_WEIGHTS
    v13.train_predict_seed = train_predict_seed
    v13.write_report = write_report


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    training = config["training"]
    sd = training["spectral_decoupling"]
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
        != "weighted_SmoothL1_beta_1.0_plus_fixed_normalized_residual_output_spectral_decoupling"
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
        or sd["coefficient"] != 0.01
        or sd["formula"]
        != "0.5_times_coefficient_times_weighted_mean_prediction_squared"
        or sd["prediction_space"]
        != "normalized_residual_before_profile_scale_and_blend"
        or not sd["target_independent"]
        or sd["parameter_norm_constraint"]
        or sd["input_gradient"]
        or sd["environment_alignment"]
        or sd["sweep"]
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
        raise v12.ContractError("v38 fixed scientific contract drift")
    return config


def spectral_decoupling_penalty(
    prediction: torch.Tensor,
    weights: torch.Tensor,
    coefficient: float = 0.01,
) -> torch.Tensor:
    if prediction.ndim != 1 or weights.ndim != 1 or len(prediction) != len(weights):
        raise v12.ContractError("v38 output penalty tensor contract failed")
    if coefficient != 0.01 or bool(torch.any(weights <= 0.0)):
        raise v12.ContractError("v38 output penalty coefficient/weight drift")
    weighted_square = (prediction.square() * weights).sum() / weights.sum().clamp_min(
        1e-12
    )
    penalty = 0.5 * coefficient * weighted_square
    if not bool(torch.isfinite(penalty)):
        raise v12.ContractError("v38 output penalty is non-finite")
    return penalty


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
        )
    )
    batch_size = int(config["training"]["batch_size"])
    coefficient = float(config["training"]["spectral_decoupling"]["coefficient"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base_losses: list[float] = []
    penalties: list[float] = []
    total_losses: list[float] = []
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
            prediction = model(batch[0], batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            base_loss = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            penalty = spectral_decoupling_penalty(
                prediction, batch[4], coefficient=coefficient
            )
            loss = base_loss + penalty
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v38 total training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
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
        and np.isfinite(penalties).all()
        and np.isfinite(total_losses).all()
        and np.isfinite(prediction_array).all()
    )
    if not finite:
        raise v12.ContractError("v38 output-penalty training contract failed")
    return prediction_array, {
        "seed": seed,
        "device": str(device),
        "epochs": len(base_losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "base_loss_first": base_losses[0],
        "base_loss_last": base_losses[-1],
        "output_penalty_first": penalties[0],
        "output_penalty_last": penalties[-1],
        "total_loss_first": total_losses[0],
        "total_loss_last": total_losses[-1],
        "optimizer_steps": optimizer_steps,
        "output_penalty_steps": optimizer_steps,
        "coefficient": coefficient,
        "target_dependent_penalty_terms": 0,
        "parameter_norm_steps": 0,
        "input_gradient_steps": 0,
        "environment_alignment_steps": 0,
        "loss": config["training"]["objective"],
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _penalty_contract_receipt() -> dict[str, Any]:
    prediction = torch.tensor([0.0, 1.0, -2.0], requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0])
    penalty = spectral_decoupling_penalty(prediction, weights)
    expected = 0.5 * 0.01 * ((0.0 + 2.0 + 12.0) / 6.0)
    penalty.backward()
    expected_gradient = 0.01 * weights * prediction.detach() / weights.sum()
    zero = torch.zeros(3, requires_grad=True)
    zero_penalty = spectral_decoupling_penalty(zero, weights)
    zero_penalty.backward()
    exact = bool(abs(float(penalty.detach()) - expected) <= 1e-8)
    gradient_exact = bool(
        prediction.grad is not None
        and torch.allclose(prediction.grad, expected_gradient, atol=1e-8, rtol=0.0)
    )
    zero_noop = bool(
        float(zero_penalty.detach()) == 0.0
        and zero.grad is not None
        and torch.equal(zero.grad, torch.zeros_like(zero.grad))
    )
    if not (exact and gradient_exact and zero_noop):
        raise v12.ContractError("v38 synthetic output penalty contract failed")
    return {
        "penalty": float(penalty.detach()),
        "expected_penalty": expected,
        "formula_exact": exact,
        "gradient_exact": gradient_exact,
        "zero_output_noop": zero_noop,
        "coefficient": 0.01,
        "target_values_used": 0,
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for key, relative in config["authorization_evidence"].items():
        if not (
            key.endswith("_result")
            or key in ("prospective_gate_amendment", "p1_v46_report")
        ):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "p1_v46_cross_problem_adjacency_disclosed": True,
        "v21_likelihood_distinguished": True,
        "v23_input_gradient_distinguished": True,
        "v27_parameter_spectral_norm_distinguished": True,
        "v32_weighted_mse_distinguished": True,
        "v37_latent_cmd_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    isolation = v37._isolation_receipt()
    if max(isolation.values()) > 1e-6:
        raise v12.ContractError("v38 masked/future or permutation isolation failed")
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "output_penalty_contract": _penalty_contract_receipt(),
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
        "# P2 v38 normalized-residual spectral decoupling\n\n"
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
        "Exact v13 weighted SmoothL1에 `0.5*0.01*weighted_mean(prediction^2)` "
        "normalized-residual output penalty만 추가했다. Pezeshki et al. (NeurIPS "
        "2021)은 동기만 제공하며 P2 regression 성능 근거가 아니다. P1-v46 인접성은 "
        "공개했고 P1 결과 기반 selection은 0이다. sweep/router/ensemble/official/"
        "hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_SPECTRAL_DECOUPLED_OUTPUT"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["output_penalty_contract"] = _penalty_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "spectral_decoupling": config["training"]["spectral_decoupling"],
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
            for name in ("v13", "v21", "v23", "v32", "v37")
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
