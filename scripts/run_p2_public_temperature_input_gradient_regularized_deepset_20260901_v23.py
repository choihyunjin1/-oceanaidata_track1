"""Run the sealed P2 v23 public-temperature input-gradient objective once."""

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
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_public_temperature_input_gradient_regularized_deepset_20260901_v23"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V23_PUBLIC_TEMP_INPUT_GRADIENT_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.public_temperature_input_gradient_regularized_deepset.result.20260901.v23"

_BASE_LOAD_CONFIG = v13.load_config
_BASE_RUN = v13.run
_V13_RUNNER = v13.RUNNER


def _bind_base() -> None:
    v13.EXPERIMENT_ID = EXPERIMENT_ID
    v13.CONFIG = CONFIG
    v13.ARTIFACT = ARTIFACT
    v13.REPORT = REPORT
    v13.RUNNER = RUNNER
    v13.PREDICTION_NAME = PREDICTION_NAME
    v13.load_config = load_config
    v13.train_predict_seed = train_predict_seed
    v13.write_report = write_report


def load_config() -> dict[str, Any]:
    config = _BASE_LOAD_CONFIG()
    training = config["training"]
    gradient = training["input_gradient"]
    if (
        training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["epochs"] != 60
        or training["seeds"] != [20260901, 20260902, 20260903]
        or training["maximum_fit_count"] != 9
        or training["maximum_final_action_C"] != 0.5
        or training["input_perturbation"]
        or gradient["coefficient"] != 0.01
        or gradient["token_channel"] != 0
        or not gradient["observed_token_mask_only"]
        or gradient["penalize_psal_depth_nominal_presence_context"]
        or gradient["coefficient_sweep"]
    ):
        raise v12.ContractError("v23 fixed scientific contract drift")
    return config


def observed_temperature_gradient_penalty(
    per_row_loss: Tensor,
    tokens: Tensor,
    token_mask: Tensor,
    row_weights: Tensor,
) -> Tensor:
    """L2 loss-gradient penalty on observed public-temperature channel only."""
    if per_row_loss.ndim != 1 or row_weights.shape != per_row_loss.shape:
        raise v12.ContractError("per-row loss/weight geometry drift")
    if tokens.ndim != 3 or token_mask.shape != tokens.shape[:2]:
        raise v12.ContractError("token geometry drift")
    gradient = torch.autograd.grad(
        per_row_loss.sum(), tokens, create_graph=True, retain_graph=True
    )[0]
    observed = token_mask.to(dtype=gradient.dtype)
    per_row = (gradient[..., 0].square() * observed).sum(dim=1)
    per_row = per_row / observed.sum(dim=1).clamp_min(1.0)
    return (per_row * row_weights).sum() / row_weights.sum().clamp_min(1e-12)


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
    coefficient = float(config["training"]["input_gradient"]["coefficient"])
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    data_losses: list[float] = []
    penalties: list[float] = []
    objectives: list[float] = []
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        data_numerator = 0.0
        penalty_numerator = 0.0
        denominator = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            batch_tokens = batch[0].detach().clone().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_tokens, batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            data_loss = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            gradient_penalty = observed_temperature_gradient_penalty(
                raw_loss, batch_tokens, batch[1], batch[4]
            )
            objective = data_loss + coefficient * gradient_penalty
            objective.backward()
            optimizer.step()
            weight_sum = float(batch[4].sum().detach().cpu())
            data_numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            penalty_numerator += float(gradient_penalty.detach().cpu())
            denominator += weight_sum
            batches += 1
        epoch_data = data_numerator / denominator
        epoch_penalty = penalty_numerator / max(batches, 1)
        data_losses.append(epoch_data)
        penalties.append(epoch_penalty)
        objectives.append(epoch_data + coefficient * epoch_penalty)
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
        np.isfinite(data_losses).all()
        and np.isfinite(penalties).all()
        and np.isfinite(objectives).all()
        and np.isfinite(prediction).all()
    )
    if not finite:
        raise v12.ContractError("v23 training or prediction is non-finite")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(data_losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "data_loss_first": data_losses[0],
        "data_loss_last": data_losses[-1],
        "input_gradient_penalty_first": penalties[0],
        "input_gradient_penalty_last": penalties[-1],
        "objective_first": objectives[0],
        "objective_last": objectives[-1],
        "input_gradient_coefficient": coefficient,
        "penalized_token_channel": 0,
        "non_temperature_channels_penalized": 0,
        "observed_token_mask_only": True,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _masked_token_isolation_receipt() -> dict[str, Any]:
    torch.manual_seed(23)
    model = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(4, 5, 8)
    mask = torch.ones(4, 5)
    mask[:, -1] = 0.0
    context = torch.randn(4, 11)
    changed = tokens.clone()
    changed[:, -1] += 1000.0
    with torch.inference_mode():
        left = model(tokens, mask, context)
        right = model(changed, mask, context)
    error = float(torch.max(torch.abs(left - right)))
    if error > 1e-6:
        raise v12.ContractError("masked public token changes prediction")
    return {"maximum_abs_error": error, "perturbed_masked_tokens": 4}


def _gradient_scope_receipt() -> dict[str, Any]:
    tokens = torch.ones(2, 3, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    weights = torch.tensor([1.0, 2.0])
    per_row = (2.0 * tokens[..., 0]).sum(dim=1).square()
    per_row = per_row + (19.0 * tokens[..., 1]).sum(dim=1).square()
    penalty = observed_temperature_gradient_penalty(per_row, tokens, mask, weights)
    if not torch.isfinite(penalty) or float(penalty.detach()) <= 0.0:
        raise v12.ContractError("synthetic gradient penalty is nonpositive or non-finite")
    changed = torch.ones(2, 3, 8, requires_grad=True)
    changed_loss = (2.0 * changed[..., 0]).sum(dim=1).square()
    changed_loss = changed_loss + (1900.0 * changed[..., 1]).sum(dim=1).square()
    changed_penalty = observed_temperature_gradient_penalty(
        changed_loss, changed, mask, weights
    )
    error = float(torch.abs(penalty - changed_penalty).detach())
    if error > 1e-8:
        raise v12.ContractError("non-temperature channel entered the penalty")
    return {
        "penalty": float(penalty.detach()),
        "non_temperature_coefficient_invariance_error": error,
        "penalized_channel": 0,
        "masked_tokens_excluded": True,
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative in (
        config["authorization_evidence"]["duplicate_audit"],
        config["authorization_evidence"]["v13_result"],
        config["authorization_evidence"]["v20_result"],
        "src/p2_restore/dynamic_sigmoid_profile.py",
        "scripts/run_p2_public_sensor_influence_shrink_20260901_v10.py",
        "scripts/run_p2_supported_layer_change_coherence_20260901_v11r1.py",
        "scripts/run_p2_fixed_physical_depth_graph_message_passing_20260901_v16.py",
        "scripts/run_p2_fixed_student_t_robust_deepset_20260901_v21.py",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v22_output_depth_derivative_distinguished": True,
        "v10_v11_post_model_action_rules_distinguished": True,
        "v16_graph_representation_distinguished": True,
        "dynamic_sigmoid_solver_conditioning_distinguished": True,
        "v21_student_t_likelihood_distinguished": True,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "gradient_scope": _gradient_scope_receipt(),
        "masked_token_isolation": _masked_token_isolation_receipt(),
        "permutation_invariance": v12.permutation_invariance_receipt(),
        "prefix_cutoffs": {
            fold: (pd.Timestamp(start) - pd.Timedelta(days=7)).isoformat()
            for fold, start in config["training"]["fold_starts_kst"].items()
        },
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "audit_result_sha256": v12.sha256_file(audit_path),
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
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v23 public-temperature input-gradient DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled ΔRMSE `{item['delta_rmse']:+.9f} C`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점.\n\n"
        f"fold ΔRMSE: Sep-Oct `{folds['2024_sep_oct']['delta_rmse']:+.9f}`, "
        f"Jul-Aug `{folds['2025_jul_aug']['delta_rmse']:+.9f}`, "
        f"Nov-Dec `{folds['2025_nov_dec']['delta_rmse']:+.9f}`.\n\n"
        "v13 architecture/domain weights/prefix+7d purge/seeds/epochs/blend/action cap을 고정하고, "
        "observed public-temperature token에 대한 loss-input-gradient L2 penalty(lambda=0.01)만 추가했다. "
        "Ross and Doshi-Velez (AAAI 2018, DOI 10.1609/aaai.v32i1.11504)는 representation "
        "동기만 제공하며 P2 성능 근거가 아니다. row deletion/parameter sweep/router/official/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    _bind_base()
    started = time.perf_counter()
    result = _BASE_RUN()
    config = load_config()
    record = result["candidate"]
    passed = bool(record["strict_exploratory_pass"] and record["safety_pass"])
    result["schema_version"] = RESULT_SCHEMA
    result["status"] = (
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if passed
        else "EXPLORATORY_NO_GO_PUBLIC_TEMPERATURE_INPUT_GRADIENT"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "input_gradient": config["training"]["input_gradient"],
            "input_perturbation": 0,
        }
    )
    result["comparison_to_preserved_candidates"] = {
        "use": "ledger_only_no_posthoc_selection_router_or_ensemble",
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
    }
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    result["hashes"]["v22_duplicate_audit"] = v12.sha256_file(
        ROOT / config["authorization_evidence"]["duplicate_audit"]
    )
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
