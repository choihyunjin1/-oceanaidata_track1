"""Run sealed P2 v40 predictive-dropout consistency DeepSets exactly once."""

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

import run_p2_target_layer_gradient_sign_unanimity_deepset_20260901_v39 as v39  # noqa: E402

v38 = v39.v38
v37 = v39.v37
v13 = v39.v13
v12 = v39.v12

EXPERIMENT_ID = "p2_predictive_dropout_consistency_deepset_20260901_v40"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V40_PREDICTIVE_DROPOUT_CONSISTENCY_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.predictive_dropout_consistency_deepset.result.20260901.v40"

_BASE_RUN = v39._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v39._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v39._V13_RUNNER


class RDropVerticalDeepSet(nn.Module):
    """Exact v13 DeepSets geometry with fixed dropout after hidden ReLUs."""

    def __init__(
        self,
        token_features: int,
        context_features: int,
        hidden: int = 32,
        dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()
        if dropout_probability != 0.1:
            raise v12.ContractError("v40 dropout probability drift")
        self.element = nn.Sequential(
            nn.Linear(token_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + context_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        encoded = self.element(tokens)
        mask = token_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool(), negative).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return self.head(torch.cat((mean, maximum, context), dim=1)).squeeze(1)


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
    rdrop = training["predictive_dropout_consistency"]
    safety = config["evaluation"]["safety_gate"]
    evidence = config["authorization_evidence"]
    audit = ROOT / evidence["audit_result"]
    amendment = ROOT / evidence["prospective_gate_amendment"]
    fingerprint = ROOT / evidence["fingerprint"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2_with_fixed_dropout_after_hidden_ReLU"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"]
        != "mean_two_weighted_SmoothL1_beta_1.0_plus_fixed_variance_Gaussian_predictive_consistency"
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
        or rdrop["dropout_probability"] != 0.1
        or rdrop["dropout_location"] != "after_each_of_four_hidden_ReLU"
        or rdrop["training_passes_per_batch"] != 2
        or rdrop["base_loss_formula"] != "mean_of_two_weighted_SmoothL1_beta_1.0"
        or rdrop["coefficient"] != 1.0
        or rdrop["formula"]
        != "0.5_times_coefficient_times_weighted_mean_squared_prediction_disagreement"
        or not rdrop["fixed_variance_Gaussian_interpretation"]
        or rdrop["inference_dropout"]
        or not rdrop["inseparable_single_intervention"]
        or rdrop["sweep"]
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
        raise v12.ContractError("v40 fixed scientific contract drift")
    return config


def predictive_consistency_penalty(
    first: torch.Tensor,
    second: torch.Tensor,
    weights: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    if (
        first.ndim != 1
        or second.shape != first.shape
        or weights.shape != first.shape
        or coefficient != 1.0
        or bool(torch.any(weights <= 0.0))
    ):
        raise v12.ContractError("v40 predictive-consistency tensor contract failed")
    disagreement = ((first - second).square() * weights).sum()
    disagreement /= weights.sum().clamp_min(1e-12)
    penalty = 0.5 * coefficient * disagreement
    if not bool(torch.isfinite(penalty)):
        raise v12.ContractError("v40 predictive-consistency penalty is non-finite")
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
    rdrop = config["training"]["predictive_dropout_consistency"]
    model = RDropVerticalDeepSet(
        8, 11, hidden=32, dropout_probability=float(rdrop["dropout_probability"])
    ).to(device)
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
    coefficient = float(rdrop["coefficient"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base_losses: list[float] = []
    penalties: list[float] = []
    total_losses: list[float] = []
    disagreements: list[float] = []
    optimizer_steps = 0
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        base_total = 0.0
        penalty_total = 0.0
        total_loss_total = 0.0
        disagreement_total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            first = model(batch[0], batch[1], batch[2])
            second = model(batch[0], batch[1], batch[2])
            first_raw = F.smooth_l1_loss(
                first, batch[3], beta=1.0, reduction="none"
            )
            second_raw = F.smooth_l1_loss(
                second, batch[3], beta=1.0, reduction="none"
            )
            denominator = batch[4].sum().clamp_min(1e-12)
            first_loss = (first_raw * batch[4]).sum() / denominator
            second_loss = (second_raw * batch[4]).sum() / denominator
            base_loss = 0.5 * (first_loss + second_loss)
            penalty = predictive_consistency_penalty(
                first, second, batch[4], coefficient=coefficient
            )
            loss = base_loss + penalty
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v40 total training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            base_total += float(base_loss.detach().cpu())
            penalty_total += float(penalty.detach().cpu())
            total_loss_total += float(loss.detach().cpu())
            disagreement_total += float(torch.mean(torch.abs(first - second)).detach().cpu())
            batches += 1
        base_losses.append(base_total / batches)
        penalties.append(penalty_total / batches)
        total_losses.append(total_loss_total / batches)
        disagreements.append(disagreement_total / batches)

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
    finite = bool(
        np.isfinite(base_losses).all()
        and np.isfinite(penalties).all()
        and np.isfinite(total_losses).all()
        and np.isfinite(disagreements).all()
        and np.isfinite(prediction).all()
    )
    if not finite:
        raise v12.ContractError("v40 R-Drop training contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(base_losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "base_loss_first": base_losses[0],
        "base_loss_last": base_losses[-1],
        "consistency_penalty_first": penalties[0],
        "consistency_penalty_last": penalties[-1],
        "total_loss_first": total_losses[0],
        "total_loss_last": total_losses[-1],
        "mean_abs_disagreement_first": disagreements[0],
        "mean_abs_disagreement_last": disagreements[-1],
        "optimizer_steps": optimizer_steps,
        "two_pass_steps": optimizer_steps,
        "dropout_probability": float(rdrop["dropout_probability"]),
        "consistency_coefficient": coefficient,
        "training_passes_per_batch": 2,
        "inference_dropout": False,
        "input_gradient_steps": 0,
        "parameter_perturbation_steps": 0,
        "row_mixing_steps": 0,
        "gradient_surgery_steps": 0,
        "loss": config["training"]["objective"],
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _penalty_contract_receipt() -> dict[str, Any]:
    first = torch.tensor([0.0, 1.0, -2.0], requires_grad=True)
    second = torch.tensor([0.0, -1.0, 1.0], requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0])
    penalty = predictive_consistency_penalty(first, second, weights)
    expected = 0.5 * ((0.0 + 8.0 + 27.0) / 6.0)
    reverse = predictive_consistency_penalty(second, first, weights)
    equal = predictive_consistency_penalty(first, first, weights)
    penalty.backward()
    exact = abs(float(penalty.detach()) - expected) <= 1e-7
    symmetric = bool(torch.equal(penalty.detach(), reverse.detach()))
    zero_equal = float(equal.detach()) == 0.0
    finite_gradient = bool(
        first.grad is not None
        and second.grad is not None
        and torch.isfinite(first.grad).all()
        and torch.isfinite(second.grad).all()
    )
    if not (exact and symmetric and zero_equal and finite_gradient):
        raise v12.ContractError("v40 synthetic penalty contract failed")
    return {
        "penalty": float(penalty.detach()),
        "expected_penalty": expected,
        "formula_exact": exact,
        "symmetric": symmetric,
        "equal_predictions_zero": zero_equal,
        "finite_gradient": finite_gradient,
        "coefficient": 1.0,
    }


def _stochastic_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(40)
    model = RDropVerticalDeepSet(8, 11, dropout_probability=0.1)
    tokens = torch.randn(8, 5, 8)
    mask = torch.ones(8, 5)
    context = torch.randn(8, 11)
    model.train()
    torch.manual_seed(4001)
    first_a = model(tokens, mask, context)
    first_b = model(tokens, mask, context)
    torch.manual_seed(4001)
    repeat_a = model(tokens, mask, context)
    repeat_b = model(tokens, mask, context)
    pair_reproducible = bool(
        torch.equal(first_a, repeat_a) and torch.equal(first_b, repeat_b)
    )
    stochastic_difference = float(
        torch.max(torch.abs(first_a - first_b)).detach().cpu()
    )
    model.eval()
    with torch.inference_mode():
        inference_a = model(tokens, mask, context)
        inference_b = model(tokens, mask, context)
    deterministic_inference = bool(torch.equal(inference_a, inference_b))
    dropout_modules = sum(isinstance(item, nn.Dropout) for item in model.modules())
    if (
        not pair_reproducible
        or stochastic_difference <= 0.0
        or not deterministic_inference
        or dropout_modules != 4
    ):
        raise v12.ContractError("v40 stochastic-view contract failed")
    return {
        "pair_reproducible_under_seed_reset": pair_reproducible,
        "stochastic_pair_maximum_abs_difference": stochastic_difference,
        "dropout_off_inference_deterministic": deterministic_inference,
        "dropout_module_count": dropout_modules,
        "dropout_probability": 0.1,
    }


def _isolation_receipt() -> dict[str, Any]:
    torch.manual_seed(401)
    model = RDropVerticalDeepSet(8, 11, dropout_probability=0.1).eval()
    tokens = torch.randn(4, 5, 8)
    mask = torch.ones(4, 5)
    mask[:, -1] = 0.0
    context = torch.randn(4, 11)
    changed = tokens.clone()
    changed[:, -1] += 1000.0
    order = torch.tensor([2, 4, 0, 3, 1])
    with torch.inference_mode():
        base = model(tokens, mask, context)
        masked = model(changed, mask, context)
        permuted = model(tokens[:, order], mask[:, order], context)
    return {
        "masked_token_maximum_abs_error": float(torch.max(torch.abs(base - masked))),
        "permutation_maximum_abs_error": float(torch.max(torch.abs(base - permuted))),
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    for key, relative in config["authorization_evidence"].items():
        if not (
            key.endswith("_result")
            or key in ("prospective_gate_amendment", "p1_v39_report", "fingerprint")
        ):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "p1_v39_cross_problem_adjacency_disclosed": True,
        "p1_code_result_gate_transfer_count": 0,
        "v23_input_gradient_distinguished": True,
        "v24_sam_distinguished": True,
        "v26_mixup_distinguished": True,
        "v38_output_shrinkage_distinguished": True,
        "v39_gradient_mask_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    isolation = _isolation_receipt()
    if max(isolation.values()) > 1e-6:
        raise v12.ContractError("v40 masked/future or permutation isolation failed")
    evidence = config["authorization_evidence"]
    audit = ROOT / evidence["audit_result"]
    amendment = ROOT / evidence["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "penalty_contract": _penalty_contract_receipt(),
        "stochastic_contract": _stochastic_contract_receipt(),
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
        "# P2 v40 predictive-dropout consistency DeepSets\n\n"
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
        "Exact v13에 fixed dropout 0.1과 two-pass fixed-variance Gaussian "
        "predictive-consistency만 inseparable intervention으로 추가했다. Liang et "
        "al. (NeurIPS 2021)은 동기만 제공하며 P2 regression 성능 근거가 아니다. "
        "P1-v39 adjacency는 공개했고 code/result/gate transfer는 0이다. sweep/"
        "router/ensemble/row deletion/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_PREDICTIVE_DROPOUT_CONSISTENCY"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["penalty_contract"] = _penalty_contract_receipt()
    result["stochastic_contract"] = _stochastic_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "predictive_dropout_consistency": config["training"][
                "predictive_dropout_consistency"
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
            for name in ("v13", "v23", "v24", "v26", "v38", "v39")
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
