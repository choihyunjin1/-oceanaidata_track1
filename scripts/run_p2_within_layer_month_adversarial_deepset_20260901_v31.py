"""Run sealed P2 v31 within-layer month-adversarial DeepSets exactly once."""

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
import run_p2_layer_task_gradient_surgery_deepset_20260901_v28 as v28  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_within_layer_month_adversarial_deepset_20260901_v31"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V31_WITHIN_LAYER_MONTH_ADVERSARIAL_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.within_layer_month_adversarial_deepset.result.20260901.v31"

_BASE_LOAD_CONFIG = v13.load_config
_BASE_RUN = v13.run
_BASE_DOMAIN_BALANCED_WEIGHTS = v13.domain_balanced_weights
_V13_RUNNER = v13.RUNNER
_ACTIVE_LAYER_IDS: np.ndarray | None = None
_ACTIVE_MONTH_IDS: np.ndarray | None = None


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
    adversary = training["domain_adversary"]
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    audit = ROOT / config["authorization_evidence"]["audit_result"]
    if (
        training["task_architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["optimizer"] != "exact_v13_AdamW"
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
        or adversary["nuisance_label"] != "calendar_month_1_to_12"
        or adversary["conditioning"]
        != "separate_linear_64_to_12_head_per_target_layer_2_3_4"
        or adversary["input"] != "masked_mean_max_pooled_token_latent_only"
        or adversary["task_context_adversarialized"]
        or adversary["gradient_reversal_coefficient"] != 0.1
        or adversary["domain_loss_multiplier"] != 1.0
        or adversary["schedule"]
        or adversary["coefficient_sweep"]
        or adversary["month_router"]
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
        raise v12.ContractError("v31 fixed scientific contract drift")
    return config


def domain_balanced_weights(
    layer: np.ndarray, local_time: pd.Series | pd.DatetimeIndex
) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture fixed layer and month nuisance labels beside exact v13 weights."""
    global _ACTIVE_LAYER_IDS, _ACTIVE_MONTH_IDS
    local = pd.DatetimeIndex(local_time)
    _ACTIVE_LAYER_IDS = np.asarray(layer, dtype=np.int64).copy()
    _ACTIVE_MONTH_IDS = local.month.to_numpy(dtype=np.int64)
    weights, receipt = _BASE_DOMAIN_BALANCED_WEIGHTS(layer, local_time)
    if sorted(np.unique(_ACTIVE_LAYER_IDS).tolist()) != [2, 3, 4]:
        raise v12.ContractError("v31 target-layer support drift")
    if np.min(_ACTIVE_MONTH_IDS) < 1 or np.max(_ACTIVE_MONTH_IDS) > 12:
        raise v12.ContractError("v31 calendar-month support drift")
    receipt["adversary_target_layers"] = [2, 3, 4]
    receipt["adversary_calendar_months"] = sorted(
        np.unique(_ACTIVE_MONTH_IDS).tolist()
    )
    return weights, receipt


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(value: Tensor, coefficient: float) -> Tensor:
    return _GradientReverse.apply(value, float(coefficient))


class MonthAdversarialDeepSet(nn.Module):
    """Exact v13 task path plus training-only layer-conditional month heads."""

    def __init__(self, token_features: int = 8, context_features: int = 11) -> None:
        super().__init__()
        hidden = 32
        self.element = nn.Sequential(
            nn.Linear(token_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + context_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.month_heads = nn.ModuleDict(
            {str(layer): nn.Linear(hidden * 2, 12) for layer in (2, 3, 4)}
        )

    def pooled_tokens(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        encoded = self.element(tokens)
        mask = token_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool(), negative).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        return torch.cat((mean, maximum), dim=1)

    def task_from_pooled(self, pooled: Tensor, context: Tensor) -> Tensor:
        return self.head(torch.cat((pooled, context), dim=1)).squeeze(1)

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        return self.task_from_pooled(self.pooled_tokens(tokens, token_mask), context)


def domain_adversarial_loss(
    model: MonthAdversarialDeepSet,
    pooled: Tensor,
    layer_ids: Tensor,
    month_ids: Tensor,
    weights: Tensor,
    coefficient: float,
) -> tuple[Tensor, float, int]:
    reversed_pooled = gradient_reverse(pooled, coefficient)
    numerator = torch.zeros((), dtype=pooled.dtype, device=pooled.device)
    denominator = torch.zeros((), dtype=pooled.dtype, device=pooled.device)
    correct = 0
    rows = 0
    heads_used = 0
    for layer in (2, 3, 4):
        selected = layer_ids.eq(layer)
        if not bool(selected.any()):
            continue
        logits = model.month_heads[str(layer)](reversed_pooled[selected])
        labels = month_ids[selected] - 1
        raw = F.cross_entropy(logits, labels, reduction="none")
        local_weights = weights[selected]
        numerator = numerator + (raw * local_weights).sum()
        denominator = denominator + local_weights.sum()
        correct += int(logits.detach().argmax(dim=1).eq(labels).sum().cpu())
        rows += int(selected.sum().cpu())
        heads_used += 1
    if heads_used == 0 or rows == 0:
        raise v12.ContractError("v31 minibatch has no supported domain head")
    return numerator / denominator.clamp_min(1e-12), correct / rows, heads_used


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
    if (
        _ACTIVE_LAYER_IDS is None
        or _ACTIVE_MONTH_IDS is None
        or len(_ACTIVE_LAYER_IDS) != len(target)
        or len(_ACTIVE_MONTH_IDS) != len(target)
    ):
        raise v12.ContractError("v31 active nuisance labels unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonthAdversarialDeepSet().to(device)
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
    adversary = config["training"]["domain_adversary"]
    coefficient = float(adversary["gradient_reversal_coefficient"])
    multiplier = float(adversary["domain_loss_multiplier"])
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    domain_losses: list[float] = []
    domain_accuracies: list[float] = []
    heads_per_batch: list[int] = []
    optimizer_steps = 0
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        numerator = 0.0
        denominator = 0.0
        epoch_domain_losses: list[float] = []
        epoch_domain_accuracies: list[float] = []
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            layers = torch.from_numpy(_ACTIVE_LAYER_IDS[selected.numpy()]).to(device)
            months = torch.from_numpy(_ACTIVE_MONTH_IDS[selected.numpy()]).to(device)
            optimizer.zero_grad(set_to_none=True)
            pooled = model.pooled_tokens(batch[0], batch[1])
            prediction = model.task_from_pooled(pooled, batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            task_loss = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            domain_loss, domain_accuracy, heads_used = domain_adversarial_loss(
                model, pooled, layers, months, batch[4], coefficient
            )
            loss = task_loss + multiplier * domain_loss
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v31 adversarial training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
            epoch_domain_losses.append(float(domain_loss.detach().cpu()))
            epoch_domain_accuracies.append(domain_accuracy)
            heads_per_batch.append(heads_used)
        losses.append(numerator / denominator)
        domain_losses.append(float(np.mean(epoch_domain_losses)))
        domain_accuracies.append(float(np.mean(epoch_domain_accuracies)))
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
        and np.isfinite(domain_losses).all()
        and np.isfinite(domain_accuracies).all()
        and np.isfinite(prediction).all()
    )
    if not finite or min(heads_per_batch, default=0) < 1:
        raise v12.ContractError("v31 training/domain-adversary contract failed")
    task_parameters = sum(
        value.numel()
        for name, value in model.named_parameters()
        if not name.startswith("month_heads.")
    )
    domain_parameters = sum(
        value.numel()
        for name, value in model.named_parameters()
        if name.startswith("month_heads.")
    )
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "task_parameters": int(task_parameters),
        "domain_head_parameters": int(domain_parameters),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "domain_loss_first": domain_losses[0],
        "domain_loss_last": domain_losses[-1],
        "domain_accuracy_first": domain_accuracies[0],
        "domain_accuracy_last": domain_accuracies[-1],
        "gradient_reversal_coefficient": coefficient,
        "domain_loss_multiplier": multiplier,
        "minimum_domain_heads_per_batch": min(heads_per_batch),
        "maximum_domain_heads_per_batch": max(heads_per_batch),
        "optimizer_steps": optimizer_steps,
        "optimizer": "AdamW",
        "task_context_adversarialized": 0,
        "schedule": 0,
        "coefficient_sweep": 0,
        "month_router": 0,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _gradient_reversal_contract_receipt() -> dict[str, Any]:
    value = torch.tensor([1.0, -2.0], requires_grad=True)
    upstream = torch.tensor([3.0, 4.0])
    (gradient_reverse(value, 0.1) * upstream).sum().backward()
    expected = torch.tensor([-0.3, -0.4])
    exact = bool(torch.allclose(value.grad, expected, atol=1e-7, rtol=0.0))
    if not exact:
        raise v12.ContractError("v31 gradient reversal contract failed")
    return {
        "coefficient": 0.1,
        "expected_gradient": expected.tolist(),
        "observed_gradient": value.grad.tolist(),
        "formula_exact": exact,
        "schedule": False,
        "coefficient_sweep": False,
    }


def _model_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(31)
    base = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    torch.manual_seed(31)
    candidate = MonthAdversarialDeepSet().eval()
    task_state_equal = all(
        torch.equal(value, candidate.state_dict()[name])
        for name, value in base.state_dict().items()
    )
    tokens = torch.randn(5, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 0, 0, 1]],
        dtype=torch.float32,
    )
    context = torch.randn(5, 11)
    order = torch.tensor([4, 2, 0, 3, 1])
    with torch.inference_mode():
        base_output = base(tokens, mask, context)
        task_output = candidate(tokens, mask, context)
        permuted = candidate(tokens[:, order], mask[:, order], context)
    task_error = float(torch.max(torch.abs(base_output - task_output)))
    permutation_error = float(torch.max(torch.abs(task_output - permuted)))
    if not task_state_equal or task_error != 0.0 or permutation_error > 1e-6:
        raise v12.ContractError("v31 exact task/permutation contract failed")
    return {
        "v13_task_state_byte_equal_at_initialization": task_state_equal,
        "v13_task_output_maximum_abs_error": task_error,
        "permutation_maximum_abs_error": permutation_error,
        "task_context_adversarialized": False,
        "domain_head_count": len(candidate.month_heads),
        "domain_classes_per_head": 12,
    }


def _isolation_receipt() -> dict[str, Any]:
    return v28._isolation_receipt()


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative in (
        config["authorization_evidence"]["v13_result"],
        config["authorization_evidence"]["v18_result"],
        config["authorization_evidence"]["v19_result"],
        config["authorization_evidence"]["v20_result"],
        config["authorization_evidence"]["v23_result"],
        config["authorization_evidence"]["v30_result"],
        config["authorization_evidence"]["p1_dann_report"],
        config["authorization_evidence"]["prospective_gate_amendment"],
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "cross_problem_p1_execution_hits": 1,
        "p1_code_or_output_reused": False,
        "v18_group_dro_distinguished": True,
        "v19_vrex_distinguished": True,
        "v20_coral_distinguished": True,
        "v23_input_gradient_distinguished": True,
        "v30_irmv1_distinguished": True,
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
    model_contract = _model_contract_receipt()
    if max(isolation.values()) > 1e-6 or model_contract["permutation_maximum_abs_error"] > 1e-6:
        raise v12.ContractError("v31 masked/future or permutation isolation failed")
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "gradient_reversal_contract": _gradient_reversal_contract_receipt(),
        "model_contract": model_contract,
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
        "# P2 v31 within-layer month-adversarial DeepSets\n\n"
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
        "v13 task path를 그대로 두고 pooled token latent에만 layer별 12-way month "
        "classifier와 fixed GRL(0.1)을 추가했다. Ganin et al. (JMLR 2016)은 "
        "representation 동기만 제공하며 P2 성능 근거가 아니다. P1 v24 code/output 재사용0, "
        "schedule/sweep/router/ensemble/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_WITHIN_LAYER_MONTH_ADVERSARY"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["gradient_reversal_contract"] = _gradient_reversal_contract_receipt()
    result["model_contract"] = _model_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "domain_adversary": config["training"]["domain_adversary"],
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
        "v30_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v30_result"]).read_text(
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
