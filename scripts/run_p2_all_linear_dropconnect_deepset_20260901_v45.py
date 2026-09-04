"""Run sealed P2 v45 all-Linear train-only DropConnect exactly once."""

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
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_target_layer_gradnorm_balanced_deepset_20260901_v44 as v44  # noqa: E402

v37 = v44.v37
v13 = v44.v13
v12 = v44.v12

EXPERIMENT_ID = "p2_all_linear_dropconnect_deepset_20260901_v45"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V45_ALL_LINEAR_DROPCONNECT_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.all_linear_dropconnect_deepset.result.20260901.v45"

_BASE_RUN = v44._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v44._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v44._V13_RUNNER


class DropConnectLinear(nn.Linear):
    """Linear with transient train-only Bernoulli weight masks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        drop_probability: float = 0.1,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if not 0.0 <= drop_probability < 1.0:
            raise v12.ContractError("v45 invalid DropConnect probability")
        self.drop_probability = float(drop_probability)
        self._mask_calls = 0
        self._mask_kept: Tensor | None = None
        self._mask_total = 0
        self._last_mask: Tensor | None = None

    @classmethod
    def from_linear(
        cls, source: nn.Linear, drop_probability: float
    ) -> DropConnectLinear:
        target = cls(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            drop_probability=drop_probability,
        ).to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            target.weight.copy_(source.weight)
            if source.bias is not None and target.bias is not None:
                target.bias.copy_(source.bias)
        return target

    def reset_mask_statistics(self) -> None:
        self._mask_calls = 0
        self._mask_kept = None
        self._mask_total = 0
        self._last_mask = None

    def forward(self, value: Tensor) -> Tensor:
        if not self.training or self.drop_probability == 0.0:
            return F.linear(value, self.weight, self.bias)
        keep_probability = 1.0 - self.drop_probability
        mask = torch.empty_like(self.weight).bernoulli_(keep_probability)
        kept = mask.detach().sum()
        self._mask_kept = kept if self._mask_kept is None else self._mask_kept + kept
        self._mask_calls += 1
        self._mask_total += mask.numel()
        self._last_mask = mask.detach().clone()
        return F.linear(value, self.weight * mask / keep_probability, self.bias)


class DropConnectVerticalDeepSet(v12.VerticalDeepSet):
    """Exact v13 geometry with all five Linear weights masked only in training."""

    def __init__(
        self,
        token_features: int,
        context_features: int,
        hidden: int = 32,
        drop_probability: float = 0.1,
    ) -> None:
        super().__init__(token_features, context_features, hidden)
        for sequence, index in (
            (self.element, 0),
            (self.element, 2),
            (self.head, 0),
            (self.head, 2),
            (self.head, 4),
        ):
            source = sequence[index]
            if not isinstance(source, nn.Linear):
                raise v12.ContractError("v45 expected exact v13 Linear geometry")
            sequence[index] = DropConnectLinear.from_linear(
                source, drop_probability=drop_probability
            )


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
    dropconnect = training["dropconnect"]
    safety = config["evaluation"]["safety_gate"]
    evidence = config["authorization_evidence"]
    audit = ROOT / evidence["audit_result"]
    amendment = ROOT / evidence["prospective_gate_amendment"]
    fingerprint = ROOT / evidence["fingerprint"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or training["architecture"]
        != "v13_exact_DeepSets_with_train_only_DropConnect_on_all_five_Linear_weights"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"] != "exact_v13_weighted_SmoothL1_beta_1.0"
        or training["optimizer"] != "exact_v13_AdamW"
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
        or training["extra_loss"]
        or training["extra_parameters"]
        or dropconnect["drop_probability"] != 0.1
        or dropconnect["keep_probability"] != 0.9
        or not dropconnect["inverted_training_scale"]
        or dropconnect["masked_module"]
        != "all_five_Linear_weight_matrices"
        or dropconnect["linear_module_count"] != 5
        or dropconnect["bias_masked"]
        or dropconnect["activation_dropout"]
        or dropconnect["prediction_consistency"]
        or dropconnect["monte_carlo_inference"]
        or dropconnect["ensemble_or_model_aggregation"]
        or not dropconnect["evaluation_uses_raw_weight_once"]
        or dropconnect["sweep"]
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
        raise v12.ContractError("v45 fixed scientific contract drift")
    return config


def mask_sha256(mask: Tensor | None) -> str | None:
    if mask is None:
        return None
    array = mask.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def rng_state_sha256(device: torch.device) -> str:
    digest = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes())
    if device.type == "cuda":
        digest.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return digest.hexdigest()


def dropconnect_modules(model: nn.Module) -> list[tuple[str, DropConnectLinear]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DropConnectLinear)
    ]


def dropconnect_statistics(model: nn.Module) -> dict[str, Any]:
    modules = dropconnect_modules(model)
    rows: dict[str, Any] = {}
    total_kept = 0.0
    total_weights = 0
    total_calls = 0
    for name, module in modules:
        kept = 0.0 if module._mask_kept is None else float(module._mask_kept.cpu())
        total = int(module._mask_total)
        rows[name] = {
            "shape": list(module.weight.shape),
            "bias_present": module.bias is not None,
            "drop_probability": module.drop_probability,
            "mask_calls": module._mask_calls,
            "kept_weights": kept,
            "total_weight_draws": total,
            "keep_share": None if total == 0 else kept / total,
            "last_mask_sha256": mask_sha256(module._last_mask),
        }
        total_kept += kept
        total_weights += total
        total_calls += module._mask_calls
    return {
        "module_count": len(modules),
        "modules": rows,
        "mask_calls": total_calls,
        "kept_weights": total_kept,
        "total_weight_draws": total_weights,
        "keep_share": None if total_weights == 0 else total_kept / total_weights,
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probability = float(config["training"]["dropconnect"]["drop_probability"])
    model = DropConnectVerticalDeepSet(
        8, 11, hidden=32, drop_probability=probability
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
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    optimizer_steps = 0
    model.train()
    initial_rng_sha256 = rng_state_sha256(device)
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        numerator = 0.0
        denominator = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            loss = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v45 training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
        losses.append(numerator / denominator)
    final_rng_sha256 = rng_state_sha256(device)
    mask_receipt = dropconnect_statistics(model)
    model.eval()
    rng_before_evaluation = rng_state_sha256(device)
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
    rng_after_evaluation = rng_state_sha256(device)
    prediction = np.concatenate(output).astype(float)
    modules = dropconnect_modules(model)
    expected_mask_calls = optimizer_steps * len(modules)
    finite = bool(np.isfinite(losses).all() and np.isfinite(prediction).all())
    if (
        not finite
        or len(modules) != 5
        or mask_receipt["mask_calls"] != expected_mask_calls
        or mask_receipt["total_weight_draws"] <= 0
        or initial_rng_sha256 == final_rng_sha256
        or rng_before_evaluation != rng_after_evaluation
    ):
        raise v12.ContractError("v45 DropConnect training contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "parameter_tensors": len(list(model.parameters())),
        "buffers": len(list(model.buffers())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "optimizer_steps": optimizer_steps,
        "drop_probability": probability,
        "keep_probability": 1.0 - probability,
        "dropconnect_statistics": mask_receipt,
        "expected_mask_calls": expected_mask_calls,
        "initial_rng_sha256": initial_rng_sha256,
        "final_rng_sha256": final_rng_sha256,
        "evaluation_rng_unchanged": rng_before_evaluation == rng_after_evaluation,
        "dropout_module_count": sum(
            isinstance(module, nn.Dropout) for module in model.modules()
        ),
        "prediction_consistency_loss": 0,
        "monte_carlo_inference": 0,
        "ensemble_models": 1,
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _dropconnect_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(45)
    base = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    torch.manual_seed(45)
    candidate = DropConnectVerticalDeepSet(8, 11, drop_probability=0.1).eval()
    tokens = torch.randn(16, 5, 8)
    mask = torch.ones(16, 5)
    context = torch.randn(16, 11)
    with torch.inference_mode():
        expected = base(tokens, mask, context)
        actual = candidate(tokens, mask, context)
    evaluation_error = float(torch.max(torch.abs(expected - actual)))
    parameter_names_equal = [name for name, _ in base.named_parameters()] == [
        name for name, _ in candidate.named_parameters()
    ]

    left = DropConnectVerticalDeepSet(8, 11, drop_probability=0.1).train()
    right = DropConnectVerticalDeepSet(8, 11, drop_probability=0.1).train()
    right.load_state_dict(left.state_dict())
    torch.manual_seed(451)
    left_output = left(tokens, mask, context)
    left_first = dropconnect_statistics(left)
    left_first_hashes = [
        value["last_mask_sha256"] for value in left_first["modules"].values()
    ]
    torch.manual_seed(451)
    right_output = right(tokens, mask, context)
    right_first = dropconnect_statistics(right)
    right_first_hashes = [
        value["last_mask_sha256"] for value in right_first["modules"].values()
    ]
    deterministic_error = float(
        torch.max(torch.abs(left_output - right_output)).detach()
    )
    left(tokens, mask, context)
    left_second = dropconnect_statistics(left)
    left_second_hashes = [
        value["last_mask_sha256"] for value in left_second["modules"].values()
    ]

    zero = DropConnectVerticalDeepSet(8, 11, drop_probability=0.0).train()
    zero.load_state_dict(base.state_dict())
    base.train()
    zero_error = float(
        torch.max(
            torch.abs(base(tokens, mask, context) - zero(tokens, mask, context))
        ).detach()
    )

    candidate.eval()
    rng_before = rng_state_sha256(torch.device("cpu"))
    with torch.inference_mode():
        first_eval = candidate(tokens, mask, context)
        second_eval = candidate(tokens, mask, context)
    rng_after = rng_state_sha256(torch.device("cpu"))
    modules = dropconnect_modules(candidate)
    return {
        "module_count": len(modules),
        "module_names": [name for name, _ in modules],
        "module_shapes": [list(module.weight.shape) for _, module in modules],
        "all_biases_unmasked": all(module.bias is not None for _, module in modules),
        "drop_probability": 0.1,
        "keep_probability": 0.9,
        "parameters": int(sum(value.numel() for value in candidate.parameters())),
        "parameter_tensors": len(list(candidate.parameters())),
        "buffers": len(list(candidate.buffers())),
        "parameter_names_equal_v13": parameter_names_equal,
        "evaluation_initial_function_maximum_abs_error": evaluation_error,
        "deterministic_same_seed_training_maximum_abs_error": deterministic_error,
        "deterministic_same_seed_mask_hashes": left_first_hashes == right_first_hashes,
        "consecutive_step_masks_distinct": left_first_hashes != left_second_hashes,
        "first_step_keep_share": left_first["keep_share"],
        "zero_probability_training_maximum_abs_error": zero_error,
        "evaluation_repeat_maximum_abs_error": float(
            torch.max(torch.abs(first_eval - second_eval))
        ),
        "evaluation_rng_unchanged": rng_before == rng_after,
        "dropout_module_count": sum(
            isinstance(module, nn.Dropout) for module in candidate.modules()
        ),
    }


def _isolation_receipt() -> dict[str, float]:
    torch.manual_seed(54)
    model = DropConnectVerticalDeepSet(8, 11, drop_probability=0.1).eval()
    tokens = torch.randn(4, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    context = torch.randn(4, 11)
    changed = tokens.clone()
    changed[mask == 0] = 1e6
    order = torch.tensor([4, 2, 0, 3, 1])
    with torch.inference_mode():
        base = model(tokens, mask, context)
        masked = model(changed, mask, context)
        permuted = model(tokens[:, order], mask[:, order], context)
        repeated = model(tokens, mask, context)
    return {
        "masked_or_future_token_maximum_abs_error": float(
            torch.max(torch.abs(base - masked))
        ),
        "permutation_maximum_abs_error": float(torch.max(torch.abs(base - permuted))),
        "repeat_maximum_abs_error": float(torch.max(torch.abs(base - repeated))),
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    for key, relative in config["authorization_evidence"].items():
        if not (
            key.endswith("_result")
            or key.endswith("_receipt")
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
        "v24_sam_distinguished": True,
        "v27_spectral_norm_distinguished": True,
        "v40_activation_dropout_consistency_distinguished": True,
        "v41_weight_norm_distinguished": True,
        "v43_v44_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    contract = _dropconnect_contract_receipt()
    isolation = _isolation_receipt()
    if (
        contract["module_count"] != 5
        or contract["module_shapes"]
        != [[32, 8], [32, 32], [32, 75], [32, 32], [1, 32]]
        or not contract["all_biases_unmasked"]
        or contract["parameters"] != 4865
        or contract["parameter_tensors"] != 10
        or contract["buffers"] != 0
        or not contract["parameter_names_equal_v13"]
        or contract["evaluation_initial_function_maximum_abs_error"] != 0.0
        or contract["deterministic_same_seed_training_maximum_abs_error"] != 0.0
        or not contract["deterministic_same_seed_mask_hashes"]
        or not contract["consecutive_step_masks_distinct"]
        or not 0.85 <= contract["first_step_keep_share"] <= 0.95
        or contract["zero_probability_training_maximum_abs_error"] != 0.0
        or contract["evaluation_repeat_maximum_abs_error"] != 0.0
        or not contract["evaluation_rng_unchanged"]
        or contract["dropout_module_count"] != 0
        or max(isolation.values()) > 1e-6
    ):
        raise v12.ContractError("v45 target-free DropConnect preflight failed")
    evidence = config["authorization_evidence"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "dropconnect_contract": contract,
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
        "# P2 v45 all-Linear train-only DropConnect DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled delta RMSE "
        f"`{item['delta_rmse']:+.9f} C`, canonical nominal "
        f"`{item['canonical_nominal_pooled_points_delta']:+.6f}` points, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}` points.\n\n"
        f"prospective fold x layer gate: `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, max cell "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "Exact v13의 다섯 Linear weight에 train-only DropConnect p=0.1만 적용했다. "
        "activation dropout/R-Drop/MC inference/ensemble/sweep/router/row deletion/"
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
        else "EXPLORATORY_NO_GO_ALL_LINEAR_DROPCONNECT"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["dropconnect_contract"] = _dropconnect_contract_receipt()
    result["isolation"] = _isolation_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer": config["training"]["optimizer"],
            "dropconnect": config["training"]["dropconnect"],
            "row_deletion": 0,
            "input_perturbation": 0,
            "data_augmentation": 0,
            "extra_loss": 0,
            "extra_parameters": 0,
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
            for name in ("v13", "v24", "v40", "v41", "v43", "v44")
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
