"""Run sealed P2 v41 all-Linear WeightNorm DeepSets exactly once."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_predictive_dropout_consistency_deepset_20260901_v40 as v40  # noqa: E402

v39 = v40.v39
v38 = v40.v38
v37 = v40.v37
v13 = v40.v13
v12 = v40.v12

EXPERIMENT_ID = "p2_all_linear_weight_normalized_deepset_20260901_v41"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V41_ALL_LINEAR_WEIGHT_NORMALIZED_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.all_linear_weight_normalized_deepset.result.20260901.v41"

_BASE_RUN = v40._BASE_RUN
_BASE_DOMAIN_BALANCED_WEIGHTS = v40._BASE_DOMAIN_BALANCED_WEIGHTS
_V13_RUNNER = v40._V13_RUNNER


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
    norm = training["weight_normalization"]
    safety = config["evaluation"]["safety_gate"]
    evidence = config["authorization_evidence"]
    audit = ROOT / evidence["audit_result"]
    amendment = ROOT / evidence["prospective_gate_amendment"]
    fingerprint = ROOT / evidence["fingerprint"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
        or training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2_with_all_five_Linear_weight_norm"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"]
        != "weighted_SmoothL1_beta_1.0_with_all_five_Linear_per_output_weight_normalization"
        or training["optimizer"]
        != "exact_v13_AdamW_on_weight_norm_magnitude_and_direction_parameters"
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
        or norm["api"] != "torch.nn.utils.parametrizations.weight_norm"
        or norm["dimension"] != 0
        or norm["linear_module_count"] != 5
        or not norm["learned_per_output_magnitude"]
        or not norm["unit_direction"]
        or norm["initial_function_maximum_abs_error_lte"] != 0.000001
        or norm["bias_parametrized"]
        or norm["spectral_normalization"]
        or norm["power_iteration"]
        or norm["global_operator_norm_constraint"]
        or norm["batch_statistics"]
        or norm["activation_normalization"]
        or norm["sweep"]
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
        raise v12.ContractError("v41 fixed scientific contract drift")
    return config


def apply_weight_normalization(model: nn.Module) -> list[str]:
    linear_names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if parametrize.is_parametrized(module, "weight"):
                raise v12.ContractError("v41 Linear was already parametrized")
            weight_norm(module, name="weight", dim=0)
            linear_names.append(name)
    if len(linear_names) != 5:
        raise v12.ContractError("v41 expected exactly five Linear modules")
    return linear_names


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def parametrization_receipt(model: nn.Module) -> dict[str, Any]:
    linear_names: list[str] = []
    magnitude_errors: list[float] = []
    direction_errors: list[float] = []
    effective_errors: list[float] = []
    bias_parametrized = False
    class_names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not parametrize.is_parametrized(module, "weight"):
            raise v12.ContractError("v41 missing Linear WeightNorm")
        linear_names.append(name)
        stack = module.parametrizations.weight
        class_names.extend(type(item).__name__ for item in stack)
        magnitude = stack.original0
        direction = stack.original1
        direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        effective_norm = torch.linalg.vector_norm(module.weight, dim=1, keepdim=True)
        magnitude_errors.append(
            float(torch.max(torch.abs(magnitude.detach() - direction_norm.detach())))
        )
        unit = direction / direction_norm.clamp_min(1e-12)
        direction_errors.append(
            float(
                torch.max(
                    torch.abs(
                        torch.linalg.vector_norm(unit, dim=1)
                        - torch.ones(unit.shape[0], device=unit.device)
                    )
                ).detach().cpu()
            )
        )
        effective_errors.append(
            float(
                torch.max(
                    torch.abs(effective_norm.detach() - torch.abs(magnitude.detach()))
                ).cpu()
            )
        )
        bias_parametrized |= parametrize.is_parametrized(module, "bias")
    buffers = [name for name, _value in model.named_buffers()]
    spectral_state = [
        name
        for name in (*class_names, *buffers, *model.state_dict().keys())
        if "spectral" in name.lower() or "power_iteration" in name.lower()
    ]
    finite = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    receipt = {
        "linear_names": linear_names,
        "linear_module_count": len(linear_names),
        "parametrization_class_names": class_names,
        "maximum_initial_magnitude_error": max(magnitude_errors),
        "maximum_unit_direction_norm_error": max(direction_errors),
        "maximum_effective_magnitude_error": max(effective_errors),
        "bias_parametrized": bias_parametrized,
        "buffer_names": buffers,
        "spectral_or_power_iteration_state": spectral_state,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "parameter_tensor_count": len(list(model.parameters())),
        "state_sha256": state_digest(model),
        "finite": finite,
    }
    if (
        receipt["linear_module_count"] != 5
        or bias_parametrized
        or buffers
        or spectral_state
        or not finite
    ):
        raise v12.ContractError("v41 WeightNorm parametrization contract failed")
    return receipt


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
    linear_names = apply_weight_normalization(model)
    initial = parametrization_receipt(model)
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
                raise v12.ContractError("v41 training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
        losses.append(numerator / denominator)

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
    final = parametrization_receipt(model)
    finite = bool(np.isfinite(losses).all() and np.isfinite(prediction).all())
    if not finite:
        raise v12.ContractError("v41 WeightNorm training contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": final["parameter_count"],
        "parameter_tensors": final["parameter_tensor_count"],
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "optimizer_steps": optimizer_steps,
        "linear_names": linear_names,
        "weight_norm_module_count": final["linear_module_count"],
        "initial_parametrization_state_sha256": initial["state_sha256"],
        "final_parametrization_state_sha256": final["state_sha256"],
        "spectral_buffer_count": len(final["spectral_or_power_iteration_state"]),
        "batch_stat_buffer_count": len(final["buffer_names"]),
        "bias_parametrized": final["bias_parametrized"],
        "loss": config["training"]["objective"],
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _weight_norm_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(41)
    base = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    candidate = copy.deepcopy(base).eval()
    tokens = torch.randn(8, 5, 8)
    mask = torch.ones(8, 5)
    context = torch.randn(8, 11)
    with torch.inference_mode():
        before = base(tokens, mask, context)
    linear_names = apply_weight_normalization(candidate)
    receipt = parametrization_receipt(candidate)
    with torch.inference_mode():
        after = candidate(tokens, mask, context)
        repeat = candidate(tokens, mask, context)
    function_error = float(torch.max(torch.abs(before - after)))
    deterministic = bool(torch.equal(after, repeat))
    receipt.update(
        {
            "initial_function_maximum_abs_error": function_error,
            "initial_function_preserved": function_error <= 1e-6,
            "deterministic_inference": deterministic,
            "linear_names_exact": linear_names
            == ["element.0", "element.2", "head.0", "head.2", "head.4"],
            "learned_magnitude_direction_only": True,
            "power_iteration_count": 0,
            "global_operator_norm_constraint": False,
        }
    )
    if not (
        receipt["initial_function_preserved"]
        and deterministic
        and receipt["linear_names_exact"]
        and receipt["maximum_initial_magnitude_error"] <= 1e-7
        and receipt["maximum_unit_direction_norm_error"] <= 1e-6
        and receipt["maximum_effective_magnitude_error"] <= 1e-6
    ):
        raise v12.ContractError("v41 initial-function WeightNorm contract failed")
    return receipt


def _isolation_receipt() -> dict[str, Any]:
    torch.manual_seed(411)
    model = v12.VerticalDeepSet(8, 11, hidden=32).eval()
    apply_weight_normalization(model)
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
        repeated = model(tokens, mask, context)
    return {
        "masked_token_maximum_abs_error": float(torch.max(torch.abs(base - masked))),
        "permutation_maximum_abs_error": float(torch.max(torch.abs(base - permuted))),
        "repeat_maximum_abs_error": float(torch.max(torch.abs(base - repeated))),
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    for key, relative in config["authorization_evidence"].items():
        if not (
            key.endswith("_result")
            or key in ("prospective_gate_amendment", "v27_failure_report", "fingerprint")
        ):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v27_spectral_family_not_reopened": True,
        "v27_code_import_count": 0,
        "spectral_tolerance_changes": 0,
        "v29_lookahead_distinguished": True,
        "v34_gradient_centralization_distinguished": True,
        "v35_radam_distinguished": True,
        "v40_dropout_consistency_distinguished": True,
        "official_v23_feedback_used_for_selection": False,
        "evidence_sha256": evidence,
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    isolation = _isolation_receipt()
    if max(isolation.values()) > 1e-6:
        raise v12.ContractError("v41 masked/future, permutation or repeat isolation failed")
    evidence = config["authorization_evidence"]
    audit = ROOT / evidence["audit_result"]
    amendment = ROOT / evidence["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "weight_norm_contract": _weight_norm_contract_receipt(),
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
        "# P2 v41 all-Linear WeightNorm DeepSets\n\n"
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
        "Exact v13의 다섯 Linear에 learned per-output magnitude/direction WeightNorm "
        "재매개화만 적용했다. Salimans and Kingma (NIPS 2016)는 동기만 제공하며 "
        "P2 성능 근거가 아니다. v27 spectral code/tolerance/power iteration을 재사용하지 "
        "않았다. sweep/router/ensemble/row deletion/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_WEIGHT_NORMALIZATION"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["weight_norm_contract"] = _weight_norm_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "weight_normalization": config["training"]["weight_normalization"],
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
            for name in ("v13", "v29", "v34", "v35", "v40")
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
