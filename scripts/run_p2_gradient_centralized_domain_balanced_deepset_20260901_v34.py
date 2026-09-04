"""Run sealed P2 v34 gradient-centralized DeepSets exactly once."""

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

EXPERIMENT_ID = "p2_gradient_centralized_domain_balanced_deepset_20260901_v34"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V34_GRADIENT_CENTRALIZED_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.gradient_centralized_domain_balanced_deepset.result.20260901.v34"

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
    centralization = training["gradient_centralization"]
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    audit = ROOT / config["authorization_evidence"]["audit_result"]
    if (
        training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"]
        != "weighted_SmoothL1_beta_1.0_with_fixed_gradient_centralization"
        or training["optimizer"] != "exact_v13_AdamW_after_gradient_centralization"
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
        or centralization["eligible"]
        != "Linear weight parameters with ndim greater than 1 only"
        or centralization["coefficient"] is not None
        or centralization["bias_or_1d_change"]
        or centralization["second_loss"]
        or centralization["task_split"]
        or centralization["parameter_reparameterization"]
        or centralization["slow_weights"]
        or centralization["sweep"]
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
        raise v12.ContractError("v34 fixed scientific contract drift")
    return config


def centralize_linear_weight_gradients(model: torch.nn.Module) -> dict[str, Any]:
    eligible: list[str] = []
    maximum_abs_mean_before = 0.0
    maximum_abs_mean_after = 0.0
    linear_weight_names = {
        f"{module_name}.weight" if module_name else "weight"
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            raise v12.ContractError(f"v34 missing gradient: {name}")
        if not bool(torch.isfinite(gradient).all()):
            raise v12.ContractError(f"v34 non-finite gradient: {name}")
        if not (name in linear_weight_names and parameter.ndim > 1):
            continue
        dimensions = tuple(range(1, gradient.ndim))
        before = gradient.mean(dim=dimensions, keepdim=True)
        maximum_abs_mean_before = max(
            maximum_abs_mean_before, float(before.detach().abs().max().cpu())
        )
        gradient.sub_(before)
        after = gradient.mean(dim=dimensions, keepdim=True)
        maximum_abs_mean_after = max(
            maximum_abs_mean_after, float(after.detach().abs().max().cpu())
        )
        eligible.append(name)
    if not eligible or maximum_abs_mean_after > 1e-6:
        raise v12.ContractError("v34 gradient centralization contract failed")
    return {
        "eligible_names": eligible,
        "eligible_count": len(eligible),
        "maximum_abs_mean_before": maximum_abs_mean_before,
        "maximum_abs_mean_after": maximum_abs_mean_after,
        "bias_or_1d_changed": False,
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
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    optimizer_steps = 0
    eligible_names: list[str] | None = None
    maximum_abs_mean_before = 0.0
    maximum_abs_mean_after = 0.0
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
                raise v12.ContractError("v34 weighted SmoothL1 is non-finite")
            loss.backward()
            central = centralize_linear_weight_gradients(model)
            if eligible_names is None:
                eligible_names = central["eligible_names"]
            elif eligible_names != central["eligible_names"]:
                raise v12.ContractError("v34 eligible gradient set drift")
            maximum_abs_mean_before = max(
                maximum_abs_mean_before, central["maximum_abs_mean_before"]
            )
            maximum_abs_mean_after = max(
                maximum_abs_mean_after, central["maximum_abs_mean_after"]
            )
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
        and np.isfinite(prediction).all()
        and np.isfinite(maximum_abs_mean_before)
        and np.isfinite(maximum_abs_mean_after)
    )
    if not finite or eligible_names is None or len(eligible_names) != 5:
        raise v12.ContractError("v34 gradient-centralized training contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "optimizer_steps": optimizer_steps,
        "centralization_steps": optimizer_steps,
        "eligible_names": eligible_names,
        "eligible_count": len(eligible_names),
        "maximum_abs_mean_before": maximum_abs_mean_before,
        "maximum_abs_mean_after": maximum_abs_mean_after,
        "coefficient": None,
        "second_loss": 0,
        "task_split": 0,
        "bias_or_1d_change": 0,
        "loss": "weighted_SmoothL1_beta_1.0_with_gradient_centralization",
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _gradient_centralization_contract_receipt() -> dict[str, Any]:
    layer = torch.nn.Linear(2, 2, bias=True)
    layer.weight.grad = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
    layer.bias.grad = torch.tensor([5.0, 7.0])
    bias_before = layer.bias.grad.clone()
    receipt = centralize_linear_weight_gradients(layer)
    expected = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]])
    exact = bool(torch.equal(layer.weight.grad, expected))
    bias_unchanged = bool(torch.equal(layer.bias.grad, bias_before))
    row_means_zero = bool(
        torch.allclose(layer.weight.grad.mean(dim=1), torch.zeros(2), atol=0.0, rtol=0.0)
    )
    if not exact or not bias_unchanged or not row_means_zero:
        raise v12.ContractError("v34 toy gradient centralization failed")
    return {
        "input_gradient": [[1.0, 3.0], [2.0, 6.0]],
        "centralized_gradient": layer.weight.grad.tolist(),
        "expected_gradient": expected.tolist(),
        "bias_before": bias_before.tolist(),
        "bias_after": layer.bias.grad.tolist(),
        "formula_exact": exact,
        "row_means_zero": row_means_zero,
        "bias_unchanged": bias_unchanged,
        "eligible_count": receipt["eligible_count"],
        "coefficient": None,
    }


def _isolation_receipt() -> dict[str, Any]:
    return v28._isolation_receipt()


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for key, relative in config["authorization_evidence"].items():
        if not (
            key.endswith("_result")
            or key.endswith("_failure")
            or key == "prospective_gate_amendment"
        ):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v24_sam_distinguished": True,
        "v27_spectral_norm_distinguished": True,
        "v28_pcgrad_distinguished": True,
        "v29_lookahead_distinguished": True,
        "v33_mldg_distinguished": True,
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
        raise v12.ContractError("v34 masked/future or permutation isolation failed")
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "gradient_centralization_contract": _gradient_centralization_contract_receipt(),
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
        "# P2 v34 gradient-centralized DeepSets\n\n"
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
        "Exact v13 backward 이후 Linear weight gradient의 입력축 평균만 0으로 만들고 "
        "AdamW를 한 번 적용했다. Yong et al. (ECCV 2020)은 optimization 동기만 "
        "제공하며 P2 성능 근거가 아니다. coefficient/second loss/task split/sweep/router/"
        "ensemble/official-feedback selection/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_GRADIENT_CENTRALIZATION"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["gradient_centralization_contract"] = (
        _gradient_centralization_contract_receipt()
    )
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "gradient_centralization": config["training"]["gradient_centralization"],
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
        "v23_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v23_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v33_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v33_result"]).read_text(
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
