"""Run the sealed P2 v24 vanilla-SAM optimizer geometry exactly once."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
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
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_sharpness_aware_domain_balanced_deepset_20260901_v24"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V24_SHARPNESS_AWARE_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.sharpness_aware_domain_balanced_deepset.result.20260901.v24"

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
    geometry = training["optimizer_geometry"]
    if (
        training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["epochs"] != 60
        or training["seeds"] != [20260901, 20260902, 20260903]
        or training["maximum_fit_count"] != 9
        or training["maximum_final_action_C"] != 0.5
        or training["model_minus_champion_clip_C"] != 2.5
        or training["champion_preserving_weight"] != 0.8
        or training["model_weight"] != 0.2
        or training["row_deletion"]
        or training["input_perturbation"]
        or training["data_augmentation"]
        or geometry["name"] != "vanilla_SAM"
        or geometry["rho"] != 0.05
        or geometry["gradient_norm"]
        != "global_L2_over_all_trainable_parameters"
        or geometry["adaptive_asam"]
        or geometry["rho_sweep"]
        or geometry["scheduler"]
        or geometry["base_optimizer"] != "AdamW"
        or not geometry["apply_every_batch"]
    ):
        raise v12.ContractError("v24 fixed scientific contract drift")
    return config


def global_gradient_l2_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    """Return one L2 norm across every existing parameter gradient."""
    gradients = [
        parameter.grad.detach().norm(p=2)
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    return torch.stack(gradients).norm(p=2)


@torch.no_grad()
def sam_ascent_step(
    parameters: Iterable[nn.Parameter], rho: float
) -> tuple[list[tuple[nn.Parameter, Tensor]], float, float]:
    """Perturb parameters along the global normalized gradient."""
    items = list(parameters)
    norm = global_gradient_l2_norm(items)
    norm_value = float(norm.detach().cpu())
    if norm_value <= 0.0:
        return [], norm_value, 0.0
    scale = float(rho) / (norm_value + 1e-12)
    backups: list[tuple[nn.Parameter, Tensor]] = []
    squared_radius = 0.0
    for parameter in items:
        if not parameter.requires_grad or parameter.grad is None:
            continue
        backup = parameter.detach().clone()
        perturbation = parameter.grad.detach() * scale
        parameter.add_(perturbation)
        backups.append((parameter, backup))
        squared_radius += float(perturbation.square().sum().detach().cpu())
    return backups, norm_value, float(np.sqrt(squared_radius))


@torch.no_grad()
def sam_restore(backups: list[tuple[nn.Parameter, Tensor]]) -> None:
    """Restore the unperturbed parameters exactly before the base step."""
    for parameter, backup in backups:
        parameter.copy_(backup)


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
    rho = float(config["training"]["optimizer_geometry"]["rho"])
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first_losses: list[float] = []
    second_losses: list[float] = []
    gradient_norms: list[float] = []
    perturbation_radii: list[float] = []
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        first_numerator = 0.0
        second_numerator = 0.0
        denominator = 0.0
        epoch_norms: list[float] = []
        epoch_radii: list[float] = []
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            first_prediction = model(batch[0], batch[1], batch[2])
            first_raw = F.smooth_l1_loss(
                first_prediction, batch[3], beta=1.0, reduction="none"
            )
            first_loss = (first_raw * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            first_loss.backward()
            backups, gradient_norm, radius = sam_ascent_step(model.parameters(), rho)
            optimizer.zero_grad(set_to_none=True)
            second_prediction = model(batch[0], batch[1], batch[2])
            second_raw = F.smooth_l1_loss(
                second_prediction, batch[3], beta=1.0, reduction="none"
            )
            second_loss = (second_raw * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            second_loss.backward()
            sam_restore(backups)
            optimizer.step()
            weight_sum = float(batch[4].sum().detach().cpu())
            first_numerator += float((first_raw.detach() * batch[4]).sum().cpu())
            second_numerator += float((second_raw.detach() * batch[4]).sum().cpu())
            denominator += weight_sum
            epoch_norms.append(gradient_norm)
            epoch_radii.append(radius)
        first_losses.append(first_numerator / denominator)
        second_losses.append(second_numerator / denominator)
        gradient_norms.append(float(np.mean(epoch_norms)))
        perturbation_radii.append(float(np.mean(epoch_radii)))
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
        np.isfinite(first_losses).all()
        and np.isfinite(second_losses).all()
        and np.isfinite(gradient_norms).all()
        and np.isfinite(perturbation_radii).all()
        and np.isfinite(prediction).all()
    )
    if not finite:
        raise v12.ContractError("v24 training or prediction is non-finite")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(first_losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": first_losses[0],
        "loss_last": first_losses[-1],
        "sam_second_loss_first": second_losses[0],
        "sam_second_loss_last": second_losses[-1],
        "global_gradient_norm_first": gradient_norms[0],
        "global_gradient_norm_last": gradient_norms[-1],
        "perturbation_radius_first": perturbation_radii[0],
        "perturbation_radius_last": perturbation_radii[-1],
        "sam_rho": rho,
        "adaptive_asam": False,
        "parameter_restore_before_adamw_step": True,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _masked_token_isolation_receipt() -> dict[str, Any]:
    torch.manual_seed(24)
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
        raise v12.ContractError("masked/future token changes prediction")
    return {"maximum_abs_error": error, "perturbed_masked_tokens": 4}


def _sam_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(24)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    features = torch.tensor([[0.5, -1.0, 2.0], [1.0, 0.25, -0.5]])
    target = torch.tensor([0.75, -0.25])
    original = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    first_loss = F.smooth_l1_loss(model(features).squeeze(-1), target, beta=1.0)
    first_loss.backward()
    backups, gradient_norm, radius = sam_ascent_step(model.parameters(), 0.05)
    optimizer.zero_grad(set_to_none=True)
    second_loss = F.smooth_l1_loss(model(features).squeeze(-1), target, beta=1.0)
    second_loss.backward()
    sam_restore(backups)
    restore_exact = all(
        torch.equal(parameter.detach(), before)
        for parameter, before in zip(model.parameters(), original, strict=True)
    )
    if abs(radius - 0.05) > 1e-6 or not restore_exact or not torch.isfinite(second_loss):
        raise v12.ContractError("SAM perturbation/restoration contract failed")
    zero_model = nn.Linear(2, 1)
    zero_before = [parameter.detach().clone() for parameter in zero_model.parameters()]
    empty, zero_norm, zero_radius = sam_ascent_step(zero_model.parameters(), 0.05)
    zero_noop = not empty and all(
        torch.equal(parameter.detach(), before)
        for parameter, before in zip(zero_model.parameters(), zero_before, strict=True)
    )
    if not zero_noop or zero_norm != 0.0 or zero_radius != 0.0:
        raise v12.ContractError("SAM zero-gradient branch is not an exact no-op")
    return {
        "rho": 0.05,
        "first_gradient_global_l2_norm": gradient_norm,
        "actual_parameter_perturbation_l2_norm": radius,
        "parameter_restore_bit_exact": restore_exact,
        "second_loss_finite": bool(torch.isfinite(second_loss)),
        "zero_gradient_exact_noop": zero_noop,
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative in (
        config["authorization_evidence"]["v13_result"],
        config["authorization_evidence"]["v18_result"],
        config["authorization_evidence"]["v19_result"],
        config["authorization_evidence"]["v20_result"],
        config["authorization_evidence"]["v23_result"],
        "scripts/run_p2_regularized_layer_month_group_dro_20260901_v18.py",
        "scripts/run_p2_layer_month_risk_variance_rex_20260901_v19.py",
        "scripts/run_p2_public_temperature_input_gradient_regularized_deepset_20260901_v23.py",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v23_input_gradient_distinguished": True,
        "v18_worst_group_distinguished": True,
        "v19_risk_variance_distinguished": True,
        "adamw_weight_decay_distinguished": True,
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
        "sam_contract": _sam_contract_receipt(),
        "masked_future_token_isolation": _masked_token_isolation_receipt(),
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
        "# P2 v24 sharpness-aware domain-balanced DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled ΔRMSE `{item['delta_rmse']:+.9f} C`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점.\n\n"
        f"fold ΔRMSE: Sep-Oct `{folds['2024_sep_oct']['delta_rmse']:+.9f}`, "
        f"Jul-Aug `{folds['2025_jul_aug']['delta_rmse']:+.9f}`, "
        f"Nov-Dec `{folds['2025_nov_dec']['delta_rmse']:+.9f}`.\n\n"
        "v13 architecture/loss/domain weights/prefix+7d purge/seeds/epochs/blend/action cap을 고정하고, "
        "rho=0.05 vanilla SAM parameter-neighborhood optimizer geometry만 추가했다. Foret et al. "
        "(ICLR 2021)는 optimizer 동기만 제공하며 P2 성능 근거가 아니다. ASAM/rho sweep/"
        "scheduler/router/ensemble/row deletion/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_SHARPNESS_AWARE_OPTIMIZER"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["sam_contract"] = _sam_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer_geometry": config["training"]["optimizer_geometry"],
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
    }
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    result["hashes"]["v23_result"] = v12.sha256_file(
        ROOT / config["authorization_evidence"]["v23_result"]
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
