"""Run sealed P2 v27 spectrally normalized DeepSets exactly once."""

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
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrizations

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_spectral_normalized_domain_balanced_deepset_20260901_v27"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V27_SPECTRAL_NORMALIZED_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.spectral_normalized_domain_balanced_deepset.result.20260901.v27"

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
    spectral = training["spectral_normalization"]
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    if (
        training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
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
        or spectral["implementation"]
        != "torch.nn.utils.parametrizations.spectral_norm"
        or spectral["target_operator_norm"] != 1.0
        or spectral["n_power_iterations_per_training_forward"] != 1
        or spectral["initialization_power_iterations"] != 15
        or spectral["preflight_contract_forwards"] != 15
        or spectral["eps"] != 1e-12
        or spectral["normalize_bias"]
        or spectral["coefficient_sweep"]
        or spectral["selective_layer_application"]
        or safety["minimum_fold_layer_non_harm_cells"] != 8
        or safety["total_fold_layer_cells"] != 9
        or safety["maximum_any_fold_layer_delta_rmse_C"] != 0.003
        or not amendment.is_file()
        or v12.sha256_file(amendment)
        != config["authorization_evidence"]["prospective_gate_amendment_sha256"]
    ):
        raise v12.ContractError("v27 fixed scientific contract drift")
    return config


def build_spectral_model() -> v12.VerticalDeepSet:
    """Apply the one fixed spectral reparameterization to all five Linear maps."""
    model = v12.VerticalDeepSet(8, 11, hidden=32)
    linear_count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parametrizations.spectral_norm(
                module,
                name="weight",
                n_power_iterations=1,
                eps=1e-12,
                dim=None,
            )
            linear_count += 1
    if linear_count != 5:
        raise v12.ContractError("v27 expected exactly five Linear maps")
    return model


def _effective_spectral_norms(model: nn.Module) -> list[float]:
    norms = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            norms.append(float(torch.linalg.matrix_norm(module.weight.detach(), ord=2)))
    return norms


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
    model = build_spectral_model().to(device)
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
            loss.backward()
            optimizer.step()
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
    norms = _effective_spectral_norms(model)
    finite = bool(np.isfinite(losses).all() and np.isfinite(prediction).all())
    if not finite or len(norms) != 5 or max(abs(value - 1.0) for value in norms) > 0.02:
        raise v12.ContractError("v27 training/spectral contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "spectral_normalized_linear_count": len(norms),
        "effective_spectral_norm_min": min(norms),
        "effective_spectral_norm_max": max(norms),
        "n_power_iterations_per_training_forward": 1,
        "input_gradient_penalty": 0,
        "parameter_neighborhood_perturbation": 0,
        "data_augmentation": 0,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _spectral_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(27)
    model = build_spectral_model()
    tokens = torch.randn(5, 5, 8)
    mask = torch.ones(5, 5)
    mask[:, -1] = 0.0
    context = torch.randn(5, 11)
    changed = tokens.clone()
    changed[:, -1] += 1000.0
    permutation = torch.tensor([4, 1, 3, 0, 2])
    model.train()
    with torch.no_grad():
        for _ in range(15):
            model(tokens, mask, context)
    model.eval()
    with torch.inference_mode():
        base = model(tokens, mask, context)
        masked = model(changed, mask, context)
        permuted = model(tokens[:, permutation], mask[:, permutation], context)
    norms = _effective_spectral_norms(model)
    mask_error = float(torch.max(torch.abs(base - masked)))
    permutation_error = float(torch.max(torch.abs(base - permuted)))
    parametrized_count = sum(
        int(isinstance(module, nn.Linear) and hasattr(module, "parametrizations"))
        for module in model.modules()
    )
    if (
        parametrized_count != 5
        or len(norms) != 5
        or max(abs(value - 1.0) for value in norms) > 1e-4
        or mask_error > 1e-6
        or permutation_error > 1e-6
    ):
        raise v12.ContractError("v27 spectral preflight contract failed")
    return {
        "spectral_normalized_linear_count": parametrized_count,
        "effective_spectral_norms": norms,
        "maximum_operator_norm_abs_error_from_one": max(
            abs(value - 1.0) for value in norms
        ),
        "masked_token_isolation_maximum_abs_error": mask_error,
        "permutation_invariance_maximum_abs_error": permutation_error,
        "bias_normalized": False,
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative in (
        config["authorization_evidence"]["v13_result"],
        config["authorization_evidence"]["v20_result"],
        config["authorization_evidence"]["v23_result"],
        config["authorization_evidence"]["v24_result"],
        config["authorization_evidence"]["v26_result"],
        config["authorization_evidence"]["prospective_gate_amendment"],
        "scripts/run_p2_layer_conditioned_month_covariance_alignment_20260901_v20.py",
        "scripts/run_p2_public_temperature_input_gradient_regularized_deepset_20260901_v23.py",
        "scripts/run_p2_sharpness_aware_domain_balanced_deepset_20260901_v24.py",
        "scripts/run_p2_layer_month_group_preserving_mixup_deepset_20260901_v26.py",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v20_coral_distinguished": True,
        "v23_input_gradient_distinguished": True,
        "v24_sam_distinguished": True,
        "v26_mixup_distinguished": True,
        "evidence_sha256": evidence,
    }


def prospective_fold_layer_gate(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    values = [
        float(layer["delta_rmse"])
        for fold in record["by_fold_layer"].values()
        for layer in fold.values()
    ]
    gate = config["evaluation"]["safety_gate"]
    non_harm = int(sum(value <= 0.0 for value in values))
    maximum = float(max(values))
    checks = {
        "minimum_eight_of_nine_non_harm": non_harm
        >= int(gate["minimum_fold_layer_non_harm_cells"]),
        "all_cells_within_plus_0_003C": maximum
        <= float(gate["maximum_any_fold_layer_delta_rmse_C"]),
    }
    return {
        "source": config["authorization_evidence"]["prospective_gate_amendment"],
        "source_sha256": config["authorization_evidence"][
            "prospective_gate_amendment_sha256"
        ],
        "non_harm_cells": non_harm,
        "total_cells": len(values),
        "non_harm_coverage": non_harm / len(values),
        "maximum_cell_delta_rmse_C": maximum,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def preflight() -> dict[str, Any]:
    _bind_base()
    config = load_config()
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "spectral_contract": _spectral_contract_receipt(),
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
    folds = item["by_fold"]
    local = item["prospective_fold_layer_gate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 v27 spectrally normalized domain-balanced DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled ΔRMSE `{item['delta_rmse']:+.9f} C`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점.\n\n"
        f"fold ΔRMSE: Sep-Oct `{folds['2024_sep_oct']['delta_rmse']:+.9f}`, "
        f"Jul-Aug `{folds['2025_jul_aug']['delta_rmse']:+.9f}`, "
        f"Nov-Dec `{folds['2025_nov_dec']['delta_rmse']:+.9f}`.\n\n"
        f"prospective fold×layer gate: `{local['pass']}`, non-harm "
        f"`{local['non_harm_cells']}/9`, maximum cell ΔRMSE "
        f"`{local['maximum_cell_delta_rmse_C']:+.9f} C`.\n\n"
        "v13 architecture/loss/domain weights/prefix+7d purge/seeds/epochs/blend/action cap을 "
        "고정하고 다섯 Linear weight의 operator norm을 1로 재매개화했다. Miyato et al. "
        "(ICLR 2018)은 표현 동기만 제공하며 P2 성능 근거가 아니다. selective application/"
        "coefficient sweep/router/ensemble/row deletion/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_SPECTRAL_NORMALIZATION"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["spectral_contract"] = _spectral_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "spectral_normalization": config["training"]["spectral_normalization"],
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
        "v23_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v23_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v24_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v24_result"]).read_text(
                encoding="utf-8"
            )
        )["candidate"]["delta_rmse"],
        "v26_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v26_result"]).read_text(
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
