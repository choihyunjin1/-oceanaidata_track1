"""Run sealed P2 v25 conditional heteroscedastic Gaussian DeepSets once."""

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

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_heteroscedastic_gaussian_domain_balanced_deepset_20260901_v25"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V25_HETEROSCEDASTIC_GAUSSIAN_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.heteroscedastic_gaussian_domain_balanced_deepset.result.20260901.v25"

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
    gaussian = training["heteroscedastic_gaussian"]
    if (
        training["architecture"]
        != "v13_exact_shared_element_mean_max_pool_head32x2_with_final_mean_logvariance_outputs"
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
        or gaussian["mean_output_index"] != 0
        or gaussian["log_variance_output_index"] != 1
        or gaussian["log_variance_min"] != -6.0
        or gaussian["log_variance_max"] != 3.0
        or not gaussian["inference_uses_mean_only"]
        or gaussian["variance_router"]
        or gaussian["variance_abstention"]
        or gaussian["variance_head_sweep"]
    ):
        raise v12.ContractError("v25 fixed scientific contract drift")
    return config


def make_model() -> v12.VerticalDeepSet:
    """Keep v13 body exactly and replace only the final scalar head by two outputs."""
    model = v12.VerticalDeepSet(8, 11, hidden=32)
    model.head[-1] = nn.Linear(32, 2)
    return model


def heteroscedastic_gaussian_nll(
    output: Tensor, target: Tensor, log_variance_min: float, log_variance_max: float
) -> tuple[Tensor, Tensor]:
    if output.ndim != 2 or output.shape[1] != 2 or target.shape != output[:, 0].shape:
        raise v12.ContractError("v25 mean/log-variance output geometry drift")
    mean = output[:, 0]
    log_variance = output[:, 1].clamp(log_variance_min, log_variance_max)
    per_row = 0.5 * torch.exp(-log_variance) * (target - mean).square()
    per_row = per_row + 0.5 * log_variance
    return per_row, log_variance


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
    model = make_model().to(device)
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
    gaussian = config["training"]["heteroscedastic_gaussian"]
    low = float(gaussian["log_variance_min"])
    high = float(gaussian["log_variance_max"])
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
            per_row, _log_variance = heteroscedastic_gaussian_nll(
                model(batch[0], batch[1], batch[2]), batch[3], low, high
            )
            loss = (per_row * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
            numerator += float((per_row.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
        losses.append(numerator / denominator)
    model.eval()
    means: list[np.ndarray] = []
    raw_log_variances: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_tokens), batch_size):
            stop = start + batch_size
            output = model(
                torch.from_numpy(query_tokens[start:stop]).to(device),
                torch.from_numpy(query_mask[start:stop]).to(device),
                torch.from_numpy(query_context[start:stop]).to(device),
            )
            means.append(output[:, 0].cpu().numpy())
            raw_log_variances.append(output[:, 1].cpu().numpy())
    prediction = np.concatenate(means).astype(float)
    raw_log_variance = np.concatenate(raw_log_variances).astype(float)
    clipped_log_variance = np.clip(raw_log_variance, low, high)
    finite = bool(
        np.isfinite(losses).all()
        and np.isfinite(prediction).all()
        and np.isfinite(raw_log_variance).all()
    )
    if not finite:
        raise v12.ContractError("v25 training/prediction/log-variance is non-finite")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "mean_only_inference": True,
        "log_variance_min_contract": low,
        "log_variance_max_contract": high,
        "query_log_variance_min": float(np.min(clipped_log_variance)),
        "query_log_variance_p50": float(np.median(clipped_log_variance)),
        "query_log_variance_p99": float(np.quantile(clipped_log_variance, 0.99)),
        "query_log_variance_max": float(np.max(clipped_log_variance)),
        "query_log_variance_clipped_share": float(
            np.mean((raw_log_variance < low) | (raw_log_variance > high))
        ),
        "variance_router": 0,
        "variance_abstention": 0,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _nll_formula_receipt() -> dict[str, Any]:
    output = torch.tensor([[2.0, 0.0], [1.0, float(np.log(4.0))]])
    target = torch.tensor([0.0, 3.0])
    actual, log_variance = heteroscedastic_gaussian_nll(output, target, -6.0, 3.0)
    expected = torch.tensor([2.0, 0.5 + 0.5 * float(np.log(4.0))])
    error = float(torch.max(torch.abs(actual - expected)))
    if error > 1e-6 or float(log_variance.min()) < -6.0 or float(log_variance.max()) > 3.0:
        raise v12.ContractError("v25 Gaussian NLL formula contract failed")
    return {
        "maximum_abs_formula_error": error,
        "log_variance_min": float(log_variance.min()),
        "log_variance_max": float(log_variance.max()),
    }


def _model_contract_receipt() -> dict[str, Any]:
    torch.manual_seed(25)
    model = make_model().eval()
    tokens = torch.randn(4, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    context = torch.randn(4, 11)
    order = torch.tensor([4, 2, 0, 3, 1])
    changed_masked = tokens.clone()
    changed_masked[~mask.bool()] += 1000.0
    with torch.inference_mode():
        original = model(tokens, mask, context)
        permuted = model(tokens[:, order], mask[:, order], context)
        masked_changed = model(changed_masked, mask, context)
    permutation_error = float(torch.max(torch.abs(original - permuted)))
    masked_error = float(torch.max(torch.abs(original - masked_changed)))
    mean_before = original[:, 0].clone()
    final = model.head[-1]
    with torch.no_grad():
        final.weight[1].add_(1000.0)
        final.bias[1].add_(1000.0)
        mean_after = model(tokens, mask, context)[:, 0]
    mean_error = float(torch.max(torch.abs(mean_before - mean_after)))
    if max(permutation_error, masked_error, mean_error) > 1e-6:
        raise v12.ContractError("v25 model invariance/mean-only contract failed")
    return {
        "output_width": 2,
        "permutation_maximum_abs_error": permutation_error,
        "masked_future_token_maximum_abs_error": masked_error,
        "mean_invariance_to_variance_head_perturbation_error": mean_error,
    }


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative in (
        config["authorization_evidence"]["v13_result"],
        config["authorization_evidence"]["v20_result"],
        config["authorization_evidence"]["v21_result"],
        config["authorization_evidence"]["v23_result"],
        config["authorization_evidence"]["v24_result"],
        "scripts/run_p2_fixed_student_t_robust_deepset_20260901_v21.py",
        "scripts/run_p2_public_temperature_input_gradient_regularized_deepset_20260901_v23.py",
        "scripts/run_p2_sharpness_aware_domain_balanced_deepset_20260901_v24.py",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v21_fixed_student_t_distinguished": True,
        "v23_input_gradient_distinguished": True,
        "v24_sam_distinguished": True,
        "post_model_outlier_rules_distinguished": True,
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
        "nll_formula": _nll_formula_receipt(),
        "model_contract": _model_contract_receipt(),
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
        "# P2 v25 heteroscedastic Gaussian domain-balanced DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled ΔRMSE `{item['delta_rmse']:+.9f} C`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점.\n\n"
        f"fold ΔRMSE: Sep-Oct `{folds['2024_sep_oct']['delta_rmse']:+.9f}`, "
        f"Jul-Aug `{folds['2025_jul_aug']['delta_rmse']:+.9f}`, "
        f"Nov-Dec `{folds['2025_nov_dec']['delta_rmse']:+.9f}`.\n\n"
        "v13 shared element/pooling/hidden head/domain weights/prefix+7d purge/seeds/epochs/blend/"
        "action cap을 고정하고 final mean+conditional log-variance head와 bounded Gaussian NLL만 "
        "추가했다. inference는 mean만 사용한다. Kendall and Gal (NeurIPS 2017)은 learned "
        "attenuation 동기만 제공하며 P2 성능 근거가 아니다. variance router/abstention/sweep/"
        "row deletion/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_HETEROSCEDASTIC_GAUSSIAN"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["nll_formula"] = _nll_formula_receipt()
    result["model_contract"] = _model_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "heteroscedastic_gaussian": config["training"]["heteroscedastic_gaussian"],
            "input_perturbation": 0,
            "data_augmentation": 0,
        }
    )
    result["comparison_to_preserved_candidates"] = {
        "use": "ledger_only_no_posthoc_selection_router_or_ensemble",
        "v13_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v13_result"]).read_text(encoding="utf-8")
        )["candidate"]["delta_rmse"],
        "v20_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v20_result"]).read_text(encoding="utf-8")
        )["candidate"]["delta_rmse"],
        "v23_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v23_result"]).read_text(encoding="utf-8")
        )["candidate"]["delta_rmse"],
        "v24_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v24_result"]).read_text(encoding="utf-8")
        )["candidate"]["delta_rmse"],
    }
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    result["hashes"]["v24_result"] = v12.sha256_file(
        ROOT / config["authorization_evidence"]["v24_result"]
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
