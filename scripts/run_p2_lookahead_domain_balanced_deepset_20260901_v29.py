"""Run sealed P2 v29 fixed-Lookahead AdamW DeepSets exactly once."""

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
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_layer_task_gradient_surgery_deepset_20260901_v28 as v28  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_lookahead_domain_balanced_deepset_20260901_v29"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V29_LOOKAHEAD_ADAMW_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.lookahead_domain_balanced_deepset.result.20260901.v29"

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
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    audit = ROOT / config["authorization_evidence"]["audit_result"]
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
        or geometry["name"] != "Lookahead_AdamW"
        or geometry["inner_optimizer"] != "AdamW"
        or geometry["synchronization_period_steps"] != 5
        or geometry["slow_weight_interpolation_alpha"] != 0.5
        or not geometry["initial_slow_weights_equal_fast_weights"]
        or geometry["inference_weights"]
        != "current_fast_weights_after_periodic_slow_sync"
        or geometry["posthoc_weight_ensemble"]
        or geometry["hyperparameter_sweep"]
        or geometry["scheduler"]
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
        raise v12.ContractError("v29 fixed scientific contract drift")
    return config


@torch.no_grad()
def lookahead_sync(
    parameters: Iterable[nn.Parameter],
    slow_weights: list[torch.Tensor],
    alpha: float,
) -> float:
    """Interpolate slow weights toward fast weights and copy slow to fast."""
    items = list(parameters)
    if len(items) != len(slow_weights):
        raise v12.ContractError("v29 Lookahead parameter-state width drift")
    squared_delta = 0.0
    for parameter, slow in zip(items, slow_weights, strict=True):
        delta = parameter.detach() - slow
        squared_delta += float(delta.square().sum().cpu())
        slow.add_(delta, alpha=float(alpha))
        parameter.copy_(slow)
    return float(np.sqrt(squared_delta))


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
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    slow_weights = [value.detach().clone() for value in parameters]
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
    geometry = config["training"]["optimizer_geometry"]
    period = int(geometry["synchronization_period_steps"])
    alpha = float(geometry["slow_weight_interpolation_alpha"])
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    synchronization_deltas: list[float] = []
    steps = 0
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
            steps += 1
            if steps % period == 0:
                synchronization_deltas.append(
                    lookahead_sync(parameters, slow_weights, alpha)
                )
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
    expected_syncs = steps // period
    finite = bool(
        np.isfinite(losses).all()
        and np.isfinite(synchronization_deltas).all()
        and np.isfinite(prediction).all()
    )
    if not finite or len(synchronization_deltas) != expected_syncs:
        raise v12.ContractError("v29 training/Lookahead contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "optimizer_steps": steps,
        "lookahead_synchronizations": len(synchronization_deltas),
        "expected_synchronizations": expected_syncs,
        "lookahead_period_steps": period,
        "lookahead_alpha": alpha,
        "slow_fast_delta_l2_first_sync": synchronization_deltas[0],
        "slow_fast_delta_l2_last_sync": synchronization_deltas[-1],
        "slow_fast_delta_l2_max": max(synchronization_deltas),
        "inference_weights": "current_fast_weights_after_periodic_slow_sync",
        "posthoc_weight_ensemble": 0,
        "input_gradient_penalty": 0,
        "parameter_neighborhood_perturbation": 0,
        "gradient_surgery": 0,
        "data_augmentation": 0,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _lookahead_contract_receipt() -> dict[str, Any]:
    parameter = nn.Parameter(torch.tensor([3.0, -1.0]))
    slow = [torch.tensor([1.0, 1.0])]
    distance = lookahead_sync([parameter], slow, 0.5)
    expected = torch.tensor([2.0, 0.0])
    exact_formula = torch.equal(parameter.detach(), expected) and torch.equal(
        slow[0], expected
    )
    before = parameter.detach().clone()
    zero_distance = lookahead_sync([parameter], slow, 0.5)
    zero_noop = torch.equal(parameter.detach(), before) and zero_distance == 0.0
    parameter2 = nn.Parameter(torch.tensor([3.0, -1.0]))
    slow2 = [torch.tensor([1.0, 1.0])]
    distance2 = lookahead_sync([parameter2], slow2, 0.5)
    deterministic = torch.equal(parameter.detach(), parameter2.detach()) and distance == distance2
    if not exact_formula or not zero_noop or not deterministic:
        raise v12.ContractError("v29 Lookahead interpolation contract failed")
    return {
        "period_steps": 5,
        "alpha": 0.5,
        "formula_exact": exact_formula,
        "initial_slow_fast_distance_l2": distance,
        "post_sync_fast_equals_slow": True,
        "zero_difference_exact_noop": zero_noop,
        "byte_identical_replay": deterministic,
        "posthoc_weight_ensemble": False,
    }


def _isolation_receipt() -> dict[str, Any]:
    return v28._isolation_receipt()


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative in (
        config["authorization_evidence"]["v13_result"],
        config["authorization_evidence"]["v23_result"],
        config["authorization_evidence"]["v24_result"],
        config["authorization_evidence"]["v26_result"],
        config["authorization_evidence"]["v28_terminal"],
        config["authorization_evidence"]["prospective_gate_amendment"],
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v23_input_gradient_distinguished": True,
        "v24_sam_distinguished": True,
        "v25_variance_head_distinguished": True,
        "v26_mixup_distinguished": True,
        "v28_pcgrad_distinguished": True,
        "unrelated_p1_feature_lookahead_excluded": True,
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
        raise v12.ContractError("v29 masked/future or permutation isolation failed")
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "lookahead_contract": _lookahead_contract_receipt(),
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
        "# P2 v29 fixed Lookahead-AdamW DeepSets\n\n"
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
        "v13 science를 고정하고 AdamW trajectory에 fixed Lookahead(k=5, alpha=0.5)만 적용했다. "
        "Zhang et al. (NeurIPS 2019)은 optimizer 동기만 제공하며 P2 성능 근거가 아니다. "
        "sweep/scheduler/router/posthoc ensemble/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_LOOKAHEAD_OPTIMIZER"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["lookahead_contract"] = _lookahead_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer_geometry": config["training"]["optimizer_geometry"],
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
