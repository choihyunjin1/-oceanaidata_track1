"""Run sealed P2 v35 fixed-RAdam DeepSets exactly once."""

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

EXPERIMENT_ID = "p2_rectified_adam_domain_balanced_deepset_20260901_v35"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V35_FIXED_RADAM_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.rectified_adam_domain_balanced_deepset.result.20260901.v35"

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
    optimizer = training["optimizer"]
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    audit = ROOT / config["authorization_evidence"]["audit_result"]
    if (
        training["architecture"]
        != "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2"
        or training["weighting"]
        != "equal_total_mass_per_target_layer_x_calendar_month_then_equal_KST_day_then_equal_row"
        or training["objective"] != "weighted_SmoothL1_beta_1.0_with_fixed_RAdam"
        or optimizer["class"] != "torch.optim.RAdam"
        or optimizer["learning_rate"] != 0.001
        or optimizer["betas"] != [0.9, 0.999]
        or optimizer["epsilon"] != 1e-8
        or optimizer["weight_decay"] != 0.0001
        or optimizer["decoupled_weight_decay"] is not True
        or optimizer["warmup"]
        or optimizer["scheduler"]
        or optimizer["lookahead"]
        or optimizer["gradient_projection"]
        or optimizer["sweep"]
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
        raise v12.ContractError("v35 fixed scientific contract drift")
    return config


def _make_optimizer(
    parameters: Any, config: dict[str, Any]
) -> torch.optim.RAdam:
    optimizer = config["training"]["optimizer"]
    return torch.optim.RAdam(
        parameters,
        lr=float(optimizer["learning_rate"]),
        betas=tuple(float(value) for value in optimizer["betas"]),
        eps=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
        decoupled_weight_decay=bool(optimizer["decoupled_weight_decay"]),
        foreach=False,
    )


def _optimizer_state_receipt(optimizer: torch.optim.RAdam) -> dict[str, Any]:
    states = list(optimizer.state.values())
    if not states:
        raise v12.ContractError("v35 RAdam has no optimizer state")
    steps = [int(state["step"].detach().cpu().item()) for state in states]
    finite = all(
        bool(torch.isfinite(state[name]).all())
        for state in states
        for name in ("exp_avg", "exp_avg_sq")
    )
    group = optimizer.param_groups[0]
    return {
        "class": type(optimizer).__name__,
        "learning_rate": float(group["lr"]),
        "betas": [float(value) for value in group["betas"]],
        "epsilon": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "decoupled_weight_decay": bool(group["decoupled_weight_decay"]),
        "foreach": group["foreach"],
        "state_count": len(states),
        "state_step_min": min(steps),
        "state_step_max": max(steps),
        "state_keys": sorted(states[0]),
        "state_finite": finite,
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
    optimizer = _make_optimizer(model.parameters(), config)
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
                raise v12.ContractError("v35 weighted SmoothL1 is non-finite")
            loss.backward()
            if not all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise v12.ContractError("v35 gradient contract failed")
            optimizer.step()
            optimizer_steps += 1
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
        losses.append(numerator / denominator)

    state = _optimizer_state_receipt(optimizer)
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
        and state["state_finite"]
    )
    if (
        not finite
        or state["class"] != "RAdam"
        or state["state_count"] != 10
        or state["state_step_min"] != optimizer_steps
        or state["state_step_max"] != optimizer_steps
    ):
        raise v12.ContractError("v35 fixed-RAdam training contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "optimizer_steps": optimizer_steps,
        "optimizer": state,
        "warmup_steps": 0,
        "scheduler_steps": 0,
        "lookahead_steps": 0,
        "gradient_projection_steps": 0,
        "loss": "weighted_SmoothL1_beta_1.0_with_fixed_RAdam",
        "loss_finite": finite,
        "row_deletion": 0,
    }


def _radam_contract_receipt(config: dict[str, Any]) -> dict[str, Any]:
    def run_once() -> tuple[list[float], dict[str, Any]]:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = _make_optimizer([parameter], config)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (parameter.square() * torch.tensor([1.0, 0.5])).sum()
            loss.backward()
            optimizer.step()
        return parameter.detach().tolist(), _optimizer_state_receipt(optimizer)

    first_values, first = run_once()
    second_values, second = run_once()
    deterministic = first_values == second_values and first == second
    keys_ok = first["state_keys"] == ["exp_avg", "exp_avg_sq", "step"]
    contract = bool(
        deterministic
        and keys_ok
        and first["class"] == "RAdam"
        and first["learning_rate"] == 0.001
        and first["betas"] == [0.9, 0.999]
        and first["epsilon"] == 1e-8
        and first["weight_decay"] == 0.0001
        and first["decoupled_weight_decay"] is True
        and first["foreach"] is False
        and first["state_step_min"] == 2
        and first["state_step_max"] == 2
        and first["state_finite"]
    )
    if not contract:
        raise v12.ContractError("v35 toy RAdam contract failed")
    return {
        "parameter_after_two_steps": first_values,
        "repeat_parameter_after_two_steps": second_values,
        "deterministic": deterministic,
        "state_keys_exact": keys_ok,
        "contract_exact": contract,
        **first,
        "warmup": False,
        "scheduler": False,
        "sweep": False,
    }


def _isolation_receipt() -> dict[str, Any]:
    return v28._isolation_receipt()


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for key, relative in config["authorization_evidence"].items():
        if not (key.endswith("_result") or key == "prospective_gate_amendment"):
            continue
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v13_adamw_distinguished": True,
        "v24_sam_distinguished": True,
        "v29_lookahead_distinguished": True,
        "v34_gradient_centralization_distinguished": True,
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
        raise v12.ContractError("v35 masked/future or permutation isolation failed")
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "radam_contract": _radam_contract_receipt(config),
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
        "# P2 v35 fixed-RAdam DeepSets\n\n"
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
        "Exact v13 AdamW만 fixed RAdam(lr=.001, betas=.9/.999, eps=1e-8, "
        "decoupled WD=.0001)으로 교체했다. Liu et al. (ICLR 2020)은 "
        "optimization 동기만 제공하며 P2 성능 근거가 아니다. warmup/scheduler/"
        "sweep/router/ensemble/official-feedback selection/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_FIXED_RADAM"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["radam_contract"] = _radam_contract_receipt(config)
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "optimizer": config["training"]["optimizer"],
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
            for name in ("v13", "v23", "v24", "v29", "v34")
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
