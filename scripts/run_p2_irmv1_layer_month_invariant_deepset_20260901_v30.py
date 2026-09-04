"""Run sealed P2 v30 fixed IRMv1 layer-month DeepSets exactly once."""

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

EXPERIMENT_ID = "p2_irmv1_layer_month_invariant_deepset_20260901_v30"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V30_IRMV1_LAYER_MONTH_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.irmv1_layer_month_invariant_deepset.result.20260901.v30"

_BASE_LOAD_CONFIG = v13.load_config
_BASE_RUN = v13.run
_BASE_DOMAIN_BALANCED_WEIGHTS = v13.domain_balanced_weights
_V13_RUNNER = v13.RUNNER
_ACTIVE_ENVIRONMENT_IDS: np.ndarray | None = None


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
    irm = training["irmv1"]
    safety = config["evaluation"]["safety_gate"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    audit = ROOT / config["authorization_evidence"]["audit_result"]
    if (
        training["architecture"]
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
        or irm["environment_axis"] != "target_layer_x_calendar_month"
        or irm["dummy_classifier_scale"] != 1.0
        or irm["coefficient"] != 1.0
        or irm["annealing"]
        or irm["coefficient_sweep"]
        or irm["environment_router"]
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
        raise v12.ContractError("v30 fixed scientific contract drift")
    return config


def domain_balanced_weights(
    layer: np.ndarray, local_time: pd.Series | pd.DatetimeIndex
) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture preregistered layer-month environments beside exact v13 weights."""
    global _ACTIVE_ENVIRONMENT_IDS
    local = pd.DatetimeIndex(local_time)
    layers = np.asarray(layer, dtype=np.int64)
    _ACTIVE_ENVIRONMENT_IDS = layers * 100 + local.month.to_numpy(dtype=np.int64)
    weights, receipt = _BASE_DOMAIN_BALANCED_WEIGHTS(layer, local_time)
    labels = sorted(np.unique(_ACTIVE_ENVIRONMENT_IDS).tolist())
    if sorted(np.unique(layers).tolist()) != [2, 3, 4] or not labels:
        raise v12.ContractError("v30 layer-month environment support drift")
    receipt["irmv1_environment_ids"] = labels
    receipt["irmv1_environment_count"] = len(labels)
    return weights, receipt


def irmv1_penalty(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    environment_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Mean squared group-risk gradient at the fixed dummy scale w=1."""
    if not (
        prediction.ndim == target.ndim == weights.ndim == environment_ids.ndim == 1
        and len(prediction) == len(target) == len(weights) == len(environment_ids)
    ):
        raise v12.ContractError("v30 IRMv1 vector contract drift")
    scale = torch.ones((), dtype=prediction.dtype, device=prediction.device)
    scale.requires_grad_(True)
    scaled = prediction * scale
    raw = F.smooth_l1_loss(scaled, target, beta=1.0, reduction="none")
    penalties: list[torch.Tensor] = []
    for label in torch.unique(environment_ids, sorted=True):
        selected = environment_ids.eq(label)
        local_weights = weights[selected]
        group_loss = (raw[selected] * local_weights).sum()
        group_loss = group_loss / local_weights.sum().clamp_min(1e-12)
        gradient = torch.autograd.grad(
            group_loss, scale, create_graph=True, retain_graph=True
        )[0]
        penalties.append(gradient.square())
    if not penalties:
        raise v12.ContractError("v30 minibatch has no IRMv1 environments")
    penalty = torch.stack(penalties).mean()
    return penalty, len(penalties)


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
    if _ACTIVE_ENVIRONMENT_IDS is None or len(_ACTIVE_ENVIRONMENT_IDS) != len(target):
        raise v12.ContractError("v30 active layer-month environments unavailable")
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
    coefficient = float(config["training"]["irmv1"]["coefficient"])
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    penalties: list[float] = []
    environment_counts: list[int] = []
    optimizer_steps = 0
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        numerator = 0.0
        denominator = 0.0
        epoch_penalties: list[float] = []
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            environment = torch.from_numpy(
                _ACTIVE_ENVIRONMENT_IDS[selected.numpy()]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction, batch[3], beta=1.0, reduction="none"
            )
            pooled = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            penalty, environment_count = irmv1_penalty(
                prediction, batch[3], batch[4], environment
            )
            loss = pooled + coefficient * penalty
            if not bool(torch.isfinite(loss)):
                raise v12.ContractError("v30 IRMv1 training loss is non-finite")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            numerator += float((raw_loss.detach() * batch[4]).sum().cpu())
            denominator += float(batch[4].sum().cpu())
            epoch_penalties.append(float(penalty.detach().cpu()))
            environment_counts.append(environment_count)
        losses.append(numerator / denominator)
        penalties.append(float(np.mean(epoch_penalties)))
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
        and np.isfinite(penalties).all()
        and np.isfinite(prediction).all()
    )
    if not finite or min(environment_counts, default=0) < 1:
        raise v12.ContractError("v30 training/IRMv1 contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "irmv1_penalty_first": penalties[0],
        "irmv1_penalty_last": penalties[-1],
        "irmv1_penalty_min": min(penalties),
        "irmv1_penalty_max": max(penalties),
        "irmv1_coefficient": coefficient,
        "dummy_classifier_scale": 1.0,
        "minimum_batch_environment_count": min(environment_counts),
        "maximum_batch_environment_count": max(environment_counts),
        "optimizer_steps": optimizer_steps,
        "optimizer": "AdamW",
        "annealing": 0,
        "coefficient_sweep": 0,
        "environment_router": 0,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _irmv1_contract_receipt() -> dict[str, Any]:
    prediction = torch.tensor([0.0, 2.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0])
    weights = torch.ones(2)
    environment = torch.tensor([201, 202])
    first, count = irmv1_penalty(prediction, target, weights, environment)
    second, _ = irmv1_penalty(prediction, target, weights, environment)
    exact = bool(torch.isclose(first, torch.tensor(2.0), atol=1e-7, rtol=0.0))
    deterministic = bool(torch.equal(first.detach(), second.detach()))
    if not exact or not deterministic or count != 2:
        raise v12.ContractError("v30 IRMv1 penalty formula contract failed")
    return {
        "environment_axis": "target_layer_x_calendar_month",
        "dummy_classifier_scale": 1.0,
        "coefficient": 1.0,
        "toy_environment_count": count,
        "toy_expected_penalty": 2.0,
        "toy_observed_penalty": float(first.detach()),
        "formula_exact": exact,
        "byte_identical_replay": deterministic,
        "annealing": False,
        "coefficient_sweep": False,
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
        config["authorization_evidence"]["v28_result"],
        config["authorization_evidence"]["v29_result"],
        config["authorization_evidence"]["prospective_gate_amendment"],
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v18_group_dro_distinguished": True,
        "v19_vrex_distinguished": True,
        "v20_coral_distinguished": True,
        "v23_input_gradient_distinguished": True,
        "v28_pcgrad_distinguished": True,
        "v29_lookahead_distinguished": True,
        "p1_report_only_irm_mention_excluded": True,
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
        raise v12.ContractError("v30 masked/future or permutation isolation failed")
    audit_path = ROOT / config["authorization_evidence"]["audit_result"]
    amendment = ROOT / config["authorization_evidence"]["prospective_gate_amendment"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "semantic_audit": semantic_audit(config),
        "irmv1_contract": _irmv1_contract_receipt(),
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
        "# P2 v30 fixed IRMv1 layer-month invariant DeepSets\n\n"
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
        "v13 science를 고정하고 layer×calendar-month group risk의 fixed classifier-scale "
        "gradient-square IRMv1 penalty(coefficient=1.0)만 추가했다. Arjovsky et al. "
        "(2019)은 representation 동기만 제공하며 P2 성능 근거가 아니다. "
        "anneal/sweep/router/ensemble/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_IRMV1_LAYER_MONTH_INVARIANCE"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["irmv1_contract"] = _irmv1_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "irmv1": config["training"]["irmv1"],
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
        "v29_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v29_result"]).read_text(
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
