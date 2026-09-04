"""Run sealed P2 v26 layer-month group-preserving MixUp once."""

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
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

EXPERIMENT_ID = "p2_layer_month_group_preserving_mixup_deepset_20260901_v26"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V26_LAYER_MONTH_GROUP_MIXUP_DEEPSET_BLEND020"
RESULT_SCHEMA = "p2.layer_month_group_preserving_mixup_deepset.result.20260901.v26"

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
    mixup = training["mixup"]
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
        or mixup["alpha"] != 0.2
        or mixup["partner_group"] != "exact_target_layer_x_calendar_month"
        or mixup["partner_rule"]
        != "deterministic_cyclic_within_each_minibatch_group"
        or mixup["token_mask_rule"] != "intersection"
        or mixup["minimum_intersection_tokens"] != 2
        or mixup["insufficient_intersection_rule"]
        != "bit_exact_original_row_noop"
        or mixup["cross_group_mixing"]
        or mixup["alpha_sweep"]
        or mixup["inference_augmentation"]
    ):
        raise v12.ContractError("v26 fixed scientific contract drift")
    return config


def domain_balanced_weights(
    layer: np.ndarray, local_time: pd.Series | pd.DatetimeIndex
) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture truth-independent layer-month IDs next to exact v13 weights."""
    global _ACTIVE_ENVIRONMENT_IDS
    local = pd.DatetimeIndex(local_time)
    layer_values = np.asarray(layer, dtype=np.int64)
    _ACTIVE_ENVIRONMENT_IDS = layer_values * 100 + local.month.to_numpy(np.int64)
    weights, receipt = _BASE_DOMAIN_BALANCED_WEIGHTS(layer_values, local)
    receipt["mixup_environment_count"] = int(len(np.unique(_ACTIVE_ENVIRONMENT_IDS)))
    receipt["mixup_environment_encoding"] = "target_layer_times_100_plus_calendar_month"
    return weights, receipt


def group_preserving_mixup(
    batch: list[torch.Tensor],
    environment_ids: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Create same-environment convex examples with intersection masks."""
    rows = len(environment_ids)
    if rows != len(batch[0]) or len(batch) != 5:
        raise v12.ContractError("v26 batch/environment geometry drift")
    partner = np.arange(rows, dtype=np.int64)
    for environment in np.unique(environment_ids):
        positions = np.flatnonzero(environment_ids == environment)
        if len(positions) > 1:
            partner[positions] = np.roll(positions, 1)
    cross_group = int(np.sum(environment_ids != environment_ids[partner]))
    if cross_group:
        raise v12.ContractError("v26 cross-group partner detected")
    lam_np = rng.beta(alpha, alpha, size=rows).astype(np.float32)
    device = batch[0].device
    partner_tensor = torch.from_numpy(partner).to(device=device)
    identity = torch.arange(rows, device=device)
    intersection = batch[1] * batch[1][partner_tensor]
    valid = intersection.sum(dim=1) >= 2.0
    partner_tensor = torch.where(valid, partner_tensor, identity)
    lam = torch.from_numpy(lam_np).to(device=device)
    lam = torch.where(valid, lam, torch.ones_like(lam))
    intersection = batch[1] * batch[1][partner_tensor]
    token_lam = lam[:, None, None]
    row_lam = lam[:, None]
    mixed = [
        token_lam * batch[0] + (1.0 - token_lam) * batch[0][partner_tensor],
        intersection,
        row_lam * batch[2] + (1.0 - row_lam) * batch[2][partner_tensor],
        lam * batch[3] + (1.0 - lam) * batch[3][partner_tensor],
        lam * batch[4] + (1.0 - lam) * batch[4][partner_tensor],
    ]
    mixed_rows = valid & partner_tensor.ne(identity)
    receipt = {
        "rows": rows,
        "mixed_rows": int(mixed_rows.sum().detach().cpu()),
        "mask_intersection_fallback_rows": int((~valid).sum().detach().cpu()),
        "singleton_group_noop_rows": int(
            (valid & partner_tensor.eq(identity)).sum().detach().cpu()
        ),
        "cross_group_pairs": cross_group,
        "minimum_output_mask_tokens": float(mixed[1].sum(dim=1).min().detach().cpu()),
        "lambda_mean_mixed": (
            float(lam[mixed_rows].mean().detach().cpu())
            if bool(mixed_rows.any())
            else 1.0
        ),
    }
    return mixed, receipt


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
        raise v12.ContractError("v26 active layer-month environment IDs unavailable")
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
    alpha = float(config["training"]["mixup"]["alpha"])
    batch_size = int(config["training"]["batch_size"])
    order_generator = torch.Generator(device="cpu").manual_seed(seed)
    mix_generator = np.random.default_rng(seed)
    losses: list[float] = []
    total_rows = 0
    total_mixed = 0
    total_fallback = 0
    total_singleton = 0
    total_cross_group = 0
    minimum_mask_tokens = float("inf")
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=order_generator)
        numerator = 0.0
        denominator = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            environments = _ACTIVE_ENVIRONMENT_IDS[selected.numpy()]
            mixed, receipt = group_preserving_mixup(
                batch, environments, alpha, mix_generator
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(mixed[0], mixed[1], mixed[2])
            raw_loss = F.smooth_l1_loss(
                prediction, mixed[3], beta=1.0, reduction="none"
            )
            loss = (raw_loss * mixed[4]).sum() / mixed[4].sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
            numerator += float((raw_loss.detach() * mixed[4]).sum().cpu())
            denominator += float(mixed[4].sum().cpu())
            total_rows += receipt["rows"]
            total_mixed += receipt["mixed_rows"]
            total_fallback += receipt["mask_intersection_fallback_rows"]
            total_singleton += receipt["singleton_group_noop_rows"]
            total_cross_group += receipt["cross_group_pairs"]
            minimum_mask_tokens = min(
                minimum_mask_tokens, receipt["minimum_output_mask_tokens"]
            )
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
    finite = bool(np.isfinite(losses).all() and np.isfinite(prediction).all())
    if not finite or total_cross_group != 0 or minimum_mask_tokens < 2.0:
        raise v12.ContractError("v26 training/mixup contract failed")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "mixup_alpha": alpha,
        "mixup_rows_seen": total_rows,
        "mixed_rows": total_mixed,
        "mask_intersection_fallback_rows": total_fallback,
        "singleton_group_noop_rows": total_singleton,
        "cross_group_pairs": total_cross_group,
        "minimum_output_mask_tokens": minimum_mask_tokens,
        "inference_augmentation": 0,
        "row_deletion": 0,
        "loss_finite": finite,
    }


def _mixup_contract_receipt() -> dict[str, Any]:
    tokens = torch.arange(6 * 5 * 8, dtype=torch.float32).reshape(6, 5, 8)
    mask = torch.ones(6, 5)
    mask[0, 2:] = 0.0
    mask[2, :3] = 0.0
    context = torch.arange(6 * 11, dtype=torch.float32).reshape(6, 11)
    target = torch.arange(6, dtype=torch.float32)
    weights = torch.arange(1, 7, dtype=torch.float32)
    groups = np.array([201, 201, 201, 302, 302, 403], dtype=np.int64)
    first, receipt = group_preserving_mixup(
        [tokens, mask, context, target, weights],
        groups,
        0.2,
        np.random.default_rng(26),
    )
    second, second_receipt = group_preserving_mixup(
        [tokens, mask, context, target, weights],
        groups,
        0.2,
        np.random.default_rng(26),
    )
    deterministic = all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    fallback_exact = bool(
        torch.equal(first[0][0], tokens[0])
        and torch.equal(first[2][0], context[0])
        and torch.equal(first[3][0], target[0])
        and torch.equal(first[4][0], weights[0])
    )
    if not deterministic or receipt != second_receipt or not fallback_exact:
        raise v12.ContractError("v26 deterministic/fallback MixUp contract failed")
    return {
        **receipt,
        "byte_identical_seed_replay": deterministic,
        "insufficient_intersection_exact_noop": fallback_exact,
    }


def _group_capture_receipt() -> dict[str, Any]:
    layers = np.array([2, 2, 3, 4])
    local = pd.DatetimeIndex(
        ["2024-05-01T00:00:00+09:00", "2024-06-01T00:00:00+09:00", "2024-05-02T00:00:00+09:00", "2024-05-03T00:00:00+09:00"]
    )
    _weights, _receipt = domain_balanced_weights(layers, local)
    expected = np.array([205, 206, 305, 405])
    exact = bool(np.array_equal(_ACTIVE_ENVIRONMENT_IDS, expected))
    if not exact:
        raise v12.ContractError("v26 layer-month group encoding drift")
    return {"encoded": expected.tolist(), "exact": exact}


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence = {}
    for relative in (
        config["authorization_evidence"]["v13_result"],
        config["authorization_evidence"]["v17_result"],
        config["authorization_evidence"]["v23_result"],
        config["authorization_evidence"]["v24_result"],
        config["authorization_evidence"]["v25_result"],
        "scripts/run_p2_local_prefix_masked_public_auxiliary_20260901_v17.py",
        "scripts/run_p2_public_temperature_input_gradient_regularized_deepset_20260901_v23.py",
        "scripts/run_p2_sharpness_aware_domain_balanced_deepset_20260901_v24.py",
        "scripts/run_p2_heteroscedastic_gaussian_domain_balanced_deepset_20260901_v25.py",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "repository_p2_exact_execution_hits": 0,
        "v17_masked_auxiliary_distinguished": True,
        "v23_input_gradient_distinguished": True,
        "v24_sam_distinguished": True,
        "v25_heteroscedastic_distinguished": True,
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
        "mixup_contract": _mixup_contract_receipt(),
        "layer_month_group_capture": _group_capture_receipt(),
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
        "# P2 v26 layer-month group-preserving MixUp DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{result['status']}`. pooled ΔRMSE `{item['delta_rmse']:+.9f} C`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"transport `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점.\n\n"
        f"fold ΔRMSE: Sep-Oct `{folds['2024_sep_oct']['delta_rmse']:+.9f}`, "
        f"Jul-Aug `{folds['2025_jul_aug']['delta_rmse']:+.9f}`, "
        f"Nov-Dec `{folds['2025_nov_dec']['delta_rmse']:+.9f}`.\n\n"
        "v13 model/loss/domain weights/prefix+7d purge/seeds/epochs/blend/action cap을 고정하고 "
        "같은 target-layer×calendar-month 안의 training rows에 alpha=0.2 MixUp만 적용했다. "
        "Zhang et al. (ICLR 2018)은 vicinal training 동기만 제공하며 P2 성능 근거가 아니다. "
        "cross-group mixing/alpha sweep/router/ensemble/row deletion/official/hidden/CSV/upload=0.\n",
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
        else "EXPLORATORY_NO_GO_LAYER_MONTH_GROUP_MIXUP"
    )
    result["runtime_seconds"] = time.perf_counter() - started
    result["semantic_audit"] = semantic_audit(config)
    result["mixup_contract"] = _mixup_contract_receipt()
    result["training"].update(
        {
            "objective": config["training"]["objective"],
            "mixup": config["training"]["mixup"],
        }
    )
    result["comparison_to_preserved_candidates"] = {
        "use": "ledger_only_no_posthoc_selection_router_or_ensemble",
        "v13_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v13_result"]).read_text(encoding="utf-8")
        )["candidate"]["delta_rmse"],
        "v23_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v23_result"]).read_text(encoding="utf-8")
        )["candidate"]["delta_rmse"],
        "v25_delta_rmse": json.loads(
            (ROOT / config["authorization_evidence"]["v25_result"]).read_text(encoding="utf-8")
        )["candidate"]["delta_rmse"],
    }
    result["hashes"]["v13_runner"] = v12.sha256_file(_V13_RUNNER)
    result["hashes"]["v25_result"] = v12.sha256_file(
        ROOT / config["authorization_evidence"]["v25_result"]
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
