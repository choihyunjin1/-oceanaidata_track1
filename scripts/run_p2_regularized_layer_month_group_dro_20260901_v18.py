"""Run one sealed regularized layer-month Group-DRO candidate for P2."""

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

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)

EXPERIMENT_ID = "p2_regularized_layer_month_group_dro_20260901_v18"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V18_REGULARIZED_LAYER_MONTH_GROUP_DRO_BLEND020"


def load_config() -> dict[str, Any]:
    """Load and fail closed on every scientific field of the v18 contract."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
    ):
        raise v12.ContractError("v18 config identity/status drift")
    auth = config["authorization_evidence"]
    result_path = ROOT / auth["closed_predecessor"]
    qa_path = ROOT / auth["independent_qa"]
    if (
        v12.sha256_file(result_path) != auth["closed_predecessor_sha256"]
        or v12.sha256_file(qa_path) != auth["independent_qa_sha256"]
    ):
        raise v12.ContractError("v17 authorization hash drift")
    if (
        json.loads(result_path.read_text(encoding="utf-8"))["status"]
        != auth["required_status"]
        or json.loads(qa_path.read_text(encoding="utf-8"))["status"] != "PASS"
    ):
        raise v12.ContractError("v17 authorization state drift")
    training = config["training"]
    expected = {
        "architecture": "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2",
        "group_definition": "target_layer_x_calendar_month",
        "within_group_weighting": "equal_KST_day_then_equal_row",
        "group_dro_update": "epoch_end_exponentiated_gradient",
        "group_dro_eta": 0.1,
        "weight_decay": 0.001,
        "epochs": 60,
        "model_weight": 0.2,
        "champion_preserving_weight": 0.8,
    }
    if any(training[key] != value for key, value in expected.items()):
        raise v12.ContractError("v18 scientific contract drift")
    if (
        training["row_deletion"]
        or training["early_stopping"]
        or training["outer_fold_tuning"]
        or len(training["seeds"]) * 3 != training["maximum_fit_count"]
        or training["maximum_fit_count"] != 9
    ):
        raise v12.ContractError("v18 fit/no-deletion contract drift")
    return config


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Prove that no prior P2 exponentiated Group-DRO execution exists."""
    evidence_hashes: dict[str, str] = {}
    for relative in config["semantic_audit"]["evidence_paths"]:
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence_hashes[relative] = v12.sha256_file(path)
    prior_runners = sorted(
        path.name
        for path in (ROOT / "scripts").glob("run_p2*group*dro*.py")
        if path.resolve() != RUNNER.resolve()
    )
    prior_artifacts = sorted(
        path.name
        for path in (ROOT / "artifacts").glob("p2*group*dro*")
        if path.name != EXPERIMENT_ID
    )
    prior_reports = sorted(
        path.name
        for path in (ROOT / "reports").glob("p2*group*dro*")
        if path.name != EXPERIMENT_ID
    )
    if prior_runners or prior_artifacts or prior_reports:
        raise v12.ContractError("prior P2 Group-DRO execution namespace exists")
    v13_source = (ROOT / "scripts/run_p2_prefix_safe_domain_balanced_deepset_20260901_v13.py").read_text(
        encoding="utf-8"
    )
    if "exponentiated" in v13_source.lower() or "group_dro" in v13_source.lower():
        raise v12.ContractError("v13 unexpectedly contains adaptive Group-DRO")
    return {
        "classification": config["semantic_audit"]["classification"],
        "reason": config["semantic_audit"]["reason"],
        "evidence_sha256": evidence_hashes,
        "prior_p2_group_dro_runners": prior_runners,
        "prior_p2_group_dro_artifacts": prior_artifacts,
        "prior_p2_group_dro_reports": prior_reports,
        "v13_static_domain_balance_only": True,
        "p1_group_dro_is_cross_problem_only": True,
        "result_adaptive_group_or_hyperparameter_selection": False,
    }


def build_group_contract(
    layer: np.ndarray,
    local_time: pd.Series | pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Build fixed layer-month ids and equal-day weights inside each group."""
    local = pd.DatetimeIndex(local_time)
    frame = pd.DataFrame(
        {
            "layer": np.asarray(layer, dtype=int),
            "calendar_month": local.month.astype(int),
            "kst_date": local.date,
        }
    )
    pairs = sorted(
        frame[["layer", "calendar_month"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    pair_to_index = {pair: index for index, pair in enumerate(pairs)}
    group_index = np.fromiter(
        (
            pair_to_index[(int(target_layer), int(month))]
            for target_layer, month in frame[["layer", "calendar_month"]].itertuples(
                index=False, name=None
            )
        ),
        dtype=np.int64,
        count=len(frame),
    )
    raw = np.zeros(len(frame), dtype=float)
    receipt: dict[str, Any] = {}
    labels: list[str] = []
    for group_id, (target_layer, month) in enumerate(pairs):
        label = f"layer{target_layer}:month{month:02d}"
        labels.append(label)
        selected_group = group_index == group_id
        days = sorted(frame.loc[selected_group, "kst_date"].unique())
        for day in days:
            selected_day = selected_group & frame["kst_date"].eq(day).to_numpy()
            raw[selected_day] = 1.0 / (
                len(pairs) * len(days) * int(selected_day.sum())
            )
        receipt[label] = {
            "group_id": group_id,
            "rows": int(selected_group.sum()),
            "days": len(days),
            "raw_weight_sum": float(raw[selected_group].sum()),
        }
    if not (np.isfinite(raw).all() and np.all(raw > 0)):
        raise v12.ContractError("v18 within-group base weights are invalid")
    weights = raw / raw.mean()
    return group_index, weights.astype(np.float32), labels, {
        "groups": receipt,
        "group_count": len(labels),
        "normalized_mean": float(weights.mean()),
        "normalized_min": float(weights.min()),
        "normalized_max": float(weights.max()),
    }


def exponentiated_group_update(
    probabilities: torch.Tensor,
    losses: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Apply one stable, fixed exponentiated-gradient adversary update."""
    if probabilities.ndim != 1 or losses.shape != probabilities.shape:
        raise ValueError("group probability/loss shape mismatch")
    logits = torch.log(probabilities.clamp_min(1e-30)) + float(eta) * losses
    return torch.softmax(logits, dim=0)


def train_predict_seed(
    tokens: np.ndarray,
    mask: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    base_weights: np.ndarray,
    group_index: np.ndarray,
    group_labels: list[str],
    query_tokens: np.ndarray,
    query_mask: np.ndarray,
    query_context: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fixed DeepSets model with the preregistered Group-DRO objective."""
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
            base_weights.astype(np.float32),
            group_index.astype(np.int64),
        )
    )
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["training"]["epochs"])
    group_count = len(group_labels)
    eta = float(config["training"]["group_dro_eta"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probabilities = torch.full(
        (group_count,), 1.0 / group_count, dtype=torch.float32, device=device
    )
    initial_probabilities = probabilities.detach().cpu().numpy().astype(float)
    robust_losses: list[float] = []
    worst_group_losses: list[float] = []
    last_group_losses = torch.zeros(group_count, device=device)
    model.train()
    for _epoch in range(epochs):
        order = torch.randperm(len(target), generator=generator)
        robust_numerator = 0.0
        robust_denominator = 0.0
        group_numerator = torch.zeros(group_count, device=device)
        group_denominator = torch.zeros(group_count, device=device)
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            raw_loss = F.smooth_l1_loss(
                prediction,
                batch[3],
                beta=1.0,
                reduction="none",
            )
            adversarial = probabilities[batch[5]] * group_count
            effective_weight = batch[4] * adversarial
            loss = (raw_loss * effective_weight).sum() / effective_weight.sum().clamp_min(
                1e-12
            )
            loss.backward()
            optimizer.step()
            robust_numerator += float((raw_loss.detach() * effective_weight).sum().cpu())
            robust_denominator += float(effective_weight.sum().cpu())
            group_numerator.scatter_add_(0, batch[5], raw_loss.detach() * batch[4])
            group_denominator.scatter_add_(0, batch[5], batch[4])
        last_group_losses = group_numerator / group_denominator.clamp_min(1e-12)
        probabilities = exponentiated_group_update(
            probabilities,
            last_group_losses,
            eta,
        ).detach()
        robust_losses.append(robust_numerator / robust_denominator)
        worst_group_losses.append(float(last_group_losses.max().detach().cpu()))
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
    final_probabilities = probabilities.cpu().numpy().astype(float)
    final_losses = last_group_losses.detach().cpu().numpy().astype(float)
    return np.concatenate(output).astype(float), {
        "seed": seed,
        "device": str(device),
        "epochs": len(robust_losses),
        "group_count": group_count,
        "group_dro_eta": eta,
        "weight_decay": float(config["training"]["weight_decay"]),
        "robust_loss_first": robust_losses[0],
        "robust_loss_last": robust_losses[-1],
        "worst_group_loss_first": worst_group_losses[0],
        "worst_group_loss_last": worst_group_losses[-1],
        "initial_group_probabilities": {
            label: float(initial_probabilities[index])
            for index, label in enumerate(group_labels)
        },
        "final_group_probabilities": {
            label: float(final_probabilities[index])
            for index, label in enumerate(group_labels)
        },
        "final_group_losses": {
            label: float(final_losses[index])
            for index, label in enumerate(group_labels)
        },
        "final_probability_sum": float(final_probabilities.sum()),
        "final_probability_min": float(final_probabilities.min()),
        "final_probability_max": float(final_probabilities.max()),
        "final_probability_entropy": float(
            -(final_probabilities * np.log(final_probabilities)).sum()
        ),
        "probability_l1_change": float(
            np.abs(final_probabilities - initial_probabilities).sum()
        ),
        "parameters": int(sum(value.numel() for value in model.parameters())),
    }


def preflight() -> dict[str, Any]:
    """Return a deterministic zero-operation science/novelty receipt."""
    config = load_config()
    semantic = semantic_audit(config)
    permutation = v12.permutation_invariance_receipt()
    synthetic_probabilities = torch.tensor([0.5, 0.5])
    synthetic_losses = torch.tensor([0.1, 0.4])
    updated = exponentiated_group_update(
        synthetic_probabilities,
        synthetic_losses,
        float(config["training"]["group_dro_eta"]),
    )
    if not updated[1] > updated[0]:
        raise v12.ContractError("v18 adversarial update contract failed")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "semantic_audit": semantic,
        "semantic_audit_sha256": v12.sha256_json(semantic),
        "permutation_invariance": permutation,
        "worst_synthetic_group_upweighted": True,
        "fixed_group_dro_eta": float(config["training"]["group_dro_eta"]),
        "fixed_weight_decay": float(config["training"]["weight_decay"]),
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "v13_runner_sha256": v12.sha256_file(v13.RUNNER),
        "v12_runner_sha256": v12.sha256_file(v12.RUNNER),
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
    """Write a concise local-only terminal report and source ledger."""
    item = result["candidate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 regularized layer-month Group-DRO v18\n\n## 결론\n\n"
        f"상태 `{result['status']}`; pooled ΔRMSE `{item['delta_rmse']:+.9f}℃`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"fixed-penalty `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점. "
        "모든 수치는 반복 노출된 historical surface의 탐색 증거이며 official 성능 주장이 아니다.\n\n"
        "v13 representation/architecture/prefix/purge/seeds/epochs/blend/action cap은 고정했다. "
        "유일한 과학 변화는 layer×calendar-month별 epoch-end exponentiated Group-DRO "
        "adversary와 고정 0.001 weight decay다. group/eta/regularization/blend sweep은 없다.\n\n"
        "official/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )
    (REPORT / "claim-source-ledger.md").write_text(
        "# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n"
        "| Group DRO minimizes worst pre-defined group risk and regularization matters in overparameterized models | Sagawa et al., ICLR 2020, https://openreview.net/forum?id=ryxGuJrFvS | objective motivation only; no P2 performance transfer |\n"
        "| v18 metrics and access | `result.json`, `independent-qa.json` | local exposed historical evidence only |\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    """Execute the sealed nine-fit v18 evaluation exactly once."""
    started = time.perf_counter()
    if ARTIFACT.exists():
        raise FileExistsError(ARTIFACT)
    config = load_config()
    semantic = semantic_audit(config)
    ARTIFACT.mkdir(parents=True)
    v12.atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": v12.sha256_file(CONFIG),
            "runner_sha256": v12.sha256_file(RUNNER),
            "candidate": PREDICTION_NAME,
            "fit_plan": "3_folds_x_3_seeds",
            "group_definition": config["training"]["group_definition"],
            "group_dro_eta": config["training"]["group_dro_eta"],
            "weight_decay": config["training"]["weight_decay"],
            "result_adaptive_tuning": False,
        },
    )
    observations_path = v12.resolve_observations(config)
    scoring_path = ROOT / config["source_contract"]["scoring_frame"]
    if v12.sha256_file(scoring_path) != config["source_contract"]["scoring_frame_sha256"]:
        raise v12.ContractError("scoring frame hash drift")
    observations = pd.read_csv(
        observations_path,
        dtype={"station": "string", "time": "string"},
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    scored = pd.read_parquet(scoring_path)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    blind, reference = v12.metric_engine.make_reference(observations, scored)
    table = build_training_features(observations)
    design = build_normalized_curvature_design(table.frame)
    tokens, token_mask, context = v12.build_arrays(table.frame)
    local = design.keys["time"].dt.tz_convert("Asia/Seoul")
    design_index = pd.MultiIndex.from_arrays(
        [
            v12.metric_engine.canonical_time_ns(design.keys["time"]),
            design.keys["layer"],
        ]
    )
    query_index = pd.MultiIndex.from_arrays(
        [v12.metric_engine.canonical_time_ns(blind["time"]), blind["layer"]]
    )
    positions = design_index.get_indexer(query_index)
    if design_index.has_duplicates or np.any(positions < 0):
        raise v12.ContractError("historical query alignment failed")
    model_prediction = np.full(len(blind), np.nan, dtype=float)
    fold_receipts: dict[str, Any] = {}
    fit_count = 0
    start = pd.Timestamp(config["training"]["training_start_inclusive_kst"])
    for fold in v12.metric_engine.FOLD_ORDER:
        fold_start = pd.Timestamp(config["training"]["fold_starts_kst"][fold])
        cutoff = fold_start - pd.Timedelta(
            days=int(config["training"]["embargo_days"])
        )
        train_mask = (local >= start) & (local < cutoff)
        group_index, base_weights, group_labels, group_receipt = build_group_contract(
            design.keys.loc[train_mask, "layer"].to_numpy(int),
            local[train_mask],
        )
        query_mask = blind["fold"].eq(fold).to_numpy()
        predictions: list[np.ndarray] = []
        fit_receipts: list[dict[str, Any]] = []
        for seed in config["training"]["seeds"]:
            prediction, receipt = train_predict_seed(
                tokens[train_mask],
                token_mask[train_mask],
                context[train_mask],
                design.normalized_target[train_mask],
                base_weights,
                group_index,
                group_labels,
                tokens[positions[query_mask]],
                token_mask[positions[query_mask]],
                context[positions[query_mask]],
                config,
                int(seed),
            )
            predictions.append(prediction)
            fit_receipts.append(receipt)
        model_prediction[query_mask] = np.mean(np.vstack(predictions), axis=0)
        fit_count += len(fit_receipts)
        fold_receipts[fold] = {
            "training_start_kst": start.isoformat(),
            "training_cutoff_exclusive_kst": cutoff.isoformat(),
            "training_rows": int(train_mask.sum()),
            "training_calendar_months": sorted(
                set(local[train_mask].dt.month.astype(int))
            ),
            "group_receipt": group_receipt,
            "fit_receipts": fit_receipts,
        }
    if fit_count != 9 or not np.isfinite(model_prediction).all():
        raise v12.ContractError("v18 fit/prediction contract failed")
    absolute_model = (
        design.baseline[positions] + model_prediction * design.profile_scale[positions]
    )
    clipped = np.clip(
        absolute_model - reference,
        -float(config["training"]["model_minus_champion_clip_C"]),
        float(config["training"]["model_minus_champion_clip_C"]),
    )
    candidate = reference + float(config["training"]["model_weight"]) * clipped
    prediction_path = ARTIFACT / f"{PREDICTION_NAME}.npz"
    np.savez_compressed(
        prediction_path,
        time_ns=v12.metric_engine.canonical_time_ns(blind["time"]),
        layer=blind["layer"].to_numpy(np.int16),
        fold=blind["fold"].to_numpy(str),
        reference=reference,
        candidate=candidate,
    )
    commitment = {
        "path": str(prediction_path),
        "sha256": v12.sha256_file(prediction_path),
        "rows": len(candidate),
        "metric_computed_at_commitment": False,
    }
    v12.atomic_json(ARTIFACT / "prediction_commitment.json", commitment)
    truth = design.truth[positions]
    spec = v12.metric_engine.CandidateSpec(
        name=PREDICTION_NAME,
        objective="regularized_layer_month_exponentiated_group_DRO_SmoothL1",
        conditional=False,
    )
    record = v12.metric_engine.evaluate_candidate(
        spec,
        blind,
        truth,
        reference,
        candidate,
        config,
    )
    record["by_month"] = v12.by_month_metrics(blind, truth, reference, candidate)
    record["action_geometry"] = v12.action_geometry(
        truth,
        reference,
        candidate,
    )
    slope = float(config["evaluation"]["points_per_rmse_C"])
    penalty = float(config["evaluation"]["transport_penalty_points"])
    nominal = -float(record["delta_rmse"]) * slope
    transport = nominal - penalty
    record["canonical_nominal_pooled_points_delta"] = nominal
    record["canonical_transport_adjusted_pooled_points_delta"] = transport
    record["legacy_engine_score_translation"] = {
        "raw_expected_points_delta": record["raw_expected_points_delta"],
        "transport_calibrated_expected_points_delta": record[
            "transport_calibrated_expected_points_delta"
        ],
        "meaning": "official-like Sep-Oct day-bootstrap CI90-high translation",
    }
    record["prediction_commitment"] = commitment
    safety = v13.safety_gate(record, transport)
    record["safety_gate_checks"] = safety
    record["safety_pass"] = bool(all(safety.values()))
    status = (
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if record["strict_exploratory_pass"] and record["safety_pass"]
        else "EXPLORATORY_NO_GO_REGULARIZED_LAYER_MONTH_GROUP_DRO"
    )
    result = {
        "schema_version": "p2.regularized_layer_month_group_dro.result.20260901.v18",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": fit_count,
        "semantic_audit": semantic,
        "permutation_invariance": v12.permutation_invariance_receipt(),
        "training": {
            "row_deletion": 0,
            "group_definition": config["training"]["group_definition"],
            "group_dro_eta": float(config["training"]["group_dro_eta"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "folds": fold_receipts,
        },
        "candidate": record,
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "historical_scoring_rows_read": int(len(scored)),
            "official_test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_file_rows_read": 0,
            "score_file_rows_read": 0,
            "query_support_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config": v12.sha256_file(CONFIG),
            "runner": v12.sha256_file(RUNNER),
            "v13_runner": v12.sha256_file(v13.RUNNER),
            "v12_runner": v12.sha256_file(v12.RUNNER),
            "observations": v12.sha256_file(observations_path),
            "scoring_frame": v12.sha256_file(scoring_path),
            "prediction_npz": commitment["sha256"],
        },
    }
    v12.atomic_json(ARTIFACT / "result.json", result)
    REPORT.mkdir(parents=True, exist_ok=True)
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
    payload = preflight() if args.preflight else run()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
