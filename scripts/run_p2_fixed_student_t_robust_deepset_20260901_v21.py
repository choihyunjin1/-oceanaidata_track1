"""Run one sealed fixed-Student-t robust DeepSets objective for P2."""

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

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_layer_conditioned_month_covariance_alignment_20260901_v20 as v20  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402
import run_p2_regularized_layer_month_group_dro_20260901_v18 as v18  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)

EXPERIMENT_ID = "p2_fixed_student_t_robust_deepset_20260901_v21"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V21_FIXED_STUDENT_T4_ROBUST_DEEPSET_BLEND020"


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
    ):
        raise v12.ContractError("v21 config identity/status drift")
    auth = config["authorization_evidence"]
    result_path = ROOT / auth["predecessor"]
    qa_path = ROOT / auth["independent_qa"]
    if (
        v12.sha256_file(result_path) != auth["predecessor_sha256"]
        or v12.sha256_file(qa_path) != auth["independent_qa_sha256"]
    ):
        raise v12.ContractError("v20 authorization hash drift")
    if (
        json.loads(result_path.read_text(encoding="utf-8"))["status"]
        != auth["required_status"]
        or json.loads(qa_path.read_text(encoding="utf-8"))["status"] != "PASS"
    ):
        raise v12.ContractError("v20 authorization state drift")
    training = config["training"]
    expected = {
        "architecture": "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2",
        "environment_definition": "target_layer_x_calendar_month",
        "within_environment_weighting": "equal_KST_day_then_equal_row",
        "objective": "fixed_homoscedastic_student_t_location_negative_log_likelihood",
        "student_t_degrees_of_freedom": 4.0,
        "student_t_normalized_scale": 1.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 60,
        "batch_size": 4096,
        "model_weight": 0.2,
        "champion_preserving_weight": 0.8,
    }
    if any(training[key] != value for key, value in expected.items()):
        raise v12.ContractError("v21 scientific contract drift")
    if (
        training["row_deletion"]
        or training["early_stopping"]
        or training["outer_fold_tuning"]
        or len(training["seeds"]) * 3 != 9
        or training["maximum_fit_count"] != 9
    ):
        raise v12.ContractError("v21 fit/no-deletion contract drift")
    return config


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    evidence_hashes: dict[str, str] = {}
    for relative in config["semantic_audit"]["evidence_paths"]:
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        evidence_hashes[relative] = v12.sha256_file(path)
    prior_runners: list[str] = []
    for path in (ROOT / "scripts").glob("run_p2*.py"):
        if path.resolve() == RUNNER.resolve():
            continue
        source = path.read_text(encoding="utf-8").lower()
        if "student_t_location" in source or "student-t likelihood" in source:
            prior_runners.append(path.name)
    prior_artifacts = sorted(
        path.name
        for path in (ROOT / "artifacts").glob("p2*student_t*")
        if path.name != EXPERIMENT_ID
    )
    prior_reports = sorted(
        path.name
        for path in (ROOT / "reports").glob("p2*student_t*")
        if path.name != EXPERIMENT_ID
    )
    if prior_runners or prior_artifacts or prior_reports:
        raise v12.ContractError("prior P2 Student-t likelihood execution exists")
    v10_source = (
        ROOT / "configs/experiments/p2_public_sensor_influence_shrink_20260901_v10.json"
    ).read_text(encoding="utf-8").lower()
    if "model_fit_count\": 0" not in v10_source or "first_difference" not in v10_source:
        raise v12.ContractError("v10 semantic distinction drift")
    return {
        "classification": config["semantic_audit"]["classification"],
        "reason": config["semantic_audit"]["reason"],
        "evidence_sha256": evidence_hashes,
        "prior_p2_student_t_runners": sorted(prior_runners),
        "prior_p2_student_t_artifacts": prior_artifacts,
        "prior_p2_student_t_reports": prior_reports,
        "v10_zero_fit_public_difference_action_shrink_distinguished": True,
        "v13_fixed_beta_huber_distinguished": True,
        "v18_v19_environment_risk_objectives_distinguished": True,
        "v20_latent_covariance_alignment_distinguished": True,
        "row_deletion": False,
        "degrees_of_freedom_or_scale_tuned": False,
        "result_adaptive_hyperparameter_selection": False,
    }


def student_t_location_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    degrees_of_freedom: float,
    scale: float,
) -> torch.Tensor:
    residual = (prediction - target) / float(scale)
    return 0.5 * (float(degrees_of_freedom) + 1.0) * torch.log1p(
        residual.square() / float(degrees_of_freedom)
    )


def student_t_absolute_influence(
    residual: torch.Tensor,
    degrees_of_freedom: float,
    scale: float,
) -> torch.Tensor:
    numerator = (float(degrees_of_freedom) + 1.0) * residual.abs()
    denominator = float(degrees_of_freedom) * float(scale) ** 2 + residual.square()
    return numerator / denominator


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
    degrees_of_freedom = float(config["training"]["student_t_degrees_of_freedom"])
    scale = float(config["training"]["student_t_normalized_scale"])
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
            raw_loss = student_t_location_loss(
                prediction,
                batch[3],
                degrees_of_freedom,
                scale,
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
    return np.concatenate(output).astype(float), {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "student_t_degrees_of_freedom": degrees_of_freedom,
        "student_t_normalized_scale": scale,
        "weight_decay": float(config["training"]["weight_decay"]),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_finite": bool(np.isfinite(losses).all()),
        "row_deletion": 0,
        "parameters": int(sum(value.numel() for value in model.parameters())),
    }


def synthetic_student_t_receipt(config: dict[str, Any]) -> dict[str, Any]:
    nu = float(config["training"]["student_t_degrees_of_freedom"])
    scale = float(config["training"]["student_t_normalized_scale"])
    residual = torch.tensor([0.0, 1.0, 2.0, 10.0], dtype=torch.float64)
    loss = student_t_location_loss(residual, torch.zeros_like(residual), nu, scale)
    influence = student_t_absolute_influence(residual, nu, scale)
    if not (
        loss[0] == 0.0
        and torch.all(loss[1:] > loss[:-1])
        and influence[-1] < influence[1]
    ):
        raise v12.ContractError("v21 synthetic Student-t influence contract failed")
    return {
        "residuals": residual.tolist(),
        "losses": loss.tolist(),
        "absolute_influence": influence.tolist(),
        "extreme_influence_below_unit_residual_influence": True,
        "row_deletion": 0,
    }


def preflight() -> dict[str, Any]:
    config = load_config()
    semantic = semantic_audit(config)
    synthetic = synthetic_student_t_receipt(config)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "semantic_audit": semantic,
        "semantic_audit_sha256": v12.sha256_json(semantic),
        "permutation_invariance": v12.permutation_invariance_receipt(),
        "synthetic_student_t": synthetic,
        "fixed_degrees_of_freedom": 4.0,
        "fixed_normalized_scale": 1.0,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "v13_runner_sha256": v12.sha256_file(v13.RUNNER),
        "v18_runner_sha256": v12.sha256_file(v18.RUNNER),
        "v20_runner_sha256": v12.sha256_file(v20.RUNNER),
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
    comparison = result["comparison_to_preserved_candidates"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 fixed Student-t robust DeepSets v21\n\n## 결론\n\n"
        f"상태 `{result['status']}`; pooled ΔRMSE `{item['delta_rmse']:+.9f}℃`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"fixed-penalty `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점. "
        "반복 노출 historical surface의 탐색 증거이며 official 성능 주장이 아니다.\n\n"
        "v13 architecture/input/domain-balanced weights/prefix/purge/seeds/epochs/blend/action "
        "cap은 고정했다. 유일한 변화는 SmoothL1을 ν=4, normalized scale=1의 고정 "
        "Student-t location NLL로 교체한 것이다. row deletion, df/scale learning, sweep, "
        "router, ensemble은 없다.\n\n"
        f"v13 대비 ΔRMSE `{comparison['v21_minus_v13_delta_rmse']:+.9f}℃`, v18 대비 "
        f"`{comparison['v21_minus_v18_delta_rmse']:+.9f}℃`, v20 대비 "
        f"`{comparison['v21_minus_v20_delta_rmse']:+.9f}℃`다. 비교는 원장용이다.\n\n"
        "official/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )
    (REPORT / "claim-source-ledger.md").write_text(
        "# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n"
        "| Student-t noise can make neural regression less sensitive to outliers | Briegel and Tresp, NeurIPS 1999, https://papers.neurips.cc/paper_files/paper/1999/hash/db29450c3f5e97f97846693611f98c15-Abstract.html | robust-loss motivation only; no P2 performance transfer |\n"
        "| v21 metrics and access | `result.json`, `independent-qa.json` | local exposed historical evidence only |\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
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
            "student_t_degrees_of_freedom": 4.0,
            "student_t_normalized_scale": 1.0,
            "row_deletion": 0,
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
        [v12.metric_engine.canonical_time_ns(design.keys["time"]), design.keys["layer"]]
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
        cutoff = fold_start - pd.Timedelta(days=int(config["training"]["embargo_days"]))
        train_mask = (local >= start) & (local < cutoff)
        weights, weight_receipt = v13.domain_balanced_weights(
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
                weights,
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
            "training_calendar_months": sorted(set(local[train_mask].dt.month.astype(int))),
            "weight_receipt": weight_receipt,
            "fit_receipts": fit_receipts,
        }
    if fit_count != 9 or not np.isfinite(model_prediction).all():
        raise v12.ContractError("v21 fit/prediction contract failed")
    absolute_model = design.baseline[positions] + model_prediction * design.profile_scale[positions]
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
        objective="fixed_Student_t4_location_NLL_scale1_domain_balanced",
        conditional=False,
    )
    record = v12.metric_engine.evaluate_candidate(
        spec, blind, truth, reference, candidate, config
    )
    record["by_month"] = v12.by_month_metrics(blind, truth, reference, candidate)
    record["action_geometry"] = v12.action_geometry(truth, reference, candidate)
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
        else "EXPLORATORY_NO_GO_FIXED_STUDENT_T_ROBUST_OBJECTIVE"
    )
    preserved_paths = {
        "v13": ROOT / "reports/p2_prefix_safe_domain_balanced_deepset_20260901_v13/result.json",
        "v18": ROOT / "reports/p2_regularized_layer_month_group_dro_20260901_v18/result.json",
        "v20": ROOT / "reports/p2_layer_conditioned_month_covariance_alignment_20260901_v20/result.json",
    }
    preserved = {
        name: json.loads(path.read_text(encoding="utf-8"))["candidate"]
        for name, path in preserved_paths.items()
    }
    comparison: dict[str, Any] = {"use": "ledger_only_no_posthoc_selection_or_ensemble"}
    for name, item in preserved.items():
        comparison[f"{name}_delta_rmse"] = float(item["delta_rmse"])
        comparison[f"{name}_canonical_nominal_points"] = float(
            item["canonical_nominal_pooled_points_delta"]
        )
        comparison[f"v21_minus_{name}_delta_rmse"] = float(record["delta_rmse"]) - float(
            item["delta_rmse"]
        )
        comparison[f"v21_minus_{name}_canonical_nominal_points"] = nominal - float(
            item["canonical_nominal_pooled_points_delta"]
        )
    result = {
        "schema_version": "p2.fixed_student_t_robust_deepset.result.20260901.v21",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": fit_count,
        "semantic_audit": semantic,
        "permutation_invariance": v12.permutation_invariance_receipt(),
        "training": {
            "row_deletion": 0,
            "environment_definition": config["training"]["environment_definition"],
            "objective": config["training"]["objective"],
            "student_t_degrees_of_freedom": 4.0,
            "student_t_normalized_scale": 1.0,
            "weight_decay": float(config["training"]["weight_decay"]),
            "folds": fold_receipts,
        },
        "candidate": record,
        "comparison_to_preserved_candidates": comparison,
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
            "v18_runner": v12.sha256_file(v18.RUNNER),
            "v20_runner": v12.sha256_file(v20.RUNNER),
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
