"""Run one sealed P2 causal multiscale domain-stat DeepSets candidate."""

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
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402

from p2_restore.causal_multiscale_domain_stats import augment_tokens  # noqa: E402
from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import build_normalized_curvature_design  # noqa: E402

EXPERIMENT_ID = "p2_causal_multiscale_domain_stat_deepset_20260901_v14"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V14_CAUSAL_MULTISCALE_DOMAIN_STAT_DEEPSET_BLEND020"


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise v12.ContractError("experiment ID drift")
    if config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED":
        raise v12.ContractError("config is not preregistered")
    authorization = config["authorization_evidence"]
    audit_path = ROOT / authorization["fresh_surface_audit"]
    if v12.sha256_file(audit_path) != authorization["fresh_surface_audit_sha256"]:
        raise v12.ContractError("fresh-surface audit hash drift")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != authorization["required_status"]:
        raise v12.ContractError("fresh-surface audit status drift")
    representation = config["representation"]
    training = config["training"]
    if (
        representation["total_token_features"] != 20
        or not representation["statistics_are_fixed_not_learned"]
        or representation["target_layer_values_used_as_features"]
        or training["model_weight"] != 0.2
        or training["champion_preserving_weight"] != 0.8
        or training["row_deletion"]
        or training["outer_fold_tuning"]
        or len(training["seeds"]) * 3 > training["maximum_fit_count"]
    ):
        raise v12.ContractError("v14 scientific contract drift")
    return config


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for relative in config["semantic_audit"]["evidence_paths"]:
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        hashes[relative] = v12.sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "reason": config["semantic_audit"]["reason"],
        "evidence_sha256": hashes,
        "not_v13_posthoc_router": True,
        "not_structured_tcn": True,
        "not_dtw_or_analog": True,
        "not_latent_state_model": True,
    }


def permutation_invariance_receipt(token_features: int, context_features: int) -> dict[str, Any]:
    torch.manual_seed(7)
    model = v12.VerticalDeepSet(token_features, context_features, hidden=32).eval()
    tokens = torch.randn(4, 5, token_features)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    context = torch.randn(4, context_features)
    order = torch.tensor([4, 2, 0, 3, 1])
    with torch.inference_mode():
        left = model(tokens, mask, context)
        right = model(tokens[:, order], mask[:, order], context)
    error = float(torch.max(torch.abs(left - right)))
    if error > 1e-6:
        raise v12.ContractError("v14 token encoder is not permutation invariant")
    return {"public_layer_permutation": order.tolist(), "maximum_abs_error": error}


def synthetic_representation_receipt(config: dict[str, Any]) -> dict[str, Any]:
    rows = 240
    timeline = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    payload: dict[str, Any] = {"time": timeline}
    phase = np.linspace(0.0, 8.0 * np.pi, rows)
    for layer in (1, 5, 6, 7, 8):
        payload[f"temp_{layer}"] = 10.0 + layer / 10.0 + np.sin(phase)
        payload[f"psal_{layer}"] = 34.0 + layer / 100.0 + 0.05 * np.cos(phase)
    base = np.zeros((rows, 5, 8), dtype=np.float32)
    representation = config["representation"]
    values, receipt = augment_tokens(
        pd.DataFrame(payload),
        base,
        representation["half_life_steps_10min"],
        int(representation["minimum_periods"]),
        float(representation["temp_scale_floor"]),
        float(representation["psal_scale_floor"]),
    )
    supports = list(receipt["support_share"].values())
    return {
        "finite": bool(np.isfinite(values).all()),
        "shape": list(values.shape),
        "support_share_min": float(min(supports)),
        "support_share_max": float(max(supports)),
        "feature_receipt_sha256": v12.sha256_json(receipt),
    }


def preflight() -> dict[str, Any]:
    config = load_config()
    audit = semantic_audit(config)
    receipt = permutation_invariance_receipt(
        int(config["training"]["token_features"]),
        int(config["training"]["context_features"]),
    )
    representation = synthetic_representation_receipt(config)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "semantic_audit": audit,
        "semantic_audit_sha256": v12.sha256_json(audit),
        "synthetic_representation": representation,
        "permutation_invariance": receipt,
        "fixed_half_life_steps": config["representation"]["half_life_steps_10min"],
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
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
    model = v12.VerticalDeepSet(
        int(config["training"]["token_features"]),
        int(config["training"]["context_features"]),
        hidden=32,
    ).to(device)
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
            raw_loss = F.smooth_l1_loss(prediction, batch[3], beta=1.0, reduction="none")
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
                ).cpu().numpy()
            )
    return np.concatenate(output).astype(float), {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "parameters": int(sum(value.numel() for value in model.parameters())),
    }


def write_report(result: dict[str, Any]) -> None:
    item = result["candidate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 causal multiscale domain-stat DeepSets v14\n\n"
        "## 결론\n\n"
        f"상태 `{result['status']}`; pooled ΔRMSE `{item['delta_rmse']:+.9f}℃`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"fixed-penalty `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점. "
        "모든 값은 노출된 historical surface의 탐색 지표이며 official 성능 주장이 아니다.\n\n"
        "Wild-Time은 시간창을 domain으로 평가할 필요를 보이고, EvoS는 과거 domain 통계의 "
        "다중시간창 진화를 동기로 제공한다. v14는 이들의 학습법을 복제하지 않고 public-layer-only "
        "shifted EWM 통계를 1/7/30일 고정 반감기로 사용한다. v13 결과 기반 router나 gate 완화는 없다.\n\n"
        "official/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )
    (REPORT / "claim-source-ledger.md").write_text(
        "# Claim-source ledger\n\n"
        "| Claim | Source | Use |\n|---|---|---|\n"
        "| Time windows can be evaluated as domains under temporal shift | Wild-Time, NeurIPS 2022, https://proceedings.neurips.cc/paper_files/paper/2022/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html | representation motivation only |\n"
        "| Multi-window past domain statistics can model temporal evolution | EvoS, NeurIPS 2023, https://proceedings.neurips.cc/paper_files/paper/2023/hash/459a911eb49cd2e0192055ee156d04e5-Abstract-Conference.html | representation motivation only; learned method not copied |\n"
        "| v14 metric and access claims | `result.json` and `independent-qa.json` | local exposed historical evidence only |\n",
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
            "result_adaptive_tuning": False,
        },
    )
    observations_path = v12.resolve_observations(config)
    scoring_path = ROOT / config["source_contract"]["scoring_frame"]
    if v12.sha256_file(scoring_path) != config["source_contract"]["scoring_frame_sha256"]:
        raise v12.ContractError("scoring frame hash drift")
    observations = pd.read_csv(observations_path, dtype={"station": "string", "time": "string"})
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    scored = pd.read_parquet(scoring_path)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    blind, reference = v12.metric_engine.make_reference(observations, scored)
    table = build_training_features(observations)
    design = build_normalized_curvature_design(table.frame)
    base_tokens, token_mask, context = v12.build_arrays(table.frame)
    representation = config["representation"]
    tokens, representation_receipt = augment_tokens(
        table.frame,
        base_tokens,
        representation["half_life_steps_10min"],
        int(representation["minimum_periods"]),
        float(representation["temp_scale_floor"]),
        float(representation["psal_scale_floor"]),
    )
    if tokens.shape[2] != int(config["training"]["token_features"]):
        raise v12.ContractError("augmented token width drift")
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
            design.keys.loc[train_mask, "layer"].to_numpy(int), local[train_mask]
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
        raise v12.ContractError("v14 fit/prediction contract failed")
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
        objective="domain_balanced_SmoothL1_beta_1.0",
        conditional=False,
    )
    record = v12.metric_engine.evaluate_candidate(spec, blind, truth, reference, candidate, config)
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
        else "EXPLORATORY_NO_GO_CAUSAL_MULTISCALE_DOMAIN_STATS"
    )
    result = {
        "schema_version": "p2.causal_multiscale_domain_stat_deepset.result.20260901.v14",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": fit_count,
        "semantic_audit": semantic,
        "representation": representation_receipt,
        "permutation_invariance": permutation_invariance_receipt(
            int(config["training"]["token_features"]),
            int(config["training"]["context_features"]),
        ),
        "training": {"row_deletion": 0, "folds": fold_receipts},
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
            "feature_module": v12.sha256_file(
                ROOT / "src/p2_restore/causal_multiscale_domain_stats.py"
            ),
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
