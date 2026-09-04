"""Run one sealed exploratory P2 compact same-time Set Transformer."""

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

from p2_restore.compact_set_transformer import (  # noqa: E402
    CompactSetTransformer,
    architecture_receipt,
)
from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import build_normalized_curvature_design  # noqa: E402

EXPERIMENT_ID = "p2_compact_pair_interaction_set_transformer_20260901_v15"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
MODEL_MODULE = ROOT / "src/p2_restore/compact_set_transformer.py"
PREDICTION_NAME = "P2_V15_COMPACT_SAME_TIME_SET_TRANSFORMER_BLEND020"


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise v12.ContractError("experiment ID drift")
    if config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED":
        raise v12.ContractError("config is not preregistered")
    authorization = config["authorization_evidence"]
    result_path = ROOT / authorization["closed_predecessor"]
    qa_path = ROOT / authorization["independent_qa"]
    if v12.sha256_file(result_path) != authorization["closed_predecessor_sha256"]:
        raise v12.ContractError("v14 result hash drift")
    if v12.sha256_file(qa_path) != authorization["independent_qa_sha256"]:
        raise v12.ContractError("v14 independent-QA hash drift")
    predecessor = json.loads(result_path.read_text(encoding="utf-8"))
    predecessor_qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if predecessor["status"] != authorization["required_status"]:
        raise v12.ContractError("v14 is not the sealed NO_GO predecessor")
    if predecessor_qa["status"] != "PASS":
        raise v12.ContractError("v14 independent QA is not PASS")
    representation = config["representation"]
    training = config["training"]
    if (
        representation["attention_blocks"] != 1
        or representation["attention_heads"] != 2
        or representation["positional_encoding"]
        or representation["temporal_axis"]
        or representation["target_layer_values_used_as_features"]
        or training["model_weight"] != 0.2
        or training["champion_preserving_weight"] != 0.8
        or training["row_deletion"]
        or training["outer_fold_tuning"]
        or len(training["seeds"]) * 3 > training["maximum_fit_count"]
    ):
        raise v12.ContractError("v15 scientific contract drift")
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
        "v12_v13_pointwise_before_pooling": True,
        "prior_attention_is_temporal_patch_or_channel_sequence": True,
        "same_time_public_depth_pair_interaction_previously_executed": False,
        "not_v13_posthoc_router": True,
        "not_v14_ewm_retune": True,
    }


def architecture_contract_receipt(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(7)
    representation = config["representation"]
    model = CompactSetTransformer(
        int(representation["token_features"]),
        int(representation["context_features"]),
        hidden=int(representation["hidden_width"]),
        heads=int(representation["attention_heads"]),
        blocks=int(representation["attention_blocks"]),
    ).eval()
    tokens = torch.randn(6, 5, int(representation["token_features"]))
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    context = torch.randn(6, int(representation["context_features"]))
    order = torch.tensor([4, 2, 0, 3, 1])
    with torch.inference_mode():
        encoded = model.encode(tokens, mask)
        permuted_encoded = model.encode(tokens[:, order], mask[:, order])
        prediction = model(tokens, mask, context)
        permuted_prediction = model(tokens[:, order], mask[:, order], context)
        changed_tokens = tokens.clone()
        changed_context = context.clone()
        changed_tokens[3:] += 1000.0
        changed_context[3:] -= 1000.0
        changed_prediction = model(changed_tokens, mask, changed_context)
    equivariance_error = float(
        torch.max(torch.abs(permuted_encoded - encoded[:, order]))
    )
    invariance_error = float(torch.max(torch.abs(permuted_prediction - prediction)))
    batch_isolation_error = float(torch.max(torch.abs(changed_prediction[:3] - prediction[:3])))
    if max(equivariance_error, invariance_error, batch_isolation_error) > 1e-6:
        raise v12.ContractError("v15 set/leakage architecture receipt failed")
    return {
        **architecture_receipt(model),
        "public_layer_permutation": order.tolist(),
        "encoder_equivariance_maximum_abs_error": equivariance_error,
        "prediction_invariance_maximum_abs_error": invariance_error,
        "future_batch_perturbation_maximum_abs_error_on_prior_rows": batch_isolation_error,
        "time_axis_present": False,
        "current_target_temp_or_psal_feature_present": False,
        "same_timestamp_public_layer_features_only": True,
    }


def preflight() -> dict[str, Any]:
    config = load_config()
    audit = semantic_audit(config)
    architecture = architecture_contract_receipt(config)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "semantic_audit": audit,
        "semantic_audit_sha256": v12.sha256_json(audit),
        "architecture_contract": architecture,
        "architecture_contract_sha256": v12.sha256_json(architecture),
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "model_module_sha256": v12.sha256_file(MODEL_MODULE),
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
    representation = config["representation"]
    model = CompactSetTransformer(
        int(representation["token_features"]),
        int(representation["context_features"]),
        hidden=int(representation["hidden_width"]),
        heads=int(representation["attention_heads"]),
        blocks=int(representation["attention_blocks"]),
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
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(output).astype(float), {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "attention_blocks": int(representation["attention_blocks"]),
        "attention_heads": int(representation["attention_heads"]),
    }


def write_report(result: dict[str, Any]) -> None:
    item = result["candidate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 compact same-time Set Transformer v15\n\n"
        "## 결론\n\n"
        f"상태 `{result['status']}`; pooled ΔRMSE `{item['delta_rmse']:+.9f}℃`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"fixed-penalty `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점. "
        "모든 값은 노출된 historical surface의 탐색 지표이며 official 성능 주장이 아니다.\n\n"
        "v12/v13의 독립 element map 뒤 pooling과 달리, v15는 같은 timestamp의 public-depth "
        "tokens 사이에 1 block/2 heads self-attention을 적용한다. time axis와 positional encoding은 "
        "없으며 public-layer permutation equivariance/invariance를 단위검사했다. Set Transformer "
        "논문은 구조 동기만 제공하고 P2 성능을 보증하지 않는다.\n\n"
        "official/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )
    (REPORT / "claim-source-ledger.md").write_text(
        "# Claim-source ledger\n\n"
        "| Claim | Source | Use |\n|---|---|---|\n"
        "| Attention-based set models can preserve permutation invariance | Lee et al., Set Transformer, ICML/PMLR 2019, https://proceedings.mlr.press/v97/lee19d.html | architecture motivation only |\n"
        "| v15 metric, permutation, and access claims | `result.json` and `independent-qa.json` | local exposed historical evidence only |\n",
        encoding="utf-8",
    )


def run() -> dict[str, Any]:
    started = time.perf_counter()
    if ARTIFACT.exists():
        raise FileExistsError(ARTIFACT)
    config = load_config()
    semantic = semantic_audit(config)
    architecture = architecture_contract_receipt(config)
    ARTIFACT.mkdir(parents=True)
    v12.atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": v12.sha256_file(CONFIG),
            "runner_sha256": v12.sha256_file(RUNNER),
            "model_module_sha256": v12.sha256_file(MODEL_MODULE),
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
    tokens, token_mask, context = v12.build_arrays(table.frame)
    if (
        tokens.shape[1:] != (5, int(config["training"]["token_features"]))
        or context.shape[1] != int(config["training"]["context_features"])
        or not np.isfinite(tokens).all()
        or not np.isfinite(context).all()
    ):
        raise v12.ContractError("v15 pointwise representation contract failed")
    representation_receipt = {
        "token_shape": list(tokens.shape),
        "context_shape": list(context.shape),
        "valid_token_share": float(token_mask.mean()),
        "finite": True,
        "same_timestamp_public_layer_features_only": True,
        "temporal_axis_present": False,
        "positional_encoding_present": False,
        "current_target_temp_or_psal_feature_present": False,
    }
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
        raise v12.ContractError("v15 fit/prediction contract failed")
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
        objective="domain_balanced_SmoothL1_beta_1.0_same_time_set_interaction",
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
        else "EXPLORATORY_NO_GO_COMPACT_SET_TRANSFORMER"
    )
    result = {
        "schema_version": "p2.compact_pair_interaction_set_transformer.result.20260901.v15",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": fit_count,
        "semantic_audit": semantic,
        "representation": representation_receipt,
        "architecture_contract": architecture,
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
            "model_module": v12.sha256_file(MODEL_MODULE),
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
