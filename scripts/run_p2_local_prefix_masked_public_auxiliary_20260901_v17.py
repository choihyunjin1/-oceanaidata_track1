"""Run one sealed P2 local prefix-only masked-public auxiliary candidate."""

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
from p2_restore.masked_public_auxiliary_encoder import (  # noqa: E402
    MaskedPublicAuxiliaryEncoder,
    mask_public_index,
)
from p2_restore.normalized_curvature_residual import build_normalized_curvature_design  # noqa: E402

EXPERIMENT_ID = "p2_local_prefix_masked_public_auxiliary_20260901_v17"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
MODEL_MODULE = ROOT / "src/p2_restore/masked_public_auxiliary_encoder.py"
PREDICTION_NAME = "P2_V17_LOCAL_PREFIX_MASKED_PUBLIC_AUX_BLEND020"


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED":
        raise v12.ContractError("v17 config identity/status drift")
    auth = config["authorization_evidence"]
    result_path = ROOT / auth["closed_predecessor"]
    qa_path = ROOT / auth["independent_qa"]
    if v12.sha256_file(result_path) != auth["closed_predecessor_sha256"] or v12.sha256_file(qa_path) != auth["independent_qa_sha256"]:
        raise v12.ContractError("v16 authorization hash drift")
    if json.loads(result_path.read_text(encoding="utf-8"))["status"] != auth["required_status"] or json.loads(qa_path.read_text(encoding="utf-8"))["status"] != "PASS":
        raise v12.ContractError("v16 authorization state drift")
    representation = config["representation"]
    auxiliary = config["auxiliary"]
    training = config["training"]
    if (
        representation["external_data"]
        or representation["temporal_axis"]
        or representation["target_layer_values_used_as_features"]
        or auxiliary["weight"] != 0.25
        or auxiliary["masked_slots_per_training_forward"] != 1
        or auxiliary["result_adaptive"]
        or training["champion_preserving_weight"] != 0.8
        or training["model_weight"] != 0.2
        or training["row_deletion"]
        or training["outer_fold_tuning"]
        or len(training["seeds"]) * 3 > training["maximum_fit_count"]
    ):
        raise v12.ContractError("v17 scientific contract drift")
    return config


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for relative in config["semantic_audit"]["evidence_paths"]:
        path = ROOT / relative
        if not path.is_file():
            raise v12.ContractError(f"semantic evidence missing: {relative}")
        hashes[relative] = v12.sha256_file(path)
    external_path = ROOT / "configs/experiments/p2_external_depth_query_v1.json"
    external = json.loads(external_path.read_text(encoding="utf-8"))
    external_execution = external["execution"]
    if any(bool(value) for value in external_execution.values()):
        raise v12.ContractError("external depth-query plan unexpectedly executed")
    if (ROOT / "artifacts/p2_external_depth_query_v1").exists() or (ROOT / "reports/p2_external_depth_query_v1").exists():
        raise v12.ContractError("external depth-query execution output unexpectedly exists")
    return {
        "classification": config["semantic_audit"]["classification"],
        "reason": config["semantic_audit"]["reason"],
        "evidence_sha256": hashes,
        "external_depth_query_execution": external_execution,
        "external_depth_query_artifact_exists": False,
        "external_depth_query_report_exists": False,
        "uses_external_data": False,
        "reconstructs_target_layer_values": False,
        "uses_temporal_masking": False,
        "target_masked_tcn_or_csdi_overlap": False,
        "local_prefix_public_auxiliary_execution_found": False,
    }


def architecture_contract_receipt(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(13)
    representation = config["representation"]
    model = MaskedPublicAuxiliaryEncoder(
        int(representation["token_features"]),
        len(representation["public_layers"]),
        int(representation["context_features"]),
        hidden=int(representation["hidden_width"]),
        latent=int(representation["latent_width"]),
    ).eval()
    tokens = torch.randn(8, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 0, 1], [0, 1, 1, 0, 1]], dtype=torch.float32)
    context = torch.randn(8, 11)
    masked_tokens, masked_mask, eligible = mask_public_index(tokens, mask, 2)
    with torch.inference_mode():
        prediction = model(tokens, mask, context)
        reconstruction = model.reconstruct(masked_tokens, masked_mask, context)
        changed_tokens = tokens.clone()
        changed_context = context.clone()
        changed_tokens[4:] += 10000
        changed_context[4:] -= 10000
        changed_prediction = model(changed_tokens, mask, changed_context)
    isolation_error = float(torch.max(torch.abs(changed_prediction[:4] - prediction[:4])))
    if isolation_error != 0.0 or not torch.isfinite(reconstruction).all() or not torch.any(eligible):
        raise v12.ContractError("v17 architecture/leakage receipt failed")
    return {
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "ordered_public_slots": True,
        "masked_slot": 2,
        "masked_slot_tokens_zero": bool(torch.count_nonzero(masked_tokens[:, 2]) == 0),
        "masked_slot_mask_zero": bool(torch.count_nonzero(masked_mask[:, 2]) == 0),
        "eligible_rows": int(eligible.sum()),
        "reconstruction_width": int(reconstruction.shape[1]),
        "future_batch_perturbation_maximum_abs_error_on_prior_rows": isolation_error,
        "time_axis_present": False,
        "current_target_temp_or_psal_feature_present": False,
        "external_data_used": False,
        "mask_cycle_counts_over_60_epochs": {str(index): 12 for index in range(5)},
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
        "external_rows_read": 0,
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
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representation = config["representation"]
    model = MaskedPublicAuxiliaryEncoder(8, 5, 11, hidden=int(representation["hidden_width"]), latent=int(representation["latent_width"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    train = tuple(torch.from_numpy(value) for value in (tokens, mask, context, target.astype(np.float32), weights.astype(np.float32)))
    batch_size = int(config["training"]["batch_size"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    main_losses: list[float] = []
    auxiliary_losses: list[float] = []
    cycle_counts = {index: 0 for index in range(5)}
    model.train()
    for epoch in range(int(config["training"]["epochs"])):
        masked_index = epoch % 5
        cycle_counts[masked_index] += 1
        order = torch.randperm(len(target), generator=generator)
        main_numerator = aux_numerator = 0.0
        main_denominator = aux_denominator = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            raw_main = F.smooth_l1_loss(prediction, batch[3], beta=1.0, reduction="none")
            main_loss = (raw_main * batch[4]).sum() / batch[4].sum().clamp_min(1e-12)
            masked_tokens, masked_mask, eligible = mask_public_index(batch[0], batch[1], masked_index)
            reconstruction = model.reconstruct(masked_tokens, masked_mask, batch[2])[:, masked_index]
            raw_aux = F.smooth_l1_loss(reconstruction, batch[0][:, masked_index, 0], beta=1.0, reduction="none")
            aux_weight = batch[4] * eligible.to(batch[4].dtype)
            aux_loss = (raw_aux * aux_weight).sum() / aux_weight.sum().clamp_min(1e-12)
            loss = main_loss + float(config["auxiliary"]["weight"]) * aux_loss
            loss.backward()
            optimizer.step()
            main_numerator += float((raw_main.detach() * batch[4]).sum().cpu())
            main_denominator += float(batch[4].sum().cpu())
            aux_numerator += float((raw_aux.detach() * aux_weight).sum().cpu())
            aux_denominator += float(aux_weight.sum().cpu())
        main_losses.append(main_numerator / main_denominator)
        auxiliary_losses.append(aux_numerator / max(aux_denominator, 1e-12))
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_tokens), batch_size):
            stop = start + batch_size
            output.append(model(torch.from_numpy(query_tokens[start:stop]).to(device), torch.from_numpy(query_mask[start:stop]).to(device), torch.from_numpy(query_context[start:stop]).to(device)).cpu().numpy())
    return np.concatenate(output).astype(float), {
        "seed": seed,
        "device": str(device),
        "epochs": len(main_losses),
        "main_loss_first": main_losses[0],
        "main_loss_last": main_losses[-1],
        "auxiliary_loss_first": auxiliary_losses[0],
        "auxiliary_loss_last": auxiliary_losses[-1],
        "auxiliary_weight": float(config["auxiliary"]["weight"]),
        "mask_cycle_counts": {str(key): value for key, value in cycle_counts.items()},
        "parameters": int(sum(value.numel() for value in model.parameters())),
    }


def write_report(result: dict[str, Any]) -> None:
    item = result["candidate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text(
        "# P2 local prefix masked-public auxiliary v17\n\n## 결론\n\n"
        f"상태 `{result['status']}`; pooled ΔRMSE `{item['delta_rmse']:+.9f}℃`, canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, fixed-penalty `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점. 모든 값은 노출된 historical surface의 탐색 지표이며 official 성능 주장이 아니다.\n\n"
        "External depth-query plan은 downloaded/trained/evaluated가 모두 false이며 실행 output이 없다. v17은 external data나 target reconstruction 없이 각 fold prefix 안에서 current public node 하나만 fixed cycle로 마스킹한다. Vincent et al.은 denoising objective 동기만 제공한다.\n\nofficial/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )
    (REPORT / "claim-source-ledger.md").write_text(
        "# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| A denoising criterion can guide representation learning | Vincent et al., JMLR 2010, https://www.jmlr.org/papers/v11/vincent10a.html | auxiliary-objective motivation only |\n| v17 metrics and access | `result.json`, `independent-qa.json` | local exposed historical evidence only |\n",
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
    v12.atomic_json(ARTIFACT / "attempt_lock.json", {"experiment_id": EXPERIMENT_ID, "config_sha256": v12.sha256_file(CONFIG), "runner_sha256": v12.sha256_file(RUNNER), "model_module_sha256": v12.sha256_file(MODEL_MODULE), "candidate": PREDICTION_NAME, "fit_plan": "3_folds_x_3_seeds", "result_adaptive_tuning": False})
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
    if tokens.shape[1:] != (5, 8) or context.shape[1] != 11 or not np.isfinite(tokens).all() or not np.isfinite(context).all():
        raise v12.ContractError("v17 pointwise representation contract failed")
    representation_receipt = {"token_shape": list(tokens.shape), "context_shape": list(context.shape), "valid_token_share": float(token_mask.mean()), "finite": True, "ordered_public_slots": [1, 5, 6, 7, 8], "temporal_axis_present": False, "external_rows_read": 0, "current_target_temp_or_psal_feature_present": False}
    local = design.keys["time"].dt.tz_convert("Asia/Seoul")
    design_index = pd.MultiIndex.from_arrays([v12.metric_engine.canonical_time_ns(design.keys["time"]), design.keys["layer"]])
    query_index = pd.MultiIndex.from_arrays([v12.metric_engine.canonical_time_ns(blind["time"]), blind["layer"]])
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
        weights, weight_receipt = v13.domain_balanced_weights(design.keys.loc[train_mask, "layer"].to_numpy(int), local[train_mask])
        query_mask = blind["fold"].eq(fold).to_numpy()
        predictions: list[np.ndarray] = []
        fit_receipts: list[dict[str, Any]] = []
        for seed in config["training"]["seeds"]:
            prediction, receipt = train_predict_seed(tokens[train_mask], token_mask[train_mask], context[train_mask], design.normalized_target[train_mask], weights, tokens[positions[query_mask]], token_mask[positions[query_mask]], context[positions[query_mask]], config, int(seed))
            predictions.append(prediction)
            fit_receipts.append(receipt)
        model_prediction[query_mask] = np.mean(np.vstack(predictions), axis=0)
        fit_count += len(fit_receipts)
        fold_receipts[fold] = {"training_start_kst": start.isoformat(), "training_cutoff_exclusive_kst": cutoff.isoformat(), "training_rows": int(train_mask.sum()), "training_calendar_months": sorted(set(local[train_mask].dt.month.astype(int))), "weight_receipt": weight_receipt, "fit_receipts": fit_receipts}
    if fit_count != 9 or not np.isfinite(model_prediction).all():
        raise v12.ContractError("v17 fit/prediction contract failed")
    absolute_model = design.baseline[positions] + model_prediction * design.profile_scale[positions]
    clipped = np.clip(absolute_model - reference, -float(config["training"]["model_minus_champion_clip_C"]), float(config["training"]["model_minus_champion_clip_C"]))
    candidate = reference + float(config["training"]["model_weight"]) * clipped
    prediction_path = ARTIFACT / f"{PREDICTION_NAME}.npz"
    np.savez_compressed(prediction_path, time_ns=v12.metric_engine.canonical_time_ns(blind["time"]), layer=blind["layer"].to_numpy(np.int16), fold=blind["fold"].to_numpy(str), reference=reference, candidate=candidate)
    commitment = {"path": str(prediction_path), "sha256": v12.sha256_file(prediction_path), "rows": len(candidate), "metric_computed_at_commitment": False}
    v12.atomic_json(ARTIFACT / "prediction_commitment.json", commitment)
    truth = design.truth[positions]
    spec = v12.metric_engine.CandidateSpec(name=PREDICTION_NAME, objective="domain_balanced_target_SmoothL1_plus_fixed_masked_public_auxiliary", conditional=False)
    record = v12.metric_engine.evaluate_candidate(spec, blind, truth, reference, candidate, config)
    record["by_month"] = v12.by_month_metrics(blind, truth, reference, candidate)
    record["action_geometry"] = v12.action_geometry(truth, reference, candidate)
    slope = float(config["evaluation"]["points_per_rmse_C"])
    penalty = float(config["evaluation"]["transport_penalty_points"])
    nominal = -float(record["delta_rmse"]) * slope
    transport = nominal - penalty
    record["canonical_nominal_pooled_points_delta"] = nominal
    record["canonical_transport_adjusted_pooled_points_delta"] = transport
    record["legacy_engine_score_translation"] = {"raw_expected_points_delta": record["raw_expected_points_delta"], "transport_calibrated_expected_points_delta": record["transport_calibrated_expected_points_delta"], "meaning": "official-like Sep-Oct day-bootstrap CI90-high translation"}
    record["prediction_commitment"] = commitment
    safety = v13.safety_gate(record, transport)
    record["safety_gate_checks"] = safety
    record["safety_pass"] = bool(all(safety.values()))
    status = "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION" if record["strict_exploratory_pass"] and record["safety_pass"] else "EXPLORATORY_NO_GO_MASKED_PUBLIC_AUXILIARY"
    result = {"schema_version": "p2.local_prefix_masked_public_auxiliary.result.20260901.v17", "experiment_id": EXPERIMENT_ID, "status": status, "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION", "runtime_seconds": time.perf_counter() - started, "fit_count": fit_count, "semantic_audit": semantic, "representation": representation_receipt, "architecture_contract": architecture, "training": {"row_deletion": 0, "auxiliary_weight": float(config["auxiliary"]["weight"]), "folds": fold_receipts}, "candidate": record, "operation_counters": {"observations_rows_read": int(len(observations)), "historical_scoring_rows_read": int(len(scored)), "external_rows_read": 0, "official_test_index_rows_read": 0, "sample_rows_read": 0, "baseline_file_rows_read": 0, "score_file_rows_read": 0, "query_support_rows_read": 0, "hidden_truth_rows_read": 0, "submission_csv_created": 0, "uploads": 0}, "hashes": {"config": v12.sha256_file(CONFIG), "runner": v12.sha256_file(RUNNER), "model_module": v12.sha256_file(MODEL_MODULE), "observations": v12.sha256_file(observations_path), "scoring_frame": v12.sha256_file(scoring_path), "prediction_npz": commitment["sha256"]}}
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
