"""Run one sealed layer-conditioned month-latent Deep-CORAL objective for P2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402
import run_p2_layer_month_risk_variance_rex_20260901_v19 as v19  # noqa: E402
import run_p2_prefix_safe_domain_balanced_deepset_20260901_v13 as v13  # noqa: E402
import run_p2_regularized_layer_month_group_dro_20260901_v18 as v18  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)

EXPERIMENT_ID = "p2_layer_conditioned_month_covariance_alignment_20260901_v20"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
PREDICTION_NAME = "P2_V20_LAYER_CONDITIONED_MONTH_CORAL_BLEND020"


class LatentVerticalDeepSet(v12.VerticalDeepSet):
    """The exact v13 network with its first head hidden state exposed."""

    def latent_and_prediction(
        self,
        tokens: Tensor,
        token_mask: Tensor,
        context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        encoded = self.element(tokens)
        mask = token_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool(), negative).amax(dim=1)
        maximum = torch.where(
            torch.isfinite(maximum),
            maximum,
            torch.zeros_like(maximum),
        )
        joined = torch.cat((mean, maximum, context), dim=1)
        latent = self.head[1](self.head[0](joined))
        hidden = self.head[3](self.head[2](latent))
        prediction = self.head[4](hidden).squeeze(1)
        return prediction, latent

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        prediction, _latent = self.latent_and_prediction(tokens, token_mask, context)
        return prediction


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
    ):
        raise v12.ContractError("v20 config identity/status drift")
    auth = config["authorization_evidence"]
    result_path = ROOT / auth["predecessor"]
    qa_path = ROOT / auth["independent_qa"]
    if (
        v12.sha256_file(result_path) != auth["predecessor_sha256"]
        or v12.sha256_file(qa_path) != auth["independent_qa_sha256"]
    ):
        raise v12.ContractError("v19 authorization hash drift")
    if (
        json.loads(result_path.read_text(encoding="utf-8"))["status"]
        != auth["required_status"]
        or json.loads(qa_path.read_text(encoding="utf-8"))["status"] != "PASS"
    ):
        raise v12.ContractError("v19 authorization state drift")
    training = config["training"]
    expected = {
        "architecture": "v13_exact_DeepSets_shared_element_MLP32x2_masked_mean_max_head32x2",
        "alignment_latent": "first_head_hidden_relu_width32",
        "environment_definition": "target_layer_x_calendar_month",
        "within_environment_weighting": "equal_KST_day_then_equal_row",
        "objective": "weighted_SmoothL1_plus_layer_conditioned_pairwise_month_CORAL",
        "coral_coefficient": 1.0,
        "coral_normalization": "frobenius_squared_divided_by_4d_squared",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 60,
        "batch_size": 4096,
        "model_weight": 0.2,
        "champion_preserving_weight": 0.8,
    }
    if any(training[key] != value for key, value in expected.items()):
        raise v12.ContractError("v20 scientific contract drift")
    if (
        training["row_deletion"]
        or training["early_stopping"]
        or training["outer_fold_tuning"]
        or len(training["seeds"]) * 3 != 9
        or training["maximum_fit_count"] != 9
    ):
        raise v12.ContractError("v20 fit/no-deletion contract drift")
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
        if "deep_coral" in source or "month_covariance_alignment" in source:
            prior_runners.append(path.name)
    prior_artifacts = sorted(
        path.name
        for pattern in ("p2*coral*", "p2*covariance_alignment*")
        for path in (ROOT / "artifacts").glob(pattern)
        if path.name != EXPERIMENT_ID
    )
    prior_reports = sorted(
        path.name
        for pattern in ("p2*coral*", "p2*covariance_alignment*")
        for path in (ROOT / "reports").glob(pattern)
        if path.name != EXPERIMENT_ID
    )
    if prior_runners or prior_artifacts or prior_reports:
        raise v12.ContractError("prior P2 CORAL execution namespace exists")
    for predecessor in (v18.RUNNER, v19.RUNNER):
        source = predecessor.read_text(encoding="utf-8").lower()
        if "deep_coral" in source or "month_covariance_alignment" in source:
            raise v12.ContractError("predecessor unexpectedly contains CORAL")
    return {
        "classification": config["semantic_audit"]["classification"],
        "reason": config["semantic_audit"]["reason"],
        "evidence_sha256": evidence_hashes,
        "prior_p2_coral_runners": sorted(prior_runners),
        "prior_p2_coral_artifacts": sorted(set(prior_artifacts)),
        "prior_p2_coral_reports": sorted(set(prior_reports)),
        "v18_minimax_loss_distinguished": True,
        "v19_label_risk_variance_distinguished": True,
        "alignment_is_within_target_layer_only": True,
        "result_adaptive_environment_or_hyperparameter_selection": False,
    }


def layer_groups_from_labels(labels: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        layer, month = label.split(":")
        if not layer.startswith("layer") or not month.startswith("month"):
            raise v12.ContractError("invalid layer-month environment label")
        groups.setdefault(layer, []).append(index)
    if set(groups) != {"layer2", "layer3", "layer4"}:
        raise v12.ContractError("v20 target-layer environment contract drift")
    return groups


def covariance_matrix(values: Tensor) -> Tensor:
    if values.ndim != 2 or len(values) < 2:
        raise v12.ContractError("CORAL covariance requires at least two rows")
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1) @ centered / float(len(values) - 1)


def layer_conditioned_month_coral(
    latent: Tensor,
    environment_index: Tensor,
    layer_groups: dict[str, list[int]],
) -> tuple[Tensor, int, int]:
    """Pairwise Deep-CORAL loss across months, never across target layers."""
    pair_losses: list[Tensor] = []
    minimum_rows = len(latent)
    dimension = latent.shape[1]
    for group_indices in layer_groups.values():
        present: list[tuple[int, Tensor]] = []
        for group_index in group_indices:
            selected = latent[environment_index == group_index]
            if len(selected) >= 2:
                present.append((group_index, covariance_matrix(selected)))
                minimum_rows = min(minimum_rows, len(selected))
        for (_left_id, left), (_right_id, right) in combinations(present, 2):
            pair_losses.append(
                (left - right).square().sum() / float(4 * dimension * dimension)
            )
    if not pair_losses:
        raise v12.ContractError("CORAL batch has no within-layer month pair")
    return torch.stack(pair_losses).mean(), len(pair_losses), minimum_rows


def train_predict_seed(
    tokens: np.ndarray,
    mask: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    base_weights: np.ndarray,
    environment_index: np.ndarray,
    environment_labels: list[str],
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
    model = LatentVerticalDeepSet(8, 11, hidden=32).to(device)
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
            environment_index.astype(np.int64),
        )
    )
    layer_groups = layer_groups_from_labels(environment_labels)
    batch_size = int(config["training"]["batch_size"])
    coefficient = float(config["training"]["coral_coefficient"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    objective_history: list[float] = []
    supervised_history: list[float] = []
    coral_history: list[float] = []
    minimum_pair_count = 1 << 30
    minimum_rows_per_covariance = 1 << 30
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        objective_sum = supervised_sum = coral_sum = 0.0
        batch_count = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train]
            optimizer.zero_grad(set_to_none=True)
            prediction, latent = model.latent_and_prediction(
                batch[0], batch[1], batch[2]
            )
            raw_loss = F.smooth_l1_loss(
                prediction,
                batch[3],
                beta=1.0,
                reduction="none",
            )
            supervised = (raw_loss * batch[4]).sum() / batch[4].sum().clamp_min(
                1e-12
            )
            coral, pair_count, minimum_rows = layer_conditioned_month_coral(
                latent,
                batch[5],
                layer_groups,
            )
            objective = supervised + coefficient * coral
            objective.backward()
            optimizer.step()
            objective_sum += float(objective.detach().cpu())
            supervised_sum += float(supervised.detach().cpu())
            coral_sum += float(coral.detach().cpu())
            batch_count += 1
            minimum_pair_count = min(minimum_pair_count, pair_count)
            minimum_rows_per_covariance = min(
                minimum_rows_per_covariance,
                minimum_rows,
            )
        objective_history.append(objective_sum / batch_count)
        supervised_history.append(supervised_sum / batch_count)
        coral_history.append(coral_sum / batch_count)
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
        "epochs": len(objective_history),
        "environment_count": len(environment_labels),
        "coral_coefficient": coefficient,
        "weight_decay": float(config["training"]["weight_decay"]),
        "objective_first": objective_history[0],
        "objective_last": objective_history[-1],
        "supervised_loss_first": supervised_history[0],
        "supervised_loss_last": supervised_history[-1],
        "coral_loss_first": coral_history[0],
        "coral_loss_last": coral_history[-1],
        "minimum_within_layer_month_pairs_per_batch": minimum_pair_count,
        "minimum_rows_per_covariance": minimum_rows_per_covariance,
        "latent_covariance_finite": bool(np.isfinite(coral_history).all()),
        "alignment_crosses_target_layers": False,
        "parameters": int(sum(value.numel() for value in model.parameters())),
    }


def synthetic_coral_receipt() -> dict[str, Any]:
    latent = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [0.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [3.0, 2.0],
            [2.0, 0.0],
            [4.0, 3.0],
        ],
        dtype=torch.float64,
    )
    environment = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    groups = {"layer2": [0, 1], "layer3": [2, 3], "layer4": []}
    loss, pairs, minimum_rows = layer_conditioned_month_coral(
        latent,
        environment,
        groups,
    )
    order = torch.tensor([7, 2, 4, 0, 5, 1, 3, 6])
    permuted, permuted_pairs, permuted_minimum = layer_conditioned_month_coral(
        latent[order],
        environment[order],
        groups,
    )
    if not torch.allclose(loss, permuted, atol=1e-12, rtol=0.0):
        raise v12.ContractError("synthetic CORAL row permutation drift")
    return {
        "loss": float(loss),
        "within_layer_pair_count": pairs,
        "minimum_rows_per_covariance": minimum_rows,
        "permuted_pair_count": permuted_pairs,
        "permuted_minimum_rows": permuted_minimum,
        "row_permutation_invariant": True,
        "cross_layer_pairs": 0,
    }


def preflight() -> dict[str, Any]:
    config = load_config()
    semantic = semantic_audit(config)
    synthetic = synthetic_coral_receipt()
    if (
        synthetic["within_layer_pair_count"] != 2
        or synthetic["minimum_rows_per_covariance"] != 2
        or not np.isfinite(synthetic["loss"])
    ):
        raise v12.ContractError("v20 synthetic CORAL contract failed")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate_count": 1,
        "maximum_fit_count": 9,
        "semantic_audit": semantic,
        "semantic_audit_sha256": v12.sha256_json(semantic),
        "permutation_invariance": v12.permutation_invariance_receipt(),
        "synthetic_layer_conditioned_coral": synthetic,
        "fixed_coral_coefficient": 1.0,
        "fixed_weight_decay": 0.0001,
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
        "v13_runner_sha256": v12.sha256_file(v13.RUNNER),
        "v18_runner_sha256": v12.sha256_file(v18.RUNNER),
        "v19_runner_sha256": v12.sha256_file(v19.RUNNER),
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
        "# P2 layer-conditioned month covariance alignment v20\n\n## 결론\n\n"
        f"상태 `{result['status']}`; pooled ΔRMSE `{item['delta_rmse']:+.9f}℃`, "
        f"canonical nominal `{item['canonical_nominal_pooled_points_delta']:+.6f}`점, "
        f"fixed-penalty `{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점. "
        "반복 노출 historical surface의 탐색 증거이며 official 성능 주장이 아니다.\n\n"
        "v13 architecture/input/prefix/purge/seeds/epochs/blend/action cap은 고정했고, "
        "유일한 과학 변화는 같은 target layer 안의 calendar-month latent covariance에 "
        "고정 1.0 Deep-CORAL penalty를 적용한 것이다. cross-layer alignment, sweep, "
        "router, ensemble은 없다.\n\n"
        f"v13 대비 pooled ΔRMSE 차이 `{comparison['v20_minus_v13_delta_rmse']:+.9f}℃`, "
        f"v18 대비 `{comparison['v20_minus_v18_delta_rmse']:+.9f}℃`, v19 대비 "
        f"`{comparison['v20_minus_v19_delta_rmse']:+.9f}℃`다. 비교는 후보 원장용이다.\n\n"
        "official/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )
    (REPORT / "claim-source-ledger.md").write_text(
        "# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n"
        "| Deep CORAL aligns second-order feature statistics through covariance discrepancy | Sun and Saenko, ECCV Workshops 2016, https://arxiv.org/abs/1607.01719 | representation-objective motivation only; no P2 performance transfer |\n"
        "| v20 metrics and access | `result.json`, `independent-qa.json` | local exposed historical evidence only |\n",
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
            "alignment": "same_target_layer_calendar_month_covariance_only",
            "coral_coefficient": config["training"]["coral_coefficient"],
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
        cutoff = fold_start - pd.Timedelta(days=int(config["training"]["embargo_days"]))
        train_mask = (local >= start) & (local < cutoff)
        group_index, base_weights, group_labels, group_receipt = v18.build_group_contract(
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
            "training_calendar_months": sorted(set(local[train_mask].dt.month.astype(int))),
            "environment_receipt": group_receipt,
            "fit_receipts": fit_receipts,
        }
    if fit_count != 9 or not np.isfinite(model_prediction).all():
        raise v12.ContractError("v20 fit/prediction contract failed")
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
        objective="weighted_SmoothL1_plus_same_layer_month_Deep_CORAL",
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
        else "EXPLORATORY_NO_GO_LAYER_CONDITIONED_MONTH_COVARIANCE_ALIGNMENT"
    )
    preserved_paths = {
        "v13": ROOT / "reports/p2_prefix_safe_domain_balanced_deepset_20260901_v13/result.json",
        "v18": ROOT / "reports/p2_regularized_layer_month_group_dro_20260901_v18/result.json",
        "v19": ROOT / "reports/p2_layer_month_risk_variance_rex_20260901_v19/result.json",
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
        comparison[f"v20_minus_{name}_delta_rmse"] = float(record["delta_rmse"]) - float(
            item["delta_rmse"]
        )
        comparison[f"v20_minus_{name}_canonical_nominal_points"] = nominal - float(
            item["canonical_nominal_pooled_points_delta"]
        )
    result = {
        "schema_version": "p2.layer_conditioned_month_covariance_alignment.result.20260901.v20",
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
            "alignment": "same_target_layer_calendar_month_covariance_only",
            "coral_coefficient": float(config["training"]["coral_coefficient"]),
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
            "v19_runner": v12.sha256_file(v19.RUNNER),
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
