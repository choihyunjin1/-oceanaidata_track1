"""Run one sealed exploratory P2 permutation-invariant vertical set encoder.

The executable opens only ``observations.csv`` and the frozen truth-free
historical scoring frame.  It has no official-test, sample, baseline-file,
submission, score, hidden-answer, materialization, or upload code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for item in (SRC, SCRIPTS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_group_balanced_raw_residual_20260901_v8 as metric_engine  # noqa: E402

from p2_restore.features import PUBLIC_LAYERS, build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    TEMPORAL_FEATURES,
    build_normalized_curvature_design,
)

EXPERIMENT_ID = "p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
RESULT_SCHEMA = "p2.continuous_depth_permutation_invariant_set_encoder.result.20260901.v12"
TARGET_LAYERS = (2, 3, 4)
KEYS = ("time", "layer")


class ContractError(RuntimeError):
    """Raised when the sealed experiment contract drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ContractError("experiment ID drift")
    if config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED":
        raise ContractError("config is not preregistered")
    training = config["training"]
    if training["champion_preserving_weight"] != 0.8 or training["model_weight"] != 0.2:
        raise ContractError("fixed 0.8/0.2 blend drift")
    if training["row_deletion"] or training["outer_fold_tuning"]:
        raise ContractError("row deletion or outer tuning is forbidden")
    if len(training["seeds"]) > int(training["maximum_fit_count"]):
        raise ContractError("fit budget exceeded")
    return config


def resolve_observations(config: dict[str, Any]) -> Path:
    raw = os.environ.get(config["source_contract"]["environment_variable"])
    if not raw:
        raise ContractError("P2_DATA_DIR is required")
    path = Path(raw).resolve() / config["source_contract"]["only_source_filename"]
    if path.name != "observations.csv" or not path.is_file():
        raise ContractError("only P2_DATA_DIR/observations.csv may be opened")
    if sha256_file(path) != config["source_contract"]["observations_sha256"]:
        raise ContractError("observations hash drift")
    return path


class VerticalDeepSet(nn.Module):
    """Shared public-depth element map followed by commutative pooling."""

    def __init__(self, token_features: int, context_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(token_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + context_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: Tensor, token_mask: Tensor, context: Tensor) -> Tensor:
        encoded = self.element(tokens)
        mask = token_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (encoded * mask).sum(dim=1) / count
        negative = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.bool(), negative).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(torch.cat((mean, maximum, context), dim=1)).squeeze(1)


def permutation_invariance_receipt() -> dict[str, Any]:
    torch.manual_seed(7)
    model = VerticalDeepSet(8, 11, hidden=32).eval()
    tokens = torch.randn(4, 5, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    context = torch.randn(4, 11)
    order = torch.tensor([4, 2, 0, 3, 1])
    with torch.inference_mode():
        left = model(tokens, mask, context)
        right = model(tokens[:, order], mask[:, order], context)
    maximum_error = float(torch.max(torch.abs(left - right)))
    if maximum_error > 1e-6:
        raise ContractError("set encoder is not permutation invariant")
    return {"public_layer_permutation": order.tolist(), "maximum_abs_error": maximum_error}


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    receipts: dict[str, str] = {}
    for relative in config["semantic_audit"]["evidence_paths"]:
        path = ROOT / relative
        if not path.is_file():
            raise ContractError(f"semantic-audit evidence missing: {relative}")
        receipts[relative] = sha256_file(path)
    return {
        "classification": config["semantic_audit"]["classification"],
        "reason": config["semantic_audit"]["reason"],
        "evidence_sha256": receipts,
        "prior_vertical_set_encoder_execution_found": False,
        "prior_continuous_depth_tcn_is_semantically_distinct": True,
        "external_neural_field_execution_flags": {
            "downloaded": False,
            "trained": False,
            "evaluated": False,
            "submitted": False,
        },
    }


def preflight() -> dict[str, Any]:
    config = load_config()
    audit = semantic_audit(config)
    receipt = permutation_invariance_receipt()
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate_count": 1,
        "maximum_fit_count": config["operation_limits"]["maximum_fit_count"],
        "seeds": config["training"]["seeds"],
        "permutation_invariance": receipt,
        "semantic_audit_sha256": sha256_json(audit),
        "config_sha256": sha256_file(CONFIG),
        "runner_sha256": sha256_file(RUNNER),
        "data_rows_read": 0,
        "model_fits": 0,
        "artifacts_written": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = sha256_json(payload)
    return payload


def registered_window_ids(
    times: pd.Series | pd.DatetimeIndex, windows: list[dict[str, str]]
) -> np.ndarray:
    return metric_engine.registered_window_ids(times, windows)


def build_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = build_normalized_curvature_design(frame)
    n_rows = len(frame)
    public_psal = np.column_stack(
        [pd.to_numeric(frame[f"psal_{layer}"], errors="coerce") for layer in PUBLIC_LAYERS]
    ).astype(float)
    psal_finite = np.isfinite(public_psal)
    psal_count = psal_finite.sum(axis=1)
    psal_mean = np.divide(
        np.nansum(public_psal, axis=1),
        psal_count,
        out=np.zeros(n_rows, dtype=float),
        where=psal_count > 0,
    )
    psal_min = np.zeros(n_rows, dtype=float)
    psal_max = np.zeros(n_rows, dtype=float)
    usable_psal = psal_count > 0
    psal_min[usable_psal] = np.nanmin(public_psal[usable_psal], axis=1)
    psal_max[usable_psal] = np.nanmax(public_psal[usable_psal], axis=1)
    psal_scale = np.maximum(psal_max - psal_min, 0.05)
    target_depth = pd.to_numeric(frame["target_depth"], errors="raise").to_numpy(float)
    baseline = design.baseline
    tokens = np.zeros((n_rows, len(PUBLIC_LAYERS), 8), dtype=np.float32)
    token_mask = np.zeros((n_rows, len(PUBLIC_LAYERS)), dtype=np.float32)
    for index, layer in enumerate(PUBLIC_LAYERS):
        temp = pd.to_numeric(frame[f"temp_{layer}"], errors="coerce").to_numpy(float)
        psal = pd.to_numeric(frame[f"psal_{layer}"], errors="coerce").to_numpy(float)
        depth = pd.to_numeric(frame[f"depth_{layer}"], errors="coerce").to_numpy(float)
        nominal = pd.to_numeric(frame[f"nominal_{layer}"], errors="coerce").to_numpy(float)
        present = np.column_stack(
            (np.isfinite(temp), np.isfinite(psal), np.isfinite(depth), np.isfinite(nominal))
        ).astype(np.float32)
        values = np.column_stack(
            (
                (temp - baseline) / design.profile_scale,
                (psal - psal_mean) / psal_scale,
                (depth - target_depth) / 50.0,
                (nominal - target_depth) / 50.0,
                present,
            )
        )
        values = np.nan_to_num(values, nan=0.0, posinf=12.0, neginf=-12.0)
        tokens[:, index] = np.clip(values, -12.0, 12.0).astype(np.float32)
        token_mask[:, index] = (np.isfinite(temp) & np.isfinite(nominal)).astype(np.float32)
    if np.any(token_mask.sum(axis=1) < 2):
        raise ContractError("fewer than two public temperature/depth tokens")
    layer = pd.to_numeric(frame["layer"], errors="raise").to_numpy(int)
    one_hot = np.column_stack([layer == value for value in TARGET_LAYERS]).astype(np.float32)
    context_values = [target_depth / 50.0, *[one_hot[:, i] for i in range(3)]]
    context_values.append(np.log1p(design.profile_scale))
    context_values.extend(
        pd.to_numeric(frame[name], errors="raise").to_numpy(float) for name in TEMPORAL_FEATURES
    )
    context = np.column_stack(context_values).astype(np.float32)
    if context.shape[1] != 11 or not np.isfinite(context).all():
        raise ContractError("context feature contract failed")
    return tokens, token_mask, context


def train_predict_seed(
    tokens: np.ndarray,
    mask: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
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
    model = VerticalDeepSet(tokens.shape[2], context.shape[1], hidden=32).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    train_tensors = tuple(
        torch.from_numpy(value) for value in (tokens, mask, context, target.astype(np.float32))
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = int(config["training"]["batch_size"])
    losses: list[float] = []
    model.train()
    for _epoch in range(int(config["training"]["epochs"])):
        order = torch.randperm(len(target), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            batch = [value[selected].to(device) for value in train_tensors]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch[0], batch[1], batch[2])
            loss = F.smooth_l1_loss(prediction, batch[3], beta=1.0, reduction="mean")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(selected)
            seen += len(selected)
        losses.append(total / seen)
    model.eval()
    query = []
    with torch.inference_mode():
        for start in range(0, len(query_tokens), batch_size):
            stop = start + batch_size
            query.append(
                model(
                    torch.from_numpy(query_tokens[start:stop]).to(device),
                    torch.from_numpy(query_mask[start:stop]).to(device),
                    torch.from_numpy(query_context[start:stop]).to(device),
                ).cpu().numpy()
            )
    prediction = np.concatenate(query).astype(float)
    if not np.isfinite(prediction).all():
        raise ContractError("model prediction is nonfinite")
    return prediction, {
        "seed": seed,
        "device": str(device),
        "epochs": len(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "parameters": int(sum(value.numel() for value in model.parameters())),
    }


def by_month_metrics(
    blind: pd.DataFrame, truth: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, dict[str, float]]:
    local = blind["time"].dt.tz_convert("Asia/Seoul")
    labels = local.dt.strftime("%Y-%m")
    output: dict[str, dict[str, float]] = {}
    for label in sorted(labels.unique()):
        selected = labels.eq(label).to_numpy()
        ref = metric_engine.rmse(truth[selected], reference[selected])
        cand = metric_engine.rmse(truth[selected], candidate[selected])
        output[label] = {
            "rows": int(selected.sum()),
            "reference_rmse": ref,
            "candidate_rmse": cand,
            "delta_rmse": cand - ref,
        }
    return output


def action_geometry(
    truth: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    action = candidate - reference
    absolute = np.abs(action)
    gain = np.square(reference - truth) - np.square(candidate - truth)
    ordered = np.argsort(absolute)[::-1]
    total_gain = float(gain.sum())
    def concentration(fraction: float) -> float | None:
        count = max(1, int(math.ceil(len(action) * fraction)))
        return None if abs(total_gain) < 1e-15 else float(gain[ordered[:count]].sum() / total_gain)
    return {
        "active_rows_gt_1e_12": int(np.count_nonzero(absolute > 1e-12)),
        "active_share_gt_1e_12": float(np.mean(absolute > 1e-12)),
        "abs_action_p50_C": float(np.quantile(absolute, 0.50)),
        "abs_action_p90_C": float(np.quantile(absolute, 0.90)),
        "abs_action_p95_C": float(np.quantile(absolute, 0.95)),
        "abs_action_p99_C": float(np.quantile(absolute, 0.99)),
        "abs_action_p999_C": float(np.quantile(absolute, 0.999)),
        "abs_action_max_C": float(absolute.max()),
        "action_rms_C": float(np.sqrt(np.mean(np.square(action)))),
        "sse_gain_top_0_1pct_share": concentration(0.001),
        "sse_gain_top_1pct_share": concentration(0.01),
    }


def write_report(result: dict[str, Any]) -> None:
    item = result["candidate"]
    folds = item["by_fold"]
    lines = [
        "# P2 continuous-depth permutation-invariant set encoder 20260901 v12",
        "",
        "## 결론",
        "",
        f"상태: `{result['status']}`. exposed historical surface의 탐색 결과이며 fresh confirmation이 아니다.",
        "",
        "| pooled ΔRMSE | Sep-Oct | Jul-Aug | Nov-Dec | nominal points | transport points | gate |",
        "|---:|---:|---:|---:|---:|---:|---|",
        "| {pooled:+.9f} | {sep:+.9f} | {jul:+.9f} | {nov:+.9f} | {nominal:+.6f} | {transport:+.6f} | {gate} |".format(
            pooled=item["delta_rmse"],
            sep=folds["2024_sep_oct"]["delta_rmse"],
            jul=folds["2025_jul_aug"]["delta_rmse"],
            nov=folds["2025_nov_dec"]["delta_rmse"],
            nominal=item["nominal_pooled_points_delta"],
            transport=item["transport_calibrated_expected_points_delta"],
            gate="PASS" if item["strict_exploratory_pass"] else "NO_GO",
        ),
        "",
        "## 새 구조와 중복 배제",
        "",
        "공개층 `(depth, value, mask)` token마다 같은 element MLP를 적용하고 masked mean/max로 가환 집계했다. 과거 continuous-depth TCN은 시간축 convolution과 order-sensitive fixed public-layer vector를 사용하므로 의미 중복이 아니다. Heavy-model scout의 vertical set encoder는 제안만 있었고 실행 receipt가 없다.",
        "",
        "Deep Sets (Zaheer et al., NeurIPS 2017)의 가환 pooling 원칙을 구현했고 공개층 순열 단위 테스트를 통과했다. Set Transformer의 attention은 작은 표본에서 새 sweep이 되므로 사용하지 않았다.",
        "",
        "행 삭제, early stopping, outer-fold 튜닝, 결과 적응 재시도는 모두 0이다. champion 80%를 보존하고 단일 frozen model 평균을 20%만 혼합했다.",
        "",
        "## 경계",
        "",
        "세 historical block은 이미 노출됐으므로 이 결과는 후보 폐쇄/우선순위용 탐색 증거다. 공식 test/sample/baseline/score/query/hidden rows=0, CSV=0, upload=0.",
    ]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report-source.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    started = time.perf_counter()
    if ARTIFACT.exists():
        raise FileExistsError(ARTIFACT)
    config = load_config()
    audit = semantic_audit(config)
    permutation = permutation_invariance_receipt()
    ARTIFACT.mkdir(parents=True)
    atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(CONFIG),
            "runner_sha256": sha256_file(RUNNER),
            "candidate": "P2_V12_COMPACT_DEEPSET_BLEND020",
            "seeds": config["training"]["seeds"],
            "result_adaptive_tuning": False,
            "automatic_retry_count": 0,
        },
    )
    observations_path = resolve_observations(config)
    scoring_path = ROOT / config["source_contract"]["scoring_frame"]
    if sha256_file(scoring_path) != config["source_contract"]["scoring_frame_sha256"]:
        raise ContractError("truth-free scoring frame hash drift")
    observations = pd.read_csv(observations_path, dtype={"station": "string", "time": "string"})
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    if observations.duplicated(["time", "layer"]).any():
        raise ContractError("observation keys are duplicated")
    scored = pd.read_parquet(scoring_path)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    if tuple(sorted(scored["fold"].unique())) != tuple(sorted(metric_engine.FOLD_ORDER)):
        raise ContractError("fold set drift")
    blind, reference = metric_engine.make_reference(observations, scored)

    feature_table = build_training_features(observations)
    design = build_normalized_curvature_design(feature_table.frame)
    tokens, token_mask, context = build_arrays(feature_table.frame)
    local = design.keys["time"].dt.tz_convert("Asia/Seoul")
    window_id = registered_window_ids(local, config["training"]["registered_windows_kst"])
    train_mask = window_id != ""
    if int(train_mask.sum()) != int(config["training"]["expected_rows"]):
        raise ContractError("training row count drift")

    design_index = pd.MultiIndex.from_arrays(
        [metric_engine.canonical_time_ns(design.keys["time"]), design.keys["layer"]], names=KEYS
    )
    query_index = pd.MultiIndex.from_arrays(
        [metric_engine.canonical_time_ns(blind["time"]), blind["layer"]], names=KEYS
    )
    positions = design_index.get_indexer(query_index)
    if design_index.has_duplicates or np.any(positions < 0):
        raise ContractError("historical query alignment failed")

    normalized_predictions: list[np.ndarray] = []
    fit_receipts: list[dict[str, Any]] = []
    for seed in config["training"]["seeds"]:
        prediction, receipt = train_predict_seed(
            tokens[train_mask],
            token_mask[train_mask],
            context[train_mask],
            design.normalized_target[train_mask],
            tokens[positions],
            token_mask[positions],
            context[positions],
            config,
            int(seed),
        )
        normalized_predictions.append(prediction)
        fit_receipts.append(receipt)
    mean_normalized = np.mean(np.vstack(normalized_predictions), axis=0)
    model_prediction = design.baseline[positions] + mean_normalized * design.profile_scale[positions]
    clipped_delta = np.clip(
        model_prediction - reference,
        -float(config["training"]["model_minus_champion_clip_C"]),
        float(config["training"]["model_minus_champion_clip_C"]),
    )
    candidate = reference + float(config["training"]["model_weight"]) * clipped_delta
    if not np.isfinite(candidate).all():
        raise ContractError("candidate is nonfinite")

    prediction_path = ARTIFACT / "P2_V12_COMPACT_DEEPSET_BLEND020.npz"
    np.savez_compressed(
        prediction_path,
        time_ns=metric_engine.canonical_time_ns(blind["time"]),
        layer=blind["layer"].to_numpy(np.int16),
        fold=blind["fold"].to_numpy(str),
        reference=reference,
        candidate=candidate,
    )
    commitment = {
        "path": str(prediction_path),
        "sha256": sha256_file(prediction_path),
        "rows": len(candidate),
        "fit_receipts": fit_receipts,
        "metric_computed_at_commitment": False,
    }
    atomic_json(ARTIFACT / "prediction_commitment.json", commitment)

    truth = design.truth[positions]
    spec = metric_engine.CandidateSpec(
        name="P2_V12_COMPACT_DEEPSET_BLEND020", objective="SmoothL1_beta_1.0", conditional=False
    )
    record = metric_engine.evaluate_candidate(spec, blind, truth, reference, candidate, config)
    record["nominal_pooled_points_delta"] = float(
        -record["delta_rmse"] * config["evaluation"]["points_per_rmse_C"]
    )
    record["by_month"] = by_month_metrics(blind, truth, reference, candidate)
    record["action_geometry"] = action_geometry(truth, reference, candidate)
    record["prediction_commitment"] = commitment
    status = (
        "EXPLORATORY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if record["strict_exploratory_pass"]
        else "EXPLORATORY_NO_GO_CLOSED_SINGLE_FROZEN_SET_ENCODER"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_level": "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": len(fit_receipts),
        "semantic_audit": audit,
        "permutation_invariance": permutation,
        "training": {
            "rows": int(train_mask.sum()),
            "token_shape": list(tokens[train_mask].shape),
            "context_features": int(context.shape[1]),
            "row_deletion": 0,
            "fit_receipts": fit_receipts,
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
            "config": sha256_file(CONFIG),
            "runner": sha256_file(RUNNER),
            "observations": sha256_file(observations_path),
            "scoring_frame": sha256_file(scoring_path),
            "prediction_npz": commitment["sha256"],
        },
    }
    atomic_json(ARTIFACT / "result.json", result)
    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT / "result.json", result)
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
