"""Exactly-once P2 public-sensor influence sensitivity experiment.

Only observations.csv and a truth-free historical prediction frame are inputs.
The candidate is sealed before target truth is consulted.  There is no code
path for official inputs, submission materialization, or upload.
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p2_group_balanced_raw_residual_20260901_v8 as metric_engine  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import (  # noqa: E402
    build_normalized_curvature_design,
)

EXPERIMENT_ID = "p2_public_sensor_influence_shrink_20260901_v10"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
FOLD_ORDER = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")
TARGET_LAYERS = (2, 3, 4)
SCHEMA_VERSION = "p2.public_sensor_influence_shrink.result.20260901.v10"


class ContractError(RuntimeError):
    """Fail-closed contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    value = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


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
        raise ContractError("experiment id drift")
    if config["status"] != "PREREGISTERED_EXPLORATORY_NOT_EXECUTED":
        raise ContractError("experiment is not preregistered")
    if config["result_adaptive_tuning"] or config["automatic_retry_count"] != 0:
        raise ContractError("adaptive tuning or retry is forbidden")
    if config["training_only_influence"]["row_deletion"]:
        raise ContractError("row deletion is forbidden")
    if config["candidate"]["candidate_count"] != 1:
        raise ContractError("exactly one candidate is required")
    if config["operation_limits"]["maximum_fit_count"] != 0:
        raise ContractError("this deterministic sensitivity must have zero fits")
    return config


def resolve_observations(config: dict[str, Any]) -> Path:
    raw = os.environ.get(config["source_contract"]["environment_variable"])
    if not raw:
        raise ContractError("P2_DATA_DIR is required")
    path = Path(raw).resolve() / config["source_contract"]["only_source_filename"]
    if path.name != "observations.csv" or not path.is_file():
        raise ContractError("only P2_DATA_DIR/observations.csv may be opened")
    if sha256_file(path) != config["source_contract"]["observations_sha256"]:
        raise ContractError("observations.csv hash drift")
    return path


def semantic_audit(config: dict[str, Any]) -> dict[str, Any]:
    paths = [ROOT / item for item in config["negative_registry_audit"]["evidence_paths"]]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise ContractError(f"semantic-audit evidence missing: {missing}")
    return {
        "status": "PASS_NON_DUPLICATE_SINGLE_UNUSED_AXIS",
        "fingerprint": config["semantic_fingerprint"],
        "evidence": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for path in paths
        ],
        "closed_or_excluded": config["negative_registry_audit"]["closed_or_excluded"],
        "nonduplicate_reason": config["negative_registry_audit"]["nonduplicate_reason"],
        "row_deletion": False,
    }


def preflight() -> dict[str, Any]:
    config = load_config()
    audit = semantic_audit(config)
    candidate = config["candidate"]["name"]
    if candidate != "P2_V10_HUBER6_PUBLIC_TEMP_DIFF_INFLUENCE_SHRINK":
        raise ContractError("candidate drift")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate": candidate,
        "semantic_fingerprint": config["semantic_fingerprint"],
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


def _training_window_mask(
    times: pd.Series, windows: list[dict[str, str]]
) -> np.ndarray:
    local = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).tz_convert("Asia/Seoul")
    mask = np.zeros(len(local), dtype=bool)
    for window in windows:
        start = pd.Timestamp(window["start_inclusive"]).tz_convert("Asia/Seoul")
        end = pd.Timestamp(window["end_exclusive"]).tz_convert("Asia/Seoul")
        selected = (local >= start) & (local < end)
        if np.any(mask & selected):
            raise ContractError("training windows overlap")
        mask |= selected
    return mask


def build_public_influence(
    observations: pd.DataFrame, blind: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return truth-free aggregate robust-z and Huber influence weights."""

    contract = config["training_only_influence"]
    layers = tuple(int(value) for value in contract["public_layers"])
    public = observations.loc[
        observations["layer"].isin(layers), ["station", "time", "layer", "temp"]
    ].copy()
    public = public.sort_values(["station", "layer", "time"], kind="stable")
    grouped = public.groupby(["station", "layer"], sort=False, observed=True)
    previous_time = grouped["time"].shift(1)
    elapsed_seconds = (public["time"] - previous_time).dt.total_seconds()
    difference = grouped["temp"].diff()
    public["temp_diff"] = difference.where(elapsed_seconds.eq(600.0))
    train_mask = _training_window_mask(public["time"], contract["registered_windows_kst"])

    stats: dict[int, tuple[float, float]] = {}
    receipts: dict[str, Any] = {}
    for layer in layers:
        values = public.loc[train_mask & public["layer"].eq(layer), "temp_diff"].dropna()
        if len(values) < 100:
            raise ContractError(f"insufficient exact-10-minute training diffs for layer {layer}")
        median = float(values.median())
        mad = float((values - median).abs().median())
        scale = max(1.4826 * mad, 0.01)
        stats[layer] = (median, scale)
        receipts[str(layer)] = {
            "training_exact10min_differences": int(len(values)),
            "median_signed_difference_C": median,
            "mad_C": mad,
            "robust_scale_C": scale,
        }
        mask = public["layer"].eq(layer) & public["temp_diff"].notna()
        public.loc[mask, "robust_abs_z"] = (
            (public.loc[mask, "temp_diff"] - median).abs() / scale
        )

    wide = public.pivot(index=["station", "time"], columns="layer", values="robust_abs_z")
    wide = wide.reindex(columns=list(layers))
    key = pd.MultiIndex.from_frame(blind[["station", "time"]])
    aligned = wide.reindex(key)
    available = aligned.notna().sum(axis=1).to_numpy(int)
    score = aligned.max(axis=1, skipna=True).to_numpy(float)
    score[available < int(contract["minimum_available_public_layers"])] = np.nan
    cutoff = float(contract["huber_cutoff_robust_z"])
    floor = float(contract["minimum_influence_weight"])
    weight = np.ones(len(blind), dtype=float)
    active = np.isfinite(score) & (score > cutoff)
    weight[active] = np.maximum(floor, cutoff / score[active])
    if not (np.isfinite(weight).all() and np.all((weight >= floor) & (weight <= 1.0))):
        raise ContractError("influence weights violate the sealed bounds")
    return score, weight, {
        "public_layers": list(layers),
        "per_layer_training_stats": receipts,
        "query_rows": int(len(blind)),
        "query_rows_minimum_layers_available": int(np.count_nonzero(np.isfinite(score))),
        "active_rows": int(active.sum()),
        "active_share": float(active.mean()),
        "weight_min": float(weight.min()),
        "weight_mean": float(weight.mean()),
        "rows_deleted": 0,
        "target_truth_used": False,
    }


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _metric_slice(
    truth: np.ndarray, reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    rows = int(mask.sum())
    if rows == 0:
        return {"rows": 0, "reference_rmse": None, "candidate_rmse": None, "delta_rmse": None}
    reference_rmse = _rmse(truth[mask], reference[mask])
    candidate_rmse = _rmse(truth[mask], candidate[mask])
    return {
        "rows": rows,
        "reference_rmse": reference_rmse,
        "candidate_rmse": candidate_rmse,
        "delta_rmse": candidate_rmse - reference_rmse,
    }


def _day_bootstrap(
    blind: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    replicates: int,
    fold: str | None = None,
) -> dict[str, Any]:
    selected = np.ones(len(blind), dtype=bool) if fold is None else blind["fold"].eq(fold).to_numpy()
    work = blind.loc[selected, ["fold", "kst_date"]].copy()
    work["truth"] = truth[selected]
    work["reference"] = reference[selected]
    work["candidate"] = candidate[selected]
    group_columns = ["fold", "kst_date"] if fold is None else ["kst_date"]
    pieces = list(work.groupby(group_columns, sort=True, observed=True))
    ref_sse = np.asarray([np.square(part.reference - part.truth).sum() for _, part in pieces])
    cand_sse = np.asarray([np.square(part.candidate - part.truth).sum() for _, part in pieces])
    counts = np.asarray([len(part) for _, part in pieces], dtype=float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draw = rng.integers(0, len(pieces), len(pieces))
        count = counts[draw].sum()
        deltas[index] = math.sqrt(cand_sse[draw].sum() / count) - math.sqrt(
            ref_sse[draw].sum() / count
        )
    return {
        "unit": "fold_x_kst_day" if fold is None else "kst_day",
        "groups": len(pieces),
        "replicates": replicates,
        "mean_delta_rmse": float(deltas.mean()),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def evaluate(
    blind: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    active = weight < 1.0
    pooled = _metric_slice(truth, reference, candidate, np.ones(len(blind), dtype=bool))
    by_fold = {
        fold: _metric_slice(truth, reference, candidate, blind["fold"].eq(fold).to_numpy())
        for fold in FOLD_ORDER
    }
    by_layer = {
        str(layer): _metric_slice(
            truth, reference, candidate, blind["layer"].eq(layer).to_numpy()
        )
        for layer in TARGET_LAYERS
    }
    local = blind["time"].dt.tz_convert("Asia/Seoul")
    month_id = local.dt.strftime("%Y-%m")
    by_month = {
        month: _metric_slice(truth, reference, candidate, month_id.eq(month).to_numpy())
        for month in sorted(month_id.unique())
    }
    slices = {
        "active_influence": _metric_slice(truth, reference, candidate, active),
        "inactive_exact_champion": _metric_slice(truth, reference, candidate, ~active),
    }
    evaluation = config["evaluation"]
    pooled_bootstrap = _day_bootstrap(
        blind,
        truth,
        reference,
        candidate,
        seed=int(evaluation["bootstrap_seed"]),
        replicates=int(evaluation["bootstrap_replicates"]),
    )
    official_like_bootstrap = _day_bootstrap(
        blind,
        truth,
        reference,
        candidate,
        seed=int(evaluation["bootstrap_seed"]) + 1,
        replicates=int(evaluation["bootstrap_replicates"]),
        fold=evaluation["official_like_fold"],
    )
    points_per = float(evaluation["points_per_rmse_C"])
    nominal_points = -float(pooled["delta_rmse"]) * points_per
    conservative_points = max(0.0, -official_like_bootstrap["ci90_high"] * points_per)
    transport_points = conservative_points - float(evaluation["transport_penalty_points"])
    active_delta = slices["active_influence"]["delta_rmse"]
    checks = {
        "pooled_improves": pooled["delta_rmse"] < evaluation["strict_gate"]["pooled_delta_rmse_lt"],
        "minimum_two_folds_improve": sum(item["delta_rmse"] < 0.0 for item in by_fold.values())
        >= int(evaluation["strict_gate"]["minimum_improved_folds"]),
        "official_like_fold_improves": by_fold[evaluation["official_like_fold"]]["delta_rmse"] < evaluation["strict_gate"]["official_like_delta_rmse_lt"],
        "all_layers_within_limit": max(item["delta_rmse"] for item in by_layer.values())
        <= float(evaluation["strict_gate"]["maximum_layer_delta_rmse_C"]),
        "pooled_ci90_high_below_zero": pooled_bootstrap["ci90_high"]
        < float(evaluation["strict_gate"]["pooled_bootstrap_ci90_high_lt"]),
        "active_slice_improves": active_delta is not None
        and active_delta < float(evaluation["strict_gate"]["active_slice_delta_rmse_lt"]),
        "minimum_active_rows": int(active.sum())
        >= int(evaluation["strict_gate"]["minimum_active_rows"]),
        "transport_adjusted_points_gte": transport_points
        >= float(evaluation["strict_gate"]["transport_adjusted_points_gte"]),
    }
    concentration = {
        "by_fold": {
            fold: int((active & blind["fold"].eq(fold).to_numpy()).sum())
            for fold in FOLD_ORDER
        },
        "by_layer": {
            str(layer): int((active & blind["layer"].eq(layer).to_numpy()).sum())
            for layer in TARGET_LAYERS
        },
        "by_month": {
            month: int((active & month_id.eq(month).to_numpy()).sum())
            for month in sorted(month_id.unique())
        },
    }
    return {
        "name": config["candidate"]["name"],
        "reference_rmse": pooled["reference_rmse"],
        "candidate_rmse": pooled["candidate_rmse"],
        "delta_rmse": pooled["delta_rmse"],
        "by_fold": by_fold,
        "by_layer": by_layer,
        "by_month": by_month,
        "worst_and_identity_slices": slices,
        "bootstrap": pooled_bootstrap,
        "official_like_bootstrap": official_like_bootstrap,
        "nominal_expected_points_delta": nominal_points,
        "conservative_official_like_points_delta": conservative_points,
        "transport_adjusted_points_delta": transport_points,
        "nominal_champion_if_delta_transported": {
            "rmse_C": float(evaluation["current_champion_rmse_C"]) - nominal_points / points_per,
            "points": float(evaluation["current_champion_points"]) + nominal_points,
        },
        "active_rows": int(active.sum()),
        "active_share": float(active.mean()),
        "finite_anomaly_score_rows": int(np.isfinite(score).sum()),
        "anomaly_score_active_median": float(np.median(score[active])) if active.any() else None,
        "influence_weight_min": float(weight.min()),
        "change_rms_C": float(np.sqrt(np.mean(np.square(candidate - reference)))),
        "change_abs_max_C": float(np.max(np.abs(candidate - reference))),
        "active_concentration": concentration,
        "gate_checks": checks,
        "strict_exploratory_pass": bool(all(checks.values())),
    }


def v7_priority_audit() -> dict[str, Any]:
    path = REPORT.parent / "p2_domain_invariant_vertical_curvature_20260901_v9r1" / "v7-readiness-audit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source_path": str(path.relative_to(ROOT)),
        "source_sha256": sha256_file(path),
        "status": value["status"],
        "candidate": value["candidate"],
        "attested_csv_sha256": value["attested_csv_sha256"],
        "transport_calibrated_expected_points_delta": value["transport_calibrated_expected_points_delta"],
        "priority_before_v10_result": "ONLY_REPORT_ATTESTED_READY_INFORMATION_PROBE_CHAMPION_REMAINS_DEFAULT",
        "pack_values_or_hash_reread": False,
        "materialized": False,
        "uploaded": False,
    }


def write_report(result: dict[str, Any]) -> None:
    item = result["candidate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    fold_text = ", ".join(
        f"{key} {value['delta_rmse']:+.9f}" for key, value in item["by_fold"].items()
    )
    layer_text = ", ".join(
        f"L{key} {value['delta_rmse']:+.9f}" for key, value in item["by_layer"].items()
    )
    lines = [
        "# P2 public-sensor influence sensitivity cycle 20260901 v10",
        "",
        "## 결론",
        "",
        f"상태: `{result['status']}`. pooled ΔRMSE {item['delta_rmse']:+.9f}°C, 명목 {item['nominal_expected_points_delta']:+.6f}점, transport-adjusted {item['transport_adjusted_points_delta']:+.6f}점이다.",
        f"fold: {fold_text}. layer: {layer_text}. active {item['active_rows']:,}/{result['rows']:,}행.",
        "세 fold는 모두 이미 노출된 exploratory surface이므로 fresh confirmation이나 공식 개선 보장이 아니다.",
        "",
        "## v9r1 폐쇄",
        "",
        "v9r1 두 후보는 각각 1/3 fold만 개선했고 모두 L2를 악화했다. 따라서 normalized vertical-curvature Ridge family는 `EXPLORATORY_NO_GO_BOTH_SEALED_CANDIDATES`로 닫는다.",
        "",
        "## v10 과학 축",
        "",
        "공개층 온도의 exact-10-minute first difference만 사용해 training-only layer median/MAD robust z를 계산했다. 6을 넘는 행에서만 endpoint baseline 대비 챔피언 보정량을 Huber weight로 감쇠했고 weight는 0.5 아래로 내리지 않았다.",
        "행 삭제와 model fit은 0이다. 정상 행은 bit-exact 챔피언이고 action NPZ/hash를 target metric 전에 봉인했다. learned soft/benefit gate, raw residual, curvature, heave, copula, GP, CatBoost, PAVA와 의미적으로 다르다.",
        "",
        "## v7 우선순위",
        "",
        f"기존 v7 pack은 값/해시를 다시 읽거나 materialize/upload하지 않았다. 원장상 `{result['v7_readiness']['status']}`, transport 보정 +{result['v7_readiness']['transport_calibrated_expected_points_delta']:.6f}점의 유일한 ready information probe지만 챔피언 보존이 기본이다.",
        "",
        "## 접근·재현 경계",
        "",
        f"fits=0, observations rows={result['operation_counters']['observations_rows_read']:,}, historical rows={result['operation_counters']['historical_truth_free_scoring_rows_read']:,}; official/test/sample/baseline/score/query/hidden=0, CSV=0, upload=0.",
    ]
    (REPORT / "report-source.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    started = time.perf_counter()
    if ARTIFACT.exists():
        raise FileExistsError(f"exactly-once artifact already exists: {ARTIFACT}")
    config = load_config()
    audit = semantic_audit(config)
    ARTIFACT.mkdir(parents=True)
    atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(CONFIG),
            "runner_sha256": sha256_file(RUNNER),
            "semantic_fingerprint": config["semantic_fingerprint"],
            "semantic_audit_sha256": sha256_json(audit),
            "candidate": config["candidate"]["name"],
            "candidate_count": 1,
            "fit_count": 0,
            "row_deletion": False,
            "result_adaptive_tuning": False,
            "official_access_before_lock": 0,
        },
    )
    observations_path = resolve_observations(config)
    scoring_path = ROOT / config["source_contract"]["scoring_frame"]
    if sha256_file(scoring_path) != config["source_contract"]["scoring_frame_sha256"]:
        raise ContractError("truth-free scoring frame hash drift")
    observations = pd.read_csv(
        observations_path, dtype={"station": "string", "time": "string"}
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    if observations.duplicated(["time", "layer"]).any():
        raise ContractError("observation keys are duplicated")
    scored = pd.read_parquet(scoring_path)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    if tuple(sorted(scored["fold"].unique())) != tuple(sorted(FOLD_ORDER)):
        raise ContractError("historical fold set drift")
    blind, reference = metric_engine.make_reference(observations, scored)

    feature_table = build_training_features(observations)
    design = build_normalized_curvature_design(feature_table.frame)
    design_index = pd.MultiIndex.from_arrays(
        [metric_engine.canonical_time_ns(design.keys["time"]), design.keys["layer"]]
    )
    query_index = pd.MultiIndex.from_arrays(
        [metric_engine.canonical_time_ns(blind["time"]), blind["layer"]]
    )
    positions = design_index.get_indexer(query_index)
    if design_index.has_duplicates or np.any(positions < 0):
        raise ContractError("historical feature alignment failed")
    endpoint_baseline = design.baseline[positions]
    score, weight, influence_receipt = build_public_influence(observations, blind, config)
    candidate = endpoint_baseline + weight * (reference - endpoint_baseline)
    inactive = weight == 1.0
    if not np.array_equal(candidate[inactive], reference[inactive]):
        raise ContractError("inactive rows are not bit-exact champion")
    if not np.isfinite(candidate).all():
        raise ContractError("candidate is nonfinite")

    action_path = ARTIFACT / f"{config['candidate']['name']}.npz"
    np.savez_compressed(
        action_path,
        time_ns=metric_engine.canonical_time_ns(blind["time"]),
        layer=blind["layer"].to_numpy(np.int16),
        fold=blind["fold"].to_numpy(str),
        reference=reference,
        endpoint_baseline=endpoint_baseline,
        anomaly_score=score,
        influence_weight=weight,
        candidate=candidate,
    )
    commitment = {
        "path": str(action_path),
        "sha256": sha256_file(action_path),
        "rows": int(len(candidate)),
        "action_formula": config["candidate"]["formula"],
        "influence_receipt": influence_receipt,
        "metric_computed_at_commitment": False,
        "target_truth_used_to_choose_action": False,
    }
    atomic_json(ARTIFACT / "action_commitment.json", commitment)

    # Only after the candidate vector and its hash are sealed is target truth used.
    truth = design.truth[positions]
    metric = evaluate(blind, truth, reference, candidate, score, weight, config)
    metric["action_commitment"] = commitment
    v7 = v7_priority_audit()
    status = (
        "EXPLORATORY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if metric["strict_exploratory_pass"]
        else "EXPLORATORY_NO_GO_PUBLIC_SENSOR_INFLUENCE_SHRINK"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_level": config["claim_level"],
        "runtime_seconds": time.perf_counter() - started,
        "rows": int(len(blind)),
        "fit_count": 0,
        "candidate_count": 1,
        "semantic_audit": audit,
        "influence_receipt": influence_receipt,
        "candidate": metric,
        "v7_readiness": v7,
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "historical_truth_free_scoring_rows_read": int(len(scored)),
            "official_test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_file_rows_read": 0,
            "score_file_rows_read": 0,
            "query_support_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "existing_v7_submission_csv_value_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config": sha256_file(CONFIG),
            "runner": sha256_file(RUNNER),
            "observations": sha256_file(observations_path),
            "scoring_frame": sha256_file(scoring_path),
            "action_npz": commitment["sha256"],
            "metric_engine": sha256_file(
                ROOT / "scripts" / "run_p2_group_balanced_raw_residual_20260901_v8.py"
            ),
        },
    }
    atomic_json(ARTIFACT / "result.json", result)
    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT / "result.json", result)
    atomic_json(REPORT / "semantic-audit.json", audit)
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
