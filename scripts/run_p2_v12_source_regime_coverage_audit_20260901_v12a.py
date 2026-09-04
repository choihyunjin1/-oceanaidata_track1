"""Zero-fit public-covariate and calendar support audit for P2 v12."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import build_normalized_curvature_design  # noqa: E402

EXPERIMENT_ID = "p2_v12_source_regime_coverage_audit_20260901_v12a"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
REPORT = ROOT / "reports" / EXPERIMENT_ID
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
RUNNER = Path(__file__)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID or config["operation_limits"]["model_fits"] != 0:
        raise v12.ContractError("audit contract drift")
    return config


def descriptor(tokens: np.ndarray, mask: np.ndarray, context: np.ndarray) -> np.ndarray:
    continuous = tokens[:, :, :4].astype(float)
    usable = mask.astype(bool)
    count = usable.sum(axis=1, keepdims=True)
    mean = np.divide(
        (continuous * usable[:, :, None]).sum(axis=1),
        count,
        out=np.zeros((len(tokens), 4), dtype=float),
        where=count > 0,
    )
    maximum = np.where(usable[:, :, None], continuous, -np.inf).max(axis=1)
    maximum[~np.isfinite(maximum)] = 0.0
    presence = tokens[:, :, 4:8].mean(axis=1)
    result = np.column_stack((mean, maximum, presence, context)).astype(float)
    if result.shape[1] != 23 or not np.isfinite(result).all():
        raise v12.ContractError("coverage descriptor contract failed")
    return result


def robust_distance_receipt(train: np.ndarray, query: np.ndarray, floor: float) -> dict[str, float]:
    center = np.median(train, axis=0)
    scale = np.maximum(np.median(np.abs(train - center), axis=0) * 1.4826, floor)
    standardized = np.clip((query - center) / scale, -50.0, 50.0)
    distance = np.sqrt(np.mean(np.square(standardized), axis=1))
    lower = np.quantile(train, 0.01, axis=0)
    upper = np.quantile(train, 0.99, axis=0)
    outside = np.any((query < lower) | (query > upper), axis=1)
    return {
        "rows": int(len(query)),
        "robust_distance_p50": float(np.quantile(distance, 0.50)),
        "robust_distance_p90": float(np.quantile(distance, 0.90)),
        "robust_distance_p99": float(np.quantile(distance, 0.99)),
        "outside_training_1_99_envelope_share": float(outside.mean()),
    }


def run() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError(ARTIFACT)
    config = load_config()
    ARTIFACT.mkdir(parents=True)
    v12.atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": v12.sha256_file(CONFIG),
            "runner_sha256": v12.sha256_file(RUNNER),
            "model_fits": 0,
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
    table = build_training_features(observations)
    design = build_normalized_curvature_design(table.frame)
    tokens, token_mask, context = v12.build_arrays(table.frame)
    values = descriptor(tokens, token_mask, context)
    local = design.keys["time"].dt.tz_convert("Asia/Seoul")
    training_id = v12.registered_window_ids(local, config["v12_training_windows_kst"])
    training = training_id != ""
    training_months = sorted(set(local[training].dt.month.astype(int)))
    design_index = pd.MultiIndex.from_arrays(
        [v12.metric_engine.canonical_time_ns(design.keys["time"]), design.keys["layer"]]
    )
    query_index = pd.MultiIndex.from_arrays(
        [v12.metric_engine.canonical_time_ns(scored["time"]), scored["layer"]]
    )
    positions = design_index.get_indexer(query_index)
    if np.any(positions < 0):
        raise v12.ContractError("coverage query alignment failed")
    fold_receipts: dict[str, Any] = {}
    absent_count = 0
    prefix_support_count = 0
    for fold in v12.metric_engine.FOLD_ORDER:
        selected = scored["fold"].eq(fold).to_numpy()
        fold_months = sorted(set(scored.loc[selected, "time"].dt.tz_convert("Asia/Seoul").dt.month))
        absent = sorted(set(fold_months).difference(training_months))
        if absent:
            absent_count += 1
        start = pd.Timestamp(config["outer_fold_starts_kst"][fold]).tz_convert("Asia/Seoul")
        cutoff = start - pd.Timedelta(days=int(config["prefix_embargo_days"]))
        prefix = (local >= pd.Timestamp("2024-05-01T00:00:00+09:00")) & (local < cutoff)
        same_month = prefix & local.dt.month.isin(fold_months).to_numpy()
        same_month_rows = int(same_month.sum())
        if same_month_rows > 0:
            prefix_support_count += 1
        fold_receipts[fold] = {
            "calendar_months": [int(value) for value in fold_months],
            "months_absent_from_v12_training": [int(value) for value in absent],
            "prefix_cutoff_kst": cutoff.isoformat(),
            "prefix_rows": int(prefix.sum()),
            "prior_same_calendar_month_rows": same_month_rows,
            "current_v12_training_distance": robust_distance_receipt(
                values[training], values[positions[selected]], float(config["robust_scale_floor"])
            ),
        }
    gate = {
        "minimum_two_folds_have_absent_training_months": absent_count
        >= int(
            config["authorization_gate"][
                "minimum_outer_folds_with_calendar_months_absent_from_v12_training"
            ]
        ),
        "minimum_two_folds_have_prior_same_month_support": prefix_support_count
        >= int(
            config["authorization_gate"][
                "minimum_outer_folds_with_prior_same_calendar_month_support"
            ]
        ),
    }
    result = {
        "schema_version": "p2.v12_source_regime_coverage_audit.result.20260901.v12a",
        "experiment_id": EXPERIMENT_ID,
        "status": "DOMAIN_BALANCED_PREFIX_SAFE_V13_AUTHORIZED" if all(gate.values()) else "NO_V13_SUPPORT",
        "model_fits": 0,
        "v12_training_rows": int(training.sum()),
        "v12_training_calendar_months": training_months,
        "folds_with_absent_training_months": absent_count,
        "folds_with_prior_same_month_support": prefix_support_count,
        "folds": fold_receipts,
        "authorization_gate": gate,
        "prohibitions": {
            "posthoc_nov_dec_router": True,
            "gate_lowering": True,
            "threshold_search": True,
        },
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "historical_scoring_rows_read": int(len(scored)),
            "model_fits": 0,
            "official_rows_read": 0,
            "hidden_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config": v12.sha256_file(CONFIG),
            "runner": v12.sha256_file(RUNNER),
            "observations": v12.sha256_file(observations_path),
            "scoring_frame": v12.sha256_file(scoring_path),
        },
    }
    v12.atomic_json(ARTIFACT / "result.json", result)
    REPORT.mkdir(parents=True, exist_ok=True)
    v12.atomic_json(REPORT / "result.json", result)
    (REPORT / "report-source.md").write_text(
        "# P2 v12 source-regime coverage audit\n\n"
        f"## 결론\n\n상태: `{result['status']}`. model fit=0. v12 training months="
        f"`{training_months}`, absent-month folds=`{absent_count}`, prefix same-month support folds="
        f"`{prefix_support_count}`. 이 감사로 gate를 낮추거나 Nov-Dec를 posthoc routing하지 않는다.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Use --execute")
    print(json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
