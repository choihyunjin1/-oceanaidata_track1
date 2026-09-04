"""One-line ndarray mutability repair for the zero-fit P2 v11 experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_supported_layer_change_coherence_20260901_v11 as predecessor  # noqa: E402

engine = predecessor.engine
EXPERIMENT_ID = "p2_supported_layer_change_coherence_20260901_v11r1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)

engine.EXPERIMENT_ID = EXPERIMENT_ID
engine.CONFIG = CONFIG
engine.ARTIFACT = ARTIFACT
engine.REPORT = REPORT
engine.RUNNER = RUNNER
engine.SCHEMA_VERSION = "p2.supported_layer_change_coherence.result.20260901.v11r1"
predecessor.REPORT = REPORT


def load_config() -> dict[str, Any]:
    recovery = json.loads(CONFIG.read_text(encoding="utf-8"))
    predecessor_path = ROOT / recovery["predecessor"]["config_path"]
    if engine.sha256_file(predecessor_path) != recovery["predecessor"]["config_sha256"]:
        raise engine.ContractError("predecessor config hash drift")
    config = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if config["experiment_id"] != recovery["predecessor"]["experiment_id"]:
        raise engine.ContractError("predecessor experiment drift")
    forbidden_changes = (
        "scientific_contract_changed",
        "candidate_changed",
        "support_changed",
        "folds_changed",
        "huber_changed",
        "gate_changed",
        "operation_counters_changed",
    )
    if any(recovery[key] for key in forbidden_changes):
        raise engine.ContractError("technical recovery changed scientific contract")
    config = copy.deepcopy(config)
    config["experiment_id"] = EXPERIMENT_ID
    config["status"] = "PREREGISTERED_EXPLORATORY_NOT_EXECUTED"
    config["contract_repair"] = recovery
    return config


def build_supported_layer_coherence_writeable(
    observations: pd.DataFrame, blind: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Bit-equivalent v11 action with an explicit writeable score buffer."""

    contract = config["training_only_influence"]
    layers = tuple(int(value) for value in contract["public_layers"])
    if layers != (1, 5, 6, 7):
        raise engine.ContractError("supported public-layer set drift")
    public = observations.loc[
        observations["layer"].isin(layers), ["station", "time", "layer", "temp"]
    ].copy()
    public = public.sort_values(["station", "layer", "time"], kind="stable")
    grouped = public.groupby(["station", "layer"], sort=False, observed=True)
    previous_time = grouped["time"].shift(1)
    elapsed_seconds = (public["time"] - previous_time).dt.total_seconds()
    public["temp_diff"] = grouped["temp"].diff().where(elapsed_seconds.eq(600.0))
    train_mask = engine._training_window_mask(
        public["time"], contract["registered_windows_kst"]
    )
    receipts: dict[str, Any] = {}
    for layer in layers:
        values = public.loc[
            train_mask & public["layer"].eq(layer), "temp_diff"
        ].dropna()
        expected = int(contract["support_receipt_exact10min_counts"][str(layer)])
        if len(values) != expected:
            raise engine.ContractError(
                f"supported layer {layer} count drift: {len(values)} != {expected}"
            )
        median = float(values.median())
        mad = float((values - median).abs().median())
        scale = max(1.4826 * mad, 0.01)
        mask = public["layer"].eq(layer) & public["temp_diff"].notna()
        public.loc[mask, "standardized_signed_change"] = (
            public.loc[mask, "temp_diff"] - median
        ) / scale
        receipts[str(layer)] = {
            "training_exact10min_differences": int(len(values)),
            "median_signed_difference_C": median,
            "mad_C": mad,
            "robust_scale_C": scale,
        }
    wide = public.pivot(
        index=["station", "time"], columns="layer", values="standardized_signed_change"
    ).reindex(columns=list(layers))
    key = pd.MultiIndex.from_frame(blind[["station", "time"]])
    aligned = wide.reindex(key)
    available = aligned.notna().sum(axis=1).to_numpy(int)
    cross_layer_median = aligned.median(axis=1, skipna=True)
    deviation = aligned.sub(cross_layer_median, axis=0).abs()
    # The only v11 -> v11r1 executable change: request an explicit writeable copy.
    score = deviation.max(axis=1, skipna=True).to_numpy(dtype=float, copy=True)
    if not score.flags.writeable:
        raise engine.ContractError("recovery score buffer is not writeable")
    score[available < int(contract["minimum_available_public_layers"])] = np.nan
    cutoff = float(contract["huber_cutoff_coherence_score"])
    floor = float(contract["minimum_influence_weight"])
    weight = np.ones(len(blind), dtype=float)
    active = np.isfinite(score) & (score > cutoff)
    weight[active] = np.maximum(floor, cutoff / score[active])
    if not (np.isfinite(weight).all() and np.all((weight >= floor) & (weight <= 1.0))):
        raise engine.ContractError("coherence influence weights violate bounds")
    return score, weight, {
        "signal": contract["signal"],
        "public_layers": list(layers),
        "minimum_available_public_layers": int(contract["minimum_available_public_layers"]),
        "per_layer_training_stats": receipts,
        "query_rows": int(len(blind)),
        "query_rows_minimum_layers_available": int(np.count_nonzero(np.isfinite(score))),
        "active_rows": int(active.sum()),
        "active_share": float(active.mean()),
        "weight_min": float(weight.min()),
        "weight_mean": float(weight.mean()),
        "rows_deleted": 0,
        "target_truth_used": False,
        "technical_recovery": "explicit_to_numpy_copy_true",
    }


def preflight() -> dict[str, Any]:
    config = load_config()
    audit = engine.semantic_audit(config)
    recovery = config["contract_repair"]
    if config["candidate"]["name"] != "P2_V11_HUBER6_SUPPORTED_LAYER_CHANGE_COHERENCE":
        raise engine.ContractError("candidate changed")
    test_frame = pd.DataFrame({"x": [1.0, 2.0]})
    buffer = test_frame.max(axis=1).to_numpy(dtype=float, copy=True)
    buffer[0] = np.nan
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "predecessor_config_sha256": recovery["predecessor"]["config_sha256"],
        "predecessor_runner_sha256": recovery["predecessor"]["runner_sha256"],
        "only_change": recovery["only_change"],
        "scientific_contract_changed": False,
        "candidate": config["candidate"]["name"],
        "semantic_fingerprint": config["semantic_fingerprint"],
        "semantic_audit_sha256": engine.sha256_json(audit),
        "writeable_buffer_contract": bool(buffer.flags.writeable and np.isnan(buffer[0])),
        "config_sha256": engine.sha256_file(CONFIG),
        "runner_sha256": engine.sha256_file(RUNNER),
        "data_rows_read": 0,
        "model_fits": 0,
        "artifacts_written": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = engine.sha256_json(payload)
    return payload


def write_report(result: dict[str, Any]) -> None:
    predecessor.write_report(result)
    path = REPORT / "report-source.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("20260901 v11", "20260901 v11r1", 1)
    text += (
        "\n## 기술 recovery\n\n"
        "v11은 fit/prediction/metric 0에서 read-only ndarray masking으로 종료됐다. "
        "v11r1은 `.to_numpy(dtype=float, copy=True)`만 변경했고 candidate, support, "
        "fold, Huber, gate, operation counters는 predecessor config hash로 고정했다.\n"
    )
    path.write_text(text, encoding="utf-8")


engine.load_config = load_config
engine.build_public_influence = build_supported_layer_coherence_writeable
engine.write_report = write_report


def run() -> dict[str, Any]:
    result = engine.run()
    recovery = load_config()["contract_repair"]
    receipt = {
        "predecessor": recovery["predecessor"],
        "only_change": recovery["only_change"],
        "scientific_contract_changed": False,
        "predecessor_model_fits": 0,
        "predecessor_predictions": 0,
        "predecessor_metrics": 0,
        "official_access": 0,
    }
    engine.atomic_json(ARTIFACT / "technical-recovery.json", receipt)
    engine.atomic_json(REPORT / "technical-recovery.json", receipt)
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
