"""Exactly-once historical validation of an I-ORS E150 microfragment veto."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as e150_source  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v13 as truth_source  # noqa: E402

EXPERIMENT_ID = "p1_iors_e150_microfragment_veto_20260901_v34a"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
KEYS = ["station", "year", "layer", "time"]
FOLDS = ["2025_q2", "2025_q3", "2025_q4"]


class ContractError(RuntimeError):
    """Raised when a frozen v34a input or rule changes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def resolve_source(spec: dict[str, str]) -> Path:
    path = (ROOT / spec["path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise ContractError(f"historical source unavailable or changed: {path}")
    return path


def load_contract() -> tuple[dict[str, Any], dict[str, Path]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    decoder = config["decoder"]
    checks = {
        "schema": config["schema_version"] == "p1.iors_e150_microfragment_veto.20260901.v34a",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "station": decoder["station"] == "I-ORS",
        "length": decoder["maximum_segment_length_inclusive"] == 2,
        "continuity": decoder["continuity_minutes"] == 10,
        "fit0": decoder["model_fits"] == 0,
        "tuning0": decoder["threshold_tuning"] == 0,
        "folds": config["validation"]["diagnostic_folds"] == FOLDS,
        "primary": config["validation"]["primary_outer_folds"] == FOLDS[1:],
        "official0": config["authorization"]["official_reads"] == 0,
        "hidden0": config["authorization"]["hidden_truth_reads"] == 0,
        "csv0": config["authorization"]["submission_csv_created"] == 0,
        "upload0": config["authorization"]["uploads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v34a contract failed: {checks}")
    paths = {
        name: resolve_source(config["historical_sources"][name])
        for name in ["anchor", "truth", *FOLDS]
    }
    return config, paths


def microfragment_mask(
    frame: pd.DataFrame,
    incumbent: np.ndarray,
    raw_e150: np.ndarray,
    *,
    station: str = "I-ORS",
    maximum_length: int = 2,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return rows in short contiguous E150-only segments, without labels."""

    addition = (incumbent == 0) & (raw_e150 == 1)
    eligible = addition & frame["station"].astype(str).eq(station).to_numpy()
    parsed_time = pd.to_datetime(frame["time"], utc=True)
    same_previous = (
        frame["fold"].astype(str).eq(frame["fold"].astype(str).shift()).to_numpy()
        & frame["station"].astype(str).eq(frame["station"].astype(str).shift()).to_numpy()
        & frame["layer"].eq(frame["layer"].shift()).to_numpy()
        & parsed_time.diff().eq(pd.Timedelta(minutes=10)).to_numpy()
    )
    continuation = eligible & pd.Series(eligible).shift(fill_value=False).to_numpy() & same_previous
    segment_id = np.cumsum(eligible & ~continuation)
    removal = np.zeros(len(frame), dtype=bool)
    inventory: list[dict[str, Any]] = []
    for value in np.unique(segment_id[eligible]):
        rows = np.flatnonzero(eligible & (segment_id == value))
        selected = len(rows) <= maximum_length
        if selected:
            removal[rows] = True
        inventory.append(
            {
                "fold": str(frame.loc[int(rows[0]), "fold"]),
                "station": str(frame.loc[int(rows[0]), "station"]),
                "layer": int(frame.loc[int(rows[0]), "layer"]),
                "length": int(len(rows)),
                "selected": bool(selected),
            }
        )
    return removal, inventory


def build_truth_blind_action(
    anchor_frame: pd.DataFrame, paths: dict[str, Path], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    incumbent = anchor_frame["current_router_prediction"].to_numpy(np.int8)
    raw_e150 = np.empty(len(anchor_frame), dtype=np.int8)
    receipts: list[dict[str, Any]] = []
    for fold in FOLDS:
        mask = anchor_frame["fold"].eq(fold).to_numpy()
        raw = np.asarray(e150_source._select_archive_arrays(fold, paths[fold])["candidate"], dtype=np.int8)
        if raw.shape != (int(mask.sum()),) or not np.isin(raw, [0, 1]).all():
            raise ContractError(f"raw E150 alignment failed: {fold}")
        raw_e150[mask] = raw
        receipts.append({"fold": fold, "rows": int(mask.sum()), "truth_reads_before_seal": 0})
    if not np.all(raw_e150[incumbent == 1] == 1):
        raise ContractError("raw E150 does not preserve incumbent positives")
    removal, inventory = microfragment_mask(
        anchor_frame,
        incumbent,
        raw_e150,
        station=config["decoder"]["station"],
        maximum_length=int(config["decoder"]["maximum_segment_length_inclusive"]),
    )
    candidate = raw_e150.copy()
    candidate[removal] = 0
    if np.any(candidate[incumbent == 1] == 0) or np.any(candidate > raw_e150):
        raise ContractError("candidate is not an incumbent-preserving veto")
    return raw_e150, candidate, removal, [*receipts, {"segment_inventory": inventory}]


def metric_block(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    removal: np.ndarray,
) -> dict[str, Any]:
    removed = mask & removal
    reference_f1 = float(f1_score(truth[mask], reference[mask]))
    candidate_f1 = float(f1_score(truth[mask], candidate[mask]))
    tp = int((removed & (truth == 1)).sum())
    fp = int((removed & (truth == 0)).sum())
    return {
        "rows": int(mask.sum()),
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": candidate_f1 - reference_f1,
        "removed_rows": int(removed.sum()),
        "removed_true_positive": tp,
        "removed_false_positive": fp,
        "removed_true_positive_share": tp / int(removed.sum()) if removed.any() else None,
    }


def official_geometry(config: dict[str, Any]) -> dict[str, Any]:
    evidence = config["official_aggregate_evidence"]
    before = float(evidence["champion_public_f1"])
    after = float(evidence["v33a_public_f1"])
    positives = int(evidence["champion_positive_rows"])
    removed = int(evidence["v33a_changed_rows"])
    minimum_truth_positives = before * positives / (2.0 - before)
    maximum_possible_drop = removed * (2.0 - before) / (
        minimum_truth_positives + positives - removed
    )
    tolerance = 0.5e-6
    feasible_tp: list[int] = []
    for truth_positives in range(1, 169012):
        lower_tp = max(0, int(np.ceil((before - tolerance) * (truth_positives + positives) / 2.0)))
        upper_tp = min(
            truth_positives,
            positives,
            int(np.floor((before + tolerance) * (truth_positives + positives) / 2.0)),
        )
        for tp in range(lower_tp, upper_tp + 1):
            if not (before - tolerance <= 2 * tp / (truth_positives + positives) < before + tolerance):
                continue
            for removed_tp in range(removed + 1):
                candidate_f1 = 2 * (tp - removed_tp) / (truth_positives + positives - removed)
                if after - tolerance <= candidate_f1 < after + tolerance:
                    feasible_tp.append(removed_tp)
    return {
        "assumed_metric": "pooled binary micro-F1",
        "observed_drop": before - after,
        "maximum_possible_drop_if_all_80_removed_rows_are_true_positive": maximum_possible_drop,
        "observed_to_maximum_ratio": (before - after) / maximum_possible_drop,
        "feasible_removed_true_positive_counts_at_six_decimal_rounding": sorted(set(feasible_tp)),
        "metric_geometry_consistent": bool(feasible_tp),
        "interpretation": "Aggregate v33a score and 80-row 1->0 diff cannot jointly arise from pooled binary micro-F1; no row-level I-ORS TP inversion is identifiable until reconciled.",
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or (REPORT / "result.json").exists():
        raise FileExistsError("v34a exactly-once namespace already exists")
    started = time.perf_counter()
    config, paths = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json_new(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "fits": 0,
            "retries": 0,
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
    )
    anchor = pd.read_parquet(paths["anchor"], columns=[*KEYS, "fold", "current_router_prediction"])
    raw, candidate, removal, receipts = build_truth_blind_action(anchor, paths, config)
    np.savez_compressed(ARTIFACT / "sealed_action.npz", raw_e150=raw, candidate=candidate, removal=removal)
    seal = {
        "experiment_id": EXPERIMENT_ID,
        "rule": config["decoder"],
        "rows": len(candidate),
        "removals": int(removal.sum()),
        "raw_sha256": sha256_array(raw),
        "candidate_sha256": sha256_array(candidate),
        "removal_sha256": sha256_array(removal.astype(np.uint8)),
        "sealed_npz_sha256": sha256_file(ARTIFACT / "sealed_action.npz"),
        "receipts": receipts,
        "truth_reads_before_action_seal": 0,
        "official_reads": 0,
    }
    write_json_new(ARTIFACT / "action-seal.json", seal)
    historical, candidate = truth_source.attach_truth(anchor, candidate)
    truth = historical["label_base"].to_numpy(np.int8)
    by_fold = {
        fold: metric_block(truth, raw, candidate, historical["fold"].eq(fold).to_numpy(), removal)
        for fold in FOLDS
    }
    primary = historical["fold"].isin(FOLDS[1:]).to_numpy()
    pooled = metric_block(truth, raw, candidate, primary, removal)
    bootstrap = truth_source.base.day_bootstrap(historical, raw, candidate, config)
    removed_tp_share = pooled["removed_true_positive_share"]
    gates = {
        "primary_delta_f1_strictly_positive": pooled["delta_f1"] > 0.0,
        "q3_delta_f1_nonnegative": by_fold["2025_q3"]["delta_f1"] >= 0.0,
        "q4_delta_f1_nonnegative": by_fold["2025_q4"]["delta_f1"] >= 0.0,
        "primary_day_block_ci90_low_nonnegative": bootstrap["ci90_low"] >= 0.0,
        "removed_true_positive_share_strictly_below_reference_f1_half": removed_tp_share is not None and removed_tp_share < pooled["reference_f1"] / 2.0,
        "anchor_removals_zero": not np.any(removal & (anchor["current_router_prediction"].to_numpy(np.int8) == 1)),
        "primary_removed_rows_at_least": pooled["removed_rows"] >= 1,
    }
    geometry = official_geometry(config)
    internal_pass = all(gates.values())
    status = "INTERNAL_PASS_OFFICIAL_GEOMETRY_BLOCKED" if internal_pass else "TERMINAL_NO_GO"
    result = {
        "schema_version": "p1.iors_e150_microfragment_veto.result.v34a",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 0,
        "by_fold": by_fold,
        "pooled_primary_q3_q4": pooled,
        "day_block_bootstrap": bootstrap,
        "expected_points_delta": pooled["delta_f1"] * float(config["score"]["points_per_f1"]),
        "gates": gates,
        "internal_pass": internal_pass,
        "official_metric_geometry": geometry,
        "official_materializer_execute_allowed": internal_pass and geometry["metric_geometry_consistent"],
        "operations": {"official_reads": 0, "hidden_truth_reads": 0, "test_reads": 0, "sample_reads": 0, "submission_csv_created": 0, "uploads": 0, "retries": 0},
        "hashes": {
            "config": sha256_file(CONFIG_PATH),
            "runner": sha256_file(Path(__file__)),
            "lock": sha256_file(ARTIFACT / "attempt_lock.json"),
            "action_seal": sha256_file(ARTIFACT / "action-seal.json"),
            "sealed_action": sha256_file(ARTIFACT / "sealed_action.npz"),
        },
    }
    write_json_new(ARTIFACT / "result.json", result)
    write_json_new(REPORT / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-metric-only", action="store_true")
    args = parser.parse_args()
    if not args.execute_metric_only:
        raise SystemExit("--execute-metric-only required")
    try:
        result = execute()
    except Exception as exc:
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            write_json_new(ARTIFACT / "terminal_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
