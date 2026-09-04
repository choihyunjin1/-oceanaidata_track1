"""Exploratory P1 anchor false-negative oracle audit for v25 design only.

This script reads registered historical OOF surfaces only.  It does not create
a candidate, threshold, attempt lock, official prediction, or submission file.
"""

from __future__ import annotations

import hashlib
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

import run_p1_public_transport_repair_cycle_20260831_v16 as source  # noqa: E402

EXPERIMENT_ID = "p1_anchor_false_negative_oracle_audit_20260831_v25"
REPORT = ROOT / "reports" / EXPERIMENT_ID
CORRECTED_RESULT = REPORT / "result.corrected.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "-Infinity" if value < 0 else "Infinity"
    if pd.isna(value):
        return None
    return value


def summary_table(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(columns, sort=True, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        eligible = group["anchor"] == 0
        positive = group["truth"] == 1
        fn = eligible & positive
        eligible_rows = int(eligible.sum())
        positive_rows = int(positive.sum())
        fn_rows = int(fn.sum())
        record = {column: native(key) for column, key in zip(columns, keys, strict=True)}
        record.update(
            {
                "anchor_negative_rows": eligible_rows,
                "false_negative_rows": fn_rows,
                "marginal_precision_if_all_anchor_negative_selected": (
                    fn_rows / eligible_rows if eligible_rows else None
                ),
                "truth_positive_rows": positive_rows,
                "anchor_false_negative_rate": fn_rows / positive_rows if positive_rows else None,
            }
        )
        rows.append(record)
    return rows


def attach_gap_minutes(frame: pd.DataFrame) -> None:
    frame["gap_minutes"] = 0.0
    for _, group in frame.groupby(["station", "layer"], sort=False, observed=True):
        ordered = group.sort_values("time", kind="stable")
        difference = ordered["time"].diff().dt.total_seconds().div(60.0).fillna(0.0)
        frame.loc[ordered.index, "gap_minutes"] = difference.to_numpy(np.float64)
    frame["gap_bin"] = pd.cut(
        frame["gap_minutes"],
        bins=[-np.inf, 10.0, 30.0, 120.0, np.inf],
        labels=["le10", "11_30", "31_120", "gt120"],
        include_lowest=True,
    )


def attach_truth_event_geometry(frame: pd.DataFrame) -> None:
    state = np.full(len(frame), "negative", dtype=object)
    length = np.zeros(len(frame), dtype=np.int64)
    event_counter = 0
    for _, group in frame.groupby(["fold", "station", "layer"], sort=False, observed=True):
        ordered = group.sort_values("time", kind="stable")
        indices = ordered.index.to_numpy(np.int64)
        truth = ordered["truth"].to_numpy(np.int8)
        elapsed_seconds = ordered["time"].diff().dt.total_seconds().to_numpy(np.float64)
        positive_positions = np.flatnonzero(truth == 1)
        if not len(positive_positions):
            continue
        breaks = np.ones(len(positive_positions), dtype=bool)
        if len(positive_positions) > 1:
            consecutive = positive_positions[1:] == positive_positions[:-1] + 1
            cadence = elapsed_seconds[positive_positions[1:]] == 600.0
            breaks[1:] = ~(consecutive & cadence)
        event_local = np.cumsum(breaks) - 1
        for local_id in np.unique(event_local):
            positions = positive_positions[event_local == local_id]
            rows = indices[positions]
            event_counter += 1
            length[rows] = len(rows)
            if len(rows) == 1:
                state[rows] = "singleton"
            else:
                state[rows[0]] = "onset"
                state[rows[-1]] = "end"
                if len(rows) > 2:
                    state[rows[1:-1]] = "interior"
    frame["truth_event_state"] = state
    frame["truth_event_run_length_rows"] = length
    frame["truth_event_run_length_bin"] = pd.cut(
        length,
        bins=[-1, 0, 1, 3, 6, 12, 24, np.inf],
        labels=["negative", "1", "2_3", "4_6", "7_12", "13_24", "25_plus"],
    )
    frame.attrs["truth_event_count"] = event_counter


def attach_probability_bins(frame: pd.DataFrame) -> list[str]:
    output: list[str] = []
    edges = [-np.inf, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, np.inf]
    labels = ["lt_0.10", "0.10_0.25", "0.25_0.50", "0.50_0.75", "0.75_0.90", "0.90_0.95", "0.95_0.99", "ge_0.99"]
    for source_name, column in (
        ("base", "probability_base"),
        ("peer", "probability_peer"),
        ("e150", "e150_probability"),
    ):
        target = f"{source_name}_probability_bin"
        frame[target] = pd.cut(frame[column], bins=edges, labels=labels, right=False)
        output.append(target)
    return output


def attach_q2_residual_bins(frame: pd.DataFrame) -> tuple[list[str], dict[str, list[float]]]:
    output: list[str] = []
    edge_receipt: dict[str, list[float]] = {}
    q2_negative = (frame["fold"] == "2025_q2") & (frame["anchor"] == 0)
    for column in (
        "temp_abs_median_resid_24h",
        "temp_abs_peer_residual",
        "temp_robust_z_24h",
    ):
        values = np.abs(frame[column].to_numpy(np.float64))
        fit_values = values[q2_negative.to_numpy() & np.isfinite(values)]
        raw_edges = np.quantile(fit_values, [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        edges = np.unique(raw_edges)
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        target = f"{column}_q2_distribution_bin"
        frame[target] = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
        output.append(target)
        edge_receipt[target] = [float(value) for value in edges]
    return output, edge_receipt


def top_rows(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["false_negative_rows"] > 0]
    return sorted(
        eligible,
        key=lambda row: (
            -(row["marginal_precision_if_all_anchor_negative_selected"] or 0.0),
            -row["false_negative_rows"],
        ),
    )[:limit]


def main() -> None:
    if CORRECTED_RESULT.exists():
        raise FileExistsError("v25 corrected oracle audit result already exists")
    frame, anchor, _, dependency = source.load_feature_surface()
    frame = frame.copy().reset_index(drop=True)
    frame["anchor"] = anchor.astype(np.int8)
    frame["truth"] = frame["label_base"].to_numpy(np.int8)
    frame["quarter"] = frame["fold"].astype(str)
    attach_gap_minutes(frame)
    attach_truth_event_geometry(frame)
    probability_bins = attach_probability_bins(frame)
    residual_bins, residual_edges = attach_q2_residual_bins(frame)
    frame["missingness_pattern"] = (
        "psal" + frame["psal_missing"].astype(int).astype(str)
        + "_depth" + frame["depth_missing"].astype(int).astype(str)
        + "_gap" + frame["has_gap_before"].astype(int).astype(str)
    )

    tables = {
        "station_layer_quarter": summary_table(frame, ["station", "layer", "quarter"]),
        "truth_event_state": summary_table(frame.loc[frame["truth"] == 1], ["truth_event_state"]),
        "truth_event_run_length": summary_table(
            frame.loc[frame["truth"] == 1], ["truth_event_run_length_bin"]
        ),
        "missingness_pattern": summary_table(frame, ["missingness_pattern"]),
        "gap_minutes": summary_table(frame, ["gap_bin"]),
        "probability_bins": {
            column: summary_table(frame, [column]) for column in probability_bins
        },
        "causal_residual_bins": {
            column: summary_table(frame, [column]) for column in residual_bins
        },
    }
    fn = (frame["anchor"] == 0) & (frame["truth"] == 1)
    result = {
        "schema_version": "p1.anchor-fn-oracle-audit.v25.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_EXPLORATORY_ORACLE_ONLY",
        "purpose": "Describe where the historical OOF incumbent misses positives; never select a candidate or threshold on outer labels.",
        "rows": len(frame),
        "anchor_false_negative_rows": int(fn.sum()),
        "truth_positive_rows": int((frame["truth"] == 1).sum()),
        "truth_event_count": int(frame.attrs["truth_event_count"]),
        "anchor_false_negative_rate": float(fn.sum() / (frame["truth"] == 1).sum()),
        "tables": tables,
        "high_precision_descriptive_cells": {
            "station_layer_quarter": top_rows(tables["station_layer_quarter"]),
            "missingness_pattern": top_rows(tables["missingness_pattern"]),
            "gap_minutes": top_rows(tables["gap_minutes"]),
            "probability_bins": {
                key: top_rows(value) for key, value in tables["probability_bins"].items()
            },
            "causal_residual_bins": {
                key: top_rows(value) for key, value in tables["causal_residual_bins"].items()
            },
        },
        "residual_bin_edges_fit_on_q2_anchor_negative_covariates_only": residual_edges,
        "source_feature_dependency_receipt": dependency,
        "operations": {
            "historical_oof_reads": 1,
            "candidate_count": 0,
            "model_fits": 0,
            "threshold_searches": 0,
            "attempt_locks_created": 0,
            "official_reads": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "leakage_boundary": "All Q2/Q3/Q4 labels are oracle diagnostics. No cell, bin edge, rank, or threshold is authorized for a v25 candidate without a new result-before-seal design using train-prefix inner selection only.",
        "hashes": {
            "runner_sha256": sha256(Path(__file__)),
            "anchor_sha256": sha256(ROOT / "artifacts/p1_current_router_oof_anchor_v1/anchor.parquet"),
        },
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    CORRECTED_RESULT.write_text(
        json.dumps(native(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "false_negatives": int(fn.sum())}))


if __name__ == "__main__":
    main()
