"""Execute the preregistered e150 F1-aware long-event CP rescue once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from p1_qc.long_event_change_point_rescue import (
    KEY_COLUMNS,
    TARGET_CELLS,
    additions_from_scores,
    anchor_preserving_union,
    binary_metrics,
    build_past_only_row_features,
    cell_fp_diagnostics,
    expanding_cross_fit_scores,
    fit_scorer,
    generate_proposals,
    ordered_changed_key_sha,
    proposal_bank_sha,
    proposal_targets,
    select_threshold_arm,
    stable_sha256,
)

EXPERIMENT_ID = "p1_e150_f1aware_long_event_cp_rescue_20260828_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
SOURCE_RUNNER = ROOT / "scripts" / "run_p1_incumbent_preserving_mstcn_asrf_v2.py"
SOURCE_CONFIG = ROOT / "configs" / "experiments" / "p1_incumbent_preserving_mstcn_asrf_v2.json"
SOURCE_ARTIFACT = ROOT / "artifacts" / "p1_incumbent_preserving_mstcn_asrf_v2"
CHECKPOINT_ARTIFACT = ROOT / "artifacts" / "p1_mstcn_checkpoint_diagnostic_20260827_v2"


class ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    if tuple((str(station), int(layer)) for station, layer in config["target_cells"]) != TARGET_CELLS:
        raise ContractError("target cell scope changed")
    if config["anchor_contract"]["anchor_positive_removal_allowed"] is not False:
        raise ContractError("anchor deletion prohibition changed")
    if config["proposal_scorer"]["zero_add_no_op_arm_required"] is not True:
        raise ContractError("zero-add no-op arm removed")
    return config


def _load_source() -> Any:
    name = f"{EXPERIMENT_ID}_source"
    spec = importlib.util.spec_from_file_location(name, SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise ContractError("cannot import source runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _surface_frame(surface: Any, numeric_names: Sequence[str]) -> pd.DataFrame:
    index = {name: offset for offset, name in enumerate(numeric_names)}
    required = ("temp_raw", "psal_raw", "depth_raw")
    if not set(required).issubset(index):
        raise ContractError("source surface lacks raw current-row fields")
    frame = surface.keys.loc[:, list(KEY_COLUMNS)].copy()
    for name in required:
        frame[name] = surface.numeric[:, index[name]]
    return frame


def _load_base_predictions(source: Any, surfaces: Any, config: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    anchor = config["anchor_contract"]
    grid = source.load_sealed_q2_grid(SOURCE_ARTIFACT / "q2_qualification_grid_receipt.json")
    capacity = np.flatnonzero((grid.widths == int(anchor["width"])) & (grid.epochs == int(anchor["epoch"])))
    threshold = np.flatnonzero(np.isclose(grid.thresholds, float(anchor["threshold"]), atol=0.0, rtol=0.0))
    if len(capacity) != 1 or len(threshold) != 1:
        raise ContractError("Q2 e150 anchor cell is not unique")
    result = {
        "q2": {
            "probability": grid.row_probability[int(capacity[0])].astype(np.float64),
            "anchor": grid.candidate[int(capacity[0]), int(threshold[0])].astype(np.int8),
        }
    }
    for phase in ("q3", "q4"):
        receipt = json.loads((CHECKPOINT_ARTIFACT / f"{phase}_blind_checkpoint_curve_receipt.json").read_text(encoding="utf-8"))
        score_path = CHECKPOINT_ARTIFACT / receipt["score_path"]
        if score_path.stat().st_size != int(receipt["score_bytes"]) or _sha256(score_path) != receipt["score_sha256"]:
            raise ContractError(f"{phase} checkpoint curve identity changed")
        with np.load(score_path, allow_pickle=False) as archive:
            epochs = archive["epochs"]
            location = np.flatnonzero(epochs == int(anchor["epoch"]))
            if len(location) != 1:
                raise ContractError(f"{phase} epoch 150 row is not unique")
            position = int(location[0])
            result[phase] = {
                "probability": archive["row_probability"][position].astype(np.float64),
                "anchor": archive["candidate"][position].astype(np.int8),
            }
    expected = {"q2": surfaces.q2.rows, "q3": surfaces.q3.rows, "q4": surfaces.q4.rows}
    for phase, rows in expected.items():
        if result[phase]["anchor"].shape != (rows,) or result[phase]["probability"].shape != (rows,):
            raise ContractError(f"{phase} base prediction shape changed")
        if not np.isin(result[phase]["anchor"], [0, 1]).all() or not np.isfinite(result[phase]["probability"]).all():
            raise ContractError(f"{phase} base prediction values invalid")
    return result


def _read_truth(source_config: dict[str, Any], surface: Any, fold: str) -> pd.DataFrame:
    record = source_config["immutable_inputs"]["frozen_truth_and_folds"]
    path = ROOT / record["path"]
    if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
        raise ContractError("frozen OOF identity changed")
    table = ds.dataset(path, format="parquet").scanner(
        columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
        filter=ds.field("fold") == fold,
        use_threads=True,
    ).to_table().to_pandas().reset_index(drop=True)
    expected = surface.keys.loc[:, list(KEY_COLUMNS)].astype(str).reset_index(drop=True)
    observed = table.loc[:, list(KEY_COLUMNS)].astype(str).reset_index(drop=True)
    if not expected.equals(observed):
        raise ContractError(f"{fold} truth keys differ from blind surface")
    if not np.isin(table["label"].to_numpy(), [0, 1]).all():
        raise ContractError(f"{fold} truth is not binary")
    return table


def _proposal_receipt(phase: str, keys: pd.DataFrame, proposals: Sequence[Any], feature_names: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": "p1.e150_f1aware_cp.proposal_receipt.v1",
        "phase": phase,
        "rows": int(len(keys)),
        "ordered_key_sha256": source_ordered_key_sha(keys),
        "proposal_count": int(len(proposals)),
        "proposal_bank_sha256": proposal_bank_sha(proposals),
        "feature_names": list(feature_names),
        "target_columns_read_before_receipt": 0,
        "future_rows_used_by_row_features": 0,
        "target_cells": [f"{station}/L{layer}" for station, layer in TARGET_CELLS],
    }


def source_ordered_key_sha(keys: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in keys.loc[:, list(KEY_COLUMNS)].astype(str).itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _evaluate_fold(keys: pd.DataFrame, truth: pd.DataFrame, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    y = truth["label"].to_numpy(dtype=np.int8)
    base = binary_metrics(y, anchor)
    new = binary_metrics(y, candidate)
    changed = (anchor == 0) & (candidate == 1)
    added_tp = int(np.sum(changed & (y == 1)))
    added_fp = int(np.sum(changed & (y == 0)))
    return {
        "rows": int(len(y)),
        "anchor": base,
        "candidate": new,
        "delta_f1": float(new["f1"] - base["f1"]),
        "added_rows": int(changed.sum()),
        "added_tp": added_tp,
        "added_fp": added_fp,
        "added_precision": float(added_tp / (added_tp + added_fp)) if added_tp + added_fp else 1.0,
        "anchor_positive_removed_rows": int(np.sum((anchor == 1) & (candidate == 0))),
        "changed_outside_target_cells": int(
            np.sum(
                changed
                & ~np.logical_or.reduce(
                    [
                        (keys["station"].astype(str).to_numpy() == station)
                        & (keys["layer"].to_numpy(dtype=int) == layer)
                        for station, layer in TARGET_CELLS
                    ]
                )
            )
        ),
        "cell_fp": cell_fp_diagnostics(keys, y, anchor, candidate),
        "changed_key_sha256": ordered_changed_key_sha(keys, changed),
    }


def _bootstrap(fold_payloads: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]], config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for phase, (keys, y, anchor, candidate) in fold_payloads.items():
        dates = pd.to_datetime(keys["time"], utc=True, format="mixed").dt.tz_convert("Asia/Seoul").dt.date.astype(str)
        frame = pd.DataFrame({"phase": phase, "date": dates, "y": y, "a": anchor, "c": candidate})
        for (_phase, _date), group in frame.groupby(["phase", "date"], sort=True, observed=True):
            yy = group["y"].to_numpy(dtype=np.int8)
            aa = group["a"].to_numpy(dtype=np.int8)
            cc = group["c"].to_numpy(dtype=np.int8)
            am = binary_metrics(yy, aa)
            cm = binary_metrics(yy, cc)
            rows.append((int(am["tp"]), int(am["fp"]), int(am["fn"]), int(cm["tp"]), int(cm["fp"]), int(cm["fn"])))
    counts = np.asarray(rows, dtype=np.int64)
    spec = config["bootstrap"]
    replicas = int(spec["replicates"])
    block = int(spec["block_days"])
    rng = np.random.default_rng(int(spec["seed"]))
    n = len(counts)
    deltas = np.empty(replicas, dtype=np.float64)
    blocks = int(np.ceil(n / block))
    offsets = np.arange(block, dtype=np.int64)
    for index in range(replicas):
        starts = rng.integers(0, n, size=blocks)
        sampled = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        total = counts[sampled].sum(axis=0)
        atp, afp, afn, ctp, cfp, cfn = total.tolist()
        af1 = 2 * atp / (2 * atp + afp + afn)
        cf1 = 2 * ctp / (2 * ctp + cfp + cfn)
        deltas[index] = cf1 - af1
    return {
        "method": "paired circular moving-block bootstrap over whole fold x KST-day cross-sections",
        "replicates": replicas,
        "block_days": block,
        "day_cross_sections": n,
        "ci90_lower": float(np.quantile(deltas, 0.05)),
        "ci90_upper": float(np.quantile(deltas, 0.95)),
        "mean": float(np.mean(deltas)),
        "probability_positive": float(np.mean(deltas > 0.0)),
    }


def check_only() -> dict[str, Any]:
    _load_config()
    required = [SOURCE_RUNNER, SOURCE_CONFIG, SOURCE_ARTIFACT / "q2_qualification_grid_blind.npz", CHECKPOINT_ARTIFACT / "q3_blind_checkpoint_curve.npz", CHECKPOINT_ARTIFACT / "q4_blind_checkpoint_curve.npz"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ContractError(f"required artifacts missing: {missing}")
    if ARTIFACT_DIR.exists():
        raise ContractError("append-only artifact namespace already exists")
    return {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "module_sha256": _sha256(ROOT / "src" / "p1_qc" / "long_event_change_point_rescue.py"),
        "official_test_reads": 0,
        "official_uploads": 0,
        "result": "PASS",
    }


def execute() -> dict[str, Any]:
    preflight = check_only()
    config = _load_config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    _atomic_json(ARTIFACT_DIR / "preflight.json", preflight)
    source_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    source = _load_source()
    surfaces = source.load_blind_surfaces(source_config, root=ROOT)
    base = _load_base_predictions(source, surfaces, config)

    phase_surfaces = {"q2": surfaces.q2, "q3": surfaces.q3, "q4": surfaces.q4}
    frames: dict[str, pd.DataFrame] = {}
    proposals: dict[str, list[Any]] = {}
    feature_names: tuple[str, ...] | None = None
    for phase, surface in phase_surfaces.items():
        frame = _surface_frame(surface, surfaces.numeric_names)
        frame["anchor_probability"] = base[phase]["probability"]
        frame["anchor"] = base[phase]["anchor"]
        frames[phase] = frame
        row_features = build_past_only_row_features(frame)
        proposal_spec = config["past_only_proposals"]
        proposal_bank, names = generate_proposals(
            row_features,
            score_thresholds=proposal_spec["score_thresholds"],
            minimum_support_rows=proposal_spec["minimum_support_rows"],
            maximum_gap_rows=int(proposal_spec["maximum_gap_rows"]),
            padding_rows=int(proposal_spec["padding_rows"]),
            minimum_interval_rows=int(proposal_spec["minimum_interval_rows"]),
            maximum_interval_rows=int(proposal_spec["maximum_interval_rows"]),
        )
        proposals[phase] = proposal_bank
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise ContractError("proposal feature schema changed across phases")
        _atomic_json(ARTIFACT_DIR / f"{phase}_proposal_receipt.json", _proposal_receipt(phase, surface.keys, proposal_bank, names))

    # Q2 is the only design fold.  Q3/Q4 truth remains unopened here.
    q2_truth = _read_truth(source_config, surfaces.q2, "2025_q2")
    q2_y = q2_truth["label"].to_numpy(dtype=np.int8)
    q2_targets, q2_target_diagnostics = proposal_targets(proposals["q2"], q2_y, base["q2"]["anchor"])
    q2_features = np.vstack([item.features for item in proposals["q2"]])
    q2_oof_scores, cross_fit_receipts = expanding_cross_fit_scores(proposals["q2"], q2_targets, seed=int(config["proposal_scorer"]["seed"]))
    selection, _q2_additions = select_threshold_arm(
        surfaces.q2.keys,
        q2_y,
        base["q2"]["anchor"],
        proposals["q2"],
        q2_oof_scores,
        threshold_candidates=config["proposal_scorer"]["threshold_candidates"],
        maximum_added_fp_per_day=float(config["selection_guards"]["maximum_added_false_positives_per_day_per_station_level"]),
        minimum_added_precision=float(config["selection_guards"]["minimum_added_row_precision"]),
    )
    scorer = fit_scorer(q2_features, q2_targets, seed=int(config["proposal_scorer"]["seed"]))
    joblib.dump(scorer, ARTIFACT_DIR / "proposal_scorer.joblib")
    frozen_selection = {
        "schema_version": "p1.e150_f1aware_cp.frozen_selection.v1",
        "design_fold": "2025_q2",
        "selected": selection,
        "proposal_count": len(proposals["q2"]),
        "beneficial_proposal_count": int(q2_targets.sum()),
        "proposal_target_sha256": hashlib.sha256(q2_targets.tobytes()).hexdigest(),
        "cross_fit_receipts": cross_fit_receipts,
        "q3_q4_truth_columns_read_before_selection": 0,
        "feature_names": list(feature_names or ()),
        "scorer_sha256": _sha256(ARTIFACT_DIR / "proposal_scorer.joblib"),
        "individual_target_diagnostic_aggregate": {
            "tp_added_sum_over_overlapping_proposals": int(sum(int(item["tp_added"]) for item in q2_target_diagnostics)),
            "fp_added_sum_over_overlapping_proposals": int(sum(int(item["fp_added"]) for item in q2_target_diagnostics)),
        },
    }
    _atomic_json(ARTIFACT_DIR / "frozen_selection.json", frozen_selection)

    blind_arrays: dict[str, np.ndarray] = {}
    selected_threshold = selection["threshold"]
    for phase in ("q3", "q4"):
        if selected_threshold is None:
            additions = np.zeros(phase_surfaces[phase].rows, dtype=np.int8)
            scores = scorer.predict_proba(np.vstack([item.features for item in proposals[phase]])) if proposals[phase] else np.empty(0)
        else:
            features = np.vstack([item.features for item in proposals[phase]]) if proposals[phase] else np.empty((0, len(feature_names or ())))
            scores = scorer.predict_proba(features)
            additions = additions_from_scores(phase_surfaces[phase].rows, proposals[phase], scores, float(selected_threshold), base[phase]["anchor"])
        blind_arrays[f"{phase}_proposal_score"] = scores.astype(np.float32)
        blind_arrays[f"{phase}_additions"] = additions.astype(np.int8)
        blind_arrays[f"{phase}_candidate"] = anchor_preserving_union(base[phase]["anchor"], additions)
    blind_path = ARTIFACT_DIR / "q3_q4_blind_predictions.npz"
    np.savez_compressed(blind_path, **blind_arrays)
    blind_receipt = {
        "schema_version": "p1.e150_f1aware_cp.blind_predictions.v1",
        "score_path": blind_path.name,
        "score_bytes": blind_path.stat().st_size,
        "score_sha256": _sha256(blind_path),
        "array_inventory": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in blind_arrays.items()},
        "q3_q4_truth_columns_read_before_receipt": 0,
        "selected_threshold_sha256": stable_sha256(selection),
        "anchor_positive_removal_allowed": False,
    }
    _atomic_json(ARTIFACT_DIR / "q3_q4_blind_predictions_receipt.json", blind_receipt)

    # Only now may blind truth be opened.
    truths = {phase: _read_truth(source_config, phase_surfaces[phase], f"2025_{phase}") for phase in ("q3", "q4")}
    fold_metrics = {
        phase: _evaluate_fold(
            phase_surfaces[phase].keys,
            truths[phase],
            base[phase]["anchor"],
            blind_arrays[f"{phase}_candidate"],
        )
        for phase in ("q3", "q4")
    }
    pooled_y = np.concatenate([truths[phase]["label"].to_numpy(dtype=np.int8) for phase in ("q3", "q4")])
    pooled_anchor = np.concatenate([base[phase]["anchor"] for phase in ("q3", "q4")])
    pooled_candidate = np.concatenate([blind_arrays[f"{phase}_candidate"] for phase in ("q3", "q4")])
    pooled_base_metrics = binary_metrics(pooled_y, pooled_anchor)
    pooled_candidate_metrics = binary_metrics(pooled_y, pooled_candidate)
    changed = (pooled_anchor == 0) & (pooled_candidate == 1)
    added_tp = int(np.sum(changed & (pooled_y == 1)))
    added_fp = int(np.sum(changed & (pooled_y == 0)))
    bootstrap = _bootstrap(
        {
            phase: (
                phase_surfaces[phase].keys,
                truths[phase]["label"].to_numpy(dtype=np.int8),
                base[phase]["anchor"],
                blind_arrays[f"{phase}_candidate"],
            )
            for phase in ("q3", "q4")
        },
        config,
    )
    all_cells = {f"{phase}:{cell}": value for phase in ("q3", "q4") for cell, value in fold_metrics[phase]["cell_fp"].items()}
    gates = config["blind_decision_gates"]
    gate_checks = {
        "selected_non_noop_arm": selection["arm"] != "ZERO_ADD_NO_OP",
        "pooled_delta_f1": float(pooled_candidate_metrics["f1"] - pooled_base_metrics["f1"]) >= float(gates["pooled_delta_f1_gte"]),
        "improving_blind_folds": sum(fold_metrics[p]["delta_f1"] > 0.0 for p in ("q3", "q4")) >= int(gates["minimum_improving_blind_folds_of_2"]),
        "q4_nonnegative": float(fold_metrics["q4"]["delta_f1"]) >= float(gates["q4_delta_f1_gte"]),
        "bootstrap_ci90_lower": float(bootstrap["ci90_lower"]) > float(gates["paired_block_bootstrap_ci90_lower_gt"]),
        "cell_fp_cap": all(float(value["added_fp_per_day"]) <= float(gates["maximum_added_false_positives_per_day_per_station_level"]) for value in all_cells.values()),
        "added_precision": (float(added_tp / (added_tp + added_fp)) if added_tp + added_fp else 1.0) >= float(gates["minimum_added_row_precision"]),
        "anchor_preserved": all(fold_metrics[p]["anchor_positive_removed_rows"] == int(gates["anchor_positive_removed_rows_eq"]) for p in ("q3", "q4")),
        "scope_preserved": all(fold_metrics[p]["changed_outside_target_cells"] == 0 for p in ("q3", "q4")),
    }
    passed = all(gate_checks.values())
    metrics = {
        "schema_version": "p1.e150_f1aware_cp.metrics.v1",
        "experiment_id": EXPERIMENT_ID,
        "selection": selection,
        "folds": fold_metrics,
        "pooled": {
            "rows": int(len(pooled_y)),
            "anchor": pooled_base_metrics,
            "candidate": pooled_candidate_metrics,
            "delta_f1": float(pooled_candidate_metrics["f1"] - pooled_base_metrics["f1"]),
            "added_rows": int(changed.sum()),
            "added_tp": added_tp,
            "added_fp": added_fp,
            "added_precision": float(added_tp / (added_tp + added_fp)) if added_tp + added_fp else 1.0,
            "anchor_positive_removed_rows": int(np.sum((pooled_anchor == 1) & (pooled_candidate == 0))),
        },
        "bootstrap": bootstrap,
        "gate_checks": gate_checks,
        "passed_all_gates": passed,
        "decision": "GO_LOCAL_BLIND_GATE" if passed else "NO_GO_LOCAL_BLIND_GATE",
        "official_probe_authorized": False,
        "surface_role": "retrospective blind-to-this-run Q3/Q4; historically exposed, not fresh confirmation",
    }
    _atomic_json(ARTIFACT_DIR / "metrics.json", metrics)
    changed_manifest = {
        "schema_version": "p1.e150_f1aware_cp.changed_rows.v1",
        "folds": {phase: {"added_rows": fold_metrics[phase]["added_rows"], "changed_key_sha256": fold_metrics[phase]["changed_key_sha256"], "changed_outside_target_cells": fold_metrics[phase]["changed_outside_target_cells"]} for phase in ("q3", "q4")},
        "anchor_positive_removed_rows": int(metrics["pooled"]["anchor_positive_removed_rows"]),
        "raw_keys_persisted": False,
    }
    _atomic_json(ARTIFACT_DIR / "changed_rows_manifest.json", changed_manifest)
    terminal = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_LOCAL_BLIND_SCREEN",
        "decision": metrics["decision"],
        "passed_all_gates": passed,
        "official_test_reads": 0,
        "submission_files_created": 0,
        "uploads": 0,
        "result_driven_reruns_allowed": False,
    }
    _atomic_json(ARTIFACT_DIR / "terminal_result.json", terminal)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(CONFIG_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "module_sha256": _sha256(ROOT / "src" / "p1_qc" / "long_event_change_point_rescue.py"),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(ARTIFACT_DIR.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    _atomic_json(ARTIFACT_DIR / "manifest.json", manifest)
    return terminal | {"metrics_path": str(ARTIFACT_DIR / "metrics.json"), "manifest_sha256": _sha256(ARTIFACT_DIR / "manifest.json")}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = check_only() if args.check_only else execute()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
