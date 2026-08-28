"""Run a Q2-selected, Q3/Q4-blind partial-pooling score calibration screen.

No official P1 test/sample/submission path is opened and no deployable CSV is
created.  Existing e150 positives are immutable; the challenger can only add
decoded long-event rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from p1_qc.mstcn_partial_pooling_calibrator import (
    PartialPoolingState,
    fit_partial_pooling,
    predict_partial_pooling,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_partial_pooling_calibrator_shadow_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
SHADOW_BASE = ROOT / "scripts" / "run_p1_mstcn_type_boundary_cascade_shadow_20260829_v1.py"


class ExperimentError(RuntimeError):
    """Raised when a pinned input or blind boundary changes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_shadow_base() -> Any:
    spec = importlib.util.spec_from_file_location("p1_partial_pooling_shadow_base", SHADOW_BASE)
    if spec is None or spec.loader is None:
        raise ExperimentError("cannot load the shared shadow utilities")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ExperimentError("experiment identity changed")
    if value.get("operation_boundaries") != {
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_csv_creation": False,
        "upload": False,
        "result_driven_rerun": False,
    }:
        raise ExperimentError("operation boundary changed")
    return value


def _pinned_path(config: dict[str, Any], phase: str) -> Path:
    pin = config["pinned_blind_inputs"][phase]
    path = ROOT / pin["path"]
    if not path.is_file() or path.stat().st_size != int(pin["bytes"]):
        raise ExperimentError(f"{phase} blind input size changed")
    if _sha256(path) != pin["sha256"]:
        raise ExperimentError(f"{phase} blind input hash changed")
    return path


def _score_bundle(config: dict[str, Any], phase: str) -> dict[str, np.ndarray]:
    path = _pinned_path(config, phase)
    epoch = int(config["scientific_epoch"])
    threshold = float(config["baseline_threshold"])
    with np.load(path, allow_pickle=False) as archive:
        if phase == "q2":
            selected = np.flatnonzero(
                (archive["widths"] == 512) & (archive["epochs"] == epoch)
            )
            threshold_index = np.flatnonzero(np.isclose(archive["thresholds"], threshold))
            if len(selected) != 1 or len(threshold_index) != 1:
                raise ExperimentError("Q2 e150 baseline identity is not unique")
            index = int(selected[0])
            score = archive["row_probability"][index].astype(np.float32)
            boundary = archive["boundary_probability"][index].astype(np.float32)
            baseline = archive["candidate"][index, int(threshold_index[0])].astype(np.int8)
        else:
            selected = np.flatnonzero(archive["epochs"] == epoch)
            if len(selected) != 1:
                raise ExperimentError(f"{phase} e150 baseline identity is not unique")
            index = int(selected[0])
            raw = archive["row_probability"][index].astype(np.float32)
            types = archive["type_probability"][index].astype(np.float32)
            long_type = np.max(types[:, [1, 3, 4]], axis=1)
            score = (raw * (0.75 + 0.25 * long_type)).astype(np.float32)
            boundary = archive["boundary_probability"][index].astype(np.float32)
            baseline = archive["candidate"][index].astype(np.int8)
    if not np.isfinite(score).all() or not np.isfinite(boundary).all():
        raise ExperimentError(f"{phase} score contains nonfinite values")
    return {"score": score, "boundary": boundary, "baseline": baseline}


def _decode(
    runner: Any,
    encoded: Any,
    score: np.ndarray,
    boundary: np.ndarray,
    *,
    threshold: float,
    config: dict[str, Any],
) -> np.ndarray:
    decoder = config["decoder"]
    return runner.decode_long_event_segments(
        score,
        boundary,
        encoded.layout,
        high_threshold=float(threshold),
        snap_radius=int(decoder["boundary_peak_snap_radius_rows"]),
        minimum_rows=int(decoder["minimum_added_segment_rows"]),
        maximum_rows=None,
    ).astype(np.int8)


def _metrics(runner: Any, truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = runner.binary_metrics(truth, baseline)
    right = runner.binary_metrics(truth, candidate)
    added = (candidate == 1) & (baseline == 0)
    removed = (candidate == 0) & (baseline == 1)
    return {
        "baseline": left,
        "candidate": right,
        "delta_f1": float(right["f1"] - left["f1"]),
        "added_rows": int(added.sum()),
        "added_true_rows": int(truth[added].sum()),
        "added_precision": float(truth[added].mean()) if bool(added.any()) else None,
        "removed_rows": int(removed.sum()),
    }


def _validate_baseline(
    runner: Any,
    encoded: Any,
    bundle: dict[str, np.ndarray],
    config: dict[str, Any],
    phase: str,
) -> None:
    proposal = _decode(
        runner,
        encoded,
        bundle["score"],
        bundle["boundary"],
        threshold=float(config["baseline_threshold"]),
        config=config,
    )
    reconstructed = runner.anchor_preserving_union(encoded.surface.anchor, proposal).astype(np.int8)
    if not np.array_equal(reconstructed, bundle["baseline"]):
        raise ExperimentError(f"{phase} archived e150 candidate did not reproduce")


def _select_q2(
    runner: Any,
    encoded: Any,
    bundle: dict[str, np.ndarray],
    truth: np.ndarray,
    config: dict[str, Any],
) -> tuple[PartialPoolingState, float, list[dict[str, Any]]]:
    eligible = bundle["baseline"] == 0
    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, int, float], PartialPoolingState, float] | None = None
    station = encoded.surface.keys["station"].astype(str).to_numpy()
    layer = encoded.surface.keys["layer"].to_numpy()
    for regularization_c in config["regularization_c_grid"]:
        state = fit_partial_pooling(
            bundle["score"],
            station,
            layer,
            truth,
            eligible=eligible,
            regularization_c=float(regularization_c),
            epsilon=float(config["epsilon"]),
        )
        calibrated = predict_partial_pooling(
            state, bundle["score"], station, layer
        )
        for threshold in config["calibrated_threshold_grid"]:
            proposal = _decode(
                runner,
                encoded,
                calibrated,
                bundle["boundary"],
                threshold=float(threshold),
                config=config,
            )
            candidate = np.logical_or(bundle["baseline"] == 1, proposal == 1).astype(np.int8)
            metric = _metrics(runner, truth, bundle["baseline"], candidate)
            added_precision = metric["added_precision"]
            rank = (
                float(metric["candidate"]["f1"]),
                -1.0 if added_precision is None else float(added_precision),
                -int(metric["added_rows"]),
                -float(regularization_c),
            )
            rows.append(
                {
                    "regularization_c": float(regularization_c),
                    "threshold": float(threshold),
                    **metric,
                }
            )
            if best is None or rank > best[0]:
                best = (rank, state, float(threshold))
    if best is None:
        raise ExperimentError("Q2 selection grid is empty")
    return best[1], best[2], rows


def execute() -> dict[str, Any]:
    if ARTIFACT_DIR.exists():
        raise FileExistsError(f"one-shot artifact namespace already exists: {ARTIFACT_DIR}")
    config = _config()
    base = _load_shadow_base()
    runner = base._load_runner()  # noqa: SLF001
    base_config = runner._canonical_config()  # noqa: SLF001
    runner.verify_immutable_inputs(base_config, root=ROOT)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    base._atomic_json(ARTIFACT_DIR / "preregistration.json", config)  # noqa: SLF001

    surfaces = runner.load_blind_surfaces(base_config, root=ROOT)
    encoded: dict[str, Any] = {}
    split_receipts: dict[str, Any] = {}
    for phase in ("q2", "q3", "q4"):
        _encoder, _training, holdout, receipt = runner._prepare_phase_surfaces(  # noqa: SLF001
            surfaces, base_config, phase, root=ROOT
        )
        encoded[phase] = holdout
        split_receipts[phase] = receipt
    bundles = {phase: _score_bundle(config, phase) for phase in ("q2", "q3", "q4")}
    for phase in ("q2", "q3", "q4"):
        _validate_baseline(runner, encoded[phase], bundles[phase], config, phase)

    q2_truth_frame = base._read_fold_truth(  # noqa: SLF001
        runner, base_config, encoded["q2"], "2025_q2"
    )
    q2_truth = q2_truth_frame["label"].to_numpy(dtype=np.int8)
    state, threshold, selection_grid = _select_q2(
        runner, encoded["q2"], bundles["q2"], q2_truth, config
    )
    selected_row = next(
        row
        for row in selection_grid
        if row["regularization_c"] == state.regularization_c
        and row["threshold"] == threshold
    )
    selection = {
        "schema_version": "p1.mstcn_partial_pooling.selection.v1",
        "state": state.as_dict(),
        "threshold": threshold,
        "selected_metrics": selected_row,
        "grid": selection_grid,
        "q2_truth_reads": 1,
        "q3_q4_truth_reads": 0,
    }
    base._atomic_json(ARTIFACT_DIR / "q2_selection.json", selection)  # noqa: SLF001

    blind_receipts: dict[str, Any] = {}
    for phase in ("q3", "q4"):
        keys = encoded[phase].surface.keys
        calibrated = predict_partial_pooling(
            state,
            bundles[phase]["score"],
            keys["station"].astype(str).to_numpy(),
            keys["layer"].to_numpy(),
        )
        proposal = _decode(
            runner,
            encoded[phase],
            calibrated,
            bundles[phase]["boundary"],
            threshold=threshold,
            config=config,
        )
        candidate = np.logical_or(bundles[phase]["baseline"] == 1, proposal == 1).astype(np.int8)
        blind_path = ARTIFACT_DIR / f"{phase}_blind.npz"
        base._atomic_npz(  # noqa: SLF001
            blind_path,
            baseline=bundles[phase]["baseline"],
            calibrated_probability=calibrated,
            calibrated_proposal=proposal,
            candidate=candidate,
        )
        receipt = {
            "schema_version": "p1.mstcn_partial_pooling.blind.v1",
            "phase": phase,
            "fold": base_config["phase_protocols"][phase]["fold"],
            "blind_path": blind_path.name,
            "blind_sha256": _sha256(blind_path),
            "ordered_holdout_key_sha256": runner._ordered_key_sha(keys),  # noqa: SLF001
            "split_receipt": split_receipts[phase],
            "baseline_positive_rows": int(bundles[phase]["baseline"].sum()),
            "candidate_positive_rows": int(candidate.sum()),
            "baseline_positive_removed_rows": int(
                ((bundles[phase]["baseline"] == 1) & (candidate == 0)).sum()
            ),
            "same_fold_truth_reads_before_receipt": 0,
            "official_test_rows_read": 0,
            "submission_created": False,
            "upload_performed": False,
        }
        receipt_path = ARTIFACT_DIR / f"{phase}_blind_receipt.json"
        base._atomic_json(receipt_path, receipt)  # noqa: SLF001
        blind_receipts[phase] = {"path": blind_path, "receipt": receipt_path}

    for phase in ("q3", "q4"):
        receipt = json.loads(blind_receipts[phase]["receipt"].read_text(encoding="utf-8"))
        if receipt["blind_sha256"] != _sha256(blind_receipts[phase]["path"]):
            raise ExperimentError(f"{phase} blind payload changed after sealing")
        if receipt["same_fold_truth_reads_before_receipt"] != 0:
            raise ExperimentError(f"{phase} truth was opened before blind seal")

    metrics: dict[str, Any] = {}
    pooled_truth: list[np.ndarray] = []
    pooled_baseline: list[np.ndarray] = []
    pooled_candidate: list[np.ndarray] = []
    for phase in ("q3", "q4"):
        truth_frame = base._read_fold_truth(  # noqa: SLF001
            runner,
            base_config,
            encoded[phase],
            base_config["phase_protocols"][phase]["fold"],
        )
        truth = truth_frame["label"].to_numpy(dtype=np.int8)
        with np.load(blind_receipts[phase]["path"], allow_pickle=False) as archive:
            baseline = archive["baseline"].astype(np.int8)
            candidate = archive["candidate"].astype(np.int8)
        metric = _metrics(runner, truth, baseline, candidate)
        by_station: dict[str, Any] = {}
        station_values = truth_frame["station"].astype(str).to_numpy()
        for station in sorted(set(station_values)):
            mask = station_values == station
            by_station[station] = _metrics(
                runner, truth[mask], baseline[mask], candidate[mask]
            )
        metric["by_station"] = by_station
        metrics[phase] = metric
        pooled_truth.append(truth)
        pooled_baseline.append(baseline)
        pooled_candidate.append(candidate)

    truth = np.concatenate(pooled_truth)
    baseline = np.concatenate(pooled_baseline)
    candidate = np.concatenate(pooled_candidate)
    pooled = _metrics(runner, truth, baseline, candidate)
    gate = config["confirmation_gate"]
    added_precision = pooled["added_precision"]
    checks = {
        "pooled_delta_f1": pooled["delta_f1"] >= float(gate["pooled_delta_f1_min"]),
        "each_fold_delta_f1": all(
            value["delta_f1"] >= float(gate["each_fold_delta_f1_min"])
            for value in metrics.values()
        ),
        "minimum_positive_folds": sum(value["delta_f1"] > 0.0 for value in metrics.values())
        >= int(gate["minimum_positive_folds"]),
        "marginal_added_precision": added_precision is not None
        and float(added_precision)
        >= float(pooled["baseline"]["f1"])
        * float(gate["minimum_added_precision_relative_to_baseline_f1"]),
        "baseline_positive_deletions": pooled["removed_rows"] == 0,
    }
    status = (
        "GO_LOCAL_REPLICATION_REQUIRED_NOT_SUBMISSION_AUTHORIZED"
        if all(checks.values())
        else "NO_GO_Q3_Q4_CONFIRMATION"
    )
    result = {
        "schema_version": "p1.mstcn_partial_pooling.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "selection": {
            "regularization_c": state.regularization_c,
            "threshold": threshold,
            "q2": selected_row,
        },
        "metrics": metrics,
        "pooled": pooled,
        "gate_checks": checks,
        "q2_truth_reads": 1,
        "q3_q4_truth_reads_after_both_blind_seals": 2,
        "official_test_rows_read": 0,
        "submission_created": False,
        "upload_performed": False,
        "claim_limit": "Retrospective Q2-selected Q3/Q4 confirmation; no official transport claim.",
    }
    base._atomic_json(ARTIFACT_DIR / "result.json", result)  # noqa: SLF001
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "CHECK_ONLY",
                    "config_sha256": _sha256(CONFIG_PATH),
                    "official_test_reads": 0,
                    "submission_created": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
