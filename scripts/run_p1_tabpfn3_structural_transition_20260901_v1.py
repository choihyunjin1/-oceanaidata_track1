"""Exactly-once historical P1 TabPFN-3 structural transition.

The runner changes only the classifier family.  It reuses the frozen 165-column
split-local feature surface and current-router anchor.  Q2 is a screening fold;
Q3 and Q4 are produced and sealed together only if the screen improves F1.
No official P1 test/sample/submission file is resolved by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ocean_tabpfn3.offline import canonical_json_bytes, make_classifier, require_ready, sha256

EXPERIMENT_ID = "p1_tabpfn3_structural_transition_20260901_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
OUTPUT = ROOT / "artifacts" / EXPERIMENT_ID
LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
KEY_COLUMNS = ("station", "year", "layer", "time")


class P1TabPFNContractError(RuntimeError):
    """Raised when the frozen P1 transition contract is violated."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P1TabPFNContractError(f"JSON object required: {path}")
    return value


def _base_module() -> Any:
    name = f"{EXPERIMENT_ID}_base"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "scripts" / "run_p1_incumbent_preserving_mstcn_asrf_v2.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise P1TabPFNContractError("cannot import frozen P1 base runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, Any]:
    config = _load_json(CONFIG)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise P1TabPFNContractError("experiment id mismatch")
    if config.get("status") != "READY_GUARDED":
        raise P1TabPFNContractError("transition status changed")
    model = config["model"]
    if not (
        model["n_estimators"] == 8
        and model["training_cap_per_cell"] == 49_000
        and model["threshold"] == 0.5
        and model["maximum_cell_fits"] == 48
    ):
        raise P1TabPFNContractError("frozen model contract changed")
    for name, record in config["pinned_inputs"].items():
        path = (ROOT / record["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != record["sha256"]:
            raise P1TabPFNContractError(f"pinned input mismatch: {name}")
    return config


def deterministic_binary_sample(
    labels: np.ndarray,
    keys: pd.DataFrame,
    *,
    cap: int,
    seed: int,
) -> np.ndarray:
    """Retain all positive rows and deterministically rank only excess negatives."""

    target = np.asarray(labels, dtype=np.int8)
    if target.ndim != 1 or len(target) != len(keys) or not np.isin(target, [0, 1]).all():
        raise ValueError("labels and keys must be aligned binary rows")
    if cap < 2:
        raise ValueError("cap must be at least two")
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("each fitted cell requires both classes")
    if len(positive) >= cap:
        raise P1TabPFNContractError("positive rows alone exceed the TabPFN sample limit")
    negative_budget = cap - len(positive)
    if len(negative) > negative_budget:
        key_text = keys.astype(str).agg("|".join, axis=1).to_numpy()
        ranked = sorted(
            negative.tolist(),
            key=lambda index: hashlib.sha256(
                f"{seed}|{key_text[index]}".encode()
            ).digest(),
        )
        negative = np.asarray(ranked[:negative_budget], dtype=np.int64)
    selected = np.sort(np.concatenate([positive, negative])).astype(np.int64)
    if len(selected) > cap or int(target[selected].sum()) != len(positive):
        raise AssertionError("deterministic binary sampler violated its preservation contract")
    return selected


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def preflight() -> dict[str, Any]:
    config = _config()
    tabpfn = require_ready(workspace=ROOT)
    base = _base_module()
    base_config = base._canonical_config()
    immutable = base.verify_immutable_inputs(base_config)
    if OUTPUT.exists() or LOCK.exists():
        raise P1TabPFNContractError("exactly-once output namespace already exists")
    return {
        "schema_version": "p1.tabpfn3.structural_transition.preflight.v1",
        "status": "READY",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256(CONFIG),
        "tabpfn": tabpfn,
        "base_immutable_inputs": immutable,
        "features": 165,
        "official_access": config["official_access_budget"],
    }


def _open_truth(base: Any, base_config: dict[str, Any], holdout: Any, phase: str) -> pd.DataFrame:
    import pyarrow.dataset as dataset

    fold = base_config["phase_protocols"][phase]["fold"]
    with base._verified_immutable_read(base_config, "frozen_truth_and_folds", root=ROOT) as path:
        table = dataset.dataset(path, format="parquet").scanner(
            columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
            filter=dataset.field("fold") == fold,
            use_threads=True,
        ).to_table()
    truth = table.to_pandas().reset_index(drop=True)
    truth, _ = base._validate_registered_holdout_membership(truth, base_config, fold=fold)
    if not base._keys_equal(holdout.surface.keys, truth):
        raise P1TabPFNContractError(f"{phase} truth keys differ from blind prediction keys")
    return truth


def _phase_prediction(
    *,
    base: Any,
    surfaces: Any,
    base_config: dict[str, Any],
    phase: str,
    config: dict[str, Any],
    weight_path: Path,
    artifact_dir: Path,
) -> tuple[Any, np.ndarray, Path, dict[str, Any], int]:
    _encoder, training, holdout, split = base._prepare_phase_surfaces(
        surfaces, base_config, phase, root=ROOT
    )
    if training.features.shape[1] != 165 or holdout.features.shape[1] != 165:
        raise P1TabPFNContractError("encoded feature width is not 165")
    if holdout.surface.anchor is None or training.surface.labels is None:
        raise P1TabPFNContractError("training labels or holdout anchor are absent")

    probability = np.zeros(holdout.surface.rows, dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    fit_count = 0
    holdout_cells = sorted(
        set(zip(holdout.surface.station.tolist(), holdout.surface.layer_category.tolist(), strict=True))
    )
    for cell_index, (station, layer) in enumerate(holdout_cells):
        train_ids = np.flatnonzero(
            (training.surface.station == station)
            & (training.surface.layer_category == layer)
        )
        valid_ids = np.flatnonzero(
            (holdout.surface.station == station)
            & (holdout.surface.layer_category == layer)
        )
        if len(valid_ids) == 0:
            continue
        labels = training.surface.labels[train_ids]
        if len(train_ids) < 2 or len(np.unique(labels)) != 2:
            receipts.append(
                {
                    "station": str(station),
                    "layer": str(layer),
                    "status": "NO_FIT_MISSING_CLASS_OR_TRAIN_ROWS",
                    "train_rows": int(len(train_ids)),
                    "validation_rows": int(len(valid_ids)),
                }
            )
            continue
        sample_local = deterministic_binary_sample(
            labels,
            training.surface.keys.iloc[train_ids].reset_index(drop=True),
            cap=int(config["model"]["training_cap_per_cell"]),
            seed=20260901 + cell_index,
        )
        sample_ids = train_ids[sample_local]
        model = make_classifier(
            weight_path,
            seed=20260901 + cell_index,
            n_estimators=int(config["model"]["n_estimators"]),
        )
        model.fit(training.features[sample_ids], training.surface.labels[sample_ids])
        classes = np.asarray(model.classes_)
        positive_columns = np.flatnonzero(classes == 1)
        if len(positive_columns) != 1:
            raise P1TabPFNContractError("TabPFN classifier lacks the positive class")
        probability[valid_ids] = model.predict_proba(holdout.features[valid_ids])[
            :, int(positive_columns[0])
        ]
        fit_count += 1
        receipts.append(
            {
                "station": str(station),
                "layer": str(layer),
                "status": "FIT",
                "train_rows_before_cap": int(len(train_ids)),
                "train_rows": int(len(sample_ids)),
                "positive_rows": int(training.surface.labels[sample_ids].sum()),
                "validation_rows": int(len(valid_ids)),
            }
        )
        (artifact_dir / "progress.json").write_text(
            json.dumps(
                {"phase": phase, "cell": cell_index + 1, "cells": len(holdout_cells), "fits": fit_count},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    proposal = (probability >= float(config["model"]["threshold"])).astype(np.int8)
    candidate = base.anchor_preserving_union(holdout.surface.anchor, proposal)
    score_path = artifact_dir / f"{phase}_blind.npz"
    np.savez_compressed(score_path, probability=probability, proposal=proposal, candidate=candidate)
    receipt = {
        "schema_version": "p1.tabpfn3.blind_receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "ordered_holdout_key_sha256": base._ordered_key_sha(holdout.surface.keys),
        "score": {"path": score_path.name, "bytes": score_path.stat().st_size, "sha256": sha256(score_path)},
        "fit_count": fit_count,
        "cells": receipts,
        "same_fold_truth_rows_read_before_seal": 0,
        "official_rows_read": 0,
    }
    receipt_path = artifact_dir / f"{phase}_blind_receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    return holdout, candidate, receipt_path, split, fit_count


def _metrics(base: Any, truth: pd.DataFrame, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    y = truth["label"].to_numpy(dtype=np.int8)
    anchor_score = base.binary_metrics(y, anchor)
    candidate_score = base.binary_metrics(y, candidate)
    return {
        "rows": int(len(y)),
        "anchor": anchor_score,
        "candidate": candidate_score,
        "delta_f1": float(candidate_score["f1"] - anchor_score["f1"]),
        "added_rows": int(np.sum((anchor == 0) & (candidate == 1))),
        "anchor_positive_removed_rows": int(np.sum((anchor == 1) & (candidate == 0))),
    }


def execute() -> dict[str, Any]:
    started = time.perf_counter()
    first = preflight()
    second = preflight()
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise P1TabPFNContractError("two preflights are not byte-identical")
    config = _config()
    lock = {
        "schema_version": "p1.tabpfn3.attempt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256(CONFIG),
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
    }
    _exclusive_json(LOCK, lock)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "preflight.json").write_bytes(canonical_json_bytes(first))
    fit_count = 0
    try:
        base = _base_module()
        base_config = base._canonical_config()
        surfaces = base.load_blind_surfaces(base_config, root=ROOT)
        classifier_path = Path(first["tabpfn"]["weights"]["classifier"]["path"])

        q2, q2_candidate, _q2_receipt, q2_split, q2_fits = _phase_prediction(
            base=base,
            surfaces=surfaces,
            base_config=base_config,
            phase="q2",
            config=config,
            weight_path=classifier_path,
            artifact_dir=OUTPUT,
        )
        fit_count += q2_fits
        q2_truth = _open_truth(base, base_config, q2, "q2")
        q2_metrics = _metrics(base, q2_truth, q2.surface.anchor, q2_candidate)
        if q2_metrics["delta_f1"] <= 0.0:
            result = {
                "schema_version": "p1.tabpfn3.structural_transition.result.v1",
                "experiment_id": EXPERIMENT_ID,
                "status": "COMPLETE_NO_GO_Q2",
                "fit_count": fit_count,
                "runtime_seconds": time.perf_counter() - started,
                "q2": q2_metrics,
                "q2_split": q2_split,
                "weights": first["tabpfn"]["weights"],
                "official_access": config["official_access_budget"],
                "submission_ready": False,
                "automatic_retry_forbidden": True,
            }
        else:
            sealed: dict[str, tuple[Any, np.ndarray, Path, dict[str, Any], int]] = {}
            for phase in ("q3", "q4"):
                sealed[phase] = _phase_prediction(
                    base=base,
                    surfaces=surfaces,
                    base_config=base_config,
                    phase=phase,
                    config=config,
                    weight_path=classifier_path,
                    artifact_dir=OUTPUT,
                )
                fit_count += sealed[phase][4]
            fold_metrics: dict[str, Any] = {}
            truth_parts: list[np.ndarray] = []
            anchor_parts: list[np.ndarray] = []
            candidate_parts: list[np.ndarray] = []
            for phase in ("q3", "q4"):
                holdout, candidate, _receipt, _split, _fits = sealed[phase]
                truth = _open_truth(base, base_config, holdout, phase)
                fold_metrics[phase] = _metrics(
                    base, truth, holdout.surface.anchor, candidate
                )
                truth_parts.append(truth["label"].to_numpy(dtype=np.int8))
                anchor_parts.append(holdout.surface.anchor)
                candidate_parts.append(candidate)
            y = np.concatenate(truth_parts)
            anchor = np.concatenate(anchor_parts)
            candidate = np.concatenate(candidate_parts)
            pooled_anchor = base.binary_metrics(y, anchor)
            pooled_candidate = base.binary_metrics(y, candidate)
            pooled = {
                "rows": int(len(y)),
                "anchor": pooled_anchor,
                "candidate": pooled_candidate,
                "delta_f1": float(pooled_candidate["f1"] - pooled_anchor["f1"]),
                "anchor_positive_removed_rows": int(np.sum((anchor == 1) & (candidate == 0))),
            }
            result = {
                "schema_version": "p1.tabpfn3.structural_transition.result.v1",
                "experiment_id": EXPERIMENT_ID,
                "status": (
                    "COMPLETE_GO_HISTORICAL_CONFIRMATION"
                    if pooled["delta_f1"] > 0.0
                    else "COMPLETE_NO_GO_CONFIRMATION"
                ),
                "fit_count": fit_count,
                "runtime_seconds": time.perf_counter() - started,
                "q2": q2_metrics,
                "confirmation": {"folds": fold_metrics, "pooled": pooled},
                "weights": first["tabpfn"]["weights"],
                "official_access": config["official_access_budget"],
                "submission_ready": False,
                "automatic_retry_forbidden": True,
            }
        result_path = OUTPUT / "result.json"
        result_path.write_bytes(canonical_json_bytes(result))
        terminal = {**result, "result_sha256": sha256(result_path), "terminal": True}
        (OUTPUT / "terminal_result.json").write_bytes(canonical_json_bytes(terminal))
        return terminal
    except BaseException as exc:
        failure = {
            "schema_version": "p1.tabpfn3.structural_transition.terminal.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fit_count": fit_count,
            "runtime_seconds": time.perf_counter() - started,
            "official_access": config["official_access_budget"],
            "automatic_restart_forbidden": True,
            "terminal": True,
        }
        (OUTPUT / "terminal_result.json").write_bytes(canonical_json_bytes(failure))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute == args.check_only:
        parser.error("choose exactly one of --check-only or --execute")
    result = execute() if args.execute else preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
