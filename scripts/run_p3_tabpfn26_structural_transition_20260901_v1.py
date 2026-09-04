"""Exactly-once clean historical P3 TabPFN-2.6 residual confirmation."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ocean_tabpfn26.offline import canonical_json_bytes, make_regressor, require_ready, sha256
from p3_wave.clean_fractional_change_residual_20260901_c1 import blend_with_clean_fallback
from p3_wave.clean_fractional_change_residual_20260901_c1r1 import (
    _input_paths,
    assert_validation_surface_matches,
)
from p3_wave.corrected_repeated_forward import build_corrected_repeated_forward_folds
from p3_wave.models import compact_feature_columns
from p3_wave.revin_patch import assign_storm_episodes_from_wave
from p3_wave.validation import LEADS, expand_leads, metric_slices

EXPERIMENT_ID = "p3_tabpfn26_structural_transition_20260901_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
CLEAN_CONFIG = ROOT / "configs" / "experiments" / (
    "p3_clean_fractional_change_residual_20260901_c1r1.json"
)
OUTPUT = ROOT / "artifacts" / EXPERIMENT_ID
LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"


class P3TabPFNContractError(RuntimeError):
    """Raised when the clean P3 transition contract is violated."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P3TabPFNContractError(f"JSON object required: {path}")
    return value


def _config() -> dict[str, Any]:
    config = _json(CONFIG)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise P3TabPFNContractError("experiment id mismatch")
    if config.get("status") != "READY_PENDING_USER_LICENSE_AND_LOCAL_WEIGHTS":
        raise P3TabPFNContractError("transition status changed")
    model = config["model"]
    if not (
        model["n_estimators"] == 8
        and model["leads_h"] == [1, 2, 3, 6, 12, 24]
        and model["clean_fallback_weight"] == 0.75
        and model["tabpfn_weight"] == 0.25
        and model["maximum_fits"] == 18
    ):
        raise P3TabPFNContractError("frozen P3 model contract changed")
    for name, record in config["pinned_inputs"].items():
        path = (ROOT / record["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != record["sha256"]:
            raise P3TabPFNContractError(f"pinned input mismatch: {name}")
    return config


def lead_specific_matrix(frame: pd.DataFrame, *, lead: int) -> np.ndarray:
    """Encode the fixed station category and numeric clean surface for one lead."""

    selected = frame.loc[frame["lead_h"].eq(lead)].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"empty lead surface: {lead}")
    station_map = {"G-ORS": 0.0, "S-ORS": 1.0}
    station = selected["station"].map(station_map)
    if station.isna().any():
        raise P3TabPFNContractError("unexpected clean P3 station")
    numeric = selected.drop(columns=["station", "lead_h"]).apply(
        pd.to_numeric, errors="coerce"
    )
    matrix = np.column_stack(
        [station.to_numpy(dtype=np.float32), numeric.to_numpy(dtype=np.float32)]
    )
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix.astype(np.float32, copy=False)


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


def _data_dir() -> Path:
    raw = os.environ.get("P3_DATA_DIR")
    if not raw:
        raise P3TabPFNContractError("P3_DATA_DIR is required")
    path = Path(raw).expanduser().resolve(strict=True)
    for filename in ("README.md", "train_wave.csv", "train_atmos.csv"):
        if not (path / filename).is_file():
            raise P3TabPFNContractError(f"P3_DATA_DIR is missing {filename}")
    return path


def preflight() -> dict[str, Any]:
    config = _config()
    tabpfn = require_ready(workspace=ROOT)
    data_dir = _data_dir()
    clean = _json(CLEAN_CONFIG)
    paths = _input_paths(ROOT, data_dir)
    verified: dict[str, dict[str, Any]] = {}
    for label, expected in clean["inputs"].items():
        path = paths[label].resolve(strict=True)
        observed = {"bytes": int(path.stat().st_size), "sha256": sha256(path)}
        if observed != {"bytes": int(expected["bytes"]), "sha256": expected["sha256"]}:
            raise P3TabPFNContractError(f"clean input mismatch: {label}")
        verified[label] = observed
    if OUTPUT.exists() or LOCK.exists():
        raise P3TabPFNContractError("exactly-once output namespace already exists")
    return {
        "schema_version": "p3.tabpfn26.structural_transition.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY",
        "config_sha256": sha256(CONFIG),
        "tabpfn": tabpfn,
        "verified_clean_inputs": verified,
        "expected_train_shape": [24_360, 591],
        "expected_validation": {"cases": 181, "rows": 1_086},
        "official_access": config["official_access_budget"],
    }


def execute() -> dict[str, Any]:
    started = time.perf_counter()
    first = preflight()
    second = preflight()
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise P3TabPFNContractError("two preflights are not byte-identical")
    config = _config()
    lock = {
        "schema_version": "p3.tabpfn26.attempt.v1",
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
        data_dir = _data_dir()
        clean = _json(CLEAN_CONFIG)
        cache = ROOT / "artifacts" / "p3" / "features_all20_v1"
        fallback_dir = ROOT / "artifacts" / "p3_corrected_repeated_forward_catboost_v2"
        features = pd.read_parquet(cache / "train_features.parquet")
        anchors = pd.read_parquet(cache / "train_anchors.parquet")
        fallback_oof = pd.read_parquet(fallback_dir / "oof.parquet")
        reference_keys = pd.read_parquet(fallback_dir / "validation_keys.parquet")
        if len(features) != 24_360 or len(anchors) != 24_360 or len(fallback_oof) != 1_086:
            raise P3TabPFNContractError("clean train or validation row count changed")
        if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
            raise P3TabPFNContractError("clean feature/anchor alignment changed")

        wave = pd.read_csv(data_dir / "train_wave.csv")
        wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
        anchors = assign_storm_episodes_from_wave(anchors, wave)
        folds, selected, split_audit = build_corrected_repeated_forward_folds(
            anchors,
            windows=clean["validation"]["windows"],
            gap_hours=int(clean["validation"]["embargo_hours"]),
            footprint_hours=int(clean["validation"]["footprint_hours"]),
        )
        assert_validation_surface_matches(selected, reference_keys)
        feature_columns = compact_feature_columns(
            [column for column in features if column not in {"anchor_id", "station"}]
        )
        if len(feature_columns) != 591:
            raise P3TabPFNContractError("clean feature surface is not 591 columns")

        regressor_path = Path(first["tabpfn"]["weights"]["regressor"]["path"])
        predictions: list[pd.DataFrame] = []
        fold_receipts: list[dict[str, Any]] = []
        for fold_index, fold in enumerate(folds):
            train_x, train_delta, train_meta = expand_leads(
                features, anchors, fold.train_ids, feature_columns
            )
            valid_x, _valid_delta, valid_meta = expand_leads(
                features, anchors, fold.validation_ids, feature_columns
            )
            challenger = np.empty(len(valid_x), dtype=np.float64)
            lead_receipts: list[dict[str, Any]] = []
            for lead_index, lead in enumerate(LEADS):
                train_mask = train_x["lead_h"].eq(lead).to_numpy()
                valid_mask = valid_x["lead_h"].eq(lead).to_numpy()
                train_matrix = lead_specific_matrix(train_x, lead=int(lead))
                valid_matrix = lead_specific_matrix(valid_x, lead=int(lead))
                if train_matrix.shape[1] != 593 or valid_matrix.shape[1] != 593:
                    raise P3TabPFNContractError("runtime P3 feature width is not 593")
                model = make_regressor(
                    regressor_path,
                    seed=20260901 + fold_index * 10 + lead_index,
                    categorical_features_indices=[0],
                    n_estimators=int(config["model"]["n_estimators"]),
                )
                model.fit(train_matrix, train_delta[train_mask])
                residual = np.asarray(model.predict(valid_matrix), dtype=np.float64)
                current = valid_meta.loc[valid_mask, "current_hs"].to_numpy(dtype=np.float64)
                challenger[valid_mask] = current + residual
                fit_count += 1
                lead_receipts.append(
                    {
                        "lead_h": int(lead),
                        "train_rows": int(len(train_matrix)),
                        "validation_rows": int(len(valid_matrix)),
                    }
                )
                (OUTPUT / "progress.json").write_text(
                    json.dumps(
                        {
                            "fold": fold.name,
                            "lead_h": int(lead),
                            "fit_count": fit_count,
                            "maximum_fits": 18,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            challenger = np.clip(
                challenger,
                float(config["model"]["prediction_clip_m"][0]),
                float(config["model"]["prediction_clip_m"][1]),
            )
            frame = valid_meta.copy()
            frame["fold"] = fold.name
            frame["tabpfn_prediction"] = challenger
            fallback = fallback_oof[
                ["anchor_id", "station", "lead_h", "target_hs", "final_prediction"]
            ].rename(
                columns={
                    "target_hs": "fallback_target_hs",
                    "final_prediction": "clean_fallback_prediction",
                }
            )
            frame = frame.merge(
                fallback,
                on=["anchor_id", "station", "lead_h"],
                how="left",
                validate="one_to_one",
            )
            if frame["clean_fallback_prediction"].isna().any():
                raise P3TabPFNContractError("clean fallback merge is incomplete")
            if not np.array_equal(
                frame["target_hs"].to_numpy(dtype=np.float64),
                frame["fallback_target_hs"].to_numpy(dtype=np.float64),
            ):
                raise P3TabPFNContractError("historical target differs from clean fallback")
            frame["candidate_prediction"] = blend_with_clean_fallback(
                frame["clean_fallback_prediction"].to_numpy(dtype=np.float64),
                frame["tabpfn_prediction"].to_numpy(dtype=np.float64),
                challenger_weight=float(config["model"]["tabpfn_weight"]),
            )
            predictions.append(frame)
            fold_receipts.append(
                {
                    "fold": fold.name,
                    "train_anchor_count": int(len(fold.train_ids)),
                    "validation_anchor_count": int(len(fold.validation_ids)),
                    "leads": lead_receipts,
                }
            )

        oof = pd.concat(predictions, ignore_index=True)
        if len(oof) != 1_086 or oof.duplicated(["anchor_id", "station", "lead_h"]).any():
            raise P3TabPFNContractError("TabPFN OOF surface changed")
        fallback_metrics = metric_slices(
            oof, oof["clean_fallback_prediction"].to_numpy(dtype=np.float64)
        )
        candidate_metrics = metric_slices(
            oof, oof["candidate_prediction"].to_numpy(dtype=np.float64)
        )
        delta = float(candidate_metrics["rmse"] - fallback_metrics["rmse"])
        oof_path = OUTPUT / "oof.parquet"
        oof.to_parquet(oof_path, index=False)
        result = {
            "schema_version": "p3.tabpfn26.structural_transition.result.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": (
                "COMPLETE_GO_CLEAN_HISTORICAL"
                if delta < 0.0
                else "COMPLETE_NO_GO_CLEAN_HISTORICAL"
            ),
            "fit_count": fit_count,
            "runtime_seconds": time.perf_counter() - started,
            "metrics": {
                "clean_fallback": fallback_metrics,
                "candidate": candidate_metrics,
                "delta_rmse_m": delta,
            },
            "fold_receipts": fold_receipts,
            "split_audit": split_audit,
            "oof": {"rows": int(len(oof)), "sha256": sha256(oof_path)},
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
            "schema_version": "p3.tabpfn26.structural_transition.terminal.v1",
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
