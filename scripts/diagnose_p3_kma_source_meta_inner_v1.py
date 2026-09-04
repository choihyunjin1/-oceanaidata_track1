"""Read-only diagnostics for the completed P3 KMA source-meta inner gate.

This script never fits a model and never opens a current-fold validation target.
It evaluates the already sealed source predictions only on each outer fold's
inner-calibration IDs, matching the rolling-origin label scope of the one-shot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset

from p3_wave.kma_source_meta import (
    LEADS,
    load_preregistration,
    read_frozen_outer_key_membership,
    sha256_file,
    validate_outer_membership_against_anchors,
)
from p3_wave.revin_patch import (
    assign_storm_episodes_from_wave,
    build_episode_disjoint_folds_from_ids,
    build_inner_episode_split,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/p3_kma_source_prediction_meta_v1.json"
ONE_SHOT = ROOT / "artifacts/p3_kma_source_prediction_meta_v1/one_shot"
OUTPUT = (
    ROOT
    / "artifacts/p3_kma_source_prediction_meta_v1/diagnostics/inner_source_meta_diagnostics.json"
)


class InnerTargetVault:
    """Filtered target reader enforcing the preregistered rolling label scope."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.access_log: list[dict[str, Any]] = []

    def read_inner_calibration(
        self,
        anchor_ids: np.ndarray,
        *,
        forbidden_current_validation_ids: np.ndarray,
        all_outer_validation_ids: np.ndarray,
        allowed_prior_validation_ids: np.ndarray,
        fold: str,
    ) -> pd.DataFrame:
        ids = np.asarray(anchor_ids, dtype=np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("inner target request must contain unique IDs")
        if np.intersect1d(ids, forbidden_current_validation_ids).size:
            raise PermissionError("current-fold validation target requested by diagnostics")
        outer_overlap = np.intersect1d(ids, all_outer_validation_ids)
        if np.setdiff1d(outer_overlap, allowed_prior_validation_ids).size:
            raise PermissionError("future outer validation target requested by diagnostics")
        columns = ["anchor_id", *[f"target_{lead}" for lead in LEADS]]
        dataset = pyarrow_dataset.dataset(self.path, format="parquet")
        table = dataset.to_table(
            columns=columns,
            filter=pyarrow_dataset.field("anchor_id").isin(ids.tolist()),
        )
        frame = table.to_pandas().set_index("anchor_id").loc[ids].reset_index()
        if len(frame) != len(ids) or frame["anchor_id"].duplicated().any():
            raise ValueError("filtered inner target read is incomplete")
        self.access_log.append(
            {
                "purpose": "fold_local_inner_calibration_diagnostic",
                "fold": fold,
                "rows": int(len(frame)),
                "current_validation_overlap_rows": 0,
                "permitted_prior_validation_history_rows": int(len(outer_overlap)),
            }
        )
        return frame


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    source = frame["source_prediction"].to_numpy(dtype=np.float64)
    persistence = frame["current_hs"].to_numpy(dtype=np.float64)
    true_residual = truth - persistence
    source_residual = source - persistence
    source_rmse = _rmse(truth, source)
    persistence_rmse = _rmse(truth, persistence)
    if len(frame) > 1 and np.std(source_residual) > 0.0 and np.std(true_residual) > 0.0:
        pearson = float(np.corrcoef(source_residual, true_residual)[0, 1])
        spearman = float(
            pd.Series(source_residual).corr(pd.Series(true_residual), method="spearman")
        )
        scale = float(
            np.cov(source_residual, true_residual, ddof=0)[0, 1] / np.var(source_residual)
        )
    else:
        pearson = None
        spearman = None
        scale = None
    intercept = (
        None if scale is None else float(np.mean(true_residual) - scale * np.mean(source_residual))
    )
    return {
        "rows": int(len(frame)),
        "cases": int(frame["anchor_id"].nunique()),
        "source_rmse": source_rmse,
        "persistence_rmse": persistence_rmse,
        "source_minus_persistence_rmse": float(source_rmse - persistence_rmse),
        "source_mae": float(np.mean(np.abs(source - truth))),
        "persistence_mae": float(np.mean(np.abs(persistence - truth))),
        "source_absolute_bias": float(np.mean(source - truth)),
        "source_residual_mean": float(np.mean(source_residual)),
        "true_residual_mean": float(np.mean(true_residual)),
        "residual_pearson": pearson,
        "residual_spearman": spearman,
        "calibration_definition": "true_residual = intercept + scale * source_residual",
        "calibration_intercept": intercept,
        "calibration_scale": scale,
    }


def _by(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, group in frame.groupby(grouper, observed=True, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        name = "|".join(str(value) for value in values)
        output[name] = _metrics(group)
    return output


def run(*, p3_data_dir: Path, output: Path) -> dict[str, Any]:
    config = load_preregistration(CONFIG)
    source_meta_path = ONE_SHOT / "source_meta_predictions.parquet"
    gate_path = ONE_SHOT / "inner_utility_gate.json"
    result_path = ONE_SHOT / "result.json"
    attempt_lock = ROOT / config["artifacts"]["global_attempt_lock"]
    forbidden_outer_artifacts = (
        ROOT / config["artifacts"]["global_outer_lock"],
        ROOT / config["artifacts"]["canonical_outer_lock"],
        ONE_SHOT / "blind_predictions.parquet",
        ONE_SHOT / "blind_manifest.json",
        ONE_SHOT / "outer_exposure_receipt.json",
    )
    if not attempt_lock.is_file() or any(path.exists() for path in forbidden_outer_artifacts):
        raise PermissionError("diagnostics require a pre-outer stop with only the attempt lock")

    wave = pd.read_csv(p3_data_dir / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchor_path = ROOT / config["frozen_inputs"]["anchor_metadata_and_vault"]["path"]
    anchors = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", "station", "anchor_time", "grid_position", "current_hs"],
    )
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    fold_names = [str(item[0]) for item in config["validation"]["windows"]]
    frozen_keys, membership = read_frozen_outer_key_membership(
        ROOT / config["frozen_inputs"]["incumbent_oof_keys"]["path"],
        expected_folds=fold_names,
    )
    validate_outer_membership_against_anchors(frozen_keys, anchors)
    folds = build_episode_disjoint_folds_from_ids(
        anchors,
        windows=config["validation"]["windows"],
        validation_ids_by_fold=membership,
        embargo_hours=78,
    )
    all_outer_ids = np.unique(np.concatenate([fold.validation_ids for fold in folds]))
    source_meta = pd.read_parquet(source_meta_path)
    if source_meta["anchor_id"].duplicated().any():
        raise ValueError("source meta contains duplicate anchor IDs")

    vault = InnerTargetVault(anchor_path)
    prior_validation_ids = np.asarray([], dtype=np.int64)
    rows: list[pd.DataFrame] = []
    for fold in folds:
        inner = build_inner_episode_split(
            anchors,
            fold.train_ids,
            validation_days=45,
            embargo_hours=78,
        )
        targets = vault.read_inner_calibration(
            inner.validation_ids,
            forbidden_current_validation_ids=fold.validation_ids,
            all_outer_validation_ids=all_outer_ids,
            allowed_prior_validation_ids=prior_validation_ids,
            fold=f"{fold.name}_postrun_inner_diagnostic",
        )
        label_free = anchors.loc[
            anchors["anchor_id"].isin(inner.validation_ids),
            ["anchor_id", "station", "current_hs"],
        ]
        base = label_free.merge(targets, on="anchor_id", how="inner", validate="one_to_one")
        base = base.merge(source_meta, on="anchor_id", how="inner", validate="one_to_one")
        if len(base) != len(inner.validation_ids):
            raise ValueError("inner diagnostic source/target coverage is incomplete")
        for lead in LEADS:
            block = base[
                [
                    "anchor_id",
                    "station",
                    "current_hs",
                    f"target_{lead}",
                    f"kma_source_hs_pred_{lead}h",
                ]
            ].rename(
                columns={
                    f"target_{lead}": "target_hs",
                    f"kma_source_hs_pred_{lead}h": "source_prediction",
                }
            )
            block.insert(0, "fold", fold.name)
            block["lead_h"] = lead
            rows.append(block)
        prior_validation_ids = np.union1d(prior_validation_ids, fold.validation_ids)

    evaluated = pd.concat(rows, ignore_index=True)
    values = evaluated[["target_hs", "source_prediction", "current_hs"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("inner diagnostic frame contains non-finite values")
    pooled = _metrics(evaluated)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    pooled["source_minus_saved_control_rmse"] = float(
        pooled["source_rmse"] - gate["pooled_control_rmse"]
    )
    pearson = pooled["residual_pearson"]
    scale = pooled["calibration_scale"]
    if pearson is not None and abs(pearson) < 0.1 and pooled["source_minus_persistence_rmse"] >= 0:
        diagnosis = "primarily_weak_or_absent_transfer_signal"
    elif (
        pearson is not None
        and abs(pearson) >= 0.2
        and scale is not None
        and abs(scale - 1.0) > 0.25
    ):
        diagnosis = "transfer_signal_present_but_calibration_mismatch_is_plausible"
    else:
        diagnosis = "mixed_weak_signal_and_calibration_mismatch"

    payload = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "scope": "postrun_read_only_inner_calibration_diagnostics",
        "outer_truth_opened": False,
        "model_refit_count": 0,
        "threshold_or_router_selection_count": 0,
        "inputs_sha256": {
            "config": sha256_file(CONFIG),
            "source_meta_predictions": sha256_file(source_meta_path),
            "inner_utility_gate": sha256_file(gate_path),
            "one_shot_result": sha256_file(result_path),
            "global_attempt_lock": sha256_file(attempt_lock),
        },
        "fold_local_access_log": vault.access_log,
        "saved_target_model_ablation": gate,
        "raw_source_meta_diagnostics": {
            "pooled": pooled,
            "by_fold": _by(evaluated, ["fold"]),
            "by_station": _by(evaluated, ["station"]),
            "by_lead": _by(evaluated, ["lead_h"]),
            "by_fold_station_lead": _by(evaluated, ["fold", "station", "lead_h"]),
        },
        "unavailable_without_prohibited_refit": {
            "control_vs_challenger_delta_by_station_or_lead": True,
            "six_meta_target_catboost_feature_importance": True,
            "reason": "inner target models and row-level predictions were not persisted before the preregistered pre-outer stop",
        },
        "diagnosis": diagnosis,
        "interpretation_limit": "diagnoses the fixed raw source predictions and saved aggregate inner ablation only; it does not authorize a calibration fit, new model, or outer access",
    }
    _atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p3-data-dir", required=True)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    payload = run(p3_data_dir=Path(args.p3_data_dir).resolve(), output=Path(args.output).resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
