"""Reproduce the frozen P1 e125-only disagreement cell on historical OOF folds."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "reports"
    / "parallel_breakthrough_deep_research_20260828_v14"
    / "p1_checkpoint_disagreement_retroaudit.json"
)
KEYS = ["station", "year", "layer", "time", "fold"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def f1_counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "f1": float(2 * tp / denominator)}


def event_lengths(frame: pd.DataFrame, mask: np.ndarray) -> list[int]:
    selected = frame.loc[mask, ["station", "layer", "time"]].copy()
    if selected.empty:
        return []
    selected["parsed"] = pd.to_datetime(selected["time"], utc=True)
    selected = selected.sort_values(["station", "layer", "parsed"])
    same_station = selected["station"].astype(str).eq(selected["station"].astype(str).shift())
    same_layer = selected["layer"].eq(selected["layer"].shift())
    same_gap = selected["parsed"].diff().eq(pd.Timedelta(minutes=10))
    new_event = ~(same_station & same_layer & same_gap)
    event_id = new_event.cumsum()
    return [int(value) for value in selected.groupby(event_id, sort=False).size().tolist()]


def select_predictions(path: Path, fold: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if fold == "2025_q2":
            threshold = int(np.flatnonzero(np.isclose(archive["thresholds"], 0.8))[0])
            epoch125 = int(
                np.flatnonzero((archive["widths"] == 512) & (archive["epochs"] == 125))[0]
            )
            epoch150 = int(
                np.flatnonzero((archive["widths"] == 512) & (archive["epochs"] == 150))[0]
            )
            return (
                archive["candidate"][epoch125, threshold].astype(np.int8),
                archive["candidate"][epoch150, threshold].astype(np.int8),
            )
        epoch125 = int(np.flatnonzero(archive["epochs"] == 125)[0])
        epoch150 = int(np.flatnonzero(archive["epochs"] == 150)[0])
        return (
            archive["candidate"][epoch125].astype(np.int8),
            archive["candidate"][epoch150].astype(np.int8),
        )


def main() -> None:
    anchor_path = ROOT / "artifacts" / "p1_current_router_oof_anchor_v1" / "anchor.parquet"
    truth_path = ROOT / "artifacts" / "runs" / "20260813T153038+0900_cv_378a4e89" / "oof.parquet"
    paths = {
        "2025_q2": ROOT
        / "artifacts"
        / "p1_incumbent_preserving_mstcn_asrf_v2"
        / "q2_qualification_grid_blind.npz",
        "2025_q3": ROOT
        / "artifacts"
        / "p1_mstcn_checkpoint_diagnostic_20260827_v2"
        / "q3_blind_checkpoint_curve.npz",
        "2025_q4": ROOT
        / "artifacts"
        / "p1_mstcn_checkpoint_diagnostic_20260827_v2"
        / "q4_blind_checkpoint_curve.npz",
    }
    anchor = pd.read_parquet(anchor_path)
    truth = pd.read_parquet(truth_path)
    require(anchor[KEYS].astype(str).equals(truth[KEYS].astype(str)), "anchor/truth keys differ")
    fold_results = {}
    pooled_truth = []
    pooled_e150 = []
    pooled_union = []
    pooled_signature_truth = []
    all_event_lengths = []
    all_event_types = []
    for fold, path in paths.items():
        prediction125, prediction150 = select_predictions(path, fold)
        anchor_fold = anchor.loc[anchor["fold"].eq(fold)].reset_index(drop=True)
        truth_fold = truth.loc[truth["fold"].eq(fold), ["label", "anomaly_type"]].reset_index(
            drop=True
        )
        frame = pd.concat([anchor_fold, truth_fold], axis=1)
        require(len(frame) == len(prediction125) == len(prediction150), f"fold length: {fold}")
        router = frame["current_router_prediction"].to_numpy(dtype=np.int8)
        signature = (
            (prediction125 == 1)
            & (prediction150 == 0)
            & (router == 0)
            & frame["station"].astype(str).eq("I-ORS").to_numpy()
            & frame["layer"].eq(5).to_numpy()
        )
        labels = frame["label"].to_numpy(dtype=np.int8)
        union = np.maximum(prediction150, signature.astype(np.int8))
        baseline = f1_counts(labels, prediction150)
        candidate = f1_counts(labels, union)
        lengths = event_lengths(frame, signature)
        types = sorted(frame.loc[signature, "anomaly_type"].dropna().astype(str).unique().tolist())
        fold_results[fold] = {
            "rows": len(frame),
            "signature_rows": int(signature.sum()),
            "signature_true_positive_rows": int(labels[signature].sum()),
            "signature_precision": float(labels[signature].mean()) if signature.any() else None,
            "event_lengths": lengths,
            "event_types": types,
            "epoch150": baseline,
            "limited_union": candidate,
            "delta_f1": float(candidate["f1"] - baseline["f1"]),
        }
        pooled_truth.append(labels)
        pooled_e150.append(prediction150)
        pooled_union.append(union)
        pooled_signature_truth.append(labels[signature])
        all_event_lengths.extend(lengths)
        all_event_types.extend(types)

    truth_all = np.concatenate(pooled_truth)
    e150_all = np.concatenate(pooled_e150)
    union_all = np.concatenate(pooled_union)
    signature_truth = np.concatenate(pooled_signature_truth)
    baseline_all = f1_counts(truth_all, e150_all)
    candidate_all = f1_counts(truth_all, union_all)
    require([fold_results[fold]["signature_rows"] for fold in paths] == [8, 0, 36], "signature checksum")
    require(int(signature_truth.sum()) == 44 and len(signature_truth) == 44, "TP checksum")
    require(all_event_lengths == [8, 36], "event checksum")
    result = {
        "schema_version": "p1.checkpoint_disagreement_cell.retroaudit.20260828.v1",
        "status": "PASS_GO_BOUNDED_FULL_REPLAY",
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "folds": fold_results,
        "pooled": {
            "signature_rows": len(signature_truth),
            "signature_true_positive_rows": int(signature_truth.sum()),
            "signature_precision": float(signature_truth.mean()),
            "event_lengths": all_event_lengths,
            "event_types": sorted(set(all_event_types)),
            "epoch150": baseline_all,
            "limited_union": candidate_all,
            "delta_f1": float(candidate_all["f1"] - baseline_all["f1"]),
        },
        "input_hashes": {
            "anchor": sha256(anchor_path),
            "truth": sha256(truth_path),
            **{fold: sha256(path) for fold, path in paths.items()},
        },
        "selection_limitations": {
            "retrospective_subgroup": True,
            "independent_events": 2,
            "fresh_confirmatory_fold": False,
            "official_upload_performed": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(json.dumps(result["pooled"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
