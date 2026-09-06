"""Audit the historical P1 composition on TRAIN OOF; never emit an answer.

Restores source-defined Boolean operations, NOT the lost policy-selection
algorithm. Historical cells are passed explicitly for retrospective audit only.
No model unpickling, fitting, official inputs, archived answer CSV, or upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

KEYS = ["station", "year", "layer", "time"]
O_COL = "incumbent_offline_xgboost__default"
B_COL = "event_day_balanced_lightgbm__default"
ADD_CELLS = frozenset({("G-ORS", 1), ("I-ORS", 2)})
REMOVE_CELLS = frozenset({("S-ORS", 1), ("S-ORS", 5), ("S-ORS", 6), ("I-ORS", 4)})
PINS = {
    "artifacts/p1_matched_budget_local_compare_20260825_v1/predictions.parquet":
        "23f9b59cc54a7502c87786280ef76319e64288a95a44f8c9ec37188a761033c5",
    "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet":
        "d1b9439db6d0d906fa080bd01f1eb8fc21d051c3d056a274e2b02e43c1e55f4a",
    "artifacts/p1_current_router_oof_anchor_v1/anchor.parquet":
        "fc5c594aadabec98e0fbf032ff33422691d94d8525d134495a81a47586024536",
    "scripts/package_preregistered_submission_20260826.py":
        "36903d7693a28f43fe62ec8f28ca80ed4c002c5ad3a7ac8b126e32346ce1357e",
    "scripts/build_p1_current_router_oof_anchor_v1.py":
        "6235536f4cde74394579a1732225bb2de3375e28628cb962b3c410cc6335b08d",
    "scripts/build_preregistered_3x3_evidence_round_20260827.py":
        "82b4aaba2cf9e03c223142d12904dba572457aebe74b49d4cdc0b30d44f7b27f",
    "scripts/build_deadline_probe_set_20260828.py":
        "9687ce3a7ac7a457894215759da8a701fcd525e9c3aba70025278e7fab161cfd",
    "artifacts/p1_target_covariate_density_ratio_xgb_v1/models/"
    "P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1.joblib":
        "5f7933a6fa7e2c03a84ce255538a74fa4e697052c546e87af2d8c8a3982f4895",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def binary(values) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not np.isin(result, [0, 1]).all():
        raise ValueError("Expected finite binary vector")
    return result.astype(bool)


def compose_tree(station, layer, o, b, *, add_cells, remove_cells):
    """General original router and GI-without-removals, without answer keys."""
    o, b = binary(o), binary(b)
    station, layer = np.asarray(station), np.asarray(layer)
    if station.ndim != 1 or layer.ndim != 1 or not (
        len(station) == len(layer) == len(o) == len(b)
    ):
        raise ValueError("Mismatched input vectors")
    add = np.asarray([
        (str(s), int(depth_layer)) in add_cells
        for s, depth_layer in zip(station, layer, strict=True)
    ])
    remove = np.asarray([
        (str(s), int(depth_layer)) in remove_cells
        for s, depth_layer in zip(station, layer, strict=True)
    ])
    additions = o & ~b & add.astype(bool)
    removals = ~o & b & remove.astype(bool)
    gi = b | additions
    router = gi & ~removals
    return router.astype(np.int8), gi.astype(np.int8)


def compose_mstcn_spike(router, gi, mstcn, gi_type):
    """Original general novel-and-spike rule, no fixed row count or keys."""
    router, gi, mstcn = binary(router), binary(gi), binary(mstcn)
    types = np.asarray(gi_type)
    if types.ndim != 1 or not (len(router) == len(gi) == len(mstcn) == len(types)):
        raise ValueError("Mismatched composition vectors")
    base = router | mstcn
    additions = ~base & gi & (types == "spike")
    return (base | additions).astype(np.int8)


def metrics(y, p):
    y, p = binary(y), binary(p)
    if len(y) != len(p):
        raise ValueError("Mismatched metric vectors")
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    return {"rows": len(y), "tp": tp, "fp": fp, "fn": fn,
            "positive_rows": int(p.sum()), "f1": 2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn else 0.0}


def run(root: Path, output: Path):
    import pandas as pd

    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    hashes = {name: digest(root / name) for name in PINS}
    if hashes != PINS:
        raise RuntimeError("Historical source/artifact hash drift")
    paths = list(PINS)
    pred = pd.read_parquet(root / paths[0], columns=[*KEYS, "fold", O_COL, B_COL])
    truth = pd.read_parquet(root / paths[1], columns=[*KEYS, "fold", "label"])
    anchor = pd.read_parquet(root / paths[2])
    for frame in (pred, truth, anchor):
        if len(frame) != 421032 or frame.duplicated(KEYS).any():
            raise RuntimeError("Historical row/unique key drift")
    # Normalize only key representation, never reorder or drop rows.
    reference = pred[[*KEYS, "fold"]].astype(str)
    if not all(reference.equals(f[[*KEYS, "fold"]].astype(str)) for f in (truth, anchor)):
        raise RuntimeError("Ordered historical keys/folds differ")
    o, b = pred[O_COL].to_numpy(), pred[B_COL].to_numpy()
    router, _gi = compose_tree(
        pred.station, pred.layer, o, b, add_cells=ADD_CELLS, remove_cells=REMOVE_CELLS
    )
    if not np.array_equal(router, anchor.current_router_prediction.to_numpy()):
        raise RuntimeError("Historical router bits do not replay")
    y = truth.label.to_numpy()
    table = {name: metrics(y, value) for name, value in {
        "O": o, "B": b, "router": router, "union": binary(o) | binary(b)
    }.items()}
    folds = {}
    for fold in sorted(pred.fold.astype(str).unique()):
        mask = pred.fold.astype(str).eq(fold).to_numpy()
        folds[fold] = {name: metrics(y[mask], value[mask]) for name, value in
                       {"B": b, "router": router}.items()}
    result = {
        "status": "EXACT_HISTORICAL_ROUTER_OOF_REPLAY_PASS_NOT_FULL_RESTORATION",
        "hashes": hashes, "script_sha256": digest(Path(__file__)),
        "ordered_keys_and_folds_exact": True, "router_mismatch_rows": 0,
        "additions_vs_B": int(((router == 1) & (b == 0)).sum()),
        "removals_vs_B": int(((router == 0) & (b == 1)).sum()),
        "metrics": table, "folds": folds,
        "selection_scope": "Historical prescribed cells; original selector NOT recovered. "
                           "Same development OOF was used for historical selection; not fresh.",
        "original_O_model": {
            "path": "output/2026-08-20/retrain/P1/model.joblib",
            "current_audit_sha256": digest(root / "output/2026-08-20/retrain/P1/model.joblib"),
            "historical_receipt_hash_match": "NOT_YET_LINKED", "loaded": False},
        "GI_general_rule": "SOURCE_RESTORED_SYNTHETIC_ONLY_NOT_REAL_INFERENCE_VERIFIED",
        "official_inputs_read": 0, "archived_answer_csv_reads": 0,
        "model_deserializations": 0, "model_fits": 0, "candidate_csv_writes": 0,
        "uploads": 0, "hidden_truth_reads": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "metrics": table, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
