"""Verify and extract only frozen e150 historical proposals, not old answers.

This reuses already exposed train-derived Q3/Q4 predictions. It is not fresh
validation or a new model fit. Old router/candidate arrays are never consumed.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import mstcn
import numpy as np
import pandas as pd

HISTORY = ROOT / "artifacts/p1_mstcn_checkpoint_diagnostic_20260827_v2"
OOF = ROOT / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet"
OOF_SHA = "d1b9439db6d0d906fa080bd01f1eb8fc21d051c3d056a274e2b02e43c1e55f4a"


def extract(output: Path):
    if output.exists():
        raise FileExistsError("new output directory required")
    started = time.monotonic()
    source, config = mstcn.load_source(ROOT)
    if mstcn.sha(OOF) != OOF_SHA:
        raise ValueError("historical key source SHA mismatch")
    manifest = mstcn.read_json(HISTORY / "manifest.json")
    inventory = {item["path"]: item for item in manifest["files"]}
    checks = {"historical_oof_key_source_sha": True}

    def verify(name):
        path = (HISTORY / name).resolve()
        if path.parent != HISTORY.resolve() or name not in inventory:
            raise ValueError("unregistered historical artifact")
        item = inventory[name]
        passed = path.stat().st_size == item["bytes"] and mstcn.sha(path) == item["sha256"]
        checks[name] = passed
        if not passed:
            raise ValueError(f"historical SHA mismatch: {name}")
        return path

    recipe = mstcn.read_json(verify("selected_recipe.json"))
    parts, folds = [], {}
    for phase in ("q3", "q4"):
        receipt = mstcn.read_json(verify(f"{phase}_blind_checkpoint_curve_receipt.json"))
        split = mstcn.read_json(verify(f"{phase}_split.json"))
        verify(f"{phase}_encoder.json")
        if (receipt["scientific_metric_epoch"] != 150 or
                receipt["recipe_sha256"] != mstcn.sha(HISTORY / "selected_recipe.json")):
            raise ValueError("historical recipe identity mismatch")
        seeds = []
        for fit in receipt["fit_receipts"]:
            seeds.append(fit["seed"])
            if (fit["epochs_trained"], fit["source_schedule_horizon_epochs"],
                    fit["nonfinite_count_total"], fit["fresh_refit"]) != (150, 300, 0, True):
                raise ValueError("historical fit recipe mismatch")
            verify(fit["history_artifact"]["path"])
            states = [s for s in fit["state_artifacts"] if s["epoch"] == 150]
            if len(states) != 1:
                raise ValueError("missing fixed-epoch state")
            verify(states[0]["path"])
        if tuple(seeds) != mstcn.SEEDS:
            raise ValueError("historical seed mismatch")
        keys = source._read_fold_membership_without_truth(OOF, f"2025_{phase}")
        if (len(keys) != receipt["holdout_rows"] or
                source._ordered_key_sha(keys) != receipt["ordered_holdout_key_sha256"]):
            raise ValueError("historical prediction key order mismatch")
        path = verify(receipt["score_path"])
        if mstcn.sha(path) != receipt["score_sha256"]:
            raise ValueError("blind receipt probability linkage mismatch")
        with np.load(path, allow_pickle=False) as archive:
            epochs = archive["epochs"]
            selected = np.flatnonzero(epochs == 150)
            if len(selected) != 1:
                raise ValueError("fixed epoch missing or duplicated")
            index = int(selected[0])
            arrays = [archive[name][index].copy() for name in
                      ("row_probability", "boundary_probability", "type_probability")]
            stored = archive["proposal"][index].copy()
        if any(not np.isfinite(v).all() or not ((v >= 0) & (v <= 1)).all() for v in arrays):
            raise ValueError("invalid historical probabilities")
        if not np.isin(stored, [0, 1]).all():
            raise ValueError("nonbinary historical proposal")
        layout = source._load_scientific()[4].SegmentLayout.from_aligned(
            *(keys[k] for k in mstcn.KEYS))
        bundle = source.PredictionBundle(*arrays)
        decoded = source.decode_long_event_segments(
            source._decoder_row_probability(bundle, config), bundle.boundary_probability,
            layout, high_threshold=0.8, low_threshold=0.4,
            snap_radius=12, minimum_rows=19, maximum_rows=None)
        checks[f"{phase}_e150_decoder_semantic_replay"] = bool(np.array_equal(decoded, stored))
        if not checks[f"{phase}_e150_decoder_semantic_replay"]:
            raise ValueError("fixed decoder semantic replay mismatch")
        keys["proposal"] = decoded
        parts.append(keys)
        folds[phase] = {"rows": len(keys), "proposal_positive": int(decoded.sum()),
                        "ordered_source_key_sha256": source._ordered_key_sha(keys),
                        "training_max_time_utc": split["training_max_time_utc"],
                        "training_rows": split["training_rows"],
                        "fit_count_reused": 3, "new_fits": 0}
    joined = pd.concat(parts, ignore_index=True)
    if joined.duplicated(list(mstcn.KEYS)).any():
        raise ValueError("historical fold keys overlap")
    output.mkdir(parents=True)
    joined.to_parquet(output / "proposals.parquet", index=False)
    mstcn.save_json(output / "qa.json", {
        "status": "PASS", "role": "previously_exposed_retrospective_predictions_only",
        "new_fits": 0, "reused_historical_fits": 6, "checks": checks,
        "folds": folds, "rows": len(joined), "official_rows_read": 0,
        "historical_router_candidate_arrays_read": 0, "training_truth_columns_read": 0,
        "epochs_selected": [150], "recipe_sha256": mstcn.sha(HISTORY / "selected_recipe.json"),
        "recipe_evidence": recipe, "historical_manifest_sha256": mstcn.sha(HISTORY / "manifest.json"),
        "proposal_sha256": mstcn.sha(output / "proposals.parquet"),
        "runtime_seconds": time.monotonic() - started,
        "next_check": "Match new tree OOF keys; explicitly report any excluded rows, never implicit intersection",
    })
    print(json.dumps({"status": "PASS", "rows": len(joined), "checks": len(checks),
                      "new_fits": 0, "official_rows": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    extract(parser.parse_args().output.resolve())
