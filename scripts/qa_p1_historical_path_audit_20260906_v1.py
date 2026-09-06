"""Separate-process receipt/hash/count QA; never fit or produce a candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def run(root, directory):
    import pandas as pd

    output = directory / "independent-qa.json"
    if output.exists():
        raise FileExistsError(output)
    names = ["result.json", "saved-replay-result.json", "full-replay-preflight.json",
             "full-replay-result.json"]
    receipts = {name: json.loads((directory / name).read_text(encoding="utf-8")) for name in names}
    historical, saved, preflight, full = (receipts[name] for name in names)
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    scripts = {
        "result.json": "p1_historical_path_audit_20260906_v1.py",
        "saved-replay-result.json": "p1_historical_saved_replay_20260906_v1.py",
        "full-replay-preflight.json": "p1_historical_full_replay_20260906_v1.py",
        "full-replay-result.json": "p1_historical_full_replay_20260906_v1.py",
    }
    for name, script in scripts.items():
        check(f"script_pin:{name}", sha(root / "scripts" / script) == receipts[name]["script_sha256"])
    for name, digest in full["pinned_sha256"].items():
        check(f"input_pin:{Path(name).relative_to(root) if Path(name).is_relative_to(root) else Path(name).name}",
              sha(Path(name)) == digest)
    check("preflight_execution_pins_equal", preflight["pinned_sha256"] == full["pinned_sha256"])
    check("full_replay_pass", full["status"] == "FULL_ORIGINAL_SAVED_WEIGHT_REPLAY_EXACT_PASS")
    check("saved_tree_pass", saved["status"] == "SAVED_O_B_ROUTER_EXACT_REPLAY_PASS")
    check("independent_tree_runs_equal", full["checks"]["tree_sha256"] ==
          {k: saved["checks"][k]["sha256"] for k in ["O", "B", "router"]})
    for stage in ["base", "final"]:
        check(f"{stage}_hash_match", full["checks"][f"{stage}_hash_match"])
        validator = full["checks"][f"{stage}_validator"]
        check(f"{stage}_rows_keys", validator["rows"] == 169011 and validator["test_order_match"])
    check("general_rule_count", full["checks"]["general_GI_rule_additions"] ==
          full["checks"]["final_validator"]["positive"] -
          full["checks"]["base_validator"]["positive"] == 2)
    # Read historical training OOF only, independently recompute the confusion counts.
    paths = list(historical["hashes"])
    pred = pd.read_parquet(root / paths[0])
    truth = pd.read_parquet(root / paths[1], columns=["label"])
    anchor = pd.read_parquet(root / paths[2], columns=["current_router_prediction"])
    y = truth.label.astype(int)
    for name, bits in {"O": pred["incumbent_offline_xgboost__default"],
                       "B": pred["event_day_balanced_lightgbm__default"],
                       "router": anchor.current_router_prediction}.items():
        tp = int(((y == 1) & (bits == 1)).sum())
        fp = int(((y == 0) & (bits == 1)).sum())
        fn = int(((y == 1) & (bits == 0)).sum())
        expected = historical["metrics"][name]
        check(f"oof_counts:{name}", [tp, fp, fn] == [expected[k] for k in ["tp", "fp", "fn"]])
        check(f"oof_f1:{name}", math.isclose(2 * tp / (2 * tp + fp + fn), expected["f1"],
                                           rel_tol=0, abs_tol=1e-15))
    o_receipt = root / "artifacts/runs/20260813T155254+0900_train_378a4e89/model_metadata.json"
    o_meta = json.loads(o_receipt.read_text(encoding="utf-8"))
    o_model = root / "output/2026-08-20/retrain/P1/model.joblib"
    check("O_original_training_receipt_linked", o_meta["model_sha256"] == sha(o_model))
    for key in ["model_fits", "archived_answer_csv_reads", "row_patch_json_reads",
                "hidden_truth_reads", "candidate_csv_writes", "uploads"]:
        check(f"full_declared_zero:{key}", full[key] == 0)
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Separate-process receipt/hash/count verification, not a second full GPU "
                 "replay, model quality confirmation, or scratch-training audit.",
        "checks": checks, "check_count": len(checks), "pid": os.getpid(),
        "receipts_sha256": {name: sha(directory / name) for name in names},
        "qa_script_sha256": sha(Path(__file__)),
        "O_original_receipt_sha256": sha(o_receipt),
        "supersedes_initial_NOT_YET_LINKED_O_field": True,
        "access_scope": "Pinned artifact byte hashing, historical TRAIN OOF aggregates, "
                        "JSON receipts. No official predictions or hidden truth used as QA targets.",
        "remaining": ["Source-only feature regeneration not proved by this audit",
                      "Full empty-model-folder retraining not done here",
                      "Original executable OOF cell-selection algorithm not recovered",
                      "Organizer final reproducibility approval not claimed"],
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "checks": len(checks)}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.directory.resolve())
