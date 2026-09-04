"""Exactly-once historical audit of a fixed E150/CatBoost precision union.

The action mask is formed and sealed without loading target labels.  The
candidate only adds fixed-threshold (0.8) Ordered-CatBoost positives to the
frozen E150 prediction and never removes an E150 positive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_p1_ordered_catboost_eventday_20260831_v32a as v32a  # noqa: E402

EXPERIMENT_ID = "p1_e150_catboost_precision_union_20260831_v32g"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
RESULT_PATH = ARTIFACT_DIR / "result.json"
LOCK_PATH = ARTIFACT_DIR / "attempt_lock.json"
PROPOSAL_PATH = ARTIFACT_DIR / "proposal.json"
SEALED_PATH = ARTIFACT_DIR / "sealed_candidate.npz"
KEYS = ["station", "year", "layer", "time"]
FOLDS = ["2025_q2", "2025_q3", "2025_q4"]
PUBLIC_BEST_F1 = 0.833548
PUBLIC_BEST_POINTS = 28.909341
PUBLIC_SCORE_SLOPE = 26.578120867377286


class ContractError(RuntimeError):
    """Raised when the frozen experiment contract is violated."""


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def wilson_lower(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((center - radius) / denominator)


def execute() -> dict[str, Any]:
    started_wall = now_kst()
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if started_wall >= datetime.fromisoformat(config["deadline_kst"]):
        raise ContractError("deadline already passed")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_new(
        LOCK_PATH,
        {
            "experiment_id": EXPERIMENT_ID,
            "started_kst": started_wall.isoformat(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "exactly_once": True,
        },
    )
    oof_path = ROOT / config["inputs"]["catboost_oof"]
    if sha256_file(oof_path) != config["inputs"]["catboost_oof_sha256"]:
        raise ContractError("CatBoost OOF hash mismatch")

    # Phase 1: no label column is loaded.  Form and persist the complete action mask.
    features = pd.read_parquet(
        oof_path,
        columns=[*KEYS, "fold", "deployment_prediction", "candidate_probability"],
    )
    if len(features) != 421_032 or features.duplicated(KEYS).any():
        raise ContractError("OOF key contract failed")
    if features["fold"].drop_duplicates().tolist() != FOLDS:
        raise ContractError("unexpected fold order")
    _, e150_prediction = v32a.aligned_references(features)
    threshold = float(config["candidate"]["catboost_probability_threshold"])
    catboost_positive = features["candidate_probability"].to_numpy(np.float64) >= threshold
    additions = (e150_prediction == 0) & catboost_positive
    candidate = np.maximum(e150_prediction, catboost_positive.astype(np.int8)).astype(np.int8)
    if np.any((e150_prediction == 1) & (candidate == 0)):
        raise ContractError("add-only contract violated")
    np.savez_compressed(
        SEALED_PATH,
        candidate=candidate,
        additions=additions.astype(np.uint8),
        e150_prediction=e150_prediction,
    )
    proposal = {
        "rows": int(len(features)),
        "threshold": threshold,
        "additions": int(additions.sum()),
        "candidate_sha256": sha256_array(candidate),
        "additions_sha256": sha256_array(additions.astype(np.uint8)),
        "sealed_npz_sha256": sha256_file(SEALED_PATH),
        "target_columns_read_before_seal": 0,
        "official_reads": 0,
    }
    write_json_new(PROPOSAL_PATH, proposal)

    # Phase 2: evaluate the immutable mask against historical labels exactly once.
    labelled = pd.read_parquet(oof_path, columns=[*KEYS, "fold", "label"])
    if not features[[*KEYS, "fold"]].equals(labelled[[*KEYS, "fold"]]):
        raise ContractError("label alignment failed")
    truth = labelled["label"].to_numpy(np.int8)
    metadata = features[["station", "layer", "time"]].reset_index(drop=True)
    by_fold: dict[str, Any] = {}
    for fold in FOLDS:
        mask = features["fold"].eq(fold).to_numpy()
        candidate_metric = v32a.metric(truth[mask], candidate[mask])
        reference_metric = v32a.metric(truth[mask], e150_prediction[mask])
        fold_add = additions & mask
        tp_add = int(np.sum(fold_add & (truth == 1)))
        fp_add = int(np.sum(fold_add & (truth == 0)))
        by_fold[fold] = {
            "candidate": candidate_metric,
            "reference": reference_metric,
            "delta_f1": float(candidate_metric["f1"] - reference_metric["f1"]),
            "additions": int(fold_add.sum()),
            "true_positive_additions": tp_add,
            "false_positive_additions": fp_add,
        }
    candidate_metric = v32a.metric(truth, candidate)
    reference_metric = v32a.metric(truth, e150_prediction)
    bootstrap = v32a.paired_bootstrap(
        truth,
        candidate,
        e150_prediction,
        metadata,
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["validation"]["bootstrap_seed"]),
    )
    q34 = features["fold"].isin(["2025_q3", "2025_q4"]).to_numpy()
    q34_candidate = v32a.metric(truth[q34], candidate[q34])
    q34_reference = v32a.metric(truth[q34], e150_prediction[q34])
    q34_bootstrap = v32a.paired_bootstrap(
        truth[q34],
        candidate[q34],
        e150_prediction[q34],
        metadata.loc[q34].reset_index(drop=True),
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["validation"]["bootstrap_seed"]) + 1,
    )
    true_additions = int(np.sum(additions & (truth == 1)))
    false_additions = int(np.sum(additions & (truth == 0)))
    total_additions = true_additions + false_additions
    addition_precision = true_additions / total_additions if total_additions else 0.0
    addition_precision_lcb90 = wilson_lower(true_additions, total_additions)
    reference_half = float(reference_metric["f1"]) / 2.0
    changed_fraction = float(additions.mean())
    runtime = float(time.perf_counter() - started)
    gates = {
        "positive_additions": total_additions > 0,
        "anchor_removals_zero": True,
        "all_q2_q3_q4_nonnegative": all(item["delta_f1"] >= 0.0 for item in by_fold.values()),
        "q3_q4_each_nonnegative": all(by_fold[name]["delta_f1"] >= 0.0 for name in ("2025_q3", "2025_q4")),
        "pooled_delta_positive": candidate_metric["f1"] > reference_metric["f1"],
        "pooled_ci90_low_positive": bootstrap["difference_ci90"][0] > 0.0,
        "q3_q4_delta_positive": q34_candidate["f1"] > q34_reference["f1"],
        "q3_q4_ci90_low_positive": q34_bootstrap["difference_ci90"][0] > 0.0,
        "addition_precision_lcb_above_reference_f1_half": addition_precision_lcb90 > reference_half,
        "changed_fraction_at_most_limit": changed_fraction <= float(config["validation"]["maximum_changed_fraction"]),
        "runtime_within_cap": runtime <= float(config["maximum_runtime_seconds"]),
        "official_access_zero": True,
    }
    q34_delta = float(q34_candidate["f1"] - q34_reference["f1"])
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PROMOTE" if all(gates.values()) else "NO_GO_INTERNAL_GATE",
        "fit_count": 0,
        "runtime_seconds": runtime,
        "candidate": {
            "operation": "e150_or_catboost_probability_at_least_0.8",
            "threshold": threshold,
            "candidate_uses_incumbent_as_input": True,
            "anchor_removals": 0,
            "additions": total_additions,
            "true_positive_additions": true_additions,
            "false_positive_additions": false_additions,
            "addition_precision": addition_precision,
            "addition_precision_lcb90": addition_precision_lcb90,
            "reference_f1_half": reference_half,
            "changed_fraction": changed_fraction,
        },
        "by_fold": by_fold,
        "pooled": {
            "candidate": candidate_metric,
            "reference": reference_metric,
            "delta_f1": float(candidate_metric["f1"] - reference_metric["f1"]),
            "bootstrap": bootstrap,
        },
        "q3_q4": {
            "candidate": q34_candidate,
            "reference": q34_reference,
            "delta_f1": q34_delta,
            "bootstrap": q34_bootstrap,
        },
        "public_score_translation": {
            "assumption": "empirical local linear slope only; not a guarantee",
            "expected_points_center": PUBLIC_BEST_POINTS + PUBLIC_SCORE_SLOPE * q34_delta,
            "expected_points_ci90": [
                PUBLIC_BEST_POINTS + PUBLIC_SCORE_SLOPE * q34_bootstrap["difference_ci90"][0],
                PUBLIC_BEST_POINTS + PUBLIC_SCORE_SLOPE * q34_bootstrap["difference_ci90"][1],
            ],
            "target_32_f1": PUBLIC_BEST_F1 + (32.0 - PUBLIC_BEST_POINTS) / PUBLIC_SCORE_SLOPE,
        },
        "gates": gates,
        "seal": proposal,
        "official_access": config["official_access"],
        "hashes": {
            "config": sha256_file(CONFIG_PATH),
            "runner": sha256_file(Path(__file__)),
            "catboost_oof": sha256_file(oof_path),
            "proposal": sha256_file(PROPOSAL_PATH),
        },
        "completed_kst": now_kst().isoformat(),
    }
    write_json_new(RESULT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(CONFIG_PATH.read_text(encoding="utf-8"))
        return 0
    try:
        result = execute()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_kst": now_kst().isoformat(),
        }
        if not RESULT_PATH.exists():
            write_json_new(RESULT_PATH, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
