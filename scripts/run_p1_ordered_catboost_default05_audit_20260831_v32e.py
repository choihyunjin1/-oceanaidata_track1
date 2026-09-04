"""One-shot threshold-0.5 audit of the immutable v32a historical OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_ordered_catboost_eventday_20260831_v32a as source  # noqa: E402

EXPERIMENT_ID = "p1_ordered_catboost_default05_audit_20260831_v32e"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_contract() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    path = ROOT / config["source_oof"]
    checks = {
        "id": config["experiment_id"] == EXPERIMENT_ID,
        "threshold": config["decision_threshold"] == 0.5,
        "fit0": config["fit_budget"] == 0,
        "source_hash": source.sha256_file(path) == config["source_oof_sha256"],
        "official0": all(value == 0 for value in config["official_budget"].values()),
    }
    if not all(checks.values()):
        raise source.ContractError(f"v32e contract mismatch: {checks}")
    return config


def concentration(metadata: pd.DataFrame, changed: np.ndarray) -> dict:
    count = int(changed.sum())
    if count == 0:
        return {"maximum_station_layer_fold_share": 0.0, "maximum_station_layer_fold_rows": 0}
    grouped = metadata.loc[changed].groupby(["station", "layer", "fold"], observed=True).size()
    maximum = int(grouped.max())
    return {"maximum_station_layer_fold_share": maximum / count, "maximum_station_layer_fold_rows": maximum}


def execute() -> dict:
    started = time.perf_counter()
    config = load_contract()
    source_path = ROOT / config["source_oof"]
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "fit_budget": 0,
        "config_sha256": source.sha256_file(CONFIG),
        "runner_sha256": source.sha256_file(Path(__file__)),
    }
    source.write_json_new(ARTIFACT / "attempt_lock.json", lock)

    # Deliberately exclude the historical label until the action mask is sealed.
    blind = pd.read_parquet(
        source_path,
        columns=[*source.KEYS, "fold", "deployment_prediction", "candidate_probability"],
    )
    if len(blind) != 421_032 or blind.duplicated(source.KEYS).any():
        raise source.ContractError("v32a OOF key contract changed")
    probability = blind["candidate_probability"].to_numpy(np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise source.ContractError("invalid sealed probability")
    action = (probability >= 0.5).astype(np.int8)
    action_path = ARTIFACT / "sealed_default05_action.npz"
    np.savez_compressed(action_path, action=action)
    seal = {
        "rows": len(action),
        "threshold": 0.5,
        "positive_actions": int(action.sum()),
        "action_sha256": sha256_array(action),
        "npz_sha256": source.sha256_file(action_path),
        "config_sha256": source.sha256_file(CONFIG),
        "runner_sha256": source.sha256_file(Path(__file__)),
        "truth_columns_read_before_seal": 0,
        "official_reads": 0,
    }
    source.write_json_new(ARTIFACT / "action_seal.json", seal)

    truth_frame = pd.read_parquet(source_path, columns=[*source.KEYS, "fold", "label"])
    if not blind[[*source.KEYS, "fold"]].equals(truth_frame[[*source.KEYS, "fold"]]):
        raise source.ContractError("truth attachment changed sealed row order")
    oof = blind.copy()
    oof["label"] = truth_frame["label"].to_numpy(np.int8)
    tabular, e150 = source.aligned_references(oof)
    truth = oof["label"].to_numpy(np.int8)
    metadata = oof[["station", "layer", "time"]].reset_index(drop=True)
    by_fold = {}
    fold_gates = []
    for fold in config["folds"]:
        mask = oof["fold"].eq(fold).to_numpy()
        candidate_metric = source.metric(truth[mask], action[mask])
        reference_metric = source.metric(truth[mask], tabular[mask])
        delta = candidate_metric["f1"] - reference_metric["f1"]
        by_fold[fold] = {
            "candidate": candidate_metric,
            "tabular_reference": reference_metric,
            "delta_f1_vs_tabular": delta,
        }
        fold_gates.append(delta >= 0.0)
    all_mask = oof["fold"].isin(config["folds"]).to_numpy()
    pooled_candidate = source.metric(truth[all_mask], action[all_mask])
    pooled_reference = source.metric(truth[all_mask], tabular[all_mask])
    pooled_delta = pooled_candidate["f1"] - pooled_reference["f1"]
    pooled_bootstrap = source.paired_bootstrap(
        truth[all_mask],
        action[all_mask],
        tabular[all_mask],
        metadata.loc[all_mask].reset_index(drop=True),
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["bootstrap_seed"]),
    )
    q34 = oof["fold"].isin(["2025_q3", "2025_q4"]).to_numpy()
    q34_candidate = source.metric(truth[q34], action[q34])
    q34_reference = source.metric(truth[q34], e150[q34])
    q34_delta = q34_candidate["f1"] - q34_reference["f1"]
    q34_bootstrap = source.paired_bootstrap(
        truth[q34],
        action[q34],
        e150[q34],
        metadata.loc[q34].reset_index(drop=True),
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["bootstrap_seed"]) + 1,
    )
    changed = action != tabular
    removals = (tabular == 1) & (action == 0)
    additions = (tabular == 0) & (action == 1)
    gates = {
        "each_q2_q3_q4_nonnegative_vs_tabular": all(fold_gates),
        "pooled_positive_vs_tabular": pooled_delta > 0.0,
        "pooled_ci90_low_positive_vs_tabular": pooled_bootstrap["difference_ci90"][0] > 0.0,
        "q3_q4_positive_vs_e150": q34_delta > 0.0,
        "q3_q4_ci90_low_positive_vs_e150": q34_bootstrap["difference_ci90"][0] > 0.0,
    }
    result = {
        "status": "PASS_INTERNAL_GATE" if all(gates.values()) else "TERMINAL_NO_GO",
        "fit_count": 0,
        "runtime_seconds": time.perf_counter() - started,
        "threshold": 0.5,
        "seal": seal,
        "by_fold": by_fold,
        "pooled_vs_tabular": {
            "candidate": pooled_candidate,
            "reference": pooled_reference,
            "delta_f1": pooled_delta,
            "bootstrap": pooled_bootstrap,
        },
        "q3_q4_vs_e150": {
            "candidate": q34_candidate,
            "reference": q34_reference,
            "delta_f1": q34_delta,
            "bootstrap": q34_bootstrap,
        },
        "expected_public_score": {
            "reference_points": source.PUBLIC_BEST_POINTS,
            "center": source.PUBLIC_BEST_POINTS + source.PUBLIC_SCORE_SLOPE * pooled_delta,
            "ci90": [
                source.PUBLIC_BEST_POINTS + source.PUBLIC_SCORE_SLOPE * pooled_bootstrap["difference_ci90"][0],
                source.PUBLIC_BEST_POINTS + source.PUBLIC_SCORE_SLOPE * pooled_bootstrap["difference_ci90"][1],
            ],
            "linear_translation_only": True,
        },
        "changes_vs_tabular": {
            "rows": int(changed.sum()),
            "fraction": float(changed.mean()),
            "additions": int(additions.sum()),
            "anchor_removals": int(removals.sum()),
            **concentration(oof, changed),
        },
        "gates": gates,
        "operations": {"official_reads": 0, "hidden_reads": 0, "csv": 0, "uploads": 0},
        "hashes": {
            "source_oof": source.sha256_file(source_path),
            "config": source.sha256_file(CONFIG),
            "runner": source.sha256_file(Path(__file__)),
            "lock": source.sha256_file(ARTIFACT / "attempt_lock.json"),
            "action": source.sha256_file(action_path),
        },
    }
    source.write_json_new(ARTIFACT / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PREFLIGHT_ONLY", "contract": load_contract()}, indent=2))
        return
    print(json.dumps(execute(), indent=2))


if __name__ == "__main__":
    main()
