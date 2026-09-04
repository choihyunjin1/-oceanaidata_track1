"""One-shot union of frozen E150 and frozen CatBoost-default-0.5 actions."""

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
from scipy.stats import beta as beta_distribution

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as e150_source  # noqa: E402
import run_p1_ordered_catboost_default05_audit_20260831_v32e as audit_source  # noqa: E402
import run_p1_ordered_catboost_eventday_20260831_v32a as metric_source  # noqa: E402

EXPERIMENT_ID = "p1_e150_catboost_default05_union_20260831_v32h"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
FOLDS = metric_source.FOLDS


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_contract() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {
        "id": config["experiment_id"] == EXPERIMENT_ID,
        "fit0": config["fit_budget"] == 0,
        "search0": config["threshold_searches"] == 0,
        "action_hash": metric_source.sha256_file(ROOT / config["catboost_action"])
        == config["catboost_action_sha256"],
        "oof_hash": metric_source.sha256_file(ROOT / config["source_oof"])
        == config["source_oof_sha256"],
        "official0": all(value == 0 for value in config["official_budget"].values()),
    }
    if not all(checks.values()):
        raise metric_source.ContractError(f"v32h contract mismatch: {checks}")
    return config


def prediction_only_e150(blind: pd.DataFrame) -> np.ndarray:
    anchor = pd.read_parquet(e150_source.ANCHOR_PATH)
    output = np.full(len(blind), -1, dtype=np.int8)
    for fold, path in e150_source.FOLD_PATHS.items():
        archive = e150_source._select_archive_arrays(fold, path)  # noqa: SLF001
        fold_anchor = anchor.loc[anchor["fold"].eq(fold), [*metric_source.KEYS, "fold"]].copy()
        fold_anchor["e150"] = np.asarray(archive["candidate"], dtype=np.int8)
        aligned = blind.loc[blind["fold"].eq(fold), [*metric_source.KEYS, "fold"]].merge(
            fold_anchor,
            on=[*metric_source.KEYS, "fold"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if aligned["e150"].isna().any():
            raise metric_source.ContractError(f"E150 prediction-only alignment failed: {fold}")
        output[blind["fold"].eq(fold)] = aligned["e150"].to_numpy(np.int8)
    if (output < 0).any():
        raise metric_source.ContractError("E150 prediction-only surface incomplete")
    return output


def execute() -> dict:
    started = time.perf_counter()
    config = load_contract()
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "fit_budget": 0,
        "threshold_searches": 0,
        "config_sha256": metric_source.sha256_file(CONFIG),
        "runner_sha256": metric_source.sha256_file(Path(__file__)),
    }
    metric_source.write_json_new(ARTIFACT / "attempt_lock.json", lock)
    oof_path = ROOT / config["source_oof"]
    blind = pd.read_parquet(oof_path, columns=[*metric_source.KEYS, "fold"])
    catboost = np.load(ROOT / config["catboost_action"])["action"].astype(np.int8)
    if len(blind) != len(catboost) or not np.isin(catboost, [0, 1]).all():
        raise metric_source.ContractError("CatBoost action alignment contract failed")
    e150 = prediction_only_e150(blind)
    union = np.maximum(e150, catboost).astype(np.int8)
    union_path = ARTIFACT / "sealed_union_action.npz"
    np.savez_compressed(union_path, action=union)
    seal = {
        "rows": len(union),
        "positive_actions": int(union.sum()),
        "action_sha256": sha256_array(union),
        "npz_sha256": metric_source.sha256_file(union_path),
        "config_sha256": metric_source.sha256_file(CONFIG),
        "runner_sha256": metric_source.sha256_file(Path(__file__)),
        "truth_columns_read_before_seal": 0,
        "official_reads": 0,
    }
    metric_source.write_json_new(ARTIFACT / "action_seal.json", seal)

    truth_frame = pd.read_parquet(oof_path, columns=[*metric_source.KEYS, "fold", "label"])
    if not blind.equals(truth_frame[[*metric_source.KEYS, "fold"]]):
        raise metric_source.ContractError("truth attachment changed union order")
    truth = truth_frame["label"].to_numpy(np.int8)
    metadata = blind[["station", "layer", "time"]].reset_index(drop=True)
    by_fold = {}
    fold_nonnegative = []
    for fold in FOLDS:
        mask = blind["fold"].eq(fold).to_numpy()
        cand = metric_source.metric(truth[mask], union[mask])
        ref = metric_source.metric(truth[mask], e150[mask])
        delta = cand["f1"] - ref["f1"]
        fold_nonnegative.append(delta >= 0.0)
        by_fold[fold] = {"candidate": cand, "e150_reference": ref, "delta_f1": delta}

    def comparison(mask: np.ndarray, seed: int) -> dict:
        candidate_metric = metric_source.metric(truth[mask], union[mask])
        reference_metric = metric_source.metric(truth[mask], e150[mask])
        bootstrap = metric_source.paired_bootstrap(
            truth[mask],
            union[mask],
            e150[mask],
            metadata.loc[mask].reset_index(drop=True),
            replicates=int(config["bootstrap_replicates"]),
            seed=seed,
        )
        return {
            "candidate": candidate_metric,
            "reference": reference_metric,
            "delta_f1": candidate_metric["f1"] - reference_metric["f1"],
            "bootstrap": bootstrap,
        }

    pooled_mask = blind["fold"].isin(FOLDS).to_numpy()
    q34_mask = blind["fold"].isin(["2025_q3", "2025_q4"]).to_numpy()
    pooled = comparison(pooled_mask, int(config["bootstrap_seed"]))
    q34 = comparison(q34_mask, int(config["bootstrap_seed"]) + 1)
    additions = (union == 1) & (e150 == 0)
    removals = (union == 0) & (e150 == 1)
    tp = int((additions & (truth == 1)).sum())
    fp = int((additions & (truth == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    precision_lcb = float(beta_distribution.ppf(0.1, tp, fp + 1)) if tp else 0.0
    concentration = audit_source.concentration(blind, additions)
    gates = {
        "each_q2_q3_q4_nonnegative": all(fold_nonnegative),
        "q3_q4_each_nonnegative": by_fold["2025_q3"]["delta_f1"] >= 0
        and by_fold["2025_q4"]["delta_f1"] >= 0,
        "pooled_positive": pooled["delta_f1"] > 0,
        "pooled_ci90_low_positive": pooled["bootstrap"]["difference_ci90"][0] > 0,
        "q3_q4_positive": q34["delta_f1"] > 0,
        "q3_q4_ci90_low_positive": q34["bootstrap"]["difference_ci90"][0] > 0,
        "addition_precision_lcb_above_reference_f1_half": precision_lcb > q34["reference"]["f1"] / 2,
        "changed_fraction_at_most_0_005": float(additions.mean()) <= float(config["maximum_changed_fraction"]),
        "official_zero": True,
    }
    result = {
        "status": "PASS_INTERNAL_GATE" if all(gates.values()) else "TERMINAL_NO_GO",
        "fit_count": 0,
        "threshold_searches": 0,
        "runtime_seconds": time.perf_counter() - started,
        "seal": seal,
        "by_fold": by_fold,
        "pooled_vs_e150": pooled,
        "q3_q4_vs_e150": q34,
        "additions": {
            "rows": int(additions.sum()),
            "fraction": float(additions.mean()),
            "true_positive": tp,
            "false_positive": fp,
            "precision": precision,
            "precision_lcb90": precision_lcb,
            "anchor_removals": int(removals.sum()),
            **concentration,
        },
        "expected_points": {
            "reference": metric_source.PUBLIC_BEST_POINTS,
            "center": metric_source.PUBLIC_BEST_POINTS + metric_source.PUBLIC_SCORE_SLOPE * q34["delta_f1"],
            "ci90": [
                metric_source.PUBLIC_BEST_POINTS
                + metric_source.PUBLIC_SCORE_SLOPE * q34["bootstrap"]["difference_ci90"][0],
                metric_source.PUBLIC_BEST_POINTS
                + metric_source.PUBLIC_SCORE_SLOPE * q34["bootstrap"]["difference_ci90"][1],
            ],
        },
        "gates": gates,
        "operations": {"official_reads": 0, "hidden_reads": 0, "csv": 0, "uploads": 0},
        "hashes": {
            "config": metric_source.sha256_file(CONFIG),
            "runner": metric_source.sha256_file(Path(__file__)),
            "lock": metric_source.sha256_file(ARTIFACT / "attempt_lock.json"),
            "union": metric_source.sha256_file(union_path),
        },
    }
    metric_source.write_json_new(ARTIFACT / "result.json", result)
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
