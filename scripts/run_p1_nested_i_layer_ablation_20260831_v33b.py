"""Exactly-once nested causal I-layer E150 ablation."""

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

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as e150  # noqa: E402
import run_p1_ordered_catboost_eventday_20260831_v32a as metrics  # noqa: E402

EXPERIMENT_ID = "p1_nested_i_layer_ablation_20260831_v33b"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
FOLDS = metrics.FOLDS


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_contract() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    select = config["selection"]
    checks = {
        "id": config["experiment_id"] == EXPERIMENT_ID,
        "station": select["station"] == "I-ORS",
        "support": select["minimum_layer_addition_support_inclusive"] == 10,
        "half": select["remove_when_marginal_precision_strictly_below_prefix_incumbent_f1_divided_by"] == 2,
        "q2_abstain": select["q2"] == "abstain",
        "fit0": config["fit_budget"] == 0,
        "official0": all(value == 0 for value in config["official_budget"].values()),
    }
    if not all(checks.values()):
        raise metrics.ContractError(f"v33b contract mismatch: {checks}")
    return config


def select_layers(parts: list[e150.FoldBundle], minimum_support: int) -> tuple[list[int], dict]:
    labels = np.concatenate([part.labels for part in parts])
    incumbent = np.concatenate([part.incumbent for part in parts])
    raw = np.concatenate([part.raw_candidate for part in parts])
    station = np.concatenate([part.frame["station"].astype(str).to_numpy() for part in parts])
    layer = np.concatenate([part.frame["layer"].to_numpy(int) for part in parts])
    reference_f1 = metrics.metric(labels, incumbent)["f1"]
    threshold = reference_f1 / 2.0
    records = {}
    selected = []
    additions = (station == "I-ORS") & (raw == 1) & (incumbent == 0)
    for value in sorted(np.unique(layer[station == "I-ORS"]).tolist()):
        mask = additions & (layer == value)
        support = int(mask.sum())
        tp = int((mask & (labels == 1)).sum())
        precision = tp / support if support else None
        eligible = support >= minimum_support and precision is not None and precision < threshold
        if eligible:
            selected.append(int(value))
        records[str(value)] = {
            "support": support,
            "true_positive": tp,
            "false_positive": support - tp,
            "precision": precision,
            "threshold": threshold,
            "selected": eligible,
        }
    return selected, {"prefix_incumbent_f1": reference_f1, "threshold": threshold, "layers": records}


def execute() -> dict:
    started = time.perf_counter()
    config = load_contract()
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": metrics.sha256_file(CONFIG),
        "runner_sha256": metrics.sha256_file(Path(__file__)),
        "fit_budget": 0,
    }
    metrics.write_json_new(ARTIFACT / "attempt_lock.json", lock)
    bundles = e150.load_bundles()
    support = int(config["selection"]["minimum_layer_addition_support_inclusive"])
    q3_layers, q3_audit = select_layers([bundles["2025_q2"]], support)
    q4_layers, q4_audit = select_layers([bundles["2025_q2"], bundles["2025_q3"]], support)
    deployment_layers, deployment_audit = select_layers([bundles[fold] for fold in FOLDS], support)
    selected = {"2025_q2": [], "2025_q3": q3_layers, "2025_q4": q4_layers}
    candidates = {}
    removals = {}
    for fold in FOLDS:
        bundle = bundles[fold]
        station = bundle.frame["station"].astype(str).to_numpy()
        layer = bundle.frame["layer"].to_numpy(int)
        removal = (
            (station == "I-ORS")
            & np.isin(layer, selected[fold])
            & (bundle.raw_candidate == 1)
            & (bundle.incumbent == 0)
        )
        candidate = bundle.raw_candidate.copy()
        candidate[removal] = bundle.incumbent[removal]
        candidates[fold] = candidate
        removals[fold] = removal
    sealed = np.concatenate([candidates[fold] for fold in FOLDS])
    seal_path = ARTIFACT / "sealed_nested_candidate.npz"
    np.savez_compressed(seal_path, candidate=sealed)
    seal = {
        "rows": len(sealed),
        "candidate_sha256": sha256_array(sealed),
        "npz_sha256": metrics.sha256_file(seal_path),
        "selected_layers": selected,
        "q3_selection_audit": q3_audit,
        "q4_selection_audit": q4_audit,
        "deployment_layers": deployment_layers,
        "deployment_selection_audit": deployment_audit,
        "outer_result_reselection": 0,
        "official_reads": 0,
    }
    metrics.write_json_new(ARTIFACT / "selection_and_prediction_seal.json", seal)

    by_fold = {}
    labels = []
    references = []
    predictions = []
    metadata = []
    all_removals = []
    for fold in FOLDS:
        bundle = bundles[fold]
        ref = metrics.metric(bundle.labels, bundle.raw_candidate)
        cand = metrics.metric(bundle.labels, candidates[fold])
        removal = removals[fold]
        by_fold[fold] = {
            "selected_layers": selected[fold],
            "reference_f1": ref["f1"],
            "candidate_f1": cand["f1"],
            "delta_f1": cand["f1"] - ref["f1"],
            "removed_rows": int(removal.sum()),
            "removed_true_positive": int((removal & (bundle.labels == 1)).sum()),
            "removed_false_positive": int((removal & (bundle.labels == 0)).sum()),
        }
        labels.append(bundle.labels)
        references.append(bundle.raw_candidate)
        predictions.append(candidates[fold])
        metadata.append(bundle.frame[["station", "layer", "time"]])
        all_removals.append(removal)
    truth = np.concatenate(labels)
    reference = np.concatenate(references)
    candidate = np.concatenate(predictions)
    meta = pd.concat(metadata, ignore_index=True)
    removed = np.concatenate(all_removals)

    def compare(mask: np.ndarray, seed: int) -> dict:
        cand = metrics.metric(truth[mask], candidate[mask])
        ref = metrics.metric(truth[mask], reference[mask])
        bootstrap = metrics.paired_bootstrap(
            truth[mask], candidate[mask], reference[mask], meta.loc[mask].reset_index(drop=True),
            replicates=int(config["validation"]["bootstrap_replicates"]), seed=seed,
        )
        return {"candidate": cand, "reference": ref, "delta_f1": cand["f1"] - ref["f1"], "bootstrap": bootstrap}

    pooled_mask = np.ones(len(truth), dtype=bool)
    q34_mask = np.concatenate([np.zeros(len(labels[0]), dtype=bool), np.ones(len(labels[1]) + len(labels[2]), dtype=bool)])
    pooled = compare(pooled_mask, int(config["validation"]["bootstrap_seed"]))
    q34 = compare(q34_mask, int(config["validation"]["bootstrap_seed"]) + 1)
    changed_layers = meta.loc[removed, "layer"].value_counts()
    max_share = float(changed_layers.max() / removed.sum()) if removed.any() else 0.0
    gates = {
        "q3_nonnegative": by_fold["2025_q3"]["delta_f1"] >= 0,
        "q4_nonnegative": by_fold["2025_q4"]["delta_f1"] >= 0,
        "pooled_positive": pooled["delta_f1"] > 0,
        "pooled_ci90_low_positive": pooled["bootstrap"]["difference_ci90"][0] > 0,
        "q34_positive": q34["delta_f1"] > 0,
        "q34_ci90_low_positive": q34["bootstrap"]["difference_ci90"][0] > 0,
        "changed_fraction_at_most_0_005": float(removed.mean()) <= float(config["guards"]["maximum_changed_fraction"]),
        "max_layer_share_at_most_0_5": max_share <= float(config["guards"]["maximum_single_layer_change_share"]),
        "anchor_removals_zero": True,
    }
    result = {
        "status": "PASS_INTERNAL_GATE" if all(gates.values()) else "TERMINAL_NO_GO",
        "fit_count": 0,
        "runtime_seconds": time.perf_counter() - started,
        "seal": seal,
        "by_fold": by_fold,
        "pooled": pooled,
        "q3_q4": q34,
        "removed_rows": int(removed.sum()),
        "changed_fraction": float(removed.mean()),
        "maximum_layer_concentration": max_share,
        "expected_points_delta": {
            "center": metrics.PUBLIC_SCORE_SLOPE * q34["delta_f1"],
            "ci90": [metrics.PUBLIC_SCORE_SLOPE * value for value in q34["bootstrap"]["difference_ci90"]],
        },
        "gates": gates,
        "materializer_preparation": {
            "deployment_layers": deployment_layers,
            "execute": False,
            "formula": "remove official I-ORS E150 incumbent-negative additions whose layer is in deployment_layers",
        },
        "operations": {"official_reads": 0, "hidden_reads": 0, "csv": 0, "uploads": 0},
        "hashes": {
            "config": metrics.sha256_file(CONFIG),
            "runner": metrics.sha256_file(Path(__file__)),
            "lock": metrics.sha256_file(ARTIFACT / "attempt_lock.json"),
            "candidate": metrics.sha256_file(seal_path),
        },
    }
    metrics.write_json_new(ARTIFACT / "result.json", result)
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
