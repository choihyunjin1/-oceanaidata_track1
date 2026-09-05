"""Independent aggregate checks of frozen P2-A results without retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "p2_objective_alignment_20260905_v2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.sum(np.square(p - y), dtype=np.float64) / len(y)))


def execute():
    report, artifact = ROOT / "reports" / ID, ROOT / "artifacts" / ID
    result = json.loads((report / "result.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "configs/experiments" / f"{ID}.json").read_text(encoding="utf-8"))
    seal = json.loads((report / "preregistration-seal.json").read_text(encoding="utf-8"))
    fits = json.loads((artifact / "03_training/fit-receipts.json").read_text(encoding="utf-8"))
    path = artifact / "03_training/raw_oof.npz"
    data = np.load(path, allow_pickle=False)
    old = np.load(ROOT / "artifacts/p2_score_repair_20260905_v1/raw_oof.npz", allow_pickle=False)
    y, fold = data["truth"], data["fold"]
    primary = fold == "2024_sep_oct"
    checks = {
        "artifact_hash": digest(path) == result["artifact_sha256"],
        "runner_hash": digest(ROOT / "scripts" / f"run_{ID}.py") == seal["hashes"]["runner"],
        "config_hash": digest(ROOT / "configs/experiments" / f"{ID}.json") == seal["hashes"]["config"],
        "dependencies_hash": all(digest(ROOT / rel) == value for rel, value in seal["hashes"]["dependencies"].items()),
        "keys_unique": len(np.unique(data["key"])) == len(y),
        "old_keys_truth_fold_exact": all(np.array_equal(data[name], old[name]) for name in ("key", "truth", "fold")),
        "control_mean_exact": np.array_equal(data["C_mean"], old["v23_blockmask_mean"]),
        "row_count": len(y) == result["rows"] == 69850,
        "primary_count": int(primary.sum()) == 26273,
        "fit_ids_unique": len(set(item["fit_id"] for item in fits)) == len(fits),
        "fit_cap_and_report": len(fits) == result["new_historical_fits"] <= 15,
        "fit_seed_epochs": all(item["seed"] in cfg["seeds"] and item["epochs"] == 60 for item in fits),
        "saved_models_hash_replay": all(item["replay_exact"] and digest(artifact / "04_models" / f"{item['fit_id']}.pt") == item["model_sha256"] for item in fits),
        "official_csv_upload_zero": result["official_access_rows"] == result["csv_written"] == result["upload"] == 0,
        "calibration_fullfit_zero": result["calibration_fits"] == result["fulltrain_fits"] == 0,
        "control_reuse_pass": json.loads((report / "control-reuse-qa.json").read_text(encoding="utf-8"))["passed"],
    }
    recalculated = {}
    for name, value in result["metrics"].items():
        pred = data[name]
        recalculated[name] = {"primary_rmse_C": rmse(y[primary], pred[primary]), "pooled_rmse_C": rmse(y, pred)}
        checks[f"finite_{name}"] = bool(np.isfinite(pred).all())
        checks[f"metrics_{name}"] = abs(recalculated[name]["primary_rmse_C"] - value["primary"]["rmse"]) < 1e-12 and abs(recalculated[name]["pooled_rmse_C"] - value["pooled"]["rmse"]) < 1e-12
    for scope, metric_table in result["stress_metrics"].items():
        mask = primary & data["stress_supported"]
        if scope == "fall_masked_supported":
            mask &= data["stress_selected"]
        checks[f"stress_{scope}"] = all(
            values["n"] == int(mask.sum())
            and abs(rmse(y[mask], data[f"stress_{name}"][mask]) - values["rmse"]) < 1e-12
            for name, values in metric_table.items()
        )
    chosen, reference = data[result["chosen"]], data["C_mean"]
    expected_delta = rmse(y[primary], chosen[primary]) - rmse(y[primary], reference[primary])
    checks["primary_delta"] = abs(expected_delta - result["primary_delta_rmse_C"]) < 1e-12
    checks["decision_mean_not_ci"] = (expected_delta < 0) == (result["status"] == "PRIMARY_IMPROVEMENT_INTERNAL_ONLY")
    # Fixed seven-calendar-day blocks within autumn; descriptive bootstrap,
    # not a fresh independent holdout and not used for candidate selection.
    local = pd.DatetimeIndex(pd.to_datetime(data["time"][primary], utc=True)).tz_convert("Asia/Seoul")
    block = (local.normalize() - local.normalize().min()).days // 7
    stats = np.asarray([[np.sum((chosen[primary][block == b] - y[primary][block == b]) ** 2), np.sum((reference[primary][block == b] - y[primary][block == b]) ** 2), np.sum(block == b)] for b in np.unique(block)])
    rng = np.random.default_rng(20260905)
    picks = rng.integers(0, len(stats), (2000, len(stats)))
    totals = stats[picks].sum(axis=1)
    deltas = np.sqrt(totals[:, 0] / totals[:, 2]) - np.sqrt(totals[:, 1] / totals[:, 2])
    output = {"experiment_id": ID, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "check_count": len(checks), "recalculated": recalculated, "primary_delta_rmse_C": expected_delta, "descriptive_autumn_week_block_bootstrap_95_CI": np.quantile(deltas, [0.025, 0.975]).tolist(), "historical_block_count": len(stats), "new_training_fits": 0, "official_access_rows": 0, "csv_written": 0, "upload": 0}
    if not all(checks.values()):
        raise RuntimeError(json.dumps({key: value for key, value in checks.items() if not value}))
    (report / "independent-recalculation.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("status", "check_count", "primary_delta_rmse_C", "descriptive_autumn_week_block_bootstrap_95_CI")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()
