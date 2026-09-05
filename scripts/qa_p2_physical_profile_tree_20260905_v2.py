"""Independent numerical/hash QA for P2-B; no model fits or official input."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "p2_physical_profile_tree_20260905_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.sum((p - y) ** 2, dtype=np.float64) / len(y)))


def execute():
    report, artifact = ROOT / "reports" / ID, ROOT / "artifacts" / ID
    result = json.loads((report / "result.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "configs/experiments" / f"{ID}.json").read_text(encoding="utf-8"))
    seal = json.loads((report / "preregistration-seal.json").read_text(encoding="utf-8"))
    fits = json.loads((artifact / "03_training/fit-receipts.json").read_text(encoding="utf-8"))
    data_path = artifact / "03_training/raw_oof.npz"
    data = np.load(data_path, allow_pickle=False)
    old = np.load(ROOT / "artifacts/p2_score_repair_20260905_v1/raw_oof.npz", allow_pickle=False)
    y, fold = data["truth"], data["fold"]
    primary = fold == "2024_sep_oct"
    checks = {
        "runner_hash": sha(ROOT / "scripts" / f"run_{ID}.py") == seal["hashes"]["runner"],
        "config_hash": sha(ROOT / "configs/experiments" / f"{ID}.json") == seal["hashes"]["config"],
        "dependencies_hash": all(sha(ROOT / p) == expected for p, expected in seal["hashes"]["dependencies"].items()),
        "raw_hash": sha(data_path) == result["raw_oof_sha256"],
        "keys_truth_fold_exact": all(np.array_equal(data[name], old[name]) for name in ("key", "truth", "fold")),
        "row_unique_count": len(np.unique(data["key"])) == len(y) == result["rows"] == 69850,
        "primary_rows": primary.sum() == 26273,
        "control_exact": np.array_equal(data["C_mean"], old["v23_blockmask_mean"]),
        "fit_bound_unique": len(set(f["fit_id"] for f in fits)) == len(fits) == result["new_historical_fits"] <= 9,
        "fixed_iterations_seeds": all(f["iterations"] == 400 and f["seed"] in cfg["seeds"] for f in fits),
        "model_replay_hash": all(f["replay_max_abs_error_C"] <= 1e-12 and sha(artifact / "04_models" / f"{f['fit_id']}.txt") == f["model_sha256"] for f in fits),
        "original_weight_mass": all(f["training"]["weight_sum"] == f["training"]["original_rows"] for f in fits),
        "prohibited_action_zero": result["official_access_rows"] == result["csv_written"] == result["upload"] == result["calibration_fits"] == result["fulltrain_fits"] == 0,
        "feature_dependency_purge": cfg["feature_dependency_hours"] < cfg["purge_days"] * 24,
    }
    recalculated = {}
    for name, metrics in result["metrics"].items():
        pred = data[name]
        recalculated[name] = {"primary_rmse_C": rmse(y[primary], pred[primary]), "pooled_rmse_C": rmse(y, pred)}
        checks[f"finite_{name}"] = bool(np.isfinite(pred).all())
        checks[f"rmse_{name}"] = abs(recalculated[name]["primary_rmse_C"] - metrics["primary"]["rmse"]) < 1e-12 and abs(recalculated[name]["pooled_rmse_C"] - metrics["pooled"]["rmse"]) < 1e-12
        if name.startswith("fixed_half"):
            suffix = name.removeprefix("fixed_half")
            checks[f"no_fit_equal_mean_{name}"] = np.array_equal(pred, (data[f"tree{suffix}"] + data[f"C{suffix}"]) * 0.5) if suffix != "_mean" else np.allclose(pred, (data[f"tree{suffix}"] + data[f"C{suffix}"]) * 0.5, atol=1e-14, rtol=0)
    for scope, metric_table in result["stress_metrics"].items():
        mask = primary & data["stress_supported"]
        if scope == "fall_masked_supported":
            mask &= data["stress_selected"]
        checks[f"stress_{scope}"] = all(abs(rmse(y[mask], data[f"stress_{name}"][mask]) - values["rmse"]) < 1e-12 for name, values in metric_table.items())
    reference, pred = data["C_mean"], data[result["chosen"]]
    delta = rmse(y[primary], pred[primary]) - rmse(y[primary], reference[primary])
    checks["primary_delta_status"] = abs(delta - result["primary_delta_rmse_C"]) < 1e-12 and (delta < 0) == (result["status"] == "PRIMARY_IMPROVEMENT_INTERNAL_ONLY")
    local = pd.DatetimeIndex(pd.to_datetime(data["time"][primary], utc=True)).tz_convert("Asia/Seoul")
    block = (local.normalize() - local.normalize().min()).days // 7
    block_stats = np.asarray([[np.sum((pred[primary][block == b] - y[primary][block == b]) ** 2), np.sum((reference[primary][block == b] - y[primary][block == b]) ** 2), np.sum(block == b)] for b in np.unique(block)])
    indices = np.random.default_rng(20260905).integers(0, len(block_stats), (2000, len(block_stats)))
    totals = block_stats[indices].sum(axis=1)
    bootstrap = np.sqrt(totals[:, 0] / totals[:, 2]) - np.sqrt(totals[:, 1] / totals[:, 2])
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {"experiment_id": ID, "status": "PASS" if all(checks.values()) else "FAIL", "check_count": len(checks), "checks": checks, "recalculated": recalculated, "primary_delta_rmse_C": delta, "descriptive_autumn_week_block_bootstrap_95_CI": np.quantile(bootstrap, [0.025, 0.975]).tolist(), "new_training_fits": 0, "official_access_rows": 0, "csv_written": 0, "upload": 0}
    if not all(checks.values()):
        raise RuntimeError(json.dumps({k: v for k, v in checks.items() if not v}))
    (report / "independent-recalculation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "check_count", "primary_delta_rmse_C", "descriptive_autumn_week_block_bootstrap_95_CI")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute()
