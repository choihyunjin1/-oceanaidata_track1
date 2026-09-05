"""Independent zero-fit arithmetic, public-only routing and chronology QA."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "p2_missingness_conditional_validation_20260905_v3"
OUT = ROOT / "artifacts" / ID
REPORT = ROOT / "reports" / ID


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(y, p):
    sse = float(np.sum(np.square(y-p), dtype=np.float64))
    return {"n": len(y), "sse": sse, "rmse": float(np.sqrt(sse/len(y)))}


def main():
    cfg_path = ROOT / "configs/experiments" / f"{ID}.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    result_path = REPORT / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    seal = json.loads((REPORT / "preregistration-seal.json").read_text(encoding="utf-8"))
    raw_path = OUT / "validation_predictions.npz"
    data = np.load(raw_path, allow_pickle=False)
    old = np.load(ROOT / "artifacts/p2_objective_alignment_20260905_v2/03_training/raw_oof.npz", allow_pickle=False)
    source = Path(os.environ["P2_DATA_DIR"]).resolve() / "observations.csv"
    checks = {
        "runner_frozen": sha(ROOT / "scripts" / f"run_{ID}.py") == seal["hashes"]["runner"],
        "config_frozen": sha(cfg_path) == seal["hashes"]["config"],
        "dependencies_frozen": all(sha(ROOT / path) == value for path, value in seal["hashes"]["dependencies"].items()),
        "six_source_models_exact": len(seal["hashes"]["models"]) == 6 and all(sha(ROOT / m["path"]) == m["sha256"] for m in seal["hashes"]["models"]),
        "source_hash": sha(source) == cfg["source_sha256"],
        "output_hash": sha(raw_path) == result["prediction_sha256"],
        "same_69850_keys_truth_fold": len(data["truth"]) == 69850 and all(np.array_equal(data[name], old[name]) for name in ("key", "truth", "fold")),
        "unique_keys": len(np.unique(data["key"])) == 69850,
        "first_seed_control_reuse": np.array_equal(data["intact_C"], old["C_seed20260901"]),
        "first_seed_R_reuse": np.array_equal(data["intact_R"], old["R_seed20260901"]),
        "zero_fits_official_csv_upload_deleted_rows": all(result[name] == 0 for name in ("new_backbone_fits", "new_rule_fits", "fulltrain_fits", "official_access_rows", "csv_written", "upload", "evaluation_rows_deleted")),
    }
    public = pd.read_csv(source, usecols=["time", "layer", "temp", "psal"])
    public = public.loc[public.layer.eq(5)].copy()
    public["time"] = pd.to_datetime(public.time, utc=True)
    observed = public.set_index("time")[["temp", "psal"]].reindex(pd.to_datetime(data["time"], utc=True))
    expected_intact = ~(np.isfinite(observed.temp.to_numpy()) & np.isfinite(observed.psal.to_numpy()))
    checks["trigger_independent_public_only_rebuild"] = np.array_equal(expected_intact, data["intact_trigger"])
    y, fold = data["truth"], data["fold"]
    times = pd.to_datetime(data["time"], utc=True)
    temp_missing = ~np.isfinite(observed.temp.to_numpy())
    psal_missing = ~np.isfinite(observed.psal.to_numpy())
    types = {"temp_only_missing": temp_missing & ~psal_missing, "psal_only_missing": ~temp_missing & psal_missing, "both_missing": temp_missing & psal_missing, "both_available": ~temp_missing & ~psal_missing}
    natural = {}
    for name, mask_type in types.items():
        for group in (*np.unique(fold), "all"):
            mask = mask_type & (np.ones(len(y), bool) if group == "all" else fold == group)
            natural[f"{group}/{name}"] = {"n": int(mask.sum()), "metrics": {arm: metric(y[mask], data[f"intact_{arm}"][mask]) for arm in ("C", "R", "conditional")} if mask.any() else None}
    cases = [("intact", result["intact"], data["intact_trigger"], None)]
    for episode in result["episodes"]:
        name = episode["id"]
        selected = np.asarray((times >= pd.Timestamp(episode["start"])) & (times < pd.Timestamp(episode["stop"])))
        checks[f"{name}_onset_offset"] = np.array_equal(selected, data[f"{name}_selected"])
        expected_trigger = expected_intact | selected
        checks[f"{name}_trigger"] = np.array_equal(expected_trigger, data[f"{name}_trigger"])
        checks[f"{name}_rows_preserved"] = episode["all_keys"] == len(y) and episode["evaluation_rows_deleted"] == 0
        if episode["status"] == "SUPPORT_BLOCKED":
            checks[f"{name}_not_silently_scored"] = "metrics" not in episode and episode["unsupported_rows"] > 0
            continue
        checks[f"{name}_support_complete"] = data[f"{name}_supported"].all()
        cases.append((name, episode["metrics"], expected_trigger, selected))
    for name, scopes, trigger, selected in cases:
        c, r, policy = (data[f"{name}_{arm}"] for arm in ("C", "R", "conditional"))
        checks[f"{name}_policy_exact"] = np.array_equal(policy, np.where(trigger, r, c))
        checks[f"{name}_finite"] = all(np.isfinite(v).all() for v in (c, r, policy))
        if selected is not None:
            checks[f"{name}_outside_episode_invariant"] = all(np.array_equal(data[f"{name}_{arm}"][~selected], data[f"intact_{arm}"][~selected]) for arm in ("C", "R", "conditional"))
        masks = {"all_intact_keys": np.ones(len(y), bool), "autumn_primary": fold == "2024_sep_oct", "trigger": trigger}
        if selected is not None:
            masks.update(injected_episode=selected, outside_injected_episode=~selected)
        for scope, mask in masks.items():
            reported = scopes[scope]
            passed = reported["n"] == int(mask.sum())
            if mask.any():
                for arm in ("C", "R", "conditional"):
                    rec = metric(y[mask], data[f"{name}_{arm}"][mask])
                    passed &= all(abs(rec[k]-reported["metrics"][arm][k]) < 1e-9 for k in ("n", "sse", "rmse"))
            else:
                passed &= reported["metrics"] is None
            checks[f"{name}_{scope}_metrics"] = passed
    eligible = [e for e in result["episodes"] if not e["development"] and e["status"] == "COMPLETE"]
    for group, table in result["additional_episode_metrics"].items():
        episodes = [e for e in eligible if group == "all_new_episodes" or e["fold"] == group]
        for arm in ("C", "R", "conditional"):
            error = np.concatenate([data[f"{e['id']}_{arm}"][data[f"{e['id']}_selected"]]-y[data[f"{e['id']}_selected"]] for e in episodes]) if episodes else np.array([])
            passed = len(error) == table[arm]["n"]
            if len(error):
                passed &= abs(float(np.sqrt(np.mean(error**2)))-table[arm]["rmse"]) < 1e-12
            checks[f"aggregate_{group}_{arm}"] = passed
    full_support = all(e["status"] == "COMPLETE" for e in result["episodes"])
    primary = fold == "2024_sep_oct"
    intact_delta = metric(y[primary], data["intact_conditional"][primary])["rmse"]-metric(y[primary], data["intact_C"][primary])["rmse"]
    autumn = result["additional_episode_metrics"]["2024_sep_oct"]
    information = full_support and intact_delta <= 0 and autumn["conditional"]["rmse"] < autumn["C"]["rmse"]
    checks["decision_independent_recalculation"] = information == (result["status"] == "INFORMATION_VALUE_INTERNAL_ONLY")
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {"experiment_id": ID, "status": "PASS" if all(checks.values()) else "FAIL", "check_count": len(checks), "checks": checks, "natural_availability_breakdown": natural, "result_sha256": sha(result_path), "new_fits": 0, "official_access_rows": 0, "csv_written": 0, "upload": 0}
    with (REPORT / "independent-qa.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps({k: payload[k] for k in ("status", "check_count", "new_fits")}))
    if not all(checks.values()):
        raise AssertionError([k for k, v in checks.items() if not v])


if __name__ == "__main__":
    main()
