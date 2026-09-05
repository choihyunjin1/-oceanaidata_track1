"""Read-only independent arithmetic/replay receipts for the frozen P3 v4 run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import run_p3_wind_only_dropout_20260905_v4 as run

ROOT, OUT, REPORT = run.ROOT, run.OUT, run.REPORT


def main():
    config = json.loads(run.CONFIG.read_text(encoding="utf-8"))
    run.install_guard(Path(os.environ["P3_DATA_DIR"]).resolve(), config)
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    seal = json.loads((OUT / "seal.json").read_text(encoding="utf-8"))
    replay = json.loads((REPORT / "fresh-process-replay.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(OUT / "reference.parquet")
    oof = pd.read_parquet(OUT / "oof.parquet")
    checks = []

    def check(name, condition):
        checks.append({"check": name, "pass": bool(condition)})

    check("frozen_runner", result["runner_sha256"] == seal["runner_sha256"] == run.sha(ROOT / "scripts/run_p3_wind_only_dropout_20260905_v4.py"))
    check("frozen_config", result["config_sha256"] == seal["config_sha256"] == run.sha(run.CONFIG))
    check("seal_hash", result["seal_sha256"] == run.sha(OUT / "seal.json"))
    check("reference_hash", result["reference_sha256"] == run.sha(OUT / "reference.parquet"))
    check("oof_hash", result["oof_sha256"] == run.sha(OUT / "oof.parquet"))
    check("six_fits_zero_new_router", result["historical_fit_count"] == 6 and result["new_router_fit_count"] == 0 and len(result["fits"]) == 6)
    check("1086rows181cases", len(frame) == 1086 and frame.anchor_id.nunique() == 181 and len(oof) == 2172)
    check("unique_keys", not frame.duplicated(run.KEYS).any() and not oof.duplicated(run.KEYS + ["arm"]).any())
    check("all_six_leads", frame.groupby("anchor_id").lead_h.agg(lambda x: sorted(x) == run.LEADS).all())
    check("budget_runtime", result["runtime_seconds"] <= 1800)
    check("official_zero", all(x == 0 for x in result["official_access"].values()))
    for name, expected in seal["integrity"]["verified_inputs"].items():
        path = Path(os.environ["P3_DATA_DIR"]) / name.split("/", 1)[1] if name.startswith("source/") else ROOT / name
        check(f"pinned/{name}", run.sha(path) == expected)
    chronology = seal["integrity"]["reference_chronology"]["chronology"]
    check("past_router_counts", [x["past_cases"] for x in chronology] == [0, 49, 128])
    for row in chronology[1:]:
        check(f"chronology/{row['fold']}", row["global_past_target_ready_before_fold_start_h"] >= 0 and row["minimum_same_station_anchor_gap_h"] >= 78 and row["same_station_episode_overlap"] == 0)
    ordered = frame.sort_values(run.KEYS).reset_index(drop=True)
    truth = ordered.target_hs.to_numpy()
    base = ordered.final_prediction.to_numpy()
    policies = {"no_op": base}
    for arm in ("control", "wind_only"):
        one = oof[oof.arm.eq(arm)].sort_values(run.KEYS).reset_index(drop=True)
        check(f"{arm}/exact_keys", ordered[run.KEYS].equals(one[run.KEYS]))
        check(f"{arm}/exact_targets", np.array_equal(truth, one.target_hs))
        check(f"{arm}/finite_clip", np.isfinite(one.raw_prediction).all() and np.array_equal(np.clip(one.raw_prediction, 0, 30), one.prediction))
        policies[arm] = one.prediction.to_numpy()
        policies[f"{arm}_half"] = (base + policies[arm]) / 2
        raw_sse = float(np.square(truth - one.raw_prediction.to_numpy()).sum())
        check(f"{arm}/raw_sse", np.isclose(raw_sse, result["raw_metrics"][arm]["sse_m2"], rtol=0, atol=1e-9))
    for name, prediction in policies.items():
        sse = float(np.dot(truth - prediction, truth - prediction))
        check(f"{name}/pooled_sse", np.isclose(sse, result["metrics"][name]["sse_m2"], rtol=0, atol=1e-9))
        check(f"{name}/pooled_rmse", np.isclose(np.sqrt(sse / len(truth)), result["metrics"][name]["rmse_m"], rtol=0, atol=1e-12))
        for column in ("fold", "station", "lead_h", "wind_observed", "onset_3h", "nonwind_atmos_observed"):
            for group, indices in ordered.groupby(column).indices.items():
                expected = float(np.sqrt(np.dot(truth[indices] - prediction[indices], truth[indices] - prediction[indices]) / len(indices)))
                check(f"{name}/{column}/{group}", np.isclose(expected, result["metrics"][name][f"by_{column}"][str(group)]["rmse_m"], rtol=0, atol=1e-12))
    actual_winner = min(run.POLICIES, key=lambda name: np.square(truth - policies[name]).sum())
    check("fixed_policy_selection", actual_winner == result["selected_policy"])
    for fit in result["fits"]:
        check(f"model/{fit['fold']}/{fit['arm']}", run.sha(ROOT / fit["model_path"]) == fit["model_sha256"] and fit["native_reload_max_abs_m"] == 0)
        aug = fit["augmentation"]
        check(f"weight/{fit['fold']}/{fit['arm']}", aug["max_per_original_row_weight_error"] <= 1e-12 and np.isclose(aug["weight_sum_before"], aug["weight_sum_after"], rtol=0, atol=1e-8) and aug["target_copy_exact"])
    check("fresh_pid", replay["fresh_pid"] != replay["training_pid"] == result["pid"])
    check("replay_links", replay["result_sha256"] == run.sha(REPORT / "result.json") and replay["runner_sha256"] == result["runner_sha256"])
    check("six_saved_models2172rows_exact", replay["models"] == 6 and replay["rows_replayed"] == 2172 and all(x["max_abs_raw_prediction_error_m"] == 0 for x in replay["checks"]))
    failed = [x["check"] for x in checks if not x["pass"]]
    receipt = {"status": "PASS" if not failed else "FAIL", "independent_checks": len(checks), "failed_checks": failed, "checks": checks, "result_sha256": run.sha(REPORT / "result.json"), "replay_sha256": run.sha(REPORT / "fresh-process-replay.json"), "qa_runner_sha256": run.sha(Path(__file__)), "source_rows_printed": 0, "official_access": run.official_zero(), "empty_folder_fulltrain_inference_pass": False, "scope": "independent formulas and all aggregate slices; six historical saved models fresh-process replay; not complete final training/inference package"}
    run.save(REPORT / "independent-qa.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "failed_checks": failed, "result_sha256": receipt["result_sha256"]}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
