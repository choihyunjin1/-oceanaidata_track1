"""Read-only independent arithmetic/hash QA; no fitting or data-row output."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/p3_direct_sse_meta_20260905_v2/result.json"
OUT = ROOT / "artifacts/p3_direct_sse_meta_20260905_v2"


def run():
    result = json.loads(REPORT.read_text(encoding="utf-8"))
    data = pd.read_parquet(OUT / "oof.parquet")
    checks = {}
    checks["key_unique_complete_181x6"] = len(data) == 1086 and not data.duplicated(["anchor_id", "station", "lead_h"]).any() and data.anchor_id.nunique() == 181
    checks["metrics_independently_recomputed"] = all(abs(float(np.sqrt(np.square(data.target_hs-data[name]).mean()))-value["rmse_m"]) < 1e-14 for name, value in result["candidates"].items())
    checks["first_fold_exact_noop"] = all(np.array_equal(data.loc[data.fold.eq("2024_h2_storm"), name], data.loc[data.fold.eq("2024_h2_storm"), "final_prediction"]) for name in ("no_op", "long_simplex", "global_bias"))
    checks["simplex_short_leads_exact_noop"] = np.array_equal(data.loc[data.lead_h.le(9), "long_simplex"], data.loc[data.lead_h.le(9), "final_prediction"])
    checks["four_historical_fits_zero_backbone"] = result["fit_count"] == {"historical_meta": 4, "selected_full_meta": 0, "backbone": 0}
    checks["past_fit_cases_49_128"] = [row["fit_cases"] for row in result["meta_fit_receipts"]] == [49, 49, 128, 128]
    checks["no_op_wins"] = result["winner"] == "no_op" and min(result["candidates"], key=lambda name: result["candidates"][name]["rmse_m"]) == "no_op"
    checks["artifact_hashes_match"] = all(hashlib.sha256((OUT / name).read_bytes()).hexdigest() == value for name, value in result["artifact_sha256"].items())
    checks["input_hashes_match"] = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in result["manifest"]["inputs"].items() if not name.startswith("source/"))
    predictions = data[["no_op", "long_simplex", "global_bias"]]
    checks["finite_range"] = bool(np.isfinite(predictions).all().all() and predictions.ge(0).all().all() and predictions.le(30).all().all())
    checks["zero_official_counts"] = not any(result["official_access"].values())
    checks["zero_public_fit"] = result["public_score_fitting"] is False
    checks["reload_exact"] = result["selected_parameters_reload_max_abs_m"] == 0
    output = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": {key: bool(value) for key, value in checks.items()}, "passed": int(sum(checks.values())), "total": len(checks), "result_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(), "elapsed_s": result["elapsed_seconds"], "bootstrap": {name: value["paired_bootstrap"]["delta_rmse_ci95_m"] for name, value in result["candidates"].items()}}
    assert all(checks.values())
    return output


if __name__ == "__main__":
    print(json.dumps(run()))
