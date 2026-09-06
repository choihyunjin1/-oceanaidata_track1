"""Independent aggregate QA of completed P2 predictions; no model or source-data I/O."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/p2_c3_multiseed_completion_20260906_v1"
PRIOR = ROOT / "artifacts/p2_c3_training_comparison_20260906_v1"
REPORT = ROOT / "reports/parallel_completion_training_20260906_v2/p2-root-independent-qa.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    terminal = json.loads((OUT / "terminal_result.json").read_text())
    replay = json.loads((OUT / "fresh-replay.json").read_text())
    checks = []

    def check(name, condition):
        checks.append(name)
        if not condition:
            raise AssertionError(name)

    check("complete_28", terminal["new_fits"] == terminal["attempted_fits"] == len(terminal["fits"]) == 28)
    check("48_models", terminal["combined_models"] == replay["models"] == 48)
    check("zero_external_actions", terminal["official_rows"] == terminal["csv_written"] == terminal["upload"] == 0)
    check("replay_pass", replay["status"] == "PASS" and replay["max_absolute_error"] == 0)
    check("replay_link", replay["terminal_result_sha256"] == sha(OUT / "terminal_result.json"))
    check("fresh_pid", replay["pid"] != replay["training_pid"] == terminal["pid"])
    check("evaluation_hash", terminal["evaluation_sha256"] == sha(OUT / "evaluation.npz"))
    metrics = {}
    with np.load(OUT / "evaluation.npz", allow_pickle=False) as values, np.load(PRIOR / "evaluation.npz", allow_pickle=False) as prior:
        for name in ("key", "truth", "fold", "time", "layer", "natural_T5_missing"):
            check(name + "_unchanged", np.array_equal(values[name], prior[name]))
        y, folds = values["truth"], values["fold"]
        check("complete_unique_rows", len(y) == len(np.unique(values["key"])) == 166268)
        for arm in ("C60", "L120"):
            prediction = values["natural_" + arm]
            check(arm + "_finite", np.isfinite(prediction).all())
            for scope, mask in (("all8_pooled", np.ones(len(y), dtype=bool)), ("B3_primary", folds == "B3")):
                # Scalar fsum, independent of production numpy mean/dot reductions.
                sse = math.fsum(float(error) ** 2 for error in y[mask] - prediction[mask])
                rmse = math.sqrt(sse / int(mask.sum()))
                expected = terminal["summary"]["natural"][arm][scope]
                check(arm + scope + "_n", expected["n"] == int(mask.sum()))
                check(arm + scope + "_sse", math.isclose(sse, expected["sse"], rel_tol=1e-12))
                check(arm + scope + "_rmse", math.isclose(rmse, expected["rmse_C"], abs_tol=1e-12))
                metrics.setdefault(scope, {})[arm] = rmse
            for surface in ("natural", "outage"):
                check(arm + surface + "_primary_exact", np.array_equal(values[surface + "_" + arm][folds == "B3"], prior[f"B3_{arm}_{surface}_all3"]))
        for row in metrics.values():
            row["delta_L120_minus_C60_C"] = row["L120"] - row["C60"]
    result = {"status": "PASS", "checks": checks, "metrics": metrics,
              "terminal_sha256": sha(OUT / "terminal_result.json"), "replay_sha256": sha(OUT / "fresh-replay.json"),
              "new_fits": 0, "official_source_reads": 0, "model_reads": 0,
              "scope": "aggregate arithmetic and exact prior-primary/key comparison, not new generalization evidence"}
    with REPORT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "checks": len(checks), "metrics": metrics}))


if __name__ == "__main__":
    run()
