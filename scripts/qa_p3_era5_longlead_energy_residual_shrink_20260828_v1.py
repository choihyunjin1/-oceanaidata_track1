"""Independent QA for the fixed P3 long-lead energy residual probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p3_era5_longlead_energy_residual_shrink_20260828_v1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p3_energy_residual_runner_qa", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = _load_runner()
    config = runner._json(runner.CONFIG)
    contract = runner.validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    result_path = output / "result.json"
    manifest_path = output / "manifest.json"
    sealed_path = output / "sealed_candidate_predictions.parquet"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed = pd.read_parquet(sealed_path)
    inactive = ~sealed["active"].to_numpy(dtype=bool)
    recomputed = runner._metric(
        np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]), np.asarray([0.0, 1.0])
    )
    checks = {
        "contract_pass": contract["status"] == "PASS",
        "sealed_rows_1086": len(sealed) == 1086,
        "sealed_has_no_target": "target_hs" not in sealed.columns,
        "active_rows_exact_362": int(sealed["active"].sum()) == 362,
        "inactive_bit_exact": bool(np.array_equal(sealed.loc[inactive, "candidate_prediction"].to_numpy(), sealed.loc[inactive, "incumbent_prediction"].to_numpy())),
        "zero_fits_and_searches": result["fits"] == 0 and result["parameter_searches"] == 0,
        "result_hash_matches": manifest["result_sha256"] == runner.sha256_file(result_path),
        "sealed_hash_matches": manifest["sealed_candidate_sha256"] == runner.sha256_file(sealed_path),
        "status_matches_checks": result["status"] == ("LOCAL_PROMISING_ADAPTED_PROBE" if all(result["checks"].values()) else "NO_GO_LOCAL_GATE"),
        "metric_helper_sanity": bool(
            np.isclose(recomputed["candidate_rmse_m"], 0.0)
            and np.isclose(recomputed["incumbent_rmse_m"], np.sqrt(0.5))
        ),
        "official_rows_zero": result["official_rows_read"] == 0,
        "no_submission": result["candidate_or_submission_created"] is False and result["upload_count"] == 0,
    }
    qa = {
        "schema_version": "p3.era5_longlead_energy_residual_shrink.qa.v1",
        "experiment_id": runner.EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "terminal_decision": result["status"],
        "checks": checks,
        "artifacts": {
            "result.json": {"bytes": result_path.stat().st_size, "sha256": runner.sha256_file(result_path)},
            "sealed_candidate_predictions.parquet": {"bytes": sealed_path.stat().st_size, "sha256": runner.sha256_file(sealed_path)},
        },
    }
    qa_path = output / "independent_qa.json"
    runner._atomic_json(qa_path, qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    if qa["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
