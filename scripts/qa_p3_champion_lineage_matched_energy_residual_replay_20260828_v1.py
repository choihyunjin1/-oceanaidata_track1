"""Independent QA for the P3 champion-lineage historical replay."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_champion_lineage_matched_energy_residual_replay_20260828_v1"
RUNNER_PATH = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p3_champion_lineage_replay_qa", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import replay runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same_metric(observed: dict[str, object], expected: dict[str, object]) -> bool:
    return bool(
        int(observed["rows"]) == int(expected["rows"])
        and np.isclose(observed["champion_rmse_m"], expected["champion_rmse_m"], atol=1e-15)
        and np.isclose(observed["candidate_rmse_m"], expected["candidate_rmse_m"], atol=1e-15)
        and np.isclose(observed["delta_m"], expected["delta_m"], atol=1e-15)
    )


def main() -> None:
    runner = _load_runner()
    config = runner._json(runner.CONFIG)
    contract = runner.validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    report_path = ROOT / config["artifacts"]["report"]
    result_path = output / "result.json"
    manifest_path = output / "manifest.json"
    seal_path = output / "PREDICTION_SEAL.json"
    sealed_path = output / "sealed_candidate_predictions.parquet"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    sealed = pd.read_parquet(sealed_path)

    replayed, lineage_checks = runner._load_prediction_surface(config)
    prediction_columns = [
        "o_prediction",
        "a_prediction",
        "champion_prediction",
        "transfer_prediction",
        "candidate_prediction",
    ]
    prediction_reproduction = all(
        np.array_equal(sealed[column].to_numpy(), replayed[column].to_numpy())
        for column in prediction_columns
    )
    original_path = ROOT / config["immutable_inputs"]["original_oof"]["path"]
    truth = pd.read_parquet(original_path, columns=runner.KEYS + ["target_hs"])
    scored = sealed.merge(truth, on=runner.KEYS, how="inner", validate="one_to_one", sort=False)
    recomputed_overall = runner._metric(
        scored["target_hs"].to_numpy(),
        scored["champion_prediction"].to_numpy(),
        scored["candidate_prediction"].to_numpy(),
    )
    recomputed_by_fold = {
        str(name): runner._metric(
            group["target_hs"].to_numpy(),
            group["champion_prediction"].to_numpy(),
            group["candidate_prediction"].to_numpy(),
        )
        for name, group in scored.groupby("fold", sort=True)
    }
    recomputed_by_station = {
        str(name): runner._metric(
            group["target_hs"].to_numpy(),
            group["champion_prediction"].to_numpy(),
            group["candidate_prediction"].to_numpy(),
        )
        for name, group in scored.groupby("station", sort=True)
    }
    recomputed_by_lead = {
        str(int(name)): runner._metric(
            group["target_hs"].to_numpy(),
            group["champion_prediction"].to_numpy(),
            group["candidate_prediction"].to_numpy(),
        )
        for name, group in scored.groupby("lead_h", sort=True)
    }
    slices_match = (
        all(_same_metric(result["by_fold"][key], value) for key, value in recomputed_by_fold.items())
        and all(
            _same_metric(result["by_station"][key], value)
            for key, value in recomputed_by_station.items()
        )
        and all(_same_metric(result["by_lead"][key], value) for key, value in recomputed_by_lead.items())
    )
    inactive = ~sealed["energy_active"].to_numpy(dtype=bool)
    checks = {
        "contract_pass": contract["status"] == "PASS",
        "sealed_rows_1086_cases_181": len(sealed) == 1086 and sealed["anchor_id"].nunique() == 181,
        "sealed_has_no_target": "target_hs" not in sealed.columns,
        "prediction_reproduction_bit_exact": prediction_reproduction,
        "lineage_checks_reproduce": lineage_checks == result["lineage_checks"],
        "inactive_724_bit_exact": int(inactive.sum()) == 724
        and np.array_equal(
            sealed.loc[inactive, "candidate_prediction"].to_numpy(),
            sealed.loc[inactive, "champion_prediction"].to_numpy(),
        ),
        "active_362": int(sealed["energy_active"].sum()) == 362,
        "overall_metric_reproduces": _same_metric(result["overall"], recomputed_overall),
        "slice_metrics_reproduce": slices_match,
        "prediction_seal_matches": seal["sha256"] == runner.sha256_file(sealed_path)
        and seal["truth_attached_after_this_seal"] is True,
        "manifest_hashes_match": manifest["sealed_candidate_sha256"]
        == runner.sha256_file(sealed_path)
        and manifest["result_sha256"] == runner.sha256_file(result_path)
        and manifest["report_sha256"] == runner.sha256_file(report_path),
        "status_matches_gate": result["status"]
        == (
            "GO_OFFICIAL_PROBE_LINEAGE_MATCHED"
            if all(result["gate"].values())
            else "NO_GO_LINEAGE_MATCHED_LOCAL_GATE"
        ),
        "old_gen6_excluded": result["old_gen6_delta_used_as_promotion_evidence"] is False,
        "zero_fit_search_official_submission_upload": result["fits"] == 0
        and result["parameter_searches"] == 0
        and result["official_rows_read"] == 0
        and result["candidate_or_submission_created"] is False
        and result["upload_count"] == 0,
    }
    qa = {
        "schema_version": "p3.champion_lineage_matched_energy_residual_replay.qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "terminal_decision": result["status"],
        "checks": checks,
        "artifacts": {
            "result.json": {
                "bytes": result_path.stat().st_size,
                "sha256": runner.sha256_file(result_path),
            },
            "sealed_candidate_predictions.parquet": {
                "bytes": sealed_path.stat().st_size,
                "sha256": runner.sha256_file(sealed_path),
            },
            "summary.md": {
                "bytes": report_path.stat().st_size,
                "sha256": runner.sha256_file(report_path),
            },
        },
    }
    runner._atomic_json(output / "independent_qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    if qa["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
