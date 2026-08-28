"""Execute one fixed P3 long-lead Hs-squared residual shrink probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.longlead_energy_residual import apply_longlead_energy_residual  # noqa: E402

CONFIG = ROOT / "configs/experiments/p3_era5_longlead_energy_residual_shrink_20260828_v1.json"
MODULE = ROOT / "src/p3_wave/longlead_energy_residual.py"
RUNNER = Path(__file__).resolve()
EXPERIMENT_ID = "p3_era5_longlead_energy_residual_shrink_20260828_v1"
FORBIDDEN = ("test_context", "test_index", "sample_submission", "submission.csv")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("experiment ID drifted")
    if config["model"]["parameter_search_count"] != 0 or config["model"]["new_model_fit_count"] != 0:
        raise RuntimeError("this probe must not fit or search")
    policy = config["execution_policy"]
    if any(policy.get(name) is not False for name in ("result_based_weight_or_lead_rerun", "official_input_read_authorized", "submission_generation_authorized", "official_upload_authorized")):
        raise RuntimeError("a prohibited execution boundary was opened")
    checked: dict[str, Any] = {}
    for name, spec in config["immutable_inputs"].items():
        path = ROOT / spec["path"]
        if any(token in str(path).lower() for token in FORBIDDEN):
            raise RuntimeError(f"forbidden path: {path}")
        if not path.is_file() or path.stat().st_size != int(spec["bytes"]):
            raise RuntimeError(f"immutable input missing or size drifted: {name}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(f"immutable hash drifted: {name}")
        checked[name] = {"bytes": path.stat().st_size, "sha256": digest}
    return {"status": "PASS", "checked": checked, "official_rows_read": 0}


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(prediction)) ** 2)))


def _metric(truth: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    incumbent_rmse = _rmse(truth, incumbent)
    candidate_rmse = _rmse(truth, candidate)
    return {
        "rows": int(len(truth)),
        "incumbent_rmse_m": incumbent_rmse,
        "candidate_rmse_m": candidate_rmse,
        "delta_m": candidate_rmse - incumbent_rmse,
    }


def _bootstrap_cases(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, Any]:
    case_ids = np.asarray(sorted(frame["anchor_id"].unique()))
    groups = {case: frame.index[frame["anchor_id"] == case].to_numpy() for case in case_ids}
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    truth = frame["target_hs"].to_numpy()
    incumbent = frame["incumbent_prediction"].to_numpy()
    candidate = frame["candidate_prediction"].to_numpy()
    for replicate in range(replicates):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        rows = np.concatenate([groups[value] for value in sampled])
        deltas[replicate] = _rmse(truth[rows], candidate[rows]) - _rmse(truth[rows], incumbent[rows])
    return {
        "unit": "anchor_id_complete_six_lead_case",
        "cases": int(len(case_ids)),
        "replicates": int(replicates),
        "seed": int(seed),
        "ci90_lower_m": float(np.quantile(deltas, 0.05)),
        "ci90_upper_m": float(np.quantile(deltas, 0.95)),
        "probability_candidate_improves": float(np.mean(deltas < 0.0)),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    receipt = validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    if output.exists():
        raise FileExistsError("one-shot energy residual artifact already exists")
    output.mkdir(parents=True, exist_ok=False)
    sealed_path = ROOT / config["immutable_inputs"]["sealed_transfer_predictions"]["path"]
    incumbent_path = ROOT / config["immutable_inputs"]["incumbent_oof"]["path"]
    sealed = pd.read_parquet(sealed_path)
    oof = pd.read_parquet(incumbent_path)
    oof = oof.loc[np.isclose(oof["prefix_fraction"], 1.0)].copy()
    keys = ["fold", "anchor_id", "station", "lead_h"]
    truth = oof[keys + ["incumbent_prediction", "target_hs"]].copy()
    merged = sealed.merge(truth, on=keys, how="inner", validate="one_to_one", suffixes=("_sealed", "_oof"))
    if len(merged) != len(sealed) or len(merged) != 1086:
        raise RuntimeError("sealed and OOF validation surfaces do not align")
    if not np.array_equal(
        merged["incumbent_prediction_sealed"].to_numpy(),
        merged["incumbent_prediction_oof"].to_numpy(),
    ):
        raise RuntimeError("sealed incumbent is not bit-exact to full-prefix OOF")
    merged = merged.rename(columns={"incumbent_prediction_oof": "incumbent_prediction"})
    candidate, active = apply_longlead_energy_residual(
        merged["incumbent_prediction"].to_numpy(),
        merged["transfer_prediction"].to_numpy(),
        merged["lead_h"].to_numpy(),
        active_leads=tuple(int(value) for value in config["model"]["active_leads_h"]),
        energy_weight=float(config["model"]["energy_weight"]),
    )
    merged["candidate_prediction"] = candidate
    merged["active"] = active
    sealed_candidate = merged[keys + ["current_hs", "incumbent_prediction", "transfer_prediction", "candidate_prediction", "active", "episode_id"]].copy()
    candidate_path = output / "sealed_candidate_predictions.parquet"
    sealed_candidate.to_parquet(candidate_path, index=False)

    overall = _metric(merged["target_hs"].to_numpy(), merged["incumbent_prediction"].to_numpy(), candidate)
    by_fold = {
        str(name): _metric(group["target_hs"].to_numpy(), group["incumbent_prediction"].to_numpy(), group["candidate_prediction"].to_numpy())
        for name, group in merged.groupby("fold", sort=True)
    }
    by_lead = {
        str(int(name)): _metric(group["target_hs"].to_numpy(), group["incumbent_prediction"].to_numpy(), group["candidate_prediction"].to_numpy())
        for name, group in merged.groupby("lead_h", sort=True)
    }
    by_station = {
        str(name): _metric(group["target_hs"].to_numpy(), group["incumbent_prediction"].to_numpy(), group["candidate_prediction"].to_numpy())
        for name, group in merged.groupby("station", sort=True)
    }
    bootstrap = _bootstrap_cases(
        merged.reset_index(drop=True),
        int(config["validation"]["bootstrap_replicates"]),
        int(config["validation"]["bootstrap_seed"]),
    )
    inactive = ~active
    maximum_regression = max(
        [value["delta_m"] for value in by_station.values()]
        + [value["delta_m"] for value in by_lead.values()]
    )
    checks = {
        "overall_improves": overall["delta_m"] < 0.0,
        "minimum_improved_folds": sum(value["delta_m"] < 0.0 for value in by_fold.values()) >= int(config["promotion_gate"]["minimum_improved_folds"]),
        "maximum_station_or_lead_regression": maximum_regression <= float(config["promotion_gate"]["maximum_any_station_or_lead_regression_m"]),
        "inactive_rows_bit_exact": bool(np.array_equal(candidate[inactive], merged.loc[inactive, "incumbent_prediction"].to_numpy())),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "p3.era5_longlead_energy_residual_shrink.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "LOCAL_PROMISING_ADAPTED_PROBE" if passed else "NO_GO_LOCAL_GATE",
        "promotion": "research-only" if passed else "no-go",
        "adaptation_disclosure": config["adaptation_disclosure"],
        "model": config["model"],
        "surface": {"rows": int(len(merged)), "cases": int(merged["anchor_id"].nunique()), "active_rows": int(active.sum()), "inactive_rows": int(inactive.sum())},
        "overall": overall,
        "by_fold": by_fold,
        "by_lead": by_lead,
        "by_station": by_station,
        "paired_case_bootstrap": bootstrap,
        "maximum_station_or_lead_regression_m": float(maximum_regression),
        "checks": checks,
        "fits": 0,
        "parameter_searches": 0,
        "official_rows_read": 0,
        "candidate_or_submission_created": False,
        "upload_count": 0,
    }
    _atomic_json(output / "result.json", result)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "module_sha256": sha256_file(MODULE),
        "runner_sha256": sha256_file(RUNNER),
        "sealed_candidate_sha256": sha256_file(candidate_path),
        "result_sha256": sha256_file(output / "result.json"),
        "preflight": receipt,
        "sealed_candidate_contains_target": False,
        "submission_generated_or_uploaded": False,
    }
    _atomic_json(output / "manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.check == arguments.execute:
        raise SystemExit("choose exactly one mode")
    config = _json(CONFIG)
    result = validate_contract(config) if arguments.check else run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
