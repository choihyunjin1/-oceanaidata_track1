"""Execute the fixed P3 champion-lineage historical energy-residual replay."""

from __future__ import annotations

import argparse
import hashlib
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

from p3_wave.champion_lineage_energy_residual import (  # noqa: E402
    apply_champion_energy_residual,
    reconstruct_champion_lineage,
)

EXPERIMENT_ID = "p3_champion_lineage_matched_energy_residual_replay_20260828_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
MODULE = ROOT / "src/p3_wave/champion_lineage_energy_residual.py"
RUNNER = Path(__file__).resolve()
FORBIDDEN = ("test_context", "test_index", "sample_submission", "submission.csv")
KEYS = ["fold", "anchor_id", "station", "lead_h"]
LEADS = (3, 6, 9, 12, 18, 24)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("experiment ID drifted")
    candidate = config["candidate"]
    if candidate["new_model_fit_count"] != 0 or candidate["parameter_search_count"] != 0:
        raise RuntimeError("this replay must not fit or search")
    if tuple(candidate["active_leads_h"]) != (18, 24) or candidate["energy_weight"] != 0.25:
        raise RuntimeError("candidate contract drifted")
    lineage = config["champion_lineage"]
    if tuple(lineage["active_leads_h"]) != (12, 18, 24):
        raise RuntimeError("champion lead support drifted")
    if float(lineage["alpha"]) != -10.21743189862218:
        raise RuntimeError("champion alpha drifted")
    policy = config["execution_policy"]
    false_fields = (
        "result_based_rerun",
        "old_gen6_delta_as_promotion_evidence",
        "official_input_read_authorized",
        "submission_generation_authorized",
        "official_upload_authorized",
    )
    if any(policy.get(field) is not False for field in false_fields):
        raise RuntimeError("a prohibited boundary was opened")
    checked: dict[str, Any] = {}
    for name, spec in config["immutable_inputs"].items():
        path = ROOT / spec["path"]
        if any(token in str(path).lower() for token in FORBIDDEN):
            raise RuntimeError(f"forbidden input path: {path}")
        if not path.is_file() or path.stat().st_size != int(spec["bytes"]):
            raise RuntimeError(f"immutable input missing or size drifted: {name}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(f"immutable input hash drifted: {name}")
        checked[name] = {"bytes": path.stat().st_size, "sha256": digest}
    return {"status": "PASS", "checked": checked, "official_rows_read": 0}


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((truth_array - prediction_array) ** 2)))


def _metric(truth: np.ndarray, champion: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    champion_rmse = _rmse(truth, champion)
    candidate_rmse = _rmse(truth, candidate)
    return {
        "rows": int(len(truth)),
        "champion_rmse_m": champion_rmse,
        "candidate_rmse_m": candidate_rmse,
        "delta_m": candidate_rmse - champion_rmse,
    }


def _bootstrap_cases(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, Any]:
    case_ids = np.asarray(sorted(frame["anchor_id"].unique()))
    groups = {case: frame.index[frame["anchor_id"] == case].to_numpy() for case in case_ids}
    truth = frame["target_hs"].to_numpy(dtype=np.float64)
    champion = frame["champion_prediction"].to_numpy(dtype=np.float64)
    candidate = frame["candidate_prediction"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        rows = np.concatenate([groups[value] for value in sampled])
        deltas[replicate] = _rmse(truth[rows], candidate[rows]) - _rmse(
            truth[rows], champion[rows]
        )
    return {
        "unit": "complete_six_lead_anchor_id_case",
        "cases": int(len(case_ids)),
        "replicates": int(replicates),
        "seed": int(seed),
        "ci90_lower_m": float(np.quantile(deltas, 0.05)),
        "ci90_upper_m": float(np.quantile(deltas, 0.95)),
        "probability_candidate_improves": float(np.mean(deltas < 0.0)),
    }


def _load_prediction_surface(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_path = ROOT / config["immutable_inputs"]["original_oof"]["path"]
    axis_a_path = ROOT / config["immutable_inputs"]["axis_a_oof"]["path"]
    transfer_path = ROOT / config["immutable_inputs"]["sealed_transfer_predictions"]["path"]
    original = pd.read_parquet(original_path, columns=KEYS + ["final_prediction"])
    axis_a = pd.read_parquet(
        axis_a_path,
        columns=KEYS + ["final_prediction", "candidate_prediction"],
    )
    transfer = pd.read_parquet(
        transfer_path,
        columns=KEYS + ["current_hs", "transfer_prediction", "episode_id"],
    )
    expected_rows = int(config["validation"]["expected_rows"])
    if any(len(frame) != expected_rows for frame in (original, axis_a, transfer)):
        raise RuntimeError("BLOCKED_LINEAGE: historical surfaces have different row counts")
    if any(frame.duplicated(KEYS).any() for frame in (original, axis_a, transfer)):
        raise RuntimeError("BLOCKED_LINEAGE: duplicate historical keys")
    original = original.rename(columns={"final_prediction": "o_prediction"})
    axis_a = axis_a.rename(
        columns={
            "final_prediction": "axis_a_embedded_o_prediction",
            "candidate_prediction": "a_prediction",
        }
    )
    merged = original.merge(axis_a, on=KEYS, how="inner", validate="one_to_one", sort=False)
    merged = merged.merge(transfer, on=KEYS, how="inner", validate="one_to_one", sort=False)
    if len(merged) != expected_rows:
        raise RuntimeError("BLOCKED_LINEAGE: O/A/ERA5 key sets do not align")
    embedded_o_exact = bool(
        np.array_equal(
            merged["o_prediction"].to_numpy(),
            merged["axis_a_embedded_o_prediction"].to_numpy(),
        )
    )
    if not embedded_o_exact:
        raise RuntimeError("BLOCKED_LINEAGE: axis A is not anchored to exact historical O")
    lead_contract = merged.groupby("anchor_id", observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    expected_cases = int(config["validation"]["expected_cases"])
    if len(lead_contract) != expected_cases or not lead_contract.map(lambda value: value == LEADS).all():
        raise RuntimeError("BLOCKED_LINEAGE: complete six-lead case contract failed")
    champion, champion_active = reconstruct_champion_lineage(
        merged["o_prediction"].to_numpy(),
        merged["a_prediction"].to_numpy(),
        merged["lead_h"].to_numpy(),
        alpha=float(config["champion_lineage"]["alpha"]),
        active_leads=tuple(int(value) for value in config["champion_lineage"]["active_leads_h"]),
    )
    inactive_axis = ~champion_active
    axis_inactive_exact = bool(
        np.array_equal(champion[inactive_axis], merged.loc[inactive_axis, "o_prediction"].to_numpy())
    )
    if not axis_inactive_exact or not np.isfinite(champion).all():
        raise RuntimeError("BLOCKED_LINEAGE: champion reconstruction failed")
    candidate, energy_active = apply_champion_energy_residual(
        champion,
        merged["transfer_prediction"].to_numpy(),
        merged["lead_h"].to_numpy(),
        energy_weight=float(config["candidate"]["energy_weight"]),
        active_leads=tuple(int(value) for value in config["candidate"]["active_leads_h"]),
    )
    inactive_energy = ~energy_active
    energy_inactive_exact = bool(
        np.array_equal(candidate[inactive_energy], champion[inactive_energy])
    )
    if not energy_inactive_exact or not np.isfinite(candidate).all():
        raise RuntimeError("BLOCKED_LINEAGE: candidate reconstruction failed")
    merged["champion_prediction"] = champion
    merged["candidate_prediction"] = candidate
    merged["champion_axis_active"] = champion_active
    merged["energy_active"] = energy_active
    checks = {
        "o_a_embedded_anchor_bit_exact": embedded_o_exact,
        "champion_inactive_3_6_9_bit_exact": axis_inactive_exact,
        "candidate_inactive_3_6_9_12_bit_exact": energy_inactive_exact,
        "complete_181_case_surface": len(lead_contract) == expected_cases,
        "old_gen6_delta_used_as_promotion_evidence": False,
    }
    return merged, checks


def _result_report(result: dict[str, Any]) -> str:
    overall = result["overall"]
    bootstrap = result["paired_case_bootstrap"]
    return (
        f"# {EXPERIMENT_ID}\n\n"
        f"- terminal decision: `{result['status']}`\n"
        f"- champion RMSE: `{overall['champion_rmse_m']:.12f} m`\n"
        f"- candidate RMSE: `{overall['candidate_rmse_m']:.12f} m`\n"
        f"- delta candidate - champion: `{overall['delta_m']:+.12f} m`\n"
        f"- improved folds: `{result['improved_fold_count']}/3`\n"
        f"- maximum station/lead regression: `{result['maximum_station_or_lead_regression_m']:+.12f} m`\n"
        f"- case bootstrap CI90: `[{bootstrap['ci90_lower_m']:+.12f}, "
        f"{bootstrap['ci90_upper_m']:+.12f}] m`\n"
        f"- official rows read / submission generated / upload: `0 / false / 0`\n\n"
        "This is an aggregate-only historical replay. It does not reuse the old Gen6 delta as promotion evidence.\n"
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    receipt = validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    report_path = ROOT / config["artifacts"]["report"]
    if output.exists() or report_path.exists():
        raise FileExistsError("one-shot output already exists")
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        output / "ATTEMPT_LOCK.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "ONE_SHOT_ATTEMPT_CONSUMED",
            "official_scope": False,
        },
    )
    prediction_surface, lineage_checks = _load_prediction_surface(config)
    sealed_columns = KEYS + [
        "current_hs",
        "o_prediction",
        "a_prediction",
        "champion_prediction",
        "transfer_prediction",
        "candidate_prediction",
        "champion_axis_active",
        "energy_active",
        "episode_id",
    ]
    sealed = prediction_surface[sealed_columns].copy()
    sealed_path = output / "sealed_candidate_predictions.parquet"
    sealed.to_parquet(sealed_path, index=False)
    seal = {
        "experiment_id": EXPERIMENT_ID,
        "rows": int(len(sealed)),
        "cases": int(sealed["anchor_id"].nunique()),
        "target_columns_present": [],
        "sha256": sha256_file(sealed_path),
        "truth_attached_after_this_seal": True,
        "official_rows_read": 0,
    }
    _atomic_json(output / "PREDICTION_SEAL.json", seal)

    original_path = ROOT / config["immutable_inputs"]["original_oof"]["path"]
    truth = pd.read_parquet(original_path, columns=KEYS + ["target_hs"])
    scored = sealed.merge(truth, on=KEYS, how="inner", validate="one_to_one", sort=False)
    if len(scored) != len(sealed):
        raise RuntimeError("truth attach key mismatch after prediction seal")
    overall = _metric(
        scored["target_hs"].to_numpy(),
        scored["champion_prediction"].to_numpy(),
        scored["candidate_prediction"].to_numpy(),
    )
    by_fold = {
        str(name): _metric(
            group["target_hs"].to_numpy(),
            group["champion_prediction"].to_numpy(),
            group["candidate_prediction"].to_numpy(),
        )
        for name, group in scored.groupby("fold", sort=True)
    }
    by_station = {
        str(name): _metric(
            group["target_hs"].to_numpy(),
            group["champion_prediction"].to_numpy(),
            group["candidate_prediction"].to_numpy(),
        )
        for name, group in scored.groupby("station", sort=True)
    }
    by_lead = {
        str(int(name)): _metric(
            group["target_hs"].to_numpy(),
            group["champion_prediction"].to_numpy(),
            group["candidate_prediction"].to_numpy(),
        )
        for name, group in scored.groupby("lead_h", sort=True)
    }
    bootstrap = _bootstrap_cases(
        scored.reset_index(drop=True),
        int(config["validation"]["bootstrap_replicates"]),
        int(config["validation"]["bootstrap_seed"]),
    )
    improved_folds = sum(metric["delta_m"] < 0.0 for metric in by_fold.values())
    maximum_regression = max(
        [metric["delta_m"] for metric in by_station.values()]
        + [metric["delta_m"] for metric in by_lead.values()]
    )
    gate = {
        "overall_delta_below_zero": overall["delta_m"] < 0.0,
        "minimum_improved_folds": improved_folds
        >= int(config["promotion_gate"]["minimum_improved_folds"]),
        "maximum_station_or_lead_regression": maximum_regression
        <= float(config["promotion_gate"]["maximum_any_station_or_lead_regression_m"]),
        "lineage_preflight": all(value is True or value is False for value in lineage_checks.values())
        and all(value is True for key, value in lineage_checks.items() if key != "old_gen6_delta_used_as_promotion_evidence"),
    }
    passed = all(gate.values())
    result = {
        "schema_version": "p3.champion_lineage_matched_energy_residual_replay.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "GO_OFFICIAL_PROBE_LINEAGE_MATCHED" if passed else "NO_GO_LINEAGE_MATCHED_LOCAL_GATE",
        "promotion": "official-probe-eligible-not-generated" if passed else "no-go",
        "champion_lineage": config["champion_lineage"],
        "candidate": config["candidate"],
        "lineage_checks": lineage_checks,
        "surface": {
            "rows": int(len(scored)),
            "cases": int(scored["anchor_id"].nunique()),
            "active_rows": int(scored["energy_active"].sum()),
            "inactive_rows": int((~scored["energy_active"]).sum()),
        },
        "overall": overall,
        "by_fold": by_fold,
        "by_station": by_station,
        "by_lead": by_lead,
        "paired_case_bootstrap": bootstrap,
        "improved_fold_count": int(improved_folds),
        "maximum_station_or_lead_regression_m": float(maximum_regression),
        "gate": gate,
        "fits": 0,
        "parameter_searches": 0,
        "official_rows_read": 0,
        "candidate_or_submission_created": False,
        "upload_count": 0,
        "old_gen6_delta_used_as_promotion_evidence": False,
    }
    result_path = output / "result.json"
    _atomic_json(result_path, result)
    _atomic_text(report_path, _result_report(result))
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "module_sha256": sha256_file(MODULE),
        "runner_sha256": sha256_file(RUNNER),
        "immutable_inputs": receipt["checked"],
        "prediction_seal_sha256": sha256_file(output / "PREDICTION_SEAL.json"),
        "sealed_candidate_sha256": sha256_file(sealed_path),
        "result_sha256": sha256_file(result_path),
        "report_sha256": sha256_file(report_path),
        "sealed_candidate_contains_target": False,
        "official_rows_read": 0,
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
