"""Exactly-once zero-fit G-ORS causal one-step run extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_full_internal_submission_cycle_20260831_v2 as source_cycle  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v19"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
PREFLIGHT_REPORT = REPORT / "preflight-report.json"
DUPLICATE_AUDIT = REPORT / "duplicate-audit.md"
CALIBRATION_PATH = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
OFFICIAL_PRIOR_PATH = ROOT / "reports/official_information_probe_cycle_20260830_v1/p1-official-result.json"
NEGATIVE_REGISTRY_PATH = ROOT / "reports/negative_evidence_registry_20260830_v1/failure-ledger.json"
FOLDS = ("2025_q2", "2025_q3", "2025_q4")
CONFIRMATORY_FOLDS = FOLDS[1:]
KEYS = ["station", "year", "layer", "time"]


class ContractError(RuntimeError):
    """Frozen rule or access contract violation."""


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(native(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    family = config["transport_family"]
    policy = config["decision_policy"]
    family_gate = calibration["family_gates"][family["family_id"]]
    checks = {
        "calibration_sha": family["selected_penalty_provenance_sha256"]
        == sha256_file(CALIBRATION_PATH),
        "family": family["family_id"] == "P1_FIXED_ADD_ONLY_UNION",
        "tier": family["tier_id"] == "LOW_DOF_FIXED",
        "representation": family["representation_changed"] is False,
        "routing": family["routing_discontinuous"] is False,
        "penalty": np.isclose(
            policy["transport_penalty_points"],
            family_gate["transport_penalty_points"],
            atol=1e-15,
        ),
        "raw": np.isclose(
            policy["minimum_raw_expected_point_delta_inclusive"],
            family_gate["minimum_raw_expected_points_delta"],
            atol=1e-15,
        ),
        "station": config["candidate"]["station"] == "G-ORS",
        "cadence": config["candidate"]["cadence_minutes"] == 10,
        "span": config["candidate"]["span_rows"] == 1,
        "nonrecursive": config["candidate"]["recursive_extension"] is False,
        "zero_fit": config["fit_budget"]["maximum_model_fits"] == 0,
        "one_pass": config["validation"]["deterministic_evaluation_passes"] == 1,
    }
    if not all(checks.values()):
        raise ContractError(f"v19 contract mismatch: {checks}")
    return config


def load_historical_frame() -> pd.DataFrame:
    frame, _ = source_cycle.p1_frame()
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    if len(frame) != 421_032:
        raise ContractError("historical frame row contract changed")
    if frame.duplicated([*KEYS, "fold"]).any():
        raise ContractError("historical keys are not unique")
    return frame


def gors_causal_lag1_mask(
    frame: pd.DataFrame, reference: np.ndarray
) -> np.ndarray:
    """Return one non-recursive G-ORS row after an exact-cadence reference run."""
    if len(frame) != len(reference):
        raise ContractError("reference length mismatch")
    work = pd.DataFrame(
        {
            "station": frame["station"].astype(str),
            "year": frame["year"].astype(int),
            "layer": frame["layer"].astype(int),
            "time": pd.to_datetime(frame["time"], utc=True),
            "position": np.arange(len(frame), dtype=np.int64),
            "reference": np.asarray(reference, dtype=np.int8),
        }
    )
    work.sort_values(
        ["station", "year", "layer", "time", "position"],
        kind="stable",
        inplace=True,
    )
    grouped = work.groupby(
        ["station", "year", "layer"], sort=False, observed=True
    )
    prior_reference = grouped["reference"].shift(1).fillna(0).to_numpy(np.int8)
    exact_cadence = grouped["time"].diff().eq(pd.Timedelta(minutes=10)).to_numpy()
    additions_sorted = (
        work["station"].eq("G-ORS").to_numpy()
        & (prior_reference == 1)
        & exact_cadence
        & work["reference"].eq(0).to_numpy()
    )
    additions = np.zeros(len(frame), dtype=bool)
    additions[work["position"].to_numpy(np.int64)] = additions_sorted
    return additions


def build_candidate(frame: pd.DataFrame, reference: np.ndarray) -> np.ndarray:
    candidate = np.asarray(reference, dtype=np.int8).copy()
    candidate[gors_causal_lag1_mask(frame, reference)] = 1
    return candidate


def f1_counts(truth: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int]:
    return (
        int(((truth == 1) & (prediction == 1)).sum()),
        int(((truth == 0) & (prediction == 1)).sum()),
        int(((truth == 1) & (prediction == 0)).sum()),
    )


def f1_from_counts(values: np.ndarray) -> float:
    tp, fp, fn = values
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def day_bootstrap(
    frame: pd.DataFrame,
    reference: np.ndarray,
    candidate: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    evaluated = frame["fold"].isin(CONFIRMATORY_FOLDS).to_numpy()
    day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
    )
    table = pd.DataFrame(
        {
            "fold": frame["fold"],
            "day": day,
            "truth": truth,
            "reference": reference,
            "candidate": candidate,
            "evaluated": evaluated,
        }
    )
    blocks = []
    for _, group in table.loc[table.evaluated].groupby(
        ["fold", "day"], sort=True
    ):
        y = group["truth"].to_numpy(np.int8)
        blocks.append(
            (
                *f1_counts(y, group["reference"].to_numpy(np.int8)),
                *f1_counts(y, group["candidate"].to_numpy(np.int8)),
            )
        )
    counts = np.asarray(blocks, dtype=np.int64)
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    delta = np.empty(
        int(config["validation"]["bootstrap_replicates"]), dtype=np.float64
    )
    for index in range(len(delta)):
        total = counts[
            rng.integers(0, len(counts), size=len(counts))
        ].sum(axis=0)
        delta[index] = f1_from_counts(total[3:]) - f1_from_counts(total[:3])
    return {
        "method": "KST calendar-day block bootstrap within Q3/Q4",
        "blocks": len(blocks),
        "replicates": len(delta),
        "mean_delta_f1": float(delta.mean()),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta > 0.0)),
    }


def evaluate(
    frame: pd.DataFrame, candidate: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    reference = frame["e150_prediction"].to_numpy(np.int8)
    evaluated = frame["fold"].isin(CONFIRMATORY_FOLDS).to_numpy()
    additions = evaluated & (candidate == 1) & (reference == 0)
    removals = evaluated & (candidate == 0) & (reference == 1)
    reference_f1 = float(f1_score(truth[evaluated], reference[evaluated]))
    candidate_f1 = float(f1_score(truth[evaluated], candidate[evaluated]))
    delta_f1 = candidate_f1 - reference_f1

    by_fold: dict[str, Any] = {}
    for fold in FOLDS:
        mask = frame["fold"].eq(fold).to_numpy()
        fold_additions = mask & (candidate == 1) & (reference == 0)
        base = float(f1_score(truth[mask], reference[mask]))
        score = float(f1_score(truth[mask], candidate[mask]))
        by_fold[fold] = {
            "rows": int(mask.sum()),
            "reference_f1": base,
            "candidate_f1": score,
            "delta_f1": score - base,
            "additions": int(fold_additions.sum()),
            "true_positive_additions": int(
                (fold_additions & (truth == 1)).sum()
            ),
            "false_positive_additions": int(
                (fold_additions & (truth == 0)).sum()
            ),
            "anchor_removals": int((mask & removals).sum()),
        }

    bootstrap = day_bootstrap(frame, reference, candidate, config)
    day = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
    )
    addition_frame = frame.loc[additions, ["station"]].copy()
    addition_frame["day"] = day[additions].to_numpy()
    maximum_gors_day_additions = (
        int(addition_frame.groupby(["station", "day"], observed=True).size().max())
        if len(addition_frame)
        else 0
    )

    station_layer_delta: dict[str, float] = {}
    for station, layer in (
        frame.loc[evaluated, ["station", "layer"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        mask = (
            evaluated
            & frame["station"].eq(station).to_numpy()
            & frame["layer"].eq(layer).to_numpy()
        )
        base = float(f1_score(truth[mask], reference[mask]))
        score = float(f1_score(truth[mask], candidate[mask]))
        station_layer_delta[f"{station}|{layer}"] = score - base

    tp_additions = int((additions & (truth == 1)).sum())
    fp_additions = int((additions & (truth == 0)).sum())
    precision = tp_additions / int(additions.sum()) if additions.any() else None
    precision_lcb = (
        float(
            beta_distribution.ppf(
                float(config["safety"]["precision_lcb_quantile"]),
                tp_additions,
                fp_additions + 1,
            )
        )
        if tp_additions
        else 0.0
    )
    policy = config["decision_policy"]
    raw_points = delta_f1 * float(policy["score_points_per_f1"])
    calibrated = raw_points - float(policy["transport_penalty_points"])
    other = frame["station"].ne("G-ORS").to_numpy()
    other_bit_exact = bool(np.array_equal(candidate[other], reference[other]))
    gors_deltas = [
        value
        for key, value in station_layer_delta.items()
        if key.startswith("G-ORS|")
    ]
    gates = {
        "positive_additions": int(additions.sum()) > 0,
        "anchor_removals_zero": int(removals.sum()) == 0,
        "q3_q4_each_nonnegative": min(
            by_fold[fold]["delta_f1"] for fold in CONFIRMATORY_FOLDS
        )
        >= float(policy["minimum_each_q3_q4_delta_f1"]),
        "pooled_delta_positive": delta_f1
        > float(policy["minimum_pooled_delta_f1_exclusive"]),
        "bootstrap_probability_at_least_0_8": bootstrap["probability_improved"]
        >= float(policy["bootstrap_probability_improved_minimum_inclusive"]),
        "bootstrap_ci90_low_strictly_positive": bootstrap["ci90_low"]
        > float(policy["bootstrap_ci90_low_exclusive"]),
        "raw_expected_points_at_least_0_015383691": raw_points
        >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_at_least_0_01": calibrated
        >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "marginal_precision_lcb_above_reference_f1_half": precision_lcb
        > reference_f1 / 2.0,
        "changed_fraction_at_most_0_005": float(additions.sum() / evaluated.sum())
        <= float(config["safety"]["maximum_changed_fraction"]),
        "maximum_five_additions_per_gors_day": maximum_gors_day_additions
        <= int(config["safety"]["maximum_additions_per_gors_kst_day"]),
        "supported_gors_nonnegative": bool(gors_deltas)
        and min(gors_deltas)
        >= float(config["safety"]["minimum_supported_gors_delta_f1"]),
        "other_stations_bit_exact": other_bit_exact,
    }
    return {
        "name": config["candidate"]["name"],
        "family": config["transport_family"]["family_id"],
        "development_confirmation_only": True,
        "past_only": True,
        "learned_parameters": 0,
        "fit_count": 0,
        "deterministic_evaluation_passes": 1,
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": delta_f1,
        "raw_expected_points_delta": raw_points,
        "transport_penalty_points": float(policy["transport_penalty_points"]),
        "calibrated_conservative_expected_points_delta": calibrated,
        "additions": int(additions.sum()),
        "true_positive_additions": tp_additions,
        "false_positive_additions": fp_additions,
        "additions_precision": precision,
        "additions_precision_lcb90": precision_lcb,
        "reference_f1_divided_by_2": reference_f1 / 2.0,
        "anchor_removals": int(removals.sum()),
        "changed_fraction": float(additions.sum() / evaluated.sum()),
        "maximum_gors_day_additions": maximum_gors_day_additions,
        "station_layer_delta_f1": station_layer_delta,
        "other_stations_bit_exact": other_bit_exact,
        "by_fold": by_fold,
        "day_block_bootstrap": bootstrap,
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }


def _resolve_p1_data_dir() -> Path:
    configured = os.environ.get("P1_DATA_DIR")
    return Path(configured) if configured else Path(source_cycle.P1_DATA)


def materialize_if_pass(
    record: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counters = {
        "official_test_key_rows_read": 0,
        "official_champion_rows_read": 0,
        "submission_csv_created": 0,
    }
    if not record["strict_internal_pass"]:
        return [], counters

    champion_path = Path(source_cycle.P1_CHAMPION)
    if sha256_file(champion_path) != config["official_policy"][
        "current_champion_sha256"
    ]:
        raise ContractError("current champion hash changed")
    test_path = _resolve_p1_data_dir() / "test.csv"
    test = pd.read_csv(
        test_path,
        usecols=KEYS,
        dtype={"station": "string", "year": "int64", "layer": "int64", "time": "string"},
    )
    counters["official_test_key_rows_read"] = len(test)
    champion = pd.read_csv(
        champion_path,
        usecols=[*KEYS, "label"],
        dtype={"station": "string", "year": "int64", "layer": "int64", "time": "string", "label": "int8"},
    )
    counters["official_champion_rows_read"] = len(champion)
    if len(test) != 169_011 or not champion[KEYS].equals(test[KEYS]):
        raise ContractError("official champion key/order mismatch")
    reference = champion["label"].to_numpy(np.int8)
    candidate = build_candidate(test, reference)
    if not np.all(candidate >= reference):
        raise ContractError("deployment candidate removed a champion row")
    output = test.copy()
    output["label"] = candidate
    output_path = ARTIFACT / "submission" / "P1_submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    counters["submission_csv_created"] = 1
    verify = pd.read_csv(
        output_path,
        dtype={"station": "string", "year": "int64", "layer": "int64", "time": "string", "label": "int8"},
    )
    structural = {
        "rows_169011": len(verify) == 169_011,
        "schema_exact": list(verify.columns) == [*KEYS, "label"],
        "keys_exact_order": verify[KEYS].equals(test[KEYS]),
        "keys_unique": not verify.duplicated(KEYS).any(),
        "labels_binary": set(verify["label"].unique()).issubset({0, 1}),
        "labels_finite": bool(np.isfinite(verify["label"].to_numpy(float)).all()),
        "champion_removals_zero": bool(np.all(verify["label"].to_numpy(np.int8) >= reference)),
    }
    if not all(structural.values()):
        raise ContractError(f"submission structural QA failed: {structural}")
    return [
        {
            "name": config["candidate"]["name"],
            "path": str(output_path),
            "rows": len(verify),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "positive_rows": int(verify["label"].sum()),
            "additions_vs_champion": int(
                (verify["label"].to_numpy(np.int8) > reference).sum()
            ),
            "structural_qa": structural,
        }
    ], counters


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    record = result["candidate"]
    checks = {
        "one_candidate": result["candidate_count"] == 1,
        "zero_model_fits": result["fit_count"] == 0,
        "one_deterministic_pass": result["deterministic_evaluation_passes"] == 1,
        "exact_fixed_family": result["transport_family"]["family_id"]
        == "P1_FIXED_ADD_ONLY_UNION",
        "gors_only_formula": result["candidate_contract"]["station"] == "G-ORS",
        "exact_ten_minute_cadence": result["candidate_contract"]["cadence_minutes"]
        == 10,
        "one_row_nonrecursive": result["candidate_contract"]["span_rows"] == 1
        and result["candidate_contract"]["recursive_extension"] is False,
        "anchor_removals_zero": record["anchor_removals"] == 0,
        "other_stations_bit_exact": record["other_stations_bit_exact"],
        "calibration_hash": result["transport_family"][
            "selected_penalty_provenance_sha256"
        ]
        == result["hashes"]["root_calibration_sha256"],
        "official_prior_hash": result["official_support_prior"]["source_sha256"]
        == result["hashes"]["official_support_prior_sha256"],
        "development_not_independent": result["independent_confirmation_claimed"]
        is False,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
        "materialization_only_if_pass": bool(result["outputs"])
        == bool(record["strict_internal_pass"]),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def synthetic_preflight() -> dict[str, Any]:
    config = load_contract()
    frame = pd.DataFrame(
        {
            "station": ["G-ORS", "G-ORS", "G-ORS", "G-ORS", "I-ORS", "I-ORS"],
            "year": [2025] * 6,
            "layer": [1] * 6,
            "time": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:20:00Z",
                    "2025-01-01T01:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:10:00Z",
                ],
                utc=True,
            ),
        }
    )
    reference = np.array([1, 0, 0, 1, 1, 0], dtype=np.int8)
    additions = gors_causal_lag1_mask(frame, reference)
    candidate = build_candidate(frame, reference)
    other = frame["station"].ne("G-ORS").to_numpy()
    checks = {
        "contract_valid": config["candidate"]["name"]
        == "P1_1_GORS_CAUSAL_ONE_STEP_RUN_EXTENSION",
        "synthetic_exact_mask": additions.tolist()
        == [False, True, False, False, False, False],
        "nonrecursive": candidate.tolist() == [1, 1, 0, 1, 1, 0],
        "other_stations_bit_exact": bool(
            np.array_equal(candidate[other], reference[other])
        ),
        "add_only": bool(np.all(candidate >= reference)),
        "duplicate_audit_exists": DUPLICATE_AUDIT.is_file(),
        "frozen_inputs_exist": all(
            path.is_file()
            for path in (
                CALIBRATION_PATH,
                OFFICIAL_PRIOR_PATH,
                NEGATIVE_REGISTRY_PATH,
            )
        ),
        "artifact_absent_before_execution": not ARTIFACT.exists(),
    }
    return {
        "schema_version": "p1.v19.synthetic_preflight.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "duplicate_audit_sha256": sha256_file(DUPLICATE_AUDIT),
            "calibration_sha256": sha256_file(CALIBRATION_PATH),
            "official_support_prior_sha256": sha256_file(OFFICIAL_PRIOR_PATH),
        },
        "fit_budget": 0,
        "access": {
            "historical_rows_read": 0,
            "official_rows_read": 0,
            "hidden_truth_reads": 0,
            "attempt_locks_created": 0,
            "uploads": 0,
        },
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError("v19 exactly-once artifact path exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "duplicate_audit_sha256": sha256_file(DUPLICATE_AUDIT),
        "model_fit_budget": 0,
        "deterministic_evaluation_passes": 1,
        "official_reads_before_internal_pass": 0,
        "hidden_truth_reads": 0,
        "uploads": 0,
    }
    write_json(ARTIFACT / "attempt_lock.json", lock)
    write_json(
        ARTIFACT / "progress.json",
        {"phase": "historical_scoring", "fit_count": 0, "evaluation_passes": 0},
    )
    frame = load_historical_frame()
    reference = frame["e150_prediction"].to_numpy(np.int8)
    candidate = build_candidate(frame, reference)
    record = evaluate(frame, candidate, config)
    outputs, official_counters = materialize_if_pass(record, config)
    operations = {
        "historical_rows_read": len(frame),
        "official_covariate_reads_after_internal_pass": int(
            official_counters["official_test_key_rows_read"] > 0
        ),
        **official_counters,
        "hidden_truth_reads": 0,
        "uploads": 0,
    }
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v19.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_AND_MATERIALIZED"
        if outputs
        else "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "candidate_contract": config["candidate"],
        "transport_family": config["transport_family"],
        "decision_policy": config["decision_policy"],
        "official_support_prior": config["official_support_prior"],
        "duplicate_provenance": config["duplicate_provenance"],
        "independent_confirmation_claimed": False,
        "candidate_count": 1,
        "pass_count": int(record["strict_internal_pass"]),
        "fit_count": 0,
        "deterministic_evaluation_passes": 1,
        "candidate": record,
        "outputs": outputs,
        "operations": operations,
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "lock_sha256": sha256_file(ARTIFACT / "attempt_lock.json"),
            "duplicate_audit_sha256": sha256_file(DUPLICATE_AUDIT),
            "root_calibration_sha256": sha256_file(CALIBRATION_PATH),
            "official_support_prior_sha256": sha256_file(OFFICIAL_PRIOR_PATH),
            "negative_registry_sha256": sha256_file(NEGATIVE_REGISTRY_PATH),
        },
    }
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    write_json(
        ARTIFACT / "progress.json",
        {"phase": "terminal", "fit_count": 0, "evaluation_passes": 1, "pass_count": result["pass_count"]},
        exclusive=False,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        parser.error("choose exactly one mode")
    try:
        if args.preflight:
            payload = synthetic_preflight()
            write_json(PREFLIGHT_REPORT, payload, exclusive=not PREFLIGHT_REPORT.exists())
        else:
            payload = execute()
        print(json.dumps(native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if payload.get("status") in {"PASS", "COMPLETE_INTERNAL_ONLY", "COMPLETE_INTERNAL_AND_MATERIALIZED"} else 1
    except Exception as exc:  # noqa: BLE001
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "official_covariate_reads": 0,
            "hidden_truth_reads": 0,
            "uploads": 0,
        }
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
