"""Exactly-once chronological P1 anchor false-positive suppressor."""

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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_full_internal_submission_cycle_20260831_v2 as source_cycle  # noqa: E402
import run_p1_parallel_candidate_cycle_20260831_v4 as feature_cycle  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v11"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CALIBRATION_PATH = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
AUTHORITATIVE_PATH = ROOT / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json"
ANCHOR_PATH = ROOT / "artifacts/p1_current_router_oof_anchor_v1/anchor.parquet"
FOLDS = ("2025_q2", "2025_q3", "2025_q4")


class ContractError(RuntimeError):
    """Frozen experiment contract violation."""


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
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
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(native(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    policy = config["decision_policy"]
    family = config["transport_family"]
    tier = calibration["tier_gates"][family["tier_id"]]
    checks = {
        "calibration_sha": family["calibration_sha256"] == sha256_file(CALIBRATION_PATH),
        "tier": family["tier_id"] == "HARD_CONDITIONAL_ROUTER",
        "representation": family["representation_changed"] is True,
        "routing": family["routing_discontinuous"] is True,
        "penalty": np.isclose(policy["transport_penalty_points"], tier["transport_penalty_points"], atol=1e-15),
        "raw": np.isclose(policy["minimum_raw_expected_point_delta_inclusive"], tier["minimum_raw_expected_points_delta"], atol=1e-15),
    }
    if not all(checks.values()):
        raise ContractError(f"family-aware transport mismatch: {checks}")
    if len(config["models"]) != 3 or config["fit_budget"]["maximum"] != 18:
        raise ContractError("model or fit budget changed")
    return config


def load_frame(config: dict[str, Any]) -> pd.DataFrame:
    frame, _ = source_cycle.p1_frame()
    frame, actual = feature_cycle.add_causal_features(frame)
    missing = sorted(set(config["features"]) - set(actual))
    if missing:
        raise ContractError(f"missing frozen causal features: {missing}")
    anchor = pd.read_parquet(ANCHOR_PATH)
    anchor["time"] = pd.to_datetime(anchor["time"], utc=True)
    keys = ["station", "year", "layer", "time", "fold"]
    frame = frame.merge(anchor, on=keys, validate="one_to_one")
    if len(frame) != 421_032 or not set(frame["current_router_prediction"].unique()).issubset({0, 1}):
        raise ContractError("current-router historical anchor differs")
    return frame


def inner_split(
    frame: pd.DataFrame,
    train_folds: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    timestamp = pd.to_datetime(frame["time"], utc=True)
    outer_train = frame["fold"].isin(train_folds).to_numpy()
    calibration_start = timestamp[outer_train].max() - pd.Timedelta(days=int(config["validation"]["inner_calibration_days"]))
    fit_end = calibration_start - pd.Timedelta(hours=int(config["validation"]["purge_hours"]))
    fit_mask = outer_train & timestamp.lt(fit_end).to_numpy()
    calibration_mask = outer_train & timestamp.ge(calibration_start).to_numpy()
    return fit_mask, calibration_mask, {
        "fit_end_exclusive": fit_end.isoformat(),
        "calibration_start_inclusive": calibration_start.isoformat(),
        "purge_hours": int(config["validation"]["purge_hours"]),
    }


def build_model(spec: dict[str, Any], seed: int) -> Any:
    if spec["family"] == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(spec["C"]), max_iter=int(spec["max_iter"]), class_weight="balanced", random_state=seed),
        )
    if spec["family"] == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(spec["learning_rate"]),
            max_iter=int(spec["max_iter"]),
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2_regularization"]),
            random_state=seed,
        )
    if spec["family"] == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=int(spec["n_jobs"]),
        )
    raise ContractError(f"unknown model family: {spec['family']}")


def fit_models(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    config: dict[str, Any],
) -> list[Any]:
    x = frame[config["features"]].to_numpy(np.float64)
    risk = frame["label_base"].eq(0).to_numpy(np.int8)
    source = fit_mask & frame["current_router_prediction"].eq(1).to_numpy()
    if source.sum() < 500 or np.unique(risk[source]).size != 2:
        raise ContractError("anchor-positive fit surface lacks two-class support")
    models = []
    for seed in config["validation"]["seeds"]:
        model = build_model(spec, int(seed))
        if spec["family"] == "hist_gradient_boosting":
            model.fit(x[source], risk[source], sample_weight=compute_sample_weight("balanced", risk[source]))
        else:
            model.fit(x[source], risk[source])
        models.append(model)
    return models


def score_models(models: list[Any], x: np.ndarray) -> np.ndarray:
    return np.vstack([model.predict_proba(x)[:, 1] for model in models])


def guarded_removals(
    frame: pd.DataFrame,
    scope: np.ndarray,
    score: np.ndarray,
    fraction: float,
    denominator_rows: int,
    config: dict[str, Any],
    *,
    threshold: float = -np.inf,
) -> tuple[np.ndarray, dict[str, Any]]:
    budget = int(np.floor(float(fraction) * denominator_rows))
    candidates = np.flatnonzero(scope & (score >= threshold))
    removals = np.zeros(len(frame), dtype=bool)
    if budget < 2 or len(candidates) < 2:
        return removals, {"budget": budget, "status": "INSUFFICIENT_MULTI_STATION_BUDGET"}
    ordered = candidates[np.argsort(-score[candidates], kind="stable")]
    station = frame["station"].astype(str).to_numpy()
    station_cap = max(1, int(np.floor(float(config["decision_policy"]["maximum_station_intervention_share"]) * budget)))
    used: dict[str, int] = {}
    for index in ordered:
        key = station[index]
        if used.get(key, 0) >= station_cap:
            continue
        removals[index] = True
        used[key] = used.get(key, 0) + 1
        if removals.sum() >= budget:
            break
    shares = [count / int(removals.sum()) for count in used.values()] if removals.any() else []
    if removals.any() and (len(used) < 2 or max(shares) > float(config["decision_policy"]["maximum_station_intervention_share"])):
        removals[:] = False
        return removals, {"budget": budget, "status": "STATION_CONCENTRATION_ABSTAIN", "provisional_station_counts": used}
    return removals, {"budget": budget, "status": "SELECTED", "station_counts": used, "maximum_station_share": max(shares, default=0.0)}


def select_inner_rule(
    frame: pd.DataFrame,
    calibration_mask: np.ndarray,
    score: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    anchor = frame["current_router_prediction"].to_numpy(np.int8)
    scope = calibration_mask & (anchor == 1)
    reference = float(f1_score(truth[calibration_mask], anchor[calibration_mask]))
    records = []
    for fraction in config["validation"]["budget_fractions"]:
        removals, guard = guarded_removals(frame, scope, score, float(fraction), int(calibration_mask.sum()), config)
        candidate = anchor.copy()
        candidate[removals] = 0
        tp_removals = int((removals & (truth == 1)).sum())
        delta = float(f1_score(truth[calibration_mask], candidate[calibration_mask]) - reference)
        selected_scores = score[removals]
        records.append(
            {
                "fraction": float(fraction),
                "threshold": float(selected_scores.min()) if len(selected_scores) else float("inf"),
                "removals": int(removals.sum()),
                "false_positive_removals": int((removals & (truth == 0)).sum()),
                "true_positive_removals": tp_removals,
                "delta_f1": delta,
                "guard": guard,
            }
        )
    eligible = [item for item in records if item["removals"] > 0 and item["true_positive_removals"] == 0 and item["delta_f1"] > 0]
    if not eligible:
        return {"status": "INNER_TP_GUARD_ABSTAIN", "fraction": 0.0, "threshold": float("inf"), "records": records}
    best = max(eligible, key=lambda item: (item["delta_f1"], item["false_positive_removals"], -item["fraction"]))
    return {"status": "INNER_RULE_FROZEN", **best, "records": records}


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


def day_bootstrap(frame: pd.DataFrame, evaluated: np.ndarray, anchor: np.ndarray, candidate: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    truth = frame["label_base"].to_numpy(np.int8)
    local_day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    table = pd.DataFrame({"fold": frame["fold"], "day": local_day, "truth": truth, "anchor": anchor, "candidate": candidate, "evaluated": evaluated})
    blocks = []
    for _, group in table.loc[table["evaluated"]].groupby(["fold", "day"], sort=True):
        y = group.truth.to_numpy(np.int8)
        blocks.append((*f1_counts(y, group.anchor.to_numpy(np.int8)), *f1_counts(y, group.candidate.to_numpy(np.int8))))
    counts = np.asarray(blocks, dtype=np.int64)
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    delta = np.empty(int(config["validation"]["bootstrap_replicates"]), dtype=float)
    for index in range(len(delta)):
        total = counts[rng.integers(0, len(counts), size=len(counts))].sum(axis=0)
        delta[index] = f1_from_counts(total[3:6]) - f1_from_counts(total[0:3])
    return {
        "blocks": len(blocks),
        "replicates": len(delta),
        "mean_delta_f1": float(delta.mean()),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta > 0)),
    }


def evaluate(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    x = frame[config["features"]].to_numpy(np.float64)
    truth = frame["label_base"].to_numpy(np.int8)
    anchor = frame["current_router_prediction"].to_numpy(np.int8)
    ensemble = anchor.copy()
    seed_predictions = [anchor.copy() for _ in config["validation"]["seeds"]]
    receipts = {}
    fit_count = 0
    for outer in config["validation"]["outer_forward_tests"]:
        fit_mask, calibration_mask, split = inner_split(frame, outer["train_folds"], config)
        models = fit_models(spec, frame, fit_mask, config)
        fit_count += len(models)
        scores = score_models(models, x)
        rule = select_inner_rule(frame, calibration_mask, scores.mean(axis=0), config)
        outer_mask = frame["fold"].eq(outer["test_fold"]).to_numpy()
        scope = outer_mask & (anchor == 1)
        removals, guard = guarded_removals(frame, scope, scores.mean(axis=0), float(rule["fraction"]), int(outer_mask.sum()), config, threshold=float(rule["threshold"]))
        ensemble[outer_mask] = anchor[outer_mask]
        ensemble[removals] = 0
        seed_guards = []
        for ordinal, seed_score in enumerate(scores):
            seed_removals, seed_guard = guarded_removals(frame, scope, seed_score, float(rule["fraction"]), int(outer_mask.sum()), config, threshold=float(rule["threshold"]))
            seed_predictions[ordinal][outer_mask] = anchor[outer_mask]
            seed_predictions[ordinal][seed_removals] = 0
            seed_guards.append(seed_guard)
        receipts[outer["test_fold"]] = {
            "train_folds": outer["train_folds"],
            "split": split,
            "inner_rule": rule,
            "outer_guard": guard,
            "outer_anchor_positive_rows": int(scope.sum()),
            "outer_removals": int(removals.sum()),
            "seed_guards": seed_guards,
        }
    evaluated = frame["fold"].isin(FOLDS[1:]).to_numpy()
    reference_f1 = float(f1_score(truth[evaluated], anchor[evaluated]))
    candidate_f1 = float(f1_score(truth[evaluated], ensemble[evaluated]))
    delta_f1 = candidate_f1 - reference_f1
    removals = evaluated & (anchor == 1) & (ensemble == 0)
    by_fold = {}
    for fold in FOLDS[1:]:
        mask = frame["fold"].eq(fold).to_numpy()
        fold_removals = mask & removals
        base = float(f1_score(truth[mask], anchor[mask]))
        value = float(f1_score(truth[mask], ensemble[mask]))
        by_fold[fold] = {
            "rows": int(mask.sum()),
            "reference_f1": base,
            "candidate_f1": value,
            "delta_f1": value - base,
            "removals": int(fold_removals.sum()),
            "false_positive_removals": int((fold_removals & (truth == 0)).sum()),
            "true_positive_removals": int((fold_removals & (truth == 1)).sum()),
        }
    station_counts = frame.loc[removals, "station"].astype(str).value_counts()
    max_station_share = float(station_counts.max() / removals.sum()) if removals.any() else 0.0
    bootstrap = day_bootstrap(frame, evaluated, anchor, ensemble, config)
    seed_deltas = [float(f1_score(truth[evaluated], prediction[evaluated]) - reference_f1) for prediction in seed_predictions]
    seed_tp_removals = [int((evaluated & (anchor == 1) & (prediction == 0) & (truth == 1)).sum()) for prediction in seed_predictions]
    policy = config["decision_policy"]
    raw_points = delta_f1 * float(policy["score_points_per_f1"])
    calibrated = raw_points - float(policy["transport_penalty_points"])
    required_f1 = float(policy["minimum_raw_expected_point_delta_inclusive"]) / float(policy["score_points_per_f1"])
    gates = {
        "positive_removals": int(removals.sum()) > 0,
        "true_positive_removals_zero": int((removals & (truth == 1)).sum()) == 0,
        "q3_q4_each_nonnegative": min(item["delta_f1"] for item in by_fold.values()) >= float(policy["minimum_each_forward_fold_delta_f1"]),
        "raw_expected_points_at_least_0_331905690": raw_points >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_at_least_0_01": calibrated >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "block_bootstrap_lcb_meets_raw_transport_equivalent": bootstrap["ci90_low"] >= required_f1,
        "bootstrap_probability_at_least_0_8": bootstrap["probability_improved"] >= float(policy["bootstrap_probability_improved_minimum"]),
        "intervention_fraction_at_most_0_005": float(removals.sum() / evaluated.sum()) <= float(policy["maximum_intervention_fraction"]),
        "station_concentration_at_most_0_8": max_station_share <= float(policy["maximum_station_intervention_share"]),
        "all_seed_deltas_nonnegative": min(seed_deltas) >= 0.0,
        "all_seed_true_positive_removals_zero": max(seed_tp_removals) == 0,
    }
    return {
        "name": spec["name"],
        "family": spec["family"],
        "past_only": True,
        "fit_count": fit_count,
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": delta_f1,
        "raw_expected_points_delta": raw_points,
        "transport_penalty_points": float(policy["transport_penalty_points"]),
        "calibrated_conservative_expected_points_delta": calibrated,
        "removals": int(removals.sum()),
        "false_positive_removals": int((removals & (truth == 0)).sum()),
        "true_positive_removals": int((removals & (truth == 1)).sum()),
        "intervention_fraction": float(removals.sum() / evaluated.sum()),
        "station_counts": station_counts.to_dict(),
        "maximum_station_share": max_station_share,
        "by_fold": by_fold,
        "seed_delta_f1": seed_deltas,
        "seed_true_positive_removals": seed_tp_removals,
        "day_block_bootstrap": bootstrap,
        "prequential_receipts": receipts,
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }, fit_count


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    family = result["transport_family"]
    policy = result["decision_policy"]
    checks = {
        "candidate_count_three": len(result["candidates"]) == 3,
        "fit_count_exact_eighteen": result["fit_count"] == 18,
        "hard_router_family_registered": family["tier_id"] == "HARD_CONDITIONAL_ROUTER" and family["representation_changed"] and family["routing_discontinuous"],
        "calibration_hash_exact": family["calibration_sha256"] == result["hashes"]["root_calibration_sha256"],
        "penalty_exact": np.isclose(policy["transport_penalty_points"], 0.3219056897594759),
        "raw_gate_exact": np.isclose(policy["minimum_raw_expected_point_delta_inclusive"], 0.33190568975947593),
        "all_past_only": all(item["past_only"] for item in result["candidates"]),
        "only_strict_passes_in_pass_count": result["pass_count"] == sum(item["strict_internal_pass"] for item in result["candidates"]),
        "official_reads_zero": result["operations"]["official_covariate_reads"] == 0,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    config = load_contract()
    for path in (AUTHORITATIVE_PATH, ANCHOR_PATH, CALIBRATION_PATH):
        if not path.is_file():
            raise ContractError(f"missing frozen input: {path}")
    return {
        "status": "VALID",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "calibration_sha256": sha256_file(CALIBRATION_PATH),
        "candidate_count": len(config["models"]),
        "fit_budget": config["fit_budget"]["maximum"],
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("exactly-once artifact/report path already exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    started = time.perf_counter()
    write_json(ARTIFACT / "attempt_lock.json", {"experiment_id": EXPERIMENT_ID, "pid": os.getpid(), "config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__)), "fit_budget": 18, "performance_withheld_until_terminal": True})
    write_json(ARTIFACT / "progress.json", {"phase": "loading_historical_oof_only", "fit_count": 0, "performance_withheld_until_terminal": True}, exclusive=False)
    frame = load_frame(config)
    candidates = []
    fit_count = 0
    for spec in config["models"]:
        record, fits = evaluate(spec, frame, config)
        candidates.append(record)
        fit_count += fits
        write_json(ARTIFACT / "progress.json", {"phase": "historical_forward_validation", "completed_candidates": len(candidates), "total_candidates": 3, "fit_count": fit_count, "performance_withheld_until_terminal": True}, exclusive=False)
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v11.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_ONLY",
        "runtime_seconds": time.perf_counter() - started,
        "transport_family": config["transport_family"],
        "decision_policy": config["decision_policy"],
        "candidate_count": 3,
        "pass_count": sum(item["strict_internal_pass"] for item in candidates),
        "fit_count": fit_count,
        "candidates": candidates,
        "outputs": [],
        "operations": {"official_covariate_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0},
        "hashes": {"config_sha256": sha256_file(CONFIG_PATH), "runner_sha256": sha256_file(Path(__file__)), "root_calibration_sha256": sha256_file(CALIBRATION_PATH), "authoritative_results_sha256": sha256_file(AUTHORITATIVE_PATH), "anchor_sha256": sha256_file(ANCHOR_PATH)},
    }
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    write_json(ARTIFACT / "progress.json", {"phase": "terminal", "fit_count": fit_count, "pass_count": result["pass_count"], "performance_withheld_until_terminal": False}, exclusive=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.execute:
        parser.error("choose exactly one mode")
    try:
        payload = validate_only() if args.validate_only else execute()
        print(json.dumps(native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {"experiment_id": EXPERIMENT_ID, "status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "official_covariate_reads": 0, "hidden_truth_reads": 0, "uploads": 0}
        if ARTIFACT.exists() and not (ARTIFACT / "terminal_failure.json").exists():
            write_json(ARTIFACT / "terminal_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
