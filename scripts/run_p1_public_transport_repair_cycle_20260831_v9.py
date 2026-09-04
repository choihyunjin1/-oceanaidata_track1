"""Exactly-once P1 prequential CAPA benefit-selector evidence cycle.

The proposal bank was generated without targets and is treated as frozen.  Target
labels are used only in an earlier historical fold to estimate proposal benefit;
the learned selector is then applied unchanged to the next chronological fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v9"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CALIBRATION_PATH = ROOT / "reports/public_transport_calibration_20260831_v2/calibration.json"
AUTHORITATIVE_PATH = ROOT / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json"
CAPA_DIR = ROOT / "artifacts/p1_clean_state_capa_falsification_20260831_v1"
CAPA_PREFLIGHT = CAPA_DIR / "preflight.json"
FOLDS = ("2025_q2", "2025_q3", "2025_q4")


class ContractError(RuntimeError):
    """A frozen scientific or access contract was violated."""


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
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(native(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def duration_bin(rows: int) -> str:
    if rows <= 96:
        return "le96"
    if rows <= 384:
        return "192_to_384"
    return "ge519"


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    family = config["transport_family"]
    policy = config["decision_policy"]
    tier = calibration["tier_gates"][family["tier_id"]]
    checks = {
        "calibration_sha": family["calibration_sha256"] == sha256_file(CALIBRATION_PATH),
        "tier": family["tier_id"] == "HARD_CONDITIONAL_ROUTER",
        "representation_changed": family["representation_changed"] is True,
        "routing_discontinuous": family["routing_discontinuous"] is True,
        "penalty": np.isclose(policy["transport_penalty_points"], tier["transport_penalty_points"], atol=1e-15),
        "raw_floor": np.isclose(policy["minimum_raw_expected_point_delta_inclusive"], tier["minimum_raw_expected_points_delta"], atol=1e-15),
        "calibrated_floor": np.isclose(policy["minimum_calibrated_expected_point_delta_inclusive"], calibration["minimum_calibrated_expected_points_delta"], atol=1e-15),
    }
    if not all(checks.values()):
        raise ContractError(f"family-aware calibration mismatch: {checks}")
    if len(config["selectors"]) != 3 or config["fit_budget"]["maximum"] != 6:
        raise ContractError("selector count or fit budget changed")
    return config


def load_fold(fold: str) -> dict[str, Any]:
    preflight = json.loads(CAPA_PREFLIGHT.read_text(encoding="utf-8"))
    raw = pd.read_csv(
        preflight["train_csv"],
        usecols=["station", "year", "layer", "time", "label"],
    )
    arrays = np.load(CAPA_DIR / f"{fold}_blind_predictions.npz", allow_pickle=False)
    positions = arrays["row_position"].astype(np.int64, copy=False)
    frame = raw.iloc[positions].reset_index(drop=True)
    incumbent = arrays["incumbent"].astype(np.int8, copy=False)
    if len(frame) != len(incumbent) or not np.array_equal(positions, np.unique(positions)):
        raise ContractError(f"invalid frozen fold row positions: {fold}")
    proposals_raw = json.loads(
        (CAPA_DIR / f"{fold}_proposals.json").read_text(encoding="utf-8")
    )["proposals"]
    proposals: list[dict[str, Any]] = []
    for ordinal, item in enumerate(proposals_raw):
        if item["station"] != "I-ORS" or int(item["layer"]) != 1:
            continue
        start = int(item["start_row_in_frame"])
        stop = int(item["last_row_in_frame"]) + 1
        if start < 0 or stop > len(frame) or stop - start != int(item["window_rows"]):
            raise ContractError(f"proposal interval mismatch: {fold}/{ordinal}")
        local = frame.iloc[start:stop]
        if not (local["station"].eq("I-ORS").all() and local["layer"].eq(1).all()):
            raise ContractError(f"proposal scope mismatch: {fold}/{ordinal}")
        proposals.append(
            {
                "id": f"{fold}:{ordinal}",
                "fold": fold,
                "model": str(item["model"]),
                "month": int(pd.Timestamp(item["start_time"]).month),
                "duration_bin": duration_bin(int(item["window_rows"])),
                "window_rows": int(item["window_rows"]),
                "start": start,
                "stop": stop,
                "score": float(item["score"]),
                "score_per_row": float(item["score"]) / int(item["window_rows"]),
                "start_time": str(item["start_time"]),
            }
        )
    return {
        "frame": frame,
        "truth": frame["label"].to_numpy(np.int8),
        "incumbent": incumbent,
        "proposals": proposals,
        "row_positions": positions,
    }


def proposal_counts(fold_data: dict[str, Any], proposal: dict[str, Any]) -> tuple[int, int]:
    truth = fold_data["truth"]
    incumbent = fold_data["incumbent"]
    mask = np.zeros(len(truth), dtype=bool)
    mask[proposal["start"] : proposal["stop"]] = True
    additions = mask & (incumbent == 0)
    tp = int((additions & (truth == 1)).sum())
    fp = int((additions & (truth == 0)).sum())
    return tp, fp


def empirical_rate(tp: int, fp: int) -> float:
    return (tp + 1.0) / (tp + fp + 2.0)


def selector_state(
    kind: str,
    training: list[dict[str, Any]],
    folds: dict[str, dict[str, Any]],
    prior_strength: float,
) -> dict[str, Any]:
    enriched = []
    for proposal in training:
        tp, fp = proposal_counts(folds[proposal["fold"]], proposal)
        enriched.append({**proposal, "tp": tp, "fp": fp})
    global_tp = sum(item["tp"] for item in enriched)
    global_fp = sum(item["fp"] for item in enriched)
    global_mean = empirical_rate(global_tp, global_fp)
    state: dict[str, Any] = {
        "kind": kind,
        "global_tp": global_tp,
        "global_fp": global_fp,
        "global_mean": global_mean,
        "prior_strength": prior_strength,
        "groups": {},
    }
    if kind == "flat":
        state["global_alpha"] = 1.0 + global_tp
        state["global_beta"] = 1.0 + global_fp
        return state
    parent_counts: dict[str, list[int]] = {}
    child_counts: dict[str, list[int]] = {}
    for item in enriched:
        parent = f"{item['model']}|{item['duration_bin']}"
        child = f"{parent}|m{item['month']:02d}"
        parent_counts.setdefault(parent, [0, 0])
        parent_counts[parent][0] += item["tp"]
        parent_counts[parent][1] += item["fp"]
        child_counts.setdefault(child, [0, 0])
        child_counts[child][0] += item["tp"]
        child_counts[child][1] += item["fp"]
    for key, (tp, fp) in parent_counts.items():
        state["groups"][key] = {
            "alpha": prior_strength * global_mean + tp,
            "beta": prior_strength * (1.0 - global_mean) + fp,
            "tp": tp,
            "fp": fp,
        }
    if kind == "model_month_duration":
        for key, (tp, fp) in child_counts.items():
            parent = key.rsplit("|", 1)[0]
            parent_item = state["groups"][parent]
            parent_mean = parent_item["alpha"] / (parent_item["alpha"] + parent_item["beta"])
            state["groups"][key] = {
                "alpha": prior_strength * parent_mean + tp,
                "beta": prior_strength * (1.0 - parent_mean) + fp,
                "tp": tp,
                "fp": fp,
                "parent": parent,
            }
    return state


def proposal_lcb(proposal: dict[str, Any], state: dict[str, Any], quantile: float) -> tuple[float, str]:
    if state["kind"] == "flat":
        alpha, beta = state["global_alpha"], state["global_beta"]
        return float(beta_distribution.ppf(quantile, alpha, beta)), "global"
    parent = f"{proposal['model']}|{proposal['duration_bin']}"
    key = parent
    if state["kind"] == "model_month_duration":
        child = f"{parent}|m{proposal['month']:02d}"
        key = child if child in state["groups"] else parent
    if key not in state["groups"]:
        alpha = state["prior_strength"] * state["global_mean"]
        beta = state["prior_strength"] * (1.0 - state["global_mean"])
        return float(beta_distribution.ppf(quantile, alpha, beta)), "global_fallback"
    item = state["groups"][key]
    return float(beta_distribution.ppf(quantile, item["alpha"], item["beta"])), key


def apply_selector(
    fold_data: dict[str, Any],
    state: dict[str, Any],
    reference_threshold: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = fold_data["frame"]
    incumbent = fold_data["incumbent"]
    selected = np.zeros(len(frame), dtype=bool)
    day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d").to_numpy()
    day_rows = pd.Series(day).value_counts().to_dict()
    used: dict[str, int] = {}
    cap_fraction = float(config["scope"]["daily_full_surface_fraction_cap"])
    scored = []
    for proposal in fold_data["proposals"]:
        lcb, source = proposal_lcb(proposal, state, float(config["proposal_policy"]["posterior_precision_lcb_quantile"]))
        scored.append({**proposal, "precision_lcb90": lcb, "lookup": source})
    scored.sort(key=lambda item: (-item["precision_lcb90"], -item["score_per_row"], item["start_time"]))
    decisions = []
    for item in scored:
        interval = np.zeros(len(frame), dtype=bool)
        interval[item["start"] : item["stop"]] = True
        additions = interval & (incumbent == 0)
        proposal_days, proposal_counts_per_day = np.unique(day[additions], return_counts=True)
        reason = "ACCEPTED"
        if item["precision_lcb90"] <= reference_threshold:
            reason = "LCB_NOT_ABOVE_F1_HALF"
        elif np.any(selected & additions):
            reason = "OVERLAP_WITH_HIGHER_RANKED_PROPOSAL"
        else:
            for local_day, count in zip(proposal_days, proposal_counts_per_day, strict=True):
                cap = int(np.floor(cap_fraction * int(day_rows[local_day])))
                if used.get(str(local_day), 0) + int(count) > cap:
                    reason = "WHOLE_PROPOSAL_EXCEEDS_DAILY_CAP"
                    break
        if reason == "ACCEPTED":
            selected |= additions
            for local_day, count in zip(proposal_days, proposal_counts_per_day, strict=True):
                used[str(local_day)] = used.get(str(local_day), 0) + int(count)
        decisions.append(
            {
                "proposal_id": item["id"],
                "precision_lcb90": item["precision_lcb90"],
                "threshold": reference_threshold,
                "lookup": item["lookup"],
                "eligible_additions": int(additions.sum()),
                "decision": reason,
            }
        )
    prediction = incumbent.copy()
    prediction[selected] = 1
    return prediction, {
        "proposals": len(scored),
        "accepted_proposals": sum(item["decision"] == "ACCEPTED" for item in decisions),
        "additions": int(selected.sum()),
        "maximum_day_additions": max(used.values(), default=0),
        "decisions": decisions,
    }


def f1_counts(truth: np.ndarray, prediction: np.ndarray) -> tuple[int, int, int]:
    return (
        int(((truth == 1) & (prediction == 1)).sum()),
        int(((truth == 0) & (prediction == 1)).sum()),
        int(((truth == 1) & (prediction == 0)).sum()),
    )


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def day_bootstrap(
    evaluated: list[tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    blocks = []
    for frame, truth, incumbent, candidate in evaluated:
        day = pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
        local = pd.DataFrame({"day": day, "truth": truth, "anchor": incumbent, "candidate": candidate})
        for _, group in local.groupby("day", sort=True):
            blocks.append((*f1_counts(group.truth.to_numpy(), group.anchor.to_numpy()), *f1_counts(group.truth.to_numpy(), group.candidate.to_numpy())))
    counts = np.asarray(blocks, dtype=np.int64)
    replicates = int(config["validation"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    delta = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = counts[rng.integers(0, len(counts), size=len(counts))].sum(axis=0)
        delta[index] = f1_from_counts(*sampled[3:6]) - f1_from_counts(*sampled[0:3])
    return {
        "method": "KST-day paired block bootstrap",
        "blocks": int(len(blocks)),
        "replicates": replicates,
        "mean_delta_f1": float(delta.mean()),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta > 0.0)),
    }


def evaluate_selector(
    selector: dict[str, Any],
    folds: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    outer_predictions: dict[str, np.ndarray] = {}
    receipts: dict[str, Any] = {}
    fit_count = 0
    for outer in config["validation"]["outer_forward_tests"]:
        training = [proposal for fold in outer["train_folds"] for proposal in folds[fold]["proposals"]]
        train_truth = np.concatenate([folds[fold]["truth"] for fold in outer["train_folds"]])
        train_anchor = np.concatenate([folds[fold]["incumbent"] for fold in outer["train_folds"]])
        reference_f1 = float(f1_score(train_truth, train_anchor))
        state = selector_state(selector["kind"], training, folds, float(selector.get("prior_strength", 0.0)))
        fit_count += 1
        prediction, receipt = apply_selector(folds[outer["test_fold"]], state, reference_f1 / 2.0, config)
        outer_predictions[outer["test_fold"]] = prediction
        receipts[outer["test_fold"]] = {
            "train_folds": outer["train_folds"],
            "training_proposals": len(training),
            "train_anchor_f1": reference_f1,
            "precision_threshold_strict": reference_f1 / 2.0,
            "selector_state": state,
            "application": receipt,
        }
    evaluated = []
    by_fold = {}
    total_tp_add = 0
    total_add = 0
    for fold in FOLDS[1:]:
        data = folds[fold]
        candidate = outer_predictions[fold]
        truth = data["truth"]
        anchor = data["incumbent"]
        additions = (candidate == 1) & (anchor == 0)
        reference_f1 = float(f1_score(truth, anchor))
        candidate_f1 = float(f1_score(truth, candidate))
        tp_add = int((additions & (truth == 1)).sum())
        total_tp_add += tp_add
        total_add += int(additions.sum())
        by_fold[fold] = {
            "rows": len(truth),
            "reference_f1": reference_f1,
            "candidate_f1": candidate_f1,
            "delta_f1": candidate_f1 - reference_f1,
            "additions": int(additions.sum()),
            "true_positive_additions": tp_add,
            "false_positive_additions": int(additions.sum()) - tp_add,
            "anchor_removals": 0,
        }
        evaluated.append((data["frame"], truth, anchor, candidate))
    truth = np.concatenate([item[1] for item in evaluated])
    anchor = np.concatenate([item[2] for item in evaluated])
    candidate = np.concatenate([item[3] for item in evaluated])
    reference_f1 = float(f1_score(truth, anchor))
    candidate_f1 = float(f1_score(truth, candidate))
    delta_f1 = candidate_f1 - reference_f1
    raw_points = delta_f1 * float(config["decision_policy"]["score_points_per_f1"])
    calibrated = raw_points - float(config["decision_policy"]["transport_penalty_points"])
    bootstrap = day_bootstrap(evaluated, config)
    precision = total_tp_add / total_add if total_add else None
    policy = config["decision_policy"]
    gates = {
        "positive_additions": total_add > 0,
        "additions_precision_above_training_rule": all(
            all(
                decision["decision"] != "ACCEPTED" or decision["precision_lcb90"] > receipt["precision_threshold_strict"]
                for decision in receipt["application"]["decisions"]
            )
            for receipt in receipts.values()
        ),
        "anchor_removals_zero": True,
        "q3_q4_each_nonnegative": min(item["delta_f1"] for item in by_fold.values()) >= float(policy["minimum_each_forward_fold_delta_f1"]),
        "pooled_delta_strictly_positive": delta_f1 > float(policy["minimum_pooled_delta_f1_strict"]),
        "bootstrap_ci90_low_at_least_0_000579": bootstrap["ci90_low"] >= float(policy["bootstrap_ci90_low_minimum"]),
        "bootstrap_probability_at_least_0_8": bootstrap["probability_improved"] >= float(policy["bootstrap_probability_improved_minimum"]),
        "raw_expected_points_at_least_0_331905690": raw_points >= float(policy["minimum_raw_expected_point_delta_inclusive"]),
        "calibrated_expected_points_at_least_0_01": calibrated >= float(policy["minimum_calibrated_expected_point_delta_inclusive"]),
        "deterministic_stability": True,
    }
    abstain_reasons = sorted({decision["decision"] for receipt in receipts.values() for decision in receipt["application"]["decisions"] if decision["decision"] != "ACCEPTED"})
    return {
        "name": selector["name"],
        "selector_kind": selector["kind"],
        "past_only": True,
        "add_only": True,
        "fit_count": fit_count,
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": delta_f1,
        "raw_expected_points_delta": raw_points,
        "transport_penalty_points": float(policy["transport_penalty_points"]),
        "calibrated_conservative_expected_points_delta": calibrated,
        "additions": total_add,
        "true_positive_additions": total_tp_add,
        "false_positive_additions": total_add - total_tp_add,
        "additions_precision": precision,
        "anchor_removals": 0,
        "by_fold": by_fold,
        "prequential_receipts": receipts,
        "day_block_bootstrap": bootstrap,
        "abstention_reasons": abstain_reasons,
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    policy = result["decision_policy"]
    family = result["transport_family"]
    checks = {
        "three_candidates": len(result["candidates"]) == 3,
        "fit_count_exact_six": result["fit_count"] == 6,
        "hard_router_tier": family["tier_id"] == "HARD_CONDITIONAL_ROUTER",
        "representation_changed": family["representation_changed"] is True,
        "routing_discontinuous": family["routing_discontinuous"] is True,
        "calibration_sha_exact": family["calibration_sha256"] == result["hashes"]["root_calibration_sha256"],
        "penalty_exact": np.isclose(policy["transport_penalty_points"], 0.3219056897594759),
        "raw_gate_exact": np.isclose(policy["minimum_raw_expected_point_delta_inclusive"], 0.33190568975947593),
        "all_past_only_add_only": all(item["past_only"] and item["add_only"] for item in result["candidates"]),
        "anchor_removals_zero": all(item["anchor_removals"] == 0 for item in result["candidates"]),
        "only_passes_materialized": len(result["outputs"]) == result["pass_count"],
        "official_reads_zero_without_pass": result["pass_count"] > 0 or result["operations"]["official_covariate_reads"] == 0,
        "hidden_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
        "prior_tie_excluded": result["prior_tie_disposition"] == "EXCLUDED_FROM_PASS_COUNT",
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    config = load_contract()
    required = [CAPA_PREFLIGHT, AUTHORITATIVE_PATH, *(CAPA_DIR / f"{fold}_proposals.json" for fold in FOLDS), *(CAPA_DIR / f"{fold}_blind_predictions.npz" for fold in FOLDS)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ContractError(f"missing frozen inputs: {missing}")
    return {
        "status": "VALID",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "calibration_sha256": sha256_file(CALIBRATION_PATH),
        "candidate_count": len(config["selectors"]),
        "fit_budget": config["fit_budget"]["maximum"],
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("exactly-once artifact/report path already exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    started = time.perf_counter()
    write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "fit_budget": 6,
            "performance_withheld_until_terminal": True,
        },
    )
    write_json(ARTIFACT / "progress.json", {"phase": "loading_frozen_historical_proposals", "fit_count": 0, "performance_withheld_until_terminal": True}, exclusive=False)
    folds = {fold: load_fold(fold) for fold in FOLDS}
    records = []
    fit_count = 0
    for selector in config["selectors"]:
        record = evaluate_selector(selector, folds, config)
        records.append(record)
        fit_count += int(record["fit_count"])
        write_json(ARTIFACT / "progress.json", {"phase": "historical_prequential_scoring", "completed_candidates": len(records), "total_candidates": 3, "fit_count": fit_count, "performance_withheld_until_terminal": True}, exclusive=False)
    pass_count = sum(item["strict_internal_pass"] for item in records)
    outputs: list[dict[str, Any]] = []
    operations = {"official_covariate_reads": 0, "hidden_truth_reads": 0, "submission_csv_created": 0, "uploads": 0}
    if pass_count:
        raise ContractError("scientific PASS requires a separately preregistered official CAPA proposal generator; no official values were opened")
    result: dict[str, Any] = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v9.result",
        "experiment_id": EXPERIMENT_ID,
        "status": "NO_GO_BUDGET_SUPPORT" if all(item["additions"] == 0 for item in records) else "NO_GO_TRANSPORT_CALIBRATED",
        "runtime_seconds": time.perf_counter() - started,
        "transport_family": config["transport_family"],
        "decision_policy": config["decision_policy"],
        "candidate_count": len(records),
        "pass_count": pass_count,
        "prior_tie_disposition": "EXCLUDED_FROM_PASS_COUNT",
        "fit_count": fit_count,
        "candidates": records,
        "outputs": outputs,
        "operations": operations,
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "root_calibration_sha256": sha256_file(CALIBRATION_PATH),
            "authoritative_results_sha256": sha256_file(AUTHORITATIVE_PATH),
            "capa_preflight_sha256": sha256_file(CAPA_PREFLIGHT),
            "proposal_sha256": {fold: sha256_file(CAPA_DIR / f"{fold}_proposals.json") for fold in FOLDS},
            "blind_prediction_sha256": {fold: sha256_file(CAPA_DIR / f"{fold}_blind_predictions.npz") for fold in FOLDS},
        },
        "source_provenance": [str(AUTHORITATIVE_PATH), str(CALIBRATION_PATH), str(CAPA_PREFLIGHT), "historical frozen target-free CAPA proposal banks and blind incumbent predictions"],
    }
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    write_json(ARTIFACT / "progress.json", {"phase": "terminal", "fit_count": fit_count, "pass_count": pass_count, "performance_withheld_until_terminal": False}, exclusive=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.execute:
        parser.error("choose exactly one of --validate-only or --execute")
    try:
        payload = validate_only() if args.validate_only else execute()
        print(json.dumps(native(payload), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = {
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
            write_json(ARTIFACT / "terminal_failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
