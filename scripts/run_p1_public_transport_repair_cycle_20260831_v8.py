"""Exactly-once P1 layer-2 direct-rank transport repair."""

from __future__ import annotations

import argparse
import copy
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v5 as base  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v8"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260831_P1_PUBLIC_TRANSPORT_REPAIR_CYCLE_V8"
)
NAMES = {
    "pooled": "P1_1_L2_DRIFT_STATION_POOLED_RANK",
    "shrunk": "P1_2_L2_DRIFT_STATION_SHRUNK_RANK",
    "consensus": "P1_3_L2_DRIFT_RANK_CONSENSUS",
}
MODEL_FEATURES = base.MODEL_FEATURES + [
    "station_code",
    "layer_code",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
]


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


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    root_gate = json.loads(base.CALIBRATION_PATH.read_text(encoding="utf-8"))["gates"]["P1"]
    policy = config["decision_policy"]
    if not (
        np.isclose(
            policy["minimum_raw_expected_point_delta_inclusive"],
            root_gate["minimum_uncalibrated_expected_points_delta"],
            atol=1e-15,
        )
        and np.isclose(
            policy["bootstrap_ci90_low_minimum"],
            root_gate["full_transport_metric_improvement_equivalent"],
            atol=1e-15,
        )
    ):
        raise RuntimeError("root transport gate mismatch")
    return config


def source_fit(frame: pd.DataFrame) -> np.ndarray:
    return frame["e150_prediction"].eq(0).to_numpy()


def deployable(frame: pd.DataFrame) -> np.ndarray:
    return (
        source_fit(frame)
        & frame["layer"].eq(2).to_numpy()
        & frame["pmax"].ge(0.01).to_numpy()
    )


def fit_models(
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    config: dict[str, Any],
) -> list[Any]:
    x = frame[MODEL_FEATURES].to_numpy(np.float64)
    y = frame["label_base"].to_numpy(np.int8)
    source = fit_mask & source_fit(frame)
    spec = config["model"]
    models = []
    for seed in config["validation"]["seeds"]:
        sampled = base.day_subsample_mask(
            frame,
            source,
            float(config["validation"]["training_day_subsample_fraction"]),
            int(seed),
        )
        model = ExtraTreesClassifier(
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            max_features=float(spec["max_features"]),
            random_state=int(seed),
            n_jobs=int(spec["n_jobs_per_seed"]),
        )
        model.fit(
            x[sampled],
            y[sampled],
            sample_weight=compute_sample_weight("balanced", y[sampled]),
        )
        models.append(model)
    return models


def model_scores(models: list[Any], x: np.ndarray) -> np.ndarray:
    return np.vstack([model.predict_proba(x)[:, 1] for model in models])


def rank_score(
    frame: pd.DataFrame,
    mask: np.ndarray,
    score: np.ndarray,
    mode: str,
    shrinkage: float,
) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return result
    global_rank = pd.Series(score[indices]).rank(method="average", pct=True).to_numpy()
    if mode == "pooled":
        result[indices] = global_rank
        return result
    local = pd.DataFrame(
        {
            "station": frame.iloc[indices]["station"].to_numpy(),
            "score": score[indices],
        }
    )
    station_rank = local.groupby("station", observed=True)["score"].rank(
        method="average", pct=True
    )
    result[indices] = (1.0 - shrinkage) * global_rank + shrinkage * station_rank
    return result


def budgeted_additions(
    frame: pd.DataFrame,
    scope: np.ndarray,
    rank: np.ndarray,
    fraction: float,
    denominator_rows: int,
    daily_cap: int,
) -> np.ndarray:
    additions = np.zeros(len(frame), dtype=bool)
    budget = min(int(np.ceil(fraction * denominator_rows)), int(scope.sum()))
    if budget <= 0:
        return additions
    candidates = np.flatnonzero(scope)
    ordered = candidates[np.argsort(-rank[candidates], kind="stable")]
    timestamp = (
        pd.to_datetime(frame["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    used: dict[tuple[str, str], int] = {}
    selected = 0
    for index in ordered:
        key = (str(frame.iloc[index]["station"]), str(timestamp[index]))
        if used.get(key, 0) >= daily_cap:
            continue
        additions[index] = True
        used[key] = used.get(key, 0) + 1
        selected += 1
        if selected >= budget:
            break
    return additions


def select_fraction(
    mode: str,
    frame: pd.DataFrame,
    calibration_mask: np.ndarray,
    score: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    truth = frame["label_base"].to_numpy(np.int8)
    scope = calibration_mask & deployable(frame)
    rank = rank_score(
        frame,
        scope,
        score,
        mode,
        float(config["rank_selection"]["station_layer_shrinkage"]),
    )
    inner_config = copy.deepcopy(config)
    inner_config["validation"]["bootstrap_replicates"] = int(
        config["validation"]["inner_bootstrap_replicates"]
    )
    best = None
    for fraction in config["rank_selection"]["top_fractions"]:
        additions = budgeted_additions(
            frame,
            scope,
            rank,
            float(fraction),
            int(calibration_mask.sum()),
            int(config["rank_selection"]["maximum_additions_per_station_day"]),
        )
        candidate = anchor.copy()
        candidate[additions] = 1
        bootstrap = base.day_block_bootstrap(
            frame, calibration_mask, anchor, candidate, inner_config
        )
        reference = float(f1_score(truth[calibration_mask], anchor[calibration_mask]))
        candidate_f1 = float(f1_score(truth[calibration_mask], candidate[calibration_mask]))
        tp = int((additions & (truth == 1)).sum())
        fp = int(additions.sum()) - tp
        record = {
            "fraction": float(fraction),
            "calibration_rows_denominator": int(calibration_mask.sum()),
            "eligible_rows": int(scope.sum()),
            "additions": int(additions.sum()),
            "true_positive_additions": tp,
            "false_positive_additions": fp,
            "reference_f1": reference,
            "candidate_f1": candidate_f1,
            "delta_f1": candidate_f1 - reference,
            "day_block_bootstrap": bootstrap,
        }
        key = (
            bootstrap["ci90_low"],
            bootstrap["mean_delta_f1"],
            record["delta_f1"],
            -float(fraction),
        )
        if best is None or key > best[0]:
            best = (key, record)
    if best is None or best[1]["day_block_bootstrap"]["ci90_low"] < float(
        config["rank_selection"]["inner_f1_gain_lcb_minimum"]
    ):
        return {
            "status": "INNER_LCB_ABSTAIN",
            "fraction": 0.0,
            "eligible_rows": int(scope.sum()),
            "additions": 0,
            "inner_grid": config["rank_selection"]["top_fractions"],
        }
    return {
        "status": "INNER_LCB_SELECTED",
        **best[1],
        "inner_grid": config["rank_selection"]["top_fractions"],
    }


def evaluate(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    x = frame[MODEL_FEATURES].to_numpy(np.float64)
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    predictions = {mode: anchor.copy() for mode in ("pooled", "shrunk")}
    seed_predictions = {
        mode: [anchor.copy() for _ in config["validation"]["seeds"]]
        for mode in ("pooled", "shrunk")
    }
    receipts = {mode: {} for mode in ("pooled", "shrunk")}
    fits = 0
    for outer_index, outer in enumerate(config["validation"]["outer_forward_tests"], start=1):
        outer_train = frame["fold"].isin(outer["train_folds"]).to_numpy()
        outer_test = frame["fold"].eq(outer["test_fold"]).to_numpy()
        fit_mask, calibration_mask, split = base.chronological_inner_split(
            frame,
            outer_train,
            calibration_days=int(config["validation"]["inner_calibration_days"]),
            purge_hours=int(config["validation"]["purge_hours"]),
        )
        models = fit_models(frame, fit_mask, config)
        fits += len(models)
        scores = model_scores(models, x)
        for mode in ("pooled", "shrunk"):
            receipt = select_fraction(
                mode, frame, calibration_mask, scores.mean(axis=0), config
            )
            fraction = float(receipt["fraction"])
            scope = outer_test & deployable(frame)
            rank = rank_score(
                frame,
                scope,
                scores.mean(axis=0),
                mode,
                float(config["rank_selection"]["station_layer_shrinkage"]),
            )
            additions = budgeted_additions(
                frame,
                scope,
                rank,
                fraction,
                int(outer_test.sum()),
                int(config["rank_selection"]["maximum_additions_per_station_day"]),
            )
            predictions[mode][outer_test] = np.maximum(
                anchor[outer_test], additions[outer_test]
            )
            for seed_index in range(len(models)):
                seed_rank = rank_score(
                    frame,
                    scope,
                    scores[seed_index],
                    mode,
                    float(config["rank_selection"]["station_layer_shrinkage"]),
                )
                seed_additions = budgeted_additions(
                    frame,
                    scope,
                    seed_rank,
                    fraction,
                    int(outer_test.sum()),
                    int(config["rank_selection"]["maximum_additions_per_station_day"]),
                )
                seed_predictions[mode][seed_index][outer_test] = np.maximum(
                    anchor[outer_test], seed_additions[outer_test]
                )
            receipts[mode][outer["test_fold"]] = {
                **receipt,
                "inner_split": split,
                "outer_test_fold": outer["test_fold"],
                "outer_eligible_rows": int(scope.sum()),
            }
        write_json(
            ARTIFACT / "progress.json",
            {
                "phase": "historical_forward_validation",
                "completed_outer_tests": outer_index,
                "total_outer_tests": 2,
                "fit_count": fits,
                "performance_withheld_until_terminal": True,
            },
            exclusive=False,
        )
    evaluated = frame["fold"].isin(base.FOLD_ORDER[1:]).to_numpy()
    consensus = anchor.copy()
    consensus[evaluated] = np.maximum(
        anchor[evaluated],
        (
            (predictions["pooled"][evaluated] == 1)
            & (predictions["shrunk"][evaluated] == 1)
            & (anchor[evaluated] == 0)
        ),
    )
    consensus_seeds = [anchor.copy() for _ in config["validation"]["seeds"]]
    for index in range(len(consensus_seeds)):
        consensus_seeds[index][evaluated] = np.maximum(
            anchor[evaluated],
            (
                (seed_predictions["pooled"][index][evaluated] == 1)
                & (seed_predictions["shrunk"][index][evaluated] == 1)
                & (anchor[evaluated] == 0)
            ),
        )
    records = [
        base.score_record(
            NAMES[mode],
            frame,
            predictions[mode],
            seed_predictions[mode],
            receipts[mode],
            config,
        )
        for mode in ("pooled", "shrunk")
    ]
    records.append(
        base.score_record(
            NAMES["consensus"],
            frame,
            consensus,
            consensus_seeds,
            {"component_receipts": receipts},
            config,
        )
    )
    for record in records:
        if "component_receipts" in record["thresholds"]:
            inner = [
                receipt
                for family in record["thresholds"]["component_receipts"].values()
                for receipt in family.values()
            ]
        else:
            inner = list(record["thresholds"].values())
        record["gates"]["all_inner_fractions_selected_without_outer_labels"] = all(
            item["status"] == "INNER_LCB_SELECTED" for item in inner
        )
        record["gates"]["all_forward_blocks_nonnegative"] = all(
            item["delta_f1"] >= 0.0 for item in record["by_fold"].values()
        )
        record["gates"]["pooled_bootstrap_lcb_at_least_transport_equivalent"] = (
            record["day_block_bootstrap"]["ci90_low"]
            >= float(config["decision_policy"]["bootstrap_ci90_low_minimum"])
        )
        record["strict_internal_pass"] = bool(all(record["gates"].values()))
    return native(records), fits


def final_models_and_receipts(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[list[Any], dict[str, Any], int]:
    all_history = np.ones(len(frame), dtype=bool)
    fit_mask, calibration_mask, split = base.chronological_inner_split(
        frame,
        all_history,
        calibration_days=int(config["validation"]["inner_calibration_days"]),
        purge_hours=int(config["validation"]["purge_hours"]),
    )
    models = fit_models(frame, fit_mask, config)
    x = frame[MODEL_FEATURES].to_numpy(np.float64)
    score = model_scores(models, x).mean(axis=0)
    receipts = {
        mode: {**select_fraction(mode, frame, calibration_mask, score, config), "inner_split": split}
        for mode in ("pooled", "shrunk")
    }
    return models, receipts, len(models)


def materialize(
    frame: pd.DataFrame,
    actual_columns: list[str],
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    passes = [item for item in records if item["strict_internal_pass"]]
    if not passes:
        return [], {"official_covariate_reads": 0, "paths": []}, 0
    models, receipts, deployment_fits = final_models_and_receipts(frame, config)
    raw_test, official = base.prior_p1.official_frame(actual_columns)
    x = official[MODEL_FEATURES].to_numpy(np.float64)
    score = model_scores(models, x).mean(axis=0)
    scope = deployable(official)
    additions = {}
    for mode in ("pooled", "shrunk"):
        rank = rank_score(
            official,
            scope,
            score,
            mode,
            float(config["rank_selection"]["station_layer_shrinkage"]),
        )
        additions[mode] = budgeted_additions(
            official,
            scope,
            rank,
            float(receipts[mode]["fraction"]),
            len(official),
            int(config["rank_selection"]["maximum_additions_per_station_day"]),
        )
    candidate_additions = {
        NAMES["pooled"]: additions["pooled"],
        NAMES["shrunk"]: additions["shrunk"],
        NAMES["consensus"]: additions["pooled"] & additions["shrunk"],
    }
    anchor = official["e150_prediction"].to_numpy(np.int8)
    outputs = []
    for record in passes:
        added = candidate_additions[record["name"]]
        label = np.maximum(anchor, added).astype(np.int8)
        submission = raw_test[base.P1_KEYS].copy()
        submission["label"] = label
        if len(submission) != 169_011 or submission.duplicated(base.P1_KEYS).any():
            raise RuntimeError("official key contract failed")
        path = DELIVERY / record["name"] / "P1_submission.csv"
        path.parent.mkdir(parents=True, exist_ok=False)
        submission.to_csv(path, index=False, lineterminator="\n")
        outputs.append(
            {
                "name": record["name"],
                "path": str(path),
                "rows": 169_011,
                "sha256": sha256_file(path),
                "positive_rows": int(label.sum()),
                "additions_vs_champion": int((added & (anchor == 0)).sum()),
                "anchor_removals": 0,
                "final_inner_receipts": receipts,
                "upload_performed": False,
            }
        )
    return (
        outputs,
        {
            "official_covariate_reads": 1,
            "paths": [str(base.prior_p1.P1_DATA / "test.csv"), str(base.prior_p1.P1_CHAMPION)],
        },
        deployment_fits + 2,
    )


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "candidate_count": len(result["candidates"]) == 3,
        "fit_budget": result["fit_count"] <= 11,
        "all_anchor_removals_zero": all(item["anchor_removals"] == 0 for item in result["candidates"]),
        "only_passes_materialized": {item["name"] for item in result["outputs"]}
        == {item["name"] for item in result["candidates"] if item["strict_internal_pass"]},
        "all_other_rows_no_op": result["transport_scope"]["all_other_rows_exact_no_op"],
        "official_after_pass_only": bool(result["outputs"])
        == bool(result["operations"]["official_covariate_reads"]),
        "hidden_zero": result["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    config = load_contract()
    return {
        "status": "VALID",
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "candidate_count": len(config["candidates"]),
        "total_fit_budget": 11,
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists() or DELIVERY.exists():
        raise FileExistsError("exactly-once v8 path exists")
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    started = time.perf_counter()
    write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "pid": os.getpid(),
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    write_json(
        ARTIFACT / "progress.json",
        {"phase": "loading_historical_oof_only", "fit_count": 0},
        exclusive=False,
    )
    historical, _ = base.prior_cycle.p1_frame()
    historical, actual_columns = base.prior_p1.add_causal_features(historical)
    records, historical_fits = evaluate(historical, config)
    outputs, official_access, deployment_fits = materialize(
        historical, actual_columns, records, config
    )
    result = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v8",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "decision_policy": config["decision_policy"],
        "validation_contract": config["validation"],
        "rank_selection": config["rank_selection"],
        "transport_scope": config["transport_scope"],
        "candidates": records,
        "pass_count": sum(item["strict_internal_pass"] for item in records),
        "outputs": outputs,
        "historical_fit_count": historical_fits,
        "deployment_fit_count": deployment_fits,
        "fit_count": historical_fits + deployment_fits,
        "operations": {**official_access, "hidden_truth_reads": 0, "uploads": 0},
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "v7_result_sha256": sha256_file(
                ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v7/result.json"
            ),
        },
    }
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    if outputs:
        write_json(DELIVERY / "SET_MANIFEST.json", result)
    write_json(
        ARTIFACT / "progress.json",
        {
            "phase": "terminal",
            "fit_count": result["fit_count"],
            "pass_count": result["pass_count"],
            "outputs": len(outputs),
        },
        exclusive=False,
    )
    return native(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_only(), ensure_ascii=False, indent=2))
        return 0
    try:
        result = execute()
    except Exception as exc:
        if ARTIFACT.exists():
            write_json(
                ARTIFACT / "terminal_failure.json",
                {
                    "status": "TERMINAL_TECHNICAL_FAILURE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "automatic_restart_allowed": False,
                    "hidden_truth_reads": 0,
                    "uploads": 0,
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
