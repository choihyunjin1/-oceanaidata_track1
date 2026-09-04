"""Nested group-bootstrap lower-bound veto for P1 MSTCN added segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_deployable_type_veto_stability_20260829_v1 as prior  # noqa: E402

EXPERIMENT_ID = "p1_mstcn_bootstrap_lower_bound_veto_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
PRIOR_CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "p1_mstcn_deployable_type_veto_stability_20260829_v1.json"
)
OUTPUT_DIR = ROOT / "artifacts" / EXPERIMENT_ID


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_acceptance_frequency(
    *,
    training: prior.base.FoldBundle,
    evaluation_frame,
    fit_positions: np.ndarray,
    groups: np.ndarray,
    utility: np.ndarray,
    model_config: dict,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    targets = training.segments["beneficial"].to_numpy(np.int8)
    available_groups = np.unique(groups[fit_positions])
    rng = np.random.default_rng(seed)
    accepted = np.zeros((replicates, len(evaluation_frame)), dtype=bool)
    completed = 0
    attempts = 0
    while completed < replicates:
        attempts += 1
        require(attempts <= replicates * 30, "bootstrap class balance failure")
        sampled_groups = rng.choice(
            available_groups, size=len(available_groups), replace=True
        )
        sampled = np.concatenate(
            [
                fit_positions[groups[fit_positions] == group]
                for group in sampled_groups
            ]
        )
        if np.unique(targets[sampled]).size < 2:
            continue
        cls = prior.classifier(model_config)
        reg = prior.regressor(model_config)
        cls.fit(training.segments.iloc[sampled][prior.FEATURES], targets[sampled])
        reg.fit(training.segments.iloc[sampled][prior.FEATURES], utility[sampled])
        cls_accept = cls.predict_proba(evaluation_frame[prior.FEATURES])[:, 1] >= 0.5
        reg_accept = reg.predict(evaluation_frame[prior.FEATURES]) > 0.0
        accepted[completed] = cls_accept & reg_accept
        completed += 1
    return np.mean(accepted, axis=0), attempts


def nested_q3(training: prior.base.FoldBundle, config: dict, model_config: dict) -> dict:
    groups = prior.truth_event_groups(training)
    utility = prior.marginal_utility(training)
    frequencies = np.full(len(training.segments), np.nan, dtype=np.float64)
    total_attempts = 0
    replicates = int(config["nested_group_bootstrap_replicates"])
    for group_index, group in enumerate(np.unique(groups)):
        validation = groups == group
        fit_positions = np.flatnonzero(~validation)
        fold_frequency, attempts = bootstrap_acceptance_frequency(
            training=training,
            evaluation_frame=training.segments.loc[validation],
            fit_positions=fit_positions,
            groups=groups,
            utility=utility,
            model_config=model_config,
            replicates=replicates,
            seed=int(config["seed"]) + group_index * 1009,
        )
        frequencies[validation] = fold_frequency
        total_attempts += attempts
    require(np.isfinite(frequencies).all(), "nested frequencies incomplete")
    acceptance = frequencies >= float(config["acceptance_frequency"])
    candidate, additions = prior.candidate_from_acceptance(training, acceptance)
    incumbent = prior.base.metric(training.labels, training.incumbent)
    scored = prior.base.metric(training.labels, candidate)
    return {
        "groups": int(np.unique(groups).size),
        "segments": int(len(training.segments)),
        "replicates_per_group": replicates,
        "model_fits": int(2 * replicates * np.unique(groups).size),
        "resampling_attempts": total_attempts,
        "acceptance_frequency_quantiles": [
            float(value) for value in np.quantile(frequencies, [0.0, 0.25, 0.5, 0.75, 1.0])
        ],
        "consensus": additions,
        "incumbent": incumbent,
        "candidate": scored,
        "delta_f1_vs_incumbent": float(scored["f1"] - incumbent["f1"]),
    }


def full_q4(
    training: prior.base.FoldBundle,
    evaluation: prior.base.FoldBundle,
    config: dict,
    model_config: dict,
) -> dict:
    groups = prior.truth_event_groups(training)
    utility = prior.marginal_utility(training)
    fit_positions = np.arange(len(training.segments), dtype=int)
    frequencies, attempts = bootstrap_acceptance_frequency(
        training=training,
        evaluation_frame=evaluation.segments,
        fit_positions=fit_positions,
        groups=groups,
        utility=utility,
        model_config=model_config,
        replicates=int(config["full_group_bootstrap_replicates"]),
        seed=int(config["seed"]) + 99991,
    )
    acceptance = frequencies >= float(config["acceptance_frequency"])
    candidate, additions = prior.candidate_from_acceptance(evaluation, acceptance)
    incumbent = prior.base.metric(evaluation.labels, evaluation.incumbent)
    raw = prior.base.metric(evaluation.labels, evaluation.raw_candidate)
    scored = prior.base.metric(evaluation.labels, candidate)
    return {
        "segments": int(len(evaluation.segments)),
        "replicates": int(config["full_group_bootstrap_replicates"]),
        "model_fits": int(2 * int(config["full_group_bootstrap_replicates"])),
        "resampling_attempts": attempts,
        "acceptance_frequency_quantiles": [
            float(value) for value in np.quantile(frequencies, [0.0, 0.25, 0.5, 0.75, 1.0])
        ],
        "consensus": additions,
        "incumbent": incumbent,
        "raw_e150": raw,
        "candidate": scored,
        "delta_f1_vs_incumbent": float(scored["f1"] - incumbent["f1"]),
        "delta_f1_vs_raw_e150": float(scored["f1"] - raw["f1"]),
    }


def execute() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(config["experiment_id"] == EXPERIMENT_ID, "config id")
    require(config["official_test_reads_allowed"] == 0, "official read contract")
    require(config["submission_creation_allowed"] is False, "submission contract")
    require("incumbent_probability_mean" not in prior.FEATURES, "feature parity")
    model_config = json.loads(PRIOR_CONFIG_PATH.read_text(encoding="utf-8"))
    require(
        model_config["experiment_id"]
        == "p1_mstcn_deployable_type_veto_stability_20260829_v1",
        "prior model config id",
    )
    bundles = prior.base.load_bundles()
    q3 = bundles[str(config["training_fold"])]
    q4 = bundles[str(config["evaluation_fold"])]
    q3_result = nested_q3(q3, config, model_config)
    q4_result = full_q4(q3, q4, config, model_config)
    minimum_precision = float(
        config["diagnostic_gate"]["q3_accepted_row_precision_minimum"]
    )
    gate_checks = {
        "q3_grouped_nested_candidate_nonnegative_vs_incumbent": q3_result[
            "delta_f1_vs_incumbent"
        ]
        >= 0.0,
        "q3_accepted_row_precision": q3_result["consensus"]["accepted_row_precision"]
        >= minimum_precision,
        "q3_accepts_at_least_one_segment": q3_result["consensus"]["accepted_segments"]
        > 0,
        "q4_exact_incumbent_fallback": q4_result["candidate"] == q4_result["incumbent"],
        "q4_improves_raw_e150": q4_result["delta_f1_vs_raw_e150"] > 0.0,
        "deployment_feature_parity": "incumbent_probability_mean" not in prior.FEATURES,
        "no_official_test_or_submission_access": True,
    }
    passed = all(gate_checks.values())
    return {
        "schema_version": "p1.mstcn_bootstrap_lower_bound_veto.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "PASS_SHADOW_AUDIT_ELIGIBLE" if passed else "NO_GO_LOWER_BOUND_GATE",
        "passed_all_diagnostic_gates": passed,
        "q3_nested": q3_result,
        "q4_once": q4_result,
        "gate_checks": gate_checks,
        "input_hashes": {
            "config": sha256(CONFIG_PATH),
            "prior_config": sha256(PRIOR_CONFIG_PATH),
            "prior_runner": sha256(Path(prior.__file__)),
        },
        "operation_counters": {
            "official_test_rows_read": 0,
            "submission_files_created": 0,
            "uploads": 0,
            "model_fits": int(q3_result["model_fits"] + q4_result["model_fits"]),
        },
        "claim_limit": "Adaptive retrospective robustness evidence only; a pass permits label-free shadow audit, not submission promotion.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(CONFIG_PATH.exists(), "missing config")
    if not args.execute:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "READY_LOWER_BOUND_ONLY",
                    "official_test_rows_read": 0,
                    "submission_files_created": 0,
                },
                indent=2,
            )
        )
        return
    result = execute()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "result.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed_all_diagnostic_gates": result["passed_all_diagnostic_gates"],
                "q3": result["q3_nested"],
                "q4": result["q4_once"],
                "gate_checks": result["gate_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
