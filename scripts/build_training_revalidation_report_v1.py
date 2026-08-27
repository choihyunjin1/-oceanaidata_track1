"""Build the append-only training/revalidation registry and technical report.

The builder consumes SHA-pinned aggregate receipts for P1/P2/P3 and performs
one narrowly bounded P3 candidate-vs-current aggregate recomputation.  It does
not retain or emit prediction rows.  P2 and P3 remain corrected research
candidates; this builder cannot mutate or auto-promote the preregistered
official candidate pools.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
REPORT_TITLE = "교정 학습·재검증 레지스트리 — 연구 근거와 공식 후보를 분리한다"
REPORT_ID = "training-revalidation-2026-08-22-r1"
DEFAULT_REGISTRY = Path("artifacts/training_revalidation_20260822/registry.json")
DEFAULT_ARTIFACT = Path("reports/generated/training_revalidation_2026-08-22_r1/artifact.json")

EXPECTED_SHA256 = {
    "p1_pool_manifest": ("c8fb911c60e593caefbe43ed1192eacc3b372d0a2ba69ba7b02e7e57dbac903e"),
    "p1_causal_receipt": ("987f16076ff9728fdd8ebaca973bec68cfb26915ffa25cabc1dd9160fa280aa3"),
    "p1_duration_failure": ("090896b5bd1dc323df04cc7367f483d27643d1826be63b1561a8b0202a2d771d"),
    "p2_manifest": ("fbc9dbbfd5fdbe644faa94a42c498d82b20fe2e4b0a8577c4c8a113d0e6b76f1"),
    "p2_metrics": ("f76f0fb5dcbb6084dcd414578826d89f97aa545569c4510091fb1e6819e38eef"),
    "p2_completion": ("415160d4c991ebb878ca466509ab44eb3ba90c104ad0cc4fcc6338489da9b4c6"),
    "p3_manifest": ("ea6d6c7eda174e43493c7db3fb716d97c4fc74f1db2c3a4ab839bd04acb3cf9f"),
    "p3_metrics": ("2c797e6169b7af27d343edb31fae5acfd4ce704149c63b732480fe33692c22e6"),
    "cross_problem_policy": ("9529aa4aad806799dd3ed410e55bd4d3563d857358ae4e7c99f59789db4a27e7"),
}

RELATIVE_PATHS = {
    "p1_pool_manifest": Path("artifacts/p1_candidate_pool_20260822/pool_manifest.json"),
    "p1_causal_receipt": Path(
        "artifacts/p1_candidate_pool_20260822/P1_LIGHTGBM_CAUSAL_V1/candidate_receipt.json"
    ),
    "p1_duration_failure": Path(
        "artifacts/p1_candidate_pool_20260822/sequence_duration_slot_failure_receipt.json"
    ),
    "p2_manifest": Path("artifacts/p2_corrected_repeated_forward_v2/manifest.json"),
    "p2_metrics": Path("artifacts/p2_corrected_repeated_forward_v2/metrics.json"),
    "p2_completion": Path(
        "artifacts/p2_corrected_repeated_forward_v2/resume_completion_status.json"
    ),
    "p3_manifest": Path("artifacts/p3_corrected_repeated_forward_catboost_v2/manifest.json"),
    "p3_metrics": Path("artifacts/p3_corrected_repeated_forward_catboost_v2/metrics.json"),
    "cross_problem_policy": Path(
        "artifacts/validation_system_audit_20260822/cross_problem_policy.json"
    ),
}

P3_COMPARISON_INPUTS = {
    "candidate": Path(
        "artifacts/p3_corrected_repeated_forward_catboost_v2/candidate/submission.csv"
    ),
    "current": Path("output/2026-08-20/ready/P3_submission.csv"),
}
P3_COMPARISON_SHA256 = {
    "candidate": "24a360dd85978155b883378459f6d4d46a6b847569f1c3b6636a728c96e5ba11",
    "current": "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
}

FROZEN_BASELINE_SHA256 = {
    "P1": "28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3",
    "P2": "1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf",
    "P3": "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
}

P2_QA_ATTESTATION = {
    "medium": "message_only",
    "receipt_created": False,
    "verdict": "PASS",
    "p0_findings": 0,
    "p1_findings": 0,
    "scope": "independent post-run integrity review",
}
P3_QA_ATTESTATION = {
    "medium": "message_only",
    "receipt_created": False,
    "verdict": "POST_RUN_INTEGRITY_GO",
    "p0_findings": 0,
    "p1_findings": 0,
    "mismatches": 0,
    "scope": "artifact, metric, key, model, input, split, bootstrap, and access checks",
    "nonblocking_caveat": (
        "Package versions and six transitive module SHAs are absent from the run manifest."
    ),
}


class TrainingRevalidationError(RuntimeError):
    """Raised when a source, decision, or report contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingRevalidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"aggregate JSON must be an object: {path}")
    return payload


def _validate_path_contract() -> None:
    _require(set(RELATIVE_PATHS) == set(EXPECTED_SHA256), "pin/path set drifted")
    for name, relative in RELATIVE_PATHS.items():
        _require(not relative.is_absolute(), f"absolute evidence path forbidden: {name}")
        _require(relative.suffix.lower() == ".json", f"non-JSON receipt forbidden: {name}")
        _require(".." not in relative.parts, f"parent traversal forbidden: {name}")
    for name, relative in P3_COMPARISON_INPUTS.items():
        _require(not relative.is_absolute(), f"absolute comparison path forbidden: {name}")
        _require(".." not in relative.parts, f"parent traversal forbidden: {name}")


def collect_evidence(root: Path) -> dict[str, Any]:
    """Read only the pinned aggregate receipts and fail closed on drift."""

    _validate_path_contract()
    evidence: dict[str, Any] = {}
    for name, relative in RELATIVE_PATHS.items():
        expected = EXPECTED_SHA256[name]
        _require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"bad pin: {name}")
        path = root / relative
        _require(path.is_file(), f"missing aggregate evidence: {relative.as_posix()}")
        actual = _sha256(path)
        _require(actual == expected, f"SHA mismatch for {name}: {actual} != {expected}")
        evidence[name] = _read_json(path)
    evidence["hashes"] = dict(EXPECTED_SHA256)
    return evidence


def recompute_p3_candidate_distance(root: Path) -> dict[str, Any]:
    """Recompute aggregate distance without retaining or emitting prediction rows.

    The exact nonzero count is intentionally parser-qualified.  Pandas 3.0.1's
    default float64 CSV parser reproduces the independent QA count of 748,
    whereas decimal-text inequality is 825.  The tolerance-aware count is also
    retained so a parser-specific exact count is not mistaken for material
    prediction movement.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise TrainingRevalidationError(
            "pandas is required for the parser-qualified P3 aggregate recomputation"
        ) from exc

    _require(pd.__version__ == "3.0.1", f"P3 comparison parser drifted: pandas {pd.__version__}")
    paths = {name: root / relative for name, relative in P3_COMPARISON_INPUTS.items()}
    for name, path in paths.items():
        _require(path.is_file(), f"missing P3 comparison input: {P3_COMPARISON_INPUTS[name]}")
        actual = _sha256(path)
        expected = P3_COMPARISON_SHA256[name]
        _require(actual == expected, f"P3 comparison SHA mismatch for {name}: {actual}")

    columns = ["case_id", "station", "lead_h", "hs_pred"]
    candidate = pd.read_csv(paths["candidate"], usecols=columns)
    current = pd.read_csv(paths["current"], usecols=columns)
    _require(list(candidate.columns) == columns, "P3 candidate schema/order drifted")
    _require(list(current.columns) == columns, "P3 current schema/order drifted")
    _require(len(candidate) == len(current) == 1200, "P3 comparison row count drifted")
    keys = columns[:3]
    key_mismatch = int((candidate[keys] != current[keys]).any(axis=1).sum())
    _require(key_mismatch == 0, "P3 candidate/current key order mismatch")
    candidate_values = candidate["hs_pred"].to_numpy(dtype=float)
    current_values = current["hs_pred"].to_numpy(dtype=float)
    delta = candidate_values - current_values
    _require(bool(pd.notna(delta).all()), "P3 comparison contains non-finite values")
    exact_nonzero = int((delta != 0.0).sum())
    material_nonzero = int((abs(delta) > 1e-12).sum())
    rmse = math.sqrt(float((delta * delta).mean()))
    max_abs = float(abs(delta).max())

    decimal_text_unequal = 0
    with (
        paths["candidate"].open(encoding="utf-8", newline="") as candidate_handle,
        paths["current"].open(encoding="utf-8", newline="") as current_handle,
    ):
        candidate_reader = csv.DictReader(candidate_handle)
        current_reader = csv.DictReader(current_handle)
        for candidate_row, current_row in zip(candidate_reader, current_reader, strict=True):
            decimal_text_unequal += candidate_row["hs_pred"] != current_row["hs_pred"]

    _require(exact_nonzero == 748, f"P3 pandas exact-change count drifted: {exact_nonzero}")
    _require(material_nonzero == 660, f"P3 material-change count drifted: {material_nonzero}")
    _require(
        decimal_text_unequal == 825,
        f"P3 decimal-text inequality count drifted: {decimal_text_unequal}",
    )
    _require(
        math.isclose(rmse, 0.001052575987759441, rel_tol=0.0, abs_tol=1e-15),
        f"P3 candidate/current RMSE drifted: {rmse}",
    )
    _require(
        math.isclose(max_abs, 0.00716521632450795, rel_tol=0.0, abs_tol=1e-15),
        f"P3 candidate/current max delta drifted: {max_abs}",
    )
    return {
        "method": "builder_independent_recompute",
        "parser": f"pandas-{pd.__version__} default CSV parser to float64",
        "candidate_sha256": P3_COMPARISON_SHA256["candidate"],
        "current_sha256": P3_COMPARISON_SHA256["current"],
        "rows": 1200,
        "key_order_mismatches": key_mismatch,
        "prediction_rmse_distance_m": rmse,
        "exact_parsed_float_nonzero_rows": exact_nonzero,
        "absolute_delta_gt_1e_12_rows": material_nonzero,
        "decimal_text_unequal_rows": decimal_text_unequal,
        "maximum_absolute_delta_m": max_abs,
        "raw_rows_retained_or_emitted": 0,
    }


def _validate_p1(evidence: dict[str, Any]) -> dict[str, Any]:
    pool = evidence["p1_pool_manifest"]
    receipt = evidence["p1_causal_receipt"]
    failure = evidence["p1_duration_failure"]
    _require(pool["problem"] == receipt["problem"] == failure["problem"] == "P1", "P1 id drifted")
    _require(
        pool["status"] == "TWO_VALID_CANDIDATES_SEALED__SEQUENCE_DURATION_SLOT_FAILED_CLOSED",
        "P1 pool status drifted",
    )
    candidates = pool["candidates"]
    _require(len(candidates) == 2, "P1 sealed candidate count drifted")
    current, causal = candidates
    _require(current["sha256"] == FROZEN_BASELINE_SHA256["P1"], "P1 baseline drifted")
    _require(
        current["unchanged"] is True and current["eligible"] is True, "P1 current state drifted"
    )
    _require(
        causal["candidate_order"] == 2 and causal["eligible"] is True,
        "P1 causal eligibility drifted",
    )
    _require(receipt["candidate_id"] == causal["id"], "P1 causal identity drifted")
    _require(receipt["candidate"]["sha256"] == causal["sha256"], "P1 candidate SHA drifted")
    _require(receipt["eligibility"]["eligible"] is True, "P1 receipt eligibility drifted")
    _require(receipt["created_before_first_official_score"] is True, "P1 timing drifted")
    _require(receipt["strict_validation"]["passed"] is True, "P1 validation drifted")
    _require(
        receipt["reproduction"]["sha_identical_to_candidate"] is True, "P1 reproduction drifted"
    )
    _require(
        failure["status"] == "FAILED_CLOSED_NO_EXECUTABLE_FROZEN_DEPLOYMENT"
        and failure["eligible"] is False,
        "P1 duration failure drifted",
    )
    _require(pool["non_actions"]["submission_uploads"] == 0, "P1 upload count drifted")
    _require(receipt["operation_counters"]["submission_uploads"] == 0, "P1 receipt upload drifted")
    _require(failure["operation_counters"]["uploads"] == 0, "P1 failure upload drifted")
    return {
        "current_sha256": current["sha256"],
        "causal_candidate_sha256": causal["sha256"],
        "causal_candidate_rows": int(causal["rows"]),
        "causal_status": "ELIGIBLE_PRE_FIRST_SCORE_CANDIDATE_2",
        "duration_status": "INELIGIBLE_FAILED_CLOSED",
        "upload_count": 0,
    }


def _validate_p2(evidence: dict[str, Any]) -> dict[str, Any]:
    manifest = evidence["p2_manifest"]
    metrics = evidence["p2_metrics"]
    completion = evidence["p2_completion"]
    policy = evidence["cross_problem_policy"]["problem_policies"]["P2"]
    _require(manifest["experiment_id"] == metrics["experiment_id"], "P2 experiment drifted")
    _require(completion["experiment_id"] == manifest["experiment_id"], "P2 completion drifted")
    _require(manifest["research_only"] is True, "P2 research-only flag drifted")
    _require(manifest["upload_allowed"] is False, "P2 upload authorization drifted")
    _require(manifest["adaptive_research"] is True, "P2 adaptive flag drifted")
    _require(manifest["fresh_holdout_claimed"] is False, "P2 holdout claim drifted")
    _require(manifest["hyperparameter_searches"] == 0, "P2 search count drifted")
    _require(manifest["frozen_submission_snapshot_unchanged"] is True, "P2 frozen baseline drifted")
    _require(completion["same_original_attempt"] == 1, "P2 attempt identity drifted")
    _require(completion["new_generation_or_attempt"] is False, "P2 generation drifted")
    _require(completion["status"] == "complete", "P2 completion status drifted")
    _require(completion["upload_performed"] is False, "P2 upload count drifted")
    candidate_sha = manifest["artifacts"]["candidate"]["sha256"]
    _require(candidate_sha == completion["candidate"]["sha256"], "P2 candidate pin drifted")
    _require(
        manifest["artifacts"]["metrics"]["sha256"] == EXPECTED_SHA256["p2_metrics"],
        "P2 metrics pin drifted",
    )
    pool_hashes = {item["sha256"] for item in policy["candidate_pool"]["candidates"]}
    _require(len(pool_hashes) == 3 and candidate_sha not in pool_hashes, "P2 pool boundary drifted")
    _require(policy["candidate_pool"]["available_candidate_count"] == 3, "P2 pool count drifted")
    _require(
        policy["candidate_pool"]["candidates"][0]["sha256"] == FROZEN_BASELINE_SHA256["P2"],
        "P2 baseline pin drifted",
    )

    inner = metrics["inner_diagnostic"]
    inner_baseline = float(inner["baseline"]["fold_equal_official_layer_weighted_rmse_c"])
    inner_candidate = float(inner["candidate"]["fold_equal_official_layer_weighted_rmse_c"])
    inner_delta = inner_candidate - inner_baseline
    _require(inner_delta > 0, "P2 inner diagnostic must remain adverse")

    outer = metrics["outer_repeated_forward"]
    outer_baseline = outer["baseline"]
    outer_candidate = outer["candidate"]
    fold_labels = [
        ("2024 Sep–Oct", "outer_2024_sep_oct"),
        ("2025 May–Jun", "outer_2025_may_jun"),
        ("2025 Jul–Aug", "outer_2025_jul_aug"),
    ]
    folds = []
    for sequence, (label, key) in enumerate(fold_labels, start=2):
        baseline_rmse = float(outer_baseline["by_fold"][key]["official_layer_weighted_rmse_c"])
        candidate_rmse = float(outer_candidate["by_fold"][key]["official_layer_weighted_rmse_c"])
        delta = candidate_rmse - baseline_rmse
        folds.append(
            {
                "sequence": sequence,
                "scope": label,
                "evidence_kind": "outer fold",
                "baseline_rmse_c": baseline_rmse,
                "candidate_rmse_c": candidate_rmse,
                "delta_rmse_c": delta,
                "signed_delta_label": f"{delta:+.6f}°C",
                "direction": "개선 (Δ<0)" if delta < 0 else "악화 (Δ>0)",
            }
        )
    _require(sum(row["delta_rmse_c"] > 0 for row in folds) == 1, "P2 adverse-fold count drifted")

    bootstrap = outer["candidate_vs_baseline_bootstrap"]
    overall_delta = float(bootstrap["delta_rmse_c"])
    overall = {
        "sequence": 1,
        "scope": "Fold-equal aggregate",
        "evidence_kind": "aggregate",
        "baseline_rmse_c": float(bootstrap["reference_rmse_c"]),
        "candidate_rmse_c": float(bootstrap["candidate_rmse_c"]),
        "delta_rmse_c": overall_delta,
        "signed_delta_label": f"{overall_delta:+.6f}°C",
        "direction": "개선 (Δ<0)",
    }
    ci90 = [float(value) for value in bootstrap["delta_interval"]]
    _require(overall_delta < 0 and ci90[1] < 0, "P2 outer aggregate direction drifted")
    return {
        "baseline_sha256": FROZEN_BASELINE_SHA256["P2"],
        "candidate_sha256": candidate_sha,
        "candidate_rows": int(manifest["artifacts"]["candidate"]["validation"]["rows"]),
        "status": "CORRECTED_RESEARCH_CANDIDATE_NOT_AUTO_PROMOTED",
        "pool_candidate_count_before_after": [3, 3],
        "inner_delta_rmse_c": inner_delta,
        "outer_overall_delta_rmse_c": overall_delta,
        "outer_ci90_c": ci90,
        "outer_improved_fold_count": 2,
        "outer_adverse_fold_count": 1,
        "chart_rows": [overall, *folds],
        "upload_count": 0,
    }


def _validate_p3(evidence: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    manifest = evidence["p3_manifest"]
    metrics = evidence["p3_metrics"]
    policy = evidence["cross_problem_policy"]["problem_policies"]["P3"]
    _require(manifest["experiment_id"] == metrics["experiment_id"], "P3 experiment drifted")
    expected_status = "CORRECTED_RESEARCH_EVIDENCE_GATE_PASS_CANDIDATE_CREATED_NOT_UPLOADED"
    _require(manifest["status"] == metrics["status"] == expected_status, "P3 status drifted")
    _require(manifest["append_only_generation"] is True, "P3 append-only flag drifted")
    _require(manifest["source_cache_current_frozen_unchanged"] is True, "P3 inputs drifted")
    _require(manifest["gate_passed"] is True, "P3 gate drifted")
    _require(manifest["candidate_created"] is True, "P3 candidate state drifted")
    _require(manifest["candidate_uploaded"] is False, "P3 upload state drifted")
    _require(metrics["gate"]["passed"] is True, "P3 metrics gate drifted")
    _require(metrics["invariants"]["hyperparameter_search_run"] is False, "P3 search drifted")
    _require(metrics["candidate_validation"]["key_order_exact"] is True, "P3 key order drifted")
    _require(metrics["candidate_validation"]["finite"] is True, "P3 finite check drifted")
    _require(metrics["candidate_validation"]["same_case_only"] is True, "P3 case scope drifted")
    _require(metrics["access_counters"]["upload_attempts"] == 0, "P3 upload count drifted")
    candidate_sha = metrics["candidate_validation"]["sha256"]["candidate/submission.csv"]
    _require(candidate_sha == P3_COMPARISON_SHA256["candidate"], "P3 candidate SHA drifted")
    before = manifest["input_sha256_before"]["current/ready_submission.csv"]
    after = manifest["input_sha256_after"]["current/ready_submission.csv"]
    _require(before == after == FROZEN_BASELINE_SHA256["P3"], "P3 current baseline drifted")
    pool_hashes = {item["sha256"] for item in policy["candidate_pool"]["candidates"]}
    _require(len(pool_hashes) == 3 and candidate_sha not in pool_hashes, "P3 pool boundary drifted")

    gate = metrics["gate"]
    overall_delta = float(gate["delta_candidate_minus_persistence_m"])
    ci90 = [
        float(value)
        for value in metrics["paired_case_bootstrap"]["delta_candidate_minus_persistence_ci90_m"]
    ]
    _require(overall_delta < 0 and ci90[1] < 0, "P3 overall direction drifted")
    fold_labels = [
        ("2024 H2 storm", "2024_h2_storm"),
        ("Winter transition", "winter_transition"),
        ("2025 H1", "2025_h1"),
    ]
    folds = []
    metric_folds = metrics["metrics"]["folds"]
    for sequence, (label, key) in enumerate(fold_labels, start=2):
        candidate_rmse = float(metric_folds[key]["final"]["rmse"])
        persistence_rmse = float(metric_folds[key]["persistence"]["rmse"])
        delta = float(metric_folds[key]["delta_final_minus_persistence_m"])
        _require(
            math.isclose(delta, candidate_rmse - persistence_rmse, abs_tol=1e-15),
            f"P3 fold delta drifted: {key}",
        )
        folds.append(
            {
                "sequence": sequence,
                "scope": label,
                "evidence_kind": "corrected fold",
                "baseline_rmse_m": persistence_rmse,
                "candidate_rmse_m": candidate_rmse,
                "delta_rmse_m": delta,
                "signed_delta_label": f"{delta:+.6f}m",
                "direction": "개선 (Δ<0)",
            }
        )
    _require(all(row["delta_rmse_m"] < 0 for row in folds), "P3 fold direction drifted")
    overall = {
        "sequence": 1,
        "scope": "Case-pooled aggregate",
        "evidence_kind": "aggregate",
        "baseline_rmse_m": float(gate["persistence_rmse_m"]),
        "candidate_rmse_m": float(gate["candidate_rmse_m"]),
        "delta_rmse_m": overall_delta,
        "signed_delta_label": f"{overall_delta:+.6f}m",
        "direction": "개선 (Δ<0)",
    }
    split = metrics["split_audit"]
    _require(split["validation_case_count"] == 181, "P3 case count drifted")
    _require(split["repeated_station_episode_count"] == 0, "P3 episode overlap drifted")
    _require(split["cross_window_pairs_below_78h"] == 0, "P3 global spacing drifted")
    _require(split["context48_plus_target24_footprint_overlap_pairs"] == 0, "P3 footprint drifted")
    return {
        "baseline_sha256": FROZEN_BASELINE_SHA256["P3"],
        "candidate_sha256": candidate_sha,
        "candidate_rows": int(metrics["candidate_validation"]["rows"]),
        "candidate_cases": int(metrics["candidate_validation"]["cases"]),
        "status": "CORRECTED_RESEARCH_CANDIDATE_NOT_AUTO_PROMOTED",
        "pool_candidate_count_before_after": [3, 3],
        "validation_cases": int(split["validation_case_count"]),
        "validation_rows": int(split["validation_row_count"]),
        "overall_delta_rmse_m": overall_delta,
        "ci90_m": ci90,
        "improved_fold_count": 3,
        "chart_rows": [overall, *folds],
        "candidate_vs_current": comparison,
        "upload_count": 0,
    }


def _validate_policy(evidence: dict[str, Any]) -> None:
    policy = evidence["cross_problem_policy"]
    _require(
        policy["schema_version"] == "cross_problem_official_scoring_preregistration.v1",
        "policy schema drifted",
    )
    _require(
        policy["governing_rules"]["maximum_scored_candidates_per_problem"] == 3, "score cap drifted"
    )
    _require(
        policy["governing_rules"]["within_family_leaderboard_tuning"] is False,
        "tuning rule drifted",
    )
    for problem, expected in FROZEN_BASELINE_SHA256.items():
        actual = policy["problem_policies"][problem]["candidate_pool"]["candidates"][0]["sha256"]
        _require(actual == expected, f"frozen baseline policy pin drifted: {problem}")
    _require(
        policy["non_actions_in_this_policy_run"]["uploads"] == 0,
        "policy-run upload count drifted",
    )


def build_registry(
    evidence: dict[str, Any], comparison: dict[str, Any], *, generated_at: str
) -> dict[str, Any]:
    """Build the aggregate-only operational registry."""

    _validate_policy(evidence)
    p1 = _validate_p1(evidence)
    p2 = _validate_p2(evidence)
    p3 = _validate_p3(evidence, comparison)
    registry = {
        "schema_version": "training_revalidation.registry.v1",
        "registry_id": REPORT_ID,
        "generated_at_kst": generated_at,
        "status": "SEALED_RESEARCH_EVIDENCE__OFFICIAL_POOLS_UNCHANGED__NO_UPLOAD",
        "purpose": (
            "Separate corrected training/revalidation evidence from official candidate eligibility "
            "and preserve immutable baseline identities."
        ),
        "source_contract": {
            "sha_pinned_aggregate_json_only": True,
            "exception": (
                "P3 candidate-vs-current reads two SHA-pinned prediction CSVs only to compute "
                "aggregate distance and change counts."
            ),
            "raw_rows_retained_or_emitted": 0,
            "absolute_paths_emitted": 0,
            "secrets_emitted": 0,
        },
        "source_pins": {
            name: {
                "logical_path": RELATIVE_PATHS[name].as_posix(),
                "sha256": EXPECTED_SHA256[name],
            }
            for name in RELATIVE_PATHS
        },
        "independent_qa": {"P2": dict(P2_QA_ATTESTATION), "P3": dict(P3_QA_ATTESTATION)},
        "frozen_baselines": {
            problem: {
                "sha256": sha256,
                "unchanged": True,
                "role": "immutable current/rollback baseline",
            }
            for problem, sha256 in FROZEN_BASELINE_SHA256.items()
        },
        "eligibility_records": [
            {
                "problem": "P1",
                "track": "causal LightGBM candidate 2",
                "candidate_sha256": p1["causal_candidate_sha256"],
                "research_result": "trained, validated, and byte-identically reproduced",
                "official_eligibility": p1["causal_status"],
                "pool_effect": "P1 pre-score pool has two sealed eligible candidates",
                "upload_count": 0,
            },
            {
                "problem": "P1",
                "track": "sequence/duration structural slot",
                "candidate_sha256": None,
                "research_result": "no executable frozen deployment; failed closed",
                "official_eligibility": p1["duration_status"],
                "pool_effect": "no candidate added for the failed slot",
                "upload_count": 0,
            },
            {
                "problem": "P2",
                "track": "corrected repeated-forward candidate",
                "candidate_sha256": p2["candidate_sha256"],
                "research_result": "outer research comparison improved overall with one adverse fold",
                "official_eligibility": p2["status"],
                "pool_effect": "existing three-candidate preregistered pool unchanged",
                "upload_count": 0,
            },
            {
                "problem": "P3",
                "track": "corrected repeated-forward CatBoost/router/shrink candidate",
                "candidate_sha256": p3["candidate_sha256"],
                "research_result": "corrected research gate passed versus persistence",
                "official_eligibility": p3["status"],
                "pool_effect": "existing three-candidate preregistered pool unchanged",
                "upload_count": 0,
            },
        ],
        "evidence": {"P1": p1, "P2": p2, "P3": p3},
        "policy_effect": {
            "p1_causal_candidate2_is_pre_score_eligible": True,
            "p1_duration_slot_failed_closed": True,
            "p2_corrected_candidate_auto_promoted": False,
            "p3_corrected_candidate_auto_promoted": False,
            "existing_p2_p3_preregistered_pools_changed": False,
            "official_submission_choice_changed": False,
        },
        "operation_counters": {
            "frozen_baseline_mutations": 0,
            "official_pool_mutations_p2_p3": 0,
            "submission_uploads": 0,
        },
    }
    _validate_registry(registry)
    return registry


def _validate_registry(registry: dict[str, Any]) -> None:
    _require(
        registry["schema_version"] == "training_revalidation.registry.v1", "registry schema drifted"
    )
    _require(registry["operation_counters"]["submission_uploads"] == 0, "registry upload drifted")
    records = registry["eligibility_records"]
    _require(len(records) == 4, "eligibility record count drifted")
    _require(
        records[0]["official_eligibility"] == "ELIGIBLE_PRE_FIRST_SCORE_CANDIDATE_2",
        "P1 causal status drifted",
    )
    _require(
        records[1]["official_eligibility"] == "INELIGIBLE_FAILED_CLOSED",
        "P1 duration status drifted",
    )
    _require(
        all("NOT_AUTO_PROMOTED" in records[index]["official_eligibility"] for index in (2, 3)),
        "P2/P3 research status drifted",
    )
    _require(
        registry["evidence"]["P3"]["candidate_vs_current"]["exact_parsed_float_nonzero_rows"]
        == 748,
        "P3 change-count contract drifted",
    )
    serialized = json.dumps(registry, ensure_ascii=False)
    for forbidden in ("C:/Users/", "C:\\Users\\", "api_key", "access_token", "password"):
        _require(
            forbidden.lower() not in serialized.lower(), f"unsafe registry content: {forbidden}"
        )


def _source(source_id: str, label: str, path: str, sha256: str, note: str) -> dict[str, Any]:
    return {"id": source_id, "label": label, "path": path, "sha256": sha256, "note": note}


def _sql_text(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _rows_to_union_sql(rows: list[dict[str, Any]], fields: list[str]) -> str:
    selects = []
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, bool):
                rendered = "1" if value else "0"
            elif isinstance(value, (int, float)):
                rendered = repr(value)
            elif value is None:
                rendered = "NULL"
            else:
                rendered = _sql_text(value)
            values.append(f"{rendered} AS {field}")
        selects.append("SELECT " + ", ".join(values))
    return " UNION ALL ".join(selects)


def build_artifact(
    evidence: dict[str, Any], registry: dict[str, Any], *, generated_at: str
) -> dict[str, Any]:
    """Build the canonical technical report artifact from the sealed registry."""

    registry_sha = hashlib.sha256(_canonical_bytes(registry)).hexdigest()
    p2 = registry["evidence"]["P2"]
    p3 = registry["evidence"]["P3"]
    decision_rows = []
    for sequence, record in enumerate(registry["eligibility_records"], start=1):
        decision_rows.append(
            {
                "sequence": sequence,
                "problem": record["problem"],
                "track": record["track"],
                "research_result": record["research_result"],
                "official_eligibility": record["official_eligibility"],
                "pool_effect": record["pool_effect"],
                "upload_count": record["upload_count"],
            }
        )
    baseline_rows = [
        {
            "sequence": sequence,
            "problem": problem,
            "frozen_sha256": FROZEN_BASELINE_SHA256[problem],
            "unchanged": "YES",
            "role": "immutable current/rollback baseline",
        }
        for sequence, problem in enumerate(("P1", "P2", "P3"), start=1)
    ]

    sources = [
        _source(
            "training_registry",
            "Training/revalidation aggregate registry",
            DEFAULT_REGISTRY.as_posix(),
            registry_sha,
            "Builder-generated aggregate-only registry; no prediction rows are retained.",
        )
    ]
    source_labels = {
        "p1_pool_manifest": "P1 pre-score candidate-pool manifest",
        "p1_causal_receipt": "P1 causal candidate receipt",
        "p1_duration_failure": "P1 duration-slot failed-closed receipt",
        "p2_manifest": "P2 corrected repeated-forward manifest",
        "p2_metrics": "P2 corrected repeated-forward aggregate metrics",
        "p2_completion": "P2 same-attempt completion receipt",
        "p3_manifest": "P3 corrected repeated-forward manifest",
        "p3_metrics": "P3 corrected repeated-forward aggregate metrics",
        "cross_problem_policy": "Cross-problem preregistered scoring policy",
    }
    for name in RELATIVE_PATHS:
        sources.append(
            _source(
                name,
                source_labels[name],
                RELATIVE_PATHS[name].as_posix(),
                EXPECTED_SHA256[name],
                "SHA-pinned source; report adapter consumes aggregate fields only.",
            )
        )
    sources.extend(
        [
            {
                "id": "independent_qa_message",
                "label": "Independent post-run QA attestations",
                "note": (
                    "Message-only attestations: P2 PASS P0=0/P1=0; P3 "
                    "POST_RUN_INTEGRITY_GO P0=0/P1=0/mismatch=0. No receipt file was created. "
                    "The builder independently rechecks source hashes and P3 candidate distance."
                ),
            },
            {
                "id": "method_note",
                "label": "Technical delivery and chart-contract note",
                "note": (
                    "Technical audience; Required Structure maps to visible Technical Summary, Key "
                    "Findings, Scope/Definitions, Methodology, Limitations/Robustness, Recommended "
                    "Next Steps, and Further Questions blocks. MCP report tools were unavailable, so "
                    "the canonical artifact uses the technical portable HTML fallback. P2 and P3 "
                    "fold-delta charts are separated because °C and m are not commensurate. P1 has "
                    "no metric chart because its receipts establish structural eligibility and "
                    "failure state, not a comparable scored metric. Both charts use signed labels "
                    "and a neutral zero reference; P2 uses at most two direction roots and P3 a "
                    "single root, so color is not the sole encoding."
                ),
            },
        ]
    )

    p2_rows = p2["chart_rows"]
    p3_rows = p3["chart_rows"]
    p2_fields = [
        "sequence",
        "scope",
        "evidence_kind",
        "baseline_rmse_c",
        "candidate_rmse_c",
        "delta_rmse_c",
        "signed_delta_label",
        "direction",
    ]
    p3_fields = [
        "sequence",
        "scope",
        "evidence_kind",
        "baseline_rmse_m",
        "candidate_rmse_m",
        "delta_rmse_m",
        "signed_delta_label",
        "direction",
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "training_registry",
            "body": (
                "## Technical Summary — 통과한 연구 gate와 공식 후보 자격은 같지 않다\n\n"
                "**현재 공식적으로 추가 자격이 생긴 것은 P1 causal candidate 2뿐입니다.** 이 후보는 "
                "첫 official score 전에 고정된 선택을 그대로 학습·재현해 eligible로 봉인됐습니다. 반면 P1 "
                "sequence/duration slot은 실행 가능한 frozen deployment가 없어 failed closed입니다.\n\n"
                "P2와 P3의 교정 실험은 각각 outer/persistence 기준 연구 gate를 통과했지만 "
                "**corrected research candidates**입니다. 기존 3개짜리 preregistered official pool은 어느 "
                "쪽도 바뀌지 않았고 auto-promotion도 없습니다. 세 frozen baseline SHA는 불변이며 upload는 "
                "0회입니다. 따라서 이 보고서는 새 연구 근거를 등록하지만 현재 official submission 선택을 "
                "바꾸지 않습니다."
            ),
        },
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "training_registry",
            "body": (
                "## Key Findings — eligibility를 행 단위로 고정한다\n\n"
                "아래 표는 연구 결과와 official eligibility를 분리한 exact decision record입니다. "
                "`gate pass`는 local validation contract 통과를 뜻할 뿐, preregistered pool 편입이나 hidden "
                "성능 승인을 뜻하지 않습니다."
            ),
        },
        {"id": "eligibility_table", "type": "table", "tableId": "eligibility_decisions"},
        {
            "id": "baseline_integrity_intro",
            "type": "markdown",
            "sourceId": "cross_problem_policy",
            "body": (
                "### Frozen baseline identity는 세 문제 모두 유지됐다\n\n"
                "아래 SHA는 official 선택이 아니라 exact-byte current/rollback identity입니다. 이번 학습·재검증 "
                "과정은 이 파일들을 수정하지 않았고, P2/P3 연구 후보도 그 자리를 자동으로 대체하지 않습니다."
            ),
        },
        {"id": "baseline_table", "type": "table", "tableId": "frozen_baselines"},
        {
            "id": "p1_result",
            "type": "markdown",
            "sourceId": "p1_pool_manifest",
            "body": (
                "### P1 — causal candidate 2는 eligible, duration slot은 failed closed\n\n"
                "Causal LightGBM candidate 2는 169,011행의 key/order/schema 검사를 통과했고 saved-model "
                "재현 SHA가 candidate와 일치합니다. 새 outer-label search 없이 첫 score 전에 봉인되어 "
                "`ELIGIBLE_PRE_FIRST_SCORE_CANDIDATE_2`입니다. Sequence/duration slot은 full-train epoch, "
                "deployment runner, compatible checkpoint가 고정되지 않아 후보를 만들어내지 않고 "
                "`INELIGIBLE_FAILED_CLOSED`로 끝났습니다. 이 두 receipt에는 서로 비교 가능한 metric이 없어 "
                "P1 chart는 의도적으로 생략했습니다."
            ),
        },
        {
            "id": "p2_result",
            "type": "markdown",
            "sourceId": "p2_metrics",
            "body": (
                "### P2 — outer aggregate는 개선됐지만 inner와 한 fold가 반대다\n\n"
                f"Corrected outer fold-equal RMSE는 baseline 1.259103°C에서 candidate 1.115888°C로 "
                f"{p2['outer_overall_delta_rmse_c']:+.6f}°C 변했고, paired KST-day bootstrap CI90은 "
                f"[{p2['outer_ci90_c'][0]:+.6f}, {p2['outer_ci90_c'][1]:+.6f}]°C입니다. 그러나 inner "
                f"diagnostic aggregate는 {p2['inner_delta_rmse_c']:+.6f}°C로 열세이고, May–Jun outer "
                "fold도 +0.247666°C 악화됩니다. 즉 aggregate gain은 fold-stable evidence가 아니며, adaptive "
                "research·no fresh holdout 조건에서 얻은 local 결과입니다. Candidate는 연구 artifact로만 "
                "남고 기존 official pool은 그대로입니다."
            ),
        },
        {"id": "p2_chart", "type": "chart", "chartId": "p2_fold_delta_rmse"},
        {
            "id": "p3_result",
            "type": "markdown",
            "sourceId": "p3_metrics",
            "body": (
                "### P3 — corrected split에서는 persistence를 이겼지만 current와 거의 같은 후보이다\n\n"
                f"Station-global ≥78h, episode-disjoint, 72h footprint-disjoint인 181-case/1,086-row "
                f"validation에서 final RMSE는 0.779105m, persistence는 0.863497m로 "
                f"{p3['overall_delta_rmse_m']:+.6f}m 개선됐습니다. Case bootstrap CI90은 "
                f"[{p3['ci90_m'][0]:+.6f}, {p3['ci90_m'][1]:+.6f}]m이고 세 fold 모두 음수입니다. "
                "다만 이것은 persistence 대비 연구 근거이지 current official candidate 대비 local-label 승리가 "
                "아닙니다."
            ),
        },
        {"id": "p3_chart", "type": "chart", "chartId": "p3_fold_delta_rmse"},
        {
            "id": "p3_distance_caveat",
            "type": "markdown",
            "sourceId": "training_registry",
            "body": (
                "### P3 candidate-current 차이는 작고 ‘변경 행’ 정의는 parser-sensitive하다\n\n"
                f"두 SHA-pinned CSV의 prediction RMSE distance는 "
                f"{p3['candidate_vs_current']['prediction_rmse_distance_m']:.13f}m입니다. Pandas 3.0.1 "
                f"default float64 parser에서 `dx != 0`인 행은 "
                f"{p3['candidate_vs_current']['exact_parsed_float_nonzero_rows']}/1,200, "
                f"`|dx| > 1e-12`는 {p3['candidate_vs_current']['absolute_delta_gt_1e_12_rows']}/1,200입니다. "
                f"Decimal text inequality는 {p3['candidate_vs_current']['decimal_text_unequal_rows']}/1,200이므로 "
                "단순 ‘changed rows’는 parser-dependent합니다. 핵심은 이 후보가 current와 매우 가깝고, 이 "
                "distance 자체는 hidden 성능 우열 근거가 아니라는 점입니다."
            ),
        },
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## Scope, Data, and Metric Definitions\n\n"
                "- **Official eligibility:** 첫 score 전 봉인, 구조적 slot, exact SHA, governing policy를 모두 "
                "충족해 score candidate가 될 수 있는 상태입니다. 연구 gate pass와 다릅니다.\n"
                "- **Corrected research candidate:** 교정된 local validation에서 평가·생성됐지만 기존 official "
                "pool에 자동 편입되지 않은 artifact입니다.\n"
                "- **ΔRMSE:** candidate − baseline입니다. 음수는 개선, 양수는 악화입니다. P2 단위는 °C, "
                "P3 단위는 유의파고(hs) m이며 문제 간 수치를 한 축에서 비교하지 않습니다.\n"
                "- **P2 baseline:** 같은 corrected outer fold의 fixed baseline. **P3 baseline:** 같은 corrected "
                "cases의 persistence입니다. P3 current-vs-new comparison은 label-free prediction distance입니다.\n"
                "- **Frozen baseline:** exact-byte current/rollback 파일로, hidden 성능 승인과 동의어가 아닙니다."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "method_note",
            "body": (
                "## Methodology — pinned receipts, schema assertions, bounded recomputation\n\n"
                "Builder는 9개 aggregate JSON을 logical relative path와 exact SHA-256으로 읽고, problem별 "
                "adapter가 status, candidate identity, frozen immutability, upload=0, fold metric, bootstrap "
                "interval, official-pool boundary를 assertion으로 검증합니다. P3 candidate-current distance만 "
                "예외적으로 두 SHA-pinned CSV에서 집계하며, key order를 대조한 뒤 RMSE와 세 가지 change count만 "
                "남기고 row values는 registry/report에 저장하지 않습니다.\n\n"
                "시각화 계약은 두 horizontal signed bar입니다. 각 chart는 overall과 세 fold의 같은 단위만 "
                "포함하고 neutral zero line과 exact signed label을 사용합니다. CI는 bar geometry로 흉내내지 않고 "
                "인접 문단과 source dataset에 보존합니다. P1은 scored metric 근거가 없어 chart를 만들지 않았고, "
                "cross-problem scale chart도 단위·baseline·validation grain이 달라 금지했습니다."
            ),
        },
        {
            "id": "limitations_robustness",
            "type": "markdown",
            "sourceId": "independent_qa_message",
            "body": (
                "## Limitations, Uncertainty, and Robustness Checks\n\n"
                "P2와 P3 독립 post-run QA는 각각 P0=0/P1=0으로 통과했지만 receipt 파일은 생성되지 않아 "
                "message-only attestation으로 명시했습니다. Builder는 그 대신 source SHA와 핵심 aggregate를 "
                "다시 검산합니다. P3 manifest에는 package versions와 6개 transitive module SHA가 없어 exact "
                "environment replay 증거는 완전하지 않습니다.\n\n"
                "P2는 inner aggregate 열세와 1/3 outer fold 악화가 있어 positive overall CI만으로 robustness를 "
                "과장할 수 없습니다. P3의 all-fold gain은 persistence 대비이며 current 대비 labeled comparison이 "
                "아닙니다. 두 corrected validations 모두 official hidden calibration을 제공하지 않고, 새 candidate를 "
                "기존 pool에 자동 승격할 근거도 제공하지 않습니다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended Next Steps — 연구 registry와 score registry를 계속 분리한다\n\n"
                "1. P1 candidate 2를 score 대상으로 고려할 때만 daily slot, exact SHA, schema/key, platform split, "
                "명시적 사용자 승인을 다시 확인합니다.\n"
                "2. P1 duration slot은 새 frozen deployment contract 없이는 계속 failed closed로 둡니다.\n"
                "3. P2/P3 corrected candidates는 research registry에만 유지하고, 기존 preregistered pool이나 current "
                "submission을 변경하지 않습니다.\n"
                "4. P2는 inner/May–Jun failure를 설명할 독립 계절 block이 생기기 전 승격 논의를 중단합니다.\n"
                "5. P3는 current 대비 labeled, predeclared comparison이 생기기 전 persistence 승리를 current 승리로 "
                "해석하지 않습니다. 어떤 upload도 별도 사용자 승인 없이는 수행하지 않습니다."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further Questions\n\n"
                "- P2의 May–Jun 역전이 seasonal covariate shift인지 model projection failure인지 어느 untouched "
                "block에서 분리할 수 있는가?\n"
                "- P3 새 후보와 current의 작은 prediction distance를 평가할 independent labeled surface가 있는가?\n"
                "- P1 candidate 2를 실제로 score할 조건과 candidate 1 유지 margin을 언제 적용할 것인가?\n"
                "- Message-only QA를 장기 보존 가능한 aggregate receipt로 전환할 필요가 있는가?"
            ),
        },
    ]

    charts = [
        {
            "id": "p2_fold_delta_rmse",
            "title": "P2 corrected repeated-forward ΔRMSE by fold",
            "subtitle": (
                "°C; candidate − baseline; negative=improvement; fold-equal aggregate + 3 outer folds"
            ),
            "showDescription": True,
            "intent": "comparison",
            "question": "P2 candidate gain이 corrected outer folds에서 일관적인가?",
            "rationale": (
                "Four same-unit signed comparisons use a horizontal bar, exact labels, direction tone, "
                "and a neutral zero line; CI remains in the adjacent narrative and reviewed dataset."
            ),
            "type": "horizontalBar",
            "dataset": "p2_fold_delta_rmse",
            "sourceId": "p2_metrics",
            "source": {
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": _rows_to_union_sql(p2_rows, p2_fields),
                    "description": "Deterministic aggregate projection of P2 outer fold metrics.",
                    "tables_used": [RELATIVE_PATHS["p2_metrics"].as_posix()],
                    "filters": [
                        "candidate minus baseline",
                        "fold-equal aggregate plus three corrected outer folds",
                        "raw prediction rows excluded",
                    ],
                    "metric_definitions": {
                        "delta_rmse_c": "candidate RMSE minus baseline RMSE in °C; negative is improvement",
                        "overall_ci90_c": (
                            f"paired KST-day bootstrap [{p2['outer_ci90_c'][0]}, "
                            f"{p2['outer_ci90_c'][1]}]"
                        ),
                    },
                }
            },
            "valueFormat": "number",
            "unit": "°C",
            "layout": "full",
            "maxRows": 4,
            "settings": {"orientation": "horizontal", "groupMode": "single", "showValues": True},
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0,
                    "label": "no change",
                    "color": "neutral",
                    "lineStyle": "solid",
                }
            ],
            "encodings": {
                "x": {"field": "scope", "type": "nominal", "label": "Validation scope"},
                "y": {
                    "field": "delta_rmse_c",
                    "type": "quantitative",
                    "label": "ΔRMSE (°C)",
                    "format": "number",
                },
                "color": {"field": "direction", "type": "nominal", "label": "Direction"},
                "label": {
                    "field": "signed_delta_label",
                    "type": "nominal",
                    "label": "Signed ΔRMSE",
                },
                "tooltip": [
                    {
                        "field": "baseline_rmse_c",
                        "type": "quantitative",
                        "label": "Baseline",
                        "unit": "°C",
                    },
                    {
                        "field": "candidate_rmse_c",
                        "type": "quantitative",
                        "label": "Candidate",
                        "unit": "°C",
                    },
                    {"field": "evidence_kind", "type": "nominal", "label": "Evidence grain"},
                ],
            },
        },
        {
            "id": "p3_fold_delta_rmse",
            "title": "P3 corrected repeated-forward ΔRMSE by fold",
            "subtitle": (
                "m of significant wave height (hs); final − persistence; negative=improvement; "
                "case-pooled aggregate + 3 corrected folds"
            ),
            "showDescription": True,
            "intent": "comparison",
            "question": "P3 final candidate gain versus persistence가 corrected folds에서 일관적인가?",
            "rationale": (
                "Four same-unit signed comparisons use a single-root horizontal bar, exact labels, "
                "and a neutral zero line; case-bootstrap CI remains adjacent rather than simulated."
            ),
            "type": "horizontalBar",
            "dataset": "p3_fold_delta_rmse",
            "sourceId": "p3_metrics",
            "source": {
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": _rows_to_union_sql(p3_rows, p3_fields),
                    "description": "Deterministic aggregate projection of P3 corrected fold metrics.",
                    "tables_used": [RELATIVE_PATHS["p3_metrics"].as_posix()],
                    "filters": [
                        "final candidate minus persistence",
                        "case-pooled aggregate plus three corrected folds",
                        "181 complete six-lead validation cases",
                        "raw prediction rows excluded",
                    ],
                    "metric_definitions": {
                        "delta_rmse_m": (
                            "final candidate RMSE minus persistence RMSE in meters of significant "
                            "wave height; negative is improvement"
                        ),
                        "overall_ci90_m": (
                            f"paired case bootstrap [{p3['ci90_m'][0]}, {p3['ci90_m'][1]}]"
                        ),
                    },
                }
            },
            "valueFormat": "number",
            "unit": "m",
            "layout": "full",
            "maxRows": 4,
            "settings": {"orientation": "horizontal", "groupMode": "single", "showValues": True},
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0,
                    "label": "no change",
                    "color": "neutral",
                    "lineStyle": "solid",
                }
            ],
            "encodings": {
                "x": {"field": "scope", "type": "nominal", "label": "Validation scope"},
                "y": {
                    "field": "delta_rmse_m",
                    "type": "quantitative",
                    "label": "ΔRMSE (m)",
                    "format": "number",
                },
                "label": {
                    "field": "signed_delta_label",
                    "type": "nominal",
                    "label": "Signed ΔRMSE",
                },
                "tooltip": [
                    {
                        "field": "baseline_rmse_m",
                        "type": "quantitative",
                        "label": "Persistence",
                        "unit": "m",
                    },
                    {
                        "field": "candidate_rmse_m",
                        "type": "quantitative",
                        "label": "Final candidate",
                        "unit": "m",
                    },
                    {"field": "evidence_kind", "type": "nominal", "label": "Evidence grain"},
                ],
            },
        },
    ]
    decision_fields = [
        "sequence",
        "problem",
        "track",
        "research_result",
        "official_eligibility",
        "pool_effect",
        "upload_count",
    ]
    baseline_fields = ["sequence", "problem", "frozen_sha256", "unchanged", "role"]
    tables = [
        {
            "id": "eligibility_decisions",
            "title": "Training/revalidation decision registry",
            "subtitle": "Research result, official eligibility, pool effect, and upload count",
            "dataset": "eligibility_decisions",
            "sourceId": "training_registry",
            "source": {
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": _rows_to_union_sql(decision_rows, decision_fields),
                    "description": "Exact four-row eligibility projection from the sealed registry.",
                    "tables_used": [DEFAULT_REGISTRY.as_posix()],
                    "filters": ["P1 causal", "P1 duration", "P2 corrected", "P3 corrected"],
                    "metric_definitions": {
                        "official_eligibility": "policy status; not inferred from local metric direction",
                        "upload_count": "submission uploads performed in the registered runs",
                    },
                }
            },
            "density": "spacious",
            "defaultSort": {"field": "sequence", "direction": "asc"},
            "columns": [
                {"field": "problem", "label": "문제", "type": "text"},
                {"field": "track", "label": "Track", "type": "text"},
                {"field": "research_result", "label": "Research result", "type": "text"},
                {"field": "official_eligibility", "label": "Official eligibility", "type": "text"},
                {"field": "pool_effect", "label": "Pool effect", "type": "text"},
                {"field": "upload_count", "label": "Uploads", "type": "number"},
            ],
        },
        {
            "id": "frozen_baselines",
            "title": "Immutable frozen baseline identities",
            "subtitle": "Exact SHA-256 values; all unchanged; upload count remains zero",
            "dataset": "frozen_baselines",
            "sourceId": "cross_problem_policy",
            "source": {
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": _rows_to_union_sql(baseline_rows, baseline_fields),
                    "description": "Exact current/rollback identities from the pinned policy and run manifests.",
                    "tables_used": [
                        RELATIVE_PATHS["cross_problem_policy"].as_posix(),
                        RELATIVE_PATHS["p1_pool_manifest"].as_posix(),
                        RELATIVE_PATHS["p2_manifest"].as_posix(),
                        RELATIVE_PATHS["p3_manifest"].as_posix(),
                    ],
                    "filters": ["immutable current candidate only", "no raw rows"],
                    "metric_definitions": {
                        "frozen_sha256": "exact byte identity of the current rollback baseline"
                    },
                }
            },
            "density": "dense",
            "defaultSort": {"field": "sequence", "direction": "asc"},
            "columns": [
                {"field": "problem", "label": "문제", "type": "text"},
                {"field": "frozen_sha256", "label": "Frozen SHA-256", "type": "text"},
                {"field": "unchanged", "label": "Unchanged", "type": "text"},
                {"field": "role", "label": "Role", "type": "text"},
            ],
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "Technical training/revalidation registry for P1–P3",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "eligibility_decisions": decision_rows,
                "frozen_baselines": baseline_rows,
                "p2_fold_delta_rmse": p2_rows,
                "p3_fold_delta_rmse": p3_rows,
            },
            "accessIssues": [],
        },
        "sources": [
            {
                "id": source["id"],
                "label": source["label"],
                **({"path": source["path"]} if "path" in source else {}),
                **({"sha256": source["sha256"]} if "sha256" in source else {}),
            }
            for source in sources
        ],
        "package_info": {
            "originUrl": f"artifact://{REPORT_ID}",
            "controls": {"edit": False, "refresh": False},
            "delivery": "technical portable HTML fallback",
        },
    }
    _validate_artifact(artifact, registry_sha=registry_sha)
    return artifact


def _validate_artifact(artifact: dict[str, Any], *, registry_sha: str) -> None:
    manifest = artifact["manifest"]
    _require(artifact["surface"] == "report", "artifact surface drifted")
    _require(manifest["blocks"][0]["body"] == f"# {REPORT_TITLE}", "title drifted")
    _require(
        manifest["blocks"][1]["body"].startswith("## Technical Summary"), "summary order drifted"
    )
    required_roles = (
        "## Technical Summary",
        "## Key Findings",
        "## Scope, Data, and Metric Definitions",
        "## Methodology",
        "## Limitations, Uncertainty, and Robustness Checks",
        "## Recommended Next Steps",
        "## Further Questions",
    )
    serialized = json.dumps(artifact, ensure_ascii=False)
    for heading in required_roles:
        _require(heading in serialized, f"technical report role missing: {heading}")
    for phrase in (
        "ELIGIBLE_PRE_FIRST_SCORE_CANDIDATE_2",
        "INELIGIBLE_FAILED_CLOSED",
        "CORRECTED_RESEARCH_CANDIDATE_NOT_AUTO_PROMOTED",
        "MCP report tools were unavailable",
        "technical portable HTML fallback",
        "0.0010525759878m",
        "748/1,200",
        "660/1,200",
        "825/1,200",
    ):
        _require(phrase in serialized, f"required report phrase missing: {phrase}")
    _require(len(manifest["charts"]) == 2, "exactly two unit-separated charts required")
    chart_ids = [chart["id"] for chart in manifest["charts"]]
    _require(chart_ids == ["p2_fold_delta_rmse", "p3_fold_delta_rmse"], "chart order drifted")
    _require(
        all("p1" not in chart_id.lower() for chart_id in chart_ids), "P1 metric chart forbidden"
    )
    _require(manifest["charts"][0]["unit"] == "°C", "P2 chart unit drifted")
    _require(manifest["charts"][1]["unit"] == "m", "P3 chart unit drifted")
    for chart in manifest["charts"]:
        _require(chart["type"] == "horizontalBar", f"chart type drifted: {chart['id']}")
        _require(chart["settings"]["showValues"] is True, f"signed labels missing: {chart['id']}")
        _require(chart["referenceLines"][0]["value"] == 0, f"zero line missing: {chart['id']}")
        _require(
            chart["referenceLines"][0]["color"] == "neutral",
            f"zero line color drifted: {chart['id']}",
        )
    p2_rows = artifact["snapshot"]["datasets"]["p2_fold_delta_rmse"]
    p3_rows = artifact["snapshot"]["datasets"]["p3_fold_delta_rmse"]
    _require(len(p2_rows) == len(p3_rows) == 4, "chart row count drifted")
    _require(sum(row["delta_rmse_c"] > 0 for row in p2_rows) == 1, "P2 direction mix drifted")
    _require(all(row["delta_rmse_m"] < 0 for row in p3_rows), "P3 fold direction drifted")
    _require(
        all(row["signed_delta_label"].startswith(("+", "-")) for row in p2_rows + p3_rows),
        "signed labels drifted",
    )
    _require(len(manifest["tables"]) == 2, "table count drifted")
    _require(
        len(artifact["snapshot"]["datasets"]["eligibility_decisions"]) == 4,
        "decision table drifted",
    )
    _require(
        len(artifact["snapshot"]["datasets"]["frozen_baselines"]) == 3, "baseline table drifted"
    )
    registry_source = next(
        source for source in manifest["sources"] if source["id"] == "training_registry"
    )
    _require(registry_source["sha256"] == registry_sha, "registry source pin drifted")
    source_ids = {source["id"] for source in manifest["sources"]}
    _require(len(source_ids) == len(manifest["sources"]), "duplicate source ids")
    for block in manifest["blocks"]:
        if "sourceId" in block:
            _require(block["sourceId"] in source_ids, f"missing block source: {block['id']}")
    for visual in [*manifest["charts"], *manifest["tables"]]:
        _require(visual["sourceId"] in source_ids, f"missing visual source: {visual['id']}")
    for forbidden in ("C:/Users/", "C:\\Users\\", "api_key", "access_token", "password"):
        _require(
            forbidden.lower() not in serialized.lower(), f"unsafe artifact content: {forbidden}"
        )


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"append-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _require(not temporary.exists(), f"stale temporary output exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(_canonical_bytes(payload))
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate pins, schemas, aggregate recomputation, registry, and artifact without writing.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    artifact_path = args.artifact if args.artifact.is_absolute() else root / args.artifact
    _require(
        registry_path.resolve() == (root / DEFAULT_REGISTRY).resolve(), "registry path is frozen"
    )
    _require(
        artifact_path.resolve() == (root / DEFAULT_ARTIFACT).resolve(), "artifact path is frozen"
    )
    evidence = collect_evidence(root)
    comparison = recompute_p3_candidate_distance(root)
    generated_at = args.generated_at or datetime.now(KST).isoformat()
    registry = build_registry(evidence, comparison, generated_at=generated_at)
    artifact = build_artifact(evidence, registry, generated_at=generated_at)
    registry_sha = hashlib.sha256(_canonical_bytes(registry)).hexdigest()
    if args.check_only:
        print(
            "PASS: validated 9 SHA-pinned aggregate sources, P3 bounded recomputation, "
            f"registry {registry_sha}, and complete technical report artifact"
        )
        return 0
    _require(not registry_path.exists(), f"append-only registry exists: {DEFAULT_REGISTRY}")
    _require(not artifact_path.exists(), f"append-only artifact exists: {DEFAULT_ARTIFACT}")
    _write_new(registry_path, registry)
    _write_new(artifact_path, artifact)
    print(f"PASS: wrote {DEFAULT_REGISTRY.as_posix()} ({registry_sha})")
    print(f"PASS: wrote {DEFAULT_ARTIFACT.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
