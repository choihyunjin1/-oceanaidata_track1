"""Build the append-only full-improvement-cycle registry and technical report.

Only SHA-pinned aggregate JSON receipts are read.  The builder never reads or
emits raw training, OOF, prediction, test, or submission rows.  P1, P2, and P3
use different metric units, so their signed validation comparisons remain in
three separate chart roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
REPORT_TITLE = "Full Improvement Cycle — 실제 개선 학습·full fit·candidate 감사"
REPORT_ID = "full-improvement-cycle-2026-08-22-r1"
DEFAULT_REGISTRY = Path("artifacts/full_improvement_cycle_20260822/registry.json")
DEFAULT_ARTIFACT = Path("reports/generated/full_improvement_cycle_2026-08-22_r1/artifact.json")

RELATIVE_PATHS = {
    "p1_metrics": Path("artifacts/p1_full_improvement_cycle_20260822_v1/metrics.json"),
    "p1_result": Path("artifacts/p1_full_improvement_cycle_20260822_v1/result.json"),
    "p1_manifest": Path("artifacts/p1_full_improvement_cycle_20260822_v1/manifest.json"),
    "p1_qa": Path("artifacts/p1_full_improvement_cycle_20260822_v1/qa/independent_validation.json"),
    "p1_failure": Path("artifacts/p1_full_improvement_cycle_20260822_v1/execution_failure.json"),
    "p2_metrics": Path("artifacts/p2_conservative_stack_improvement_v1/metrics.json"),
    "p2_manifest": Path("artifacts/p2_conservative_stack_improvement_v1/manifest.json"),
    "p2_completion": Path(
        "artifacts/p2_conservative_stack_improvement_v1/resume_completion_status.json"
    ),
    "p2_timestamp_correction": Path(
        "artifacts/p2_conservative_stack_improvement_v1/provenance_timestamp_correction.json"
    ),
    "p2_addendum_seal": Path(
        "artifacts/p2_conservative_stack_improvement_v1/provenance_addendum_seal.json"
    ),
    "p3_metrics": Path("artifacts/p3_corrected_fixed_long_shrink_v5_full_refit/metrics.json"),
    "p3_manifest": Path("artifacts/p3_corrected_fixed_long_shrink_v5_full_refit/manifest.json"),
    "p3_completion": Path(
        "artifacts/p3_corrected_fixed_long_shrink_v5_full_refit/resume_completion_status.json"
    ),
    "p3_qa": Path(
        "artifacts/p3_corrected_fixed_long_shrink_v5_full_refit_QA/independent_aggregate_audit.json"
    ),
}

EXPECTED_SHA256 = {
    "p1_metrics": "ebf8240ff1eeb249c12c8cc03c5e220272f87ce02ae7dcee4c0d2a67a5fd5a41",
    "p1_result": "222fae7b2fa49c3d24ee3c01b25667f31e008b4ffcb8b9663b39c7c78ac28f08",
    "p1_manifest": "08345c3350b5b28cca9b8fb9511cc555a3e9f870aa5f6362829a617782880438",
    "p1_qa": "d10f36f1f5a97f46178ba24ed46a6692baf6b3279bf2e9bbe228d90754d09b13",
    "p1_failure": "0a57ff37d3fd9f5a82438ad21b20790df3dcc6d5b08060484b27ae1829d2e9c0",
    "p2_metrics": "b8614593b69cd7dad3f7efa05c5902f4b9e134467ce198a585620fea6069ef99",
    "p2_manifest": "6c744182595cbf37beae01cd3395c20e6de6cdd10a72f20aed36705956713d3e",
    "p2_completion": "cd2f52dabeeca8024c63d6450bc0085d0c823f6273bfc5e51f1aaed42ea02ed5",
    "p2_timestamp_correction": ("89845c446d89b8361853ba0b35ad26076fe31aa70d829460730151dd14390d8e"),
    "p2_addendum_seal": "6d8f324c79feaf00b37cd3d8ff979d6cdeb18057c0fd0366d80929727c5e4b4a",
    "p3_metrics": "427349bce4bf97ad6d71d3760fe1be199c912afc6c61a4fa1e61d3db7c668303",
    "p3_manifest": "f21a4a986c7540184441f92d81a59ee70e5b9aa6a030de9b5d8073e3e42ebd56",
    "p3_completion": "7b5a97f1eff012efd5856f5690f881c2af3dad2bc3b8367046cbaef26060da21",
    "p3_qa": "fc51109c33301f90e68a36b6c3e2ebadfe3d5aeffce213adc127e210520d5dc5",
}

FROZEN_SHA256 = {
    "P1": "28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3",
    "P2": "1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf",
    "P3": "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
}

CANDIDATE_SHA256 = {
    "P1": "8c0f4627b8da2af1c8362233e6a563e1033dd7119db126d6097272a0cc37bd6e",
    "P2": "3960660b1e4076c88efdb927a50073aa2d8f1435bc1c7d6f2f40885aea2f2350",
    "P3": "6f9d10e6efeb7e884629d1595d7c76afdc10ac1d89f7738e5e66f4d613c9786f",
}

FULL_CYCLE_QA_ATTESTATION = {
    "medium": "message_only",
    "receipt_created": False,
    "verdict": "YES",
    "p0_findings": 0,
    "p1_findings": 0,
    "mismatches": 0,
    "scope": "full-cycle aggregate source, metric, model, candidate, and immutability review",
    "p1_strict_clock_caveat": (
        "Event-protected fold assignment leaves the 119-row tail of the Q3 I-ORS layer-1 "
        "positive event (2025-10-01 00:00–19:40 KST) after the earliest global Q4 timestamp, "
        "so globally strict wall-clock-earlier labels must not be claimed. Fold order and each "
        "station-layer chronology remain safe. Removing those 119 rows from calibration leaves "
        "Q4 parameters 0.75/0.15 and Q4 predictions unchanged; the performance verdict is not blocked."
    ),
}


class FullCycleReportError(RuntimeError):
    """Raised when an evidence pin or reporting contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullCycleReportError(message)


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


def _close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    _require(
        math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance),
        f"{label} drifted: {actual} != {expected}",
    )


def _validate_path_contract() -> None:
    _require(set(RELATIVE_PATHS) == set(EXPECTED_SHA256), "source pin/path set drifted")
    for name, path in RELATIVE_PATHS.items():
        _require(not path.is_absolute(), f"absolute source path forbidden: {name}")
        _require(path.suffix.lower() == ".json", f"non-JSON source forbidden: {name}")
        _require(".." not in path.parts, f"parent traversal forbidden: {name}")


def collect_evidence(root: Path) -> dict[str, Any]:
    """Load only final aggregate JSON receipts and verify every byte pin."""

    _validate_path_contract()
    evidence: dict[str, Any] = {}
    for name, relative in RELATIVE_PATHS.items():
        expected = EXPECTED_SHA256[name]
        _require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"bad SHA pin: {name}")
        path = root / relative
        _require(path.is_file(), f"missing aggregate source: {relative.as_posix()}")
        actual = _sha256(path)
        _require(actual == expected, f"SHA mismatch for {name}: {actual} != {expected}")
        evidence[name] = _read_json(path)
    evidence["hashes"] = dict(EXPECTED_SHA256)
    return evidence


def _validate_qa_attestation(*, allow_pending: bool) -> None:
    attestation = FULL_CYCLE_QA_ATTESTATION
    if attestation["verdict"] == "PENDING_FINAL":
        _require(allow_pending, "full-cycle independent QA final verdict is still pending")
        return
    _require(attestation["medium"] == "message_only", "QA medium drifted")
    _require(attestation["receipt_created"] is False, "do not invent a QA receipt")
    _require(attestation["p0_findings"] == 0, "full-cycle QA P0 findings must be zero")
    _require(attestation["p1_findings"] == 0, "full-cycle QA P1 findings must be zero")
    _require(attestation["mismatches"] == 0, "full-cycle QA mismatch count must be zero")


def _p1_record(evidence: dict[str, Any]) -> dict[str, Any]:
    metrics = evidence["p1_metrics"]
    result = evidence["p1_result"]
    manifest = evidence["p1_manifest"]
    qa = evidence["p1_qa"]
    failure = evidence["p1_failure"]
    _require(metrics["experiment_id"] == result["experiment_id"], "P1 experiment drifted")
    _require(manifest["experiment_id"] == result["experiment_id"], "P1 manifest drifted")
    _require(metrics["winner"] == result["winner"], "P1 winner drifted")
    _require(
        metrics["guard"]["passed"] is True and result["guard_passed"] is True, "P1 gate failed"
    )
    _require(metrics["outer_key_alignment_exact"] is True, "P1 key alignment drifted")
    _require(metrics["target_fold_label_reads_before_prediction"] == 0, "P1 leakage drifted")
    _close(result["outer_incumbent_f1"], 0.8603708380408055, "P1 incumbent F1")
    _close(result["outer_candidate_f1"], 0.8609416445623342, "P1 candidate F1")
    _close(result["outer_f1_delta"], 0.0005708065215287439, "P1 delta F1")
    _require(result["candidate"]["sha256"] == CANDIDATE_SHA256["P1"], "P1 candidate drifted")
    _require(result["reproduction"]["byte_identical"] is True, "P1 reproduction drifted")
    _require(result["strict_validation"]["test_order_match"] is True, "P1 order drifted")
    _require(result["strict_validation"]["rows"] == 169011, "P1 candidate rows drifted")
    _require(result["full_fit"]["new_resume_model_fits"] == 1, "P1 resume fit count drifted")
    _require(result["full_fit"]["train_rows"] == 776706, "P1 train rows drifted")
    _require(
        result["full_fit"]["invalid_partial_causal_preserved"]["eligible"] is False,
        "P1 invalid partial eligibility drifted",
    )
    counters = result["operation_counters"]
    _require(counters["submission_uploads"] == 0, "P1 upload count drifted")
    _require(counters["source_mutations"] == 0, "P1 source mutation drifted")
    _require(counters["frozen_submission_mutations"] == 0, "P1 frozen mutation drifted")
    _require(counters["test_label_reads"] == 0, "P1 test label read drifted")
    before = result["protected_hashes"]["before"]["frozen"]
    after = result["protected_hashes"]["after"]["frozen"]
    _require(before == after == FROZEN_SHA256["P1"], "P1 frozen SHA drifted")
    _require(qa["decision"] == "QA_PASS", "P1 QA verdict drifted")
    _require(qa["P0_finding_count"] == qa["P1_finding_count"] == 0, "P1 QA findings")
    _require(qa["artifact_sha256"]["metrics"] == EXPECTED_SHA256["p1_metrics"], "P1 QA metric pin")
    _require(qa["artifact_sha256"]["result"] == EXPECTED_SHA256["p1_result"], "P1 QA result pin")
    _require(failure["error"] == "model and inference feature modes differ", "P1 failure drifted")
    _require(failure["submission_uploads"] == 0, "P1 failure upload drifted")
    ci90 = [float(value) for value in metrics["bootstrap"]["delta_ci90"]]
    _require(ci90[0] < 0 < ci90[1], "P1 CI must include zero")
    incumbent = metrics["guard"]["by_fold_incumbent"]
    candidate = metrics["guard"]["by_fold_candidate"]
    deltas = metrics["guard"]["by_fold_f1_delta"]
    fold_labels = [("2025 Q2", "2025_q2"), ("2025 Q3", "2025_q3"), ("2025 Q4", "2025_q4")]
    rows = [
        {
            "sequence": 1,
            "scope": "Pooled outer",
            "baseline": float(result["outer_incumbent_f1"]),
            "candidate": float(result["outer_candidate_f1"]),
            "delta": float(result["outer_f1_delta"]),
            "signed_delta_label": f"{float(result['outer_f1_delta']):+.7f}",
            "direction": "improved (Δ>0)",
        }
    ]
    for sequence, (label, key) in enumerate(fold_labels, start=2):
        delta = float(deltas[key])
        rows.append(
            {
                "sequence": sequence,
                "scope": label,
                "baseline": float(incumbent[key]["f1"]),
                "candidate": float(candidate[key]["f1"]),
                "delta": delta,
                "signed_delta_label": f"{delta:+.7f}",
                "direction": "improved (Δ>0)" if delta > 0 else "unchanged (Δ=0)",
            }
        )
    return {
        "problem": "P1",
        "metric": "micro F1",
        "unit": "F1",
        "better_direction": "higher",
        "baseline_value": float(result["outer_incumbent_f1"]),
        "candidate_value": float(result["outer_candidate_f1"]),
        "delta_candidate_minus_baseline": float(result["outer_f1_delta"]),
        "ci90": ci90,
        "ci90_includes_zero": True,
        "uncertainty_interpretation": "descriptive local gain; CI90 includes zero",
        "validation_surface": {"rows": int(metrics["outer_rows"]), "folds": 3},
        "chart_rows": rows,
        "winner": result["winner"],
        "full_fit": {
            "status": "actual corrected full fit complete",
            "train_rows": int(result["full_fit"]["train_rows"]),
            "model_sha256": {
                "corrected_causal_lightgbm": result["full_fit"]["corrected_causal_lightgbm"][
                    "sha256"
                ],
                "corrected_ensemble": result["full_fit"]["corrected_ensemble"]["sha256"],
                "reused_valid_offline_xgboost": result["full_fit"][
                    "valid_offline_xgboost_reused_from_initial_attempt"
                ]["sha256"],
            },
        },
        "candidate": {
            "sha256": CANDIDATE_SHA256["P1"],
            "rows": 169011,
            "reproduced_byte_identical": True,
            "key_order_valid": True,
        },
        "failure_resume": (
            "Initial full-fit inference stopped because model and inference feature modes differed; "
            "the invalid partial causal model stayed ineligible. One append-only corrective causal fit "
            "completed the ensemble and byte-identical candidate reproduction."
        ),
        "frozen_sha256": FROZEN_SHA256["P1"],
        "frozen_unchanged": True,
        "upload_count": 0,
        "independent_qa": "QA_PASS P0=0 P1=0",
    }


def _p2_record(evidence: dict[str, Any]) -> dict[str, Any]:
    metrics = evidence["p2_metrics"]
    manifest = evidence["p2_manifest"]
    completion = evidence["p2_completion"]
    correction = evidence["p2_timestamp_correction"]
    addendum = evidence["p2_addendum_seal"]
    _require(metrics["experiment_id"] == manifest["experiment_id"], "P2 experiment drifted")
    _require(completion["experiment_id"] == manifest["experiment_id"], "P2 completion drifted")
    _require(
        metrics["research_only"] is True and metrics["adaptive_research"] is True,
        "P2 scope drifted",
    )
    _require(metrics["fresh_holdout_claimed"] is False, "P2 holdout claim drifted")
    _require(manifest["frozen_submission_snapshot_unchanged"] is True, "P2 frozen flag drifted")
    _require(
        manifest["upload_allowed"] is False and manifest["upload_performed"] is False,
        "P2 upload drifted",
    )
    _require(completion["same_original_attempt"] == 1, "P2 attempt identity drifted")
    _require(completion["new_generation_or_attempt"] is False, "P2 generation drifted")
    _require(completion["status"] == "complete", "P2 completion status drifted")
    _require(completion["correction_model_refits"] == 0, "P2 correction refit drifted")
    _require(completion["upload_performed"] is False, "P2 completion upload drifted")
    _require(completion["candidate"]["sha256"] == CANDIDATE_SHA256["P2"], "P2 candidate drifted")
    _require(completion["candidate"]["byte_identical"] is True, "P2 reproduction drifted")
    _require(
        completion["final_stack_model"]["sha256"]
        == manifest["final_saved_model_checks"]["stack_model_sha256"],
        "P2 model SHA drifted",
    )
    _require(completion["final_stack_model"]["load_check"] == "PASS", "P2 load check drifted")
    _require(
        correction["behavior_changes"] == correction["metric_changes"] == 0,
        "P2 metadata correction changed behavior",
    )
    _require(
        correction["candidate_changes"] == correction["model_refits"] == 0,
        "P2 metadata correction changed candidate",
    )
    _require(correction["upload_performed"] is False, "P2 correction upload drifted")
    _require(
        addendum["timestamp_correction_sha256"] == EXPECTED_SHA256["p2_timestamp_correction"],
        "P2 addendum pin drifted",
    )
    _require(
        addendum["behavior_changes"] == addendum["metric_changes"] == 0,
        "P2 addendum behavior drifted",
    )
    winner = metrics["winner"]
    predecessor = metrics["predecessor"]["outer_candidate"]
    _require(winner["id"] == "STACK_W0625", "P2 winner drifted")
    _require(winner["guard"]["eligible"] is True, "P2 winner gate drifted")
    baseline_value = float(predecessor["fold_equal_official_layer_weighted_rmse_c"])
    candidate_value = float(winner["outer"]["fold_equal_official_layer_weighted_rmse_c"])
    delta = float(winner["winner_vs_predecessor_bootstrap"]["delta_rmse_c"])
    _close(baseline_value, 1.1158878559665548, "P2 predecessor RMSE")
    _close(candidate_value, 1.042512377552349, "P2 winner RMSE")
    _close(delta, -0.07337547841420577, "P2 delta RMSE")
    ci90 = [float(value) for value in winner["winner_vs_predecessor_bootstrap"]["delta_interval"]]
    _require(ci90[1] < 0, "P2 CI must exclude zero on the improvement side")
    fold_labels = [
        ("2024 Sep–Oct", "outer_2024_sep_oct"),
        ("2025 May–Jun", "outer_2025_may_jun"),
        ("2025 Jul–Aug", "outer_2025_jul_aug"),
    ]
    rows = [
        {
            "sequence": 1,
            "scope": "Fold-equal aggregate",
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
            "signed_delta_label": f"{delta:+.6f}°C",
            "direction": "improved (Δ<0)",
        }
    ]
    for sequence, (label, key) in enumerate(fold_labels, start=2):
        fold_delta = float(winner["guard"]["outer_fold_delta_vs_predecessor_c"][key])
        rows.append(
            {
                "sequence": sequence,
                "scope": label,
                "baseline": float(predecessor["by_fold"][key]["official_layer_weighted_rmse_c"]),
                "candidate": float(
                    winner["outer"]["by_fold"][key]["official_layer_weighted_rmse_c"]
                ),
                "delta": fold_delta,
                "signed_delta_label": f"{fold_delta:+.6f}°C",
                "direction": "improved (Δ<0)" if fold_delta < 0 else "worsened (Δ>0)",
            }
        )
    _require(sum(row["delta"] < 0 for row in rows[1:]) == 2, "P2 improved-fold count drifted")
    return {
        "problem": "P2",
        "metric": "fold-equal official-layer-weighted RMSE",
        "unit": "°C",
        "better_direction": "lower",
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta_candidate_minus_baseline": delta,
        "ci90": ci90,
        "ci90_includes_zero": False,
        "uncertainty_interpretation": "statistically strong local gain; CI90 excludes zero",
        "validation_surface": {"rows": int(winner["outer"]["rows"]), "folds": 3},
        "chart_rows": rows,
        "winner": "STACK_W0625; candidate weight 0.625",
        "full_fit": {
            "status": "actual final stack model complete; load check PASS",
            "underlying_lightgbm_fits": int(
                completion["initial_underlying_lightgbm_estimator_fits"]
            ),
            "correction_refits": 0,
            "model_sha256": {"final_stack": completion["final_stack_model"]["sha256"]},
        },
        "candidate": {
            "sha256": CANDIDATE_SHA256["P2"],
            "rows": int(manifest["candidate"]["validation"]["rows"]),
            "reproduced_byte_identical": True,
            "key_order_valid": True,
        },
        "failure_resume": (
            "A Windows text-mode descriptor expanded LF bytes in binary artifacts after metrics and "
            "candidate creation. The same locked attempt repaired pinned bytes to new append-only paths; "
            "there were zero refits and zero branch, weight, feature, fold, threshold, or metric changes. "
            "A sealed addendum separately corrects a non-authoritative rounded timestamp."
        ),
        "research_caveat": (
            "Adaptive research with no fresh holdout; the inner aggregate remains worse than its baseline "
            f"by {float(winner['guard']['inner_excess_over_baseline_c']):+.9f}°C and one outer fold worsens."
        ),
        "frozen_sha256": FROZEN_SHA256["P2"],
        "frozen_unchanged": True,
        "upload_count": 0,
        "independent_qa": "message-only final PASS P0=0 P1=0",
    }


def _p3_record(evidence: dict[str, Any]) -> dict[str, Any]:
    metrics = evidence["p3_metrics"]
    manifest = evidence["p3_manifest"]
    completion = evidence["p3_completion"]
    qa = evidence["p3_qa"]
    expected_status = "ACTUAL_FULL_REFIT_SAME_ATTEMPT_RESUMED_CANDIDATE_CREATED_NOT_UPLOADED"
    _require(metrics["status"] == manifest["status"] == expected_status, "P3 status drifted")
    _require(metrics["experiment_id"] == manifest["experiment_id"], "P3 experiment drifted")
    _require(manifest["append_only_generation"] is True, "P3 append-only flag drifted")
    _require(manifest["same_attempt_resume"] is True, "P3 resume identity drifted")
    _require(manifest["input_unchanged"] is True, "P3 input mutation drifted")
    _require(manifest["candidate_created"] is True, "P3 candidate state drifted")
    _require(manifest["candidate_uploaded"] is False, "P3 upload state drifted")
    _require(manifest["test_target_or_hidden_labels_used"] == 0, "P3 hidden-label access drifted")
    _require(
        manifest["absolute_test_timestamp_recovered"] is False, "P3 timestamp recovery drifted"
    )
    _require(manifest["current_or_frozen_mutated"] is False, "P3 frozen mutation drifted")
    before = manifest["input_sha256_before"]["frozen_current"]
    after = manifest["input_sha256_after"]["frozen_current"]
    _require(before == after == FROZEN_SHA256["P3"], "P3 frozen SHA drifted")
    _require(completion["status"] == "SAME_ATTEMPT_RESUME_COMPLETE", "P3 completion drifted")
    _require(
        completion["load_only"] is True and completion["fit_count"] == 0, "P3 resume refit drifted"
    )
    _require(completion["candidate_sha256"] == CANDIDATE_SHA256["P3"], "P3 candidate drifted")
    _require(completion["candidate_reproduction_byte_identical"] is True, "P3 reproduction drifted")
    _require(completion["uploaded"] is False, "P3 completion upload drifted")
    _require(qa["status"] == "PASS_P0_0_P1_0", "P3 QA verdict drifted")
    _require(qa["p0_count"] == qa["p1_count"] == 0, "P3 QA findings")
    _require(qa["checks"]["metrics_sha256"] == EXPECTED_SHA256["p3_metrics"], "P3 QA metric pin")
    _require(
        qa["checks"]["manifest_sha256"] == EXPECTED_SHA256["p3_manifest"], "P3 QA manifest pin"
    )
    evaluation = metrics["evaluation"]
    baseline_value = float(evaluation["incumbent"]["rmse_m"])
    candidate_value = float(evaluation["candidate"]["rmse_m"])
    delta = float(evaluation["delta_candidate_minus_incumbent_m"])
    _close(baseline_value, 0.7791048399763751, "P3 incumbent RMSE")
    _close(candidate_value, 0.7786608799293823, "P3 candidate RMSE")
    _close(delta, -0.00044396004699287506, "P3 delta RMSE")
    ci90 = [
        float(value)
        for value in evaluation["paired_case_bootstrap"]["ci90_delta_candidate_minus_incumbent_m"]
    ]
    _require(ci90[0] < 0 < ci90[1], "P3 CI must include zero")
    fold_labels = [
        ("2024 H2 storm", "2024_h2_storm"),
        ("Winter transition", "winter_transition"),
        ("2025 H1", "2025_h1"),
    ]
    rows = [
        {
            "sequence": 1,
            "scope": "Pooled corrected OOF",
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
            "signed_delta_label": f"{delta:+.7f}m",
            "direction": "improved (Δ<0)",
        }
    ]
    for sequence, (label, key) in enumerate(fold_labels, start=2):
        fold_delta = float(evaluation["fold_delta_m"][key])
        rows.append(
            {
                "sequence": sequence,
                "scope": label,
                "baseline": float(evaluation["incumbent"]["by_fold"][key]),
                "candidate": float(evaluation["candidate"]["by_fold"][key]),
                "delta": fold_delta,
                "signed_delta_label": f"{fold_delta:+.7f}m",
                "direction": "improved (Δ<0)" if fold_delta < 0 else "worsened (Δ>0)",
            }
        )
    _require(sum(row["delta"] < 0 for row in rows[1:]) == 2, "P3 improved-fold count drifted")
    receipt = metrics["training_receipt"]
    counts = receipt["original_attempt_actual_fit_counts"]
    _require(
        counts == {"single_catboost": 1, "multi_catboost": 1, "router": 1}, "P3 fit counts drifted"
    )
    _require(
        receipt["resume_fit_counts"] == {"single_catboost": 0, "multi_catboost": 0, "router": 0},
        "P3 resume counts drifted",
    )
    validation = metrics["candidate_validation"]
    _require(validation["candidate_sha256"] == CANDIDATE_SHA256["P3"], "P3 validation SHA drifted")
    _require(
        validation["fresh_saved_model_reproduction_byte_identical"] is True,
        "P3 model reproduction drifted",
    )
    _require(validation["rows"] == 1200 and validation["cases"] == 200, "P3 candidate size drifted")
    _require(
        validation["key_order_exact"] is True and validation["finite"] is True,
        "P3 candidate validation drifted",
    )
    _require(
        metrics["access_counters_total_attempt"]["upload_attempts"] == 0,
        "P3 upload counter drifted",
    )
    return {
        "problem": "P3",
        "metric": "significant-wave-height (hs) RMSE",
        "unit": "m",
        "better_direction": "lower",
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta_candidate_minus_baseline": delta,
        "ci90": ci90,
        "ci90_includes_zero": True,
        "uncertainty_interpretation": "descriptive small local gain; CI90 includes zero",
        "validation_surface": {
            "rows": int(evaluation["surface"]["rows"]),
            "cases": int(evaluation["surface"]["cases"]),
            "folds": 3,
        },
        "chart_rows": rows,
        "winner": "fixed long-horizon shrink structure",
        "full_fit": {
            "status": "actual full refit complete; same-attempt load-only resume",
            "training_anchor_count": int(receipt["full_training_anchor_count"]),
            "training_rows": int(receipt["single_training_rows"]),
            "feature_count": int(receipt["feature_count"]),
            "model_sha256": dict(receipt["fresh_model_sha256"]),
        },
        "candidate": {
            "sha256": CANDIDATE_SHA256["P3"],
            "rows": 1200,
            "cases": 200,
            "reproduced_byte_identical": True,
            "key_order_valid": True,
            "finite_and_range_valid": True,
        },
        "failure_resume": (
            "The actual fit completed before a post-fit prior-model 1e-12 equality guard stopped the "
            "runner. A same-attempt, load-only resume performed zero fits, preserved the fixed winner "
            "structure and weight, and produced the byte-identical candidate."
        ),
        "frozen_sha256": FROZEN_SHA256["P3"],
        "frozen_unchanged": True,
        "upload_count": 0,
        "independent_qa": "PASS_P0_0_P1_0",
    }


def build_registry(
    evidence: dict[str, Any], *, generated_at: str, allow_pending_qa: bool = False
) -> dict[str, Any]:
    _validate_qa_attestation(allow_pending=allow_pending_qa)
    records = {
        "P1": _p1_record(evidence),
        "P2": _p2_record(evidence),
        "P3": _p3_record(evidence),
    }
    registry = {
        "schema_version": "full_improvement_cycle.registry.v1",
        "generated_at_kst": generated_at,
        "status": (
            "ONE_CYCLE_ACTUAL_IMPROVEMENT_MODELS_FULL_FIT_CANDIDATES_CREATED__"
            "LOCAL_UNCERTAINTY_MIXED__FROZEN_UNCHANGED__UPLOAD_0"
        ),
        "scope": {
            "claim": "actual local validation improvement, actual full fit, and validated candidate creation for P1/P2/P3 in one cycle",
            "not_claimed": [
                "official hidden-test improvement",
                "automatic promotion to an official candidate pool",
                "submission upload",
            ],
            "raw_training_oof_test_submission_rows_read_by_builder": 0,
        },
        "source_pins": [
            {
                "id": source_id,
                "path": RELATIVE_PATHS[source_id].as_posix(),
                "sha256": evidence["hashes"][source_id],
                "kind": "aggregate_json_receipt",
            }
            for source_id in RELATIVE_PATHS
        ],
        "independent_qa_attestation": dict(FULL_CYCLE_QA_ATTESTATION),
        "problems": records,
        "cross_problem_controls": {
            "frozen_sha256": dict(FROZEN_SHA256),
            "all_frozen_or_current_unchanged": True,
            "submission_uploads": 0,
            "official_pool_auto_promotions": 0,
            "candidate_count": 3,
            "actual_full_fit_count": 3,
            "ci90_excludes_zero_problem_count": 1,
            "ci90_excludes_zero_problems": ["P2"],
            "cross_problem_metric_chart_forbidden": True,
            "reason": "F1, °C RMSE, and significant-wave-height RMSE in m are not commensurate.",
        },
    }
    _validate_registry(registry, allow_pending_qa=allow_pending_qa)
    return registry


def _validate_registry(registry: dict[str, Any], *, allow_pending_qa: bool = False) -> None:
    _validate_qa_attestation(allow_pending=allow_pending_qa)
    _require(
        registry["schema_version"] == "full_improvement_cycle.registry.v1",
        "registry schema drifted",
    )
    _require(list(registry["problems"]) == ["P1", "P2", "P3"], "problem order drifted")
    _require(
        registry["cross_problem_controls"]["submission_uploads"] == 0, "registry upload drifted"
    )
    _require(
        registry["cross_problem_controls"]["actual_full_fit_count"] == 3, "full-fit count drifted"
    )
    _require(
        registry["cross_problem_controls"]["ci90_excludes_zero_problems"] == ["P2"],
        "CI classification drifted",
    )
    for problem, record in registry["problems"].items():
        _require(
            record["candidate"]["sha256"] == CANDIDATE_SHA256[problem],
            f"{problem} candidate registry drifted",
        )
        _require(
            record["frozen_sha256"] == FROZEN_SHA256[problem], f"{problem} frozen registry drifted"
        )
        _require(record["frozen_unchanged"] is True, f"{problem} frozen state drifted")
        _require(record["upload_count"] == 0, f"{problem} upload registry drifted")
        _require(len(record["chart_rows"]) == 4, f"{problem} chart row count drifted")
    _require(registry["problems"]["P1"]["ci90_includes_zero"] is True, "P1 CI flag drifted")
    _require(registry["problems"]["P2"]["ci90_includes_zero"] is False, "P2 CI flag drifted")
    _require(registry["problems"]["P3"]["ci90_includes_zero"] is True, "P3 CI flag drifted")
    serialized = json.dumps(registry, ensure_ascii=False)
    for forbidden in ("C:/Users/", "C:\\Users\\", "api_key", "access_token", "password"):
        _require(
            forbidden.lower() not in serialized.lower(), f"unsafe registry content: {forbidden}"
        )


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), "non-finite SQL value")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _rows_to_union_sql(rows: list[dict[str, Any]], fields: list[str]) -> str:
    statements = []
    for index, row in enumerate(rows):
        values = [_sql_literal(row[field]) for field in fields]
        if index == 0:
            projection = ", ".join(
                f'{value} AS "{field}"' for value, field in zip(values, fields, strict=True)
            )
        else:
            projection = ", ".join(values)
        statements.append(f"SELECT {projection}")
    return "\nUNION ALL\n".join(statements)


def _source_specs(registry_sha: str) -> list[dict[str, Any]]:
    specs = [
        {
            "id": "full_cycle_registry",
            "label": "Sealed full-improvement-cycle registry",
            "path": DEFAULT_REGISTRY.as_posix(),
            "sha256": registry_sha,
        }
    ]
    specs.extend(
        {
            "id": source_id,
            "label": source_id.replace("_", " ").upper(),
            "path": path.as_posix(),
            "sha256": EXPECTED_SHA256[source_id],
        }
        for source_id, path in RELATIVE_PATHS.items()
    )
    specs.extend(
        [
            {
                "id": "full_cycle_independent_qa_message",
                "label": "Independent full-cycle QA attestation",
                "note": (
                    f"Message-only {FULL_CYCLE_QA_ATTESTATION['verdict']}; P0=0, P1=0, mismatch=0; "
                    "no receipt was created or invented. P1 caveat: "
                    f"{FULL_CYCLE_QA_ATTESTATION['p1_strict_clock_caveat']}"
                ),
            },
            {
                "id": "method_note",
                "label": "Report delivery and visualization method",
                "note": (
                    "MCP report tools were unavailable, so this uses the technical portable HTML fallback. "
                    "Three separate signed horizontal bars preserve incompatible units; CI90 is stated in "
                    "each chart subtitle and adjacent technical text, not simulated as unsupported error bars. "
                    "The builder reads aggregate JSON only and retains no raw rows."
                ),
            },
        ]
    )
    return specs


def _summary_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sequence, problem in enumerate(("P1", "P2", "P3"), start=1):
        record = registry["problems"][problem]
        rows.append(
            {
                "sequence": sequence,
                "problem": problem,
                "metric_unit": f"{record['metric']} ({record['unit']})",
                "baseline": record["baseline_value"],
                "candidate": record["candidate_value"],
                "signed_delta": record["delta_candidate_minus_baseline"],
                "ci90": f"[{record['ci90'][0]:+.10f}, {record['ci90'][1]:+.10f}]",
                "ci_zero": "includes zero" if record["ci90_includes_zero"] else "excludes zero",
                "evidence": record["uncertainty_interpretation"],
                "full_fit": record["full_fit"]["status"],
                "candidate_sha256": record["candidate"]["sha256"],
                "frozen_unchanged": "yes",
                "uploads": 0,
            }
        )
    return rows


def _chart(problem: str, record: dict[str, Any], source_id: str) -> dict[str, Any]:
    unit = record["unit"]
    positive_is_improvement = record["better_direction"] == "higher"
    direction = "positive=improvement" if positive_is_improvement else "negative=improvement"
    ci_state = "includes 0" if record["ci90_includes_zero"] else "excludes 0"
    ci = record["ci90"]
    delta_label = "ΔF1" if problem == "P1" else f"ΔRMSE ({unit})"
    subtitle = (
        f"{unit}; candidate − baseline; {direction}; aggregate + 3 folds; "
        f"CI90 [{ci[0]:+.10f}, {ci[1]:+.10f}] {ci_state}"
    )
    rows = record["chart_rows"]
    fields = [
        "sequence",
        "scope",
        "baseline",
        "candidate",
        "delta",
        "signed_delta_label",
        "direction",
    ]
    return {
        "id": f"{problem.lower()}_signed_validation_delta",
        "title": f"{problem} signed local validation comparison",
        "subtitle": subtitle,
        "showDescription": True,
        "intent": "comparison",
        "question": f"{problem} point improvement is distributed how across aggregate and folds?",
        "rationale": (
            "A horizontal signed bar is appropriate for one aggregate plus three folds in one unit. "
            "Exact labels and direction text make the comparison readable without color; the neutral "
            "zero line marks no change. CI is textual because this chart has no reviewed error-bar contract."
        ),
        "type": "horizontalBar",
        "dataset": f"{problem.lower()}_signed_validation_delta",
        "sourceId": source_id,
        "source": {
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": _rows_to_union_sql(rows, fields),
                "description": f"Deterministic aggregate projection of {problem} validation metrics.",
                "tables_used": [RELATIVE_PATHS[source_id].as_posix()],
                "filters": [
                    "candidate minus baseline",
                    "aggregate plus three folds",
                    "raw rows excluded",
                ],
                "metric_definitions": {
                    "delta": f"candidate minus baseline in {unit}; {direction}",
                    "ci90": f"[{ci[0]}, {ci[1]}]; {ci_state}",
                },
            }
        },
        "valueFormat": "number",
        "unit": unit,
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
                "field": "delta",
                "type": "quantitative",
                "label": delta_label,
                "format": "number",
            },
            "color": {"field": "direction", "type": "nominal", "label": "Direction"},
            "label": {"field": "signed_delta_label", "type": "nominal", "label": "Signed delta"},
            "tooltip": [
                {"field": "baseline", "type": "quantitative", "label": "Baseline", "unit": unit},
                {"field": "candidate", "type": "quantitative", "label": "Candidate", "unit": unit},
                {"field": "direction", "type": "nominal", "label": "Interpretation"},
            ],
        },
    }


def build_artifact(
    registry: dict[str, Any], *, generated_at: str, allow_pending_qa: bool = False
) -> dict[str, Any]:
    _validate_registry(registry, allow_pending_qa=allow_pending_qa)
    registry_sha = hashlib.sha256(_canonical_bytes(registry)).hexdigest()
    sources = _source_specs(registry_sha)
    records = registry["problems"]
    summary_rows = _summary_rows(registry)
    charts = [
        _chart("P1", records["P1"], "p1_metrics"),
        _chart("P2", records["P2"], "p2_metrics"),
        _chart("P3", records["P3"], "p3_metrics"),
    ]
    table_fields = [
        "sequence",
        "problem",
        "metric_unit",
        "baseline",
        "candidate",
        "signed_delta",
        "ci90",
        "ci_zero",
        "evidence",
        "full_fit",
        "candidate_sha256",
        "frozen_unchanged",
        "uploads",
    ]
    tables = [
        {
            "id": "full_cycle_exact_registry",
            "title": "Exact full-cycle result registry",
            "subtitle": "Unit-specific metrics, uncertainty, full-fit state, candidate identity, and controls",
            "dataset": "full_cycle_exact_registry",
            "sourceId": "full_cycle_registry",
            "source": {
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": _rows_to_union_sql(summary_rows, table_fields),
                    "description": "Exact three-problem projection from the sealed aggregate registry.",
                    "tables_used": [DEFAULT_REGISTRY.as_posix()],
                    "filters": ["P1", "P2", "P3", "raw rows excluded"],
                    "metric_definitions": {
                        "signed_delta": "candidate minus baseline in the unit named by metric_unit",
                        "ci90": "problem-specific paired/bootstrap 90% interval",
                        "uploads": "submission uploads performed in this cycle",
                    },
                }
            },
            "density": "dense",
            "defaultSort": {"field": "problem", "direction": "asc"},
            "columns": [
                {"field": "problem", "label": "Problem", "type": "text"},
                {"field": "metric_unit", "label": "Metric (unit)", "type": "text"},
                {"field": "baseline", "label": "Baseline", "type": "number"},
                {"field": "candidate", "label": "Candidate", "type": "number"},
                {"field": "signed_delta", "label": "Candidate − baseline", "type": "number"},
                {"field": "ci90", "label": "CI90", "type": "text"},
                {"field": "ci_zero", "label": "Zero relation", "type": "text"},
                {"field": "evidence", "label": "Evidence strength", "type": "text"},
                {"field": "full_fit", "label": "Full fit", "type": "text"},
                {"field": "candidate_sha256", "label": "Candidate SHA-256", "type": "text"},
                {"field": "frozen_unchanged", "label": "Frozen unchanged", "type": "text"},
                {"field": "uploads", "label": "Uploads", "type": "number"},
            ],
        }
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "full_cycle_registry",
            "body": (
                "## Technical Summary\n\n"
                "요청한 한 사이클의 산출 범위는 충족됐다: P1·P2·P3 모두 local validation에서 "
                "점추정 개선을 보인 구조를 실제 full fit했고, key/order·finite/range 및 byte-level "
                "reproduction을 통과한 새 candidate를 만들었다. 기존 frozen/current 세 파일은 SHA-256 "
                "기준 불변이며 upload는 0이다. 다만 근거 강도는 같지 않다. P2의 ΔRMSE CI90만 0을 "
                "배제한다; P1과 P3의 CI90은 0을 포함하므로 그 개선은 corrected local evidence이지 "
                "official hidden-test 보장이 아니다."
            ),
        },
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "full_cycle_registry",
            "body": (
                "## Key Findings\n\n"
                "- **P1:** micro F1 0.8603708380 → 0.8609416446, Δ +0.0005708065. "
                "KST-day bootstrap CI90 [-0.0010006259, +0.0019853049]로 0을 포함한다.\n"
                "  Event-protected fold assignment의 Q3 I-ORS layer-1 positive event tail 119 rows"
                "(2025-10-01 00:00–19:40 KST)가 global Q4 earliest timestamp 이후까지 남으므로 "
                "globally strict wall-clock-earlier labels라고 부를 수 없다. 다만 station-layer별 chronology와 "
                "fold order는 안전하고, 119 rows를 calibration에서 뺀 counterfactual에서도 Q4 params "
                "0.75/0.15와 Q4 predictions는 불변이었다.\n"
                "- **P2:** fold-equal official-layer-weighted RMSE 1.1158878560 → 1.0425123776°C, "
                "Δ -0.0733754784°C. CI90 [-0.1016906444, -0.0446891572]로 세 문제 중 유일하게 "
                "0을 배제한다. 다만 inner aggregate는 baseline보다 +0.0276402153°C 열세이고 outer 한 "
                "fold는 +0.0770355591°C 악화했다.\n"
                "- **P3:** 유의파고(hs) RMSE 0.7791048400 → 0.7786608799m, Δ -0.0004439600m. "
                "case-bootstrap CI90 [-0.0020481402, +0.0011163551]로 0을 포함한다.\n\n"
                "아래 세 chart는 동일한 읽기 질문(전체 효과와 fold 일관성)에 반복 bar family를 쓰되, "
                "F1/°C/m를 절대 한 축에 섞지 않는다. 각 bar에는 signed exact label과 방향 문구가 "
                "있고 neutral zero line이 no-change를 표시한다."
            ),
        },
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": "full_cycle_registry",
            "body": (
                "## Scope, Data, and Metric Definitions\n\n"
                "이 보고서는 최종 aggregate metrics, result/manifest/completion/failure·provenance receipts, "
                "독립 QA만 사용한다. builder는 raw train/OOF/test/submission 행을 읽지 않았다. P1은 outer "
                "micro F1(클수록 좋음), P2는 세 outer fold의 official-layer-weighted MSE를 fold-equal로 "
                "평균한 뒤 제곱근을 취한 RMSE(°C, 작을수록 좋음), P3는 181 complete six-lead case의 "
                "유의파고(hs) RMSE(m, 작을수록 좋음)다. 모든 Δ는 candidate − baseline이다. 후보 생성은 "
                "official pool 승격이나 upload를 뜻하지 않는다."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "full_cycle_registry",
            "body": (
                "## Methodology, Model, and Validation Design\n\n"
                "P1은 421,032-row repeated walk-forward outer surface에서 causal event rescue ensemble을 "
                "검증한 뒤 776,706 training rows로 corrected causal LightGBM과 ensemble을 full fit했다. "
                "P2는 세 conservative stack weight 중 locked guard를 통과한 STACK_W0625를 선택하고 "
                "14 underlying LightGBM fits의 final stack model을 저장·load 검증했다. P3는 corrected "
                "181-case/1,086-row surface에서 fixed long-horizon shrink 구조를 평가하고 24,360 anchors, "
                "146,160 training rows, 591 features로 single/multi CatBoost와 router를 실제 refit했다. "
                "각 candidate는 원래 test key/order와 동일하고 byte-identical reproduction을 통과했다."
            ),
        },
        {
            "id": "operation_failures_resume",
            "type": "markdown",
            "sourceId": "full_cycle_registry",
            "body": (
                "## Operation Failures and Resume Integrity\n\n"
                "P1은 model/inference feature-mode 불일치로 첫 full-fit inference가 멈췄고 invalid partial "
                "모델을 ineligible로 보존한 뒤 append-only corrective causal fit 1회로 완료했다. P2는 "
                "Windows text-mode descriptor가 binary LF를 CRLF로 확장한 serialization 오류였으며, "
                "same locked attempt에서 새 append-only 경로로 raw-byte repair했다; refit과 branch/weight/"
                "metric 변경은 0이다. 비권위 rounded timestamp는 별도 sealed addendum으로만 정정했다. "
                "P3는 실제 fit 완료 후 prior-model equality guard에서 멈췄고 same-attempt load-only resume "
                "(fit 0)로 candidate를 만들었다. 어느 경로도 frozen/current를 수정하거나 upload하지 않았다."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "full_cycle_independent_qa_message",
            "body": (
                "## Limitations, Uncertainty, and Robustness Checks\n\n"
                "P1과 P3의 90% interval은 0을 포함하므로 작은 positive point estimate를 확정 개선으로 "
                "표현하지 않는다. P2만 CI90이 0 아래에 있지만 adaptive research이며 fresh holdout이 "
                "아니고 inner aggregate 열세와 한 outer-fold 악화가 남는다. 세 결과 모두 local labeled "
                "surface의 상대 비교다. hidden-test absolute score calibration과 leaderboard 효과는 이 "
                "자료로 식별되지 않는다. P1의 119-row strict-wall-clock boundary는 성능 verdict를 바꾸지 "
                "않지만 validation을 globally strict time split이라고 부르는 것은 금지한다. Independent "
                "full-cycle QA는 message-only attestation이며 이 "
                "보고서는 존재하지 않는 receipt를 만들지 않는다. Portable HTML 검증은 structural "
                "validator 중심이고 실제 브라우저의 clipping/theme/hover는 별도 visual QA 대상이다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "full_cycle_registry",
            "body": (
                "## Recommended Next Steps\n\n"
                "1. 세 candidate를 immutable research registry에 유지하고 frozen/current를 rollback 기준으로 보존한다.\n"
                "2. P1·P3는 새로운 independent labeled surface가 생길 때까지 CI-zero-crossing을 해소할 "
                "evidence를 기다린다.\n"
                "3. P2는 untouched seasonal block에서 Sep–Oct regression과 inner/outer 방향 불일치를 재검증한다.\n"
                "4. Official pool 반영이나 upload는 별도 정책 판단과 사용자 승인 뒤에만 수행한다."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further Questions\n\n"
                "- P1의 개선이 Q3에만 집중된 원인은 event selection인가, calibration margin인가?\n"
                "- P2의 inner 열세와 Sep–Oct outer 악화가 어떤 seasonal shift에서 동시에 나타나는가?\n"
                "- P3의 작은 long-horizon gain을 검증할 truly untouched storm episodes가 추가될 수 있는가?\n"
                "- 세 candidate 중 official score budget을 배분한다면 불확실성·rollback 비용을 어떻게 함께 제한할 것인가?"
            ),
        },
        {
            "id": "source_method_note",
            "type": "markdown",
            "sourceId": "method_note",
            "body": (
                "### Source and delivery note\n\n"
                "MCP report tools were unavailable, so this is a technical portable HTML fallback. "
                "Cross-problem metric charts are intentionally forbidden because F1, °C RMSE, and "
                "유의파고(hs) RMSE(m) are not commensurate."
            ),
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "Technical audit of one full P1–P3 improvement, full-fit, and candidate cycle",
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
                "full_cycle_exact_registry": summary_rows,
                "p1_signed_validation_delta": records["P1"]["chart_rows"],
                "p2_signed_validation_delta": records["P2"]["chart_rows"],
                "p3_signed_validation_delta": records["P3"]["chart_rows"],
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
    _validate_artifact(artifact, registry_sha=registry_sha, allow_pending_qa=allow_pending_qa)
    return artifact


def _validate_artifact(
    artifact: dict[str, Any], *, registry_sha: str, allow_pending_qa: bool = False
) -> None:
    _validate_qa_attestation(allow_pending=allow_pending_qa)
    manifest = artifact["manifest"]
    serialized = json.dumps(artifact, ensure_ascii=False)
    _require(artifact["surface"] == manifest["surface"] == "report", "surface drifted")
    _require(manifest["blocks"][0]["body"] == f"# {REPORT_TITLE}", "title drifted")
    required_roles = (
        "## Technical Summary",
        "## Key Findings",
        "## Scope, Data, and Metric Definitions",
        "## Methodology, Model, and Validation Design",
        "## Limitations, Uncertainty, and Robustness Checks",
        "## Recommended Next Steps",
        "## Further Questions",
    )
    for role in required_roles:
        _require(role in serialized, f"technical report role missing: {role}")
    for phrase in (
        "P2만 CI90이 0 아래",
        "119-row strict-wall-clock boundary",
        "MCP report tools were unavailable",
        "technical portable HTML fallback",
        "유의파고(hs) RMSE(m)",
        "upload는 0",
        CANDIDATE_SHA256["P1"],
        CANDIDATE_SHA256["P2"],
        CANDIDATE_SHA256["P3"],
    ):
        _require(phrase in serialized, f"required phrase missing: {phrase}")
    _require(len(manifest["charts"]) == 3, "exactly three unit-separated charts required")
    expected_units = {
        "p1_signed_validation_delta": "F1",
        "p2_signed_validation_delta": "°C",
        "p3_signed_validation_delta": "m",
    }
    for chart in manifest["charts"]:
        _require(chart["type"] == "horizontalBar", f"chart type drifted: {chart['id']}")
        _require(chart["unit"] == expected_units[chart["id"]], f"chart unit drifted: {chart['id']}")
        _require(chart["settings"]["showValues"] is True, f"exact labels missing: {chart['id']}")
        _require(chart["referenceLines"][0]["value"] == 0, f"zero line missing: {chart['id']}")
        _require(
            chart["referenceLines"][0]["color"] == "neutral",
            f"neutral zero line missing: {chart['id']}",
        )
        _require("CI90" in chart["subtitle"], f"adjacent CI missing: {chart['id']}")
        _require("aggregate + 3 folds" in chart["subtitle"], f"chart grain missing: {chart['id']}")
        rows = artifact["snapshot"]["datasets"][chart["dataset"]]
        _require(len(rows) == 4, f"chart rows drifted: {chart['id']}")
        _require(
            all(row["signed_delta_label"].startswith(("+", "-")) for row in rows),
            f"signed labels drifted: {chart['id']}",
        )
    _require("includes 0" in manifest["charts"][0]["subtitle"], "P1 CI label drifted")
    _require("excludes 0" in manifest["charts"][1]["subtitle"], "P2 CI label drifted")
    _require("includes 0" in manifest["charts"][2]["subtitle"], "P3 CI label drifted")
    _require(len(manifest["tables"]) == 1, "table count drifted")
    table = manifest["tables"][0]
    declared = {column["field"] for column in table["columns"]}
    _require(table["defaultSort"]["field"] in declared, "table defaultSort field undeclared")
    _require(
        len(artifact["snapshot"]["datasets"][table["dataset"]]) == 3, "summary table rows drifted"
    )
    source_ids = {source["id"] for source in manifest["sources"]}
    _require(len(source_ids) == len(manifest["sources"]), "duplicate source IDs")
    for block in manifest["blocks"]:
        if "sourceId" in block:
            _require(block["sourceId"] in source_ids, f"missing block source: {block['id']}")
    for visual in [*manifest["charts"], *manifest["tables"]]:
        _require(visual["sourceId"] in source_ids, f"missing visual source: {visual['id']}")
    registry_source = next(
        source for source in manifest["sources"] if source["id"] == "full_cycle_registry"
    )
    _require(registry_source["sha256"] == registry_sha, "registry source SHA drifted")
    source_projection = [
        {
            "id": source["id"],
            "label": source["label"],
            **({"path": source["path"]} if "path" in source else {}),
            **({"sha256": source["sha256"]} if "sha256" in source else {}),
        }
        for source in manifest["sources"]
    ]
    _require(source_projection == artifact["sources"], "top-level source projection drifted")
    for forbidden in ("C:/Users/", "C:\\Users\\", "api_key", "access_token", "password"):
        _require(
            forbidden.lower() not in serialized.lower(), f"unsafe artifact content: {forbidden}"
        )
    for raw_marker in ("winner_oof.parquet", "submission.csv", "train.csv", "test.csv"):
        _require(raw_marker not in serialized, f"raw-row artifact path leaked: {raw_marker}")


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
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--allow-pending-qa",
        action="store_true",
        help="Validate builder/source contracts before final QA; canonical writes still fail closed.",
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
    _require(
        args.check_only or not args.allow_pending_qa, "pending QA is never allowed on a write path"
    )
    evidence = collect_evidence(root)
    generated_at = args.generated_at or datetime.now(KST).isoformat()
    registry = build_registry(
        evidence, generated_at=generated_at, allow_pending_qa=args.allow_pending_qa
    )
    artifact = build_artifact(
        registry, generated_at=generated_at, allow_pending_qa=args.allow_pending_qa
    )
    registry_sha = hashlib.sha256(_canonical_bytes(registry)).hexdigest()
    artifact_sha = hashlib.sha256(_canonical_bytes(artifact)).hexdigest()
    if args.check_only:
        qa_state = FULL_CYCLE_QA_ATTESTATION["verdict"]
        print(
            f"PASS: validated {len(EXPECTED_SHA256)} aggregate pins, registry {registry_sha}, "
            f"artifact {artifact_sha}, QA={qa_state}; no files written"
        )
        return 0
    _require(
        FULL_CYCLE_QA_ATTESTATION["verdict"] != "PENDING_FINAL", "final QA required before write"
    )
    _require(not registry_path.exists(), f"append-only registry exists: {DEFAULT_REGISTRY}")
    _require(not artifact_path.exists(), f"append-only artifact exists: {DEFAULT_ARTIFACT}")
    _write_new(registry_path, registry)
    _write_new(artifact_path, artifact)
    print(f"PASS: wrote {DEFAULT_REGISTRY.as_posix()} ({registry_sha})")
    print(f"PASS: wrote {DEFAULT_ARTIFACT.as_posix()} ({artifact_sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
