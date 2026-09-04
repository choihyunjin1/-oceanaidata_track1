"""Build the aggregate-only external-data final decision report v2.

This builder is intentionally narrow.  It reads only SHA-pinned aggregate JSON
receipts and hashes (without parsing) the three frozen incumbent submissions.
It never opens observations, row-level OOF predictions, model checkpoints,
test data, or submission values.  The generated artifact keeps each problem's
metric in a separate card because P1 F1, P2 temperature RMSE, and P3 wave-height
RMSE do not share a defensible quantitative axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
REPORT_TITLE = "외부 데이터 최종 판정 — 세 문제 모두 현 동결 제출 유지"
DEFAULT_OUTPUT = Path(
    "reports/generated/external_data_final_decision_v2_2026-08-21_r2/artifact.json"
)

RELATIVE_PATHS = {
    "p1_point": Path(
        "artifacts/p1_iors_external_point_residual_oof_v1/20260821T174744+0900/result.json"
    ),
    "p2_nasa": Path("artifacts/p2_nasa_power_residual_meta_v1/result.json"),
    "p2_era5": Path("artifacts/p2_era5_mixing_gate_actual_v1_run1/result.json"),
    "p2_era5_blind_seal": Path(
        "artifacts/p2_era5_mixing_gate_actual_v1_run1/blind_outer_predictions.seal.json"
    ),
    "p2_era5_qa": Path("artifacts/p2_era5_mixing_gate_actual_v1_qa/qa_receipt.json"),
    "p3_domain": Path("artifacts/p3_kma_domain_shift_gate_v1/result.json"),
    "p3_meta": Path("artifacts/p3_kma_source_prediction_meta_v1/one_shot/result.json"),
    "p3_calibrated": Path("artifacts/p3_kma_calibrated_longlead_blend_v2/one_shot/result.json"),
    "p3_alpha_0p4": Path(
        "artifacts/p3_kma_calibrated_longlead_blend_v2/"
        "posthoc_deployment_consistency_alpha_0p4.json"
    ),
    "p3_alpha_grid": Path(
        "artifacts/p3_kma_calibrated_longlead_blend_v2/posthoc_adaptive_global_alpha_grid_v1.json"
    ),
    "p3_withdrawal": Path("submissions/p3_kma_calibrated_longlead_secondary_v1/DO_NOT_SUBMIT.json"),
    "p1_incumbent": Path("submissions/frozen/P1_FROZEN_READY_TO_UPLOAD_28243fda.csv"),
    "p2_incumbent": Path("submissions/p2/P2_EXTRAPOLATED_SOFT_GATE_V2.csv"),
    "p3_incumbent": Path("submissions/p3_long_persistence_shrink/submission.csv"),
}

EXPECTED_SHA256 = {
    "p1_point": "dd7f720da5c40a63fcba54921b6b503479601514c89e41b780e21ee6749b6e01",
    "p2_nasa": "846a246f6af7fadb0b33be78bb45a46049f92528534c8368c2ab7c14070db75f",
    "p2_era5": "b94fba56dbdc2b2485ca3aabdd91e93b996d51ad00715d39519cf616925dec4d",
    "p2_era5_blind_seal": ("91ccb5c137700d76a0b72afc780ad7d5b75a6ee9f7c94abc02190fc442735c34"),
    "p2_era5_qa": "6b2c31deb3261d59b866941219ee36b3d0e0f4b96e878e4b56fbdf553695cb4b",
    "p3_domain": "cb97bb2e104ceed157705ec920b915c9a1b55d2ee4469d1996e282ccb0c562ca",
    "p3_meta": "e1b1eed9568cc0879a866ca2fbb3954b4437bd3140174c4bb72a683a22edd0c5",
    "p3_calibrated": ("f3b00dcb6a8134148edbdd92d03282472caf4b9b0ba41cbc1b45b2fb362838b9"),
    "p3_alpha_0p4": ("1dc9a1e745a6ad170a8d078d4eed4e3c786a3481df0e11d36acf4367c9cab0fa"),
    "p3_alpha_grid": ("45e004b075cd262daeb6d38d46a59f98da1d77df21c8b1b14e304826038fcc34"),
    "p3_withdrawal": ("6eb84cec7c4a211ba8dc3ec6bfe5b2778dce40f80c962cdbea450371163dbb11"),
    "p1_incumbent": ("28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3"),
    "p2_incumbent": ("1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf"),
    "p3_incumbent": ("d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7"),
}

JSON_EVIDENCE = tuple(
    key for key in RELATIVE_PATHS if key not in {"p1_incumbent", "p2_incumbent", "p3_incumbent"}
)
INCUMBENT_KEYS = ("p1_incumbent", "p2_incumbent", "p3_incumbent")


class FinalDecisionReportError(RuntimeError):
    """Raised when sealed evidence or the report contract has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalDecisionReportError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FinalDecisionReportError(f"unable to hash sealed evidence: {path}") from exc
    return digest.hexdigest()


def _read_sealed_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing required aggregate evidence: {label}")
    actual = _sha256(path)
    _require(
        actual == expected_sha256,
        f"sealed SHA mismatch for {label}: {actual} != {expected_sha256}",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalDecisionReportError(f"invalid aggregate JSON evidence: {label}") from exc
    _require(isinstance(value, dict), f"aggregate evidence must be an object: {label}")
    return value


def _mapping(value: Any, *, role: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{role} must be an object")
    return value


def _sequence(value: Any, *, role: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{role} must be an array",
    )
    return value


def _number(value: Any, *, role: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{role} drifted")
    numeric = float(value)
    _require(math.isfinite(numeric), f"{role} must be finite")
    return numeric


def _close(value: Any, expected: float, *, role: str, tolerance: float = 1e-15) -> float:
    numeric = _number(value, role=role)
    _require(
        math.isclose(numeric, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{role} drifted: {numeric!r} != {expected!r}",
    )
    return numeric


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_p1(result: Mapping[str, Any]) -> dict[str, float]:
    _require(
        result.get("experiment_id") == "p1_iors_external_point_residual_oof_v1",
        "P1 experiment drifted",
    )
    _require(result.get("decision") == "NO_GO_POINT_RESIDUAL", "P1 decision drifted")
    metrics = _mapping(result.get("metrics"), role="P1 metrics")
    gate = _mapping(metrics.get("gate"), role="P1 gate")
    diagnostics = _mapping(gate.get("diagnostics"), role="P1 gate diagnostics")
    _require(gate.get("passed") is False, "P1 gate unexpectedly passed")
    _require(metrics.get("official_hidden_test_used") is False, "P1 hidden test was used")
    _require(metrics.get("outer_is_independent_holdout") is False, "P1 independence claim drifted")
    delta = _close(
        diagnostics.get("overall_weighted_f1_delta"),
        -0.04899753214538305,
        role="P1 weighted F1 delta",
    )
    bootstrap = _mapping(metrics.get("paired_block_bootstrap"), role="P1 bootstrap")
    interval = _sequence(bootstrap.get("difference_ci90"), role="P1 CI90")
    _require(len(interval) == 2, "P1 CI90 length drifted")
    ci_low = _close(interval[0], -0.10108725665784712, role="P1 CI90 lower")
    ci_high = _close(interval[1], -0.029788056150546964, role="P1 CI90 upper")
    _close(bootstrap.get("probability_improved"), 0.0, role="P1 P(improved)")
    _require(
        int(_number(bootstrap.get("blocks"), role="P1 bootstrap blocks")) == 3089,
        "P1 block count drifted",
    )
    return {"delta": delta, "ci_low": ci_low, "ci_high": ci_high}


def _validate_p2_nasa(result: Mapping[str, Any]) -> dict[str, float]:
    _require(
        result.get("experiment_id") == "p2_nasa_power_residual_meta_v1",
        "P2 NASA experiment drifted",
    )
    promotion = _mapping(result.get("promotion"), role="P2 NASA promotion")
    _require(promotion.get("passed") is False, "P2 NASA unexpectedly passed")
    _require(
        promotion.get("decision") == "NO_GO_EXTERNAL_METEOROLOGY_FAMILY", "P2 NASA decision drifted"
    )
    metrics = _mapping(result.get("metrics"), role="P2 NASA metrics")
    incremental = _mapping(
        metrics.get("external_incremental_candidate_vs_control"),
        role="P2 NASA incremental metrics",
    )
    delta = _close(incremental.get("delta_rmse"), 0.0, role="P2 NASA delta")
    _require(
        int(_number(incremental.get("rows"), role="P2 NASA rows")) == 69850,
        "P2 NASA row count drifted",
    )
    bootstrap = _mapping(result.get("bootstrap"), role="P2 NASA bootstrap")
    external_bootstrap = _mapping(
        bootstrap.get("external_incremental_candidate_vs_control"),
        role="P2 NASA incremental bootstrap",
    )
    ci_low = _close(external_bootstrap.get("ci90_low"), 0.0, role="P2 NASA CI low")
    ci_high = _close(external_bootstrap.get("ci90_high"), 0.0, role="P2 NASA CI high")
    return {"delta": delta, "ci_low": ci_low, "ci_high": ci_high}


def _validate_p2_era5(
    result: Mapping[str, Any], seal: Mapping[str, Any], qa: Mapping[str, Any]
) -> dict[str, float]:
    _require(
        result.get("experiment_id") == "p2_era5_mixing_gate_actual_v1", "P2 ERA5 experiment drifted"
    )
    _require(
        result.get("decision") == "REJECT_ERA5_INCREMENT_KEEP_CONTROL", "P2 ERA5 decision drifted"
    )
    _require(result.get("submission_created") is False, "P2 ERA5 created submission")
    _require(result.get("uploaded") is False, "P2 ERA5 uploaded")
    _require(result.get("test_reads") == 0, "P2 ERA5 read test")
    _require(result.get("submission_reads_or_writes") == 0, "P2 ERA5 touched submissions")
    _require(result.get("uploads") == 0, "P2 ERA5 upload count drifted")
    metrics = _mapping(result.get("metrics"), role="P2 ERA5 metrics")
    pooled = _mapping(metrics.get("pooled"), role="P2 ERA5 pooled metrics")
    delta = _close(
        pooled.get("delta_rmse"),
        8.348053914808418e-09,
        role="P2 ERA5 pooled delta",
        tolerance=1e-20,
    )
    _require(
        int(_number(pooled.get("rows"), role="P2 ERA5 rows")) == 69850, "P2 ERA5 row count drifted"
    )
    bootstrap = _mapping(result.get("bootstrap"), role="P2 ERA5 bootstrap")
    ci_low = _close(
        bootstrap.get("delta_rmse_ci_lower"),
        2.8345666036599936e-09,
        role="P2 ERA5 CI lower",
        tolerance=1e-20,
    )
    ci_high = _close(
        bootstrap.get("delta_rmse_ci_upper"),
        1.498833469559813e-08,
        role="P2 ERA5 CI upper",
        tolerance=1e-20,
    )
    promotion = _mapping(result.get("promotion"), role="P2 ERA5 promotion")
    _require(promotion.get("promoted") is False, "P2 ERA5 unexpectedly promoted")
    _require(promotion.get("improved_outer_blocks") == 0, "P2 ERA5 improved block count drifted")
    join = _mapping(result.get("era5_join"), role="P2 ERA5 join")
    _close(join.get("join_coverage"), 1.0, role="P2 ERA5 join coverage")
    _require(join.get("feature_count") == 30, "P2 ERA5 feature count drifted")
    _require(join.get("future_era5_rows") == 0, "P2 ERA5 future rows detected")

    _require(seal.get("truth_columns_present") is False, "P2 ERA5 blind seal contains truth")
    _require(seal.get("rows") == 69850, "P2 ERA5 blind seal rows drifted")
    for flag in (
        "parquet_sha256_unchanged_after_reload",
        "reload_schema_equal",
        "reload_dtype_equal",
        "reload_key_order_equal",
        "reload_values_exact_equal",
    ):
        _require(seal.get(flag) is True, f"P2 ERA5 blind seal failed {flag}")
    _require(
        seal.get("current_outer_truth_rows_seen_during_fit") == 0, "P2 ERA5 current-fold truth leak"
    )
    _require(
        seal.get("designated_scoring_truth_open_count_before_seal") == 0,
        "P2 ERA5 scoring truth opened before seal",
    )
    _require(
        result.get("designated_outer_truth_open_count") == 1,
        "P2 ERA5 scoring truth open count drifted",
    )

    _require(qa.get("status") == "passed", "P2 ERA5 QA did not pass")
    _require(qa.get("independent") is True, "P2 ERA5 QA is not independent")
    _require(qa.get("actual_execution_performed") is False, "P2 ERA5 QA executed actual")
    _require(
        qa.get("test_or_submission_access_performed") is False,
        "P2 ERA5 QA accessed test/submission",
    )
    return {"delta": delta, "ci_low": ci_low, "ci_high": ci_high}


def _validate_p3(
    domain: Mapping[str, Any],
    meta: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    alpha: Mapping[str, Any],
    grid: Mapping[str, Any],
    withdrawal: Mapping[str, Any],
) -> dict[str, float]:
    domain_decision = _mapping(domain.get("decision"), role="P3 domain decision")
    _require(
        domain_decision.get("tier") == "no_go_source_concat_or_full_finetune",
        "P3 domain decision drifted",
    )
    domain_classifier = _mapping(domain.get("domain_classifier"), role="P3 domain classifier")
    domain_auc = _close(
        domain_classifier.get("oof_auc"),
        0.9967791897555192,
        role="P3 domain AUC",
    )
    access = _mapping(domain.get("access_audit"), role="P3 domain access audit")
    _require(
        access.get("frozen_model_oof_or_submission_read") is False,
        "P3 domain gate touched frozen artifacts",
    )

    _require(
        meta.get("decision") == "NO_GO_HIGH_AUC_INNER_NO_INCREMENTAL_SIGNAL",
        "P3 meta decision drifted",
    )
    inner_meta = _mapping(meta.get("inner_gate"), role="P3 meta inner gate")
    meta_delta = _close(
        inner_meta.get("pooled_delta_rmse"),
        0.00935173423193969,
        role="P3 meta inner delta",
    )
    _require(inner_meta.get("pass") is False, "P3 meta gate unexpectedly passed")
    _require(meta.get("designated_outer_scoring_open_count") == 0, "P3 meta opened outer scoring")

    _require(
        calibrated.get("decision") == "NO_GO_EXACT_INCUMBENT", "P3 calibrated decision drifted"
    )
    inner = _mapping(calibrated.get("inner_gate"), role="P3 calibrated inner gate")
    _require(inner.get("pass") is True, "P3 calibrated inner gate drifted")
    outer = _mapping(calibrated.get("outer_promotion"), role="P3 outer promotion")
    local_delta = _close(
        outer.get("candidate_minus_incumbent_rmse"),
        -0.0025293726008623896,
        role="P3 fold-alpha local delta",
    )
    paired = _mapping(outer.get("paired_case_bootstrap"), role="P3 outer bootstrap")
    local_ci_low = _close(paired.get("ci90_lower"), -0.008046136259744468, role="P3 local CI lower")
    local_ci_high = _close(paired.get("ci90_upper"), 0.002910861118130509, role="P3 local CI upper")
    _require(outer.get("decision") == "NO_GO_EXACT_INCUMBENT", "P3 outer decision drifted")

    reconstruction = _mapping(alpha.get("reconstruction"), role="P3 alpha reconstruction")
    _close(reconstruction.get("global_deployment_alpha"), 0.4, role="P3 deployment alpha")
    deployed_delta = _close(
        _mapping(alpha.get("candidate_minus_incumbent"), role="P3 deployed delta").get(
            "pooled_rmse"
        ),
        0.0006208732949318785,
        role="P3 deployed alpha delta",
    )
    deployed_bootstrap = _mapping(alpha.get("paired_case_bootstrap"), role="P3 deployed bootstrap")
    deployed_ci_low = _close(
        deployed_bootstrap.get("ci90_lower"),
        -0.005987352810260832,
        role="P3 deployed CI lower",
    )
    deployed_ci_high = _close(
        deployed_bootstrap.get("ci90_upper"),
        0.007159215462548102,
        role="P3 deployed CI upper",
    )
    alpha_scope = _mapping(alpha.get("scope"), role="P3 alpha scope")
    for key in (
        "model_fit_or_refit_count",
        "new_target_shard_or_vault_open_count",
        "test_context_read_count",
        "submission_read_or_write_count",
    ):
        _require(alpha_scope.get(key) == 0, f"P3 alpha diagnostic scope drifted: {key}")

    adaptive = _mapping(grid.get("adaptive_best_on_same_rows"), role="P3 alpha grid best")
    _close(adaptive.get("alpha"), 0.2, role="P3 posthoc alpha")
    posthoc_delta = _close(
        adaptive.get("delta_vs_alpha_0"),
        -0.001088204044226826,
        role="P3 posthoc delta",
    )
    adaptive_bootstrap = _mapping(
        adaptive.get("paired_case_bootstrap"), role="P3 posthoc bootstrap"
    )
    posthoc_ci_low = _close(
        adaptive_bootstrap.get("ci90_lower"),
        -0.004432220722349717,
        role="P3 posthoc CI lower",
    )
    posthoc_ci_high = _close(
        adaptive_bootstrap.get("ci90_upper"),
        0.0021951761590787654,
        role="P3 posthoc CI upper",
    )
    conclusion = _mapping(grid.get("conclusion"), role="P3 alpha grid conclusion")
    _require(
        conclusion.get("any_single_global_alpha_supported_for_promotion") is False,
        "P3 grid unexpectedly supports promotion",
    )
    grid_values = _mapping(grid.get("grid"), role="P3 alpha grid")
    expected_grid = {
        "0.0": (0.7801609198910191, 0.0, 0.0, 0.0),
        "0.1": (
            0.7792671840956024,
            -0.0008937357954167391,
            0.00197258064402539,
            -0.0003584923639932125,
        ),
        "0.2": (
            0.7790727158467923,
            -0.001088204044226826,
            0.00457004407342565,
            0.0009821222013871589,
        ),
        "0.3": (
            0.7795780384483798,
            -0.0005828814426392936,
            0.007786246247941331,
            0.004014256142726436,
        ),
        "0.4": (
            0.780781793185951,
            0.0006208732949318785,
            0.011613690573430868,
            0.008720913859646573,
        ),
    }
    _require(set(grid_values) == set(expected_grid), "P3 alpha grid keys drifted")
    alpha_grid_rows: list[dict[str, Any]] = []
    for alpha_label, expected in expected_grid.items():
        entry = _mapping(grid_values[alpha_label], role=f"P3 alpha {alpha_label}")
        delta = _mapping(entry.get("delta_vs_alpha_0"), role=f"P3 alpha {alpha_label} delta")
        alpha_grid_rows.append(
            {
                "alpha": float(alpha_label),
                "alpha_label": f"α={alpha_label}",
                "pooled_rmse_m": _close(
                    entry.get("pooled_rmse"),
                    expected[0],
                    role=f"P3 alpha {alpha_label} RMSE",
                ),
                "delta_rmse_mm": 1000.0
                * _close(
                    delta.get("pooled_rmse"),
                    expected[1],
                    role=f"P3 alpha {alpha_label} pooled delta",
                ),
                "winter_delta_mm": 1000.0
                * _close(
                    _mapping(
                        delta.get("by_fold"),
                        role=f"P3 alpha {alpha_label} fold delta",
                    ).get("winter_transition"),
                    expected[2],
                    role=f"P3 alpha {alpha_label} winter delta",
                ),
                "lead18_delta_mm": 1000.0
                * _close(
                    _mapping(
                        delta.get("by_lead"),
                        role=f"P3 alpha {alpha_label} lead delta",
                    ).get("18"),
                    expected[3],
                    role=f"P3 alpha {alpha_label} lead 18 delta",
                ),
                "policy_role": (
                    "incumbent"
                    if alpha_label == "0.0"
                    else "posthoc minimum"
                    if alpha_label == "0.2"
                    else "deployed candidate"
                    if alpha_label == "0.4"
                    else "diagnostic"
                ),
            }
        )

    _require(withdrawal.get("receipt_type") == "DO_NOT_SUBMIT", "P3 withdrawal type drifted")
    _require(
        withdrawal.get("decision") == "WITHDRAWN_GLOBAL_DEPLOYMENT_MISMATCH",
        "P3 withdrawal decision drifted",
    )
    _require(withdrawal.get("upload_allowed") is False, "P3 withdrawal allows upload")
    _require(withdrawal.get("upload_performed") is False, "P3 withdrawal reports upload")
    candidate = _mapping(withdrawal.get("candidate"), role="P3 withdrawn candidate")
    _require(
        candidate.get("sha256")
        == "89105c18120a2e260d8467f15b68d8b0724b52d241b8667e361dbecc710f5421",
        "P3 withdrawn candidate SHA drifted",
    )
    _require(candidate.get("status") == "DO_NOT_SUBMIT", "P3 candidate status drifted")
    incumbent = _mapping(withdrawal.get("incumbent_to_keep"), role="P3 incumbent receipt")
    _require(
        incumbent.get("sha256") == EXPECTED_SHA256["p3_incumbent"],
        "P3 incumbent receipt SHA drifted",
    )
    original = _mapping(withdrawal.get("original_generation"), role="P3 original generation")
    v2 = _mapping(original.get("v2_validation_result"), role="P3 v2 withdrawal binding")
    _require(v2.get("sha256") == EXPECTED_SHA256["p3_calibrated"], "P3 v2 result binding drifted")
    mismatch = _mapping(withdrawal.get("deployment_mismatch_evidence"), role="P3 mismatch evidence")
    fixed = _mapping(mismatch.get("fixed_global_alpha_0p4"), role="P3 fixed alpha binding")
    adaptive_receipt = _mapping(mismatch.get("adaptive_global_alpha_grid"), role="P3 grid binding")
    _require(
        fixed.get("sha256") == EXPECTED_SHA256["p3_alpha_0p4"],
        "P3 alpha diagnostic binding drifted",
    )
    _require(
        adaptive_receipt.get("sha256") == EXPECTED_SHA256["p3_alpha_grid"],
        "P3 grid diagnostic binding drifted",
    )
    return {
        "domain_auc": domain_auc,
        "meta_delta": meta_delta,
        "local_delta": local_delta,
        "local_ci_low": local_ci_low,
        "local_ci_high": local_ci_high,
        "deployed_delta": deployed_delta,
        "deployed_ci_low": deployed_ci_low,
        "deployed_ci_high": deployed_ci_high,
        "posthoc_delta": posthoc_delta,
        "posthoc_ci_low": posthoc_ci_low,
        "posthoc_ci_high": posthoc_ci_high,
        "alpha_grid": alpha_grid_rows,
    }


def collect_evidence(root: Path) -> dict[str, Any]:
    root = root.resolve()
    loaded: dict[str, Mapping[str, Any]] = {}
    hashes: dict[str, str] = {}
    for key in JSON_EVIDENCE:
        path = root / RELATIVE_PATHS[key]
        loaded[key] = _read_sealed_json(path, EXPECTED_SHA256[key], key)
        hashes[key] = EXPECTED_SHA256[key]
    for key in INCUMBENT_KEYS:
        path = root / RELATIVE_PATHS[key]
        _require(path.is_file(), f"missing frozen incumbent: {key}")
        actual = _sha256(path)
        _require(
            actual == EXPECTED_SHA256[key],
            f"frozen incumbent SHA mismatch for {key}: {actual} != {EXPECTED_SHA256[key]}",
        )
        hashes[key] = actual

    p1 = _validate_p1(loaded["p1_point"])
    nasa = _validate_p2_nasa(loaded["p2_nasa"])
    era5 = _validate_p2_era5(loaded["p2_era5"], loaded["p2_era5_blind_seal"], loaded["p2_era5_qa"])
    p3 = _validate_p3(
        loaded["p3_domain"],
        loaded["p3_meta"],
        loaded["p3_calibrated"],
        loaded["p3_alpha_0p4"],
        loaded["p3_alpha_grid"],
        loaded["p3_withdrawal"],
    )
    return {"p1": p1, "p2_nasa": nasa, "p2_era5": era5, "p3": p3, "hashes": hashes}


def _source(
    *,
    source_id: str,
    label: str,
    paths: Sequence[Path],
    description: str,
    filters: Sequence[str],
    sql: str,
    metric_definitions: Mapping[str, str],
) -> dict[str, Any]:
    normalized = [path.as_posix() for path in paths]
    return {
        "id": source_id,
        "label": label,
        "path": normalized[0],
        "query": {
            "engine": "sqlite",
            "sql": sql,
            "description": description,
            "tables_used": normalized,
            "filters": list(filters),
            "metric_definitions": dict(metric_definitions),
        },
    }


def _sql_text(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_artifact(evidence: Mapping[str, Any], *, generated_at: str) -> dict[str, Any]:
    p1 = _mapping(evidence.get("p1"), role="P1 report evidence")
    nasa = _mapping(evidence.get("p2_nasa"), role="P2 NASA report evidence")
    era5 = _mapping(evidence.get("p2_era5"), role="P2 ERA5 report evidence")
    p3 = _mapping(evidence.get("p3"), role="P3 report evidence")
    hashes = _mapping(evidence.get("hashes"), role="report evidence hashes")

    p1_delta = _number(p1.get("delta"), role="report P1 delta")
    nasa_delta = _number(nasa.get("delta"), role="report NASA delta")
    era5_delta = _number(era5.get("delta"), role="report ERA5 delta")
    deployed_delta = _number(p3.get("deployed_delta"), role="report P3 deployed delta")
    local_delta = _number(p3.get("local_delta"), role="report P3 local delta")
    posthoc_delta = _number(p3.get("posthoc_delta"), role="report P3 posthoc delta")

    decision_rows = [
        {
            "problem": "P1",
            "hypothesis": "I-ORS external q50 residual 특징으로 eligible incumbent 예측을 교체",
            "local_metric_delta": f"{p1_delta:+.9f} weighted F1 (candidate − incumbent)",
            "uncertainty": (
                f"paired-block CI90 [{p1['ci_low']:+.9f}, {p1['ci_high']:+.9f}]; P(improve)=0"
            ),
            "deployment_consistency": "Q2 exact no-op; Q3·Q4 모두 악화; hidden/test/submission 미사용",
            "decision": "REJECT — NO_GO_POINT_RESIDUAL",
            "current_incumbent": ("P1 frozen XGBoost QC — " + str(hashes["p1_incumbent"])),
        },
        {
            "problem": "P2",
            "hypothesis": "NASA residual-meta와 ERA5 wind-stress/Qnet가 frozen gate를 개선",
            "local_metric_delta": (
                f"NASA {nasa_delta:+.9f} °C; ERA5 {era5_delta:+.12f} °C "
                "(candidate − control; 양수는 악화)"
            ),
            "uncertainty": (
                "NASA CI90 [0, 0]; ERA5 CI90 "
                f"[{era5['ci_low']:+.12f}, {era5['ci_high']:+.12f}] °C; 0/3 blocks improve"
            ),
            "deployment_consistency": "동일 frozen experts·convex weight; test/submission/upload 0회",
            "decision": "REJECT ERA5 INCREMENT — KEEP CONTROL",
            "current_incumbent": ("P2 extrapolated soft-gate v2 — " + str(hashes["p2_incumbent"])),
        },
        {
            "problem": "P3",
            "hypothesis": "KMA source-meta calibrated long-lead blend가 incumbent를 개선",
            "local_metric_delta": (
                f"fold-specific α {local_delta:+.9f} m; deployable global α=0.4 "
                f"{deployed_delta:+.9f} m"
            ),
            "uncertainty": (
                f"local CI90 [{p3['local_ci_low']:+.9f}, {p3['local_ci_high']:+.9f}] m; "
                f"global α=0.4 CI90 [{p3['deployed_ci_low']:+.9f}, {p3['deployed_ci_high']:+.9f}] m; "
                f"posthoc α=0.2 CI90 [{p3['posthoc_ci_low']:+.9f}, {p3['posthoc_ci_high']:+.9f}] m"
            ),
            "deployment_consistency": "fold별 α와 global α 불일치; secondary candidate 철회·업로드 금지",
            "decision": "WITHDRAWN_GLOBAL_DEPLOYMENT_MISMATCH",
            "current_incumbent": (
                "P3 long-lead persistence shrink — " + str(hashes["p3_incumbent"])
            ),
        },
    ]

    evidence_rows = [
        {
            "evidence": key,
            "path": RELATIVE_PATHS[key].as_posix(),
            "sha256": str(hashes[key]),
            "read_mode": "aggregate JSON"
            if key in JSON_EVIDENCE
            else "hash-only; CSV values not parsed",
        }
        for key in RELATIVE_PATHS
    ]

    p1_card_rows = [{"weighted_f1_delta": p1_delta}]
    p2_card_rows = [{"era5_delta_micro_c": era5_delta * 1_000_000.0}]
    p3_card_rows = [{"deployed_delta_mm": deployed_delta * 1_000.0}]
    visual_contract_rows = [
        {
            "problem": "P1",
            "form": "independent delta metric card",
            "unit": "weighted F1",
            "scale": "native",
            "reason": "single decision delta; no defensible shared axis with temperature or wave RMSE",
        },
        {
            "problem": "P2",
            "form": "independent delta metric card",
            "unit": "microdegrees Celsius",
            "scale": "ERA5 °C delta × 1,000,000",
            "reason": "effect is too small for an honest shared axis; exact °C remains in decision table",
        },
        {
            "problem": "P3",
            "form": "independent delta metric card plus same-unit alpha-grid bar",
            "unit": "millimetres",
            "scale": "wave-height RMSE metres delta × 1,000",
            "reason": (
                "deployment-consistent delta is the decision metric; the only chart stays within "
                "P3 millimetres and shows all five audited alpha policies"
            ),
        },
    ]
    decision_columns = (
        "problem",
        "hypothesis",
        "local_metric_delta",
        "uncertainty",
        "deployment_consistency",
        "decision",
        "current_incumbent",
    )
    decision_sql = "\nUNION ALL\n".join(
        "SELECT "
        + ", ".join(f"{_sql_text(row[column])} AS {column}" for column in decision_columns)
        for row in decision_rows
    )
    p3_grid_rows = list(_sequence(p3.get("alpha_grid"), role="P3 alpha grid rows"))
    p3_grid_values_sql = ",\n  ".join(
        "("
        + ", ".join(
            (
                str(row[field])
                if field
                in {
                    "alpha",
                    "pooled_rmse_m",
                    "delta_rmse_mm",
                    "winter_delta_mm",
                    "lead18_delta_mm",
                }
                else _sql_text(row[field])
            )
            for field in (
                "alpha",
                "alpha_label",
                "pooled_rmse_m",
                "delta_rmse_mm",
                "winter_delta_mm",
                "lead18_delta_mm",
                "policy_role",
            )
        )
        + ")"
        for row in p3_grid_rows
    )

    p1_source = _source(
        source_id="p1_point_result",
        label="P1 external point-residual aggregate result",
        paths=[RELATIVE_PATHS["p1_point"]],
        description=(
            "SHA-pinned aggregate nested-OOF metrics only; no prediction rows, model values, "
            "test data, or submission values."
        ),
        filters=["official_hidden_test_used = false", "outer_is_independent_holdout = false"],
        sql=(
            f"SELECT {p1_delta!r} AS weighted_f1_delta, "
            f"{p1['ci_low']!r} AS ci90_lower, {p1['ci_high']!r} AS ci90_upper"
        ),
        metric_definitions={
            "weighted_f1_delta": "candidate minus incumbent weighted F1 over the frozen chronological OOF membership",
            "ci90_lower": "paired block bootstrap 90% lower bound for weighted F1 delta",
            "ci90_upper": "paired block bootstrap 90% upper bound for weighted F1 delta",
        },
    )
    p2_source = _source(
        source_id="p2_external_derivation",
        label="P2 NASA and ERA5 aggregate decisions",
        paths=[
            RELATIVE_PATHS["p2_nasa"],
            RELATIVE_PATHS["p2_era5"],
            RELATIVE_PATHS["p2_era5_blind_seal"],
            RELATIVE_PATHS["p2_era5_qa"],
        ],
        description=(
            "SHA-pinned aggregate NASA and ERA5 results plus blind-seal and independent-QA "
            "receipts; no OOF value pages are included in the report."
        ),
        filters=["69,850 fixed OOF keys", "test/submission/upload access = 0"],
        sql=(
            f"SELECT {nasa_delta!r} AS nasa_delta_rmse_c, "
            f"{era5_delta!r} AS era5_delta_rmse_c, "
            f"{era5_delta * 1_000_000.0!r} AS era5_delta_micro_c, "
            f"{era5['ci_low']!r} AS era5_ci90_lower_c, "
            f"{era5['ci_high']!r} AS era5_ci90_upper_c"
        ),
        metric_definitions={
            "nasa_delta_rmse_c": "NASA candidate minus no-external control pooled RMSE in degrees Celsius",
            "era5_delta_rmse_c": "ERA5 selected gate minus identical control pooled RMSE in degrees Celsius",
            "era5_delta_micro_c": "ERA5 delta RMSE multiplied by one million for a readable card",
        },
    )
    p3_source = _source(
        source_id="p3_deployment_derivation",
        label="P3 KMA signal, calibration, deployment diagnostics, and withdrawal",
        paths=[
            RELATIVE_PATHS["p3_domain"],
            RELATIVE_PATHS["p3_meta"],
            RELATIVE_PATHS["p3_calibrated"],
            RELATIVE_PATHS["p3_alpha_0p4"],
            RELATIVE_PATHS["p3_alpha_grid"],
            RELATIVE_PATHS["p3_withdrawal"],
        ],
        description=(
            "SHA-pinned aggregate P3 domain, meta, calibrated validation, deployment-consistency, "
            "posthoc grid, and immutable withdrawal receipts."
        ),
        filters=["182 cases / 1,092 lead rows", "posthoc alpha grid is not promotion evidence"],
        sql=(
            "WITH alpha_grid(alpha, alpha_label, pooled_rmse_m, delta_rmse_mm, "
            "winter_delta_mm, lead18_delta_mm, policy_role) AS (VALUES\n  "
            + p3_grid_values_sql
            + ")\nSELECT *, "
            + f"{deployed_delta * 1_000.0!r} AS deployed_delta_mm, "
            + f"{local_delta!r} AS fold_specific_delta_m, "
            + f"{posthoc_delta!r} AS posthoc_best_delta_m FROM alpha_grid"
        ),
        metric_definitions={
            "delta_rmse_mm": "global-alpha candidate minus incumbent RMSE in millimetres on the same 182 evaluated cases",
            "deployed_delta_mm": "global alpha 0.4 candidate minus incumbent RMSE in millimetres",
            "fold_specific_delta_m": "original fold-specific-alpha candidate minus incumbent RMSE in metres",
            "posthoc_best_delta_m": "same-row posthoc alpha 0.2 candidate minus incumbent RMSE in metres; not promotion evidence",
        },
    )
    final_source = _source(
        source_id="final_decision_derivation",
        label="Cross-problem aggregate final-decision derivation",
        paths=[RELATIVE_PATHS[key] for key in RELATIVE_PATHS],
        description=(
            "Deterministic report transformation over exact SHA-pinned aggregate receipts and "
            "hash-only incumbent submissions."
        ),
        filters=[
            "raw observation rows = 0",
            "row-level OOF values = 0",
            "submission values parsed = 0",
            "personal paths and credentials = 0",
        ],
        sql=decision_sql,
        metric_definitions={
            "local_metric_delta": "problem-native candidate-minus-incumbent or candidate-minus-control metric with explicit unit and direction",
            "uncertainty": "problem-native paired bootstrap interval and consistency count",
            "deployment_consistency": "whether the validated local policy matches the policy that can be deployed",
            "current_incumbent": "immutable incumbent identity bound to its exact SHA-256",
        },
    )
    sources = [p1_source, p2_source, p3_source, final_source]

    cards = [
        {
            "id": "p1_delta_card",
            "description": "candidate − incumbent; negative weighted F1 means worse",
            "dataset": "p1_delta_card",
            "sourceId": "p1_point_result",
            "metrics": [
                {
                    "label": "P1 weighted F1 delta",
                    "field": "weighted_f1_delta",
                    "format": "number",
                    "signed": True,
                }
            ],
        },
        {
            "id": "p2_delta_card",
            "description": "ERA5 selected − control; +0.00835 µ°C is a tiny regression",
            "dataset": "p2_delta_card",
            "sourceId": "p2_external_derivation",
            "metrics": [
                {
                    "label": "P2 ERA5 ΔRMSE",
                    "field": "era5_delta_micro_c",
                    "format": "number",
                    "unit": "µ°C",
                    "signed": True,
                }
            ],
        },
        {
            "id": "p3_delta_card",
            "description": "deployable global α=0.4 candidate − incumbent; positive means worse",
            "dataset": "p3_delta_card",
            "sourceId": "p3_deployment_derivation",
            "metrics": [
                {
                    "label": "P3 deployed ΔRMSE",
                    "field": "deployed_delta_mm",
                    "format": "number",
                    "unit": "mm",
                    "signed": True,
                }
            ],
        },
    ]

    decision_table = {
        "id": "final_decision_register",
        "title": "문제별 외부 데이터 최종 판정",
        "subtitle": (
            "Delta는 각 문제의 로컬 지표와 단위를 유지합니다. F1과 °C·m RMSE를 한 축으로 비교하지 않습니다."
        ),
        "dataset": "decision_register",
        "sourceId": "final_decision_derivation",
        "defaultSort": {"field": "problem", "direction": "asc"},
        "columns": [
            {"field": "problem", "label": "Problem", "type": "text"},
            {"field": "hypothesis", "label": "Hypothesis", "type": "text"},
            {"field": "local_metric_delta", "label": "Local metric delta", "type": "text"},
            {"field": "uncertainty", "label": "Uncertainty", "type": "text"},
            {
                "field": "deployment_consistency",
                "label": "Deployment consistency",
                "type": "text",
            },
            {"field": "decision", "label": "Decision", "type": "text"},
            {"field": "current_incumbent", "label": "Current incumbent", "type": "text"},
        ],
    }

    alpha_grid_chart = {
        "id": "p3_global_alpha_grid",
        "title": "P3 global alpha grid ΔRMSE",
        "subtitle": (
            "Same 182 evaluated cases; candidate − incumbent in mm, negative improves; "
            "posthoc diagnostic only"
        ),
        "type": "bar",
        "dataset": "p3_alpha_grid",
        "sourceId": "p3_deployment_derivation",
        "valueFormat": "number",
        "encodings": {
            "x": {"field": "alpha_label", "type": "nominal", "label": "Global alpha"},
            "y": {
                "field": "delta_rmse_mm",
                "type": "quantitative",
                "label": "ΔRMSE (mm)",
            },
            "tooltip": [
                {
                    "field": "pooled_rmse_m",
                    "type": "quantitative",
                    "label": "Pooled RMSE (m)",
                },
                {
                    "field": "winter_delta_mm",
                    "type": "quantitative",
                    "label": "Winter Δ (mm)",
                },
                {
                    "field": "lead18_delta_mm",
                    "type": "quantitative",
                    "label": "+18h Δ (mm)",
                },
                {"field": "policy_role", "type": "nominal", "label": "Policy role"},
            ],
        },
    }

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 기술 요약 — 외부 데이터 신호는 있었지만 제출을 바꿀 증거는 없습니다\n\n"
                "- **P1은 기각합니다.** I-ORS external q50 residual 통합은 weighted F1을 "
                f"{p1_delta:+.6f} 낮췄고 paired CI90 전체가 0 아래였습니다.\n"
                "- **P2는 ERA5 증분을 기각하고 기존 control을 유지합니다.** NASA residual-meta는 "
                f"ΔRMSE 0.0°C였고, ERA5 mixing gate는 {era5_delta:+.3e}°C로 미세하게 악화했습니다.\n"
                "- **P3는 KMA 신호를 인정하되 후보를 철회합니다.** fold별 α 후보는 로컬 RMSE를 "
                f"{local_delta:+.6f}m 낮췄지만 CI90이 0을 포함했고, 실제 배포한 global α=0.4는 "
                f"{deployed_delta:+.6f}m 악화했습니다.\n"
                "- **현재 동결 제출 세 개를 그대로 유지합니다.** 보고서는 고정 SHA의 집계 영수증과 "
                "제출 파일의 byte hash만 읽었으며 원자료·OOF 행값·test 값·submission 값은 읽지 않았습니다."
            ),
        },
        {
            "id": "delta_cards_intro",
            "type": "markdown",
            "body": (
                "## 세 delta는 단위를 보존한 독립 카드로 읽어야 합니다\n\n"
                "P1은 F1이 높을수록 좋고 P2·P3 RMSE는 낮을수록 좋습니다. 따라서 공통 축이나 "
                "정규화 순위를 만들지 않았습니다. P2와 P3 카드는 매우 작은 차이를 읽을 수 있도록 각각 "
                "µ°C와 mm로만 재표현하며, exact 원 단위 값은 바로 아래 판정표에 남깁니다."
            ),
        },
        {
            "id": "delta_cards",
            "type": "metric-strip",
            "cardIds": ["p1_delta_card", "p2_delta_card", "p3_delta_card"],
        },
        {
            "id": "p1_finding",
            "type": "markdown",
            "sourceId": "p1_point_result",
            "body": (
                "## P1 external residual 교체는 모든 실질 개선 관문에서 실패했습니다\n\n"
                f"전체 weighted F1 delta는 **{p1_delta:+.9f}**이고 CI90은 "
                f"**[{p1['ci_low']:+.9f}, {p1['ci_high']:+.9f}]**입니다. Q2는 external-eligible "
                "학습 이력이 없어 exact no-op이었고 Q3·Q4는 모두 악화했습니다. 이 결과는 external q50 "
                "표현 자체보다, 그것으로 eligible incumbent를 wholesale replacement하는 결합을 기각합니다."
            ),
        },
        {
            "id": "p2_finding",
            "type": "markdown",
            "sourceId": "p2_external_derivation",
            "body": (
                "## P2 ERA5 물리 변수는 정렬·누출 검증을 통과했지만 점수를 움직이지 못했습니다\n\n"
                f"NASA 증분은 **{nasa_delta:+.9f}°C**, ERA5 wind-stress/Qnet 증분은 "
                f"**{era5_delta:+.12f}°C**입니다. ERA5 CI90은 "
                f"**[{era5['ci_low']:+.12f}, {era5['ci_high']:+.12f}]°C**이고 개선 block은 0/3입니다. "
                "ERA5 30개 특징은 69,850개 OOF key에 100% causal join됐지만, 동일 frozen experts의 convex "
                "weight를 더 잘 고르지 못했습니다. 물리 가설의 구현은 검증됐고 성능 가설은 기각됐습니다."
            ),
        },
        {
            "id": "p3_finding",
            "type": "markdown",
            "sourceId": "p3_deployment_derivation",
            "body": (
                "## P3 KMA는 학습 신호와 배포 가능한 신호가 달랐습니다\n\n"
                f"Source/target domain AUC는 **{p3['domain_auc']:.6f}**로 직접 concat/full finetune이 부적합했고, "
                f"source-meta inner gate는 **{p3['meta_delta']:+.6f}m** 악화했습니다. 보정된 fold별 α 후보는 "
                f"outer ΔRMSE **{local_delta:+.9f}m**였지만 CI90 상한이 "
                f"**{p3['local_ci_high']:+.9f}m**로 0을 넘었습니다. 더 중요하게 실제 global α=0.4는 "
                f"**{deployed_delta:+.9f}m** 악화했습니다. 같은 평가행에서 고른 posthoc α=0.2의 "
                f"**{posthoc_delta:+.9f}m**도 CI90이 0을 포함하므로 승격 근거가 아닙니다. 철회 영수증에 따라 "
                "secondary candidate는 업로드 금지입니다."
            ),
        },
        {
            "id": "p3_grid_chart_intro",
            "type": "markdown",
            "sourceId": "p3_deployment_derivation",
            "body": (
                "## P3의 사후 alpha 곡선은 작은 이득과 배포 실패를 동시에 보여줍니다\n\n"
                "아래 막대는 동일한 182개 평가 case에서 global α만 0.0–0.4로 바꾼 진단입니다. "
                "음수는 incumbent 대비 RMSE 개선, 양수는 악화입니다. α=0.2가 사후 최저지만 CI90이 "
                "0을 포함하고 +18h와 winter fold가 악화했으며, 실제 secondary 생성에 사용한 α=0.4는 "
                "pooled RMSE도 악화했습니다. 따라서 이 곡선은 새 α 선택 근거가 아니라 철회 근거입니다."
            ),
        },
        {"id": "p3_grid_chart", "type": "chart", "chartId": "p3_global_alpha_grid"},
        {
            "id": "decision_table_intro",
            "type": "markdown",
            "body": (
                "## 최종 판정표는 로컬 효과·불확실성·배포 일관성을 함께 요구합니다\n\n"
                "Local delta 하나가 좋아 보여도 CI와 fold 일관성, 배포 때 재현되는 단일 정책을 모두 통과해야 "
                "현재 제출을 바꿉니다. 아래 표는 그 세 조건과 유지할 exact incumbent를 한 행에 묶습니다."
            ),
        },
        {"id": "decision_table", "type": "table", "tableId": "final_decision_register"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## 범위와 지표 정의\n\n"
                "- **P1 delta:** candidate − incumbent weighted F1입니다. 음수는 악화입니다. 421,032개 "
                "chronological OOF 행을 사용했지만 non-virgin follow-up이며 independent outer holdout 주장이 아닙니다.\n"
                "- **P2 delta:** candidate − control RMSE(°C)입니다. 음수가 개선입니다. NASA와 ERA5는 동일한 "
                "69,850개 key 및 3개 block 계약을 사용합니다.\n"
                "- **P3 delta:** candidate − incumbent 유의파고 RMSE(m)입니다. 음수가 개선입니다. outer 결과는 "
                "182개 case × 6 lead = 1,092행입니다.\n"
                "- 모든 수치는 로컬 검증 근거이며 official hidden leaderboard 성능이 아닙니다."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "final_decision_derivation",
            "body": (
                "## 검증 설계와 증거 봉인\n\n"
                "각 source는 builder에 고정된 SHA-256과 byte-for-byte 일치해야만 열립니다. JSON은 집계 영수증만 "
                "읽고, 세 incumbent CSV는 내용을 parse하지 않고 SHA만 계산합니다. P2 ERA5는 current outer truth를 "
                "보지 않고 세 fold 예측을 Parquet로 fsync·reload·hash 봉인한 뒤 designated truth를 전역 한 번만 "
                "열었습니다. P3는 domain gate→source-meta inner gate→calibrated outer gate→deployment-consistency "
                "diagnostic→withdrawal 순서를 보존합니다. 보고서 생성은 model fit, target open, submission 생성, "
                "upload 권한을 갖지 않습니다."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 한계·불확실성·강건성\n\n"
                "- **P1은 독립 재검증이 아닙니다.** 같은 outer 결과를 보고 threshold만 다시 맞추면 과적합입니다.\n"
                "- **P2 결론은 현재 gate 구조에 한정됩니다.** ERA5가 해양 혼합과 무관하다는 결론이 아니라, "
                "현재 30개 특징이 physical/deep blend weight를 개선하지 못했다는 결론입니다.\n"
                "- **P3 local 신호는 배포 일관성이 없습니다.** fold별 α의 결과와 global α=0.4 결과가 방향부터 "
                "다르고, 사후 α=0.2는 같은 평가행에서 선택되어 독립 promotion evidence가 아닙니다.\n"
                "- 서로 다른 단위와 목적함수를 한 점수나 한 축으로 합치지 않았습니다. 문제별 delta 카드와 "
                "exact audit 표를 사용하고, 유일한 차트는 P3 내부의 동일 단위 alpha grid로 제한했습니다."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 권고 다음 단계 — 세 incumbent를 유지하고 같은 outer 재튜닝을 중단합니다\n\n"
                "1. **P1:** frozen XGBoost QC 제출을 유지합니다. 외부 q50을 다시 쓴다면 wholesale replacement가 "
                "아닌 합성 anomaly 학습 또는 별도 veto처럼 구조가 다른 사전등록 가설만 허용합니다.\n"
                "2. **P2:** extrapolated soft-gate v2를 유지합니다. ERA5 특징·optimizer·gate를 같은 3 block에 다시 "
                "맞추지 말고, 새 독립 split이나 다른 예측 대상이 생길 때만 재개합니다.\n"
                "3. **P3:** long-lead persistence-shrink를 유지하고 secondary candidate는 제출하지 않습니다. KMA를 "
                "재사용하려면 배포 가능한 단일 global calibration을 먼저 고정하고 fresh holdout에서 검증합니다."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 다음 판단을 바꿀 질문\n\n"
                "- P1 external representation이 실제 관측을 교체하지 않고 합성 anomaly 다양성만 늘릴 때도 F1을 "
                "악화시키는가?\n"
                "- P2 ERA5 신호가 blend weight가 아니라 성층 붕괴 event 분류처럼 더 직접적인 물리 target에서 "
                "fresh split 개선을 보이는가?\n"
                "- P3 KMA gain을 하나의 사전 고정 global policy로 변환했을 때 18h와 winter fold guard를 동시에 "
                "지킬 수 있는가?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "P1/P2/P3 외부 데이터 실험의 aggregate-only 최종 유지·기각·철회 판정",
            "generatedAt": generated_at,
            "filters": [],
            "cards": cards,
            "charts": [alpha_grid_chart],
            "tables": [decision_table],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "p1_delta_card": p1_card_rows,
                "p2_delta_card": p2_card_rows,
                "p3_delta_card": p3_card_rows,
                "p3_alpha_grid": p3_grid_rows,
                "decision_register": decision_rows,
                "evidence_registry": evidence_rows,
                "visual_contract": visual_contract_rows,
            },
            "accessIssues": [],
        },
        "sources": [
            {"id": source["id"], "label": source["label"], "path": source["path"]}
            for source in sources
        ],
        "package_info": {
            "originUrl": "artifact://external-data-final-decision-v2-2026-08-21-r2",
            "controls": {"edit": False, "refresh": False},
        },
    }
    _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    _require(artifact.get("surface") == "report", "surface must be report")
    manifest = _mapping(artifact.get("manifest"), role="artifact manifest")
    snapshot = _mapping(artifact.get("snapshot"), role="artifact snapshot")
    _require(manifest.get("title") == REPORT_TITLE, "manifest title drifted")
    blocks = _sequence(manifest.get("blocks"), role="report blocks")
    _require(
        blocks and _mapping(blocks[0], role="title block").get("body") == f"# {REPORT_TITLE}",
        "visible title mismatch",
    )
    _require(
        len(_sequence(manifest.get("cards"), role="report cards")) == 3,
        "exactly three problem delta cards required",
    )
    charts = _sequence(manifest.get("charts"), role="report charts")
    _require(len(charts) == 1, "exactly one same-unit P3 diagnostic chart required")
    chart = _mapping(charts[0], role="P3 alpha chart")
    _require(chart.get("dataset") == "p3_alpha_grid", "P3 alpha chart dataset drifted")
    _require(chart.get("type") == "bar", "P3 alpha chart type drifted")
    tables = _sequence(manifest.get("tables"), role="report tables")
    _require(len(tables) == 1, "exactly one final decision table required")
    table = _mapping(tables[0], role="decision table")
    expected_fields = [
        "problem",
        "hypothesis",
        "local_metric_delta",
        "uncertainty",
        "deployment_consistency",
        "decision",
        "current_incumbent",
    ]
    columns = _sequence(table.get("columns"), role="decision columns")
    _require(
        [_mapping(column, role="decision column").get("field") for column in columns]
        == expected_fields,
        "decision table columns drifted",
    )
    datasets = _mapping(snapshot.get("datasets"), role="snapshot datasets")
    rows = _sequence(datasets.get("decision_register"), role="decision rows")
    _require(len(rows) == 3, "decision table must contain one row per problem")
    _require(
        [_mapping(row, role="decision row").get("problem") for row in rows] == ["P1", "P2", "P3"],
        "problem order drifted",
    )
    _require(snapshot.get("status") == "ready", "snapshot must be ready")

    required_blocks = {
        "technical_summary",
        "delta_cards_intro",
        "p1_finding",
        "p2_finding",
        "p3_finding",
        "p3_grid_chart_intro",
        "decision_table_intro",
        "scope_definitions",
        "methodology",
        "limitations",
        "next_steps",
        "further_questions",
    }
    block_ids = {_mapping(block, role="report block").get("id") for block in blocks}
    _require(required_blocks <= block_ids, "technical report structure is incomplete")

    source_ids = {
        _mapping(source, role="manifest source").get("id")
        for source in _sequence(manifest.get("sources"), role="manifest sources")
    }
    for card in _sequence(manifest.get("cards"), role="report cards"):
        card_map = _mapping(card, role="metric card")
        _require(card_map.get("sourceId") in source_ids, "metric card lacks canonical provenance")
        _require(
            len(_sequence(card_map.get("metrics"), role="card metrics")) == 1,
            "each card needs one headline metric",
        )
    _require(chart.get("sourceId") in source_ids, "P3 alpha chart lacks canonical provenance")
    _require(table.get("sourceId") in source_ids, "decision table lacks canonical provenance")

    serialized = json.dumps(artifact, ensure_ascii=False)
    forbidden_fragments = (
        "C:/Users/",
        "C:\\Users\\",
        "/Users/",
        "file://",
        "api_key",
        "authorization: bearer",
        "station,time,temp",
        "case_id,station,lead_h",
        "incumbent_probability",
        "candidate_probability",
        "target_hs",
    )
    lowered = serialized.lower()
    for forbidden in forbidden_fragments:
        _require(
            forbidden.lower() not in lowered,
            f"forbidden local, secret, or row-level content: {forbidden}",
        )
    for source in _sequence(artifact.get("sources"), role="top-level sources"):
        path = Path(str(_mapping(source, role="top-level source").get("path")))
        _require(not path.is_absolute(), "absolute source path leaked")
        _require(".." not in path.parts, "parent traversal leaked")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    _require(_path_within(output, root), "output must stay within repository root")
    generated_at = args.generated_at or datetime.now(KST).isoformat(timespec="seconds")
    evidence = collect_evidence(root)
    artifact = build_artifact(evidence, generated_at=generated_at)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "validated_not_written",
                    "output": output.relative_to(root).as_posix(),
                    "evidence_count": len(RELATIVE_PATHS),
                    "decision_rows": 3,
                },
                ensure_ascii=False,
            )
        )
        return 0

    _require(not output.exists(), f"refusing to overwrite report generation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        with output.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError as exc:
        raise FinalDecisionReportError(f"report output collision: {output}") from exc
    print(
        json.dumps(
            {
                "status": "written",
                "artifact": output.relative_to(root).as_posix(),
                "sha256": _sha256(output),
                "evidence_count": len(RELATIVE_PATHS),
                "decision_rows": 3,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
