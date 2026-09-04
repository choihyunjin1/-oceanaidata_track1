"""Materialize the exact frozen v5 ExtraTrees router under competition adjudication.

This script does not reinterpret the scientific result as significant.  It
records scientific INCONCLUSIVE and a separate positive expected-value
competition action under the governing expiring-quota policy, then performs
one exact full fit and creates one non-uploaded official submission candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for entry in (SCRIPTS, SRC):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

from run_p3_parallel_candidate_cycle_20260831_v4 import (  # noqa: E402
    ACTIVE_LEADS,
    BOOTSTRAP_SEED,
    KEYS,
)
from run_p3_physical_expert_router_cycle_20260831_v5 import (  # noqa: E402
    ALPHA_MAX,
    CHAMPION_PATH,
    SPECS,
    _sample_weight,
    build_estimator,
    build_historical_cases,
    json_bytes,
    load_official_cases,
    route_alpha,
    sha256,
    write_new,
)

EXPERIMENT_ID = "p3_v5_extratrees_competition_adjudication_20260831_v7"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
DELIVERY_DIR = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_V5_EXTRATREES_ADJUDICATED_V7"
)
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
V5_RUNNER = ROOT / "scripts/run_p3_physical_expert_router_cycle_20260831_v5.py"
V5_RESULT = ROOT / "artifacts/p3_physical_expert_router_cycle_20260831_v5/result.json"
POLICY_PATH = ROOT / "configs/goals/tolerance_recalibration_and_failure_replay_20260830_v2.json"
TARGET_NAME = "P3_2_EXTRATREES_HARD_PHYSICAL_ROUTER"
TARGET_SPEC_INDEX = 1
FULL_FIT_SEED = BOOTSTRAP_SEED + 1_000 + TARGET_SPEC_INDEX


class ContractError(RuntimeError):
    """Raised when exact frozen lineage or output validity is violated."""


def adjudicate() -> tuple[dict[str, Any], list[str], pd.DataFrame, dict[str, Any]]:
    v5 = json.loads(V5_RESULT.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    matches = [record for record in v5["candidates"] if record["spec"]["name"] == TARGET_NAME]
    if len(matches) != 1:
        raise ContractError("frozen v5 target record missing or duplicated")
    record = matches[0]
    spec = SPECS[TARGET_SPEC_INDEX]
    if spec.name != TARGET_NAME or spec.family != "extra_trees" or spec.policy != "hard_0p50":
        raise ContractError("frozen v5 ExtraTrees contract changed")
    if record["strict_internal_pass"]:
        raise ContractError("v5 scientific status unexpectedly changed")
    score = record["conditional_leaderboard_translation"]["scenarios"]
    central_gain = float(score["central"]["projected_point_delta"])
    conservative_gain = float(score["conservative"]["projected_point_delta"])
    optimistic_gain = float(score["optimistic"]["projected_point_delta"])
    probability = float(record["bootstrap"]["probability_improved"])
    heuristic_action_points = probability * central_gain + (1.0 - probability) * conservative_gain
    competition_rule = policy["tolerance_layers"]["competition_action"]
    level_zero = policy["tolerance_layers"]["level_0_validity"]
    scientific = {
        "status": "SCIENTIFIC_INCONCLUSIVE_NOT_STRONG_CHALLENGER",
        "reason": "The dependence-preserving 90% interval crosses zero.",
        "delta_rmse_candidate_minus_champion_m": float(record["delta_rmse"]),
        "ci90_low_m": float(record["bootstrap"]["ci90_low"]),
        "ci90_high_m": float(record["bootstrap"]["ci90_high"]),
        "probability_improved": probability,
    }
    competition = {
        "status": "COMPETITION_EXPECTED_VALUE_PASS_READY_NOT_UPLOADED",
        "policy_rule": competition_rule["expiring_quota_rule"],
        "candidate_validity_policy": level_zero,
        "central_projected_point_delta": central_gain,
        "conservative_projected_point_delta": conservative_gain,
        "optimistic_projected_point_delta": optimistic_gain,
        "heuristic_probability_weighted_action_points": heuristic_action_points,
        "decision_changing_information_value": True,
        "reason": (
            "P3 has no strict internal PASS candidate; this exact frozen nonduplicate router has "
            "a favorable pooled point estimate, 5/6 diagnostic blocks, P(improve)=0.9156, and a "
            "material central planning gain. The Public query can resolve a known transport uncertainty."
        ),
        "transport_warning": (
            "The bootstrap probability is historical, not an official improvement probability. "
            "Past P3 local-to-official direction reversals make both sign and magnitude uncertain."
        ),
    }
    if not (
        record["delta_rmse"] < 0.0
        and central_gain > 0.0
        and heuristic_action_points > 0.0
        and record["improved_block_count"] == 5
    ):
        raise ContractError("competition expected-value prerequisites changed")
    frame, cases, profile, features = build_historical_cases()
    return {"scientific": scientific, "competition": competition, "v5_record": record}, features, cases, profile


def full_fit_and_materialize(
    adjudication: dict[str, Any], features: list[str], cases: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, int]]:
    spec = SPECS[TARGET_SPEC_INDEX]
    model = build_estimator(spec, FULL_FIT_SEED)
    model.fit(
        cases[features],
        cases["champion_better"].to_numpy(dtype=np.int8),
        extratreesclassifier__sample_weight=_sample_weight(cases["advantage_mse_gain"]),
    )
    official, official_cases, champion = load_official_cases(features)
    probability = model.predict_proba(official_cases[features])[:, 1]
    alpha = route_alpha(probability, spec.policy)
    if alpha.min() < 0.0 or alpha.max() > ALPHA_MAX:
        raise ContractError("official physical alpha bound failed")
    case_alpha = official_cases[["case_id", "station"]].copy()
    case_alpha["router_alpha"] = alpha
    mapping = official[["case_id", "station"]].merge(
        case_alpha,
        on=["case_id", "station"],
        how="left",
        validate="many_to_one",
    )
    active = official["lead_h"].isin(ACTIVE_LEADS).to_numpy()
    prediction = official["reference"].to_numpy(float).copy()
    prediction[active] = (
        official.loc[active, "base"].to_numpy(float)
        + mapping.loc[active, "router_alpha"].to_numpy(float)
        * official.loc[active, "delta"].to_numpy(float)
    )
    prediction = np.clip(prediction, 0.0, 30.0)
    champion_prediction = champion["hs_pred"].to_numpy(float)
    if not np.array_equal(prediction[~active], champion_prediction[~active]):
        raise ContractError("official short lead exact no-op failed")
    submission = official[KEYS].copy()
    submission["hs_pred"] = prediction
    if len(submission) != 1200 or submission.duplicated(KEYS).any() or not np.isfinite(prediction).all():
        raise ContractError("official submission structure failed")
    payload = submission.to_csv(index=False, lineterminator="\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    champion_digest = sha256(CHAMPION_PATH)
    if digest == champion_digest or np.array_equal(prediction, champion_prediction):
        raise ContractError("adjudicated candidate duplicates the current champion")
    DELIVERY_DIR.mkdir(parents=True, exist_ok=False)
    csv_path = DELIVERY_DIR / "P3_submission.csv"
    write_new(csv_path, payload)
    output = {
        "name": TARGET_NAME,
        "title": "P3 V5 ExtraTrees Physical Router Competition Probe",
        "summary": (
            "Frozen v5 train-only case router; scientific CI crosses zero, but competition "
            "expected-value adjudication is positive. No upload performed."
        ),
        "path": str(csv_path),
        "rows": 1200,
        "sha256": digest,
        "bytes": len(payload),
        "champion_sha256": champion_digest,
        "nonduplicate_vs_champion": True,
        "minimum_m": float(prediction.min()),
        "maximum_m": float(prediction.max()),
        "changed_rows_vs_champion": int(np.sum(np.abs(prediction - champion_prediction) > 1e-12)),
        "short_lead_exact_noop": True,
        "official_case_route_share": float(np.mean(alpha > 0.0)),
        "official_probability_min": float(probability.min()),
        "official_probability_median": float(np.median(probability)),
        "official_probability_max": float(probability.max()),
        "full_fit_seed": FULL_FIT_SEED,
        "full_fit_count": 1,
        "adjudication": adjudication,
    }
    write_new(
        DELIVERY_DIR / "submission-info.txt",
        (
            f"title: {output['title']}\n"
            f"summary: {output['summary']}\n"
            f"sha256: {digest}\n"
            "scientific: INCONCLUSIVE\n"
            "competition: EXPECTED_VALUE_PASS\n"
            "risk: conservative projected point delta is negative; transport can reverse sign\n"
            "status: READY_NOT_UPLOADED\n"
        ).encode(),
    )
    write_new(
        DELIVERY_DIR / "SET_MANIFEST.json",
        json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "COMPETITION_EXPECTED_VALUE_PASS_READY_NOT_UPLOADED",
                "output": output,
                "hidden_truth_rows_read": 0,
                "uploads": 0,
            }
        ),
    )
    access = {
        "test_index_rows_read": 1200,
        "base_prediction_rows_read": 1200,
        "source_prediction_rows_read": 1200,
        "champion_prediction_rows_read": 1200,
        "official_test_feature_rows_read": 200,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    return output, access


def make_report(result: dict[str, Any]) -> str:
    adjudication = result["adjudication"]
    competition = adjudication["competition"]
    scientific = adjudication["scientific"]
    output = result["output"]
    return "\n".join(
        [
            "# P3 v5 ExtraTrees competition adjudication v7",
            "",
            "## 결론",
            "",
            "- **Scientific: INCONCLUSIVE.** CI90이 0을 교차하므로 강한 과학적 challenger로 승격하지 않는다.",
            "- **Competition: EXPECTED_VALUE PASS / READY_NOT_UPLOADED.** 만료성 제출 슬롯의 정보가치를 포함한 별도 행동 판정이다.",
            f"- frozen v5 ExtraTrees를 seed {FULL_FIT_SEED}로 정확히 1회 full-fit했고 1,200행 비중복 CSV를 만들었다.",
            "",
            "## 내부 근거와 점수 범위",
            "",
            f"- delta RMSE(candidate-reference): {scientific['delta_rmse_candidate_minus_champion_m']:+.9f}m",
            f"- episode bootstrap CI90: [{scientific['ci90_low_m']:+.9f}, {scientific['ci90_high_m']:+.9f}]m; P(improve)={scientific['probability_improved']:.4f}",
            f"- 예상 점수 변화 conservative/central/optimistic: {competition['conservative_projected_point_delta']:+.6f} / {competition['central_projected_point_delta']:+.6f} / {competition['optimistic_projected_point_delta']:+.6f}",
            f"- heuristic probability-weighted action points: {competition['heuristic_probability_weighted_action_points']:+.6f}",
            "- 보수 시나리오는 손실이며 실제 official 수송은 과거 방향 반전 때문에 부호도 보장되지 않는다.",
            "",
            "## 구조 QA",
            "",
            f"- CSV SHA-256: `{output['sha256']}`",
            f"- champion 대비 changed rows: {output['changed_rows_vs_champion']}; short leads exact no-op: {output['short_lead_exact_noop']}",
            f"- finite/domain: min={output['minimum_m']:.9f}m, max={output['maximum_m']:.9f}m",
            "- hidden truth 0, upload 0.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or DELIVERY_DIR.exists() or ATTEMPT_LOCK.exists():
        raise FileExistsError("v7 output or attempt lock already exists")
    for path in (V5_RUNNER, V5_RESULT, POLICY_PATH, CHAMPION_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    runner_hash = sha256(Path(__file__))
    lock = {
        "schema_version": "p3.v5_extratrees_competition_adjudication.attempt.20260831.v7",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "ATTEMPT_CONSUMED",
        "runner_sha256": runner_hash,
        "v5_runner_sha256": sha256(V5_RUNNER),
        "v5_result_sha256": sha256(V5_RESULT),
        "governing_policy_sha256": sha256(POLICY_PATH),
        "target_name": TARGET_NAME,
        "target_spec_index": TARGET_SPEC_INDEX,
        "full_fit_seed": FULL_FIT_SEED,
        "planned_full_fit_count": 1,
        "threshold_or_model_change_from_v5": False,
        "scientific_and_competition_decisions_separated": True,
        "hidden_truth_rows": 0,
        "uploads": 0,
    }
    write_new(ATTEMPT_LOCK, json_bytes(lock))
    started = time.perf_counter()
    adjudication, features, cases, profile = adjudicate()
    output, access = full_fit_and_materialize(adjudication, features, cases)
    result = {
        "schema_version": "p3.v5_extratrees_competition_adjudication.result.20260831.v7",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPETITION_EXPECTED_VALUE_PASS_READY_NOT_UPLOADED",
        "runtime_seconds": float(time.perf_counter() - started),
        "runner_sha256": runner_hash,
        "attempt_lock_sha256": sha256(ATTEMPT_LOCK),
        "v5_runner_sha256": sha256(V5_RUNNER),
        "v5_result_sha256": sha256(V5_RESULT),
        "governing_policy_sha256": sha256(POLICY_PATH),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data_profile": profile,
        "adjudication": adjudication,
        "full_fit_count": 1,
        "output": output,
        "official_access": access,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
        "threshold_or_model_change_from_v5": False,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    write_new(ARTIFACT_DIR / "result.json", json_bytes(result))
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    write_new(REPORT_DIR / "report-source.md", make_report(result).encode())
    write_new(
        REPORT_DIR / "run-manifest.json",
        json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(ARTIFACT_DIR / "result.json"),
                "runner_sha256": runner_hash,
                "attempt_lock_sha256": sha256(ATTEMPT_LOCK),
                "submission_sha256": output["sha256"],
                "full_fit_count": 1,
                "hidden_truth_rows_read": 0,
                "uploads": 0,
            }
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
