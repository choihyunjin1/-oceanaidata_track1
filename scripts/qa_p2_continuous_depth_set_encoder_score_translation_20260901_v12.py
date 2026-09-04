"""Seal the canonical score translation amendment for terminal P2 v12."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12"
REPORT = ROOT / "reports" / EXPERIMENT_ID
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"


def main() -> None:
    result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    candidate = result["candidate"]
    slope = float(config["evaluation"]["points_per_rmse_C"])
    penalty = float(config["evaluation"]["transport_penalty_points"])
    delta = float(candidate["delta_rmse"])
    nominal = -delta * slope
    transport = nominal - penalty
    checks = {
        "pooled_delta_finite": abs(delta) < float("inf"),
        "canonical_nominal_matches_recorded_explicit_field": abs(
            nominal - float(candidate["nominal_pooled_points_delta"])
        )
        <= 1e-12,
        "legacy_raw_identified_as_official_like_ci_translation": abs(
            float(candidate["raw_expected_points_delta"])
            - max(0.0, -float(candidate["official_like_bootstrap"]["ci90_high"]) * slope)
        )
        <= 1e-12,
        "legacy_transport_identified": abs(
            float(candidate["transport_calibrated_expected_points_delta"])
            - (float(candidate["raw_expected_points_delta"]) - penalty)
        )
        <= 1e-12,
        "nov_dec_safety_regression_positive": float(
            candidate["by_fold"]["2025_nov_dec"]["delta_rmse"]
        )
        > 0.0,
        "official_access_zero": all(
            int(result["operation_counters"][name]) == 0
            for name in (
                "official_test_index_rows_read",
                "hidden_truth_rows_read",
                "submission_csv_created",
                "uploads",
            )
        ),
    }
    amendment = {
        "schema_version": "p2.score_translation_amendment.20260901.v12",
        "experiment_id": EXPERIMENT_ID,
        "status": "CANONICAL_TRANSLATION_AMENDED_RESULT_IMMUTABLE",
        "formal_registered_gate": "PASS",
        "transport_safety": "NOT_READY",
        "reason": "Nov-Dec regressed despite the registered aggregate gate passing.",
        "pooled_delta_rmse_C": delta,
        "canonical_points_per_rmse_C": slope,
        "canonical_nominal_planning_points": nominal,
        "fixed_transport_penalty_points": penalty,
        "canonical_transport_adjusted_planning_points": transport,
        "legacy_engine_fields": {
            "raw_expected_points_delta": candidate["raw_expected_points_delta"],
            "transport_calibrated_expected_points_delta": candidate[
                "transport_calibrated_expected_points_delta"
            ],
            "meaning": "official-like Sep-Oct day-bootstrap CI90-high translation, not pooled-delta canonical planning translation",
        },
        "posthoc_routing_or_gate_change": False,
        "result_json_modified": False,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    qa = {
        "schema_version": "p2.score_translation_amendment.independent_qa.20260901.v12",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "canonical_nominal_planning_points": nominal,
        "canonical_transport_adjusted_planning_points": transport,
    }
    (REPORT / "score-translation-amendment.json").write_text(
        json.dumps(amendment, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (REPORT / "score-translation-amendment-qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (REPORT / "score-translation-amendment.md").write_text(
        "# P2 v12 score translation amendment\n\n"
        "## 결론\n\n"
        f"등록 gate는 PASS지만 transport safety는 NOT_READY다. pooled ΔRMSE `{delta:+.9f}℃`의 "
        f"canonical planning 환산은 명목 `{nominal:+.6f}`점, 고정 penalty 후 "
        f"`{transport:+.6f}`점이다.\n\n"
        "기존 result의 raw/transport 필드는 Sep-Oct day-bootstrap CI90 상단 환산인 legacy "
        "engine field이며 pooled-delta planning 값으로 사용하지 않는다. Nov-Dec 회귀를 근거로 "
        "posthoc 라우팅하거나 gate를 바꾸지 않았다. result.json은 수정하지 않았다.\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
