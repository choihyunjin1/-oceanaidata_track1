"""Finalize committed P2 v28 metrics after a report-order technical failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_layer_task_gradient_surgery_deepset_20260901_v28 as runner  # noqa: E402


def main() -> None:
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    artifact_result_path = runner.ARTIFACT / "result.json"
    report_result_path = runner.REPORT / "result.json"
    prediction_path = runner.ARTIFACT / f"{runner.PREDICTION_NAME}.npz"
    lock_path = runner.ARTIFACT / "attempt_lock.json"
    result = json.loads(artifact_result_path.read_text(encoding="utf-8"))
    if result["fit_count"] != 9 or result["operation_counters"]["uploads"] != 0:
        raise runner.v12.ContractError("v28 committed base result contract failed")
    if runner.v12.sha256_file(prediction_path) != result["hashes"]["prediction_npz"]:
        raise runner.v12.ContractError("v28 committed prediction hash drift")
    if runner.v12.sha256_file(artifact_result_path) != runner.v12.sha256_file(
        report_result_path
    ):
        raise runner.v12.ContractError("v28 base result copies differ")
    gate = runner.prospective_fold_layer_gate(result["candidate"], config)
    legacy_safety = bool(result["candidate"]["safety_pass"])
    effective_safety = bool(legacy_safety and gate["pass"])
    status = (
        "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION"
        if result["candidate"]["strict_exploratory_pass"] and effective_safety
        else "EXPLORATORY_NO_GO_LAYER_TASK_PCGRAD"
    )
    amendment = {
        "schema_version": "p2.layer_task_gradient_surgery_deepset.terminal_amendment.20260901.v28",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": status,
        "scientific_result_available": True,
        "technical_event": {
            "classification": "POST_METRIC_REPORT_ORDER_FAILURE",
            "error": "KeyError: prospective_fold_layer_gate",
            "location": "scripts/run_p2_layer_task_gradient_surgery_deepset_20260901_v28.py:453",
            "same_id_rerun": False,
            "metrics_retrained": False,
            "artifact_result_modified": False,
        },
        "fit_count": result["fit_count"],
        "runtime_seconds": result["runtime_seconds"],
        "candidate": {
            "reference_rmse": result["candidate"]["reference_rmse"],
            "candidate_rmse": result["candidate"]["candidate_rmse"],
            "delta_rmse": result["candidate"]["delta_rmse"],
            "canonical_nominal_pooled_points_delta": result["candidate"][
                "canonical_nominal_pooled_points_delta"
            ],
            "canonical_transport_adjusted_pooled_points_delta": result["candidate"][
                "canonical_transport_adjusted_pooled_points_delta"
            ],
            "legacy_safety_pass_without_v26a_amendment": legacy_safety,
            "prospective_fold_layer_gate": gate,
            "safety_pass_with_v26a_amendment": effective_safety,
        },
        "operation_counters": result["operation_counters"],
        "hashes": {
            "artifact_result": runner.v12.sha256_file(artifact_result_path),
            "report_result": runner.v12.sha256_file(report_result_path),
            "prediction_npz": runner.v12.sha256_file(prediction_path),
            "prediction_commitment": runner.v12.sha256_file(
                runner.ARTIFACT / "prediction_commitment.json"
            ),
            "attempt_lock": runner.v12.sha256_file(lock_path),
            "config": runner.v12.sha256_file(runner.CONFIG),
            "runner": runner.v12.sha256_file(runner.RUNNER),
            "v13_runner": runner.v12.sha256_file(runner._V13_RUNNER),
            "prospective_gate_amendment": config["authorization_evidence"][
                "prospective_gate_amendment_sha256"
            ],
        },
    }
    runner.v12.atomic_json(runner.REPORT / "terminal-amendment.json", amendment)
    item = amendment["candidate"]
    (runner.REPORT / "report-source.md").write_text(
        "# P2 v28 fixed target-layer PCGrad DeepSets\n\n"
        "## 결론\n\n"
        f"상태: `{status}`. 9 fits와 prediction/result는 기술 오류 전에 봉인됐다. "
        f"pooled ΔRMSE `{item['delta_rmse']:+.9f} C`, nominal "
        f"`{item['canonical_nominal_pooled_points_delta']:+.6f}`점, transport "
        f"`{item['canonical_transport_adjusted_pooled_points_delta']:+.6f}`점.\n\n"
        f"v26a prospective fold×layer gate는 `{gate['pass']}`: non-harm "
        f"`{gate['non_harm_cells']}/9`, max cell `{gate['maximum_cell_delta_rmse_C']:+.9f} C`. "
        "따라서 aggregate 개선에도 안전 후보가 아니다.\n\n"
        "최초 실행은 base result 저장 뒤 Markdown report 순서에서 KeyError로 종료됐다. "
        "학습이나 metric을 재실행하지 않고 immutable result/prediction을 독립 QA한다. "
        "official/query/hidden/CSV/upload=0.\n",
        encoding="utf-8",
    )
    print(json.dumps(amendment, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
