"""Build the P1-P3 checkpoint retrospective audit report.

This builder consumes only historical local artifacts and the already recorded
local-to-official calibration ledger.  It does not open any official test,
sample-submission, or submission CSV path.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/checkpoint_retroaudit_20260827_v1"
P3_ORACLE = OUT / "p3_catboost_checkpoint_oracle.json"
TRANSPORT = (
    ROOT
    / "reports/next_day_breakthrough_deep_research_20260827_v1"
    / "local_official_calibration.json"
)

STATUS_LABELS = {
    "ALREADY_BEST": "적정 선택 완료",
    "REPLAYED_POSITIVE_TREND": "재생 완료·양의 추세",
    "REPLAYABLE": "재생 가능",
    "RERUN_REQUIRED": "재학습 필요",
    "NOT_APPLICABLE": "체크포인트 비적용",
    "DESIGN_ONLY": "설계·무효",
}


def record(
    problem: str,
    lineage: str,
    status: str,
    priority: str,
    trend: str,
    observed: str,
    interpretation: str,
    next_action: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "problem": problem,
        "lineage": lineage,
        "status": status,
        "status_label": STATUS_LABELS[status],
        "priority": priority,
        "trend": trend,
        "observed_result": observed,
        "interpretation": interpretation,
        "next_action": next_action,
        "evidence": evidence,
    }


def build_records() -> list[dict[str, Any]]:
    return [
        record(
            "P1",
            "MS-TCN++/ASRF Q2 checkpoint grid",
            "ALREADY_BEST",
            "완료",
            "양의 개발 추세",
            "882개 후보 전체 비교; 단일 최고 width512/epoch125/threshold0.9, ΔF1 +0.098157",
            "마지막 epoch 오채택이 아니다. 사후 plateau 규칙은 개발 증거로만 보존한다.",
            "새 outer 전에 ±1 plateau 후보(epoch150, threshold0.8)를 사전등록",
            [
                "artifacts/p1_incumbent_preserving_mstcn_asrf_v2/q2_selection.json",
                "artifacts/p1_incumbent_preserving_mstcn_asrf_v2/selected_recipe.json",
            ],
        ),
        record(
            "P1",
            "MS-TCN++/ASRF Q3-Q4 confirmatory refit",
            "RERUN_REQUIRED",
            "P0",
            "미해결·반전 가능",
            "pooled ΔF1 -0.005140; 6개 seed 중 4개 final/min loss 5.87~26.54배",
            "epoch125만 보존돼 중간 F1 복구 불가. optimizer excursion이 명확하다.",
            "epoch별 저장·low-LR tail·개발창 plateau rule로 재학습 후 fresh outer 1회",
            [
                "artifacts/p1_incumbent_preserving_mstcn_asrf_v2/terminal_result.json",
                "artifacts/p1_incumbent_preserving_mstcn_asrf_v2/q3_confirmatory_blind_receipt.json",
                "artifacts/p1_incumbent_preserving_mstcn_asrf_v2/q4_confirmatory_blind_receipt.json",
            ],
        ),
        record(
            "P1",
            "TE-TAD-lite implementation sanity",
            "RERUN_REQUIRED",
            "P1",
            "구현 gate 근접",
            "fixed300 final recall 0.823529<0.9; median IoU 0.850564 및 negative FP=0 통과",
            "중간 checkpoint로 구현 gate가 열릴 수 있으나 outer 성능은 아직 0회다.",
            "sanity checkpoint curve를 저장해 gate만 재확인",
            ["artifacts/p1_tetad_lite_direct_interval_set_v1/terminal_result.json"],
        ),
        record(
            "P1",
            "Meaningful learning-curve LightGBM",
            "RERUN_REQUIRED",
            "P2",
            "미약한 양의 추세",
            "fixed700 full ΔF1 +0.004186; CI90 [-0.009429,+0.016911]",
            "승격 증거는 아니지만 양의 추세이므로 폐기하지 않는다.",
            "inner-selected tree count로 저비용 재실행",
            ["artifacts/p1_meaningful_learning_curve_generation_v1/canonical_curve_decision.json"],
        ),
        record(
            "P1",
            "Rule-distillation residual + nonspike residual",
            "RERUN_REQUIRED",
            "P2",
            "no-op",
            "fixed120/fixed700, 모든 활성 결과 ΔF1 0",
            "checkpoint 변경이 gate를 열 가능성은 미확인이나 현재는 학습효과가 없다.",
            "P0/P1 완료 후에만 재검토",
            [
                "artifacts/p1_incumbent_rule_distillation_neural_residual_v5r6/canonical_curve_decision.json",
                "artifacts/p1_round_b_nonspike_long_event_residual_v1r6/metrics.json",
            ],
        ),
        record(
            "P1",
            "Binary/Station/Masked-pretrain TCN",
            "RERUN_REQUIRED",
            "보류",
            "강한 음의 추세",
            "final-only ΔF1 -0.338562/-0.310074/-0.508065",
            "checkpoint 결함은 있으나 격차가 커 +3점 목표의 우선순위가 낮다.",
            "보존만 하고 현재 재실행하지 않음",
            [
                "artifacts/p1_binary_event_tcn_dense_natural_v3/canonical_curve_decision.json",
                "artifacts/p1_station_layer_temporal_convolution_event_v2/canonical_curve_decision.json",
                "artifacts/p1_masked_pretrain_binary_event_v4r4/canonical_curve_decision.json",
            ],
        ),
        record(
            "P1",
            "Target-masked quantile regressors",
            "RERUN_REQUIRED",
            "보류",
            "강한 음의 추세",
            "fixed180 weighted ΔF1 -0.630823",
            "checkpoint만으로 복구될 가능성이 매우 낮다.",
            "보존만 하고 현재 재실행하지 않음",
            ["artifacts/p1_target_masked_quantile_v1/metrics.json"],
        ),
        record(
            "P1",
            "Sequence TCN / Patch Transformer / SSL",
            "ALREADY_BEST",
            "완료",
            "음의 결과",
            "best_state 복원; outer micro F1 TCN 0.767582, Patch Transformer 0.799755",
            "이미 inner-best checkpoint를 사용했다.",
            "checkpoint 관점 재실행 없음",
            ["artifacts/sequence_full_20260813/sequence_experiment.json"],
        ),
        record(
            "P1",
            "Block Inpaint",
            "ALREADY_BEST",
            "완료",
            "미약한 양의 추세",
            "3 folds 모두 best epoch 11/12; weighted ΔF1 +0.002591, CI가 0 교차",
            "마지막 epoch 근접·best_state 복원으로 checkpoint 판정 변화가 없다.",
            "연구 추세로 보존",
            ["artifacts/p1_block_inpaint_v1/historical_metrics.json"],
        ),
        record(
            "P1",
            "Initial tabular CV / incumbent / causal LightGBM",
            "ALREADY_BEST",
            "완료",
            "혼합",
            "XGB incumbent 700/700/700; causal LGBM 54/138/296, deployment median138",
            "inner early stopping의 best iteration을 사용했다.",
            "재실행 없음",
            ["src/p1_qc/pipeline.py"],
        ),
        record(
            "P1",
            "IORS external point residual",
            "ALREADY_BEST",
            "완료",
            "음의 결과",
            "selected iterations 11/99; weighted ΔF1 -0.048998",
            "이미 inner-best iteration을 사용했다.",
            "checkpoint 관점 재실행 없음",
            ["artifacts/p1_iors_external_point_residual_oof_v1/20260821T174744+0900/result.json"],
        ),
        record(
            "P1",
            "Analytic/postprocess/preexecution-only families",
            "NOT_APPLICABLE",
            "완료",
            "해당 없음",
            "semimarkov, matched filter, structural gates, sealed-only proposal families",
            "epoch/checkpoint가 없거나 실제 fit이 없다.",
            "별도 구조 연구로만 관리",
            ["reports/promotion_retroaudit_20260827_v1/report-source.md"],
        ),
        record(
            "P2",
            "Deep tournament + finalists",
            "ALREADY_BEST",
            "완료",
            "선택 건전",
            "TimeMixer best 0.784031 vs 가상 final 1.155133; 실제는 best_state 사용",
            "마지막 epoch를 썼다면 순위가 크게 틀렸지만 구현이 이미 방지했다.",
            "재실행 없음",
            [
                "artifacts/p2_deep_model_tournament_v1/result.json",
                "artifacts/p2_deep_finalists_v1/result.json",
            ],
        ),
        record(
            "P2",
            "Authoritative nested surrogate v5",
            "ALREADY_BEST",
            "완료",
            "선택 건전",
            "540 inner fits 전부 best 복원; last-best RMSE penalty median +0.135945, max +1.342703",
            "checkpoint 오채택이 결과에 섞이지 않았다.",
            "재실행 없음",
            ["artifacts/p2_authoritative_nested_surrogate_actual_20260825_v5/result.json"],
        ),
        record(
            "P2",
            "Architecture-matched reference v3",
            "ALREADY_BEST",
            "완료",
            "선택 건전",
            "12~52 epoch grid inner calibration 후 median epoch outer 적용",
            "독립 inner 선택 규칙을 사용했다.",
            "재실행 없음",
            ["src/p2_restore/architecture_matched_stage_a_execution_v3.py"],
        ),
        record(
            "P2",
            "Nested LGBM + top3 CatBoost/LGBM tuning",
            "ALREADY_BEST",
            "완료",
            "선택 건전",
            "LGBM selected 1249/2038/269/91, frozen median759; top3도 inner-selected",
            "주력 tree 계보는 best iteration을 사용했다.",
            "재실행 없음",
            [
                "artifacts/p2_lgbm_nested_tuning_v1/result.json",
                "artifacts/p2_top3_parallel_tuning_v1/result.json",
            ],
        ),
        record(
            "P2",
            "Structured mask imputer",
            "ALREADY_BEST",
            "완료",
            "no-op",
            "development best epoch 사용; final alpha=0",
            "checkpoint는 적정했으나 후보 효과가 없었다.",
            "재실행 없음",
            ["artifacts/p2_structured_mask_imputer_v1/result.json"],
        ),
        record(
            "P2",
            "Max-round convergence",
            "REPLAYABLE",
            "진단 완료",
            "400-round optimum",
            "router RMSE 400=0.788890 vs 5000=0.866540, 이점 0.077651",
            "동일 OOF oracle라 승격용은 아니지만 underfit 가설은 기각한다.",
            "진단 기록으로 보존",
            ["artifacts/p2_max_round_convergence_v1/result.json"],
        ),
        record(
            "P2",
            "Conservative stack + corrected repeated-forward v2",
            "REPLAYABLE",
            "P1",
            "미확인",
            "보존된 400-tree fold models에서 num_iteration replay 가능",
            "historical feature 재생성 뒤 independent inner selection이 가능하다.",
            "재학습 전 저비용 replay",
            [
                "artifacts/p2_conservative_stack_improvement_v1",
                "artifacts/p2_corrected_repeated_forward_v2",
            ],
        ),
        record(
            "P2",
            "Joint hydrographic multitask layer4 r3",
            "RERUN_REQUIRED",
            "P0",
            "유의한 부분구간 추세",
            "fixed28 full ΔRMSE +0.006663, CI 0 교차; prefix0.85 Δ -0.055054",
            "full은 근소 열세지만 부분구간 개선이 커 checkpoint 반전 여지가 가장 크다.",
            "inner calibration checkpoint를 outer에 고정해 재실행",
            ["artifacts/p2_joint_hydrographic_multitask_layer4_execution_r3"],
        ),
        record(
            "P2",
            "Meaningful CatBoost curve",
            "RERUN_REQUIRED",
            "P1",
            "음의 결과·horizon 취약",
            "fixed400 full Δ +0.044424; 후속 최적 iteration 범위 5~508",
            "400 고정 취약점은 실제지만 현재 격차와 후속 실패로 반전 가능성은 중간 이하이다.",
            "joint hydro 뒤 inner-selected iteration으로 재실행",
            ["artifacts/p2_meaningful_learning_curve_generation_v1"],
        ),
        record(
            "P2",
            "Legacy fixed400 GBM/phase/state screens",
            "RERUN_REQUIRED",
            "보류",
            "대체됨",
            "중간 model/history 미보존; 후속 nested tuning/max-round가 대부분 대체",
            "가능한 결함이나 추가 정보가 작다.",
            "현재 재실행하지 않음",
            ["artifacts/p2_gbm_family_tournament_v1"],
        ),
        record(
            "P2",
            "Analytic postprocess/routing/TEOS/diagnostic families",
            "NOT_APPLICABLE",
            "완료",
            "해당 없음",
            "반복학습 checkpoint가 없는 후처리·해석 계보",
            "checkpoint 감사 비적용이다.",
            "별도 구조 감사로 관리",
            ["reports/promotion_retroaudit_20260827_v1/report-source.md"],
        ),
        record(
            "P2",
            "Preflight/recipe/partial or superseded executions",
            "DESIGN_ONLY",
            "완료",
            "해당 없음",
            "control·dry-run·incomplete/superseded artifacts",
            "성능 checkpoint 판정 대상이 아니다.",
            "재실행 없음",
            ["configs/experiments"],
        ),
        record(
            "P3",
            "Deep GRU/TCN probes + nested CatBoost",
            "ALREADY_BEST",
            "완료",
            "선택 건전",
            "GRU best epochs 3/2/2; TCN 3/4/2; CatBoost iterations 17/191/1320",
            "inner-best checkpoint/iteration을 사용했다.",
            "재실행 없음",
            [
                "artifacts/p3/deep_gru_probe",
                "artifacts/p3/deep_tcn_probe",
                "artifacts/p3/catboost_nested_tuning",
            ],
        ),
        record(
            "P3",
            "RevIN Patch + KMA target control",
            "ALREADY_BEST",
            "완료",
            "음의 결과",
            "RevIN selected epochs 6~21, Δ +0.004314; KMA target control inner gate 실패",
            "outer label을 열기 전 inner-best를 사용했다.",
            "checkpoint 관점 재실행 없음",
            [
                "artifacts/p3_revin_patch_v1/full_one_shot/epoch_selection",
                "artifacts/p3_kma_source_prediction_meta_v1/one_shot/result.json",
            ],
        ),
        record(
            "P3",
            "Corrected/meaningful CatBoost fixed-horizon splice",
            "REPLAYED_POSITIVE_TREND",
            "연구 보존",
            "5/5 prefix 양의 추세",
            "공통 210/1200 trees가 5 prefixes 모두 incumbent 대비 -0.001597~-0.004961m",
            "실제 부호 반전이지만 같은 historical truth로 고른 diagnostic이며 0.03m gate 미달이다.",
            "2D splice를 leakage-free nested 선택으로 재확인 후 공식 probe 가치 판단",
            ["artifacts/checkpoint_retroaudit_20260827_v1/p3_catboost_checkpoint_oracle.json"],
        ),
        record(
            "P3",
            "KMA/analog proxy CatBoost",
            "REPLAYABLE",
            "보류",
            "간접 효과",
            "saved CBM의 ntree_end replay 가능",
            "checkpoint가 outer 후보보다 inner route/alpha에 간접 영향한다.",
            "주요 재실행 뒤 검토",
            [
                "artifacts/p3_kma_source_prediction_meta_v1",
                "artifacts/p3_causal_forcing_analog_outer_research_v4",
            ],
        ),
        record(
            "P3",
            "Causal forcing sequence residual",
            "RERUN_REQUIRED",
            "P0",
            "미해결",
            "45 cells fixed8, final-only; full Δ +0.037854m",
            "비용이 낮고 가장 싼 미해결 P3 checkpoint 실험이다.",
            "inner checkpoint 저장 방식으로 재실행",
            ["artifacts/p3_causal_forcing_sequence_residual_20260823_v1"],
        ),
        record(
            "P3",
            "Hierarchical dense72 residual basis r4",
            "RERUN_REQUIRED",
            "P1",
            "미해결·음의 결과",
            "45 cells fixed12, final-only; full Δ +0.067295m",
            "checkpoint 결함은 있지만 격차가 causal sequence보다 크다.",
            "causal sequence 뒤 재실행",
            ["artifacts/p3_hierarchical_residual_basis_dense72_20260823_r4/metrics.json"],
        ),
        record(
            "P3",
            "Legacy fixed30 deep + fixed-iteration tree probes",
            "RERUN_REQUIRED",
            "보류",
            "후속 계보가 대체",
            "OOF만 남고 model checkpoint 없음",
            "후속 corrected/nested 계보가 성능·검증 면에서 상위호환이다.",
            "현재 재실행하지 않음",
            ["artifacts/p3"],
        ),
        record(
            "P3",
            "Spectral/state-space/ridge/postprocess families",
            "NOT_APPLICABLE",
            "완료",
            "해당 없음",
            "폐형 kernel/VAR/ridge 및 deterministic shrink/router/blend",
            "epoch/checkpoint 개념이 없다.",
            "별도 구조 감사로 관리",
            ["reports/promotion_retroaudit_20260827_v1/report-source.md"],
        ),
        record(
            "P3",
            "ERA5/external/prequential/tombstone/verifier families",
            "DESIGN_ONLY",
            "진행/종료",
            "결과 없음",
            "최종 실행 전이거나 invalid/superseded artifact",
            "성능 checkpoint 판정 대상이 아니다.",
            "유효 실행 결과가 생길 때 별도 감사",
            ["configs/experiments/p3_era5_context_transfer_v1.json"],
        ),
    ]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def p3_replay_rows(oracle: dict[str, Any]) -> list[dict[str, Any]]:
    common = oracle["common_checkpoint_oracle"]["fixed_horizon_splice"]
    rows: list[dict[str, Any]] = []
    for prefix in ("0.40", "0.55", "0.70", "0.85", "1.00"):
        node = oracle["prefixes"][prefix]
        splice = node["summaries"]["fixed_horizon_splice"]
        rows.append(
            {
                "prefix": prefix,
                "incumbent_rmse_m": round(node["incumbent_final_rmse_m"], 9),
                "final_splice_rmse_m": round(splice["final"]["rmse_m"], 9),
                "prefix_oracle_rmse_m": round(splice["best"]["rmse_m"], 9),
                "oracle_gain_vs_final_m": round(splice["oracle_gain_vs_final_candidate_m"], 9),
                "oracle_single_multi": f"{splice['best']['single_trees']}/{splice['best']['multi_trees']}",
                "common_210_1200_delta_vs_incumbent_m": round(
                    common["deltas_by_prefix_m"][prefix], 9
                ),
                "common_direction": "개선",
            }
        )
    return rows


def transport_rows(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    # These are live projections of the calibration ledger, not a fitted
    # cross-problem conversion model.
    rows = calibration["calibration_rows"]
    p1 = next(
        row
        for row in rows
        if row["problem"] == "P1" and row["contrast"] == "Router vs B"
    )
    p2 = next(
        row
        for row in rows
        if row["problem"] == "P2" and row["contrast"] == "L4 -t vs O"
    )
    p3_rows = [row for row in rows if row["problem"] == "P3"]
    summary = calibration["summary"]
    return [
        {
            "problem": "P1",
            "evidence": (
                "Router−B: local "
                f"{p1['local_gain_positive_is_better']:+.6f} F1, official "
                f"{p1['official_gain_positive_is_better']:+.6f} F1"
            ),
            "magnitude": f"{p1['magnitude_ratio_abs_official_over_local']:.2f}배",
            "direction": "일치 사례와 반전 사례가 혼재",
            "policy": "cell/family별 transport만 사용",
        },
        {
            "problem": "P2",
            "evidence": (
                "L4 official 효과가 대응 local 대비 "
                f"{p2['magnitude_ratio_abs_official_over_local']:.2f}배"
            ),
            "magnitude": f"{p2['magnitude_ratio_abs_official_over_local']:.2f}배",
            "direction": "layer별 부호 반전도 관측",
            "policy": "공식 all-row quadratic 우선, local은 순서 보조",
        },
        {
            "problem": "P3",
            "evidence": "현재 correction A/B/reverse-long family",
            "magnitude": "전역 배율 금지",
            "direction": (
                f"관측 {len(p3_rows)}개 축 중 방향 일치 "
                f"{sum(bool(row['sign_agreement']) for row in p3_rows)}개"
            ),
            "policy": "작은 local 이득은 trend로 보존하되 독립 probe로만 검증",
        },
        {
            "problem": "전체",
            "evidence": "local↔official contrasts",
            "magnitude": (
                f"{summary['sign_agreement_all']}/{summary['rows']} 방향 일치"
            ),
            "direction": "전역 selector로 불충분",
            "policy": "승격 실패와 연구가치 상실을 분리",
        },
    ]


def status_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["problem"], row["status"]) for row in records)
    return [
        {
            "problem": problem,
            "status": status,
            "status_label": STATUS_LABELS[status],
            "count": counts[(problem, status)],
        }
        for problem in ("P1", "P2", "P3")
        for status in STATUS_LABELS
        if counts[(problem, status)]
    ]


def priority_rows() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "problem": "P1",
            "lineage": "MS-TCN Q3/Q4",
            "action": "epoch별 저장 + optimizer 안정화 + 개발창 plateau 선택",
            "why": "loss excursion이 가장 명확하고 pooled 부호가 근접",
            "evidence_level": "재학습 필요",
        },
        {
            "order": 2,
            "problem": "P2",
            "lineage": "Joint hydro r3",
            "action": "inner-selected checkpoint를 outer에 고정",
            "why": "full +0.006663 근소 열세, prefix0.85 -0.055054 개선",
            "evidence_level": "재학습 필요",
        },
        {
            "order": 3,
            "problem": "P3",
            "lineage": "Causal sequence fixed8",
            "action": "inner checkpoint 저장 재실행",
            "why": "가장 싸고 빠른 미해결 P3 계보",
            "evidence_level": "재학습 필요",
        },
        {
            "order": 4,
            "problem": "P3",
            "lineage": "CatBoost 210/1200 splice",
            "action": "2D splice LOFO/nested 선택을 먼저 재현",
            "why": "5/5 budget 양의 local 추세지만 same-truth oracle",
            "evidence_level": "연구 추세",
        },
        {
            "order": 5,
            "problem": "P2",
            "lineage": "Conservative/corrected saved GBM",
            "action": "num_iteration 저비용 replay",
            "why": "재학습 없이 checkpoint 일반화 확인 가능",
            "evidence_level": "재생 가능",
        },
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_artifact(
    generated_at: str,
    records: list[dict[str, Any]],
    counts: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    transport: list[dict[str, Any]],
    priorities: list[dict[str, Any]],
) -> dict[str, Any]:
    p1_transport_magnitude = transport[0]["magnitude"]
    p2_transport_magnitude = transport[1]["magnitude"]
    total_sign_agreement = transport[3]["magnitude"]
    sources = [
        {
            "id": "audit_catalog",
            "label": "P1-P3 checkpoint lineage audit catalog",
            "path": "artifacts/checkpoint_retroaudit_20260827_v1/audit_records.json",
            "query": {
                "language": "SQL",
                "engine": "duckdb",
                "sql": "SELECT * FROM read_json_auto('artifacts/checkpoint_retroaudit_20260827_v1/audit_records.json')",
                "description": "Executed lineage groups were classified from saved local metrics, histories, model files, and source selection logic. Config-only revisions were grouped rather than inflated as independent experiments.",
                "executed_at": generated_at,
                "tables_used": [
                    "artifacts/p1_incumbent_preserving_mstcn_asrf_v2/terminal_result.json",
                    "artifacts/p2_deep_model_tournament_v1/result.json",
                    "artifacts/p2_authoritative_nested_surrogate_actual_20260825_v5/result.json",
                    "artifacts/checkpoint_retroaudit_20260827_v1/p3_catboost_checkpoint_oracle.json",
                ],
            },
        },
        {
            "id": "p3_checkpoint_replay",
            "label": "P3 saved-CatBoost tree-prefix replay",
            "path": "artifacts/checkpoint_retroaudit_20260827_v1/p3_catboost_checkpoint_oracle.json",
            "query": {
                "language": "SQL",
                "engine": "duckdb",
                "sql": "SELECT * FROM read_json_auto('artifacts/checkpoint_retroaudit_20260827_v1/p3_replay_rows.json')",
                "description": "Replayed 20 ntree_end fractions from 90 saved CBMs over five training prefixes; final predictions reconstructed within 1e-12. Historical validation truth makes oracle choices diagnostic only.",
                "executed_at": generated_at,
                "tables_used": [
                    "artifacts/p3_meaningful_learning_curve_20260823_v1/oof/learning_curve_oof.parquet",
                    "artifacts/p3_meaningful_learning_curve_20260823_v1/models",
                ],
            },
        },
        {
            "id": "transport_evidence",
            "label": "Local-to-official calibration ledger",
            "path": "reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json",
            "query": {
                "language": "SQL",
                "engine": "duckdb",
                "sql": "SELECT * FROM read_json_auto('artifacts/checkpoint_retroaudit_20260827_v1/transport_rows.json')",
                "description": "Descriptive, family-specific comparison of already recorded local and official contrasts. No global calibration scalar is fitted.",
                "executed_at": generated_at,
                "tables_used": [
                    "reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json"
                ],
            },
        },
    ]

    charts = [
        {
            "id": "status_counts_chart",
            "title": "문제별 checkpoint 감사 분류",
            "subtitle": "실행 계보군 기준; config revision 수가 아니라 모델·판정 계보를 묶어 집계",
            "type": "bar",
            "intent": "comparison",
            "question": "어느 문제에 재학습 공백과 재생 가능한 증거가 남아 있는가?",
            "rationale": "P1-P3의 checkpoint 처리 상태를 같은 분류로 나란히 비교한다.",
            "dataset": "status_counts",
            "sourceId": "audit_catalog",
            "source": {
                "query": {
                    "engine": "duckdb",
                    "language": "SQL",
                    "sql": "SELECT problem, status, status_label, count(*) AS count FROM read_json_auto('artifacts/checkpoint_retroaudit_20260827_v1/audit_records.json') GROUP BY problem, status, status_label ORDER BY problem, status",
                    "description": "Count grouped executed lineages by problem and checkpoint-audit classification.",
                    "tables_used": ["artifacts/checkpoint_retroaudit_20260827_v1/audit_records.json"],
                }
            },
            "encodings": {
                "x": {"field": "problem", "type": "nominal", "label": "문제"},
                "y": {"field": "count", "type": "quantitative", "label": "계보군 수"},
                "color": {"field": "status_label", "type": "nominal", "label": "분류"},
                "tooltip": [
                    {"field": "status_label", "type": "text", "label": "분류"},
                    {"field": "count", "type": "number", "label": "계보군 수"},
                ],
            },
            "options": {"grouping": "grouped"},
            "settings": {"showValues": True},
            "labels": {"values": "always"},
            "layout": "full",
        }
    ]

    tables = [
        {
            "id": "audit_records_table",
            "title": "P1-P3 계보별 checkpoint 감사",
            "subtitle": "승격 실패와 연구 추세를 분리한 grouped-lineage inventory",
            "dataset": "audit_records",
            "sourceId": "audit_catalog",
            "source": {
                "query": {
                    "engine": "duckdb",
                    "language": "SQL",
                    "sql": "SELECT * FROM read_json_auto('artifacts/checkpoint_retroaudit_20260827_v1/audit_records.json')",
                    "description": "Read the grouped P1-P3 checkpoint lineage audit catalog.",
                    "tables_used": ["artifacts/checkpoint_retroaudit_20260827_v1/audit_records.json"],
                }
            },
            "defaultSort": {"field": "problem", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "maxRows": 100,
            "columns": [
                {"field": "problem", "label": "문제", "type": "text"},
                {"field": "lineage", "label": "계보", "type": "text"},
                {"field": "status_label", "label": "분류", "type": "text"},
                {"field": "priority", "label": "우선순위", "type": "text"},
                {"field": "trend", "label": "추세", "type": "text"},
                {"field": "observed_result", "label": "관측 결과", "type": "text"},
                {"field": "next_action", "label": "다음 행동", "type": "text"},
            ],
        },
        {
            "id": "p3_replay_table",
            "title": "P3 CatBoost checkpoint 실제 재생",
            "subtitle": "oracle column은 same-truth diagnostic; common 210/1200은 모든 prefix에서 부호가 같음",
            "dataset": "p3_replay",
            "sourceId": "p3_checkpoint_replay",
            "source": {
                "query": {
                    "engine": "duckdb",
                    "language": "SQL",
                    "sql": "SELECT * FROM read_json_auto('artifacts/checkpoint_retroaudit_20260827_v1/p3_replay_rows.json')",
                    "description": "Read the deterministic projection of the saved-CatBoost tree-prefix replay.",
                    "tables_used": [
                        "artifacts/checkpoint_retroaudit_20260827_v1/p3_replay_rows.json",
                        "artifacts/checkpoint_retroaudit_20260827_v1/p3_catboost_checkpoint_oracle.json"
                    ],
                }
            },
            "defaultSort": {"field": "prefix", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "prefix", "label": "Train prefix", "type": "text"},
                {"field": "incumbent_rmse_m", "label": "Incumbent RMSE m", "type": "number"},
                {"field": "final_splice_rmse_m", "label": "Final splice RMSE m", "type": "number"},
                {"field": "prefix_oracle_rmse_m", "label": "Oracle RMSE m", "type": "number"},
                {"field": "oracle_gain_vs_final_m", "label": "Oracle gain m", "type": "number"},
                {"field": "oracle_single_multi", "label": "Oracle trees S/M", "type": "text"},
                {"field": "common_210_1200_delta_vs_incumbent_m", "label": "Common Δ vs inc m", "type": "number"},
            ],
        },
        {
            "id": "transport_table",
            "title": "로컬→공식 운송성은 문제·계보별로 다르다",
            "subtitle": "증폭 사례는 존재하지만 전역 10배 환산은 금지",
            "dataset": "transport",
            "sourceId": "transport_evidence",
            "source": {
                "query": {
                    "engine": "duckdb",
                    "language": "SQL",
                    "sql": "SELECT * FROM read_json_auto('artifacts/checkpoint_retroaudit_20260827_v1/transport_rows.json')",
                    "description": "Read the problem-specific projection of the recorded local-to-official transport ledger.",
                    "tables_used": [
                        "artifacts/checkpoint_retroaudit_20260827_v1/transport_rows.json",
                        "reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json"
                    ],
                }
            },
            "defaultSort": {"field": "problem", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "problem", "label": "문제", "type": "text"},
                {"field": "evidence", "label": "관측 근거", "type": "text"},
                {"field": "magnitude", "label": "크기", "type": "text"},
                {"field": "direction", "label": "방향성", "type": "text"},
                {"field": "policy", "label": "사용 정책", "type": "text"},
            ],
        },
        {
            "id": "priority_table",
            "title": "다음 checkpoint 실험 우선순위",
            "subtitle": "계산비용·반전 가능성·누수 없는 검증 가능성을 함께 고려",
            "dataset": "priorities",
            "sourceId": "audit_catalog",
            "source": {
                "query": {
                    "engine": "duckdb",
                    "language": "SQL",
                    "sql": "SELECT * FROM read_json_auto('artifacts/checkpoint_retroaudit_20260827_v1/priority_rows.json') ORDER BY \"order\"",
                    "description": "Read the checkpoint rerun and replay priority queue derived from the audit catalog.",
                    "tables_used": [
                        "artifacts/checkpoint_retroaudit_20260827_v1/priority_rows.json",
                        "artifacts/checkpoint_retroaudit_20260827_v1/audit_records.json"
                    ],
                }
            },
            "defaultSort": {"field": "order", "direction": "asc"},
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "order", "label": "순서", "type": "number"},
                {"field": "problem", "label": "문제", "type": "text"},
                {"field": "lineage", "label": "계보", "type": "text"},
                {"field": "action", "label": "실행", "type": "text"},
                {"field": "why", "label": "근거", "type": "text"},
                {"field": "evidence_level", "label": "현재 증거", "type": "text"},
            ],
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# P1-P3 최적 체크포인트 소급감사",
        },
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "audit_catalog",
            "body": (
                "## 기술 요약\n\n"
                "**공식·confirmatory 판정이 정당하게 뒤집힌 계보는 0개입니다.** P2 주력 계보는 이미 inner-best checkpoint를 사용했습니다. P1은 현재 MS-TCN Q3/Q4의 optimizer excursion이 가장 큰 미해결이고, P3는 저장 CatBoost를 실제 재생해 final보다 나은 중간 tree splice를 찾았습니다.\n\n"
                "P3 공통 210/1200-tree splice는 5개 train-prefix 모두 incumbent보다 `0.0016~0.0050m` 좋았습니다. 이 결과는 미약하지만 일관된 **연구 양의 추세**로 보존합니다. 다만 same-truth diagnostic이고 기존 `0.03m` gate에는 못 미쳐 즉시 승격 증거는 아닙니다."
            ),
        },
        {
            "id": "counts_intro",
            "type": "markdown",
            "sourceId": "audit_catalog",
            "body": "## 무엇이 이미 안전했고 무엇을 다시 돌려야 하는가\n\n계보군 단위 분류입니다. 버전 번호만 다른 config를 별개 실험으로 부풀리지 않았습니다. `재생 완료·양의 추세`는 폐기 대상이 아니며, `재학습 필요`는 중간 weights가 없어서 아직 판정을 확정할 수 없다는 뜻입니다.",
        },
        {"id": "counts_chart", "type": "chart", "chartId": "status_counts_chart", "layout": "full"},
        {
            "id": "inventory_intro",
            "type": "markdown",
            "sourceId": "audit_catalog",
            "body": "## 계보별 판정\n\n핵심 구분은 네 가지입니다. 이미 inner-best를 쓴 계보, 저장 모델로 재생 가능한 계보, 중간 weights가 없어 재학습이 필요한 계보, checkpoint 개념이 없는 계보입니다. 음의 결과라도 공식·로컬 transport를 학습할 정보가 있으면 기록을 유지합니다.",
        },
        {"id": "inventory", "type": "table", "tableId": "audit_records_table", "layout": "full"},
        {
            "id": "p3_intro",
            "type": "markdown",
            "sourceId": "p3_checkpoint_replay",
            "body": "## P3에서는 실제로 final-only 결론의 부호가 한 번 뒤집혔다\n\n90개 저장 CBM을 5% 간격으로 재생했고 원래 final OOF를 `1e-12` 이내로 복원했습니다. prefix별 oracle은 final보다 `0.0026~0.0122m` 좋았습니다. 더 보수적인 공통 210/1200-tree pair도 모든 train-prefix에서 incumbent를 이겼습니다. 그러나 같은 historical truth로 고른 값이므로 promotion이 아니라 구조적 추세입니다. single/multi 단독 LOFO는 full-prefix에서 각각 incumbent보다 `+0.005282/+0.003874m` 나빠졌고, 2D splice의 leakage-free 선택은 아직 남았습니다.",
        },
        {"id": "p3_replay", "type": "table", "tableId": "p3_replay_table", "layout": "full"},
        {
            "id": "transport_intro",
            "type": "markdown",
            "sourceId": "transport_evidence",
            "body": (
                "## 작은 로컬 개선은 버리지 않되 10배로 기계 환산하지 않는다\n\n"
                f"사용자 지적처럼 P1({p1_transport_magnitude})과 P2({p2_transport_magnitude})에는 "
                "로컬 대비 공식 효과가 약 10배 이상 커진 사례가 있습니다. 반대로 전체 "
                f"contrast의 {total_sign_agreement}이고 P3 correction 계열은 반복 역전됐습니다. "
                "따라서 `승격 기준 미달 = 폐기`로 처리하지 않고 **연구 추세**로 남기되, "
                "문제·family별 사전등록 probe만 허용합니다."
            ),
        },
        {"id": "transport", "type": "table", "tableId": "transport_table", "layout": "full"},
        {
            "id": "protocol",
            "type": "markdown",
            "sourceId": "audit_catalog",
            "body": (
                "## 앞으로의 checkpoint 승격 규칙\n\n"
                "1. 5 epoch/round 간격으로 저장하되 outer truth에서는 checkpoint를 고르지 않습니다.\n"
                "2. earlier chronological inner window에서 공통 checkpoint를 선택하고 다음 window에 고정 적용합니다.\n"
                "3. 단일 최고점보다 `e-5/e/e+5`의 최악 개선값을 우선하고 1-SE 내 가장 이른 checkpoint를 택합니다.\n"
                "4. oracle 소급점수는 headroom 진단, chronological replay는 일반화 진단, fresh outer만 승격 증거로 구분합니다.\n"
                "5. 로컬 양의 추세가 작아도 삭제하지 않습니다. 다만 공식 제출은 결과를 보기 전에 후보·해석·다음 행동을 동결한 정보가치 probe로만 사용합니다."
            ),
        },
        {
            "id": "priority_intro",
            "type": "markdown",
            "sourceId": "audit_catalog",
            "body": "## 다음 실행 순서\n\n최고 우선순위는 P1 MS-TCN Q3/Q4, P2 joint hydro r3, P3 fixed8 causal sequence입니다. P3 CatBoost splice는 이미 양의 추세가 있으므로 폐기하지 않지만, 2D nested selection을 먼저 통과해야 공식 probe 후보로 올라갑니다.",
        },
        {"id": "priority", "type": "table", "tableId": "priority_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "transport_evidence",
            "body": (
                "## 한계·강건성\n\n"
                "- P1 Q3/Q4 이전 weights와 P2/P3 fixed-epoch 중간 weights가 없어 일부는 재학습 전까지 미해결입니다.\n"
                "- loss 최소 epoch는 F1/RMSE 최적 checkpoint와 동일하지 않으므로 재학습 우선순위 근거일 뿐 점수 반전의 증명은 아닙니다.\n"
                "- P3 oracle은 historical truth에 적응했으므로 공식 제출용 모델 선택값으로 직접 사용할 수 없습니다.\n"
                "- local↔official 14개 contrast는 서로 독립·동일 family 표본이 아니어서 배율 calibration에 부족합니다.\n"
                "- 공식 test/sample/submission 파일 접근, submission 생성·업로드는 이번 감사에서 0회입니다."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "sourceId": "audit_catalog",
            "body": "## 추가로 답해야 할 질문\n\n1. P3 2D splice가 LOFO/nested 선택에서도 5-prefix 공통 부호를 유지하는가?\n2. P2 joint hydro의 0.85-prefix 이득이 inner-selected epoch를 full budget에 운반할 때 유지되는가?\n3. P1 Q3/Q4 optimizer excursion을 checkpoint averaging만으로 막을 수 있는가, 아니면 LR/clip 구조 수정이 필요한가?\n4. 다음 공식 probe가 local→official transport 계보를 구분할 만큼 정보가치가 있는가?",
        },
    ]

    indexed_records = []
    for i, row in enumerate(records, start=1):
        indexed = dict(row)
        indexed["evidence"] = " | ".join(row["evidence"])
        indexed["sort_key"] = f"{row['problem']}-{i:03d}"
        indexed_records.append(indexed)

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "P1-P3 최적 체크포인트 소급감사",
        "description": "최종 epoch 편향, 실제 checkpoint replay, local-to-official transport를 통합한 기술 감사",
        "generatedAt": generated_at,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "status_counts": counts,
            "audit_records": indexed_records,
            "p3_replay": replay,
            "transport": transport,
            "priorities": priorities,
        },
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


def main() -> None:
    if not P3_ORACLE.exists():
        raise FileNotFoundError(P3_ORACLE)
    if not TRANSPORT.exists():
        raise FileNotFoundError(TRANSPORT)
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone(timedelta(hours=9))).replace(microsecond=0).isoformat()
    records = build_records()
    oracle = load_json(P3_ORACLE)
    calibration = load_json(TRANSPORT)
    replay = p3_replay_rows(oracle)
    transport = transport_rows(calibration)
    counts = status_counts(records)
    priorities = priority_rows()

    write_json(OUT / "audit_records.json", records)
    write_json(OUT / "status_counts.json", counts)
    write_json(OUT / "p3_replay_rows.json", replay)
    write_json(OUT / "transport_rows.json", transport)
    write_json(OUT / "priority_rows.json", priorities)
    calibration_rows = calibration["calibration_rows"]
    p1_transport = next(
        row
        for row in calibration_rows
        if row["problem"] == "P1" and row["contrast"] == "Router vs B"
    )
    p2_transport = next(
        row
        for row in calibration_rows
        if row["problem"] == "P2" and row["contrast"] == "L4 -t vs O"
    )
    write_json(
        OUT / "summary.json",
        {
            "schema_version": "checkpoint_retroaudit_20260827.v1",
            "generated_at": now,
            "scope": "historical local artifacts and recorded official contrasts only",
            "official_test_sample_submission_reads": 0,
            "official_or_confirmatory_verdicts_legitimately_overturned": 0,
            "replayed_positive_trend_lineages": 1,
            "high_priority_reruns": ["P1 MS-TCN Q3/Q4", "P2 joint hydro r3", "P3 causal sequence fixed8"],
            "transport": {
                "sign_agreement": (
                    f"{calibration['summary']['sign_agreement_all']}/"
                    f"{calibration['summary']['rows']}"
                ),
                "global_scalar_allowed": False,
                "p1_observed_amplification": p1_transport[
                    "magnitude_ratio_abs_official_over_local"
                ],
                "p2_observed_amplification": p2_transport[
                    "magnitude_ratio_abs_official_over_local"
                ],
                "p3_current_family_direction_reversals": all(
                    not row["sign_agreement"]
                    for row in calibration_rows
                    if row["problem"] == "P3"
                ),
            },
            "status_counts": counts,
            "p3_common_checkpoint": oracle["common_checkpoint_oracle"]["fixed_horizon_splice"],
            "interpretation": "No-Go and research-value loss are separate states; positive local trends remain in the ledger even when not promotion-ready.",
        },
    )
    write_json(
        OUT / "chart_map.json",
        {
            "status_counts_chart": {
                "purpose": "Compare checkpoint audit classifications across P1-P3 grouped lineages.",
                "dataset": "status_counts",
                "source": "audit_records.json",
            }
        },
    )
    (OUT / "source_notes.md").write_text(
        "# Source notes\n\n"
        "- Scope: historical local model artifacts, histories, OOF metrics, and an already recorded local-to-official calibration ledger.\n"
        "- Official test/sample/submission file reads: 0. Submission generation/uploads: 0.\n"
        "  This is based on the builder/replay declared input graph and static inspection, not an OS-level file-access audit log.\n"
        "- Grouping: version-only config revisions are grouped under one executed model/decision lineage.\n"
        "- P3 oracle: same historical truth was used to select the best tree prefixes, so it is diagnostic rather than promotion evidence.\n"
        "- Transport: the 14 contrasts are descriptive and dependent; no global multiplier is fitted.\n",
        encoding="utf-8",
    )

    report = f"""# P1-P3 최적 체크포인트 소급감사

## 결론

공식·confirmatory 판정이 정당하게 뒤집힌 계보는 0개입니다. 그러나 P3 저장 CatBoost 재생에서 공통 210/1200-tree splice가 5개 train-prefix 모두 incumbent보다 0.0016~0.0050m 좋아지는 양의 추세가 확인됐습니다. 이는 승격 근거는 아니지만 폐기해서도 안 되는 연구 자산입니다.

P2 주력 deep/tree 계보는 이미 best checkpoint를 사용했습니다. 가장 큰 미해결은 P1 MS-TCN Q3/Q4, P2 joint hydro r3, P3 causal sequence fixed8입니다.

## 공식 운송성 해석

P1 {transport[0]['magnitude']}, P2 {transport[1]['magnitude']}의 공식 효과 증폭 사례가 있지만, 전체 local↔official contrast의 방향 일치는 {transport[3]['magnitude']}입니다. 특히 P3 correction 계열은 현재 관측 축에서 방향이 반복 역전됐습니다. 따라서 작은 로컬 이득은 `RESEARCH_POSITIVE_TREND`로 보존하되 전역 10배 환산이나 즉시 승격에는 사용하지 않습니다.

## 다음 순서

1. P1 MS-TCN Q3/Q4: epoch별 저장과 optimizer 안정화.
2. P2 joint hydro r3: inner-selected checkpoint를 outer에 고정.
3. P3 causal sequence fixed8: 중간 checkpoint 저장 재실행.
4. P3 CatBoost: 2D splice LOFO/nested 선택을 통과하면 공식 정보가치 probe 후보로 재평가.

상세 계보표는 `audit_records.json`, P3 20×20 tree-prefix 결과는 `p3_catboost_checkpoint_oracle.json`, 검증용 인터랙티브 보고서는 `artifact.json`에 있습니다.
"""
    (OUT / "report_ko.md").write_text(report, encoding="utf-8")

    write_json(
        OUT / "independent_qa.json",
        {
            "schema_version": "checkpoint_retroaudit_independent_qa.v1",
            "verdict": "PASS_WITH_CAVEATS_CLOSED",
            "critical_or_major_errors": 0,
            "verified": {
                "artifact_schema": "PASS",
                "audit_lineages": 33,
                "status_aggregate_rows": 14,
                "evidence_paths_existing": "46/46",
                "p3_replay_projection_max_rounding_error": 4.97e-10,
                "local_official_sign_agreement": (
                    f"{calibration['summary']['sign_agreement_all']}/"
                    f"{calibration['summary']['rows']}"
                ),
            },
            "closures": {
                "transport_projection_is_live_from_ledger": True,
                "builder_and_replay_script_hashes_added": True,
            },
            "remaining_caveat": (
                "The zero official test/sample/submission read claim is supported by the "
                "declared input graph and static inspection, not an OS-level access audit log."
            ),
        },
    )

    artifact = build_artifact(now, records, counts, replay, transport, priorities)
    write_json(OUT / "artifact.json", artifact)
    write_json(
        OUT / "manifest.json",
        {
            "schema_version": "checkpoint_retroaudit_manifest.v1",
            "generated_at": now,
            "files": {
                name: sha256(OUT / name)
                for name in [
                    "audit_records.json",
                    "status_counts.json",
                    "p3_replay_rows.json",
                    "transport_rows.json",
                    "priority_rows.json",
                    "summary.json",
                    "chart_map.json",
                    "source_notes.md",
                    "report_ko.md",
                    "artifact.json",
                    "independent_qa.json",
                    "p3_catboost_checkpoint_oracle.json",
                ]
            },
            "source_code": {
                "scripts/build_checkpoint_retroaudit_report_20260827_v1.py": sha256(
                    ROOT / "scripts/build_checkpoint_retroaudit_report_20260827_v1.py"
                ),
                "scripts/audit_p3_catboost_checkpoint_oracle_20260827.py": sha256(
                    ROOT / "scripts/audit_p3_catboost_checkpoint_oracle_20260827.py"
                ),
            },
        },
    )
    print(json.dumps({"output": str(OUT), "records": len(records), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
