"""Build a portable technical report for P3 failure-mode and method reconnaissance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _source(
    source_id: str,
    label: str,
    path: str,
    rows: list[dict[str, object]],
    description: str,
) -> dict[str, object]:
    columns = list(rows[0])
    sql = " UNION ALL ".join(
        "SELECT " + ", ".join(f"{_literal(row.get(column))} AS {column}" for column in columns)
        for row in rows
    )
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "tables_used": [path],
            "filters": [
                "182 independent local validation cases",
                "Six official lead times and pooled row RMSE",
                "Case-local past 48 hours only",
                "No external observations or hidden test labels",
            ],
        },
    }


def build_artifact(root: Path) -> dict[str, object]:
    failure_path = root / "artifacts/p3/failure_recon/diagnostics.json"
    bias_path = root / "artifacts/p3/bias_correction_probe/metrics.json"
    router_path = root / "artifacts/p3/trajectory_router_probe/metrics.json"
    sea_state_path = root / "artifacts/p3/sea_state_probe/metrics.json"
    sea_state_validation_path = root / "artifacts/p3/sea_state_probe/paired_validation.json"
    amplitude_path = root / "artifacts/p3/amplitude_weight_probe/metrics.json"
    amplitude_validation_path = root / "artifacts/p3/amplitude_weight_probe/paired_validation.json"
    validation_path = root / "artifacts/p3/failure_recon/validation.json"
    failure = _read(failure_path)
    bias = _read(bias_path)
    router = _read(router_path)
    sea_state = _read(sea_state_path)
    sea_state_validation = _read(sea_state_validation_path)
    amplitude = _read(amplitude_path)
    amplitude_validation = _read(amplitude_validation_path)
    validation = _read(validation_path)

    ensemble = float(failure["overall"]["ensemble"]["rmse"])
    persistence = float(failure["overall"]["persistence"]["rmse"])
    case_oracle = float(failure["overall"]["case_oracle_rmse_unimplementable"])
    domain_auc = float(failure["data_shift"]["domain_classifier"]["mean_auc"])
    bias_corrected = float(bias["metrics"]["corrected"]["rmse"])
    sea_state_delta = float(sea_state["metrics"]["delta_candidate_minus_base"]["rmse"])
    amplitude_delta = float(amplitude["metrics"]["delta_candidate_minus_base"]["rmse"])
    router_balanced = float(router["overall"]["balanced_accuracy"])
    router_macro_f1 = float(router["overall"]["macro_f1"])
    router_gate = router_balanced >= float(
        router["interpretation_gate"]["minimum_balanced_accuracy_for_moe_followup"]
    ) and router_macro_f1 >= float(
        router["interpretation_gate"]["minimum_macro_f1_for_moe_followup"]
    )

    headline = [
        {
            "ensemble_rmse": ensemble,
            "persistence_rmse": persistence,
            "case_oracle_rmse": case_oracle,
            "case_oracle_gap": ensemble - case_oracle,
            "domain_auc": domain_auc,
            "bias_corrected_rmse": bias_corrected,
            "router_balanced_accuracy": router_balanced,
            "router_macro_f1": router_macro_f1,
            "router_gate_passed": router_gate,
            "sea_state_delta_rmse": sea_state_delta,
            "amplitude_weight_delta_rmse": amplitude_delta,
        }
    ]
    lead_rows = [
        {
            "lead_h": int(row["segment"]),
            "rmse": float(row["rmse"]),
            "bias": float(row["bias"]),
            "squared_error_share_pct": 100.0 * float(row["squared_error_share"]),
            "rows": int(row["rows"]),
        }
        for row in failure["cuts"]["lead"]
    ]
    trajectory_rows: list[dict[str, object]] = []
    for row in failure["cuts"]["future_trajectory_research_only"]:
        for method, field in (("Frozen ensemble", "rmse"), ("Persistence", "persistence_rmse")):
            trajectory_rows.append(
                {
                    "trajectory": str(row["segment"]),
                    "method": method,
                    "rmse": float(row[field]),
                    "bias": float(row["bias"]) if method == "Frozen ensemble" else None,
                    "cases": int(row["rows"]) // 6,
                    "squared_error_share_pct": 100.0 * float(row["squared_error_share"]),
                }
            )
    shift_rows = [
        {
            "feature": str(row["feature"]),
            "ks_statistic": float(row["ks_statistic"]),
            "standardized_mean_difference": float(row["standardized_mean_difference"]),
            "validation_finite_pct": 100.0 * float(row["validation_finite_share"]),
            "test_finite_pct": 100.0 * float(row["test_finite_share"]),
        }
        for row in failure["data_shift"]["feature_shift"][:8]
    ]
    bias_rows = [
        {
            "fold": fold,
            "inner_correction_m": float(bias["selections"][fold]["correction_m"]),
            "inner_raw_bias_m": float(bias["selections"][fold]["inner_raw_bias_m"]),
            "frozen_outer_rmse": float(values["frozen_rmse"]),
            "corrected_outer_rmse": float(values["corrected_rmse"]),
            "outer_delta_rmse": float(values["corrected_rmse"] - values["frozen_rmse"]),
        }
        for fold, values in bias["metrics"]["folds"].items()
    ]
    fold_router_rows = [
        {
            "fold": fold,
            "cases": int(values["validation_cases"]),
            "balanced_accuracy": float(values["balanced_accuracy"]),
            "macro_f1": float(values["macro_f1"]),
            "log_loss": float(values["log_loss"]),
        }
        for fold, values in router["folds"].items()
    ]
    ablation_rows = [
        {
            "experiment": "Global additive correction",
            "base_rmse": ensemble,
            "candidate_rmse": bias_corrected,
            "delta_rmse": bias_corrected - ensemble,
            "ci90_low": None,
            "ci90_high": None,
            "probability_improved": None,
            "decision": "Reject",
        },
        {
            "experiment": "Inverse-wave-age proxy features",
            "base_rmse": float(sea_state["metrics"]["base_gpu"]["rmse"]),
            "candidate_rmse": float(sea_state["metrics"]["sea_state_gpu"]["rmse"]),
            "delta_rmse": sea_state_delta,
            "ci90_low": float(sea_state_validation["paired_case_bootstrap"]["ci90"][0]),
            "ci90_high": float(sea_state_validation["paired_case_bootstrap"]["ci90"][1]),
            "probability_improved": float(
                sea_state_validation["paired_case_bootstrap"]["probability_candidate_improved"]
            ),
            "decision": "Reject",
        },
        {
            "experiment": "Amplitude-emphasis training weight",
            "base_rmse": float(amplitude["metrics"]["base_gpu"]["rmse"]),
            "candidate_rmse": float(amplitude["metrics"]["amplitude_weight_gpu"]["rmse"]),
            "delta_rmse": amplitude_delta,
            "ci90_low": float(amplitude_validation["paired_case_bootstrap"]["ci90"][0]),
            "ci90_high": float(amplitude_validation["paired_case_bootstrap"]["ci90"][1]),
            "probability_improved": float(
                amplitude_validation["paired_case_bootstrap"]["probability_candidate_improved"]
            ),
            "decision": "Reject",
        },
    ]
    priority_rows = [
        {
            "priority": 1,
            "candidate": "Cross-fitted component-loss soft router",
            "local_evidence": (
                f"Case-oracle gap {ensemble - case_oracle:.3f} m; hard outcome router failed, so train "
                "directly on cross-fitted single/multi/persistence loss differences"
            ),
            "decision": "Next bounded structural experiment",
        },
        {
            "priority": 2,
            "candidate": "Compact inverse-wave-age / wind-sea state features",
            "local_evidence": f"Matched ΔRMSE {sea_state_delta:+.4f} m; CI crosses zero",
            "decision": "Reject current eight-feature form",
        },
        {
            "priority": 3,
            "candidate": "Large-amplitude sample weighting",
            "local_evidence": f"Matched ΔRMSE {amplitude_delta:+.4f} m; CI crosses zero",
            "decision": "Reject current fixed weight",
        },
        {
            "priority": 4,
            "candidate": "N-HiTS or PatchTST revisit",
            "local_evidence": "Existing GRU/TCN and forced-long training did not beat CatBoost",
            "decision": "Defer until routing headroom is exhausted",
        },
    ]

    sources = [
        _source(
            "headline_source",
            "Validated P3 reconnaissance headline",
            "artifacts/p3/failure_recon/validation.json",
            headline,
            "Reconciles frozen RMSE, oracle diagnostic, shift AUC, and two follow-up probes.",
        ),
        _source(
            "lead_source",
            "P3 error concentration by lead",
            "artifacts/p3/failure_recon/diagnostics.json",
            lead_rows,
            "Materializes pooled error contribution and bias for each official lead.",
        ),
        _source(
            "trajectory_source",
            "Research-only future trajectory error cuts",
            "artifacts/p3/failure_recon/diagnostics.json",
            trajectory_rows,
            "Outcome-defined diagnostic used only to locate failure modes, never as inference input.",
        ),
        _source(
            "shift_source",
            "Validation-to-test covariate shift",
            "artifacts/p3/failure_recon/diagnostics.json",
            shift_rows,
            "Top label-free feature distribution shifts by two-sample KS statistic.",
        ),
        _source(
            "bias_source",
            "Inner-only additive correction probe",
            "artifacts/p3/bias_correction_probe/metrics.json",
            bias_rows,
            "Shows each fold's past calibration correction and frozen outer effect.",
        ),
        _source(
            "router_source",
            "Past-only trajectory router feasibility",
            "artifacts/p3/trajectory_router_probe/metrics.json",
            fold_router_rows,
            "Chronological classification using case-local past features only.",
        ),
        _source(
            "ablation_source",
            "Matched structural ablation decisions",
            "artifacts/p3/*_probe/{metrics,paired_validation}.json",
            ablation_rows,
            "Reconciles the three bounded follow-up experiments and paired uncertainty where available.",
        ),
        _source(
            "priority_source",
            "Evidence-ranked next experiments",
            "artifacts/p3/failure_recon/diagnostics.json",
            priority_rows,
            "Separates tested failures from bounded next structural experiments.",
        ),
    ]

    cards = [
        {
            "id": "frozen_rmse",
            "description": "Frozen local candidate on 182 independent cases",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {"label": "Frozen RMSE", "field": "ensemble_rmse", "format": "number", "unit": " m"}
            ],
        },
        {
            "id": "oracle_gap",
            "description": "Research-only case router upper-bound gap",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Case-oracle gap",
                    "field": "case_oracle_gap",
                    "format": "number",
                    "unit": " m",
                }
            ],
        },
        {
            "id": "shift_auc",
            "description": "5-fold validation-vs-test covariate classifier",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [{"label": "Domain AUC", "field": "domain_auc", "format": "number"}],
        },
        {
            "id": "router_score",
            "description": "Past-only four-state chronological probe",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [
                {
                    "label": "Router balanced accuracy",
                    "field": "router_balanced_accuracy",
                    "format": "percent",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "lead_error_chart",
            "title": "Squared-error contribution by forecast lead",
            "subtitle": "Share of frozen ensemble squared error across 1,092 rows; shares sum to 100%.",
            "type": "bar",
            "dataset": "leads",
            "sourceId": "lead_source",
            "encodings": {
                "x": {"field": "lead_h", "type": "ordinal", "label": "Lead (hours)"},
                "y": {
                    "field": "squared_error_share_pct",
                    "type": "quantitative",
                    "label": "Squared-error share (%)",
                },
                "tooltip": [
                    {"field": "rmse", "type": "quantitative", "label": "RMSE (m)"},
                    {"field": "bias", "type": "quantitative", "label": "Bias (m)"},
                ],
            },
        },
        {
            "id": "trajectory_chart",
            "title": "RMSE by future trajectory class",
            "subtitle": "Outcome-defined research diagnostic only; classes are never available at inference.",
            "type": "bar",
            "dataset": "trajectories",
            "sourceId": "trajectory_source",
            "settings": {"groupMode": "grouped"},
            "encodings": {
                "x": {"field": "trajectory", "type": "nominal", "label": "Future trajectory"},
                "y": {"field": "rmse", "type": "quantitative", "label": "RMSE (m)"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "cases", "type": "quantitative", "label": "Cases"},
                    {
                        "field": "squared_error_share_pct",
                        "type": "quantitative",
                        "label": "Frozen error share (%)",
                    },
                ],
            },
        },
        {
            "id": "shift_chart",
            "title": "Largest validation-to-test feature shifts",
            "subtitle": "Two-sample KS on label-free case features; higher means greater distribution difference.",
            "type": "bar",
            "dataset": "shift",
            "sourceId": "shift_source",
            "encodings": {
                "x": {"field": "feature", "type": "nominal", "label": "Feature"},
                "y": {"field": "ks_statistic", "type": "quantitative", "label": "KS statistic"},
                "tooltip": [
                    {
                        "field": "standardized_mean_difference",
                        "type": "quantitative",
                        "label": "Standardized mean difference",
                    }
                ],
            },
        },
    ]
    tables = [
        {
            "id": "bias_table",
            "title": "Inner-only additive bias correction by outer fold",
            "subtitle": "The correction is learned on the preceding 60-day calibration block only.",
            "dataset": "bias",
            "sourceId": "bias_source",
            "columns": [
                {"field": "fold", "label": "Fold", "type": "text"},
                {"field": "inner_correction_m", "label": "Correction (m)", "type": "number"},
                {"field": "frozen_outer_rmse", "label": "Frozen RMSE", "type": "number"},
                {"field": "corrected_outer_rmse", "label": "Corrected RMSE", "type": "number"},
                {
                    "field": "outer_delta_rmse",
                    "label": "ΔRMSE",
                    "type": "number",
                    "movement": True,
                },
            ],
        },
        {
            "id": "router_table",
            "title": "Past-only trajectory router by chronological fold",
            "subtitle": "Four outcome classes; random balanced accuracy is 25%.",
            "dataset": "router",
            "sourceId": "router_source",
            "columns": [
                {"field": "fold", "label": "Fold", "type": "text"},
                {"field": "cases", "label": "Cases", "type": "number"},
                {
                    "field": "balanced_accuracy",
                    "label": "Balanced accuracy",
                    "type": "percent",
                },
                {"field": "macro_f1", "label": "Macro F1", "type": "percent"},
                {"field": "log_loss", "label": "Log loss", "type": "number"},
            ],
        },
        {
            "id": "priority_table",
            "title": "Evidence-ranked next experiments",
            "subtitle": "Only one structural hypothesis should be promoted at a time.",
            "dataset": "priorities",
            "sourceId": "priority_source",
            "columns": [
                {"field": "priority", "label": "Priority", "type": "number"},
                {"field": "candidate", "label": "Candidate", "type": "text"},
                {"field": "local_evidence", "label": "Local evidence", "type": "text"},
                {"field": "decision", "label": "Decision", "type": "text"},
            ],
        },
        {
            "id": "ablation_table",
            "title": "Bounded follow-up experiments",
            "subtitle": "Positive ΔRMSE is worse; matched GPU arms use identical rows and seeds.",
            "dataset": "ablations",
            "sourceId": "ablation_source",
            "columns": [
                {"field": "experiment", "label": "Experiment", "type": "text"},
                {"field": "base_rmse", "label": "Base RMSE", "type": "number"},
                {"field": "candidate_rmse", "label": "Candidate RMSE", "type": "number"},
                {
                    "field": "delta_rmse",
                    "label": "ΔRMSE",
                    "type": "number",
                    "movement": True,
                },
                {"field": "ci90_low", "label": "CI90 low", "type": "number"},
                {"field": "ci90_high", "label": "CI90 high", "type": "number"},
                {"field": "decision", "label": "Decision", "type": "text"},
            ],
        },
    ]

    router_decision = (
        "사전 기준을 통과했지만, 다음 단계는 outcome class가 아니라 component loss를 직접 예측하는 soft router다."
        if router_gate
        else "과거 48시간 hard 상태 라우터는 사전 기준을 넘지 못했다. 4개 class expert는 중단하고 component loss를 직접 예측하는 soft router만 남긴다."
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": "# P3 유의파고 예측 돌파 정찰"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "headline_source",
            "body": (
                "## 기술 요약: 더 큰 단일 모델보다 해상 상태 전환을 구분하는 구조가 핵심이다\n\n"
                f"동일 182개 case의 frozen 앙상블은 **{ensemble:.6f}m**, persistence는 "
                f"**{persistence:.6f}m**다. 그러나 case별로 single·multi·persistence 중 최적 하나를 "
                f"사후 선택하는 구현 불가 oracle은 **{case_oracle:.6f}m**여서 조합 가능한 여지가 "
                f"**{ensemble - case_oracle:.6f}m** 남는다. {router_decision}"
            ),
        },
        {
            "id": "cards",
            "type": "metric-strip",
            "cardIds": ["frozen_rmse", "oracle_gap", "shift_auc", "router_score"],
        },
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 범위·정의·데이터 계약\n\n"
                "공식 pooled RMSE와 동일하게 6개 lead의 모든 행을 합쳐 계산했다. 검증은 3개 고정 시간 외삽 "
                "구간에서 정점별 78시간 이상 떨어진 182 case×6=1,092행이다. 모든 실제 예측 특징은 같은 "
                "case의 과거 48시간에만 의존한다. 미래 trajectory class는 실패 위치를 설명하거나 training "
                "target을 만들 때만 쓰며 inference 입력에는 포함하지 않는다. 외부 관측과 test 절대시각은 0건 사용했다."
            ),
        },
        {"id": "lead_error_block", "type": "chart", "chartId": "lead_error_chart"},
        {
            "id": "lead_finding",
            "type": "markdown",
            "sourceId": "lead_source",
            "body": (
                "## 장기 lead가 총 오차의 대부분을 만든다\n\n"
                "+12·+18·+24시간이 전체 squared error의 약 **63%**를 차지한다. +18시간 RMSE는 "
                "0.920m, bias는 -0.233m다. 따라서 +3시간을 조금 더 정교하게 맞추는 모델보다 미래의 "
                "성장·정점·감쇠 전환을 장기 lead에 맞게 분기하는 구조가 기대값이 크다."
            ),
        },
        {"id": "trajectory_block", "type": "chart", "chartId": "trajectory_chart"},
        {
            "id": "trajectory_finding",
            "type": "markdown",
            "sourceId": "trajectory_source",
            "body": (
                "## peak-then-decay와 continued-growth가 모델의 평균회귀를 드러낸다\n\n"
                "미래 정답으로만 사후 분류했을 때 peak-then-decay는 frozen squared error의 약 60%, "
                "continued-growth는 약 14%를 만든다. 두 상태에서 bias는 약 -0.55m다. 반면 decay에서는 "
                "+0.34m로 부호가 반대다. 이 때문에 전역 절편 보정은 구조적으로 불안정하다."
            ),
        },
        {"id": "bias_block", "type": "table", "tableId": "bias_table"},
        {
            "id": "bias_finding",
            "type": "markdown",
            "sourceId": "bias_source",
            "body": (
                "## 한 개 절편 보정은 시간 외삽에 실패했다\n\n"
                f"각 outer-train의 마지막 60일에서만 추정한 절편을 frozen 앙상블에 적용했지만 RMSE가 "
                f"**{ensemble:.6f} → {bias_corrected:.6f}m**로 악화했다. 첫 fold의 과거 calibration은 "
                "-0.35m를 요구했으나 outer에서는 이미 과소예측이었다. 이 계열은 기각한다."
            ),
        },
        {"id": "shift_block", "type": "chart", "chartId": "shift_chart"},
        {
            "id": "shift_finding",
            "type": "markdown",
            "sourceId": "shift_source",
            "body": (
                "## hidden test 입력은 로컬 validation과 완전히 같지 않다\n\n"
                f"17개 사전고정 물리 특징으로 validation/test를 구분한 AUC는 **{domain_auc:.3f}**다. "
                "가장 큰 차이는 현재 평균 파주기, 12시간 파고 변동성, 12시간 평균 풍속이다. "
                "이는 hidden target shift를 증명하지는 않지만, 계절·폭풍 표본이 달라질 가능성을 수치로 보여준다."
            ),
        },
        {"id": "router_block", "type": "table", "tableId": "router_table"},
        {"id": "ablation_block", "type": "table", "tableId": "ablation_table"},
        {
            "id": "ablation_finding",
            "type": "markdown",
            "sourceId": "ablation_source",
            "body": (
                "## 세 개의 단순 보정은 모두 기각됐다\n\n"
                f"전역 절편은 ΔRMSE **{bias_corrected - ensemble:+.6f}m**, inverse-wave-age proxy는 "
                f"**{sea_state_delta:+.6f}m**, 큰 진폭 sample weight는 **{amplitude_delta:+.6f}m**였다. "
                "후자의 90% paired CI도 모두 0을 걸쳤고 개선확률은 각각 35.7%, 30.5%였다. "
                "따라서 단순한 위쪽 bias 이동이나 feature 한 묶음으로는 상태 혼합을 풀 수 없다."
            ),
        },
        {
            "id": "literature",
            "type": "markdown",
            "body": (
                "## 1차 문헌에서 가져온 구조적 단서\n\n"
                "[MoLE (AISTATS 2024)](https://proceedings.mlr.press/v238/ni24a.html)는 서로 다른 시간 패턴에 "
                "전문가를 특화하고 router가 soft 결합하는 구조로 다수 장기예측 설정에서 단일 linear head를 개선했다. "
                "[JGR Oceans의 해상 상태 분류](https://doi.org/10.1029/2023JC020686)는 풍속·파고·inverse wave age·"
                "spectral width의 다변량 조합이 단순 wind-sea/swell 이분법보다 전이 상태를 더 잘 표현함을 보였다. "
                "P3에는 peak period와 spectrum이 없으므로 `tp` 기반 inverse-wave-age는 근사 feature로만 시험해야 한다.\n\n"
                "[N-HiTS](https://doi.org/10.1609/aaai.v37i6.25854)는 multi-rate sampling과 hierarchical "
                "interpolation, [PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol)는 patching과 channel-independent "
                "attention을 제안한다. 하지만 우리 실제 GRU/TCN 장기학습 실패와 독립 case 수를 고려하면 곧바로 더 큰 "
                "딥 모델을 돌리는 우선순위는 낮다. [2026 attention-LSTM SWH 연구](https://doi.org/10.1016/j.apor.2026.105016)는 "
                "미래 바람 입력의 이득을 보고했지만, P3는 미래 기상장을 제공하지 않으므로 그대로 재현할 수 없다."
            ),
        },
        {"id": "priority_block", "type": "table", "tableId": "priority_table"},
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## 권장 다음 실험 사양\n\n"
                "다음 한 번만 `single CatBoost`, `multi-output CatBoost`, `persistence`의 case별 6-lead loss를 outer-train "
                "내부에서 cross-fit으로 생성한다. 작은 gate는 과거 48시간 특징으로 세 loss를 회귀하고, 예측 loss의 softmax로 "
                "세 component를 결합한다. hard trajectory class와 미래 정답 class는 사용하지 않는다. gate temperature와 shrinkage는 "
                "inner block에서만 정하고 outer는 한 번 평가한다. +12/+18/+24와 세 fold가 함께 개선되지 않으면 routing 계열을 닫는다."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 한계·강건성·해석 경계\n\n"
                f"독립 validator 상태는 **{validation['status']}**이며 OOF grain·중복·RMSE·오차 share·bias probe를 "
                "재계산했다. 다만 182 case는 공식 hidden 200 case와 다르고, 공식 stretch 0.624165m와 로컬 RMSE를 "
                "직접 비교할 수 없다. case oracle과 future trajectory 분석은 구현 가능한 성능이 아니라 연구용 상한·분해다. "
                "domain AUC는 입력 분포 차이만 측정한다. router는 GPU feasibility run이라 bitwise 재현이 아니며, 승격 전 CPU "
                "또는 3-seed 확인이 필요하다."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 남은 질문\n\n"
                "- 과거 48시간만으로 성장과 peak-then-decay를 구분하는 router signal이 fold마다 안정적인가?\n"
                "- `tp`가 평균주기라는 제약 아래 inverse-wave-age 근사가 장기 lead를 실제로 개선하는가?\n"
                "- 첫 공식 제출이 로컬 개선 방향을 지지하는가, 아니면 2025-07~2026-06 shift가 더 큰가?"
            ),
        },
    ]

    generated = datetime.now().astimezone().isoformat()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "P3 유의파고 예측 돌파 정찰",
            "description": "실패 모드, 분포 이동, 편향 보정, 상태 라우터 및 1차 문헌의 기술 검증",
            "generatedAt": generated,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "leads": lead_rows,
                "trajectories": trajectory_rows,
                "shift": shift_rows,
                "bias": bias_rows,
                "router": fold_router_rows,
                "ablations": ablation_rows,
                "priorities": priority_rows,
            },
        },
        "sources": [{"id": source["id"], "path": source["path"]} for source in sources],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_artifact(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": args.output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
