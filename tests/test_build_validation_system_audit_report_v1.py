from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_validation_system_audit_report_v1.py"
SPEC = importlib.util.spec_from_file_location("build_validation_system_audit_report_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _payload(name: str) -> dict[str, object]:
    return json.loads(
        (ROOT / f"artifacts/validation_system_audit_20260822/{name}.json").read_text(
            encoding="utf-8"
        )
    )


def test_final_p1_p2_p3_and_superseded_report_pins_validate() -> None:
    evidence = report.collect_evidence(ROOT)
    assert evidence["hashes"] == report.EXPECTED_SHA256
    assert report.EXPECTED_SHA256["p1_audit"].startswith("0752941c")
    assert report.EXPECTED_SHA256["p2_audit"].startswith("ea09df2b")
    assert report.EXPECTED_SHA256["p3_audit"].startswith("4e2f2a74")
    assert report.EXPECTED_SHA256["cross_problem_policy"].startswith("9529aa4a")


def test_cross_problem_policy_is_portable_and_preregistered() -> None:
    evidence = report.collect_evidence(ROOT)
    report._validate_cross_problem_policy(evidence["cross_problem_policy"])
    serialized = json.dumps(evidence["cross_problem_policy"], ensure_ascii=False).lower()
    assert "c:/users/" not in serialized
    assert "c:\\users\\" not in serialized


def test_p1_adapter_separates_local_rank_from_hidden_calibration() -> None:
    row = report._adapt_p1(_payload("p1"))
    assert row["problem"] == "P1"
    assert row["decision"] == "LOCAL_RELATIVE_RANKING_SUPPORTED__ABSOLUTE_HIDDEN_UNCALIBRATED"
    assert "Δ +0.044511" in row["local_relative_ranking"]
    assert "43.61%" in row["hidden_calibration"]
    assert "outer-result exposure 13회" in row["adaptive_exposure"]
    assert "hidden F1/최적성 주장 금지" in row["action"]


def test_p2_adapter_uses_real_nested_decision_schema() -> None:
    row = report._adapt_p2(_payload("p2"))
    assert row["problem"] == "P2"
    assert row["decision"] == "DO_NOT_PROMOTE_FROM_THIS_VALIDATION_SYSTEM_ALONE"
    assert "2/5" in row["local_relative_ranking"]
    assert "same-season CI90" in row["local_relative_ranking"]
    assert "1/5" in row["hidden_calibration"]
    assert "executed generation ≥18" in row["adaptive_exposure"]
    assert "untouched seasonal validation" in row["action"]


def test_p3_adapter_preserves_validation_not_endorsement_conclusion() -> None:
    row = report._adapt_p3(_payload("p3"))
    assert row["problem"] == "P3"
    assert "34.667h" in row["implementation_integrity"]
    assert "55/100" in row["local_relative_ranking"]
    assert "25/100" in row["hidden_calibration"]
    assert "OOF ≥10" in row["adaptive_exposure"]
    assert "risk-control baseline" in row["action"]


def test_report_is_technical_supersession_without_cross_problem_chart() -> None:
    artifact = report.build_artifact(
        report.collect_evidence(ROOT),
        generated_at="2026-08-22T18:00:00+09:00",
    )
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert artifact["surface"] == "report"
    assert artifact["manifest"]["description"].endswith("technical audience")
    assert [chart["id"] for chart in artifact["manifest"]["charts"]] == ["p2_delta_rmse"]
    assert len(artifact["manifest"]["tables"]) == 1
    assert "## Technical Summary" in serialized
    assert "## Key Findings" in serialized
    assert "## Scope, Data, and Metric Definitions" in serialized
    assert "## Methodology" in serialized
    assert "## Limitations, Uncertainty, and Robustness Checks" in serialized
    assert "## Recommended Next Steps" in serialized
    assert "## Further Questions" in serialized
    assert "명시적으로 대체" in serialized
    assert "freeze = immutable baseline/risk control, not validation endorsement" in serialized
    assert "T=0.624165" in serialized
    assert "MCP report tools were unavailable" in serialized
    assert "technical portable HTML fallback" in serialized
    assert "executive/product-stakeholder portable HTML fallback" not in serialized
    assert "P1 pool은 현재 incomplete" in serialized


def test_only_chart_is_within_p2_same_unit_delta_rmse() -> None:
    artifact = report.build_artifact(
        report.collect_evidence(ROOT),
        generated_at="2026-08-22T18:00:00+09:00",
    )
    chart = artifact["manifest"]["charts"][0]
    assert chart["type"] == "horizontalBar"
    assert chart["sourceId"] == "p2_validation_audit"
    assert chart["unit"] == "°C"
    assert chart["referenceLines"] == [
        {
            "axis": "y",
            "value": 0,
            "label": "no change",
            "color": "neutral",
            "lineStyle": "solid",
        }
    ]
    rows = artifact["snapshot"]["datasets"]["p2_delta_rmse"]
    assert [row["scope"] for row in rows] == [
        "Pooled",
        "2024 Sep–Oct",
        "2025 Jul–Aug",
        "2025 Nov–Dec",
    ]
    assert [row["kst_days"] for row in rows] == [163, 61, 62, 40]
    assert [row["signed_delta_label"] for row in rows] == [
        "-0.006050°C",
        "-0.001251°C",
        "-0.011469°C",
        "+0.001257°C",
    ]
    assert all("ci90_lower_c" in row and "ci90_upper_c" in row for row in rows)
    assert "Cross-problem" not in chart["title"]


def test_trust_matrix_has_exact_requested_columns_and_three_rows() -> None:
    artifact = report.build_artifact(
        report.collect_evidence(ROOT),
        generated_at="2026-08-22T18:00:00+09:00",
    )
    columns = [column["field"] for column in artifact["manifest"]["tables"][0]["columns"]]
    assert columns == [
        "problem",
        "implementation_integrity",
        "local_relative_ranking",
        "hidden_calibration",
        "adaptive_exposure",
        "action",
    ]
    rows = artifact["snapshot"]["datasets"]["trust_matrix"]
    assert [row["problem"] for row in rows] == ["P1", "P2", "P3"]


def test_artifact_contains_only_relative_aggregate_source_paths() -> None:
    artifact = report.build_artifact(
        report.collect_evidence(ROOT),
        generated_at="2026-08-22T18:00:00+09:00",
    )
    for source in artifact["sources"]:
        if "path" in source:
            path = Path(source["path"])
            assert not path.is_absolute()
            assert path.suffix == ".json"
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert "C:\\Users\\" not in serialized
    assert ".parquet" not in serialized
    assert ".csv" not in serialized
    assert "test_context" not in serialized
    assert "test_index" not in serialized
