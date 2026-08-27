from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_full_improvement_cycle_report_v3.py"
SPEC = importlib.util.spec_from_file_location("build_full_improvement_cycle_report_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

FIXED_TIME = "2026-08-23T01:00:00+09:00"


def _payloads():
    evidence, registry, r2, audit = builder.collect_inputs(ROOT)
    artifact = builder.build_artifact(r2, audit, generated_at=FIXED_TIME)
    return evidence, registry, r2, audit, artifact


def test_audit_and_r2_are_exactly_pinned() -> None:
    _, _, _, audit, _ = _payloads()
    assert hashlib.sha256((ROOT / builder.R2_ARTIFACT).read_bytes()).hexdigest() == (
        builder.EXPECTED_R2_ARTIFACT_SHA256
    )
    assert hashlib.sha256((ROOT / builder.AUDIT_RELATIVE_PATH).read_bytes()).hexdigest() == (
        builder.EXPECTED_AUDIT_SHA256
    )
    assert audit["schema_version"] == "meaningful_improvement_audit.v1"
    assert audit["summary"]["confirmed_meaningful_generalization_improvement_count"] == 0


def test_exact_meaningful_decisions_fail_all_three() -> None:
    _, _, _, audit, artifact = _payloads()
    assert audit["summary"]["candidate_point_estimate_improvement_count"] == 3
    assert audit["summary"]["failed_confirmatory_problems"] == ["P1", "P2", "P3"]
    for problem in ("P1", "P2", "P3"):
        decision = audit["problems"][problem]
        assert decision["overall_verdict"] == "FAIL"
        assert decision["confirmed_meaningful_generalization_improvement"] is False
        assert decision["candidate_status_recommendation"] == "research_only"
        assert decision["do_not_promote"] is True
    assert artifact["manifest"]["title"].endswith("confirmed meaningful improvements: 0/3")
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert "P2 — PROMISING LOCAL ONLY / FAIL CONFIRMED" in serialized
    assert "research_only / do_not_promote" in serialized
    assert "r3는 r2" in serialized


def test_r3_preserves_all_charts_and_quantitative_chart_datasets() -> None:
    _, _, r2, _, artifact = _payloads()
    assert artifact["manifest"]["charts"] == r2["manifest"]["charts"]
    for dataset_id in (
        "p1_signed_validation_delta",
        "p2_signed_validation_delta",
        "p3_signed_validation_delta",
    ):
        assert (
            artifact["snapshot"]["datasets"][dataset_id] == r2["snapshot"]["datasets"][dataset_id]
        )
    assert [chart["unit"] for chart in artifact["manifest"]["charts"]] == ["F1", "°C", "m"]
    assert all(chart["referenceLines"][0]["value"] == 0 for chart in artifact["manifest"]["charts"])


def test_summary_table_changes_only_decision_evidence_cells() -> None:
    _, _, r2, _, artifact = _payloads()
    old_rows = r2["snapshot"]["datasets"]["full_cycle_exact_registry"]
    new_rows = artifact["snapshot"]["datasets"]["full_cycle_exact_registry"]
    for old, new in zip(old_rows, new_rows, strict=True):
        normalized = dict(new)
        normalized["evidence"] = old["evidence"]
        assert normalized == old
        assert new["evidence"] == builder.DECISION_EVIDENCE[new["problem"]]
    table = artifact["manifest"]["tables"][0]
    assert table["sourceId"] == builder.SYNTHESIS_SOURCE_ID
    assert table["defaultSort"]["field"] in {column["field"] for column in table["columns"]}


def test_unrelated_blocks_and_existing_sources_are_preserved() -> None:
    _, _, r2, _, artifact = _payloads()
    unchanged = {
        "full_cycle_exact_table",
        "p1_signed_chart",
        "p2_signed_chart",
        "p3_signed_chart",
        "methodology",
        "operation_failures_resume",
        "further_questions",
        "source_method_note",
    }
    old_blocks = {block["id"]: block for block in r2["manifest"]["blocks"]}
    new_blocks = {block["id"]: block for block in artifact["manifest"]["blocks"]}
    for block_id in unchanged:
        assert new_blocks[block_id] == old_blocks[block_id]
    assert artifact["manifest"]["sources"][:-2] == r2["manifest"]["sources"]
    assert artifact["manifest"]["sources"][-2]["sha256"] == builder.EXPECTED_AUDIT_SHA256
    assert artifact["manifest"]["sources"][-1]["id"] == builder.SYNTHESIS_SOURCE_ID


def test_strict_clock_caveat_and_p2_local_only_are_visible() -> None:
    _, _, _, _, artifact = _payloads()
    serialized = json.dumps(artifact, ensure_ascii=False)
    for phrase in (
        "119-row boundary",
        "0.75/0.15",
        "adaptive/repeatedly exposed",
        "+0.0276402153°C",
        "+0.0770355591°C",
        "CI90 [-0.0020481402, +0.0011163551]",
    ):
        assert phrase in serialized


def test_technical_roles_visual_blocks_and_hygiene() -> None:
    _, _, _, _, artifact = _payloads()
    serialized = json.dumps(artifact, ensure_ascii=False)
    for heading in (
        "## Technical Summary",
        "## Key Findings",
        "## Scope, Data, and Metric Definitions",
        "## Methodology, Model, and Validation Design",
        "## Limitations, Uncertainty, and Robustness Checks",
        "## Recommended Next Steps",
        "## Further Questions",
    ):
        assert heading in serialized
    blocks = artifact["manifest"]["blocks"]
    assert len([block for block in blocks if block["type"] == "chart"]) == 3
    assert len([block for block in blocks if block["type"] == "table"]) == 1
    lowered = serialized.lower()
    for forbidden in ("c:/users/", "c:\\users\\", "api_key", "access_token", "password"):
        assert forbidden not in lowered


def test_check_only_preserves_output() -> None:
    output = ROOT / builder.DEFAULT_OUTPUT
    before = (
        output.exists(),
        hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None,
    )
    assert builder.main(["--root", str(ROOT), "--check-only", "--generated-at", FIXED_TIME]) == 0
    after = (
        output.exists(),
        hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None,
    )
    assert after == before
