from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_full_improvement_cycle_report_v1.py"
SPEC = importlib.util.spec_from_file_location("build_full_improvement_cycle_report_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

FIXED_TIME = "2026-08-22T23:59:00+09:00"


def _allow_pending() -> bool:
    return builder.FULL_CYCLE_QA_ATTESTATION["verdict"] == "PENDING_FINAL"


def _payloads():
    evidence = builder.collect_evidence(ROOT)
    registry = builder.build_registry(
        evidence,
        generated_at=FIXED_TIME,
        allow_pending_qa=_allow_pending(),
    )
    artifact = builder.build_artifact(
        registry,
        generated_at=FIXED_TIME,
        allow_pending_qa=_allow_pending(),
    )
    return evidence, registry, artifact


def test_all_aggregate_source_pins_match_current_bytes() -> None:
    evidence = builder.collect_evidence(ROOT)
    assert len(evidence["hashes"]) == 14
    assert set(evidence["hashes"]) == set(builder.RELATIVE_PATHS)
    for source_id, relative in builder.RELATIVE_PATHS.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == builder.EXPECTED_SHA256[source_id]


def test_problem_metrics_and_uncertainty_classification_are_exact() -> None:
    _, registry, _ = _payloads()
    p1, p2, p3 = (registry["problems"][key] for key in ("P1", "P2", "P3"))

    assert p1["baseline_value"] == pytest.approx(0.8603708380408055, abs=1e-15)
    assert p1["candidate_value"] == pytest.approx(0.8609416445623342, abs=1e-15)
    assert p1["delta_candidate_minus_baseline"] == pytest.approx(0.0005708065215287439, abs=1e-15)
    assert p1["ci90"] == pytest.approx([-0.001000625887793999, 0.0019853048894418186], abs=1e-15)
    assert p1["ci90_includes_zero"] is True

    assert p2["baseline_value"] == pytest.approx(1.1158878559665548, abs=1e-15)
    assert p2["candidate_value"] == pytest.approx(1.042512377552349, abs=1e-15)
    assert p2["delta_candidate_minus_baseline"] == pytest.approx(-0.07337547841420577, abs=1e-15)
    assert p2["ci90"] == pytest.approx([-0.10169064438448504, -0.04468915720888248], abs=1e-15)
    assert p2["ci90_includes_zero"] is False
    assert [row["delta"] for row in p2["chart_rows"][1:]] == pytest.approx(
        [0.07703555910367743, -0.19971867719510983, -0.03512559955712091],
        abs=1e-15,
    )

    assert p3["baseline_value"] == pytest.approx(0.7791048399763751, abs=1e-15)
    assert p3["candidate_value"] == pytest.approx(0.7786608799293823, abs=1e-15)
    assert p3["delta_candidate_minus_baseline"] == pytest.approx(-0.00044396004699287506, abs=1e-15)
    assert p3["ci90"] == pytest.approx([-0.0020481402282327087, 0.0011163550501478439], abs=1e-15)
    assert p3["ci90_includes_zero"] is True
    assert [row["delta"] for row in p3["chart_rows"][1:]] == pytest.approx(
        [-0.0011343494229109075, 0.00026745801480099196, -0.0009048926732522222],
        abs=1e-15,
    )
    assert registry["cross_problem_controls"]["ci90_excludes_zero_problems"] == ["P2"]


def test_full_fit_candidate_frozen_and_upload_contracts() -> None:
    _, registry, _ = _payloads()
    assert registry["cross_problem_controls"]["actual_full_fit_count"] == 3
    assert registry["cross_problem_controls"]["candidate_count"] == 3
    assert registry["cross_problem_controls"]["submission_uploads"] == 0
    assert registry["cross_problem_controls"]["official_pool_auto_promotions"] == 0
    for problem, record in registry["problems"].items():
        assert record["candidate"]["sha256"] == builder.CANDIDATE_SHA256[problem]
        assert record["candidate"]["reproduced_byte_identical"] is True
        assert record["candidate"]["key_order_valid"] is True
        assert record["frozen_sha256"] == builder.FROZEN_SHA256[problem]
        assert record["frozen_unchanged"] is True
        assert record["upload_count"] == 0
        assert record["full_fit"]["model_sha256"]


def test_charts_are_unit_separated_signed_and_zero_anchored() -> None:
    _, _, artifact = _payloads()
    charts = artifact["manifest"]["charts"]
    assert [chart["id"] for chart in charts] == [
        "p1_signed_validation_delta",
        "p2_signed_validation_delta",
        "p3_signed_validation_delta",
    ]
    assert [chart["unit"] for chart in charts] == ["F1", "°C", "m"]
    assert "positive=improvement" in charts[0]["subtitle"]
    assert "negative=improvement" in charts[1]["subtitle"]
    assert "negative=improvement" in charts[2]["subtitle"]
    assert "includes 0" in charts[0]["subtitle"]
    assert "excludes 0" in charts[1]["subtitle"]
    assert "includes 0" in charts[2]["subtitle"]
    for chart in charts:
        assert chart["type"] == "horizontalBar"
        assert chart["settings"]["showValues"] is True
        assert chart["referenceLines"] == [
            {
                "axis": "y",
                "value": 0,
                "label": "no change",
                "color": "neutral",
                "lineStyle": "solid",
            }
        ]
        rows = artifact["snapshot"]["datasets"][chart["dataset"]]
        assert len(rows) == 4
        assert all(row["signed_delta_label"].startswith(("+", "-")) for row in rows)


def test_technical_structure_table_and_sources_are_valid() -> None:
    _, registry, artifact = _payloads()
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
    table = artifact["manifest"]["tables"][0]
    declared = {column["field"] for column in table["columns"]}
    assert table["defaultSort"]["field"] == "problem"
    assert table["defaultSort"]["field"] in declared
    assert len(artifact["snapshot"]["datasets"]["full_cycle_exact_registry"]) == 3
    registry_sha = hashlib.sha256(builder._canonical_bytes(registry)).hexdigest()
    registry_source = next(
        source
        for source in artifact["manifest"]["sources"]
        if source["id"] == "full_cycle_registry"
    )
    assert registry_source["sha256"] == registry_sha
    assert "MCP report tools were unavailable" in serialized
    assert "cross_problem_metric_chart_forbidden" in json.dumps(registry)


def test_hygiene_excludes_absolute_paths_secrets_and_raw_row_artifacts() -> None:
    _, registry, artifact = _payloads()
    serialized = json.dumps({"registry": registry, "artifact": artifact}, ensure_ascii=False)
    lowered = serialized.lower()
    for forbidden in (
        "c:/users/",
        "c:\\users\\",
        "api_key",
        "access_token",
        "password",
        "winner_oof.parquet",
        "submission.csv",
        "train.csv",
        "test.csv",
    ):
        assert forbidden not in lowered
    assert registry["scope"]["raw_training_oof_test_submission_rows_read_by_builder"] == 0


def test_pending_qa_fail_closes_write_contract() -> None:
    if not _allow_pending():
        builder._validate_qa_attestation(allow_pending=False)
        return
    with pytest.raises(builder.FullCycleReportError, match="still pending"):
        builder._validate_qa_attestation(allow_pending=False)
    builder._validate_qa_attestation(allow_pending=True)


def test_check_only_preserves_any_existing_outputs() -> None:
    paths = [ROOT / builder.DEFAULT_REGISTRY, ROOT / builder.DEFAULT_ARTIFACT]
    before = {
        path: (
            path.exists(),
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
        )
        for path in paths
    }
    args = ["--root", str(ROOT), "--check-only", "--generated-at", FIXED_TIME]
    if _allow_pending():
        args.append("--allow-pending-qa")
    assert builder.main(args) == 0
    after = {
        path: (
            path.exists(),
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
        )
        for path in paths
    }
    assert after == before


def test_source_hash_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    corrupted = dict(builder.EXPECTED_SHA256)
    corrupted["p1_metrics"] = "0" * 64
    monkeypatch.setattr(builder, "EXPECTED_SHA256", corrupted)
    with pytest.raises(builder.FullCycleReportError, match="SHA mismatch for p1_metrics"):
        builder.collect_evidence(ROOT)
