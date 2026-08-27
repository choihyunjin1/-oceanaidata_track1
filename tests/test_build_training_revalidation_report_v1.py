from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_training_revalidation_report_v1.py"
SPEC = importlib.util.spec_from_file_location("build_training_revalidation_report_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


@lru_cache(maxsize=1)
def _products() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    evidence = report.collect_evidence(ROOT)
    comparison = report.recompute_p3_candidate_distance(ROOT)
    generated_at = "2026-08-22T20:00:00+09:00"
    registry = report.build_registry(evidence, comparison, generated_at=generated_at)
    artifact = report.build_artifact(evidence, registry, generated_at=generated_at)
    return evidence, registry, artifact


def test_all_file_sources_are_relative_exact_sha_pins() -> None:
    evidence, _, _ = _products()
    assert set(report.RELATIVE_PATHS) == set(report.EXPECTED_SHA256)
    assert evidence["hashes"] == report.EXPECTED_SHA256
    for name, relative in report.RELATIVE_PATHS.items():
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == report.EXPECTED_SHA256[name]


def test_source_hash_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(report.EXPECTED_SHA256, "p1_pool_manifest", "0" * 64)
    with pytest.raises(report.TrainingRevalidationError, match="SHA mismatch"):
        report.collect_evidence(ROOT)


def test_p3_candidate_distance_is_bounded_and_parser_qualified() -> None:
    comparison = report.recompute_p3_candidate_distance(ROOT)
    assert comparison["method"] == "builder_independent_recompute"
    assert comparison["parser"] == "pandas-3.0.1 default CSV parser to float64"
    assert comparison["rows"] == 1200
    assert comparison["key_order_mismatches"] == 0
    assert comparison["exact_parsed_float_nonzero_rows"] == 748
    assert comparison["absolute_delta_gt_1e_12_rows"] == 660
    assert comparison["decimal_text_unequal_rows"] == 825
    assert comparison["prediction_rmse_distance_m"] == pytest.approx(
        0.001052575987759441, abs=1e-15
    )
    assert comparison["maximum_absolute_delta_m"] == pytest.approx(0.00716521632450795, abs=1e-15)
    assert comparison["raw_rows_retained_or_emitted"] == 0


def test_registry_separates_research_evidence_from_official_eligibility() -> None:
    _, registry, _ = _products()
    records = registry["eligibility_records"]
    assert records[0]["official_eligibility"] == "ELIGIBLE_PRE_FIRST_SCORE_CANDIDATE_2"
    assert records[1]["official_eligibility"] == "INELIGIBLE_FAILED_CLOSED"
    assert records[2]["official_eligibility"] == ("CORRECTED_RESEARCH_CANDIDATE_NOT_AUTO_PROMOTED")
    assert records[3]["official_eligibility"] == ("CORRECTED_RESEARCH_CANDIDATE_NOT_AUTO_PROMOTED")
    assert registry["policy_effect"]["existing_p2_p3_preregistered_pools_changed"] is False
    assert registry["policy_effect"]["official_submission_choice_changed"] is False
    assert registry["operation_counters"]["submission_uploads"] == 0
    assert registry["independent_qa"]["P2"]["p0_findings"] == 0
    assert registry["independent_qa"]["P3"]["p1_findings"] == 0


def test_frozen_baselines_and_candidate_hashes_are_exact() -> None:
    _, registry, _ = _products()
    assert {
        problem: payload["sha256"] for problem, payload in registry["frozen_baselines"].items()
    } == report.FROZEN_BASELINE_SHA256
    assert all(payload["unchanged"] for payload in registry["frozen_baselines"].values())
    assert registry["evidence"]["P1"]["causal_candidate_sha256"] == (
        "f76c91d2ce7db41542d51b698f5a376f327a9bc7a69b71f72d0b395ab2648da1"
    )
    assert registry["evidence"]["P2"]["candidate_sha256"] == (
        "d58ce75f76a54e9448e80505684a45c78e8cc7c5ad7c8fc51470012e81a4008a"
    )
    assert registry["evidence"]["P3"]["candidate_sha256"] == (
        "24a360dd85978155b883378459f6d4d46a6b847569f1c3b6636a728c96e5ba11"
    )


def test_p2_and_p3_charts_are_unit_separated_signed_and_zero_referenced() -> None:
    _, _, artifact = _products()
    charts = artifact["manifest"]["charts"]
    assert [chart["id"] for chart in charts] == [
        "p2_fold_delta_rmse",
        "p3_fold_delta_rmse",
    ]
    assert [chart["unit"] for chart in charts] == ["°C", "m"]
    assert all(chart["type"] == "horizontalBar" for chart in charts)
    assert all(chart["settings"]["showValues"] is True for chart in charts)
    assert all(chart["referenceLines"][0]["value"] == 0 for chart in charts)
    assert all(chart["referenceLines"][0]["color"] == "neutral" for chart in charts)
    assert all("p1" not in chart["id"].lower() for chart in charts)

    p2_rows = artifact["snapshot"]["datasets"]["p2_fold_delta_rmse"]
    p3_rows = artifact["snapshot"]["datasets"]["p3_fold_delta_rmse"]
    assert len(p2_rows) == len(p3_rows) == 4
    assert sum(row["delta_rmse_c"] > 0 for row in p2_rows) == 1
    assert all(row["delta_rmse_m"] < 0 for row in p3_rows)
    assert p2_rows[0]["delta_rmse_c"] == pytest.approx(-0.14321500084463312)
    assert p3_rows[0]["delta_rmse_m"] == pytest.approx(-0.0843925348838056)


def test_report_has_exact_decision_tables_and_technical_structure() -> None:
    _, _, artifact = _products()
    manifest = artifact["manifest"]
    assert manifest["blocks"][0]["body"] == f"# {report.REPORT_TITLE}"
    assert manifest["blocks"][1]["body"].startswith("## Technical Summary")
    assert [table["id"] for table in manifest["tables"]] == [
        "eligibility_decisions",
        "frozen_baselines",
    ]
    assert len(artifact["snapshot"]["datasets"]["eligibility_decisions"]) == 4
    assert len(artifact["snapshot"]["datasets"]["frozen_baselines"]) == 3
    serialized = json.dumps(artifact, ensure_ascii=False)
    for heading in (
        "## Technical Summary",
        "## Key Findings",
        "## Scope, Data, and Metric Definitions",
        "## Methodology",
        "## Limitations, Uncertainty, and Robustness Checks",
        "## Recommended Next Steps",
        "## Further Questions",
    ):
        assert heading in serialized
    assert "inner diagnostic aggregate" in serialized
    assert "+0.247666°C" in serialized
    assert "0.0010525759878m" in serialized


def test_artifact_registry_pin_and_safety_contract() -> None:
    _, registry, artifact = _products()
    expected_registry_sha = hashlib.sha256(report._canonical_bytes(registry)).hexdigest()
    registry_source = next(
        source for source in artifact["manifest"]["sources"] if source["id"] == "training_registry"
    )
    assert registry_source["sha256"] == expected_registry_sha
    assert artifact["package_info"]["delivery"] == "technical portable HTML fallback"
    serialized = json.dumps({"registry": registry, "artifact": artifact}, ensure_ascii=False)
    assert "MCP report tools were unavailable" in serialized
    assert "C:/Users/" not in serialized
    assert "C:\\Users\\" not in serialized
    assert "api_key" not in serialized.lower()
    assert "access_token" not in serialized.lower()
    assert registry["source_contract"]["raw_rows_retained_or_emitted"] == 0


def test_check_only_validates_without_creating_canonical_files() -> None:
    before = {
        "registry": report.DEFAULT_REGISTRY.exists(),
        "artifact": report.DEFAULT_ARTIFACT.exists(),
    }
    result = report.main(
        [
            "--root",
            str(ROOT),
            "--check-only",
            "--generated-at",
            "2026-08-22T20:00:00+09:00",
        ]
    )
    after = {
        "registry": (ROOT / report.DEFAULT_REGISTRY).exists(),
        "artifact": (ROOT / report.DEFAULT_ARTIFACT).exists(),
    }
    assert result == 0
    assert after == before
