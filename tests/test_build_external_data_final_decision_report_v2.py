from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_external_data_final_decision_report_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_external_data_final_decision_report_v2", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
REPO_ROOT = SCRIPT.parents[1]


@pytest.fixture(scope="module")
def evidence() -> dict[str, object]:
    return report.collect_evidence(REPO_ROOT)


@pytest.fixture(scope="module")
def artifact(evidence: dict[str, object]) -> dict[str, object]:
    return report.build_artifact(evidence, generated_at="2026-08-21T23:20:00+09:00")


def test_all_required_sources_are_exact_sha_pinned(
    evidence: dict[str, object],
) -> None:
    assert set(report.RELATIVE_PATHS) == set(report.EXPECTED_SHA256)
    assert len(report.RELATIVE_PATHS) == 14
    assert evidence["hashes"] == report.EXPECTED_SHA256
    assert report.EXPECTED_SHA256["p1_point"].startswith("dd7f720d")
    assert report.EXPECTED_SHA256["p2_era5"].startswith("b94fba56")
    assert report.EXPECTED_SHA256["p3_withdrawal"] == (
        "6eb84cec7c4a211ba8dc3ec6bfe5b2778dce40f80c962cdbea450371163dbb11"
    )


def test_final_decisions_and_exact_aggregate_metrics(
    evidence: dict[str, object],
) -> None:
    assert evidence["p1"] == {
        "delta": -0.04899753214538305,
        "ci_low": -0.10108725665784712,
        "ci_high": -0.029788056150546964,
    }
    assert evidence["p2_nasa"] == {
        "delta": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
    }
    assert evidence["p2_era5"]["delta"] == 8.348053914808418e-09
    assert evidence["p2_era5"]["ci_low"] == 2.8345666036599936e-09
    assert evidence["p2_era5"]["ci_high"] == 1.498833469559813e-08
    assert evidence["p3"]["local_delta"] == -0.0025293726008623896
    assert evidence["p3"]["deployed_delta"] == 0.0006208732949318785
    assert evidence["p3"]["posthoc_delta"] == -0.001088204044226826


def test_report_has_three_unit_separated_cards_and_one_requested_table(
    artifact: dict[str, object],
) -> None:
    manifest = artifact["manifest"]
    assert manifest["title"] == report.REPORT_TITLE
    assert len(manifest["cards"]) == 3
    assert len(manifest["charts"]) == 1
    assert manifest["charts"][0]["dataset"] == "p3_alpha_grid"
    assert manifest["charts"][0]["encodings"]["y"]["label"] == "ΔRMSE (mm)"
    assert len(manifest["tables"]) == 1
    assert [column["field"] for column in manifest["tables"][0]["columns"]] == [
        "problem",
        "hypothesis",
        "local_metric_delta",
        "uncertainty",
        "deployment_consistency",
        "decision",
        "current_incumbent",
    ]
    rows = artifact["snapshot"]["datasets"]["decision_register"]
    assert [row["problem"] for row in rows] == ["P1", "P2", "P3"]
    assert rows[0]["decision"] == "REJECT — NO_GO_POINT_RESIDUAL"
    assert rows[1]["decision"] == "REJECT ERA5 INCREMENT — KEEP CONTROL"
    assert rows[2]["decision"] == "WITHDRAWN_GLOBAL_DEPLOYMENT_MISMATCH"


def test_visual_contract_forbids_mixed_unit_axis(
    artifact: dict[str, object],
) -> None:
    datasets = artifact["snapshot"]["datasets"]
    contract = datasets["visual_contract"]
    assert {row["problem"] for row in contract} == {"P1", "P2", "P3"}
    assert all(row["form"] == "independent delta metric card" for row in contract[:2])
    assert "same-unit alpha-grid bar" in contract[2]["form"]
    assert datasets["p2_delta_card"][0]["era5_delta_micro_c"] == pytest.approx(0.008348053914808418)
    assert datasets["p3_delta_card"][0]["deployed_delta_mm"] == pytest.approx(0.6208732949318785)
    assert [row["alpha"] for row in datasets["p3_alpha_grid"]] == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert datasets["p3_alpha_grid"][2]["delta_rmse_mm"] == pytest.approx(-1.088204044226826)
    assert datasets["p3_alpha_grid"][4]["delta_rmse_mm"] == pytest.approx(0.6208732949318785)


def test_report_is_aggregate_only_portable_and_technically_complete(
    artifact: dict[str, object],
) -> None:
    serialized = json.dumps(artifact, ensure_ascii=False)
    lowered = serialized.lower()
    for forbidden in (
        "c:\\users\\",
        "c:/users/",
        "file://",
        "api_key",
        "station,time,temp",
        "case_id,station,lead_h",
        "target_hs",
    ):
        assert forbidden not in lowered
    assert all(not Path(source["path"]).is_absolute() for source in artifact["sources"])
    block_ids = {block["id"] for block in artifact["manifest"]["blocks"]}
    assert {
        "technical_summary",
        "p1_finding",
        "p2_finding",
        "p3_finding",
        "p3_grid_chart_intro",
        "scope_definitions",
        "methodology",
        "limitations",
        "next_steps",
        "further_questions",
    } <= block_ids
    registry = artifact["snapshot"]["datasets"]["evidence_registry"]
    assert len(registry) == 14
    assert all(len(row["sha256"]) == 64 for row in registry)
    assert all("hash-only" in row["read_mode"] for row in registry[-3:])


def test_sealed_reader_fails_closed_on_sha_mismatch(tmp_path: Path) -> None:
    fixture = tmp_path / "aggregate.json"
    fixture.write_text('{"decision":"REJECT"}', encoding="utf-8")
    actual = hashlib.sha256(fixture.read_bytes()).hexdigest()

    assert report._read_sealed_json(fixture, actual, "fixture")["decision"] == "REJECT"
    with pytest.raises(report.FinalDecisionReportError, match="sealed SHA mismatch"):
        report._read_sealed_json(fixture, "0" * 64, "fixture")


def test_check_only_validates_without_writing() -> None:
    output = Path("reports/generated/external_data_final_decision_v2_check_only.json")
    target = REPO_ROOT / output
    assert not target.exists()
    assert (
        report.main(
            [
                "--root",
                str(REPO_ROOT),
                "--output",
                str(output),
                "--generated-at",
                "2026-08-21T23:20:00+09:00",
                "--check-only",
            ]
        )
        == 0
    )
    assert not target.exists()
